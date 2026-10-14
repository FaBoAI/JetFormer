#!/usr/bin/env python3
# 確認 r3 の追加の組の集計: REG(確認 r2 で崩れた所の作り直し)・REPL(試験の行の作り直し)・EDGE2・NBR・FB(替えなかったフレームも含むシールドの内訳)
import json, glob, sys
from collections import Counter, defaultdict
O = sys.argv[1]
rows = [json.loads(l) for d in ['runs_x', 'runs_nbr'] for f in glob.glob(f'{O}/{d}/rows_*.jsonl') for l in open(f) if l.strip()]
errs = [r for r in rows if 'error' in r]; rows = [r for r in rows if 'error' not in r]
by = defaultdict(list)
for r in rows: by[r['set']].append(r)
S = {'rows': len(rows), 'errors': len(errs)}
for s, g in sorted(by.items()):
    S[s] = {'n': len(g), 'collisions': sum(r['collisions'] > 0 for r in g), 'stalls': sum(r['stall'] for r in g),
            'min_cm': round(100 * min(r['min_clearance'] for r in g), 3), 'lt8': sum(r['min_clearance'] < .08 for r in g), 'valid_r3': sum(r['valid_episode_r3'] for r in g),
            'first_lap_max': max((r['first_lap'] or 99) for r in g), 'bad': [{k: r[k] for k in ['id', 'reason', 'collision', 'min_clearance', 'stall', 'valid_laps', 'fb']} for r in g if r['collisions'] or r['stall'] or r['min_clearance'] < .08]}
# REG を組ごとに
reg = defaultdict(list)
for r in rows:
    if r['set'].startswith('REG'): reg[r['set']].append(r)
S['REG_by'] = {k: {'n': len(v), 'coll': sum(r['collisions'] > 0 for r in v), 'min_cm': round(100 * min(r['min_clearance'] for r in v), 2), 'stalls': sum(r['stall'] for r in v),
                   'valid': sum(r['valid_episode_r3'] for r in v), 'first_lap_max': round(max((r['first_lap'] or 99) for r in v), 2)} for k, v in reg.items()}
# REPL: 試験の行との比べ
jobs = {json.loads(l)['id']: json.loads(l) for l in open(f'{O}/jobs_repl_ck3.jsonl')}
rep = []
for r in by.get('REPL', []):
    t = jobs[r['id']]['test_row']
    same = r['reason'] == t['reason'] and r['frames'] == t['frames'] and r['collisions'] == t['collisions'] and r['valid_laps'] == t['valid_laps'] \
        and r['stall'] == bool(t['stall']) and r['valid_episode_r3'] == bool(t['valid_episode_r3']) and (r['sh'] == t['shield_frames'] or r['mode'] == 'fastest')
    d = abs(r['min_clearance'] - t['min_clearance']) if r['collisions'] == 0 and t['collisions'] == 0 else None
    rep.append({'id': r['id'], 'exact': same and (d is None or d < 1e-6), 'same_outcome': r['collisions'] == t['collisions'] and r['stall'] == bool(t['stall']) and r['valid_episode_r3'] == bool(t['valid_episode_r3']),
                'fc': r['fc'], 'dmin_cm': None if d is None else round(100 * d, 4), 'mine': [r['reason'], r['frames'], r['sh'], round(r['min_clearance'], 5)], 'test': [t['reason'], t['frames'], t['shield_frames'], round(t['min_clearance'], 5)]})
S['REPL_compare'] = {'n': len(rep), 'exact': sum(x['exact'] for x in rep), 'same_outcome': sum(x['same_outcome'] for x in rep),
                     'not_exact': [x for x in rep if not x['exact']], 'not_exact_all_fc': all(x['fc'] for x in rep if not x['exact'])}
# FB: 替えなかったフレームも含むシールドの内訳(標準の物理・FS の最初の 200)
fb = by.get('FB', [])
if fb:
    tot = Counter(); runs_giveup = 0; frames = sum(r['frames'] for r in fb)
    for r in fb:
        c = Counter(r.get('fb_all', {})); tot += c
        if c.get('proposal_best', 0) or c.get('max_clearance', 0): runs_giveup += 1
    S['FB'] = {'runs': len(fb), 'frames': frames, 'fb_all_total': dict(tot), 'runs_with_no_candidate_passing (proposal_best/max_clearance)': runs_giveup,
               'frac_frames_no_candidate': round((tot.get('proposal_best', 0) + tot.get('max_clearance', 0)) / frames, 5),
               'frac_frames_proposal_nominal': round(tot.get('proposal_nominal', 0) / frames, 5)}
json.dump(S, open(f'{O}/summary_extra.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False)[:6000])
