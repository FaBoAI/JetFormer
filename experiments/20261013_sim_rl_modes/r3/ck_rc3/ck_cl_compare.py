"""離脱復活 r3 の確認: ブラウザ(Firefox)の閉ループと Node(自前の作り方 FP)の同じ開始・同じ種の走りを比べる(2026-10-14 21:50)。
記録の前の予想(Node)が、ブラウザで記録したときの「有効(valid_episode_r3)・ゲートの外し・出来事」をどれだけ当てるか。
  python3 -I ck_cl_compare.py <browser_cl60.json> <runs のフォルダ> <out.json>
"""
import glob
import json
import sys
from collections import Counter

B, RUNS, OUT = sys.argv[1:4]
L = 26.119648724030924
br = {r['id']: r for r in json.load(open(B))['closed_loop_batch']['results']}
node = {}
for p in glob.glob(RUNS + '/rows_*.jsonl'):
    for line in open(p):
        r = json.loads(line)
        if r['set'] == 'FP' and r['id'] in br:
            node[r['id']] = r


def stall(v, events):
    T = len(v)
    ex = [False] * T
    for e in events:
        typ, kind, ps, pe = e['type'], e.get('kind'), e['ps'], e['pe']
        if typ == 'speed_pulse' and kind == 'brake':
            for f in range(ps, min(T, (pe if pe is not None else T) + 30)):
                ex[f] = True
    run = mx = 0
    for f in range(30, T):
        if ex[f]:
            continue
        if abs(v[f]) < 0.2:
            run += 1
            mx = max(mx, run)
        else:
            run = 0
    return mx >= 30


def gate_miss(s, g):
    ls, lg, n = s[0], g[0], 0
    for f in range(len(s)):
        if g[f] != lg:
            lg, ls = g[f], s[f]
        elif s[f] - ls > 2.02:
            n += 1
            ls = s[f]
    return n > 0


def summ_browser(b):
    evs = [{'ps': e[0], 'pe': e[1], 'type': e[2], 'kind': e[3]} for e in b['events']]
    valid = b['reason'] == 'collection_duration_complete' and b['frames'] == 600 and b['collisions'] == 0
    st = stall(b['v'], evs)
    prog = b['s'][-1] - b['s'][0]
    return {'valid_r3': bool(valid and not st and prog >= L), 'collision': b['collisions'] > 0, 'stall': st, 'events': len(evs),
            'gate_miss': gate_miss(b['s'], b['g']), 'ev_sig': [(e['ps'], e['type']) for e in evs]}


def summ_node(r):
    valid = r['valid_episode']
    st = stall(r['pf']['v'], r['events'])
    prog = r['final']['s'] - r['pf']['s'][0]
    return {'valid_r3': bool(valid and not st and prog >= L), 'collision': r['collisions'] > 0, 'stall': st, 'events': len(r['events']),
            'gate_miss': gate_miss(r['pf']['s'], r['pf']['g']), 'ev_sig': [(e['ps'], e['type']) for e in r['events']]}


rows = []
for k in sorted(br, key=lambda x: int(x[1:])):
    if br[k].get('invalid_spawn') or k not in node:
        continue
    a, b = summ_node(node[k]), summ_browser(br[k])
    first = next((i for i, (x, y) in enumerate(zip(node[k]['pf']['st'], [None] * 0)) if False), None)
    rows.append({'id': k, 'node': {x: a[x] for x in ('valid_r3', 'collision', 'stall', 'events', 'gate_miss')},
                 'browser': {x: b[x] for x in ('valid_r3', 'collision', 'stall', 'events', 'gate_miss')},
                 'same_events': a['ev_sig'] == b['ev_sig'], 'same_frames': node[k]['frames'] == br[k]['frames']})
c = Counter()
for r in rows:
    c['n'] += 1
    c['same_event_list'] += r['same_events']
    c['same_valid'] += r['node']['valid_r3'] == r['browser']['valid_r3']
    c['node_valid'] += r['node']['valid_r3']
    c['browser_valid'] += r['browser']['valid_r3']
    c['node_valid_browser_invalid'] += r['node']['valid_r3'] and not r['browser']['valid_r3']
    c['node_invalid_browser_valid'] += (not r['node']['valid_r3']) and r['browser']['valid_r3']
    c['node_ok_nogm'] += r['node']['valid_r3'] and not r['node']['gate_miss']
    c['browser_ok_nogm'] += r['browser']['valid_r3'] and not r['browser']['gate_miss']
    c['pred_ok_nogm_browser_not'] += (r['node']['valid_r3'] and not r['node']['gate_miss']) and not (r['browser']['valid_r3'] and not r['browser']['gate_miss'])
    c['browser_stall'] += r['browser']['stall']
    c['node_stall'] += r['node']['stall']
res = {'counts': dict(c), 'diff_rows': [r for r in rows if r['node']['valid_r3'] != r['browser']['valid_r3'] or r['node']['gate_miss'] != r['browser']['gate_miss']]}
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])
