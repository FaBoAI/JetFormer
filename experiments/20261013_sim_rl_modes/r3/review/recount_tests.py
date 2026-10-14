"""第 3〜4 回の審査: 画面・status・報告に出ている試験の主な数字を、試験の回ごとの行から数え直す(読むだけ・判定のコードは使わない)

    python3 -I recount_tests.py > ~/jetformer_reports/sim_rl_modes_20261013/r3/review/recount_tests.json

no_wall: r3/no_wall_r4/test/rows_test.jsonl(4,010 行)
recover: r3/rc4/eval_test.json の tests(609 回)
line:    r3/line_r4b/eval_test_r4b.json の tests(542 回)
"""
import json
from pathlib import Path

R3 = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3')

NW_N2_SETS = ('NW-N300-R4', 'NW-FS1000-R4', 'NW-LOWC-R4', 'NW-EDGE-R4', 'NW-STD')
NW_N3_SETS = ('NW-G', 'NW-R200-R4', 'NW-VX-R4', 'NW-HC-R4', 'NW-FC30-R4')


def no_wall():
    rows = [json.loads(x) for x in (R3 / 'no_wall_r4/test/rows_test.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    judged = [r for r in rows if r.get('judged')]
    sets = sorted({r['set'] for r in rows})
    by = {}
    for s in sets:
        rr = [r for r in rows if r['set'] == s]
        jj = [r for r in rr if r.get('judged')]
        by[s] = {'rows': len(rr), 'judged': len(jj), 'collisions_judged': sum(1 for r in jj if r.get('collisions')),
                 'min_clearance_judged_cm': round(100 * min(r['min_clearance'] for r in jj), 3) if jj else None,
                 'physics': sorted({str(r.get('physics')) for r in rr})}
    nominal = [r for r in judged if r.get('physics') == 'nominal']
    shifted = [r for r in judged if r.get('physics') != 'nominal']
    n300 = [r for r in rows if r['set'] == 'NW-N300-R4']
    sh = sum(r.get('shield_frames', 0) for r in n300)
    fr = sum(r.get('frames', 0) for r in n300)
    return {'rows': len(rows), 'judged': len(judged), 'judged_collisions': sum(1 for r in judged if r.get('collisions')),
            'nominal_judged': len(nominal), 'nominal_min_cm': round(100 * min(r['min_clearance'] for r in nominal), 4),
            'nominal_min_at': min(nominal, key=lambda r: r['min_clearance'])['id'],
            'shifted_judged': len(shifted), 'shifted_min_cm': round(100 * min(r['min_clearance'] for r in shifted), 4),
            'shifted_min_at': min(shifted, key=lambda r: r['min_clearance'])['id'],
            'nominal_8_to_9cm': sum(1 for r in nominal if 0.08 <= r['min_clearance'] < 0.09),
            'N7_N300_fraction': sh / fr if fr else None, 'N7_N300_run_max': max(r.get('shield_frames', 0) / r['frames'] for r in n300),
            'all_rows_collisions': sum(1 for r in rows if r.get('collisions')),
            'by_set': by}


def recover():
    d = json.loads((R3 / 'rc4/eval_test.json').read_text(encoding='utf-8'))
    t = d['tests']
    out = {'sets': {k: len(v) for k, v in t.items()}, 'runs': sum(len(v) for v in t.values())}
    p = t['RC-P100-R4']
    out['P100_valid_r3'] = sum(1 for r in p if r.get('valid_episode_r3'))
    out['P100_collisions'] = sum(1 for r in p if r.get('collisions'))
    out['P100_stall'] = sum(1 for r in p if r.get('stall'))
    out['P100_invalid_ids'] = [r['id'] for r in p if not r.get('valid_episode_r3')]
    ev = [e for r in p for e in (r.get('recovery') or [])]
    out['P100_events'] = len(ev)
    # recovered の鍵の名前を探す
    keys = sorted({k for e in ev for k in e})
    out['event_keys'] = keys
    for k in ('recovered', 'success', 'ok'):
        if k in keys:
            out['P100_events_recovered'] = sum(1 for e in ev if e.get(k))
            out['recovered_key'] = k
            break
    out['P100_events_per_run'] = len([e for r in p for e in (r.get('events') or [])]) / len(p)
    out['P100_mask_frac_mean'] = sum(r.get('mask_frac', 0) for r in p) / len(p)
    valid = [r for r in p if r.get('valid_episode_r3')]
    out['P100_gate_miss_in_valid'] = sum(1 for r in valid if r.get('gate_misses'))
    u = t['RC-U60-R4']
    out['U60_valid_r3'] = sum(1 for r in u if r.get('valid_episode_r3'))
    out['U60_min_cm'] = round(100 * min(r['min_clearance'] for r in u), 4)
    rec = t['RC-REC60']
    out['REC60_valid_r3'] = sum(1 for r in rec if r.get('valid_episode_r3'))
    out['REC60_min_cm'] = round(100 * min(r['min_clearance'] for r in rec), 4)
    out['all_collisions'] = {k: sum(1 for r in v if r.get('collisions')) for k, v in t.items()}
    return out


def line():
    d = json.loads((R3 / 'line_r4b/eval_test_r4b.json').read_text(encoding='utf-8'))
    t = d['tests']
    out = {'sets': {k: len(v) for k, v in t.items()}, 'runs': sum(len(v) for v in t.values())}
    allr = [r for v in t.values() for r in v]
    out['collisions'] = sum(1 for r in allr if r.get('collisions'))
    out['stalls'] = sum(1 for r in allr if r.get('stall'))
    out['escape_runs'] = sum(1 for r in allr if r.get('escape_frames'))
    out['reverse_frames'] = sum(r.get('reverse_frames', 0) for r in allr)
    out['valid_r3_by_set'] = {k: '%d/%d' % (sum(1 for r in v if r.get('valid_r3')), len(v)) for k, v in t.items()}
    valid = [r for r in allr if r.get('valid_r3')]
    out['min_valid_cm'] = round(100 * min(r['min_clearance'] for r in valid), 4)
    out['min_all_cm'] = round(100 * min(r['min_clearance'] for r in allr), 4)
    out['invalid'] = [(r['test'], r['id'], r.get('reason'), r.get('gate_misses')) for r in allr if not r.get('valid_r3')]
    sh = sum(r.get('shield_frames', 0) for r in t['LN-P60-R4'])
    fr = sum(r.get('log_frames', r.get('frames', 0)) for r in t['LN-P60-R4'])
    out['P60_shield_fraction'] = sh / fr
    return out


def main():
    print(json.dumps({'schema': 'jf-sim-rl-r3-review-recount/v1', 'no_wall': no_wall(), 'recover': recover(), 'line': line()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
