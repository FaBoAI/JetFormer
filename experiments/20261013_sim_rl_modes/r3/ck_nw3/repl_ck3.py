#!/usr/bin/env python3
# 試験の行の作り直し(自前の走らせ方で、試験の JSON の条件だけを読む): 各組の最小すき間の回・止まりの回・VX の「確認 r2 の隅」の回・NW-OFF の衝突の回・各組から乱数で 4 回
import json, sys, random
T = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r3/test/rows_test.jsonl'
rows = [json.loads(l) for l in open(T)]
pick = {}
def add(r): pick[f"{r['set']}:{r['id']}"] = r
for s in {r['set'] for r in rows}:
    g = [r for r in rows if r['set'] == s and r['set'] not in ('NW-LONG-R3',)]
    if not g: continue
    add(min(g, key=lambda r: r['min_clearance']))
    random.seed(20261014 + len(s))
    for r in random.sample(g, min(4, len(g))): add(r)
for r in rows:
    if r.get('stall') or (r['collisions'] and r['set'] == 'NW-OFF-R3'): add(r)
    d = r.get('dr') or {}
    p = d.get('physics') or {}
    if r['set'] == 'NW-VX-R3' and p.get('steerRate') == 1.8 and p.get('brakeDeceleration') == .7 and p.get('frictionMultiplier') == .85 and d.get('frictionCircle') and d.get('speedGain') == 1.04: add(r)
jobs = []
for k, r in pick.items():
    d = r.get('dr') or {}
    j = {'set': 'REPL', 'id': f'REPL:{k}', 'seed': r['start'].get('seed', 1), 'start': {'gate': r['start']['gate'], 'off': r['start']['left_offset_m'], 'yaw': r['start']['left_yaw_deg']},
         'phys': d.get('physics') or None, 'delay': d.get('delay', 0), 'sg': d.get('steerGain', 1), 'vg': d.get('speedGain', 1), 'fc': bool(d.get('frictionCircle')),
         'T': r['time_limit'], 'episode': r['episode_mode'], 'shield': r['shield'], 'test_row': {x: r.get(x) for x in ['reason', 'frames', 'collisions', 'shield_frames', 'valid_laps', 'min_clearance', 'stall', 'valid_episode_r3']}}
    if r['set'] == 'FAST':
        j.update({'mode': 'fastest', 'policy': '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/bundle/fastest/policy.json', 'rtDir': '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl', 'shield': True})
    if d.get('physics') == {} : j['phys'] = None
    if 'LOWMU' in k: j['phys'] = {'frictionMultiplier': .4}
    jobs.append(j)
open(sys.argv[1], 'w').write(''.join(json.dumps(j) + '\n' for j in jobs))
print(len(jobs))
