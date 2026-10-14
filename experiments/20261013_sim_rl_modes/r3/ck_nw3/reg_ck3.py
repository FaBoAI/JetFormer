#!/usr/bin/env python3
# 確認 r2 で崩れた所の作り直し(退行の確かめ。新しい開始ではない): VX の衝突 10 回(同じ開始・同じ物理)、
# K0 × 33 ゲート(横 0・向き 0)、K0 の遅延 1 版(r3 の学習の段の K7)× 33、「標準の舵・ゲイン + 摩擦円・μ 0.85・ブレーキ 0.7」× 33、内側 2 × 33
import json, sys
R2 = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_no_wall_r2/runs/rows_VX_NG.jsonl'
jobs = []
seed = 4400000
def add(set_, id_, start, d):
    global seed
    jobs.append({'set': set_, 'id': f'{set_}:{id_}', 'seed': seed, 'start': start, 'phys': d['physics'], 'delay': d['delay'], 'sg': d['steerGain'], 'vg': d['speedGain'], 'fc': d['frictionCircle']}); seed += 1
for l in open(R2):
    r = json.loads(l)
    if r.get('set') == 'VX' and r['collisions'] > 0:
        add('REG-VX', r['id'], {'gate': r['start']['gate'], 'off': r['start']['left_offset_m'], 'yaw': r['start']['left_yaw_deg']}, r['dr'])
K0 = {'physics': {'steerRate': 1.8, 'brakeDeceleration': .7, 'driveTau': .28, 'driveAccelerationCap': 3.5, 'frictionMultiplier': .85}, 'delay': 0, 'steerGain': 1.06, 'speedGain': 1.04, 'frictionCircle': True}
K7 = {**K0, 'delay': 1}
KS = {'physics': {'frictionMultiplier': .85, 'brakeDeceleration': .7}, 'delay': 0, 'steerGain': 1, 'speedGain': 1, 'frictionCircle': True}   # 確認 r2 の K11
IN2 = {'physics': {'steerRate': 1.7, 'brakeDeceleration': .75, 'driveTau': .3, 'driveAccelerationCap': 4, 'frictionMultiplier': .87}, 'delay': 0, 'steerGain': 1.04, 'speedGain': 1.03, 'frictionCircle': True}   # 確認 r2 の K10
for name, d in [('K0', K0), ('K7', K7), ('KS', KS), ('IN2', IN2)]:
    for g in range(33): add(f'REG-{name}', f'g{g}', {'gate': g, 'off': 0, 'yaw': 0}, d)
open(sys.argv[1], 'w').write(''.join(json.dumps(j) + '\n' for j in jobs))
print(len(jobs))
