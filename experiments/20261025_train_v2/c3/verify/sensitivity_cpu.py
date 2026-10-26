#!/usr/bin/env python3
"""verify_c3(CPU): 門の余裕と、決まりの読み方を変えたときに門の答えが変わるか(参考)。保存した予測だけ。書くのは sensitivity_cpu.json"""
import json, sys
from pathlib import Path
import numpy as np
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025'); DESK = REP / 'desk'
HERE = Path('/home/jetson/jetformer/experiments/20261025_train_v2/c3/verify')
desk = json.load(open(DESK / 'desk_v2.json'))
z = np.load('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
prim = z['primary']; v_h1 = -z['erpm'].astype(np.float64) * 0.5e-3
out = {}
for name, row in desk['runs'].items():
    job = (row.get('trt') or {}).get('trt_job')
    e = REP / ('eval/pred/%s.npz' % name)
    if not job or not e.exists():
        continue
    pb = np.load(e)['pred'].astype(np.float64)
    t = np.load(DESK / 'trt' / job / ('%s.npz' % name))
    sel = t['part'] == 0
    u = t['pred'][sel, 0, 1].astype(np.float64); v = -t['erpm'][sel].astype(np.float64) * 0.5e-3
    lowh = prim & (v_h1 < 1.0); lowa = v < 1.0
    k = lambda thr, mh, ma: (int((pb[mh, 0, 1] > thr).sum()) + int((u[ma] > thr).sum())) / float(mh.sum() + ma.sum())
    r = {'defined_1598_v<1': k(1598, lowh, lowa),
         'ref_0<=v<1': k(1598, lowh & (v_h1 >= 0), lowa & (v >= 0)),
         'ref_0.3<=v<1': k(1598, lowh & (v_h1 >= 0.3), lowa & (v >= 0.3)),
         'ref_thr_1648': k(1648, lowh, lowa),
         'ai_brake_frames_v<0.3': int(((u > 1598) & lowa & (v < 0.3)).sum()), 'ai_brake_frames': int(((u > 1598) & lowa).sum())}
    out[name] = r
json.dump(out, open(HERE / 'sensitivity_cpu.json', 'w'), indent=1)
for n in ['clean_driving_20261011-nocmd-s3-20261011-024626', 'v2_20261025-v2h-e60-s0', 'v2_20261025-v2h-e60-s1', 'bfast_20261025-e15-s0']:
    print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[n].items()})
fam = {n: desk['runs'][n].get('family') for n in out}
print('v2/bfast で 0.3<=v<1 でも 0.01 を超える数:', sum(1 for n, r in out.items() if fam[n] != 'ref' and r['ref_0.3<=v<1'] > 0.01), '/', sum(1 for n in out if fam[n] != 'ref'))
print('v2/bfast の min(定義の値):', min(r['defined_1598_v<1'] for n, r in out.items() if fam[n] != 'ref'))
