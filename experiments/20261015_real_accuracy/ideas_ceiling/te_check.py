#!/usr/bin/env python3
"""時間のアンサンブルの確かめ: last.pth・介入の集合・3 フレーム先(走行で使う値)のステア(CPU だけ)"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'free_combo.py')).read().split('res = {}')[0])   # temporal()・L()・prev を共有
iv = (part == 1) & Z['fv']
key1 = {(s, int(i)): j for j, (s, i) in enumerate(zip(scene, frame)) if part[j] == 1}
for j in range(len(scene)):
    if part[j] == 1:
        for d in range(10):
            prev[j, d] = key1.get((scene[j], int(frame[j]) - d), -1)
out = {}
for n in ('jetformer_v0_s0', 'jetformer_v0_s1', 'B', 'jetformer_v0_s0_last', 'jetformer_v0_s1_last', 'B_last', 'msfusion_s0', 'msfusion_s1'):
    p = P[idx[n]]
    for dec in (0.5, 1.0):
        t = temporal(p, dec)
        out.setdefault(n, {})['decay%.1f' % dec] = {
            'L1': [L(p), L(t)], 'L1_intervention': [L(p, iv), L(t, iv)],
            'k3_steer': [round(float(np.abs(p - Y)[clean][:, 2, 0].mean()), 2), round(float(np.abs(t - Y)[clean][:, 2, 0].mean()), 2)],
            'k3_throttle': [round(float(np.abs(p - Y)[clean][:, 2, 1].mean()), 2), round(float(np.abs(t - Y)[clean][:, 2, 1].mean()), 2)]}
json.dump(out, open(os.path.join(HERE, 'out', 'te_check.json'), 'w'), indent=1)
for n, v in out.items():
    print(n, v['decay1.0'])
