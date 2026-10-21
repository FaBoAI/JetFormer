#!/usr/bin/env python3
"""V-models: AI 走行の状態での SLOW と B の差を、R1 の手前・速さの帯で分ける。best と last の違いがどのフレームで大きいか。numpy だけ"""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path(sys.argv[1]); LAB = Path('/home/jetson/jetformer/data/labels')
Z = {c: np.load(OUT / ('gpu_%s.npz' % c), allow_pickle=False) for c in ('B', 'SLOW', 'WGT', 'AUG', 'ENS', 'ACT')}
z0 = Z['B']; t = z0['ai_t_us'] / 1e6; v = -z0['ai_erpm'] * 0.5e-3
lg = np.array([str(s)[:15] for s in z0['ai_scene']])
pre = np.zeros(len(t), bool); turn = np.zeros(len(t), bool)
for L in np.unique(lg):
    zl = np.load(LAB / (L + '.npz')); det, ts = zl['loc_detail'], zl['t_us'] / 1e6
    for ta in ts[np.nonzero((det[1:] == 1) & (det[:-1] == 7))[0] + 1] + 0.05:
        m = lg == L
        pre |= m & (t >= ta - 1.0) & (t < ta); turn |= m & (t >= ta) & (t < ta + 0.5)
res = {}
for name, m in (('pre1s_all', pre), ('pre1s_v>=2.75', pre & (v >= 2.75)), ('pre1s_v<2.75', pre & (v < 2.75) & (v >= 1.0)),
                ('turn0.5s_all', turn), ('all_v>=1', v >= 1.0), ('all_v<1', v < 1.0)):
    r = {'n': int(m.sum())}
    for c in Z:
        if c == 'B':
            r['B_thr_row2'] = round(float(z0['ai_pred'][m][:, 2, 1].mean()), 1); continue
        d = Z[c]['ai_pred'][m].astype(np.float64) - z0['ai_pred'][m].astype(np.float64)
        r[c] = {'thr_minus_B': round(float(d[:, 2, 1].mean()), 1), 'steer_abs_minus_B': round(float(np.abs(d[:, 2, 0]).mean()), 1)}
    res[name] = r
# best と last の差(GPU の段の参照フレーム 64 個: AI の低速 24・AI のそれ以外 24・保留の R1 の手前 16)
fp = np.load(OUT / 'gpu_fp32_ref.npz')
gj = json.load(open(OUT / 'gpu.json'))
keys = gj['ref_keys']
kind = np.array([k[0] for k in keys]); kidx = np.array([k[1] for k in keys])
for c in ('SLOW', 'AUG', 'WGT'):
    d = np.abs(fp['%s_last' % c] - fp['%s_best' % c])
    out = {}
    for nm, m in (('ai_lowspeed', (kind == 'ai') & (v[np.where(kind == 'ai', kidx, 0)] < 0.3)),
                  ('ai_moving', (kind == 'ai') & (v[np.where(kind == 'ai', kidx, 0)] >= 0.3)), ('heldout_r1pre', kind == 'ho')):
        out[nm] = {'n': int(m.sum()), 'steer_max': round(float(d[m][:, :, 0].max()), 1), 'thr_max': round(float(d[m][:, :, 1].max()), 1),
                   'steer_mean': round(float(d[m][:, :, 0].mean()), 1), 'thr_mean': round(float(d[m][:, :, 1].mean()), 1)}
    res['best_vs_last_' + c] = out
json.dump(res, open(OUT / 'slow_ai.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=0))
