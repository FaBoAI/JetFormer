#!/usr/bin/env python3
"""駆動の効き(スロットル → 速さ)の変化で、スロットルの予測を直せるかの CPU の下見(学習なし)

g = 直前 5 秒(同じシーンの前のフレームだけ)の 速さの中央値 / (1500 − 実際に出したスロットル)の中央値 [m/s/µs]
直し方(物理の素朴な仮定 = 人は効きが落ちた分だけスロットルを深くする): 1500 − thr' = (1500 − thr) × (g_ref / g)^α
g_ref = 学習のシーンの g の中央値(ジャンル別)。α = 0(直さない)・0.5・1。倍率は 0.7〜1.5 に制限。
"""
import json
import os
import sys

import numpy as np

REPO = '/home/jetson/jetformer'
sys.path.insert(0, os.path.join(REPO, 'tools'))
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(rp.rl_dir(ev.DATASET) / 'frames.npz', allow_pickle=False)
f = {k: z[k] for k in ('scene', 'frame', 'mode', 'erpm', 'action_label', 'split', 'vehicle_valid')}
C = np.load(os.path.join(HERE, 'cache', 'heldout_cache.npz'), allow_pickle=False)
names = [str(x).split('/')[-1] for x in C['names']]
clean_scenes = set(x[:-4] for x in os.listdir(os.path.join(REPO, 'data/datasets/clean_driving_20261011/aligned')))
spd = -f['erpm'] * 0.5e-3
thr = f['action_label'][:, 1].astype(np.float64)
g = np.full(len(spd), np.nan)
for s in clean_scenes:
    r = np.nonzero(f['scene'] == s)[0]
    r = r[np.argsort(f['frame'][r])]
    for i in range(len(r)):
        w = r[max(0, i - 150):i]
        ok = f['vehicle_valid'][w] & (spd[w] > 0.8) & (thr[w] < 1450)
        if ok.sum() >= 60:
            g[r[i]] = np.median(spd[w][ok]) / np.median(1500.0 - thr[w][ok])
res = {}
gref = {}
for gen in (4, 5):
    m = np.isin(f['scene'], list(clean_scenes)) & (f['split'] == 'train') & (f['mode'] == gen) & np.isfinite(g)
    gref[gen] = float(np.median(g[m]))
res['g_ref'] = gref
cl = C['clean']
rows = C['frow'][cl]
Y = np.nan_to_num(C['y'][cl].astype(np.float64))
gg = g[rows]
mode = f['mode'][rows]
ratio = np.array([gref[int(m)] for m in mode]) / gg
res['val_rows_with_g'] = int(np.isfinite(gg).sum())
res['ratio_by_scene_median'] = {s: round(float(np.nanmedian(ratio[C['scene'][cl] == s])), 3) for s in np.unique(C['scene'][cl])}
for n in ('jetformer_v0_s0', 'jetformer_v0_s1', 'B'):
    P = C['preds'][names.index(n)][cl].astype(np.float64)
    out = {}
    for a in (0.0, 0.5, 1.0):
        fac = np.clip(np.where(np.isfinite(ratio), ratio, 1.0) ** a, 0.7, 1.5)
        Q = P.copy()
        Q[:, :, 1] = 1500.0 - (1500.0 - P[:, :, 1]) * fac[:, None]
        m193 = C['scene'][cl] == '20261010-193638-0003'
        out['alpha%.1f' % a] = {'L1': round(float(np.abs(Q - Y).mean()), 3),
                                'L1_throttle': round(float(np.abs(Q - Y)[:, :, 1].mean()), 2),
                                'L1_193638_0003': round(float(np.abs(Q - Y)[m193].mean()), 2),
                                'L1_other_scenes': round(float(np.abs(Q - Y)[~m193].mean()), 3)}
    res[n] = out
json.dump(res, open(os.path.join(HERE, 'out', 'plant_gain_probe.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
