#!/usr/bin/env python3
"""下調べ 4(CPU だけ): 10/12 の方策 5 種 × 種 2 本 + モデル B の、2 本の組のアンサンブルの L1 の表(best)。
同じ形の種どうしと、形の違う組でどれだけ違うか。保留を見たあとの下調べ(判定ではない)。"""
import json, os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np
ROOT = Path('/home/jetson/jetformer'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import rl_evaluate as ev
tab = ev.pred_table(); held = ev._heldout(); cols, f = ev._val_arrays(tab)
m0 = tab['part'] == 0
clean = np.isin(tab['scene'], held['human_clean_scenes']) & m0 & cols['fv'] & ~cols['impact']
y = np.nan_to_num(cols['y'])
zp = ROOT / 'data/zoo/policy_zoo_20261012/pred'
names = ['jetformer_v0_s0', 'jetformer_v0_s1', 'B', 'msfusion_s0', 'msfusion_s1', 'tcp_s0', 'tcp_s1', 'act_s0', 'act_s1', 'dinov2_s0', 'dinov2_s1']
P = {n: np.load(zp / (n + '.npz'))['pred'].astype(np.float64)[clean] for n in names}
Y = y[clean]
L = lambda p: float(np.abs(p - Y).mean())
single = {n: L(P[n]) for n in names}
rows = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        e = L(0.5 * (P[a] + P[b]))
        rows.append({'a': a, 'b': b, 'ens': round(e, 3), 'gain_vs_mean': round(0.5 * (single[a] + single[b]) - e, 3),
                     'disagree': round(float(np.abs(P[a] - P[b]).mean()), 2),
                     'disagree_steer': round(float(np.abs(P[a] - P[b])[..., 0].mean()), 2),
                     'disagree_thr': round(float(np.abs(P[a] - P[b])[..., 1].mean()), 2)})
rows.sort(key=lambda r: r['ens'])
# 平均の時間のずれ: 予測のステア列の、正解との相互相関の最大のずれ(フレーム)
def lag_of(p):
    best = None
    for d in range(-4, 5):
        # p[:, k] を y[:, k+d] と比べる(k+d が 0..9 に入るところだけ)
        ks = [k for k in range(10) if 0 <= k + d < 10]
        v = float(np.mean([np.abs(p[:, k, 0] - Y[:, k + d, 0]).mean() for k in ks]))
        if best is None or v < best[1]:
            best = (d, v)
    return best
lags = {n: lag_of(P[n]) for n in names}
out = {'single': {n: round(v, 3) for n, v in single.items()}, 'pairs': rows,
       'steer_best_lag_frames': {n: [l[0], round(l[1], 2)] for n, l in lags.items()}}
Path('/home/jetson/jetformer_reports/real_accuracy_20261015/ideas_data/probe_pairs.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for r in rows[:20]: print(r)
print('...')
for r in rows[-8:]: print(r)
print(out['steer_best_lag_frames'])
