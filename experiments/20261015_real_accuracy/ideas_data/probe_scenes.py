#!/usr/bin/env python3
"""下調べ 5(CPU だけ): v0+msfusion のアンサンブルと時間方向のアンサンブルは、11 シーンのうち何シーンで良くなるか(種 0・1 の平均)。"""
import json, os, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np

ROOT = Path('/home/jetson/jetformer'); sys.path.insert(0, str(ROOT / 'tools'))
import rl_evaluate as ev
tab = ev.pred_table(); held = ev._heldout(); cols, f = ev._val_arrays(tab)
m0 = tab['part'] == 0
clean = np.isin(tab['scene'], held['human_clean_scenes']) & m0 & cols['fv'] & ~cols['impact']
y = np.nan_to_num(cols['y']); sc, fr = tab['scene'], tab['frame']
zp = ROOT / 'data/zoo/policy_zoo_20261012/pred'
ld = lambda n: np.load(zp / (n + '.npz'))['pred'].astype(np.float64)
row_of = {(s, int(i)): k for k, (s, i) in enumerate(zip(sc, fr)) if m0[k]}
prev = np.full((len(sc), 3), -1, int)
for k in np.nonzero(m0)[0]:
    for j in range(3):
        prev[k, j] = row_of.get((sc[k], int(fr[k]) - j), -1)
def temporal(p, J=2):
    out = p.copy()
    for k in range(10):
        num, den = p[:, k, :].copy(), np.ones(len(p))
        for j in range(1, J + 1):
            if k + j > 9: break
            ok = prev[:, j] >= 0
            num[ok] += p[prev[ok, j], k + j, :]; den[ok] += 1
        out[:, k, :] = num / den[:, None]
    return out
res = {}
for s in sorted(set(sc[clean].tolist())):
    m = clean & (sc == s)
    L = lambda p: float(np.abs(p - y)[m].mean())
    base = np.mean([L(ld('jetformer_v0_s%d' % i)) for i in (0, 1)])
    e2 = np.mean([L(0.5 * (ld('jetformer_v0_s%d' % i) + ld('msfusion_s%d' % i))) for i in (0, 1)])
    t = np.mean([L(temporal(ld('jetformer_v0_s%d' % i))) for i in (0, 1)])
    res[s] = {'n': int(m.sum()), 'A': round(base, 2), 'E2_gain': round(base - e2, 2), 'T_gain': round(base - t, 2)}
for s, v in res.items(): print(s, v)
print('E2 better scenes', sum(v['E2_gain'] > 0 for v in res.values()), '/', len(res), ' T better scenes', sum(v['T_gain'] > 0 for v in res.values()))
Path('/home/jetson/jetformer_reports/real_accuracy_20261015/ideas_data/probe_scenes.json').write_text(json.dumps(res, indent=1))
