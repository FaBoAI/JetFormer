#!/usr/bin/env python3
"""案出しの小さな確かめ 4(CPU・保存済みの予測だけ・判定には使わない)。
重ならない 5 本の組(10/12 の 5 つの方策の種 0 / 種 1)と、5 つの方策 + シミュレーターの腕の組の平均を、規則の形(best・last)で見る。
出力: probe_pred_cache3.json"""
import json, os, sys
import numpy as np
JF = '/home/jetson/jetformer'
sys.path.insert(0, JF); sys.path.insert(0, JF + '/tools')
import rl_evaluate as ev  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
Z = JF + '/data/zoo/'
TH = {'best': 0.626, 'last': 1.322}
def f(name, s, ck):
    sfx = '' if ck == 'best' else '_last'
    if name in ('jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2'):
        p = Z + 'policy_zoo_20261012/pred/%s_s%d%s.npz' % (name, s, sfx)
    elif name == 'C':
        p = Z + 'sim_cotrain_20261013/pred/C_s%d%s.npz' % (s, sfx)
    else:
        p = Z + 'sim_diverse_train_20261015/pred/CDIV_s%d%s.npz' % (s, sfx)
    with np.load(p) as z:
        return z['pred'].astype(np.float64)
held = ev._heldout(); tab = ev.pred_table(); cols, _ = ev._val_arrays(tab)
clean = np.isin(tab['scene'], held['human_clean_scenes']) & (tab['part'] == 0) & cols['fv'] & ~cols['impact']
y = np.nan_to_num(cols['y'].astype(np.float64))
L = lambda p: float(np.abs(p - y).mean(axis=(1, 2))[clean].mean())
out = {}
ref = {ck: [L(f('jetformer_v0', s, ck)) for s in (0, 1)] for ck in ('best', 'last')}
sets = {'zoo5(v0,act,tcp,msfusion,dinov2)': ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2'],
        'zoo5+C+CDIV': ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2', 'C', 'CDIV'],
        'v0+msfusion': ['jetformer_v0', 'msfusion'], 'v0+dinov2': ['jetformer_v0', 'dinov2'],
        'v0+C+CDIV': ['jetformer_v0', 'C', 'CDIV']}
for k, names in sets.items():
    row = {}
    for ck in ('best', 'last'):
        vals = [L(np.mean([f(n, s, ck) for n in names], axis=0)) for s in (0, 1)]
        g = np.mean(ref[ck]) - np.mean(vals)
        row[ck] = {'vals': [round(v, 3) for v in vals], 'gain_vs_A_us': round(g, 3),
                   'all_below_A_min': bool(max(vals) < min(ref[ck])), 'gain_gt_threshold': bool(g > TH[ck])}
    out[k] = row
json.dump(out, open(os.path.join(HERE, 'probe_pred_cache3.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
