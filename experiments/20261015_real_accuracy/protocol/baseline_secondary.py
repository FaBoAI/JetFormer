#!/usr/bin/env python3
"""手順を固める前の基準値(CPU だけ)。読むのは基準 A(v0 種 0・1)とモデル B の予測だけ(10/12 の npz。腕の予測は読まない)。
副指標の閾値 = 2 × s_A(s_A = {A 種 0, A 種 1, B} の範囲)を、主指標と同じ考え方で先に決めるために使う。

  cd ~/jetformer && CUDA_VISIBLE_DEVICES= jf-heavy --mem 1500M --lock none -- nice -n 10 python experiments/20261015_real_accuracy/protocol/baseline_secondary.py
"""
import json, os, sys
import numpy as np
REPO = '/home/jetson/jetformer'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/tools')
import rl_evaluate as ev  # noqa
import policy_zoo as pz  # noqa
OUT = '/home/jetson/jetformer_reports/real_accuracy_20261015/protocol_baseline_secondary.json'
BATTERY_SCENE = '20261010-193638-0003'
tab = ev.pred_table(); held = ev._heldout(); cols, f = ev._val_arrays(tab)
assert tab['key'] == held['pred_table_key'] == '3d339512828b17195d568f1b93ee4a28208f2e85'
part, scene = tab['part'], tab['scene']
yfin = np.isfinite(cols['y']).all(axis=(1, 2))
clean = np.isin(scene, held['human_clean_scenes']) & (part == 0) & cols['fv'] & ~cols['impact']
sets = {
    'primary': clean,
    'S1_no_battery_scene': clean & (scene != BATTERY_SCENE),
    'S3_intervention_part1': (part == 1) & cols['fv'] & yfin,
    'battery_scene_only': clean & (scene == BATTERY_SCENE),
    'genre4': clean & (cols['mode'] == 4),
    'genre5': clean & (cols['mode'] == 5),
}
y = np.nan_to_num(cols['y'].astype(np.float64))
names = {'A_s0': 'jetformer_v0_s0', 'A_s1': 'jetformer_v0_s1', 'B': 'B'}
out = {'pred_table_key': tab['key'], 'frames': {k: int(v.sum()) for k, v in sets.items()}, 'values': {}, 'thresholds_2xsA': {}}
for ck in ('best', 'last'):
    vals = {}
    for k, z in names.items():
        p = np.load(pz.pred_path(z, ck))['pred'].astype(np.float64)
        d = np.abs(p - y)
        row = {}
        for sn, m in sets.items():
            row[sn] = round(float(d.mean(axis=(1, 2))[m].mean()), 3)
        row['S2_k3_steer'] = round(float(d[clean][:, 2, 0].mean()), 3)
        row['k3_throttle'] = round(float(d[clean][:, 2, 1].mean()), 3)
        vals[k] = row
    out['values'][ck] = vals
    thr = {}
    for m in list(sets) + ['S2_k3_steer', 'k3_throttle']:
        v = [vals[k][m] for k in names]
        thr[m] = {'s_A': round(max(v) - min(v), 3), 'threshold': round(2 * (max(v) - min(v)), 3)}
    out['thresholds_2xsA'][ck] = thr
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
