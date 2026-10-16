#!/usr/bin/env python3
"""手順を固める前の数え上げ(CPU だけ)。モデルの予測は読まない(誤差は見ない)。
保留の表(10,985 フレーム)のうち、主指標の 4,670 フレーム以外の部分集合の大きさを数える。"""
import json, os, sys
import numpy as np
REPO = '/home/jetson/jetformer'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/tools')
import rl_evaluate as ev  # noqa
tab = ev.pred_table(); held = ev._heldout(); cols, f = ev._val_arrays(tab)
assert tab['key'] == held['pred_table_key']
part, scene = tab['part'], tab['scene']
m0 = part == 0
human_scene = np.isin(scene, held['human_clean_scenes'])
clean = human_scene & m0 & cols['fv'] & ~cols['impact']
yfin = np.isfinite(cols['y']).all(axis=(1, 2))
out = {'pred_table_key': tab['key'], 'rows': int(len(part)), 'primary_clean': int(clean.sum())}
ai_scene = m0 & ~human_scene
out['ai_val_scenes'] = sorted(set(scene[ai_scene].tolist()))
for src in (-1, 0, 1, 2):
    out['ai_scene_rows_source_%d' % src] = int((ai_scene & (cols['source'] == src)).sum())
h2 = ai_scene & (cols['source'] == 0) & cols['fv'] & ~cols['impact'] & yfin
out['H2_ai_scene_human_fv_noimpact'] = int(h2.sum())
out['H2_per_scene'] = {s: int((h2 & (scene == s)).sum()) for s in out['ai_val_scenes']}
out['H2_mode_counts'] = {int(k): int((h2 & (cols['mode'] == k)).sum()) for k in np.unique(cols['mode'][h2])} if h2.any() else {}
iv = (part == 1) & cols['fv'] & yfin
out['part1_scored'] = int(iv.sum())
out['part1_scenes'] = sorted(set(scene[part == 1].tolist()))
# battery scene rows
bat = clean & (scene == '20261010-193638-0003')
out['primary_battery_scene_rows'] = int(bat.sum())
out['primary_without_battery_scene'] = int((clean & ~(scene == '20261010-193638-0003')).sum())
print(json.dumps(out, ensure_ascii=False, indent=1))
s0 = ai_scene & (cols['source'] == 0)
print('src0', int(s0.sum()), 'fv', int((s0 & cols['fv']).sum()), 'noimp', int((s0 & ~cols['impact']).sum()), 'yfin', int((s0 & yfin).sum()))
s1 = ai_scene & (cols['source'] == 1)
print('src1', int(s1.sum()), 'fv', int((s1 & cols['fv']).sum()), 'noimp', int((s1 & ~cols['impact']).sum()), 'yfin', int((s1 & yfin).sum()))
