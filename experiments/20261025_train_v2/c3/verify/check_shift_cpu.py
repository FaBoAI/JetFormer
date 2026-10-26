#!/usr/bin/env python3
"""verify_c3(CPU): 自分のずれ(repredict_c3_gpu.my_shift)を、C3 の desk_v2.perturb・tools/policy_zoo.perturb と画素で照らす。
H1 の 4,670 フレームから 300 枚(等間隔)。上 1/3 は deviations.md 7 の式(行 r に 1 + (f − 1)·w(r))を別に画素の式で書いて照らす。
torch を読まない。書くのは c3/verify/check_shift_cpu.json だけ。"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import cv2
REPO = Path('/home/jetson/jetformer'); sys.path.insert(0, str(REPO))
HERE = REPO / 'experiments/20261025_train_v2/c3/verify'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mine = load('mine_gpu', HERE / 'repredict_c3_gpu.py')
c3 = load('c3_desk_v2', REPO / 'experiments/20261025_train_v2/c3/desk_v2.py')
pz = load('pz', REPO / 'tools/policy_zoo.py')
z = np.load('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz', allow_pickle=False)
rows = np.nonzero(z['primary'])[0]
pick = rows[np.linspace(0, len(rows) - 1, 300).astype(int)]
CLEAN = REPO / 'data/datasets/clean_driving_20261011'
res = {k: {'max_abs_vs_c3': 0, 'max_abs_vs_policy_zoo': None, 'n_px_diff_vs_c3': 0} for k in mine.KINDS}
res_formula = {'top_0.85': 0, 'top_1.15': 0}
rows_changed = {}
cache = {}
for i in pick:
    s, f = str(z['scene'][i]), int(z['frame'][i])
    if s not in cache:
        a = np.load(CLEAN / 'aligned' / (s + '.npz'), allow_pickle=False); cache[s] = (a['cam_files'], a['tof'])
    cf, tf = cache[s]
    img = cv2.imread(str(CLEAN / str(cf[f]))); tof = tf[f]
    assert img.shape == (224, 224, 3)
    for k in mine.KINDS:
        a_img, a_tof = mine.my_shift(k, img.copy(), tof.copy())
        b_img, b_tof = c3.perturb(k, img.copy(), tof.copy())
        d = max(int(np.abs(a_img.astype(int) - b_img.astype(int)).max()), int(np.abs(a_tof.astype(np.int64) - b_tof.astype(np.int64)).max()))
        res[k]['max_abs_vs_c3'] = max(res[k]['max_abs_vs_c3'], d)
        res[k]['n_px_diff_vs_c3'] += int((a_img != b_img).sum())
        if not k.startswith('top_'):
            p_img, p_tof = pz.perturb(k, img.copy(), tof.copy())
            dp = max(int(np.abs(a_img.astype(int) - p_img.astype(int)).max()), int(np.abs(a_tof.astype(np.int64) - p_tof.astype(np.int64)).max()))
            res[k]['max_abs_vs_policy_zoo'] = max(res[k]['max_abs_vs_policy_zoo'] or 0, dp)
        else:
            fct = float(k.split('_')[1])
            # deviations 7 を画素ごとに書いた式(ループ)
            ref = img.astype(np.float64).copy()
            for r in range(224):
                w = 1.0 - r / (224 / 3.0)
                if w > 0:
                    ref[r] = ref[r] * (1.0 + (fct - 1.0) * w)
            ref = np.clip(np.rint(ref), 0, 255).astype(np.uint8)
            res_formula[k] = max(res_formula[k], int(np.abs(ref.astype(int) - a_img.astype(int)).max()))
            ch = np.nonzero((a_img != img).any(axis=(1, 2)))[0]
            if len(ch):
                rows_changed[k] = [int(min(rows_changed.get(k, [999, 0])[0], ch.min())), int(max(rows_changed.get(k, [999, 0])[1], ch.max()))]
out = {'frames': len(pick), 'per_shift': res, 'top_vs_formula_max_abs': res_formula, 'top_rows_changed_range': rows_changed}
json.dump(out, open(HERE / 'check_shift_cpu.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
