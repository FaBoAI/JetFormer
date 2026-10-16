#!/usr/bin/env python3
"""見直し: GAIN の g_t は「衝突の前後」(1 秒前〜2 秒後)のフレームを除いて計算する(protocol の式)。
「1 秒前」を除くには、これから起きる衝突を知っている必要がある(走行中には分からない = 因果でない)。
衝突の「後」だけを除く因果の形(前 0 秒・後 2 秒)にしたとき、GAIN の数字がどれだけ変わるかを確かめる(CPU だけ)。

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \\
        python experiments/20261015_real_accuracy/review/gain_causal_check.py
"""
import json
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / 'recount.py'), run_name='recount_lib')     # recount.py の表・正解・予測を作り直す
OUT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/review/gain_causal_check.json')
np_ = np
part, scene, frame, N = R['part'], R['scene'], R['frame'], R['N']
mode, Y, sets, primary = R['mode'], R['Y'], R['sets'], R['primary']
G_REF, NEUTRAL = R['G_REF'], R['NEUTRAL']


def g_series_causal(a):
    thr = a['vehicle'][:, 1].astype(np.float64)
    fwd = -a['vehicle'][:, 3].astype(np.float64)
    imp_after = R['my_impact'](a['timestamps_us'], a['imu'], 2.5, 0.0, 2.0)   # 衝突の後 2 秒だけ(前は見ない)
    ok = a['vehicle_valid'] & (fwd > 2000.0) & ~imp_after
    g = np.full(len(thr), np.nan)
    g[ok] = (NEUTRAL - thr[ok]) / fwd[ok] * 1000.0
    return g


gt = np.full(N, np.nan)
for (p, s), a in R['scene_cache'].items():
    idx = np.nonzero((part == p) & (scene == s))[0]
    gt[idx] = R['running_median'](g_series_causal(a), a['timestamps_us'], 5.0)[frame[idx]]
ref = np.array([G_REF.get(int(m), np.nan) for m in mode])
sc = np.clip(gt / ref, 0.8, 2.0)
sc = np.where(np.isfinite(sc), sc, 1.0)
out = {'scale_diff_vs_protocol': {'max_abs': float(np.abs(sc - R['SC5']).max()),
                                  'frames_changed_primary': int((np.abs(sc - R['SC5']) > 1e-9)[primary].sum())}}
for ck in ('best', 'last'):
    for pair in (('A_s0', 'A_s1'), ('v0-s2', 'v0-s3')):
        for n in pair:
            p0 = R['V'][ck][n]
            q = R['apply_gain'](p0, sc)
            d0 = np.abs(p0 - Y).mean(axis=(1, 2))
            d1 = np.abs(q - Y).mean(axis=(1, 2))
            d2 = np.abs(R['V'][ck]['GAIN(%s)' % n] - Y).mean(axis=(1, 2))
            out['%s/%s' % (ck, n)] = {m: {'raw': round(float(d0[sets[m]].mean()), 4),
                                          'gain_protocol': round(float(d2[sets[m]].mean()), 4),
                                          'gain_causal': round(float(d1[sets[m]].mean()), 4)}
                                      for m in ('primary', 'S1_no_battery_scene', 'S3_intervention_part1')}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
