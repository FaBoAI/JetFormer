#!/usr/bin/env python3
"""原因 A の診断 M3(numpy だけ・読むだけ): 保存した H1(10/10 の人・保留 4,670)と介入(10/10 の交代の後 1,570)の予測で、
スロットルの偏り(予測 − 正解。+ = 緩める)を速さの帯で分ける。介入の正解がどれだけ「緩める・止まる」かも数える。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I m3_h1_interv.py
"""
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
EVP = Path('/home/jetson/jetformer_reports/train_v2_20261025/eval/pred')
ZOO = Path('/home/jetson/jetformer_reports/policy_zoo_20261012')
OUT = Path(__file__).resolve().parent / 'out'
ROW2 = 2
BANDS = [(-9, 0.3, '<0.3'), (0.3, 1.0, '0.3-1.0'), (1.0, 2.0, '1-2'), (2.0, 3.0, '2-3'), (3.0, 99, '>=3')]


def main():
    b = np.load(BUNDLE, allow_pickle=False)
    y = b['y'].astype(np.float64)
    v = -b['erpm'].astype(np.float64) * 0.5e-3
    prim, interv = b['primary'].astype(bool), b['interv'].astype(bool)
    preds = {'B': b['pred_B_best'].astype(np.float64)}
    for f in sorted(EVP.glob('*.npz')):
        if f.stem.endswith('_last'):
            continue
        with np.load(f, allow_pickle=False) as z:
            preds[f.stem] = z['pred'].astype(np.float64)
    out = {'labels': {}, 'runs': {}}
    for name, m in (('H1', prim), ('介入', interv)):
        lab = {}
        for lo, hi, bn in BANDS:
            mm = m & (v >= lo) & (v < hi)
            if mm.any():
                lab[bn] = {'n': int(mm.sum()), 'thr_row2_mean': float(y[mm, ROW2, 1].mean()),
                           'brake_frac_row0(>1598)': float((y[mm, 0, 1] > 1598).mean())}
        out['labels'][name] = lab
    for r, p in preds.items():
        row = {}
        for name, m in (('H1', prim), ('介入', interv)):
            d = p[:, ROW2, 1] - y[:, ROW2, 1]
            row[name] = {'thr_bias_all': float(d[m].mean())}
            for lo, hi, bn in BANDS:
                mm = m & (v >= lo) & (v < hi)
                if mm.sum() >= 20:
                    row[name][bn] = [float(d[mm].mean()), float((p[mm, ROW2, 1] - preds['B'][mm, ROW2, 1]).mean()), int(mm.sum())]
        out['runs'][r] = row
    OUT.mkdir(exist_ok=True)
    (OUT / 'm3_h1_interv.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out['labels'], ensure_ascii=False))
    for r, row in out['runs'].items():
        print('%-34s H1 %6.2f | %s || 介入 %6.2f | %s' % (
            r[:34], row['H1']['thr_bias_all'],
            ' '.join('%s:%.1f/%.1f' % (k, x[0], x[1]) for k, x in row['H1'].items() if k != 'thr_bias_all'),
            row['介入']['thr_bias_all'],
            ' '.join('%s:%.1f/%.1f' % (k, x[0], x[1]) for k, x in row['介入'].items() if k != 'thr_bias_all')))


if __name__ == '__main__':
    sys.exit(main())
