#!/usr/bin/env python3
"""原因 A の診断 D3(numpy だけ・読むだけ): 直し方の案ごとに、学習のサンプルの正解の形(数・直線の前進の量・
動いているのにブレーキ側・低速のブレーキ側・止まった割合)を出す。書くのは out/d3_variants.json だけ。"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'out'
TK_SUBS = ('交代の直後3秒', '走り+交代の直後')


def main():
    D = np.load(OUT / 'd1_samples.npz')
    new = D['group'] != 'B41(10/10 人)'
    tk = new & np.isin(D['sub'], TK_SUBS)
    run_ = new & (D['sub'] == '走りの区間')
    n2324 = new & np.isin(D['sess'], ['10/23', '10/24-2'])
    v, t0, t2, st = D['v'], D['thr0'], D['thr2'], D['straight']
    stop = v < 0.3
    low_brake = (v < 1.0) & (t0 > 1598)
    V = {'B(41 シーン)': ~new, 'v2-H のまま': np.ones(len(v), bool), '交代の直後を外す': ~tk,
         '交代の直後を外す + 新しい人 10/23・24 の走りを外す': ~tk & ~(run_ & n2324),
         '0.3 m/s 未満を外す(だけ)': ~stop, '交代の直後を外す + 0.3 m/s 未満を外す': ~tk & ~stop,
         '交代の直後を外す + 低速のブレーキの正解を外す': ~tk & ~low_brake}
    out = {}
    for k, m in V.items():
        mv = m & (v >= 1.0)
        ms = m & st & (v >= 1.0)
        lo = m & (v < 1.0)
        out[k] = {'n': int(m.sum()), 'new_human': int((m & new).sum()),
                  'fwd_straight_us': float(np.mean(1496 - t2[ms])), 'fwd_moving_us': float(np.mean(1496 - t2[mv])),
                  'brake_frac_moving': float(np.mean(t0[mv] > 1598)), 'brake_frac_v<1': float(np.mean(t0[lo] > 1598)) if lo.any() else None,
                  'stop<0.3_frac': float(np.mean(stop[m])), 'n_v<1': int(lo.sum())}
    (OUT / 'd3_variants.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, r in out.items():
        print('%-44s %s' % (k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}))


if __name__ == '__main__':
    sys.exit(main())
