"""指摘 tools:F9 の確かめ(読むだけ。2026-10-26 fix2): collect.py の collisions_first_lap(発進 → A2)と、
standing_3lap の 1 周目(発進 → A2 の d_pre 手前 = 線)の衝突の違いを、10/23・10/24 の記録で数える。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/diag_launch')
import collect as C  # noqa: E402
le = C.le

days = tuple(sys.argv[1:]) or ('20261023', '20261024')
blocks, names = C.bench_blocks()
tot = {'launch_A2': 0, 'diff_launches': 0, 'diff_hits': 0}
diff_loc = {}
for log in C.logs(days):
    try:
        d = C.load(log, True, blocks, names)
    except Exception as e:  # noqa: BLE001
        print('skip', log, e)
        continue
    if not d['z']['ai'].any():
        continue
    rows = C.launch_rows(d, [], 'none')
    vt, Df = d['vt'], d['Df']
    for r in rows:
        if r.get('to_r1_s') is None or not r.get('first_lap_full'):
            continue
        t0 = r['t_ai']
        a1 = t0 + r['to_r1_s']
        a2 = r['first_lap_end_s']
        # 線 = 動いた場所(standing_3lap の line_ahead 0)。d_pre = 線 → A1
        mv = np.nonzero((vt >= t0) & (-d['erpm'] >= 400))[0]
        tm = float(vt[mv[0]])
        D = lambda x: float(np.interp(x, vt, Df))  # noqa: E731
        d_pre = D(a1) - D(tm)
        m = (vt >= a1) & (vt <= a2)
        tl = float(np.interp(D(a2) - d_pre, Df[m], vt[m]))
        fl = r['collisions_first_lap']
        inl = [h for h in fl if h[0] < tl]
        extra = [h for h in fl if h[0] >= tl]
        tot['launch_A2'] += 1
        if extra:
            tot['diff_launches'] += 1
            tot['diff_hits'] += len(extra)
            for h in extra:
                diff_loc[h[2]] = diff_loc.get(h[2], 0) + 1
        print('%s No.%-3s t_ai %7.2f d_pre %.2f 線 %7.2f A2 %7.2f  線まで %d 本 %s | 線〜A2 %d 本 %s' % (
            log, r['no'], t0, d_pre, tl, a2, len(inl), [(round(h[0] - t0, 2), round(h[1], 1), h[2]) for h in inl],
            len(extra), [(round(h[0] - t0, 2), round(h[1], 1), h[2]) for h in extra]))
print(tot, diff_loc)
