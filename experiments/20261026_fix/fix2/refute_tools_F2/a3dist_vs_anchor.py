#!/usr/bin/env python3
"""指摘 tools:F2 の確かめ(読むだけ): A4 がある発進で、A4 を隠して A3+dist で出し直し、起点の方法との差を出す"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import standing_3lap as S  # noqa: E402

orig = S._line_ends
MODE = {'hide': False}

def patched(vt, Df, A_, d_pre, t_line0, t1, Lrun):
    if MODE['hide'] and len(A_) == 4:
        return orig(vt, Df, A_[:3], d_pre, t_line0, t1, Lrun)
    return orig(vt, Df, A_, d_pre, t_line0, t1, Lrun)

S._line_ends = patched
names = S.candidate_names()
rows = []
for day in ('20261023', '20261024'):
    logs, _ = S.day_logs(day)
    for lg in logs:
        MODE['hide'] = False
        try:
            r0 = S.analyze_log(lg, names=names)
        except Exception as e:  # noqa
            print('skip', lg, e); continue
        if 'error' in r0:
            continue
        MODE['hide'] = True
        r1 = S.analyze_log(lg, names=names)
        for s0, s1 in zip(r0['standing'], r1['standing']):
            if s0.get('method') != 'anchor' or s0.get('from_move_s') is None:
                continue
            L = r0['lap_dist_median_m']
            dA = s0.get('dist_anchor_m') or []
            rows.append((day, r0.get('ai_no'), lg, s0['t_ai'], s0['valid'], s0['from_move_s'], s1.get('from_move_s'),
                         s1.get('method'), s1.get('valid'), (s1['from_move_s'] - s0['from_move_s']) if s1.get('from_move_s') is not None else None,
                         L, dA[-1] if dA else None, s0.get('note')))
print('| 日 | No. | 記録 | t_ai | valid(元) | 起点の方法 | A3+dist(A4 を隠す) | method | valid | 差 | 中央値 m | A3→A4 m | note |')
for x in rows:
    print('| %s | %s | %s | %.2f | %s | %.3f | %s | %s | %s | %s | %s | %s | %s |' % (
        x[0], x[1], x[2], x[3], x[4], x[5], '—' if x[6] is None else '%.3f' % x[6], x[7], x[8],
        '—' if x[9] is None else '%+.3f' % x[9], x[10], x[11], x[12]))
d = [x[9] for x in rows if x[9] is not None]
dv = [x[9] for x in rows if x[9] is not None and x[4]]
print('n=%d median=%+.3f min=%+.3f max=%+.3f' % (len(d), np.median(d), min(d), max(d)))
if dv:
    print('valid only: n=%d median=%+.3f min=%+.3f max=%+.3f sorted=%s' % (len(dv), np.median(dv), min(dv), max(dv), sorted(round(v, 3) for v in dv)))
