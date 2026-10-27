#!/usr/bin/env python3
"""指摘 R2 の確かめ(読むだけ): A4 がある発進で、「3 周目の線の直後に人側にして止めた」形を作り直す。
A4 を隠し、1 周の距離の中央値はその本の A1→A2・A2→A3 だけ(3 周だけの本ならそうなる)、AI の区間の終わりを
本当の 3 周目の線 + delay 秒にして、A3+dist の値と起点の方法の値の差を出す。比べにログ全体の中央値も。"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import standing_3lap as S  # noqa: E402

orig = S._line_ends
REC = []
DELAYS = (0.3, 0.6, 1.0, 1.5)

def patched(vt, Df, A_, d_pre, t_line0, t1, Lrun):
    e0, m0 = orig(vt, Df, A_, d_pre, t_line0, t1, Lrun)
    if len(A_) == 4 and len(e0) == 3 and m0[2] == 'anchor':
        D = lambda x: float(np.interp(x, vt, Df))  # noqa: E731
        L12, L23, L34 = D(A_[1]) - D(A_[0]), D(A_[2]) - D(A_[1]), D(A_[3]) - D(A_[2])
        L2 = 0.5 * (L12 + L23)
        r = {'A': A_, 'e3': e0[2], 'L12': L12, 'L23': L23, 'L34': L34, 'Llog': Lrun, 'L2': L2, 'own': {}, 'log': {}}
        for dl in DELAYS:
            for key, LL in (('own', L2), ('log', Lrun)):
                e1, m1 = orig(vt, Df, A_[:3], d_pre, t_line0, e0[2] + dl, LL)
                r[key][dl] = (e1[2] - e0[2]) if len(e1) == 3 else None
        REC.append(r)
    return e0, m0

S._line_ends = patched
names = S.candidate_names()
rows = []
for day in ('20261023', '20261024'):
    for lg in S.day_logs(day)[0]:
        REC.clear()
        try:
            r = S.analyze_log(lg, names=names)
        except Exception as e:  # noqa
            continue
        if 'error' in r:
            continue
        byend = {round(x['e3'], 3): x for x in REC}
        for s in r['standing']:
            if s.get('method') != 'anchor' or s.get('end_s') is None:
                continue
            x = byend.get(s['end_s'])
            if x is None:
                continue
            rows.append((day, r.get('ai_no'), lg, s['valid'], s['from_move_s'], s.get('note'), x))

def f(v):
    return '—(数えない)' if v is None else '%+.2f' % v

print('| 日 | No. | valid | 起点の方法 | L1・L2・L3 m | 本の中央値 m | ' + ' | '.join('本の中央値・線+%.1f 秒で人側' % d for d in DELAYS) + ' | ログの中央値・線+0.6 |')
print('|' + '---|' * (7 + len(DELAYS)))
for day, no, lg, ok, fm, note, x in rows:
    print('| %s | %s | %s | %.2f | %.1f・%.1f・%.1f | %.2f | %s | %s |%s' % (
        day[4:], no, ok, fm, x['L12'], x['L23'], x['L34'], x['L2'], ' | '.join(f(x['own'][d]) for d in DELAYS),
        f(x['log'][0.6]), '  note=%s' % note if note else ''))
for d in DELAYS:
    v = [x['own'][d] for _, _, _, ok, _, note, x in rows if ok and not (note and '確かめる' in note)]
    c = [a for a in v if a is not None]
    print('delay %.1f: n=%d 数えた %d 数えない %d / 差 min %+.2f max %+.2f 絶対値の中央値 %.2f 短く出た %d' % (
        d, len(v), len(c), len(v) - len(c), min(c), max(c), float(np.median(np.abs(c))), sum(1 for a in c if a < 0)))
