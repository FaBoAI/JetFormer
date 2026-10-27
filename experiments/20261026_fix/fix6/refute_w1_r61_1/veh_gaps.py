"""反証 w1-safety-r61-safety-1 の打ち消し: 記録の車両の値(VEHICLE.npz)の止まりの長さを日ごとに数える(読むだけ)。
AI 側(select True)の前後で、隣り合うサンプルの時刻の間が 0.15 秒を超えたものを数え、0.45〜0.65 秒のものを数える。
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')
res = defaultdict(lambda: {'ai_s': 0.0, 'gaps': []})
for ep in sorted(RAW.glob('2026*/2026*-*')):
    f = ep / 'VEHICLE.npz'
    if not f.exists():
        continue
    try:
        z = np.load(f)
        t = z["timestamps_us"]
    except Exception:
        continue
    t = z['timestamps_us'].astype(np.int64)
    sel = z['select'].astype(bool)
    erpm = z['erpm']
    if len(t) < 2:
        continue
    day = ep.name[:8]
    dt = np.diff(t) / 1e6
    ai = sel[:-1] & sel[1:]
    ok = (dt > 0) & (dt < 1.0)
    res[day]['ai_s'] += float(dt[ai & ok].sum())
    for i in np.where(ai & (dt > 0.15))[0]:
        res[day]['gaps'].append((round(float(dt[i]), 3), ep.name, int(erpm[i]), int(erpm[i + 1])))
for day, r in sorted(res.items()):
    g = [x[0] for x in r['gaps']]
    print(day, 'ai_s=%.0f' % r['ai_s'], 'gaps>0.15:', len(g), '0.45-0.65:', sum(0.45 <= x <= 0.65 for x in g),
          '>0.5:', sum(x > 0.5 for x in g))
    print('   ', sorted(g))
    for x in r['gaps']:
        if 0.4 <= x[0] <= 0.7:
            print('    ', x)
print('--- all gaps with run (AI side)')
for day, r in sorted(res.items()):
    for x in r['gaps']:
        print(day, x)
# AI 秒を 10/10 16:10(I2C の直し 99df0d2)の前と後で分ける
pre = post = 0.0
for ep in sorted(RAW.glob('20261010*/2026*-*')):
    f = ep / 'VEHICLE.npz'
    try:
        z = np.load(f); t = z['timestamps_us'].astype(np.int64); sel = z['select'].astype(bool)
    except Exception:
        continue
    dt = np.diff(t) / 1e6; ai = sel[:-1] & sel[1:]; ok = (dt > 0) & (dt < 1.0)
    s = float(dt[ai & ok].sum())
    if ep.name[9:15] < '161024':
        pre += s
    else:
        post += s
print('10/10 AI s before 16:10 fix: %.0f  after: %.0f' % (pre, post))
