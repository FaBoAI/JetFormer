"""反証 w1-safety-r61-safety-1 の打ち消し: 車両の値の止まり(人側・AI 側どちらも)を日ごと・10/10 の直し(16:10 と 18:39)の前後で数える(読むだけ)。"""
from pathlib import Path
from collections import defaultdict
import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')
tot = defaultdict(float)
gaps = defaultdict(list)
for ep in sorted(RAW.glob('2026*/2026*-*')):
    try:
        z = np.load(ep / 'VEHICLE.npz')
        t = z['timestamps_us'].astype(np.int64)
    except Exception:
        continue
    day = ep.name[:8]
    hm = ep.name[9:15]
    key = day if day != '20261010' else ('20261010<16:10' if hm < '161024' else ('20261010<18:39' if hm < '183920' else '20261010>=18:39'))
    dt = np.diff(t) / 1e6
    ok = (dt > 0) & (dt < 5.0)
    tot[key] += float(dt[ok].sum())
    for i in np.where(dt > 0.15)[0]:
        if dt[i] < 5.0:
            gaps[key].append(round(float(dt[i]), 3))
for k in sorted(tot):
    g = sorted(gaps[k])
    print(k, 'all_s=%.0f' % tot[k], 'n>0.15=%d' % len(g), 'n0.40-0.70=%d' % sum(0.40 <= x <= 0.70 for x in g), g[:40])
