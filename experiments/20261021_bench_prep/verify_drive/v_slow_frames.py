#!/usr/bin/env python3
"""V-drive: 区間の速度上限のつなぎ(drive._ZoneCaps)の遅いフレームを探す(10/10 の記録・バッファに記録の時刻どおりに入れる)"""
import collections
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import drive  # noqa: E402

RAW = Path('/home/jetson/jetformer/data/raw')
log = sys.argv[1]
parts = collections.defaultdict(list)
for s in sorted(d for d in (RAW / log).iterdir() if d.is_dir() and d.name.startswith(log + '-')):
    for n in ('IMU', 'VEHICLE', 'AI_CMD'):
        z = np.load(s / (n + '.npz'))
        parts[n].append({k: z[k] for k in z.files if z[k].ndim >= 1})
cat = lambda n, k: np.concatenate([p[k] for p in parts[n]])  # noqa: E731
it, gz = cat('IMU', 'timestamps_us').astype(np.int64), cat('IMU', 'gyro')[:, 2]
vt, er = cat('VEHICLE', 'timestamps_us').astype(np.int64), cat('VEHICLE', 'erpm')
at = cat('AI_CMD', 'timestamps_us').astype(np.int64)
o = np.argsort(at, kind='stable')
at, pred, out, sel = at[o], cat('AI_CMD', 'pred')[o], cat('AI_CMD', 'out')[o], cat('AI_CMD', 'select')[o]
print('IMU monotonic:', bool(np.all(np.diff(it) >= 0)), 'n back:', int(np.sum(np.diff(it) < 0)),
      'max gap s:', float(np.max(np.diff(it)) / 1e6))
oi = np.argsort(it, kind='stable')
it, gz = it[oi], gz[oi]
zc = drive._ZoneCaps()
ib, vb = collections.deque(maxlen=60), collections.deque(maxlen=50)
ki = kv = 0
times = []
for i in range(len(at)):
    n0 = ki
    while ki < len(it) and it[ki] <= at[i] + 2000:
        ib.append((int(it[ki]) * 1000, np.array([0, 0, 0, 0, 0, np.deg2rad(float(gz[ki]))])))
        ki += 1
    while kv < len(vt) and vt[kv] <= at[i]:
        vb.append((int(vt[kv]) * 1000, {'erpm': int(er[kv])}))
        kv += 1
    veh = next((d for tv, d in reversed(vb) if tv <= at[i] * 1000), None)
    if veh is None:
        continue
    t0 = time.perf_counter()
    z = zc.frame(int(at[i]) * 1000, (float(out[i, 0]), float(out[i, 1])), veh, pred[i], bool(sel[i]), ib, vb)
    dt = (time.perf_counter() - t0) * 1000
    times.append((dt, i, ki - n0, len(zc.state.phase.h_hist)))
times.sort(reverse=True)
print('frames', len(times), 'mean ms %.3f' % np.mean([t[0] for t in times]))
for t in times[:8]:
    print('ms %.2f frame %d imu_fed %d h_hist %d' % t)
