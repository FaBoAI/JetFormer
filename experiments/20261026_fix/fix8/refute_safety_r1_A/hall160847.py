"""票 A: 20261024-160847 の 40 秒あたりの後ろへの動きの量(読むだけ)。hall_count の進み・eRPM の積分(KV 0.5e-3 m/s/eRPM)・
1 本全体で求めた「1 カウントあたりの m」(|Δhall| の和 / ∫|eRPM|·KV dt)。→ 標準出力"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rec_io  # noqa: E402

KV = 0.5e-3
cat, L, t0 = rec_io.load_log('20261024-160847')
V, I = cat['VEHICLE'], cat['IMU']
t = (V['timestamps_us'] - t0) / 1e6
h = V['hall_count'].astype(np.int64)
e = V['erpm'].astype(float)
# 1 本全体: 隣どうしの間が 0.05 秒以内のところだけ
dt = np.diff(t)
ok = dt < 0.05
dist = np.sum(np.abs(e[1:][ok]) * KV * dt[ok])
cnt = np.sum(np.abs(np.diff(h))[ok])
mpc = dist / cnt
print('whole log: |dhall| %d counts, int|erpm|KV %.2f m -> %.4f m/count' % (cnt, dist, mpc))
for a, b in ((40.03, 40.46), (40.46, 41.21), (40.03, 41.21)):
    s = (t >= a) & (t <= b)
    dh = h[s][-1] - h[s][0]
    steps = np.diff(h[s])
    sd = (t[s][1:] - t[s][:-1])
    x = float(np.sum(e[s][1:] * KV * sd))
    print('%.2f-%.2f s: hall %d -> %d (%+d; steps +%d / -%d), int erpm*KV %.3f m (back), hall*mpc %.3f m, erpm range %d..%d' % (
        a, b, h[s][0], h[s][-1], dh, int(np.sum(steps > 0)), int(np.sum(steps < 0)), x, dh * mpc, e[s].min(), e[s].max()))
ti = (I['timestamps_us'] - t0) / 1e6
s = (ti >= 40.0) & (ti <= 40.2)
ay = I['accel'][s][:, 1]
print('IMU ay 40.0-40.2: min %.2f m/s2 (%.2f G) at %.3f s' % (ay.min(), ay.min() / 9.80665, ti[s][np.argmin(ay)]))
