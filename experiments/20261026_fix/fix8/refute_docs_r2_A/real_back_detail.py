"""打ち消しの票 A: real_back.py の 20261023-211828 の 2 回(127.13・146.22 秒)と 20261023-205458(23.45 秒)の、
リカバリーの間の assist・select・書いた値・eRPM・下がった距離を 0.1 秒ごとに出す(読むだけ)。
    python -I -B real_back_detail.py → 標準出力
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_back as rb  # noqa: E402

for lg, ts in (('20261023-211828', 127.131), ('20261023-211828', 146.217), ('20261023-205458', 23.453)):
    cat = rb.load(lg)
    z, v = cat['AI_CMD'], cat['VEHICLE']
    o = np.argsort(z['timestamps_us']); ov = np.argsort(v['timestamps_us'])
    ta = z['timestamps_us'][o].astype(np.int64); tv = v['timestamps_us'][ov].astype(np.int64)
    T0 = int(min(ta[0], tv[0]))
    a = (ta - T0) / 1e6; b = (tv - T0) / 1e6
    er = v['erpm'][ov].astype(float); vel = -er * rb.KV
    x = np.concatenate([[0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(b))])
    thr = z['out'][o][:, 1] if 'out' in z and z['out'].ndim == 2 else None
    x0 = float(np.interp(ts, b, x))
    print('==', lg, ts)
    for t in np.arange(ts - 0.2, ts + 2.0, 0.1):
        i = int(np.searchsorted(a, t)); i = min(i, len(a) - 1)
        print('  %7.2f assist %d sel %d thr %s erpm %6.0f back %.3f' % (
            t, z['assist'][o][i], z['select'][o][i], ('%.0f' % thr[i]) if thr is not None else '?',
            float(np.interp(t, b, er)), x0 - float(np.interp(t, b, x))))
