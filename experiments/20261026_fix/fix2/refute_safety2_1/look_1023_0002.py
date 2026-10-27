"""反証 safety2-1: 10/23 20261023-205458-0002 の行 55〜135 を、記録の値だけで並べる(assist は通さない)。
eRPM は 100 Hz の VEHICLE を細かく、IMU は水平の G。読むだけ。装置は開かない"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/jetson/jetformer')
D = ROOT / 'data/raw/20261023-205458/20261023-205458-0002'
z = np.load(D / 'AI_CMD.npz'); v = np.load(D / 'VEHICLE.npz'); im = np.load(D / 'IMU.npz')
t = z['timestamps_us'].astype(np.int64); mo = z['model_out']; out = z['out']; kind = z['assist']; sel = z['select']; hold = z['hold']
lat = z['latency_ms']
vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float); pwm = v['pwm']
it = im['timestamps_us'].astype(np.int64); acc = im['accel'].astype(float)
hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
t0 = t[103]
print('VEHICLE dt median ms', np.median(np.diff(vt)) / 1e3, ' IMU dt ms', np.median(np.diff(it)) / 1e3)
for i in range(55, 136):
    jv = np.searchsorted(vt, t[i], side='right') - 1
    m = (it > t[i] - 150_000) & (it <= t[i] + lat[i] * 1000)
    print(f"{i:3d} {(t[i]-t0)/1e6:+.3f} sel={int(sel[i])} hold={int(hold[i])} mo_thr={mo[i,1]:7.1f} out={out[i,0]:6.0f},{out[i,1]:6.0f} kind={int(kind[i])} erpm={ve[jv]:6.0f} pwm_thr={pwm[jv,1]:5.0f} g={hg[m].max() if m.any() else 0:4.2f}")
