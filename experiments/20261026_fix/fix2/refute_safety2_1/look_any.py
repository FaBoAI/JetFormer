"""反証 safety2-1: 記録の行の範囲を並べる(引数: 記録のパス 始め 終わり)。読むだけ"""
import json, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); a0, a1 = int(sys.argv[2]), int(sys.argv[3])
z = np.load(D / 'AI_CMD.npz'); v = np.load(D / 'VEHICLE.npz'); im = np.load(D / 'IMU.npz')
t = z['timestamps_us'].astype(np.int64); mo = z['model_out']; out = z['out']; kind = z['assist']; sel = z['select']; lat = z['latency_ms']
vt = v['timestamps_us'].astype(np.int64); ve = v['erpm'].astype(float)
it = im['timestamps_us'].astype(np.int64); hg = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
for i in range(a0, min(a1, len(t))):
    jv = np.searchsorted(vt, t[i], side='right') - 1
    m = (it > t[i] - 150_000) & (it <= t[i] + lat[i] * 1000)
    print(f"{i:3d} {(t[i]-t[a0])/1e6:+.3f} sel={int(sel[i])} mo={mo[i,1]:7.1f} out={out[i,1]:6.0f} kind={int(kind[i])} erpm={ve[jv]:6.0f} g={hg[m].max() if m.any() else 0:4.2f}")
