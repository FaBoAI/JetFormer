import json
from pathlib import Path
import numpy as np
R = Path('/home/jetson/jetformer'); DS = R / 'data/datasets/dataset_v2_20261025'; T3 = R / 'data/derived/dataset_v3_20261026'
splits = json.loads((DS / 'splits.json').read_text())
def table(p):
    z = np.load(p); sc, fr, v = z['scene'].astype(str), z['frame'].astype(np.int64), z['drop']; t = {}
    for s in np.unique(sc):
        m = sc == s; t[s] = (fr[m], v[m])
    return t
def impact(t_us, imu):
    t = t_us.astype(np.float64) / 1e6; hg = np.hypot(imu[..., 0], imu[..., 1]).max(axis=1) / 9.80665
    m = np.zeros(len(t), bool)
    for ti in t[hg >= 2.5]: m |= (t >= ti - 1.0) & (t <= ti + 2.0)
    return m
out = {}
for name in ('drop_v3n.npz', 'drop_v3d.npz', 'drop_v3b.npz', '../dataset_v2_20261025/drop_v2h.npz'):
    dt = table(T3 / name); rows_s, rows_all = [], []
    for s in splits['train']:
        a = np.load(DS / 'aligned' / (s + '.npz'))
        if 'vehicle' not in a.files or 'tof' not in a.files or 'imu' not in a.files: continue
        n = len(a['future_valid']); ok = a['future_valid'] & a['vehicle_valid']
        ok &= (a['mode'] if 'mode' in a.files else np.zeros(n, np.int16)) != 0
        d = np.zeros(n, bool); fr, v = dt[s]; d[fr] = v.astype(bool); ok &= ~d
        ok &= ~impact(a['timestamps_us'], a['imu'].astype(np.float32)[:, :, :2]); ok &= (np.arange(n) % 3) == 0
        ev = a['erpm_valid'] if 'erpm_valid' in a.files else np.ones(n, bool)
        stand = a['vehicle_valid'] & ev & (np.abs(a['vehicle'][:, 3]) < 1)
        rows_s.append(a['vehicle'][stand & ok]); rows_all.append(a['vehicle'][stand])
    rs = np.concatenate(rows_s); ra = np.concatenate(rows_all)
    out[name] = {'from_samples': [float(np.median(rs[:, 0])), float(np.median(rs[:, 1])), int(len(rs))],
                 'scene_all': [float(np.median(ra[:, 0])), float(np.median(ra[:, 1])), int(len(ra))]}
print(json.dumps(out, ensure_ascii=False))
