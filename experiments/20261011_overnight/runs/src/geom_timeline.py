"""Sensor-geometry timeline over all 2026-10-10 logs:
 - camera: rows of the car's own blue bumper at the bottom of the 224x224 image (bumper.py method, 150 frames/log)
 - ToF: floor distance (rows 9 / 18, centre columns) vs data/tof_floor_ref.npy on straight fast driving
Writes geom_timeline.json and fig/sensor_geometry_timeline.png"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bumper import bumper_rows  # noqa
RAW = '/home/jetson/jetformer/data/raw/'
REF = np.load('/home/jetson/jetformer/data/tof_floor_ref.npy').astype(float)
KV = 0.5e-3
logs = sorted(x for x in os.listdir(RAW) if x.startswith('20261010-') and len(x) == 15)
out = {}
for log in logs:
    root = RAW + log
    sc = sorted(s for s in os.listdir(root) if s.startswith(log + '-'))
    files, tt, dep, vt, er, it, gz = [], [], [], [], [], [], []
    for s in sc:
        p = os.path.join(root, s)
        try:
            if os.path.exists(p + '/CAM_FRONT.npz'):
                files += [os.path.join(p, str(x)) for x in np.load(p + '/CAM_FRONT.npz', allow_pickle=True)['files']]
            if os.path.exists(p + '/TOF_FRONT.npz'):
                d = np.load(p + '/TOF_FRONT.npz'); tt.append(d['timestamps_us']); dep.append(d['depth'][:, [9, 18], 20:35])
            d = np.load(p + '/VEHICLE.npz'); vt.append(d['timestamps_us']); er.append(d['erpm'])
            d = np.load(p + '/IMU.npz'); it.append(d['timestamps_us']); gz.append(d['gyro'][:, 2])
        except Exception as e:
            print('skip', s, e)
    if not files:
        continue
    idx = np.linspace(0, len(files) - 1, 150).astype(int)
    rows = np.array([bumper_rows(files[i]) for i in idx], float)
    meta = json.load(open(root + '/log.json'))
    r = {'desc': meta.get('description') or '', 'bumper_rows_median': float(np.nanmedian(rows)),
         'bumper_rows_iqr': [float(np.nanpercentile(rows, 25)), float(np.nanpercentile(rows, 75))]}
    if tt:
        tt = np.concatenate(tt).astype(np.int64); dep = np.concatenate(dep)
        vt = np.concatenate(vt).astype(np.int64); er = np.concatenate(er).astype(float)
        it = np.concatenate(it).astype(np.int64); gz = np.concatenate(gz).astype(float)
        e = er[np.clip(np.searchsorted(vt, tt) - 1, 0, len(vt) - 1)]
        g = np.interp(tt, it, gz)
        m = (e < -1500) & (np.abs(g) < 40)
        dd = (dep & 0x1FFF).astype(float); conf = dep >> 13
        dd[(conf != 4) | (dd == 0x1FFF)] = np.nan
        for k, row in enumerate((9, 18)):
            if m.sum() > 30:
                ratio = np.nanmedian(dd[m, k, :], axis=1) / np.median(REF[row, 20:35])
                r['tof_row%d_pct' % row] = round(float((np.nanmedian(ratio) - 1) * 100), 1)
        r['tof_frames'] = int(m.sum())
    out[log] = r
    print(log, r)
S = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(S + '/geom_timeline.json', 'w'), indent=1, ensure_ascii=False)
