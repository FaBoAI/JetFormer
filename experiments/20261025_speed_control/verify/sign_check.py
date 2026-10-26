"""反証 V-drive: 左旋回の決まりの符号(舵 + = 左・gz + = 左)を 10/24 の記録で確かめる(読むだけ)"""
import glob, json, os, sys
import numpy as np
RAW = '/home/jetson/jetformer/data/raw'
for log in sys.argv[1:]:
    A, I, V = [], [], []
    for d in sorted(glob.glob(os.path.join(RAW, log, log + '-*'))):
        try:
            a = np.load(d + '/AI_CMD.npz'); i = np.load(d + '/IMU.npz'); v = np.load(d + '/VEHICLE.npz')
            A.append((a['timestamps_us'], a['out'], a['select'], a['hold'], a['dry_run']))
            I.append((i['timestamps_us'], i['gyro'], i['accel'])); V.append((v['timestamps_us'], v['erpm']))
        except Exception as e:
            pass
    at = np.concatenate([x[0] for x in A]); out = np.concatenate([x[1] for x in A]); sel = np.concatenate([x[2] for x in A])
    hold = np.concatenate([x[3] for x in A]); dry = np.concatenate([x[4] for x in A])
    it = np.concatenate([x[0] for x in I]); gy = np.concatenate([x[1] for x in I]); acc = np.concatenate([x[2] for x in I])
    vt = np.concatenate([x[0] for x in V]); er = np.concatenate([x[1] for x in V])
    o = np.argsort(it); it, gy, acc = it[o], gy[o], acc[o]
    o = np.argsort(vt); vt, er = vt[o], er[o]
    meta = json.load(open(os.path.join(RAW, log, 'log.json')))
    c = float(meta['drive'].get('steer_center_us') or 1441); trim = float(meta['drive']['params'].get('steer_trim_us', 0))
    gz = np.interp(at + 100000, it, gy[:, 2])   # 0.1 秒後のヨー
    v = -np.interp(at, vt, er) * 0.5e-3
    m = sel & ~hold & ~dry & (v > 1.5)
    s = out[m, 0] - c - trim
    r = np.corrcoef(s, gz[m])[0, 1]
    left = s >= 100
    print('%s: n=%d corr(steer_rel, gz+0.1s)=%.2f  steer>=+100: gz median %.0f deg/s  steer<=-100: gz median %.0f  '
          'az median %.2f m/s2  ax %.2f ay %.2f' % (log, m.sum(), r, np.median(gz[m][left]) if left.any() else np.nan,
          np.median(gz[m][s <= -100]) if (s <= -100).any() else np.nan, np.median(acc[:, 2]), np.median(acc[:, 0]), np.median(acc[:, 1])))
