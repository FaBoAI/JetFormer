"""読むだけ: 記録 LOG の秒 A〜B の AI_CMD(out・model_out・assist・select・hold)と VEHICLE の eRPM を 1 行ずつ。
    python -I -B look_rec.py LOG A B"""
import sys
from pathlib import Path
import numpy as np
RAW = Path('/home/jetson/jetformer/data/raw')
log, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ld = RAW / log
sc = [d for d in sorted(ld.glob('2026*')) if (d / 'AI_CMD.npz').exists()]
def cat(name, keys):
    ps = [np.load(d / (name + '.npz')) for d in sc]
    return {k: np.concatenate([p[k] for p in ps]) for k in keys if all(k in p for p in ps)}
z = cat('AI_CMD', ('timestamps_us', 'out', 'model_out', 'assist', 'select', 'hold', 'ttc'))
v = cat('VEHICLE', ('timestamps_us', 'erpm', 'pwm'))
im = cat('IMU', ('timestamps_us', 'accel'))
t0 = int(z['timestamps_us'][0])
t = (z['timestamps_us'] - t0) / 1e6
vt = (v['timestamps_us'] - t0) / 1e6
it = (im['timestamps_us'] - t0) / 1e6
hg = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
for i in np.nonzero((t >= a) & (t <= b))[0]:
    j = np.searchsorted(vt, t[i], side='right') - 1
    k = (it > t[i] - 0.034) & (it <= t[i])
    print('%.3f' % t[i], 'out', [round(float(x)) for x in z['out'][i]], 'mo', [round(float(x)) for x in z['model_out'][i]],
          'as', int(z['assist'][i]), 'sel', int(z['select'][i]), 'hold', int(z['hold'][i]), 'erpm', int(v['erpm'][j]),
          'pwm', v['pwm'][j].tolist(), 'g', round(float(hg[k].max()), 2) if k.any() else None)
