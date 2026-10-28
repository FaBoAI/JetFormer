"""読むだけ: 20261024-160847 の VEHICLE の pwm(人の送信機の値か書いた値)・moving・quality を 39.9〜41.3 秒で並べ、
IMU の前後の加速度も 0.1 秒ごとに出す。 python -I -B look160847_pwm.py"""
from pathlib import Path
import numpy as np
RAW = Path('/home/jetson/jetformer/data/raw/20261024-160847')
def load(name):
    out = []
    for d in sorted(RAW.glob('20261024-160847-*')):
        f = d / (name + '.npz')
        if f.exists():
            z = np.load(f, allow_pickle=False)
            out.append((d.name, {k: z[k] for k in z.files}))
    return out
ai = load('AI_CMD'); veh = load('VEHICLE'); imu = load('IMU')
t0 = min(int(z['timestamps_us'][0]) for _, z in ai)
print('IMU keys', list(imu[0][1].keys()))
for n, z in veh:
    ts = (z['timestamps_us'] - t0) / 1e6
    m = (ts >= 39.9) & (ts <= 41.3)
    for i in np.nonzero(m)[0][::3]:
        print(n[-4:], '%.3f' % ts[i], 'pwm', z['pwm'][i].tolist(), 'sel', bool(z['select'][i]), 'mov', z['moving'][i].tolist(), 'erpm', int(z['erpm'][i]), 'hall', int(z['hall_count'][i]))
for n, z in imu:
    ts = (z['timestamps_us'] - t0) / 1e6
    m = (ts >= 39.9) & (ts <= 41.3)
    keys = [k for k in z if k != 'timestamps_us']
    idx = np.nonzero(m)[0]
    for i in idx[::10]:
        print('IMU', n[-4:], '%.3f' % ts[i], {k: np.round(z[k][i], 2).tolist() for k in keys[:3]})
