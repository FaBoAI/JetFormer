"""読むだけ: 10/23 No.6(20261023-205458)のリカバリー 23.45 秒からの、書いた値と後ろへ下がった距離(eRPM の積分。
1000 eRPM = 0.5 m/s)・速さを車両の値ごとに出す。 python -I -B no6_profile.py"""
import json
from pathlib import Path
import numpy as np
ld = Path('/home/jetson/jetformer/data/raw/20261023-205458')
sc = [d for d in sorted(ld.glob('2026*')) if (d / 'AI_CMD.npz').exists()]
def cat(name, keys):
    ps = [np.load(d / (name + '.npz')) for d in sc]
    return {k: np.concatenate([p[k] for p in ps]) for k in keys}
z = cat('AI_CMD', ('timestamps_us', 'out'))
v = cat('VEHICLE', ('timestamps_us', 'erpm'))
im = cat('IMU', ('timestamps_us', 'accel'))
t0 = int(z['timestamps_us'][0])
zt = (z['timestamps_us'] - t0) / 1e6
vt = (v['timestamps_us'] - t0) / 1e6
it = (im['timestamps_us'] - t0) / 1e6
hg = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
m = (vt >= 23.45) & (vt <= 24.36)
x = 0.0
rows = []
prev = None
for tt, e in zip(vt[m], v['erpm'][m]):
    if prev is not None:
        x += max(0.0, float(e)) * 0.0005 * (tt - prev)
    prev = tt
    k = np.searchsorted(zt, tt, side='right') - 1
    g = hg[(it > tt - 0.015) & (it <= tt + 0.015)]
    rows.append((round(float(tt), 3), int(z['out'][k][1]), int(e), round(float(e) * 0.0005, 2), round(x, 3), round(float(g.max()), 2) if len(g) else None))
for r in rows[::2]:
    print(r)
Path(__file__).with_name('out').mkdir(exist_ok=True)
(Path(__file__).with_name('out') / 'no6_profile.json').write_text(json.dumps(rows))
