"""反証 w1-safety-r61-safety-1: 10/23・24 の車両の値の止まり(>0.15 秒)の前後(読むだけ)。"""
from pathlib import Path
import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')
for ep in sorted(list(RAW.glob('20261023*/2026*-*')) + list(RAW.glob('20261024*/2026*-*'))):
    try:
        z = np.load(ep / 'VEHICLE.npz')
        t = z['timestamps_us'].astype(np.int64)
    except Exception:
        continue
    dt = np.diff(t) / 1e6
    for i in np.where((dt > 0.15) & (dt < 5.0))[0]:
        print(ep.name, 'gap=%.3f' % dt[i], 'sel', bool(z['select'][i]), bool(z['select'][i + 1]), 'erpm', int(z['erpm'][i]), int(z['erpm'][i + 1]),
              'seq', int(z['seq'][i]), int(z['seq'][i + 1]), 'board_t', int(z['board_t_us'][i]), int(z['board_t_us'][i + 1]),
              'pwm', z['pwm'][i].tolist(), z['pwm'][i + 1].tolist(), 'i', i, len(t))
