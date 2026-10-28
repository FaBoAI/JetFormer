"""票 A: 10/24 20261024-160847 の 40.07 秒あたりを data/raw から読むだけで並べる(書かない)。
python -I -B look160847.py → 標準出力"""
import json
import sys
from pathlib import Path
import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')
LOG = sys.argv[1] if len(sys.argv) > 1 else '20261024-160847'
T_A = float(sys.argv[2]) if len(sys.argv) > 2 else 39.3
T_B = float(sys.argv[3]) if len(sys.argv) > 3 else 41.6


def load(log):
    scenes = sorted(p for p in (RAW / log).iterdir() if p.is_dir())
    t0 = json.loads((scenes[0] / 'scene.json').read_text())['start_us']
    cat = {}
    for s in scenes:
        for k in ('AI_CMD', 'VEHICLE', 'IMU'):
            z = np.load(s / ('%s.npz' % k), allow_pickle=False)
            d = cat.setdefault(k, {})
            for n in z.files:
                if z[n].ndim == 0:
                    continue
                d.setdefault(n, []).append(z[n])
            d.setdefault('_scene', []).append(np.full(len(z['timestamps_us']), int(s.name[-4:])))
    for k in cat:
        cat[k] = {n: np.concatenate(v) for n, v in cat[k].items()}
        cat[k]['t'] = (cat[k]['timestamps_us'] - t0) / 1e6
    return cat, scenes


cat, scenes = load(LOG)
for s in scenes:
    j = json.loads((s / 'scene.json').read_text())
    print(s.name, 'start %.3f end %.3f' % ((j['start_us'] - json.loads((scenes[0] / 'scene.json').read_text())['start_us']) / 1e6,
                                           (j['end_us'] - json.loads((scenes[0] / 'scene.json').read_text())['start_us']) / 1e6))
A, V, I = cat['AI_CMD'], cat['VEHICLE'], cat['IMU']
print('--- AI_CMD')
for i in np.where((A['t'] >= T_A) & (A['t'] <= T_B))[0]:
    print('%7.3f sc%d out=%6.1f/%6.1f model=%6.1f/%6.1f assist=%d hold=%d sel=%d guard=%d ttc=%.2f' % (
        A['t'][i], A['_scene'][i], A['out'][i][0], A['out'][i][1], A['model_out'][i][0], A['model_out'][i][1],
        A['assist'][i], A['hold'][i], A['select'][i], A['guard'][i], A['ttc'][i]))
print('--- VEHICLE (erpm, hall_count, pwm)')
for i in np.where((V['t'] >= T_A) & (V['t'] <= T_B))[0]:
    print('%7.3f sc%d erpm=%6d hall=%d err=%d st=%d pwm=%s sel=%d' % (V['t'][i], V['_scene'][i], V['erpm'][i], V['hall_count'][i],
                                                          V['hall_errors'][i], V['hall_state'][i], list(V['pwm'][i]), V['select'][i]))
print('--- IMU (horizontal G, ax, ay, gz)')
for i in np.where((I['t'] >= T_A) & (I['t'] <= T_B))[0]:
    a = I['accel'][i]
    print('%7.3f hG=%.2f ax=%6.2f ay=%6.2f az=%6.2f gz=%7.2f' % (I['t'][i], np.hypot(a[0], a[1]) / 9.80665, a[0], a[1], a[2], I['gyro'][i][2]))
