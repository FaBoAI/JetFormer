"""読むだけ: data/raw/20261024-160847 の 5 場面をつなぎ、39.3〜41.5 秒の VEHICLE(eRPM・hall_count・select)と
AI_CMD(out・assist などの欄)を並べる。場面のさかい目の時刻も出す。
    python -I -B look160847.py
"""
import json
import sys
from pathlib import Path

import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw/20261024-160847')


def load(name):
    out = []
    for d in sorted(RAW.glob('20261024-160847-*')):
        f = d / (name + '.npz')
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        out.append((d.name, {k: z[k] for k in z.files}))
    return out


def main():
    ai = load('AI_CMD')
    veh = load('VEHICLE')
    t0 = min(int(z['timestamps_us'][0]) for _, z in ai)
    print('AI_CMD keys', list(ai[0][1].keys()))
    print('VEHICLE keys', list(veh[0][1].keys()))
    for n, z in ai:
        ts = (z['timestamps_us'] - t0) / 1e6
        print('scene', n, 'AI_CMD %.3f..%.3f n=%d' % (ts[0], ts[-1], len(ts)))
    lo, hi = float(sys.argv[1]) if len(sys.argv) > 1 else 39.3, float(sys.argv[2]) if len(sys.argv) > 2 else 41.5
    print('--- VEHICLE', lo, hi)
    for n, z in veh:
        ts = (z['timestamps_us'] - t0) / 1e6
        m = (ts >= lo) & (ts <= hi)
        cols = [k for k in ('erpm', 'hall_count', 'select', 'throttle_us', 'steer_us', 'thr', 'vbat') if k in z]
        for i in np.nonzero(m)[0][::3]:
            print(n[-4:], '%.3f' % ts[i], {k: (z[k][i].tolist() if hasattr(z[k][i], 'tolist') else z[k][i]) for k in cols})
    print('--- AI_CMD')
    for n, z in ai:
        ts = (z['timestamps_us'] - t0) / 1e6
        m = (ts >= lo) & (ts <= hi)
        cols = [k for k in z if k not in ('timestamps_us',)]
        for i in np.nonzero(m)[0]:
            row = {}
            for k in cols:
                v = z[k][i]
                v = v.tolist() if hasattr(v, 'tolist') else v
                if isinstance(v, list) and len(v) > 6:
                    continue
                row[k] = [round(x, 1) if isinstance(x, float) else x for x in v] if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v)
            print(n[-4:], '%.3f' % ts[i], json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
