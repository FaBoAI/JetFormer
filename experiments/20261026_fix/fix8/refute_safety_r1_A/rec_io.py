"""票 A の共通: data/raw の 1 本(log)を、場面(scene)をつないで読む(読むだけ)。"""
import json
from pathlib import Path

import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')


def scenes_of(log):
    return sorted(p for p in (RAW / log).iterdir() if p.is_dir() and (p / 'AI_CMD.npz').exists()
                  and (p / 'VEHICLE.npz').exists() and (p / 'IMU.npz').exists())


def load_scene(d):
    out = {}
    for k in ('AI_CMD', 'VEHICLE', 'IMU'):
        z = np.load(d / ('%s.npz' % k), allow_pickle=False)
        out[k] = {n: z[n] for n in z.files if z[n].ndim > 0}
    return out


def load_log(log, scenes=None):
    """場面をつないだ dict(AI_CMD・VEHICLE・IMU)。'_scene' に場面の番号。scenes を渡せばその場面だけ"""
    ss = scenes if scenes is not None else scenes_of(log)
    cat = {}
    for s in ss:
        z = load_scene(s)
        for k, d in z.items():
            c = cat.setdefault(k, {})
            n0 = len(d['timestamps_us'])
            for n, v in d.items():
                c.setdefault(n, []).append(v)
            c.setdefault('_scene', []).append(np.full(n0, int(s.name[-4:])))
    for k in cat:
        cat[k] = {n: np.concatenate(v) for n, v in cat[k].items()}
    L = json.loads((RAW / log / 'log.json').read_text(encoding='utf-8'))
    t0 = json.loads((ss[0] / 'scene.json').read_text(encoding='utf-8'))['start_us'] if (ss[0] / 'scene.json').exists() else None
    return cat, L, t0
