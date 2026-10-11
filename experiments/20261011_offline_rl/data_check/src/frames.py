"""ai_mix_20261011 のフレームの表(必要な配列だけ)を作ってキャッシュする。CPU だけ・読み取りだけ

    python frames.py            # → ../frames_ai_mix.npz(あれば作り直さない。--force で作り直す)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
DS = ROOT / 'data' / 'datasets' / 'ai_mix_20261011'
OUT = Path(__file__).resolve().parent.parent / 'frames_ai_mix.npz'

KEYS1 = ['timestamps_us', 'mode', 'future_valid', 'vehicle_valid', 'erpm_valid', 'source', 'label_valid', 'reward',
         'mc_return', 'segment_id', 'segment_end', 'end_code', 'time_to_end_s', 'collision', 'near_collision',
         'failure_takeover', 'loc_code', 'progress_m', 'label_index', 'tof_age_ms', 'vehicle_age_ms']
KEYS2 = ['action_label', 'action_exec', 'action_sat', 'vehicle']


def build(force=False):
    if OUT.exists() and not force:
        return dict(np.load(OUT, allow_pickle=False))
    splits = json.loads((DS / 'splits.json').read_text())
    split_of = {s: k for k in ('train', 'val') for s in splits[k]}
    cols = {k: [] for k in KEYS1 + KEYS2 + ['fut', 'gz', 'ax', 'ay', 'scene', 'log', 'split', 'img_ok']}
    scenes = sorted(p.stem for p in (DS / 'aligned').glob('*.npz'))
    problems = []
    for s in scenes:
        try:
            with np.load(DS / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                a = {k: z[k] for k in KEYS1 + KEYS2 + ['future_actions', 'imu', 'cam_files'] if k in z.files}
            n = len(a['timestamps_us'])
            a.setdefault('vehicle_age_ms', np.full(n, np.nan, np.float32))   # AI だけのシーン(人の側の車両データなし)には無い
        except Exception as e:          # 電源断で壊れたファイル
            problems.append('%s: %s' % (s, e))
            continue
        for k in KEYS1 + KEYS2:
            cols[k].append(a[k])
        cols['fut'].append(a['future_actions'])
        cols['gz'].append(np.rad2deg(a['imu'][:, -1, 5]))          # 最新の gyro z [deg/s](左 +)
        cols['ax'].append(a['imu'][:, -1, 0])
        cols['ay'].append(a['imu'][:, -1, 1])
        cols['scene'].append(np.full(n, s))
        cols['log'].append(np.full(n, s.rsplit('-', 1)[0]))
        cols['split'].append(np.full(n, split_of.get(s, '?')))
        ok = np.array([os.path.exists(DS / str(f)) for f in a['cam_files']])
        cols['img_ok'].append(ok)
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    out['problems'] = np.array(problems if problems else [''])
    tmp = str(OUT) + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, OUT)
    return out


if __name__ == '__main__':
    f = build('--force' in sys.argv)
    print({k: v.shape for k, v in f.items()})
    print('problems', f['problems'])
