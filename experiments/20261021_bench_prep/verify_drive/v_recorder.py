#!/usr/bin/env python3
"""V-drive: 影の判定(cap_*)が本物の Recorder で AI_CMD.npz まで残るか。走行ループの記録の行(v_replay.py の結果。
ON・乱れ・途中の例外 -2・中立で止めた行を含む)を、何シーンにもまたがって入れ、保存したファイルと行ごとに比べる。
センサーは偽物(_Hub)。書くのは一時フォルダだけ。

    python -I v_recorder.py <new_*_perturb_z1_zfail.pkl> <一時フォルダ>
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import clock, config, drive  # noqa: E402
from jetformer.recorder import Recorder  # noqa: E402
from jetformer.sensors.base import Sample  # noqa: E402

src, tmp = Path(sys.argv[1]), Path(sys.argv[2])
tmp.mkdir(parents=True, exist_ok=True)
with open(src, 'rb') as f:
    R = pickle.load(f)
rows = R['rows'][:1500]                       # 約 50 秒
config.RAW_DIR = tmp
config.SCENE_SECONDS = 10.0


class _Hub(object):
    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


rec = Recorder(_Hub())
rec._warm_up = lambda: None
log = rec.start(description='v-drive recorder check')
t0 = rec.log['start_ns']
base = rows[0][0]
sent = []
for k, (t, r) in enumerate(rows):
    tt = t0 + (t - base) + 1_000_000
    rec._on_sample('CAM_FRONT', Sample(tt, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': k}))
    rec.add_ai(tt, r)
    sent.append((tt, r))
rec.stop(only_log=log)
CAP = {'cap_on': bool, 'cap_zone': np.int8, 'cap_zone_gyro': np.int8, 'cap_pred_sig': bool, 'cap_rel_psi': np.float32,
       'cap_mode': np.int8, 'cap_limit': np.float32, 'cap_in_us': np.float32, 'cap_thr_us': np.float32,
       'cap_brake_start': bool, 'cap_applied': bool}
scenes = sorted(p for p in (tmp / log).iterdir() if p.is_dir())
got = {k: [] for k in CAP}
got['out'] = []
n_files = 0
for s in scenes:
    p = s / 'AI_CMD.npz'
    if not p.exists():
        continue
    n_files += 1
    z = np.load(p)
    for k in got:
        got[k].append(z[k])
    for k, dt in CAP.items():
        assert z[k].dtype == np.dtype(dt), (k, z[k].dtype)
got = {k: np.concatenate(v) for k, v in got.items()}
bad = {}
for k, dt in CAP.items():
    want = np.array([r[k] for _, r in sent], dtype=dt)
    ok = np.array_equal(got[k], want, equal_nan=True) if want.dtype.kind == 'f' else np.array_equal(got[k], want)
    if not ok:
        bad[k] = int(np.sum(~((got[k] == want) | (np.isnan(got[k]) & np.isnan(want))) if want.dtype.kind == 'f'
                             else (got[k] != want)))
modes = {int(m): int(c) for m, c in zip(*np.unique(got['cap_mode'], return_counts=True))}
print(json.dumps({'rows': len(sent), 'saved_rows': int(len(got['cap_mode'])), 'scenes_with_ai': n_files, 'mismatch': bad,
                  'modes': modes, 'applied': int(got['cap_applied'].sum()), 'on': int(got['cap_on'].sum()),
                  'limit_inf': int(np.isinf(got['cap_limit']).sum()),
                  'ok': (not bad) and len(got['cap_mode']) == len(sent)}, ensure_ascii=False))
