"""反証 4 回目・安全(fix4/review_safety): 共通の部品。直さない・装置に触れない・GPU を使わない。

- load_snapshot(name): tests/fixtures/<name>/ の drive.py・assist.py を別のモジュールとして読む(sha256 を確かめる)
- run_open(mod, p, frames, imu): tests/test_launch_drive.run(開いたループ。フレームは前もって作る)
- run_closed(mod, amod, p, scene): fix3/refute2_safety/closed_loop の偽物の ESC・車で本物の drive._loop を回す
- same_rows(a, b): 記録の行が(infer_ms のほか)ビット単位で同じか
一時の場所は fix4/review_safety/tmp_*(終わったら消す)。
"""
import collections
import hashlib
import importlib.util
import shutil
import sys
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
FIX = ROOT / 'experiments' / '20261026_fix'
for p in (ROOT / 'tests', ROOT, FIX / 'fix2' / 'safety2', FIX / 'fix3' / 'refute2_safety', FIX / 'fix3' / 'safety'):
    sys.path.insert(0, str(p))

from jetformer import assist as A          # noqa: E402
from jetformer import clock, config, drive  # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize   # noqa: E402
import test_launch_drive as TLD            # noqa: E402
import closed_loop as CL                   # noqa: E402

C, N, PARAMS, T0, NS = TLD.C, TLD.N, TLD.PARAMS, TLD.T0, TLD.NS
TMP = HERE / 'tmp_rs'
CL.TMP = TMP

SNAP_SHA = {
    'fix3_before': {'drive.py': '216efc09981394f7213d846a065c55ca0b4e4c2f722a0a71980e15dbb44dad56',
                    'assist.py': '6ee550c0b8e09881de3b4231254ec5a247bf209c0f2bdc454381ae2848fc9cfb'},
    'fix3_r3_1_before': {'drive.py': 'b8c3ddba39681d81f14249981c770a9349e6dba3bcf5b4a8d88ee88afbc85db6',
                         'assist.py': '6b68e813533c52427e73746b5644183db4eb3c6a9af59b90e5df4e0180e38a31'},
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_snapshot(name):
    d = ROOT / 'tests' / 'fixtures' / name
    for fn, h in SNAP_SHA[name].items():
        assert sha(d / fn) == h, (name, fn)
    aspec = importlib.util.spec_from_loader('jetformer._rs_assist_%s' % name, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((d / 'assist.py').read_text(encoding='utf-8'), str(d / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._rs_drive_%s' % name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((d / 'drive.py').read_text(encoding='utf-8'), str(d / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    return mod, amod


def run_open(mod, p, frames, imu, hook=None, dry_run=False):
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        d, pca, rec = TLD.run(mod, mp, Path(td), p, frames, imu, hook=hook, dry_run=dry_run)
    return d, pca, rec


def run_closed(p, scene, mod=None, amod=None, hook=None, floor=True, wrap=None):
    """closed_loop.run と同じ(mod・amod を選べる)。wrap = Recovery.step を包む関数(段の記録)"""
    mod = mod or drive
    amod = amod or A
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        if wrap is not None:
            mp.setattr(amod.Recovery, 'step', wrap(amod.Recovery.step))
        tmp_path = Path(td)
        box = {'ns': T0, 'm': 100.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                      time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        if floor:
            np.save(fp, np.full((42, 54), float(CL.FLOOR_MM), np.float32))
        mp.setattr(mod, 'FLOOR_REF_PATH', fp if floor else tmp_path / 'none.npy')
        rec = FakeRecorder()
        d = mod.Driver(hub=None, recorder=rec)
        d.params = dict(p)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = CL.TimedPCA(box)
        out = mod._Output(pca, config.PCA9685, (C, N), lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        sim = CL.Sim(d, scene, bufs, cam_slot, box, pca, hook)
        d._loop(d.st, CL.Runner(sim), fake_normalize, pca, out, (C, N), 6, 0.0, cam_slot, sim, *bufs, False)
        d._stop.clear()
    return sim, rec, pca, d


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) == type(b) and a == b


def same_rows(ra, rb):
    """(同じか, 最初の違い)。infer_ms は計測なので除く"""
    if len(ra) != len(rb):
        return False, ('len', len(ra), len(rb))
    for (t0, r0), (t1, r1) in zip(ra, rb):
        if t0 != t1:
            return False, ('t', t0, t1)
        keys = set(r0) | set(r1)
        for k in sorted(keys):
            if k == 'infer_ms':
                continue
            if k not in r0 or k not in r1 or not _same(r0[k], r1[k]):
                return False, ('row', (t0 - T0) / 1e9, k, r0.get(k), r1.get(k))
    return True, None


def same_events(ea, eb):
    if len(ea) != len(eb):
        return False, ('len', len(ea), len(eb))
    for x, y in zip(ea, eb):
        if x[0] != y[0] or not _same(dict(x[1]), dict(y[1])):
            return False, (x, y)
    return True, None


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)
