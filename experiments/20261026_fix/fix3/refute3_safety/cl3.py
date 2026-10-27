"""反証 3 回目・safety: 閉じたループ(refute2_safety/closed_loop.py の偽物の車・ESC)で、本物の drive._loop を回す。
今の drive.py か、手を入れる前の写し(tests/fixtures/*/drive.py・assist.py)を選べる。装置・GPU・API には触れない。直さない。"""
import collections
import importlib.util
import sys
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/jetson/jetformer')
FIX3 = ROOT / 'experiments' / '20261026_fix' / 'fix3'
sys.path.insert(0, str(FIX3 / 'refute2_safety'))
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import closed_loop as CL                      # noqa: E402
from closed_loop import C, N, PARAMS, T0, NS, FLOOR_MM   # noqa: E402
from jetformer import assist as A             # noqa: E402
from jetformer import clock, config, drive    # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize   # noqa: E402

TMP = FIX3 / 'tmp_refute3_safety'
CL.TMP = TMP


def load_fixture(name):
    d = ROOT / 'tests' / 'fixtures' / name
    aspec = importlib.util.spec_from_loader('jetformer._assist_%s' % name, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((d / 'assist.py').read_text(encoding='utf-8'), str(d / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_%s' % name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((d / 'drive.py').read_text(encoding='utf-8'), str(d / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    return mod, amod


PH = []      # (t, 段の前, 段のあと, hold_g, impact_g, hold_since が渡ったか)


def _wrap_step(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 4), p0, self.phase, round(float(kw.get('hold_g', 0.0)), 2), round(float(impact_g), 2),
                   kw.get('hold_since') is not None))
        return r
    return step


def run(params, scene, hook=None, floor=True, mod=None, amod=None, wrap=True):
    mod = mod or drive
    amod = amod or A
    TMP.mkdir(parents=True, exist_ok=True)
    del PH[:]
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        if wrap and amod is A:
            mp.setattr(A.Recovery, 'step', _wrap_step(A.Recovery.step))
        tmp_path = Path(td)
        box = {'ns': T0, 'm': 100.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                      time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        if floor:
            np.save(fp, np.full((42, 54), float(FLOOR_MM), np.float32))
        mp.setattr(mod, 'FLOOR_REF_PATH', fp if floor else tmp_path / 'none.npy')
        rec = FakeRecorder()
        d = mod.Driver(hub=None, recorder=rec)
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = CL.TimedPCA(box)
        out = mod._Output(pca, config.PCA9685, (C, N), lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        sim = CL.Sim(d, scene, bufs, cam_slot, box, pca, hook)
        d._loop(d.st, CL.Runner(sim), fake_normalize, pca, out, (C, N), 6, 0.0, cam_slot, sim, *bufs, False)
        d._stop.clear()
        return sim, rec, pca, list(PH)


def phases(ph):
    return [(t, p0, p1) for t, p0, p1, *_ in ph if p0 != p1]
