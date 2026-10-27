"""fix6 係 safety 直す 6-1 回目(指摘 w1-safety-r61-safety-1)の台本の共通の部品。装置に触れない・GPU を使わない。

- load(name): 'now'(jetformer/ の今の版)・'cand'(この係の直しの候補 cand/)・'fix5_before'(tests/fixtures/fix5_safety_before)
  の drive.py・assist.py をモジュールにする(now 以外は別のモジュールとして読む)
- run(mods, params, scene): 本物の drive._loop を tests/test_fix3_r2_1 の閉じたループ(偽物の ESC と車)で回す。
  偽物の ESC が後退で走らせていた時刻も残す(sim.revlog = [(t, rev_on, contact)])
- first_rec(sim, pca, xf): 最初のリカバリーの 'brake' の始め・最初の escape・下がった距離・escape が前の壁を押した秒・後ろの当たり
一時の場所は fix6/tmp_safety_r1(終わったら消す)。
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
TMP = ROOT / 'experiments' / '20261026_fix' / 'fix6' / 'tmp_safety_r1' / 'exp'
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import assist as A      # noqa: E402
from jetformer import drive            # noqa: E402
import test_fix3_r2_1 as R2            # noqa: E402
import test_rec_fix5_safety as F5      # noqa: E402
from test_launch_drive import C, N, PARAMS   # noqa: E402

DT = R2.DT
_MODS = {}


def load(name):
    if name == 'now':
        return drive, A
    if name in _MODS:
        return _MODS[name]
    d = {'cand': HERE / 'cand', 'final': HERE / 'final', 'fix5_before': ROOT / 'tests' / 'fixtures' / 'fix5_safety_before',
         'fix6_before': ROOT / 'tests' / 'fixtures' / 'fix6_safety_before'}[name]
    aspec = importlib.util.spec_from_loader('jetformer._exp_assist_%s' % name, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((d / 'assist.py').read_text(encoding='utf-8'), str(d / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._exp_drive_%s' % name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((d / 'drive.py').read_text(encoding='utf-8'), str(d / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    _MODS[name] = (mod, amod)
    return mod, amod


class LogCar(R2.Car):
    def step(self, t, thr):
        r = super().step(t, thr)
        self.revlog.append((t, self.rev_on, self.contact))
        return r


FLAGS = {'ff': ('d', 'fast_full', True), 'ch': ('a', 'coast_hold', True), 'dd': ('d', 'no_double', False),
         'ff3': ('d', 'fast_thr', 300.0), 'sep': ('d', 'sep', True), 'mr2': ('a', 'min_left', 0.2),
         'mr15': ('a', 'min_left', 0.15), 'mr0': ('a', 'min_left', 0.0), 'mr25': ('a', 'min_left', 0.25),
         'mr3': ('a', 'min_left', 0.3), 'd06': ('t', None, None)}


def run(name, params, scene):
    base, *fl = name.split('+')
    mod, amod = load(base)
    if hasattr(mod, 'EXP'):
        mod.EXP.update({'fast_full': False, 'fast_thr': 1000.0, 'no_double': True, 'sep': False})
        amod.EXP.update({'coast_hold': False, 'min_left': None})
        amod.EscTrack.DELAY_S, amod.EscTrack.READY_NEUTRAL_S = 0.08, 0.1
        for f in fl:
            w, k, v = FLAGS[f]
            if w == 't':
                amod.EscTrack.DELAY_S, amod.EscTrack.READY_NEUTRAL_S = 0.06, 0.08
                continue
            (mod if w == 'd' else amod).EXP[k] = v
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        mp.setattr(R2, 'Sim', F5.SpikeSim)

        class _Car(LogCar):
            def __init__(self, s):
                super().__init__(s)
                self.revlog = []
        mp.setattr(R2, 'Car', _Car)
        sim, rec, pca = R2.run_cl(mod, mp, Path(td), params, scene)
    return sim, rec, pca


def first_rec(sim, pca, xf, t_from=1.5, rev_us=300.0):
    """最初のリカバリー: 'brake' の始め(リカバリーの後退側 = 中立 + rev_us を最初に書いた)・最初の escape(前進 130 µs)・
    その 0.6 秒あとまでに 'brake' の始めの位置から下がった最大の距離・escape が前の壁を押した秒(escape を書いていて前の壁に
    接している)・後ろの当たり"""
    tb = next((t for t, s, th in pca.w if t > t_from and abs(th - (N + rev_us)) < 1e-6), None)
    if tb is None:
        return None
    te = next((t for t, s, th in pca.w if t > tb and abs(th - (N - A.Recovery.ESCAPE_US)) < 1e-6), None)
    end = (te + 0.6) if te is not None else tb + 3.0
    xs = [x for t, x, v, th in sim.log if tb <= t <= end]
    xf = next(x for t, x, v, th in sim.log if t >= tb)
    push = sum(DT for t, x, v, th in sim.log if te is not None and te <= t < te + 0.6 and x >= sim.s['xf'] - 1e-9
               and abs(th - (N - A.Recovery.ESCAPE_US)) < 1e-6)
    rev_on = next((t for t, r, c in sim.car.revlog if t >= tb and r), None)
    return {'t_brake': tb, 't_escape': te, 'back_m': round(xf - min(xs), 3), 'esc_push_s': round(push, 2),
            'rear_hits': [h for h in sim.car.hits if h[1] == 'rear' and h[0] <= end], 't_rev_on': rev_on,
            'front_hits': [h for h in sim.car.hits if h[1] == 'front' and tb < h[0] <= end],
            'rev_max': max((th for t, s, th in pca.w if tb <= t <= (te or end)), default=None)}


def guards(rec):
    return [(e[1]['what'], e[1]['phase'], e[1]['backed_s']) for e in rec.events if e[0] == 'recovery_guard']


def P1(**kw):
    return dict(dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1), **kw)


def cleanup():
    shutil.rmtree(TMP.parent, ignore_errors=True)
