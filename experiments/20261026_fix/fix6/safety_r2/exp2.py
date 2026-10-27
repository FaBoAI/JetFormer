"""fix6 係 safety 直す 6-2 回目の台本の共通の部品(装置に触れない・GPU を使わない・読むだけ)。

- load(name): 'now'(jetformer/ の今の版)・'before'(tests/fixtures/fix6_safety_before = fix5 のあと。指摘の「直す前の写し」)・
  'r1'(before/*.orig = fix6 r1 のあと。この係が直す前)・'cand'(cand/。EXP のスイッチつき)・'final'(final/)
- run(name, params, scene, esc=None, flags=''): 本物の drive._loop を tests/test_fix3_r2_1 の閉じたループで回す。
  esc = None → 元の偽物の ESC(R2.Car そのもの。中立は 0.1 秒の決まりを 10 ms 刻みで数えるので 11 刻み)。
  esc = (neu, rdelay) → 中立で下がれる秒・後退側から下がり始める秒だけ変えた偽物(反証の tf62.Car と同じ形。1e-9 の余裕つき)
  flags(cand だけ): so = back_hold は後ろへ動いていたサンプルのときだけ・oc = CHECK_S のやり直しは見た後退で決める・
  dr = やり直すとき見積もりを捨てる・dg = 'wait' → 'gap' で確かめられなかった見積もりを捨てる・
  dv = 'gap' → 'reverse' で確かめられなかった(見た後退が 0 の)見積もりを捨てる・
  dw = dv と同じ。ただし見積もりのフレームからあとに 300 eRPM を超えたサンプル(back_t)があれば捨てない・
  dz = dv と同じ。ただし見積もりのフレームからあとに z_erpm(ztXXX。既定 0)を超えて後ろへ動いたサンプルがあれば捨てない・rnXX = EscTrack.READY_NEUTRAL_S
  を 0.XX 秒・dlXX = EscTrack.DELAY_S を 0.XX 秒
- metrics(res, params): 最初のリカバリーの 'brake' の始め・escape・下がった距離・escape が前の壁を押した秒・前と後ろの当たり・
  偽物の ESC が後退に入った時刻・後退側を書いているのに止まっていた秒・やり直し(+50 µs)の有無・出来事
一時の場所は fix6/tmp_safety_r2/<R2_TAG>(終わったら消す)。
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
TMP = ROOT / 'experiments' / '20261026_fix' / 'fix6' / 'tmp_safety_r2' / os.environ.get('R2_TAG', 'main')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import assist as A      # noqa: E402
from jetformer import drive            # noqa: E402
import test_fix3_r2_1 as R2            # noqa: E402
import test_rec_fix5_safety as F5      # noqa: E402
from test_launch_drive import C, N, PARAMS   # noqa: E402

DT = R2.DT
_MODS = {}
DIRS = {'before': ROOT / 'tests' / 'fixtures' / 'fix6_safety_before', 'r1': HERE / 'before', 'cand': HERE / 'cand',
        'final': HERE / 'final'}


def load(name):
    if name == 'now':
        return drive, A
    if name in _MODS:
        return _MODS[name]
    d = DIRS[name]
    suf = '.orig' if name == 'r1' else ''
    aspec = importlib.util.spec_from_loader('jetformer._s2_assist_%s' % name, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    ap = d / ('assist.py' + suf)
    exec(compile(ap.read_text(encoding='utf-8'), str(ap), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._s2_drive_%s' % name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    dp = d / ('drive.py' + suf)
    exec(compile(dp.read_text(encoding='utf-8'), str(dp), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    _MODS[name] = (mod, amod)
    return mod, amod


FLAG = {'so': 'seen_only', 'oc': 'obs_check', 'dr': 'drop_retry', 'dg': 'drop_gap', 'dv': 'drop_rev', 'dw': 'drop_rev_w', 'dz': 'drop_rev_z'}


def setup(name, flags=''):
    mod, amod = load(name)
    if name in ('cand', 'final'):
        for k in FLAG.values():
            if hasattr(amod, 'EXP'):
                amod.EXP[k] = False
        if hasattr(amod, 'EXP'):
            amod.EXP['z_erpm'] = 0.0
        amod.EscTrack.READY_NEUTRAL_S, amod.EscTrack.DELAY_S = 0.08, 0.06
        for f in [x for x in flags.split(',') if x]:
            if f.startswith('rn'):
                amod.EscTrack.READY_NEUTRAL_S = int(f[2:]) / 100.0
            elif f.startswith('dl'):
                amod.EscTrack.DELAY_S = int(f[2:]) / 100.0
            elif f.startswith('zt'):
                amod.EXP['z_erpm'] = float(f[2:])
            else:
                amod.EXP[FLAG[f]] = True
    return mod, amod


def make_car(esc):
    if esc is None:
        class OrigCar(R2.Car):
            def __init__(self, s):
                super().__init__(s)
                self.revlog = []

            def step(self, t, thr):
                r = super().step(t, thr)
                self.revlog.append((t, self.rev_on))
                return r
        return OrigCar
    neu, rdelay = esc

    class EscCar(R2.Car):
        """R2.Car と同じ(中立で ready になる秒・後退側から下がり始める秒だけ変える。反証の tf62.Car と同じ)"""

        def __init__(self, s):
            super().__init__(s)
            self.revlog = []

        def step(self, t, thr):
            s = self.s
            fwd, rev = thr < N - 20.0, thr > N + 40.0
            v0 = self.v
            if fwd:
                self.ready = self.braked = self.rev_on = False
                self.rev_wait = self.neu_t = 0.0
                if self.v < 0:
                    self.v = min(0.0, self.v + 8.0 * DT)
                else:
                    vt = min((N - thr) / 80.0 * 1.2, 6.0)
                    self.v += float(np.clip((vt - self.v) / 0.3, -6.0, 6.0)) * DT
            elif rev:
                self.neu_t = 0.0
                if self.rev_on:
                    a = (thr - N) / 300.0 * 5.0
                    self.v = max(self.v - a * DT, -(thr - N) / 300.0 * 3.0)
                elif self.ready and self.v <= 0.075:
                    self.rev_wait += DT
                    if self.rev_wait >= rdelay - 1e-9:
                        self.rev_on = True
                else:
                    self.braked = True
                    self.v = max(0.0, self.v - 6.0 * DT) if self.v > 0 else min(0.0, self.v + 6.0 * DT)
            else:
                self.rev_wait = 0.0
                self.neu_t += DT
                if self.braked and self.neu_t >= neu - 1e-9:
                    self.ready, self.braked = True, False
                if self.rev_on:
                    self.rev_on = False
                    self.ready = True
                self.v = max(0.0, self.v - 1.5 * DT) if self.v > 0 else min(0.0, self.v + 1.5 * DT)
            nx = self.x + self.v * DT
            prev_con = self.contact
            self.contact = None
            if nx >= s['xf']:
                if self.v > 0.05 and prev_con != 'front':
                    g = min(2.0 + 2.5 * self.v, 9.0)
                    self.hits.append((round(t, 3), 'front', round(self.v, 2), round(g, 2)))
                    self.spike = [g, g]
                nx, self.v = s['xf'], 0.0
                self.contact = 'front'
            if s.get('xr') is not None:
                if not self.rear_active and nx > s['xr'] + 0.1:
                    self.rear_active = True
                if self.rear_active and nx <= s['xr']:
                    if self.v < -0.05 and prev_con != 'rear':
                        g = min(2.0 + 2.5 * -self.v, 9.0)
                        self.hits.append((round(t, 3), 'rear', round(self.v, 2), round(g, 2)))
                        self.spike = [g, g]
                    nx, self.v = s['xr'], 0.0
                    self.contact = 'rear'
            self.x = nx
            acc = (self.v - v0) / DT
            g = self.spike.pop(0) if self.spike else 0.0
            self.revlog.append((t, self.rev_on))
            return acc, g
    return EscCar


def run(name, params, scene, esc=None, flags=''):
    mod, amod = setup(name, flags)
    TMP.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        mp.setattr(R2, 'Sim', F5.SpikeSim)
        mp.setattr(R2, 'Car', make_car(esc))
        return R2.run_cl(mod, mp, Path(td), params, scene)


def guards(rec):
    return [(e[1]['what'], e[1]['phase'], e[1]['backed_s']) for e in rec.events if e[0] == 'recovery_guard']


def metrics(res, params, t_from=1.5):
    sim, rec, pca = res
    rev_us = params['rev_us']
    tb = next((t for t, s, th in pca.w if t > t_from and abs(th - (N + rev_us)) < 1e-6), None)
    if tb is None:
        return None
    esc = N - A.Recovery.ESCAPE_US
    te = next((t for t, s, th in pca.w if t > tb and abs(th - esc) < 1e-6), None)
    end = (te + 0.6) if te is not None else tb + 3.0
    x_b = next(x for t, x, v, th in sim.log if t >= tb)
    xs = [x for t, x, v, th in sim.log if tb <= t <= end]
    push = sum(DT for t, x, v, th in sim.log if te is not None and te <= t < te + 0.6 and x >= sim.s['xf'] - 1e-9
               and abs(th - esc) < 1e-6)
    end2 = te if te is not None else tb + 3.0
    dead = sum(DT for t, x, v, th in sim.log if tb <= t < end2 and th > N + 40 and abs(v) < 0.05)
    retry = any(abs(th - (N + rev_us + A.Recovery.RETRY_EXTRA_US)) < 1e-6 for t, s, th in pca.w if tb <= t <= end2)
    vmin = min((v for t, x, v, th in sim.log if tb <= t <= end), default=0.0)
    return {'t_brake': tb, 't_escape': te, 'back': round(x_b - min(xs), 3), 'push': round(push, 2),
            'front': [h for h in sim.car.hits if h[1] == 'front' and tb < h[0] <= end],
            'rear': [h for h in sim.car.hits if h[1] == 'rear' and tb <= h[0] <= end],
            't_on': next((t for t, r in sim.car.revlog if t >= tb and r), None), 'dead': round(dead, 2),
            'retry': retry, 'vmin': round(vmin, 2), 'guards': guards(rec)[:4]}


def P1(**kw):
    return dict(dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, rev_s=0.4, rev_us=300.0), **kw)


XF, XB = R2.XF, 2.4
MODEL = (C, N - 120.0)


def model_brake(t, car):
    """場面 brake: 壁の 0.45 m 手前からブレーキ側(中立 + 200)= 止まると ESC は後退を待っている(10/23 の形)"""
    return (C, N - 120.0) if car.x < XB - 0.45 and t < 6.0 else (C, N + 200.0)


def scene(kind, skips=(), drops=(), rear=None, dur=7.0):
    xf, model = (XF, MODEL) if kind == 'rev' else (XB, model_brake)
    sc = {'dur': dur, 'xf': xf, 'model': model, 'drops': list(drops), 'veh_skip': list(skips)}
    if rear is not None:
        sc['xr'] = xf - rear
    return sc


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)
    for p in (TMP.parent,):
        try:
            p.rmdir()
        except OSError:
            pass
