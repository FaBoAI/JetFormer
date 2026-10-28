"""fix8 係 safety-proposal(fix8/proposal)の共通の部品。fix7/safety_r2/h9.py の写し。変えたのは: 一時の場所(fix8/tmp_proposal)・
写し base / pA / pB / pC(fix8/proposal/<名前>/ の assist.py・drive.py。base = 今の jetformer/ の写し。'now' も base を読む = 走りの途中で
jetformer/ が書き換わっても比べがずれない)・patch の OFFTOO(写しの P8_OFF_TOO を走りの間だけ True)だけ。
元の説明: 直す 7-2 回目・係 safety(fix7/safety_r2)の共通の部品。反証 7-2 回目の review_safety_r2/h8.py の写し。変えたのは: 一時の場所
(fix7/tmp_safety_r2)・写し 'r1'(tests/fixtures/fix7_safety_r2_before = 7-1 回目のあと = この係が直す前)を足した・run の patch
(読み込んだ assist の定数をその走りの間だけ変える: {'READY': EscMayReverse.READY_NEUTRAL_S, 'SURE': SURE_BACK_S})だけ。
元の説明: 直す 7-1 回目・係 safety(fix7/safety)の共通の部品。反証 7-1 回目の review_safety_r1/rv7.py の写し(一時の場所・写しの名前だけ変えた)。
装置に触れない・GPU を使わない・API を呼ばない。
- 足した写し: 'f7b' = tests/fixtures/fix7_safety_before(この係が直す前 = fix6 r2 のあと)

- load(name): 'now'(jetformer/ の今の版)・'f6b'(tests/fixtures/fix6_safety_before = fix5 のあと)・
  'f6r2b'(tests/fixtures/fix6_safety_r2_before = fix6 r1 のあと)・'f5b'(tests/fixtures/fix5_safety_before = fix4 のあと)
  の drive.py・assist.py をモジュールにする(sha256 を照らす)
- run(name, params, scene): 本物の drive._loop を fix3/refute2_safety/closed_loop.py の閉じたループ(偽物の ESC と車)で回す。
  scene に足せるもの: spikes(止まらない衝撃)・esc=(neu, rdelay)(中立で下がれる秒・後退側から下がり始める秒。1e-9 の余裕)・
  bounce(前の壁に当たったとき、当たった速さ × bounce で後ろへ跳ね返る)・cam0(最初のフレームの時刻)
  戻り値 (sim, rec, pca, ph)。ph = Recovery.step の (t, 段の前, 段のあと, hold_since が渡った, hold_rev_s, est_s, backed_s)
- metrics(sim, rec, pca): 後ろの当たり・前の壁を押した秒・後ろの物を押した秒・中止・後退側を止まったまま書いた秒 など
一時の場所は fix7/tmp_review_safety_r1(終わったら消す)。
"""
import collections
import hashlib
import importlib.util
import math
import os
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
FIXD = ROOT / 'experiments' / '20261026_fix'
TMP = FIXD / 'fix8' / 'tmp_proposal' / 'h' / os.environ.get('RV9_TAG', 'main')
PROP = FIXD / 'fix8' / 'proposal'
for p in (ROOT / 'tests', ROOT, FIXD / 'fix3' / 'refute2_safety'):
    sys.path.insert(0, str(p))

from jetformer import assist as A          # noqa: E402
from jetformer import clock, config, drive  # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize   # noqa: E402
import test_launch_drive as TLD            # noqa: E402
import closed_loop as CL                   # noqa: E402

C, N, PARAMS, T0, NS = TLD.C, TLD.N, TLD.PARAMS, TLD.T0, TLD.NS
DT = CL.DT
PSHA = {   # fix8 の写し(make_copies.py の出力。drive.py はどれも今と同じ)
    'base': {'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
             'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'},
    'pA': {'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
           'assist.py': '82e2f8a584c66fe292d2122800dcacb928d93c24aba78ca53faa5331915528d3'},
    'pB': {'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
           'assist.py': '05d78b8aeaf09a64364e284191ce19259b3b0baee4d449674056f3fe6228e5b6'},
    'pC': {'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
           'assist.py': '7ac97688a93713704762dc29f3ff64d17dc2ec4b758332ec4200ffd2240585b8'},
}
SNAP = {
    'r1': ('fix7_safety_r2_before', {'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
                                     'assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f'}),
    'f7b': ('fix7_safety_before', {'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
                                    'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'}),
    'f6b': ('fix6_safety_before', {'drive.py': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c',
                                   'assist.py': '80f20890804170bede9c7df6e7f05892ab4e5397f8c5f837fda49ddba64d1276'}),
    'f6r2b': ('fix6_safety_r2_before', {'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
                                        'assist.py': 'd1d62c66bef856002c14db8dd2fd9c35279a36878b2e4817a279e9fb8bf64165'}),
    'f5b': ('fix5_safety_before', {'drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
                                   'assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6'}),
}
NOW_SHA = {'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
           'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'}
_MODS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_now():
    """今の版の sha256 を返す(直したあとは NOW_SHA と違ってよい)"""
    return {fn: sha(ROOT / 'jetformer' / fn) for fn in NOW_SHA}


def load(name):
    if name == 'now':
        name = 'base'
    if name in _MODS:
        return _MODS[name]
    if name in PSHA:
        d, sums = PROP / name, PSHA[name]
    else:
        dn, sums = SNAP[name]
        d = ROOT / 'tests' / 'fixtures' / dn
    for fn, h in sums.items():
        assert sha(d / fn) == h, (name, fn)
    aspec = importlib.util.spec_from_loader('jetformer._h8_assist_%s' % name, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((d / 'assist.py').read_text(encoding='utf-8'), str(d / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._h8_drive_%s' % name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((d / 'drive.py').read_text(encoding='utf-8'), str(d / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    _MODS[name] = (mod, amod)
    return mod, amod


class Car(CL.Car):
    """closed_loop.Car と同じ。scene['esc'] = (neu, rdelay) なら中立・下がり始めの秒だけ変える(fix6 r2 の exp2.EscCar と同じ)。
    scene['bounce'] = b なら、前の壁に当たったとき 当たった速さ × b で後ろへ跳ね返る(壁から離れる)"""

    def __init__(self, s):
        super().__init__(s)
        self.revlog = []
        self._bounce_v = 0.0

    def step(self, t, thr):
        s = self.s
        esc = s.get('esc')
        b = float(s.get('bounce', 0.0) or 0.0)
        if s.get('spring') is not None:
            return self._spring_step(t, thr)
        if esc is None and not b:
            r = super().step(t, thr)
            self.revlog.append((t, self.rev_on, self.contact))
            return r
        neu, rdelay = esc if esc is not None else (0.1, 0.08)
        tol = 1e-9 if esc is not None else 0.0
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
                if self.rev_wait >= rdelay - tol:
                    self.rev_on = True
            else:
                self.braked = True
                self.v = max(0.0, self.v - 6.0 * DT) if self.v > 0 else min(0.0, self.v + 6.0 * DT)
        else:
            self.rev_wait = 0.0
            self.neu_t += DT
            if self.braked and self.neu_t >= neu - tol:
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
                g = min(2.0 + 2.5 * self.v, 9.0) * s.get('front_soft', 1.0)
                self.hits.append((round(t, 3), 'front', round(self.v, 2), round(g, 2)))
                self.spike = [g, g]
                vb = -b * self.v
                nx, self.v = s['xf'], vb
                self.contact = 'front'
            else:
                nx, self.v = s['xf'], 0.0
                self.contact = 'front'
        if s.get('xr') is not None:
            if not self.rear_active and nx > s['xr'] + 0.1:
                self.rear_active = True
            if self.rear_active and nx <= s['xr']:
                if self.v < -0.05 and prev_con != 'rear':
                    g = min(2.0 + 2.5 * -self.v, 9.0) * s.get('rear_soft', 1.0)
                    self.hits.append((round(t, 3), 'rear', round(self.v, 2), round(g, 2)))
                    self.spike = [g, g]
                nx, self.v = s['xr'], 0.0
                self.contact = 'rear'
        self.x = nx
        acc = (self.v - v0) / DT
        g = self.spike.pop(0) if self.spike else 0.0
        self.revlog.append((t, self.rev_on, self.contact))
        return acc, g


    def _spring_step(self, t, thr):
        """scene['spring'] = (v, dur, push_s): 前の壁に押しつけて前進を push_s 秒以上出していたあと、前進でない値に変わったら、
        dur 秒だけ v [m/s] で壁から離れる(バンパー・タイヤのたわみの戻り)。ほかは closed_loop.Car と同じ"""
        vs, dur, push_s = self.s['spring']
        if not hasattr(self, '_push'):
            self._push, self._spr_until = 0.0, -1.0
        fwd = thr < N - 20.0
        if self.contact == 'front' and fwd:
            self._push += DT
        elif not fwd and self._push >= push_s and self.contact == 'front':
            self._spr_until = t + dur
            self._push = 0.0
        elif not fwd:
            self._push = 0.0
        if t <= self._spr_until + 1e-9:
            # ESC の状態は closed_loop.Car と同じに進め、速さだけ上書き
            v_save = self.v
            r = CL.Car.step(self, t, thr)
            self.v = -vs
            self.x = min(self.x, self.s['xf']) - vs * DT
            self.contact = None
            self.revlog.append((t, self.rev_on, self.contact))
            return (self.v - v_save) / DT, r[1]
        r = CL.Car.step(self, t, thr)
        self.revlog.append((t, self.rev_on, self.contact))
        return r


class Sim(CL.Sim):
    """closed_loop.Sim + 止まらない衝撃(scene['spikes'] = [(t, g)])・最初のフレームの時刻 scene['cam0']"""

    def __init__(self, d, sc_, bufs, cam_slot, box, pca, hook=None):
        super().__init__(d, sc_, bufs, cam_slot, box, pca, hook)
        self.car = Car(sc_)
        self.next_cam = sc_.get('cam0', 0.15)

    def _advance(self, t_end):
        sp = self.s.get('spikes') or []
        gl = self.s.get('glitch') or []      # [(t, erpm)]: その時刻の車両の値のサンプルの eRPM だけを差し替える(車は動かない)
        while self.t + DT <= t_end + 1e-9:
            nt = round(self.t + DT, 6)
            for ts, g in sp:
                if abs(nt - ts) < DT / 2 and not self.car.spike:
                    self.car.spike = [g, g]
            last = self.veh_buf[-1] if self.veh_buf else None
            if self.s.get('veh_dt'):
                self._tick(nt)
            else:
                super()._advance(nt)
            if gl and self.veh_buf and self.veh_buf[-1] is not last:
                for ts, e in gl:
                    if abs(nt - ts) < DT / 2:
                        self.veh_buf[-1][1]['erpm'] = int(e)
        if self.s.get('veh_dt'):
            self._tick(t_end)
        else:
            super()._advance(t_end)

    def _tick(self, t_end):
        """closed_loop.Sim._advance と同じ。ただし車両の値の間隔を scene['veh_dt'](本物は 100 Hz = 0.01)にする"""
        vdt = float(self.s['veh_dt'])
        while self.t + DT <= t_end + 1e-9:
            self.t = round(self.t + DT, 6)
            acc, g = self.car.step(self.t, self.pca.thr)
            if len(self.car.hits) > self.n_hits:
                self.n_hits = len(self.car.hits)
                if self.hit_drops:
                    dl, ln = self.hit_drops
                    self.drops.append((self.t + dl, self.t + dl + ln))
            t_ns = T0 + int(round(self.t * NS))
            ay = acc + g * 9.80665
            self.imu_buf.append((t_ns, np.array([0.0, ay, 9.8, 0.0, 0.0, 0.0], np.float32)))
            if self.t >= self.next_veh - 1e-9:
                self.next_veh += vdt
                if not any(a <= self.t < b for a, b in self.s.get('veh_skip', [])):
                    sl = self.sel(self.t)
                    self.veh_buf.append((t_ns, {'pwm': [1441, 1496, 2000 if sl else 1000], 'valid': [True, True, True],
                                                'select': sl, 'moving': abs(self.car.v) > 0.01,
                                                'erpm': int(round(-self.car.v / CL.KV))}))
            if self.t >= self.next_tof - 1e-9:
                self.next_tof += 1.0 / 30
                dist = self.s['xf'] - self.car.x
                self.tof_buf.append((t_ns, CL.tof_near(int(dist * 1000)) if 0.05 < dist < 1.95 else CL.TOF_FREE))
            self.log.append((self.t, self.car.x, self.car.v, self.pca.thr, self.car.contact, self.car.rev_on,
                             self.sel(self.t)))
        self.box['ns'] = T0 + int(round(self.t * NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']


def _wrap(orig, ph, force=None):
    """force = Recovery.step に上書きして渡す引数(例 {'esc_rev': None} = drive の EscTrack を使わない形)"""
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        if force:
            kw = dict(kw, **force)
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        ph.append((round(now - 100.0, 4), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_rev_s', 0.0)), 4), round(float(getattr(self, 'est_s', 0.0)), 4),
                   round(float(self.backed_s), 4), round(float(erpm), 1)))
        return r
    return step


def run(name, params, scene, wrap=True, floor=True, force=None, patch=None):
    mod, amod = load(name)
    TMP.mkdir(parents=True, exist_ok=True)
    ph = []
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        for k, v in (patch or {}).items():          # この走りの間だけ定数を変える(ファイルは変えない)
            if k == 'READY':
                mp.setattr(amod.EscMayReverse, 'READY_NEUTRAL_S', float(v))
            elif k == 'OFFTOO':
                assert hasattr(amod, 'P8_OFF_TOO'), name
                mp.setattr(amod, 'P8_OFF_TOO', bool(v))
            elif k == 'SURE':
                assert hasattr(amod, 'SURE_BACK_S'), name
                mp.setattr(amod, 'SURE_BACK_S', float(v))
            else:
                raise KeyError(k)
        if wrap:
            mp.setattr(amod.Recovery, 'step', _wrap(amod.Recovery.step, ph, force))
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
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = CL.TimedPCA(box)
        out = mod._Output(pca, config.PCA9685, (C, N), lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        sim = Sim(d, scene, bufs, cam_slot, box, pca)
        d._loop(d.st, CL.Runner(sim), fake_normalize, pca, out, (C, N), 6, 0.0, cam_slot, sim, *bufs, False)
        d._stop.clear()
    return sim, rec, pca, ph


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) == type(b) and a == b


def same_run(ra, rb):
    """PCA9685 に書いた値・記録の行(infer_ms のほか。欄の名前も)・出来事がビット単位で同じか → (同じか, 最初の違い)"""
    (sa, reca, pa, _), (sb, recb, pb, _) = ra, rb
    if pa.w != pb.w:
        i = next((k for k, (x, y) in enumerate(zip(pa.w, pb.w)) if x != y), min(len(pa.w), len(pb.w)))
        return False, ('pca', i, pa.w[i:i + 1], pb.w[i:i + 1])
    if len(reca.rows) != len(recb.rows):
        return False, ('rows_len', len(reca.rows), len(recb.rows))
    for (t0, r0), (t1, r1) in zip(reca.rows, recb.rows):
        if t0 != t1 or set(r0) != set(r1):
            return False, ('row_keys', (t0 - T0) / 1e9)
        for k in r0:
            if k != 'infer_ms' and not _same(r0[k], r1[k]):
                return False, ('row', (t0 - T0) / 1e9, k)
    if len(reca.events) != len(recb.events):
        return False, ('ev_len', len(reca.events), len(recb.events))
    for x, y in zip(reca.events, recb.events):
        if x[0] != y[0] or not _same(dict(x[1]), dict(y[1])):
            return False, ('ev', x, y)
    return True, None


def metrics(sim, rec, pca, t_from=0.0):
    """1 本の走りの数(sim.log = (t, x, v, thr, contact, rev_on, sel))"""
    log = sim.log
    m = collections.Counter()
    for (t0, x, v, thr, con, rev_on, sl), (t1, *_r) in zip(log, log[1:]):
        if t0 < t_from:
            continue
        dt = t1 - t0
        if con == 'front' and thr < N - 20 and sl:
            m['front_push_s'] += dt
        if con == 'rear' and thr > N + 40 and sl:
            m['rear_push_s'] += dt
        if thr > N + 40 and sl and abs(v) < 0.05 and con is None:
            m['rev_dead_s'] += dt
        if thr > N + 40 and sl:
            m['rev_side_s'] += dt
    rear = [h for h in sim.car.hits if h[1] == 'rear' and h[0] >= t_from]
    front = [h for h in sim.car.hits if h[1] == 'front' and h[0] >= t_from]
    gu = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_GIVEUP]
    out = {k: round(float(v), 3) for k, v in m.items()}
    out.update(rear=rear, rear_max_v=max([-h[2] for h in rear], default=0.0), front=front,
               front_max_v=max([h[2] for h in front], default=0.0), giveup_t=gu[0] if gu else None,
               min_x=round(min(x for _, x, *_ in log), 3),
               guards=[(e[1]['what'], e[1]['phase']) for e in rec.events if e[0] == 'recovery_guard'],
               n_rec_rows=sum(1 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY))
    for k in ('front_push_s', 'rear_push_s', 'rev_dead_s', 'rev_side_s'):
        out.setdefault(k, 0.0)
    return out


def ON(p):
    return dict(p, rec_back_guard=1, rec_hold_keep=1)


def OFF(p):
    return dict(p, rec_back_guard=0, rec_hold_keep=0)


def base_params(**kw):
    """11/3 の回 1〜4 の形(押すと: start_kick 1・lookahead 3・ToF ガード 1・rev_s 0.4・rev_us 300)"""
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, lookahead=3, tof_guard=1, throttle_gain=0.9,
             rev_s=0.4, rev_us=300.0, launch_boost=0, launch_no_kick=0, speed_ceiling=0, speed_auto=0, zone_caps=0,
             ceil_left_mps=4.1, rec_back_guard=0, rec_hold_keep=0)
    p.update(kw)
    return p


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)
    for q in (TMP.parent,):
        try:
            q.rmdir()
        except OSError:
            pass
