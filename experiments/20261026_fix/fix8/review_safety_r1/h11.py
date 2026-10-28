"""反証 8-1 回目・係 safety(fix8/review_safety_r1)の閉じたループの部品(h11)。直さない・コードも文書も変えない。
本物の drive._loop(jetformer/ の今の版。写し 'f7b'・'r1'・'f6b'・'f5b' も読める)を、fix3/refute2_safety/closed_loop.py の
閉じたループ(偽物の ESC と車。PCA9685 に書いた値で車が動き、その結果が次の eRPM・IMU・ToF になる)で回す。
fix7/safety_r2/h9.py(= 反証 7-2 回目の h8 の写し)と同じ考えで、足したもの:
  - ESC: scene['esc'] = (中立で下がれる秒, 後退側から下がり始める秒)・scene['no_rev'] = True(後退に入らない ESC。後退側は
    いつもブレーキ)
  - 車: scene['bounce'](前の壁で跳ね返る)・scene['slope'](m/s²。負 = 後ろへ転がる坂)と scene['coast'](中立の惰性の減速。
    既定 1.5)。坂は前進・後退・中立のどれにも足す(ブレーキは 6 m/s² で 0 へ寄せる = 坂に勝つ)
  - 人側: scene['human'] = [(a, b)] の間は CH3 が人側(select False)。車に届くのは PCA9685 ではなく人の操作
    scene['human_thr'](f(t) か数。既定 中立)
  - センサー: 車両の値の間隔 scene['veh_dt'](既定 0.02 = 50 Hz。本物は 0.01)・eRPM の跳ね scene['glitch'] = [(t, eRPM)]・
    車両の値の止まり scene['veh_skip']・ToF の欠け scene['tof_skip']・IMU の欠け scene['imu_skip']・カメラの止まり
    scene['drops']・当たりに続くカメラの止まり scene['hit_drop'] = (遅れ, 長さ)・止まらない衝撃 scene['spikes']
  - モデル: scene['model'] = f(t, car) か (舵, スロットル)
装置に触れない・GPU を使わない・API を呼ばない。一時の場所は fix8/tmp_review_safety_r1(終わったら消す)。
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
TMP = FIXD / 'fix8' / 'tmp_review_safety_r1' / 'h' / os.environ.get('RS8_TAG', 'main')
for _p in (ROOT / 'tests', ROOT, FIXD / 'fix3' / 'refute2_safety'):
    sys.path.insert(0, str(_p))

from jetformer import assist as A          # noqa: E402
from jetformer import clock, config, drive  # noqa: E402
from jetformer.sensors.base import Sample   # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize   # noqa: E402
import test_launch_drive as TLD            # noqa: E402
import closed_loop as CL                   # noqa: E402

C, N, PARAMS, T0, NS = TLD.C, TLD.N, TLD.PARAMS, TLD.T0, TLD.NS
DT = CL.DT
KV = CL.KV
NOW_SHA = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
           'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}
SNAP = {
    'r1': ('fix7_safety_r2_before', {'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f',
                                     'assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f'}),
    'f7b': ('fix7_safety_before', {'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
                                    'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'}),
    'f6b': ('fix6_safety_before', {'drive.py': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c',
                                   'assist.py': '80f20890804170bede9c7df6e7f05892ab4e5397f8c5f837fda49ddba64d1276'}),
    'f5b': ('fix5_safety_before', {'drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
                                   'assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6'}),
}
_MODS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_now():
    got = {fn: sha(ROOT / 'jetformer' / fn) for fn in NOW_SHA}
    assert got == NOW_SHA, got
    return got


def load(name):
    if name == 'now':
        return drive, A
    if name in _MODS:
        return _MODS[name]
    dn, sums = SNAP[name]
    d = ROOT / 'tests' / 'fixtures' / dn
    for fn, h in sums.items():
        assert sha(d / fn) == h, (name, fn)
    aspec = importlib.util.spec_from_loader('jetformer._h11_assist_%s' % name, loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((d / 'assist.py').read_text(encoding='utf-8'), str(d / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._h11_drive_%s' % name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((d / 'drive.py').read_text(encoding='utf-8'), str(d / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    _MODS[name] = (mod, amod)
    return mod, amod


def in_any(t, spans):
    return any(a <= t < b for a, b in spans)


class Car(CL.Car):
    """closed_loop.Car と同じ ESC の模型 + scene['esc']・['no_rev']・['bounce']・['slope']・['coast']。
    slope・bounce・no_rev・coast が既定(0・0・False・1.5)で esc が無ければ closed_loop.Car と同じ値になる"""

    def __init__(self, s):
        super().__init__(s)
        self.revlog = []

    def step(self, t, thr):
        s = self.s
        esc = s.get('esc')
        neu, rdelay = esc if esc is not None else (0.1, 0.08)
        tol = 1e-9 if esc is not None else 0.0
        no_rev = bool(s.get('no_rev', False))
        coast = float(s.get('coast', 1.5))
        slope = float(s.get('slope', 0.0))
        if slope and s.get('slope_after_hit') and not self.hits:
            slope = 0.0                 # 坂は前の壁に当たったあとから(当たる前は平ら)
        b = float(s.get('bounce', 0.0) or 0.0)
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
            if slope:
                self.v += slope * DT
        elif rev:
            self.neu_t = 0.0
            if self.rev_on:
                a = (thr - N) / 300.0 * 5.0
                self.v = max(self.v - a * DT, -(thr - N) / 300.0 * 3.0)
                if slope:
                    self.v += slope * DT
            elif self.ready and self.v <= 0.075 and not no_rev:
                self.rev_wait += DT
                if self.rev_wait >= rdelay - tol:
                    self.rev_on = True
                if slope:
                    self.v += slope * DT
            else:
                self.braked = True
                if slope:
                    self.v += slope * DT
                self.v = max(0.0, self.v - 6.0 * DT) if self.v > 0 else min(0.0, self.v + 6.0 * DT)
        else:
            self.rev_wait = 0.0
            self.neu_t += DT
            if self.braked and self.neu_t >= neu - tol:
                self.ready, self.braked = True, False
            if self.rev_on:
                self.rev_on = False
                self.ready = True
            if slope:
                self.v += slope * DT
            self.v = max(0.0, self.v - coast * DT) if self.v > 0 else min(0.0, self.v + coast * DT)
        nx = self.x + self.v * DT
        prev_con = self.contact
        self.contact = None
        if nx >= s['xf']:
            if self.v > 0.05 and prev_con != 'front':
                g = min(2.0 + 2.5 * self.v, 9.0) * s.get('front_soft', 1.0)
                self.hits.append((round(t, 3), 'front', round(self.v, 2), round(g, 2)))
                self.spike = [g, g]
                nx, self.v = s['xf'], -b * self.v
            else:
                nx, self.v = s['xf'], min(self.v, 0.0) if b else 0.0
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


class Sim(CL.Sim):
    def __init__(self, d, sc_, bufs, cam_slot, box, pca, hook=None):
        super().__init__(d, sc_, bufs, cam_slot, box, pca, hook)
        self.car = Car(sc_)
        self.next_cam = sc_.get('cam0', 0.15)
        self.cam_timeouts = []

    def wait(self, timeout):
        r = super().wait(timeout)
        if not r and not self.d._stop.is_set():
            self.cam_timeouts.append(round(self.t, 3))
        return r

    def human_thr(self, t):
        h = self.s.get('human_thr', N)
        return float(h(t)) if callable(h) else float(h)

    def _advance(self, t_end):
        sp = self.s.get('spikes') or []
        gl = self.s.get('glitch') or []
        vdt = float(self.s.get('veh_dt', 0.02))
        while self.t + DT <= t_end + 1e-9:
            self.t = round(self.t + DT, 6)
            for ts, g in sp:
                if abs(self.t - ts) < DT / 2 and not self.car.spike:
                    self.car.spike = [g, g]
            sl = self.sel(self.t)
            thr = self.pca.thr if sl else self.human_thr(self.t)
            lag = float(self.s.get('lag', 0.0) or 0.0)
            if lag:                     # ESC が指令に遅れて応える(scene['lag'] 秒前の指令で動く)
                if not hasattr(self, '_lagq'):
                    self._lagq = collections.deque()
                self._lagq.append((self.t, thr))
                while len(self._lagq) > 1 and self._lagq[1][0] <= self.t - lag + 1e-9:
                    self._lagq.popleft()
                thr = self._lagq[0][1] if self._lagq[0][0] <= self.t - lag + 1e-9 else N
            acc, g = self.car.step(self.t, thr)
            if len(self.car.hits) > self.n_hits:
                self.n_hits = len(self.car.hits)
                if self.hit_drops:
                    dl, ln = self.hit_drops
                    self.drops.append((self.t + dl, self.t + dl + ln))
            t_ns = T0 + int(round(self.t * NS))
            ay = acc + g * 9.80665
            if not in_any(self.t, self.s.get('imu_skip', [])):
                self.imu_buf.append((t_ns, np.array([0.0, ay, 9.8, 0.0, 0.0, 0.0], np.float32)))
            if self.t >= self.next_veh - 1e-9:
                self.next_veh += vdt
                if not in_any(self.t, self.s.get('veh_skip', [])):
                    e = int(round(-self.car.v / KV))
                    for ts, ge in gl:
                        if abs(self.t - ts) < DT / 2:
                            e = int(ge)
                    self.veh_buf.append((t_ns, {'pwm': [1441, 1496, 2000 if sl else 1000], 'valid': [True, True, True],
                                                'select': sl, 'moving': abs(self.car.v) > 0.01, 'erpm': e}))
            if self.t >= self.next_tof - 1e-9:
                self.next_tof += 1.0 / 30
                if not in_any(self.t, self.s.get('tof_skip', [])):
                    dist = self.s['xf'] - self.car.x
                    self.tof_buf.append((t_ns, CL.tof_near(int(dist * 1000)) if 0.05 < dist < 1.95 else CL.TOF_FREE))
            self.log.append((self.t, self.car.x, self.car.v, thr, self.car.contact, self.car.rev_on, sl))
        self.box['ns'] = T0 + int(round(self.t * NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']


def _wrap(orig, ph, force=None):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        if force:
            kw = dict(kw, **force)
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        ph.append((round(now - 100.0, 4), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 4), round(float(self.backed_s), 4), round(float(erpm), 1),
                   kw.get('esc_rev'), round(float(impact_g), 2), round(float(kw.get('hold_g', 0.0)), 2)))
        return r
    return step


def run(name, params, scene, wrap=True, force=None, patch=None):
    """→ (sim, rec, pca, ph, stats)"""
    mod, amod = load(name)
    TMP.mkdir(parents=True, exist_ok=True)
    ph = []
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        for k, v in (patch or {}).items():
            obj, attr = k.split('.') if '.' in k else (None, k)
            mp.setattr(getattr(amod, obj) if obj else amod, attr, v)
        if wrap:
            mp.setattr(amod.Recovery, 'step', _wrap(amod.Recovery.step, ph, force))
        tmp_path = Path(td)
        box = {'ns': T0, 'm': 100.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                      time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        np.save(fp, np.full((42, 54), float(CL.FLOOR_MM), np.float32))
        mp.setattr(mod, 'FLOOR_REF_PATH', fp)
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
        stats = dict(d.st.get('stats') or {})
    return sim, rec, pca, ph, stats


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
    sa, reca, pa = ra[:3]
    sb, recb, pb = rb[:3]
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
            return False, ('ev', x[0], y[0])
    return True, None


ESC_FWD = N - A.Recovery.ESCAPE_US


def recov_segments(res):
    """リカバリーの手順ごと(PCA9685 に書いた値の列から): 'brake' の始め(中立 + rev_us 以上の後退側を書き始めた AI 側の時刻で、
    前の値が後退側でない)〜 次の escape の前進(中立 − 130)の終わり。戻り値 [{t0, x0, min_x, back, t_esc, esc_push, ...}]"""
    sim, rec, pca = res[:3]
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows]
    segs = []
    cur = None
    for (t, r) in rows:
        if r['hold']:
            continue
        k = int(r['assist'])
        if k == A.ASSIST_RECOVERY and cur is None:
            cur = {'t0': round(t, 3)}
        elif k != A.ASSIST_RECOVERY and cur is not None:
            cur['t1'] = round(t, 3)
            cur['end_kind'] = k
            segs.append(cur)
            cur = None
    if cur is not None:
        cur['t1'] = None
        cur['end_kind'] = None
        segs.append(cur)
    log = sim.log
    xf = sim.s['xf']
    for sg in segs:
        t0, t1 = sg['t0'], sg['t1'] if sg['t1'] is not None else log[-1][0]
        part = [r for r in log if t0 <= r[0] <= t1]
        if not part:
            continue
        x0 = part[0][1]
        sg['x0'] = round(x0, 3)
        sg['back'] = round(x0 - min(r[1] for r in part), 3)
        sg['vmin'] = round(min(r[2] for r in part), 3)
        sg['esc_push'] = round(sum(DT for r in part if r[4] == 'front' and r[3] < N - 20 and r[6]), 3)
        sg['rev_side_s'] = round(sum(DT for r in part if r[3] > N + 40 and r[6]), 3)
        sg['rev_dead_s'] = round(sum(DT for r in part if r[3] > N + 40 and r[6] and abs(r[2]) < 0.05 and r[4] is None), 3)
        sg['at_wall'] = abs(x0 - xf) < 0.02
    return segs


def metrics(res, t_from=0.0):
    sim, rec, pca, ph, stats = res
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
        if thr > N + 40 and sl:
            m['rev_side_s'] += dt
        if thr > N + 40 and sl and abs(v) < 0.05 and con is None:
            m['rev_dead_s'] += dt
        if sl and v < -0.05:
            m['back_move_s'] += dt
    rear = [h for h in sim.car.hits if h[1] == 'rear' and h[0] >= t_from]
    front = [h for h in sim.car.hits if h[1] == 'front' and h[0] >= t_from]
    gu = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_GIVEUP]
    segs = recov_segments(res)
    out = {k: round(float(v), 3) for k, v in m.items()}
    for k in ('front_push_s', 'rear_push_s', 'rev_side_s', 'rev_dead_s', 'back_move_s'):
        out.setdefault(k, 0.0)
    out.update(rear=rear, rear_n=len(rear), rear_max_v=max([-h[2] for h in rear], default=0.0), front=front,
               front_n=len(front), front_max_v=max([h[2] for h in front], default=0.0),
               giveup_t=round(gu[0], 3) if gu else None, giveup_rows=len(gu),
               min_x=round(min(r[1] for r in log), 3), x_end=round(log[-1][1], 3), v_end=round(log[-1][2], 3),
               guards=[(e[1]['what'], e[1]['phase']) for e in rec.events if e[0] == 'recovery_guard'],
               recoveries=int(stats.get('recoveries', 0)), holds=int(stats.get('holds', 0)),
               segs=segs, zero_push=sum(1 for s in segs if s.get('at_wall') and s.get('back', 1) < 0.1
                                        and s.get('esc_push', 0) >= 0.05),
               launch_frames=sum(1 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_LAUNCH),
               kick_frames=sum(1 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_START))
    return out


def ON(p):
    return dict(p, rec_back_guard=1, rec_hold_keep=1)


def OFF(p):
    return dict(p, rec_back_guard=0, rec_hold_keep=0)


FORMS = {   # 11/3 の回 1〜4 の形(押すと速さ 0.90・start_kick 1・lookahead 3・ToF ガード 1・rev_s 0.4・rev_us 300)
    'kai1': {},
    'kai1_ceil': {'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1},
    'kai2_auto42': {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.2},
    'kai3_lb_ceil': {'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'launch_boost': 1},
    'kai4_lb': {'launch_boost': 1},
}


def base_params(**kw):
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, lookahead=3, tof_guard=1, throttle_gain=0.9,
             rev_s=0.4, rev_us=300.0, launch_boost=0, launch_no_kick=0, speed_ceiling=0, speed_auto=0, zone_caps=0,
             ceil_left_mps=4.1, rec_back_guard=0, rec_hold_keep=0)
    p.update(kw)
    return p


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)
    for q in (TMP.parent, TMP.parent.parent):
        try:
            q.rmdir()
        except OSError:
            pass
