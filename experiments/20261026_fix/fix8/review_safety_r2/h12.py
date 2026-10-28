"""反証 8-2 回目・係 safety(fix8/review_safety_r2)の閉じたループの部品(h12)。直さない・コードも文書も変えない。
8-1 の h11(fix8/review_safety_r1/h11.py。本物の drive._loop を fix3 closed_loop の偽物の ESC と車で回す)を読み込んで使い、
足したもの:
  - ESC の模型 18 通り(fix8/refute_safety_r1_A/cl_A.py の ESC と esc_variants をそのまま写した。cl_A を import すると
    closed_loop.Car を差し替えるので、式だけ写した): scene['esc18'] = 模型の名前。後ろの物・跳ね返り・後ろ向きの坂も効く
  - 記録の照らし: 記録の行(add_ai)ごとに、その時点の PCA9685 の書き込みの数と最後の値を残す(RecX)
  - コース(1 次元。fix8/decided_safety/h10.py の TrackSim と同じ考え): scene['corners'] があれば、ToF は次の角の壁を見る・
    舵と速さからヨーと横の加速(速さの天井の左・周の数え・発進の前進の ③ が動く)。前の壁 scene['xf'] は物理の当たり
  - 人側: h11 と同じ(scene['human'] = [(a, b)]・scene['human_thr'])
装置に触れない・GPU を使わない・API を呼ばない。一時の場所は fix8/tmp_review_safety_r2(終わったら消す)。
"""
import collections
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / 'review_safety_r1'
sys.path.insert(0, str(R1))
import h11  # noqa: E402

ROOT = h11.ROOT
FIXD = h11.FIXD
TMP = FIXD / 'fix8' / 'tmp_review_safety_r2' / 'h' / os.environ.get('RS82_TAG', 'main')
h11.TMP = TMP
CL = h11.CL
A = h11.A
C, N, T0, NS, DT, KV = h11.C, h11.N, h11.T0, h11.NS, h11.DT, h11.KV

# ---- ESC の模型 18 通り(cl_A.py の ESC・esc_variants の写し)
ESC18 = {
    'fit': dict(acc='fo', A=3.0, Vmax=3.0, coast=0.4, neu_ready=0.1, rdelay=0.01, rdelay2=0.0, lag=0.0, brake=6.0),
    'cl': dict(acc='const', A=5.0, Vmax=3.0, coast=1.5, neu_ready=0.1, rdelay=0.08, rdelay2=0.08, lag=0.0, brake=6.0),
    'human': dict(acc='const', A=5.0, Vmax=3.0, coast=1.5, neu_ready=0.12, rdelay=0.10, rdelay2=0.10, lag=0.0, brake=6.0),
}


def _variants():
    f = ESC18['fit']
    out = {}
    for k, vals in (('coast', (0.0, 1.5, 3.0)), ('rdelay', (0.05, 0.1)), ('rdelay2', (0.05, 0.1)), ('lag', (0.05, 0.1, 0.15)),
                    ('A', (2.0, 4.5)), ('Vmax', (2.5, 3.5))):
        for v in vals:
            out['fit_%s%s' % (k, v)] = dict(f, **{k: v})
    out['fit_const'] = dict(f, acc='const', A=7.0, Vmax=2.0, coast=1.5)
    return out


ESC18.update(_variants())
assert len(ESC18) == 18, len(ESC18)


class Car18(CL.Car):
    """cl_A.Car と同じ式(ESC の模型 scene['esc18'])。足したもの: scene['bounce'](前の壁の跳ね返り)・scene['slope'](m/s²。
    負 = 後ろへ転がる。scene['slope_after_hit'] なら前の壁に当たったあとから)・scene['volt'](前進の速さ・加速の倍率)。
    どれも既定なら cl_A.Car と同じ値"""

    def __init__(self, s):
        super().__init__(s)
        self.e = dict(ESC18[s['esc18']])
        self.ran = False
        self.hist = collections.deque()
        self.revlog = []

    def step(self, t, thr_now):
        s, e = self.s, self.e
        lag = float(e.get('lag', 0.0))
        if lag > 0:
            self.hist.append((t, thr_now))
            thr = N
            while self.hist and self.hist[0][0] <= t - lag + 1e-9:
                thr = self.hist[0][1]
                if len(self.hist) > 1 and self.hist[1][0] <= t - lag + 1e-9:
                    self.hist.popleft()
                else:
                    break
            if not self.hist or self.hist[0][0] > t - lag + 1e-9:
                thr = getattr(self, '_last_thr', N)
            self._last_thr = thr
        else:
            thr = thr_now
        slope = float(s.get('slope', 0.0) or 0.0)
        if slope and s.get('slope_after_hit') and not self.hits:
            slope = 0.0
        volt = float(s.get('volt', 1.0))
        b = float(s.get('bounce', 0.0) or 0.0)
        fwd, rev = thr < N - 20.0, thr > N + 40.0
        v0 = self.v
        if fwd:
            self.ready = self.braked = self.rev_on = False
            self.rev_wait = self.neu_t = 0.0
            if self.v < 0:
                self.v = min(0.0, self.v + 8.0 * DT)
            else:
                vt = min((N - thr) / 80.0 * 1.2, 6.0) * volt
                self.v += float(np.clip((vt - self.v) / 0.3, -6.0 * volt, 6.0 * volt)) * DT
        elif rev:
            self.neu_t = 0.0
            sc = (thr - N) / 300.0
            if self.rev_on:
                if e['acc'] == 'const':
                    self.v = max(self.v - e['A'] * sc * DT, -e['Vmax'] * sc)
                else:
                    self.v = self.v - e['A'] * (e['Vmax'] * sc + self.v) * DT
            elif self.ready and self.v <= 0.075:
                self.rev_wait += DT
                if self.rev_wait >= (e['rdelay2'] if self.ran else e['rdelay']) - 1e-9:
                    self.rev_on, self.ran = True, True
            else:
                self.braked = True
                bb = e['brake']
                self.v = max(0.0, self.v - bb * DT) if self.v > 0 else min(0.0, self.v + bb * DT)
        else:
            self.rev_wait = 0.0
            self.neu_t += DT
            if self.braked and self.neu_t >= e['neu_ready'] - 1e-9:
                self.ready, self.braked = True, False
            if self.rev_on:
                self.rev_on = False
                self.ready = True
            c = e['coast']
            self.v = max(0.0, self.v - c * DT) if self.v > 0 else min(0.0, self.v + c * DT)
        if slope:
            self.v += slope * DT
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


_SimBase = h11.Sim


class Sim12(_SimBase):
    """h11.Sim + ESC 18 模型 + コース(corners)。h11.run がこのクラスを使う(h11.Sim を差し替える)"""

    def __init__(self, d, sc_, bufs, cam_slot, box, pca, hook=None):
        super().__init__(d, sc_, bufs, cam_slot, box, pca, hook)
        if sc_.get('esc18'):
            self.car = Car18(sc_)
        self.hook12 = sc_.get('hook')

    def wait(self, timeout):
        if self.hook12 is not None:
            self.hook12(self.t, self.d)
        return super().wait(timeout)

    def wall_ahead(self):
        x = self.car.x
        gap = float(self.s.get('wall_gap', 0.6))
        for c in self.s.get('corners', []):
            if x < c:
                return c + gap - x
        return None

    def _advance(self, t_end):
        if not self.s.get('corners'):
            return super()._advance(t_end)
        sp = self.s.get('spikes') or []
        gl = self.s.get('glitch') or []
        vdt = float(self.s.get('veh_dt', 0.01))
        while self.t + DT <= t_end + 1e-9:
            self.t = round(self.t + DT, 6)
            for ts, g in sp:
                if abs(self.t - ts) < DT / 2 and not self.car.spike:
                    self.car.spike = [g, g]
            sl = self.sel(self.t)
            thr = self.pca.thr if sl else self.human_thr(self.t)
            steer = self.pca.steer if sl else C
            acc, g = self.car.step(self.t, thr)
            if len(self.car.hits) > self.n_hits:
                self.n_hits = len(self.car.hits)
                if self.hit_drops:
                    dl, ln = self.hit_drops
                    self.drops.append((self.t + dl, self.t + dl + ln))
            v = self.car.v
            yaw = float(np.clip((C - steer) / 400.0 * 45.0 * max(v, 0.0), -250.0, 250.0))
            t_ns = T0 + int(round(self.t * NS))
            ay = acc + g * 9.80665
            lat = math.radians(yaw) * max(v, 0.0)
            if not h11.in_any(self.t, self.s.get('imu_skip', [])):
                self.imu_buf.append((t_ns, np.array([lat, ay, 9.8, 0.0, 0.0, math.radians(yaw)], np.float32)))
            if self.t >= self.next_veh - 1e-9:
                self.next_veh += vdt
                if not h11.in_any(self.t, self.s.get('veh_skip', [])):
                    e = int(round(-v / KV))
                    for ts, ge in gl:
                        if abs(self.t - ts) < DT / 2:
                            e = int(ge)
                    self.veh_buf.append((t_ns, {'pwm': [1441, 1496, 2000 if sl else 1000], 'valid': [True, True, True],
                                                'select': sl, 'moving': abs(v) > 0.01, 'erpm': e}))
            if self.t >= self.next_tof - 1e-9:
                self.next_tof += 1.0 / 30
                if not h11.in_any(self.t, self.s.get('tof_skip', [])):
                    dist = self.wall_ahead()
                    if dist is None:
                        dist = self.s['xf'] - self.car.x
                    self.tof_buf.append((t_ns, CL.tof_near(int(dist * 1000)) if 0.05 < dist < 1.95 else CL.TOF_FREE))
            self.log.append((self.t, self.car.x, self.car.v, thr, self.car.contact, self.car.rev_on, sl))
        self.box['ns'] = T0 + int(round(self.t * NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']


h11.Sim = Sim12

# ---- 記録の照らし(行ごとに PCA9685 の書き込みの数と最後の値)
_CUR = {'pca': None}
_PCABase = CL.TimedPCA


class PCAX(_PCABase):
    def __init__(self, box):
        super().__init__(box)
        _CUR['pca'] = self


class RecX(h11.FakeRecorder):
    def add_ai(self, t_ns, data):
        p = _CUR['pca']
        self.rows.append((t_ns, data))
        if not hasattr(self, 'wmark'):
            self.wmark = []
        self.wmark.append((len(p.w), p.w[-1] if p.w else None))


CL.TimedPCA = PCAX
h11.FakeRecorder = RecX


def run(params, scene, ver='now', wrap=True):
    return h11.run(ver, params, scene, wrap=wrap)


def check_rows(res):
    """記録の行と PCA9685 に書いた値が合うか(行ごと)。→ 合わない行の [(秒, 何)]"""
    sim, rec, pca = res[:3]
    bad = []
    prev_n = 0
    for (t_ns, r), (n, last) in zip(rec.rows, getattr(rec, 'wmark', [])):
        t = round((t_ns - T0) / 1e9, 3)
        o = tuple(float(x) for x in r['out'])
        if n == prev_n:
            bad.append((t, 'no_write'))
        elif last is None or (round(last[1], 6), round(last[2], 6)) != (round(o[0], 6), round(o[1], 6)):
            bad.append((t, 'out_ne_write', o, last))
        prev_n = n
        k = int(r['assist'])
        if bool(r.get('launch_on')) != (k == A.ASSIST_LAUNCH and not r['hold']):
            bad.append((t, 'launch_on_ne_assist5', k, bool(r.get('launch_on'))))
        if r['hold'] and (k != A.ASSIST_NONE or o[1] != N):
            bad.append((t, 'hold_row', k, o))
    return bad


def phase_at(ph, t):
    """t より前の最後の Recovery.step のあとの段(ph は h11 の _wrap の並び)"""
    last = None
    for x in ph:
        if x[0] <= t + 1e-9:
            last = x
        else:
            break
    return None if last is None else last[2]


def ON(p):
    return h11.ON(p)


def OFF(p):
    return h11.OFF(p)


def base_params(**kw):
    return h11.base_params(**kw)


FORMS = dict(h11.FORMS)


def cleanup():
    h11.cleanup()
    for q in (TMP.parent, TMP.parent.parent):
        try:
            q.rmdir()
        except OSError:
            pass
