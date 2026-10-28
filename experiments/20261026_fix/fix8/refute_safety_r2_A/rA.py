"""票 A(反証 8-2・safety・rs8-2-reverse-end-stall-overrun を打ち消すつもりで確かめる)の部品。直さない・装置に触れない。

本物の drive._loop(今の jetformer/)を fix3/refute2_safety/closed_loop.py の Sim・run で回す。ESC の模型 18 通りは
fix8/refute_safety_r1_A/cl_A.py から写した(同じ式)。この台本が足したもの:
  - ESC の模型に fwdb(後ろへ動いている車に前進側を書いたときの減速 m/s²。前は 8 固定)と fdelay(後退のあとの前進側が
    効くまでの秒。その間は中立と同じ惰性)を足した版
  - カメラの位相(phase。最初のフレームを 0.15 + phase 秒に)
  - 処理の時間(proc。処理したフレームのあと、cam_evt.wait(0.1) が始まるまでの秒。本物は推論・JPEG・記録のぶん遅い)
  - カメラの止まりはフレームの番号で置く(drop_frames = [(最初の番号, 長さ)])
"""
import collections
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
FIXD = ROOT / 'experiments' / '20261026_fix'
for _p in (ROOT / 'tests', ROOT, FIXD / 'fix3' / 'refute2_safety'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import closed_loop as CL                   # noqa: E402
from jetformer import assist as A          # noqa: E402
from jetformer import drive                # noqa: E402

C, N, DT = CL.C, CL.N, CL.DT
TMP = FIXD / 'fix8' / 'tmp_refute_safety_r2_A' / os.environ.get('RA_TAG', 'main')
CL.TMP = TMP
FP = 1.0 / 30

NOW = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
       'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}

# --- ESC の模型(cl_A.py の写し) ---
ESC = {
    'fit': dict(acc='fo', A=3.0, Vmax=3.0, coast=0.4, neu_ready=0.1, rdelay=0.01, rdelay2=0.0, lag=0.0, brake=6.0),
    'cl': dict(acc='const', A=5.0, Vmax=3.0, coast=1.5, neu_ready=0.1, rdelay=0.08, rdelay2=0.08, lag=0.0, brake=6.0),
    'human': dict(acc='const', A=5.0, Vmax=3.0, coast=1.5, neu_ready=0.12, rdelay=0.10, rdelay2=0.10, lag=0.0, brake=6.0),
}


def _variants():
    f = ESC['fit']
    out = {}
    for k, vals in (('coast', (0.0, 1.5, 3.0)), ('rdelay', (0.05, 0.1)), ('rdelay2', (0.05, 0.1)), ('lag', (0.05, 0.1, 0.15)),
                    ('A', (2.0, 4.5)), ('Vmax', (2.5, 3.5))):
        for v in vals:
            out['fit_%s%s' % (k, v)] = dict(f, **{k: v})
    out['fit_const'] = dict(f, acc='const', A=7.0, Vmax=2.0, coast=1.5)
    return out


ESC.update(_variants())
ESC18 = sorted(ESC)
# この台本で足した: 後ろへの惰性を止める前進側の効き(fwdb)・効くまでの遅れ(fdelay)
for base in ('fit', 'cl', 'human'):
    ESC['%s_fwdb3' % base] = dict(ESC[base], fwdb=3.0)
    ESC['%s_fwd015' % base] = dict(ESC[base], fdelay=0.15)


class Car(CL.Car):
    def __init__(self, s):
        super().__init__(s)
        self.e = dict(s['esc'])
        self.ran = False
        self.hist = collections.deque()
        self.fwd_t = 0.0        # 前進側を書き続けた秒(fdelay 用)

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
        fwd, rev = thr < N - 20.0, thr > N + 40.0
        v0 = self.v
        if fwd:
            self.fwd_t += DT
        else:
            self.fwd_t = 0.0
        fdelay = float(e.get('fdelay', 0.0))
        if fwd and self.v < 0 and self.fwd_t < fdelay - 1e-9:
            # 前進側が効くまで: 中立の惰性と同じ(ESC の決まりは前進に移る)
            self.ready = self.braked = self.rev_on = False
            self.rev_wait = self.neu_t = 0.0
            c = e['coast']
            self.v = min(0.0, self.v + c * DT)
        elif fwd:
            self.ready = self.braked = self.rev_on = False
            self.rev_wait = self.neu_t = 0.0
            if self.v < 0:
                self.v = min(0.0, self.v + float(e.get('fwdb', 8.0)) * DT)
            else:
                vt = min((N - thr) / 80.0 * 1.2, 6.0)
                self.v += float(np.clip((vt - self.v) / 0.3, -6.0, 6.0)) * DT
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
                b = e['brake']
                self.v = max(0.0, self.v - b * DT) if self.v > 0 else min(0.0, self.v + b * DT)
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
        nx = self.x + self.v * DT
        prev_con = self.contact
        self.contact = None
        if nx >= s['xf']:
            if self.v > 0.05 and prev_con != 'front':
                g = min(2.0 + 2.5 * self.v, 9.0) * s.get('front_soft', 1.0)
                self.hits.append((round(t, 3), 'front', round(self.v, 2), round(g, 2)))
                self.spike = [g, g]
            nx, self.v = s['xf'], 0.0
            self.contact = 'front'
        self.x = nx
        acc = (self.v - v0) / DT
        g = self.spike.pop(0) if self.spike else 0.0
        return acc, g


class Sim(CL.Sim):
    """カメラの位相・処理の時間・フレームの番号で置く止まり"""

    def __init__(self, d, sc_, bufs, cam_slot, box, pca, hook=None):
        super().__init__(d, sc_, bufs, cam_slot, box, pca, hook)
        self.phase0 = 0.15 + float(sc_.get('phase', 0.0))
        self.next_cam = self.phase0
        self.idx = 0                    # 次のフレームの番号
        self.dropset = set()
        for i0, ln in sc_.get('drop_frames', []):
            self.dropset.update(range(int(i0), int(i0) + int(ln)))
        self.proc = float(sc_.get('proc', 0.0))
        self.after_frame = False
        self.frames = []                # (番号, 撮像時刻 s, 届いた時刻 s)

    def wait(self, timeout):
        if self.t >= self.s['dur']:
            self.d._stop.set()
            return False
        if self.hook is not None:
            self.hook(self.t, self.d)
        while self.idx in self.dropset:
            self.idx += 1
        cam_t_s = self.phase0 + self.idx * FP
        arrive = cam_t_s + 0.008
        if arrive - self.t > timeout + 1e-9:
            self._advance(self.t + timeout)
            return False
        # 届いたフレームの処理は届いた時刻 + proc で終わる(書く値・helper.step の now はそこ。wait(0.1) はそこから)
        self._advance(max(arrive, self.t) + self.proc)
        self.cam_slot[0] = CL.Sample(CL.T0 + int(round(cam_t_s * CL.NS)), {'image': np.zeros((8, 8, 3), np.uint8),
                                                                             'seq': self.seq})
        self.frames.append((self.idx, round(cam_t_s, 6), round(self.t, 6)))
        self.seq += 1
        self.idx += 1
        self.after_frame = True
        self.box['ns'] = CL.T0 + int(round(self.t * CL.NS))
        return True


CL.Car = Car
CL.Sim = Sim


def params(sw, **kw):
    """11/3 の回 1 の形(速さ 0.90・start_kick 1・lookahead 3・ToF ガード 1・rev_s 0.4・rev_us 300)"""
    p = dict(CL.PARAMS, recovery=1, esc_reverse=1, start_kick=1, lookahead=3, tof_guard=1, throttle_gain=0.9,
             rev_s=0.4, rev_us=300.0, launch_boost=0, launch_no_kick=0, speed_ceiling=0, speed_auto=0, zone_caps=0)
    p.update({'on': dict(rec_back_guard=1, rec_hold_keep=1), 'off': dict(rec_back_guard=0, rec_hold_keep=0),
              'guard': dict(rec_back_guard=1, rec_hold_keep=0)}[sw])
    p.update(kw)
    return p


def model_fwd(m):
    def f(t, car):
        return (C, N - m)
    return f


def model_brakewall(m, xf, db, brake_us=200.0):
    def f(t, car):
        return (C, N - m) if car.x < xf - db else (C, N + brake_us)
    return f


def run(esc, sw, model, xf=2.0, dur=6.0, phase=0.0, proc=0.0, drop_frames=(), **pkw):
    sc = {'dur': dur, 'xf': xf, 'ai_from': 1.0, 'model': model, 'esc': ESC[esc] if isinstance(esc, str) else esc,
          'phase': phase, 'proc': proc, 'drop_frames': list(drop_frames)}
    sim, rec, pca = CL.run(params(sw, **pkw), sc)
    return sim, rec, pca, list(CL.PH)


def first_rec(ph):
    t0 = next((t for t, p0, p1, *_ in ph if p0 is None and p1 == 'brake'), None)
    t1 = None
    if t0 is not None:
        t1 = next((t for t, p0, p1, *_ in ph if t > t0 and p0 is not None and p1 is None), None)
    return t0, t1


def rev_end(ph, t0):
    """最初のリカバリーで 'reverse' → 'coast'(か 'escape')に移ったフレームの時刻(step の now - 100 = 届いた時刻)"""
    for t, p0, p1, *_ in ph:
        if t0 is not None and t >= t0 and p0 == 'reverse' and p1 in ('coast', 'escape'):
            return t
    return None


def frame_at(sim, t_arr):
    for i, ct, at in sim.frames:
        if abs(at - t_arr) < 1e-6:
            return i
    return None


def profile(sim, t_rec, ds, win=3.5):
    """リカバリーの始めの位置 x0 から d [m] 後ろに物があったら最初に当たる速さ(物なしの走りから。当たらない = 0)"""
    log = [r for r in sim.log if t_rec - 1e-9 <= r[0] <= t_rec + win]
    x0 = log[0][1]
    hits = {}
    for d in ds:
        v = 0.0
        for t, x, vv, *_ in log:
            if x <= x0 - d:
                v = -vv if vv < -0.05 else 0.0
                break
        hits[d] = round(v, 3)
    back = round(x0 - min(r[1] for r in log), 4)
    vpk = round(-min(r[2] for r in log), 3)
    return x0, back, vpk, hits


def writes(pca, a, b):
    out, prev = [], None
    for tt, s, th in pca.w:
        if a <= tt <= b and th != prev:
            out.append((round(tt, 3), round(th, 1)))
            prev = th
    return out


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)
    try:
        TMP.parent.rmdir()
    except OSError:
        pass
