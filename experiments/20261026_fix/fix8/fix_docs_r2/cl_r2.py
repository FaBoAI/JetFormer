"""(fix8 係 docs 直す 8-2 回目: refute_docs_r2_A/cl_mine.py = fix8/refute_safety_r1_A/cl_A.py の写し。TMP だけ自分の場所に)票 A の閉じたループ: 本物の drive._loop(今の jetformer/)を fix3/refute2_safety/closed_loop.py の Sim・run で回し、
車(ESC の模型)だけを自分の Car に差し替える(CL.Car を入れ替える)。直さない・装置に触れない・API を呼ばない。

ESC の模型 esc = dict(
  acc='const'|'fo', A, Vmax: 後退で走っている間。const: |v| を A·s [m/s²] で Vmax·s まで。fo: dv/dt = −k·(Vmax·s + v)(A = k)
                   (s = (thr − 中立)/300)
  coast: 中立の帯の惰性の減速 [m/s²]
  neu_ready: ブレーキのあと中立の帯がこれだけ続いたら後退を受け付ける [s]
  rdelay: 受け付けているとき、止まって(か後ろへ動いて)いる車に後退側を書いてから走り出すまで [s]
  rdelay2: 後退で走ったあとの中立のあと、後退側でもう一度走り出すまで [s]
  lag: ESC が見る値の遅れ [s](書いた値が lag 秒あとに効く)
  brake: ブレーキの減速 [m/s²])
前進・ブレーキ・壁・後ろの物・衝撃は closed_loop.Car と同じ式。
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
TMP = FIXD / 'fix8' / 'tmp_docs_r2' / 'cl' / os.environ.get('RA_TAG', 'main')
CL.TMP = TMP

ESC = {
    # 10/23 No.6 の本物の記録に合わせた模型(fit1023.py: rmse 0.13 m/s・下がった距離 1.042 m / 本物 1.039 m)
    'fit': dict(acc='fo', A=3.0, Vmax=3.0, coast=0.4, neu_ready=0.1, rdelay=0.01, rdelay2=0.0, lag=0.0, brake=6.0),
    # closed_loop.py の既定の模型(fix3 から使っている形)
    'cl': dict(acc='const', A=5.0, Vmax=3.0, coast=1.5, neu_ready=0.1, rdelay=0.08, rdelay2=0.08, lag=0.0, brake=6.0),
    # 指摘の「人に寄せた ESC」(中立 0.12 秒・後退側から 0.10 秒)を自分の式で
    'human': dict(acc='const', A=5.0, Vmax=3.0, coast=1.5, neu_ready=0.12, rdelay=0.10, rdelay2=0.10, lag=0.0, brake=6.0),
}


def esc_variants():
    """fit のまわりを 1 つずつ動かした模型"""
    f = ESC['fit']
    out = {}
    for k, vals in (('coast', (0.0, 1.5, 3.0)), ('rdelay', (0.05, 0.1)), ('rdelay2', (0.05, 0.1)), ('lag', (0.05, 0.1, 0.15)),
                    ('A', (2.0, 4.5)), ('Vmax', (2.5, 3.5))):
        for v in vals:
            out['fit_%s%s' % (k, v)] = dict(f, **{k: v})
    out['fit_const'] = dict(f, acc='const', A=7.0, Vmax=2.0, coast=1.5)      # fit1023 の const のいちばん(rmse 0.18)
    return out


ESC.update(esc_variants())


class Car(CL.Car):
    def __init__(self, s):
        super().__init__(s)
        self.e = dict(s['esc'])
        self.ran = False
        self.hist = collections.deque()     # (t, thr) ESC が見る値の遅れ用

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
            self.ready = self.braked = self.rev_on = False
            self.rev_wait = self.neu_t = 0.0
            if self.v < 0:
                self.v = min(0.0, self.v + 8.0 * DT)
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
        if not self.rev_on and not (rev and self.ready) and not fwd and not rev:
            pass
        # 壁・後ろの物(closed_loop.Car と同じ)
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
        return acc, g


CL.Car = Car        # closed_loop.Sim は Car(sc_) をモジュールの名前で引く


def base_params(**kw):
    """11/3 の回 1 の形(速さ 0.90・start_kick 1・lookahead 3・ToF ガード 1・rev_s 0.4・rev_us 300)"""
    p = dict(CL.PARAMS, recovery=1, esc_reverse=1, start_kick=1, lookahead=3, tof_guard=1, throttle_gain=0.9,
             rev_s=0.4, rev_us=300.0, launch_boost=0, launch_no_kick=0, speed_ceiling=0, speed_auto=0, zone_caps=0,
             rec_back_guard=0, rec_hold_keep=0)
    p.update(kw)
    return p


SW = {'on': dict(rec_back_guard=1, rec_hold_keep=1), 'off': dict(rec_back_guard=0, rec_hold_keep=0),
      'guard': dict(rec_back_guard=1, rec_hold_keep=0)}


def brakewall(m, xf, db, brake_us=200.0):
    def f(t, car):
        return (C, N - m) if car.x < xf - db else (C, N + brake_us)
    return f


def run(esc, sw, m=120.0, xf=2.4, db=0.45, rear=None, veh_dt=0.01, dur=6.0, brake_us=200.0):
    sc = {'dur': dur, 'xf': xf, 'ai_from': 1.0, 'model': brakewall(m, xf, db, brake_us), 'esc': esc}
    if rear is not None:
        sc['xr'] = rear
    if veh_dt != 0.02:
        sc['veh_dt'] = veh_dt
    sim, rec, pca = _run(base_params(**SW[sw]), sc)
    return sim, rec, pca, list(CL.PH)


def _run(params, sc):
    # closed_loop.Sim は車両の値を 0.02 秒ごと。veh_dt を変えるときは Sim._advance の 0.02 を差し替える
    vdt = sc.get('veh_dt', 0.02)
    if vdt == 0.02:
        return CL.run(params, sc)
    orig = CL.Sim._advance

    def adv(self, t_end):
        while self.t + DT <= t_end + 1e-9:
            self.t = round(self.t + DT, 6)
            acc, g = self.car.step(self.t, self.pca.thr)
            t_ns = CL.T0 + int(round(self.t * CL.NS))
            ay = acc + g * 9.80665
            self.imu_buf.append((t_ns, np.array([0.0, ay, 9.8, 0.0, 0.0, 0.0], np.float32)))
            if self.t >= self.next_veh - 1e-9:
                self.next_veh += vdt
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
        self.box['ns'] = CL.T0 + int(round(self.t * CL.NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']
    CL.Sim._advance = adv
    try:
        return CL.run(params, sc)
    finally:
        CL.Sim._advance = orig


def first_rec(ph):
    """最初のリカバリーの始め(段 None → 'brake')の時刻 [s] と、そのあと段が None に戻った時刻"""
    t0 = next((t for t, p0, p1, *_ in ph if p0 is None and p1 == 'brake'), None)
    t1 = None
    if t0 is not None:
        t1 = next((t for t, p0, p1, *_ in ph if t > t0 and p0 is not None and p1 is None), None)
    return t0, t1


def hit_profile(sim, t_rec, t_end, ds):
    """後ろの物なしの走り: リカバリーの始めの位置から d [m] 後ろに物があったら、最初に当たる速さ [m/s](当たらない = None)。
    closed_loop.Car と同じく、その位置を越えた 10 ms の v(< −0.05 のときだけ当たり)"""
    log = [r for r in sim.log if t_rec - 1e-9 <= r[0] <= t_end + 1e-9]
    x0 = next(r[1] for r in sim.log if r[0] >= t_rec - 1e-9)
    out = {}
    for d in ds:
        xr = x0 - d
        v = None
        for t, x, vv, *_ in log:
            if x <= xr:
                v = -vv if vv < -0.05 else None
                break
        out[d] = None if v is None else round(v, 3)
    xmin = min(r[1] for r in log)
    vmin = min(r[2] for r in log)
    return out, x0, round(x0 - xmin, 3), round(-vmin, 3)


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)
    try:
        TMP.parent.rmdir()
    except OSError:
        pass
