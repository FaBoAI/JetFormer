"""反証 2 回目・safety: 本物の drive._loop を「閉じたループ」の偽物の車で回す(装置は開かない・直さない)。
PCA9685 に書いた値で偽物の ESC と車が動き、その結果が次のセンサー(eRPM・IMU・ToF)になる。前の壁・後ろの物(通り過ぎた
あとに現れる)・当たりの衝撃・カメラが来ない区間(当たりに続けて起きることも)・車両の値が来ない区間を入れる。

ESC の模型(assist.py の説明・tests/test_recovery_back_guard.esc_sim と同じ考え):
  前進側(< 中立 − 20): ready を消す。下がっていれば 8 m/s² で止める。前へは (中立 − thr)/80 × 1.2 m/s に 0.3 秒で寄る
  後退側(> 中立 + 40): 後退中ならそのまま後ろへ。ready かつ止まっている(v ≥ −0.075 の前)なら 0.08 秒あとから後ろへ。
                        ほかはブレーキ(6 m/s² で 0 へ)。ブレーキをしたしるし braked
  中立の帯: 0.1 秒続いて braked なら ready。惰性 1.5 m/s²
"""
import collections
import math
import sys
import tempfile
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))

from jetformer import assist as A          # noqa: E402
from jetformer import clock, config, drive  # noqa: E402
from jetformer import safety_caps as sc     # noqa: E402
from jetformer.sensors.base import Sample   # noqa: E402
from test_drive_caps import FakeRecorder, fake_normalize   # noqa: E402
from test_launch_drive import C, N, PARAMS, TOF_FREE, FLOOR_MM, tof_near, T0, NS   # noqa: E402

TMP = ROOT / 'experiments' / '20261026_fix' / 'fix3' / 'tmp_refute2_safety'
DT = 0.01
KV = sc.KV


class Car(object):
    def __init__(self, sc_):
        self.s = sc_
        self.x, self.v = 0.0, 0.0
        self.ready, self.braked, self.rev_on, self.rev_wait, self.neu_t = False, False, False, 0.0, 0.0
        self.rear_active = False
        self.hits = []          # (t, 'front'/'rear', speed, g)
        self.spike = []         # 残りの衝撃サンプル [g, ...]
        self.contact = None     # 'front' / 'rear' / None

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
                if self.rev_wait >= 0.08:
                    self.rev_on = True
            else:
                self.braked = True
                self.v = max(0.0, self.v - 6.0 * DT) if self.v > 0 else min(0.0, self.v + 6.0 * DT)
        else:
            self.rev_wait = 0.0
            self.neu_t += DT
            if self.braked and self.neu_t >= 0.1:
                self.ready, self.braked = True, False
            if self.rev_on:          # 後退のあとは中立 → 後退側ですぐ後ろへ(ready のまま)
                self.rev_on = False
                self.ready = True
            self.v = max(0.0, self.v - 1.5 * DT) if self.v > 0 else min(0.0, self.v + 1.5 * DT)
        # 壁
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


class TimedPCA(object):
    def __init__(self, box):
        self.box, self.w = box, []
        self.thr, self.steer = N, C

    def set_pair_us(self, sc_, s, tc, t):
        self.thr, self.steer = float(t), float(s)
        self.w.append(((self.box['ns'] - T0) / 1e9, float(s), float(t)))

    def set_us(self, ch, us):
        if ch == config.PCA9685['throttle_ch']:
            self.thr = float(us)

    def set_osc_scale(self, scale):
        pass


class Runner(object):
    def __init__(self, sim):
        self.sim = sim

    def run_numpy(self, **kw):
        st, th = self.sim.model()
        p = np.tile([[st, th]], (10, 1))
        return {'actions': ((np.asarray(p, np.float32) - drive.PWM_CENTER) / drive.PWM_SCALE)[None]}


class Sim(object):
    """cam_evt の代わり(wait/clear/set)。時計・センサー・車をまとめて進める"""

    def __init__(self, d, sc_, bufs, cam_slot, box, pca, hook=None):
        self.d, self.s, self.box, self.pca, self.hook = d, sc_, box, pca, hook
        self.tof_buf, self.imu_buf, self.veh_buf = bufs
        self.cam_slot = cam_slot
        self.car = Car(sc_)
        self.t = 0.0                    # 秒(T0 から)
        self.next_cam = 0.15
        self.next_veh = 0.0
        self.next_tof = 0.0
        self.seq = 0
        self.log = []                   # (t, x, v, thr, contact, rev_on, sel)
        self.drops = list(sc_.get('drops', []))
        self.hit_drops = sc_.get('hit_drop')      # 当たりに続くカメラの止まり (遅れ, 長さ)
        self.n_hits = 0

    def sel(self, t):
        return t >= self.s.get('ai_from', 1.0) and not any(a <= t < b for a, b in self.s.get('human', []))

    def model(self):
        f = self.s['model']
        return f(self.t, self.car) if callable(f) else f

    def dropped(self, t):
        return any(a <= t < b for a, b in self.drops)

    def _advance(self, t_end):
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
                self.next_veh += 0.02
                if not any(a <= self.t < b for a, b in self.s.get('veh_skip', [])):
                    sl = self.sel(self.t)
                    self.veh_buf.append((t_ns, {'pwm': [1441, 1496, 2000 if sl else 1000], 'valid': [True, True, True],
                                                'select': sl, 'moving': abs(self.car.v) > 0.01,
                                                'erpm': int(round(-self.car.v / KV))}))
            if self.t >= self.next_tof - 1e-9:
                self.next_tof += 1.0 / 30
                dist = self.s['xf'] - self.car.x
                self.tof_buf.append((t_ns, tof_near(int(dist * 1000)) if 0.05 < dist < 1.95 else TOF_FREE))
            self.log.append((self.t, self.car.x, self.car.v, self.pca.thr, self.car.contact, self.car.rev_on,
                             self.sel(self.t)))
        self.box['ns'] = T0 + int(round(self.t * NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']

    def wait(self, timeout):
        if self.t >= self.s['dur']:
            self.d._stop.set()
            return False
        if self.hook is not None:
            self.hook(self.t, self.d)
        # 来ないフレームは飛ばす
        while self.dropped(self.next_cam):
            self.next_cam += 1.0 / 30
        arrive = self.next_cam + 0.008
        if arrive - self.t > timeout + 1e-9:
            self._advance(self.t + timeout)
            return False
        self._advance(arrive)
        cam_t = T0 + int(round(self.next_cam * NS))
        self.cam_slot[0] = Sample(cam_t, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': self.seq})
        self.seq += 1
        self.next_cam += 1.0 / 30
        self.box['ns'] = T0 + int(round(self.t * NS))
        return True

    def clear(self):
        pass

    def set(self):
        pass


PH = []          # (t, 前の段, あとの段, hold_g)


def _wrap_step(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 4), p0, self.phase, round(float(kw.get('hold_g', 0.0)), 2), round(float(impact_g), 2)))
        return r
    return step


def run(params, scene, hook=None, floor=True):
    TMP.mkdir(parents=True, exist_ok=True)
    del PH[:]
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory(dir=str(TMP)) as td:
        mp.setattr(A.Recovery, 'step', _wrap_step(A.Recovery.step))
        tmp_path = Path(td)
        box = {'ns': T0, 'm': 100.0}
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(drive, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                        time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        if floor:
            np.save(fp, np.full((42, 54), float(FLOOR_MM), np.float32))
        mp.setattr(drive, 'FLOOR_REF_PATH', fp if floor else tmp_path / 'none.npy')
        rec = FakeRecorder()
        d = drive.Driver(hub=None, recorder=rec)
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = TimedPCA(box)
        out = drive._Output(pca, config.PCA9685, (C, N), lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        sim = Sim(d, scene, bufs, cam_slot, box, pca, hook)
        d._loop(d.st, Runner(sim), fake_normalize, pca, out, (C, N), 6, 0.0, cam_slot, sim, *bufs, False)
        d._stop.clear()
        return sim, rec, pca


def metrics(sim, rec):
    """押した秒・当たり・中止のあと前を押した秒"""
    log = sim.log
    m = collections.Counter()
    for (t0, x, v, thr, con, rev_on, sl), (t1, *_r) in zip(log, log[1:]):
        dt = t1 - t0
        if con == 'front' and thr < N - 20:
            m['front_push_s'] += dt
        if con == 'rear' and rev_on and thr > N + 40:
            m['rear_push_s'] += dt
        if thr > N + 40 and sl and v < -0.05:
            m['rev_while_backing_s'] += dt
        if thr > N + 40 and sl and v > 1.0:
            m['rev_side_while_fwd_fast_s'] += dt
    # 中止(ASSIST_GIVEUP)の行のあと、同じ AI の区間で前を押した秒
    gu = [t for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_GIVEUP]
    if gu:
        tg = (gu[0] - T0) / 1e9
        for (t0, x, v, thr, con, rev_on, sl), (t1, *_r) in zip(log, log[1:]):
            if t0 >= tg and not sl:
                break                       # 人側にした = 区間の終わり
            if t0 >= tg and con == 'front' and thr < N - 20:
                m['front_push_after_giveup_s'] += t1 - t0
    m['min_x'] = min(x for _, x, *_ in log)
    out = {k: round(float(v), 3) for k, v in m.items()}
    out['hits'] = sim.car.hits
    out['guard_events'] = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
    out['giveup_rows'] = len(gu)
    out['holds'] = sum(1 for t, r in rec.rows if r['hold'])
    return out
