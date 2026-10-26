"""反証 V-drive: 閉ループの偽物の車とコース(走行ループが書いたスロットルで速さが変わる)。
コース(左回り、1 周 約 29.7 m・向き +360 度。tests の LAP_FAST と同じ形を距離で):
  R1 右 −180 度 1.7 m → GM 左 +160 度 2.6 m → 直線 +60 度 9.4 m → HS/LH 左 +135 度 6.45 m → +40 度 4.4 m
  → SW 左 +145 度 4.1 m → 直線 1.05 m
前後: 1 次の遅れ τ 0.32 秒・むだ時間 30 ms・駆動の最大 4.12 m/s²。定常の eRPM = b × (26.314 × 前進の指令 + 1343.5)。
中立・ブレーキでは惰性 0.58 m/s² + こすれ 0.40·v²·|κ|。b = 電池の強さ(1.0 = No.9 の満充電の仮定)。
モデルの偽物: 区間ごとの前進の指令(µs)× 曲がりで減らす + 舵 = 中心 + 320 × κ(0.3 m 先)。
"""
import math

import numpy as np

from jetformer import safety_caps as sc

KV = sc.KV
SEG = [(1.7, -180.0, 'R1'), (2.6, 160.0, 'GM'), (9.4, 60.0, 'DE'), (6.45, 135.0, 'HS'), (4.4, 40.0, 'LH'),
       (4.1, 145.0, 'SW'), (1.05, 0.0, 'X')]
L = sum(s[0] for s in SEG)
NS = 1_000_000_000


def seg_at(s):
    s = s % L
    acc = 0.0
    for ln, deg, name in SEG:
        if s < acc + ln:
            return math.radians(deg) / ln, name
        acc += ln
    return 0.0, SEG[-1][2]


# 区間ごとのモデルの前進の指令 [µs](gain 1.0 の前)。直線は強く、R1・GM は弱く
F_MODEL = {'R1': 140.0, 'GM': 170.0, 'DE': 330.0, 'HS': 320.0, 'LH': 320.0, 'SW': 300.0, 'X': 60.0}


class SimSource(object):
    def __init__(self, t_end_s=60.0, b=1.0, vbat=None, select=None, drop=(), bump_g=0.0, neutral=1496.0,
                 center=1441.0, f_scale=1.0, brake_at=(), seed=0, stand_s=1.0, crash=(), lap_noise=0.0):
        self.t0 = 1_800_000_000 * NS
        self.t_end = t_end_s
        self.b, self.vbat, self.select, self.drop = b, vbat, select, drop
        self.bump_g = bump_g
        self.neutral, self.center = neutral, center
        self.f_scale = f_scale
        self.brake_at = brake_at
        self.rng = np.random.RandomState(seed)
        self.stand_s = stand_s
        self.lap_noise = lap_noise
        self.lap_fac = np.exp(np.random.RandomState(seed + 7).normal(0, lap_noise, 200)) if lap_noise else np.ones(200)
        self.crash = crash               # [(t, 止める秒)]: 水平 4G の衝撃 → 止まる(人が持つ)
        self.reset()

    def reset(self):
        self.t = 0.0                        # [秒]
        self.s, self.v, self.psi = 0.6, 0.0, 0.0     # R1 の手前から
        self.cmd_hist = [(-1.0, self.neutral)]
        self.queue_imu, self.queue_veh, self.queue_tof = [], [], []
        self.pca = None
        self.n_pairs = 0
        self.log = []                       # (t, s, v, seg)
        self.laps = []                      # 周の区切りの時刻(位置が L の倍数を越えたとき)
        self.lap_k = 0

    def attach(self, pca, ref):
        self.reset()
        self.pca = pca
        self.neutral = ref[1]
        self.center = ref[0]

    def _keep(self, ch, t):
        return not any(a <= t < b and (c is None or c == ch) for c, a, b in self.drop)

    def _cmd_at(self, t):
        u = self.neutral
        for tt, x in self.cmd_hist:
            if tt <= t:
                u = x
            else:
                break
        return u

    def _step(self, dt):
        """物理を dt 進める(書いたスロットルは 30 ms 遅れて効く)"""
        for tc, dur in self.crash:
            if tc <= self.t < tc + dur:
                self.v = 0.0
                kap, name = seg_at(self.s)
                return 0.0, kap, name
        thr = self._cmd_at(self.t - 0.030)
        f = self.neutral - thr
        kap, name = seg_at(self.s)
        if f > 0:
            v_ss = max(0.0, self.b * (26.314 * f + 1343.5) * KV)
            a = (v_ss - self.v) / 0.32
            a = min(a, 4.12)
            if a < 0:                       # 指令より速い: 惰性で落ちる分だけ(ESC はブレーキしない)
                a = max(a, -(0.58 + 0.40 * self.v * self.v * abs(kap)))
        elif f < 0:
            a = -3.0 if self.v > 0.05 else 0.0       # ブレーキ
        else:
            a = -(0.58 + 0.40 * self.v * self.v * abs(kap)) if self.v > 0.02 else 0.0
        a_eff = a if self.v > 0 or a > 0 else 0.0
        if self.v > 0:
            a_eff -= 0.40 * self.v * self.v * abs(kap) * (1.0 if f > 0 else 0.0)
        self.v = max(0.0, self.v + a_eff * dt)
        s_old = self.s
        self.s += self.v * dt
        if int((self.s - 0.6 + 0.0) // L) > int((s_old - 0.6) // L):
            self.laps.append(self.t)
        self.psi += self.v * kap * dt
        return a_eff, kap, name

    def _advance_to(self, t_target):
        if self.pca is not None and len(self.pca.pairs) > self.n_pairs:
            for p in self.pca.pairs[self.n_pairs:]:
                self.cmd_hist.append((self.t, p[1]))
            self.n_pairs = len(self.pca.pairs)
            self.cmd_hist = self.cmd_hist[-50:]
        while self.t + 0.01 <= t_target + 1e-12:
            a, kap, name = self._step(0.01)
            self.t = round(self.t + 0.01, 6)
            t_ns = self.t0 + int(round(self.t * NS))
            gz = math.degrees(self.v * kap) + self.rng.normal(0, 3.0) + 0.4
            ay = max(-12.0, min(12.0, self.v * self.v * kap))     # 滑り(横 1.2 G まで)
            az = 9.80665
            for tc, dur in self.crash:
                if abs(self.t - tc) < 0.005:
                    a = 40.0                                     # 水平 約 4G
            if self.bump_g and name == 'R1' and abs((self.s % L) - 0.3) < self.v * 0.01 + 1e-9:
                az += self.bump_g * 9.80665                      # R1 の入口の板の上下の衝撃
            if self._keep('IMU', self.t):
                self.queue_imu.append((t_ns, np.array([a, ay, az, 0.0, 0.0, math.radians(gz)], np.float32)))
            if self._keep('VEHICLE', self.t):
                sel = True if self.select is None else bool(self.select(self.t))
                if any(tc + 0.3 <= self.t < tc + dur for tc, dur in self.crash):
                    sel = False                                  # 人が取り返して持っている
                d = {'pwm': [1441, 1496, 2000 if sel else 1000], 'valid': [True, True, True], 'select': sel,
                     'moving': self.v > 0.05, 'erpm': int(round(-self.v / KV))}
                if self.vbat is not None:
                    vb = self.vbat(self.t)
                    d['vbat_ok'] = vb is not None
                    d['vbat_v'] = float(vb) if vb is not None else float('nan')
                self.queue_veh.append((t_ns, d))
            if self._keep('TOF', self.t) and int(round(self.t * 100)) % 7 == 0:
                self.queue_tof.append((t_ns, np.full((42, 54), 3000, np.uint16)))
            self.log.append((self.t, self.s, self.v, name))

    def next_cam(self, i):
        t = 0.15 + i / 30.0
        if t > self.t_end:
            return None
        self._advance_to(t + 0.005)
        return self.t0 + int(round(t * NS))

    def _pop(self, q, t):
        out = [x for x in q if x[0] <= t]
        del q[:len(out)]
        return out

    def imu_upto(self, t):
        return self._pop(self.queue_imu, t)

    def veh_upto(self, t):
        return self._pop(self.queue_veh, t)

    def tof_upto(self, t):
        return self._pop(self.queue_tof, t)

    def pred_of(self, cam_t, feeder):
        t = (cam_t - self.t0) / 1e9
        kap_ahead, _ = seg_at(self.s + 0.3)
        _, name = seg_at(self.s)
        f = F_MODEL[name] * self.f_scale * self.lap_fac[int(max(self.s - 0.6, 0) // L) % 200]
        if t < self.stand_s:
            f = 0.0                         # 止まった状態から(初めの 1 秒は中立)
        for a, b in self.brake_at:
            if a <= t < b:
                f = -150.0                  # ブレーキ・後退側の予測
        steer = self.center + 320.0 * kap_ahead
        p = np.tile([[steer, self.neutral - f]], (10, 1))
        return p

    # ---- 集計
    def lap_times(self):
        return np.diff(self.laps)

    def seg_max(self, name, t_from=0.0):
        xs = [v for t, s, v, n in self.log if n == name and t >= t_from]
        return max(xs) if xs else float('nan')
