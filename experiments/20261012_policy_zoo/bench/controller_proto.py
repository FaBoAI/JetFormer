# -*- coding: utf-8 -*-
"""TCP / TransFuser++ 型の「通過点 + 目標速度 → 操作」の制御器と、行動のかたまりの時間方向の平均(ACT の temporal ensembling)の試作。
numpy だけ(torch を読まない = drive.py からも評価の道具からも同じコードを呼べる)。

車の決まり(CAR CONVENTIONS):
    ステアリング: 中心 ~1441µs、中心より大きい = 左。経路の曲率 κ>0 = 左(y 左が正)なので µs = 中心 + κ/傾き
    スロットル: 前進は中立 ~1496µs より小さい側(dt = 中立 − µs > 0 が前進)。前進側だけにゲインが掛かる
    eRPM: 負 = 前進。速さ [m/s] ≈ −eRPM × 0.5e-3(65mm のタイヤ)
"""

import math

import numpy as np

ERPM_TO_MPS = -0.5e-3
WP_DT = 0.1                 # 通過点の間隔 [s](10点 = 1秒先まで)


class Calib(object):
    """学習データから合わせる係数(tcp_oracle.py の fit_calibration が作る)"""

    def __init__(self, steer_center_us=1441.0, curv_per_us=0.0, curv_bias=0.0, throttle_neutral_us=1496.0,
                 ff_a0=0.0, ff_a1=0.0, steer_min_us=1000.0, steer_max_us=1900.0):
        self.steer_center_us = steer_center_us
        self.curv_per_us = curv_per_us            # κ [1/m] = curv_bias + curv_per_us × (µs − 中心)
        self.curv_bias = curv_bias
        self.throttle_neutral_us = throttle_neutral_us
        self.ff_a0, self.ff_a1 = ff_a0, ff_a1      # 前進の量 dt [µs] ≈ a0 + a1 × 速さ [m/s](定常)
        self.steer_min_us, self.steer_max_us = steer_min_us, steer_max_us

    def to_dict(self):
        return dict(self.__dict__)


def pure_pursuit_curvature(waypoints, speed, t_look=0.5, l_min=0.6, l_max=2.0):
    """自車の座標(x 前・y 左、m)の通過点 [N,2] → 追う曲率 κ [1/m]。
    先を見る距離 L = clip(速さ × t_look, l_min, l_max)。通過点の折れ線の上で、原点から距離 L の点を補間して κ = 2y/L²"""
    wp = np.asarray(waypoints, np.float64).reshape(-1, 2)
    pts = np.vstack([[0.0, 0.0], wp])
    d = np.hypot(pts[:, 0], pts[:, 1])
    look = min(max(speed * t_look, l_min), l_max)
    k = int(np.searchsorted(d, look))
    if k >= len(pts):                     # 経路が短い(遅い・止まる)→ 最後の点
        x, y = pts[-1]
    elif k == 0:
        x, y = pts[1]
    else:
        f = (look - d[k - 1]) / max(d[k] - d[k - 1], 1e-6)
        x, y = pts[k - 1] + f * (pts[k] - pts[k - 1])
    l2 = x * x + y * y
    return 0.0 if l2 < 1e-4 else 2.0 * y / l2


class TCPController(object):
    """通過点 + 目標速度 → [10,2] の操作(µs)。行はすべて同じ値(先読みは純追従の距離で行うので、走行の lookahead は効かない)。
    speed_cap [m/s] があれば目標速度をそこで頭打ちにする(管理画面の速度の上限)。PI の積分だけが状態"""

    def __init__(self, calib, kp=40.0, ki=20.0, i_limit=60.0, t_look=0.5, l_min=0.6, dt_max_us=250.0):
        self.c = calib
        self.kp, self.ki, self.i_limit, self.t_look, self.dt_max_us = kp, ki, i_limit, t_look, dt_max_us
        self.l_min = l_min
        self.integ = 0.0

    def reset(self):
        self.integ = 0.0

    def step(self, waypoints, target_speed, erpm, speed_cap=None, dt=1 / 30.0):
        c = self.c
        speed = max(float(erpm) * ERPM_TO_MPS, 0.0)
        kappa = pure_pursuit_curvature(waypoints, speed, self.t_look, self.l_min)
        steer = c.steer_center_us + (kappa - c.curv_bias) / c.curv_per_us if c.curv_per_us else c.steer_center_us
        steer = min(max(steer, c.steer_min_us), c.steer_max_us)
        v_t = max(float(target_speed), 0.0)
        if speed_cap is not None:
            v_t = min(v_t, float(speed_cap))
        err = v_t - speed
        self.integ = min(max(self.integ + self.ki * err * dt, -self.i_limit), self.i_limit)
        fwd = (c.ff_a0 + c.ff_a1 * v_t if v_t > 0.05 else 0.0) + self.kp * err + self.integ
        fwd = min(max(fwd, 0.0), self.dt_max_us)            # 制御器は後退・ブレーキを出さない(ESC の決まり)
        throttle = c.throttle_neutral_us - fwd
        out = np.empty((10, 2), np.float32)
        out[:, 0], out[:, 1] = steer, throttle
        return out, {'kappa': kappa, 'speed': speed, 'target_speed': v_t, 'fwd_us': fwd}


class TemporalEnsembler(object):
    """ACT の temporal ensembling を走行用に: 30fps のフレームごとの予測 [10,2](µs)と撮像時刻を貯め、
    今の時刻から見た各ステップ s の値を、過去の予測 p(j ステップ前)の p[j+s] の重み付き平均にする。
    重み w = exp(−m × j)(m=0 で単純平均。ACT の元の式は古い予測ほど重い exp(−m × 新しさの順) で m=0.01 = ほぼ平均)。
    出力は [10,2] のまま(j+s が 10 を超える分は使える予測だけで平均)→ 先読み・ゲイン・補助はそのまま使える。
    フレームが抜けても時刻で数えるので、ずれない"""

    def __init__(self, horizon=10, m=0.0, max_age=10, step_us=33333.0, newest_heavier=True):
        self.horizon, self.m, self.max_age, self.step_us = horizon, m, max_age, step_us
        self.sign = 1.0 if newest_heavier else -1.0
        self.buf = []                          # [(撮像時刻 µs, 予測 [10,2])]

    def reset(self):
        self.buf = []

    def push(self, t_us, pred):
        self.buf.append((float(t_us), np.asarray(pred, np.float64)))
        self.buf = [(t, p) for t, p in self.buf if (t_us - t) / self.step_us < self.max_age + 0.5]
        acc = np.zeros((self.horizon, 2))
        wsum = np.zeros((self.horizon, 1))
        for t, p in self.buf:
            j = int(round((t_us - t) / self.step_us))
            if j >= self.horizon:
                continue
            w = math.exp(-self.m * j * self.sign)
            n = self.horizon - j
            acc[:n] += w * p[j:j + n]
            wsum[:n] += w
        return (acc / np.maximum(wsum, 1e-12)).astype(np.float32)
