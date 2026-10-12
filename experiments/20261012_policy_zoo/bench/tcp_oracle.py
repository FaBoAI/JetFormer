# -*- coding: utf-8 -*-
"""TCP 型の下調べ(CPU だけ): 学習データから
  1. 曲率 ← ステアリング µs の換算(IMU のヨーレート ÷ eRPM の速さ)と、スロットルの定常の換算を合わせる
  2. IMU のヨー + eRPM の速さの推測航法で、各フレームの「1秒先までの通過点(0.1秒おき 10点)」を作る(= 学習のラベル)
  3. 正解の通過点・正解の未来の速さを controller_proto.TCPController に入れたとき、人の操作をどれだけ再現できるか
     (経路と速度に分けて表すことの上限。予測が完璧でもこれ以上は人の操作に近づかない)
  4. 制御器と temporal ensembling の CPU の時間

    python tcp_oracle.py          → results/tcp_oracle.json
"""

import json
import math
import os
import time

import numpy as np

from common import DATASET, RESULTS, save_json
from controller_proto import ERPM_TO_MPS, Calib, TCPController, TemporalEnsembler

FPS = 30.0
STEER_CENTER, THROTTLE_NEUTRAL = 1441.0, 1496.0       # モデル B の model_config.json(止まっている間の人の操作の中央値)
GRAVITY = 9.80665


def load_scene(name):
    with np.load(os.path.join(DATASET, 'aligned', name + '.npz')) as z:
        a = {k: z[k] for k in ('timestamps_us', 'imu', 'vehicle', 'future_actions', 'future_valid', 'erpm_valid',
                               'vehicle_valid', 'mode')}
    a['yaw'] = a['imu'][:, -3:, 5].mean(axis=1).astype(np.float64)       # 直近 30ms のジャイロ z(rad/s、左が正)
    a['v'] = a['vehicle'][:, 3].astype(np.float64) * ERPM_TO_MPS        # m/s(前進が正)
    acc = a['imu'][:, :, :2]
    a['hg'] = np.hypot(acc[..., 0], acc[..., 1]).max(axis=1) / GRAVITY
    a['wps'] = [waypoints_of(a, i) for i in range(len(a['v']))]
    from jetformer.train_data import impact_mask
    a['impact'] = impact_mask(a['timestamps_us'], a['imu'], 2.5)     # モデル B と同じ 2.5G の衝突の前後
    return a


def waypoints_of(a, i, n=10, step=3):
    """フレーム i から先の推測航法 → 自車の座標(x 前・y 左)の [n,2]。フレーム i+n*step が無ければ None"""
    if i + n * step >= len(a['v']):
        return None
    x = y = th = 0.0
    out = []
    dt = 1.0 / FPS
    for j in range(i, i + n * step):
        v, w = a['v'][j + 1], a['yaw'][j + 1]           # 区間 (t_j, t_j+1] を次のフレームの値で
        th += w * dt
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        if (j - i + 1) % step == 0:
            out.append((x, y))
    return np.array(out)


def fit_calibration(scenes):
    """κ = bias + slope × (steer(j−lag) − 中心)。lag 0〜8 フレームで決定係数が最大のもの。速さ 0.8m/s 以上・衝突の近くを除く"""
    best = None
    for lag in range(0, 9):
        X, Y = [], []
        for a in scenes:
            v, w = a['v'], a['yaw']
            ok = (v > 0.8) & a['erpm_valid'] & a['vehicle_valid'] & (a['hg'] < 1.5)
            idx = np.nonzero(ok)[0]
            idx = idx[idx >= lag]
            X.append(a['vehicle'][idx - lag, 0] - STEER_CENTER)
            Y.append(w[idx] / v[idx])
        X, Y = np.concatenate(X), np.concatenate(Y)
        A = np.stack([np.ones_like(X), X], 1)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        r2 = 1 - ((Y - A @ coef) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        if best is None or r2 > best['r2']:
            best = {'lag_frames': lag, 'curv_bias': float(coef[0]), 'curv_per_us': float(coef[1]), 'r2': float(r2),
                    'n': int(len(X))}
    # スロットル: 前進の量 dt(j−lag) ≈ a0 + a1 × v(j)(加速が小さいフレーム = ほぼ定常)
    tb = None
    for lag in range(0, 16, 3):
        X, Y = [], []
        for a in scenes:
            v = a['v']
            dv = np.gradient(v) * FPS
            fwd = THROTTLE_NEUTRAL - a['vehicle'][:, 1]
            ok = a['erpm_valid'] & a['vehicle_valid'] & (np.abs(dv) < 0.5) & (v > 0.3) & (a['hg'] < 1.5)
            idx = np.nonzero(ok)[0]
            idx = idx[idx >= lag]
            X.append(v[idx])
            Y.append(fwd[idx - lag])
        X, Y = np.concatenate(X), np.concatenate(Y)
        A = np.stack([np.ones_like(X), X], 1)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        r2 = 1 - ((Y - A @ coef) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        if tb is None or r2 > tb['r2']:
            tb = {'lag_frames': lag, 'ff_a0': float(coef[0]), 'ff_a1': float(coef[1]), 'r2': float(r2), 'n': int(len(X))}
    return best, tb


def oracle(scenes, calib, t_look, kp, ki, stride=3, l_min=0.6):
    """正解の通過点と正解の 0.5 秒先の速さ → 制御器 → 人の未来の操作 10ステップとの L1(µs)。
    学習の検証と同じく future_valid・vehicle_valid のフレームだけ・間引き 3・衝突(2.5G)の前後を除く(モデル B の検証と同じ条件)。
    比べる基準: 「今の人の操作をそのまま 10 ステップ」"""
    err_s, err_t, hold_s, hold_t = [], [], [], []
    for a in scenes:
        ctl = TCPController(calib, kp=kp, ki=ki, t_look=t_look, l_min=l_min)
        for i in range(0, len(a['v']) - 31):
            wp = a['wps'][i]
            if wp is None:
                break
            tgt_v = a['v'][i + 15]
            act, _ = ctl.step(wp, tgt_v, a['vehicle'][i, 3])
            if i % stride or not a['future_valid'][i] or not a['vehicle_valid'][i] or a['impact'][i]:
                continue
            fa = a['future_actions'][i]
            err_s.append(np.abs(act[:, 0] - fa[:, 0]).mean())
            err_t.append(np.abs(act[:, 1] - fa[:, 1]).mean())
            hold_s.append(np.abs(a['vehicle'][i, 0] - fa[:, 0]).mean())
            hold_t.append(np.abs(a['vehicle'][i, 1] - fa[:, 1]).mean())
    f = lambda e: round(float(np.mean(e)), 1)     # noqa: E731
    return {'steer_l1_us': f(err_s), 'throttle_l1_us': f(err_t), 'l1_us': round((f(err_s) + f(err_t)) / 2, 1),
            'hold_now_steer_l1_us': f(hold_s), 'hold_now_throttle_l1_us': f(hold_t),
            'hold_now_l1_us': round((f(hold_s) + f(hold_t)) / 2, 1), 'n': len(err_s)}


def timing():
    rng = np.random.default_rng(0)
    calib = Calib(curv_per_us=0.004, ff_a0=40, ff_a1=30)
    ctl = TCPController(calib)
    wps = [np.cumsum(rng.random((10, 2)) * [0.25, 0.02], 0) for _ in range(200)]
    t0 = time.perf_counter()
    for k in range(2000):
        ctl.step(wps[k % 200], 2.0, -4000, speed_cap=2.5)
    ctl_us = (time.perf_counter() - t0) / 2000 * 1e6
    te = TemporalEnsembler(m=0.0)
    preds = [rng.random((10, 2)) * 100 + 1450 for _ in range(200)]
    t0 = time.perf_counter()
    for k in range(2000):
        te.push(k * 33333.0, preds[k % 200])
    te_us = (time.perf_counter() - t0) / 2000 * 1e6
    return {'controller_step_us': round(ctl_us, 1), 'temporal_ensemble_push_us': round(te_us, 1)}


def main():
    splits = json.load(open(os.path.join(DATASET, 'splits.json')))
    train = [load_scene(s) for s in splits['train']]
    val = [load_scene(s) for s in splits['val']]
    steer_fit, thr_fit = fit_calibration(train)
    calib = Calib(STEER_CENTER, steer_fit['curv_per_us'], steer_fit['curv_bias'], THROTTLE_NEUTRAL,
                  thr_fit['ff_a0'], thr_fit['ff_a1'])
    # 先を見る時間と PI の係数は学習の分割で選び、検証の分割で測る
    grid = []
    for l_min in (0.2, 0.6):
        for t_look in (0.15, 0.3, 0.5, 0.7):
            for kp, ki in ((0.0, 0.0), (30.0, 0.0), (30.0, 20.0)):
                grid.append((oracle(train[::3], calib, t_look, kp, ki, l_min=l_min)['l1_us'], t_look, kp, ki, l_min))
    grid.sort()
    _, t_look, kp, ki, l_min = grid[0]
    res = {'calibration': {'steer': steer_fit, 'throttle': thr_fit, 'calib': calib.to_dict()},
           'chosen': {'t_look_s': t_look, 'kp': kp, 'ki': ki, 'l_min_m': l_min},
           'grid_train_subset': [{'l1_us': g[0], 't_look_s': g[1], 'kp': g[2], 'ki': g[3], 'l_min_m': g[4]} for g in grid],
           'oracle_val': oracle(val, calib, t_look, kp, ki, l_min=l_min),
           'oracle_train': oracle(train, calib, t_look, kp, ki, l_min=l_min),
           'oracle_val_long_look': oracle(val, calib, 0.5, kp, ki, l_min=0.6)}
    # 通過点のラベルの大きさ(検証の分割): 1秒先の前後・左右
    ends = []
    for a in val:
        for i in range(0, len(a['v']) - 31, 3):
            wp = a['wps'][i]
            if wp is not None:
                ends.append(wp[-1])
    ends = np.array(ends)
    res['waypoint_label_val'] = {'x_1s_m_median': round(float(np.median(ends[:, 0])), 2),
                                 'y_1s_m_abs_p90': round(float(np.percentile(np.abs(ends[:, 1]), 90)), 2),
                                 'n': int(len(ends))}
    res['cpu_timing'] = timing()
    os.makedirs(RESULTS, exist_ok=True)
    save_json(os.path.join(RESULTS, 'tcp_oracle.json'), res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
