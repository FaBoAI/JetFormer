#!/usr/bin/env python3
"""TCP / TransFuser++ 型の方策のラベル(1秒先までの通過点・目標速度)を、clean_driving_20261011 から作れるかを確かめる
(読み取りのみ・CPU・numpy だけ。torch・GPU は使わない)。

    nice -n 15 python tcp_label_check.py > tcp_label_check.json

使うもの: data/datasets/clean_driving_20261011/
  aligned/<scene>.npz                  カメラ 30fps の各フレーム(timestamps_us, mode, future_valid, vehicle_valid, imu, future_actions)
  can_bus/<scene>_ms_imu.json          IMU 全サンプル(約100Hz、rotation_rate は rad/s。z 正 = 左旋回)
  can_bus/<scene>_vehicle_monitor.json 操作値・eRPM(約100Hz。eRPM 負 = 前進)

決めごと(車の約束): 速度 v = -eRPM × 0.5e-3 [m/s]、自車座標 x = 前・y = 左、ヨー ψ = ∫(gz - バイアス)。
ステア > 中心(約1443µs)= 左。スロットルは中立(約1498µs)より下 = 前進。

調べること:
  A 符号の確認(ステアとヨー、ヨーと横加速度、周回の向き)
  B 速度・通過点の分布(ジャンル別)、1秒の窓が取れるフレームの割合
  C ステア → 曲率の較正(遅れを探す。左右で別の傾き)。tof_guard の K_LEFT/K_RIGHT と比べる
  D スロットル ↔ 速度(前向きの写像: スロットル = f(今の速度, 目標速度))
  E 「完全な予測」をしたときの制御器の上限: 正解の通過点と目標速度から、制御器(pure pursuit + 前向き写像)で作った
    10ステップの [ステア, スロットル] が、人の操作(future_actions)とどれだけ違うか(µs)。モデル B の検証の誤差と比べる
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/jetson/jetformer/data/datasets/clean_driving_20261011'
KV = 0.5e-3                  # m/s per eRPM(前進 = eRPM 負)
CENTER, NEUTRAL = 1441.0, 1496.0     # モデル B の model_config(steer_center_us / throttle_neutral_us)
K_LEFT, K_RIGHT = 0.00275, 0.0039    # tof_guard.py の曲率/µs
FPS = 30.0
WP_DT, WP_N = 0.1, 10        # 通過点: 0.1秒刻み × 10 = 1秒
GRAV = 9.80665


def impact_mask(ts_us, imu, g=2.5, before=1.0, after=2.0):
    t = ts_us / 1e6
    hg = np.hypot(imu[..., 0], imu[..., 1]).max(axis=1) / GRAV
    m = np.zeros(len(t), bool)
    for ti in t[hg >= g]:
        m |= (t >= ti - before) & (t <= ti + after)
    return m


def load_scene(name):
    a = dict(np.load(os.path.join(ROOT, 'aligned', name + '.npz')))
    imu = json.load(open(os.path.join(ROOT, 'can_bus', name + '_ms_imu.json')))
    veh = json.load(open(os.path.join(ROOT, 'can_bus', name + '_vehicle_monitor.json')))
    it = np.array([r['utime'] for r in imu], np.int64)
    gz = np.array([r['rotation_rate'][2] for r in imu], np.float64)
    ax = np.array([r['linear_accel'][0] for r in imu], np.float64)
    vt = np.array([r['utime'] for r in veh], np.int64)
    erpm = np.array([r['erpm'] for r in veh], np.float64)
    st = np.array([r['steering_us'] for r in veh], np.float64)
    th = np.array([r['throttle_us'] for r in veh], np.float64)
    ok = np.array([r['valid'][0] and r['valid'][1] and not r['ai_driving'] for r in veh])
    return a, dict(it=it, gz=gz, ax=ax, vt=vt, erpm=erpm, st=st, th=th, vok=ok)


def dead_reckon(s, bias):
    """IMU の時刻で ψ, x, y(シーンの最初を原点)"""
    t = s['it'] / 1e6
    v = -np.interp(s['it'], s['vt'], s['erpm']) * KV
    dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.05)
    w = s['gz'] - bias
    psi = np.cumsum(w * dt)
    x = np.cumsum(v * np.cos(psi) * dt)
    y = np.cumsum(v * np.sin(psi) * dt)
    return t, v, w, psi, x, y


def waypoints(t, psi, x, y, t0, dts):
    """時刻 t0 の自車座標(x 前・y 左)での、t0 + dts の位置 [len(dts), 2] と向きの変化"""
    p0, x0, y0 = np.interp(t0, t, psi), np.interp(t0, t, x), np.interp(t0, t, y)
    tt = t0 + dts
    dx, dy = np.interp(tt, t, x) - x0, np.interp(tt, t, y) - y0
    c, s = np.cos(p0), np.sin(p0)
    return np.stack([c * dx + s * dy, -s * dx + c * dy], 1), np.interp(tt, t, psi) - p0


def steer_from_curv(k):
    """曲率 [1/m] → ステア µs(中心 + 左右で違う傾き)。tof_guard と同じ形の逆"""
    return np.where(k > 0, CENTER + k / KL, CENTER + k / KR)


def steer_from_curv_table(k, v):
    """速度帯の表で逆引き(帯の間は線形に混ぜる)"""
    vm = np.array([t[0] for t in TABLE])
    offs = np.array([np.interp(k, kk, cc) for _, cc, kk in TABLE])
    return CENTER + float(np.interp(v, vm, offs))


def pure_pursuit(path, ld):
    """path: 自車座標の点列 [M,2](0 = 今の位置を含む)。弧長 ld の点への円弧の曲率 κ = 2y/L²"""
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 1e-3:
        return 0.0
    ld = min(ld, s[-1])
    px, py = np.interp(ld, s, path[:, 0]), np.interp(ld, s, path[:, 1])
    L2 = px * px + py * py
    return 2.0 * py / max(L2, 1e-6)


def main():
    splits = json.load(open(os.path.join(ROOT, 'splits.json')))
    names = sorted(splits['train'] + splits['val'])
    val = set(splits['val'])
    scenes = {n: load_scene(n) for n in names}
    out = {'definitions': __doc__}

    # ---- ジャイロのバイアス(ログごと: 止まっている間 |eRPM|<1 かつ |gz|<5°/s の中央値)
    bias_log = {}
    for n, (a, s) in scenes.items():
        er = np.interp(s['it'], s['vt'], s['erpm'])
        still = (np.abs(er) < 1) & (np.abs(s['gz']) < np.deg2rad(5))
        bias_log.setdefault(n.rsplit('-', 1)[0], []).append(s['gz'][still])
    bias_log = {k: (float(np.median(np.concatenate(v))) if sum(len(x) for x in v) > 50 else 0.0, int(sum(len(x) for x in v)))
                for k, v in bias_log.items()}
    out['gyro_bias_rad_s'] = {k: [round(b, 5), c] for k, (b, c) in bias_log.items()}

    # ---- フレームごとのラベル
    dts = WP_DT * np.arange(1, WP_N + 1)
    rows = []          # 1フレーム = dict
    sign = {'steer_gz_corr': [], 'gz_ax_corr': [], 'scene_yaw_deg': []}
    calib = []         # (scene, t, steer(t), v, w) 高速の区間の生サンプル
    for n, (a, s) in scenes.items():
        bias = bias_log[n.rsplit('-', 1)[0]][0]
        t, v, w, psi, x, y = dead_reckon(s, bias)
        sign['scene_yaw_deg'].append(float(np.rad2deg(psi[-1])))
        st_i = np.interp(s['it'], s['vt'][s['vok']], s['st'][s['vok']]) if s['vok'].sum() > 10 else None
        mv = v > 0.8
        if st_i is not None and mv.sum() > 50:
            sign['steer_gz_corr'].append(float(np.corrcoef(st_i[mv] - CENTER, w[mv])[0, 1]))
            sign['gz_ax_corr'].append(float(np.corrcoef(w[mv] * v[mv], s['ax'][mv])[0, 1]))
            calib.append((n, t, st_i, v, w, np.interp(s['it'], s['vt'][s['vok']], s['th'][s['vok']])))
        ts = a['timestamps_us']
        base = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0) & ~impact_mask(ts, a['imu'])
        t_end = min(t[-1], s['vt'][-1] / 1e6)
        gaps = np.diff(t)
        for i in np.nonzero(base)[0]:
            t0 = ts[i] / 1e6
            ok1s = t0 + dts[-1] <= t_end
            if ok1s:
                m = (t >= t0) & (t <= t0 + dts[-1])
                ok1s = bool(m.sum() > 50 and gaps[m[1:]].max(initial=0) < 0.05)
            wp, dpsi = waypoints(t, psi, x, y, t0, dts) if ok1s else (None, None)
            # 目標速度の候補: 0.5〜1.0秒の通過点の間の速さ(TF++ の waypoints 版の定義)・1秒の平均・0.33秒先の瞬時の速さ
            vi = lambda tt: float(np.interp(tt, t, v))
            rows.append(dict(scene=n, val=n in val, i=int(i), mode=int(a['mode'][i]), stride3=(i % 3 == 0), ok1s=ok1s,
                             v0=vi(t0), wp=wp, dpsi=dpsi, fut=a['future_actions'][i].astype(np.float64),
                             veh=a['vehicle'][i].astype(np.float64), t0=t0, sc=(t, v, w, psi, x, y)))

    out['sign_checks'] = {
        'steer_minus_center_vs_yawrate_corr(>0 なら ステア大 = 左 = gz 正)': np.round(np.percentile(sign['steer_gz_corr'], [0, 50, 100]), 3).tolist(),
        'yawrate_x_speed_vs_accel_x_corr(<0 なら 左旋回で加速度 x 負 = +x は右)': np.round(np.percentile(sign['gz_ax_corr'], [0, 50, 100]), 3).tolist(),
        'scene_total_yaw_deg(左回りのコースなので正が多いはず)': {'median': round(float(np.median(sign['scene_yaw_deg'])), 1),
                                                             'positive_scenes': int(np.sum(np.array(sign['scene_yaw_deg']) > 0)),
                                                             'scenes': len(sign['scene_yaw_deg'])},
    }

    # ---- B 分布
    R = [r for r in rows]
    def pct(x, q=(10, 50, 90)):
        x = np.asarray(x, float)
        return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else None
    dist = {}
    for mode in (4, 5):
        rr = [r for r in R if r['mode'] == mode]
        ok = [r for r in rr if r['ok1s']]
        d1 = [np.hypot(*np.diff(np.vstack([[0, 0], r['wp']]), axis=0).T).sum() for r in ok]
        dist[str(mode)] = {
            'frames_base': len(rr), 'frames_base_stride3': sum(r['stride3'] for r in rr),
            'frac_with_1s_window': round(len(ok) / max(len(rr), 1), 3),
            'v0_m_s_p10_50_90': pct([r['v0'] for r in rr]),
            'frac_v0_below_0.3': round(float(np.mean([r['v0'] < 0.3 for r in rr])), 3),
            'frac_reverse_in_1s(v<-0.1)': round(float(np.mean([np.interp(r['t0'] + dts, r['sc'][0], r['sc'][1]).min() < -0.1 for r in ok])), 3),
            'dist_1s_m_p10_50_90': pct(d1),
            'lat_y_at_1s_m_p10_50_90': pct([r['wp'][-1, 1] for r in ok]),
            'abs_lat_y_at_1s_m_p50_90_99': pct([abs(r['wp'][-1, 1]) for r in ok], (50, 90, 99)),
            'dpsi_1s_deg_p10_50_90': pct([np.rad2deg(r['dpsi'][-1]) for r in ok]),
        }
    out['distribution'] = dist

    # ---- C ステア → 曲率の較正(遅れ L: κ(t+L) = ω/v と ステア(t))
    res_c = {}
    best = None
    for lag_ms in (0, 40, 80, 100, 120, 140, 160, 200, 250):
        S, K, V = [], [], []
        for (n, t, st_i, v, w, th_i) in calib:
            if n in val:
                continue
            tt = t + lag_ms / 1e3
            vv = np.interp(tt, t, v); ww = np.interp(tt, t, w)
            m = (vv > 1.0) & (tt <= t[-1])
            S.append(st_i[m] - CENTER); K.append(ww[m] / vv[m]); V.append(vv[m])
        S, K, V = np.concatenate(S), np.concatenate(K), np.concatenate(V)
        lin = np.abs(S) < 250          # 頭打ちの手前だけで傾きを見る
        kl = float(np.sum(S[lin & (S > 0)] * K[lin & (S > 0)]) / np.sum(S[lin & (S > 0)] ** 2))
        kr = float(np.sum(S[lin & (S < 0)] * K[lin & (S < 0)]) / np.sum(S[lin & (S < 0)] ** 2))
        pred = np.where(S > 0, kl * S, kr * S)
        r2 = 1 - np.sum((K[lin] - pred[lin]) ** 2) / np.sum((K[lin] - K[lin].mean()) ** 2)
        res_c[lag_ms] = {'k_left_per_us': round(kl, 5), 'k_right_per_us': round(kr, 5), 'r2_linear_region': round(float(r2), 3), 'n': int(lin.sum())}
        if best is None or r2 > res_c[best]['r2_linear_region']:
            best = lag_ms
    # 頭打ち(いっぱいに切ったとき)の曲率と、速度帯別の傾き(アンダーステア)
    lag = best
    S, K, V = [], [], []
    for (n, t, st_i, v, w, th_i) in calib:
        if n in val:
            continue
        tt = t + lag / 1e3
        vv = np.interp(tt, t, v); ww = np.interp(tt, t, w)
        m = (vv > 1.0) & (tt <= t[-1])
        S.append(st_i[m] - CENTER); K.append(ww[m] / vv[m]); V.append(vv[m])
    S, K, V = np.concatenate(S), np.concatenate(K), np.concatenate(V)
    bins = [-600, -450, -350, -250, -150, -75, -25, 25, 75, 150, 250, 350, 450, 600]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (S >= lo) & (S < hi)
        if m.sum() > 30:
            table.append({'steer_offset_us': [lo, hi], 'median_curv_1_per_m': round(float(np.median(K[m])), 3), 'n': int(m.sum())})
    speed_slope = {}
    for lo, hi in ((1.0, 2.0), (2.0, 3.0), (3.0, 5.0)):
        m = (V >= lo) & (V < hi) & (np.abs(S) < 250)
        if m.sum() > 100:
            kl = float(np.sum(S[m & (S > 0)] * K[m & (S > 0)]) / np.sum(S[m & (S > 0)] ** 2))
            kr = float(np.sum(S[m & (S < 0)] * K[m & (S < 0)]) / np.sum(S[m & (S < 0)] ** 2))
            speed_slope['%.0f-%.0f m/s' % (lo, hi)] = {'k_left': round(kl, 5), 'k_right': round(kr, 5), 'n': int(m.sum())}
    out['steer_to_curvature'] = {'by_lag_ms(train)': res_c, 'best_lag_ms': lag, 'binned_at_best_lag': table,
                                 'slope_by_speed': speed_slope, 'tof_guard_constants': {'K_LEFT': K_LEFT, 'K_RIGHT': K_RIGHT}}
    global KL, KR, TABLE
    KL, KR = res_c[lag]['k_left_per_us'], res_c[lag]['k_right_per_us']
    # 速度帯ごとの表(ステアのずれ 50µs 刻みの曲率の中央値。単調にそろえる)→ 逆引き。頭打ち(いっぱい)も表に入る
    TABLE = []
    edges = np.arange(-475, 626, 50)
    for lo, hi in ((0.0, 2.0), (2.0, 3.0), (3.0, 9.0)):
        cs, ks = [], []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            m = (S >= a_) & (S < b_) & (V >= max(lo, 1.0)) & (V < hi)
            if m.sum() > 20:
                cs.append(0.5 * (a_ + b_)); ks.append(float(np.median(K[m])))
        ks = np.maximum.accumulate(np.array(ks))
        TABLE.append(((lo + min(hi, 4.5)) / 2, np.array(cs), ks + np.arange(len(ks)) * 1e-6))
    out['steer_to_curvature']['inverse_table_by_speed'] = [
        {'v_mid': round(vm, 2), 'offset_us': c.tolist(), 'curv': np.round(k, 3).tolist()} for vm, c, k in TABLE]

    # ---- D スロットル ↔ 速度(前向きの写像)。thr(t) = c0 + c1·v(t) + c2·(v(t+T) - v(t))/T(人の操作から最小二乗)
    res_d = {}
    for T in (0.2, 0.33, 0.5):
        X, Y, Xv, Yv = [], [], [], []
        for (n, t, st_i, v, w, th_i) in calib:
            vf = np.interp(t + T, t, v)
            m = (t + T <= t[-1]) & (v > 0.3) & (vf > 0.3)
            A = np.stack([np.ones(m.sum()), v[m], (vf[m] - v[m]) / T], 1)
            (Xv if n in val else X).append(A); (Yv if n in val else Y).append(th_i[m])
        X, Y, Xv, Yv = np.concatenate(X), np.concatenate(Y), np.concatenate(Xv), np.concatenate(Yv)
        c, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res_d['T=%.2fs' % T] = {'coef[c0,c1(us per m/s),c2(us per m/s2)]': np.round(c, 2).tolist(),
                                'train_mae_us': round(float(np.mean(np.abs(X @ c - Y))), 1),
                                'val_mae_us': round(float(np.mean(np.abs(Xv @ c - Yv))), 1)}
    out['throttle_feedforward'] = res_d
    cT = np.array(res_d['T=0.33s']['coef[c0,c1(us per m/s),c2(us per m/s2)]'])
    T_FF = 0.33

    # ---- E 制御器の上限(正解の 1秒の通過点 → pure pursuit・前向き写像 → 10ステップ)
    def controller(r, ld_s, ld_min, delay_s, table=False):
        t, v, w, psi, x, y = r['sc']
        t0 = r['t0']
        fine = t0 + np.arange(0, 1.0001, 0.02)
        P, _ = waypoints(t, psi, x, y, t0, fine - t0)                 # 正解の道(細かい点列)
        P = np.vstack([[0, 0], P])
        vs = np.interp(fine, t, v)
        acts = np.zeros((10, 2))
        for k in range(1, 11):
            tk = t0 + k / FPS                                          # このステップの時刻
            # その時刻の自車の位置・向き(計画どおりに進んだとする)
            pk, dpk = waypoints(t, psi, x, y, t0, np.array([k / FPS]))
            th0 = dpk[0]
            Q = P - pk[0]
            c, s = np.cos(th0), np.sin(th0)
            Q = np.stack([c * Q[:, 0] + s * Q[:, 1], -s * Q[:, 0] + c * Q[:, 1]], 1)
            Q = Q[Q[:, 0] > -0.05]
            vk = float(np.interp(tk + delay_s, t, v))
            ld = max(ld_min, vk * ld_s)
            kap = pure_pursuit(np.vstack([[0, 0], Q]), ld) if len(Q) > 1 else 0.0
            acts[k - 1, 0] = np.clip(steer_from_curv_table(kap, vk) if table else steer_from_curv(np.array(kap)), 1000, 2000)
            vnow = float(np.interp(tk, t, v)); vtar = float(np.interp(min(tk + T_FF, t0 + 1.0), t, v))
            acts[k - 1, 1] = cT[0] + cT[1] * vnow + cT[2] * (vtar - vnow) / T_FF
        return acts

    def evaluate(rs, ld_s, ld_min, delay_s, table=False):
        e = np.array([np.abs(controller(r, ld_s, ld_min, delay_s, table) - r['fut']) for r in rs])   # [N,10,2]
        return e
    tr = [r for r in R if r['ok1s'] and not r['val'] and r['stride3'] and r['v0'] > 0.3][::4]
    va = [r for r in R if r['ok1s'] and r['val'] and r['stride3']]
    grid = {}
    for ld_s in (0.05, 0.1, 0.15, 0.25, 0.35, 0.5):
        for ld_min in (0.1, 0.2, 0.4):
            e = evaluate(tr, ld_s, ld_min, 0.0)
            grid['ld=%.2fs,min=%.1fm' % (ld_s, ld_min)] = round(float(e[:, :, 0].mean()), 1)
    bk = min(grid, key=grid.get)
    ld_s, ld_min = float(bk.split('=')[1].split('s')[0]), float(bk.split('min=')[1].rstrip('m'))
    e = evaluate(va, ld_s, ld_min, 0.0)
    gridt = {}
    for ld_s_ in (0.1, 0.15, 0.25, 0.35):
        gridt['ld=%.2fs,min=0.2m' % ld_s_] = round(float(evaluate(tr, ld_s_, 0.2, 0.0, True)[:, :, 0].mean()), 1)
    bkt = min(gridt, key=gridt.get)
    et = evaluate(va, float(bkt.split('=')[1].split('s')[0]), 0.2, 0.0, True)
    e035 = evaluate(va, 0.35, 0.2, 0.0, True)
    v0 = np.array([r['v0'] for r in va]); md = np.array([r['mode'] for r in va])
    # 参考: 曲率の逆写像の上限(未来の実際の曲率 κ(t+k/30+遅れ) をそのままステアに戻す)
    inv = []
    for r in va:
        t, v, w, psi, x, y = r['sc']
        tk = r['t0'] + np.arange(1, 11) / FPS + lag / 1e3
        vv = np.maximum(np.interp(tk, t, v), 0.3)
        inv.append(np.abs(np.clip(steer_from_curv(np.interp(tk, t, w) / vv), 1000, 2000) - r['fut'][:, 0]))
    inv = np.array(inv)
    out['oracle_controller'] = {
        'note': '正解の通過点(未来の実際の道)と未来の実際の速度から制御器で作った操作と、人の操作(future_actions)の差。'
                'モデルの誤差ではなく「制御器と較正だけで生じる誤差の下限」。val の stride 3、1秒の窓が取れるフレームだけ',
        'grid_train_steer_mae_us': grid, 'chosen': bk,
        'val_n': len(va),
        'val_steer_mae_us_k1_k3_k10': [round(float(e[:, j, 0].mean()), 1) for j in (0, 2, 9)],
        'val_throttle_mae_us_k1_k3_k10': [round(float(e[:, j, 1].mean()), 1) for j in (0, 2, 9)],
        'val_mean_us(steer+throttle, 10 steps)': round(float(e.mean()), 1),
        'val_by_mode_mean_us': {str(m): round(float(e[md == m].mean()), 1) for m in (4, 5) if (md == m).any()},
        'val_moving(v0>0.5)_steer_mae_k3': round(float(e[v0 > 0.5, 2, 0].mean()), 1),
        'table_grid_train_steer_mae_us': gridt, 'table_chosen': bkt,
        'val_table_steer_mae_us_k1_k3_k10': [round(float(et[:, j, 0].mean()), 1) for j in (0, 2, 9)],
        'val_table_ld0.35s_steer_mae_us_k1_k3_k10': [round(float(e035[:, j, 0].mean()), 1) for j in (0, 2, 9)],
        'val_table_by_mode_steer_mae_k3': {str(m): round(float(et[md == m, 2, 0].mean()), 1) for m in (4, 5) if (md == m).any()},
        'val_inverse_curvature_steer_mae_us_k1_k3_k10': [round(float(inv[:, j].mean()), 1) for j in (0, 2, 9)],
        'model_B_reference(compare.json clean val, teacher-forced = self-fed)': {'steer_k1_k3_k10': [42.7, 48.7, 56.1], 'throttle_k1_k3_k10': [41.5, 44.9, 50.7], 'val_l1_us': 49.0},
    }
    # 目標速度のクラス分け(TF++ 型)の目安: 0.33秒先の速さの分位
    vt = [float(np.interp(r['t0'] + 0.33, r['sc'][0], r['sc'][1])) for r in R if r['ok1s'] and not r['val']]
    out['target_speed_quantiles_m_s(train, v at t+0.33s)'] = pct(vt, (5, 10, 25, 50, 75, 90, 95, 99))
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
