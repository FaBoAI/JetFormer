#!/usr/bin/env python3
"""シミュレーター較正(2026-10-16)の物理の再生(Python、numpy でたくさんの窓を同時に)

2 つの物理を持つ。
  - 'sim'  : 今の sim-core.js(real-drive-response-estimate-v1)の式そのまま。P_default と P_cfg(設定欄だけ + アプリの遅れ)
  - 'cal'  : 較正の模型 jf-calib-v1(protocol.json model.P_cal)。P_cal

時間の進め方は sim-core.js の step と同じ: 1 フレーム dt の中を h = min(substep, 0.012/最大の動きの速さ, 残り) の
小区間で進める(maxSpeed 7.24・maxSteer 0.52 では h ≈ 1.0994 ms)。指令はフレームごと(10 ms)に 1 つ。
遅れ d は「小区間の始まりの時刻 τ − d を含むフレームの指令」を使う(履歴の指令を含む)。
この離散の決まりは、S2 で作る JS の較正版のコア(calib_runtime.mjs)と同じにする(試験 T3)。

窓の配列(calib_data.make_windows / stack_windows): 格子 k = −70〜100、10 ms。
  replay の始まり = k = −50(準備の始まり、τ = 0)。k = 0 が窓の始まり t0(τ = 0.5)。配列の添字 = k + 70。
  準備の間は速さを実際の値に固定し、舵の状態だけを流す(protocol fidelity.replay.initial_state)。
"""
import math

import numpy as np

G = 9.80665
WHEELBASE = 0.26
MAX_STEER = 0.52
MAX_SPEED = 7.24
DT = 0.01
K_START = 50          # 再生の始まり = 窓の始まりの 50 フレーム前
IDX_OFF = 20          # 配列の添字 = (τ を含むフレームの番号) + 20(k = −70 が添字 0、再生の始まり k = −50 が添字 20)
N_WIN = 100
BRAKE_US_PER_MPS = 627.5   # off = −u_v × 627.5 µs(ブリッジの逆)


def h_sub(max_speed=MAX_SPEED, max_steer=MAX_STEER, length=0.42, width=0.19, wheelbase=WHEELBASE):
    """sim-core.js の小区間の上限 .012/maximumMotionRate(同じ順で計算)"""
    vehicle_radius = math.hypot(length, width) / 2
    rate = max_speed + vehicle_radius * max_speed / wheelbase * math.tan(max_steer)
    return .012 / rate


def schedule(n_frames, dt=DT, substep=None):
    """小区間の予定: [(フレーム番号, τ, h)]。sim-core.js の step の while と同じ足し算の順"""
    substep = min(0.01, dt) if substep is None else substep
    hs = h_sub()
    out = []
    sim_time = 0.0
    for f in range(n_frames):
        remaining = dt
        elapsed = 0.0
        while elapsed < remaining - 1e-12:
            h = min(substep, hs, remaining - elapsed)
            out.append((f, sim_time + elapsed, h))
            elapsed += h
        sim_time = sim_time + elapsed
    return out


SCHED = schedule(K_START + N_WIN)


def delayed_index(tau, d, dt=DT):
    """時刻 τ − d を含むフレームの、窓の配列での添字"""
    return IDX_OFF + int(math.floor((tau - d) / dt + 1e-9))


# ------------------------------------------------------------------ 較正の模型の部品
def pwl(x, xs, ys):
    """区分線形(端の外は端の値)"""
    return np.interp(x, xs, ys)


STEER_KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])


def steer_target(u_s, prm):
    """δ_target = g_L(u_s/0.52)(左)・−g_R(|u_s|/0.52)(右)。prm['steer_L']・['steer_R'] は節の δ [rad](節 0 の値は左右で共通 = 中心のずれ)"""
    n = np.clip(np.asarray(u_s, float) / MAX_STEER, -1.0, 1.0)
    xs = np.concatenate([-STEER_KNOTS[::-1], STEER_KNOTS[1:]])
    ys = np.concatenate([-np.asarray(prm['steer_R'], float)[::-1], np.asarray(prm['steer_L'], float)[1:]])
    # 節 0 は左右共通: steer_R[0] は −steer_L[0] と同じ扱い(下で揃える)
    ys[len(STEER_KNOTS) - 1] = prm['steer_L'][0]
    return np.interp(n, xs, ys)


SPEED_KNOTS = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.24])


def speed_target(u_v, prm, k_batt=1.0):
    """v_tgt = k_batt × T(u_v)(u_v ≥ 0)。T: (0,0)・(u_db,0)・(0.5,T0.5)…(7.24,T7.24) の区分線形"""
    u = np.asarray(u_v, float)
    xs = np.concatenate([[0.0, prm['u_db']], SPEED_KNOTS])
    ys = np.concatenate([[0.0, 0.0], np.asarray(prm['T'], float)])
    return k_batt * np.interp(np.maximum(u, 0.0), xs, ys)


def brake_decel(off_us, prm):
    """a_b(off) = s_b × max(0, off − db_b)"""
    return prm['s_b'] * np.maximum(0.0, off_us - prm['db_b'])


def approach(value, target, max_delta):
    return value + np.clip(target - value, -max_delta, max_delta)


# ------------------------------------------------------------------ 再生
def replay(W, physics, prm, mode='coupled', mu=None, k_batt=None, n_frames=N_WIN, record=('v', 'r', 'psi', 'x', 'y')):
    """窓の配列 W を物理で流す。
    physics: 'sim'(P_default/P_cfg)か 'cal'(P_cal)
    mode: 'coupled'(速さもヨーもシミュレーター)・'lat'(速さは実際の値、ヨーだけ)・'lon'(ヨーは実際の値、速さだけ)
    mu: 窓ごとの μ(路面 × 倍率を掛けた後)。k_batt: 窓ごとの電池の係数('cal' だけ)
    返り値: 窓の始まり(k = 0)から 1 フレームごとの値(k = 0〜n_frames、形 [窓, n_frames+1])"""
    U_s, U_v, V_real, R_real = W['u_s'], W['u_v'], W['v'], W['r']
    n = U_s.shape[0]
    mu = np.full(n, 0.8) if mu is None else np.asarray(mu, float)
    mug = mu * G
    if physics == 'sim':
        cfg = prm
        max_steer = cfg.get('maxSteer', MAX_STEER)
        d_s = d_v = cfg.get('control_delay_frames', 0) / 30.0
        u_s_all = np.clip(U_s, -max_steer, max_steer)
        u_v_all = np.clip(U_v, -cfg.get('maxSpeed', MAX_SPEED), cfg.get('maxSpeed', MAX_SPEED))
    else:
        d_s, d_v = prm.get('d_s', 0.0), prm.get('d_v', 0.0)
        if mode != 'lon':
            tgt_s_all = steer_target(U_s, prm)
            r_s = prm['r_s']
            K_us = prm['K_us']
        if mode != 'lat':
            kb = np.ones(n) if k_batt is None else np.asarray(k_batt, float)
            tgt_v_all = kb[:, None] * speed_target(U_v, prm)
            off_all = -np.minimum(U_v, 0.0) * BRAKE_US_PER_MPS
            ab_all = brake_decel(off_all, prm)
    # 状態
    v = V_real[:, IDX_OFF].copy()
    delta = np.zeros(n)
    yaw = np.zeros(n); x = np.zeros(n); y = np.zeros(n)
    r_last = np.zeros(n)
    esc_seen = np.zeros(n, bool); esc_neutral = np.zeros(n); esc_armed = np.zeros(n, bool)
    rec = {k: np.zeros((n, n_frames + 1)) for k in record}
    sched = SCHED if n_frames == N_WIN else schedule(K_START + n_frames)
    start_f = 0 if mode != 'lon' else K_START
    cur_frame = -1
    for (f, tau, h) in sched:
        if f < start_f:
            continue
        warm = f < K_START
        kf = f - K_START          # 窓の格子の番号(k)
        if f != cur_frame:
            if cur_frame >= K_START:
                kk = cur_frame - K_START + 1
                for key, val in (('v', v), ('r', r_last), ('psi', yaw), ('x', x), ('y', y)):
                    if key in rec:
                        rec[key][:, kk] = val
            cur_frame = f
            if f == K_START:
                v = V_real[:, IDX_OFF + K_START].copy()
                yaw[:] = 0; x[:] = 0; y[:] = 0
                for key, val in (('v', v), ('r', R_real[:, IDX_OFF + K_START]), ('psi', yaw), ('x', x), ('y', y)):
                    if key in rec:
                        rec[key][:, 0] = val
        ia = IDX_OFF + f            # このフレームの実際の値の添字
        v_ext = V_real[:, ia]
        if warm or mode == 'lat':
            v = v_ext.copy()
        js = delayed_index(tau, d_s)
        jv = delayed_index(tau, d_v)
        # ---------------- 前後
        if warm or mode == 'lat':
            v_next = v_ext
        elif physics == 'sim':
            u = u_v_all[:, jv]
            fwd = (v >= 0) & (u > v)
            rate = np.where(fwd, np.minimum(np.minimum(cfg['driveAccelerationCap'], mug), (u - v) / cfg['driveTau']),
                            cfg['brakeDeceleration'])
            v_next = approach(v, u, rate * h)
        else:
            u = U_v[:, jv]
            vt = tgt_v_all[:, jv]
            if mode == 'lon':
                alat = np.abs(v * R_real[:, ia])
            else:
                alat = np.abs(v * r_last)
            scrub = prm['k_s'] * alat
            v_next = v.copy()
            fwd_cmd = u >= 0
            # 前進側の指令: 駆動(v_tgt > v ≥ 0)
            drive = fwd_cmd & (vt > v) & (v >= 0)
            a_dr = np.minimum(np.minimum(prm['a_cap'], mug), (vt - v) / prm['tau_up']) - scrub
            vd = v + a_dr * h
            vd = np.where(a_dr > 0, np.minimum(vd, vt), vd)
            v_next = np.where(drive, np.maximum(vd, 0.0), v_next)
            # 前進側の指令: 惰性(0 ≤ v_tgt ≤ v)。リフトの項は v_tgt より下へは押さない。こすれは効く
            coast = fwd_cmd & (vt <= v) & (v >= 0)
            lift = np.minimum(prm['c0'], (v - vt) / prm['tau_down']) if prm['tau_down'] > 0 else np.where(v > vt, prm['c0'], 0.0)
            vc = v - (lift + scrub) * h
            v_next = np.where(coast, np.maximum(vc, 0.0), v_next)
            # 後退中(v < 0)に前進側の指令: 1.5 m/s² で 0 へ
            back = fwd_cmd & (v < 0)
            v_next = np.where(back, approach(v, 0.0, 1.5 * h), v_next)
            # ブレーキ側の指令
            brk = ~fwd_cmd
            moving = brk & (v > 0.05)
            vb = np.maximum(v - (prm['c0'] + scrub + ab_all[:, jv]) * h, 0.0)
            v_next = np.where(moving, vb, v_next)
            stopped = brk & (v <= 0.05)
            rev = stopped & esc_armed
            v_next = np.where(rev, approach(v, -0.8, 1.5 * h), v_next)
            hold = stopped & ~esc_armed
            v_next = np.where(hold, np.maximum(v - (prm['c0'] + ab_all[:, jv]) * h, np.minimum(v, 0.0)), v_next)
            # ESC の手順: 止まってブレーキ側を受けた → 中立(前進側)0.12 秒以上 → 後退できる
            esc_seen = np.where(hold, True, esc_seen)
            esc_neutral = np.where(hold | brk, 0.0, np.where(esc_seen & fwd_cmd & (v <= 0.05), esc_neutral + h, esc_neutral))
            esc_armed = esc_armed | (esc_seen & (esc_neutral >= 0.12 - 1e-12))
            reset = v_next > 0.05
            esc_seen &= ~reset; esc_armed &= ~reset; esc_neutral = np.where(reset, 0.0, esc_neutral)
        # ---------------- 舵とヨー
        if mode == 'lon':
            yr = R_real[:, ia]
            delta_next = delta
        else:
            if physics == 'sim':
                delta_next = approach(delta, u_s_all[:, js], cfg['steerRate'] * h)
                raw = v_next / WHEELBASE * np.tan(delta_next)
            else:
                tgt = tgt_s_all[:, js]
                delta_next = tgt if not np.isfinite(r_s) else approach(delta, tgt, r_s * h)
                raw = v_next * np.tan(delta_next) / WHEELBASE / (1.0 + K_us * v_next * v_next)
            lim = mug / np.maximum(np.abs(v_next), .01)
            yr = np.clip(raw, -lim, lim)
        if not warm:
            ym = yaw + yr * h / 2
            x = x + v_next * np.cos(ym) * h
            y = y + v_next * np.sin(ym) * h
            yaw = yaw + yr * h
        v = v_next
        delta = delta_next
        r_last = yr
    kk = cur_frame - K_START + 1
    for key, val in (('v', v), ('r', r_last), ('psi', yaw), ('x', x), ('y', y)):
        if key in rec:
            rec[key][:, kk] = val
    return rec


def truth(W, n_frames=N_WIN):
    """実際の値(k = 0〜n_frames): 速さ・ヨーレート・向き(∫r)・位置(∫v(cos ψ, sin ψ))。台形で積分"""
    i0 = IDX_OFF + K_START
    v = W['v'][:, i0:i0 + n_frames + 1]
    r = W['r'][:, i0:i0 + n_frames + 1]
    psi = np.concatenate([np.zeros((len(v), 1)), np.cumsum((r[:, 1:] + r[:, :-1]) / 2 * DT, axis=1)], axis=1)
    vx = v * np.cos(psi); vy = v * np.sin(psi)
    x = np.concatenate([np.zeros((len(v), 1)), np.cumsum((vx[:, 1:] + vx[:, :-1]) / 2 * DT, axis=1)], axis=1)
    y = np.concatenate([np.zeros((len(v), 1)), np.cumsum((vy[:, 1:] + vy[:, :-1]) / 2 * DT, axis=1)], axis=1)
    return {'v': v, 'r': r, 'psi': psi, 'x': x, 'y': y}


def errors(sim, tru):
    """窓ごとの指標: e_v(1 秒後の速さ)・e_r(窓の中のヨーの RMS)・e_psi(1 秒後の向き、度)・e_p(1 秒後の位置)・e_vw(窓の中の速さの RMS)"""
    out = {}
    if 'v' in sim:
        out['e_v'] = sim['v'][:, -1] - tru['v'][:, -1]
        out['e_vw'] = np.sqrt(np.mean((sim['v'][:, 1:] - tru['v'][:, 1:]) ** 2, axis=1))
    if 'psi' in sim:
        # ヨーレートはフレーム(10 ms)の平均で比べる: シミュレーター = Δψ/dt、実際 = 台形の平均(向きの積分と同じ)
        r_sim = np.diff(sim['psi'], axis=1) / DT
        r_tru = (tru['r'][:, 1:] + tru['r'][:, :-1]) / 2
        out['e_r'] = np.sqrt(np.mean((r_sim - r_tru) ** 2, axis=1))
    if 'psi' in sim:
        out['e_psi'] = np.degrees(sim['psi'][:, -1] - tru['psi'][:, -1])
    if 'x' in sim:
        out['e_p'] = np.hypot(sim['x'][:, -1] - tru['x'][:, -1], sim['y'][:, -1] - tru['y'][:, -1])
    return out


# ------------------------------------------------------------------ 道具
def nelder_mead(f, x0, step, max_iter=2000, tol=1e-7, callback=None):
    """簡単な Nelder–Mead(scipy が無いため)。返り値 (x, fx, 回数)"""
    x0 = np.asarray(x0, float)
    nd = len(x0)
    pts = [x0] + [x0 + np.eye(nd)[i] * step[i] for i in range(nd)]
    vals = [f(p) for p in pts]
    it = 0
    while it < max_iter:
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + 1e-12) and it > 50:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            if fe < fr:
                pts[-1], vals[-1] = xe, fe
            else:
                pts[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c) if fr >= vals[-1] else c + 0.5 * (xr - c)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, nd + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
        it += 1
        if callback is not None and it % 100 == 0:
            callback(it, vals[0], pts[0])
    i = int(np.argmin(vals))
    return pts[i], vals[i], it


def nnls_monotone(A, b, w=None, n_iter=3000):
    """b ≈ A @ (v0 + cumsum(Δ)), Δ ≥ 0 の重みつき最小二乗(座標降下)。A は節の補間の行列。返り値は節の値"""
    m, k = A.shape
    w = np.ones(m) if w is None else np.asarray(w, float)
    # 値 = C @ p、p = [v0, Δ1..Δ(k-1)]、C は下三角の 1
    C = np.tril(np.ones((k, k)))
    M = A @ C
    Wm = M * np.sqrt(w)[:, None]
    bw = b * np.sqrt(w)
    H = Wm.T @ Wm + 1e-9 * np.eye(k)
    gvec = Wm.T @ bw
    p = np.zeros(k)
    p[0] = np.linalg.lstsq(Wm[:, :1], bw, rcond=None)[0][0]
    for _ in range(n_iter):
        p_old = p.copy()
        for j in range(k):
            r = gvec[j] - H[j] @ p + H[j, j] * p[j]
            p[j] = r / H[j, j]
            if j > 0 and p[j] < 0:
                p[j] = 0.0
        if np.max(np.abs(p - p_old)) < 1e-12:
            break
    return C @ p


def interp_matrix(x, knots):
    """区分線形の補間 = A @ 節の値 の行列 A(端の外は端の値)"""
    x = np.clip(np.asarray(x, float), knots[0], knots[-1])
    A = np.zeros((len(x), len(knots)))
    j = np.clip(np.searchsorted(knots, x, side='right') - 1, 0, len(knots) - 2)
    t = (x - knots[j]) / (knots[j + 1] - knots[j])
    A[np.arange(len(x)), j] = 1 - t
    A[np.arange(len(x)), j + 1] = t
    return A


def lon_fast(W, prm, mu, k_batt=None, h=0.005, n_frames=N_WIN):
    """当てはめ用の速い『速さだけの再生』(較正の模型)。小区間を h(既定 5 ms)にし、後退の手順は省く(窓の中ではほぼ起きない)。
    同じ式。最終の値は replay(..., mode='lon') で出し直す。返り値: 速さ [窓, n_frames+1]"""
    n = W['u_v'].shape[0]
    kb = np.ones(n) if k_batt is None else np.asarray(k_batt, float)
    VT = kb[:, None] * speed_target(W['u_v'], prm)
    BRK = W['u_v'] < 0
    AB = brake_decel(-np.minimum(W['u_v'], 0.0) * BRAKE_US_PER_MPS, prm)
    R = W['r']
    mug = np.asarray(mu, float) * G
    cap = np.minimum(prm['a_cap'], mug)
    c0, ks, tu, td = prm['c0'], prm['k_s'], prm['tau_up'], prm['tau_down']
    d_v = prm['d_v']
    v = W['v'][:, IDX_OFF + K_START].copy()
    out = np.zeros((n, n_frames + 1))
    out[:, 0] = v
    per = int(round(DT / h))
    for s in range(n_frames * per):
        tau = K_START * DT + s * h
        jv = IDX_OFF + int(math.floor((tau - d_v) / DT + 1e-9))
        ia = IDX_OFF + K_START + s // per
        vt = VT[:, jv]
        scrub = ks * np.abs(v * R[:, ia])
        up = vt > v
        a_dr = np.minimum(cap, (vt - v) / tu) - scrub
        lift = np.minimum(c0, (v - vt) / td) if td > 0 else np.where(v > vt, c0, 0.0)
        a = np.where(up, a_dr, -lift - scrub)
        a = np.where(BRK[:, jv], -(c0 + scrub + AB[:, jv]), a)
        vn = v + a * h
        vn = np.where(up & (a > 0), np.minimum(vn, vt), vn)
        v = np.maximum(vn, 0.0)
        if (s + 1) % per == 0:
            out[:, (s + 1) // per] = v
    return out


def nm_restarts(f, x0, step, max_iter=1500, max_restarts=8, rel_tol=1e-4, log=None):
    """Nelder–Mead を、良くならなくなるまで新しい単体でやり直す(早く縮んで止まるのを防ぐ)"""
    x, fx, it = nelder_mead(f, x0, step, max_iter=max_iter, tol=1e-8)
    total = it
    for k in range(max_restarts):
        x2, f2, it2 = nelder_mead(f, x, step * (0.5 if k % 2 else 1.0), max_iter=max_iter, tol=1e-8)
        total += it2
        if log:
            log('   やり直し %d: %.6f → %.6f' % (k + 1, fx, f2))
        improved = f2 < fx - rel_tol * abs(fx)
        if f2 < fx:
            x, fx = x2, f2
        if not improved:
            break
    return x, fx, total
