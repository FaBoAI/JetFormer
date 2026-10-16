"""較正の模型(calib_model.py)と同定の道具の試験(S1)。手計算の値と比べる。

    cd ~/jetformer/experiments/20261016_sim_calibration
    CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/test_calib_model.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import calib_model as cm  # noqa: E402

N_ALL = 171            # 格子 k = −70〜100
I0 = cm.IDX_OFF + cm.K_START   # 窓の始まり(k = 0)の添字


def windows(n=1, u_s=0.0, u_v=0.0, v=2.0, r=0.0):
    """一定の指令・一定の実際の値の窓"""
    W = {'u_s': np.full((n, N_ALL), u_s, float), 'u_v': np.full((n, N_ALL), u_v, float),
         'v': np.full((n, N_ALL), v, float), 'r': np.full((n, N_ALL), r, float), 'n': n}
    return W


def cal_prm(**kw):
    p = {'steer_L': [0.0, 0.13, 0.26, 0.39, 0.468, 0.52], 'steer_R': [0.0, 0.13, 0.26, 0.39, 0.468, 0.52],
         'K_us': 0.0, 'd_s': 0.0, 'r_s': np.inf, 'u_db': 0.0, 'T': [0.5, 1, 2, 3, 4, 5, 6, 7.24], 'tau_up': 0.4,
         'a_cap': 6.0, 'tau_down': 0.0, 'c0': 0.5, 'k_s': 0.0, 'd_v': 0.0, 's_b': 0.01, 'db_b': 50.0}
    p.update(kw)
    return p


def test_substep_matches_sim_core_formula():
    r = math.hypot(0.42, 0.19) / 2
    rate = 7.24 + r * 7.24 / 0.26 * math.tan(0.52)
    assert cm.h_sub() == pytest.approx(0.012 / rate, rel=0, abs=1e-15)
    sched = cm.schedule(3)
    f0 = [h for f, t, h in sched if f == 0]
    assert len(f0) == 10                       # 9 × 1.0994 ms + 残り
    assert sum(f0) == pytest.approx(0.01, abs=1e-12)


def test_delayed_index():
    # τ = 0.5(窓の始まり)、遅れ 0 → k = 0 の添字
    assert cm.delayed_index(0.5, 0.0) == I0
    assert cm.delayed_index(0.5, 0.03) == I0 - 3
    assert cm.delayed_index(0.5049, 0.0) == I0
    assert cm.delayed_index(0.5051, 0.001) == I0


def test_sim_physics_constant_brake_and_drive():
    # シミュレーターの今の物理: 目標より遅いと min(cap, μg, (u−v)/τ)、速いと一律 1.2 m/s²
    cfg = {'maxSteer': 0.52, 'maxSpeed': 7.24, 'steerRate': 1.2, 'driveTau': 0.4, 'driveAccelerationCap': 6.0,
           'brakeDeceleration': 1.2, 'frictionMultiplier': 1.0, 'control_delay_frames': 0}
    W = windows(u_v=0.0, v=3.0)
    v = cm.replay(W, 'sim', cfg, 'lon', mu=[0.8], record=('v',))['v'][0]
    assert v[-1] == pytest.approx(3.0 - 1.2 * 1.0, abs=1e-9)
    W = windows(u_v=4.0, v=1.0)
    v = cm.replay(W, 'sim', cfg, 'lon', mu=[0.8], record=('v',))['v'][0]
    # 始めは上限 6 m/s² で 0.1 秒(v = 1.6 まで)、そのあと τ = 0.4 の指数
    assert v[-1] == pytest.approx(4.0 - 2.4 * math.exp(-0.9 / 0.4), abs=2e-3)


def test_cal_coast_and_scrub():
    # 中立(v_tgt = 0)・τ_down = 0: 直線の減速 c0。こすれ: 実際のヨー r で k_s |v r| が足される
    W = windows(u_v=0.0, v=3.0)
    v = cm.replay(W, 'cal', cal_prm(c0=0.5), 'lon', record=('v',))['v'][0]
    assert v[-1] == pytest.approx(2.5, abs=1e-9)
    W = windows(u_v=0.0, v=3.0, r=1.0)
    v = cm.replay(W, 'cal', cal_prm(c0=0.5, k_s=0.2), 'lon', record=('v',))['v'][0]
    # dv/dt = −0.5 − 0.2 v → v(1) = (3 + 2.5) e^{−0.2} − 2.5
    assert v[-1] == pytest.approx(5.5 * math.exp(-0.2) - 2.5, abs=2e-3)


def test_cal_scrub_steady_state_same_from_above_and_below():
    # 曲がりながら一定の指令: 下から近づいても上から近づいても、平衡は v_tgt − τ_up k_s |a_lat| のあたり
    p = cal_prm(k_s=0.3, tau_down=0.3, c0=0.5)
    lo = cm.replay(windows(u_v=3.0, v=1.5, r=1.0), 'cal', p, 'lon', record=('v',))['v'][0]
    hi = cm.replay(windows(u_v=3.0, v=3.5, r=1.0), 'cal', p, 'lon', record=('v',))['v'][0]
    # 解析の平衡: (3 − v)/0.4 = 0.3 v → v = 3/(1 + 0.12) = 2.679
    assert lo[-1] < 2.70 and hi[-1] > 2.65
    assert abs(hi[-1] - 2.679) < 0.15


def test_cal_speed_map_and_dead_band_and_battery():
    p = cal_prm(u_db=0.3, T=[0.5, 1.0, 2.0, 3.0, 3.8, 4.2, 4.3, 4.4])
    assert cm.speed_target(0.2, p) == 0.0
    assert cm.speed_target(0.4, p) == pytest.approx(0.25)
    assert cm.speed_target(7.0, p) == pytest.approx(4.3 + 0.1 * (1.0 / 1.24))
    assert cm.speed_target(2.0, p, k_batt=0.8) == pytest.approx(1.6)


def test_cal_brake_table():
    # ブレーキ: off = −u_v × 627.5 µs。u_v = −200/627.5 → off 200 → a = c0 + s_b (200 − db_b)
    u = -200.0 / 627.5
    p = cal_prm(c0=0.5, s_b=0.01, db_b=50.0)
    v = cm.replay(windows(u_v=u, v=3.0), 'cal', p, 'lon', record=('v',))['v'][0]
    assert v[50] == pytest.approx(3.0 - 0.5 * (0.5 + 1.5), abs=1e-6)
    assert cm.brake_decel(np.array([30.0, 50.0, 250.0]), p).tolist() == pytest.approx([0.0, 0.0, 2.0])


def test_esc_no_reverse_until_neutral():
    # 止まってからブレーキ側を出し続けても後退しない
    p = cal_prm(c0=2.0, s_b=0.02, db_b=0.0)
    W = windows(u_v=-0.5, v=0.6)
    v = cm.replay(W, 'cal', p, 'lon', record=('v',))['v'][0]
    assert v.min() >= 0.0 and v[-1] == 0.0
    # ブレーキ側で止まる → 中立 0.2 秒 → ブレーキ側 → 後退(−0.8 m/s へ 1.5 m/s²)
    W = windows(u_v=-0.5, v=0.6)
    W['u_v'][0, I0 + 40:I0 + 60] = 0.0
    v = cm.replay(W, 'cal', p, 'lon', record=('v',))['v'][0]
    assert v[-1] < -0.3
    assert v[-1] >= -0.8 - 1e-9


def test_cal_steering_steady_state_understeer_and_grip():
    # 舵いっぱい(左 δ = 0.52)・2 m/s・K = 0.05: κ = tan(0.52)/0.26/(1 + 0.05 × 4)
    p = cal_prm(K_us=0.05)
    W = windows(u_s=0.52, v=2.0)
    rec = cm.replay(W, 'cal', p, 'lat', mu=[3.0], record=('psi',))
    r = np.diff(rec['psi'][0]) / 0.01
    assert r[-1] == pytest.approx(2.0 * math.tan(0.52) / 0.26 / 1.2, rel=1e-6)
    # 横の限界: μ = 0.5 → r = μ g / v
    rec = cm.replay(windows(u_s=0.52, v=3.0), 'cal', p, 'lat', mu=[0.5], record=('psi',))
    r = np.diff(rec['psi'][0]) / 0.01
    assert r[-1] == pytest.approx(0.5 * cm.G / 3.0, rel=1e-6)


def test_cal_steering_delay_and_map_asymmetry():
    # 右の対応を小さく(いっぱいで 0.3 rad)。窓の始まりで右いっぱいへ段の指令、遅れ 0.05 秒
    p = cal_prm(steer_R=[0.0, 0.1, 0.2, 0.25, 0.28, 0.3], d_s=0.05)
    W = windows(u_s=0.0, v=1.0)
    W['u_s'][0, I0:] = -0.52
    rec = cm.replay(W, 'cal', p, 'lat', mu=[3.0], record=('psi',))
    r = np.diff(rec['psi'][0]) / 0.01
    assert np.all(np.abs(r[:4]) < 1e-12)                 # 40 ms までは動かない
    assert r[-1] == pytest.approx(-math.tan(0.3) / 0.26, rel=1e-6)


def test_lon_fast_close_to_exact():
    rng = np.random.default_rng(0)
    W = windows(n=20, v=2.0)
    W['u_v'] = np.clip(np.cumsum(rng.normal(0, 0.3, (20, N_ALL)), axis=1) + 2.5, -0.5, 6.0)
    W['r'] = rng.normal(0, 1.0, (20, N_ALL))
    p = cal_prm(k_s=0.3, tau_down=0.2, d_v=0.03, u_db=0.2, T=[0.5, 1.0, 2.0, 3.0, 3.8, 4.2, 4.3, 4.4])
    mu = np.full(20, 0.7)
    ve = cm.replay(W, 'cal', p, 'lon', mu=mu, record=('v',))['v']
    vf = cm.lon_fast(W, p, mu, h=0.0025)
    assert np.abs(ve - vf).max() < 0.02


def test_monotone_fit():
    knots = np.array([0.0, 0.5, 1.0])
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y = np.array([0.0, 0.3, 0.2, 0.6, 0.5])
    vals = cm.nnls_monotone(cm.interp_matrix(x, knots), y)
    assert np.all(np.diff(vals) >= -1e-12)


def test_fit_split_has_no_heldout_values():
    # 当てはめの読み込みでは、保留のシーン(k % 3 == 1)の値が 1 つも残らない
    import calib_data as cd
    rec = cd.load_record('20261010-172910', 'fit')
    held = rec['held']
    for k in ('v5', 'r', 'a_lat', 'u_s', 'u_v'):
        assert not np.isfinite(rec[k][held & ~rec['brake_only']]).any()
    assert not rec['clean'][held].any()
    ws = cd.make_windows(rec)
    assert all(w['scene'] % 3 != 1 for w in ws)


def test_zero_center_refit_and_profile_shape():
    import ident
    tab = [[-0.5, 100, -0.3], [-0.25, 100, -0.1], [-0.05, 100, 0.02], [0.05, 100, 0.03], [0.25, 100, 0.15], [0.5, 100, 0.3], [1.0, 100, 0.4]]
    vals = ident.refit_steer_zero_center(tab)
    assert vals[5] == 0.0
    assert np.all(np.diff(vals) >= -1e-12)
    L, R = ident.steer6(vals)
    assert L[0] == 0.0 and len(L) == 6 and len(R) == 6


def test_ai_input_variant_maps_through_human_curve():
    import ident
    xh = np.array([0.2, 1.0, 2.2, 3.3, 4.1, 5.0, 5.0, 5.0, 0.4, 0.3, 4.0, 0.5, 0.0])
    v = ident.ai_input_variant(xh, {})
    x_id = v['to_full'](np.array([1.0, 0.0]))
    assert np.allclose(x_id, xh)
    x2 = v['to_full'](np.array([1.0, 1.0]))
    assert x2[1] == pytest.approx(2.2)      # T_AI(1) = T_H(2)


def test_frozen_profiles_sha256():
    import hashlib
    d = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/profiles')
    for line in (d / 'SHA256SUMS').read_text().splitlines():
        h, name = line.split('  ', 1)
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == h
