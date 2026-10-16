"""安全層 v1(jetformer/safety_caps.py)の試験。実機・GPU・ログは使わない(合成の値だけ)

    cd ~/jetformer && python -m pytest -q tests/test_safety_caps.py
"""
import math
import random

import pytest

from jetformer import safety_caps as sc


# ------------------------------------------------------------------ 単位の換算
def test_units_match_bridge():
    assert sc.thr_us_to_cmd(1498) == 0.0
    assert sc.thr_us_to_cmd(1000) == pytest.approx(7.24)
    assert sc.thr_us_to_cmd(2000) == pytest.approx(-0.8)
    assert sc.brake_cmd_of_us(200) == pytest.approx(-200 / 627.5)
    for thr in (1000, 1200, 1342.5, 1497, 1498, 1520, 1698, 2000):
        assert sc.cmd_to_thr_us(sc.thr_us_to_cmd(thr)) == pytest.approx(thr)
    assert sc.cmd_to_thr_us(sc.brake_cmd_of_us(200)) == pytest.approx(1698)


def test_calibrated_cap_command_inverts_speed_map():
    # T(2) = 2.1918、T(3) = 3.3368 の間で線形
    u = sc.speed_to_cmd_calibrated(2.5)
    assert u == pytest.approx(2 + (2.5 - 2.191791568056699) / (3.336828397976429 - 2.191791568056699))
    assert sc.speed_to_cmd_calibrated(2.2) < sc.speed_to_cmd_calibrated(2.5) < sc.speed_to_cmd_calibrated(3.0)
    assert sc.speed_to_cmd_calibrated(math.inf) == math.inf
    # PWM にすると R1 の 2.5 m/s は約 1342 µs
    assert sc.cmd_to_thr_us(u) == pytest.approx(1498 - u / 7.24 * 498)
    assert 1335 < sc.cmd_to_thr_us(u) < 1350


# ------------------------------------------------------------------ 上限の層
def run(seq, p=sc.params_sim(), st=None):
    st = st or sc.CapState()
    outs = []
    for t, zone, v, u in seq:
        st, o = sc.cap_step(st, t, zone, v, u, p)
        outs.append(o)
    return st, outs


def test_modes_pass_cap_neutral():
    p = sc.params_sim()
    _, (o,) = run([(0.0, 'none', 3.5, 4.0)], p)
    assert o.mode == 'pass' and o.u == 4.0
    _, (o,) = run([(0.0, 'GM', 2.0, 4.0)], p)
    assert o.mode == 'cap' and o.u == pytest.approx(2.2)
    _, (o,) = run([(0.0, 'GM', 2.3, 4.0)], p)
    assert o.mode == 'neutral' and o.u == 0.0
    _, (o,) = run([(0.0, 'DE', 3.2, 2.0)], p)          # 指令が上限より下でも、速すぎれば中立
    assert o.mode == 'neutral' and o.u == 0.0
    _, (o,) = run([(0.0, 'unknown', 2.0, 5.0)], p)
    assert o.u == pytest.approx(3.0)
    # 実車の換算: 上限 2.5 m/s の指令は T⁻¹(2.5)
    _, (o,) = run([(0.0, 'R1_app', 2.0, 5.0)], sc.params_real_car())
    assert o.u == pytest.approx(sc.speed_to_cmd_calibrated(2.5))


def test_brake_only_in_r1_app_above_cap_and_1mps():
    p = sc.params_sim()
    for zone in ('R1_turn', 'GM', 'DE', 'unknown', 'none'):
        _, (o,) = run([(0.0, zone, 3.5, 4.0)], p)
        assert o.mode != 'brake'
    _, (o,) = run([(0.0, 'R1_app', 2.75, 4.0)], p)       # 上限 + 0.3 以下: 中立まで
    assert o.mode == 'neutral'
    _, (o,) = run([(0.0, 'R1_app', 2.85, 4.0)], p)
    assert o.mode == 'brake' and o.brake_start and o.u == pytest.approx(sc.brake_cmd_of_us(200))
    # 上限を低くした試験用: 1 m/s 未満では始めない
    q = sc.params_sim(cap_R1_app=0.5)
    _, (o,) = run([(0.0, 'R1_app', 0.95, 1.0)], q)
    assert o.mode == 'neutral'


def test_brake_hold_release_and_lockout():
    p = sc.params_sim()
    seq = [(0.0, 'R1_app', 3.0, 4.0), (0.1, 'R1_app', 2.4, 4.0)]     # 0.1 秒で上限を下回っても 0.2 秒は保つ
    st, outs = run(seq, p)
    assert [o.mode for o in outs] == ['brake', 'brake']
    st, (o,) = run([(0.2, 'R1_app', 2.4, 4.0)], p, st)
    assert o.mode == 'cap' and not st.braking
    # やめてから 0.5 秒は、上限 + 0.3 を超えても始めない(中立)
    st, (o,) = run([(0.45, 'R1_app', 3.0, 4.0)], p, st)
    assert o.mode == 'neutral'
    st, (o,) = run([(0.71, 'R1_app', 3.0, 4.0)], p, st)
    assert o.mode == 'brake'
    # R1 の旋回に入ったら、保つ時間の中でもすぐやめる
    st, (o,) = run([(0.75, 'R1_turn', 3.0, 4.0)], p, st)
    assert o.mode == 'neutral' and not st.braking
    # 1 m/s を下回ったらすぐやめる
    st2, _ = run([(0.0, 'R1_app', 3.0, 4.0)], sc.params_sim(cap_R1_app=0.2))
    st2, (o,) = run([(0.05, 'R1_app', 0.9, 4.0)], sc.params_sim(cap_R1_app=0.2), st2)
    assert o.mode != 'brake'


def test_model_brake_is_kept_and_output_never_exceeds_input():
    p = sc.params_sim()
    _, (o,) = run([(0.0, 'R1_app', 3.0, -0.6)], p)      # モデルの方が強いブレーキ → そのまま
    assert o.u == -0.6
    _, (o,) = run([(0.0, 'GM', 1.0, -0.2)], p)
    assert o.u == -0.2 and o.mode == 'pass'


def test_release_ramp_1p5():
    p = sc.params_sim()
    st, _ = run([(0.0, 'GM', 2.0, 4.0)], p)
    st, (o,) = run([(0.5, 'none', 2.0, 6.0)], p, st)
    assert o.cap_eff == pytest.approx(2.2 + 1.5 * 0.5)
    assert o.u == pytest.approx(2.95)
    t = 0.5
    while math.isfinite(o.cap_eff):
        t += 0.1
        st, (o,) = run([(t, 'none', 2.0, 6.0)], p, st)
    assert t == pytest.approx(0.5 + (8.0 - 2.95) / 1.5, abs=0.11)
    assert o.mode == 'pass'


def test_invariants_random_sequences():
    """乱数の入力 3 万フレーム: 出力 ≤ 入力・ブレーキは R1_app だけ・始まりは 1 m/s 以上・往復は 0.5 秒以上あく・保つ時間"""
    rng = random.Random(20261016)
    for p in (sc.params_sim(), sc.params_real_car(), sc.params_r1_275()):
        st = sc.CapState()
        t, zone, v = 0.0, 'none', 2.0
        brake_on, last_end, n_brake = None, -1e9, 0
        for i in range(30000):
            t += rng.choice((1 / 30, 1 / 30, 0.02, 0.05))
            if rng.random() < 0.05:
                zone = rng.choice(sc.ZONES)
            v = max(0.0, min(6.0, v + rng.gauss(0, 0.25)))
            u = rng.uniform(-0.8, 7.24)
            st, o = sc.cap_step(st, t, zone, v, u, p)
            assert o.u <= u + 1e-12
            if o.mode == 'brake':
                assert zone in p.brake_zones
                if o.brake_start:
                    assert v >= 1.0 and v > o.cap_eff + 0.3
                    assert t - last_end >= 0.5 - 1e-9
                    brake_on = t
                    n_brake += 1
            elif brake_on is not None:
                # ブレーキが終わった: 0.2 秒未満で終わるのは、区間を出た・1 m/s 未満のときだけ
                if t - brake_on < 0.2 - 1e-9:
                    assert zone not in p.brake_zones or v < 1.0
                brake_on, last_end = None, t
        assert n_brake > 20


def test_real_car_wrapper_never_more_forward():
    rng = random.Random(1)
    st = sc.CapState()
    p = sc.params_real_car()
    t = 0.0
    for _ in range(5000):
        t += 1 / 30
        thr = rng.uniform(1000, 2000)
        st, out, o = sc.cap_throttle_us(st, t, rng.choice(sc.ZONES), rng.uniform(0, 5), thr, p)
        assert out >= thr - 1e-9
        if o.mode == 'brake':
            assert out == pytest.approx(max(thr, 1698.0))


# ------------------------------------------------------------------ ジャイロの位相
# 合成の周(左回り。R1 は右 −180 度、GM 左 +160 度、DE、SW、HS、LH の左 U ターン)。(秒, ヨー deg/s, 速さ m/s)
LAP = [(0.6, -300.0, 2.5), (1.0, 160.0, 2.2), (2.0, 30.0, 3.0), (1.5, 90.0, 3.2), (1.0, 40.0, 3.5), (1.0, 145.0, 2.8), (0.3, 0.0, 3.0)]
START = [(1.0, 0.0, 0.0), (1.0, 0.0, 2.0), (1.0, 180.0, 2.5), (0.3, 0.0, 3.0)]


def synth(n_laps=3, hz=100.0):
    segs = START + LAP * n_laps
    out, t = [], 0.0
    r1_starts = []
    for k, (dur, g, v) in enumerate(segs):
        if g == -300.0:
            r1_starts.append(t)
        n = int(round(dur * hz))
        for _ in range(n):
            out.append((t, g + 0.4, v))       # ずれ +0.4 deg/s
            t += 1 / hz
    return out, r1_starts


def run_phase(samples):
    st = sc.PhaseState()
    outs = []
    for t, g, v in samples:
        st, o = sc.phase_step(st, t, g, v)
        outs.append((t, o))
    return st, outs


def test_phase_detects_r1_each_lap_and_zones_in_order():
    samples, r1 = synth(3)
    st, outs = run_phase(samples)
    conf = [t for t, o in outs if o.r1_confirmed]
    assert len(conf) == 3
    anchors = sorted({o.anchor_t for t, o in outs if o.r1_confirmed})
    for a, r in zip(anchors, r1):
        assert abs(a - r) <= 0.05                        # 起点は旋回の始まり(5 サンプル平均の遅れ以内)
    zones = [o.zone for t, o in outs]
    assert zones[0] == 'unknown'
    # 2 周目の R1 の前後: R1_app → R1_turn → GM → DE → none → R1_app
    # ψ ≥ 300 になるのは LH の 0.586 秒目 = R1 の起点の 0.714 秒前
    t_app = next(t for t, o in outs if t >= r1[1] - 1.5 and o.zone == 'R1_app')
    assert t_app == pytest.approx(r1[1] - (0.414 + 0.3), abs=0.07)   # 起点の向きの取り方とずれの分
    i0 = next(i for i, (t, o) in enumerate(outs) if t >= r1[1] - 0.6)
    i1 = next(i for i, (t, o) in enumerate(outs) if t >= r1[2] - 0.05)
    seq = []
    for z in zones[i0:i1]:
        if not seq or seq[-1] != z:
            seq.append(z)
    assert seq == ['R1_app', 'R1_turn', 'GM', 'DE', 'none', 'R1_app']
    # R1_turn は旋回の始まりから 30 ms 以内に始まる(確定より前)
    t_turn = next(t for t, o in outs if t >= r1[1] - 0.1 and o.zone == 'R1_turn')
    assert t_turn - r1[1] <= 0.031
    # DE は 1.5 秒
    de = [t for t, o in outs if o.zone == 'DE' and r1[1] < t < r1[2]]
    assert max(de) - min(de) == pytest.approx(1.5, abs=0.02)


def test_phase_lost_after_16s_and_bias():
    samples, r1 = synth(1)
    t_end = samples[-1][0]
    # R1 が来ないまま直線を 20 秒
    samples += [(t_end + 0.01 * (k + 1), 0.4, 3.0) for k in range(2000)]
    st, outs = run_phase(samples)
    assert outs[-1][1].zone == 'unknown'
    assert any(o.zone == 'R1_app' for t, o in outs)
    # 止まっている間にずれを覚える
    st2, _ = run_phase([(0.01 * k, 0.4, 0.0) for k in range(300)])
    assert st2.bias == pytest.approx(0.4)


def test_phase_no_false_r1_on_short_right_turns():
    """DE の戻し舵のような短い右(−120 deg/s、30 度)や、直前に左旋回の無い右 U ターンは R1 にしない"""
    samples, _ = synth(1)
    t0 = samples[-1][0]
    extra = [(1.0, 0.0, 3.0), (0.25, -120.0, 3.0), (1.0, 0.0, 3.0), (3.0, 0.0, 3.0), (0.6, -300.0, 2.5), (1.0, 0.0, 3.0)]
    t = t0
    for dur, g, v in extra:
        for _ in range(int(round(dur * 100))):
            t += 0.01
            samples.append((t, g, v))
    st, outs = run_phase(samples)
    assert sum(o.r1_confirmed for t, o in outs) == 1


def test_combine_zone():
    assert sc.combine_zone('none', False, True) == 'R1_app'
    assert sc.combine_zone('none', True, True) == 'R1_turn'
    assert sc.combine_zone('unknown', False, True) == 'R1_app'
    assert sc.combine_zone('GM', False, True) == 'GM'
    assert sc.combine_zone('DE', True, True) == 'DE'
    assert sc.combine_zone('R1_app', True, False) == 'R1_turn'
    assert sc.combine_zone('none', True, False) == 'none'
    assert sc.pred_signal_of([1400, 1200, 1149]) and not sc.pred_signal_of([1400, 1151])


# ------------------------------------------------------------------ v1.1 の直し(G1 の再生のあと、1 回だけ)
def test_v11_brake_released_by_right_steer_command():
    p = sc.params_sim()
    st, (o,) = run([(0.0, 'R1_app', 3.0, 4.0)], p)
    assert o.mode == 'brake'
    st, o = sc.cap_step(st, 0.05, 'R1_app', 3.0, 4.0, p, steer=-0.35)     # 右へ 35%: 保つ時間の中でもやめる
    assert o.mode == 'neutral' and not st.braking
    st, o = sc.cap_step(st, 0.6, 'R1_app', 3.0, 4.0, p, steer=-0.31)      # 右へ切っている間は始めない
    assert o.mode == 'neutral'
    st, o = sc.cap_step(st, 0.7, 'R1_app', 3.0, 4.0, p, steer=-0.1)
    assert o.mode == 'brake'
    # v1 には舵の条件が無い
    st1, o1 = sc.cap_step(sc.CapState(), 0.0, 'R1_app', 3.0, 4.0, sc.params_v1(cap_command='identity'), steer=-0.9)
    assert o1.mode == 'brake'
    # 実車の形: 舵の PWM 1443 − 0.3 × 443 ≈ 1310 µs より右なら出さない
    _, thr, o = sc.cap_throttle_us(sc.CapState(), 0.0, 'R1_app', 3.0, 1300.0, sc.params_real_car(), steer_us=1300.0)
    assert o.mode == 'neutral' and thr == pytest.approx(1498.0)
    _, thr, o = sc.cap_throttle_us(sc.CapState(), 0.0, 'R1_app', 3.0, 1300.0, sc.params_real_car(), steer_us=1400.0)
    assert o.mode == 'brake' and thr == pytest.approx(1698.0)


def test_v11_phase_unknown_after_stop_and_turnback():
    samples, r1 = synth(2)
    t_end = samples[-1][0]
    # 1.2 秒止まる(置き直したかもしれない)→ 位相は不確か。v1 では残る
    stop = [(t_end + 0.01 * (k + 1), 0.4, 0.0) for k in range(120)]
    st, outs = run_phase(samples + stop)
    assert outs[-1][1].zone == 'unknown'
    st1 = sc.PhaseState()
    for t, g, v in samples + stop:
        st1, o1 = sc.phase_step(st1, t, g, v, sc.PHASE_V1)
    assert o1.zone != 'unknown'
    # 0.8 秒の停止では残る
    st, outs = run_phase(samples + stop[:80])
    assert outs[-1][1].zone != 'unknown'


def test_v11_pred_signal_needs_1mps():
    assert sc.combine_zone('unknown', False, True, v=0.8) == 'unknown'
    assert sc.combine_zone('unknown', False, True, v=1.2) == 'R1_app'
    assert sc.combine_zone('unknown', False, True, v=0.8, pred_min_speed=0.0) == 'R1_app'
    assert sc.combine_zone('R1_app', False, False, v=0.5) == 'R1_app'      # ジャイロの合図は速さによらない
