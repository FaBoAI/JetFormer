"""速さの天井(safety_caps.ceil_step)・速さのゲインの自動合わせ(jetformer.speed_auto)の純粋な関数の試験(2026-10-25)。
仕様: jetformer_reports/speed_control_20261025/spec.md 6.1 節の D-c1・D-c4・D-a1〜a4・D-t1(純粋な関数)。
実機・GPU・torch は使わない(偽物の車は 1 次の遅れ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_speed_ctrl_pure.py
"""
import collections
import math
import random
import time

import pytest

from jetformer import safety_caps as sc
from jetformer import speed_auto as sa

N = 1496.0                 # モデル B のスロットルの中立
P = sc.CeilParams()


def step(st, t, v, a=0.0, steer=0.0, gz=0.0, thr=1100.0, active=True, p=P):
    return sc.ceil_step(st, t, v, a, steer, gz, thr, N, active, p)


# ================================================================ D-c1 天井の決まり
def test_hold_us_and_params():
    assert sc.ceil_hold_us(4.0) == pytest.approx(252.96, abs=0.01)
    assert sc.ceil_hold_us(4.5) == pytest.approx(290.97, abs=0.01)
    assert sc.ceil_hold_us(0.1) == 0.0
    p = sc.ceil_params(4.3, 4.6)                 # 左旋回中の天井は全体より大きくしない
    assert p.cap_all == 4.3 and p.cap_left == 4.3 and p.cap_left_eff() == 4.3
    assert sc.ceil_params().cap_all == 4.5 and sc.ceil_params().cap_left == 4.0
    assert (P.left_steer_us, P.left_gz_dps, P.left_hold_s, P.release_accel) == (100.0, 45.0, 0.25, 1.5)
    assert (P.kp, P.ki, P.pred_s, P.accel_max, P.neutral_margin) == (120.0, 380.0, 0.10, 3.0, 0.20)


def test_invariants_random_and_deterministic():
    """出力は いつも 入力以上・中立以下(前進の入力)。前進でない入力はそのまま。同じ入力の列で同じ出力"""
    rng = random.Random(3)
    seq = []
    t = 0.0
    for i in range(20000):
        t += rng.choice([1 / 30, 1 / 30, 1 / 30, 0.07, 0.6])
        a = rng.choice([float('nan'), rng.uniform(-6, 6)])
        seq.append((t, rng.uniform(0, 6.5), a, rng.uniform(-400, 400), rng.uniform(-300, 300),
                    rng.choice([rng.uniform(1000, 1496), rng.uniform(1100, 1400), N, rng.uniform(1496, 2000)]),
                    rng.random() < 0.9))
    outs = []
    st = sc.CeilState()
    modes = collections.Counter()
    for t, v, a, s, g, thr, act in seq:
        st, o = step(st, t, v, a, s, g, thr, act)
        outs.append(o)
        modes[o.mode] += 1
        assert o.thr_us >= thr
        if thr < N:
            assert o.thr_us <= N
        else:
            assert o.thr_us == thr and o.mode in (sc.CEIL_PASS, sc.CEIL_OFF)
        if not act:
            assert o.mode == sc.CEIL_OFF and o.thr_us == thr and st == sc.CeilState()
        if o.mode == sc.CEIL_PASS:
            assert o.thr_us == thr                   # そのまま = ビット単位で同じ
        if o.mode == sc.CEIL_NEUTRAL:
            assert o.thr_us == N
    assert all(modes[m] > 100 for m in (sc.CEIL_OFF, sc.CEIL_PASS, sc.CEIL_LIMIT, sc.CEIL_NEUTRAL))
    st = sc.CeilState()
    for (t, v, a, s, g, thr, act), o0 in zip(seq, outs):
        st, o = step(st, t, v, a, s, g, thr, act)
        assert o == o0 or (o.mode == o0.mode and o.thr_us == o0.thr_us and o.allow_us == o0.allow_us)


def test_left_rule_hold_and_release_ramp():
    st = sc.CeilState()
    st, o = step(st, 0.0, 3.0)
    assert not o.left and o.cap == 4.5
    st, o = step(st, 0.033, 3.0, steer=99.9, gz=44.9)
    assert not o.left
    st, o = step(st, 0.066, 3.0, steer=100.0)
    assert o.left and o.cap == 4.0                                       # 下げるのはすぐ
    st, o = step(st, 0.1, 3.0, gz=45.0)
    assert o.left and o.cap == 4.0
    st, o = step(st, 0.35, 3.0)                                          # 切れてから 0.25 秒は左旋回のまま
    assert o.left and o.cap == 4.0
    st, o = step(st, 0.36, 3.0)
    assert not o.left and o.cap == pytest.approx(4.0 + 1.5 * 0.01)       # 1.5 m/s² で戻す
    t, cap = 0.36, o.cap
    while cap < 4.5:
        t += 1 / 30
        st, o = step(st, t, 3.0)
        assert o.cap - cap <= 1.5 / 30 + 1e-9
        cap = o.cap
    assert cap == 4.5 and t == pytest.approx(0.36 + 0.5 / 1.5, abs=0.04)
    st, o = step(st, t + 0.01, 3.0, steer=float('nan'), gz=float('nan'))  # nan は左旋回にしない
    assert not o.left


def test_modes_engage_limit_neutral_release():
    st = sc.CeilState()
    st, o = step(st, 0.0, 4.2, a=0.0, thr=1100.0)                        # e = −0.3: まだ掛けない
    assert o.mode == sc.CEIL_PASS and not o.engaged and o.thr_us == 1100.0
    st, o = step(st, 0.033, 4.35, a=1.0, thr=1100.0)                     # v̂ = 4.45 → e = −0.05 ≥ −0.10: 掛け始め
    assert o.engaged and o.v_pred == pytest.approx(4.45)
    assert o.allow_us == pytest.approx(min(396.0, sc.ceil_hold_us(4.5) + 120.0 * 0.05))   # 保つ量から(仕様からの直し)
    assert o.mode == sc.CEIL_LIMIT and N - o.thr_us == pytest.approx(o.allow_us)
    st, o = step(st, 0.066, 4.71, a=0.0, thr=1100.0)                     # v ≥ C + 0.2 → 中立
    assert o.mode == sc.CEIL_NEUTRAL and o.thr_us == N
    st, o = step(st, 0.1, 4.5, a=0.0, thr=1300.0)                        # 入力が許す量より弱い → そのまま
    assert o.mode == sc.CEIL_PASS and o.thr_us == 1300.0 and o.engaged
    st, o = step(st, 0.133, 4.5, thr=1600.0)                             # ブレーキの入力はそのまま・掛けている状態を外す
    assert o.mode == sc.CEIL_PASS and o.thr_us == 1600.0 and not o.engaged
    st, o = step(st, 0.166, 4.5, thr=N)                                  # 中立の入力
    assert o.thr_us == N and o.mode == sc.CEIL_PASS
    # 外す: e < −0.2 が 0.2 秒続き、PI の許す量が入力まで戻ったら
    st = sc.CeilState()
    st, o = step(st, 0.0, 4.6, thr=1300.0)                               # e ≥ 0: min(入力 196, 保つ量 291) = 196
    assert o.engaged and o.allow_us == pytest.approx(196.0)
    t = 0.0
    for _ in range(30):
        t += 1 / 30
        st, o = step(st, t, 3.5, thr=1300.0)
    assert not o.engaged and o.mode == sc.CEIL_PASS
    # 0.5 秒を超えて間が空いたら状態を戻す
    st = sc.CeilState()
    st, o = step(st, 0.0, 4.6, thr=1100.0)
    assert o.engaged
    st, o = step(st, 0.6, 4.0, thr=1100.0)
    assert not o.engaged and o.cap == 4.5
    # 動かさないフレームは状態を戻す
    st, o = step(st, 0.633, 4.6, thr=1100.0, active=False)
    assert st == sc.CeilState() and o.mode == sc.CEIL_OFF and math.isnan(o.cap)


def test_spec_literal_variant_flags():
    """直しの 2 つを外すと仕様の文のまま(掛け始めは入力のまま・外すのは 0.2 秒で)"""
    lit = P._replace(start_from_hold=False, release_bumpless=False)
    st, o = step(sc.CeilState(), 0.0, 4.35, a=1.0, thr=1100.0, p=lit)
    assert o.engaged and o.allow_us == 396.0 and o.mode == sc.CEIL_PASS
    # 天井の少し下(e = −0.25)が続く: 仕様の文のままは 0.2 秒で外れて入力の前進がいきなり戻る。直しありは許す量が
    # 入力まで戻るまで外さない(大きく下がったときは P の項ですぐ入力まで戻るので、どちらも外れる)
    for p_, engaged_after in ((lit, False), (P, True)):
        st = sc.CeilState()
        st, o = step(st, 0.0, 4.6, thr=1100.0, p=p_)
        t = 0.0
        for k in range(7):
            t += 1 / 30
            st, o = step(st, t, 4.25, thr=1100.0, p=p_)
        assert o.engaged == engaged_after
        if engaged_after:
            assert o.allow_us < 396.0 and o.mode == sc.CEIL_LIMIT
    st = sc.CeilState()
    st, o = step(st, 0.0, 4.6, thr=1100.0)
    t = 0.0
    for k in range(7):
        t += 1 / 30
        st, o = step(st, t, 3.5, thr=1100.0)
    assert not o.engaged and o.thr_us == 1100.0


def test_left_drop_limits_allow_to_hold():
    st = sc.CeilState()
    st, o = step(st, 0.0, 4.45, thr=1100.0)
    assert o.engaged and o.cap == 4.5
    st, o = step(st, 0.033, 4.3, thr=1100.0, steer=150.0)                # 左旋回: 天井が 4.0 に下がる
    assert o.left and o.cap == 4.0 and o.mode == sc.CEIL_NEUTRAL        # v ≥ 4.2
    assert o.allow_us <= sc.ceil_hold_us(4.0)


# ================================================================ D-c4 閉ループ(偽物の車)
CAL_U, CAL_T = sc.CAL_T_KNOTS_U, sc.CAL_T_VALUES


def T_of(f_us):
    """前進の量 f [µs] → 較正の T(u) [m/s](u = f ÷ 498 × 7.24)"""
    u = min(max(f_us, 0.0) / sc.FWD_SPAN_US * sc.MAX_SPEED_CMD, CAL_U[-1])
    for j in range(len(CAL_U) - 1):
        if CAL_U[j] <= u <= CAL_U[j + 1]:
            return CAL_T[j] + (CAL_T[j + 1] - CAL_T[j]) * (u - CAL_U[j]) / (CAL_U[j + 1] - CAL_U[j])
    return CAL_T[-1]


def car_sim(b, cmd=400.0, v0=0.0, dur=8.0, kap_fn=None, steer_fn=None, noise=0.0, seed=0, p=P,
            tau=0.32, dead=0.06, c0=0.58, ks=0.40, a_cap=4.12):
    """1 次の遅れの車: 目標 T(u)·b へ τ 0.32 秒(駆動は最大 4.12 m/s²、P_cal)、むだ時間 0.06 秒、
    弱める・中立のときの減速は惰性 0.58 m/s² まで、曲がりのこすれ 0.40·v²·|κ|。車両のサンプル 100 Hz、カメラ 30 fps。
    天井の入力は走行ループと同じ(最新 3 サンプルの中央値・0.1 秒前との差)"""
    rnd = random.Random(seed)
    dtp = 0.001
    v = v0
    veh = []
    st = sc.CeilState()
    f_out = cmd
    pend = collections.deque([cmd] * int(round(dead / dtp)))
    rows = []
    next_v, next_c = 0.0, 0.0
    engaged_prev, engages = False, 0
    for i in range(int(round(dur / dtp))):
        t = i * dtp
        kap = kap_fn(t) if kap_fn else 0.0
        if t >= next_v - 1e-9:
            meas = v + (rnd.gauss(0.0, noise) if noise else 0.0)
            if noise and rnd.random() < 0.01:
                meas += rnd.choice((-1.5, 1.5))                       # ホールの跳ね
            veh.append((t, meas))
            next_v += 0.01
        if t >= next_c - 1e-9:
            next_c += 1 / 30
            vm = sorted(x[1] for x in veh[-3:])[min(1, len(veh) - 1)] if len(veh) >= 3 else veh[0][1]
            old = [x for x in veh if x[0] <= t - 0.1]
            a = (vm - sorted(x[1] for x in old[-3:])[1]) / 0.1 if len(old) >= 3 else float('nan')
            steer = steer_fn(t) if steer_fn else (200.0 if kap > 0 else 0.0)
            st, o = sc.ceil_step(st, t, vm, a, steer, math.degrees(v * kap), N - cmd, N, True, p)
            assert N - cmd <= o.thr_us <= N
            f_out = N - o.thr_us
            engages += int(o.engaged and not engaged_prev)
            engaged_prev = o.engaged
            rows.append((t, v, f_out, o.mode, o.cap))
        pend.append(f_out)
        f = pend.popleft()
        tgt = T_of(f) * b if f > 0 else 0.0
        ad = min((tgt - v) / tau, a_cap)
        ad = max(ad, -c0)
        v = max(0.0, v + (ad - ks * v * v * kap) * dtp)
    return rows, engages


def reversals_per_s(rows, t_from, thr_us=1.0):
    """出力の増減の向きの入れ替わり [回/秒](thr_us 未満の動きは数えない)"""
    seg = [r for r in rows if r[0] >= t_from]
    if len(seg) < 2:
        return 0.0
    signs, ref = [], seg[0][2]
    for r in seg[1:]:
        if abs(r[2] - ref) >= thr_us:
            signs.append(1 if r[2] > ref else -1)
            ref = r[2]
    n = sum(1 for x, y in zip(signs, signs[1:]) if x != y)
    return n / max(seg[-1][0] - seg[0][0], 1e-9)


@pytest.mark.parametrize('b', [0.75, 1.0])
def test_closed_loop_straight(b):
    """モデルの指令 400 µs: 行き過ぎ ≤ C + 0.15、C に届いて 1.0 秒で ±0.05、落ち着いたあとの入れ替わり ≤ 2 回/秒"""
    C = 4.5
    rows, engages = car_sim(b)
    vmax = max(r[1] for r in rows)
    assert vmax <= C + 0.15
    reach = next((r[0] for r in rows if r[1] >= C - 0.05), None)
    free_top = T_of(400.0) * b
    if free_top > C:
        assert reach is not None
        assert all(abs(r[1] - C) <= 0.05 for r in rows if r[0] >= reach + 1.0)
        assert reversals_per_s(rows, reach + 1.0) <= 2.0
        assert engages == 1 and not any(r[3] == sc.CEIL_NEUTRAL for r in rows)
    else:                                       # 電池の弱い車(b 0.75)は天井に届かない: 天井は何もしない
        assert reach is None and all(r[2] == 400.0 for r in rows)


def test_closed_loop_straight_strong_battery_bound():
    """b = 1.3(満充電より強い)・駆動の上限 4.12 m/s² で天井に近づく: 行き過ぎは C + 0.25 まで(中立 C + 0.2 が止める)。
    仕様の C + 0.15 には届かない(別の試験で xfail。build_drive.md・7 章の課題)。C に届いて 1.0 秒で ±0.05・入れ替わり ≤ 2"""
    C = 4.5
    rows, engages = car_sim(1.3)
    assert max(r[1] for r in rows) <= C + 0.25
    reach = next(r[0] for r in rows if r[1] >= C - 0.05)
    assert all(abs(r[1] - C) <= 0.05 for r in rows if r[0] >= reach + 1.0)
    assert reversals_per_s(rows, reach + 1.0) <= 2.0
    assert engages == 1


@pytest.mark.xfail(strict=True, reason='b = 1.3・4.12 m/s² で近づくと C + 0.23 m/s(仕様の C + 0.15 に届かない。build_drive.md)')
def test_closed_loop_straight_strong_battery_spec_overshoot():
    rows, _ = car_sim(1.3)
    assert max(r[1] for r in rows) <= 4.5 + 0.15


@pytest.mark.parametrize('b', [0.75, 1.0, 1.3])
def test_closed_loop_left_turn_entry(b):
    """4.4 m/s で左旋回(曲率 0.4/m)に入ったら 0.5 秒で 4.05 m/s 以下"""
    rows, _ = car_sim(b, v0=4.4, dur=2.0, kap_fn=lambda t: 0.4 if t >= 0.3 else 0.0)
    v_at = next(r[1] for r in rows if r[0] >= 0.8)
    assert v_at <= 4.05
    assert all(r[4] == 4.0 for r in rows if 0.31 <= r[0] <= 2.0)


@pytest.mark.parametrize('b', [0.9, 1.0, 1.2, 1.3])
@pytest.mark.parametrize('kap', [0.15, 0.4])
def test_closed_loop_long_left_turn_with_noise(b, kap):
    """長い左旋回(4 秒)・速さの雑音 0.03 m/s とホールの跳ね: 旋回の中は 4.0 + 0.05 以下、掛け直しは 3 回まで"""
    rows, engages = car_sim(b, cmd=450.0, kap_fn=lambda t: kap if 2.0 <= t < 6.0 else 0.0,
                            steer_fn=lambda t: 120.0 if 2.0 <= t < 6.0 else 0.0, noise=0.03, seed=2)
    turn = [r for r in rows if 2.6 <= r[0] < 6.0]
    assert max(r[1] for r in turn) <= 4.0 + 0.05
    assert engages <= 3


def test_closed_loop_spec_literal_overshoots():
    """記録用: 仕様の文のまま(直し 2 つなし)だと、b = 1.0 で C + 0.25・長い左旋回で 4.3 m/s(直しの理由)"""
    lit = P._replace(start_from_hold=False, release_bumpless=False)
    rows, _ = car_sim(1.0, p=lit)
    assert max(r[1] for r in rows) > 4.5 + 0.2
    rows, _ = car_sim(1.0, cmd=450.0, kap_fn=lambda t: 0.15 if 2.0 <= t < 6.0 else 0.0,
                      steer_fn=lambda t: 120.0 if 2.0 <= t < 6.0 else 0.0, noise=0.03, seed=2, p=lit)
    assert max(r[1] for r in rows if 2.6 <= r[0] < 6.0) > 4.25


# ================================================================ D-a1 電圧の前送り
def test_ff_of_voltage():
    assert sa.ff_of_voltage(8.35) == pytest.approx(1.000)
    assert sa.ff_of_voltage(7.80) == pytest.approx(1.0705, abs=5e-4)
    assert sa.ff_of_voltage(8.10) == pytest.approx(1.031, abs=5e-4)
    assert sa.ff_of_voltage(7.70) == pytest.approx(1.084, abs=5e-4)
    assert sa.ff_of_voltage(6.0) == 1.15 and sa.ff_of_voltage(9.0) == pytest.approx(8.35 / 9.0)
    assert sa.ff_of_voltage(float('nan')) == 1.0 and sa.ff_of_voltage(0.0) == 1.0
    assert sa.ff_of_voltage(8.0, sa.auto_params(vref_v=8.0)) == 1.0
    # 6.0〜9.0 V の外(分圧の外れ 0.5 V・3.0 V、張り付き 12.75 V)は「無し」と同じ 1.00(fix.md。前は 1.15 / 0.90 まで振れた)
    for bad in (0.5, 3.0, 5.99, 9.01, 12.0, 12.75, float('inf')):
        assert sa.ff_of_voltage(bad) == 1.0, bad


def test_vbat_level_upper_envelope():
    xs = [8.0] * 200 + [7.4] * 100                         # 3 秒のうち 1 秒だけ 0.6 V 下がる(強い加速)
    assert sa.vbat_level(xs) == pytest.approx(8.0)
    assert math.isnan(sa.vbat_level([8.0] * 99))
    assert sa.vbat_level([8.0] * 99 + [float('nan')] * 5 + [8.1]) == pytest.approx(8.0)     # nan は数えない・100 で出す
    assert sa.vbat_level([8.0] * 50 + [8.4] * 50) == pytest.approx(8.4)
    # 6.0〜9.0 V の外は数えない(fix.md): 壊れた値が混ざっても包絡は変わらず、壊れた値だけなら出さない
    assert sa.vbat_level([8.0] * 100 + [12.75] * 30 + [0.5] * 30) == pytest.approx(8.0)
    assert math.isnan(sa.vbat_level([0.5] * 300)) and math.isnan(sa.vbat_level([12.75] * 300))
    assert sa.vbat_level([8.1] * 20, min_n=20) == pytest.approx(8.1) and math.isnan(sa.vbat_level([8.1] * 19, min_n=20))


def test_v_before_of_description():
    assert sa.v_before_of('battery=#2 V_before=8.31 V_after=__') == 8.31
    assert sa.v_before_of('V_before=8.31V') == 8.31
    assert math.isnan(sa.v_before_of('V_before=__ memo=x'))
    assert math.isnan(sa.v_before_of('V_before=12.0'))                   # 6.0〜9.0 V の外
    assert sa.v_before_of('V_before=__ Vb_before=8.20 V_src=board') == 8.20
    assert sa.v_before_of('V_before=8.40 Vb_before=8.20') == 8.40        # 人の数が先
    assert math.isnan(sa.v_before_of(''))


def test_rate_limit_and_m():
    assert sa.rate_limit(None, 1.07, 0.01) == 1.07
    assert sa.rate_limit(1.0, 1.07, 0.01) == 1.01
    assert sa.rate_limit(1.0, 0.95, 0.03) == 0.97
    assert sa.m_of(1.15, 1.10) == 1.20 and sa.m_of(0.90, 0.90) == 0.85
    assert sa.percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)        # numpy.percentile と同じ


# ================================================================ D-a2 周の直し
def lapst(v90, ok=True, reason='', collision=False, ceil_frac=0.0, a3=3.0, dur=8.0, g=float('nan')):
    return sa.LapStats(0.0, dur, dur, 240, v90, ok, reason, collision, ceil_frac, a3, g)


P90 = sa.auto_params(metric=sa.METRIC_P90)      # 仕様の文のままの物差し(周の速さ p90)。D-a2・D-a4 の p90 の試験はこれで
T_STAR = sa.AutoParams().target_lap_s()         # 純粋な関数の既定の目標 8.22 秒(fix.md)。走行の既定は 8.12 秒(drive.DEFAULT_PARAMS)


def lapt(dur, **kw):
    """周の時間の物差しの周(p90 は使わないので 4.0 のまま)"""
    return lapst(4.0, dur=dur, **kw)


def test_default_metric_is_lap_time():
    p = sa.AutoParams()
    assert p.metric == sa.METRIC_LAP and T_STAR == pytest.approx(14.52 - 1.50 * 4.20) == pytest.approx(8.22)
    assert sa.auto_params(4.15).target_lap_s() == pytest.approx(8.295)        # 4.15 m/s ≈ 8.3 秒(spec 9 章の 3)
    assert (p.deadband_s, p.ceil_frac_max_lap, p.hold_max_n, p.vboard_lo, p.vboard_hi) == (0.10, 0.35, 4, 6.0, 9.0)


def test_fb_lap_rules():
    st = sa.FbState()
    st, ev, d = sa.fb_lap(st, lapst(3.9), 1.0, P90)                     # 遅い → 上げる(1 周 +0.03 まで)
    assert ev == sa.EV_FIXED and d == pytest.approx(0.03) and st.m_fb == pytest.approx(math.exp(0.03))
    st2, ev, d = sa.fb_lap(st, lapst(4.12), 1.0, P90)                    # 0.5 × ln(4.2/4.12) = 0.0096
    assert ev == sa.EV_FIXED and d == pytest.approx(0.5 * math.log(4.2 / 4.12))
    st3, ev, d = sa.fb_lap(st, lapst(4.24), 1.0, P90)                    # 不感帯 ±0.05
    assert ev == sa.EV_DEADBAND and d == 0.0 and st3.m_fb == st.m_fb
    st4, ev, d = sa.fb_lap(st, lapst(4.6), 1.0, P90)                     # 速い → 下げる
    assert ev == sa.EV_FIXED and d == pytest.approx(-0.03)
    s = sa.FbState()
    for _ in range(10):
        s, ev, d = sa.fb_lap(s, lapst(3.4), 1.0, P90)
    assert s.m_fb == pytest.approx(1.10) and s.warn == sa.WARN_RAIL and s.laps_used == 10
    s = sa.FbState()
    for _ in range(10):
        s, ev, d = sa.fb_lap(s, lapst(5.0), 1.0, P90)
    assert s.m_fb == pytest.approx(0.90) and s.warn == sa.WARN_RAIL


def test_fb_lap_rules_lap_time():
    """周の時間の物差し: 遅い周 → 上げる(1 周 +0.03 まで)・不感帯 ±0.10 秒(T* ± 0.10)・速い周 → 下げる・端で知らせ"""
    st = sa.FbState()
    st, ev, d = sa.fb_lap(st, lapt(8.70), 1.0)                          # 0.5 × ln(8.70/T*)
    assert ev == sa.EV_FIXED and d == pytest.approx(0.5 * math.log(8.70 / T_STAR))
    s, ev, d = sa.fb_lap(sa.FbState(), lapt(9.40), 1.0)
    assert ev == sa.EV_FIXED and d == pytest.approx(0.03)
    for dur in (T_STAR - 0.10 + 1e-6, T_STAR, T_STAR + 0.10 - 1e-6):
        s, ev, d = sa.fb_lap(st, lapt(dur), 1.0)
        assert ev == sa.EV_DEADBAND and d == 0.0 and s.m_fb == st.m_fb, dur
    s, ev, d = sa.fb_lap(st, lapt(7.90), 1.0)                            # 速い → 下げる
    assert ev == sa.EV_FIXED and d == pytest.approx(0.5 * math.log(7.90 / T_STAR)) and d < 0
    s = sa.FbState()
    for _ in range(10):
        s, ev, d = sa.fb_lap(s, lapt(9.5), 1.0)
    assert s.m_fb == pytest.approx(1.10) and s.warn == sa.WARN_RAIL and s.laps_used == 10
    s = sa.FbState()
    for _ in range(10):
        s, ev, d = sa.fb_lap(s, lapt(7.0), 1.0)
    assert s.m_fb == pytest.approx(0.90) and s.warn == sa.WARN_RAIL
    # p90 は物差しに使わない: 天井で 3.85 に頭打ちでも、周が目標の中なら直さない(仕様の p90 なら毎周「上げたい」)
    s, ev, d = sa.fb_lap(sa.FbState(), lapst(3.85, dur=8.25), 1.0)
    assert ev == sa.EV_DEADBAND
    s, ev, d = sa.fb_lap(sa.FbState(), lapst(3.85, dur=8.25), 1.0, P90)
    assert ev == sa.EV_FIXED and d > 0


def test_fb_lap_no_raise_conditions():
    st = sa.FbState()
    s, ev, d = sa.fb_lap(st, lapst(3.9, ceil_frac=0.10), 1.0, P90)
    assert ev == sa.EV_NORAISE_CEIL and s.m_fb == 1.0
    s, ev, d = sa.fb_lap(st, lapst(3.9, ceil_frac=0.099), 1.0, P90)
    assert ev == sa.EV_FIXED and s.m_fb > 1.0
    s, ev, d = sa.fb_lap(st, lapst(3.9, a3=6.0), 1.0, P90)
    assert ev == sa.EV_NORAISE_BOARD and s.m_fb == 1.0
    s, ev, d = sa.fb_lap(st, lapst(3.9, ok=False, reason='collision', collision=True), 1.0, P90)
    assert ev == sa.EV_UNUSED and s.m_fb == 1.0 and s.no_raise_next and dict(s.skipped) == {'collision': 1}
    s2, ev, d = sa.fb_lap(s, lapst(3.9), 1.0, P90)                       # 衝突の次の 1 周は上げない
    assert ev == sa.EV_NORAISE_COLL and s2.m_fb == 1.0 and not s2.no_raise_next
    s3, ev, d = sa.fb_lap(s2, lapst(3.9), 1.0, P90)                      # その次は上げる
    assert ev == sa.EV_FIXED and s3.m_fb > 1.0
    s4, ev, d = sa.fb_lap(s, lapst(4.6, ceil_frac=0.5, a3=7.0), 1.0, P90)   # 下げるのは、上げない条件があっても
    assert ev == sa.EV_FIXED and s4.m_fb < 1.0


def test_fb_lap_no_raise_conditions_lap_time():
    """周の時間の物差し: 天井は 35% 以上で上げない(p90 の 10% では No.9 の水準(約 11〜25%)で一度も上げられない)。
    板の衝撃・衝突の次は同じ。下げるのは、上げない条件があっても"""
    st = sa.FbState()
    s, ev, d = sa.fb_lap(st, lapt(8.7, ceil_frac=0.35), 1.0)
    assert ev == sa.EV_NORAISE_CEIL and s.m_fb == 1.0
    s, ev, d = sa.fb_lap(st, lapt(8.7, ceil_frac=0.25), 1.0)
    assert ev == sa.EV_FIXED and s.m_fb > 1.0
    s, ev, d = sa.fb_lap(st, lapt(8.7, a3=6.0), 1.0)
    assert ev == sa.EV_NORAISE_BOARD and s.m_fb == 1.0
    s, ev, d = sa.fb_lap(st, lapt(8.7, ok=False, reason='collision', collision=True), 1.0)
    s2, ev, d = sa.fb_lap(s, lapt(8.7), 1.0)
    assert ev == sa.EV_NORAISE_COLL and s2.m_fb == 1.0
    s4, ev, d = sa.fb_lap(st, lapt(7.8, ceil_frac=0.6, a3=7.0), 1.0)
    assert ev == sa.EV_FIXED and s4.m_fb < 1.0


def test_fb_abandon_collision_lap():
    """閉じられなかった周(衝突 → 人が持って止めた)に衝突があれば、使わない周(衝突)と数えて次の 1 周は上げない"""
    st = sa.FbState(m_fb=1.03)
    s, ev = sa.fb_abandon(st, False)
    assert s == st and ev == sa.EV_NONE
    s, ev = sa.fb_abandon(st, True)
    assert ev == sa.EV_ABANDON_COLL and s.no_raise_next and dict(s.skipped) == {'collision': 1} and s.m_fb == 1.03
    s2, ev, d = sa.fb_lap(s, lapt(8.9), 1.0)
    assert ev == sa.EV_NORAISE_COLL and s2.m_fb == 1.03


def frames_lap(t0, t1, v=4.0):
    n = int(math.ceil((t1 - t0) * 30 - 1e-9))
    return [sa.LapFrame(t0 + k / 30.0, v, True, 0, False, 0.2, 1.0, False) for k in range(n)]


def test_lap_stats_short_holds_and_gain():
    """周の中の中立止めは 4 フレーム(約 0.13 秒)まで使う(No.9 は 20 秒ごとに 1〜4 フレーム。fix.md)。人・dry run は 1 つでも
    使わない。掛けたゲインの平均(AI のフレーム)"""
    fs = [f._replace(gain=0.9 if f.t < 4.0 else 1.0) for f in frames_lap(0.0, 8.0)]
    st = sa.lap_stats(fs, 0.0, 8.0)
    assert st.ok and st.holds == 0 and st.g_mean == pytest.approx((120 * 0.9 + 120 * 1.0) / 240)
    hold = dict(ai=False, hold=True, v=float('nan'), gain=float('nan'))
    g = [f._replace(**hold) if 100 <= k < 104 else f for k, f in enumerate(fs)]
    st = sa.lap_stats(g, 0.0, 8.0)
    assert st.ok and st.holds == 4 and st.reason == ''
    g = [f._replace(**hold) if 100 <= k < 105 else f for k, f in enumerate(fs)]
    assert sa.lap_stats(g, 0.0, 8.0).reason == 'not_ai'
    g = [f._replace(**hold) if k in (10, 11, 120, 200, 230) else f for k, f in enumerate(fs)]     # 合わせて 5 つ
    assert sa.lap_stats(g, 0.0, 8.0).reason == 'not_ai'
    g = [f._replace(ai=False) if k == 100 else f for k, f in enumerate(fs)]                      # 人のフレーム
    assert sa.lap_stats(g, 0.0, 8.0).reason == 'not_ai'
    assert sa.lap_stats([f._replace(**hold) for f in fs], 0.0, 8.0).reason == 'not_ai'
    assert math.isnan(sa.lap_stats(frames_lap(0.0, 8.0), 0.0, 8.0).g_mean)


def test_lap_stats_unusable_laps():
    ok = sa.lap_stats(frames_lap(0.0, 8.0), 0.0, 8.0)
    assert ok.ok and ok.v90 == pytest.approx(4.0) and ok.reason == '' and ok.n == 240
    assert sa.lap_stats(frames_lap(0.0, 4.9), 0.0, 4.9).reason == 'short'
    assert sa.lap_stats(frames_lap(0.0, 16.2), 0.0, 16.2).reason == 'long'
    fs = frames_lap(0.0, 8.0)
    k = 100
    for name, change, reason in (('hold', dict(ai=False), 'not_ai'), ('assist', dict(assist=2), 'assist'),
                                 ('guard', dict(guard=True), 'guard'), ('hit', dict(impact_g=2.5), 'collision')):
        g = list(fs)
        g[k] = g[k]._replace(**change)
        st = sa.lap_stats(g, 0.0, 8.0)
        assert not st.ok and st.reason == reason, name
        assert st.collision == (reason == 'collision')
    g = fs[:100] + fs[107:]                                              # 0.2 秒を超えて空いた
    assert sa.lap_stats(g, 0.0, 8.0).reason == 'gap'
    g = [f._replace(v=0.1) if 2.0 <= f.t <= 3.05 else f for f in fs]      # 1 秒以上止まった
    assert sa.lap_stats(g, 0.0, 8.0).reason == 'stopped'
    g = [f._replace(ceil_lim=True) if f.t < 1.0 else f for f in fs]
    assert sa.lap_stats(g, 0.0, 8.0).ceil_frac == pytest.approx(30 / 240)
    g = [f._replace(a3_g=6.4) if k == 10 else f for k, f in enumerate(fs)]
    assert sa.lap_stats(g, 0.0, 8.0).a3_max == 6.4


def test_unused_laps_do_not_change_m_fb():
    st = sa.FbState(m_fb=1.04)
    for reason in sa.SKIP_REASONS:
        s, ev, d = sa.fb_lap(st, lapst(3.5, ok=False, reason=reason, collision=reason == 'collision'), 1.0)
        assert ev == sa.EV_UNUSED and s.m_fb == 1.04 and s.laps_used == 0


# ================================================================ D-a3 ゲイン
def test_gain_of_raise_limit():
    assert sa.gain_of(0.9, 1.0) == pytest.approx(0.9)
    assert sa.gain_of(0.9, 1.2) == pytest.approx(1.08)
    assert sa.gain_of(1.0, 1.2) == pytest.approx(1.10)                  # 上げるのは 1.10 まで
    assert sa.gain_of(1.2, 1.1) == pytest.approx(1.2)                   # 人が 1.10 より上にしたときは上げない
    assert sa.gain_of(1.2, 0.9) == pytest.approx(1.08)                  # 下げるのはできる
    assert sa.gain_of(0.9, 0.85) == pytest.approx(0.765)


# ================================================================ D-a4 閉ループ(周の水準)
def lap_level(eps, v0, seed, laps=12, spread=0.1):
    """v90 = v0 × m^ε + ばらつき(一様 ±spread/2)。戻り値 [(m_fb の前, 雑音なしの v, Δ)]、最後の状態"""
    r = random.Random(seed)
    st = sa.FbState()
    hist = []
    for _ in range(laps):
        m = sa.m_of(1.0, st.m_fb)
        v = v0 * m ** eps
        prev = st.m_fb
        st, ev, d = sa.fb_lap(st, lapst(v + r.uniform(-spread / 2, spread / 2)), 1.0, P90)
        assert abs(math.log(st.m_fb / prev)) <= 0.03 + 1e-12                # 1 周 0.03 以下
        assert 0.90 - 1e-12 <= st.m_fb <= 1.10 + 1e-12                       # 端を越えない
        hist.append((prev, v, d))
    return hist, st


@pytest.mark.parametrize('eps', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('v0', [3.4, 4.0, 4.6])
def test_lap_level_closed_loop(eps, v0):
    target = 4.2
    m_star = (target / v0) ** (1.0 / eps)
    for seed in range(20):
        hist, st = lap_level(eps, v0, seed)
        vs = [h[1] for h in hist]
        if 0.90 <= m_star <= 1.10:              # 合わせられる組: 6 周で目標 ± 0.08、目標 + 0.1 を越えない
            assert all(abs(v - target) <= 0.08 for v in vs[6:]), (seed, vs)
            first_ok = next(i for i, v in enumerate(vs) if v <= target + 0.1)
            assert max(vs[first_ok:]) <= target + 0.1
        else:                                   # 合わせられない組: 端で止まり、張り付きの知らせ
            m_end = sa.m_of(1.0, st.m_fb)
            v_rail = v0 * m_end ** eps
            if abs(v_rail - target) > 0.08:
                assert st.m_fb in (pytest.approx(0.90), pytest.approx(1.10)) and st.warn == sa.WARN_RAIL, (seed, vs)


@pytest.mark.parametrize('v0', [3.4, 4.0, 4.6])
def test_lap_level_strong_elasticity_does_not_grow(v0):
    """ε = 3(10/24 の見積もり 0.5〜1.7 より強い): 揺れが大きくならない(後半の誤差の最大 ≤ 前半の最大、かつ 0.2 m/s 以下)"""
    for seed in range(20):
        hist, st = lap_level(3.0, v0, seed, laps=18)
        err = [abs(h[1] - 4.2) for h in hist]
        assert max(err[9:]) <= max(max(err[3:9]), 0.2) + 1e-9 and max(err[9:]) <= 0.2


def test_oscillation_warning_rule():
    """向きが 3 周続けて入れ替わる(+−+−)と「揺れています」。直さなかった周で切れる"""
    st = sa.FbState()
    for v in (4.0, 4.4, 4.0, 4.4):
        st, ev, d = sa.fb_lap(st, lapst(v), 1.0, P90)
    assert st.warn == sa.WARN_OSC and sa.WARN_OSC in st.warn_seen
    st = sa.FbState()
    for v in (4.0, 4.4, 4.2, 4.0, 4.4):                                   # 不感帯の周で切れる
        st, ev, d = sa.fb_lap(st, lapst(v), 1.0, P90)
    assert st.warn == '' and st.warn_seen == ()
    st = sa.FbState()
    for dur in (8.9, 7.6, 8.9, 7.6):                                      # 周の時間でも(大きな直しの +−+−)
        st, ev, d = sa.fb_lap(st, lapt(dur), 1.0)
    assert st.warn == sa.WARN_OSC


def test_oscillation_warning_ignores_lap_noise():
    """周のばらつき(±0.2〜0.3 秒)くらいの小さな直し(|Δ| < 0.02)の +−+− は「揺れています」にしない(fix.md。目標の近くで
    ばらつきだけで向きが入れ替わる = 4 周に 1 回くらい出てしまい、止める条件の知らせが当てにならなくなる)"""
    st = sa.FbState()
    for dur in (8.50, 7.95, 8.50, 7.95, 8.50, 7.95):
        st, ev, d = sa.fb_lap(st, lapt(dur), 1.0)
        assert ev == sa.EV_FIXED and abs(d) < 0.02
    assert st.warn == '' and st.warn_seen == ()


def test_end_summary():
    """目安 = 使った周ごとの「目標に合ったはずのゲイン」(掛けたゲインの平均 × r)の中央値。自動合わせを使うなら ÷ m_ff。
    ON・OFF とも同じ式(仕様の ON は throttle_gain × m_fb_end。fix.md)"""
    st = sa.FbState()
    for dur in (8.70, 8.60, 8.80):                    # OFF の影: 掛けたゲインは throttle_gain 0.9 のまま
        st, ev, d = sa.fb_lap(st, lapt(dur, g=0.9), 1.03)
    r = sa.percentile([8.70 / T_STAR, 8.60 / T_STAR, 8.80 / T_STAR], 50)
    e = sa.end_summary(st, False, 0.9)
    assert e['laps_used'] == 3 and e['suggest_laps'] == 3 and e['metric'] == 'lap_s'
    assert e['target_lap_s'] == pytest.approx(T_STAR)
    assert e['suggest_throttle_gain_fixed'] == pytest.approx(0.9 * r, abs=1e-3)
    assert e['suggest_throttle_gain_auto'] == pytest.approx(0.9 * r / 1.03, abs=1e-3)
    assert e['suggest_throttle_gain'] == e['suggest_throttle_gain_auto']
    # ON: 掛けたゲインの平均(0.9 × 1.03 × m_fb)から。3 周の予選でも 1 回で合わせる値が出る(m_fb_end ではない)
    st = sa.FbState()
    for dur, g in ((8.70, 0.927), (8.60, 0.954)):
        st, ev, d = sa.fb_lap(st, lapt(dur, g=g), 1.03)
    e = sa.end_summary(st, True, 0.9)
    want = sa.percentile([0.927 * 8.70 / T_STAR / 1.03, 0.954 * 8.60 / T_STAR / 1.03], 50)
    assert e['suggest_throttle_gain'] == pytest.approx(want, abs=1e-3) and e['suggest_laps'] == 2
    assert e['suggest_throttle_gain'] > 0.9 * st.m_fb                  # m_fb_end(1 周 0.03 まで)より目標に近い
    # 丸め: 自動合わせを使うなら 0.90〜1.10 倍、使わないなら 0.85〜1.20 倍
    st = sa.FbState()
    st, ev, d = sa.fb_lap(st, lapt(11.0, g=0.9), 1.0)
    e = sa.end_summary(st, False, 0.9)
    assert e['suggest_throttle_gain_auto'] == pytest.approx(0.99) and e['suggest_throttle_gain_fixed'] == pytest.approx(1.08)
    assert sa.end_summary(sa.FbState(), True, 0.9)['suggest_throttle_gain'] is None
    # p90 の物差し(仕様の文のまま)でも同じ形
    st = sa.FbState()
    for v in (3.9, 3.95, 4.0):
        st, ev, d = sa.fb_lap(st, lapst(v), 1.03, P90)
    e = sa.end_summary(st, False, 0.9, P90)
    r = sa.percentile([4.2 / 3.9, 4.2 / 3.95, 4.2 / 4.0], 50)
    assert e['suggest_throttle_gain_fixed'] == pytest.approx(0.9 * r, abs=1e-3)
    assert e['suggest_throttle_gain_auto'] == pytest.approx(0.9 * r / 1.03, abs=1e-3)


# ================================================================ D-a4 閉ループ(周の水準・天井あり。fix.md)
# 区間ごとの速さ(No.9 の水準、29 m・約 8.6 秒)。速さ = 基の速さ × (m × s)^ε(s = 車・電池の強さ、m = 自動合わせの倍率)。
# 天井: 左旋回の区間 4.0・ほか 4.5 m/s。天井 − 0.2 以上で「出力を変えた」、天井 − 0.1 で頭打ち、超えた分の半分は戻るまでの損。
# 記録の再生(verify_replay 5.3 節: ゲイン 0.90 → 8.70 秒・11%、0.96 → 8.31 秒・22%、0.99 → 8.16 秒・25%)とおおよそ同じ形
LAP_SECTIONS = (('R1', 3.0, 2.8, 2.8, None), ('GM', 3.3, 2.5, 2.7, 0.0), ('DE', 6.0, 3.0, 3.7, None),
                ('SW', 6.0, 3.4, 3.85, 0.0), ('HS', 6.0, 3.7, 4.3, 0.75), ('LH', 4.4, 4.0, 3.0, 0.0))


def _lap_profile(ds=0.05):
    out = []
    for _, length, u0, u1, left_from in LAP_SECTIONS:
        n = int(round(length / ds))
        for i in range(n):
            x = (i + 0.5) / n
            out.append((ds, u0 + (u1 - u0) * x, 4.0 if left_from is not None and x >= left_from else 4.5))
    return out


LAP_PROFILE = _lap_profile()


def ceiling_lap(k, ceiling=True, engage=0.2, under=0.1, rec=0.5):
    """速さの倍率 k の 1 周 → (周の時間, 天井が出力を変えたフレームの割合)"""
    total = capped = 0.0
    for ds, u, cap in LAP_PROFILE:
        v = u * k
        ve = v
        if ceiling and v > cap - under:
            ve = cap - under - rec * (v - (cap - under))
        dt = ds / ve
        total += dt
        capped += dt if ceiling and v >= cap - engage else 0.0
    return total, capped / total


def test_ceiling_lap_model_is_like_the_replay():
    t_off, _ = ceiling_lap(1.0, ceiling=False)
    t_on, f = ceiling_lap(1.0)
    assert 8.5 <= t_off <= 8.7 and 0.02 <= t_on - t_off <= 0.12 and 0.05 <= f <= 0.15
    t_on2, f2 = ceiling_lap(1.067)
    assert 8.1 <= t_on2 <= 8.35 and 0.18 <= f2 <= 0.32
    # 天井 ON では周の時間がゲインに効きにくい(効き 0.6〜0.9。物差しを p90 にすると天井で頭打ち)
    eff = -math.log(ceiling_lap(1.10)[0] / t_on) / math.log(1.10)
    assert 0.5 <= eff <= 0.9


def ceiling_loop(s, eps, seed=0, laps=12, noise=0.0):
    """周の水準の閉ループ(天井あり)。戻り値 [(m_fb の前, 雑音なしの周の時間, 天井の割合, 出来事, Δ)]、最後の状態"""
    r = random.Random(seed)
    st = sa.FbState()
    hist = []
    for _ in range(laps):
        k = (sa.m_of(1.0, st.m_fb) * s) ** eps
        t_lap, frac = ceiling_lap(k)
        prev = st.m_fb
        st, ev, d = sa.fb_lap(st, lapt(t_lap + (r.gauss(0.0, noise) if noise else 0.0), ceil_frac=frac), 1.0)
        assert abs(math.log(st.m_fb / prev)) <= 0.03 + 1e-12                # 1 周 0.03 以下
        assert 0.90 - 1e-12 <= st.m_fb <= 1.10 + 1e-12                       # 端を越えない
        hist.append((prev, t_lap, frac, ev, d))
    return hist, st


def reachable(s, eps):
    """m_fb 0.90〜1.10 の中で、天井 35% 未満のまま目標 ± 0.05 秒に合う m があるか"""
    for i in range(201):
        t_lap, frac = ceiling_lap(((0.90 + 0.001 * i) * s) ** eps)
        if abs(t_lap - T_STAR) <= 0.05 and frac < 0.35:
            return True
    return False


@pytest.mark.parametrize('eps', [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize('s', [0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20])
def test_lap_time_closed_loop_with_ceiling(s, eps):
    """雑音なし: 合わせられる組は 6 周で T* ± 0.10 秒に入り、入ったら出ない(行き過ぎない)。合わせられない組は端で止まって
    張り付きの知らせか、天井 35% で止まる(それより上げない)"""
    hist, st = ceiling_loop(s, eps)
    ts = [h[1] for h in hist]
    if reachable(s, eps):
        assert all(abs(t - T_STAR) <= 0.10 + 1e-6 for t in ts[6:]), ts
        k_in = next(i for i, t in enumerate(ts) if abs(t - T_STAR) <= 0.10 + 1e-6)
        assert all(abs(t - T_STAR) <= 0.10 + 1e-6 for t in ts[k_in:]), ts
        assert st.warn != sa.WARN_OSC
    else:
        end_t = ts[-1]
        if end_t > T_STAR + 0.10:                  # 遅いまま: 上げきった(1.10・知らせ)か、天井 35% で止めた
            assert (st.m_fb == pytest.approx(1.10) and st.warn == sa.WARN_RAIL) or hist[-1][3] == sa.EV_NORAISE_CEIL, ts
        elif end_t < T_STAR - 0.10:                # 速いまま: 下げきった(0.90・知らせ)
            assert st.m_fb == pytest.approx(0.90) and st.warn == sa.WARN_RAIL, ts


@pytest.mark.parametrize('eps', [0.5, 1.0])
@pytest.mark.parametrize('s', [1.0, 1.05, 1.10, 1.15])
def test_lap_time_closed_loop_with_lap_noise(s, eps):
    """周の時間のばらつき σ 0.2 秒(No.9 のきれいな周は 8.25〜8.96 秒): 6 周目からの雑音なしの周の時間の誤差の平均 ≤ 0.18 秒・
    最大 ≤ 0.45 秒、「揺れています」は出ない"""
    assert reachable(s, eps)
    for seed in range(30):
        hist, st = ceiling_loop(s, eps, seed=seed, laps=14, noise=0.2)
        err = [abs(h[1] - T_STAR) for h in hist[6:]]
        assert sum(err) / len(err) <= 0.18 and max(err) <= 0.45, (seed, [round(h[1], 2) for h in hist])
        assert sa.WARN_OSC not in st.warn_seen, seed


@pytest.mark.parametrize('s', [0.95, 1.0, 1.05, 1.10])
def test_lap_time_strong_elasticity_does_not_grow(s):
    """ε = 3(周の時間の効き 約 2。記録の見積もり 0.5〜1.6 より強い): 雑音 σ 0.2 秒でも広がらない(端に張り付かない・
    10 周目からの誤差の平均 ≤ 0.30 秒・最大 ≤ 1.0 秒。ε 1 では平均 ≤ 0.18 秒。揺れが育つなら端まで行く)"""
    for seed in range(20):
        hist, st = ceiling_loop(s, 3.0, seed=seed, laps=18, noise=0.2)
        err = [abs(h[1] - T_STAR) for h in hist]
        assert max(err[9:]) <= 1.0 and sum(err[9:]) / 9 <= 0.30, seed
        assert sa.WARN_RAIL not in st.warn_seen, seed


# ================================================================ D-t1(純粋な関数の時間)
def test_pure_function_speed():
    rng = random.Random(1)
    seq = [(i / 30.0, rng.uniform(3.0, 5.0), rng.uniform(-1, 3), rng.uniform(-200, 200), rng.uniform(-100, 150),
            rng.uniform(1100, 1450)) for i in range(10000)]
    st = sc.CeilState()
    t0 = time.perf_counter()
    for t, v, a, s, g, thr in seq:
        st, o = sc.ceil_step(st, t, v, a, s, g, thr, N, True, P)
    per = (time.perf_counter() - t0) / len(seq) * 1000.0
    assert per <= 0.1, per
    fs = frames_lap(0.0, 16.0)
    t0 = time.perf_counter()
    for _ in range(50):
        lap = sa.lap_stats(fs, 0.0, 16.0)
        sa.fb_lap(sa.FbState(), lap, 1.0)
    per_lap = (time.perf_counter() - t0) / 50 * 1000.0
    assert per_lap <= 1.0, per_lap
