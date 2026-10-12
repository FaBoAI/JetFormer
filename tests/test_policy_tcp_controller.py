"""TCP 型の制御器(jetformer/policies/controller.py の TCPPost)のテスト(numpy だけ・CPU・実データなし)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_policy_tcp_controller.py

車の約束: ステア > 中心 = 左、スロットルは中立より下 = 前進、v_now = −eRPM × 0.5e-3(前進が正)
"""

import math
import time

import numpy as np
import pytest

from jetformer.policies import controller as C
from jetformer.policies import post

CENTER, NEUTRAL = 1441.0, 1496.0
FRAME_NS = int(1e9 / 30)
OFFSETS = np.arange(-575.0, 576.0, 50.0)


def _table(slope=0.004, sat=1.5):
    """直線 κ = slope·ずれ を ±sat で切った表(3 つの速度帯とも同じ・狭義単調)"""
    k = np.clip(slope * OFFSETS, -sat, sat)
    k = np.maximum.accumulate(k) + np.arange(len(k)) * 1e-6
    return [k.tolist()] * 3


def _pp(**over):
    pp = {'kind': 'tcp_controller', 'version': 1, 'pwm_center': 1500.0, 'pwm_scale': 500.0,
          'speed_bins': [0.5 * i for i in range(10)], 'wp_dt': 0.1, 'fps': 30,
          'steer_table': {'lag_ms': 40, 'offset_us': OFFSETS.tolist(), 'bands_mps': [2.0, 3.0], 'curv': _table()},
          'throttle_ff': {'c0': 1530.2, 'c1': -82.7, 'c2': -25.4, 'tau_s': 0.33, 'fwd_max_us': 300.0},
          'rules': dict(C.RULES), 'pursuit': dict(C.PURSUIT), 'fusion': dict(C.FUSION)}
    pp.update(over)
    return pp


def _ctl(**over):
    return C.TCPPost(_pp(**over), (CENTER, NEUTRAL))


def _logits_for(v, bins=None):
    """期待値がほぼ v になる logits(v に近い2クラスに寄せる)"""
    bins = np.asarray(bins if bins is not None else [0.5 * i for i in range(10)])
    q = np.maximum(1.0 - np.abs(bins - v) / 0.5, 1e-6)
    return np.log(q / q.sum()).astype(np.float32)


def _arc(kappa, speed, n=10, dt=0.1):
    """曲率 kappa(左 正)・速さ speed の円弧の通過点 [n,2](x 前・y 左)"""
    s = speed * dt * np.arange(1, n + 1)
    if abs(kappa) < 1e-9:
        return np.stack([s, np.zeros(n)], 1).astype(np.float32)
    return np.stack([np.sin(kappa * s) / kappa, (1 - np.cos(kappa * s)) / kappa], 1).astype(np.float32)


def _out(actions=None, v_tgt=2.0, wp=None, rng=None):
    if actions is None:
        rng = rng or np.random.default_rng(0)
        actions = rng.normal(0, 0.2, (10, 2)).astype(np.float32)
    return {'actions': np.asarray(actions, np.float32), 'waypoints': _arc(0.0, 2.0) if wp is None else wp,
            'speed_logits': _logits_for(v_tgt)}


# ---------------------------------------------------------------- 符号
def test_left_waypoints_steer_left_and_right_waypoints_steer_right():
    c = _ctl()
    left, _ = c.step(_out(wp=_arc(0.8, 2.0)), 2.0, {'tcp_mode': 3}, 0)
    c.reset()
    right, _ = c.step(_out(wp=_arc(-0.8, 2.0)), 2.0, {'tcp_mode': 3}, 0)
    c.reset()
    straight, _ = c.step(_out(wp=_arc(0.0, 2.0)), 2.0, {'tcp_mode': 3}, 0)
    assert np.all(left[:, 0] > CENTER + 100) and np.all(right[:, 0] < CENTER - 100)
    assert np.allclose(straight[:, 0], CENTER, atol=1.0)
    # 円弧の上なら追う曲率 ≒ 円の曲率 → 表の逆引き 0.8 / 0.004 = 200 µs
    assert np.allclose(left[:, 0] - CENTER, 200.0, atol=15.0)


def test_pursuit_curvature_on_circle_and_extension():
    k, total = C.pursuit_curvatures(_arc(0.5, 2.0), 0.1)
    assert total == pytest.approx(2.0, abs=0.01) and np.allclose(k, 0.5, atol=0.03)      # 弦の折れ線なので ±数%
    k_r, _ = C.pursuit_curvatures(_arc(-1.2, 2.5), 0.1)
    assert np.allclose(k_r, -1.2, rtol=0.08)
    # 道の端を超える先の点は最後の区間の向きに延ばす(有限)
    k2, _ = C.pursuit_curvatures(_arc(0.5, 0.5), 0.1, t_look=2.0, l_min=3.0)
    assert np.all(np.isfinite(k2))
    assert C.pursuit_curvatures(np.zeros((10, 2)), 0.1)[0] is None


def test_faster_target_means_forward_throttle_and_slow_target_is_neutral():
    c = _ctl()
    out, info = c.step(_out(v_tgt=3.0), 2.0, {'tcp_mode': 1}, 0)
    assert info['v_hat'] == pytest.approx(3.0, abs=0.05) and np.all(out[:, 1] < NEUTRAL)
    assert np.all(out[:, 1] == out[0, 1])                                   # 10 行同じ
    assert c.thr_ff(0.2, 0.0) == NEUTRAL                                    # v_cmd < 0.3 → 中立
    assert c.thr_ff(0.0, 0.5) == NEUTRAL                                    # 0.5 m/s で目標 0: 中立より上は出さない
    assert c.thr_ff(3.0, 2.0) < c.thr_ff(2.5, 2.0) <= NEUTRAL               # 目標が速いほど前進が強い


def test_brake_only_above_one_metre_per_second():
    c = _ctl()
    assert c.thr_ff(0.0, 1.5) == NEUTRAL + 150.0                            # 1 m/s 以上で目標 0 → ブレーキ
    assert c.thr_ff(0.0, 0.9) == NEUTRAL                                    # 1 m/s 未満は中立(後退の手順に入らない)
    assert c.thr_ff(1.4, 1.5) <= NEUTRAL                                    # 差が 0.3 以下はブレーキしない
    for v_now in np.linspace(-1.0, 4.5, 56):
        for v_t in np.linspace(0.0, 4.5, 46):
            thr = c.thr_ff(v_t, v_now)
            assert NEUTRAL - 300.0 <= thr <= NEUTRAL or (thr == NEUTRAL + 150.0 and v_now > 1.0)


def test_forward_is_saturated_at_fwd_max():
    c = _ctl()
    assert c.thr_ff(4.5, 0.0) == pytest.approx(NEUTRAL - 300.0)            # 前進の上限で切る


def test_steer_saturates_at_the_table_edge():
    c = _ctl()
    hard, _ = c.step(_out(wp=_arc(3.0, 1.0)), 1.0, {'tcp_mode': 3}, 0)     # 表の外(いっぱい以上の曲率)
    assert np.all(hard[:, 0] >= CENTER + 375.0 - 1e-3) and np.all(hard[:, 0] <= 2000.0)
    assert np.all(C.steer_from_curv(np.array([99.0, -99.0]), np.array(_table()[0]), OFFSETS, CENTER)
                  == np.array([2000.0, 1000.0]))                              # 1441 ± 575 → 1000〜2000 に切る
    assert np.all(C.steer_from_curv(np.array([99.0, -99.0]), np.array(_table()[0]), OFFSETS, 1500.0)
                  == np.array([2000.0, 1000.0]))
    assert C.steer_from_curv(np.array([99.0]), np.array(_table()[0]), OFFSETS, 1300.0)[0] == 1875.0   # 表の端 = いっぱい
    assert np.all(C.steer_from_curv(np.array([99.0]), np.array(_table()[0]), OFFSETS, 1700.0) == 2000.0)


# ---------------------------------------------------------------- 速度の上限
def test_cap_never_makes_throttle_more_forward():
    rng = np.random.default_rng(1)
    for mode in range(4):
        for _ in range(20):
            o = _out(v_tgt=float(rng.uniform(0, 4.5)), wp=_arc(rng.uniform(-1, 1), rng.uniform(0.5, 4)), rng=rng)
            v_now = float(rng.uniform(-0.5, 4.5))
            prev = None
            for cap in (0.0, 5.0, 4.0, 3.0, 2.75, 2.0, 1.0, 0.5, 0.2):      # 0 = なし → 下げていく
                c = _ctl()
                thr = c.step(o, v_now, {'tcp_mode': mode, 'speed_cap': cap}, 0)[0][:, 1]
                if prev is not None:
                    assert np.all(thr >= prev - 1e-3), (mode, cap)          # µs が減らない = 前進が強くならない
                prev = thr


def test_cap_in_control_mode_only_weakens_actions():
    c = _ctl()
    acts = np.full((10, 2), -0.5, np.float32)                                # スロットル 1250 µs(強い前進)
    acts[:, 0] = 0.1
    out, info = c.step(_out(actions=acts, v_tgt=4.0), 3.0, {'tcp_mode': 0, 'speed_cap': 2.0}, 0)
    assert info['cap'] == 2.0 and info['branch'] == 'control'
    assert np.all(out[:, 1] > 1250.0) and np.all(out[:, 0] == np.float32(1550.0))   # ステアはそのまま
    assert np.all(out[:, 1] == np.float32(NEUTRAL + 150.0))                 # 3.0 m/s で上限 2.0: ブレーキの規則
    weak = np.full((10, 2), 0.05, np.float32)                                 # 1525 µs(前進でない)はそのまま通す
    out2, _ = _ctl().step(_out(actions=weak), 1.5, {'tcp_mode': 0, 'speed_cap': 2.0}, 0)
    assert np.all(out2 == weak * 500.0 + 1500.0)
    out3, _ = _ctl().step(_out(actions=acts), 1.5, {'tcp_mode': 0, 'speed_cap': 2.0}, 0)
    assert np.all((out3[:, 1] > 1250.0) & (out3[:, 1] < NEUTRAL))            # 上限の手前: 前進を弱めるだけ


def test_cap_ramps_up_at_1_5_mps2_after_release_and_drops_at_once():
    c = _ctl()
    o = _out(v_tgt=4.4)
    assert c.step(o, 2.0, {'tcp_mode': 1, 'speed_cap': 2.0}, 0)[1]['cap'] == 2.0
    caps = [c.step(o, 2.0, {'tcp_mode': 1, 'speed_cap': 0.0}, (i + 1) * FRAME_NS)[1]['cap'] for i in range(100)]
    steps = np.diff([2.0] + [x for x in caps if x > 0])
    assert np.allclose(steps, 1.5 / 30, atol=1e-6)                            # 1.5 m/s² で上がる
    assert caps[-1] == 0.0                                                    # 5.0 m/s を超えたら「なし」(cap 0 と表示)
    assert 59 <= sum(1 for x in caps if x > 0) <= 61                          # (5 − 2)/1.5 = 2 秒 = 60 フレーム
    # 上限を上げたとき(3 → 4)も同じ速さ、下げるのはすぐ
    c = _ctl()
    c.step(o, 2.0, {'tcp_mode': 1, 'speed_cap': 3.0}, 0)
    assert c.step(o, 2.0, {'tcp_mode': 1, 'speed_cap': 4.0}, FRAME_NS)[1]['cap'] == pytest.approx(3.05)
    assert c.step(o, 2.0, {'tcp_mode': 1, 'speed_cap': 1.0}, 2 * FRAME_NS)[1]['cap'] == 1.0
    c.reset()
    assert c.step(o, 2.0, {'tcp_mode': 1, 'speed_cap': 0.0}, 3 * FRAME_NS)[1]['cap'] == pytest.approx(1.05)  # reset でも続き
    assert _ctl().step(o, 2.0, {'tcp_mode': 1, 'speed_cap': 0.0}, 0)[1]['cap'] == 0.0      # 新しいオブジェクトの最初はすぐ


def test_reset_keeps_cap_ramp():
    """中立で止めた(reset)後も、上限を上げている途中の値から 1.5 m/s² で続ける(一気に「上限なし」へ跳ばない)"""
    c = _ctl()
    o = _out(v_tgt=4.4)
    c.step(o, 2.0, {'tcp_mode': 0, 'speed_cap': 2.0}, 0)
    cap1 = c.step(o, 2.0, {'tcp_mode': 0, 'speed_cap': 0.0}, FRAME_NS)[1]['cap']
    assert cap1 == pytest.approx(2.05)
    c.reset()                                                                 # 例: 中立で止めた
    cap2 = c.step(o, 2.0, {'tcp_mode': 0, 'speed_cap': 0.0}, 90 * FRAME_NS)[1]['cap']
    assert cap2 == pytest.approx(2.1)                                         # 止めていた時間では上げず、1/30 秒ぶん
    c.reset()
    assert c.step(o, 2.0, {'tcp_mode': 0, 'speed_cap': 1.0}, 91 * FRAME_NS)[1]['cap'] == 1.0   # 下げるのはすぐ


def _after_gain(thr, gain):
    """drive.apply_gains のスロットル(前進側 = 中立より下だけにゲイン)"""
    dt = thr - NEUTRAL
    return NEUTRAL + (gain * dt if dt < 0 else dt)


@pytest.mark.parametrize('gain', [0.25, 0.5, 1.0, 1.5, 2.0])
def test_cap_holds_after_throttle_gain(gain):
    """上限はゲインの後で thr_ff(cap) になる(ゲインで割り戻す)。ゲインが 1 でなくても上限より前進を強くせず、
    上限より弱い操作の枝はゲインを掛けた値のまま(上限が操作を余計に弱めない)"""
    v_now, cap = 0.8, 1.0
    want = _ctl().thr_ff(cap, v_now)                                          # ゲインの後に出したい上限のスロットル
    strong = np.full((10, 2), -0.6, np.float32)                               # 1200 µs: 強い前進
    out, info = _ctl().step(_out(actions=strong), v_now, {'tcp_mode': 0, 'speed_cap': cap, 'throttle_gain': gain}, 0)
    assert np.isfinite(info['thr_cap'])
    assert _after_gain(float(out[0, 1]), gain) == pytest.approx(want, abs=1e-3)
    # 操作の枝(ゲインの後)が上限より弱いときは、そのまま
    weak_us = NEUTRAL - 0.5 * (NEUTRAL - want) / gain
    weak = np.full((10, 2), (weak_us - 1500.0) / 500.0, np.float32)
    out2, _ = _ctl().step(_out(actions=weak), v_now, {'tcp_mode': 0, 'speed_cap': cap, 'throttle_gain': gain}, 0)
    assert np.allclose(out2[:, 1], weak[:, 1] * 500.0 + 1500.0)


def test_gain_one_or_missing_is_bit_identical():
    """throttle_gain = 1.0(評価のゲート D)と、params に無いときは、前の制御器とビット単位で同じ出力"""
    rng = np.random.default_rng(5)
    for mode in range(4):
        for i in range(20):
            o = _out(actions=rng.normal(0, 0.3, (10, 2)).astype(np.float32), v_tgt=float(rng.uniform(0, 4.5)),
                     wp=_arc(float(rng.uniform(-1, 1)), 2.0))
            v = float(rng.uniform(0, 3.5))
            a, ia = _ctl().step(o, v, {'tcp_mode': mode, 'speed_cap': 1.7}, 0)
            b, ib = _ctl().step(o, v, {'tcp_mode': mode, 'speed_cap': 1.7, 'throttle_gain': 1.0}, 0)
            assert np.array_equal(a, b) and ia['thr_cap'] == ib['thr_cap']


def test_closed_loop_speed_reaches_cap_for_any_gain():
    """較正どおりに動く車(定常の前進 = 中立 − (c0 + c1·v)、1 次の遅れ 0.3 秒)で、強い前進の操作の枝 + 上限 1.0 m/s を
    回すと、ゲイン 0.5・1.0・2.0 のどれでも上限の近くで落ち着く(レビュー 2026-10-12: 前はゲイン 0.5 で約 0.3 m/s、
    2.0 で上限 + 0.3 m/s の頭打ちとブレーキの繰り返し)"""
    pp = _pp()
    ff = pp['throttle_ff']
    strong = np.full((10, 2), -0.6, np.float32)
    for gain in (0.5, 1.0, 2.0):
        c = C.TCPPost(pp, (CENTER, NEUTRAL))
        v = 0.0
        vs = []
        for i in range(600):                                                  # 20 秒
            out, _ = c.step(_out(actions=strong), v, {'tcp_mode': 0, 'speed_cap': 1.0, 'throttle_gain': gain}, i * FRAME_NS)
            thr = _after_gain(float(out[0, 1]), gain)
            fwd = NEUTRAL - thr                                               # 前進の量(負 = ブレーキ)
            v_ss = max((fwd - (NEUTRAL - ff['c0'])) / -ff['c1'], 0.0) if fwd > 0 else 0.0
            if fwd < 0:
                v_ss = max(v - 3.0 / 30, 0.0) if v > 0 else 0.0              # ブレーキ: 減速
            v += (v_ss - v) * (1.0 / 30) / 0.3
            vs.append(v)
        tail = np.array(vs[-150:])
        assert abs(tail.mean() - 1.0) < 0.1 and tail.max() < 1.3, (gain, tail.mean(), tail.max())


def test_constant_cap_is_stateless():
    o = _out(v_tgt=3.5)
    a = _ctl()
    seq = [a.step(o, 2.5, {'tcp_mode': 1, 'speed_cap': 2.75}, i * FRAME_NS)[0] for i in range(5)]
    for i in range(5):
        b = _ctl()
        assert np.array_equal(b.step(o, 2.5, {'tcp_mode': 1, 'speed_cap': 2.75}, i * FRAME_NS)[0], seq[i])


def test_speed_scale_multiplies_target():
    c = _ctl()
    _, info = c.step(_out(v_tgt=2.0), 2.0, {'tcp_mode': 1, 'speed_scale': 1.5}, 0)
    assert info['v_cmd'] == pytest.approx(1.5 * info['v_hat'])


# ---------------------------------------------------------------- 出し方
def test_mode0_without_cap_is_bit_identical_to_actions():
    rng = np.random.default_rng(3)
    stage = post.Stage({'postprocess': _pp()}, (CENTER, NEUTRAL))
    assert isinstance(stage.post, C.TCPPost)
    for i in range(50):
        res = {'actions': rng.normal(0, 0.3, (1, 10, 2)).astype(np.float32),
               'waypoints': rng.normal(0, 1, (1, 10, 2)).astype(np.float32),
               'speed_logits': rng.normal(0, 2, (1, 10)).astype(np.float32)}
        mp = stage.model_pred(res)
        for params in ({'tcp_mode': 0, 'speed_cap': 0.0}, {}, {'tcp_mode': 0, 'speed_cap': 0.0, 'speed_scale': 1.3}):
            pred, info = stage.apply(res, mp, float(rng.uniform(-1, 4)), dict(params, ensemble_j=0), i * FRAME_NS)
            assert pred.dtype == np.float32 and np.array_equal(pred, mp) and info['branch'] == 'control'


def test_hybrid_keeps_action_steering_rows():
    acts = np.random.default_rng(4).normal(0, 0.2, (10, 2)).astype(np.float32)
    out, info = _ctl().step(_out(actions=acts, wp=_arc(1.0, 2.0)), 2.0, {'tcp_mode': 1}, 0)
    assert info['branch'] == 'hybrid' and np.array_equal(out[:, 0], acts[:, 0] * 500.0 + 1500.0)


def test_fusion_uses_past_second_of_own_steering():
    c = _ctl()
    acts = np.zeros((10, 2), np.float32)
    acts[:, 0] = 0.6                                                         # 操作の枝: 1800 µs(左に曲がっている)
    o = _out(actions=acts, wp=_arc(0.0, 2.0))                                # 通過点: まっすぐ(中心)
    out0, info0 = c.step(o, 2.0, {'tcp_mode': 2}, 0)
    assert info0['branch'] == 'fusion:traj'                                  # 履歴が空: 通過点が主(0.3·操作 + 0.7·通過点)
    assert out0[0, 0] == pytest.approx(0.3 * 1800.0 + 0.7 * CENTER, abs=1.0)   # = 中心 + 108 µs(> 100)
    for i in range(1, 31):
        out, info = c.step(o, 2.0, {'tcp_mode': 2}, i * FRAME_NS)
    assert info['branch'] == 'fusion:ctrl'                                   # 過去 1 秒の出力の半分以上が |ずれ| > 100
    assert out[0, 0] == pytest.approx(0.7 * 1800.0 + 0.3 * CENTER, abs=1.0)
    small = dict(o, actions=np.where(np.arange(2) == 0, 0.3, 0.0).astype(np.float32) * np.ones((10, 1), np.float32))
    c2 = _ctl()                                                              # 0.3·1650 + 0.7·1441 = 中心 + 63 µs: ずっと traj
    assert all(c2.step(small, 2.0, {'tcp_mode': 2}, i * FRAME_NS)[1]['branch'] == 'fusion:traj' for i in range(40))
    assert out[0, 1] == pytest.approx(0.5 * 1500.0 + 0.5 * c.thr_ff(info['v_cmd'], 2.0), abs=1e-3)
    c.reset()
    assert c.step(o, 2.0, {'tcp_mode': 2}, 40 * FRAME_NS)[1]['branch'] == 'fusion:traj'


def test_short_path_uses_action_steering():
    acts = np.zeros((10, 2), np.float32)
    acts[:, 0] = -0.2
    out, info = _ctl().step(_out(actions=acts, wp=_arc(1.0, 0.1)), 0.1, {'tcp_mode': 3}, 0)   # 全長 0.1 m < 0.2 m
    assert np.all(out[:, 0] == np.float32(1400.0)) and info['branch'] == 'fallback'
    assert np.all(out[:, 1] == np.float32(_ctl().thr_ff(info['v_cmd'], 0.1)))   # スロットルは制御器のまま
    assert math.isnan(info['v_curv'])


def test_missing_inputs_fall_back_to_actions_without_exceptions():
    acts = np.full((10, 2), -0.1, np.float32)
    c = _ctl()
    out, info = c.step({'actions': acts}, 2.0, {'tcp_mode': 1}, 0)          # 通過点・速度の出力が無い
    assert info['branch'] == 'fallback' and np.array_equal(out, acts * 500.0 + 1500.0)
    bad = {'actions': acts, 'waypoints': np.full((10, 2), np.nan, np.float32), 'speed_logits': np.full(10, np.nan)}
    out, info = c.step(bad, float('nan'), {'tcp_mode': 3, 'speed_cap': 1.0}, None)
    assert info['branch'] == 'fallback' and np.all(np.isfinite(out)) and np.all(out[:, 1] >= 1450.0)   # 上限は掛かる
    out, info = c.step({}, 2.0, {}, 0)                                       # actions も無い → 中立
    assert info['branch'] == 'fallback' and np.all(out == np.array([CENTER, NEUTRAL], np.float32))
    out, _ = c.step(_out(), 2.0, {'tcp_mode': 'x', 'speed_cap': None, 'speed_scale': float('inf')}, 'bad')
    assert np.all(np.isfinite(out))


def test_random_inputs_give_finite_outputs():
    rng = np.random.default_rng(5)
    c = _ctl()
    for i in range(200):
        o = {'actions': rng.normal(0, 1, (10, 2)).astype(np.float32),
             'waypoints': (rng.normal(0, 2, (10, 2)) * rng.choice([0, 1e-4, 1, 10])).astype(np.float32),
             'speed_logits': (rng.normal(0, 30, 10)).astype(np.float32)}
        params = {'tcp_mode': int(rng.integers(0, 4)), 'speed_cap': float(rng.choice([0, rng.uniform(0, 5)])),
                  'speed_scale': float(rng.uniform(0.5, 1.5)), 'lookahead': int(rng.integers(1, 11))}
        out, info = c.step(o, float(rng.uniform(-2, 6)), params, int(i * FRAME_NS * rng.choice([1, -1, 3])))
        assert out.shape == (10, 2) and out.dtype == np.float32 and np.all(np.isfinite(out))
        assert np.all((out[:, 0] >= 1000.0) | (o['actions'][:, 0] * 500 + 1500 < 1000.0))
        assert set(info) >= {'mode', 'branch', 'v_hat', 'v_cmd', 'v_curv', 'cap'}
    assert len(c.hist) <= C.HIST_MAX


def test_v_curv_is_display_only_and_follows_full_lock():
    curv = [np.maximum.accumulate(np.clip(0.004 * OFFSETS, -s, s)) + np.arange(24) * 1e-6 for s in (1.6, 1.2, 0.9)]
    c = _ctl(steer_table={'lag_ms': 40, 'offset_us': OFFSETS.tolist(), 'bands_mps': [2.0, 3.0],
                          'curv': [x.tolist() for x in curv]})
    assert c.v_curv_of(np.array([0.5])) == 4.5                               # 0.9 ≥ 1.1·0.5 → 最も速い帯
    assert c.v_curv_of(np.array([1.0])) == 3.0                               # 1.2 ≥ 1.1 → 2〜3 m/s の帯の上限
    assert c.v_curv_of(np.array([-1.4])) == 2.0
    assert c.v_curv_of(np.array([2.0])) == 0.0 and math.isnan(c.v_curv_of(None))


def test_bad_postprocess_is_rejected_and_post_make_falls_back():
    with pytest.raises(ValueError):
        C.TCPPost(_pp(steer_table={'offset_us': [0.0], 'curv': [[0.0]]}), (CENTER, NEUTRAL))
    with pytest.raises(ValueError):
        C.TCPPost(_pp(throttle_ff={}), (CENTER, NEUTRAL))
    with pytest.raises(ValueError):
        C.TCPPost(_pp(version=2), (CENTER, NEUTRAL))
    obj, note = post.make({'postprocess': _pp(throttle_ff={})}, (CENTER, NEUTRAL))
    assert obj is None and '操作の枝' in note
    obj, note = post.make({'postprocess': _pp()}, (CENTER, NEUTRAL))
    assert isinstance(obj, C.TCPPost) and obj.kind == 'tcp_controller'


def test_replay_with_constant_params_matches_framewise_steps():
    rng = np.random.default_rng(6)
    n = 12
    outs = {'actions': rng.normal(0, 0.2, (n, 10, 2)).astype(np.float32),
            'waypoints': np.stack([_arc(rng.uniform(-1, 1), 2.5) for _ in range(n)]),
            'speed_logits': rng.normal(0, 1, (n, 10)).astype(np.float32)}
    erpm = rng.uniform(-8000, 0, n)
    t_ns = np.arange(n) * FRAME_NS
    params = {'tcp_mode': 1, 'speed_cap': 2.75, 'speed_scale': 1.0, 'ensemble_j': 0}
    preds, mps, infos = post.replay({'postprocess': _pp()}, (CENTER, NEUTRAL), outs, erpm, t_ns, params)
    for i in range(n):
        c = _ctl()
        p, _ = c.step({k: v[i] for k, v in outs.items()}, -erpm[i] * 0.5e-3, params, int(t_ns[i]))
        assert np.array_equal(p, preds[i])
    assert all(i['branch'] == 'hybrid' for i in infos)


def test_step_time_is_small():
    c = _ctl()
    o = _out(wp=_arc(0.5, 2.5))
    for mode in range(4):
        ts = []
        for i in range(200):
            t = time.perf_counter()
            c.step(o, 2.5, {'tcp_mode': mode, 'speed_cap': 2.5}, i * FRAME_NS)
            ts.append(time.perf_counter() - t)
        print('tcp_mode %d: 1 回 %.3f ms(中央値)' % (mode, 1e3 * float(np.median(ts))))
        assert np.median(ts) < 5e-3                                          # 目安 0.1〜0.4 ms(ほかの処理と CPU を分け合っても)
