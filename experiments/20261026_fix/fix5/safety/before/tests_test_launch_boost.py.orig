"""発進の前進(assist.LaunchBoost・Assist.step の launch)の試験。2026-10-26
仕様: jetformer_reports/fix_20261026/protocol_v3.md 14 章(T5〜T9 の純粋な部分。走行ループは tests/test_launch_drive.py)。
車・GPU・torch は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_launch_boost.py
"""
import math
import random

import numpy as np
import pytest

from jetformer import assist as A

C, N = 1441.0, 1496.0
DT = 1 / 30.0
FREE = (30.0,) * 5                     # 止まっていて前に何も無い(x_free 9 m ÷ 0.3)


def inp(**kw):
    """よい発進の入力(止まっている・前が空き・ToF・車両の値あり)。kw で替える"""
    base = dict(us=250.0, mps=2.5, max_s=1.2, v=0.0, a=0.0, gz=0.0, steer_rel=0.0, ttc=30.0, ttc5=FREE, tof_ok=True,
                veh_ok=True, impact_g=0.0, guard_changed=False)
    base.update(kw)
    return A.LaunchIn(**base)


def armed(a=None):
    """人側のフレームを 1 回見せた Assist(「AI 側にした」と分かる)"""
    a = a or A.Assist((C, N))
    a.step((C, N), 0.0, False, -0.1)
    return a


def ai(a, model=(C, N), erpm=0.0, t=0.0, launch=None, **kw):
    return a.step(model, erpm, True, t, launch=launch, **kw)


# ---------------------------------------------------------------- T5 始める・始めない
def test_t5_starts_when_stopped_free_and_tof():
    a = armed()
    out, k = ai(a, launch=inp())
    assert k == A.ASSIST_LAUNCH and out == (C, N - 250.0)
    assert a.launch_on and a.launch_end == 0 and a.boost.count == 1 and a.boost.active
    assert a.boost.event[0] == 'start' and a.boost.event[1]['ttc_med'] == 30.0


@pytest.mark.parametrize('why,kw,erpm', [
    ('near', dict(ttc5=(3.2,) * 5), 0.0),                                 # TTC の中央値 < 3.3(前 1 m 未満)
    ('near', dict(ttc5=(30.0, 30.0, 1.0, 1.0, 1.0)), 0.0),                # 中央値で見る
    ('ttc_n', dict(ttc5=(30.0,) * 4), 0.0),                               # 5 フレームそろわない(中立のあとなど)
    ('ttc_n', dict(ttc5=(30.0, 30.0, float('nan'), 30.0, 30.0)), 0.0),
    ('tof', dict(tof_ok=False), 0.0),                                     # ToF が無い・古い・ガード 0
    ('veh', dict(veh_ok=False), 0.0),                                     # 車両の値が古い
    ('moving', {}, -400.0),                                               # 動いている(|eRPM| ≥ 400)
    ('moving', {}, 400.0),
])
def test_t5_does_not_start(why, kw, erpm):
    a = armed()
    out, k = ai(a, erpm=erpm, launch=inp(**kw))
    assert k != A.ASSIST_LAUNCH and not a.launch_on and a.boost.count == 0
    assert a.boost.last_skip == why and a.boost.event == ('skip', {'why': why, 'reason': A.LAUNCH_SKIP_NAMES[why]})
    # 最初のフレームでそろわなければ、その区間ではもう出さない(あとで前が空いても)
    out, k = ai(a, t=DT, launch=inp())
    assert k != A.ASSIST_LAUNCH and a.boost.count == 0


def test_t5_ttc_just_at_threshold_and_last_five_frames():
    a = armed()
    assert ai(a, launch=inp(ttc5=(3.3,) * 5))[1] == A.ASSIST_LAUNCH            # ちょうど 3.3 秒は始める
    b = armed()
    assert ai(b, launch=inp(ttc5=(0.5,) * 3 + (30.0,) * 5))[1] == A.ASSIST_LAUNCH   # 直近 5 フレームだけ見る


def test_t5_not_without_switch_dry_run_human_hold():
    a = A.Assist((C, N))
    assert ai(a, launch=inp())[1] != A.ASSIST_LAUNCH                 # 走行の始めから AI 側(人側を見ていない)
    assert a.boost.last_skip == 'no_switch'
    b = A.Assist((C, N))
    for i in range(5):                                               # dry run = select False(人側と同じ)
        out, k = b.step((C, N), 0.0, False, i * DT, launch=inp())
        assert k == A.ASSIST_NONE and out == (C, N)
    c = armed()
    c.launch_hold()                                                  # AI 側にした最初のフレームが中立で止めたフレーム
    assert ai(c, launch=inp())[1] != A.ASSIST_LAUNCH and c.boost.last_skip == 'hold'
    d = armed()
    assert ai(d, model=(C, float('nan')), launch=inp())[1] != A.ASSIST_LAUNCH and d.boost.last_skip == 'model'


def test_t5_end_condition_on_first_frame_does_not_start():
    a = armed()
    out, k = ai(a, launch=inp(impact_g=3.0))
    assert k != A.ASSIST_LAUNCH and a.boost.count == 0 and a.boost.last_skip == 'end'
    assert a.boost.event[1]['code'] == 5


def test_t5_off_is_untouched_and_kick_still_works():
    """OFF(launch=None)は今までと同じ: 発進の補助も今まで通り出る"""
    a = armed()
    out, k = ai(a, launch=None, kick_on=True)
    assert k == A.ASSIST_START and out == (C, N - A.StartKick.KICK_US) and a.boost.count == 0
    assert a.boost.event is None and not a.launch_on and a.launch_end == 0
    b = armed()                                                     # ON でも始めなかったときは今まで通り StartKick
    out, k = ai(b, launch=inp(ttc5=(2.0,) * 5), kick_on=True)
    assert k == A.ASSIST_START


# ---------------------------------------------------------------- T6 1 回だけ・CH3 で次の 1 回
def test_t6_once_per_ai_section_and_rearm_with_ch3():
    a = armed()
    t = 0.0
    assert ai(a, t=t, launch=inp())[1] == A.ASSIST_LAUNCH
    t += DT
    out, k = ai(a, t=t, launch=inp(v=2.6))                           # 速さに届いた → 終わり
    assert k != A.ASSIST_LAUNCH and a.launch_end == 1 and a.boost.last_end == 1
    for _ in range(30):                                              # 止まり直しても同じ区間ではもう出さない
        t += DT
        assert ai(a, t=t, launch=inp(), kick_on=True)[1] not in (A.ASSIST_LAUNCH, A.ASSIST_START)
    a.step((C, N), 0.0, False, t + DT)                               # CH3 を人側へ
    assert ai(a, t=t + 2 * DT, launch=inp())[1] == A.ASSIST_LAUNCH  # 次の AI 側で 1 回
    assert a.boost.count == 2


def test_t6_ch3_human_ends_and_counts_end_reason_7():
    a = armed()
    ai(a, launch=inp())
    out, k = a.step((C, N), 0.0, False, DT, launch=inp())
    assert k == A.ASSIST_NONE and out == (C, N) and a.launch_end == 7 and not a.boost.active


# ---------------------------------------------------------------- T7 出すもの
def test_t7_output_min_of_model_and_launch():
    a = armed()
    assert ai(a, model=(1600.0, N + 300.0), launch=inp())[0] == (1600.0, N - 250.0)   # モデルがブレーキでも前進・舵はそのまま
    out, k = ai(a, model=(1300.0, N - 320.0), t=DT, launch=inp())                     # モデルが強ければモデル
    assert k == A.ASSIST_LAUNCH and out == (1300.0, N - 320.0)
    z = armed()
    out, k = ai(z, model=(C, N - 100.0), launch=inp(us=999.0))                           # 強さは 300 µs まで(念のため)
    assert out == (C, N - 300.0)
    b = armed()
    assert ai(b, launch=inp(us=10.0))[0] == (C, N - 150.0)                              # 150 µs より弱くしない
    c = armed()
    out, _ = ai(c, launch=inp(us=200.0))
    assert out == (C, N - 200.0)
    out, _ = ai(c, t=DT, launch=inp(us=300.0))                                           # 値は始めたときのまま
    assert out == (C, N - 200.0)


def test_t7_steer_is_never_changed():
    a = armed()
    for i, s in enumerate((1000.0, 1441.0, 1999.0, 1234.5)):
        out, k = ai(a, model=(s, N), t=i * DT, launch=inp())
        assert k == A.ASSIST_LAUNCH and out[0] == s


# ---------------------------------------------------------------- T8 終わりの 9 条件(+ OFF にした)
END_CASES = [
    (1, dict(v=2.5)),                                   # ① 速さ(0.1 秒先の見込み)
    (1, dict(v=2.25, a=3.0)),                           # ① v + 3 × 0.1 = 2.55
    (2, None),                                          # ② 最大の時間(下で時間を進める)
    (3, dict(gz=60.0)),                                 # ③ ヨー
    (3, dict(gz=-61.0)),
    (4, dict(v=1.0, steer_rel=200.0)),                  # ④ 1 m/s 以上で舵 200 µs
    (4, dict(v=1.2, steer_rel=-250.0)),
    (5, dict(impact_g=2.5)),                            # ⑤ 衝撃
    (6, dict(guard_changed=True)),                      # ⑥ ToF ガードが出力を変えた
    (6, dict(tof_ok=False)),                            # ⑥ ToF が無くなった(仕様に無い足し: 安全側)
    (9, dict(veh_ok=False)),                            # ⑨ 車両の値が古い
]


@pytest.mark.parametrize('code,kw', END_CASES)
def test_t8_each_end_condition(code, kw):
    a = armed()
    assert ai(a, launch=inp())[1] == A.ASSIST_LAUNCH
    if code == 2:
        assert ai(a, t=1.19, launch=inp())[1] == A.ASSIST_LAUNCH
        out, k = ai(a, t=1.2, launch=inp())
    else:
        out, k = ai(a, t=DT, launch=inp(**kw))
    assert k != A.ASSIST_LAUNCH and out == (C, N) and a.launch_end == code
    assert a.boost.event[0] == 'end' and a.boost.event[1]['code'] == code
    for i in range(10):                                  # 終わったら同じ区間では前進も StartKick も出さない
        assert ai(a, t=2.0 + i * DT, launch=inp(), kick_on=True)[1] not in (A.ASSIST_LAUNCH, A.ASSIST_START)


def test_t8_not_end_below_thresholds():
    a = armed()
    ai(a, launch=inp())
    for i, kw in enumerate([dict(v=2.39, a=1.0), dict(gz=59.9), dict(v=0.99, steer_rel=400.0), dict(v=1.5, steer_rel=199.0),
                            dict(impact_g=2.49), dict(v=0.29, ttc=0.1), dict(v=1.0, ttc=0.79)]):
        assert ai(a, t=(i + 1) * DT, launch=inp(**kw))[1] == A.ASSIST_LAUNCH, kw     # TTC < 0.8 は 1 フレームだけ


def test_t8_ttc_two_frames():
    a = armed()
    ai(a, launch=inp())
    assert ai(a, t=DT, launch=inp(v=1.0, ttc=0.7))[1] == A.ASSIST_LAUNCH
    out, k = ai(a, t=2 * DT, launch=inp(v=1.0, ttc=0.7))
    assert k != A.ASSIST_LAUNCH and a.launch_end == 6
    b = armed()                                           # 続かなければ数え直し
    ai(b, launch=inp())
    for i, ttc in enumerate((0.7, 2.0, 0.7, 2.0, 0.7)):
        assert ai(b, t=(i + 1) * DT, launch=inp(v=1.0, ttc=ttc))[1] == A.ASSIST_LAUNCH


def test_t8_backward_ch3_hold_off():
    a = armed()
    ai(a, launch=inp())
    out, k = ai(a, erpm=201.0, t=DT, launch=inp())         # ⑨ 後ろへ動いた
    assert k != A.ASSIST_LAUNCH and a.launch_end == 9
    b = armed()
    ai(b, launch=inp())
    assert b.launch_hold() == 8 and not b.boost.active      # ⑧ 中立で止めた
    b.reset()
    assert ai(b, t=DT, launch=inp())[1] != A.ASSIST_LAUNCH
    c = armed()
    ai(c, launch=inp())
    assert c.boost.stop() == 8                               # ⑧ 走行の停止
    d = armed()
    ai(d, launch=inp())
    out, k = ai(d, t=DT, launch=None)                        # 走行中に 1 → 0(仕様に無い足し: 理由 10)
    assert k != A.ASSIST_LAUNCH and d.launch_end == 10
    e = armed()
    ai(e, launch=inp())
    out, k = e.step((C, N), 0.0, False, DT, launch=inp())   # ⑦ CH3 を人側
    assert e.launch_end == 7


def test_t8_recovery_and_esc_wait_until_end():
    """発進の前進の間は、止まっている間のモデルのブレーキ → 中立(ASSIST_ESC)を出さない。終わったら今まで通り"""
    a = armed()
    out, k = ai(a, model=(C, N + 300.0), launch=inp(), rec_on=True)
    assert k == A.ASSIST_LAUNCH and out[1] == N - 250.0
    out, k = ai(a, model=(C, N + 300.0), t=DT, launch=inp(v=2.6), rec_on=True)
    assert k == A.ASSIST_ESC and out == (C, N)


def test_t8_recovery_armed_after_launch_crash():
    """出している間に 1000 eRPM を超えて走り、衝撃で終わって止まった → リカバリーが動く(出している間も動いたことを残す)"""
    a = armed()
    t = 0.0
    ai(a, launch=inp(), rec_on=True)
    for e in (-600.0, -1500.0, -2500.0):
        t += DT
        assert ai(a, erpm=e, t=t, launch=inp(v=-e * 0.5e-3), rec_on=True)[1] == A.ASSIST_LAUNCH
    t += DT
    out, k = ai(a, erpm=-2000.0, t=t, launch=inp(v=1.0, impact_g=3.0), rec_on=True, impact_g=3.0)
    assert a.launch_end == 5
    kinds = set()
    for i in range(40):                                   # ぶつかって止まった
        t += DT
        kinds.add(ai(a, erpm=0.0, t=t, launch=inp(), rec_on=True)[1])
    assert A.ASSIST_RECOVERY in kinds and a.rec.count == 1


# ---------------------------------------------------------------- T9 前進だけ(でたらめな入力 1 万回)
def test_t9_never_behind_neutral_random():
    rng = random.Random(20261026)
    a = A.Assist((C, N))
    t, n_launch, n_end = 0.0, 0, 0
    sel = False
    for i in range(10000):
        t += rng.choice((DT, DT, DT, 0.0, 0.2))
        if rng.random() < 0.05:
            sel = not sel
        thr = rng.choice((rng.uniform(1000.0, 2000.0), rng.uniform(N - 400, N + 400), float('nan'), N, float('inf'),
                          float('-inf')))
        steer = rng.uniform(1000.0, 2000.0)
        erpm = rng.choice((0.0, rng.uniform(-8000.0, 1000.0), rng.uniform(-399.0, 399.0)))
        wild = rng.random() < 0.1                        # 1 割はでたらめ・9 割は続けられる値の近く
        li = inp(us=rng.uniform(0.0, 600.0), mps=rng.uniform(1.0, 4.0), max_s=rng.uniform(0.0, 3.0),
                 v=rng.choice((0.0, rng.uniform(-1.0, 5.0), float('nan'))) if wild else rng.uniform(0.0, 2.0),
                 a=rng.choice((0.0, rng.uniform(-20, 20), float('nan'))),
                 gz=rng.uniform(-100, 100) if wild else rng.uniform(-30, 30),
                 steer_rel=rng.uniform(-500, 500) if wild else rng.uniform(-100, 100), ttc=rng.uniform(0.0, 30.0),
                 ttc5=tuple(rng.uniform(0.0, 30.0) if rng.random() < 0.9 else 30.0 for _ in range(5)),
                 tof_ok=not wild or rng.random() < 0.5, veh_ok=not wild or rng.random() < 0.5,
                 impact_g=rng.uniform(0, 4) if wild else 0.0,
                 guard_changed=wild and rng.random() < 0.2) if rng.random() < 0.99 else None
        if rng.random() < 0.01:
            a.launch_hold()
            a.reset()
            continue
        out, k = a.step((steer, thr), erpm, sel, t, launch=li, kick_on=rng.random() < 0.5, rec_on=rng.random() < 0.5,
                        esc_on=rng.random() < 0.5)
        if k == A.ASSIST_LAUNCH:
            n_launch += 1
            assert out[1] < N - 149.9 and out[0] == steer       # 中立 − 150 より前だけ・舵はそのまま
            assert math.isfinite(out[1])
        if a.launch_end:
            n_end += 1
        assert not (a.launch_on and k != A.ASSIST_LAUNCH)
    assert n_launch > 100 and n_end > 30, (n_launch, n_end)


def test_t9_ten_thousand_random_model_outputs_during_one_launch():
    """出している間のモデルの出力 1 万回(ブレーキ・後退・nan・inf を含む): いつも 中立 − U 以下・モデルが強ければモデル"""
    rng = random.Random(9)
    a = armed()
    assert ai(a, launch=inp(us=150.0))[1] == A.ASSIST_LAUNCH
    want = N - 150.0
    for i in range(10000):
        thr = rng.choice((rng.uniform(1000.0, 2000.0), rng.uniform(want - 50, N + 600), float('nan'), float('inf'),
                          float('-inf'), N, want))
        steer = rng.uniform(1000.0, 2000.0)
        out, k = ai(a, model=(steer, thr), erpm=rng.uniform(-3000.0, 200.0), t=1e-4 * (i + 1),
                    launch=inp(v=rng.uniform(0.0, 2.0), gz=rng.uniform(-59, 59), steer_rel=rng.uniform(-199, 199),
                               ttc=rng.uniform(0.8, 30.0)))
        assert k == A.ASSIST_LAUNCH and out[0] == steer
        assert math.isfinite(out[1]) and out[1] <= want < N
        assert out[1] == (thr if math.isfinite(thr) and thr < want else want)
    assert a.boost.frames == 10001


def test_names_and_codes():
    assert A.ASSIST_LAUNCH == 5 and A.ASSIST_NAMES[5] == '発進の前進' and A.ASSIST_START == 4
    assert sorted(A.LAUNCH_END_NAMES) == list(range(11))
    assert A.LaunchBoost.PRED_S == 0.1 and A.LaunchBoost.ACCEL_MAX == 3.0
    from jetformer import safety_caps
    cp = safety_caps.CeilParams()
    assert (cp.pred_s, cp.accel_max) == (A.LaunchBoost.PRED_S, A.LaunchBoost.ACCEL_MAX)   # 速さの天井と同じ見込み
    assert np.isnan(A.LaunchIn().v)


# ---------------------------------------------------------------- 念のための切り詰め(2026-10-26 fix2: 数でない値は既定値)
BAD = [float('nan'), float('inf'), float('-inf'), None, 'x']


def test_clip_defaults_match_drive_and_launchin():
    from jetformer import drive
    B = A.LaunchBoost
    assert (B.US_DEFAULT, B.MPS_DEFAULT, B.MAX_S_DEFAULT) == tuple(
        drive.DEFAULT_PARAMS[k] for k in ('launch_us', 'launch_mps', 'launch_max_s'))
    assert (B.US_DEFAULT, B.MPS_DEFAULT, B.MAX_S_DEFAULT) == (A.LaunchIn().us, A.LaunchIn().mps, A.LaunchIn().max_s)
    for k, rng in (('launch_us', B.US_RANGE), ('launch_mps', B.MPS_RANGE), ('launch_max_s', B.MAX_S_RANGE)):
        assert tuple(drive.PARAM_LIMITS[k]) == rng


def test_clip_finite_values_same_as_before():
    """数なら今まで(min(max(float(x), lo), hi))とビット単位で同じ"""
    rng = random.Random(26)
    B = A.LaunchBoost
    xs = [150.0, 300.0, 149.999, 300.001, 0.0, -0.0, -1e300, 1e300, 250, True, '275.5', np.float32(222.2), np.float64(2.0)]
    xs += [rng.uniform(-1000.0, 1000.0) for _ in range(2000)]
    for r, d in ((B.US_RANGE, B.US_DEFAULT), (B.MPS_RANGE, B.MPS_DEFAULT), (B.MAX_S_RANGE, B.MAX_S_DEFAULT)):
        for x in xs:
            old = min(max(float(x), r[0]), r[1])
            new = B._clip(x, r, d)
            assert type(new) is float and new == old and math.copysign(1.0, new) == math.copysign(1.0, old), (x, r)


@pytest.mark.parametrize('bad', BAD, ids=['nan', 'inf', '-inf', 'none', 'str'])
@pytest.mark.parametrize('field', ['us', 'mps', 'max_s'])
def test_clip_non_number_uses_default(field, bad):
    """us・mps・max_s が数でない(NaN・±inf・None・文字)なら既定値で出す(前は NaN が出力に出た・② で終わらなかった)"""
    B = A.LaunchBoost
    default = {'us': B.US_DEFAULT, 'mps': B.MPS_DEFAULT, 'max_s': B.MAX_S_DEFAULT}[field]
    a = armed()
    out, k = ai(a, model=(C, N + 100.0), launch=inp(**{field: bad}))
    assert k == A.ASSIST_LAUNCH and a.boost.count == 1
    want_cfg = {'us': 250.0, 'mps': 2.5, 'max_s': 1.2}
    want_cfg[field] = default
    assert a.boost.cfg == (want_cfg['us'], want_cfg['mps'], want_cfg['max_s'])
    assert all(type(x) is float and math.isfinite(x) for x in a.boost.cfg)
    assert out == (C, N - want_cfg['us']) and math.isfinite(out[1])
    ev = a.boost.event
    assert ev[0] == 'start' and (ev[1]['us'], ev[1]['mps'], ev[1]['max_s']) == a.boost.cfg
    # 止まったまま出し続けると ② 最大の時間(既定 1.2 秒)で終わる(前は max_s が NaN だと終わらなかった)
    n, t = 0, 0.0
    for i in range(1, 300):
        t = i * DT
        out, k = ai(a, model=(C, N + 100.0), t=t, launch=inp(**{field: bad}))
        if k != A.ASSIST_LAUNCH:
            break
        assert math.isfinite(out[1]) and out[1] == N - want_cfg['us']
        n += 1
    assert a.launch_end == 2 and abs(t - want_cfg['max_s']) <= DT + 1e-9, (n, t)


def test_clip_non_number_speed_end_uses_default_mps():
    """mps が NaN でも ① 速さに届いた(既定 2.5 m/s)で終わる"""
    a = armed()
    assert ai(a, launch=inp(mps=float('nan')))[1] == A.ASSIST_LAUNCH
    assert ai(a, t=DT, launch=inp(mps=float('nan'), v=2.3, a=1.0))[1] == A.ASSIST_LAUNCH     # 2.4 < 2.5
    out, k = ai(a, t=2 * DT, launch=inp(mps=float('nan'), v=2.45, a=1.0))                    # 2.55 ≥ 2.5
    assert k != A.ASSIST_LAUNCH and a.launch_end == 1
