"""AI走行の補助(後退の受け付け・自動リカバリー)"""

from jetformer import assist as A

N, C = 1498.0, 1443.0


def run(fn, t0, t1, dt=1 / 30.0):
    out, t = [], t0
    while t < t1 - 1e-9:
        out.append((round(t, 3), fn(t)))
        t += dt
    return out


def test_esc_reverse_inserts_neutral_after_brake():
    e = A.EscReverse(N)
    # 前進中に後退側 → ブレーキとしてそのまま
    assert e.step(1900, -3000, 0.0) == (1900, False)
    assert e.step(1900, -2000, 0.05) == (1900, False)
    # 止まった: ブレーキを BRAKE_S 出した後、中立を GAP_S 挟んでから後退(強さは上限まで)
    seq = run(lambda t: e.step(1900, 0, t), 0.1, 0.8)
    thr = [v[0] for _, v in seq]
    assert N in thr                                       # 中立を挟んだ
    first_neutral = thr.index(N)
    assert all(x == N for x in thr[first_neutral:first_neutral + 5])
    assert thr[-1] == N + A.EscReverse.REV_CAP_US          # 後退(上限で抑える)
    # 前進を求めたら、すぐそのまま
    assert e.step(1300, 0, 0.9) == (1300, False)
    assert e.state == 'idle'


def test_esc_reverse_limits_duration():
    e = A.EscReverse(N)
    e.step(1700, 500, 0.0)                                 # すでに下がっている
    assert e.state == 'reverse'
    assert e.step(1700, 1500, 0.5) == (1700, False)
    assert e.step(1700, 1500, A.EscReverse.REV_MAX_S + 0.1) == (N, True)   # 長すぎる → 中立
    assert e.step(1700, 0, 2.0) == (N, True)               # モデルが後退をやめるまで中立
    assert e.step(1450, 0, 2.1) == (1450, False)


def test_recovery_not_triggered_before_moving():
    r = A.Recovery((C, N))
    res = run(lambda t: r.step((1500, 1400), 0, t), 0.0, 3.0)  # 発進前: 止まったまま前進を指示
    assert all(o is None for _, (o, k) in res)


def test_recovery_backs_up_opposite_to_previous_steering():
    r = A.Recovery((C, N))
    run(lambda t: r.step((1000, 1350), -5000, t), 0.0, 1.0)      # 右いっぱいで走っていた
    res = run(lambda t: r.step((1000, 1450), 0, t), 1.0, 4.0)   # ぶつかって止まった
    acts = [(t, o) for t, (o, k) in res if k == A.ASSIST_RECOVERY]
    assert acts, 'リカバリーが始まらない'
    t_start = acts[0][0]
    assert 1.0 + A.Recovery.STUCK_S - 0.05 <= t_start <= 1.0 + A.Recovery.STUCK_S + 0.1
    thr = [o[1] for _, o in acts]
    assert N in thr and max(thr) == N + A.Recovery.REV_US     # ブレーキ → 中立 → 後退
    steer = [o[0] for _, o in acts if o[1] > N]
    assert all(s > C for s in steer)                           # 右に切っていた → 後退中は左
    assert r.count == 1
    assert res[-1][1] == (None, A.ASSIST_NONE)                 # 終わったらモデルへ戻す


def test_recovery_faster_after_impact():
    r = A.Recovery((C, N))
    run(lambda t: r.step((1443, 1350), -5000, t), 0.0, 1.0)
    r.step((1443, 1450), 0, 1.0, impact_g=6.0)
    res = run(lambda t: r.step((1443, 1450), 0, t), 1.0 + 1 / 30.0, 2.0)
    first = next(t for t, (o, k) in res if k == A.ASSIST_RECOVERY)
    assert first <= 1.0 + A.Recovery.STUCK_AFTER_IMPACT_S + 0.1


def test_recovery_gives_up_and_resets_on_handover():
    a = A.Assist((C, N))
    t = 0.0
    kinds = []
    for _ in range(4):                                         # 走る → 止まる を繰り返す
        for _ in range(30):
            a.step((1443, 1350), -4000, True, t)
            t += 1 / 30.0
        for _ in range(75):
            kinds.append(a.step((1443, 1450), 0, True, t)[1])
            t += 1 / 30.0
    assert A.ASSIST_GIVEUP in kinds
    assert a.step((1443, 1350), -4000, True, t) == ((C, N), A.ASSIST_GIVEUP)    # あきらめたら中立のまま
    assert a.step((1443, 1350), 0, False, t + 0.1) == ((1443, 1350), A.ASSIST_NONE)  # 人が交代 → 解除
    assert a.step((1443, 1350), -4000, True, t + 0.2) == ((1443, 1350), A.ASSIST_NONE)


def test_assist_passthrough_when_human_or_disabled():
    a = A.Assist((C, N))
    assert a.step((1600, 1900), 0, False, 0.0) == ((1600, 1900), A.ASSIST_NONE)
    assert a.step((1600, 1900), 0, True, 0.1, esc_on=False, rec_on=False) == ((1600, 1900), A.ASSIST_NONE)
