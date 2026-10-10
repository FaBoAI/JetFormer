"""AI走行の補助(自動リカバリー・後退の受け付け)"""

import pytest

from jetformer import assist as A

N, C = 1498.0, 1443.0
DT = 1 / 30.0


def sim(step, t0, t1, erpm, model=(1443.0, 1400.0), impact=lambda t: 0.0):
    """30fps で回す。erpm(t, 前回の出力) は車の反応の代わり。戻り値 [(t, 出力, 種類)]"""
    out, t, last = [], t0, None
    while t < t1 - 1e-9:
        o, k = step(model, erpm(t, last), t, impact(t))
        last = o
        out.append((round(t, 3), o, k))
        t += DT
    return out


def rec_step(r, **kw):
    return lambda m, e, t, g: r.step(m, e, t, g, **kw)


def drive_then_stop(r, steer=1000.0, t_stop=1.0, impact_at=None):
    sim(rec_step(r), 0.0, t_stop, lambda t, o: -5000, model=(steer, 1350.0))
    if impact_at is not None:
        r.step((steer, 1450.0), -3000, impact_at, 6.0)


# ---------------------------------------------------------------- リカバリー
def test_not_triggered_before_moving():
    r = A.Recovery((C, N))
    res = sim(rec_step(r), 0.0, 3.0, lambda t, o: 0)          # 発進前: 止まったまま前進を指示
    assert all(o is None for _, o, _ in res)


def test_full_recovery_after_impact():
    r = A.Recovery((C, N))
    drive_then_stop(r, steer=1000.0, impact_at=1.0)            # 右いっぱいで走っていて衝撃
    rev_since = [None]

    def erpm(t, o):                                             # 後退を出すと 0.1秒後に下がる
        if o is not None and r.phase == 'reverse':
            rev_since[0] = rev_since[0] or t
            return 1800 if t - rev_since[0] > 0.1 else 0
        if o is not None and o[1] < N - 50:
            return -2500                                        # 前進(脱出)
        return 0
    res = sim(lambda m, e, t, g: r.step((1000.0, 1450.0), e, t, g), 1.0 + DT, 4.5, erpm)
    acts = [(t, o) for t, o, k in res if k == A.ASSIST_RECOVERY]
    assert acts and acts[0][0] <= 1.0 + A.Recovery.STUCK_AFTER_IMPACT_S + 0.1   # 衝撃の後 0.5秒で始まる
    thr = [o[1] for _, o in acts]
    i_gap = thr.index(N + A.GAP_OFFSET_US)
    assert i_gap >= 6 and all(x == N + A.DEFAULT_REV_US for x in thr[:i_gap])   # ブレーキ(捨てる1回目)
    rev = [o for _, o in acts[i_gap:] if o[1] == N + A.DEFAULT_REV_US]
    assert rev and all(o[0] == pytest.approx(C + A.Recovery.REV_STEER_US) for o in rev)  # 右に曲がっていた → 後退は少し左
    esc = [o for _, o in acts if o[1] < N]
    assert esc and all(o[0] == pytest.approx(C - A.Recovery.ESCAPE_STEER_FRAC * 500) for o in esc)  # 前進は右へ
    assert res[-1][1] is None and r.count == 1                  # 終わったらモデルへ


def test_waits_until_stopped_before_neutral():
    """前へ転がっている間は 中立→後退 が効かないので、止まるまでブレーキを続ける"""
    r = A.Recovery((C, N))
    drive_then_stop(r)
    sim(rec_step(r), 1.0, 1.9, lambda t, o: 0)                  # 止まって検出
    assert r.phase is not None
    r._go('brake', 2.0)
    res = sim(rec_step(r), 2.0, 2.6, lambda t, o: -800)          # まだ前へ転がっている
    assert all(o[1] == N + A.DEFAULT_REV_US for _, o, _ in res)


def test_retries_when_reverse_does_not_engage_then_escapes():
    r = A.Recovery((C, N))
    drive_then_stop(r)
    res = sim(rec_step(r), 1.0, 6.0, lambda t, o: 0)            # 何を出しても動かない
    thr = [o[1] for _, o, k in res if k == A.ASSIST_RECOVERY]
    assert max(thr) == N + A.DEFAULT_REV_US + A.Recovery.RETRY_EXTRA_US * A.Recovery.MAX_RETRIES  # やり直しで強く
    assert any(x < N for x in thr)                               # 最後は前進で抜けてみる
    assert r.retries == A.Recovery.MAX_RETRIES


def test_reverse_strength_is_adjustable():
    r = A.Recovery((C, N))
    drive_then_stop(r)
    res = sim(rec_step(r, rev_us=450.0, rev_s=0.2), 1.0, 3.0,
              lambda t, o: 2000 if r.phase == 'reverse' else 0)
    thr = [o[1] for _, o, k in res if k == A.ASSIST_RECOVERY]
    assert max(thr) == N + 450.0


def test_impact_while_reversing_goes_forward():
    r = A.Recovery((C, N))
    drive_then_stop(r)
    sim(rec_step(r), 1.0, 1.9, lambda t, o: 0)
    t = 1.9
    while r.phase != 'reverse':
        r.step((1443.0, 1450.0), 0, t)
        t += DT
    o, k = r.step((1443.0, 1450.0), 1500, t + 0.2, 5.0)          # 後ろにぶつかった
    assert r.phase == 'escape' and o[1] < N


def test_gives_up_and_resets_on_handover():
    a = A.Assist((C, N))
    t, kinds = 0.0, []
    for _ in range(4):                                           # 走る → 止まる(動かない)を繰り返す
        for _ in range(30):
            a.step((1443, 1350), -4000, True, t)
            t += DT
        for _ in range(150):
            kinds.append(a.step((1443, 1450), 0, True, t)[1])
            t += DT
    assert A.ASSIST_GIVEUP in kinds
    assert a.step((1443, 1350), -4000, True, t) == ((C, N), A.ASSIST_GIVEUP)       # あきらめたら中立のまま
    assert a.step((1443, 1350), 0, False, t + 0.1) == ((1443, 1350), A.ASSIST_NONE)  # 人が交代 → 解除
    assert a.step((1443, 1350), -4000, True, t + 0.2) == ((1443, 1350), A.ASSIST_NONE)


# ---------------------------------------------------------------- モデル自身の後退
def test_with_recovery_model_reverse_while_stopped_is_neutral():
    a = A.Assist((C, N))
    assert a.step((1600, 1900), -3000, True, 0.0) == ((1600, 1900), A.ASSIST_NONE)   # 走っている: ブレーキとしてそのまま
    assert a.step((1600, 1900), 0, True, 0.1) == ((1600, N), A.ASSIST_ESC)            # 止まっている: 中立(下がるのはリカバリー)


def test_esc_reverse_when_recovery_off():
    e = A.EscReverse(N)
    assert e.step(1900, -3000, 0.0) == (1900, False)              # 前へ走っている: ブレーキ
    assert e.step(1900, -800, 0.3) == (1900, False)               # まだ転がっている: ブレーキのまま
    thr, last = [], None
    for i in range(20):                                          # 止まった: 中立 → 後退(上限まで)→ 下がる
        v, _ = e.step(1900, 1500 if last == N + A.DEFAULT_REV_US else 0, 0.4 + i * DT)
        thr.append(v)
        last = v
    assert thr[0] == N + A.GAP_OFFSET_US and thr[-1] == N + A.DEFAULT_REV_US and e.retries == 0
    assert e.step(1300, 0, 1.2) == (1300, False) and e.state == 'idle'


def test_esc_reverse_limits_duration():
    e = A.EscReverse(N)
    e._go('reverse', 0.0)
    e.t_last = 0.0
    t, v = 0.0, None
    while t < A.EscReverse.REV_MAX_S + 0.1:
        t += DT
        v, _ = e.step(1700, 1500, t)
    assert e.state == 'rest' and v == N
    assert e.step(1700, 0, t + 0.1) == (N, True)                  # しばらく中立


def test_assist_passthrough_when_human_or_disabled():
    a = A.Assist((C, N))
    assert a.step((1600, 1900), 0, False, 0.0) == ((1600, 1900), A.ASSIST_NONE)
    assert a.step((1600, 1900), 0, True, 0.1, esc_on=False, rec_on=False) == ((1600, 1900), A.ASSIST_NONE)
