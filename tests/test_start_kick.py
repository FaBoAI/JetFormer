"""発進の補助(assist.StartKick・Assist.step の kick_on)。2026-10-23"""
from jetformer import assist as A

N = 1496.0


def _a():
    return A.Assist((1441.0, N))


def test_kicks_when_stopped_and_model_neutral():
    a = _a()
    out, k = a.step((1441.0, N), 0.0, True, 0.0, kick_on=True)
    assert k == A.ASSIST_START and out[1] == N - A.StartKick.KICK_US
    assert a.kick_count == 1


def test_stops_after_moving_and_passes_model():
    a = _a()
    a.step((1441.0, N), 0.0, True, 0.0, kick_on=True)
    out, k = a.step((1441.0, N), -1000.0, True, 0.1, kick_on=True)
    assert k == A.ASSIST_NONE and out[1] == N
    out, k = a.step((1441.0, N), 0.0, True, 0.2, kick_on=True)     # 一度動いたら同じ AI の区間ではもう出さない
    assert k != A.ASSIST_START


def test_blocked_reverse_strong_forward_and_off():
    a = _a()
    assert a.step((1441.0, N), 0.0, True, 0.0, kick_on=True, blocked=True)[1] != A.ASSIST_START
    assert a.step((1441.0, N + 200), 0.0, True, 0.1, kick_on=True)[1] != A.ASSIST_START
    out, k = a.step((1441.0, 1350.0), 0.0, True, 0.2, kick_on=True)
    assert k != A.ASSIST_START and out[1] == 1350.0
    b = _a()
    assert b.step((1441.0, N), 0.0, True, 0.0, kick_on=False)[1] == A.ASSIST_NONE


def test_gives_up_after_max_s_and_rearms_on_ai_switch():
    a = _a()
    a.step((1441.0, N), 0.0, True, 0.0, kick_on=True)
    assert a.step((1441.0, N), 0.0, True, A.StartKick.MAX_S + 0.1, kick_on=True)[1] != A.ASSIST_START
    a.step((1441.0, N), 0.0, False, 3.0, kick_on=True)            # CH3 を人側へ
    assert a.step((1441.0, N), 0.0, True, 3.1, kick_on=True)[1] == A.ASSIST_START


def test_human_side_does_nothing():
    a = _a()
    out, k = a.step((1441.0, N), 0.0, False, 0.0, kick_on=True)
    assert k == A.ASSIST_NONE and out == (1441.0, N)
