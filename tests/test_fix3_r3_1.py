"""直す 3 回目・1 組目(2026-10-27 fix3。反証 3 回目の指摘 r3-r3-safety-1・r3-r3-safety-2)の試験。走行ループ・補助の側。

  r3-r3-safety-1: rec_hold_keep = 1・rec_back_guard = 1 で、前の壁に当たった直後にカメラが 0.6〜2 秒来ないと、当たりのあと
    1 フレーム処理できたときは中止(hold_long)、1 フレームも処理できなかったときは中止にならず(最後に処理したフレームで走っていて
    slow_since が無い)、明けてモデルの前進が壁を約 0.8 秒押したあと後退へ進み 1.1〜1.4 m 下がった(位相だけで分かれた)。
    escape のあとの cooldown の中の長い中立止めも同じ。
  r3-r3-safety-2: 'reverse' の途中(後ろへ −1.2〜−1.4 m/s)の長い中立止めで中止にすると、すぐ中立にしてブレーキも見張りも無く
    惰性で下がり、rec_hold_keep = 0・中立止めなしでは当たらない後ろの物に当たった。
  直し(assist.Recovery._after_hold・_hold_giveup・段 'halt'):
    safety-1: 段なし(止まっている時間を数えていない)・'escape' でも、armed・前に処理したフレームから 0.5 秒より長い中立止め・
      明けたフレームで止まっている(|eRPM| < 300)か中立止めの間の IMU が 2.5G 以上なら中止(出来事 hold_long)。
    safety-2: 長い中立止めで中止にするとき、明けたフレームで後ろへ動いていれば(eRPM > 300)、先に前進 ESCAPE_US(130 µs)を
      後ろへ動かなくなるか 0.3 秒まで出し('halt'。種類はリカバリー)、そのあと中立で人を待つ。

OFF(rec_hold_keep = 0・見張り OFF。今の既定)と中立止めの無い走行は、直す前の写し tests/fixtures/fix3_r3_1_before/(assist.py・
drive.py。sha256 は SHA256SUMS)とビット単位で同じ(PCA9685 に書く値・記録の行・出来事)。走行ループは tests/test_fix3_r2_1.py の
閉じたループ(本物の drive._loop を、書いた値で動く偽物の ESC と車で回す)に、当たりに続くカメラの止まり(hit_drop)と中立の惰性の
減速(dec)を足したもの(反証 3 回目の experiments/20261026_fix/fix3/refute3_safety/cl3.py・x10b と同じ作り)。実機・GPU・torch は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix3_r3_1.py
"""
import collections
import hashlib
import importlib.util
import random
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

from jetformer import assist as A
from jetformer import clock, config, drive

import test_fix3_r2_1 as R2
from test_drive_caps import FakeRecorder, assert_rows_equal_before, fake_normalize
from test_launch_drive import C, N, PARAMS, FLOOR_MM, T0, NS

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix3_r3_1_before'
# この組のあとの drive.py の版の sha256(test_before_copies_and_defaults_unchanged がその組の直す前の写しでこの組の版につなぐ。
# tests/test_fix3_r3_2_tools.py の LATER_FIX4・LATER_FIX5・LATER_FIX6 と同じ値)
LATER_DRIVE = {'fix4_small': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
               'fix5_safety': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c',
               'fix6_safety': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
               'fix7_safety': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}     # 10/27 fix7 safety 7-1 回目(LATER_FIX7_SAFETY)
DT = 0.01
FDT = 1 / 30.0
MODEL = (C, N - 120.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sums():
    return dict(reversed(line.split()) for line in (FIX / 'SHA256SUMS').read_text().splitlines() if line.strip())


_BEFORE = {}


def load_before():
    """直す前の assist.py・drive.py(写し。2026-10-27 04:0x。直す 2 回目のあと)をモジュールにする"""
    if _BEFORE:
        return _BEFORE['a'], _BEFORE['d']
    s = sums()
    for name in ('drive.py', 'assist.py'):
        assert sha(FIX / name) == s[name], name
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix3r31', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix3r31', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    assert not hasattr(amod.Recovery, 'HOLD_HALT_MAX_S')                   # 写しは直す前
    _BEFORE.update(a=amod, d=mod)
    return amod, mod


# ================================================================ 0. 写し・既定値
def test_before_copies_and_defaults_unchanged():
    s = sums()
    for name in ('assist.py', 'drive.py', 'app.js', 'index.html', 'collect.py', 'car_check.py'):
        assert sha(FIX / name) == s[name], name
    # この組は drive.py を変えていない。あとの組の版でもよい(10/27 fix4 small = 説明の文だけ・10/27 fix5 safety = 反証 4 回目の
    # s2・s3・10/27 fix6 safety = 反証 6-1 回目の s3-a・10/27 fix7 safety = 反証 7-1 回目の s4-a)。そのときは、その組の直す前の写しの
    # sha256 をたどってこの組の版
    # (SHA256SUMS)につながるか見る
    cur = sha(ROOT / 'jetformer' / 'drive.py')
    chain = {LATER_DRIVE['fix7_safety']: ROOT / 'tests' / 'fixtures' / 'fix7_safety_before' / 'drive.py',
             LATER_DRIVE['fix6_safety']: ROOT / 'tests' / 'fixtures' / 'fix6_safety_before' / 'drive.py',
             LATER_DRIVE['fix5_safety']: ROOT / 'tests' / 'fixtures' / 'fix5_safety_before' / 'drive.py',
             LATER_DRIVE['fix4_small']: ROOT / 'tests' / 'fixtures' / 'fix4_small_before' / 'drive.py'}
    while cur != s['drive.py']:
        assert cur in chain, cur
        cur = sha(chain.pop(cur))
    amod, before = load_before()
    assert drive.DEFAULT_PARAMS == before.DEFAULT_PARAMS and drive.PARAM_LIMITS == before.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == before.RESET_ON_BOOT and drive.DEFAULT_PARAMS['rec_hold_keep'] == 0
    for k, v in vars(amod.Recovery).items():                                  # 前からの定数は全部同じ
        if k.isupper():
            assert getattr(A.Recovery, k) == v, k
    # 10/27 fix5 safety(反証 4 回目 s1)で HOLD_IMPACT_FWD_ERPM を足した(tests/test_rec_fix5_safety.py)
    assert {k for k in vars(A.Recovery) if k.isupper()} - {k for k in vars(amod.Recovery) if k.isupper()} == {
        'HOLD_HALT_ERPM', 'HOLD_HALT_MAX_S', 'HOLD_IMPACT_FWD_ERPM'}
    assert A.Recovery.HOLD_HALT_ERPM == A.BACKING_ERPM == 300 and A.Recovery.HOLD_HALT_MAX_S == 0.3
    assert A.Recovery.HOLD_IMPACT_FWD_ERPM == 1000
    for k in ('BACKING_ERPM', 'CHECK_S', 'STOPPED_ERPM', 'GAP_OFFSET_US', 'DEFAULT_REV_US', 'DEFAULT_REV_S'):
        assert getattr(A, k) == getattr(amod, k)


# ================================================================ 1. Recovery の形
@pytest.mark.parametrize('seed', range(6))
def test_recovery_without_hold_bit_identical(seed):
    """hold_since を渡さない(rec_hold_keep = 0・見張り OFF・人側・dry run・中立止めの無い走行)Recovery は、直す前の写しと
    出力・状態が全部同じ。'halt' にも入らない"""
    amod, _ = load_before()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    n_phase = 0
    for model, erpm, t, ig, hg, bg, rev_us, rev_s in R2._rec_fuzz_steps(seed):
        a = r0.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg)
        b = r1.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg)
        assert a == b
        for k in R2.STATE + ('t_check',):
            assert getattr(r0, k) == getattr(r1, k), k
        assert r1.phase != 'halt'
        n_phase += r1.phase is not None
        r0.guard_event = r1.guard_event = None
    assert n_phase > 300


def _hold_fuzz(seed, n=3000):
    """でたらめな入力に、残した中立止めのあとの最初のフレーム(hold_since・hold_g・hold_back_s)を混ぜる"""
    rng = random.Random(1000 + seed)
    t, out, erpm = 0.0, [], -4000.0
    for i in range(n):
        r = rng.random()
        if r < 0.08:
            erpm = rng.choice([-5000.0, -2000.0, -400.0, 0.0, 0.0, 0.0, 200.0, 800.0, 1500.0, 3000.0])
        elif r < 0.5:
            erpm += rng.uniform(-300, 300)
        jump = rng.choice([FDT] * 18 + [0.25, 0.4, 0.6, 1.0, 2.0])
        t_prev = t
        t += jump
        held = jump > FDT and rng.random() < 0.8
        out.append(((C + rng.uniform(-300, 300), N + rng.choice([-150.0, -60.0, 0.0, 20.0, 80.0])), erpm, t,
                    rng.choice([0.0] * 12 + [3.0, 6.0]), rng.random() < 0.85,
                    dict(hold_since=t_prev + 0.1, hold_g=rng.choice([0.0] * 6 + [3.0]), hold_back_s=rng.choice([0.0, 0.0, 0.1]))
                    if held else {}))
    return out


@pytest.mark.parametrize('seed', range(8))
def test_recovery_hold_fuzz_changes_only_after_long_hold(seed):
    """残した中立止めを混ぜたでたらめな入力: 直す前の写しと違いが出るのは、前に処理したフレームから HOLD_KEEP_MAX_S(0.5 秒)より
    長い中立止めのあとの最初のフレームから。違い方は (1) 段なし・'escape'・armed で、止まっている か hold_g ≥ 2.5G → 中止 (2) 中止に
    するとき後ろへ動いている → 'halt'(前進 130 µs)。'halt' は後ろへ動かなくなるか 0.3 秒で中止(中立)になり、中止は解けない"""
    amod, _ = load_before()
    kinds = collections.Counter()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    halt_t = None
    diverged = False
    for model, erpm, t, ig, bg, kw in _hold_fuzz(seed):
        if diverged:
            break
        st0 = {k: getattr(r0, k) for k in R2.STATE}
        gap = None if r0.t_last is None else t - r0.t_last
        a = r0.step(model, erpm, t, ig, back_guard=bg, **kw)
        b = r1.step(model, erpm, t, ig, back_guard=bg, **kw)
        if a == b and all(getattr(r0, k) == getattr(r1, k) for k in R2.STATE):
            r0.guard_event = r1.guard_event = None
            continue
        diverged = True
        assert kw.get('hold_since') is not None and gap is not None and gap > A.Recovery.HOLD_KEEP_MAX_S, (seed, t, gap)
        assert r1.guard_event[0] == 'hold_long'
        if r1.phase == 'halt':
            assert b == ((C, N - A.Recovery.ESCAPE_US), A.ASSIST_RECOVERY) and erpm > A.BACKING_ERPM
            assert a == ((C, N), A.ASSIST_GIVEUP) or a[1] != A.ASSIST_GIVEUP
            kinds['halt'] += 1
            halt_t = t
        else:
            assert r1.phase == 'giveup' and b == ((C, N), A.ASSIST_GIVEUP)
            assert st0['phase'] in (None, 'escape') and st0['armed']
            assert -300 < erpm < 300 or kw['hold_g'] >= A.Recovery.IMPACT_G
            kinds['giveup_%s' % st0['phase']] += 1
    if halt_t is not None:                     # 'halt' のあと: 後ろへ動いている間・0.3 秒までは前進、そのあと中止のまま
        t = halt_t
        for e in (2000.0, 1200.0, 2500.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0):
            t += FDT
            o, k = r1.step(MODEL, e, t)
            if t - halt_t < A.Recovery.HOLD_HALT_MAX_S:
                assert (o, k) == ((C, N - 130.0), A.ASSIST_RECOVERY)
            else:
                assert (o, k) == ((C, N), A.ASSIST_GIVEUP) and r1.phase == 'giveup'
        for _ in range(30):
            t += FDT
            assert r1.step(MODEL, 0.0, t, 6.0, hold_since=t - 0.5) == ((C, N), A.ASSIST_GIVEUP)
    test_recovery_hold_fuzz_changes_only_after_long_hold.kinds.update(kinds)


test_recovery_hold_fuzz_changes_only_after_long_hold.kinds = collections.Counter()


def test_recovery_hold_fuzz_both_kinds_seen():
    """上の でたらめな試験が空振りしていない(違いの 2 つの形の両方を通った)"""
    for seed in range(8, 40):
        test_recovery_hold_fuzz_changes_only_after_long_hold(seed)
    k = test_recovery_hold_fuzz_changes_only_after_long_hold.kinds
    assert k['halt'] >= 2 and (k['giveup_None'] + k['giveup_escape']) >= 2, dict(k)


def _run_to_hit(rec_mod, case):
    """反証の unit_phase と同じ: 前へ走る(eRPM −5000)→ 当たり。case 'A' = 当たりのあと 1 フレーム処理できた(6.5G・止まった)/
    'B' = 処理できずにカメラ停止。戻り値 (Assist, 当たりの時刻, 最後に処理したフレームの時刻)"""
    a = rec_mod.Assist((C, N))
    now = 100.0
    for _ in range(60):
        a.step(MODEL, -5000.0, True, now, impact_g=0.2, rec_guard=True)
        now += FDT
    t_hit = now - 0.025
    if case == 'A':
        now = t_hit + 0.008
        a.step(MODEL, 0.0, True, now, impact_g=6.5, rec_guard=True)
    return a, t_hit, now


def _after_drop(rec_mod, a, t_hit, last, L, keep=True, erpm_after=0.0, hold_g=6.5, secs=4.0):
    a.reset(keep_recovery=keep)
    now, out, first = t_hit + L, [], True
    while now < t_hit + L + secs:
        kw = dict(hold_g=hold_g, hold_since=last + 0.1, hold_back_s=0.0) if (keep and first) else {}
        o, k = a.step(MODEL, erpm_after, True, now, impact_g=0.0, rec_guard=True, **kw)
        ev = a.rec.guard_event
        a.rec.guard_event = None
        out.append((round(now - (t_hit + L), 3), o, k, ev))
        first = False
        now += FDT
    return out


@pytest.mark.parametrize('L', [0.6, 1.0, 2.0])
@pytest.mark.parametrize('case', ['A', 'B'])
def test_safety1_unit_hit_then_camera_stall_gives_up_in_both_phases(case, L):
    """反証の確かめ 1(unit_phase): 当たりのあと 1 フレーム処理できた(A)・できなかった(B)の両方で、明けた最初のフレームで中止
    (hold_long・段なし)。4 秒の間に後退側も前進も出さない。直す前の写しは B で中止にならず、明けて 0.83 秒で後退側を書いた"""
    amod, _ = load_before()
    a, t_hit, last = _run_to_hit(A, case)
    out = _after_drop(A, a, t_hit, last, L)
    assert out[0][2] == A.ASSIST_GIVEUP and out[0][3][0] == 'hold_long' and out[0][3][1] is None
    assert all(k == A.ASSIST_GIVEUP and o == (C, N) for _, o, k, _ in out)
    b, t_hit0, last0 = _run_to_hit(amod, case)
    out0 = _after_drop(amod, b, t_hit0, last0, L)
    rev0 = [t for t, o, k, _ in out0 if k == amod.ASSIST_RECOVERY and o[1] > N + 40]
    if case == 'A':
        assert out0[0][2] == amod.ASSIST_GIVEUP                                   # 前から中止(同じ)
        assert [(t, o, k) for t, o, k, _ in out] == [(t, o, k) for t, o, k, _ in out0]
    else:
        assert not any(k == amod.ASSIST_GIVEUP for _, _, k, _ in out0) and rev0 and 0.8 < rev0[0] < 0.9    # 指摘の再現


@pytest.mark.parametrize('keep', [False, True])
def test_safety1_unit_keep_off_and_cases_left_as_before(keep):
    """変わらない場面(直す前の写しと出力が同じ): keep = 0(hold_since を渡さない)/ 明けて前へ走っていて衝撃なし /
    0.4 秒(0.5 秒まで)の中立止め / まだ armed でない"""
    amod, _ = load_before()
    scenes = [dict(L=1.0, erpm_after=-3000.0, hold_g=0.0), dict(L=0.4, erpm_after=0.0, hold_g=6.5)]
    if not keep:
        scenes.append(dict(L=1.0, erpm_after=0.0, hold_g=6.5))
    for sc_ in scenes:
        outs = []
        for mod in (A, amod):
            a, t_hit, last = _run_to_hit(mod, 'B')
            outs.append([(t, o, k) for t, o, k, _ in _after_drop(mod, a, t_hit, last, keep=keep, **sc_)])
        assert outs[0] == outs[1], sc_
    for mod in (A, amod):                      # armed でない(走り出す前): 止まったまま長い中立止め → 何もしない
        r = mod.Recovery((C, N))
        t = 0.0
        for _ in range(10):
            r.step(MODEL, -500.0, t); t += FDT
        assert not r.armed
        assert r.step(MODEL, 0.0, t + 2.0, hold_g=6.0, hold_since=t + 0.1) == (None, mod.ASSIST_NONE) and r.guard_event is None


def test_safety1_unit_moving_with_impact_during_hold_gives_up():
    """明けたフレームで動いていても、中立止めの間の IMU が 2.5G 以上なら中止(直す前は何もしない)。2.4G なら何もしない。
    2026-10-27 fix5 safety(反証 4 回目 s1)で狭めた: 衝撃だけで中止にするのは、明けたフレームで前へ 0.5 m/s(1000 eRPM)より
    遅いか後ろへ動いているときだけ(後ろへ動いていれば先に 'halt' の前進)。前へ 1.5 m/s(−3000)で走ったままならモデルのまま
    (直す前の写しと同じ。この組のときは中止にしていた。くわしくは tests/test_rec_fix5_safety.py)"""
    amod, _ = load_before()
    for e, g, want in ((-800.0, 6.5, A.ASSIST_GIVEUP), (-800.0, 2.4, A.ASSIST_NONE), (800.0, 6.5, A.ASSIST_RECOVERY),
                       (-3000.0, 6.5, A.ASSIST_NONE), (-3000.0, 2.4, A.ASSIST_NONE)):
        a, t_hit, last = _run_to_hit(A, 'B')
        out = _after_drop(A, a, t_hit, last, 1.0, erpm_after=e, hold_g=g, secs=0.2)
        assert out[0][2] == want, (e, g)
        if want == A.ASSIST_RECOVERY:
            assert out[0][1] == (C, N - A.Recovery.ESCAPE_US) and out[0][3][0] == 'hold_long'    # 'halt'
        b, t0, l0 = _run_to_hit(amod, 'B')
        assert _after_drop(amod, b, t0, l0, 1.0, erpm_after=e, hold_g=g, secs=0.2)[0][2] == amod.ASSIST_NONE


@pytest.mark.parametrize('L', [0.6, 1.0, 2.0])
def test_safety1_unit_cooldown_after_escape_gives_up(L):
    """反証の確かめ 3(unit_cooldown): escape のあとの cooldown の中(段なし・slow_since なし)に 0.6〜2.0 秒の中立止め → 明けて
    止まっている: 中止。直す前の写しは中止にならず、明けて 0.83〜1.27 秒で後退側を書いた"""
    res = {}
    for mod in (A, load_before()[0]):
        a = mod.Assist((C, N))
        r = a.rec
        now = 100.0
        for _ in range(30):
            a.step(MODEL, -5000.0, True, now, rec_guard=True); now += FDT
        r.phase, r.slow_since, r.cooldown_until, r.t_last = None, None, now + 1.0, now
        r.starts.append(now - 3.0)
        hold_t0 = now + 0.1
        a.reset(keep_recovery=True)
        now += L
        t0, first, out = now, True, []
        while now < t0 + 3.0:
            kw = dict(hold_g=0.0, hold_since=hold_t0, hold_back_s=0.0) if first else {}
            o, k = a.step(MODEL, 0.0, True, now, rec_guard=True, **kw)
            out.append((round(now - t0, 3), o, k)); first = False; now += FDT
        res[mod is A] = out
    assert all(k == A.ASSIST_GIVEUP and o == (C, N) for _, o, k in res[True])
    rev0 = [t for t, o, k in res[False] if k == A.ASSIST_RECOVERY and o[1] > N + 40]
    assert rev0 and 0.8 <= rev0[0] <= 1.3                                              # 指摘の再現


def test_safety1_unit_escape_long_hold():
    """'escape' の途中の長い中立止め: 明けて止まっている → 中止(hold_long・'escape')。前へ走っている → 前と同じ(時間で終わる)"""
    amod, _ = load_before()
    for erpm, want in ((0.0, 'giveup'), (-2000.0, None)):
        for mod in (A, amod):
            r = mod.Recovery((C, N))
            t = R2._to(r, 'reverse')
            r._go('escape', t)
            r.step((C, N), 0, t + FDT)
            o, k = r.step(MODEL, erpm, t + 2.0, hold_since=t + 0.1)
            if mod is A and want == 'giveup':
                assert r.phase == 'giveup' and (o, k) == ((C, N), A.ASSIST_GIVEUP) and r.guard_event[:2] == ('hold_long', 'escape')
            else:
                assert r.phase is None and o is None and r.guard_event is None


def _to_phase(mod, phase):
    r = mod.Recovery((C, N))
    if phase in ('brake', 'gap', 'reverse'):
        t = R2._to(r, phase)
    else:
        t = R2._to(r, 'reverse' if phase == 'coast' else 'brake')
        r._go(phase, t)
    r.step((C, N + 20.0), {'wait': -2000, 'coast': 2000}.get(phase, 0), t, back_guard=True)
    assert r.phase == phase
    return r, t


@pytest.mark.parametrize('phase', ['brake', 'wait', 'gap', 'reverse', 'coast'])
def test_safety2_unit_backing_at_long_hold_brakes_first(phase):
    """長い中立止めのあと中止にするとき、明けたフレームで後ろへ動いていれば(eRPM > 300)前進 130 µs('halt'。種類はリカバリー)。
    後ろへ動かなくなったら中立の中止。直す前の写しはすぐ中立の中止(惰性で下がった)"""
    amod, _ = load_before()
    r, t = _to_phase(A, phase)
    r0, t0 = _to_phase(amod, phase)
    assert t == t0
    t1 = t + 0.6
    o, k = r.step(MODEL, 2800.0, t1, back_guard=True, hold_since=t + 0.1, hold_back_s=0.3)
    assert (o, k) == ((C, N - 130.0), A.ASSIST_RECOVERY) and r.phase == 'halt' and r.guard_event[:2] == ('hold_long', phase)
    assert r0.step(MODEL, 2800.0, t1, back_guard=True, hold_since=t + 0.1, hold_back_s=0.3) == ((C, N), A.ASSIST_GIVEUP)  # 再現
    seq = [(1800.0, 'halt'), (700.0, 'halt'), (250.0, 'giveup'), (-300.0, 'giveup'), (2000.0, 'giveup')]
    for i, (e, want) in enumerate(seq):
        o, k = r.step(MODEL, e, t1 + (i + 1) * FDT, back_guard=True)
        assert r.phase == want
        assert (o, k) == (((C, N - 130.0), A.ASSIST_RECOVERY) if want == 'halt' else ((C, N), A.ASSIST_GIVEUP))
    ast = A.Assist((C, N))                     # CH3 を人側にすると解ける(今まで通り)
    ast.rec = r
    ast.step((C, N), 0, False, t1 + 1.0)
    assert r.phase is None


def test_safety2_unit_halt_is_capped_and_not_extended_by_holds():
    """'halt' の前進は HOLD_HALT_MAX_S(0.3 秒。'halt' に入ったフレームから)まで。途中の中立止めでも延びない。
    明けたフレームで後ろへ動いていない(eRPM ≤ 300)なら、前と同じくすぐ中立の中止"""
    r, t = _to_phase(A, 'reverse')
    t1 = t + 0.6
    assert r.step(MODEL, 3000.0, t1, hold_since=t + 0.1)[1] == A.ASSIST_RECOVERY
    tt, n_fwd = t1, 1
    while r.phase == 'halt':
        tt += FDT
        o, k = r.step(MODEL, 3000.0, tt)
        n_fwd += k == A.ASSIST_RECOVERY
    assert tt - t1 >= A.Recovery.HOLD_HALT_MAX_S - 1e-9 and tt - t1 < A.Recovery.HOLD_HALT_MAX_S + FDT + 1e-9
    assert (n_fwd - 1) * FDT < A.Recovery.HOLD_HALT_MAX_S + 1e-6 and r.phase == 'giveup'     # 最後の前進のフレームは 0.3 秒の前
    r2, t2 = _to_phase(A, 'reverse')
    r2.step(MODEL, 3000.0, t2 + 0.6, hold_since=t2 + 0.1)
    o, k = r2.step(MODEL, 3000.0, t2 + 0.6 + 0.35, hold_since=t2 + 0.65)       # 0.35 秒の中立止めのあと: 上限を過ぎた
    assert (o, k) == ((C, N), A.ASSIST_GIVEUP)
    r3, t3 = _to_phase(A, 'reverse')
    assert r3.step(MODEL, 300.0, t3 + 0.6, hold_since=t3 + 0.1) == ((C, N), A.ASSIST_GIVEUP) and r3.phase == 'giveup'


def test_safety2_unit_stuck_counting_and_impact_cases():
    """段なしで止まっている時間を数えている途中(r2-r2-rs-2 の中止)も、明けて後ろへ動いていれば先に 'halt'。後退側の段で中立止めの
    間に後ろの衝撃(2.5G 以上)があれば、前と同じく rear_impact_hold(すぐ前へ 0.5 秒。'halt' にしない)"""
    r = A.Recovery((C, N))
    t = 0.0
    for _ in range(40):
        r.step((C, N - 150.0), -5000, t); t += FDT
    r.step(MODEL, 0, t)
    assert r.slow_since is not None
    o, k = r.step(MODEL, 1500.0, t + 1.0, hold_since=t + 0.1)
    assert r.phase == 'halt' and (o, k) == ((C, N - 130.0), A.ASSIST_RECOVERY) and r.guard_event[:2] == ('hold_long', None)
    amod, _ = load_before()
    for mod in (A, amod):
        rr, tt = _to_phase(mod, 'reverse')
        o, k = rr.step(MODEL, 2800.0, tt + 1.0, hold_g=6.0, hold_since=tt + 0.1)
        assert rr.phase == 'escape' and rr.guard_event[0] == 'rear_impact_hold' and o[1] == N - 130.0


# ================================================================ 2. 走行ループ(閉じたループ)
class Car3(R2.Car):
    """R2.Car(反証 2 回目の closed_loop.py と同じ偽物の ESC と車)に、中立の惰性の減速 dec(既定 1.5 = R2 と同じ。反証の x10b と同じ
    足し方)と、当たりに続くカメラの止まり hit_drop(遅れ, 長さ。反証の cl3 と同じ)を足したもの"""
    sim = None

    def step(self, t, thr):
        neutral = not (thr < N - 20.0 or thr > N + 40.0)
        v_before, n_hits = self.v, len(self.hits)
        r = super().step(t, thr)
        dec = self.s.get('dec', 1.5)
        if neutral and dec != 1.5 and self.contact is None:
            dv = (dec - 1.5) * DT
            if v_before > 0:
                self.v = max(0.0, self.v - dv)
            elif v_before < 0:
                self.v = min(0.0, self.v + dv)
        if len(self.hits) > n_hits and self.s.get('hit_drop') and self.sim is not None:
            dl, ln = self.s['hit_drop']
            self.sim.drops.append((t + dl, t + dl + ln))
        return r


class Sim3(R2.Sim):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.car = Car3(self.s)
        self.car.sim = self


def run_cl(mod, monkeypatch, tmp_path, params, scene):
    """R2.run_cl と同じ(Sim だけ Sim3)"""
    box = {'ns': T0, 'm': 100.0}
    with monkeypatch.context() as mp:
        mp.setattr(clock, 'now_ns', lambda: box['ns'])
        mp.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                      time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
        mp.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
        fp = tmp_path / 'floor_ref.npy'
        np.save(fp, np.full((42, 54), float(FLOOR_MM), np.float32))
        mp.setattr(mod, 'FLOOR_REF_PATH', fp)
        rec = FakeRecorder()
        d = mod.Driver(hub=None, recorder=rec)
        d.params = dict(params)
        d.hb_token, d.hb_count = 'tok', 1
        d.st = {'phase': 'driving', 'mode_id': 4}
        pca = R2.TimedPCA(box)
        out = mod._Output(pca, config.PCA9685, (C, N), lambda: d.params)
        bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
        cam_slot = [None]
        sim = Sim3(d, scene, bufs, cam_slot, box, pca)
        d._loop(d.st, R2.Runner(sim), fake_normalize, pca, out, (C, N), 6, 0.0, cam_slot, sim, *bufs, False)
        d._stop.clear()
    return sim, rec, pca


def P(keep, guard=1, kick=1):
    return dict(PARAMS, recovery=1, start_kick=kick, rec_back_guard=guard, rec_hold_keep=keep)


def guards(rec):
    return [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']


def s1_scene(xf, L):
    """反証の x2c_phase・verify closed_phase: 前の壁 xf に約 1.8 m/s・6.5G で当たり、当たりと同時にカメラが L 秒来ない"""
    return {'dur': 8.0, 'xf': xf, 'xr': None, 'model': MODEL, 'drops': [], 'hit_drop': (0.0, L) if L else None}


def s1_metrics(sim, rec, pca, L):
    t_hit = sim.car.hits[0][0]
    back = t_hit + (L or 0.0)
    rev = [t for t, s, th in pca.w if t >= back and th > N + 40]
    gu = [t for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_GIVEUP]
    xs = [x for t, x, v, th in sim.log if t >= back]
    push = sum(1 for t, x, v, th in sim.log if t >= back and x >= sim.s['xf'] - 1e-9 and th < N - 20) * DT
    return {'v_hit': sim.car.hits[0][2], 'g_hit': sim.car.hits[0][3], 'giveup': bool(gu),
            'rev_after_s': round(rev[0] - back, 2) if rev else None, 'backed_m': round(sim.s['xf'] - min(xs), 3),
            'push_s': round(push, 2)}


S1_FLIP = (2.00, 2.06, 2.07)      # 反証の表で中止にならなかった 3 か所(当たりのあと 1 フレームも処理できなかった)


@pytest.mark.parametrize('L', [0.6, 2.0])
def test_safety1_loop_hit_then_camera_stall_always_gives_up(monkeypatch, tmp_path, L):
    """閉じたループ(反証の表と同じ場面。壁 2.00〜2.07 m の 8 か所・約 1.8 m/s・6.5G): keep = 1・guard = 1 で、どの位相でも
    中止(hold_long)・明けてから後退側を書かない・壁を押さない・下がらない。直す前の写しは 3 か所で 0.5〜0.87 秒後に後退へ進み
    1.1〜1.4 m 下がった(指摘の再現)"""
    _, before = load_before()
    for j in range(8):
        xf = round(2.00 + 0.01 * j, 2)
        sim, rec, pca = run_cl(drive, monkeypatch, tmp_path, P(1), s1_scene(xf, L))
        m = s1_metrics(sim, rec, pca, L)
        assert 1.7 < m['v_hit'] < 1.9 and 6.2 < m['g_hit'] < 6.8, m
        assert m['giveup'] and m['rev_after_s'] is None and m['backed_m'] <= 0.005 and m['push_s'] == 0.0, (xf, m)
        assert 'hold_long' in guards(rec)
        if xf in S1_FLIP:
            sim0, rec0, pca0 = run_cl(before, monkeypatch, tmp_path, P(1), s1_scene(xf, L))
            m0 = s1_metrics(sim0, rec0, pca0, L)
            assert not m0['giveup'] and 0.45 < m0['rev_after_s'] < 0.9 and m0['backed_m'] > 1.1, (xf, m0)


@pytest.mark.parametrize('name', ['k0_L1.0_2.00', 'k0_L1.0_2.03', 'k0_L0.6_2.06', 'k1_none_2.00', 'k1_none_2.05', 'k1g0_L1.0_2.00'])
def test_safety1_loop_off_and_no_hold_bit_identical(monkeypatch, tmp_path, name):
    """keep = 0(カメラの止まりあり)・keep = 1 の中立止めなし・keep = 1 でも見張り OFF: 直す前の写しとビット単位で同じ
    (PCA9685 に書く値と時刻・記録の行・出来事)"""
    _, before = load_before()
    k, L, xf = name.split('_')
    keep, guard = (1, 0) if k == 'k1g0' else (int(k[1]), 1)
    s = s1_scene(float(xf), None if L == 'none' else float(L[1:]))
    R2.same_cl(run_cl(drive, monkeypatch, tmp_path, P(keep, guard), s), run_cl(before, monkeypatch, tmp_path, P(keep, guard), s))


# safety-2(反証の check_giveup_coast・x10b): 前の壁 2.0 m のリカバリー。中立止めなしの段は brake 2.09・gap 2.35・reverse 2.52
TR = 2.52
REACH = {1.5: 1.304, 3.0: 1.174}       # 中立止めなしで下がる距離(dec ごと。test_safety2_loop_reference で確かめる)


def s2_scene(dec=1.5, drop=None, back=None):
    return {'dur': 8.0, 'xf': 2.0, 'xr': None if back is None else 2.0 - back, 'model': MODEL, 'drops': [drop] if drop else [],
            'dec': dec}


def s2_metrics(sim, pca, end):
    at = next(r for r in sim.log if r[0] >= end)
    xs = [r for r in sim.log if end <= r[0] <= end + 2.0]
    return {'rear': [h for h in sim.car.hits if h[1] == 'rear'], 'v_back': round(at[2], 2),
            'back_after_m': round(at[1] - min(r[1] for r in xs), 3),
            'thr_after': [(round(t - end, 2), th) for t, s, th in pca.w if end <= t <= end + 0.5]}


def test_safety2_loop_reference(monkeypatch, tmp_path):
    """比べの元: 中立止めなし(keep 0/1)の 'reverse' の始まり・下がる距離(反証の数と同じ)"""
    for dec, reach in REACH.items():
        for keep in (0, 1):
            sim, rec, pca = run_cl(drive, monkeypatch, tmp_path, P(keep, kick=0), s2_scene(dec))
            assert abs((2.0 - min(x for t, x, v, th in sim.log if t > 1.6)) - reach) < 0.005
    w = [(round(t, 2), th) for t, s, th in pca.w if 2.4 < t < 2.6]
    assert (TR, N + 300.0) in w


S2_HIT = [(1.5, 0.15, 0.40, 0.55, (-1.17, 4.93)), (1.5, 0.30, 0.40, 0.55, (-0.96, 4.4)), (1.5, 0.50, 0.40, 0.55, (-0.57, 3.43)),
          (1.5, 0.50, 0.45, 0.60, (-1.08, 4.7)), (3.0, 0.15, 0.45, 0.60, (-0.47, 3.16)), (3.0, 0.50, 0.50, 0.55, (-0.94, 4.34))]


@pytest.mark.parametrize('dec,add,o,L,hit0', S2_HIT)
def test_safety2_loop_no_rear_hit_that_keep0_and_nohold_avoid(monkeypatch, tmp_path, dec, add, o, L, hit0):
    """反証の表の場面(後ろの物は中立止めなしで下がる所の +add m・'reverse' に入って o 秒後から L 秒カメラが来ない): keep = 0 と
    中立止めなしで当たらない後ろの物に、keep = 1 でも当たらない。明けて前進 130 µs(halt)で止め、そのあと中立(中止)。
    直す前の写しは keep = 1 だけ当たった(速さ・G は反証の数と同じ)"""
    _, before = load_before()
    back, drop = REACH[dec] + add, (TR + o, TR + o + L)
    sim_n, _, _ = run_cl(drive, monkeypatch, tmp_path, P(1, kick=0), s2_scene(dec, None, back))
    sim_0, _, _ = run_cl(drive, monkeypatch, tmp_path, P(0, kick=0), s2_scene(dec, drop, back))
    assert not [h for h in sim_n.car.hits if h[1] == 'rear'] and not [h for h in sim_0.car.hits if h[1] == 'rear']
    sim, rec, pca = run_cl(drive, monkeypatch, tmp_path, P(1, kick=0), s2_scene(dec, drop, back))
    m = s2_metrics(sim, pca, drop[1])
    assert not m['rear'] and m['v_back'] < -0.4, m
    assert 'hold_long' in guards(rec) and any(int(r['assist']) == A.ASSIST_GIVEUP for _, r in rec.rows if not r['hold'])
    fwd = [t for t, th in m['thr_after'] if th == N - 130.0]
    assert fwd and fwd[0] <= 0.04 and max(fwd) < A.Recovery.HOLD_HALT_MAX_S + 0.05            # 明けてすぐ前進・0.3 秒まで
    assert m['thr_after'][-1][1] == N                                                         # そのあと中立(中止)
    sim0, rec0, pca0 = run_cl(before, monkeypatch, tmp_path, P(1, kick=0), s2_scene(dec, drop, back))
    r0 = [h for h in sim0.car.hits if h[1] == 'rear']
    assert r0 and (r0[0][2], r0[0][3]) == hit0                                                 # 指摘の再現


S2_GRID = [(dec, o, L) for dec in (1.5, 3.0) for o, L in ((0.40, 0.55), (0.45, 0.60), (0.50, 0.55), (0.50, 0.70), (0.30, 1.0))]


@pytest.mark.parametrize('dec,o,L', S2_GRID)
def test_safety2_loop_keep1_not_worse_than_keep0(monkeypatch, tmp_path, dec, o, L):
    """後ろの物なし: 明けてから下がる距離は keep = 1 でも keep = 0 以下(+ 5 mm)。keep = 0 は直す前とビット単位で同じ"""
    _, before = load_before()
    drop = (TR + o, TR + o + L)
    a = run_cl(drive, monkeypatch, tmp_path, P(0, kick=0), s2_scene(dec, drop))
    R2.same_cl(a, run_cl(before, monkeypatch, tmp_path, P(0, kick=0), s2_scene(dec, drop)))
    b = run_cl(drive, monkeypatch, tmp_path, P(1, kick=0), s2_scene(dec, drop))
    m0, m1 = s2_metrics(a[0], a[2], drop[1]), s2_metrics(b[0], b[2], drop[1])
    assert m1['back_after_m'] <= m0['back_after_m'] + 0.005, (m0, m1)
