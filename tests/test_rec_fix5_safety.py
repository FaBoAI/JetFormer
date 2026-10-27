"""10/27 fix5 係 safety(反証 4 回目 10/27 06:2x〜07:3x の指摘 s1・s2・s3。fix4/findings_round4.md)の試験。走行ループ・補助の側。

  s1: rec_hold_keep = 1・rec_back_guard = 1 で、走っている車(約 3 m/s)の長い中立止め(0.5 秒より長く見ていない)の間に、車を
    止めない 2.5G 以上の衝撃(段差・こすり。本物の 10/23・10/24 の走行中に 143 回)が来ると、明けた最初のフレームから中止
    (中立・ハンドルまっすぐ)になり、惰性で 0.4〜3 m 進んだ。ToF ガード 2・3 のブレーキも中立で上書きした。
    直し(assist.Recovery._after_hold): 衝撃だけで中止にするのは、明けたフレームで前へ HOLD_IMPACT_FWD_ERPM(1000 eRPM =
    0.5 m/s)より遅いか後ろへ動いているときだけ。明けて止まっている(|eRPM| < 300)車の中止は前と同じ。
  s2: 発進の前進(launch_boost = 1)・発進の補助のフレームは Recovery.step より前に返るので、Recovery.t_last が区間の始めから
    None のまま。発進の前進の途中のカメラの止まりの間に壁に当たっても、明けたフレームで「見ていない時間」が分からず、長い
    中立止めのあとの中止が効かなかった(モデルの前進が壁を 1.88 秒押し、そのあと後退で 1.30 m 下がった)。
    直し(assist.Assist.step の rec_keep・drive が rec_hold_keep = 1 と rec_back_guard = 1 のとき渡す): 発進の前進・補助の
    フレームでも Recovery.t_last をそのフレームにする。
  s3: リカバリーの 'brake'・'wait'・'gap'(と 'reverse' の下がり始める前)の途中で車両の値が 0.25〜0.45 秒来ないと、drive の
    veh_unseen_back が「値が来なかった間」を後ろへ動いていたとして数え、下がった時間が rev_s に届いて 'coast' → escape。
    1 cm も下がっていない車が前の壁へ前進 130 µs(ハンドル 0.85)を約 0.5 秒出した。
    直し(drive: 値が来なかっただけの秒 hold_unseen_s を分けて渡す・assist.Recovery._after_hold: それを下がった時間に数えるのは
    下がり始めたあと('reverse'・'coast' で、前から下がっている か 中立止めの間に後ろへ動いていたサンプルがある)だけ。
    'brake'・'wait'・'gap' から進めるのは後ろへ動いていたサンプルがあったときだけ)。

OFF(rec_hold_keep・rec_back_guard のどちらかが 0。今の既定)は、直す前の写し tests/fixtures/fix5_safety_before/(assist.py・
drive.py。sha256 は SHA256SUMS)とビット単位で同じ(PCA9685 に書く値・記録の行・出来事)。どの試験も「直す前の写しで指摘が
起きる(再現)」を同じ試験の中で見る(直しを戻すと落ちる)。走行ループは tests/test_fix3_r2_1.py の閉じたループ(本物の
drive._loop を、書いた値で動く偽物の ESC と車で回す。反証の experiments/20261026_fix/fix3/refute2_safety/closed_loop.py と同じ作り)。
実機・GPU・torch・jetformer-web は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_rec_fix5_safety.py
"""
import ast
import collections
import hashlib
import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import pytest

from jetformer import assist as A
from jetformer import drive

import test_fix3_r2_1 as R2
import test_rec_hold_keep as RK
from test_drive_caps import same
from test_launch_drive import C, N, PARAMS, T0

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix5_safety_before'
SHA_BEFORE = {'assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6',     # 10/27 04:10(直す 3 回目のあと)
              'drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471'}      # 10/27 06:41(fix4 small のあと)
# この組のあと 10/27 fix6 safety(反証 6-1 回目 w1-safety-r61-safety-1)が assist.py・drive.py を変えた。そのときは、その直す前の写し
# tests/fixtures/fix6_safety_before/(= この組のあとの版。sha256 を照らす)をこの組の版として照らす(fix6 の違いは
# tests/test_rec_fix6_safety.py が見る。tests/test_fix3_r3_2_tools.py の LATER_FIX5・LATER_FIX6 と同じ値)
FIX5_AFTER = {'assist.py': '80f20890804170bede9c7df6e7f05892ab4e5397f8c5f837fda49ddba64d1276',
              'drive.py': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c'}
LATER_FIX6 = {'assist.py': 'd1d62c66bef856002c14db8dd2fd9c35279a36878b2e4817a279e9fb8bf64165',
              'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec'}
# さらに 10/27 fix6 safety の直す 6-2 回目(反証 6-2 回目 w2-safety-r62-safety-1・-2)が assist.py を変えた(drive.py は同じ)。その直す前の
# 写し tests/fixtures/fix6_safety_r2_before/(= LATER_FIX6 の版)から同じようにたどる(違いは tests/test_rec_fix6_safety_r2.py が見る。
# tests/test_fix3_r3_2_tools.py の LATER_FIX6_SAFETY_R2 と同じ値)
LATER_FIX6_R2 = {'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'}
# さらに 10/27 fix7 safety の直す 7-1 回目(反証 w1-safety-rv7-safety-r1-1)が assist.py・drive.py を変えた。その直す前の写し
# tests/fixtures/fix7_safety_before/(assist.py = LATER_FIX6_R2 の版・drive.py = LATER_FIX6 の版)から同じようにたどる(違いは
# tests/test_rec_fix7_safety.py が見る。tests/test_fix3_r3_2_tools.py の LATER_FIX7_SAFETY と同じ値)
LATER_FIX7_SAFETY = {'assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f',
                     'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}
# さらに 10/27 fix7 safety の直す 7-2 回目(反証 w2-safety-rv7-safety-r2-1)が assist.py を変えた(drive.py は同じ)。その直す前の写し
# tests/fixtures/fix7_safety_r2_before/(= LATER_FIX7_SAFETY の版)から同じようにたどる(違いは tests/test_rec_fix7_safety_r2.py が見る。
# tests/test_fix3_r3_2_tools.py の LATER_FIX7_SAFETY_R2 と同じ値)
LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}
FDT = 1 / 30.0
DT = R2.DT
XF = R2.XF
MODEL = (C, N - 120.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_BEFORE = {}


def load_before():
    """直す前の assist.py・drive.py(写し)をモジュールにする(sha256 を照らしてから)"""
    if _BEFORE:
        return _BEFORE['a'], _BEFORE['d']
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sums == SHA_BEFORE
    for name, h in sums.items():
        assert sha(FIX / name) == h, name
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix5s', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix5s', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    assert not hasattr(amod.Recovery, 'HOLD_IMPACT_FWD_ERPM')                 # 写しは直す前
    assert 'rec_keep' not in amod.Assist.step.__code__.co_varnames
    _BEFORE.update(a=amod, d=mod)
    return amod, mod


# ================================================================ 0. 写し・既定値・変えた所
def _fix5_version(name):
    """この組のあとの版(jetformer/ の今の版。fix6 safety が変えたあとは、その直す前の写し = この組のあとの版。fix6 safety の
    6-2 回目が変えたあとは、その直す前の写し = fix6 の版からたどる)"""
    p = ROOT / 'jetformer' / name
    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix7_safety_r2_before' / name
    if name in LATER_FIX7_SAFETY and sha(p) == LATER_FIX7_SAFETY[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix7_safety_before' / name
    if name in LATER_FIX6_R2 and sha(p) == LATER_FIX6_R2[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix6_safety_r2_before' / name
    if sha(p) == LATER_FIX6[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix6_safety_before' / name
    assert sha(p) == FIX5_AFTER[name], name
    return p


def _defs(path):
    """モジュールの説明の文を除いた、上の段の文と(クラスの中の)関数の AST"""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    out = {}
    for i, n in enumerate(tree.body):
        if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue
        if isinstance(n, ast.ClassDef):
            for m in n.body:
                key = '%s.%s' % (n.name, m.name if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) else
                                 ast.unparse(m.targets[0]) if isinstance(m, ast.Assign) else ast.dump(m)[:80])
                out[key] = ast.dump(m)
        else:
            out['top%d:%s' % (i, getattr(n, 'name', '') or ast.dump(n)[:60])] = ast.dump(n)
    return out


def test_before_copy_defaults_and_changed_parts():
    """写しの sha256・既定値(DEFAULT_PARAMS・PARAM_LIMITS・RESET_ON_BOOT・HOLD_VEH_GAP_S)は同じ。Recovery の定数は 1 つ足しただけ。
    変えた所は assist.py の Recovery._after_hold・Recovery.step・Assist.step と定数 1 つ、drive.py の Driver._loop だけ(AST)。
    10/27 fix6 safety のあとは、その直す前の写し(= この組のあとの版)で照らす(_fix5_version)"""
    amod, before = load_before()
    assert drive.DEFAULT_PARAMS == before.DEFAULT_PARAMS and drive.PARAM_LIMITS == before.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == before.RESET_ON_BOOT and drive.HOLD_VEH_GAP_S == before.HOLD_VEH_GAP_S
    for k in ('rec_back_guard', 'rec_hold_keep', 'launch_boost', 'launch_no_kick'):
        assert drive.DEFAULT_PARAMS[k] == 0 and k in drive.RESET_ON_BOOT
    for k, v in vars(amod.Recovery).items():
        if k.isupper():
            assert getattr(A.Recovery, k) == v, k
    assert {k for k in vars(A.Recovery) if k.isupper()} - {k for k in vars(amod.Recovery) if k.isupper()} == {
        'HOLD_IMPACT_FWD_ERPM'}
    assert A.Recovery.HOLD_IMPACT_FWD_ERPM == 1000 == A.Recovery.ARM_ERPM
    a0, a1 = _defs(FIX / 'assist.py'), _defs(_fix5_version('assist.py'))
    assert set(a1) - set(a0) == {'Recovery.HOLD_IMPACT_FWD_ERPM'} and set(a0) <= set(a1)
    assert sorted(k for k in a0 if a0[k] != a1[k]) == ['Assist.step', 'Recovery._after_hold', 'Recovery.step']
    d0, d1 = _defs(FIX / 'drive.py'), _defs(_fix5_version('drive.py'))
    assert set(d0) == set(d1) and sorted(k for k in d0 if d0[k] != d1[k]) == ['Driver._loop']


# ================================================================ 1. Recovery(単体)
def _armed(mod, erpm=-6000.0, n=40):
    """前へ走っている(armed・段なし・slow_since なし)Recovery と、最後に処理したフレームの時刻"""
    r = mod.Recovery((C, N))
    t = 100.0
    for _ in range(n):
        r.step(MODEL, erpm, t, 0.2, back_guard=True)
        t += FDT
    assert r.armed and r.phase is None and r.slow_since is None
    return r, t - FDT


S1_CASES = [  # (明けたフレームの eRPM, hold_g, 今の答え, 直す前の写しの答え)。答え = 段('giveup'・'halt'・None)
    (-4400.0, 3.0, None, 'giveup'),      # 2.2 m/s で走ったまま・3.0G: 指摘(直す前は中止)→ モデルのまま
    (-6000.0, 5.7, None, 'giveup'),
    (-2001.0, 3.0, None, 'giveup'),
    (-1001.0, 3.0, None, 'giveup'),      # 0.5 m/s ちょうどより速い: モデルのまま
    (-999.0, 3.0, 'giveup', 'giveup'),   # 0.5 m/s より遅い(止まりかけ): 前と同じく中止
    (0.0, 3.0, 'giveup', 'giveup'),
    (600.0, 3.0, 'halt', 'halt'),        # 後ろへ動いている(跳ね返った): 前と同じく先に前進 130 µs で止める
    (2800.0, 6.0, 'halt', 'halt'),
    (-4400.0, 2.4, None, None),          # 衝撃なし: 前と同じ(走っていれば何もしない)
    (0.0, 0.0, 'giveup', 'giveup'),      # 衝撃なしでも止まっていれば前と同じく中止
    (-800.0, 2.4, None, None),
]


@pytest.mark.parametrize('phase', [None, 'escape'])
@pytest.mark.parametrize('L', [0.6, 1.0])
@pytest.mark.parametrize('erpm,g,want,want0', S1_CASES)
def test_s1_unit_impact_only_giveup_needs_slow_or_backing(phase, L, erpm, g, want, want0):
    """長い中立止め(L 秒)のあと、段なし・'escape' の途中で、中立止めの間の衝撃 g。今は、明けて前へ 0.5 m/s より速く走って
    いればモデルのまま(段なしは (None, なし)・'escape' は続き)。直す前の写しは衝撃だけで中止(指摘の再現)"""
    amod, _ = load_before()
    res = {}
    for mod in (A, amod):
        r, last = _armed(mod)
        if phase == 'escape':
            r._go('escape', last)
        t1 = last + L
        o, k = r.step(MODEL, erpm, t1, 0.0, back_guard=True, hold_g=g, hold_since=last + 0.1, hold_back_s=0.0)
        res[mod is A] = (r.phase, o, k, r.guard_event)
    for now, w in ((True, want), (False, want0)):
        ph, o, k, ev = res[now]
        if w == 'giveup':
            assert (ph, o, k) == ('giveup', (C, N), A.ASSIST_GIVEUP) and ev[0] == 'hold_long', (now, ph)
        elif w == 'halt':
            assert ph == 'halt' and (o, k) == ((C, N - A.Recovery.ESCAPE_US), A.ASSIST_RECOVERY) and ev[0] == 'hold_long'
        else:
            assert ph not in ('giveup', 'halt') and ev is None, (now, ph)
            if phase is None:
                assert (o, k) == (None, A.ASSIST_NONE)
    if want is None and want0 is None:
        assert res[True][:3] == res[False][:3]                 # 変わらない場面は直す前と同じ


def test_s1_unit_short_hold_and_not_armed_unchanged():
    """0.5 秒までの中立止め・まだ armed でない車は、衝撃があっても前と同じ(何もしない)"""
    amod, _ = load_before()
    for mod in (A, amod):
        r, last = _armed(mod)
        assert r.step(MODEL, -4400.0, last + 0.4, hold_g=6.0, hold_since=last + 0.1) == (None, mod.ASSIST_NONE)
        r = mod.Recovery((C, N))
        for i in range(10):
            r.step(MODEL, -500.0, 100.0 + i * FDT)
        assert not r.armed
        assert r.step(MODEL, 0.0, 102.0, hold_g=6.0, hold_since=100.4) == (None, mod.ASSIST_NONE)


def _to_phase(mod, phase, backed=0.0):
    """'brake'・'wait'・'gap'・'reverse' の Recovery(中立止めの直前のフレームまで)。backed = 'reverse' で前から下がった秒"""
    r = mod.Recovery((C, N))
    if phase == 'wait':
        t = R2._to(r, 'brake')
        r._go('wait', t)
    else:
        t = R2._to(r, phase)
    r.step((C, N + 20.0), -2000.0 if phase == 'wait' else 0.0, t, back_guard=True)
    assert r.phase == phase
    r.backed_s = backed
    return r, t


@pytest.mark.parametrize('rev_s', [0.4, 0.3, 0.2])
@pytest.mark.parametrize('phase', ['brake', 'wait', 'gap', 'reverse'])
def test_s3_unit_unseen_only_does_not_skip_reverse(phase, rev_s):
    """下がる前の段('brake'・'wait'・'gap'・'reverse' でまだ下がっていない)で、車両の値が来なかっただけの 0.45 秒(後ろへ
    動いていたサンプルなし)の中立止め: 今は下がった時間に数えず、段も進めない(そのあと止まった車で 'escape' の前進を出さない
    まま後退側の段を続ける)。直す前の写し(同じ秒を後ろへ動いていたとして渡す)は 'coast' か 'escape' へ進み、前進 130 µs
    (指摘の再現。'gap' は rev_s が 0.45 秒以下のとき)"""
    amod, _ = load_before()
    r, t = _to_phase(A, phase)
    r0, t0 = _to_phase(amod, phase)
    assert t == t0
    t1 = t + 0.45
    kw = dict(back_guard=True, rev_s=rev_s, hold_since=t + 0.1, hold_back_s=0.45)
    o, k = r.step(MODEL, 0.0, t1, hold_unseen_s=0.45, **kw)
    o0, k0 = r0.step(MODEL, 0.0, t1, **kw)
    assert r.backed_s == 0.0 and r.guard_event is None and r.phase in ('brake', 'wait', 'gap', 'reverse')
    assert o is None or o[1] > N                                   # 前進を出さない(後退側か中立の帯)
    fwd0 = o0 is not None and o0[1] < N - 100
    assert r0.backed_s >= 0.45 and (fwd0 or r0.phase == 'coast')   # 直す前: 下がっていないのに下がったとして進めた
    if phase in ('brake', 'wait') or rev_s <= 0.45:
        assert fwd0 or r0.phase in ('coast', 'escape')
    # 続き: 止まった車(eRPM 0)のまま 0.3 秒。今は 'escape' に入らない(直す前は入ったまま前進)
    for i in range(9):
        tt = t1 + (i + 1) * FDT
        o, k = r.step(MODEL, 0.0, tt, back_guard=True, rev_s=rev_s)
        assert r.phase != 'escape' and (o is None or o[1] > N), (i, r.phase, o)


@pytest.mark.parametrize('phase', ['brake', 'wait', 'gap', 'reverse'])
def test_s3_unit_seen_backing_counts_as_before(phase):
    """中立止めの間に後ろへ動いていたサンプルがあった秒(hold_back_s のうち hold_unseen_s でない分)は前と同じに数える。
    'reverse' で前から下がっていれば(backed_s > 0)、値が来なかった間も前と同じく数える(後退を短くする側)"""
    amod, _ = load_before()
    for backed, unseen, hb in ((0.0, 0.0, 0.3), (0.0, 0.1, 0.3), (0.1, 0.25, 0.25), (0.1, 0.0, 0.2), (0.0, 0.3, 0.3)):
        if backed and phase != 'reverse':
            continue
        r, t = _to_phase(A, phase, backed)
        r0, t0 = _to_phase(amod, phase, backed)
        kw = dict(back_guard=True, rev_s=0.4, hold_since=t + 0.1)
        a = r.step(MODEL, 0.0, t + 0.4, hold_back_s=hb, hold_unseen_s=unseen, **kw)
        started = phase == 'reverse' and (backed > 0.0 or hb - unseen > 0.0)     # 下がり始めたあとは値が来なかった間も数える
        b = r0.step(MODEL, 0.0, t + 0.4, hold_back_s=hb if started else hb - unseen, **kw)
        assert (a, r.phase, r.backed_s, r.guard_event) == (b, r0.phase, r0.backed_s, r0.guard_event), (backed, unseen, hb)


def _hold_fuzz(seed, n=2500):
    """でたらめな入力に、残した中立止めのあとの最初のフレーム(hold_since・hold_g・hold_back_s・hold_unseen_s)を混ぜる
    (tests/test_fix3_r3_1.py の _hold_fuzz に hold_unseen_s と速い走り・後退側を足したもの)"""
    rng = random.Random(5000 + seed)
    t, out, erpm = 0.0, [], -4000.0
    for i in range(n):
        r = rng.random()
        if r < 0.08:
            erpm = rng.choice([-6000.0, -4400.0, -2000.0, -1200.0, -900.0, -400.0, 0.0, 0.0, 0.0, 200.0, 800.0, 1500.0, 3000.0])
        elif r < 0.5:
            erpm += rng.uniform(-300, 300)
        jump = rng.choice([FDT] * 16 + [0.2, 0.3, 0.45, 0.6, 1.0, 2.0])
        t_prev = t
        t += jump
        kw = {}
        if jump > FDT and rng.random() < 0.85:
            hb = rng.choice([0.0, 0.0, 0.1, 0.3, 0.45])
            kw = dict(hold_since=t_prev + 0.1, hold_g=rng.choice([0.0] * 5 + [3.0, 6.0]), hold_back_s=hb,
                      hold_unseen_s=rng.choice([0.0, hb, hb / 2.0]))
        out.append(((C + rng.uniform(-300, 300), N + rng.choice([-150.0, -60.0, 0.0, 20.0, 80.0])), erpm, t,
                    rng.choice([0.0] * 12 + [3.0, 6.0]), rng.random() < 0.85, rng.choice([0.2, 0.4, 0.4]), kw))
    return out


_KINDS = collections.Counter()


@pytest.mark.parametrize('seed', range(24))
def test_recovery_fuzz_diverges_only_by_s1_s3(seed):
    """でたらめな入力で直す前の写しと比べる。違いが出るのは、残した中立止めのあとの最初のフレームで (s1) 長い中立止め・段なしか
    'escape'・armed・衝撃 2.5G 以上・前へ 0.5 m/s 以上で走っている(直す前は中止・今は中止にしない)か、(s3) 値が来なかっただけの秒
    (hold_unseen_s > 0)がある後退側・下がる惰性の段 だけ。hold_unseen_s = 0 で s1 の場面でなければ出力・状態は全部同じ"""
    amod, _ = load_before()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    for model, erpm, t, ig, bg, rev_s, kw in _hold_fuzz(seed):
        st0 = {k: getattr(r0, k) for k in R2.STATE}
        gap = None if r0.t_last is None else t - r0.t_last
        kw0 = {k: v for k, v in kw.items() if k != 'hold_unseen_s'}
        if kw.get('hold_unseen_s'):
            kw0['hold_back_s'] = kw['hold_back_s']             # 直す前は全部を後ろへ動いていたとして数える
        a = r0.step(model, erpm, t, ig, rev_s=rev_s, back_guard=bg, **kw0)
        b = r1.step(model, erpm, t, ig, rev_s=rev_s, back_guard=bg, **kw)
        if a == b and all(getattr(r0, k) == getattr(r1, k) for k in R2.STATE) and r0.guard_event == r1.guard_event:
            r0.guard_event = r1.guard_event = None
            continue
        assert kw.get('hold_since') is not None and gap is not None, (seed, t)
        s1 = (gap > A.Recovery.HOLD_KEEP_MAX_S and st0['phase'] in (None, 'escape') and st0['armed']
              and kw['hold_g'] >= A.Recovery.IMPACT_G and erpm <= -A.Recovery.HOLD_IMPACT_FWD_ERPM)
        s3 = kw.get('hold_unseen_s', 0.0) > 0.0 and st0['phase'] in A.Recovery.HOLD_HIT_PHASES
        assert s1 or s3, (seed, t, st0['phase'], erpm, kw)
        if s1 and not s3:
            assert r0.phase == 'giveup' and a == ((C, N), A.ASSIST_GIVEUP) and r1.phase not in ('giveup', 'halt')
            _KINDS['s1'] += 1
        else:
            assert r1.backed_s <= r0.backed_s + 1e-9                    # 今は少なく数える(値が来なかっただけの秒)
            _KINDS['s3'] += 1
        break


def test_recovery_fuzz_saw_both_kinds():
    """上の でたらめな試験が空振りしていない(s1 の形も s3 の形も通った)"""
    if not (_KINDS['s1'] and _KINDS['s3']):
        for seed in range(24, 80):
            test_recovery_fuzz_diverges_only_by_s1_s3(seed)
    assert _KINDS['s1'] >= 1 and _KINDS['s3'] >= 1, dict(_KINDS)


@pytest.mark.parametrize('seed', range(4))
def test_recovery_without_hold_bit_identical(seed):
    """hold_since を渡さない(中立止めで残さない = rec_hold_keep・見張りのどちらかが 0・人側・dry run)Recovery は直す前と全部同じ"""
    amod, _ = load_before()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    for model, erpm, t, ig, hg, bg, rev_us, rev_s in R2._rec_fuzz_steps(seed):
        assert r0.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg) == \
            r1.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg)
        for k in R2.STATE + ('t_check',):
            assert getattr(r0, k) == getattr(r1, k), k
        assert r0.guard_event == r1.guard_event
        r0.guard_event = r1.guard_event = None


# ================================================================ 2. Assist(単体): 発進の前進・補助のフレームの t_last(s2)
def _li(**kw):
    d = dict(us=250.0, v=0.0, a=0.0, gz=0.0, steer_rel=0.0, ttc=30.0, ttc5=(30.0,) * 5, tof_ok=True, veh_ok=True)
    d.update(kw)
    return A.LaunchIn(**d)


def _launch_then_stall(mod, rec_keep, L=1.0, erpm_after=0.0, hold_g=9.6):
    """人側 → AI 側で発進の前進(止まっている → 前へ 1500 eRPM まで)。そのあとカメラが L 秒止まって(中立止め・残す)明けた
    最初のフレームと 1 秒の出力"""
    a = mod.Assist((C, N))
    t = 100.0
    kw = {} if rec_keep is None else {'rec_keep': rec_keep}
    for _ in range(5):
        a.step(MODEL, 0.0, False, t, rec_guard=True, launch=_li(), **kw)
        t += FDT
    outs = []
    for i in range(8):
        e = -200.0 * i
        o = a.step(MODEL, e, True, t, rec_guard=True, launch=_li(v=-e * 0.5e-3, a=3.0), **kw)
        outs.append(o)
        t += FDT
    assert all(k == A.ASSIST_LAUNCH for _, k in outs) and a.rec.armed     # 発進の前進を出し、動いて armed
    last = t - FDT
    a.launch_hold()
    a.reset(keep_recovery=True)
    res = []
    t = last + L
    for i in range(30):
        hk = dict(hold_g=hold_g, hold_since=last + 0.1, hold_back_s=0.0) if i == 0 else {}
        res.append(a.step(MODEL, erpm_after, True, t, rec_guard=True, launch=_li(), **hk, **kw))
        t += FDT
    return a, res


@pytest.mark.parametrize('L', [0.6, 1.0, 2.0])
def test_s2_unit_launch_stall_gives_up(L):
    """発進の前進の途中にカメラが L 秒止まり、明けて止まっている(壁に当たった・9.6G): 今(rec_keep)は明けた最初のフレームから
    中止(中立)のまま。直す前の写し・rec_keep = False は何もしない(モデルの前進のまま = 指摘の再現)"""
    amod, _ = load_before()
    a, res = _launch_then_stall(A, True, L)
    assert all(r == ((C, N), A.ASSIST_GIVEUP) for r in res) and a.rec.phase == 'giveup'
    for mod, keep in ((amod, None), (A, False)):
        b, res0 = _launch_then_stall(mod, keep, L)
        assert res0[0] == (MODEL, A.ASSIST_NONE) and not any(k == A.ASSIST_GIVEUP for _, k in res0[:10])


def test_s2_unit_launch_stall_moving_fast_left_to_model():
    """同じで、明けて前へ 1.5 m/s で走っている(衝撃 3G・止まらない): 中止にしない(s1 と同じ決まり)"""
    a, res = _launch_then_stall(A, True, 1.0, erpm_after=-3000.0, hold_g=3.0)
    assert res[0] == (MODEL, A.ASSIST_NONE) and a.rec.phase is None


def test_s2_unit_kick_frames_advance_t_last():
    """発進の補助(StartKick)が出したフレームも、rec_keep なら Recovery.t_last をそのフレームにする(False は今まで通り None)"""
    for keep in (True, False):
        a = A.Assist((C, N))
        a.step((C, N), 0.0, False, 100.0, kick_on=True, rec_keep=keep)
        o, k = a.step((C, N), 0.0, True, 100.1, kick_on=True, rec_keep=keep)
        assert k == A.ASSIST_START
        assert a.rec.t_last == (100.1 if keep else None)


def _drive_assist(a, ins, rec_keep, kick_on=True):
    """tests/test_rec_hold_keep._drive_assist と同じ呼び方(中立止めは reset のまま・hold_since は渡さない)。rec_keep を渡す"""
    res = []
    for x in ins:
        if x['hold']:
            a.launch_hold()
            a.reset()
            res.append('hold')
            continue
        kw = {} if rec_keep is None else {'rec_keep': rec_keep}
        r = a.step(x['m'], x['e'], x['sel'], x['t'], esc_on=True, rec_on=True, impact_g=x['g'], rev_us=x['rev_us'],
                   rev_s=x['rev_s'], kick_on=kick_on, blocked=x['blocked'], launch=x['launch'], rec_guard=True, **kw)
        res.append((r, a.boost.event, a.launch_on, a.launch_end))
    return res


def _inputs_with_launch(seed):
    """tests/test_rec_hold_keep._random_inputs に、人側 → AI 側の区切りごとに発進の前進が始まる 8 フレーム(止まっている →
    前へ 1400 eRPM。前が空いている)を入れたもの(発進の前進と、そのあとの StartKick・リカバリーを必ず通す)"""
    ins = [dict(x) for x in RK._random_inputs(seed)]
    n_cut = 0
    for i in range(1, len(ins) - 9):
        if not ins[i - 1]['sel'] and ins[i]['sel'] and not ins[i - 1]['hold']:
            n_cut += 1
            if n_cut % 2:
                continue                # 1 つおき(残りの区間は今まで通りのでたらめ = StartKick も通る)
            for j in range(8):
                x = ins[i + j]
                if x['hold'] or not x['sel']:
                    break
                e = -200.0 * j
                x.update(m=(C, N - 120.0 if j % 2 else N - 20.0), e=e, g=0.0, blocked=False,
                         launch=_li(v=-e * 0.5e-3, a=3.0))
    return ins


@pytest.mark.parametrize('seed', range(5))
def test_s2_unit_assist_rec_keep_off_identical(seed):
    """rec_keep = False(rec_hold_keep・見張りのどちらかが 0)は直す前の写しと全部同じ(発進の前進・補助・リカバリーを混ぜる)。
    rec_keep = True でも、中立止めのあとの手順を渡さず(hold_since なし)発進の補助を使わない間は同じ(t_last は見ていない時間
    にだけ使う)"""
    amod, _ = load_before()
    ins = _inputs_with_launch(seed)
    b = _drive_assist(amod.Assist((C, N)), ins, None)
    assert _drive_assist(A.Assist((C, N)), ins, False) == b
    assert _drive_assist(A.Assist((C, N)), ins, True, kick_on=False) == _drive_assist(amod.Assist((C, N)), ins, None,
                                                                                       kick_on=False)
    kinds = collections.Counter(r[0][1] for r in b if r != 'hold')
    assert kinds[A.ASSIST_LAUNCH] >= 1 and kinds[A.ASSIST_START] >= 1 and kinds[A.ASSIST_RECOVERY] > 30, dict(kinds)


# ================================================================ 3. 走行ループ(閉じたループ。本物の drive._loop)
class SpikeSim(R2.Sim):
    """R2.Sim と同じ。scene['spikes'] = [(t, g)] の時刻に、速さを変えない衝撃(段差・こすり)を IMU に入れる"""

    def _advance(self, t_end):
        sp = self.s.get('spikes') or []
        while self.t + DT <= t_end + 1e-9:
            nt = round(self.t + DT, 6)
            for ts, g in sp:
                if abs(nt - ts) < DT / 2 and not self.car.spike:
                    self.car.spike = [g, g]
            super()._advance(nt)
        super()._advance(t_end)


def run(mod, monkeypatch, tmp_path, params, scene):
    monkeypatch.setattr(R2, 'Sim', SpikeSim)
    return R2.run_cl(mod, monkeypatch, tmp_path, params, scene)


def same_loop(a, b):
    """PCA9685 に書いた値・記録の行(infer_ms のほか。欄の名前も)・出来事がビット単位で同じ"""
    (sa, ra, pa), (sb, rb, pb) = a, b
    assert pa.w == pb.w
    assert len(ra.rows) == len(rb.rows)
    for (t0, r0), (t1, r1) in zip(ra.rows, rb.rows):
        assert t0 == t1 and set(r0) == set(r1)
        for k in r0:
            if k != 'infer_ms':
                assert same(r0[k], r1[k]), (t0, k, r0[k], r1[k])
    assert ra.events == rb.events


def rows_s(rec):
    return [((t - T0) / 1e9, r) for t, r in rec.rows]


def P1(**kw):
    return dict(dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1), **kw)


# --- s1: 走っている車の長いカメラの止まり + 止まらない衝撃
def scene_s1(L, g, xf=60.0, thr=N - 200.0, a=3.0):
    return {'dur': a + L + 3.0, 'xf': xf, 'model': (C + 180.0, thr), 'drops': [(a, a + L)], 'spikes': [(a + 0.3, g)]}


@pytest.mark.parametrize('L', [0.6, 1.0])
def test_s1_loop_moving_car_not_given_up(monkeypatch, tmp_path, L):
    """約 3 m/s(モデル 前進 200 µs・ハンドル C + 180)で、カメラが L 秒止まった間に 3.0G。今(keep 1・見張り 1)は明けても中止に
    ならず、明けた最初のフレームから出力はモデルのまま(ハンドル 1621・スロットル 1296)で、keep 0 と同じ。直す前の写しは
    明けた最初のフレームから中止(中立・まっすぐ。指摘の再現)"""
    _, before = load_before()
    sc = scene_s1(L, 3.0)
    now = run(drive, monkeypatch, tmp_path, P1(), sc)
    k0 = run(drive, monkeypatch, tmp_path, P1(rec_hold_keep=0), sc)
    old = run(before, monkeypatch, tmp_path, P1(), sc)
    end = 3.0 + L
    after = [(s, r) for s, r in rows_s(now[1]) if s >= end and not r['hold']]
    assert after and not any(int(r['assist']) == A.ASSIST_GIVEUP for _, r in rows_s(now[1]))
    assert all((float(r['out'][0]), float(r['out'][1])) == (C + 180.0, N - 200.0) for _, r in after)
    out = lambda rec: [(s, float(r['out'][0]), float(r['out'][1])) for s, r in rows_s(rec) if s >= end and not r['hold']]  # noqa: E731
    assert out(now[1]) == out(k0[1])
    old_after = [r for s, r in rows_s(old[1]) if s >= end and not r['hold']]
    assert int(old_after[0]['assist']) == A.ASSIST_GIVEUP and all(int(r['assist']) == A.ASSIST_GIVEUP for r in old_after)
    v_back = next(v for t, x, v, th in old[0].log if t >= end)
    assert v_back > 1.5                                             # 直す前は速く走ったまま中止にしていた


@pytest.mark.parametrize('guard', [2, 3])
def test_s1_loop_tof_guard_brake_not_overridden(monkeypatch, tmp_path, guard):
    """同じで、前の壁がカメラの戻った所の 1.2 m 先・ToF ガード 2・3: 今はガードのブレーキが効き、当たる速さは keep 0 と同じ。
    直す前の写しは中止の中立で上書きして、より速く当たった(指摘の再現)"""
    _, before = load_before()
    sc = scene_s1(0.6, 3.0)
    sc['xf'] = 9.16 + 1.2
    p = P1(tof_guard=guard)
    now = run(drive, monkeypatch, tmp_path, p, sc)
    k0 = run(drive, monkeypatch, tmp_path, dict(p, rec_hold_keep=0), sc)
    old = run(before, monkeypatch, tmp_path, p, sc)
    hit = [h for h in now[0].car.hits if h[1] == 'front']
    hit0 = [h for h in k0[0].car.hits if h[1] == 'front']
    hit_old = [h for h in old[0].car.hits if h[1] == 'front']
    assert hit and hit0 and hit[0] == hit0[0]
    assert hit_old and hit_old[0][2] > hit[0][2] + 0.2


def test_s1_loop_stopped_by_coasting_still_gives_up(monkeypatch, tmp_path):
    """2 秒のカメラの止まりで惰性で止まりかけた(明けて 0.13 m/s)車は、前と同じく中止(r3-r3-safety-1 の形は変えない)"""
    _, before = load_before()
    sc = scene_s1(2.0, 3.0)
    now = run(drive, monkeypatch, tmp_path, P1(), sc)
    old = run(before, monkeypatch, tmp_path, P1(), sc)
    same_loop(now, old)
    assert any(int(r['assist']) == A.ASSIST_GIVEUP for _, r in now[1].rows)


# --- s2: 発進の前進の途中のカメラの止まりの間に前の壁
def model_s2(t, car):
    return (C, N) if t < 1.0 else (C, N - 120.0)


def scene_s2(xf, L, a=1.3):
    return {'dur': a + L + 4.0, 'xf': xf, 'model': model_s2, 'drops': [(a, a + L)] if L else []}


def P2(**kw):
    return P1(**dict(dict(launch_boost=1, launch_no_kick=0), **kw))


@pytest.mark.parametrize('xf', [1.3, 1.6])
@pytest.mark.parametrize('L', [1.0, 2.0])
def test_s2_loop_launch_stall_hit_gives_up(monkeypatch, tmp_path, xf, L):
    """発進の前進の途中(1.30 秒)からカメラが L 秒止まり、その間に前の壁に当たる: 今は明けた最初のフレームから中止(中立)で、
    壁を押す前進 0 秒・後退側 0 行。直す前の写しは中止にならず、モデルの前進が壁を押したあと後退した(指摘の再現)"""
    _, before = load_before()
    sc = scene_s2(xf, L)
    now = run(drive, monkeypatch, tmp_path, P2(), sc)
    old = run(before, monkeypatch, tmp_path, P2(), sc)
    end = 1.3 + L
    for sim, rec, _pca in (now, old):
        assert any(int(r['assist']) == A.ASSIST_LAUNCH for _, r in rec.rows)
        h = [x for x in sim.car.hits if x[1] == 'front']
        assert h and 1.3 <= h[0][0] < end                       # 当たりはカメラの止まりの間
    after = [(s, r) for s, r in rows_s(now[1]) if s >= end and not r['hold']]
    assert int(after[0][1]['assist']) == A.ASSIST_GIVEUP and all(int(r['assist']) == A.ASSIST_GIVEUP for _, r in after)
    push = sum(DT for t, x, v, th in now[0].log if t >= end and x >= xf - 1e-9 and th < N - 20)
    assert push == 0.0
    old_after = [r for s, r in rows_s(old[1]) if s >= end and not r['hold']]
    assert not any(int(r['assist']) == A.ASSIST_GIVEUP for r in old_after)
    push0 = sum(DT for t, x, v, th in old[0].log if t >= end and x >= xf - 1e-9 and th < N - 20)
    rev0 = [r for r in old_after if int(r['assist']) == A.ASSIST_RECOVERY and float(r['out'][1]) > N + 40]
    assert push0 > 1.0 and rev0


@pytest.mark.parametrize('name,params,L', [('launch_no_stall', P2(), 0.0), ('launch_keep0', P2(rec_hold_keep=0), 1.0),
                                           ('launch_guard0', P2(rec_back_guard=0), 1.0),
                                           ('launch_keep0_guard0', P2(rec_hold_keep=0, rec_back_guard=0), 2.0),
                                           ('kick_keep0', P1(launch_boost=0, start_kick=1, rec_hold_keep=0), 1.0)])
def test_s2_loop_identical_when_off_or_no_stall(monkeypatch, tmp_path, name, params, L):
    """rec_hold_keep か rec_back_guard が 0・カメラの止まりなし: PCA9685 に書く値・記録・出来事は直す前の写しとビット単位で同じ"""
    _, before = load_before()
    sc = scene_s2(1.3, L)
    same_loop(run(drive, monkeypatch, tmp_path, params, sc), run(before, monkeypatch, tmp_path, params, sc))


# --- s3: リカバリーの途中の車両の値の止まり
def scene_s3(vskip=None):
    return {'dur': 6.0, 'xf': XF, 'model': MODEL, 'drops': [], 'veh_skip': [vskip] if vskip else []}


def first_rec(sim, pca, t_from=1.5):
    """最初のリカバリー: 'brake' の始め(最初の後退側)・最初の escape(前進 130 µs)・その前に下がった距離・escape で前の壁を押した秒"""
    tb = next(t for t, s, th in pca.w if t > t_from and th > N + 40)
    te = next((t for t, s, th in pca.w if t > tb and abs(th - (N - A.Recovery.ESCAPE_US)) < 1e-6), None)
    xs = [x for t, x, v, th in sim.log if tb <= t <= (te if te else tb + 2.0)]
    push = sum(DT for t, x, v, th in sim.log if te is not None and te <= t < te + 0.6 and x >= XF - 1e-9
               and abs(th - (N - A.Recovery.ESCAPE_US)) < 1e-6)
    return tb, te, round(XF - min(xs), 3), round(push, 2)


_S3_REF = {}


def _s3_ref(monkeypatch, tmp_path, rev_s):
    if rev_s not in _S3_REF:
        sim, rec, pca = run(drive, monkeypatch, tmp_path, P1(rev_s=rev_s), scene_s3())
        _S3_REF[rev_s] = first_rec(sim, pca)
    return _S3_REF[rev_s]


@pytest.mark.parametrize('rev_s', [0.4, 0.3, 0.2])
@pytest.mark.parametrize('d0', [0.05, 0.2, 0.3])
@pytest.mark.parametrize('L', [0.25, 0.35, 0.45])
def test_s3_loop_vehicle_gap_no_escape_into_wall(monkeypatch, tmp_path, rev_s, d0, L):
    """前の壁 2.0 m に当たって止まり、最初のリカバリーの 'brake' の始めから d0 秒あとに車両の値が L 秒来ない(中立止め): 今は
    下がる前に escape(前進 130 µs)で前の壁を押さず、下がる距離は値の止まりが無いときと 3 cm 以内。直す前の写しは
    下がらないまま escape を出して壁を押した(指摘の再現。このとき下がったのは 0.3 m 未満)"""
    _, before = load_before()
    tb, te, back_ref, push_ref = _s3_ref(monkeypatch, tmp_path, rev_s)
    assert push_ref == 0.0 and back_ref > 0.4
    sc = scene_s3((tb + d0, tb + d0 + L))
    sim, rec, pca = run(drive, monkeypatch, tmp_path, P1(rev_s=rev_s), sc)
    assert any(r['hold'] for _, r in rec.rows)
    tb1, te1, back, push = first_rec(sim, pca)
    assert tb1 == tb and te1 is not None and push == 0.0
    assert back_ref - 0.15 <= back <= back_ref + 0.03, (back, back_ref)
    sim0, rec0, pca0 = run(before, monkeypatch, tmp_path, P1(rev_s=rev_s), sc)
    _, _, back0, push0 = first_rec(sim0, pca0)
    test_s3_loop_vehicle_gap_no_escape_into_wall.repro.append((rev_s, d0, L, back0, push0))


test_s3_loop_vehicle_gap_no_escape_into_wall.repro = []


def test_s3_loop_before_copy_reproduces():
    """上の 27 場面で、直す前の写しは escape で前の壁を押した(下がったのは 0.3 m 未満)場面が多い(指摘の再現)"""
    rp = test_s3_loop_vehicle_gap_no_escape_into_wall.repro
    if len(rp) < 27:
        pytest.skip('上の試験を同じ回で流していない')
    bad = [x for x in rp if x[4] > 0.0 and x[3] < 0.3]
    assert len(bad) >= 20, rp


@pytest.mark.parametrize('name,params', [('keep0', P1(rec_hold_keep=0)), ('guard0', P1(rec_back_guard=0)),
                                         ('both0', P1(rec_hold_keep=0, rec_back_guard=0)),
                                         ('defaults', dict(PARAMS))])
def test_s3_loop_identical_when_off(monkeypatch, tmp_path, name, params):
    """同じ場面(車両の値の止まり 0.35 秒)で、rec_hold_keep か rec_back_guard が 0 なら直す前の写しとビット単位で同じ"""
    _, before = load_before()
    tb = 2.09
    sc = scene_s3((tb + 0.05, tb + 0.40))
    same_loop(run(drive, monkeypatch, tmp_path, params, sc), run(before, monkeypatch, tmp_path, params, sc))


# ================================================================ 4. でたらめな閉じたループ(安全の決まり)
def _rand_model(rng, dur):
    cuts, t = [], 0.0
    while t < dur:
        L = rng.uniform(0.3, 2.0)
        cuts.append((t, t + L, C + rng.choice([0.0, 0.0, 120.0, -150.0, 250.0]),
                     N - rng.choice([60.0, 100.0, 120.0, 150.0, 200.0, 250.0])))
        t += L

    def f(s, car):
        if s < 1.0:
            return (C, N)                  # 人側の間は止めておく(偽物の車は人側でも書いた値で動く)
        for a, b, st, th in cuts:
            if a <= s < b:
                return st, th
        return cuts[-1][2], cuts[-1][3]
    return f


def _rand_scene(rng):
    dur = rng.choice([7.0, 9.0])
    xf = rng.uniform(1.6, 3.2) if rng.random() < 0.85 else rng.uniform(6.0, 40.0)
    sc = {'dur': dur, 'xf': xf, 'model': _rand_model(rng, dur), 'drops': [], 'veh_skip': [], 'spikes': []}
    if rng.random() < 0.6:
        sc['xr'] = xf - rng.uniform(0.5, 2.2)
    if rng.random() < 0.5:
        a = rng.uniform(1.1, 1.6)
        sc['drops'].append((a, a + rng.choice([0.6, 1.0, 2.0])))         # 発進の途中のカメラの止まり(s2)
    for _ in range(rng.randint(0, 4)):
        a = rng.uniform(1.0, dur - 0.5)
        sc['drops'].append((a, a + rng.choice([0.12, 0.2, 0.35, 0.55, 0.7, 1.0, 1.5])))
    for _ in range(rng.randint(0, 3)):
        a = rng.uniform(1.0, dur - 0.5)
        sc['veh_skip'].append((a, a + rng.uniform(0.1, 0.5)))           # 車両の値の止まり(s3)
    for _ in range(rng.randint(0, 5)):
        sc['spikes'].append((round(rng.uniform(1.2, dur - 0.2), 2), rng.choice([2.6, 3.0, 4.0, 6.0])))   # s1
    return sc


def _rand_params(rng):
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=rng.choice([0, 1]), launch_boost=rng.choice([0, 1]),
             launch_no_kick=rng.choice([0, 1]), tof_guard=rng.choice([1, 1, 2, 3]), throttle_gain=rng.choice([0.8, 1.0, 1.2]),
             rev_s=rng.choice([0.2, 0.3, 0.4]), rev_us=rng.choice([200.0, 300.0, 400.0]), rec_back_guard=1, rec_hold_keep=1)
    return p


def check_rules(sim, rec, pca, p):
    """安全の決まり。破った所の一覧(空なら守った)
    R1 PCA9685 に書く値・記録の出力は 1000〜2000 µs の数 / R2 前進は出どころより強くならない(モデル・後退の受け付け = ゲインの
    あとのモデル、発進の補助 = 80 µs かモデル、発進の前進 = launch_us かモデル、リカバリー = 130 µs、中止 = 0)/ R3 中止は中立・
    ハンドル中心で、人側にするまで解けない(中立止めをはさんでも)/ R4 発進の前進・補助・中止は後退側を出さない。長く(0.5 秒
    より)見ていない明けから 0.3 秒、後ろへ動いている車にリカバリーの後退側を出さない / R5 人側では補助・発進の前進なし /
    R6 中止の最初の行で前へ 1.0 m/s 以上で走っていない(s1)/ R7 中立止めの行は中立"""
    bad = []
    for t, s_, th in pca.w:
        if not (math.isfinite(s_) and math.isfinite(th) and 1000.0 <= s_ <= 2000.0 and 1000.0 <= th <= 2000.0):
            bad.append(('R1_pca', t, s_, th))
    rows = rows_s(rec)
    in_gu = False
    procs = []
    for s, r in rows:
        o_s, o_t = float(r['out'][0]), float(r['out'][1])
        if not (math.isfinite(o_s) and math.isfinite(o_t) and 1000.0 <= o_s <= 2000.0 and 1000.0 <= o_t <= 2000.0):
            bad.append(('R1', s, o_s, o_t))
        if r['hold']:
            if o_t != N or int(r['assist']) != A.ASSIST_NONE or r['launch_on']:
                bad.append(('R7', s, o_t))
            if not r['select']:
                in_gu = False
            continue
        k = int(r['assist'])
        if not r['select']:
            in_gu = False
            if k != A.ASSIST_NONE or r['launch_on']:
                bad.append(('R5', s, k))
            continue
        procs.append(s)
        fwd = max(N - o_t, 0.0)
        m_fwd = max(N - float(r['model_out'][1]), 0.0)
        lim = {A.ASSIST_NONE: m_fwd, A.ASSIST_ESC: m_fwd, A.ASSIST_START: max(m_fwd, A.StartKick.KICK_US),
               A.ASSIST_LAUNCH: max(m_fwd, float(p['launch_us'])), A.ASSIST_RECOVERY: A.Recovery.ESCAPE_US,
               A.ASSIST_GIVEUP: 0.0}[k]
        if fwd > lim + 1e-6:
            bad.append(('R2', s, k, fwd, lim))
        if k in (A.ASSIST_START, A.ASSIST_LAUNCH, A.ASSIST_GIVEUP) and o_t > N + 40.0:
            bad.append(('R4_side', s, k, o_t))
        if k == A.ASSIST_GIVEUP:
            if (o_s, o_t) != (C, N):
                bad.append(('R3_neutral', s, o_s, o_t))
            if not in_gu:
                v = next(v for t, x, v, th in sim.log if t >= s - 1e-9)
                if v >= 1.0:
                    bad.append(('R6', s, round(v, 2)))
            in_gu = True
        elif in_gu:
            bad.append(('R3_released', s, k))
    ends = [b for a_, b in zip(procs, procs[1:]) if b - a_ > 0.5]
    for e in ends:
        for s, r in rows:
            if e <= s < e + 0.3 and not r['hold'] and r['select'] and int(r['assist']) == A.ASSIST_RECOVERY \
                    and float(r['out'][1]) > N + 40:
                v = next(v for t, x, v, th in sim.log if t >= s - 1e-9)
                if v < -0.15:
                    bad.append(('R4_backing', e, s, round(v, 2)))
                    break
    return bad


@pytest.mark.parametrize('seed', range(16))
def test_closed_loop_fuzz_rules_and_off_identity(monkeypatch, tmp_path, seed):
    """でたらめな場面(前の壁・後ろの物・発進の途中のカメラの止まり・カメラと車両の値の止まり・止まらない衝撃・人側にしない)で、
    rec_hold_keep = 1・見張り 1(と発進の前進・補助・launch_no_kick・ToF ガード・ゲインはでたらめ)の本物の走行ループが安全の
    決まり R1〜R6 を破らない。同じ場面で rec_hold_keep = 0 は直す前の写しとビット単位で同じ"""
    _, before = load_before()
    rng = random.Random(77000 + seed)
    for _ in range(2):
        sc = _rand_scene(rng)
        p = _rand_params(rng)
        sim, rec, pca = run(drive, monkeypatch, tmp_path, p, sc)
        assert check_rules(sim, rec, pca, p) == [], (seed, {k: p[k] for k in ('launch_boost', 'start_kick', 'tof_guard')})
        p0 = dict(p, rec_hold_keep=0)
        same_loop(run(drive, monkeypatch, tmp_path, p0, sc), run(before, monkeypatch, tmp_path, p0, sc))
