"""10/27 fix7 係 safety 直す 7-1 回目(反証 7-1 回目の指摘 w1-safety-rv7-safety-r1-1)の試験。走行ループ・補助の側。

  指摘: rec_back_guard = 1 の後ろの見張りの back_early が、eRPM の 1 サンプルの跳ね(> 300・車は動いていない)だけで 'brake' から
    'gap' の中立を飛ばして 'reverse' へ進む。前進のあとのこの ESC は中立を見ないと後退に入らない(ブレーキのまま)。'reverse' は
    そのフレームの dt を下がった時間に数えるので backed_s > 0 になり、CHECK_S のやり直し(下がらなければ中立を長くして
    やり直す)も効かない。止まったまま後退側を 1.2 秒書き、0 m のまま escape(前進 130 µs)が前の壁を 0.5 秒押す。OFF は
    brake → gap → reverse で 1.13 m 下がる。本物の記録(data/raw)でも、止まっている車の eRPM が 1 サンプルだけ 1053〜1687 を
    示す(ホールの縁の行き来)。0.1 秒に 2〜4 回続くこともある。
  直し(assist.Recovery・drive.py。見張り ON(rec_back_guard = 1)の日だけ効く。rec_hold_keep は問わない):
    s4-a  drive は rec_back_guard = 1 の日、書けた値を assist.EscMayReverse(EscTrack より多めにみる版: ブレーキのあと中立を
          1 回でも書いたら下がれる・後退側を書いたときから後退)に入れ、前に helper.step を呼んでからいままでに ESC が後退で
          走らせていたかもしれないかを esc_rev で渡す。この手順でまだ一度も True(か None)になっていない間の 'brake'・'wait'・'gap'
          では、後ろへ動いたサンプルを見張りに使わない(back_early・'gap' の惰性の数え・後ろの当たりの「直前に後ろへ動いていた」・
          中立止めのあとの数えと back_hold。出来事 back_no_esc を手順ごとに 1 回)
    s4-b  'reverse' の CHECK_S のやり直しは、'reverse' の中で見た後退(seen_rev_s。入ったフレームは除く)で決める。'gap' を
          飛ばして入った 'reverse'(back_early・back_hold)で後退を見なければ、ふつうの 'gap' からやり直す(出来事 early_none。
          やり直しの数・+50 µs は使わない)
  ESC が後退を待っていた 10/23 の形(場面 brake)は、今まで通り back_early で数える(直す前の写しとビット単位で同じ)。

OFF(rec_back_guard = 0。今の既定)は、直す前の写し tests/fixtures/fix7_safety_before/(assist.py・drive.py = fix6 r2 のあと。
sha256 は SHA256SUMS)とビット単位で同じ(PCA9685 に書く値・記録の行・出来事)。指摘の場面の試験は「直す前の写しで指摘が起きる
(再現)」を同じ試験の中で見る。走行ループは tests/test_fix3_r2_1 の閉じたループ(本物の drive._loop を、書いた値で動く偽物の
ESC と車で回す)に、車両の値の 1 サンプルの差し替え(跳ね)・車両の値の間隔・偽物の ESC の中立の秒を足した GSim で回す。
実機・GPU・torch・jetformer-web は使わない。台本と表: experiments/20261026_fix/fix7/safety/。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_rec_fix7_safety.py
"""
import hashlib
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

from jetformer import assist as A
from jetformer import drive

import test_fix3_r2_1 as R2
import test_rec_fix5_safety as F5
import test_rec_fix6_safety as T6
from test_launch_drive import C, N, PARAMS, T0, NS, TOF_FREE, tof_near
from test_launch_drive import build as build_open, run as run_open

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix7_safety_before'
SHA_BEFORE = {'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4',     # 10/27 fix6 safety r2 のあと
              'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec'}
# この組のあと 10/27 fix7 safety の直す 7-2 回目(反証 w2-safety-rv7-safety-r2-1)が assist.py を変えた(drive.py は同じ)。そのときは、
# その直す前の写し tests/fixtures/fix7_safety_r2_before/(= この組のあとの版。sha256 を照らす)をこの組の版として照らす(7-2 回目の
# 違いは tests/test_rec_fix7_safety_r2.py が見る。tests/test_fix3_r3_2_tools.py の LATER_FIX7_SAFETY・LATER_FIX7_SAFETY_R2 と同じ値)
FIX7_R1_AFTER = {'assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f',
                 'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}
LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}
# さらに 10/28 fix8 safety が drive.py の説明の文だけを変えた(AST は同じ)。その直す前の写し tests/fixtures/fix8_safety_doc_before/
# (= この組のあとの drive.py)から同じようにたどる(tests/test_fix3_r3_2_tools.py の LATER_FIX8_SAFETY_DOC と同じ値)
LATER_FIX8_SAFETY_DOC = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e'}
# さらに 10/29 fix9 の優先のモデル(main session。走行タブの一覧の並べ方だけ)が drive.py を変えた。その直す前の写し
# tests/fixtures/fix9_priority_before/(= LATER_FIX8_SAFETY_DOC の版)から同じようにたどる(変えたのが優先のモデルの所だけかは
# tests/test_fix9_priority_before.py。tests/test_fix3_r3_2_tools.py の LATER_FIX9_PRIORITY と同じ値)
LATER_FIX9_PRIORITY = {'drive.py': 'ecfb6656cae624950d5de221dfa71a5ac31b84ba82dacfbea2136832eae92315'}


def _fix7r1_version(name):
    """この組のあとの版(jetformer/ の今の版。7-2 回目が変えたあとは、その直す前の写し = この組のあとの版)"""
    p = ROOT / 'jetformer' / name
    if name in LATER_FIX9_PRIORITY and sha(p) == LATER_FIX9_PRIORITY[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix9_priority_before' / name
    if name in LATER_FIX8_SAFETY_DOC and sha(p) == LATER_FIX8_SAFETY_DOC[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / name
    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix7_safety_r2_before' / name
    assert sha(p) == FIX7_R1_AFTER[name], name
    return p
DT = R2.DT
FDT = 1 / 30.0
XF = R2.XF
MODEL = (C, N - 120.0)
REV, GAP = N + 300.0, N + A.GAP_OFFSET_US
RETRY = N + 300.0 + A.Recovery.RETRY_EXTRA_US
ESCAPE = N - A.Recovery.ESCAPE_US
STATE = R2.STATE + ('t_check', 'est_s')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_BEFORE = {}


def load_before():
    """直す前(fix6 r2 のあと)の assist.py・drive.py(写し)をモジュールにする(sha256 を照らしてから)"""
    if _BEFORE:
        return _BEFORE['a'], _BEFORE['d']
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sums == SHA_BEFORE
    for name, h in sums.items():
        assert sha(FIX / name) == h, name
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix7s', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix7s', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    r = amod.Recovery((C, N))
    assert hasattr(r, 'est_s') and not hasattr(r, 'seen_rev_s')      # 写しは fix6 r2 のあと・直す前
    _BEFORE.update(a=amod, d=mod)
    return amod, mod


# ================================================================ 0. 写し・既定値・変えた所
def test_before_copy_defaults_and_changed_parts():
    """写しの sha256(= fix6 r2 のあとの版)。既定値(DEFAULT_PARAMS・PARAM_LIMITS・RESET_ON_BOOT・HOLD_VEH_*)・Recovery・EscTrack・
    EscReverse の定数・モジュールの定数は同じ。変えたのは assist.py の Recovery.reset・_start・_guard・_after_hold・step・
    Assist.step、drive.py の Driver._loop だけ。足したのは assist.py の EscMayReverse(EscTrack の 2 つの定数を 0 にしただけ)だけ
    (消した関数や文は無い。AST を名前で照らす)。10/27 fix7 safety の 7-2 回目が変えたあとは、その直す前の写し(= この組のあとの版)
    で照らす(_fix7r1_version。モジュールの定数 SURE_BACK_S を足したのは tests/test_rec_fix7_safety_r2.py が見る)"""
    amod, before = load_before()
    assert drive.DEFAULT_PARAMS == before.DEFAULT_PARAMS and drive.PARAM_LIMITS == before.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == before.RESET_ON_BOOT
    assert drive.HOLD_VEH_GAP_S == before.HOLD_VEH_GAP_S and drive.HOLD_VEH_FAST_ERPM == before.HOLD_VEH_FAST_ERPM
    for k in ('rec_back_guard', 'rec_hold_keep', 'launch_boost', 'launch_no_kick'):
        assert drive.DEFAULT_PARAMS[k] == 0 and k in drive.RESET_ON_BOOT
    for cls in ('Recovery', 'EscTrack', 'EscReverse'):
        assert {k: v for k, v in vars(getattr(A, cls)).items() if k.isupper()} == \
            {k: v for k, v in vars(getattr(amod, cls)).items() if k.isupper()}, cls
    for k in ('BACKING_ERPM', 'CHECK_S', 'STOPPED_ERPM', 'GAP_OFFSET_US', 'DEFAULT_REV_US', 'DEFAULT_REV_S',
              'HOLD_EST_MIN_LEFT_S'):
        assert getattr(A, k) == getattr(amod, k)
    a0, a1 = T6._defs(FIX / 'assist.py'), T6._defs(_fix7r1_version('assist.py'))
    added = set(a1) - set(a0)
    assert set(a0) <= set(a1) and {k for k in added if not k.startswith('EscMayReverse.Expr(')} == {
        'class:EscMayReverse', 'EscMayReverse.READY_NEUTRAL_S', 'EscMayReverse.DELAY_S'} and len(added) == 4
    assert sorted(k for k in a0 if a0[k] != a1[k]) == [
        'Assist.step', 'Recovery._after_hold', 'Recovery._guard', 'Recovery._start', 'Recovery.reset', 'Recovery.step']
    assert issubclass(A.EscMayReverse, A.EscTrack) and {k for k in vars(A.EscMayReverse) if not k.startswith('__')} == {
        'READY_NEUTRAL_S', 'DELAY_S'} and A.EscMayReverse.READY_NEUTRAL_S == A.EscMayReverse.DELAY_S == 0.0
    d0, d1 = T6._defs(FIX / 'drive.py'), T6._defs(_fix7r1_version('drive.py'))
    assert set(d0) == set(d1) and sorted(k for k in d0 if d0[k] != d1[k]) == ['Driver._loop']
    for doc in (A.__doc__, drive.__doc__):
        assert '反証 7-1 回目の直し' in doc
    assert '反証 7-1 回目の直し' not in amod.__doc__ and '反証 7-1 回目の直し' not in before.__doc__


# ================================================================ 1. Recovery(単体)
def _brake(mod, erpm_wait=None, **kw):
    """前へ走って armed → 前の壁で止まって 'brake' を始めた Recovery(見張り ON)と、次のフレームの時刻。
    erpm_wait を渡すと 'brake' のあと前へ転がっている('wait')ところまで。kw = step に足す引数(esc_rev など)"""
    r = mod.Recovery((C, N))
    t = 100.0
    for _ in range(40):
        r.step(MODEL, -5000.0, t, back_guard=True, **kw)
        t += FDT
    while r.phase != 'brake':
        r.step(MODEL, 0.0, t, back_guard=True, **kw)
        t += FDT
        assert t < 110.0
    if erpm_wait is not None:
        while r.phase != 'wait':
            r.step(MODEL, erpm_wait, t, back_guard=True, **kw)
            t += FDT
    return r, t


def _feed(r, seq, t, **kw):
    """seq = [eRPM か (eRPM, step の追加の引数)] を FDT ごとに渡す → [(出す値, 種類, 段, 出来事)]、次の時刻"""
    out = []
    for x in seq:
        e, k2 = (x, {}) if not isinstance(x, tuple) else x
        o, k = r.step(MODEL, float(e), t, **dict(dict(back_guard=True), **dict(kw, **k2)))
        out.append((o, k, r.phase, r.guard_event))
        r.guard_event = None
        t += FDT
    return out, t


def _thr(rows):
    return [None if o is None else o[1] for o, k, ph, ev in rows]


def _events(rows):
    return [ev[0] for o, k, ph, ev in rows if ev is not None]


# --- s4-a: この手順でまだ ESC が後退で走らせていない 'brake'・'wait'・'gap' の跳ね
@pytest.mark.parametrize('e', [301.0, 313.0, 1322.0])
@pytest.mark.parametrize('k', [0, 1, 3, 6])
@pytest.mark.parametrize('where', ['brake', 'wait'])
def test_s4a_unit_spike_ignored_when_esc_not_reversing(where, k, e):
    """'brake'(k フレーム目)・'wait' の 1 サンプルの跳ね e のあと、ESC はブレーキのまま(eRPM 0)。esc_rev = False の今は、
    見張り OFF(back_guard = False)と同じ値を出す(brake → wait → gap → reverse。出来事 back_no_esc が 1 回・back_t は None)。
    直す前の写しは back_early で 'gap' を飛ばし、'reverse' の後退側 1796 µs を下がらないまま MAX_REV_TOTAL_S 書いてから escape
    (指摘の再現)"""
    amod, _ = load_before()
    ew = -2000.0 if where == 'wait' else None
    pre = [ew] * k if where == 'wait' else [0.0] * k
    seq = pre + [e] + [0.0] * 60
    r, t = _brake(A, ew, esc_rev=False)
    now, _ = _feed(r, seq, t, esc_rev=False)
    r_off, t_off = _brake(A, ew)
    off, _ = _feed(r_off, seq, t_off, back_guard=False)
    r_b, t_b = _brake(amod, ew)
    bef, _ = _feed(r_b, seq, t_b)
    assert t == t_off == t_b
    assert _thr(now) == _thr(off)
    assert [ph for _, _, ph, _ in now] == [ph for _, _, ph, _ in off]
    assert _events(now) == ['back_no_esc'] and r.back_t is None and r.backed_s == 0.0
    assert 'gap' in [ph for _, _, ph, _ in now] and GAP in _thr(now)
    # 直す前の写し: 跳ねのフレームで 'reverse'、中立(gap)を一度も出さず、後退側を 1.2 秒書いて escape
    i = len(pre)
    assert bef[i][2] == 'reverse' and _events(bef)[:1] == ['back_early']
    assert GAP not in _thr(bef)
    rev_n = sum(1 for th in _thr(bef)[i:] if th == REV)
    assert rev_n * FDT >= A.Recovery.MAX_REV_TOTAL_S - FDT
    assert ESCAPE in _thr(bef)


def _gap(mod, kw):
    """'gap' に入ったばかりの Recovery(_brake のあと eRPM 0 を 8 フレーム。'brake' 0.25 秒 → 'wait' → 'gap')"""
    r, t = _brake(mod, **{k: v for k, v in kw.items() if k == 'esc_rev'})
    rows, t = _feed(r, [0.0] * 8, t, **kw)
    assert r.phase == 'gap' and rows[-2][2] == 'brake'
    return r, t, rows


@pytest.mark.parametrize('e', [313.0, 1322.0])
@pytest.mark.parametrize('k', [0, 1, 3])
def test_s4a_unit_gap_spike_ignored_before_esc_ran(k, e):
    """この手順でまだ ESC が後退で走らせていない(esc_rev がずっと False)'gap' の k フレーム目の跳ね e: 今は惰性に数えない
    (backed_s 0・back_t なし・出来事 back_no_esc)。出す値は見張り OFF と同じ。直す前の写しは数えて(backed_s > 0)、ESC が後退に
    入らないとき CHECK_S のやり直しが効かず、後退側を MAX_REV_TOTAL_S 書いて escape"""
    amod, _ = load_before()
    seq = [0.0] * k + [e] + [0.0] * 60
    r, t, _ = _gap(A, {'esc_rev': False})
    now, _ = _feed(r, seq, t, esc_rev=False)
    r_off, t_off, _ = _gap(A, {'back_guard': False})
    off, _ = _feed(r_off, seq, t_off, back_guard=False)
    r_b, t_b, _ = _gap(amod, {})
    bef, _ = _feed(r_b, seq, t_b)
    assert now[k][2] == 'gap' and _thr(now) == _thr(off) and RETRY in _thr(now)
    assert _events(now) == ['back_no_esc'] and r.back_t is None
    assert RETRY not in _thr(bef) and ESCAPE in _thr(bef)


def test_s4a_unit_gap_counts_after_esc_ran():
    """この手順で一度 ESC が後退で走らせていたかもしれない(esc_rev が True になった)あとは、'gap'(やり直し)の後ろへの動きを
    今まで通り惰性に数える(esc_rev = None と出す値・状態が全部同じ)"""
    def go(kwf):
        r, t = _brake(A)
        seq = [(0.0, kwf(i)) for i in range(8 + 5 + 12)]           # 'brake' → 'gap' → 'reverse'(下がらない)→ やり直しの 'gap'
        rows, t = _feed(r, seq, t)
        assert r.phase == 'gap' and r.retries == 1
        rows2, t = _feed(r, [(1200.0, kwf(99))] * 3, t)
        return r, rows + rows2

    def ev(i):
        return {'esc_rev': True if 15 <= i < 25 else False}      # 'reverse' の途中から(ESC が後退で走らせていたかもしれない)
    r1, a = go(ev)
    r2, b = go(lambda i: {})
    assert _thr(a) == _thr(b) and r1.backed_s == r2.backed_s == pytest.approx(3 * FDT)
    assert r1.esc_ran and 'back_no_esc' not in _events(a)


@pytest.mark.parametrize('seed', range(4))
def test_s4a_unit_esc_rev_true_or_none_identical(seed):
    """esc_rev = True と None(分からない)は、どんな入力でも出す値・段・状態が全部同じ(s4-a は False のときだけ効く)"""
    rng = random.Random(seed)
    ra, rb = A.Recovery((C, N)), A.Recovery((C, N))
    t = 100.0
    erpm = -4000.0
    n_guarded = 0
    for i in range(2500):
        x = rng.random()
        if x < 0.1:
            erpm = rng.choice([-5000.0, -2000.0, -200.0, 0.0, 0.0, 301.0, 1322.0, 2000.0])
        elif x < 0.5:
            erpm += rng.uniform(-300, 300)
        t += rng.choice([FDT] * 20 + [0.1])
        kw = dict(back_guard=True, impact_g=rng.choice([0.0] * 30 + [3.0]))
        if rng.random() < 0.05:
            kw.update(hold_since=t - 0.05, hold_back_s=rng.choice([0.0, 0.02, 0.1]))
        a = ra.step(MODEL, erpm, t, esc_rev=True, **kw)
        b = rb.step(MODEL, erpm, t, esc_rev=None, **kw)
        assert a == b
        for k in STATE + ('seen_rev_s', 'skip_gap', 'noesc_noted'):
            assert getattr(ra, k) == getattr(rb, k), k
        n_guarded += ra.phase in ('brake', 'wait')
    assert n_guarded > 10                       # 'brake'・'wait' を通っている(試験が空振りしていない)


@pytest.mark.parametrize('where', ['brake', 'wait'])
def test_s4a_unit_hold_backing_ignored_when_esc_not_reversing(where):
    """残した中立止め(0.2 秒)のあとの最初のフレームで、その間に後ろへ動いていたサンプル 0.02 秒(跳ね)。esc_rev = False の今は
    下がった時間に数えず、back_hold で 'gap' を飛ばさない(出来事 back_no_esc)。esc_rev = None は直す前の写しと同じ back_hold →
    'reverse'"""
    amod, _ = load_before()
    ew = -2000.0 if where == 'wait' else None
    res = {}
    for nm, mod, kw in (('now_false', A, {'esc_rev': False}), ('now_none', A, {}), ('bef', amod, {})):
        r, t = _brake(mod, ew, **kw)
        t1 = t + 0.2
        r.step(MODEL, 0.0, t1, back_guard=True, hold_since=t + 0.03, hold_back_s=0.02, **kw)
        res[nm] = (r.phase, r.backed_s, r.guard_event and r.guard_event[0], getattr(r, 'skip_gap', None))
    assert res['now_false'][0] in ('brake', 'wait', 'gap') and res['now_false'][1] == 0.0
    assert res['now_false'][2] == 'back_no_esc'
    assert res['bef'][:3] == ('reverse', 0.02, 'back_hold')
    assert res['now_none'][:3] == res['bef'][:3] and res['now_none'][3] is True


def test_s4a_unit_no_rear_impact_from_ignored_spike():
    """'brake' の跳ね(esc_rev = False)の次のフレームの衝撃 3G: 今は後ろの当たりにしない(back_t が無い。段は 'brake' のまま)。
    esc_rev = None は直す前と同じく rear_impact で escape"""
    amod, _ = load_before()
    for mod, kw, want in ((A, {'esc_rev': False}, 'brake'), (A, {}, 'escape'), (amod, {}, 'escape')):
        r, t = _brake(mod, **kw)
        rows, t = _feed(r, [1322.0, (0.0, {'impact_g': 3.0})], t, **kw)
        assert rows[-1][2] == want, (mod, kw)
        if want == 'escape':
            assert rows[-1][3][0] == 'rear_impact'


# --- s4-b: 'reverse' のやり直しは 'reverse' の中で見た後退で決める
def test_s4b_unit_back_early_without_backing_restarts_from_gap():
    """esc_rev を渡さない(None)'brake' の跳ねで back_early → 'reverse'。そのあと下がらない(eRPM 0)。今は CHECK_S で 'gap'(GAP_S。
    出来事 early_none・やり直しの数 0)へ戻し、続く 'reverse' は 300 µs(+50 µs にしない)。下がり始めたあとは rev_s で 'coast'。
    直す前の写しは後退側を MAX_REV_TOTAL_S 書いて escape"""
    amod, _ = load_before()
    seq = [0.0, 1322.0] + [0.0] * 11
    r, t = _brake(A)
    rows, t = _feed(r, seq, t)
    phases = [ph for _, _, ph, _ in rows]
    assert phases[1:12] == ['reverse'] * 11 and phases[12] == 'gap'     # 跳ねのフレームから CHECK_S(11 フレーム目で > 0.35 秒)
    assert _thr(rows)[1:12] == [REV] * 11 and _thr(rows)[12] == GAP
    assert _events(rows) == ['back_early', 'early_none'] and r.retries == 0
    assert r.backed_s == 0.0 and r.est_s == 0.0 and r.seen_rev_s == 0.0 and not r.skip_gap
    rows2, t = _feed(r, [0.0] * 5 + [1400.0] * 20, t)
    th2 = _thr(rows2)
    assert th2[:4] == [GAP] * 4 and th2[4] == REV                       # 'gap' は GAP_S(0.15 秒)。続く 'reverse' は 300 µs
    assert RETRY not in th2 + _thr(rows) and r.retries == 0
    assert rows2[-1][2] in ('coast', 'escape', None)
    r_b, t_b = _brake(amod)
    bef, _ = _feed(r_b, seq + [0.0] * 40, t_b)
    tb = _thr(bef)
    assert GAP not in tb and ESCAPE in tb
    assert sum(1 for th in tb if th == REV) * FDT >= A.Recovery.MAX_REV_TOTAL_S - FDT


def test_s4b_unit_back_early_with_backing_identical():
    """ESC が本当に後退を待っていた(10/23 の形): 'brake' で下がり始め、そのまま下がる。出す値・段・状態は直す前の写しと全部同じ"""
    amod, _ = load_before()
    seq = [0.0, 0.0, 400.0, 900.0] + [1400.0] * 30 + [600.0, 0.0, 0.0] + [0.0] * 30
    r, t = _brake(A)
    rb, tb = _brake(amod)
    for x in seq:
        a = r.step(MODEL, x, t, back_guard=True)
        b = rb.step(MODEL, x, tb, back_guard=True)
        assert a == b
        for k in STATE:
            assert getattr(r, k) == getattr(rb, k), k
        t += FDT
        tb += FDT
    assert r.count == 1 and r.phase is None


@pytest.mark.parametrize('where', ['gap', 'entry'])
@pytest.mark.parametrize('e', [313.0, 1322.0])
def test_s4b_unit_gap_spike_then_no_backing_retries(where, e):
    """'gap' の 1 サンプルの跳ね(見張りの惰性の数え)か、'gap' → 'reverse' に変わるフレームの跳ねのあと、ESC が後退に入らない
    (中立 0.15 秒では下がれない ESC)。今は CHECK_S でやり直す(+50 µs・RETRY_GAP_S。見張り OFF と同じ)。直す前の写しは backed_s > 0 で
    やり直さず、後退側を MAX_REV_TOTAL_S 書いて escape"""
    amod, _ = load_before()
    res = {}
    for nm, mod, kw in (('now', A, {}), ('off', A, {'back_guard': False}), ('bef', amod, {})):
        r, t = _brake(mod)
        rows, t = _feed(r, [0.0] * 8, t, **kw)                          # 'brake' 0.25 秒 → 'wait' → 'gap'
        assert r.phase == 'gap'
        j = 2 if where == 'gap' else 4                  # 'gap' の 3 フレーム目(0.1 秒)か、'reverse' に変わるフレーム(0.167 秒)
        rows2, t = _feed(r, [0.0] * j + [e] + [0.0] * 50, t, **kw)
        assert rows2[j][2] == ('gap' if where == 'gap' else 'reverse') and rows2[j - 1][2] == 'gap'
        res[nm] = rows + rows2
    assert RETRY in _thr(res['now'])
    assert RETRY not in _thr(res['bef']) and ESCAPE in _thr(res['bef'])
    if where == 'gap':
        assert _thr(res['now']) == _thr(res['off'])                    # 見張り OFF と同じ
    else:
        assert RETRY not in _thr(res['off'])        # OFF も入ったフレームの跳ねを数えてやり直さない(OFF は変えない。今の ON はやり直す)


def test_s4b_unit_seen_backing_in_reverse_no_retry():
    """'reverse' の 2 フレーム目より後で後ろへ動いたサンプルがあれば、今までと同じくやり直さない(直す前の写しと全部同じ)"""
    amod, _ = load_before()
    seq = [0.0] * 8 + [0.0] * 5 + [0.0, 0.0, 400.0] + [0.0] * 40
    rows = {}
    for nm, mod in (('now', A), ('bef', amod)):
        r, t = _brake(mod)
        rows[nm], _ = _feed(r, seq, t)
    assert _thr(rows['now']) == _thr(rows['bef']) and RETRY not in _thr(rows['now'])


def test_s4b_unit_hold_seen_in_reverse_counts_as_before():
    """'reverse' の途中の残した中立止めの間に後ろへ動いていたサンプル: 今までと同じく見た後退(やり直さない。直す前の写しと同じ)"""
    amod, _ = load_before()
    out = {}
    for nm, mod in (('now', A), ('bef', amod)):
        r, t = _brake(mod)
        _feed(r, [0.0] * 14, t)
        assert r.phase == 'reverse'
        t += 14 * FDT
        o = r.step(MODEL, 0.0, t + 0.15, back_guard=True, hold_since=t + 0.02, hold_back_s=0.04)
        rows, _ = _feed(r, [0.0] * 30, t + 0.15 + FDT)
        out[nm] = (o, _thr(rows), r.backed_s)
    assert out['now'] == out['bef'] and RETRY not in out['now'][1]


def _fuzz(seed, n=3000, holds=True):
    """でたらめな入力(走る・止まる・跳ね・後ろへ・衝撃・中立止め・時刻の飛び)"""
    rng = random.Random(seed)
    t, erpm, out = 100.0, -4000.0, []
    for i in range(n):
        x = rng.random()
        if x < 0.06:
            erpm = rng.choice([-5000.0, -2000.0, -400.0, 0.0, 0.0, 0.0, 800.0, 1500.0, 3000.0])
        elif x < 0.4:
            erpm += rng.uniform(-300, 300)
        e = erpm
        if rng.random() < 0.04:
            e = rng.choice([301.0, 313.0, 1115.0, 1322.0, 1687.0])        # 1 サンプルの跳ね
        t += rng.choice([FDT] * 25 + [0.1, 0.25])
        kw = dict(impact_g=rng.choice([0.0] * 40 + [3.0]), rev_us=rng.choice([250.0, 300.0]), rev_s=rng.choice([0.3, 0.4]))
        if holds and rng.random() < 0.04:
            kw.update(hold_since=t - rng.uniform(0.02, 0.12), hold_back_s=rng.choice([0.0, 0.0, 0.02, 0.06]),
                      hold_rev_s=rng.choice([0.0, 0.0, 0.05]), hold_g=rng.choice([0.0] * 10 + [3.0]))
        out.append(((C + rng.uniform(-300, 300), N + rng.choice([-150.0, -60.0, 0.0, 20.0])), e, t, kw))
    return out


@pytest.mark.parametrize('holds', [False, True])
@pytest.mark.parametrize('seed', range(5))
def test_recovery_off_bit_identical(seed, holds):
    """見張り OFF(back_guard = False)の Recovery は、どんな入力でも(中立止めのあとの引数があっても)直す前の写しと出す値・状態が
    全部同じ"""
    amod, _ = load_before()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    n_phase = 0
    for model, e, t, kw in _fuzz(seed, holds=holds):
        a = r0.step(model, e, t, back_guard=False, **kw)
        b = r1.step(model, e, t, back_guard=False, **kw)
        assert a == b
        for k in STATE:
            assert getattr(r0, k) == getattr(r1, k), k
        n_phase += r1.phase is not None
        r0.guard_event = r1.guard_event = None
    assert n_phase > 100                        # 手順の途中を通っている(試験が空振りしていない)


_FUZZ_SEEN = {'diverge': 0, 'runs': 0}


@pytest.mark.parametrize('holds', [False, True])
@pytest.mark.parametrize('seed', range(8))
def test_recovery_on_diverges_only_by_fix7(seed, holds):
    """見張り ON・esc_rev を渡さない(None。s4-a は効かない)Recovery を直す前の写しと並べて回す。出す値・状態が初めて違うのは、
    今が 'reverse' の CHECK_S で 'gap' へ移った(early_none か、直す前が backed_s > 0 でやり直さなかったやり直し)フレームだけ。
    そのフレームまでは全部同じ。違ったら、そこから両方を作り直して続ける"""
    amod, _ = load_before()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    steps = _fuzz(100 + seed, holds=holds)
    for model, e, t, kw in steps:
        p_before = r1.phase
        backed_before = r1.backed_s
        est_before = r1.est_s
        a = r0.step(model, e, t, back_guard=True, **kw)
        b = r1.step(model, e, t, back_guard=True, **kw)
        same = a == b and all(getattr(r0, k) == getattr(r1, k) for k in STATE)
        if not same:
            # 違うのは: 今が 'reverse' から 'gap' へ(やり直し・early_none)、直す前は 'reverse' のまま・'coast'・'escape'
            assert p_before == 'reverse' and r1.phase == 'gap', (t, p_before, r1.phase, r0.phase)
            assert b[1] == A.ASSIST_RECOVERY and b[0][1] == N + A.GAP_OFFSET_US
            assert r1.seen_rev_s == 0.0 and r1.backed_s == 0.0
            assert backed_before - est_before > 0.0          # 直す前は「見た後退がある」としてやり直さなかった
            assert r0.phase in ('reverse', 'coast', 'escape', None)
            _FUZZ_SEEN['diverge'] += 1
            r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))     # 作り直して続ける(armed から)
        r0.guard_event = r1.guard_event = None
    _FUZZ_SEEN['runs'] += 1


def test_recovery_on_fuzz_saw_divergence():
    """上のでたらめな入力で、直しの道(違い)を通っている(試験が空振りしていない)"""
    if _FUZZ_SEEN['runs'] < 16:
        pytest.skip('上の試験と一緒に流したときだけ見る')
    assert _FUZZ_SEEN['diverge'] >= 5, _FUZZ_SEEN


# --- EscMayReverse(drive が esc_rev に使う)
def _gate(seq, cls=A.EscMayReverse):
    """seq = [(書いた時刻 s, スロットル µs)] を書いた EscMayReverse(か EscTrack)"""
    g = cls(N)
    for t, thr in seq:
        g.write(int(t * 1e9), thr)
    return g


def test_esc_may_reverse_rules():
    """前進 → 後退側(ブレーキ)の間は下がれない(EscTrack と同じ)。ブレーキのあと中立を 1 回でも書いたら、続く後退側は書いたときから
    後退とみる(EscTrack は中立 0.08 秒・遅れ 0.06 秒)。前進 → 中立の帯 → 後退側(ブレーキを挟まない)は下がれない。分からない
    (走行の始め・人側)は下がれる"""
    fwd_brake = [(0.0, N - 150.0), (0.1, N + 300.0), (0.133, N + 300.0), (0.4, N + 300.0)]
    for cls in (A.EscMayReverse, A.EscTrack):
        assert _gate(fwd_brake, cls).rev_s(0.0, 1.0e9) == 0.0
        assert _gate([(0.0, N - 150.0), (0.1, N + 20.0), (1.6, N + 300.0)], cls).rev_s(0.0, 3e9) == 0.0      # 中立の帯 1.5 秒でも
    short = [(0.0, N - 150.0), (0.1, N + 300.0), (0.2, N + 10.0), (0.233, N + 300.0)]          # ブレーキ → 中立 0.033 秒 → 後退側
    assert _gate(short).rev_s(0.0, 0.3e9) == pytest.approx(0.3 - 0.233)
    assert _gate(short, A.EscTrack).rev_s(0.0, 0.3e9) == 0.0                                     # EscTrack は 0.08 秒の中立が要る
    long_n = [(0.0, N - 150.0), (0.1, N + 300.0), (0.2, N + 10.0), (0.35, N + 300.0)]
    assert _gate(long_n).rev_s(0.35e9, 0.36e9) == pytest.approx(0.01)                           # 書いたときから
    assert _gate(long_n, A.EscTrack).rev_s(0.35e9, 0.41e9) == pytest.approx(0.0)                # EscTrack は 0.06 秒あとから
    assert _gate(long_n, A.EscTrack).rev_s(0.35e9, 0.45e9) == pytest.approx(0.04)
    assert _gate([(0.0, N + 300.0)]).rev_s(0.0, 0.1e9) == pytest.approx(0.1)                     # 分からない(始め)は下がれる


def open_scene(brake_before):
    """開いたループの場面(tests/test_recovery_back_guard.recovery_scene の形): 3 m/s → 2.0 秒で止まる → リカバリー → 2.95 秒から
    後ろへ 1 m/s → 3.15 秒に後ろの衝撃 6G。brake_before = 止まる前にブレーキ側(中立 + 200、1.8〜2.0 秒。ESC が後退を待つ 10/23 の形)。
    False は前進 → 中立の帯のまま(このESCでは 'brake' の段で後退しない。前は試験がこの形で後ろへ動かしていた)"""
    def v(s):
        if s < 2.0:
            return 3.0
        return -1.0 if 2.95 <= s < 3.15 else 0.0

    def pred(s):
        if s < 1.8:
            return (C, N - 150.0)
        return (C, N + 200.0) if (brake_before and s < 2.0) else (C, N + 20.0)
    return build_open(5.5, sel=True, v=v, pred=pred, ax=lambda s: 60.0 if 3.15 <= s < 3.17 else 0.0)


def _open_events(rec):
    return [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']


def _rev_after(rec, frames, s0):
    return sum(1 for t, r in rec.rows if not r['hold'] and r['out'][1] > N + 40.0 and (t - frames[0]['cam_t']) / 1e9 >= s0)


@pytest.mark.parametrize('keep', [0, 1])
def test_open_scene_brake_before_guard_as_before(monkeypatch, tmp_path, keep):
    """10/23 の形(止まる前にブレーキ側): 今の ON は直す前の写しの ON とビット単位で同じ(back_early → rear_impact・当たったあと
    後ろへ押さない)"""
    amod, before = load_before()
    frames, imu = open_scene(True)
    p = dict(PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=keep)
    _, pca1, rec1 = run_open(drive, monkeypatch, tmp_path, p, frames, imu)
    _, pca0, rec0 = run_open(before, monkeypatch, tmp_path, p, frames, imu)
    assert pca1.pairs == pca0.pairs and pca1.single == pca0.single and rec1.events == rec0.events
    assert _open_events(rec1)[:2] == ['back_early', 'rear_impact'] and _rev_after(rec1, frames, 3.2) == 0


def test_open_scene_forward_neutral_guard_like_off_until_reverse(monkeypatch, tmp_path):
    """前進 → 中立の帯 → 'brake' のまま後ろへ動かす場面(このESCでは起きない。人の 29 回すべて・中立 1.5 秒でも後退しなかった):
    今の ON は後ろへの動きを跳ねとして使わない(back_no_esc。back_early・rear_impact なし)。'reverse' の段に入るまで PCA9685 に書く値は
    OFF とビット単位で同じで、当たったあとに後退側を書いたフレームは OFF より多くない。直す前の写しの ON は back_early → rear_impact"""
    amod, before = load_before()
    frames, imu = open_scene(False)
    _, pca_on, rec_on = run_open(drive, monkeypatch, tmp_path, dict(PARAMS, recovery=1, rec_back_guard=1), frames, imu)
    _, pca_off, rec_off = run_open(drive, monkeypatch, tmp_path, dict(PARAMS, recovery=1, rec_back_guard=0), frames, imu)
    _, pca_b, rec_b = run_open(before, monkeypatch, tmp_path, dict(PARAMS, recovery=1, rec_back_guard=1), frames, imu)
    assert _open_events(rec_b)[:2] == ['back_early', 'rear_impact']                 # 直す前(前の試験の形)
    ev = _open_events(rec_on)
    assert ev[:1] == ['back_no_esc'] and 'back_early' not in ev and 'rear_impact' not in ev
    out_on, out_off = [r['out'] for t, r in rec_on.rows], [r['out'] for t, r in rec_off.rows]
    i_gap = next(i for i, o in enumerate(out_off) if o[1] == GAP)                  # OFF の最初の 'gap'(中立 + 10)
    i_end = next(i for i in range(i_gap, len(out_off)) if out_off[i][1] != GAP)     # その 'gap' が終わって 'reverse' の最初
    assert out_on[:i_end] == out_off[:i_end]
    assert _rev_after(rec_on, frames, 3.2) <= _rev_after(rec_off, frames, 3.2)


# ================================================================ 2. 走行ループ(閉じたループ。本物の drive._loop)
class GCar(R2.Car):
    """R2.Car と同じ。scene['esc'] = (neu, rdelay) なら中立で下がれる秒・後退側から下がり始める秒だけ変える(1e-9 の余裕)。
    scene['spring'] = (v, dur, push_s): 前の壁を前進で push_s 秒以上押していたあと前進でない値に変わったら、dur 秒だけ v [m/s] で
    壁から離れる(バンパー・タイヤのたわみの戻り。ESC の状態は R2.Car のまま)"""

    def step(self, t, thr):
        s = self.s
        if s.get('spring') is not None:
            return self._spring_step(t, thr)
        if s.get('esc') is None:
            return super().step(t, thr)
        neu, rdelay = s['esc']
        fwd, rev = thr < N - 20.0, thr > N + 40.0
        v0 = self.v
        if fwd:
            self.ready = self.braked = self.rev_on = False
            self.rev_wait = self.neu_t = 0.0
            if self.v < 0:
                self.v = min(0.0, self.v + 8.0 * DT)
            else:
                vt = min((N - thr) / 80.0 * 1.2, 6.0)
                self.v += float(np.clip((vt - self.v) / 0.3, -6.0, 6.0)) * DT
        elif rev:
            self.neu_t = 0.0
            if self.rev_on:
                a = (thr - N) / 300.0 * 5.0
                self.v = max(self.v - a * DT, -(thr - N) / 300.0 * 3.0)
            elif self.ready and self.v <= 0.075:
                self.rev_wait += DT
                if self.rev_wait >= rdelay - 1e-9:
                    self.rev_on = True
            else:
                self.braked = True
                self.v = max(0.0, self.v - 6.0 * DT) if self.v > 0 else min(0.0, self.v + 6.0 * DT)
        else:
            self.rev_wait = 0.0
            self.neu_t += DT
            if self.braked and self.neu_t >= neu - 1e-9:
                self.ready, self.braked = True, False
            if self.rev_on:
                self.rev_on = False
                self.ready = True
            self.v = max(0.0, self.v - 1.5 * DT) if self.v > 0 else min(0.0, self.v + 1.5 * DT)
        return self._walls(t, v0)

    def _walls(self, t, v0):
        s = self.s
        nx = self.x + self.v * DT
        prev_con = self.contact
        self.contact = None
        if nx >= s['xf']:
            if self.v > 0.05 and prev_con != 'front':
                g = min(2.0 + 2.5 * self.v, 9.0)
                self.hits.append((round(t, 3), 'front', round(self.v, 2), round(g, 2)))
                self.spike = [g, g]
            nx, self.v = s['xf'], 0.0
            self.contact = 'front'
        if s.get('xr') is not None:
            if not self.rear_active and nx > s['xr'] + 0.1:
                self.rear_active = True
            if self.rear_active and nx <= s['xr']:
                if self.v < -0.05 and prev_con != 'rear':
                    g = min(2.0 + 2.5 * -self.v, 9.0)
                    self.hits.append((round(t, 3), 'rear', round(self.v, 2), round(g, 2)))
                    self.spike = [g, g]
                nx, self.v = s['xr'], 0.0
                self.contact = 'rear'
        self.x = nx
        acc = (self.v - v0) / DT
        g = self.spike.pop(0) if self.spike else 0.0
        return acc, g

    def _spring_step(self, t, thr):
        vs, dur, push_s = self.s['spring']
        if not hasattr(self, '_push'):
            self._push, self._spr_until = 0.0, -1.0
        fwd = thr < N - 20.0
        if self.contact == 'front' and fwd:
            self._push += DT
        elif not fwd and self._push >= push_s and self.contact == 'front':
            self._spr_until = t + dur
            self._push = 0.0
        elif not fwd:
            self._push = 0.0
        if t <= self._spr_until + 1e-9:
            v_save = self.v
            r = R2.Car.step(self, t, thr)
            self.v = -vs
            self.x = min(self.x, self.s['xf']) - vs * DT
            self.contact = None
            return (self.v - v_save) / DT, r[1]
        return R2.Car.step(self, t, thr)


class GSim(F5.SpikeSim):
    """R2.Sim(と F5.SpikeSim の止まらない衝撃 scene['spikes'])と同じ。scene['glitch'] = [(t, eRPM)]: その時刻に来た車両の値の
    サンプルの eRPM だけを差し替える(車は動かない)。scene['veh_dt'] = 車両の値の間隔(既定 0.02。本物は 100 Hz = 0.01)。車は GCar"""

    def __init__(self, d, s, bufs, cam_slot, box, pca):
        super().__init__(d, s, bufs, cam_slot, box, pca)
        self.car = GCar(s)

    def _advance(self, t_end):
        if not self.s.get('glitch') and not self.s.get('veh_dt'):
            return super()._advance(t_end)
        vdt = float(self.s.get('veh_dt') or 0.02)
        gl = self.s.get('glitch') or []
        sp = self.s.get('spikes') or []
        while self.t + DT <= t_end + 1e-9:
            nt = round(self.t + DT, 6)
            for ts, g in sp:
                if abs(nt - ts) < DT / 2 and not self.car.spike:
                    self.car.spike = [g, g]
            self.t = nt
            acc, g = self.car.step(self.t, self.pca.thr)
            t_ns = T0 + int(round(self.t * NS))
            self.imu_buf.append((t_ns, np.array([0.0, acc + g * 9.80665, 9.8, 0.0, 0.0, 0.0], np.float32)))
            if self.t >= self.next_veh - 1e-9:
                self.next_veh += vdt
                if not any(a <= self.t < b for a, b in self.s.get('veh_skip', [])):
                    sl = self.sel(self.t)
                    e = int(round(-self.car.v / R2.KV))
                    for ts, ge in gl:
                        if abs(self.t - ts) < DT / 2:
                            e = int(ge)
                    self.veh_buf.append((t_ns, {'pwm': [1441, 1496, 2000 if sl else 1000], 'valid': [True, True, True],
                                                'select': sl, 'moving': abs(self.car.v) > 0.01, 'erpm': e}))
            if self.t >= self.next_tof - 1e-9:
                self.next_tof += 1.0 / 30
                dist = self.s['xf'] - self.car.x
                self.tof_buf.append((t_ns, tof_near(int(dist * 1000)) if 0.05 < dist < 1.95 else TOF_FREE))
            self.log.append((self.t, self.car.x, self.car.v, self.pca.thr))
        self.box['ns'] = T0 + int(round(self.t * NS))
        self.box['m'] = self.t + 100.0
        self.d.heartbeat_t = self.box['m']


def run(mod, monkeypatch, tmp_path, params, scene, force=None):
    """閉じたループ。force = Recovery.step に上書きして渡す引数(例 {'esc_rev': None} = drive の EscTrack を使わない形)"""
    monkeypatch.setattr(R2, 'Sim', GSim)
    if force:
        amod = mod.assist_mod
        orig = amod.Recovery.step

        def step(self, *a, **kw):
            return orig(self, *a, **dict(kw, **force))
        monkeypatch.setattr(amod.Recovery, 'step', step)
    try:
        return R2.run_cl(mod, monkeypatch, tmp_path, params, scene)
    finally:
        if force:
            monkeypatch.setattr(mod.assist_mod.Recovery, 'step', orig)


def P(guard=1, keep=1, **kw):
    """回 1〜4 の形の基本(start_kick 1・lookahead 3・ToF ガード 1・rev_s 0.4・rev_us 300)。guard・keep = rec_back_guard・rec_hold_keep"""
    p = dict(PARAMS, recovery=1, esc_reverse=1, start_kick=1, lookahead=3, tof_guard=1, throttle_gain=0.9, rev_s=0.4,
             rev_us=300.0, launch_boost=0, launch_no_kick=0, speed_ceiling=0, speed_auto=0, zone_caps=0, ceil_left_mps=4.1,
             rec_back_guard=guard, rec_hold_keep=keep)
    p.update(kw)
    return p


def scene_rev(dur=7.0, **kw):
    """場面 rev(反証の場面): 前の壁 2.0 m に当たって止まり、モデルは前進 120 µs で押し続ける。最初のリカバリーの 'brake' は 2.52 秒"""
    s = {'dur': dur, 'xf': XF, 'model': MODEL, 'drops': [], 'veh_skip': []}
    s.update(kw)
    return s


def model_brake(t, car):
    return (C, N - 120.0) if car.x < 2.4 - 0.45 and t < 6.0 else (C, N + 200.0)


def scene_brake(dur=7.0, rear=None, **kw):
    """場面 brake(10/23 の形): 前の壁 2.4 m・壁の 0.45 m 手前からブレーキ側(中立 + 200)。止まると中立になり ESC は後退を待つ"""
    s = {'dur': dur, 'xf': 2.4, 'model': model_brake, 'drops': [], 'veh_skip': []}
    if rear is not None:
        s['xr'] = 2.4 - rear
    s.update(kw)
    return s


T_BRAKE = 2.52


def first(sim, pca, t_from=1.5):
    """最初のリカバリー: 'brake'(後退側 中立 + 300)を書いた時刻から、最初の escape の 0.6 秒あとまでに下がった距離・escape が前の壁を
    押した秒・後退側を止まったまま書いた秒"""
    tb = next(t for t, s, th in pca.w if t > t_from and abs(th - REV) < 1e-6)
    te = next((t for t, s, th in pca.w if t > tb and abs(th - ESCAPE) < 1e-6), None)
    end = (te + 0.6) if te is not None else tb + 3.0
    x_b = next(x for t, x, v, th in sim.log if t >= tb)
    back = round(x_b - min(x for t, x, v, th in sim.log if tb <= t <= end), 3)
    push = 0.0 if te is None else round(sum(DT for t, x, v, th in sim.log if te <= t < te + 0.6 and x >= sim.s['xf'] - 1e-9
                                            and abs(th - ESCAPE) < 1e-6), 2)
    dead = round(sum(DT for t, x, v, th in sim.log if tb <= t <= end and th > N + 40 and abs(v) < 0.05), 2)
    return {'t_brake': tb, 'back': back, 'esc_push': push, 'rev_dead': dead}


GLITCH_T = [2.54, 2.58, 2.64, 2.70, 2.78]       # 50 Hz・cam0 0.15 で、跳ねがフレームの eRPM になる時刻(反証の exp_glitch)


def test_loop_reference(monkeypatch, tmp_path):
    """跳ねなしの場面 rev: 今・直す前の写し・OFF とも 2.52 秒に 'brake'、1.133 m 下がり escape は前の壁を押さない。今の ON は
    直す前の写しの ON とビット単位で同じ"""
    amod, before = load_before()
    on = run(drive, monkeypatch, tmp_path, P(), scene_rev())
    bef = run(before, monkeypatch, tmp_path, P(), scene_rev())
    off = run(drive, monkeypatch, tmp_path, P(0, 0), scene_rev())
    F5.same_loop(on, bef)
    for s, rec, pca in (on, off):
        m = first(s, pca)
        assert m['t_brake'] == pytest.approx(T_BRAKE) and m['back'] == pytest.approx(1.133, abs=1e-3) and m['esc_push'] == 0.0


@pytest.mark.parametrize('e', [301, 1322])
@pytest.mark.parametrize('tg', GLITCH_T)
@pytest.mark.parametrize('keep', [1, 0])
def test_loop_finding_reproduced_and_fixed(monkeypatch, tmp_path, keep, tg, e):
    """指摘の場面: 場面 rev・最初のリカバリーの 'brake' の間に 1 サンプル e(車は動かない)。直す前の写しの ON(見張り 1・keep)は
    back_early で 0 m・escape が前の壁を 0.5 秒押す(再現)。今の ON は OFF と同じだけ下がり(1.133 m)、escape は前の壁を押さない。
    PCA9685 に書く値は OFF とビット単位で同じ(違うのは記録の出来事 back_no_esc だけ)"""
    amod, before = load_before()
    sc = scene_rev(glitch=[(tg, e)])
    bef = run(before, monkeypatch, tmp_path, P(1, keep), sc)
    on = run(drive, monkeypatch, tmp_path, P(1, keep), sc)
    off = run(drive, monkeypatch, tmp_path, P(0, 0), sc)
    mb, mo, mf = first(bef[0], bef[2]), first(on[0], on[2]), first(off[0], off[2])
    assert mb['back'] < 0.05 and mb['esc_push'] >= 0.45 and R2.guards(bef[1])[:1] == ['back_early']     # 指摘の再現
    assert mf['back'] == pytest.approx(1.133, abs=1e-3) and mf['esc_push'] == 0.0
    assert mo == mf and on[2].w == off[2].w
    assert R2.guards(on[1]) == ['back_no_esc']


@pytest.mark.parametrize('tg', GLITCH_T)
def test_loop_s4b_alone_also_backs(monkeypatch, tmp_path, tg):
    """drive の EscTrack を使わない形(Recovery に esc_rev = None)でも、s4-b で 'gap' からやり直して下がる(1 m より多く)。
    escape は前の壁を押さない。出来事 back_early → early_none"""
    sc = scene_rev(glitch=[(tg, 1322)])
    on = run(drive, monkeypatch, tmp_path, P(), sc, force={'esc_rev': None})
    m = first(on[0], on[2])
    assert m['back'] > 1.0 and m['esc_push'] == 0.0
    assert R2.guards(on[1])[:2] == ['back_early', 'early_none']
    assert max(th for t, s, th in on[2].w if 2.4 < t < 4.5) == REV            # +50 µs のやり直しにしない


@pytest.mark.parametrize('e', [301, 313, 1322])
@pytest.mark.parametrize('k', [3, 7, 12, 17, 22])
def test_loop_100hz_human_esc(monkeypatch, tmp_path, k, e):
    """本物と同じ 100 Hz の車両の値・人に寄せた偽物の ESC(中立 0.12 秒・後退側から 0.10 秒)。'brake' の始めから k × 10 ms の
    1 サンプル e: 今の ON は OFF と同じ距離を下がり、escape は前の壁を押さない"""
    base = scene_rev(esc=(0.12, 0.10), veh_dt=0.01)
    sc = dict(base, glitch=[(round(T_BRAKE + 0.01 * k, 2), e)])
    on = run(drive, monkeypatch, tmp_path, P(), sc)
    off = run(drive, monkeypatch, tmp_path, P(0, 0), sc)
    mo, mf = first(on[0], on[2]), first(off[0], off[2])
    assert mf['back'] > 1.0 and mo == mf and on[2].w == off[2].w


BURSTS = {   # 本物の記録の跳ねの並び(10 ms ごと。data/raw を読んだだけ。車は動いていない)
    '20261010-160306-0002': [1322],
    '20261010-152707-0004': [1628, 0, 1525, 0, 0, 1572, 0, 0, 1289],
    '20261023-211828-0004': [1687, 0, 0, 0, 0, 0, 1411, -1270, 1408, -2156, 1356, 0, 0, -394, 1528, 1358, 656, 385],
}


@pytest.mark.parametrize('name', sorted(BURSTS))
@pytest.mark.parametrize('k', [0, 4, 9, 14, 20])
def test_loop_real_bursts(monkeypatch, tmp_path, name, k):
    """本物の記録の跳ねの並びを 100 Hz の車両の値で 'brake' の始めから k × 10 ms に入れる: 今の ON は OFF と同じ距離を下がり、
    escape は前の壁を押さない"""
    seq = BURSTS[name]
    t0 = round(T_BRAKE + 0.01 * k, 2)
    sc = scene_rev(veh_dt=0.01, glitch=[(round(t0 + 0.01 * i, 2), x) for i, x in enumerate(seq) if x])
    on = run(drive, monkeypatch, tmp_path, P(), sc)
    off = run(drive, monkeypatch, tmp_path, P(0, 0), sc)
    mo, mf = first(on[0], on[2]), first(off[0], off[2])
    # 'gap' に入ってからの跳ねは見張りの惰性の数えに入り、そのぶん後退が短い(1 フレーム 約 3 cm)。やり直しの決めには使わない
    assert mf['back'] > 1.0 and mo['esc_push'] == 0.0 and mo['back'] > mf['back'] - 0.1


def test_loop_100hz_bursts_spring_reproduced(monkeypatch, tmp_path):
    """上の 100 Hz・本物の跳ねの並び・壁からの戻りの組に、直す前の写しの ON で 0 m・escape が前の壁を押す組がある(空振りでない)"""
    amod, before = load_before()
    hits = {'100hz': 0, 'burst': 0, 'spring': 0}
    for k in (3, 7, 12, 17, 22):
        for e in (301, 1322):
            sc = scene_rev(esc=(0.12, 0.10), veh_dt=0.01, glitch=[(round(T_BRAKE + 0.01 * k, 2), e)])
            m = first(*run(before, monkeypatch, tmp_path, P(), sc)[0::2])
            hits['100hz'] += m['back'] < 0.05 and m['esc_push'] >= 0.45
    for k in (0, 4, 9):
        seq = BURSTS['20261010-152707-0004']
        t0 = round(T_BRAKE + 0.01 * k, 2)
        sc = scene_rev(veh_dt=0.01, glitch=[(round(t0 + 0.01 * i, 2), x) for i, x in enumerate(seq) if x])
        m = first(*run(before, monkeypatch, tmp_path, P(), sc)[0::2])
        hits['burst'] += m['back'] < 0.05 and m['esc_push'] >= 0.45
    for cam0 in (0.15, 0.16, 0.17, 0.18):
        m = first(*run(before, monkeypatch, tmp_path, P(), scene_rev(cam0=cam0, spring=(0.16, 0.05, 0.2)))[0::2])
        hits['spring'] += m['back'] < 0.05 and m['esc_push'] >= 0.4
    assert all(v >= 1 for v in hits.values()), hits


@pytest.mark.parametrize('spring', [(0.16, 0.02, 0.2), (0.16, 0.05, 0.2), (0.25, 0.04, 0.2)])
@pytest.mark.parametrize('cam0', [0.15, 0.17])
def test_loop_spring_back(monkeypatch, tmp_path, spring, cam0):
    """壁を押していた車が 'brake'(ブレーキ)に変わったときの少しの戻り(バンパー・タイヤのたわみ。0.16〜0.25 m/s・0.02〜0.05 秒。
    本当に後ろへ動く): 今の ON は 'gap' を飛ばさず、OFF と同じく下がる(1 m より多く)。escape は前の壁を押さない"""
    sc = scene_rev(cam0=cam0, spring=spring)
    on = run(drive, monkeypatch, tmp_path, P(), sc)
    off = run(drive, monkeypatch, tmp_path, P(0, 0), sc)
    mo, mf = first(on[0], on[2]), first(off[0], off[2])
    assert mf['back'] > 1.0 and mo['back'] > 1.0 and mo['esc_push'] == 0.0
    assert 'back_early' not in R2.guards(on[1])[:1]


@pytest.mark.parametrize('rear', [None, 1.35])
@pytest.mark.parametrize('vdt', [None, 0.01])
@pytest.mark.parametrize('esc', [None, (0.12, 0.10)])
def test_loop_brake_scene_identical(monkeypatch, tmp_path, esc, vdt, rear):
    """場面 brake(ESC が後退を待っている 10/23 の形): 'brake' の後退側がそのまま後退になる。今の ON は直す前の写しの ON と
    ビット単位で同じ(back_early も下がる距離も同じ)。見張りだけ(keep 0)も同じ"""
    amod, before = load_before()
    sc = scene_brake(rear=rear, esc=esc, veh_dt=vdt)
    for keep in (1, 0):
        on = run(drive, monkeypatch, tmp_path, P(1, keep), sc)
        bef = run(before, monkeypatch, tmp_path, P(1, keep), sc)
        F5.same_loop(on, bef)
        assert R2.guards(on[1])[:1] == ['back_early']


@pytest.mark.parametrize('esc', [None, (0.12, 0.10)])
def test_loop_brake_scene_spike_same_as_before(monkeypatch, tmp_path, esc):
    """場面 brake(ESC が後退を待っている)の跳ね(1322)を 2.62〜2.94 秒に(手順の前・'brake' を書いた直後・ESC が後退で走らせて
    いる間): EscMayReverse は 'brake' を書いたときから後退とみるので、どの時刻でも直す前の写しとビット単位で同じ"""
    amod, before = load_before()
    for tg in (2.62, 2.70, 2.74, 2.76, 2.78, 2.80, 2.86, 2.94):
        sc = scene_brake(esc=esc, glitch=[(tg, 1322)])
        now = run(drive, monkeypatch, tmp_path, P(), sc)
        F5.same_loop(now, run(before, monkeypatch, tmp_path, P(), sc))
        assert 'back_no_esc' not in R2.guards(now[1])


@pytest.mark.parametrize('e', [313, 1322])
@pytest.mark.parametrize('tg', [2.84, 2.88, 2.92])
@pytest.mark.parametrize('esc', [(0.2, 0.08), (0.25, 0.08)])
def test_loop_gap_spike_slow_esc(monkeypatch, tmp_path, esc, tg, e):
    """'gap' の中立 0.15 秒では下がれない ESC(中立 0.2・0.25 秒)で、'gap' の間に 1 サンプル e(見張りの惰性の数え): 直す前の写しの
    ON は backed_s > 0 でやり直さず 0 m・escape が前の壁を押す(この形の再現は時刻によるので、起きた組だけ見る)。今の ON は
    OFF と同じくやり直して下がり(1 m より多く)、escape は前の壁を押さない"""
    amod, before = load_before()
    sc = scene_rev(dur=8.0, esc=esc, glitch=[(tg, e)])
    on = run(drive, monkeypatch, tmp_path, P(), sc)
    off = run(drive, monkeypatch, tmp_path, P(0, 0), sc)
    mo, mf = first(on[0], on[2]), first(off[0], off[2])
    assert mf['back'] > 1.0 and mo['back'] > 1.0 and mo['esc_push'] == 0.0


def test_loop_gap_spike_slow_esc_reproduced(monkeypatch, tmp_path):
    """上の形が直す前の写しで起きる組がある(指摘の 'gap' の数えの道。空振りでない)"""
    amod, before = load_before()
    hit = 0
    for tg in (2.84, 2.86, 2.88, 2.90, 2.92):
        sc = scene_rev(dur=8.0, esc=(0.2, 0.08), glitch=[(tg, 1322)])
        mb = first(*run(before, monkeypatch, tmp_path, P(), sc)[0::2])
        hit += mb['back'] < 0.05 and mb['esc_push'] >= 0.45
    assert hit >= 1


OFF_SCENES = {
    'rev': scene_rev(),
    'rev_glitch': scene_rev(glitch=[(2.54, 1322), (2.88, 313)]),
    'rev_100hz': scene_rev(veh_dt=0.01, esc=(0.12, 0.10), glitch=[(2.55, 1322)]),
    'rev_hold': scene_rev(drops=[(2.6, 2.75)], veh_skip=[(2.9, 3.1)], glitch=[(2.56, 1322)]),
    'brake': scene_brake(rear=1.35),
    'brake_glitch': scene_brake(glitch=[(2.7, 1322)]),
    'spring': scene_rev(spring=(0.16, 0.03, 0.2)),
}


@pytest.mark.parametrize('name', sorted(OFF_SCENES))
@pytest.mark.parametrize('extra', [{}, {'launch_boost': 1, 'speed_ceiling': 1}, {'rec_hold_keep': 1}])
def test_loop_off_bit_identical(monkeypatch, tmp_path, name, extra):
    """見張り OFF(rec_back_guard = 0。rec_hold_keep 1 を送っても走行ループは両方 1 のときだけ残す)の走行ループは、直す前の写しと
    PCA9685 に書く値・記録の行・出来事がビット単位で同じ"""
    amod, before = load_before()
    p = P(0, 0, **extra)
    sc = OFF_SCENES[name]
    F5.same_loop(run(drive, monkeypatch, tmp_path, p, sc), run(before, monkeypatch, tmp_path, p, sc))


def test_loop_esc_rev_passed_only_with_guard(monkeypatch, tmp_path):
    """drive が Recovery に渡す esc_rev: 見張り OFF は全部 None(今まで通り)。見張り ON(keep 0 でも)は最初のフレームが None、そのあと
    True / False。場面 rev の 'brake'・'wait' は False(前進のあとのブレーキ)・最初の 'reverse' は後退側を書いた次のフレームから True(EscMayReverse は遅れ 0)"""
    seen = []
    orig = A.Recovery.step

    def spy(self, model_us, erpm, now, *a, **kw):
        seen.append((round(now - 100.0, 3), self.phase, kw.get('esc_rev', 'missing')))
        return orig(self, model_us, erpm, now, *a, **kw)
    monkeypatch.setattr(A.Recovery, 'step', spy)
    run(drive, monkeypatch, tmp_path, P(0, 0), scene_rev(dur=4.0))
    assert seen and {x[2] for x in seen} == {None}
    for keep in (1, 0):
        del seen[:]
        run(drive, monkeypatch, tmp_path, P(1, keep), scene_rev(dur=4.0))
        assert all(isinstance(x[2], bool) for x in seen[1:]) and seen[0][2] in (None, False)   # 人側の間も helper.step は呼ぶ
        assert {x[2] for x in seen if x[1] in ('brake', 'wait')} == {False}
        rev = [x for x in seen if x[1] == 'reverse']
        assert rev and rev[-1][2] is True


@pytest.mark.parametrize('seed', range(3))
def test_loop_fuzz_off_identical_and_on_rules(monkeypatch, tmp_path, seed):
    """でたらめな閉じたループ(tests/test_rec_fix5_safety の場面と決まり)に、車両の値の跳ねを足して: 見張り OFF は直す前の写しと
    ビット単位で同じ。今の ON(両方 1)は fix5 の決まり(check_rules)を破らない"""
    amod, before = load_before()
    rng = random.Random(7000 + seed)
    for i in range(4):
        sc = F5._rand_scene(rng)
        p1 = F5._rand_params(rng)
        sc['glitch'] = [(round(rng.uniform(1.5, sc['dur']), 2), rng.choice([301, 313, 1322, 1687])) for _ in range(6)]
        sc['veh_dt'] = rng.choice([0.01, 0.02])
        p0 = dict(p1, rec_back_guard=0, rec_hold_keep=0)
        F5.same_loop(run(drive, monkeypatch, tmp_path, p0, sc), run(before, monkeypatch, tmp_path, p0, sc))
        p1 = dict(p1, rec_back_guard=1, rec_hold_keep=1)
        s, rec, pca = run(drive, monkeypatch, tmp_path, p1, sc)
        bad = F5.check_rules(s, rec, pca, p1)
        assert not bad, bad[:3]
