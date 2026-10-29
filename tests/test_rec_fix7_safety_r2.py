"""10/27 fix7 係 safety 直す 7-2 回目(反証 7-2 回目の指摘 w2-safety-rv7-safety-r2-1)の試験。走行ループ・補助の側。

  指摘: rec_hold_keep = 1・rec_back_guard = 1(11/3 の形)で、最初のリカバリーの 'brake' の中に約 0.03 秒の中立止め(カメラの
    フレームが 0.1 秒来ない → drive の hold() が中立を書く。手順は残す)が入ると、drive の EscMayReverse(ブレーキのあと中立を
    1 回でも書いたら下がれる)が続く 'brake' の後退側を「後退で走らせていたかもしれない」とみて esc_rev = True・esc_ran = True に
    なり、この手順の fix7 s4-a(前進のあとの 'brake'・'wait'・'gap' の後ろへの動きを見張りに使わない)が外れる。閉じたループの
    偽物の ESC は中立 0.1 秒が要るので、まだブレーキのまま。止まっている車の eRPM の跳ね 1 つ目で back_early('gap' を飛ばして
    'reverse')、'reverse' の中の跳ね 2 つ目で seen_rev_s > 0 になり、s4-b の early_none も CHECK_S のやり直しも止まって、0 m の
    まま後退側を約 1.3 秒書き、escape が前の壁を 0.5 秒押す(OFF は 1.27 m 下がる)
  直し(assist.Recovery.step。見張り ON(back_guard = True)の日だけ効く。rec_hold_keep は問わない):
    s4-c  'gap' を飛ばして入った 'reverse'(skip_gap。back_early・back_hold)の early_none は「本当に下がっている」を見たかで決める。
          本当に下がっている = 入ったフレームのほかのフレームで、続けて assist.SURE_BACK_S(0.2 秒)後ろへ動いていた(途切れたら・
          残した中立止めのあとは数え直す)。CHECK_S のあと、まだ見ていなくて、このフレームで後ろへ動いていなければ early_none
          ('gap' から)。このフレームで後ろへ動いていれば次のフレームで決める。一度見たら、その 'reverse' ではやり直さない。
          'gap' を飛ばさずに入った 'reverse' は今まで通り(s4-b)
  EscMayReverse(drive の esc_rev)と drive.py は変えていない(本物の ESC は人の操作で中立 0.05〜0.07 秒でも後退に入った)。

OFF(rec_back_guard = 0。今の既定)は、直す前の写し tests/fixtures/fix7_safety_r2_before/(assist.py・drive.py = 7-1 回目のあと。
sha256 は SHA256SUMS)とビット単位で同じ(PCA9685 に書く値・記録の行・出来事)。指摘の場面の試験は「直す前の写しで指摘が起きる
(再現)」を同じ試験の中で見る。走行ループは tests/test_rec_fix7_safety の閉じたループ(GSim。本物の drive._loop を、書いた値で
動く偽物の ESC と車で回す)。実機・GPU・torch・jetformer-web は使わない。台本と表: experiments/20261026_fix/fix7/safety_r2/。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_rec_fix7_safety_r2.py
"""
import hashlib
import importlib.util
import math
import random
from pathlib import Path

import pytest

from jetformer import assist as A
from jetformer import drive

import test_rec_fix5_safety as F5
import test_rec_fix6_safety as T6
import test_rec_fix7_safety as F7
from test_launch_drive import C, N

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix7_safety_r2_before'
SHA_R1 = {'assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f',     # 10/27 fix7 safety 7-1 回目のあと
          'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}
# この組のあと 10/28 fix8 safety が drive.py の説明の文だけを変えた(AST は同じ)。そのときは、その直す前の写し
# tests/fixtures/fix8_safety_doc_before/(= この組の drive.py。sha256 を照らす)で照らす(tests/test_fix3_r3_2_tools.py の
# LATER_FIX8_SAFETY_DOC と同じ値)
LATER_FIX8_SAFETY_DOC = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e'}
# さらに 10/29 fix9 の優先のモデル(main session。走行タブの一覧の並べ方だけ)が drive.py を変えた。そのときは、その直す前の写し
# tests/fixtures/fix9_priority_before/(= LATER_FIX8_SAFETY_DOC の版。sha256 を照らす)からたどる(変えたのが優先のモデルの所だけかは
# tests/test_fix9_priority_before.py。tests/test_fix3_r3_2_tools.py の LATER_FIX9_PRIORITY と同じ値)
LATER_FIX9_PRIORITY = {'drive.py': 'ecfb6656cae624950d5de221dfa71a5ac31b84ba82dacfbea2136832eae92315'}
FDT = F7.FDT
MODEL = F7.MODEL
REV, GAP, RETRY, ESCAPE = F7.REV, F7.GAP, F7.RETRY, F7.ESCAPE
STATE = F7.STATE + ('seen_rev_s', 'skip_gap', 'noesc_noted', 'esc_ran')
DEC = 10                # back_early で入ったフレームの次を 0 として、CHECK_S(0.35 秒)を超える最初のフレーム(11 × FDT = 0.367 秒)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_R1 = {}


def load_r1():
    """直す前(7-1 回目のあと)の assist.py・drive.py(写し)をモジュールにする(sha256 を照らしてから)"""
    if _R1:
        return _R1['a'], _R1['d']
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sums == SHA_R1
    for name, h in sums.items():
        assert sha(FIX / name) == h, name
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix7s_r2', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix7s_r2', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    r = amod.Recovery((C, N))
    assert hasattr(r, 'seen_rev_s') and not hasattr(r, 'back_sure') and not hasattr(amod, 'SURE_BACK_S')   # 7-1 回目のあと・直す前
    _R1.update(a=amod, d=mod)
    return amod, mod


# ================================================================ 0. 写し・既定値・変えた所
def test_r1_copy_defaults_and_changed_parts():
    """写しの sha256(= 7-1 回目のあとの版)。drive.py は写しと同じ(変えていない)。既定値・Recovery・EscTrack・EscMayReverse・
    EscReverse・StartKick の定数・モジュールの定数は同じ。変えたのは assist.py の Recovery.reset・_start・step だけ、足したのは
    モジュールの定数 SURE_BACK_S だけ(消した関数や文は無い。AST を名前で照らす。モジュールの説明は別)"""
    amod, r1d = load_r1()
    pd = ROOT / 'jetformer' / 'drive.py'
    if sha(pd) == LATER_FIX9_PRIORITY['drive.py']:           # 10/29 fix9 優先のモデルの版 → 直す前の写し(= fix8 safety の説明の文だけの版)
        pd = ROOT / 'tests' / 'fixtures' / 'fix9_priority_before' / 'drive.py'
    if sha(pd) == LATER_FIX8_SAFETY_DOC['drive.py']:         # 10/28 fix8 safety の説明の文だけの版 → 直す前の写し
        pd = ROOT / 'tests' / 'fixtures' / 'fix8_safety_doc_before' / 'drive.py'
    assert sha(pd) == SHA_R1['drive.py']
    assert drive.DEFAULT_PARAMS == r1d.DEFAULT_PARAMS and drive.PARAM_LIMITS == r1d.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == r1d.RESET_ON_BOOT
    for k in ('rec_back_guard', 'rec_hold_keep', 'launch_boost', 'launch_no_kick'):
        assert drive.DEFAULT_PARAMS[k] == 0 and k in drive.RESET_ON_BOOT
    for cls in ('Recovery', 'EscTrack', 'EscMayReverse', 'EscReverse', 'StartKick'):
        assert {k: v for k, v in vars(getattr(A, cls)).items() if k.isupper()} == \
            {k: v for k, v in vars(getattr(amod, cls)).items() if k.isupper()}, cls
    assert A.EscMayReverse.READY_NEUTRAL_S == A.EscMayReverse.DELAY_S == 0.0      # drive の esc_rev は今まで通り
    mod_consts = {k for k in vars(amod) if k.isupper() and not k.startswith('_')}
    for k in mod_consts:
        assert getattr(A, k) == getattr(amod, k), k
    assert {k for k in vars(A) if k.isupper() and not k.startswith('_')} - mod_consts == {'SURE_BACK_S'}
    assert A.SURE_BACK_S == 0.2 and A.SURE_BACK_S < drive.PARAM_LIMITS['rev_s'][0] + 1e-9 and A.SURE_BACK_S < A.CHECK_S
    a0, a1 = T6._defs(FIX / 'assist.py'), T6._defs(ROOT / 'jetformer' / 'assist.py')
    assert set(a0) <= set(a1) and set(a1) - set(a0) == {'set:SURE_BACK_S'}
    assert sorted(k for k in a0 if a0[k] != a1[k]) == ['Recovery._start', 'Recovery.reset', 'Recovery.step']
    assert '反証 7-2 回目の直し' in A.__doc__ and '反証 7-2 回目の直し' not in amod.__doc__
    assert '反証 7-1 回目の直し' in A.__doc__


# ================================================================ 1. Recovery(単体)
def _skip_rev(mod, **kw):
    """'brake' の次のフレームで跳ね 1322 → back_early で 'gap' を飛ばして 'reverse' に入った Recovery(見張り ON)と、次の時刻。
    kw = step に足す引数(esc_rev など)"""
    r, t = F7._brake(mod, **kw)
    rows, t = F7._feed(r, [0.0, 1322.0], t, **kw)
    assert rows[-1][2] == 'reverse' and rows[-1][3][0] == 'back_early' and r.skip_gap
    return r, t


def _seq_after_entry(seq_rev, tail=60):
    """入ったフレームのあとの eRPM(1 フレーム目 = 入ったフレームの次)+ 下がらない tail フレーム"""
    return list(seq_rev) + [0.0] * tail


def _ph(rows):
    return [ph for _, _, ph, _ in rows]


@pytest.mark.parametrize('esc_rev', [None, True])
@pytest.mark.parametrize('e', [301.0, 1322.0])
@pytest.mark.parametrize('j', [1, 3, 6, 9, 10, 11, 12])
def test_s4c_unit_second_spike_in_skip_gap_reverse(j, e, esc_rev):
    """指摘の形(単体): back_early で入った 'reverse'(ESC はブレーキのまま = eRPM 0)の j フレーム目に 2 つ目の跳ね e。
    今は CHECK_S を超えた最初の後ろへ動いていないフレームで early_none → 'gap'(GAP_S。やり直しの数 0)→ 'reverse' は 300 µs。
    跳ねが CHECK_S を超えたフレームに入ったら(j = 11)、次のフレームで決める。直す前の写しは後退側を MAX_REV_TOTAL_S 書いて
    escape(中立を一度も出さない。指摘の再現)。CHECK_S で決めたあとの跳ね(j = 12)は直す前と同じ"""
    amod, _ = load_r1()
    kw = {} if esc_rev is None else {'esc_rev': esc_rev}
    seq = [0.0] * (j - 1) + [e] + [0.0] * 80
    r, t = _skip_rev(A, **kw)
    now, t2 = F7._feed(r, seq, t, **kw)
    rb, tb = _skip_rev(amod, **kw)
    bef, _ = F7._feed(rb, seq, tb, **kw)
    if j - 1 > DEC:             # CHECK_S で決めたあとの跳ね: 直す前と同じ(どちらも DEC で early_none)
        assert F7._thr(now) == F7._thr(bef) and _ph(now) == _ph(bef) and _ph(now).index('gap') == DEC
        return
    ph = _ph(now)
    i_gap = ph.index('gap')
    assert i_gap == (DEC + 1 if j - 1 == DEC else DEC)        # 0 始まり = 入ったフレームの次が 0。跳ねが DEC なら次のフレーム
    assert ph[:i_gap] == ['reverse'] * i_gap and F7._thr(now)[:i_gap] == [REV] * i_gap
    assert F7._events(now)[:1] == ['early_none'] and now[i_gap][3][2] > 0.0           # 跳ねを数えた backed_s(> 0)を捨てた
    th = F7._thr(now)
    assert th[i_gap:i_gap + 5] == [GAP] * 5 and th[i_gap + 5] == REV                  # ふつうの 'gap'(0.15 秒)→ 300 µs
    assert RETRY not in th[:i_gap + 6]
    # 直す前: 中立を一度も出さず、後退側を MAX_REV_TOTAL_S 書いてから escape
    tb_ = F7._thr(bef)
    assert GAP not in tb_[:40] and ESCAPE in tb_
    assert sum(1 for x in tb_ if x == REV) * FDT >= A.Recovery.MAX_REV_TOTAL_S - 2 * FDT


BURST_FRAMES = {    # フレームから見た跳ねの並び(1 = 後ろへ動いている)。本物の記録の最長は続けて 0.11 秒(4 フレーム)
    'two_consecutive': [1, 1],
    'four_consecutive': [1, 1, 1, 1],
    'six_consecutive': [1, 1, 1, 1, 1, 1],          # 0.167 秒 < SURE_BACK_S
    'alternating': [1, 0, 1, 0, 1, 0, 1],
    'scattered': [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
}


@pytest.mark.parametrize('name', sorted(BURST_FRAMES))
@pytest.mark.parametrize('j', [1, 4, 8])
def test_s4c_unit_bursts_not_sure(name, j):
    """'reverse' の中の跳ねの並び(続けて SURE_BACK_S より短い・とびとび): 今は early_none('gap' から)。直す前は後退側を
    MAX_REV_TOTAL_S 書いて escape"""
    amod, _ = load_r1()
    pat = [1322.0 if x else 0.0 for x in BURST_FRAMES[name]]
    seq = [0.0] * (j - 1) + pat + [0.0] * 80
    r, t = _skip_rev(A)
    now, _ = F7._feed(r, seq, t)
    rb, tb = _skip_rev(amod)
    bef, _ = F7._feed(rb, seq, tb)
    assert 'early_none' in F7._events(now) and GAP in F7._thr(now) and not r.back_sure
    i_gap = _ph(now).index('gap')
    assert i_gap == next(i for i in range(DEC, len(seq)) if seq[i] <= A.BACKING_ERPM)   # CHECK_S のあと、後ろへ動いていない最初のフレーム
    assert GAP not in F7._thr(bef)[:40] and ESCAPE in F7._thr(bef)


@pytest.mark.parametrize('rev_s', [0.2, 0.4, 0.8, 1.2])
@pytest.mark.parametrize('j', [1, 2, 5, 9, 11, 13])
@pytest.mark.parametrize('dip', [None, 3, 9, 14])
def test_s4c_unit_real_backing_identical(j, rev_s, dip):
    """back_early のあと本当に下がる(j フレーム目から続けて 1400 eRPM。dip = そこから dip フレーム目に 1 フレームだけ 0):
    出す値・段・状態は直す前の写しと全部同じ(一度 SURE_BACK_S 続けば、あとの 1 フレームの途切れではやり直さない。続く前の
    途切れは、直す前と同じく見た後退があるので、どちらもやり直さないか、どちらも同じフレームで 'gap')"""
    amod, _ = load_r1()
    back = [1400.0] * 50
    if dip is not None:
        back[dip] = 0.0
    seq = [0.0] * (j - 1) + back + [0.0] * 30
    kw = dict(rev_s=rev_s)
    r, t = _skip_rev(A)
    rb, tb = _skip_rev(amod)
    diverged = None
    for i, x in enumerate(seq):
        a = r.step(MODEL, x, t, back_guard=True, **kw)
        b = rb.step(MODEL, x, tb, back_guard=True, **kw)
        if a != b or any(getattr(r, k) != getattr(rb, k) for k in STATE):
            diverged = i
            break
        r.guard_event = rb.guard_event = None
        t += FDT
        tb += FDT
    if diverged is None:
        return
    # 違ってよいのは: 本当に下がっているを見る前(続けて 0.2 秒より短い)に途切れ、CHECK_S を超えていたフレームだけ(今は early_none)
    i = diverged
    assert dip is not None and dip < 6 and j - 1 + dip == i and i >= DEC, (i, j, dip)
    assert r.phase == 'gap' and rb.phase == 'reverse'


def test_s4c_unit_short_real_backing_then_stop_restarts():
    """back_early のあと 0.1 秒(3 フレーム)だけ後ろへ動いて止まる(SURE_BACK_S より短い): 今は下がっていないとみて early_none
    ('gap' から。下がった時間を捨てる)。直す前の写しは見た後退があるとして、後退側を MAX_REV_TOTAL_S 書いて escape。
    (本当に 0.1 秒だけ下がって止まった車は後ろに当たった・引っかかった形で、そのときは 'gap' からまた後退する。
    後ろの当たりの衝撃は rear_impact が先に前へ)"""
    amod, _ = load_r1()
    seq = [0.0, 0.0] + [1400.0] * 3 + [0.0] * 60
    r, t = _skip_rev(A)
    now, _ = F7._feed(r, seq, t)
    rb, tb = _skip_rev(amod)
    bef, _ = F7._feed(rb, seq, tb)
    assert F7._events(now)[:1] == ['early_none'] and _ph(now).index('gap') == DEC
    assert GAP not in F7._thr(bef)[:30] and ESCAPE in F7._thr(bef)


@pytest.mark.parametrize('hb', [0.02, 0.06])
def test_s4c_unit_hold_spike_in_skip_gap_reverse(hb):
    """'gap' を飛ばして入った 'reverse' の途中の残した中立止め(0.15 秒)の間に後ろへ動いていたサンプル hb 秒(跳ね)。そのあと
    下がらない: 今は中立止めのあとから CHECK_S で early_none。直す前の写しは見た後退として、後退側を MAX_REV_TOTAL_S 書いて escape"""
    amod, _ = load_r1()
    out = {}
    for nm, mod in (('now', A), ('bef', amod)):
        r, t = _skip_rev(mod)
        _, t = F7._feed(r, [0.0] * 3, t)
        t_h = t + 0.15
        o = r.step(MODEL, 0.0, t_h, back_guard=True, hold_since=t + 0.02, hold_back_s=hb)
        r.guard_event = None
        rows, _ = F7._feed(r, [0.0] * 60, t_h + FDT)
        out[nm] = (o, rows, r)
    now_rows, bef_rows = out['now'][1], out['bef'][1]
    assert F7._events(now_rows)[:1] == ['early_none']
    assert _ph(now_rows).index('gap') == DEC                    # 中立止めのフレームを入ったフレームとして、そこから CHECK_S
    assert GAP not in F7._thr(bef_rows)[:30] and ESCAPE in F7._thr(bef_rows)


def test_s4c_unit_hold_then_real_backing_identical():
    """'gap' を飛ばして入った 'reverse' の途中の残した中立止めのあと、本当に下がる(続けて 1400): 直す前の写しと全部同じ"""
    amod, _ = load_r1()
    res = []
    for mod in (A, amod):
        r, t = _skip_rev(mod)
        rows0, t = F7._feed(r, [0.0, 0.0, 1400.0, 1400.0], t)
        o = r.step(MODEL, 1400.0, t + 0.12, back_guard=True, hold_since=t + 0.01, hold_back_s=0.1)
        st = tuple(getattr(r, k) for k in STATE)
        rows, _ = F7._feed(r, [1400.0] * 30 + [0.0] * 30, t + 0.12 + FDT)
        res.append((F7._thr(rows0), o, st, F7._thr(rows), _ph(rows)))
    assert res[0] == res[1]


def test_s4c_unit_entry_frame_not_in_run():
    """'reverse' に入ったフレーム(back_early の跳ね)は「続けて後ろへ動いていた」に数えない(run_t0 は None のまま)。
    次のフレームから数え、途切れたら数え直す"""
    r, t = _skip_rev(A)
    assert r.run_t0 is None and not r.back_sure
    rows, t = F7._feed(r, [1322.0], t)
    t1 = t - FDT
    assert r.run_t0 == t1 and not r.back_sure
    rows, t = F7._feed(r, [0.0], t)
    assert r.run_t0 is None


def test_s4c_unit_hold_does_not_join_runs():
    """中立止めの前と後の後ろへの動きはつなげない: 中立止め(0.25 秒)の直前のフレームの跳ねと、明けたフレームの跳ねを続けて 0.2 秒
    より長いとみない(本当に下がっているにしない)。今は中立止めのあとから CHECK_S で early_none。直す前は見た後退でやり直さない"""
    amod, _ = load_r1()
    out = {}
    for nm, mod in (('now', A), ('bef', amod)):
        r, t = _skip_rev(mod)
        _, t = F7._feed(r, [0.0, 0.0, 1322.0], t)
        t_h = t + 0.25
        r.step(MODEL, 1322.0, t_h, back_guard=True, hold_since=t + 0.02, hold_back_s=0.0)
        if nm == 'now':
            assert not r.back_sure and r.run_t0 == t_h
        r.guard_event = None
        rows, _ = F7._feed(r, [0.0] * 60, t_h + FDT)
        out[nm] = rows
    assert F7._events(out['now'])[:1] == ['early_none'] and _ph(out['now']).index('gap') == DEC
    assert GAP not in F7._thr(out['bef'])[:30] and ESCAPE in F7._thr(out['bef'])


@pytest.mark.parametrize('k', [0, 2, 5, 8, 11, 14])
@pytest.mark.parametrize('e', [313.0, 1322.0])
def test_s4c_unit_normal_reverse_unchanged(k, e):
    """'gap' を飛ばさずに入った 'reverse'(brake → wait → gap → reverse)の跳ね: 直す前の写しと出す値・状態が全部同じ
    (やり直しの決めは今まで通り s4-b)"""
    amod, _ = load_r1()
    seq = [0.0] * 13 + [0.0] * k + [e] + [0.0] * 60
    res = []
    for mod in (A, amod):
        r, t = F7._brake(mod)
        rows, _ = F7._feed(r, seq, t)
        assert 'reverse' in _ph(rows) and not any(ev == 'back_early' for ev in F7._events(rows))
        res.append((F7._thr(rows), _ph(rows), F7._events(rows), tuple(getattr(r, x) for x in STATE)))
    assert res[0] == res[1]


def _fuzz_stops(seed, n=4000):
    """でたらめな入力(止まるところを多めに): 走る・止まる(跳ね・跳ねの並び・本当に後ろへ・壁からの戻り)・衝撃・中立止め・
    時刻の飛び・esc_rev(None / True / False)"""
    rng = random.Random(seed)
    t, out = 100.0, []
    mode, left = 'run', rng.randint(20, 60)
    esc_rev = rng.choice([None, True, True, False])
    for i in range(n):
        if left <= 0:
            mode = rng.choice(['run', 'stop', 'stop', 'stop', 'back', 'burst'])
            left = {'run': rng.randint(10, 60), 'stop': rng.randint(10, 90), 'back': rng.randint(1, 25),
                    'burst': rng.randint(2, 8)}[mode]
            if rng.random() < 0.2:
                esc_rev = rng.choice([None, True, False])
        left -= 1
        if mode == 'run':
            e = rng.choice([-5000.0, -3000.0, -1500.0])
        elif mode == 'stop':
            e = rng.choice([0.0] * 12 + [301.0, 1322.0, -200.0, 150.0])
        elif mode == 'back':
            e = rng.choice([1400.0, 2200.0, 900.0])
        else:
            e = rng.choice([1322.0, 1628.0, 0.0, -1270.0, 1408.0])
        t += rng.choice([FDT] * 30 + [0.05, 0.1])
        kw = dict(impact_g=rng.choice([0.0] * 60 + [3.0]), rev_us=rng.choice([250.0, 300.0]),
                  rev_s=rng.choice([0.2, 0.3, 0.4, 0.8]), esc_rev=esc_rev)
        if rng.random() < 0.03:
            kw.update(hold_since=t - rng.uniform(0.02, 0.15), hold_back_s=rng.choice([0.0, 0.0, 0.02, 0.06, 0.2]),
                      hold_rev_s=rng.choice([0.0, 0.0, 0.05]), hold_g=rng.choice([0.0] * 10 + [3.0]))
        out.append(((C + rng.uniform(-300, 300), N + rng.choice([-150.0, -60.0, 0.0, 20.0])), e, t, kw))
    return out


@pytest.mark.parametrize('seed', range(6))
def test_recovery_off_bit_identical_r1(seed):
    """見張り OFF(back_guard = False)の Recovery は、どんな入力でも直す前の写しと出す値・状態が全部同じ"""
    amod, _ = load_r1()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    n_rev = 0
    for model, e, t, kw in _fuzz_stops(seed):
        a = r0.step(model, e, t, back_guard=False, **kw)
        b = r1.step(model, e, t, back_guard=False, **kw)
        assert a == b
        for k in STATE:
            assert getattr(r0, k) == getattr(r1, k), k
        n_rev += r1.phase == 'reverse'
        r0.guard_event = r1.guard_event = None
        if r1.phase == 'giveup':                # 中止のまま(人側にするまで): 作り直して続ける(armed から)
            r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    assert n_rev > 30


_FZ = {'diverge': 0, 'runs': 0}


@pytest.mark.parametrize('seed', range(12))
def test_recovery_on_diverges_only_by_s4c(seed):
    """見張り ON の Recovery を直す前の写しと並べて回す(esc_rev はでたらめに None / True / False)。出す値・状態が初めて違うのは、
    今が 'gap' を飛ばして入った 'reverse' で、本当に下がっているを見ておらず、このフレームで後ろへ動いておらず、early_none で
    'gap' へ移ったフレームだけ(直す前はそこで見た後退があるとして 'reverse' を続けたか、'coast' へ)。違ったら作り直して続ける"""
    amod, _ = load_r1()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    for model, e, t, kw in _fuzz_stops(500 + seed):
        p_before, sk_before, sure_before = r1.phase, r1.skip_gap, r1.back_sure
        a = r0.step(model, e, t, back_guard=True, **kw)
        b = r1.step(model, e, t, back_guard=True, **kw)
        same = a == b and all(getattr(r0, k) == getattr(r1, k) for k in STATE)
        if not same:
            assert p_before == 'reverse' and sk_before and not sure_before, (t, p_before, sk_before, r1.phase)
            assert r1.phase == 'gap' and r1.guard_event[0] == 'early_none' and e <= A.BACKING_ERPM
            assert b[1] == A.ASSIST_RECOVERY and b[0][1] == N + A.GAP_OFFSET_US
            assert r0.phase in ('reverse', 'coast', 'escape')
            assert r0.seen_rev_s > 0.0          # 直す前は見た後退があるとしてやり直さなかった
            _FZ['diverge'] += 1
            r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
        r0.guard_event = r1.guard_event = None
        if r1.phase == 'giveup':
            r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    _FZ['runs'] += 1


def test_recovery_on_fuzz_saw_divergence():
    """上のでたらめな入力で、直しの道(違い)を通っている(試験が空振りしていない)"""
    if _FZ['runs'] < 12:
        pytest.skip('上の試験と一緒に流したときだけ見る')
    assert _FZ['diverge'] >= 5, _FZ


@pytest.mark.parametrize('seed', range(3))
def test_s4c_esc_rev_true_or_none_identical(seed):
    """esc_rev = True と None は、どんな入力でも出す値・段・状態(run_t0・back_sure も)が全部同じ"""
    ra, rb = A.Recovery((C, N)), A.Recovery((C, N))
    for model, e, t, kw in _fuzz_stops(900 + seed):
        kw = dict(kw, esc_rev=None)
        a = ra.step(model, e, t, back_guard=True, **dict(kw, esc_rev=True))
        b = rb.step(model, e, t, back_guard=True, **kw)
        assert a == b
        for k in STATE + ('run_t0', 'back_sure'):
            assert getattr(ra, k) == getattr(rb, k), k


# ================================================================ 2. 走行ループ(閉じたループ。本物の drive._loop)
P = F7.P
scene_rev = F7.scene_rev
XF_REV = F7.XF                              # 場面 rev の前の壁(2.0 m)
scene_brake = F7.scene_brake
CAM0 = 0.15
DROP = [(2.52, 2.62)]                       # 指摘の入力: 'brake'(2.52 秒)の始めから 0.1 秒カメラが来ない(hold が約 0.03 秒中立)
GL2 = [(2.68, 1322), (2.88, 1322)]          # 止まっている車の eRPM の跳ね 2 つ


def run(mod, monkeypatch, tmp_path, params, scene, force=None):
    return F7.run(mod, monkeypatch, tmp_path, params, scene, force=force)


def first(res):
    s, rec, pca = res
    return F7.first(s, pca)


def failed(m):
    """指摘の形: 最初のリカバリーで 0.1 m も下がらず、escape が前の壁を押した"""
    return m['back'] < 0.1 and m['esc_push'] > 0.05


def frames_after(t, n):
    """t より後のフレームの撮像時刻 n 個(フレームは CAM0 + k/30 秒)"""
    k = int(math.floor((t - CAM0) * 30)) + 1
    out = []
    while len(out) < n:
        tc = CAM0 + k / 30.0
        if tc > t + 1e-9:
            out.append(tc)
        k += 1
    return out


def sample_for(tc, st):
    """撮像時刻 tc のフレームが使う車両の値のサンプルの時刻(st 刻み。tc 以前の最新)"""
    return round(math.floor((tc + 1e-6) / st) * st, 2)


def test_loop_finding_reproduced_and_fixed(monkeypatch, tmp_path):
    """指摘の入力(場面 rev・回 1 の形・元の偽物の ESC・50 Hz。'brake' の始めから 0.1 秒カメラが来ない + 跳ね 2 つ): 直す前の写しの
    ON(両方 1)は 0 m・escape が前の壁を 0.5 秒押す(再現)。今の ON は back_early → early_none('gap' から)で 1 m より多く下がり、
    escape は前の壁を押さない。OFF(両方 0)は直す前の写しの OFF とビット単位で同じ(1.272 m)"""
    amod, r1 = load_r1()
    sc = scene_rev(dur=9.0, drops=DROP, glitch=GL2)
    bef = run(r1, monkeypatch, tmp_path, P(), sc)
    on = run(drive, monkeypatch, tmp_path, P(), sc)
    off = run(drive, monkeypatch, tmp_path, P(0, 0), sc)
    off_b = run(r1, monkeypatch, tmp_path, P(0, 0), sc)
    mb, mo, mf = first(bef), first(on), first(off)
    assert failed(mb) and mb['rev_dead'] >= 1.2 and F7.R2.guards(bef[1]) == ['back_early']          # 指摘の再現
    assert mo['back'] > 1.0 and mo['esc_push'] == 0.0 and F7.R2.guards(on[1])[:2] == ['back_early', 'early_none']
    assert max(th for t, s, th in on[2].w if 2.4 < t < 4.5) == REV                                 # +50 µs のやり直しにしない
    assert mf['back'] == pytest.approx(1.272, abs=1e-3) and mf['esc_push'] == 0.0
    F5.same_loop(off, off_b)


_TB = {}


def _base(esc, vdt):
    sc = scene_rev(dur=9.0, esc=esc)
    if vdt:
        sc['veh_dt'] = vdt
    return sc


def _tb(monkeypatch, tmp_path, esc, vdt, kw=None):
    key = (esc, vdt, tuple(sorted((kw or {}).items())))
    if key not in _TB:
        _TB[key] = first(run(drive, monkeypatch, tmp_path, P(0, 0, **(kw or {})), _base(esc, vdt)))['t_brake']
    return _TB[key]


def _aligned(tb, a0, aL, j1, j2, st):
    """'brake' の始め tb から a0 秒に aL 秒のカメラの止まり。止まりのあと j1 番目のフレームが拾う跳ね 1 つ目、その j2 フレームあとに
    2 つ目(反証の exp8_align の組)"""
    a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
    fr = frames_after(a[1], 16)
    gl = [(sample_for(fr[j1], st), 1322)]
    if j2:
        gl.append((sample_for(fr[j1 + j2], st), 1322))
    return a, gl


@pytest.mark.parametrize('j2', [1, 3, 8])
@pytest.mark.parametrize('j1', [1, 3])
@pytest.mark.parametrize('a0,aL', [(0.0, 0.10), (0.04, 0.16)])
@pytest.mark.parametrize('esc,vdt', [(None, None), ((0.12, 0.10), None), (None, 0.01), ((0.12, 0.10), 0.01)])
def test_loop_aligned_two_spikes(monkeypatch, tmp_path, esc, vdt, a0, aL, j1, j2):
    """反証の exp8_align の組(止まり 0.10・0.16 秒 × 跳ね 1 つ目の位置 × 2 つ目 1〜8 フレームあと × ESC 2 種 × 50/100 Hz):
    今の ON は 1 m より多く下がり、escape は前の壁を押さない。下がった距離は OFF と 0.3 m より違わない"""
    st = 0.01 if vdt else 0.02
    tb = _tb(monkeypatch, tmp_path, esc, vdt)
    a, gl = _aligned(tb, a0, aL, j1, j2, st)
    sc = dict(_base(esc, vdt), drops=[a], glitch=gl)
    mo = first(run(drive, monkeypatch, tmp_path, P(), sc))
    mf = first(run(drive, monkeypatch, tmp_path, P(0, 0), sc))
    assert mo['back'] > 1.0 and mo['esc_push'] == 0.0, mo
    assert abs(mo['back'] - mf['back']) < 0.3


def test_loop_aligned_reproduced(monkeypatch, tmp_path):
    """上の組の形で、直す前の写しの ON は ESC 2 種 × 50/100 Hz のどれでも 0 m・escape が前の壁を押す(空振りでない)"""
    amod, r1 = load_r1()
    for esc, vdt in ((None, None), ((0.12, 0.10), None), (None, 0.01), ((0.12, 0.10), 0.01)):
        st = 0.01 if vdt else 0.02
        a, gl = _aligned(_tb(monkeypatch, tmp_path, esc, vdt), 0.0, 0.10, 1, 3, st)
        mb = first(run(r1, monkeypatch, tmp_path, P(), dict(_base(esc, vdt), drops=[a], glitch=gl)))
        assert failed(mb), (esc, vdt, mb)


FORMS = {'kai2': {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.2}, 'kai3': {'speed_ceiling': 1, 'launch_boost': 1},
         'kai4': {'launch_boost': 1}}


@pytest.mark.parametrize('vdt', [None, 0.01])
@pytest.mark.parametrize('form', sorted(FORMS))
def test_loop_forms(monkeypatch, tmp_path, form, vdt):
    """回 2〜4 の形(天井・自動合わせ・左 4.2・発進の前進)の同じ入力(止まり 0.10 秒・跳ね 2 つ): 今の ON は 1 m より多く下がり、
    escape は前の壁を押さない。直す前の写しの ON は 0 m・壁押し(再現)"""
    amod, r1 = load_r1()
    kw = FORMS[form]
    st = 0.01 if vdt else 0.02
    a, gl = _aligned(_tb(monkeypatch, tmp_path, None, vdt, kw), 0.0, 0.10, 2, 3, st)
    sc = dict(_base(None, vdt), drops=[a], glitch=gl)
    mo = first(run(drive, monkeypatch, tmp_path, P(**kw), sc))
    mb = first(run(r1, monkeypatch, tmp_path, P(**kw), sc))
    assert mo['back'] > 1.0 and mo['esc_push'] == 0.0 and failed(mb)


BURSTS8 = {
    'b152707_0004': [1628, 0, 1525, 0, 0, 1572, 0, 0, 1289],
    'b211828_0004': [1687, 0, 0, 0, 0, 0, 1411, -1270, 1408, -2156, 1356, 0, 0, -394, 1528, 1358, 656, 385],
}


@pytest.mark.parametrize('esc', [None, (0.12, 0.10)])
@pytest.mark.parametrize('s0', [0.02, 0.06])
@pytest.mark.parametrize('name', sorted(BURSTS8))
def test_loop_real_bursts_after_short_hold(monkeypatch, tmp_path, name, s0, esc):
    """'brake' の中の短い止まり(0.10 秒)のあと、本物の記録の跳ねの並び(100 Hz): 今の ON は 1 m より多く下がり、escape は
    前の壁を押さない"""
    tb = _tb(monkeypatch, tmp_path, esc, 0.01)
    a = (round(tb, 3), round(tb + 0.10, 3))
    t1 = round(a[1] + s0, 2)
    gl = [(round(t1 + 0.01 * i, 2), e) for i, e in enumerate(BURSTS8[name]) if e]
    mo = first(run(drive, monkeypatch, tmp_path, P(), dict(_base(esc, 0.01), drops=[a], glitch=gl)))
    assert mo['back'] > 1.0 and mo['esc_push'] == 0.0, mo


@pytest.mark.parametrize('rear', [None, 1.35])
@pytest.mark.parametrize('vdt', [None, 0.01])
@pytest.mark.parametrize('esc', [None, (0.12, 0.10)])
def test_loop_brake_scene_identical_r1(monkeypatch, tmp_path, esc, vdt, rear):
    """場面 brake(ESC が後退を待っている 10/23 の形。'brake' の後退側がそのまま後退になり back_early): 今の ON は直す前の写しの
    ON とビット単位で同じ(本当に下がっているので s4-c の道は通らない)。跳ね・'brake' の中の短い止まりがあっても同じ。見張りだけ
    (keep 0)も同じ"""
    amod, r1 = load_r1()
    base = scene_brake(rear=rear, esc=esc, veh_dt=vdt)
    for extra in ({}, {'glitch': [(2.70, 1322)]}, {'glitch': [(2.78, 1322)]}, {'drops': [(2.75, 2.85)]},
                  {'drops': [(2.75, 2.85)], 'glitch': [(2.92, 1322)]}):
        sc = dict(base, **extra)
        for keep in (1, 0):
            on = run(drive, monkeypatch, tmp_path, P(1, keep), sc)
            F5.same_loop(on, run(r1, monkeypatch, tmp_path, P(1, keep), sc))
            if keep == 1 and not extra:
                assert F7.R2.guards(on[1])[:1] == ['back_early']


OFF_SCENES = {
    'rev': scene_rev(),
    'finding': scene_rev(dur=9.0, drops=DROP, glitch=GL2),
    'finding_100hz': scene_rev(dur=9.0, veh_dt=0.01, esc=(0.12, 0.10), drops=[(2.52, 2.64)], glitch=[(2.69, 1322), (2.85, 1322)]),
    'burst_hold': scene_rev(dur=9.0, veh_dt=0.01, drops=[(2.52, 2.62)],
                            glitch=[(round(2.66 + 0.01 * i, 2), e) for i, e in enumerate(BURSTS8['b152707_0004']) if e]),
    'brake': scene_brake(rear=1.35),
    'brake_hold': scene_brake(drops=[(2.75, 2.85)], glitch=[(2.92, 1322)]),
    'esc_short': scene_rev(dur=9.0, esc=(0.02, 0.08), drops=DROP, glitch=GL2, xr=XF_REV - 1.35),
}


@pytest.mark.parametrize('name', sorted(OFF_SCENES))
@pytest.mark.parametrize('extra', [{}, {'launch_boost': 1, 'speed_ceiling': 1}, {'rec_hold_keep': 1}])
def test_loop_off_bit_identical_r1(monkeypatch, tmp_path, name, extra):
    """見張り OFF(rec_back_guard = 0。rec_hold_keep 1 を送っても走行ループは両方 1 のときだけ残す)の走行ループは、直す前の写しと
    PCA9685 に書く値・記録の行・出来事がビット単位で同じ"""
    amod, r1 = load_r1()
    p = P(0, 0, **extra)
    sc = OFF_SCENES[name]
    F5.same_loop(run(drive, monkeypatch, tmp_path, p, sc), run(r1, monkeypatch, tmp_path, p, sc))


@pytest.mark.parametrize('rear', [None, 1.35])
@pytest.mark.parametrize('neu', [0.02, 0.03])
@pytest.mark.parametrize('two', [False, True])
def test_loop_short_neutral_esc_identical(monkeypatch, tmp_path, neu, rear, two):
    """中立 0.02・0.03 秒で下がれる偽物の ESC(本物は人の操作で中立 0.05 秒でも後退に入った)で、'brake' の中の短い止まり
    (約 0.03 秒の中立で本当に下がり始める)。今の ON は直す前の写しの ON とビット単位で同じ(本当に下がるので s4-c の道は
    通らない)。見張りはこの 'brake' の後退を数える(back_early。EscMayReverse を中立 0.08 秒にする案 (a) では数えず、後ろへ多く
    下がる)"""
    amod, r1 = load_r1()
    sc = scene_rev(dur=9.0, esc=(neu, 0.08), drops=DROP, glitch=GL2 if two else [])
    if rear is not None:
        sc['xr'] = XF_REV - rear
    on = run(drive, monkeypatch, tmp_path, P(), sc)
    F5.same_loop(on, run(r1, monkeypatch, tmp_path, P(), sc))
    assert F7.R2.guards(on[1])[:1] == ['back_early']


@pytest.mark.parametrize('rear', [None, 1.5])
def test_loop_neutral_005_esc(monkeypatch, tmp_path, rear):
    """中立 0.05 秒で下がれる偽物の ESC(本物で後退に入った中立のうち短い方の値)では、約 0.03 秒の中立止めではまだ下がれない。
    跳ねなし: 今の ON は直す前とビット単位で同じ。跳ね 2 つ(指摘の入力): 直す前の写しは 0 m・escape が前の壁を押す(この ESC でも
    指摘の形が起きる)。今の ON は 1 m より多く下がり、escape は前の壁を押さない"""
    amod, r1 = load_r1()
    sc = scene_rev(dur=9.0, esc=(0.05, 0.08), drops=DROP)
    if rear is not None:
        sc['xr'] = XF_REV - rear
    F5.same_loop(run(drive, monkeypatch, tmp_path, P(), sc), run(r1, monkeypatch, tmp_path, P(), sc))
    sc2 = dict(sc, glitch=GL2)
    mo = first(run(drive, monkeypatch, tmp_path, P(), sc2))
    mb = first(run(r1, monkeypatch, tmp_path, P(), sc2))
    assert failed(mb) and mo['back'] > 1.0 and mo['esc_push'] == 0.0


@pytest.mark.parametrize('seed', range(3))
def test_loop_fuzz_off_identical_and_on_rules(monkeypatch, tmp_path, seed):
    """でたらめな閉じたループ(tests/test_rec_fix5_safety の場面と決まり)に、短いカメラの止まりと車両の値の跳ねを足して: 見張り OFF は
    直す前の写しとビット単位で同じ。今の ON(両方 1)は fix5 の決まり(check_rules)を破らない"""
    amod, r1 = load_r1()
    rng = random.Random(7200 + seed)
    for i in range(4):
        sc = F5._rand_scene(rng)
        p1 = F5._rand_params(rng)
        for _ in range(3):
            a = round(rng.uniform(1.5, sc['dur'] - 0.5), 2)
            sc['drops'].append((a, round(a + rng.choice([0.10, 0.12, 0.14]), 3)))
        sc['glitch'] = [(round(rng.uniform(1.5, sc['dur']), 2), rng.choice([301, 313, 1322, 1687])) for _ in range(10)]
        sc['veh_dt'] = rng.choice([0.01, 0.02])
        p0 = dict(p1, rec_back_guard=0, rec_hold_keep=0)
        F5.same_loop(run(drive, monkeypatch, tmp_path, p0, sc), run(r1, monkeypatch, tmp_path, p0, sc))
        p1 = dict(p1, rec_back_guard=1, rec_hold_keep=1)
        s, rec, pca = run(drive, monkeypatch, tmp_path, p1, sc)
        bad = F5.check_rules(s, rec, pca, p1)
        assert not bad, bad[:3]
