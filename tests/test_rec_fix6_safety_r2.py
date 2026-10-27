"""10/27 fix6 係 safety 直す 6-2 回目(反証 6-2 回目の指摘 w2-safety-r62-safety-1・w2-safety-r62-safety-2)の試験。走行ループ・補助の側。

  指摘 1(w2-safety-r62-safety-1): rec_hold_keep = 1・rec_back_guard = 1・既定の rev_s 0.4・rev_us 300・元の偽物の ESC
    (tests/test_fix3_r2_1.Car。ブレーキのあと中立 0.1 秒の決まりを 10 ms 刻みで数える)。最初のリカバリーの 'brake' の途中の
    カメラの止まり(2.11〜2.26 秒)で 0.10 秒の中立止め → 偽物の ESC はまだ下がれないのに、fix6 r1 の EscTrack は中立 0.08 秒で
    「下がれる」とし、続く 'brake' の後退側を後退とみなす。そこで車両の値が止まる(2.26〜2.45 秒)と、明けたフレームの見積もり
    hold_rev_s 0.07 秒だけで 'brake' から 'gap' の中立を飛ばして 'reverse' へ進み(back_hold)、backed_s > 0 なので CHECK_S の
    やり直しも効かない。1796 µs を 1.2 秒書いても下がらず(0 m)、escape が前の壁を約 0.5 秒押した。直す前の写し(fix5 のあと)は
    1.272 m 下がる。
  指摘 2(w2-safety-r62-safety-2): ブレーキのあと中立 0.2 秒で下がれる偽物の ESC(最初の 'gap' 0.15 秒では下がれず、やり直しの
    0.3 秒で下がる)。'reverse' の始めに車両の値が 0.2 秒止まるだけで、見積もり 0.11 秒が backed_s に入り、やり直しが止まる。
    0 m のまま escape が前の壁を押した(直す前の写しはやり直して 1.383 m)。
  直し s3-c(assist.Recovery だけ。両方 1 の日だけ効く。drive.py・EscTrack は変えていない):
    ・見積もりで足した秒を est_s に別に持つ(backed_s には今まで通り足す。後退を 0.2 秒残すところまで)
    ・見積もりだけでは段を進めない('brake'・'wait' → 'reverse'・'coast'、'gap' → 'coast' は後ろへ動いていたサンプルのときだけ)
    ・'reverse' のやり直し(CHECK_S)は見た後退(backed_s − est_s)が 0 かで決め、やり直すときは見積もりを捨てる
    ・見積もりそのものは捨てない(外れていたら、そのぶん後退が短くなるだけ。後退を 0.2 秒残す)。見ていない間に本当に下がって
      止まった車と見分けられず、捨てると fix6 r1 の指摘の形(値の止まりなしより多く下がる)に戻るため(test_r61_window_still_fixed)

OFF(rec_hold_keep・rec_back_guard のどちらかが 0。今の既定)は、直す前の写し tests/fixtures/fix6_safety_r2_before/(assist.py・
drive.py = fix6 r1 のあと。sha256 は SHA256SUMS)とビット単位で同じ(PCA9685 に書く値・記録の行・出来事)。fix5 のあとの写しとの
ビット単位の同じは tests/test_rec_fix6_safety.py が見る。指摘の場面の試験は「直す前の写し(fix6 r1)で指摘が起きる(再現)」を同じ
試験の中で見る。走行ループは tests/test_fix3_r2_1 の閉じたループ(本物の drive._loop を、書いた値で動く偽物の ESC と車で回す)。
実機・GPU・torch・jetformer-web は使わない。台本と表: experiments/20261026_fix/fix6/safety_r2/。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_rec_fix6_safety_r2.py
"""
import copy
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
from test_launch_drive import C, N, PARAMS

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix6_safety_r2_before'
SHA_BEFORE = {'assist.py': 'd1d62c66bef856002c14db8dd2fd9c35279a36878b2e4817a279e9fb8bf64165',     # 10/27 fix6 safety r1 のあと
              'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec'}
DT = R2.DT
FDT = 1 / 30.0
MODEL = (C, N - 120.0)
REV, GAP = N + 300.0, N + A.GAP_OFFSET_US
RETRY = N + 300.0 + A.Recovery.RETRY_EXTRA_US
ESCAPE = N - A.Recovery.ESCAPE_US
STATE = R2.STATE + ('t_check',)
# この組のあと 10/27 fix7 safety の直す 7-1 回目(反証 w1-safety-rv7-safety-r1-1)が assist.py・drive.py を変えた。そのときは、その直す前の
# 写し tests/fixtures/fix7_safety_before/(= この組のあとの版。sha256 を照らす)をこの組の版として照らす(7-1 回目の違いは
# tests/test_rec_fix7_safety.py が見る。tests/test_fix3_r3_2_tools.py の LATER_FIX7_SAFETY と同じ値)
FIX6_R2_AFTER = {'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4',
                 'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec'}
LATER_FIX7_SAFETY = {'assist.py': '6a8950d9963bb4a663dd540cb4b819f13f22f9da956a56a657c98a66ea717e3f',
                     'drive.py': '6f3c437f5e9729cf5260904dbc31986c80b7dd5c7565510bef0b4c2a9b091e0f'}
# さらに 10/27 fix7 safety の直す 7-2 回目(反証 w2-safety-rv7-safety-r2-1)が assist.py を変えた(drive.py は同じ)。その直す前の写し
# tests/fixtures/fix7_safety_r2_before/(= 7-1 回目のあとの版)から同じようにたどる(違いは tests/test_rec_fix7_safety_r2.py。
# tests/test_fix3_r3_2_tools.py の LATER_FIX7_SAFETY_R2 と同じ値)
LATER_FIX7_SAFETY_R2 = {'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}


def _fix6r2_version(name):
    """この組のあとの版(jetformer/ の今の版。fix7 safety の 7-1 回目が変えたあとは、その直す前の写し = この組のあとの版)"""
    p = ROOT / 'jetformer' / name
    if name in LATER_FIX7_SAFETY_R2 and sha(p) == LATER_FIX7_SAFETY_R2[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix7_safety_r2_before' / name
    if sha(p) == LATER_FIX7_SAFETY[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix7_safety_before' / name
    assert sha(p) == FIX6_R2_AFTER[name], name
    return p


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_BEFORE = {}


def load_before():
    """直す前(fix6 r1 のあと)の assist.py・drive.py(写し)をモジュールにする(sha256 を照らしてから)"""
    if _BEFORE:
        return _BEFORE['a'], _BEFORE['d']
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sums == SHA_BEFORE
    for name, h in sums.items():
        assert sha(FIX / name) == h, name
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix6r2s', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix6r2s', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    assert hasattr(amod, 'EscTrack') and not hasattr(amod.Recovery((C, N)), 'est_s')      # 写しは fix6 r1 のあと・直す前
    _BEFORE.update(a=amod, d=mod)
    return amod, mod


def fix5():
    """fix5 のあとの写し(tests/fixtures/fix6_safety_before。指摘の「直す前の写し」。sha256 は test_rec_fix6_safety が照らす)"""
    return T6.load_before()


# ================================================================ 0. 写し・既定値・変えた所
def test_before_copy_defaults_and_changed_parts():
    """写しの sha256(= fix6 r1 のあとの版)。drive.py は変えていない(写しと同じ sha256)。既定値・Recovery の定数・EscTrack の定数・
    モジュールの定数は同じ。変えたのは assist.py の Recovery.reset・Recovery._start・Recovery._after_hold・Recovery.step だけ
    (足した・消した関数や文は無い。AST を名前で照らす)。10/27 fix7 safety(7-1 回目)が変えたあとは、その直す前の写し(= この組の
    あとの版)で照らす(_fix6r2_version)"""
    amod, before = load_before()
    assert sha(_fix6r2_version('drive.py')) == SHA_BEFORE['drive.py']
    assert drive.DEFAULT_PARAMS == before.DEFAULT_PARAMS and drive.PARAM_LIMITS == before.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == before.RESET_ON_BOOT
    for k in ('rec_back_guard', 'rec_hold_keep'):
        assert drive.DEFAULT_PARAMS[k] == 0 and k in drive.RESET_ON_BOOT
    for cls in ('Recovery', 'EscTrack', 'EscReverse'):
        assert {k: v for k, v in vars(getattr(A, cls)).items() if k.isupper()} == \
            {k: v for k, v in vars(getattr(amod, cls)).items() if k.isupper()}, cls
    for k in ('BACKING_ERPM', 'CHECK_S', 'STOPPED_ERPM', 'GAP_OFFSET_US', 'DEFAULT_REV_US', 'DEFAULT_REV_S',
              'HOLD_EST_MIN_LEFT_S'):
        assert getattr(A, k) == getattr(amod, k)
    assert A.EscTrack.READY_NEUTRAL_S == 0.08 and A.EscTrack.DELAY_S == 0.06
    a0, a1 = T6._defs(FIX / 'assist.py'), T6._defs(_fix6r2_version('assist.py'))
    assert set(a0) == set(a1)
    assert sorted(k for k in a0 if a0[k] != a1[k]) == ['Recovery._after_hold', 'Recovery._start', 'Recovery.reset',
                                                         'Recovery.step']
    doc = A.__doc__
    assert '反証 6-2 回目の直し' in doc and 's3-c' in doc and '反証 6-2 回目の直し' not in amod.__doc__


# ================================================================ 1. Recovery(単体)
def _phase(mod, phase):
    """'brake'・'wait'・'gap'・'reverse'(まだ下がっていない)・中立止めの直前のフレームまで"""
    r, t = F5._to_phase(mod, phase)
    assert r.backed_s == 0.0
    return r, t


def _state(r):
    return tuple(getattr(r, k) for k in STATE)


@pytest.mark.parametrize('rev_s', [0.3, 0.4, 0.6])
@pytest.mark.parametrize('est', [0.05, 0.11, 0.3])
@pytest.mark.parametrize('phase', ['brake', 'wait', 'gap', 'reverse'])
def test_unit_estimate_alone_does_not_change_phase(phase, est, rev_s):
    """後ろへ動いていたサンプルの無い 0.2 秒の中立止めと、見ていない間の後退の見積もり est: 今は下がった時間に min(est, rev_s − 0.2)
    を足し、同じ秒を est_s に持つ。段と出す値は、見積もりが無いときと同じ('brake'・'wait' から 'reverse' へ進めない = 出来事なし)。
    直す前の写し(fix6 r1)は 'brake'・'wait' から 'reverse' へ進めた(back_hold。指摘 1 の仕組み)"""
    amod, _ = load_before()
    r, t = _phase(A, phase)
    r1, t1 = _phase(amod, phase)
    assert t == t1
    rx = copy.deepcopy(r)
    kw = dict(back_guard=True, rev_s=rev_s, hold_since=t + 0.1, hold_back_s=0.0)
    o, k = r.step(MODEL, 0.0, t + 0.3, hold_rev_s=est, **kw)
    ox, kx = rx.step(MODEL, 0.0, t + 0.3, hold_rev_s=0.0, **kw)
    o1, k1 = r1.step(MODEL, 0.0, t + 0.3, hold_rev_s=est, **kw)
    want = min(est, max(0.0, rev_s - A.HOLD_EST_MIN_LEFT_S))
    assert r.backed_s == pytest.approx(want, abs=1e-12) and r.est_s == r.backed_s
    assert (r.phase, o, k, r.guard_event) == (rx.phase, ox, kx, None)
    assert r1.backed_s == r.backed_s
    if phase in ('brake', 'wait'):
        assert r1.phase == 'reverse' and r1.guard_event[0] == 'back_hold'        # 直す前: 'gap' の中立を飛ばした
        assert r.phase in (phase, 'gap')
    else:
        assert r1.phase == r.phase


@pytest.mark.parametrize('est', [0.0, 0.05])
@pytest.mark.parametrize('phase', ['brake', 'wait', 'gap', 'reverse'])
def test_unit_seen_backing_same_as_before(phase, est):
    """中立止めの間に後ろへ動いていたサンプルがあった(hold_back_s = 0.12 秒): 段の進め方・出す値・状態は直す前の写しと同じ
    ('brake'・'wait' はすぐ 'reverse'。出来事 back_hold)。続けて後ろへ動けば、直す前と同じフレームで惰性へ"""
    amod, _ = load_before()
    r, t = _phase(A, phase)
    r1, _ = _phase(amod, phase)
    kw = dict(back_guard=True, rev_s=0.4, hold_since=t + 0.1, hold_back_s=0.12, hold_rev_s=est)
    assert r.step(MODEL, 0.0, t + 0.3, **kw) == r1.step(MODEL, 0.0, t + 0.3, **kw)
    assert _state(r) == _state(r1) and r.guard_event == r1.guard_event
    if phase in ('brake', 'wait'):
        assert r.phase == 'reverse' and r.guard_event[0] == 'back_hold'
    for i in range(20):
        tt = t + 0.3 + (i + 1) * FDT
        assert r.step(MODEL, 2500.0, tt, back_guard=True) == r1.step(MODEL, 2500.0, tt, back_guard=True), i
        assert _state(r) == _state(r1), i


def _seq(r, t0, n, erpm, **kw):
    out = []
    for i in range(n):
        out.append(r.step(MODEL, erpm, t0 + (i + 1) * FDT, back_guard=True, **kw))
    return out


@pytest.mark.parametrize('est', [0.05, 0.11, 0.2])
def test_unit_reverse_retry_with_estimate(est):
    """指摘 2 の仕組み: 'reverse' で、後ろへ動いていたサンプルの無い中立止め + 見積もり est のあと、車が止まったまま(ESC が後退に
    入らない)。今は明けたフレームから CHECK_S でやり直し('gap' 0.3 秒 → +50 µs)、やり直すとき見積もりを捨てる。出す値は
    fix5 のあとの写し(見積もりを知らない)と全部同じ。直す前の写し(fix6 r1)はやり直さず、後退側を MAX_REV_TOTAL_S 出したあと
    escape(前進 130 µs)"""
    amod, _ = load_before()
    a5, _ = fix5()
    r, t = _phase(A, 'reverse')
    r1, _ = _phase(amod, 'reverse')
    r5, _ = _phase(a5, 'reverse')
    t1 = t + 0.3
    kw = dict(back_guard=True, rev_s=0.4, hold_since=t + 0.1, hold_back_s=0.0)
    o = [r.step(MODEL, 0.0, t1, hold_rev_s=est, **kw)]
    o1 = [r1.step(MODEL, 0.0, t1, hold_rev_s=est, **kw)]
    o5 = [r5.step(MODEL, 0.0, t1, **kw)]
    assert r.est_s == r.backed_s == est
    seen = []
    for i in range(60):
        tt = t1 + (i + 1) * FDT
        o.append(r.step(MODEL, 0.0, tt, back_guard=True))
        o1.append(r1.step(MODEL, 0.0, tt, back_guard=True))
        o5.append(r5.step(MODEL, 0.0, tt, back_guard=True))
        if r.retries and not seen:
            seen.append(tt)
            assert r.phase == 'gap' and r.backed_s == 0.0 and r.est_s == 0.0
            assert tt - t1 > A.CHECK_S and tt - FDT - t1 <= A.CHECK_S          # 明けたフレームから CHECK_S
    assert o == o5                                                          # fix5 のあとと同じ
    assert seen and any(x[0] is not None and x[0][1] == RETRY for x in o)    # やり直しの +50 µs
    assert r1.retries == 0 and not any(x[0] is not None and x[0][1] == RETRY for x in o1)     # 直す前: やり直さない
    assert any(x[0] is not None and x[0][1] == ESCAPE for x in o1)          # 直す前: 下がらないまま escape(前進)


@pytest.mark.parametrize('est', [0.05, 0.11, 0.2])
def test_unit_reverse_estimate_then_backing_same_as_before(est):
    """'reverse' の中立止め + 見積もり est のあと、ESC が後退に入って後ろへ動く(fix6 r1 の指摘の形 = 見積もりが当たっていた):
    やり直さず、見積もりを足したまま rev_s に届いたら惰性。出す値・状態は直す前の写し(fix6 r1)と同じ(est_s のほか)"""
    amod, _ = load_before()
    r, t = _phase(A, 'reverse')
    r1, _ = _phase(amod, 'reverse')
    kw = dict(back_guard=True, rev_s=0.4, hold_since=t + 0.1, hold_back_s=0.0, hold_rev_s=est)
    assert r.step(MODEL, 0.0, t + 0.3, **kw) == r1.step(MODEL, 0.0, t + 0.3, **kw)
    o, o1 = _seq(r, t + 0.3, 30, 2000.0), _seq(r1, t + 0.3, 30, 2000.0)
    assert o == o1 and _state(r) == _state(r1)
    n_rev = sum(1 for x in o if x[0] is not None and x[0][1] == REV)
    assert r.retries == 0 and abs(n_rev * FDT - (0.4 - est)) <= FDT + 1e-9


@pytest.mark.parametrize('still', [0.0, -100.0])
@pytest.mark.parametrize('est', [0.05, 0.11, 0.2])
def test_unit_gap_estimate_kept_shorter_reverse(est, still):
    """'gap' の中立止め + 見積もり est のあと、後ろへ動かない(eRPM が 0 か前へ = still)まま 'reverse' へ: 見積もりは捨てない
    (見ていない間に下がって止まった車と見分けられない。捨てると多く下がる側)。出す値・状態は直す前の写し(fix6 r1)と同じで、
    fix5 のあとの写し(見積もりを知らない)より見積もりの分だけ後退が短い(この組で残す形。後退は 0.2 秒より短くならない)"""
    amod, _ = load_before()
    a5, _ = fix5()
    r, t = _phase(A, 'gap')
    r1, _ = _phase(amod, 'gap')
    r5, _ = _phase(a5, 'gap')
    kw = dict(back_guard=True, rev_s=0.4, hold_since=t + 0.05, hold_back_s=0.0)
    o = [r.step(MODEL, still, t + 0.1, hold_rev_s=est, **kw)]
    o1 = [r1.step(MODEL, still, t + 0.1, hold_rev_s=est, **kw)]
    o5 = [r5.step(MODEL, still, t + 0.1, **kw)]
    t_rev = None
    for i in range(40):
        tt = t + 0.1 + (i + 1) * FDT
        erpm = still if r.phase in ('gap',) or t_rev is None or tt - t_rev < 0.15 else 2000.0
        o.append(r.step(MODEL, erpm, tt, back_guard=True))
        o1.append(r1.step(MODEL, erpm, tt, back_guard=True))
        o5.append(r5.step(MODEL, erpm, tt, back_guard=True))
        if t_rev is None and r.phase == 'reverse':
            t_rev = tt
            assert r.backed_s == r.est_s == r1.backed_s == est
    assert t_rev is not None and o == o1 and _state(r) == _state(r1)
    n, n5 = (sum(1 for x in oo if x[0] is not None and x[0][1] == REV) for oo in (o, o5))
    assert n5 > n and abs((n5 - n) * FDT - est) <= FDT + 1e-9 and n * FDT >= A.HOLD_EST_MIN_LEFT_S - 1e-9


@pytest.mark.parametrize('erpm_after', [60.0, 270.0, 320.0, 800.0])
def test_unit_gap_estimate_with_backing_kept(erpm_after):
    """'gap' の中立止め + 見積もり 0.11 秒の明けたフレームで後ろへ動いている(eRPM > 0。BACKING_ERPM より遅くても。見積もりが
    当たっていた): 見積もりは捨てない。出す値・状態は直す前の写し(fix6 r1)と同じ(est_s のほか)"""
    amod, _ = load_before()
    r, t = _phase(A, 'gap')
    r1, _ = _phase(amod, 'gap')
    kw = dict(back_guard=True, rev_s=0.4, hold_since=t + 0.05, hold_back_s=0.0, hold_rev_s=0.11)
    assert r.step(MODEL, erpm_after, t + 0.1, **kw) == r1.step(MODEL, erpm_after, t + 0.1, **kw)
    for i in range(30):
        tt = t + 0.1 + (i + 1) * FDT
        e = 250.0 if i < 4 else 2000.0                                       # 中立の惰性で遅くなってから、後退で速く
        assert r.step(MODEL, e, tt, back_guard=True) == r1.step(MODEL, e, tt, back_guard=True), i
        assert _state(r) == _state(r1), i
    assert r.est_s == pytest.approx(0.11)


@pytest.mark.parametrize('first', [0.0, 100.0])
def test_unit_brake_estimate_slow_backing_then_gap_kept(first):
    """fix6 r1 の指摘の形('brake' の途中の車両の値の止まりで、ESC は見ていない間に後退で走らせ、明けたときには止まっている。
    残りの 'brake' の後退側で 100 eRPM ほど後ろへ動いたところで 'gap' へ): 見積もりは捨てず、'reverse' は見積もりを足したまま
    (rev_s をまるごと出し直すと、値の止まりなしより多く下がる)。明けたフレームで動いていなくてもよい"""
    r, t = _phase(A, 'brake')
    r.step(MODEL, first, t + 0.2, back_guard=True, rev_s=0.4, hold_since=t + 0.05, hold_back_s=0.0, hold_rev_s=0.11)
    assert r.phase == 'brake' and r.backed_s == 0.11
    tt, seen_rev = t + 0.2, None
    for i in range(30):
        tt += FDT
        r.step(MODEL, 100.0 if r.phase == 'brake' else 0.0, tt, back_guard=True, rev_s=0.4)
        if r.phase == 'reverse' and seen_rev is None:
            seen_rev = (r.backed_s, r.est_s)
    assert seen_rev == (0.11, 0.11), seen_rev


def test_unit_brake_estimate_then_back_early_kept():
    """'brake' の中立止め + 見積もり 0.08 秒(ESC が後退を待っていた 'brake' = 見積もりが当たっていた)のあと、後退側で後ろへ動く:
    見張りの back_early で 'reverse' へ進み、見積もりは足したまま(rev_s − 0.08 秒の見た後退で惰性へ)"""
    r, t = _phase(A, 'brake')
    r.step(MODEL, 0.0, t + 0.2, back_guard=True, rev_s=0.4, hold_since=t + 0.05, hold_back_s=0.0, hold_rev_s=0.08)
    assert r.phase == 'brake' and r.backed_s == 0.08
    o = _seq(r, t + 0.2, 30, 1500.0)
    assert r.guard_event is None or r.guard_event[0] == 'back_early'
    n_rev = sum(1 for x in o if x[0] is not None and x[0][1] == REV)
    assert r.est_s == 0.08 and r.retries == 0 and abs(n_rev * FDT - (0.4 - 0.08)) <= 2 * FDT + 1e-9


def _closed_fuzz(seed, n=6000):
    """でたらめな閉じたループ(Recovery 単体): 今の Recovery の出力で偽物の ESC(R2.Car・壁なし)を動かし、同じ入力を直す前の写し
    (fix6 r1)にも渡す。ときどき残した中立止め(0.1〜0.45 秒。その間は中立)を入れ、明けたフレームに hold_since・後ろへ動いていた秒
    hold_back_s(ときどき 0)・見積もり hold_rev_s(奇数の種は 0)を渡す。戻り値: (違いが出たフレームの数, 見積もりのある中立止めの数,
    決まりを破ったもの)。違いが出たら両方 reset して続ける"""
    amod, _ = load_before()
    rng = random.Random(64000 + seed)
    car = R2.Car({'xf': 1e9})
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    t, t_car, thr = 100.0, 0.0, N
    model, model_left = (C, N - 120.0), 0.0
    diff, n_est, bad = 0, 0, []

    def drive_car(dur, cmd):
        nonlocal t_car
        back = 0.0
        for _ in range(int(round(dur / DT))):
            t_car = round(t_car + DT, 6)
            car.step(t_car, cmd)
            if -car.v / R2.KV > A.BACKING_ERPM:
                back += DT
        return back

    for _ in range(n):
        if model_left <= 0.0:
            model = (C + rng.choice([0.0, 150.0, -200.0]), N - rng.choice([120.0, 200.0]) if rng.random() < 0.5 else N)
            model_left = rng.uniform(0.5, 2.5)
        kw = {}
        if rng.random() < 0.08:
            L = rng.choice([0.1, 0.15, 0.2, 0.3, 0.45])
            t_prev = t
            back = drive_car(0.03, thr) + drive_car(L, N)
            t += 0.03 + L
            kw = dict(hold_since=t_prev + 0.03, hold_back_s=back if rng.random() < 0.5 else 0.0,
                      hold_rev_s=0.0 if seed % 2 else rng.choice([0.0, 0.05, 0.1, 0.15]))
        else:
            drive_car(FDT, thr)
            t += FDT
        model_left -= FDT
        erpm = -car.v / R2.KV
        rx = copy.deepcopy(r1) if kw.get('hold_rev_s', 0.0) > 0.0 else None
        ret0 = r1.retries
        a = r0.step(model, erpm, t, 0.0, back_guard=True, **kw)
        b = r1.step(model, erpm, t, 0.0, back_guard=True, **kw)
        if rx is not None:
            n_est += 1
            bx = rx.step(model, erpm, t, 0.0, back_guard=True, **dict(kw, hold_rev_s=0.0))
            if (b, r1.phase, r1.guard_event) != (bx, rx.phase, rx.guard_event):     # 見積もりだけでは段・出す値を変えない
                bad.append(('est_changed_phase', t, r1.phase, rx.phase))
        if not (0.0 <= r1.est_s <= r1.backed_s + 1e-12):
            bad.append(('est_range', t, r1.est_s, r1.backed_s))
        if r1.retries > ret0 and (r1.backed_s != 0.0 or r1.est_s != 0.0):
            bad.append(('retry_kept_est', t, r1.backed_s, r1.est_s))
        same = a == b and _state(r0) == _state(r1) and r0.guard_event == r1.guard_event
        if seed % 2 and not same:
            bad.append(('no_est_differs', t, r0.phase, r1.phase))
        if not same:
            diff += 1
            r0.reset()
            r1.reset()
        r0.guard_event = r1.guard_event = None
        if rng.random() < 0.003:
            r0.reset()
            r1.reset()
        thr = b[0][1] if b[0] is not None else model[1]
    return diff, n_est, bad


_FUZZ = {}


@pytest.mark.parametrize('seed', range(12))
def test_unit_closed_fuzz(seed):
    """でたらめな閉じたループ: 見積もりを渡さない(奇数の種)なら、直す前の写し(fix6 r1)と出す値・状態が全部同じ。見積もりを渡す
    (偶数の種)なら、見積もりのある中立止めの明けたフレームの段・出す値・出来事は見積もり 0 のときと同じ・0 ≤ est_s ≤ backed_s・
    やり直したら backed_s = est_s = 0"""
    diff, n_est, bad = _closed_fuzz(seed)
    assert bad == [], (seed, bad[:5])
    _FUZZ[seed] = (diff, n_est)
    if seed % 2:
        assert diff == 0
    else:
        assert n_est >= 20 and diff >= 1, (seed, diff, n_est)     # 空振りしていない(見積もりの形が出た)


@pytest.mark.parametrize('seed', range(4))
def test_unit_without_hold_bit_identical(seed):
    """hold_since を渡さない(中立止めで残さない = OFF・人側・dry run)Recovery は直す前の写し(fix6 r1)と全部同じ"""
    amod, _ = load_before()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    for model, erpm, t, ig, hg, bg, rev_us, rev_s in R2._rec_fuzz_steps(100 + seed):
        assert r0.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg) == \
            r1.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg)
        assert _state(r0) == _state(r1) and r0.guard_event == r1.guard_event
        assert r1.est_s == 0.0
        r0.guard_event = r1.guard_event = None


# ================================================================ 2. 走行ループ(閉じたループ。本物の drive._loop)
def esc_car(neu=None, rdelay=0.08):
    """偽物の ESC と車。neu = None なら元の R2.Car(中立 0.1 秒の決まりを 10 ms 刻みで数える = 11 刻み)。neu を渡すと、ブレーキのあと
    neu 秒の中立で下がれ、後退側から rdelay 秒で下がり始める偽物(ほかは R2.Car と同じ。反証の tf62.Car と同じ形で 1e-9 の余裕つき)。
    どちらも後退で走らせた刻みを revlog に残す"""
    if neu is None:
        return T6.LogCar

    class EscCar(T6.LogCar):
        def step(self, t, thr):
            s = self.s
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
            self.revlog.append((t, self.rev_on))
            return acc, g
    return EscCar


def run(mod, monkeypatch, tmp_path, params, scene, neu=None, rdelay=0.08):
    monkeypatch.setattr(R2, 'Car', esc_car(neu, rdelay))
    return F5.run(mod, monkeypatch, tmp_path, params, scene)


def first_rec(res, rev_us=300.0):
    m = T6.first_rec(res, rev_us)
    sim, rec, pca = res
    end = m['t_escape'] if m['t_escape'] is not None else m['t_brake'] + 3.0
    m['retry'] = any(abs(th - (N + rev_us + A.Recovery.RETRY_EXTRA_US)) < 1e-6 for t, s, th in pca.w
                     if m['t_brake'] <= t <= end)
    return m


def P1(**kw):
    return T6.P1(**kw)


def scene_rev(skip=None, drop=None, dur=7.0):
    return T6.scene('rev', skip=skip, drop=drop, dur=dur)


# 指摘 1 の場面(元の偽物の ESC): 反証の 480 組のうち、直す前の写し(fix6 r1)で「0 m・escape の壁押し」になった 12 組
F1_CELLS = [((2.09, 2.26), (2.26, 2.43)), ((2.09, 2.26), (2.26, 2.46)), ((2.09, 2.26), (2.29, 2.46)),
            ((2.09, 2.26), (2.29, 2.49)), ((2.09, 2.26), (2.29, 2.54)), ((2.09, 2.28), (2.28, 2.45)),
            ((2.09, 2.28), (2.28, 2.48)), ((2.12, 2.29), (2.29, 2.46)), ((2.12, 2.29), (2.29, 2.49)),
            ((2.12, 2.29), (2.29, 2.54)), ((2.15, 2.3), (2.3, 2.47)), ((2.15, 2.3), (2.3, 2.5))]


def _no_skip(res):
    """最初のリカバリーで 'brake'・'wait' から 'reverse' へ飛んでいない(出来事 back_hold・back_early が無い)"""
    return not [e for e in res[1].events if e[0] == 'recovery_guard' and e[1]['what'] in ('back_hold', 'back_early')]


def test_finding1_scene(monkeypatch, tmp_path):
    """指摘 1 の入力そのまま(カメラの止まり 2.11〜2.26 秒・車両の値の止まり 2.26〜2.45 秒): 今は 'gap' の中立を飛ばさず、偽物の
    ESC が後退に入って 0.9 m より多く下がり、escape は前の壁を押さない(fix5 のあとの写しは 1.272 m。外れた見積もり 0.07 秒の分だけ
    後退が短い = この組で残す形)。直す前の写し(fix6 r1)は 0 m のまま escape が前の壁を 0.4 秒より長く押した"""
    amod, before = load_before()
    _, b5 = fix5()
    sc = scene_rev(skip=(2.26, 2.45), drop=(2.11, 2.26))
    res = run(drive, monkeypatch, tmp_path, P1(), sc)
    m = first_rec(res)
    m5 = first_rec(run(b5, monkeypatch, tmp_path, P1(), sc))
    m1 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc))
    assert m['push'] == 0.0 and not m['front'] and m['t_on'] is not None and m['back'] >= 0.9 and _no_skip(res), m
    assert m5['back'] == 1.272 and m5['back'] - m['back'] <= 0.35, (m, m5)
    assert m1['back'] == 0.0 and m1['push'] >= 0.4, m1


@pytest.mark.parametrize('a,b', F1_CELLS)
def test_finding1_cells(monkeypatch, tmp_path, a, b):
    """指摘 1 の 12 組: 今は 'gap' の中立を飛ばさず、escape は前の壁を押さず 0.7 m より多く下がる(fix5 のあとの写しより短いのは
    0.45 m まで = 外れた見積もりの分。後退を 0.2 秒残す)。直す前の写し(fix6 r1)は 0 m のまま escape が前の壁を押した(再現)"""
    amod, before = load_before()
    _, b5 = fix5()
    sc = scene_rev(skip=b, drop=a)
    res = run(drive, monkeypatch, tmp_path, P1(), sc)
    m = first_rec(res)
    m5 = first_rec(run(b5, monkeypatch, tmp_path, P1(), sc))
    m1 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc))
    assert m['push'] == 0.0 and not m['front'] and m['back'] >= 0.7 and _no_skip(res), m
    assert -0.01 <= m5['back'] - m['back'] <= 0.45, (m, m5)
    assert m1['back'] < 0.1 and m1['push'] > 0.0, m1


# 人の中立(約 0.12 秒)に寄せた偽物の ESC(中立 0.12 秒・後退側から 0.10 秒)で、直す前の写しが「0 m・壁押し」になった組(反証の
# probe_false_ready_slow の 15 組から 5 組)
SLOW_CELLS = [((2.11, 2.26), (2.26, 2.45)), ((2.11, 2.26), (2.29, 2.5)), ((2.11, 2.28), (2.28, 2.47)),
              ((2.14, 2.29), (2.29, 2.48)), ((2.14, 2.29), (2.29, 2.52))]


@pytest.mark.parametrize('a,b', SLOW_CELLS)
def test_finding1_slow_esc_cells(monkeypatch, tmp_path, a, b):
    """中立 0.12 秒で下がれる偽物の ESC: 今は escape が前の壁を押さず 0.5 m より多く下がる。直す前の写し(fix6 r1)は 0 m のまま
    escape が前の壁を押した(再現)"""
    amod, before = load_before()
    sc = scene_rev(skip=b, drop=a)
    m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc, neu=0.12, rdelay=0.10))
    m1 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc, neu=0.12, rdelay=0.10))
    assert m['push'] == 0.0 and not m['front'] and m['back'] >= 0.5, m
    assert m1['back'] < 0.1 and m1['push'] > 0.0, m1


def test_finding2_scene(monkeypatch, tmp_path):
    """指摘 2 の入力そのまま(中立 0.2 秒で下がれる偽物の ESC・'reverse' の始め 2.52 秒から 0.2 秒、車両の値が来ない): 今はやり直して
    (+50 µs)fix5 のあとの写しと同じだけ下がる(1.383 m。PCA9685 に書く値も同じ)。直す前の写し(fix6 r1)はやり直さず 0 m のまま
    escape が前の壁を押した"""
    amod, before = load_before()
    _, b5 = fix5()
    sc = scene_rev(skip=(2.52, 2.72))
    now = run(drive, monkeypatch, tmp_path, P1(), sc, neu=0.2)
    old5 = run(b5, monkeypatch, tmp_path, P1(), sc, neu=0.2)
    m, m5 = first_rec(now), first_rec(old5)
    m1 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc, neu=0.2))
    assert m['retry'] and m['push'] == 0.0 and m['back'] == m5['back'] == 1.383, (m, m5)
    assert now[2].w == old5[2].w
    assert not m1['retry'] and m1['back'] == 0.0 and m1['push'] >= 0.4, m1


F2_CELLS = [(d0, L) for d0 in (-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2) for L in (0.16, 0.2, 0.25, 0.3)]


@pytest.mark.parametrize('neu', [0.2, 0.25])
def test_finding2_cells(monkeypatch, tmp_path, neu):
    """指摘 2 の広さ('reverse' の始めから d0 秒に L 秒の車両の値の止まり。28 組): 今は escape が前の壁を押さず 1.0 m より多く
    下がり、fix5 のあとの写しより 0.05 m を超えて少なくも多くもない。直す前の写し(fix6 r1)は半分より多くの組で 0 m・壁押し"""
    amod, before = load_before()
    _, b5 = fix5()
    fail1 = 0
    for d0, L in F2_CELLS:
        sc = scene_rev(skip=(round(2.52 + d0, 3), round(2.52 + d0 + L, 3)))
        m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc, neu=neu))
        m5 = first_rec(run(b5, monkeypatch, tmp_path, P1(), sc, neu=neu))
        assert m['push'] == 0.0 and not m['front'] and m['back'] >= 1.0 and abs(m['back'] - m5['back']) <= 0.05, \
            (d0, L, m, m5)
        m1 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc, neu=neu))
        fail1 += m1['back'] < 0.1 and m1['push'] > 0.0
    assert fail1 > len(F2_CELLS) // 2, fail1


OFF_SCENES = {'f1': (None, lambda: scene_rev(skip=(2.26, 2.45), drop=(2.11, 2.26))),
              'f2': (0.2, lambda: scene_rev(skip=(2.52, 2.72))),
              'slow_retry': (0.2, lambda: scene_rev()),        # 値の止まりなし。CHECK_S のやり直し(+50 µs)が OFF でも起きる
              'r61_rev': (None, lambda: T6.cell('rev', 0.0, 0.55)),
              'r61_brake': (None, lambda: T6.cell('brake', -0.04, 0.55)),
              'brake_cam': (None, lambda: T6.scene('brake', drop=(2.68, 2.81), skip=(2.87, 3.17)))}


@pytest.mark.parametrize('name', sorted(OFF_SCENES))
@pytest.mark.parametrize('pname,params', [('keep0', P1(rec_hold_keep=0)), ('guard0', P1(rec_back_guard=0)),
                                          ('both0', P1(rec_hold_keep=0, rec_back_guard=0)), ('defaults', dict(PARAMS))])
def test_off_bit_identical(monkeypatch, tmp_path, name, pname, params):
    """rec_hold_keep か rec_back_guard が 0(既定を含む): 指摘の場面でも、PCA9685 に書く値・記録の行・出来事が直す前の写し(fix6 r1)と
    ビット単位で同じ"""
    _, before = load_before()
    neu, mk = OFF_SCENES[name]
    sc = mk()
    F5.same_loop(run(drive, monkeypatch, tmp_path, params, sc, neu=neu), run(before, monkeypatch, tmp_path, params, sc, neu=neu))


ON_SAME = {'no_stall_rev': (None, lambda: scene_rev()), 'no_stall_brake': (None, lambda: T6.scene('brake')),
           'cam_only_rev': (None, lambda: scene_rev(drop=(2.62, 2.95))),
           'cam_only_brake': (None, lambda: T6.scene('brake', drop=(2.75, 3.6))),
           'cam_only_slow': (0.2, lambda: scene_rev(drop=(2.55, 2.75)))}


@pytest.mark.parametrize('name', sorted(ON_SAME))
def test_on_without_vehicle_gap_bit_identical(monkeypatch, tmp_path, name):
    """両方 1 でも、車両の値が止まらない走行(中立止めなし・カメラだけの止まり。見積もりは 0)は直す前の写し(fix6 r1)とビット単位で
    同じ(効くのは見積もりがあるときだけ)"""
    _, before = load_before()
    neu, mk = ON_SAME[name]
    sc = mk()
    F5.same_loop(run(drive, monkeypatch, tmp_path, P1(), sc, neu=neu), run(before, monkeypatch, tmp_path, P1(), sc, neu=neu))


@pytest.mark.parametrize('kind', ['rev', 'brake'])
def test_r61_window_still_fixed(monkeypatch, tmp_path, kind):
    """fix6 r1 の指摘の窓(ESC が後退に入る時刻から d0 = −0.06〜0 秒に 0.45〜0.6 秒の車両の値の止まり): 今も下がる距離は値の
    止まりなし以下(+0.05 m)・escape は前の壁を押さない(当たっていた見積もりを捨てない)"""
    for d0, L in T6.WINDOW:
        m = first_rec(run(drive, monkeypatch, tmp_path, P1(), T6.cell(kind, d0, L)))
        assert m['back'] <= T6.REF[kind] + 0.05 and m['push'] == 0.0 and not m['front'], (d0, L, m)
