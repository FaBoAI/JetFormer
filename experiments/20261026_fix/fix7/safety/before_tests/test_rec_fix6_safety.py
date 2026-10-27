"""10/27 fix6 係 safety 直す 6-1 回目(反証 6-1 回目の指摘 w1-safety-r61-safety-1)の試験。走行ループ・補助の側。

  指摘: rec_hold_keep = 1・rec_back_guard = 1・既定の rev_s 0.4・rev_us 300 で、車両の値の止まり(0.45〜0.65 秒)が ESC の後退の
    入りぎわにかかると、値が古くなってから中立止めになるまで(max_sensor_age 0.15 秒)Recovery は古い eRPM 0 のまま後退を出し続けて
    車が下がり始めるのに、明けたサンプルが 300 eRPM 未満なら fix5 s3 の決まりでその秒を 1 秒も数えず、明けてから rev_s をまるごと
    下がり直した(値の止まりなし 1.272 m → 1.606 m。後ろ 1.35〜1.6 m の物に当たる)。ESC が後退を待っていた 'brake'(back_early)
    でも同じ(1.133 m → 最大 1.586 m)。
  同じ窓の逆の形(直す中で見つけた。fix3 r2 からの数え方): 明けたサンプルが後ろへ動いていると、値が来なかった間を全部(古い値で
    Recovery が数えたフレームと二重に)数えて後退を早く切り、0.2〜0.4 m しか下がらずに escape が前の壁を押した。
  直し s3-a(drive.py・assist.EscTrack・assist.Recovery._after_hold): 両方 1 の日は PCA9685 に書けたスロットルを EscTrack に入れ、
    車両の値が来なかった間は、前に見た値が 1000 eRPM より速く後ろへ動いていたなら全部、そうでなければ EscTrack が見積もった
    「ESC が後退で走らせていたかもしれない秒」を数える(前に見た値で Recovery が数えたフレームの分は除く)。見積もりは hold_rev_s で
    渡し、Recovery は後退を HOLD_EST_MIN_LEFT_S = 0.2 秒残すところまで下がった時間に足す。
  直し s3-b(assist.Recovery._after_hold): 残した中立止めの間は中立 = 惰性。'coast' の途中の中立止めは惰性の時間に数え、中立止めの
    あいだに後退を出し終えていたら 'coast' は中立止めの始めから(前は明けてから 'coast' の 0.25 秒をやり直し、keep 0 より最大 0.37 m
    多く下がった)。

OFF(rec_hold_keep・rec_back_guard のどちらかが 0。今の既定)は、直す前の写し tests/fixtures/fix6_safety_before/(assist.py・
drive.py。sha256 は SHA256SUMS = fix5 のあとの版)とビット単位で同じ(PCA9685 に書く値・記録の行・出来事)。指摘の場面の試験は
「直す前の写しで指摘が起きる(再現)」を同じ試験の中で見る。走行ループは tests/test_fix3_r2_1 の閉じたループ(本物の drive._loop を、
書いた値で動く偽物の ESC と車で回す)。実機・GPU・torch・jetformer-web は使わない。台本と表: experiments/20261026_fix/fix6/safety_r1/。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_rec_fix6_safety.py
"""
import ast
import collections
import hashlib
import importlib.util
import random
from pathlib import Path

import pytest

from jetformer import assist as A
from jetformer import drive

import test_fix3_r2_1 as R2
import test_rec_fix5_safety as F5
from test_launch_drive import C, N, PARAMS

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests' / 'fixtures' / 'fix6_safety_before'
SHA_BEFORE = {'assist.py': '80f20890804170bede9c7df6e7f05892ab4e5397f8c5f837fda49ddba64d1276',     # 10/27 fix5 safety のあと
              'drive.py': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c'}
# この組のあと 10/27 fix6 safety の直す 6-2 回目(反証 6-2 回目 w2-safety-r62-safety-1・-2)が assist.py を変えた(drive.py は同じ)。
# そのときは、その直す前の写し tests/fixtures/fix6_safety_r2_before/(= この組のあとの版。sha256 を照らす)をこの組の版として照らす
# (6-2 回目の違いは tests/test_rec_fix6_safety_r2.py が見る。tests/test_fix3_r3_2_tools.py の LATER_FIX6・LATER_FIX6_SAFETY_R2 と同じ値)
FIX6_AFTER = {'assist.py': 'd1d62c66bef856002c14db8dd2fd9c35279a36878b2e4817a279e9fb8bf64165',
              'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec'}
LATER_FIX6_R2 = {'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'}
DT = R2.DT
FDT = 1 / 30.0
XF = R2.XF                  # 場面 rev の前の壁 2.0 m
XB = 2.4                    # 場面 brake の前の壁
MODEL = (C, N - 120.0)
T_ON = {'rev': 2.60, 'brake': 2.73}      # 偽物の ESC が最初のリカバリーで後退に入る時刻(値の止まりなし。test_reference が確かめる)
REF = {'rev': 1.272, 'brake': 1.133}     # 最初のリカバリーで下がる距離(値の止まりなし。反証と同じ)


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
    aspec = importlib.util.spec_from_loader('jetformer._assist_before_fix6s', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._drive_before_fix6s', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    assert not hasattr(amod, 'EscTrack') and not hasattr(mod, 'HOLD_VEH_FAST_ERPM')      # 写しは直す前
    _BEFORE.update(a=amod, d=mod)
    return amod, mod


# ================================================================ 0. 写し・既定値・変えた所
def _fix6_version(name):
    """この組のあとの版(jetformer/ の今の版。fix6 safety の 6-2 回目が変えたあとは、その直す前の写し = この組のあとの版)"""
    p = ROOT / 'jetformer' / name
    if name in LATER_FIX6_R2 and sha(p) == LATER_FIX6_R2[name]:
        p = ROOT / 'tests' / 'fixtures' / 'fix6_safety_r2_before' / name
    assert sha(p) == FIX6_AFTER[name], name
    return p


def _defs(path):
    """モジュールの説明の文を除いた、上の段の文と(クラスの中の)文の AST。名前で引く(並びの番号は使わない)"""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    out = {}
    for i, n in enumerate(tree.body):
        if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue
        if isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = m.name
                elif isinstance(m, ast.Assign):
                    key = ast.unparse(m.targets[0])
                else:
                    key = ast.dump(m)[:80]
                out['%s.%s' % (n.name, key)] = ast.dump(m)
            out['class:' + n.name] = ast.dump(ast.ClassDef(name=n.name, bases=n.bases, keywords=n.keywords, body=[],
                                                           decorator_list=n.decorator_list, type_params=[]))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out['def:' + n.name] = ast.dump(n)
        elif isinstance(n, ast.Assign):
            out['set:' + ast.unparse(n.targets[0])] = ast.dump(n)
        else:
            out['stmt:' + ast.dump(n)[:200]] = ast.dump(n)
    return out


def test_before_copy_defaults_and_changed_parts():
    """写しの sha256(= fix5 のあとの版)・既定値(DEFAULT_PARAMS・PARAM_LIMITS・RESET_ON_BOOT・HOLD_VEH_GAP_S)は同じ。
    Recovery の定数は 1 つも足していない・変えていない。変えた所は assist.py の Recovery._after_hold・Recovery.step・Assist.step、
    足したのは assist.py の EscTrack と HOLD_EST_MIN_LEFT_S、drive.py の HOLD_VEH_FAST_ERPM、変えたのは drive.py の
    Driver._loop だけ(AST)。10/27 fix6 r2(6-2 回目)が assist.py を変えたあとは、その直す前の写し(= この組のあとの版)で照らす"""
    amod, before = load_before()
    assert drive.DEFAULT_PARAMS == before.DEFAULT_PARAMS and drive.PARAM_LIMITS == before.PARAM_LIMITS
    assert drive.RESET_ON_BOOT == before.RESET_ON_BOOT and drive.HOLD_VEH_GAP_S == before.HOLD_VEH_GAP_S
    for k in ('rec_back_guard', 'rec_hold_keep', 'launch_boost', 'launch_no_kick'):
        assert drive.DEFAULT_PARAMS[k] == 0 and k in drive.RESET_ON_BOOT
    assert {k: v for k, v in vars(A.Recovery).items() if k.isupper()} == \
        {k: v for k, v in vars(amod.Recovery).items() if k.isupper()}
    for k in ('BACKING_ERPM', 'CHECK_S', 'STOPPED_ERPM', 'GAP_OFFSET_US', 'DEFAULT_REV_US', 'DEFAULT_REV_S'):
        assert getattr(A, k) == getattr(amod, k)
    assert A.HOLD_EST_MIN_LEFT_S == 0.2 == drive.PARAM_LIMITS['rev_s'][0]       # rev_s の下限
    assert drive.HOLD_VEH_FAST_ERPM == 1000.0 == A.Recovery.ARM_ERPM
    assert A.EscTrack.REV_US == A.EscReverse.DEADBAND_US == 40.0
    a0, a1 = _defs(FIX / 'assist.py'), _defs(_fix6_version('assist.py'))
    assert set(a0) <= set(a1)
    assert {k for k in set(a1) - set(a0) if not k.startswith('EscTrack.')} == {'set:HOLD_EST_MIN_LEFT_S', 'class:EscTrack'}
    assert sorted(k for k in a0 if a0[k] != a1[k]) == ['Assist.step', 'Recovery._after_hold', 'Recovery.step']
    d0, d1 = _defs(FIX / 'drive.py'), _defs(_fix6_version('drive.py'))
    assert set(d1) - set(d0) == {'set:HOLD_VEH_FAST_ERPM'} and set(d0) <= set(d1)
    assert sorted(k for k in d0 if d0[k] != d1[k]) == ['Driver._loop']


# ================================================================ 1. EscTrack(単体)
NS = 1_000_000_000


def _track(seq, t0=0.0, step=FDT):
    """seq = [(書く値, 何フレーム)] を 30 fps で書いた EscTrack と、終わりの時刻 [s]"""
    tr = A.EscTrack(N)
    t = t0
    for thr, n in seq:
        for _ in range(n):
            tr.write(int(round(t * NS)), thr)
            t += step
    return tr, t


def test_esctrack_forward_then_reverse_side_is_brake():
    """前進のあとの後退側はブレーキ(間の中立が何秒でも。このESCの決まり)。ブレーキのあとの中立が 0.08 秒より短ければ、次の後退側も
    ブレーキのまま。どちらも後退で走らせていない"""
    for gap_frames in (0, 1, 2, 10, 30):
        tr, t = _track([(N - 120.0, 10), (N, gap_frames), (N + 300.0, 20)])
        assert tr.rev_s(0, int(t * NS)) == 0.0 and not tr.ready, gap_frames
    for gap_frames in (1, 2):
        tr, t = _track([(N - 120.0, 10), (N + 300.0, 8), (N + 10.0, gap_frames), (N + 300.0, 20)])
        assert tr.rev_s(0, int(t * NS)) == 0.0, gap_frames
    tr, t = _track([(N - 120.0, 10), (N + 300.0, 8), (N + 10.0, 3), (N - 30.0, 1), (N + 300.0, 20)])  # 前進が挟まると数え直し
    assert tr.rev_s(0, int(t * NS)) == 0.0


def test_esctrack_brake_neutral_reverse():
    """ブレーキのあと中立の帯(中立 + 10 µs も)が 0.08 秒以上続けば、次の後退側の 0.06 秒あとから後退で走らせている。
    後退で走らせたあとは、中立が短くてもまた下がれる(このESCの決まり。モジュールの説明の始め)"""
    tr, t = _track([(N - 120.0, 5), (N + 300.0, 8), (N + 10.0, 3), (N + 300.0, 12)])
    t_rev = (5 + 8 + 3) * FDT
    assert tr.rev_s(0, int(t * NS)) == pytest.approx(12 * FDT - 0.06, abs=1e-6)
    assert tr.rev_s(0, int((t_rev + 0.06) * NS)) == 0.0
    tr, t = _track([(N - 120.0, 5), (N + 300.0, 8), (N + 10.0, 3), (N + 300.0, 12), (N, 1), (N + 300.0, 6)])
    assert tr.rev_s(0, int(t * NS)) == pytest.approx(12 * FDT - 0.06 + 6 * FDT - 0.06, abs=1e-6)


def test_esctrack_unknown_is_ready_and_window():
    """分からない始め・人側のあと(unknown)は下がれるとする(多めに数える側)。rev_s は区間の重なりだけ"""
    tr, t = _track([(N + 300.0, 10)])
    assert tr.rev_s(0, int(t * NS)) == pytest.approx(10 * FDT - 0.06, abs=1e-6)
    tr = A.EscTrack(N)
    tr.write(0, N - 200.0)
    tr.unknown(int(0.5 * NS))
    tr.write(int(1.0 * NS), N + 300.0)
    tr.write(int(1.5 * NS), N)
    assert tr.rev_s(0, int(3 * NS)) == pytest.approx(0.44, abs=1e-9)
    assert tr.rev_s(int(1.2 * NS), int(1.3 * NS)) == pytest.approx(0.1, abs=1e-9)
    assert tr.rev_s(int(1.5 * NS), int(2.0 * NS)) == 0.0 and tr.rev_s(int(0.5 * NS), int(1.06 * NS)) == 0.0
    tr.write(int(2.0 * NS), N + 300.0)                       # 続いている区間は b まで
    assert tr.rev_s(int(2.0 * NS), int(2.5 * NS)) == pytest.approx(0.44, abs=1e-9)


def _fake_esc_cover(seed, dur=6.0):
    """でたらめな書き込み(30 fps ± 4 ms・前進・後退側・中立の帯・続く長さ)で、偽物の ESC(R2.Car)が後退で走らせた 10 ms の
    刻みと、EscTrack の見積もり"""
    rng = random.Random(61000 + seed)
    car = R2.Car({'xf': 1e9})
    tr = A.EscTrack(N)
    writes, nxt = [], 0.0
    for v, n in ((N - 120.0, 10), (N + 300.0, 8), (N, 5), (N + 300.0, 10)):    # 始めに 1 回は必ず後退で走らせる並び
        for _ in range(n):
            writes.append((round(nxt, 4), v))
            nxt += FDT + rng.uniform(-0.004, 0.004)
    while nxt < dur:
        k = rng.random()
        v = (N - rng.choice([30.0, 80.0, 120.0, 200.0]) if k < 0.3 else
             N + rng.choice([45.0, 60.0, 200.0, 300.0, 400.0]) if k < 0.6 else N + rng.choice([0.0, 10.0, -10.0, 39.0]))
        for _ in range(rng.choice([1, 1, 2, 3, 3, 4, 5, 9, 15])):
            writes.append((round(nxt, 4), v))
            nxt += FDT + rng.uniform(-0.004, 0.004)
    t, thr, wi, rev = 0.0, N, 0, []
    while t < dur:
        t = round(t + DT, 6)
        while wi < len(writes) and writes[wi][0] <= t - 1e-9:
            thr = writes[wi][1]
            tr.write(int(writes[wi][0] * NS), thr)
            wi += 1
        car.step(t, thr)
        if car.rev_on:
            rev.append(t)
    return tr, rev


@pytest.mark.parametrize('seed', range(8))
def test_esctrack_covers_fake_esc_reverse(seed):
    """多めに数える側: 偽物の ESC が後退で走らせた 10 ms の刻みは(走らせ始めの 1 刻みのほか)全部、EscTrack の見積もりに入る"""
    tr, rev = _fake_esc_cover(seed)
    first = {s for i, s in enumerate(rev) if i == 0 or s - rev[i - 1] > DT + 1e-4}
    miss = [s for s in rev if s not in first and tr.rev_s(int((s - DT) * NS), int(s * NS)) < DT / 2]
    assert not miss, (seed, miss[:5])
    assert rev and tr.rev_s(0, int(7 * NS)) >= len(rev) * DT - len(first) * DT


# ================================================================ 2. Recovery(単体)
def _phase(mod, phase):
    """'brake'・'wait'・'gap'・'reverse'(まだ下がっていない)・中立止めの直前のフレームまで"""
    r, t = F5._to_phase(mod, phase)
    assert r.backed_s == 0.0
    return r, t


@pytest.mark.parametrize('rev_s', [0.2, 0.3, 0.4, 0.6])
@pytest.mark.parametrize('est', [0.05, 0.15, 0.3])
@pytest.mark.parametrize('phase', ['brake', 'wait', 'gap', 'reverse'])
def test_unit_estimate_counts_up_to_min_left(phase, est, rev_s):
    """0.3 秒の中立止め(後ろへ動いていたサンプルなし)と、見ていない間の後退の見積もり est: 今は下がった時間に
    min(est, rev_s − 0.2) を足す(後退を 0.2 秒残すところまで)。直す前の写し(見積もりを知らない)は 0 のまま(指摘の形)。
    段は進めない('brake'・'wait' も 'gap' もそのまま。10/27 fix6 r2 の s3-c: 前は 'brake'・'wait' を見積もりだけで 'reverse' へ
    進め(back_hold)、ESC がまだ下がれない場面で 'gap' の中立を飛ばした。反証 w2-safety-r62-safety-1。tests/test_rec_fix6_safety_r2.py)"""
    amod, _ = load_before()
    r, t = _phase(A, phase)
    r0, t0 = _phase(amod, phase)
    kw = dict(back_guard=True, rev_s=rev_s, hold_since=t + 0.1, hold_back_s=0.3, hold_unseen_s=0.3)
    o, k = r.step(MODEL, 0.0, t + 0.3, hold_rev_s=est, **kw)
    r0.step(MODEL, 0.0, t + 0.3, **kw)
    want = min(est, max(0.0, rev_s - A.HOLD_EST_MIN_LEFT_S))
    assert r.backed_s == pytest.approx(want, abs=1e-9) and r0.backed_s == 0.0
    assert r.backed_s <= rev_s - A.HOLD_EST_MIN_LEFT_S + 1e-9
    if phase in ('brake', 'wait'):
        assert r.phase == r0.phase and r.phase in (phase, 'gap') and r.guard_event is None    # fix6 r2: 見積もりだけでは進めない
        assert r0.phase in (phase, 'gap') and r0.guard_event is None         # 直す前: 下がっていないとして進めない
    else:
        assert r.phase == phase
    assert o is None or o[1] > N                            # 前進(escape)は出さない
    r.guard_event = None
    for i in range(int(round((rev_s - want) / FDT)) - 1):  # このあと後ろへ動けば、残りの後退を出してから惰性
        r.step(MODEL, 2000.0, t + 0.3 + (i + 1) * FDT, back_guard=True, rev_s=rev_s)
        assert r.phase != 'escape'


@pytest.mark.parametrize('hold', [0.15, 0.3, 0.45])
def test_unit_coast_hold_is_coast_time(hold):
    """'coast'(惰性)の途中の残した中立止め: 今は中立止めの間も惰性の時間に数え、明けた時点で 0.25 秒を過ぎていればすぐ escape
    (前進 130 µs。後ろへの惰性を止める)。直す前の写しは段の時間をずらし、明けてから惰性を続けた(後ろへ多く下がる)"""
    amod, _ = load_before()
    res = {}
    for mod in (A, amod):
        r, t = _phase(mod, 'reverse')
        r.backed_s = 0.4
        o, k = r.step(MODEL, 3000.0, t + FDT, back_guard=True)
        assert r.phase == 'coast' and o == (r.center - r.turn * r.REV_STEER_US, N)
        t1 = t + FDT
        t2 = t1 + FDT + hold
        o, k = r.step(MODEL, 3000.0, t2, back_guard=True, hold_since=t1 + 0.02, hold_back_s=hold)
        res[mod is A] = (r.phase, o, t2 - t1)
    ph, o, el = res[True]
    if el >= A.Recovery.COAST_S:
        assert ph == 'escape' and o[1] == N - A.Recovery.ESCAPE_US
    else:
        assert ph == 'coast' and o[1] == N
    ph0, o0, _ = res[False]
    assert ph0 == 'coast' and o0[1] == N                     # 直す前: 明けても惰性(中立)のまま


@pytest.mark.parametrize('backed,hb', [(0.38, 0.05), (0.3, 0.12), (0.39, 0.3)])
def test_unit_reverse_done_in_hold_coasts_from_hold_start(backed, hb):
    """'reverse' の途中の中立止めのあと、下がった時間が rev_s に届いた: 今は惰性を中立止めの始めからとする(0.3 秒の中立止めなら
    明けてすぐ escape)。直す前の写しは明けてから惰性の 0.25 秒を始めた"""
    amod, _ = load_before()
    res = {}
    for mod in (A, amod):
        r, t = _phase(mod, 'reverse')
        r.backed_s = backed
        o, k = r.step(MODEL, 3000.0, t + 0.33, back_guard=True, hold_since=t + 0.03, hold_back_s=hb)
        res[mod is A] = (r.phase, o, r.backed_s)
    assert res[True][2] == res[False][2] >= 0.4
    assert res[True][0] == 'escape' and res[True][1][1] == N - A.Recovery.ESCAPE_US
    assert res[False][0] == 'coast' and res[False][1][1] == N


def _closed_rec_fuzz(seed, n=10000):
    """でたらめな閉じたループ(Recovery 単体): 直す前の写しの Recovery の出力で偽物の ESC(R2.Car・壁なし)を動かし、同じ入力を
    今の Recovery にも渡す。モデルは前進・中立(止まる → stuck → リカバリー)を繰り返す。ときどき残した中立止め(0.1〜0.45 秒・
    まれに 0.6 秒。その間は中立)を入れ、明けたフレームに hold_since・その間に後ろへ動いていた秒 hold_back_s・見積もり hold_rev_s
    (奇数の種は 0)を渡す。ときどき人側(両方 reset)。戻り値: 違いが出たフレームの一覧 [(段, 種類)]。違いが出たら両方 reset して続ける"""
    amod, _ = load_before()
    rng = random.Random(62000 + seed)
    car = R2.Car({'xf': 1e9})
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    t, t_car, thr = 100.0, 0.0, N
    model, model_left = (C, N - 120.0), 0.0
    found = []

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
        k = rng.random()
        kw = {}
        if k < 0.07:                                        # 残した中立止め
            L = rng.choice([0.1, 0.15, 0.2, 0.3, 0.45, 0.45, 0.6])
            t_prev = t
            back = drive_car(0.03, thr) + drive_car(L, N)
            t += 0.03 + L
            kw = dict(hold_since=t_prev + 0.03, hold_back_s=back,
                      hold_rev_s=0.0 if seed % 2 else rng.choice([0.0, 0.05, 0.1, 0.15]))
        else:
            drive_car(FDT, thr)
            t += FDT
        model_left -= FDT
        erpm = -car.v / R2.KV
        a = r0.step(model, erpm, t, 0.0, back_guard=True, **{x: v for x, v in kw.items() if x != 'hold_rev_s'})
        b = r1.step(model, erpm, t, 0.0, back_guard=True, **kw)
        same = a == b and all(getattr(r0, x) == getattr(r1, x) for x in R2.STATE + ('t_check',)) and \
            r0.guard_event == r1.guard_event
        if not same:
            est = kw.get('hold_rev_s', 0.0) > 0.0
            found.append((r0.phase, 'est' if est else 'coast' if r0.phase == 'coast' else 'other',
                          bool(kw), round(r0.backed_s, 3), round(r1.backed_s, 3), r1.phase))
            r0.reset()
            r1.reset()
        r0.guard_event = r1.guard_event = None
        if rng.random() < 0.003:                            # 人側にして戻した
            r0.reset()
            r1.reset()
        thr = a[0][1] if a[0] is not None else model[1]
    return found


_FOUND = collections.Counter()


@pytest.mark.parametrize('seed', range(16))
def test_unit_recovery_closed_fuzz_diverges_only_after_kept_hold(seed):
    """でたらめな閉じたループで直す前の写しと比べる。違いが出るのは、残した中立止めのあとの最初のフレームで、見積もり(hold_rev_s > 0)
    があるか、直す前の写しがそのフレームで 'coast'(惰性。s3-b)にいたときだけ。今の Recovery が下がった時間を少なく数えることは
    ない(惰性の形では同じ)"""
    found = _closed_rec_fuzz(seed)
    for ph0, kind, held, b0, b1, ph1 in found:
        assert held and kind in ('est', 'coast'), (seed, found)
        assert b1 >= b0 - 1e-9
        if kind == 'coast':
            assert b1 == b0 and ph1 in ('coast', 'escape'), (seed, found)
        _FOUND[kind] += 1


def test_unit_recovery_closed_fuzz_saw_both_kinds():
    """上の でたらめな試験が空振りしていない(見積もりの形・惰性の形とも 10 回以上)"""
    if sum(_FOUND.values()) == 0:
        for seed in range(16):
            test_unit_recovery_closed_fuzz_diverges_only_after_kept_hold(seed)
    assert _FOUND['est'] >= 10 and _FOUND['coast'] >= 10, dict(_FOUND)


@pytest.mark.parametrize('seed', range(4))
def test_unit_recovery_without_hold_bit_identical(seed):
    """hold_since を渡さない(中立止めで残さない = OFF・人側・dry run)Recovery は直す前と全部同じ"""
    amod, _ = load_before()
    r0, r1 = amod.Recovery((C, N)), A.Recovery((C, N))
    for model, erpm, t, ig, hg, bg, rev_us, rev_s in R2._rec_fuzz_steps(seed):
        assert r0.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg) == \
            r1.step(model, erpm, t, ig, rev_us, rev_s, back_guard=bg, hold_g=hg)
        for k in R2.STATE + ('t_check',):
            assert getattr(r0, k) == getattr(r1, k), k
        assert r0.guard_event == r1.guard_event
        r0.guard_event = r1.guard_event = None


# ================================================================ 3. 走行ループ(閉じたループ。本物の drive._loop)
class LogCar(R2.Car):
    """R2.Car と同じ。偽物の ESC が後退で走らせていた 10 ms の刻みを残す(revlog)"""

    def __init__(self, s):
        super().__init__(s)
        self.revlog = []

    def step(self, t, thr):
        r = super().step(t, thr)
        self.revlog.append((t, self.rev_on))
        return r


def run(mod, monkeypatch, tmp_path, params, scene):
    monkeypatch.setattr(R2, 'Car', LogCar)
    return F5.run(mod, monkeypatch, tmp_path, params, scene)


def model_brake(t, car):
    """場面 brake: 壁の 0.45 m 手前からブレーキ側(中立 + 200)= 止まると ESC は後退を待っている(10/23 の形)"""
    return (C, N - 120.0) if car.x < XB - 0.45 and t < 6.0 else (C, N + 200.0)


def scene(kind, skip=None, rear=None, drop=None, dur=6.5):
    xf, model = (XF, MODEL) if kind == 'rev' else (XB, model_brake)
    sc = {'dur': dur, 'xf': xf, 'model': model, 'drops': [drop] if drop else [], 'veh_skip': [skip] if skip else []}
    if rear is not None:
        sc['xr'] = xf - rear
    return sc


def cell(kind, d0, L, **kw):
    t0 = T_ON[kind]
    return scene(kind, (round(t0 + d0, 3), round(t0 + d0 + L, 3)), **kw)


def P1(**kw):
    return dict(dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, rev_s=0.4, rev_us=300.0), **kw)


def first_rec(res, rev_us=300.0, t_from=1.5):
    """最初のリカバリー: 'brake' の始め(中立 + rev_us を最初に書いた)・最初の escape(前進 130 µs)・その 0.6 秒あとまでに
    'brake' の始めの位置から下がった最大の距離・escape が前の壁を押した秒・前と後ろの当たり・偽物の ESC が後退に入った時刻"""
    sim, rec, pca = res
    tb = next(t for t, s, th in pca.w if t > t_from and abs(th - (N + rev_us)) < 1e-6)
    esc = N - A.Recovery.ESCAPE_US
    te = next((t for t, s, th in pca.w if t > tb and abs(th - esc) < 1e-6), None)
    end = (te + 0.6) if te is not None else tb + 3.0
    x_b = next(x for t, x, v, th in sim.log if t >= tb)
    xs = [x for t, x, v, th in sim.log if tb <= t <= end]
    push = sum(DT for t, x, v, th in sim.log if te is not None and te <= t < te + 0.6 and x >= sim.s['xf'] - 1e-9
               and abs(th - esc) < 1e-6)
    return {'t_brake': tb, 't_escape': te, 'back': round(x_b - min(xs), 3), 'push': round(push, 2),
            'front': [h for h in sim.car.hits if h[1] == 'front' and tb < h[0] <= end],
            'rear': [h for h in sim.car.hits if h[1] == 'rear' and h[0] <= end],
            't_on': next((t for t, r in sim.car.revlog if t >= tb and r), None)}


_CACHE = {}


def m_of(mod, monkeypatch, tmp_path, params, sc, key=None):
    if key is not None and key in _CACHE:
        return _CACHE[key]
    m = first_rec(run(mod, monkeypatch, tmp_path, params, sc), params['rev_us'])
    if key is not None:
        _CACHE[key] = m
    return m


@pytest.mark.parametrize('kind', ['rev', 'brake'])
def test_reference_no_stall(monkeypatch, tmp_path, kind):
    """比べの元(値の止まりなし): ESC が後退に入る時刻・下がる距離は反証と同じ。直す前の写しとビット単位で同じ"""
    _, before = load_before()
    now = run(drive, monkeypatch, tmp_path, P1(), scene(kind))
    old = run(before, monkeypatch, tmp_path, P1(), scene(kind))
    F5.same_loop(now, old)
    m = first_rec(now)
    assert m['t_on'] == T_ON[kind] and m['back'] == REF[kind] and m['push'] == 0.0 and not m['front']


WINDOW = [(d0, L) for d0 in (-0.06, -0.04, -0.02, 0.0) for L in (0.45, 0.5, 0.55, 0.6)]


@pytest.mark.parametrize('kind', ['rev', 'brake'])
def test_finding_window_not_more_back_no_front_push(monkeypatch, tmp_path, kind):
    """指摘の窓(ESC が後退に入る時刻から d0 = −0.06〜0 秒に、車両の値が 0.45〜0.6 秒来ない。既定の rev_s 0.4・rev_us 300):
    今は下がる距離が値の止まりなし以下(+0.05 m 以内)・escape が前の壁を押さない・最初のリカバリーの間に前の壁に当たらない。
    直す前の写しは値の止まりなしより 0.25 m を超えて多く下がった場面がある(指摘の再現)"""
    _, before = load_before()
    worst0 = -9.0
    for d0, L in WINDOW:
        sc = cell(kind, d0, L)
        m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc))
        assert m['back'] <= REF[kind] + 0.05 and m['push'] == 0.0 and not m['front'], (d0, L, m)
        assert m['t_escape'] is not None and m['back'] >= 0.5, (d0, L, m)        # 下がってから escape
        m0 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc))
        worst0 = max(worst0, m0['back'] - REF[kind])
    assert worst0 > 0.25, worst0


@pytest.mark.parametrize('rear', [1.35, 1.4, 1.45, 1.5, 1.6])
def test_finding_rear_object_not_hit(monkeypatch, tmp_path, rear):
    """指摘の場面(場面 rev・後ろの物を前の壁から rear m): 値の止まりなしでは当たらない距離で、今も最初のリカバリーで後ろに
    当たらない。直す前の写しは d0 = 0・L = 0.55 で当たった(指摘の再現。1.35〜1.6 m)"""
    _, before = load_before()
    assert not first_rec(run(drive, monkeypatch, tmp_path, P1(), scene('rev', rear=rear)))['rear']
    for d0, L in ((0.0, 0.55), (-0.02, 0.6), (-0.04, 0.45), (-0.06, 0.5), (0.0, 0.6)):
        m = first_rec(run(drive, monkeypatch, tmp_path, P1(), cell('rev', d0, L, rear=rear)))
        assert not m['rear'] and m['push'] == 0.0, (d0, L, m)
    m0 = first_rec(run(before, monkeypatch, tmp_path, P1(), cell('rev', 0.0, 0.55, rear=rear)))
    assert m0['rear'], m0


FRONT_CELLS = [(0.0, 0.3), (0.0, 0.45), (0.0, 0.5), (0.02, 0.4), (0.02, 0.6), (0.05, 0.3), (0.1, 0.45)]


@pytest.mark.parametrize('kind', ['rev', 'brake'])
def test_same_window_no_escape_into_front_wall(monkeypatch, tmp_path, kind):
    """同じ窓の逆の形(値の止まりの始めが ESC の後退の入りぎわ〜少しあと。明けたサンプルが後ろへ動いている): 今は escape が前の壁を
    押さず、最初のリカバリーの間に前の壁に当たらない。下がる距離は値の止まりなしと同じ場面の keep 0 の多い方より +0.05 m を
    超えない。直す前の写しは値が来なかった間を全部(二重に)数えて後退を早く切り、0.2〜0.5 m しか下がらずに前の壁へ escape を
    出した(押したか当たった場面が 3 つ以上)"""
    _, before = load_before()
    bad0 = 0
    for d0, L in FRONT_CELLS:
        sc = cell(kind, d0, L)
        m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc))
        k0 = first_rec(run(drive, monkeypatch, tmp_path, P1(rec_hold_keep=0), sc))
        assert m['push'] == 0.0 and not m['front'] and m['back'] <= max(REF[kind], k0['back']) + 0.05, (d0, L, m, k0)
        assert m['back'] >= 0.5, (d0, L, m)
        m0 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc))
        bad0 += m0['push'] > 0.0 or bool(m0['front'])
    assert bad0 >= 3, bad0


@pytest.mark.parametrize('kind,cells', [
    ('rev', [(0.05, 0.3, 0.8), (0.05, 0.45, 0.8), (0.1, 0.45, 0.8), (0.15, 0.2, 0.82)]),
    ('brake', [(0.1, 0.3, 0.95), (0.1, 0.45, 0.95), (0.1, 0.55, 0.95), (0.15, 0.2, 0.75)])])
def test_backing_not_cut_short_after_stale_frames(monkeypatch, tmp_path, kind, cells):
    """値の止まりの始めに車がもう後ろへ動いていた(前に見た値 > 300 eRPM): 中立止めのあとの数えは、古い値で Recovery が dt を
    数えたフレームのあとから(二重に数えない)。値が来なかった間は、遅い(1000 eRPM 以下)なら見積もりだけ。今は下がる距離が
    lo m 以上(値の止まりなしの 0.6〜0.85 倍)。直す前の写しは全部を二重に数えて後退を早く切り、lo m より短かった(再現)。
    (0.15, 0.2) は二重に数えない決まりだけで分かれる場面(二重に数えると 0.72・0.62 m)"""
    _, before = load_before()
    for d0, L, lo in cells:
        sc = cell(kind, d0, L)
        m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc))
        assert lo <= m['back'] <= REF[kind] + 0.15 and m['push'] == 0.0, (d0, L, m)
        m0 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc))
        assert m0['back'] < lo, (d0, L, m0)


COAST_CELLS = [(0.3, 0.3), (0.3, 0.5), (0.4, 0.4), (0.4, 0.6)]


@pytest.mark.parametrize('kind', ['rev', 'brake'])
def test_coast_hold_not_more_than_keep0(monkeypatch, tmp_path, kind):
    """後退の終わり〜惰性('coast')に車両の値が来ない(s3-b): 今の keep 1 は、値の止まりなしと同じ場面の keep 0 の多い方より
    +0.05 m を超えて下がらない。直す前の写しは明けてから惰性をやり直し、0.1 m を超えて多く下がった場面がある(再現)"""
    _, before = load_before()
    worst0 = -9.0
    for d0, L in COAST_CELLS:
        sc = cell(kind, d0, L)
        m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc))
        k0 = first_rec(run(drive, monkeypatch, tmp_path, P1(rec_hold_keep=0), sc))
        lim = max(REF[kind], k0['back'])
        assert m['back'] <= lim + 0.05 and m['push'] == 0.0, (d0, L, m, k0)
        m0 = first_rec(run(before, monkeypatch, tmp_path, P1(), sc))
        worst0 = max(worst0, m0['back'] - lim)
    assert worst0 > 0.1, worst0


@pytest.mark.parametrize('d0,L', [(0.15, 0.4), (0.2, 0.45), (0.2, 0.55), (0.2, 0.6)])
def test_mid_reverse_fast_backing_not_more_than_keep0(monkeypatch, tmp_path, d0, L):
    """後退の途中(速く下がっている)に車両の値が来ない(場面 brake): 前に見た値が 1000 eRPM より速いので、来なかった間は前と同じく
    全部数える。値の止まりなしと keep 0 の多い方より +0.05 m を超えて下がらない(見積もりだけにすると惰性の分 0.2 m 多く下がった)"""
    sc = cell('brake', d0, L)
    m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc))
    k0 = first_rec(run(drive, monkeypatch, tmp_path, P1(rec_hold_keep=0), sc))
    assert m['back'] <= max(REF['brake'], k0['back']) + 0.05 and m['push'] == 0.0, (m, k0)


@pytest.mark.parametrize('kind', ['rev', 'brake'])
@pytest.mark.parametrize('rev_s,rev_us', [(0.2, 300.0), (0.3, 300.0), (0.4, 200.0), (0.6, 300.0), (0.4, 400.0)])
def test_other_rev_settings_window(monkeypatch, tmp_path, kind, rev_s, rev_us):
    """既定でない rev_s・rev_us(approvals 10)の指摘の窓: escape が前の壁を押さない。rev_s 0.3 以上は下がる距離が値の止まりなし
    以下(+0.05 m)。rev_s 0.2 は見積もりで後退を短くしない(後退を 0.2 秒残す。短くすると escape が前の壁に当たる)ので、値の
    止まりなしより最大 約 0.3 m(場面 brake は約 0.4 m)多く下がる。その最大は直す前の写しの最大を超えない(+0.02 m)。直す前の写しは
    rev_s 0.2 で escape が前の壁を押した場面がある。
    (10/27 fix6 r2: 場面 brake の上限を 0.35 → 0.42 m にした。rev_s 0.2 は後退を 0.2 秒残す決まりで見積もりを 1 秒も足さないのに、
    r1 は見積もりがあるだけで 'brake' から 'reverse' へ進め(back_hold)、たまたま短く下がっていた(最大 0.652 m)。その段の進め方は
    ESC がまだ下がれないときに 0 m のまま escape が前の壁を押す元だったので r2 でやめ(反証 w2-safety-r62-safety-1)、場面 brake の
    rev_s 0.2 の最大は直す前の写し(fix5 のあと)と同じ 0.79 m に戻った。tests/test_rec_fix6_safety_r2.py)"""
    _, before = load_before()
    p = P1(rev_s=rev_s, rev_us=rev_us)
    ref = m_of(drive, monkeypatch, tmp_path, p, scene(kind), key=('ref', kind, rev_s, rev_us))
    cells = WINDOW if rev_s < 0.3 else WINDOW[::2]
    mx, mx0, push0 = 0.0, 0.0, 0
    for d0, L in cells:
        sc = cell(kind, d0, L)
        m = first_rec(run(drive, monkeypatch, tmp_path, p, sc), rev_us)
        assert m['push'] == 0.0, (d0, L, m)
        if rev_s >= 0.3:
            assert m['back'] <= ref['back'] + 0.05, (d0, L, m, ref)
        else:
            m0 = first_rec(run(before, monkeypatch, tmp_path, p, sc), rev_us)
            mx, mx0, push0 = max(mx, m['back']), max(mx0, m0['back']), push0 + (m0['push'] > 0.0)
    if rev_s < 0.3:
        lim = 0.35 if kind == 'rev' else 0.42
        assert mx <= mx0 + 0.02 and mx <= ref['back'] + lim and push0 >= 1, (mx, mx0, ref['back'], push0)


GRID_D0 = (-0.2, -0.12, -0.08, -0.04, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
GRID_L = (0.2, 0.3, 0.45, 0.55, 0.65)


def _phase_max(monkeypatch, tmp_path, kind):
    """値の止まりなしで、フレームの並びの位相(最初のフレームの時刻)を R2.PHASES の 10 通りに変えたときの下がる距離の最大"""
    key = ('phase_max', kind)
    if key not in _CACHE:
        _CACHE[key] = max(first_rec(run(drive, monkeypatch, tmp_path, P1(), dict(scene(kind), cam0=c)))['back']
                          for c in R2.PHASES)
    return _CACHE[key]


@pytest.mark.parametrize('kind', ['rev', 'brake'])
def test_default_grid_no_front_push_and_bounded(monkeypatch, tmp_path, kind):
    """既定の rev_s・rev_us で、値の止まりの始め・長さを広く振る(10 × 5): escape が前の壁を押さない・最初のリカバリーの間に
    前の壁に当たらない・下がる距離は、値の止まりなしの位相 10 通りの最大と同じ場面の keep 0 の多い方より +0.05 m を超えない"""
    pmax = _phase_max(monkeypatch, tmp_path, kind)
    assert REF[kind] <= pmax <= REF[kind] + 0.3
    for d0 in GRID_D0:
        for L in GRID_L:
            sc = cell(kind, d0, L)
            m = first_rec(run(drive, monkeypatch, tmp_path, P1(), sc))
            k0 = first_rec(run(drive, monkeypatch, tmp_path, P1(rec_hold_keep=0), sc))
            assert m['push'] == 0.0 and not m['front'], (d0, L, m)
            assert m['back'] <= max(pmax, k0['back']) + 0.05, (d0, L, m, k0, pmax)


OFF_SCENES = {'rev_veh_on': lambda: cell('rev', 0.0, 0.55), 'rev_veh_late': lambda: cell('rev', 0.3, 0.3),
              'brake_veh': lambda: cell('brake', -0.04, 0.55), 'rev_cam': lambda: scene('rev', drop=(2.62, 2.95)),
              'brake_cam_long': lambda: scene('brake', drop=(2.75, 3.6))}


@pytest.mark.parametrize('name', sorted(OFF_SCENES))
@pytest.mark.parametrize('pname,params', [('keep0', P1(rec_hold_keep=0)), ('guard0', P1(rec_back_guard=0)),
                                          ('both0', P1(rec_hold_keep=0, rec_back_guard=0)), ('defaults', dict(PARAMS))])
def test_off_bit_identical(monkeypatch, tmp_path, name, pname, params):
    """rec_hold_keep か rec_back_guard が 0(既定を含む): 中立止めのある場面でも、PCA9685 に書く値・記録の行・出来事が直す前の写しと
    ビット単位で同じ"""
    _, before = load_before()
    sc = OFF_SCENES[name]()
    F5.same_loop(run(drive, monkeypatch, tmp_path, params, sc), run(before, monkeypatch, tmp_path, params, sc))


@pytest.mark.parametrize('kind', ['rev', 'brake'])
@pytest.mark.parametrize('rev_s', [0.2, 0.4])
def test_on_without_hold_bit_identical(monkeypatch, tmp_path, kind, rev_s):
    """両方 1 でも、中立止めの無い走行は直す前と同じ(効くのは残した中立止めのあとだけ)"""
    _, before = load_before()
    a = run(drive, monkeypatch, tmp_path, P1(rev_s=rev_s), scene(kind))
    F5.same_loop(a, run(before, monkeypatch, tmp_path, P1(rev_s=rev_s), scene(kind)))
    assert not any(row['hold'] for _, row in a[1].rows if row['select'])


@pytest.mark.parametrize('seed', range(12))
def test_closed_loop_fuzz_rules_and_off_identity(monkeypatch, tmp_path, seed):
    """でたらめな場面(tests/test_rec_fix5_safety の _rand_scene・_rand_params。前の壁・後ろの物・発進の途中のカメラの止まり・
    カメラと車両の値の止まり・止まらない衝撃)で、両方 1 の本物の走行ループが安全の決まり(F5.check_rules の R1〜R7)を破らない。
    同じ場面で rec_hold_keep = 0 は直す前の写しとビット単位で同じ"""
    _, before = load_before()
    rng = random.Random(78000 + seed)
    for _ in range(2):
        sc = F5._rand_scene(rng)
        p = F5._rand_params(rng)
        sim, rec, pca = run(drive, monkeypatch, tmp_path, p, sc)
        assert F5.check_rules(sim, rec, pca, p) == [], seed
        p0 = dict(p, rec_hold_keep=0)
        F5.same_loop(run(drive, monkeypatch, tmp_path, p0, sc), run(before, monkeypatch, tmp_path, p0, sc))
