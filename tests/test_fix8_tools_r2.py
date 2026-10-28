"""fix8 係 tools 直す 8-2 回目(2026-10-28。反証 8-2 回目 係 tools の指摘 2 件 = 打ち消しの票のあとに残った 2 件)の試験。
夜の道具 ⑤ experiments/20261026_fix/diag_launch/collect.py だけを直した(③ standing_3lap・④ quick_eval_by_params・car_check は
変えていない)。走行の動き(PCA9685 に書く値)・既定値は変えていない。

  w2-tools-r8-2-off-few-left      カメラの止まり(launch_stall)・R1 に届かない発進で OFF の本が比べから外れると、残りの OFF の中央値で
                                  「満たさない」→「使ってよい」に動いた(速い OFF 1 本が外れるだけでも)。今は、OFF の発進がカメラの止まりで
                                  外れた(1 本ごと)か、OFF の本の発進が R1 に届かず比べに 1 本も残らない(本ごと)なら「満たさない」(checks の
                                  off_all_compared。組の off_dropped と「→ OFF の発進が比べから外れた N 本 …」の行)。OFF が全部 R1 に届かない
                                  日も「満たさない」の比べを出す(前は比べの行が出なかった)
  w2-tools-r8-2-short-gaps-add-up launch_stall はいちばん長い 1 つの間しか見ないので、0.22 秒以下の止まりが続く発進は外れなかった。今は、
                                  drive が中立を書いた時間の見積もりの合計 Σ(処理したフレームの間ごとに 0.1 秒を超えた分。中立止めの行は
                                  その行から)が LAUNCH_STALL_SUM_S(0.22 秒)より長い発進も同じに外す(launch_stall に neutral_s)。0.12 秒〜
                                  0.22 秒の比べに入れた発進は注意の行(launch_sum_short)

どの試験も「直す前の写し(tests/fixtures/fix8_tools_r2_before/*.orig。sha256 を照らす)で起きる / 前と同じ」か「今の形」を見る。
作り物の記録だけ(tmp。作り方は tests/test_fix7_tools.py・test_fix8_tools_r1.py と同じ)。本物の 10/10・10/23・10/24 の記録
(data/raw・data/labels。読むだけ)を読む試験は、無ければ飛ばす。
環境変数 JF_FIX8R2_COLLECT で、試す ⑤ を差し替えられる(わざと壊した写しで試験が落ちるかを見るとき。
experiments/20261026_fix/fix8/fix_tools_r2/mutate_r2.py)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=failed \\
        --basetemp=<消せる場所> tests/test_fix8_tools_r2.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

JF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
import test_fix7_tools as t7  # noqa: E402  作り物のコース・回 3 の日(同じ作り方)
import test_fix8_tools_decided as t8  # noqa: E402  ⑤ の流し方(runc)・試し(make_kind)
import test_fix8_tools_r1 as T1  # noqa: E402  8-1 の作り物の日(sday)・行の頭

FIX = JF / 'tests' / 'fixtures' / 'fix8_tools_r2_before'
NEW_C = Path(os.environ.get('JF_FIX8R2_COLLECT') or (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'))
# t8.runc の版の表に「直す前(この直しの前 = fix8 tools r1 のあと)」と「今」を足す(③④ は変えていないので今の ③ を読ませる)
t8.VER['r2old'] = (FIX / 'collect.py.orig', JF / 'tools' / 'standing_3lap.py', JF / 'tools' / 'quick_eval_by_params.py')
t8.VER['r2new'] = (NEW_C, JF / 'tools' / 'standing_3lap.py', JF / 'tools' / 'quick_eval_by_params.py')
IG = ['--ignore-gain']
IGF = ['--ignore-gain', '--first-only']
PRM, ON = t7.PRM, t7.ON
g0, rows_of, no_path = t7.g0, t7.rows_of, t7.no_path


def runc(D, ver, args=(), tag=''):
    """⑤ を流す(ver = 'r2new' 今 / 'r2old' 直す前)。(JSON, 標準出力, JSON の文)"""
    return t8.runc(D, ver, args, tag)


def same_outputs(D, args=(), tag=''):
    """⑤ の JSON・標準出力(出力の場所の行のほか)が直す前と今でバイト単位で同じか"""
    jn, on, tn = runc(D, 'r2new', args, 'n' + tag)
    jo, oo, to = runc(D, 'r2old', args, 'o' + tag)
    return tn == to and no_path(on) == no_path(oo)


def verdict(g):
    c = g.get('compare') or {}
    if not c:
        return '比べなし'
    return '使ってよい' if c.get('all_ok') else '満たさない'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _lines(out, head):
    return [x for x in out.splitlines() if x.startswith(head)]


# ================================================================== 速さの作り(本物に近い)
# drive はフレームが 0.1 秒来ないと中立を書く(drive.py の cam_evt.wait(0.1) → hold('カメラの映像が来ていません'))。古いフレームは、その行で
# 中立(中立止めの行 = hold)。中立の間は惰性 −0.9 m/s²(本物の人側・スティック中立の 20 区間の中央値 −0.87。fix8/fix_tools_r1.md 3.2)、
# それ以外は OFF 2/0.6・ON 2/0.3 m/s² で 2 m/s へ(t7.Day の launch_keys の ramp と同じ加速)
ACC_OFF, ACC_ON, DEC = 2.0 / 0.6, 2.0 / 0.3, 0.9


def sim_vk(dur, t_move, neutral=(), acc=ACC_OFF, vmax=2.0, dt=0.001):
    """t_move に動き出し、acc で vmax へ。neutral [(始め, 終わり)] の間は DEC で惰性。vkeys [(秒, m/s)](10 ms ごと)"""
    ts = np.arange(0.0, dur + dt / 2, dt)
    v = np.zeros(len(ts))
    nm = np.zeros(len(ts), bool)
    for a, b in neutral:
        nm |= (ts >= a) & (ts < b)
    for i in range(1, len(ts)):
        if ts[i] < t_move:
            continue
        v[i] = max(0.0, v[i - 1] - DEC * dt) if nm[i] else min(vmax, v[i - 1] + acc * dt)
    k = np.arange(0, len(ts), 10)
    return [(round(float(ts[j]), 3), float(v[j])) for j in k]


def burst(k0, n_miss, count):
    """フレーム k0(k0/30 秒)から、n_miss 枚抜けて 1 枚ある、を count 回 [(残るフレームの番号 k, 抜ける枚数 n)]
    (フレームの間は (n_miss + 1)/30 秒)"""
    return [(k0 + (n_miss + 1) * i, n_miss) for i in range(count)]


def frames_iv(gaps):
    """write_log の stall / hold の間: フレーム k+1 … k+n(k の次から n 枚)"""
    return [(k / 30.0 + 0.001, (k + n) / 30.0 + 0.001) for k, n in gaps]


def neutral_iv(gaps, hold=False):
    """drive が中立を書いた間: フレームが無い(stall)なら前のフレームの 0.1 秒あと〜次のフレーム、中立止めの行(hold)なら最初の行〜次の
    処理したフレーム"""
    out = []
    for k, n in gaps:
        a, b = k / 30.0, (k + n + 1) / 30.0
        if hold:
            out.append(((k + 1) / 30.0, b))
        elif b - a > 0.1:
            out.append((a + 0.1, b))
    return out


def neutral_sum(gaps, hold=False):
    return round(sum(b - a for a, b in neutral_iv(gaps, hold)), 3)


def add_book(D, blk, prm, dm, ramp, gaps=(), hold=False, **kw):
    """回 3 の 1 本(1.0 秒に AI 側・1.0 + dm 秒に動き出す)。gaps があれば、その間はフレームが無い(hold = True なら中立止めの行)で、
    drive の中立の間は惰性(sim_vk)"""
    if not gaps:
        return D.add(blk, prm, dm, ramp, **kw)
    acc = 2.0 / ramp
    vk = sim_vk(30.0, 1.0 + dm, neutral_iv(gaps, hold), acc)
    if hold:
        return D.add(blk, prm, dm, ramp, vk=vk, hold=frames_iv(gaps), **kw)
    return D.add(blk, prm, dm, ramp, vk=vk, stall=frames_iv(gaps), **kw)


# ================================================================== 作り物の日
# (1) w2-tools-r8-2-off-few-left の日(票 A の形): 回 3 の 8 本 = OFF 3-1・3-3・3-5・3-7(起点まで 2.963・3.053・3.103・3.403 秒)と
#     ON 3-2・3-4・3-6・3-8(2.903・2.923・2.943・2.963 秒)。止まりが無ければ ON − OFF −0.145 秒「満たさない」(境目から 0.005 秒)。
#     いちばん速い OFF(3-1)が比べから外れると −0.17 秒「使ってよい」、いちばん遅い OFF(3-7)なら −0.12 秒「満たさない」
FBOOKS = ((12, 'off', 0.15), (13, 'on', 0.24), (14, 'off', 0.24), (15, 'on', 0.26), (16, 'off', 0.29), (17, 'on', 0.28),
          (18, 'off', 0.59), (19, 'on', 0.30))
# (1b) 反証の係の日(review_tools_r2/p6_offcount.py の day と同じ数): OFF 3.013・3.043・3.033・3.253・ON 2.863・2.913・2.963・2.993 = −0.10 秒
PBOOKS = ((12, 'off', 0.20), (13, 'on', 0.20), (14, 'off', 0.23), (15, 'on', 0.25), (16, 'off', 0.22), (17, 'on', 0.30),
          (18, 'off', 0.44), (19, 'on', 0.33))
LATE = [(69, 8)]                  # 2 m/s に届いたあと(2.30 秒から)フレームの間 0.300 秒(起点までは約 0.01 秒しか伸びない。8-1 の直しで外れる)
EARLY = [(48, 8)]                 # 発進の始めの加速の途中(1.60 秒から)フレームの間 0.300 秒(起点までが伸びる)


def hold_long_book(D, blk, prm, dm, ramp=0.6):
    """両方 ON の日の hold_long の形(8-1 の sday 'A' と同じ): 1.6〜2.5 秒にカメラが止まって車が止まる → 明けて中止 → 5.5 秒に人側
    (R1 に届かない)"""
    s0, s1, dur = 1.6, 2.5, 8.0
    v_s0 = 2.0 * (s0 - (1.0 + dm)) / ramp
    vk = [(0.0, 0.0), (1.0 + dm, 0.0), (s0, v_s0), (s1 - 0.1, 0.0), (dur, 0.0)]
    return D.add(blk, prm, dm, ramp, dur=dur, vk=vk, ai=[(1.0, 5.5)], stall=[(s0, s1)], giveup=[(s1, 5.5)],
                 events=[{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.9,
                          'run': 'runB', 'cam_epoch_us': int(t7.T0_US + s1 * 1e6), 't_us': int(t7.T0_US + s1 * 1e6)}])


def blip_book(D, blk, prm, dm, ramp=0.6):
    """置いて AI 側にしたが 0.5 秒で人側に戻し(動かない = 'AI の区間が短い' の R1 に届かない発進)、5.0 秒にもう一度 AI 側にして走った本"""
    return D.add(blk, prm, dm, ramp, vk=t7.launch_keys(5.0, dm, ramp, until=30.0), ai=[(1.0, 1.5), (5.0, 28.0)])


def tail_blip_book(D, blk, prm, dm, ramp=0.6):
    """ふつうに走った(1 回目の発進は R1 に届く)あと 20 秒に人側で止め、24 秒に 0.5 秒だけ AI 側にした本(2 回目の発進 = R1 に届かない)"""
    vk = [(0.0, 0.0), (1.0 + dm, 0.0), (1.0 + dm + ramp, 2.0), (19.5, 2.0), (20.5, 0.0), (30.0, 0.0)]
    return D.add(blk, prm, dm, ramp, vk=vk, ai=[(1.0, 20.0), (24.0, 24.5)])


def fday(tmp, kinds=None, books=FBOOKS):
    """kinds = {block: 'late' | 'early' | 'A' | 'sum' | 'blip' | 'tail'}(その本の形)。ほかの本はふつうに走る"""
    kinds = kinds or {}
    D = t7.Day(tmp)
    logs = {}
    for blk, st, dm in books:
        prm, ramp = (ON, 0.3) if st == 'on' else (PRM, 0.6)
        k = kinds.get(blk)
        if k == 'late':
            logs[blk] = add_book(D, blk, prm, dm, ramp, LATE)
        elif k == 'early':
            logs[blk] = add_book(D, blk, prm, dm, ramp, EARLY)
        elif k == 'sum':
            logs[blk] = add_book(D, blk, prm, dm, ramp, burst(39, 3, 9))
        elif k == 'A':
            logs[blk] = hold_long_book(D, blk, prm, dm, ramp)
        elif k == 'blip':
            logs[blk] = blip_book(D, blk, prm, dm, ramp)
        elif k == 'tail':
            logs[blk] = tail_blip_book(D, blk, prm, dm, ramp)
        else:
            logs[blk] = D.add(blk, prm, dm, ramp)
    return D, logs


# (2) w2-tools-r8-2-short-gaps-add-up の日(8-1 の sday と同じ 8 本): OFF 3.013・3.043・3-5・3.253・ON 2.813・2.863・2.943・2.973。
#     止まりが無ければ 3-5 は 3.033 秒 → ON − OFF −0.135 秒「満たさない」。gaps = {block: [(k, n)]}、hold = 中立止めの行で
SBOOKS = ((12, 'off', 0.20, 0.6), (13, 'on', 0.15, 0.3), (14, 'off', 0.23, 0.6), (15, 'on', 0.20, 0.3), (16, 'off', 0.22, 0.6),
          (17, 'on', 0.28, 0.3), (18, 'off', 0.44, 0.6), (19, 'on', 0.31, 0.3))


def gday(tmp, gaps=None, hold=False, on35=False):
    gaps = gaps or {}
    D = t7.Day(tmp)
    logs = {}
    for blk, st, dm, ramp in SBOOKS:
        if blk == 16 and on35:
            st, ramp = 'on', 0.3
        prm = ON if st == 'on' else PRM
        logs[blk] = add_book(D, blk, prm, dm, ramp, gaps.get(blk, ()), hold)
    return D, logs


OFFDROP_HEAD = '  → OFF の発進が比べから外れた '
SUM_HEAD = '  → うち短い止まりが続いた発進 '
SUMNOTE_HEAD = '  注意: 発進から R1 の起点までに短い止まりが続いた発進 '
COMPARE_HEAD = '  ON − OFF: '


def strip_r2(j, out):
    """今の ⑤ の出力から、8-2 の off-few-left で足したもの(組の off_dropped・checks の off_all_compared・「→ OFF の発進が比べから
    外れた …」の行・「ON − OFF」の行の checks の文の off_all_compared)を除く(直す前と比べるため。all_ok はそのまま)"""
    j = json.loads(json.dumps(j))
    for g in j.get('summary') or []:
        g.pop('off_dropped', None)
        (g.get('compare') or {}).get('checks', {}).pop('off_all_compared', None)
    out = '\n'.join(x.replace(", 'off_all_compared': False", '') for x in no_path(out).splitlines()
                    if not x.startswith(OFFDROP_HEAD))
    return j, out


def name_of(j, log, start_no=1):
    r = [x for x in rows_of(j, log) if x['start_no'] == start_no][0]
    return '%s@%.2f' % (log, r['t_ai'])


# ================================================================== 写し
def test_fixture_is_the_version_before():
    """直す前の写し(この直しの前 = fix8 tools r1 のあとの版 d2b210cc…・8-1 の試験 bb02bde9…・fix3 の試験 35381ec7…)の sha256 と中身。③ standing_3lap・
    ④ quick_eval_by_params・car_check はこの直しで変えていない(写しと今が同じ)"""
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sorted(sums) == ['car_check.py.orig', 'collect.py.orig', 'quick_eval_by_params.py.orig', 'standing_3lap.py.orig',
                            'test_fix3_r1_2.py.orig', 'test_fix8_tools_r1.py.orig']
    assert sums['test_fix8_tools_r1.py.orig'].startswith('bb02bde9')   # 8-1 の試験(8-2 で 1 関数を直した。5.4)
    assert sums['test_fix3_r1_2.py.orig'].startswith('35381ec7')       # fix3 r1-rec3-1 の試験(8-2 で 1 関数を直した。5.4)
    for name, h in sums.items():
        assert _sha(FIX / name) == h, name
    assert sums['collect.py.orig'].startswith('d2b210cc')
    co = (FIX / 'collect.py.orig').read_text(encoding='utf-8')
    assert 'def launch_stall(' in co and 'off_no_stall' in co and 'LAUNCH_GAP_NOTE_S = 0.15' in co
    for w in ('off_all_compared', 'off_dropped', 'def launch_neutral', 'LAUNCH_STALL_SUM_S', 'launch_sum_short'):
        assert w not in co, w
    for name, cur in (('standing_3lap.py.orig', JF / 'tools' / 'standing_3lap.py'),
                      ('quick_eval_by_params.py.orig', JF / 'tools' / 'quick_eval_by_params.py'),
                      ('car_check.py.orig', JF / 'tools' / 'car_check.py')):
        assert _sha(cur) == sums[name], name


# ================================================================== w2-tools-r8-2-off-few-left
@pytest.mark.parametrize('kind', ['late', 'early', 'A'])
def test_old_fast_off_dropped_turns_use_ok(tmp_path, kind):
    """直す前: いちばん速い OFF(3-1)がカメラの止まり(late = 2 m/s のあと・early = 発進の始め)か R1 に届かない(A = hold_long)で比べから
    外れると、残り 3 本の OFF の中央値 3.103 で −0.17 秒「使ってよい」(止まりが無ければ −0.145「満たさない」)"""
    D, logs = fday(tmp_path, {12: kind})
    j, o, _ = runc(D, 'r2old', IG, 'of' + kind)
    g = g0(j)
    assert g['by_launch']['off']['n'] == 3 and g['by_launch']['off']['to_r1_median_s'] == 3.103
    assert g['compare']['d_to_r1_s'] == -0.17 and g['compare']['all_ok'] is True and '→ 使ってよい' in o
    j0, _, _ = runc(fday(tmp_path / 'n')[0], 'r2old', IG, 'of0' + kind)
    assert g0(j0)['compare']['d_to_r1_s'] == -0.145 and g0(j0)['compare']['all_ok'] is False


@pytest.mark.parametrize('kind', ['late', 'early', 'A', 'sum'])
def test_c5_off_dropped_is_not_use_ok(tmp_path, kind):
    """今: いちばん速い OFF が外れたら「満たさない」(checks の off_all_compared だけが False。比べの数 −0.17・OFF n 3 は直す前と同じ)。
    組の off_dropped に名前と理由、「→ OFF の発進が比べから外れた 1 本(…)」の行。sum(0.133 秒 × 9 = 中立の合計 0.3 秒)は 8-2 の
    中立の合計で外れて同じになる"""
    D, logs = fday(tmp_path, {12: kind})
    jn, on, _ = runc(D, 'r2new', IG, 'nf' + kind)
    jo, _, _ = runc(D, 'r2old', IG, 'nfo' + kind)
    gn, go = g0(jn), g0(jo)
    assert gn['by_launch']['off']['n'] == 3 and gn['compare']['d_to_r1_s'] == -0.17
    assert gn['by_launch']['on'] == go['by_launch']['on']
    falses = sorted(k for k, v in gn['compare']['checks'].items() if v is False)
    assert falses == ['off_all_compared'] and gn['compare']['all_ok'] is False
    nm = name_of(jn, logs[12])
    why = 'R1 に届かなかった・15 秒以内に起点なし' if kind == 'A' else 'カメラの止まり・起点まで '
    assert len(gn['off_dropped']) == 1 and gn['off_dropped'][0].startswith('%s(%s' % (nm, why))
    ln = _lines(on, OFFDROP_HEAD)
    assert len(ln) == 1 and ln[0].startswith(OFFDROP_HEAD + '1 本(') and nm in ln[0] and '= 満たさない(checks の off_all_compared。' in ln[0]
    assert '→ 使ってよい' not in on and "'off_all_compared': False" in _lines(on, COMPARE_HEAD)[0]
    if kind != 'sum':                  # 外れ方は直す前と同じ(off-few-left の直しは比べの数を変えない)
        assert gn['by_launch'] == go['by_launch'] and gn['compare']['d_to_r1_s'] == go['compare']['d_to_r1_s']
        sj, so = strip_r2(jn, on)
        assert sj['summary'] == [dict(go, compare=dict(go['compare'], all_ok=False))]


def test_c5_slow_off_dropped_also_counts(tmp_path):
    """外れた OFF がいちばん遅い本(3-7。直す前も −0.12「満たさない」)でも off_all_compared = False(どの本が外れたかで決めない)"""
    D, logs = fday(tmp_path, {18: 'late'})
    gn = g0(runc(D, 'r2new', IG, 'slow')[0])
    assert gn['compare']['d_to_r1_s'] == -0.12 and gn['compare']['checks']['off_all_compared'] is False
    assert gn['compare']['checks']['to_r1'] is False and len(gn['off_dropped']) == 1


@pytest.mark.parametrize('name, kinds, n_drop, d_to', [
    ('late2', {12: 'late', 14: 'late'}, 2, -0.205), ('A2', {12: 'A', 14: 'A'}, 2, -0.205),
    ('mix', {12: 'late', 14: 'A'}, 2, -0.205), ('late3', {12: 'late', 14: 'late', 16: 'late'}, 3, -0.315),
    ('early2', {12: 'early', 16: 'early'}, 2, -0.21)])
def test_c5_reviewer_day(tmp_path, name, kinds, n_drop, d_to):
    """反証の係の日(review_tools_r2/p6_offcount.py と同じ数。止まりが無ければ −0.10「満たさない」): 2・3 本外れると直す前は「使ってよい」、
    今は「満たさない」(off_dropped に外れた本の数)"""
    D, logs = fday(tmp_path, kinds, PBOOKS)
    go = g0(runc(D, 'r2old', IG, 'rvo' + name)[0])
    gn = g0(runc(D, 'r2new', IG, 'rvn' + name)[0])
    assert go['compare']['d_to_r1_s'] == gn['compare']['d_to_r1_s'] == d_to
    assert go['compare']['all_ok'] is True and gn['compare']['all_ok'] is False
    assert len(gn['off_dropped']) == n_drop and gn['compare']['checks']['off_all_compared'] is False
    assert gn['by_launch']['off']['n'] == 4 - n_drop


@pytest.mark.parametrize('kind', ['blip', 'tail'])
def test_c5_book_still_compared_is_bit_identical(tmp_path, kind):
    """R1 に届かない OFF の発進があっても、その本の別の発進が比べに残る(blip = 0.5 秒だけ AI 側にした 1 回目のあとに走った 2 回目・tail =
    走ったあとの 0.5 秒だけの 2 回目)なら数えない = --ignore-gain(手順書の読み方)の出力は直す前とバイト単位で同じ。--first-only では、
    blip の本は 1 回目(届かない)だけになり比べに残らない = 満たさない、tail の本は同じ"""
    D, logs = fday(tmp_path, {12: kind})
    assert same_outputs(D, IG, 'bc' + kind)
    jn, on, _ = runc(D, 'r2new', IGF, 'bcf' + kind)
    gn = g0(jn)
    if kind == 'blip':
        jo, oo, _ = runc(D, 'r2old', IGF, 'bcfo' + kind)
        assert g0(jo)['compare']['all_ok'] is True and g0(jo)['compare']['d_to_r1_s'] == -0.17
        assert gn['compare']['checks']['off_all_compared'] is False and gn['compare']['all_ok'] is False
        assert gn['off_dropped'] == ['%s(R1 に届かなかった・AI の区間が短い)' % name_of(jn, logs[12])]
    else:
        assert same_outputs(D, IGF, 'bcf' + kind) and 'off_dropped' not in gn


def test_c5_redo_in_a_new_record_same_block(tmp_path):
    """OFF の本(3-1 = block 12)が R1 に届かず(hold_long)、同じ block で別の記録にやり直して走った: 回の計画の 1 本(block)は比べに残るので
    数えない = 直す前とバイト単位で同じ。やり直しが無ければ(test_c5_off_dropped_is_not_use_ok の A)満たさない"""
    D, logs = fday(tmp_path, {12: 'A'})
    redo = D.add(12, PRM, 0.15, 0.6)               # 同じ block 12 でやり直し(起点まで 2.963 秒 = 失敗した本と同じ速さ)
    jn, on, _ = runc(D, 'r2new', IG, 'redo')
    gn = g0(jn)
    assert 'off_dropped' not in gn and 'off_all_compared' not in gn['compare']['checks']
    assert gn['by_launch']['off']['n'] == 4 and gn['not_reached']['off']['n'] == 1 and gn['compare']['d_to_r1_s'] == -0.145
    assert any(r['log'] == redo for r in jn['launches'])
    assert same_outputs(D, IG, 'redo')


def test_c5_all_off_not_reached_gets_a_compare(tmp_path):
    """OFF の本が全部 R1 に届かない日: 直す前は「ON − OFF」の行(比べ)が出なかった。今は「満たさない」の比べ(off_all_compared。カメラの
    止まりではないので off_no_stall は付けない)と、外れた 4 本の行"""
    D, logs = fday(tmp_path, {12: 'A', 14: 'A', 16: 'A', 18: 'A'})
    jo, oo, _ = runc(D, 'r2old', IG, 'allA')
    jn, on, _ = runc(D, 'r2new', IG, 'allA')
    assert 'compare' not in g0(jo) and not _lines(oo, COMPARE_HEAD)
    gn = g0(jn)
    ck = gn['compare']['checks']
    assert gn['compare']['all_ok'] is False and ck['off_all_compared'] is False and 'off_no_stall' not in ck
    assert ck['to_r1'] is False and ck['first_lap_collisions_0'] is True and len(gn['off_dropped']) == 4
    assert len(_lines(on, COMPARE_HEAD)) == 1 and '→ 満たさない' in _lines(on, COMPARE_HEAD)[0]
    assert not _lines(on, '  → OFF の発進が全部カメラの止まりで比べから外れた') and len(_lines(on, OFFDROP_HEAD + '4 本(')) == 1


def test_c5_all_off_stalled_keeps_off_no_stall(tmp_path):
    """OFF が全部カメラの止まりで外れた日(8-1 の off_no_stall)は、off_no_stall のまま(行も)・off_all_compared も False"""
    D = t7.Day(tmp_path)
    D.add(12, PRM, 0.20, 0.6, stall=[(1.6, 2.5)])
    D.add(13, ON, 0.15, 0.3)
    D.add(14, PRM, 0.23, 0.6, stall=[(1.6, 2.5)])
    D.add(15, ON, 0.20, 0.3)
    jn, on, _ = runc(D, 'r2new', IG, 'alloff')
    jo, oo, _ = runc(D, 'r2old', IG, 'alloff')
    ckn, cko = g0(jn)['compare']['checks'], g0(jo)['compare']['checks']
    assert ckn == dict(cko, off_all_compared=False) and list(ckn)[:len(cko)] == list(cko)
    assert len(_lines(on, '  → OFF の発進が全部カメラの止まりで比べから外れた')) == 1 and len(_lines(on, OFFDROP_HEAD + '2 本(')) == 1
    sj, so = strip_r2(jn, on)
    assert sj['summary'] == jo['summary'] and so == no_path(oo)


def test_c5_no_on_compared_also_marks_off(tmp_path):
    """ON の発進が全部カメラの止まりで外れた(比べられる ON が無い = 前からの「満たさない」の比べ)日でも、外れた OFF があれば
    off_all_compared = False と行を足す(ほかは直す前と同じ)"""
    D, logs = fday(tmp_path, {12: 'late', 13: 'late', 15: 'late', 17: 'late', 19: 'late'})
    jn, on, _ = runc(D, 'r2new', IG, 'noon')
    jo, oo, _ = runc(D, 'r2old', IG, 'noon')
    gn = g0(jn)
    ck = gn['compare']['checks']
    assert 'on' not in gn['by_launch'] and ck['on_no_stall'] is False and ck['off_all_compared'] is False
    assert len(gn['off_dropped']) == 1 and len(_lines(on, OFFDROP_HEAD + '1 本(')) == 1
    sj, so = strip_r2(jn, on)
    assert sj['summary'] == jo['summary'] and so == no_path(oo)


def test_c5_human_decisions_are_not_counted(tmp_path):
    """人の決めで外した OFF(--skip・試し・途中で変えた)は数えない = 直す前とバイト単位で同じ"""
    D, logs = fday(tmp_path, {12: 'A'})
    assert same_outputs(D, IG + ['--skip', logs[12]], 'skipA')
    D2, logs2 = fday(tmp_path / 't')
    t8.make_kind(D2.raw, logs2[12], 'trial')
    assert same_outputs(D2, IG, 'trialoff')
    D3, logs3 = fday(tmp_path / 'm')
    t8.make_kind(D3.raw, logs3[12], 'midrun')
    assert same_outputs(D3, IG, 'midoff')


def test_c5_unreadable_off_is_not_counted(tmp_path):
    """読めない OFF の記録(npz が全部 0 バイト = 電源断の形。「OFF なので比べは変えない」)は数えない = 直す前とバイト単位で同じ"""
    D, logs = fday(tmp_path)
    for p in (D.raw / logs[12]).rglob('*.npz'):
        p.write_bytes(b'')
    jn, on, _ = runc(D, 'r2new', IG, 'unr')
    assert g0(jn).get('unreadable') or any('読めない' in x or 'skip' in x for x in on.splitlines())
    assert 'off_dropped' not in g0(jn) and same_outputs(D, IG, 'unr')


def test_c5_on_side_unchanged(tmp_path):
    """ON の本がカメラの止まり・R1 に届かないで外れる形は前と同じ(on_no_stall・on_all_reached_r1)= 直す前とバイト単位で同じ"""
    D, logs = fday(tmp_path, {13: 'late'})
    assert same_outputs(D, IG, 'onlate')
    assert g0(runc(D, 'r2new', IG, 'onlate2')[0])['compare']['checks']['on_no_stall'] is False
    D2, logs2 = fday(tmp_path / 'a', {13: 'A'})
    assert same_outputs(D2, IG, 'onA')


@pytest.mark.parametrize('case', ['B', 'C', 'H'])
def test_c5_r1_stall_day_differs_only_by_off_drop(tmp_path, case):
    """8-1 の日(3-5 の OFF の発進の 1.6〜2.5 秒にカメラが止まる)は、8-1 の答え(OFF n 3・−0.14「満たさない」・launch_stall の印 = 1 回の
    止まりの形 {'gap_s', 'at_s'} のまま)に、8-2 の off_all_compared・off_dropped・行が足されただけ"""
    D, log35 = T1.sday(tmp_path, case)
    jn, on, _ = runc(D, 'r2new', IG, 'r1s' + case)
    jo, oo, _ = runc(D, 'r2old', IG, 'r1s' + case)
    assert rows_of(jn, log35)[0]['launch_stall'] == {'gap_s': 0.933, 'at_s': 1.567}
    assert g0(jn)['off_dropped'] == ['%s@1.00(カメラの止まり・起点まで %.3f 秒)' % (log35, rows_of(jn, log35)[0]['to_r1_s'])]
    sj, so = strip_r2(jn, on)
    assert sj == jo and so == no_path(oo)


def test_c5_no_drop_is_bit_identical(tmp_path):
    """OFF が外れない日(止まりなし)は 3 通りの引数で直す前とバイト単位で同じ"""
    D, _ = fday(tmp_path)
    for args in ([], IG, IGF):
        assert same_outputs(D, args, 'none%d' % len(args)), args
    D2, _ = gday(tmp_path / 'g')
    for args in ([], IG, IGF):
        assert same_outputs(D2, args, 'gnone%d' % len(args)), args


# ================================================================== w2-tools-r8-2-short-gaps-add-up
def test_launch_neutral_unit():
    """launch_neutral: 間ごとに min(前のフレーム + 0.1 秒, 間の最初の中立止めの行) から次のフレームまでの合計(端の t0・t_end も)。
    しきい LAUNCH_STALL_SUM_S 0.22・注意 LAUNCH_SUM_NOTE_S 0.05・drive の待ち NEUTRAL_AFTER_S 0.1"""
    m = t8.load_mod('c8r2_unit', NEW_C)
    assert (m.LAUNCH_STALL_SUM_S, m.LAUNCH_SUM_NOTE_S, m.NEUTRAL_AFTER_S, m.LAUNCH_STALL_S) == (0.22, 0.05, 0.1, 0.22)
    t = np.round(np.arange(0.0, 6.0, 1 / 30.0), 6)

    def z(drop=(), hold=()):
        keep = np.ones(len(t), bool)
        for k, n in drop:
            keep[k + 1:k + 1 + n] = False
        h = np.zeros(len(t), bool)
        for k, n in hold:
            h[k + 1:k + 1 + n] = True
        return {'t_s': t[keep], 'hold': h[keep]}

    def nz(zz, t0=1.0, te=4.0):
        return m.launch_neutral(zz, 0, len(zz['t_s']) - 1, t0, te)
    sel = lambda r: (r['neutral_s'], r['n'], r['gap_s'])      # noqa: E731  (同じ長さの間が並ぶときの at_s は見ない)
    assert sel(nz(z())) == (0.0, 0, 0.033)
    assert nz(z([(60, 2)]))['neutral_s'] == 0.0                         # 間 0.1 秒(2 枚抜け)は中立なし
    assert nz(z([(60, 3)])) == {'neutral_s': 0.033, 'n': 1, 'gap_s': 0.133, 'at_s': 2.0}
    assert nz(z([(60, 5)])) == {'neutral_s': 0.1, 'n': 1, 'gap_s': 0.2, 'at_s': 2.0}
    assert sel(nz(z(burst(39, 3, 9)))) == (0.3, 9, 0.133)
    assert nz(z(burst(39, 3, 6)))['neutral_s'] == 0.2 and nz(z(burst(39, 3, 7)))['neutral_s'] == 0.233
    # 中立止めの行は、その行から中立(間 0.133 秒に行 3 枚 = 0.1 秒。フレームの無い間と同じに数えると 0.033 秒)
    assert nz(z(hold=[(60, 3)])) == {'neutral_s': 0.1, 'n': 1, 'gap_s': 0.133, 'at_s': 2.0}
    assert nz(z(hold=burst(39, 3, 4)))['neutral_s'] == 0.4 and nz(z(hold=[(60, 1)]))['neutral_s'] == 0.033
    # 行 1 枚(2.033 秒)のあとフレームが 4 枚抜ける(次は 2.2 秒): 行から次のフレームまで 0.167 秒(フレームの無い間と同じに数えると 0.1 秒)
    zz = z(drop=[(61, 4)], hold=[(60, 1)])
    assert nz(zz)['neutral_s'] == 0.167 and nz(zz)['gap_s'] == 0.2
    zz = z(hold=[(29, 3)])                                               # 発進(1.0 秒)の次から行: 端の 1.0 秒の次の行から
    assert nz(zz)['neutral_s'] == 0.1
    zh = {'t_s': t, 'hold': np.where((t >= 1.0) & (t < 1.1), True, False)}   # 端の t0 の行が中立止めの行: t0 から
    assert nz(zh)['neutral_s'] == 0.1
    assert nz(z([(117, 9)]), 1.0, 4.0)['neutral_s'] == 0.0               # 起点(4.0 秒)をまたぐ止まりは起点まで(3.9 → 4.0 = 0.1 秒の間)
    assert nz(z([(114, 9)]), 1.0, 4.0)['neutral_s'] == 0.1               # 3.8 → 4.0(起点)= 0.2 秒の間
    assert nz(z([(122, 9)]), 1.0, 4.0)['neutral_s'] == 0.0               # 起点のあとは見ない
    zz = z(burst(39, 3, 9))
    zz.pop('hold')                                                       # hold の欄の無いラベル
    assert nz(zz)['neutral_s'] == 0.3


def test_old_short_gaps_stay_use_ok(tmp_path):
    """直す前: 3-5(OFF)の 1.30 秒から 0.133 秒の間 × 9(中立の合計 0.3 秒)は外れず、注意の行も出ず、3-5 は 3.033 → 3.123 秒 = −0.18 秒
    「使ってよい」(止まりが無ければ −0.135「満たさない」)"""
    D, logs = gday(tmp_path, {16: burst(39, 3, 9)})
    j, o, _ = runc(D, 'r2old', IG, 'osg')
    g = g0(j)
    assert 'launch_stall' not in rows_of(j, logs[16])[0] and rows_of(j, logs[16])[0]['to_r1_s'] == 3.123
    assert g['compare']['d_to_r1_s'] == -0.18 and g['compare']['all_ok'] is True
    assert not _lines(o, T1.GAPNOTE_HEAD) and not _lines(o, T1.STALL_HEAD)


@pytest.mark.parametrize('gap, hold, ns, nn, mg', [((3, 7), False, 0.233, 7, 0.133), ((3, 9), False, 0.3, 9, 0.133),
                                                   ((4, 7), False, 0.467, 7, 0.167), ((5, 3), False, 0.3, 3, 0.2),
                                                   ((3, 4), True, 0.4, 4, 0.133)])
def test_c5_short_gaps_add_up_are_excluded(tmp_path, gap, hold, ns, nn, mg):
    """今: 1 回の間は 0.22 秒以下でも、drive の中立の合計が 0.22 秒より長い 3-5(OFF)は、カメラの止まりと同じに比べから外す(launch_stall に
    neutral_s・neutral_n。中立止めの行の続き(hold)も)→ OFF n 3・−0.14 秒(と off_all_compared)で「満たさない」。組に launch_stall_sum と
    「→ うち短い止まりが続いた発進 1 本」の行、発進の行の後ろに「カメラの止まりが続いた(…)」"""
    D, logs = gday(tmp_path, {16: burst(39, gap[0], gap[1])}, hold)
    jn, on, _ = runc(D, 'r2new', IG, 'sg%d%d%d' % (gap[0], gap[1], hold))
    gn = g0(jn)
    r = rows_of(jn, logs[16])[0]
    nm = name_of(jn, logs[16])
    st = r['launch_stall']
    assert (st['neutral_s'], st['neutral_n'], st['gap_s']) == (ns, nn, mg) and sorted(st) == ['at_s', 'gap_s', 'neutral_n', 'neutral_s']
    assert gn['by_launch']['off']['n'] == 3 and gn['compare']['d_to_r1_s'] == -0.14 and gn['compare']['all_ok'] is False
    assert sorted(k for k, v in gn['compare']['checks'].items() if v is False) == ['off_all_compared', 'to_r1']
    assert gn['launch_stall_sum'] == [nm]
    assert gn['launch_stall_excluded'] == ['%s(off・短い止まりが続いた: 中立の見積もりの合計 %.2f 秒(中立 %d 回・いちばん長いフレームの間 %.2f '
                                           '秒(%.2f 秒から))・起点まで %.3f 秒)' % (nm, ns, nn, mg, st['at_s'], r['to_r1_s'])]
    assert len(_lines(on, T1.STALL_HEAD + '1 本(')) == 1
    ln = _lines(on, SUM_HEAD)
    assert len(ln) == 1 and ln[0].startswith(SUM_HEAD + '1 本(1 回のフレームの間は 0.22 秒以下でも、') and ln[0].endswith(': ' + nm)
    assert any(x.startswith(logs[16]) and (' カメラの止まりが続いた(中立の見積もりの合計 %.2f 秒・中立 %d 回・' % (ns, nn)) in x
               and x.endswith('。比べに入れない)') for x in on.splitlines())
    assert not _lines(on, SUMNOTE_HEAD) and '→ 使ってよい' not in on


def test_c5_short_gaps_on_book_is_on_no_stall(tmp_path):
    """3-5 を ON にして 0.133 秒 × 9: カメラの止まりの ON と同じ(比べから外して on_no_stall = 満たさない。OFF は外れないので off_dropped なし)"""
    D, logs = gday(tmp_path, {16: burst(39, 3, 9)}, on35=True)
    gn = g0(runc(D, 'r2new', IG, 'sgon')[0])
    assert gn['launch_stall_on'] == 1 and gn['compare']['checks']['on_no_stall'] is False and 'off_dropped' not in gn
    assert gn['launch_stall_excluded'][0].startswith('%s@1.00(on・短い止まりが続いた: 中立の見積もりの合計 0.30 秒(' % logs[16])


@pytest.mark.parametrize('gap, hold, ns, nn, mg', [((3, 3), False, 0.1, 3, 0.133), ((3, 4), False, 0.133, 4, 0.133),
                                                   ((3, 6), False, 0.2, 6, 0.133), ((5, 2), False, 0.2, 2, 0.2),
                                                   ((1, 6), True, 0.2, 6, 0.067), ((3, 2), False, 0.067, 2, 0.133)])
def test_c5_short_gaps_noted(tmp_path, gap, hold, ns, nn, mg):
    """中立が 2 回以上で合計が 0.05〜0.22 秒の 3-5(OFF)は外さない(比べ・判定・JSON の launches は直す前と同じ)が、作り物の加速では起点
    までが伸びて「使ってよい」に変わりうる(0.133 × 3 で +0.07 秒)ので、組に launch_sum_short と注意の行。標準出力は、その行のほかは
    直す前と同じ"""
    D, logs = gday(tmp_path, {16: burst(39, gap[0], gap[1])}, hold)
    jn, on, _ = runc(D, 'r2new', IG, 'sn%d%d%d' % (gap[0], gap[1], hold))
    jo, oo, _ = runc(D, 'r2old', IG, 'sno%d%d%d' % (gap[0], gap[1], hold))
    gn, go = g0(jn), g0(jo)
    nm = name_of(jn, logs[16])
    assert 'launch_stall' not in rows_of(jn, logs[16])[0] and jn['launches'] == jo['launches']
    assert gn['launch_sum_short'] == ['%s(off・中立の見積もりの合計 %.2f 秒(中立 %d 回・いちばん長いフレームの間 %.2f 秒))' % (nm, ns, nn, mg)]
    assert {k: v for k, v in gn.items() if k != 'launch_sum_short'} == go
    ln = _lines(on, SUMNOTE_HEAD)
    assert len(ln) == 1 and ln[0].startswith(SUMNOTE_HEAD + '1 本(比べには入れた。') and ln[0].endswith(': ' + gn['launch_sum_short'][0])
    assert '合計が 0.05〜0.22 秒' in ln[0]
    assert '\n'.join(x for x in no_path(on).splitlines() if not x.startswith(SUMNOTE_HEAD)) == no_path(oo)


@pytest.mark.parametrize('gap', [(3, 1), (4, 1), (5, 1), (2, 6)])
def test_c5_one_gap_or_no_neutral_is_bit_identical(tmp_path, gap):
    """中立が 1 回だけ(間 0.133・0.167・0.200 秒が 1 つ = 8-1 の短い止まりの注意のまま)か、中立の無い間(0.1 秒 × 6)なら、直す前と
    バイト単位で同じ"""
    D, logs = gday(tmp_path, {16: burst(39, gap[0], gap[1])})
    assert same_outputs(D, IG, 'one%d%d' % gap)


def test_c5_short_gap_outside_the_comparison_not_noted(tmp_path):
    """比べに入らない本(試し)の短い止まりの続きは注意の行に出さない = 直す前とバイト単位で同じ"""
    D, logs = gday(tmp_path, {16: burst(39, 3, 4)})
    t8.make_kind(D.raw, logs[16], 'trial')
    assert same_outputs(D, IG, 'sntr')


def test_c5_short_gaps_after_r1_not_marked(tmp_path):
    """R1 の起点のあと(最初の周の途中)の短い止まりの続きは見ない = 直す前とバイト単位で同じ"""
    D, logs = gday(tmp_path, {16: burst(126, 3, 12)})          # 4.2 秒から(3-5 の R1 の起点は 4.033 秒)
    j, _, _ = runc(D, 'r2new', IG, 'after')
    r = rows_of(j, logs[16])[0]
    assert r['t_ai'] + r['to_r1_s'] < 4.2 and 'launch_stall' not in r
    assert same_outputs(D, IG, 'after')


# ================================================================== 本物の記録(読むだけ。無ければ飛ばす)
REAL_DAYS = ('20261010', '20261023', '20261024')
need_real = pytest.mark.skipif(not all(any((JF / 'data' / 'raw').glob(d + '-*')) for d in REAL_DAYS),
                               reason='本物の 10/10・10/23・10/24 の記録が無い')


@need_real
@pytest.mark.parametrize('day', REAL_DAYS)
def test_real_days_unchanged(tmp_path, day):
    """本物(読むだけ): 10/10・10/23・10/24 の ⑤(--ignore-gain)は直す前とバイト単位で同じ(比べのある組が無い = off_dropped は出ない。
    R1 に届いた発進の中立の合計はいちばん大きくて 0.100 秒 = 外れない・注意も出ない)"""
    import contextlib
    import io
    outs = []
    for ver, p in (('n', NEW_C), ('o', FIX / 'collect.py.orig')):
        m = t8.load_mod('c8r2_real_%s_%s' % (ver, day), p)
        out = tmp_path / ('%s.json' % ver)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main(['--day', day, '--out', str(out), '--ignore-gain'])
        outs.append((out.read_text(encoding='utf-8'), no_path(buf.getvalue())))
    (tn, on), (to, oo) = outs
    assert tn == to and on == oo
    j = json.loads(tn)
    assert not any((r.get('launch_stall') or {}).get('neutral_s') for r in j['launches'])
    assert not any(g.get('off_dropped') or g.get('launch_sum_short') or g.get('launch_stall_sum') for g in j['summary'])
