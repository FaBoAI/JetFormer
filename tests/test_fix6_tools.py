"""直す 6-1 回目・係 tools(2026-10-27。反証 6-1 回目の指摘 w1-tools-r6-1-tools-1)の試験。⑤ experiments/20261026_fix/diag_launch/
collect.py と手順書 7・8 章の文だけ。

  w1-tools-r6-1-tools-1(major): AI 側にしたシーンごと書けずに落ちた本(電源断・jetformer-web の落ち。recorder はシーンを 20 秒ためて
    終わりに npz を書くので、人側で置いて待った 1 つ目のシーンだけが残る)は、読めても AI のフレームが 1 つも無いので、⑤ の main が
    `if not d['z']['ai'].any(): continue` で黙って飛ばした。skip の行・読めない記録・「記録が欠けている本」のどれにも出ず、回 3 の
    ON の本が最初の周に 9G で当たっても「ON − OFF … → 使ってよい」と出た(car_check・③ は欠けを出していた)
    → AI のフレームが無く、フレームの無い欠けたシーンのある本は「skip … AI 側のフレームが無い …」と読めない記録(ON なら満たさない)。
    発進の前進の ON/OFF は log.json・計画に、events.jsonl の発進の前進の start も足す。読めた本でも、欠けたシーンの間に start が
    ある(2 回目の発進のシーンごと落ちた)なら同じ(--first-only で読めた発進のあとだけなら入れない)。「記録が欠けている本」の行は
    AI のフレームが無い本にも出す。

どの試験も「直す前の写し(tests/fixtures/fix6_tools_before/collect.py.orig。sha256 を照らす)で指摘が起きる」か「その場面のほかは
直す前と出力(JSON・標準出力)が同じ」を見る。作り物の記録だけ(tmp)。本物の data/raw・labels・装置・GPU は使わない。
環境変数 JF_FIX6_COLLECT で、試す collect.py を差し替えられる(わざと壊した写しで試験が落ちるかを見るとき)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix6_tools.py
"""
import contextlib
import hashlib
import io
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
import test_fix3_r3_2_tools as T  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402
from test_lap_anchor import T0_US  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402
import label_events as le  # noqa: E402

FIX = JF / 'tests' / 'fixtures' / 'fix6_tools_before'
OLD = FIX / 'collect.py.orig'
NEW = Path(os.environ.get('JF_FIX6_COLLECT') or (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'))
RUNBOOK = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md')
ON = dict(T.G, launch_boost=1)
IG = ['--ignore-gain']
IGF = ['--ignore-gain', '--first-only']


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_fixture_is_the_version_before():
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert _sha(OLD) == sums['collect.py.orig']
    assert 'def lost_scenes' not in OLD.read_text(encoding='utf-8')


def _n(name):
    return name.replace('-', '_').replace('.', '_')


def both(D, name, args=()):
    """(今, 直す前) の (JSON, 標準出力, JSON の文)"""
    return T.runc(D, 'n6_' + _n(name), path=NEW, args=args), T.runc(D, 'o6_' + _n(name), path=OLD, args=args)


def lose(D, log, scene, only=None, zero=False):
    """シーン scene の npz を消す(CAM_FRONT/ だけ残す = 書いている途中で落ちた形)。only = 消す npz の名前、zero = 0 バイトにする"""
    sc = D.raw / log / ('%s-%04d' % (log, scene))
    for p in sc.glob('*.npz'):
        if only and p.stem not in only:
            continue
        if zero:
            p.write_bytes(b'')
        else:
            p.unlink()
    if not only and not zero:
        (sc / 'CAM_FRONT').mkdir()


def add_start(D, log, t):
    """events.jsonl に発進の前進の start を足す(drive.py の launch_note と同じ形。時刻 = 記録の頭から t 秒)"""
    with open(D.raw / log / 'events.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'kind': 'launch_boost', 'event': 'start', 'us': 250.0, 'cam_t_us': T0_US + int(round(t * 1e6)),
                            't_us': T0_US + int(round(t * 1e6)) + 500}) + '\n')


def round3(tmp, case):
    """反証の p1 と同じ回 3: OFF 4 本(起点まで 3.013 秒)・ON 3 本(2.8・2.7・2.6 秒)・3-8 は case の形。3-8 の名前を返す"""
    tmp.mkdir(parents=True, exist_ok=True)
    D = T.RealDay(tmp)
    D.add_real(12, T.G, T.HAND)
    D.add_real(13, ON, T.HAND, **T.on_kw(36))
    D.add_real(14, T.G, T.HAND)
    D.add_real(15, ON, T.HAND, **T.on_kw(39))
    D.add_real(16, T.G, T.HAND)
    D.add_real(17, ON, T.HAND, **T.on_kw(42))
    D.add_real(18, T.G, T.HAND)
    if case == 'C':          # 1 回目の発進(1.4 秒)は読める・人側 20 秒・2 回目(45 秒・発進の前進の start)のシーン 0003 が無い
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        add_start(D, log, 45.0)
        lose(D, log, 3)
        return D, log
    if case == 'G':          # 1 回目の発進(25 秒の start)のシーン 0002 が無い・人側にして戻し、2 回目(45 秒)はシーン 0003 で読める
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, t_ai=45.0, t_move=45.1, launch=(45.0, 0.5, 1), first_anchor=48.0)
        add_start(D, log, 25.0)
        lose(D, log, 2)
        return D, log
    if case == 'D':          # AI 側にしていない本(人だけ)。シーン 0002 の VEHICLE だけ無い(AI_CMD はあり select はすべて False)
        log, _ = D.add_real(19, ON, T.HAND, dur=40.0, t_ai=99.0, t_move=99.0)
        lose(D, log, 2, only=('VEHICLE',))
        return D, log
    if case == 'F':          # AI 側にしていない本(人だけ)。欠けは無い
        log, _ = D.add_real(19, ON, T.HAND, dur=40.0, t_ai=99.0, t_move=99.0)
        return D, log
    if case == 'HEAD':       # シーン 0001 の終わりの 2 フレーム前(19.90 秒)に AI 側・発進の前進の start。シーン 0002 が無い・0003 は人
        t = 597 / 30.0 - 1e-6                  # (フレームの時刻の丸めでそのフレームが AI 側になるように 1 µs 前)
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, t_ai=t, t_move=t + 0.1, launch=(t, 0.5, 1), human_after=40.0,
                            first_anchor=23.0)
        lose(D, log, 2)
        return D, log
    if case == 'TAIL':       # シーンの区切り 20.01 秒(0003 は 40.02 秒から)。0003 の最初のフレーム(40.033 秒)に AI 側・start。0002 が無い
        t = 1201 / 30.0 - 1e-6
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, t_ai=t, t_move=t + 0.1, launch=(t, 0.5, 1), first_anchor=43.0,
                            scene_s=20.01)
        lose(D, log, 2)
        return D, log
    p = T.G if case in ('B2', 'E') else ON
    kw = dict(dur=40.0, t_ai=25.0, t_move=25.1, first_anchor=28.0, hits=[(31.0, 9.0)])
    if case != 'E':
        kw['launch'] = (25.0, 0.5, 1)      # events.jsonl に start 25.00 秒(B2 は log.json の launch_boost が 0 = 途中で ON にした形)
    log, _ = D.add_real(19, p, T.HAND, **kw)
    if case in ('B', 'B2', 'E'):
        lose(D, log, 2)
    elif case == 'B0':
        lose(D, log, 2, only=('AI_CMD',), zero=True)
    return D, log


def g0(j):
    return j['summary'][0]


def named(out, log):
    """標準出力で log の名前が出る行(発進の行を除く)"""
    return [x for x in out.splitlines() if log in x and not x.startswith(log + ' No.')]


def c1_lines(out):
    return [x for x in out.splitlines() if x.startswith('記録が欠けている本 ')]


def no_path(out):
    return '\n'.join(x for x in out.splitlines() if not x.startswith('出力 '))


# ================================================================== AI 側のシーンごと落ちた本
@pytest.mark.parametrize('args', [IG, IGF])
@pytest.mark.parametrize('case', ['B', 'B0', 'B2'])
def test_lost_ai_scene_on_book_is_unreadable_now(tmp_path, case, args):
    """直す前: 3-8 は比べから黙って消え、ON 3 本対 OFF 4 本で「使ってよい」、3-8 の名前はどこにも出ない(反証の B)。
    今: skip の行・読めない記録(ON)・「記録が欠けている本」の行に出て、組は「満たさない」(on_all_readable = False)。ほかの数は同じ"""
    D, log = round3(tmp_path, case)
    (jn, on, tn), (jo, oo, to) = both(D, case + str(len(args)), args)
    # 直す前の形(指摘の再現)
    assert g0(jo)['compare']['all_ok'] is True and g0(jo)['by_launch']['on']['n'] == 3 and g0(jo)['by_launch']['off']['n'] == 4
    assert 'unreadable' not in jo and 'unreadable' not in g0(jo) and named(oo, log) == [] and log not in jo['problems']
    assert oo.splitlines()[0].startswith('logs 7 ')
    # 今
    g = g0(jn)
    assert g['compare']['all_ok'] is False
    assert g['compare']['checks'] == dict(g0(jo)['compare']['checks'], on_all_readable=False)
    assert {k: v for k, v in g['compare'].items() if k not in ('checks', 'all_ok')} == \
        {k: v for k, v in g0(jo)['compare'].items() if k not in ('checks', 'all_ok')}
    assert g['by_launch'] == g0(jo)['by_launch'] and jn['launches'] == jo['launches']
    [u] = jn['unreadable']
    assert u['log'] == log and u['launch_state'] == 'on' and u['lost_scenes'] == ['0002'] and u['launch_starts'] == [25.0]
    assert u['launch_param'] == (0 if case == 'B2' else 1)
    assert u['why'].startswith('AI 側のフレームが無い(シーン 0002 にフレームが無い')
    assert g['unreadable']['logs'] == [log] and g['unreadable']['on'] == 1 and g['unreadable']['launch_starts'] == [[25.0]]
    assert log in jn['problems']
    lines = on.splitlines()
    assert lines[0] == 'skip %s %s。events.jsonl に発進の前進の start 25.00 秒' % (log, u['why'])
    ur = [x for x in lines if x.startswith('  読めない記録 1 本')]
    assert len(ur) == 1 and ur[0].startswith('  読めない記録 1 本(比べの数に入れない): %s(AI 側のフレームが無い' % log)
    assert ('発進の前進 ON: log.json %s・計画 ?・events.jsonl の start 25.00 秒)' % ('OFF' if case == 'B2' else 'ON')) in ur[0]
    assert any(x.startswith('  → 読めない(欠けた)ON の本がある = 満たさない') for x in lines)
    [c1] = c1_lines(on)
    assert c1.startswith('記録が欠けている本 %s(11/03 回3・発進の前進 ON): シーン 0002 の npz が無い・読めない(%d 件)。AI 側のフレームが無い'
                         % (log, 1 if case == 'B0' else 3))
    assert '「読めない記録」に入れた(ON なので満たさない。--skip 記録名 で外せる)→ 動画・紙で' in c1
    assert lines[1].startswith('logs 7 ')          # 読めた本の数は同じ(3-8 は data に入らない)


@pytest.mark.parametrize('args', [IG, IGF])
def test_lost_ai_scene_intact_is_unchanged(tmp_path, args):
    """3-8 の記録がそろっている(反証の A): JSON・標準出力とも直す前と同じ(最初の周の 9G で「満たさない」)"""
    D, log = round3(tmp_path, 'A')
    (jn, on, tn), (jo, oo, to) = both(D, 'A' + str(len(args)), args)
    assert tn == to and no_path(on) == no_path(oo)
    assert g0(jn)['compare']['checks']['first_lap_collisions_0'] is False and g0(jn)['by_launch']['on']['n'] == 4


def test_skip_removes_the_lost_book(tmp_path):
    """動画・紙で確かめた本は --skip 記録名 で外せる(外すと比べは直す前と同じ。外した名前が出る)"""
    D, log = round3(tmp_path, 'B')
    (jn, on, tn), (jo, oo, to) = both(D, 'Bskip', IG + ['--skip', log])
    assert g0(jn)['compare'] == g0(jo)['compare'] and g0(jn)['compare']['all_ok'] is True
    assert jn['summary_skipped'] == ['%s(読めない記録)' % log] and jo['summary_skipped'] == []
    assert 'unreadable' not in g0(jn) and [u['log'] for u in jn['unreadable']] == [log]


@pytest.mark.parametrize('args', [IG, IGF])
def test_lost_ai_scene_off_book(tmp_path, args):
    """OFF の本が同じ形で落ちた: 読めない記録(OFF)と行に出るが、比べ(compare)は直す前と同じ(OFF の読めない本は満たさないにしない)"""
    D, log = round3(tmp_path, 'E')
    (jn, on, tn), (jo, oo, to) = both(D, 'E' + str(len(args)), args)
    assert g0(jn)['compare'] == g0(jo)['compare'] and g0(jn)['compare']['all_ok'] is True
    [u] = jn['unreadable']
    assert u['log'] == log and u['launch_state'] == 'off' and 'launch_starts' not in u and 'unreadable' not in jo
    assert 'launch_starts' not in g0(jn)['unreadable']
    lines = on.splitlines()
    assert lines[0] == 'skip %s %s' % (log, u['why'])
    assert not any(x.startswith('  → 読めない(欠けた)ON') for x in lines)
    [c1] = c1_lines(on)
    assert c1.startswith('記録が欠けている本 %s(11/03 回3・発進の前進 OFF)' % log) and 'OFF なので比べは変えない' in c1
    assert named(oo, log) == []


# ================================================================== 読めた本の、欠けたシーンの間の発進
def test_lost_later_launch_is_unreadable_now(tmp_path):
    """3-8 の 1 回目の発進は読めるが、2 回目の発進(45 秒の start)のシーン 0003 が無い。直す前: 2 回目は黙って消え「使ってよい」。
    今: 読めない記録(ON・start 45.00 秒)で「満たさない」。「記録が欠けている本」の行の後ろに 1 文"""
    D, log = round3(tmp_path, 'C')
    (jn, on, tn), (jo, oo, to) = both(D, 'C', IG)
    assert g0(jo)['compare']['all_ok'] is True and 'unreadable' not in jo
    assert g0(jn)['compare']['all_ok'] is False and g0(jn)['compare']['checks']['on_all_readable'] is False
    assert jn['launches'] == jo['launches'] and g0(jn)['by_launch'] == g0(jo)['by_launch']
    [u] = jn['unreadable']
    assert u['log'] == log and u['launch_state'] == 'on' and u['lost_scenes'] == ['0003'] and u['launch_starts'] == [45.0]
    assert u['why'].startswith('シーン 0003 にフレームが無い')
    assert not any(x.startswith('skip ') for x in on.splitlines())      # 読めた本なので skip の行は無い
    [c1n], [c1o] = c1_lines(on), c1_lines(oo)
    assert c1n.startswith(c1o.split('→ 動画・紙で')[0])
    assert '。欠けたシーンの間に発進の前進の start 45.00 秒 = その発進の記録が無い。「読めない記録」に入れた(ON なので満たさない' in c1n


def test_lost_later_launch_first_only_is_like_a_second_launch(tmp_path):
    """--first-only: 読めた発進(1.4 秒)のあとの start だけなので入れない(読めた 2 回目以降と同じ)。JSON は直す前と同じ。
    行の後ろの 1 文だけ違う"""
    D, log = round3(tmp_path, 'C')
    (jn, on, tn), (jo, oo, to) = both(D, 'Cf', IGF)
    assert tn == to
    [c1n], [c1o] = c1_lines(on), c1_lines(oo)
    assert c1n == c1o.replace('で満たさない)→', 'で満たさない)。欠けたシーンの間に発進の前進の start 45.00 秒 = その発進の記録が無い。'
                              '読めた発進のあと(2 回目以降)なので --first-only の比べには入れない→')
    rest = lambda o: [x for x in no_path(o).splitlines() if not x.startswith('記録が欠けている本 ')]      # noqa: E731
    assert rest(on) == rest(oo)


@pytest.mark.parametrize('args', [IG, IGF])
def test_lost_first_launch_before_a_readable_one(tmp_path, args):
    """1 回目の発進(25 秒の start)のシーン 0002 が無く、2 回目(45 秒)は読める: 欠けた発進は読めた発進より前 = その本の 1 回目なので、
    --first-only でも読めない記録(ON)に入れる(直す前: 黙って消え、読めた 2 回目が「1 回目」として比べに入った)"""
    D, log = round3(tmp_path, 'G')
    (jn, on, tn), (jo, oo, to) = both(D, 'G' + str(len(args)), args)
    assert len([r for r in jo['launches'] if r['log'] == log]) == 1 and 'unreadable' not in jo
    assert jn['launches'] == jo['launches']
    [u] = jn['unreadable']
    assert u['log'] == log and u['launch_starts'] == [25.0] and u['lost_scenes'] == ['0002'] and u['launch_state'] == 'on'
    assert g0(jn)['compare']['all_ok'] is False and g0(jn)['compare']['checks']['on_all_readable'] is False
    [c1] = c1_lines(on)
    assert '欠けたシーンの間に発進の前進の start 25.00 秒 = その発進の記録が無い。「読めない記録」に入れた' in c1


def test_start_just_before_a_lost_scene_is_lost(tmp_path):
    """シーン 0001 の終わりの 2 フレーム前に AI 側・start、0002 が無い: 読める AI の区間は 0.067 秒で発進の行にならない。
    直す前: 黙って消える。今: 欠けたシーンの間(前のフレームの 0.1 秒前より後)として読めない記録(ON)"""
    D, log = round3(tmp_path, 'HEAD')
    (jn, on, tn), (jo, oo, to) = both(D, 'HEAD', IG)
    assert [r for r in jo['launches'] if r['log'] == log] == [] and g0(jo)['compare']['all_ok'] is True
    [u] = jn['unreadable']
    assert u['log'] == log and u['launch_starts'] == [19.9] and u['lost_scenes'] == ['0002'] and g0(jn)['compare']['all_ok'] is False


def test_start_at_the_first_frame_after_a_lost_scene_is_readable(tmp_path):
    """0002 が無く、0003 の最初のフレーム(40.033 秒。出来事の時刻は 1 ms に丸めて 40.033)に AI 側・start: 読める発進なので
    読めない記録にしない。JSON・標準出力とも直す前と同じ"""
    D, log = round3(tmp_path, 'TAIL')
    (jn, on, tn), (jo, oo, to) = both(D, 'TAIL', IG)
    assert len([r for r in jn['launches'] if r['log'] == log]) == 1       # 読める発進(車両の select で 1 フレームあと)
    # 出来事の時刻(1 ms に丸めた 40.033)は 0003 の最初のフレーム(40.0333…)より前 = 幅 LOST_TAIL_S が無いと「欠けたシーンの間」になる形
    ev = json.loads((D.raw / log / 'events.jsonl').read_text(encoding='utf-8').splitlines()[0])
    f0 = int(np.load(D.raw / log / ('%s-0003' % log) / 'AI_CMD.npz')['timestamps_us'][0])
    assert round((ev['cam_t_us'] - T0_US) / 1e6, 3) == 40.033 < (f0 - T0_US) / 1e6 and log in jn['problems']
    assert tn == to and no_path(on) == no_path(oo) and 'unreadable' not in jn


def test_unreadable_list_stays_in_log_order(tmp_path):
    """読めた本の欠けたシーンの間の発進(あとで足す)と、読めない本(先に足す)が混ざっても、読めない記録は記録の名前の順"""
    D, log = round3(tmp_path, 'C')
    late, _ = D.add_real(19, ON, T.HAND, dur=30.0)
    for p in (D.raw / late).rglob('*.npz'):
        p.write_bytes(b'')                          # 全部 0 バイト: load が読めない(r2-rec2-5 の読めない記録)
    jn, on, tn = T.runc(D, 'n6_order', path=NEW, args=IG)
    assert [u['log'] for u in jn['unreadable']] == sorted([log, late]) and log < late
    assert g0(jn)['unreadable']['logs'] == [log, late]


# ================================================================== その場面が無ければ前と同じ
def test_noai_book_with_frames_is_not_unreadable(tmp_path):
    """AI 側にしていない本で、欠けたシーンにもフレームがある(VEHICLE だけ無い = AI_CMD の select で人側と分かる): 読めない記録に
    しない(比べは直す前と同じ)。「記録が欠けている本」の行と JSON の problems には出す(前はどちらにも出なかった)"""
    D, log = round3(tmp_path, 'D')
    (jn, on, tn), (jo, oo, to) = both(D, 'D', IG)
    assert 'unreadable' not in jn and jn['summary'] == jo['summary']
    assert {k: v for k, v in jn.items() if k != 'problems'} == {k: v for k, v in jo.items() if k != 'problems'}
    assert jn['problems'] == {log: ['%s-0002/VEHICLE: ありません' % log]} and jo['problems'] == {}
    [c1] = c1_lines(on)
    assert c1.startswith('記録が欠けている本 %s(11/03 回3・発進の前進 ON): シーン 0002 の npz が無い・読めない(1 件)。AI 側のフレームが無い'
                         '(欠けたシーンにもフレームがあり、どこも AI 側にしていない = 発進なし。比べには入らない)' % log)
    assert c1_lines(oo) == [] and not any(x.startswith('skip ') for x in on.splitlines())


@pytest.mark.parametrize('args', [IG, IGF])
def test_noai_complete_book_is_unchanged(tmp_path, args):
    """AI 側にしていない本で欠けも無い: JSON・標準出力とも直す前と同じ(前のとおり黙って飛ばす)"""
    D, log = round3(tmp_path, 'F')
    (jn, on, tn), (jo, oo, to) = both(D, 'F' + str(len(args)), args)
    assert tn == to and no_path(on) == no_path(oo) and named(on, log) == []


def _write_labels(raw, lab, log):
    info, z = le.label_log(log, raw)
    lab.mkdir(exist_ok=True)
    (lab / (log + '.json')).write_text(json.dumps(info, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                                       encoding='utf-8')
    np.savez(lab / (log + '.npz'), **z)


def test_default_days_are_bit_identical(tmp_path):
    """引数なし(10/23・10/24 を data/labels で読む)は、AI 側のシーンごと落ちた本があっても直す前とビット単位で同じ"""
    raw, lab, out = tmp_path / 'raw', tmp_path / 'lab', tmp_path / 'out'
    raw.mkdir()
    a, b = '20261023-133000', '20261023-134000'
    write_standing_log(raw, a, dur=30.0, t_ai=1.3, t_move=1.4, ramp_s=0.5, run='runB', description='ai_no=1 test', params=dict(T.G))
    write_standing_log(raw, b, dur=40.0, t_ai=25.0, t_move=25.1, ramp_s=0.5, first_anchor=28.0, hits=[(31.0, 9.0)], run='runB',
                       description='ai_no=2 test', params=dict(ON), launch=(25.0, 0.5, 1))
    sc = raw / b / ('%s-0002' % b)
    for p in sc.glob('*.npz'):
        p.unlink()
    (sc / 'CAM_FRONT').mkdir()
    for lg in (a, b):
        _write_labels(raw, lab, lg)
    res = []
    for name, path in (('n6_default', NEW), ('o6_default', OLD)):
        m = load_mod(name, path)
        m.RAW, m.LAB, m.OUT = raw, lab, out / name
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main([])
        res.append(((out / name / 'launches.json').read_bytes(), no_path(buf.getvalue())))
    assert res[0] == res[1]
    assert b not in res[0][1]


# ================================================================== 手順書 7・8 章
def test_runbook_chapters_7_8():
    t = RUNBOOK.read_text(encoding='utf-8')
    ch7 = t[t.index('## 7. 止める条件'):t.index('## 8. 休みと夜の集計')]
    ch8 = t[t.index('## 8. 休みと夜の集計'):t.index('## 9. まだ無いもの')]
    every = [x for x in ch7.splitlines() if x.startswith('| いつでも |')][0]
    assert 'AI 側にしたシーンごと書けなかった本' in every and 'skip … AI 側のフレームが無い' in every
    head = ch8.split('### 8.1')[0]
    assert 'AI 側にしたシーンごと書けなかった本' in head and '欠けたシーンの間に発進の前進の start' in head
    [r5] = [x for x in ch8.splitlines() if x.startswith('| ⑤ |') and '「記録が欠けている本 …」=' in x]      # 8.1 の ⑤ の読み方の行
    assert '最初の周に掛かれば「最初の周の記録が欠けた発進」。10/27 fix5 c1)' not in r5          # 前の文(この形で成り立たない)は言い直した
    assert 'AI 側にしたシーンごと欠けた発進' in r5 and '発進の前進 OFF の本の、欠けたシーンの発進は数えられない' in r5
