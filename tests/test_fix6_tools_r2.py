"""直す 6-2 回目・係 tools(2026-10-27。反証 6-2 回目の指摘 w2-tools-r62-1・w2-tools-r62-2)の試験。夜の道具だけ
(⑤ experiments/20261026_fix/diag_launch/collect.py・③ tools/standing_3lap.py・④ tools/quick_eval_by_params.py)と手順書 7・8 章の文。
走行の動き(PCA9685 に書く値)・既定値は変えていない。

  w2-tools-r62-1  ⑤: fix6 の「欠けたシーンの間の events.jsonl の start で読めない記録(ON)」は events.jsonl が残る前提。recorder.add_event
    は fsync しないので、電源断では落ちる前の約 30 秒に書いた行も消え、ON の本の 2 回目の発進が最初の周で落ちると、その start の行も
    ほぼ必ず消えて「使ってよい」と出た(start の行が残った形だけ「満たさない」)
    → 発進の前進 ON の読めた本で、最後の読めるフレームのあとにフレームの無い欠けたシーン(落ちた形)があれば、start が無くても読めない
    記録(ON)。ON の手がかり = log.json・計画・events.jsonl の発進の前進の出来事・走行中に launch_boost を 1 に。--first-only で読めた
    発進のある本は入れない。OFF の本・真ん中だけ欠けた本・start で入れた本は前と同じ
  w2-tools-r62-2  ③④: 本ごとの衝突・最大 G・8G 以上・場所(④ の 1 章)に、人側にした直後(AI の区間の終わりから 0.5 秒以内)の当たりが
    入らず、car_check(r2-rec2-4 で足した)と食い違った(手順書 6 章の「左 4.2 … SW の出口・LH の当たり 0」は ④ の 1 章で読む)
    → standing_3lap.analyze_log の本の衝突と発進の「前に衝突」・その 3 周の衝突に car_check.after_ai_collisions を足す。印「うち人側に
    した直後 N」。走りながらの 3 周の窓は AI 側だけのまま

どの試験も「直す前の写し(tests/fixtures/fix6_tools_r2_before/*.orig。sha256 を照らす)で指摘が起きる」か「その場面のほかは直す前と
出力(JSON・標準出力・表)が同じ」を見る。作り物の記録だけ(tmp)。本物の data/raw・labels・装置・GPU は使わない。
環境変数 JF_FIX62_COLLECT・JF_FIX62_S3・JF_FIX62_QP で、試す版を差し替えられる(わざと壊した写しで試験が落ちるかを見るとき)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix6_tools_r2.py
"""
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

JF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
import test_fix3_r3_2_tools as T  # noqa: E402
from test_fix3_r1_2 import load_mod  # noqa: E402
from test_lap_anchor import T0_US  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402
import car_check as cc  # noqa: E402
import label_events as le  # noqa: E402
import numpy as np  # noqa: E402

FIX = JF / 'tests' / 'fixtures' / 'fix6_tools_r2_before'
OLD = FIX / 'collect.py.orig'
NEW = Path(os.environ.get('JF_FIX62_COLLECT') or (JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'))
RUNBOOK = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md')
ON = dict(T.G, launch_boost=1)
IG = ['--ignore-gain']
IGF = ['--ignore-gain', '--first-only']
DAY = '20261103'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_fixture_is_the_version_before():
    sums = dict(reversed(x.split()) for x in (FIX / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sorted(sums) == ['collect.py.orig', 'quick_eval_by_params.py.orig', 'standing_3lap.py.orig']
    for name, h in sums.items():
        assert _sha(FIX / name) == h, name
    assert 'def lost_tail' not in OLD.read_text(encoding='utf-8')
    assert 'after_ai_collisions' not in (FIX / 'standing_3lap.py.orig').read_text(encoding='utf-8')
    assert 'coll_after_ai' not in (FIX / 'quick_eval_by_params.py.orig').read_text(encoding='utf-8')


# ================================================================== ⑤ 電源断で events.jsonl の start も消えた本(w2-tools-r62-1)
def _n(name):
    return name.replace('-', '_').replace('.', '_')


def both(D, name, args=()):
    """(今, 直す前) の (JSON, 標準出力, JSON の文)"""
    return T.runc(D, 'n62_' + _n(name), path=NEW, args=args), T.runc(D, 'o62_' + _n(name), path=OLD, args=args)


def add_event(D, log, t, event='start'):
    with open(D.raw / log / 'events.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'kind': 'launch_boost', 'event': event, 'us': 250.0, 'cam_t_us': T0_US + int(round(t * 1e6)),
                            't_us': T0_US + int(round(t * 1e6)) + 500}) + '\n')


def power_cut(D, log, lost, zero):
    """電源断の本物の形(10/24 の 20261024-162305): 今のシーン lost は CAM_FRONT/ だけ、落ちる前 30 秒の中に保存したシーン zero は
    npz・scene.json まで 0 バイト"""
    sc = lambda k: D.raw / log / ('%s-%04d' % (log, k))      # noqa: E731
    for p in sc(lost).glob('*.npz'):
        p.unlink()
    (sc(lost) / 'CAM_FRONT').mkdir(exist_ok=True)
    for k in zero:
        for p in list(sc(k).glob('*.npz')) + [sc(k) / 'scene.json']:
            if p.exists():
                p.write_bytes(b'')


def round3(tmp, case):
    """反証の p1 と同じ回 3: OFF 4 本(起点まで 3.013 秒)・ON 3 本(2.8・2.7・2.6 秒)・3-8 は case の形。3-8 の名前を返す。
    C_power: ON の 80 秒の本。1 回目の発進 1.4 秒は読める(最初の周はきれい)・20 秒に人側で止めた・60 秒に 2 回目・68 秒に電源断。
    シーン 0002・0003 は 0 バイト・0004 は CAM_FRONT/ だけ。events.jsonl は 1.4 秒の start だけ(60 秒の start は同じ窓で消えた)"""
    tmp.mkdir(parents=True, exist_ok=True)
    D = T.RealDay(tmp)
    D.add_real(12, T.G, T.HAND)
    D.add_real(13, ON, T.HAND, **T.on_kw(36))
    D.add_real(14, T.G, T.HAND)
    D.add_real(15, ON, T.HAND, **T.on_kw(39))
    D.add_real(16, T.G, T.HAND)
    D.add_real(17, ON, T.HAND, **T.on_kw(42))
    D.add_real(18, T.G, T.HAND)
    if case == 'C_proc':                   # fix6 の C(jetformer-web の落ち: 45 秒の start が残り、シーン 0003 が無い)
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        add_event(D, log, 45.0)
        power_cut(D, log, 3, ())
    elif case == 'C_run':                  # AI 側のまま走っているうちに電源断(最後の読めるフレームは AI 側)
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, **T.on_kw(42))
        power_cut(D, log, 3, (2,))
    elif case == 'C_mid':                  # 真ん中のシーン 0002 だけ npz が無い(0003 は読める)。start なし
        log, _ = D.add_real(19, ON, T.HAND, dur=60.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        power_cut(D, log, 2, ())
    elif case in ('C_power_off', 'C_plan'):    # 発進の前進 OFF の本(log.json OFF・出来事なし)。C_plan は計画だけ ON
        log, _ = D.add_real(19, T.G, T.HAND, dur=80.0, human_after=20.0, stop_after=20.0, t_ai=1.4, t_move=1.5)
        power_cut(D, log, 4, (2, 3))
        if case == 'C_plan':
            b = json.loads(D.bench.read_text())
            for bl in b['days'][0]['blocks']:
                bl['expect'] = {'launch_boost': 1 if bl['block'] in (13, 15, 17, 19) else 0}
            D.bench.write_text(json.dumps(b), encoding='utf-8')
    else:
        p = dict(ON, launch_boost=0) if case in ('C_midrun', 'C_ev_only') else ON
        log, _ = D.add_real(19, p, T.HAND, dur=80.0, human_after=20.0, stop_after=20.0, **T.on_kw(42))
        if case == 'C_midrun':             # log.json は 0・30 秒に params で 0 → 1(start の行は無い)
            (D.raw / log / 'events.jsonl').write_text(json.dumps(
                {'kind': 'params', 'changed': {'launch_boost': [0, 1]}, 'epoch_us': T0_US + 30_000_000}) + '\n', encoding='utf-8')
        if case in ('C_power_ev', 'C_power_nul'):
            add_event(D, log, 60.0)
        if case == 'C_skip':
            add_event(D, log, 60.0, 'skip')
        if case == 'C_power_nul':          # 60 秒の行が NUL で残った(ext4 の大きさだけ伸びた)
            ef = D.raw / log / 'events.jsonl'
            ls_ = ef.read_bytes().split(b'\n')
            ef.write_bytes(ls_[0] + b'\n' + b'\x00' * (len(ls_[1]) + 1))
        power_cut(D, log, 4, (2, 3))
    return D, log


def g0(j):
    return j['summary'][0]


def no_path(out):
    return '\n'.join(x for x in out.splitlines() if not x.startswith('出力 '))


def c1_lines(out):
    return [x for x in out.splitlines() if x.startswith('記録が欠けている本 ')]


TAIL_SENT = re.compile(r'。最後の読めるフレーム [^→]*?(外すと読めた発進も比べから外れる\)|--first-only の比べには入れない)')


def strip_tail(j, out, log):
    """今の ⑤ の出力から、fix6-2 で足した「最後の読めるフレームのあとが欠けた ON の本」(log)の読めない記録と文を除いた形(直す前の
    出力と比べるため。その本のほかに読めない記録の無い日だけ)。ほかの試験(test_fix5_tools・test_fix3_r2_2)も使う"""
    j = json.loads(json.dumps(j))
    ur = j.pop('unreadable', None)
    if ur is not None:
        assert [u['log'] for u in ur] == [log] and 'lost_after_s' in ur[0], ur
    for g in j.get('summary') or []:
        gu = g.pop('unreadable', None)
        if gu is not None:
            assert gu['logs'] == [log]
            c = g.get('compare')
            if c:
                assert c['checks'].pop('on_all_readable') is False
                c['all_ok'] = all(c['checks'].values())
    keep = []
    for x in out.splitlines():
        if x.startswith(('  読めない記録 1 本(比べの数に入れない): %s(シーン ' % log, '  → 読めない(欠けた)ON の本がある')):
            continue
        if 'ON − OFF' in x and ", 'on_all_readable': False}" in x:
            x = x.replace(", 'on_all_readable': False}", '}')
            if 'False' not in x.split('{', 1)[1]:
                x = x.replace('→ 満たさない {', '→ 使ってよい {')
        if x.startswith('記録が欠けている本 %s(' % log):
            x = TAIL_SENT.sub('', x)
        keep.append(x)
    return j, '\n'.join(keep)


WHY3 = ('シーン 0002・0003・0004(最後の読めるフレーム 19.97 秒のあと)にフレームが無い(電源断・落ちの形。npz が無い・0 バイト・読めない)。'
        '発進の前進 ON の本なので、その間の発進(2 回目以降)の記録が無いかもしれない(電源断では events.jsonl の start も消える)')


@pytest.mark.parametrize('case,src', [
    ('C_power', ['log.json', 'events.jsonl の発進の前進の出来事(start)']),
    ('C_power_nul', ['log.json', 'events.jsonl の発進の前進の出来事(start)']),
    ('C_skip', ['log.json', 'events.jsonl の発進の前進の出来事(start・skip)']),
])
def test_power_cut_second_launch_is_unreadable_now(tmp_path, case, src):
    """直す前: 60 秒の start の行が消えた(NUL で残った・skip だけ残った)ので、2 回目の発進が黙って消え「使ってよい」(反証の C_power)。
    今: 最後の読めるフレームのあとが欠けた ON の本として読めない記録(ON)・「満たさない」。発進の行・by_launch・ほかの数は同じ"""
    D, log = round3(tmp_path, case)
    (jn, on, tn), (jo, oo, to) = both(D, case, IG)
    # 直す前(指摘の再現)
    assert g0(jo)['compare']['all_ok'] is True and 'unreadable' not in jo and 'on_all_readable' not in g0(jo)['compare']['checks']
    assert g0(jo)['by_launch']['on']['n'] == 4 and '→ 使ってよい' in oo
    # 今
    g = g0(jn)
    assert g['compare']['all_ok'] is False and g['compare']['checks'] == dict(g0(jo)['compare']['checks'], on_all_readable=False)
    assert {k: v for k, v in g['compare'].items() if k not in ('checks', 'all_ok')} == \
        {k: v for k, v in g0(jo)['compare'].items() if k not in ('checks', 'all_ok')}
    assert g['by_launch'] == g0(jo)['by_launch'] and jn['launches'] == jo['launches'] and jn['problems'] == jo['problems']
    [u] = jn['unreadable']
    assert u['log'] == log and u['launch_state'] == 'on' and u['lost_scenes'] == ['0002', '0003', '0004']
    assert u['lost_after_s'] == pytest.approx(599 / 30.0, abs=1e-3) and u['launch_on_from'] == src and 'launch_starts' not in u
    assert u['why'] == WHY3
    assert g['unreadable']['logs'] == [log] and g['unreadable']['on'] == 1 and g['unreadable']['launch_on_from'] == [src]
    assert 'launch_starts' not in g['unreadable']
    lines = on.splitlines()
    ur = [x for x in lines if x.startswith('  読めない記録 1 本')]
    other = '・'.join(x for x in src if x != 'log.json')
    assert ur == ['  読めない記録 1 本(比べの数に入れない): %s(%s。発進の前進 ON: log.json ON・計画 ?・%s)' % (log, WHY3, other)]
    assert any(x.startswith('  → 読めない(欠けた)ON の本がある = 満たさない') for x in lines)
    assert not any(x.startswith('skip ') for x in lines)              # 読めた本なので skip の行は無い
    [c1n], [c1o] = c1_lines(on), c1_lines(oo)
    assert c1n == c1o.replace('で満たさない)→', 'で満たさない)。最後の読めるフレーム 19.97 秒のあとのシーン 0002・0003・0004 にフレームが無い'
                              '(落ちた形)= 発進の前進 ON の本なので、その間の発進(2 回目以降)の記録が無いかもしれない(電源断では events.jsonl'
                              ' の start も消える)。「読めない記録」に入れた(ON なので満たさない。動画・紙で、その間に発進が無かった・当たって'
                              'いないと確かめたら --skip 記録名 で外せる。外すと読めた発進も比べから外れる)→')
    js, os_ = strip_tail(jn, on, log)             # その分を除けば直す前と同じ
    assert js == jo and no_path(os_) == no_path(oo)


@pytest.mark.parametrize('case', ['C_power', 'C_power_nul', 'C_run'])
def test_power_cut_first_only_is_a_later_launch(tmp_path, case):
    """--first-only: 読めた発進(1.4 秒)があるので、欠けた区間の発進は 2 回目以降 = 比べに入れない(fix6 の決まりと同じ)。JSON は直す前と
    同じ。「記録が欠けている本」の行の後ろの 1 文だけ違う"""
    D, log = round3(tmp_path, case)
    (jn, on, tn), (jo, oo, to) = both(D, case + 'f', IGF)
    assert tn == to and g0(jn)['compare']['all_ok'] is True
    [c1n], [c1o] = c1_lines(on), c1_lines(oo)
    assert c1n.endswith('読めた発進のあとなので --first-only の比べには入れない→ 動画・紙で(手順書 7 章「いつでも」)')
    assert TAIL_SENT.sub('', c1n) == c1o
    rest = lambda o: [x for x in no_path(o).splitlines() if not x.startswith('記録が欠けている本 ')]      # noqa: E731
    assert rest(on) == rest(oo)


@pytest.mark.parametrize('args', [IG, IGF])
@pytest.mark.parametrize('case', ['C_power_ev', 'C_proc', 'C_power_off', 'C_mid'])
def test_unchanged_shapes(tmp_path, case, args):
    """前と同じ形: start の行が残った(fix6 の決まりで読めない記録。二重に入れない)・jetformer-web の落ち(fix6 の C)・発進の前進 OFF の本
    (欠けた区間の発進は記録に手がかりが無い)・真ん中だけ欠けた本(あとのシーンが読める = 落ちた形でない)。JSON・標準出力とも直す前と同じ"""
    D, log = round3(tmp_path, case)
    (jn, on, tn), (jo, oo, to) = both(D, case + str(len(args)), args)
    assert tn == to and no_path(on) == no_path(oo)
    assert all('lost_after_s' not in u for u in jn.get('unreadable') or [])
    if case == 'C_power_ev' and args == IG:
        assert [u['log'] for u in jn['unreadable']] == [log] and jn['unreadable'][0]['launch_starts'] == [60.0]


def test_running_at_the_last_frame_is_unreadable(tmp_path):
    """AI 側のまま走っているうちに電源断(最後の読めるフレームは AI 側。最初の周は記録の中): その間に止めて戻した 2 回目の発進が
    あっても記録に無いので、同じく読めない記録(ON)"""
    D, log = round3(tmp_path, 'C_run')
    (jn, on, tn), (jo, oo, to) = both(D, 'C_run', IG)
    assert g0(jo)['compare']['all_ok'] is True and 'unreadable' not in jo
    [u] = jn['unreadable']
    assert u['log'] == log and u['lost_scenes'] == ['0002', '0003'] and g0(jn)['compare']['all_ok'] is False
    assert 'first_lap_gap_on' not in g0(jn)                 # 最初の周は記録の中
    js, os_ = strip_tail(jn, on, log)
    assert js == jo and no_path(os_) == no_path(oo)


@pytest.mark.parametrize('case,src,lj', [
    ('C_midrun', ['走行中に launch_boost を 1 に(30.0 秒)'], 'OFF'),
    ('C_ev_only', ['events.jsonl の発進の前進の出来事(start)'], 'OFF'),
    ('C_plan', ['計画'], 'OFF'),
])
def test_on_from_other_clues(tmp_path, case, src, lj):
    """log.json は OFF でも、ON の手がかり(走行中に launch_boost を 1 に・発進の前進の出来事・計画)があれば ON として読めない記録。
    行に手がかりを出し、「記録が欠けている本」の頭も「発進の前進 ON」"""
    D, log = round3(tmp_path, case)
    (jn, on, tn), (jo, oo, to) = both(D, case, IG)
    assert g0(jo)['compare']['all_ok'] is True and 'unreadable' not in jo
    [u] = jn['unreadable']
    assert u['log'] == log and u['launch_state'] == 'on' and u['launch_on_from'] == src and g0(jn)['compare']['all_ok'] is False
    [ur] = [x for x in on.splitlines() if x.startswith('  読めない記録 1 本')]
    tail = '・%s' % src[0] if src != ['計画'] else ''
    assert ur.endswith('発進の前進 ON: log.json %s・計画 %s%s)' % (lj, 'ON' if case == 'C_plan' else '?', tail))
    [c1] = c1_lines(on)
    assert c1.startswith('記録が欠けている本 %s(11/03 回3・発進の前進 ON)' % log)


def test_noai_book_on_from_skip_event(tmp_path):
    """AI のフレームが無い本(fix6 の B の形)で、log.json は OFF・start は無く、発進の前進の skip の出来事だけが残った: 前は OFF の読めない
    記録(比べは変えない)。今は ON の手がかりがあるので ON(満たさない)。ほかは同じ"""
    D, log = round3(tmp_path, 'C_power_off')            # 回 3 の 7 本(3-8 は下で作り直す)
    shutil.rmtree(D.raw / log)
    D.n -= 1
    log, _ = D.add_real(19, T.G, T.HAND, dur=40.0, t_ai=25.0, t_move=25.1, first_anchor=28.0, hits=[(31.0, 9.0)])
    add_event(D, log, 25.0, 'skip')
    sc = D.raw / log / ('%s-0002' % log)
    for p in sc.glob('*.npz'):
        p.unlink()
    (sc / 'CAM_FRONT').mkdir()
    (jn, on, tn), (jo, oo, to) = both(D, 'noai_skip', IG)
    [uo] = jo['unreadable']
    assert uo['log'] == log and uo['launch_state'] == 'off' and g0(jo)['compare']['all_ok'] is True
    [u] = jn['unreadable']
    assert u['launch_state'] == 'on' and u['launch_on_from'] == ['events.jsonl の発進の前進の出来事(skip)']
    assert {k: v for k, v in u.items() if k not in ('launch_state', 'launch_on_from')} == \
        {k: v for k, v in uo.items() if k != 'launch_state'}
    assert g0(jn)['compare']['all_ok'] is False and g0(jn)['compare']['checks']['on_all_readable'] is False
    assert ('・計画 ?・events.jsonl の発進の前進の出来事(skip))' in [x for x in on.splitlines() if x.startswith('  読めない記録')][0])


def test_noai_book_unchanged_when_already_on(tmp_path):
    """AI のフレームが無い本で、もう ON(log.json)のときは前と同じ(launch_on_from を足さない。手がかりの無い OFF の本は
    tests/test_fix6_tools.py の E の形)"""
    D, log = round3(tmp_path, 'C_power_off')
    shutil.rmtree(D.raw / log)
    D.n -= 1
    log, _ = D.add_real(19, ON, T.HAND, dur=40.0, t_ai=25.0, t_move=25.1, first_anchor=28.0)
    (D.raw / log / 'events.jsonl').unlink(missing_ok=True)
    sc = D.raw / log / ('%s-0002' % log)
    for p in sc.glob('*.npz'):
        p.unlink()
    (sc / 'CAM_FRONT').mkdir()
    (jn, on, tn), (jo, oo, to) = both(D, 'noai_on', IG)
    assert tn == to and no_path(on) == no_path(oo) and jn['unreadable'][0]['launch_state'] == 'on'


def test_skip_removes_the_tail_book(tmp_path):
    """動画・紙で確かめた本は --skip 記録名 で外せる(本ごと。外すと直す前の --skip と同じ答え・外した名前が出る)"""
    D, log = round3(tmp_path, 'C_power')
    (jn, on, tn), (jo, oo, to) = both(D, 'skip', IG + ['--skip', log])
    assert g0(jn)['compare'] == g0(jo)['compare'] and g0(jn)['by_launch']['on']['n'] == 3
    assert jn['summary_skipped'] == ['%s@1.40' % log, '%s(読めない記録)' % log] and jo['summary_skipped'] == ['%s@1.40' % log]
    assert 'unreadable' not in g0(jn) and [u['log'] for u in jn['unreadable']] == [log]


def test_unreadable_list_stays_in_log_order(tmp_path):
    """最後のフレームのあとが欠けた本(あとで足す)と、読めない本(先に足す)が混ざっても、読めない記録は記録の名前の順"""
    D, log = round3(tmp_path, 'C_power')
    late, _ = D.add_real(19, ON, T.HAND, dur=30.0)
    for p in (D.raw / late).rglob('*.npz'):
        p.write_bytes(b'')
    jn, on, tn = T.runc(D, 'n62_order', path=NEW, args=IG)
    assert [u['log'] for u in jn['unreadable']] == sorted([log, late]) and log < late
    assert g0(jn)['unreadable']['logs'] == [log, late] and 'lost_after_s' in jn['unreadable'][0]


def test_first_lap_gap_book_is_also_a_tail_book(tmp_path):
    """最初の周の途中で電源断(最初の周の記録が欠けた発進 = r2-rec2-5): 前から「満たさない」。最後のフレームのあとも欠けているので
    読めない記録にも入る(どちらも動画・紙で確かめ、--skip 記録名 で一緒に外れる)"""
    D, log = round3(tmp_path, 'C_power_off')
    shutil.rmtree(D.raw / log)
    D.n -= 1
    log, _ = D.add_real(19, ON, T.HAND, dur=40.0, t_ai=15.0, t_move=15.1, launch=(15.0, 0.5, 1), first_anchor=18.0)
    power_cut(D, log, 2, ())
    (jn, on, tn), (jo, oo, to) = both(D, 'gap', IG)
    assert g0(jo)['first_lap_gap_on'] and g0(jo)['compare']['all_ok'] is False and 'unreadable' not in jo
    assert g0(jn)['first_lap_gap_on'] == g0(jo)['first_lap_gap_on'] and [u['log'] for u in jn['unreadable']] == [log]
    js, os_ = strip_tail(jn, on, log)
    js['summary'][0]['compare']['checks']['on_all_readable'] = False          # first_lap_gap で前から False
    js['summary'][0]['compare']['all_ok'] = False
    assert js == jo
    (jk, ok, tk), (jko, oko, tko) = both(D, 'gapskip', IG + ['--skip', log])
    assert g0(jk)['compare'] == g0(jko)['compare'] and 'first_lap_gap_on' not in g0(jk) and 'unreadable' not in g0(jk)


def _write_labels(raw, lab, log):
    info, z = le.label_log(log, raw)
    lab.mkdir(exist_ok=True)
    (lab / (log + '.json')).write_text(json.dumps(info, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                                       encoding='utf-8')
    np.savez(lab / (log + '.npz'), **z)


def test_default_days_are_bit_identical(tmp_path):
    """引数なし(10/23・10/24 を data/labels で読む)は、最後のフレームのあとが欠けた ON の本があっても直す前とビット単位で同じ"""
    raw, lab, out = tmp_path / 'raw', tmp_path / 'lab', tmp_path / 'out'
    raw.mkdir()
    a, b = '20261023-133000', '20261023-134000'
    write_standing_log(raw, a, dur=30.0, t_ai=1.3, t_move=1.4, ramp_s=0.5, run='runB', description='ai_no=1 test', params=dict(T.G))
    write_standing_log(raw, b, dur=60.0, t_ai=1.4, t_move=1.5, ramp_s=0.5, human_after=20.0, stop_after=20.0, run='runB',
                       description='ai_no=2 test', params=dict(ON), launch=(1.4, 0.5, 1))
    for k in (2, 3):
        for p in (raw / b / ('%s-%04d' % (b, k))).glob('*.npz'):
            p.unlink() if k == 3 else p.write_bytes(b'')
    for lg in (a, b):
        _write_labels(raw, lab, lg)
    res = []
    for name, path in (('n62_default', NEW), ('o62_default', OLD)):
        m = load_mod(name, path)
        m.RAW, m.LAB, m.OUT = raw, lab, out / name
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main([])
        res.append(((out / name / 'launches.json').read_bytes(), no_path(buf.getvalue())))
    assert res[0] == res[1]


# ================================================================== ③④ 人側にした直後の衝突(w2-tools-r62-2)
def _pair(p3, pq, tag):
    """(standing_3lap, quick_eval_by_params) の組をファイルから読む。④ は渡した ③ を standing_3lap として読む"""
    s3m = load_mod('standing_3lap_%s' % tag, p3)
    saved = sys.modules.get('standing_3lap')
    sys.modules['standing_3lap'] = s3m
    try:
        qpm = load_mod('qp_%s' % tag, pq)
    finally:
        if saved is not None:
            sys.modules['standing_3lap'] = saved
        else:
            sys.modules.pop('standing_3lap', None)
    return s3m, qpm


@pytest.fixture(scope='module')
def new():
    """今の ③④(JF_FIX62_S3・JF_FIX62_QP があれば、その写し)"""
    if not os.environ.get('JF_FIX62_S3') and not os.environ.get('JF_FIX62_QP'):
        import standing_3lap as s3
        import quick_eval_by_params as qp
        return s3, qp
    return _pair(os.environ.get('JF_FIX62_S3') or (JF / 'tools' / 'standing_3lap.py'),
                 os.environ.get('JF_FIX62_QP') or (JF / 'tools' / 'quick_eval_by_params.py'), 'f62new')


@pytest.fixture(scope='module')
def old():
    """直す前の ③④ の写し"""
    return _pair(FIX / 'standing_3lap.py.orig', FIX / 'quick_eval_by_params.py.orig', 'f62old')


def flip_log(raw, flip, hit=(8.0, 9.0), log='20261103-140000', **kw):
    """反証の probe_flip と同じ(test_fix3_r2_2.flip_log): 8.0 秒に 9G(DE)で当たり 8.05 秒に止まる。人側 flip 秒"""
    write_standing_log(raw, log, dur=kw.pop('dur', 12.0), hits=[hit], stop_after=hit[0] + 0.05, human_after=flip, run='runB',
                       description='ai_no=15 day=2026-11-03 block=15 memo=x', params={'throttle_gain': 0.9, 'launch_boost': 1}, **kw)
    return log


def s3run(mod, raw, logs):
    return mod.run(logs, raw, raw / 'nolab', None)


@pytest.mark.parametrize('flip,dt', [(7.95, 0.07), (7.7, 0.336)])
def test_after_ai_hit_counts_in_book_collisions(tmp_path, new, old, flip, dt):
    """8.0 秒に 9G(DE)・人側 7.95 秒 / 7.7 秒(AI の終わりのすぐあとに当たる)。直す前: ③ の本の衝突 0・④ の 1 章「0(0) | —」
    (car_check は 1 回・9G・DE 1)。今: ③ の本の衝突・最大 G・8G 以上・場所が car_check と同じ・印「うち人側にした直後 1」"""
    s3n, qpn = new
    s3o, qpo = old
    log = flip_log(tmp_path, flip)
    rn, ro = s3run(s3n, tmp_path, [log]), s3run(s3o, tmp_path, [log])
    assert ro[0]['collisions']['all'] == 0 and ro[0]['collisions']['max_g'] is None              # 指摘の再現
    C = rn[0]['collisions']
    k = cc.check_log(log, tmp_path / 'nolab', tmp_path)['collisions']
    assert k['n'] == 1 and k['by_class'] == {'after_ai': 1}                                       # car_check(変えていない)
    assert (C['all'], C['strong'], C['max_g'], C['n_ge8'], C['by_place']) == (k['n'], k['strong'], k['max_g'], k['n_ge8'], k['by_place'])
    assert C['after_ai'] == k['after_ai'] and C['after_ai'][0][3] == pytest.approx(dt, abs=0.005)
    assert C['ai_all'] == 0 and C['ai_strong'] == 0
    assert C['text'] == ('1 回・最大 9.00G・8G 以上 1・場所 DE 1。うち人側にした直後(AI の区間の終わりから 0.5 秒以内)1: 8.0 秒 DE 9.00G'
                         '(AI の終わりから %.2f 秒)' % C['after_ai'][0][3])
    assert {kk: v for kk, v in rn[0].items() if kk != 'collisions'} == {kk: v for kk, v in ro[0].items() if kk != 'collisions'}
    # ④ の 1 章
    mdn, runs, groups = qpn.to_markdown(rn, DAY, None)
    mdo, runo, groupo = qpo.to_markdown(ro, DAY, None)
    rown = [x for x in mdn.splitlines() if x.startswith('| 15 | %s |' % log)][0]          # 1 章の行(3 章の発進の行より前)
    rowo = [x for x in mdo.splitlines() if x.startswith('| 15 | %s |' % log)][0]
    assert '| 0(0) | — |' in rowo
    assert '| 1(1) | 9.00G・1・DE 1(うち人側にした直後 1: DE 9.00G) |' in rown
    assert runs[0]['coll_after_ai'] == C['after_ai'] and (runs[0]['collisions'], runs[0]['coll_max_g'], runs[0]['coll_ge8'],
                                                          runs[0]['coll_places']) == (1, 9.0, 1, 'DE 1')
    assert {kk: v for kk, v in runs[0].items() if not kk.startswith('coll') and kk != 'strong'} == \
        {kk: v for kk, v in runo[0].items() if not kk.startswith('coll') and kk != 'strong'}
    assert groups[0]['collisions_after_ai'] == 1 and groups[0]['collisions'] == 1 and groups[0]['strong'] == 1
    assert groupo[0]['collisions'] == 0
    assert any(x.startswith('- 衝突(1 章の衝突(強)・最大 G・8G 以上・場所、2 章の衝突/通過・強い衝突)には、人側にした直後') for x in mdn.splitlines())
    # ③ の表(2 章の衝突(強))
    m3n, m3o = s3n.to_markdown(rn), s3o.to_markdown(ro)
    assert s3n.AFTER_AI_NOTE in m3n.splitlines() and '| 0(0) |' in [x for x in m3o.splitlines() if log in x][-1]
    assert [x for x in m3n.splitlines() if log in x][-1].endswith('| 1(1)(うち人側にした直後 1: DE 9.00G) |')
    rest = lambda m: [x for x in m.splitlines() if log not in x and x != s3n.AFTER_AI_NOTE]      # noqa: E731
    assert rest(m3n) == rest(m3o)


@pytest.mark.parametrize('flip', [8.2, 7.4, None])
def test_after_ai_unchanged_without_such_hits(tmp_path, new, old, flip):
    """当たったあとに人側(AI 側の衝突。前から数える)・0.6 秒より前に人側(数えない限り)・人側にしない: ③ の JSON・表、④ の表・本・組が
    直す前と同じ"""
    s3n, qpn = new
    s3o, qpo = old
    log = flip_log(tmp_path, flip) if flip is not None else flip_log(tmp_path, 99.0)
    rn, ro = s3run(s3n, tmp_path, [log]), s3run(s3o, tmp_path, [log])
    assert rn == ro and 'after_ai' not in rn[0]['collisions']
    assert s3n.to_markdown(rn) == s3o.to_markdown(ro)
    for ig in ((), ('gain',)):
        a, b = qpn.to_markdown(rn, DAY, None, ig), qpo.to_markdown(ro, DAY, None, ig)
        assert a == b


def test_after_ai_hit_counts_as_collision_before_the_next_launch(tmp_path, new, monkeypatch):
    """発進の「前に衝突」(coll_before = 数えない)と、その 3 周の衝突も同じ数え方: analyze_log が standing_starts に渡す衝突に、人側に
    した直後の衝突が入る(当たる直前に人側にして当て、置き直して 2 回目の発進をしたとき、2 回目を「衝突 0 の止まってから 3 周」に
    数えない)。走りながらの 3 周の窓(flying_windows)は AI 側だけのまま"""
    s3n, _ = new
    log = flip_log(tmp_path, 7.7)
    seen = {}
    real_ss, real_fw = s3n.standing_starts, s3n.flying_windows

    def ss_(*a, **k):
        seen['standing'] = list(a[7])
        return real_ss(*a, **k)

    def fw_(laps, coll):
        seen['flying'] = list(coll)
        return real_fw(laps, coll)
    monkeypatch.setattr(s3n, 'standing_starts', ss_)
    monkeypatch.setattr(s3n, 'flying_windows', fw_)
    s3n.analyze_log(log, tmp_path, tmp_path / 'nolab', None)
    assert [(c['t_s'], c.get('class')) for c in seen['standing']] == [(pytest.approx(8.0, abs=0.02), 'after_ai')]
    assert seen['flying'] == []
    # standing_starts の coll_before は渡した衝突で数える(2 回目の発進があれば 1)
    rows = real_ss(np.arange(0, 2, 1 / 30.0), np.r_[np.zeros(30, bool), np.ones(30, bool)], np.arange(0, 2, 0.01),
                   np.zeros(200), np.arange(0, 2, 0.01), np.zeros(200), [], [{'t_s': 0.5, 'g': 9.0}])
    assert rows[0]['coll_before'] == 1 and rows[0]['start_no'] == 1


# ================================================================== 手順書 7・8 章
def test_runbook_chapters_7_8():
    t = RUNBOOK.read_text(encoding='utf-8')
    ch7 = t[t.index('## 7. 止める条件'):t.index('## 8. 休みと夜の集計')]
    ch8 = t[t.index('## 8. 休みと夜の集計'):t.index('## 9. まだ無いもの')]
    every = [x for x in ch7.splitlines() if x.startswith('| いつでも |')][0]
    assert '2 回目の発進のシーンごと落ちた本も、欠けたシーンの間の start で同じ(10/27 fix6。' not in every     # 電源断では成り立たない文
    assert '電源断では events.jsonl の start も消える' in every and '最後の読めるフレームのあと' in every
    head = ch8.split('### 8.1')[0]
    assert '(出来事はすぐ書くので落ちても残る)' not in head and '電源断では' in head and '最後の読めるフレームのあと' in head
    [r5] = [x for x in ch8.splitlines() if x.startswith('| ⑤ |') and '「記録が欠けている本 …」=' in x]
    assert '最後の読めるフレームのあと' in r5 and '電源断' in r5
    [r4] = [x for x in ch8.splitlines() if x.startswith('| ④ |') and 'quick_eval_by_params' in x]
    assert '人側にした直後' in r4
    [r3] = [x for x in ch8.splitlines() if x.startswith('| ③ |') and 'standing_3lap' in x]
    assert '人側にした直後' in r3
