"""直す 2 回目・2 組目(2026-10-27 fix3。反証 2 回目の指摘 r2-rec2-2・r2-rec2-3・r2-rec2-4・r2-rec2-5)の試験。夜の道具だけ
(⑤ experiments/20261026_fix/diag_launch/collect.py と tools/car_check.py)。走行の動き(PCA9685 に書く値)は変えていない。

  r2-rec2-2  collect.py: 回 3 を全部「止まって 1 周」で走ると、走りながらの周が無く LH の入口の速さが出ず、いつも「満たさない」
  r2-rec2-3  collect.py: R1 を越えたあとの最初の周で、人が当たる直前に人側にした ON の衝突を数えず「使ってよい」
  r2-rec2-4  car_check: 人が当たる直前に人側にした 9G が「衝突 0 回」(手順書 7 章の「8G 以上 1 回」が読めない)
  r2-rec2-5  collect.py: npz が無い(電源断・落ち)ON の本が「skip」1 行だけで比べから消え「使ってよい」

どの試験も「直す前の写し(tests/fixtures/fix3_r2_2_before/*.orig。sha256 を照らす)で指摘が起きる」と「その場面のほかは直す前と
出力(JSON・標準出力・文)が同じ」を見る。作り物の記録だけ(tmp)。本物の data/raw・labels・装置・GPU は使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix3_r2_2.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from test_fix3_r1_2 import Day, ON, PRM, _row, _same_day, load_mod
import car_check as cc  # noqa: E402
from test_lap_anchor import write_lap_log  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

JF = Path(__file__).resolve().parents[1]
BEFORE = JF / 'tests' / 'fixtures' / 'fix3_r2_2_before'
COLLECT = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
OLD_COLLECT = BEFORE / 'collect.py.orig'


def test_before_copies():
    sums = dict(reversed(x.split()) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert sorted(sums) == ['car_check.py.orig', 'collect.py.orig']
    for name, h in sums.items():
        assert hashlib.sha256((BEFORE / name).read_bytes()).hexdigest() == h, name


@pytest.fixture(scope='module')
def old_cc():
    return load_mod('car_check_r22_before', BEFORE / 'car_check.py.orig')


@pytest.fixture(scope='module')
def col():
    return load_mod('collect_r22_mod', COLLECT)


def sub(tmp, name):
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def no_path(out):
    return '\n'.join(x for x in out.splitlines() if not x.startswith('出力 '))


def both(D, name, args=()):
    """今の collect.py と直す前の写し"""
    return D.run(name='n_' + name, args=args), D.run(path=OLD_COLLECT, name='o_' + name, args=args)


def same_output(D, name, args=()):
    (jn, on, tn, _), (jo, oo, to, _) = both(D, name, args)
    return tn == to and no_path(on) == no_path(oo), jn          # (JSON に出力の場所は入らない)


def cmp_line(out):
    return [x for x in out.splitlines() if 'ON − OFF' in x]


# ================================================================== r2-rec2-2 周の無い回 3 の LH
def round3(D, kw_of):
    logs = []
    for i in range(8):
        lg, _ = D.add(12 + i, ON if i % 2 else PRM, **kw_of(i))
        logs.append(lg)
    return logs


@pytest.mark.parametrize('kw', [dict(dur=13.0, human_after=12.0), dict(dur=25.0, human_after=12.0)])
def test_rec22_one_lap_round_compares_lh_at_launch_frames(tmp_path, kw):
    """反証 P1: 8 本とも 1 周(線のあと LH・A2 の前で人側)。直す前: LH — → 満たさない {'lh_speed': False}。
    今: 発進のフレームの場所(LH)で ON・OFF とも比べ、ほかの条件と同じく「使ってよい」"""
    D = Day(tmp_path)
    round3(D, lambda i: dict(kw))
    (jn, on, tn, _), (jo, oo, _, _) = both(D, 'p1')
    go = jo['summary'][0]
    assert go['compare']['checks'] == {'to_r1': True, 'first_lap_collisions_0': True, 'slip': True, 'lh_speed': False}
    assert '→ 満たさない' in oo and 'LH — m/s' in oo                                         # 指摘の再現
    assert all(r['fly_ref'] == 'none' and 'v_launch_at_lh' not in r for r in jn['launches'])
    assert all(r['lh_self']['v_mps'] == pytest.approx(2.0, abs=0.01) for r in jn['launches'])  # 周の無い発進の行に出る
    assert '"_lh_self"' not in tn                                                            # 中だけの欄は JSON に書かない
    g = jn['summary'][0]
    c = g['compare']
    assert c['lh_src'] == 'launch_loc' and c['d_v_lh_mps'] == pytest.approx(0.0, abs=1e-9)
    assert c['checks'] == {'to_r1': True, 'first_lap_collisions_0': True, 'slip': True, 'lh_speed': True} and c['all_ok'] is True
    assert g['by_launch']['on']['v_lh_self_median_mps'] == pytest.approx(2.0, abs=0.01)
    assert 'LH 0.000(発進のフレームの場所で) m/s' in on and '→ 使ってよい' in on
    assert 'LH の入口の速さは、周の無い発進があるので ON・OFF とも発進のフレームの場所(LH)で比べた' in on
    assert 'LH —(発進の場所で 2.000) m/s' in on
    # 行(lh_self のほか)・まとめのほかの数は直す前と同じ。記録が 13 秒で終わる形は A2 が無く 'ai_end'(線が分からない)なので、
    # 最初の周の終わりだけ r2-rec2-3 の 0.5 秒が足される(人が走り続ける形は A2 があり 'line' で同じ)
    strip = lambda r: {k: v for k, v in r.items() if k not in ('lh_self', 'first_lap_end_s')}      # noqa: E731
    assert [strip(r) for r in jn['launches']] == [strip(r) for r in jo['launches']]
    for a, b in zip(jn['launches'], jo['launches']):
        add = 0.5 if (kw['dur'] == 13.0) else 0.0
        assert a['first_lap_end'] == ('ai_end' if add else 'line') and a['first_lap_end_s'] == pytest.approx(b['first_lap_end_s'] + add)
    if kw['dur'] == 13.0:                     # A2 が無い = 線の前に人側にしたかは分からない(名前だけ出す)
        assert g['on_before_line'] == [] and len(g['on_no_a2']) == 4 and 'A2 が無く線が分からない' in on
    else:
        assert 'on_before_line' not in g and 'on_no_a2' not in g
    assert {k: {kk: vv for kk, vv in b.items() if kk != 'v_lh_self_median_mps'} for k, b in g['by_launch'].items()} == go['by_launch']


@pytest.mark.parametrize('which', ['two_laps', 'one_off_two_laps'])
def test_rec22_with_flying_laps_output_unchanged(tmp_path, which):
    """周がある日(8 本とも 2 周・3-1 だけ 2 周で残りは同じ設定の周): JSON・標準出力が直す前の写しとまるごと同じ"""
    D = Day(tmp_path)
    round3(D, lambda i: dict(dur=25.0, human_after=21.5 if (which == 'two_laps' or i == 0) else 12.0))
    same, jn = same_output(D, which)
    assert same and jn['summary'][0]['compare']['all_ok'] is True and 'lh_src' not in jn['summary'][0]['compare']


def test_rec22_summarize_uses_one_definition_for_on_and_off(col):
    """周の LH がある本と無い本が混ざったら、ON・OFF とも発進の場所の LH で比べる(決め方を混ぜない)。ON が LH に 0.6 m/s 速く
    入れば満たさない。全部に周の LH があれば前と同じ(周の値)"""
    def R(log, st, v_lap, v_self):
        r = _row(log, st, 2.7 if st == 'on' else 3.0)
        r['v_launch_at_lh'] = v_lap
        r['_lh_self'] = {'x_m': 3.0, 'v_mps': v_self, 't_s': 1.5}
        return r
    rows = [R('20261103-130000', 'off', 2.0, 2.0), R('20261103-131000', 'off', None, 2.0),
            R('20261103-130500', 'on', 2.1, 2.6), R('20261103-131500', 'on', None, 2.6)]
    g = col.summarize(rows)[0]
    assert g['compare']['lh_src'] == 'launch_loc' and g['compare']['d_v_lh_mps'] == pytest.approx(0.6)
    assert g['compare']['checks']['lh_speed'] is False and g['compare']['all_ok'] is False
    rows2 = [R('20261103-130000', 'off', 2.0, 9.0), R('20261103-131000', 'off', 2.0, 9.0),
             R('20261103-130500', 'on', 2.1, 0.0), R('20261103-131500', 'on', 2.1, 0.0)]
    g2 = col.summarize(rows2)[0]
    assert 'lh_src' not in g2['compare'] and g2['compare']['d_v_lh_mps'] == pytest.approx(0.1) and g2['compare']['all_ok'] is True
    assert 'v_lh_self_median_mps' not in g2['by_launch']['on']


def test_rec22_no_lh_anywhere_prints_the_reason(col):
    rows = [_row('20261103-130000', 'off', 3.0), _row('20261103-130500', 'on', 2.7)]
    for r in rows:
        r['v_launch_at_lh'] = None
        r['_lh_self'] = None
    g = col.summarize(rows)[0]
    assert g['compare']['d_v_lh_mps'] is None and g['compare']['checks']['lh_speed'] is False and g['compare']['all_ok'] is False
    txt = '\n'.join(col.summary_lines([g]))
    assert 'それでも出せない(発進のフレームに LH が無い)= 満たさない。回 3 のどれか 1 本を 2 周(A2 まで AI 側)' in txt


@pytest.mark.parametrize('loc,k', [([6, 6, 5, 5, 4, 6, 6, 6, 1], 5), ([6, 6, 6, 1], 0), ([4, 5, 6, 1], 2), ([5, 6, 5, 6, 6], 3),
                                   ([5, 5, 4, 1], None), ([], None)])
def test_rec22_lh_entry_is_the_last_lh_run_before_r1(col, loc, k):
    """起点より前のフレームの場所は向きから決めるので、置いたところが一瞬 LH になることがある(本物の 20261024-160847 は 0〜0.16 m
    が LH → 最初の LH を使うと入口の速さ 0.07 m/s)。LH の入口は R1 の前の最後の LH の続きの始め"""
    assert col.last_lh_start(np.array(loc, np.int8)) == k


# ================================================================== r2-rec2-3 'ai_end' の衝突
BASE6 = [(12, PRM, {}), (13, ON, {}), (14, PRM, {}), None, (16, PRM, {}), (17, ON, {})]


def p2_day(tmp, flip, hit_t=8.0, g=6.0):
    """反証 P2: 3-4(ON)が R1 を越えた hit_t 秒に g で当たり、0.05 秒後に止まる。人は flip 秒に人側"""
    D = Day(tmp)
    log = None
    for x in BASE6:
        if x is None:
            log, _ = D.add(15, ON, dur=12.0, hits=[(hit_t, g)], stop_after=hit_t + 0.05, human_after=flip)
        else:
            D.add(x[0], x[1], **x[2])
    return D, log


@pytest.mark.parametrize('flip', [7.95, 7.99, 7.7])
def test_rec23_on_hit_right_after_flip_counts(tmp_path, flip):
    """人が当たる 0.05・0.01・0.3 秒前に人側。直す前: 最初の周の衝突 0 → 使ってよい。今: 数えて満たさない"""
    D, bad = p2_day(tmp_path, flip)
    (jn, on, _, _), (jo, oo, _, _) = both(D, 'p2')
    assert jo['summary'][0]['compare']['all_ok'] is True and '→ 使ってよい' in oo                     # 指摘の再現
    r = [x for x in jn['launches'] if x['log'] == bad][0]
    assert r['first_lap_end'] == 'ai_end' and r['to_r1_s'] is not None
    assert r['first_lap_end_s'] == pytest.approx(r['t_ai'] + r['ai_seg_s'] + 0.5, abs=0.02)
    assert len(r['collisions_first_lap']) == 1 and r['collisions_first_lap'][0][1] == pytest.approx(6.0, abs=0.01)
    c = jn['summary'][0]['compare']
    assert c['checks']['first_lap_collisions_0'] is False and c['all_ok'] is False and '→ 満たさない' in on
    assert jn['summary'][0]['on_before_line'] == [] and jn['summary'][0]['on_no_a2'] == ['%s@1.30' % bad]   # 当たって止まった
    assert ('最初の周を AI の区間の終わり + 0.5 秒まで数えた ON 1 本(人側にした直後の衝突も数える): 線の前に人側 なし / '
            'A2 が無く線が分からない %s@1.30' % bad) in on


def test_rec23_after_hit_and_before_r1_still_fail(tmp_path):
    """人側が当たったあと(8.2 秒)・R1 の前の当たり(3.0 秒・人側 2.95 秒)は前から満たさない(今も)"""
    D, _ = p2_day(sub(tmp_path, 'a'), 8.2)
    (jn, _, _, _), (jo, _, _, _) = both(D, 'a')
    assert jn['summary'][0]['compare']['all_ok'] is False and jo['summary'][0]['compare']['all_ok'] is False
    D2, bad = p2_day(sub(tmp_path, 'b'), 2.95, hit_t=3.0)
    (jn2, _, _, _), (jo2, _, _, _) = both(D2, 'b')
    assert jn2['summary'][0]['compare']['all_ok'] is False and jo2['summary'][0]['compare']['all_ok'] is False
    assert jn2['summary'][0]['not_reached']['on']['collisions'] == 1


def test_rec23_no_r1_tail_is_now_half_a_second(tmp_path):
    """R1 の前で、人側にしてから 0.3 秒後に当たった(本物の 10/23 の先回りの形)。直す前は 0.2 秒までで数えない(届かなかったので
    比べは前から満たさない)。今は衝突 1 として名前と G が出る"""
    D, bad = p2_day(tmp_path, 2.7, hit_t=3.0, g=8.6)
    (jn, on, _, _), (jo, oo, _, _) = both(D, 'c')
    assert jo['summary'][0]['not_reached']['on']['collisions'] == 0                                    # 直す前: 数えない
    nr = jn['summary'][0]['not_reached']['on']
    assert nr['collisions'] == 1 and nr['hits'][0][2] == pytest.approx(8.6, abs=0.01) and '8.6G' in on
    r = [x for x in jn['launches'] if x['log'] == bad][0]
    assert r['first_lap_end'] == 'no_r1' and r['first_lap_end_s'] == pytest.approx(r['t_ai'] + r['ai_seg_s'] + 0.5, abs=0.02)


def test_rec23_limit_hit_more_than_half_a_second_after_flip(tmp_path):
    """人側にしてから 0.6 秒より後の当たりは数えない(限り。手順書 7 章の人の判断。car_check にも同じ 0.5 秒)"""
    D, bad = p2_day(tmp_path, 7.4)
    (jn, _, _, _), _ = both(D, 'd')
    r = [x for x in jn['launches'] if x['log'] == bad][0]
    assert r['collisions_first_lap'] == [] and jn['summary'][0]['compare']['all_ok'] is True


def test_rec23_first_lap_still_ends_at_the_line(tmp_path):
    """線が AI の区間の終わりの 0.2〜0.5 秒あと: 最初の周は線まで(線のあとの当たりは 2 周目)。線の前の当たりは数える"""
    D = Day(tmp_path)
    D.add(12, PRM)
    a, _ = D.add(13, ON, human_after=10.3, hits=[(10.70, 3.0)])       # 線 10.652 秒のあと
    b, _ = D.add(15, ON, human_after=10.3, hits=[(10.55, 3.0)])       # AI の終わり 10.267 秒のあと・線の前
    j, _, _, _ = D.run(name='e')
    R = {r['log']: r for r in j['launches']}
    for lg, n in ((a, 0), (b, 1)):
        assert R[lg]['first_lap_end'] == 'ai_end' and R[lg]['first_lap_end_s'] == pytest.approx(10.652, abs=0.02)
        assert len(R[lg]['collisions_first_lap']) == n and R[lg]['first_lap_a2_s'] == pytest.approx(13.0, abs=0.06)
    g = j['summary'][0]
    assert sorted(g['on_before_line']) == sorted(['%s@1.30' % a, '%s@1.30' % b]) and g['on_no_a2'] == []


def test_rec23_existing_line_rows_unchanged(tmp_path):
    """線まで AI 側の本('line')だけの日は、直す前と JSON・標準出力がまるごと同じ(_same_day: OFF・ON・出ず・dry run。2026-10-27
    直す 3 回目・1 組目 r3-r3-rec-1 から 0 バイトの log.json の本は作らない(⑤ が「読めない記録」に出すようになった。tests/test_fix3_r3_1_tools.py))"""
    for args in ((), ('--first-only',), ('--ignore-gain',)):
        D = _same_day(sub(tmp_path, 's%d' % len(args) + '_'.join(a.strip('-') for a in args)), zero=False)
        same, jn = same_output(D, 's', args)
        assert same and 'compare' in jn['summary'][0]


# ================================================================== r2-rec2-5 読めない記録
def kill_npz(d, scenes_from=0):
    """作り物の記録のシーンの npz を消す(log.json と dir は残す。電源断で書かれなかった形)"""
    for i, s in enumerate(sorted(p for p in Path(d).iterdir() if p.is_dir())):
        if i >= scenes_from:
            for f in s.glob('*.npz'):
                f.unlink()


def r25_day(tmp, state='on', plan=None, lb_in_log=True):
    """OFF 4・ON 4 の回 3。3-4(block 15)は 3.0 秒に 9G で当たり 3.1 秒に止まり、npz が無い(log.json だけ)"""
    D = Day(tmp)
    bad = None
    for i in range(8):
        block = 12 + i
        p = ON if i % 2 else PRM
        if block == 15:
            p = ON if (state == 'on' and lb_in_log) else PRM
            bad, d = D.add(block, p, hits=[(3.0, 9.0)], stop_after=3.1, human_after=3.6)
            kill_npz(d)
        else:
            D.add(block, p)
    if plan is not None:                    # bench_candidates.json の計画(expect.launch_boost)
        b = json.loads(D.bench.read_text())
        for bl in b['days'][0]['blocks']:
            bl['expect'] = {'launch_boost': plan if bl['block'] == 15 else (1 if bl['block'] % 2 else 0)}
        D.bench.write_text(json.dumps(b), encoding='utf-8')
    return D, bad


def test_rec25_unreadable_on_fails(tmp_path):
    D, bad = r25_day(tmp_path)
    (jn, on, tn, _), (jo, oo, _, _) = both(D, 'u')
    assert jo['summary'][0]['compare']['all_ok'] is True and '→ 使ってよい' in oo and bad not in json.dumps(jo['summary'])  # 再現
    assert 'skip %s' % bad in on                                                        # 前からの 1 行は残す
    g = jn['summary'][0]
    assert g['unreadable']['logs'] == [bad] and g['unreadable']['on'] == 1 and g['unreadable']['launch_state'] == ['on']
    assert g['compare']['checks']['on_all_readable'] is False and g['compare']['all_ok'] is False
    assert {k: v for k, v in g['compare']['checks'].items() if k != 'on_all_readable'} == jo['summary'][0]['compare']['checks']
    assert '読めない記録 1 本(比べの数に入れない): %s(' % bad in on and 'log.json ON' in on
    assert '→ 読めない(欠けた)ON の本がある = 満たさない' in on and '→ 満たさない' in on
    assert jn['unreadable'][0]['log'] == bad and jn['unreadable'][0]['launch_state'] == 'on'
    assert jn['launches'] == jo['launches']                                             # 発進ごとの行は同じ


def test_rec25_plan_on_counts_even_if_log_json_says_off(tmp_path):
    """log.json の launch_boost は 0 でも、計画(説明欄の block → expect.launch_boost)が 1 なら ON として数える"""
    D, bad = r25_day(tmp_path, state='on', plan=1, lb_in_log=False)
    (jn, on, _, _), _ = both(D, 'pl')
    g = jn['summary'][0]
    assert g['unreadable']['launch_param'] == [0] and g['unreadable']['plan_launch'] == [1] and g['unreadable']['on'] == 1
    assert g['compare']['all_ok'] is False and 'log.json OFF・計画 ON' in on


def test_rec25_unreadable_off_is_listed_only_and_skip_reads_again(tmp_path):
    D, bad = r25_day(tmp_path, state='off')
    (jn, on, _, _), _ = both(D, 'off')
    g = jn['summary'][0]
    assert g['unreadable']['on'] == 0 and g['compare']['all_ok'] is True and 'on_all_readable' not in g['compare']['checks']
    assert '読めない記録 1 本' in on and '→ 読めない(欠けた)ON' not in on
    D2, bad2 = r25_day(sub(tmp_path, 'sk'))
    j2, out2, _, _ = D2.run(name='sk', args=['--skip', bad2])
    assert 'unreadable' not in j2['summary'][0] and j2['summary'][0]['compare']['all_ok'] is True
    assert j2['summary_skipped'] == ['%s(読めない記録)' % bad2] and '%s(読めない記録)' % bad2 in out2
    assert j2['unreadable'][0]['log'] == bad2                                           # JSON の一覧には残す


def test_rec25_all_on_unreadable_still_prints_not_satisfied(tmp_path):
    D = Day(tmp_path)
    D.add(12, PRM)
    D.add(14, PRM)
    bad, d = D.add(13, ON)
    kill_npz(d)
    j, out, _, _ = D.run(name='all')
    g = j['summary'][0]
    assert 'on' not in g['by_launch'] and g['compare']['all_ok'] is False
    assert g['compare']['checks'] == {'to_r1': False, 'first_lap_collisions_0': True, 'slip': False, 'lh_speed': False,
                                      'on_all_readable': False}
    assert 'ON − OFF: 起点まで — 秒' in out


def test_rec25_unreadable_joins_the_gain_split_groups(tmp_path):
    """ゲインが違って組が分かれた日(--ignore-gain なし): 読めない ON は、ゲインのほかが同じ組の全部に入る(どの組も満たさない)。
    --ignore-gain では 1 組"""
    D = Day(tmp_path)
    for i in range(4):
        D.add(12 + i, dict(ON if i % 2 else PRM, throttle_gain=0.90 if i < 2 else 0.92))
    from test_fix3_r2_1_collect import set_gain
    for lg in sorted(p for p in D.raw.iterdir()):
        m = json.loads((lg / 'log.json').read_text())
        set_gain(lg, m['drive']['params']['throttle_gain'])
    bad, d = D.add(16, dict(ON, throttle_gain=0.95))
    kill_npz(d)
    j, _, _, _ = D.run(name='gs')
    assert [g['gain'] for g in j['summary']] == [0.9, 0.92] and all(g['unreadable']['logs'] == [bad] for g in j['summary'])
    assert all(g['compare']['all_ok'] is False for g in j['summary'])
    ji, oi, _, _ = D.run(name='gsi', args=['--ignore-gain'])
    assert len(ji['summary']) == 1 and ji['summary'][0]['unreadable']['on'] == 1 and ji['summary'][0]['compare']['all_ok'] is False


def test_rec25_partial_record_cut_in_the_first_lap_fails(tmp_path):
    """シーンが 8 秒ずつで、8 秒より後のシーンの npz が無い(最初の周の途中で電源断)。3-4(ON)の 9.0 秒の 9G は記録に無い。
    直す前: 使ってよい。今: 「最初の周の記録が欠けた発進(on)」で満たさない"""
    D = Day(tmp_path)
    bad = None
    for i in range(8):
        if 12 + i == 15:
            bad, d = D.add(15, ON, scene_s=8.0, hits=[(9.0, 9.0)])
            kill_npz(d, scenes_from=1)
        else:
            D.add(12 + i, ON if i % 2 else PRM)
    (jn, on, _, _), (jo, oo, _, _) = both(D, 'pt')
    assert jo['summary'][0]['compare']['all_ok'] is True                                              # 指摘の形の再現
    r = [x for x in jn['launches'] if x['log'] == bad][0]
    assert r['first_lap_end'] == 'ai_end' and '記録が 7.99 秒で終わった' in r['first_lap_gap']
    g = jn['summary'][0]
    assert g['first_lap_gap_on'] and g['compare']['checks']['on_all_readable'] is False and g['compare']['all_ok'] is False
    assert '最初の周の記録が欠けた発進(on)1 本' in on and jn['problems'][bad]


@pytest.mark.parametrize('i_zero', [1, 2])
def test_rec25_zero_byte_scene_after_the_data_is_unchanged(tmp_path, i_zero):
    """記録の終わりのあとに 0 バイトの npz だけのシーン(fix2 の形)。最初の周は記録の中 → 前と同じ出力
    (2026-10-27 fix5 c1: 標準出力に「記録が欠けている本 …」の 1 行が足される。その行を除けば JSON・標準出力とも前と同じ)。
    2026-10-27 fix6-2(反証 6-2 回目 w2-tools-r62-1): 発進の前進 ON の本(i_zero = 1)は、最後の読めるフレームのあとが欠けた本として
    「読めない記録」(ON。満たさない)にも入る(電源断の形 = その間の 2 回目の発進を記録で確かめられない。最初の周の記録が欠けた
    発進にはならない)。その分(tests/test_fix6_tools_r2.strip_tail)を除けば前と同じ。OFF の本(i_zero = 2)は前のまま"""
    D = Day(tmp_path)
    for i in range(4):
        D.add(12 + i, ON if i % 2 else PRM, zero_scene=(i == i_zero))
    (jn, on, tn, _), (jo, oo, to, _) = both(D, 'z')
    extra = [x for x in on.splitlines() if x.startswith('記録が欠けている本 ')]
    assert len(extra) == 1 and jn['problems'] and 'first_lap_gap_on' not in jn['summary'][0]
    assert jo['summary'][0]['compare']['all_ok'] is True
    if i_zero % 2:                   # fix6-2: ON の本は読めない記録にも入る(その分を除けば前と同じ)
        from test_fix6_tools_r2 import strip_tail
        [bad] = list(jn['problems'])
        assert [u['log'] for u in jn['unreadable']] == [bad] and jn['summary'][0]['compare']['all_ok'] is False
        jn, on = strip_tail(jn, on, bad)
        rest = '\n'.join(x for x in on.splitlines() if not x.startswith('記録が欠けている本 '))
        assert jn == jo and no_path(rest) == no_path(oo)
    else:
        rest = '\n'.join(x for x in on.splitlines() if not x.startswith('記録が欠けている本 '))
        assert tn == to and no_path(rest) == no_path(oo) and jn['summary'][0]['compare']['all_ok'] is True


def test_collect_r2_1_scenarios_unchanged(tmp_path):
    """直す 2 回目・1 組目の試験の日(ゲインが本ごとに違う・launch_us の混ざり)も、JSON・標準出力が直す前と同じ"""
    D = Day(sub(tmp_path, 'a'))
    for i in range(8):
        D.add(12 + i, dict(ON if i % 2 else PRM, throttle_gain=0.90 if i < 4 else 0.92))
    for args in ((), ('--ignore-gain',)):
        assert same_output(D, 'g%d' % len(args), args)[0]
    D2 = Day(sub(tmp_path, 'b'))
    for i in range(8):
        D2.add(12 + i, dict(ON, launch_us=250.0 if i < 5 else 200.0) if i % 2 else PRM)
    same, jn = same_output(D2, 'us')
    assert same and jn['summary'][0].get('mixed_launch_us') == [200.0, 250.0]


# ================================================================== r2-rec2-4 car_check の人側にした直後の衝突
def flip_log(raw, flip, hit=(8.0, 9.0), log='20261103-140000'):
    write_standing_log(raw, log, dur=12.0, hits=[hit], stop_after=hit[0] + 0.05, human_after=flip, run='runB',
                       description='ai_no=15 day=2026-11-03 block=15 memo=x', params={'throttle_gain': 0.9, 'launch_boost': 1})
    return log


def cc_pair(raw, log, old_cc):
    a = cc.check_log(log, raw / 'nolab', raw)
    b = old_cc.check_log(log, raw / 'nolab', raw)
    return a, b


def strip_time(r):
    # 2026-10-27 直す 3 回目・2 組目(r3-r3-rec-2): 発進の前進 ON の本の car_check に足した launch_slip(発進の空転の比)も除く
    return {k: v for k, v in r.items() if k not in ('created_at', 'seconds', 'launch_slip')}


def no_slip_line(x):
    """r3-r3-rec-2 で足した「発進の空転の比」の行(と → の行)なら True"""
    return '発進の空転の比(' in x or x.startswith(('  → 止まっているうちに', '  → もう ', '  → 比が出ない発進は'))


def text_no_time(t):
    return '\n'.join(x for x in t.splitlines() if not (x.startswith('  (') and x.endswith(' 秒)')) and not no_slip_line(x))


@pytest.mark.parametrize('flip,dt', [(7.95, 0.07), (7.7, 0.336)])
def test_rec24_hit_right_after_flip_shows_in_collision_line(tmp_path, old_cc, flip, dt):
    """反証の probe_flip: 8.0 秒に 9G(DE)。人側 7.95 秒(0.05 秒前)・7.7 秒(0.3 秒前。本物の先回りの形)。
    直す前:「衝突: 0 回」。今: 行の頭が「!」・8G 以上 1・場所 DE・「人側にした直後」の印。判定の番号は同じ"""
    log = flip_log(tmp_path, flip)
    a, b = cc_pair(tmp_path, log, old_cc)
    assert b['collisions']['n'] == 0 and '  衝突(AI・立て直し中とリカバリー中も): 0 回' in old_cc.to_text(b)   # 指摘の再現
    cs = a['collisions']
    assert cs['n'] == 1 and cs['n_ge8'] == 1 and cs['max_g'] == pytest.approx(9.0, abs=0.01) and cs['by_place']['DE'] == 1
    assert cs['by_class'] == {'after_ai': 1} and len(cs['after_ai']) == 1 and cs['after_ai'][0][3] == pytest.approx(dt, abs=0.005)
    t = cc.to_text(a)
    line = [x for x in t.splitlines() if '衝突(AI' in x][0]
    assert line.startswith('! 衝突(AI・立て直し中とリカバリー中・人側にした直後も): 1 回・最大 9.00G・8G 以上 1・場所 DE 1')
    assert '(強い: 8.0 秒 DE 9.00G・人側にした直後)' in line and 'after_ai' not in line
    assert 'うち人側にした直後(AI の区間の終わりから 0.5 秒以内)1: 8.0 秒 DE 9.00G(AI の終わりから %.2f 秒)' % dt in line
    assert a['judge'] == b['judge']                                                      # 判定の番号は変えない
    assert {k: v for k, v in strip_time(a).items() if k != 'collisions'} == {k: v for k, v in strip_time(b).items() if k != 'collisions'}
    assert [x for x in t.splitlines() if '衝突(AI' not in x and not x.startswith('  (') and not no_slip_line(x)] == \
        [x for x in old_cc.to_text(b).splitlines() if '衝突(AI' not in x and not x.startswith('  (')]


@pytest.mark.parametrize('flip', [8.2, 7.4])
def test_rec24_unchanged_when_flip_after_hit_or_long_before(tmp_path, old_cc, flip):
    """当たったあとに人側(AI 側の衝突。前から出る)・0.6 秒より前に人側(数えない限り): JSON・文が直す前と同じ"""
    log = flip_log(tmp_path, flip)
    a, b = cc_pair(tmp_path, log, old_cc)
    assert strip_time(a) == strip_time(b) and text_no_time(cc.to_text(a)) == text_no_time(old_cc.to_text(b))
    assert a['collisions']['n'] == (1 if flip > 8.0 else 0) and 'after_ai' not in a['collisions']


@pytest.mark.parametrize('human,expect', [((), 0), ((16.0, 20.0), 0), ((17.3, 20.0), 1)])
def test_rec24_lap_log_with_human_segments(tmp_path, old_cc, human, expect):
    """周回の記録(偽の起点の衝突 17.52 秒・6G)。人側の区間が衝突の 1.5 秒前から → 数えない(前と同じ)。0.25 秒前から → 足す"""
    log = '20261103-150000'
    write_lap_log(tmp_path, log, dur=30.0, human=[human] if human else [], description='ai_no=20 memo=x')
    a, b = cc_pair(tmp_path, log, old_cc)
    if not expect:
        assert strip_time(a) == strip_time(b) and text_no_time(cc.to_text(a)) == text_no_time(old_cc.to_text(b))
    else:
        assert a['collisions']['n'] == b['collisions']['n'] + 1 and len(a['collisions']['after_ai']) == 1
        assert a['collisions']['after_ai'][0][3] == pytest.approx(0.253, abs=0.02) and a['judge'] == b['judge']


def test_rec24_after_ai_collisions_unit():
    lab = {'t_s': np.arange(0, 3, 1 / 30.0), 'ai': np.r_[np.ones(60, bool), np.zeros(30, bool)], 'dry_run': np.zeros(90, bool)}
    ev = {'collisions': [{'t_s': 1.5, 'ai': True, 'g': 5.0, 'loc': 'DE'},       # AI 側(前から score_log が数える)
                         {'t_s': 2.2, 'ai': False, 'g': 9.0, 'loc': 'DE'},      # AI の終わり 1.967 秒の 0.233 秒後
                         {'t_s': 2.6, 'ai': False, 'g': 9.0, 'loc': 'HS'}]}     # 0.633 秒後
    out = cc.after_ai_collisions(ev, lab)
    assert [(c['t_s'], c['class'], c['after_ai_s']) for c in out] == [(2.2, 'after_ai', 0.233)]
    lab2 = dict(lab, ai=np.zeros(90, bool))
    assert cc.after_ai_collisions(ev, lab2) == []                                       # AI の区間が無い
    lab3 = dict(lab, dry_run=np.ones(90, bool))
    assert cc.after_ai_collisions(ev, lab3) == []                                       # dry run は AI に数えない


# ================================================================== 走行の動き・既定値に触れない
def test_tools_only_no_driving_code_or_defaults_touched(tmp_path, old_cc):
    """この組が直したのは夜の道具(読むだけ)の 2 つだけ。走行のコード(drive.py・assist.py)を読み込まない。道具を流しても
    既定値のファイル(data/bench_candidates.json・data/drive_settings.json)は変わらない。drive.DEFAULT_PARAMS などが直す前と
    同じことは tests/test_fix3_r2_1.py::test_before_copies_and_defaults_unchanged が見る(この組は drive.py を変えていない)"""
    for src in (COLLECT, JF / 'tools' / 'car_check.py'):
        txt = src.read_text(encoding='utf-8')
        assert 'import drive' not in txt and 'from jetformer import drive' not in txt and 'jetformer.drive' not in txt
        assert 'jetformer.assist import LAUNCH_END_NAMES, ASSIST_LAUNCH' in txt or src.name == 'car_check.py'   # 名前の表だけ
    files = [JF / 'data' / 'bench_candidates.json', JF / 'data' / 'drive_settings.json']
    before = [hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None for f in files]
    D = Day(tmp_path)
    for i in range(4):
        D.add(12 + i, ON if i % 2 else PRM)
    D.run(name='def')
    log = flip_log(sub(tmp_path, 'cc'), 7.95)
    cc.check_log(log, tmp_path / 'cc' / 'nolab', tmp_path / 'cc')
    assert [hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None for f in files] == before
