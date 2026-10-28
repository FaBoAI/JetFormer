"""直す 1 回目・2 組目(2026-10-26 夜 fix3)の試験。作り物の記録・作り物の行だけ(tmp)。本物の data/raw・labels・装置・GPU は使わない。

- r1-rec3-1  collect.py: R1 に届かなかった ON の発進(当たって人側・15 秒以上)を比べから黙って外し「使ってよい」と出していた
- r1-rec3-2  car_check: R1 の通過が 1〜2 の本で、板 7G 以上を「足りない(もう 1 本)」と出していた
- r1-rec3-3  collect.py: 走行中に変えた本(events.jsonl の params・params_end・天井の途中 OFF・発進の終わり 10)を比べに入れていた
- r1-rec3-4  by_params 2 章: 「25 秒以下」「最速」に衝突のある発進が入り、「ぶつからずに 25 秒以下」の数が無かった
- r1-rec3-5  rec_hold_keep・launch_no_kick が夜の道具(鍵・設定の文・計画との違い・car_check)のどれにも出なかった

どの直しも、その場面が無ければ出力は直す前の写し(tests/fixtures/fix3_r1_2_before/*.orig。sha256 を照らす)と同じことも見る。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_fix3_r1_2.py
"""
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

JF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JF / 'tools'))
sys.path.insert(0, str(JF / 'tests'))
import car_check as cc  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_lap_anchor import T0_US, write_lap_log  # noqa: E402
from test_standing_3lap import write_standing_log  # noqa: E402

BEFORE = JF / 'tests' / 'fixtures' / 'fix3_r1_2_before'
COLLECT = JF / 'experiments' / '20261026_fix' / 'diag_launch' / 'collect.py'
RUNS = JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'


def load_mod(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))          # .orig(拡張子が .py でない)も読む
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def old():
    """直す前の写し(sha256 を照らしてから読む)"""
    sums = dict(reversed(x.split()) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    for name, h in sums.items():
        assert hashlib.sha256((BEFORE / name).read_bytes()).hexdigest() == h, name
    return {'cc': load_mod('car_check_r12_before', BEFORE / 'car_check.py.orig'),
            's3': load_mod('standing_3lap_r12_before', BEFORE / 'standing_3lap.py.orig'),
            'qp': load_mod('qp_r12_before', BEFORE / 'quick_eval_by_params.py.orig')}


# ================================================================== r1-rec3-2 car_check の板
def _m(passes, a3_med, a3_max, **kw):
    m = {'turn': {'ratio': 0.95, 'n_left': 100, 'n_right': 100}, 's0': {'s0_us': 0.0, 'n': 500},
         'r1_bump': {'passes': passes, 'a3_median': a3_med, 'a3_max': a3_max}, 'vbat': {'start_v': 8.3, 'source': '基板'},
         'laps': {'p90_median_mps': 4.0, 'sw_max_median_mps': None}, 'speed_ctrl': {}}
    m.update(kw)
    return m


def _st(j, item):
    return next(x for x in j['items'] if x['item'] == item)


@pytest.mark.parametrize('passes,med,mx,status,code', [
    (1, 8.0, 8.0, 'fail', 2), (2, 7.0, 7.0, 'fail', 2), (2, 5.0, 8.0, 'fail', 2), (1, 7.01, 7.01, 'fail', 2),
    (2, 6.99, 6.99, 'insufficient', 3), (1, 6.9, 6.9, 'insufficient', 3),
    (2, 6.0, 6.5, 'insufficient', 3),          # 中央 ≥ 4.5G は通過 3 以上のときだけ(前と同じ)
    (0, None, None, 'insufficient', 3)])
def test_rec3_2_board_7g_with_few_passes_is_fail(passes, med, mx, status, code):
    j = cc.judge(_m(passes, med, mx))
    x = _st(j, 'r1_bump')
    assert x['status'] == status and j['code'] == code
    if status == 'fail':
        assert '1 回でも ≥ 7.0G' in x['text'] and 'ゲイン −0.05' in x['text'] and '通過 %d' % passes in x['text']
        assert j['headline'] == '本番の設定に進まない(下の行)'
    else:
        assert x['text'] == 'R1 の板の衝撃: 判定に足りません(AI の R1 の通過 %d、3 以上要る)' % passes


def test_rec3_2_judge_same_as_before_everywhere_else(old):
    """通過 1〜2 で最大 ≥ 7.0G のときだけ変わる。ほかは判定の返り値がまるごと直す前と同じ"""
    n_changed = n_same = 0
    for passes in range(0, 7):
        for med in (3.0, 4.49, 4.5, 6.0, 7.5):
            for mx in (5.0, 6.99, 7.0, 7.5, 9.0):
                if mx < med:
                    continue
                for extra in ({}, {'turn': {'ratio': 0.80, 'n_left': 100, 'n_right': 100}}, {'s0': {'s0_us': 0.0, 'n': 10}}):
                    m = _m(passes, med if passes else None, mx if passes else None, **extra)
                    a, b = old['cc'].judge(m), cc.judge(m)
                    if 1 <= passes < 3 and mx >= 7.0:
                        n_changed += 1
                        assert _st(a, 'r1_bump')['status'] == 'insufficient' and _st(b, 'r1_bump')['status'] == 'fail'
                        assert b['code'] == 2
                        assert [x for x in a['items'] if x['item'] != 'r1_bump'] == [x for x in b['items'] if x['item'] != 'r1_bump']
                    else:
                        n_same += 1
                        assert a == b
    assert n_changed > 0 and n_same > 100


@pytest.mark.parametrize('g,code,mark', [(8.0, 2, '× R1 の板の衝撃 最大 8.00G'), (6.9, 3, '? R1 の板の衝撃: 判定に足りません')])
def test_rec3_2_end_to_end_two_passes(tmp_path, g, code, mark):
    """回 3 の「止まって 1〜2 周」の形: 15 秒・R1 の通過 2(反証の demo_r4 と同じ作り)"""
    log = '20261103-140000'
    write_lap_log(tmp_path, log, dur=15.0, false_laps=(), r1_bump_g=g, description='ai_no=17 day=2026-11-03 block=17 memo=x')
    r = cc.check_log(log, tmp_path / 'nolab', tmp_path)
    assert r['r1_bump']['passes'] == 2 and r['r1_bump']['a3_max'] == pytest.approx(g, abs=0.05)
    assert r['judge']['code'] == code and mark in cc.to_text(r)
    assert load_mod('car_check_r12_b2', BEFORE / 'car_check.py.orig').judge(r)['code'] == 3      # 直す前の写し(再現): 「足りない」
    rc = cc.main([log, '--raw', str(tmp_path), '--labels', str(tmp_path / 'nolab'), '--no-json'])
    assert rc == code


def test_rec3_2_three_passes_unchanged(tmp_path, old):
    """通過 3 の 8G は前から不合格(文も同じ)"""
    log = '20261103-140500'
    write_lap_log(tmp_path, log, dur=26.0, false_laps=(), r1_bump_g=8.0, description='ai_no=18 memo=x')
    r = cc.check_log(log, tmp_path / 'nolab', tmp_path)
    assert r['r1_bump']['passes'] == 3 and r['judge'] == old['cc'].judge(r) and r['judge']['code'] == 2


# ================================================================== r1-rec3-5 試験の選択肢
OPT_ON = {'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 1}
OPT_0 = {'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}


def test_rec3_5_car_check_options_line(tmp_path, old):
    log = '20261103-141000'
    write_lap_log(tmp_path, log, r1_bump_g=3.0, params=dict({'throttle_gain': 0.9}, **OPT_ON))
    r = cc.check_log(log, tmp_path / 'nolab', tmp_path)
    t_new, t_old = cc.to_text(r), old['cc'].to_text(r)
    line = ('  試験の選択肢(log.json の走行開始の値): リカバリーの後ろの見張り・中立止めのあともリカバリーを残す・'
            '発進の前進を出さない区間は発進の補助も出さない が ON')
    assert line in t_new.splitlines() and line not in t_old
    assert [x for x in t_new.splitlines() if x != line] == t_old.splitlines()
    assert r['judge'] == old['cc'].judge(r)                    # 判定には入れない
    for prm in (dict({'throttle_gain': 0.9}, **OPT_0), {'throttle_gain': 0.9}, {'throttle_gain': 0.9, 'rec_hold_keep': 'x'}):
        r2 = dict(r, params=prm)
        assert cc.to_text(r2) == old['cc'].to_text(r2)          # どれも OFF・無い・読めない値なら前と同じ
    assert cc.options_on({'rec_back_guard': 1}) == ['リカバリーの後ろの見張り']


def _settings(params):
    return s3.settings_of({'drive': {'run': 'runB', 'params': params}, 'mode_id': 5}, 0.9, {'runB': 'B'})


def _res(log, params, standing=()):
    S = _settings(params)
    return {'log': log, 'ai_no': log[-2:], 'cand': 'EXP', 'settings': S, 'standing': list(standing), 'windows': [],
            'laps': {'passes': 3, 'laps': 2, 'clean': 0, 'clean_laps_s': [], 'best_clean_s': None, 'median_clean_s': None,
                     'r1_entry_median_erpm': None, 'entries_erpm': []},
            'collisions': {'all': 0, 'strong': 0, 'max_g': None, 'n_ge8': 0, 'by_place': {}},
            'flying': {'n': 0, 'best_s': None, 'best_clean_s': None}, 'ai_s': 30.0, 'midrun': [], 'speed_ctrl_flags': [],
            'zero_byte': []}


P = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 1, 'ceil_left_mps': 4.1, 'ceil_all_mps': 4.5, 'speed_auto': 0,
     'launch_boost': 0}


@pytest.mark.parametrize('extra', [{}, OPT_0, {'rec_back_guard': 1}, {'rec_back_guard': 1, 'rec_hold_keep': 0, 'launch_no_kick': 0},
                                   {'steer_trim_us': 15.0, 'rec_back_guard': 1}])
def test_rec3_5_off_is_same_as_before(old, extra):
    """rec_hold_keep・launch_no_kick が 0 か無い: 設定の文・鍵は直す前と同じ(無いときは設定の dict も同じ)"""
    prm = dict(P, **extra)
    meta = {'drive': {'run': 'runB', 'params': prm}, 'mode_id': 5}
    a, b = old['s3'].settings_of(meta, 0.9, {'runB': 'B'}), s3.settings_of(meta, 0.9, {'runB': 'B'})
    assert old['s3'].settings_text(a) == s3.settings_text(b)
    if not set(s3.OPT_KEYS) & set(extra):
        assert a == b
    r = _res('20261103-100000', prm)
    assert old['qp'].key_of(r) == qp.key_of(r)


def old_qp_key(r):
    return load_mod('qp_r12_before_k', BEFORE / 'quick_eval_by_params.py.orig').key_of(r)


@pytest.mark.parametrize('k,txt', [('rec_hold_keep', 'リカバリーを残す'), ('launch_no_kick', '見送りで補助なし')])
def test_rec3_5_new_options_show_in_key_and_text(k, txt):
    """反証の場面(b): 見張り 1 の本 2 つ。片方だけ rec_hold_keep(か launch_no_kick)1 → 鍵・文・2 章の組が分かれる"""
    on = _res('20261103-141000', dict(P, rec_back_guard=1, **{k: 1}))
    off = _res('20261103-141500', dict(P, rec_back_guard=1, **{k: 0}))
    (k1, t1), (k2, t2) = qp.key_of(on), qp.key_of(off)
    assert old_qp_key(on) == old_qp_key(off)              # 直す前の写し(再現): 鍵・文が同じで 1 つの組
    assert k1 != k2 and t1 == 'B・0.90・天井 4.1・発進 OFF・後ろの見張り・%s' % txt and t2 == 'B・0.90・天井 4.1・発進 OFF・後ろの見張り'
    assert s3.settings_text(on['settings']).endswith('後ろの見張り・%s' % txt)
    assert len(qp.group_rows([on, off])) == 2
    assert len(qp.group_rows([on, off], ignore=('trim',))) == 1        # --ignore trim でまとめる
    assert s3.midrun_text([{'key': k, 'from': 1, 'to': 0, 't_s': 3.0}]) == '%s 1→0 3.0 秒' % txt


def test_rec3_5_standing_end_to_end_same_when_absent(tmp_path, old):
    """作り物の記録: 欄が無い本は analyze_log の結果がまるごと前と同じ。欄が 1 の本は設定の文に出る"""
    write_standing_log(tmp_path, '20261103-090000', run='runB', params=dict(P))
    write_standing_log(tmp_path, '20261103-090500', run='runB', params=dict(P, **OPT_ON))
    a = old['s3'].analyze_log('20261103-090000', tmp_path, tmp_path / 'nolab', {'runB': 'B'})
    b = s3.analyze_log('20261103-090000', tmp_path, tmp_path / 'nolab', {'runB': 'B'})
    assert json.dumps(a, default=s3._json_default, sort_keys=True) == json.dumps(b, default=s3._json_default, sort_keys=True)
    c = s3.analyze_log('20261103-090500', tmp_path, tmp_path / 'nolab', {'runB': 'B'})
    assert c['settings_text'] == 'B・0.90・天井 4.1・後ろの見張り・リカバリーを残す・見送りで補助なし'
    assert c['settings']['rec_hold_keep'] == 1 and c['settings']['launch_no_kick'] == 1


# ------------------------------------------------------------------ runs_1103
def _runs_rec(raw, name, ai_no, block, tune, params):
    base = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'lookahead': 3, 'tof_guard': 1, 'zone_caps': 0, 'start_kick': 1,
            'speed_auto': 0, 'speed_ceiling': 1, 'ceil_left_mps': 4.1, 'launch_boost': 0, 'launch_us': 250}
    d = raw / name
    (d / ('%s-0001' % name)).mkdir(parents=True)
    desc = 'ai_no=%d bench=x day=2026-11-03 block=%d cand=EXP tune=%s battery=#A V_before=8.30 memo=' % (ai_no, block, tune)
    (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
        'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'dry_run': False, 'params': dict(base, **params)}},
        ensure_ascii=False), encoding='utf-8')


def _runs_main(mod_path, raw, name):
    m = load_mod(name, mod_path)
    m.RAW = raw
    argv, buf = sys.argv, io.StringIO()
    sys.argv = ['runs_1103.py', '--day', '20261103', '--json']
    try:
        with contextlib.redirect_stdout(buf):
            m.main()
    finally:
        sys.argv = argv
    out = buf.getvalue().strip().splitlines()
    return out[:-1], json.loads(out[-1])


@pytest.fixture()
def round3_blocks():
    bench = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
    blocks = {b['block']: b for d in bench['days'] if d['date'] == '2026-11-03' for b in d['blocks']}
    if not all(blocks.get(k, {}).get('run_id') == '3-%d' % (k - 11) for k in (12, 13, 14, 15)):
        pytest.skip('11/3 の回 3 のブロックが前提と違う')


def test_rec3_5_runs_1103_marks_round_option_change(tmp_path, round3_blocks):
    """朝に見張り・残すを ON → 3-3 の前に再起動して 0 に戻った(RESET_ON_BOOT)形"""
    raw = tmp_path / 'raw'
    _runs_rec(raw, '20261103-130000', 12, 12, '3-1', {'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0})
    _runs_rec(raw, '20261103-130500', 13, 13, '3-2', {'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0, 'launch_boost': 1})
    _runs_rec(raw, '20261103-131000', 14, 14, '3-3', {'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0})
    table, rows = _runs_main(RUNS, raw, 'runs_r12_a')
    by = {r['run_id']: r for r in rows}
    assert not any('試験の選択肢' in x for x in by['3-1']['diffs'] + by['3-2']['diffs'])
    # 2026-10-28 fix8 tools-decided: 印の後ろの文を「わざと OFF にした日」も分かる文に(fix7/fix_docs_r1.md 4 章 3。頭は同じ)
    assert ('回 3 の試験の選択肢が 3-1(No.12)と違う: 後ろの見張り OFF(3-1 は ON)・リカバリーを残す OFF(3-1 は ON)'
            '(再起動で OFF に戻った? 3 章で止めて OFF にした本 X から先なら数える。手順書 3・7 章)') in by['3-3']['diffs']
    assert by['3-1']['options'] == {'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0}
    line = [x for x in table if x.startswith('試験の選択肢')]
    assert line == ['試験の選択肢(log.json の走行開始の値。起動のたびに OFF に戻る。再起動のあとは朝の設定を入れ直す・手順書 7 章): '
                    '3-1(No.12)から 後ろの見張り・リカバリーを残す → 3-3(No.14)から なし(前の本と違う)']


@pytest.mark.parametrize('opt', [{}, OPT_0])
def test_rec3_5_runs_1103_same_when_off(tmp_path, round3_blocks, opt):
    raw = tmp_path / 'raw'
    _runs_rec(raw, '20261103-130000', 12, 12, '3-1', dict(opt))
    _runs_rec(raw, '20261103-130500', 13, 13, '3-2', dict(opt, launch_boost=1))
    _runs_rec(raw, '20261103-131000', 14, 14, '3-3', dict(opt, speed_ceiling=0))
    t_new, j_new = _runs_main(RUNS, raw, 'runs_r12_b')
    t_old, j_old = _runs_main(BEFORE / 'runs_1103.py.orig', raw, 'runs_r12_before')
    assert t_new == t_old
    # 2026-10-27 fix5(反証 4 回目 t3)で KEYS に rev_s・rev_us を足した(params の欄が 2 つ増える。中身は tests/test_ui_fix5.py)
    nrev = lambda p: {k: v for k, v in p.items() if k not in ('rev_s', 'rev_us')}      # noqa: E731
    assert [{k: (nrev(v) if k == 'params' else v) for k, v in r.items() if k != 'options'} for r in j_new] == j_old


# ================================================================== r1-rec3-4 ぶつからずに 25 秒以下
def _start(from_move, ncoll):
    return {'t_ai': 1.0, 'valid': True, 'counted': True, 'coll_before': 0, 'from_move_s': from_move, 'to_r1_s': 3.0,
            'move_after_s': 0.4, 'collisions': {'n': ncoll, 'strong': 0, 'gmax': 3.0 if ncoll else 0.0, 'list': []},
            'launch': {}, 'start_no': 1}


def test_rec3_4_goal_column_counts_only_clean_starts(old):
    """反証の場面(a): 24.60 秒(LH に 3.0G の軽い接触 1)と 25.40 秒(衝突なし)。目標を満たした本は 0"""
    R = [_res('20261103-140000', P, [_start(24.60, 1)]), _res('20261103-140500', P, [_start(25.40, 0)])]
    go = old['qp'].group_rows(R)[0]               # 直す前の写し(再現): 「衝突なし 1・25 秒以下 1・最速 24.60」だけ
    assert (go['standing_clean_n'], go['standing_le25'], go['standing_best_s']) == (1, 1, 24.6) and 'standing_clean_le25' not in go
    g = qp.group_rows(R)[0]
    assert (g['standing_n'], g['standing_best_s'], g['standing_clean_n'], g['standing_le25']) == (2, 24.6, 1, 1)   # 前の欄は前と同じ
    assert g['standing_clean_le25'] == 0 and g['standing_clean_best_s'] == 25.4 and g['standing_best_coll'] == 1
    md = qp.to_markdown(R, '20261103', None)[0]
    row = [x for x in md.splitlines() if x.startswith('| B・0.90')][0]
    assert '| 2・24.60・25.00・1・1・0 | 0・25.40(最速 24.60 は衝突 1) | 2.60 |' in row
    assert 'ぶつからずに 25 秒以下(本・衝突なしの最速)' in md and '- ぶつからずに 25 秒以下 = ' in md
    # 速い方が衝突なしなら、その本を数える(後ろの書き足しなし)
    R = [_res('20261103-140000', P, [_start(24.60, 0)]), _res('20261103-140500', P, [_start(25.40, 1)])]
    g = qp.group_rows(R)[0]
    assert g['standing_clean_le25'] == 1 and g['standing_clean_best_s'] == 24.6 and g['standing_best_coll'] == 0
    assert '| 1・24.60 |' in [x for x in qp.to_markdown(R, '20261103', None)[0].splitlines() if x.startswith('| B・0.90')][0]
    g = qp.group_rows([_res('20261103-140000', P, [])])[0]
    assert g['standing_clean_le25'] == 0 and g['standing_clean_best_s'] is None and g['standing_best_coll'] is None


def _strip_goal(md):
    """新しい列(2 章の最後から 2 番目)と、その説明の行を消す → 直す前の形"""
    out, ch = [], None
    for x in md.splitlines():
        if x.startswith('## '):
            ch = x[:5]
        if ch == '## 2.' and x.startswith('|'):
            c = x.split(' | ')
            x = ' | '.join(c[:-2] + c[-1:]) if not x.startswith('|---') else x[:-4]
        if x.startswith('- ぶつからずに 25 秒以下 = '):
            if out and out[-1] == '':
                out.pop()
            continue
        out.append(x)
    return '\n'.join(out) + '\n'


def test_rec3_4_rest_of_the_markdown_is_the_same(tmp_path, old):
    """作り物の記録の 1 日(P1・天井・発進 ON・出ず・衝突あり)で、新しい列のほかは md・JSON の組が直す前と同じ"""
    spec = [('20261103-100000', 1, dict(P, speed_ceiling=0), None, {}),
            ('20261103-100500', 2, dict(P), None, {}),
            ('20261103-101500', 4, dict(P, launch_boost=1), (1.0, 0.5, 1), {}),
            ('20261103-102000', 5, dict(P, launch_boost=1), None, {}),
            ('20261103-102500', 6, dict(P, speed_ceiling=0), None, {'hits': [(16.0, 5.0)]})]
    for log, no, prm, launch, kw in spec:
        write_standing_log(tmp_path, log, params=prm, launch=launch, run='runB',
                           description='ai_no=%d day=2026-11-03 block=%d cand=EXP model=runB tune=t%d' % (no, no, no), **kw)
    bench = tmp_path / 'bench.json'
    bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}]}), encoding='utf-8')
    logs, _ = s3.day_logs('20261103', tmp_path)
    res = s3.run(logs, tmp_path, tmp_path / 'nolab', str(bench))
    for ignore in ((), ('launch',), ('trim', 'ceiling')):
        md_new, runs_new, g_new = qp.to_markdown(res, '20261103', None, ignore)
        md_old, runs_old, g_old = old['qp'].to_markdown(res, '20261103', None, ignore)
        assert _strip_goal(md_new) == md_old and md_new != md_old
        assert runs_new == runs_old
        NEW = ('standing_clean_le25', 'standing_clean_best_s', 'standing_best_coll')
        assert [{k: v for k, v in g.items() if k not in NEW} for g in g_new] == g_old


# ================================================================== collect.py(r1-rec3-1・3・5)
DESC = 'ai_no=%d bench=real_bench_20261023 day=2026-11-03 block=%d cand=EXP model=runB memo=x'
PRM = {'throttle_gain': 0.9, 'steer_gain': 1.0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0,
       'auto_target_mps': 4.27, 'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2, 'zone_caps': 0}
ON = dict(PRM, launch_boost=1)


class Day(object):
    """作り物の 11/3 の回 3(block 12〜19 = 3-1〜3-8)。add で記録を足し、run で collect.main を流す"""

    def __init__(self, tmp):
        self.tmp = tmp
        self.raw = tmp / 'raw'
        self.raw.mkdir()
        self.bench = tmp / 'bench.json'
        self.bench.write_text(json.dumps({'candidates': [{'id': 'B', 'run': 'runB'}],
                                          'days': [{'date': '2026-11-03', 'blocks': [{'block': b, 'run_id': '3-%d' % (b - 11)}
                                                                                     for b in range(12, 20)]}]}), encoding='utf-8')
        self.n = 0

    def add(self, block, params, **kw):
        log = '20261103-13%02d00' % (self.n * 5)
        self.n += 1
        kw.setdefault('dur', 30.0)
        if params.get('launch_boost'):
            kw.setdefault('t_ai', 1.3)
            kw.setdefault('t_move', 1.4)
            kw.setdefault('launch', (1.3, 0.5, 1))
        d, _, _ = write_standing_log(self.raw, log, run='runB', description=DESC % (block, block), ramp_s=0.5, params=params, **kw)
        return log, d

    def run(self, path=COLLECT, name='collect_r12', args=()):
        m = load_mod(name, path)
        m.RAW, m.LAB, m.BENCH, m.OUT = self.raw, self.tmp / 'nolab', self.bench, self.tmp / 'out'
        out = self.tmp / ('%s.json' % name)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.main(['--day', '20261103', '--out', str(out)] + list(args))
        txt = out.read_text(encoding='utf-8')
        return json.loads(txt), buf.getvalue(), txt, m


def _round3_with_on_hit(tmp, hit=True):
    """反証の demo_r1: OFF 3 本・ON 3 本。3-4(ON)は発進のあと 3.0 秒に 6.0G で当たり 3.1 秒に止まり、R1 の起点(4.0 秒)の前の
    3.6 秒に人側。hit=False なら当たらずに止まって人側(人が止めた)"""
    D = Day(tmp)
    D.add(12, PRM)
    D.add(13, ON)
    D.add(14, PRM)
    bad, _ = D.add(15, ON, hits=[(3.0, 6.0)] if hit else (), stop_after=3.1, human_after=3.6)
    D.add(16, PRM)
    D.add(17, ON)
    return D, bad


def test_rec3_1_on_start_that_hit_before_r1_fails(tmp_path):
    D, bad = _round3_with_on_hit(tmp_path)
    j, out, _, _ = D.run()
    row = [r for r in j['launches'] if r['log'] == bad][0]
    assert row['to_r1_s'] is None and row['why_no_r1'] == 'AI の区間が短い' and row['launch_state'] == 'on'
    assert len(row['collisions_first_lap']) == 1 and row['collisions_first_lap'][0][1] == pytest.approx(6.0, abs=0.01)
    assert row['first_lap_end'] == 'no_r1' and row['first_lap_full'] is False
    g = j['summary'][0]
    assert g['by_launch']['on']['n'] == 2 and g['not_reached']['on']['n'] == 1 and g['not_reached']['on']['collisions'] == 1
    assert g['not_reached']['on']['logs'] == ['%s@1.30' % bad]
    c = g['compare']
    assert c['checks']['first_lap_collisions_0'] is False and c['checks']['on_all_reached_r1'] is False and c['all_ok'] is False
    assert c['checks']['to_r1'] and c['checks']['slip'] and c['checks']['lh_speed']        # ほかは届いた本で前と同じ
    assert '→ 満たさない' in out and '→ 使ってよい' not in out
    assert 'R1 に届かなかった発進(on)1 本(比べの数に入れない): %s@1.30' % bad in out and '6.0G' in out
    # 直す前の写し(再現): 当たった ON が黙って抜け「使ってよい」
    jo, oo, _, _ = D.run(path=BEFORE / 'collect.py.orig', name='collect_r12_before1')
    assert jo['summary'][0]['compare']['all_ok'] is True and '→ 使ってよい' in oo
    assert 'collisions_first_lap' not in [r for r in jo['launches'] if r['log'] == bad][0]


def test_rec3_1_long_stuck_on_start_fails(tmp_path):
    """反証の run_long: 当たって止まったまま AI 側を 20 秒まで(AI の区間 18.7 秒)→「15 秒以内に起点なし」でも同じ"""
    D = Day(tmp_path)
    D.add(12, PRM)
    D.add(13, ON)
    bad, _ = D.add(15, ON, hits=[(3.0, 6.0)], stop_after=3.1, human_after=20.0)
    j, out, _, _ = D.run()
    row = [r for r in j['launches'] if r['log'] == bad][0]
    assert row['why_no_r1'] == '15 秒以内に起点なし' and len(row['collisions_first_lap']) == 1
    c = j['summary'][0]['compare']
    assert c['all_ok'] is False and c['checks']['first_lap_collisions_0'] is False and '→ 満たさない' in out


def test_rec3_1_on_stopped_without_hit_fails_and_skip_reads_again(tmp_path, old):
    """当たらずに R1 の前で止めた ON: 衝突は 0 でも満たさない(最初の周を確かめられない)。車のせいでなければ --skip で外せる"""
    D, bad = _round3_with_on_hit(tmp_path, hit=False)
    j, out, _, _ = D.run()
    c = j['summary'][0]['compare']
    assert c['checks']['first_lap_collisions_0'] is True and c['checks']['on_all_reached_r1'] is False and c['all_ok'] is False
    assert '--skip 記録名' in out
    j2, out2, _, _ = D.run(name='collect_r12_skip', args=['--skip', bad])
    c2 = j2['summary'][0]['compare']
    assert c2['all_ok'] is True and c2['checks'] == {'to_r1': True, 'first_lap_collisions_0': True, 'slip': True, 'lh_speed': True}
    assert 'not_reached' not in j2['summary'][0] and j2['summary_skipped'] == ['%s@1.30' % bad]
    assert 'まとめから外した発進(--skip %s): %s@1.30' % (bad, bad) in out2
    assert j2['launches'] == j['launches']                      # 発進ごとの行は変えない
    # LOG@T の形でも外せる。当たらない名前なら「当たる記録なし」
    j3, _, _, _ = D.run(name='collect_r12_skip2', args=['--skip', '%s@1.30' % bad])
    assert j3['summary'] == j2['summary']
    _, out4, _, _ = D.run(name='collect_r12_skip3', args=['--skip', '20261103-999999'])
    assert '(当たる記録なし)' in out4


def test_rec3_1_all_on_unreached_still_prints_not_satisfied(tmp_path):
    D = Day(tmp_path)
    D.add(12, PRM)
    bad, _ = D.add(13, ON, hits=[(3.0, 6.0)], stop_after=3.1, human_after=3.6)
    j, out, _, _ = D.run()
    g = j['summary'][0]
    assert 'on' not in g['by_launch'] and g['compare']['all_ok'] is False
    assert g['compare']['checks'] == {'to_r1': False, 'first_lap_collisions_0': False, 'slip': False, 'lh_speed': False,
                                      'on_all_reached_r1': False}
    assert 'ON − OFF: 起点まで — 秒' in out and '→ 満たさない' in out


def _row(log, state, to_r1=3.0, hits=(), start_no=1, t_ai=1.0, why=None):
    S = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'speed_auto': 0, 'launch_boost': 0 if state == 'off' else 1, 'launch_us': 250.0}
    r = {'log': log, 'no': log[-4:], 'session': '11/03 回3', 'model': 'B', 'gain': 0.9, 'settings': S, 't_ai': t_ai,
         'to_r1_s': to_r1, 'loss_s': None if to_r1 is None else to_r1 - 2.25, 'move_0p2_s': 0.3, 'slip': {'ratio': 1.05},
         'v_launch_at_lh': 2.0, 'collisions': [], 'collisions_first_lap': [list(h) for h in hits], 'ai_seg_s': 2.3,
         'launch_on_s': 0.5 if state == 'on' else None, 'launch_end_reason': '速さに届いた' if state == 'on' else None,
         'launch_state': state, 'start_no': start_no}
    if to_r1 is None:
        r['why_no_r1'] = why or 'AI の区間が短い'
        for k in ('to_r1_s', 'loss_s', 'slip', 'v_launch_at_lh', 'collisions'):
            r.pop(k)
        r['to_r1_s'] = None
    return r


def test_rec3_1_first_only_also_reports_the_first_start():
    """--first-only: 1 回目の発進が R1 の前で当たり、人が戻した 2 回目の発進が届いた本(前は 2 回目も 1 回目も消えた)"""
    col = load_mod('collect_r12_fo', COLLECT)
    rows = [_row('20261103-133000', 'off', 3.0), _row('20261103-134000', 'off', 3.0),
            _row('20261103-133500', 'on', None, hits=[(3.0, 6.0, 'LH')]),
            _row('20261103-133500', 'on', 2.7, start_no=2, t_ai=12.0), _row('20261103-134500', 'on', 2.7)]
    for fo in (False, True):
        g = col.summarize(rows, first_only=fo)[0]
        assert g['not_reached']['on']['n'] == 1 and g['compare']['all_ok'] is False
        assert g['compare']['checks']['first_lap_collisions_0'] is False
    # 2 回目以降の届かなかった発進は --first-only なら入れない(届いた 2 回目と同じ扱い)
    rows2 = [r for r in rows if r['to_r1_s'] is not None] + [_row('20261103-134500', 'on', None, start_no=2, t_ai=20.0)]
    assert col.summarize(rows2, first_only=True)[0]['compare']['all_ok'] is True
    assert col.summarize(rows2)[0]['compare']['all_ok'] is False


def test_rec3_1_off_unreached_is_listed_but_does_not_fail_on():
    """R1 に届かなかった OFF の発進は名前を出すが、ON の条件(on_all_reached_r1・「→ ON の発進が R1 の前で終わった」の行)にはしない。
    2026-10-28 fix8 tools r2(反証 8-2 回目 w2-tools-r8-2-off-few-left の決め): その OFF の本の発進が比べに 1 本も残らない(下の 134000)と、
    残りの OFF 1 本で比べることになるので「満たさない」(checks の off_all_compared だけが False)。その本の別の発進が比べに残れば(2 つ目)
    前と同じ「使ってよい」(tests/test_fix8_tools_r2.py)"""
    col = load_mod('collect_r12_off', COLLECT)
    rows = [_row('20261103-133000', 'off', 3.0), _row('20261103-134000', 'off', None, hits=[(2.0, 5.0, 'HS')]),
            _row('20261103-133500', 'on', 2.7), _row('20261103-134500', 'on', 2.7),
            _row('20261103-135000', 'on_not_started', None)]
    g = col.summarize(rows)[0]
    assert g['not_reached']['off']['n'] == 1 and g['not_reached']['on_not_started']['n'] == 1
    assert 'on_all_reached_r1' not in g['compare']['checks']
    assert sorted(k for k, v in g['compare']['checks'].items() if v is False) == ['off_all_compared']
    assert g['compare']['all_ok'] is False and g['off_dropped'] == ['20261103-134000@1.00(R1 に届かなかった・AI の区間が短い)']
    txt = '\n'.join(col.summary_lines([g]))
    assert 'R1 に届かなかった発進(off)1 本' in txt and '→ ON の発進が R1 の前で終わった' not in txt
    rows2 = rows + [_row('20261103-134000', 'off', 3.0, start_no=2, t_ai=12.0)]     # 同じ本の 2 回目の発進が比べに残る
    g2 = col.summarize(rows2)[0]
    assert g2['not_reached']['off']['n'] == 1 and 'off_dropped' not in g2
    assert g2['compare']['all_ok'] is True and 'on_all_reached_r1' not in g2['compare']['checks']


def test_rec3_3_midrun_runs_are_left_out(tmp_path):
    """反証の demo_r3 + 天井の途中 OFF(AI_CMD)と発進の前進の終わり 10。比べに入れず、名前を出す。standing_3lap と同じ見方"""
    D = Day(tmp_path)
    PC = dict(PRM, speed_ceiling=1)
    D.add(12, PC)
    log2, d2 = D.add(13, dict(PC, launch_boost=1))
    with open(d2 / 'events.jsonl', 'a', encoding='utf-8') as f:     # drive.set_params が書く形(人側のうちの 1.1 秒に天井 OFF)
        f.write(json.dumps({'kind': 'params', 'changed': {'speed_ceiling': [1, 0]}, 'epoch_us': T0_US + 1_100_000,
                            't_us': T0_US + 1_100_000}) + '\n')
    m = json.loads((d2 / 'log.json').read_text())
    m['drive']['params_end'] = dict(PC, launch_boost=1, speed_ceiling=0)
    (d2 / 'log.json').write_text(json.dumps(m))
    D.add(14, PC)
    log4, _ = D.add(15, dict(PC, launch_boost=1))
    log5, _ = D.add(16, dict(PC, launch_boost=1), launch=(1.3, 0.5, 10))     # 発進の前進を途中で OFF にした(終わり 10)
    log6, d6 = D.add(17, dict(PC, launch_boost=1))
    for s in sorted(d6.glob('*/AI_CMD.npz')):          # 天井のスイッチを 2.0 秒から OFF(AI_CMD の ceil_on が途中から False)
        z = dict(np.load(s))
        t = (z['timestamps_us'] - T0_US) / 1e6
        n = len(t)
        z.update(ceil_on=t < 2.0, ceil_mode=np.zeros(n, np.int8), ceil_applied=np.zeros(n, bool), ceil_left=np.zeros(n, bool))
        np.savez(s, **z)
    j, out, _, _ = D.run()
    rows = {r['log']: r for r in j['launches']}
    for lg in (log2, log5, log6):
        r3 = s3.analyze_log(lg, D.raw, D.tmp / 'nolab', {'runB': 'B'})
        assert rows[lg].get('midrun') and [x['key'] for x in rows[lg]['midrun']] == [x['key'] for x in r3['midrun']]
    assert 'midrun' not in rows[log4] and 'midrun' not in rows['20261103-130000']
    assert [x['key'] for x in rows[log2]['midrun']] == ['speed_ceiling']
    assert [x['key'] for x in rows[log5]['midrun']] == ['launch_boost']
    assert [x['key'] for x in rows[log6]['midrun']] == ['speed_ceiling']
    g = j['summary'][0]
    assert g['by_launch']['on']['logs'] == ['%s@1.30' % log4] and g['by_launch']['off']['n'] == 2
    assert g['midrun_excluded'] == ['%s@1.30(天井 1→0 1.1 秒)' % log2, '%s@1.30(発進 1→0 1.8 秒)' % log5,
                                    '%s@1.30(天井 1→0 2.0 秒)' % log6]
    assert '途中で変えた 3 本(比べに入れない。ON の組にも OFF の組にも入れない。手順書 7 章)' in out
    # 直す前の写し(再現): 途中で変えた 3 本も ON の組に入り、印も無い
    jo, _, _, _ = D.run(path=BEFORE / 'collect.py.orig', name='collect_r12_before3')
    assert sorted(jo['summary'][0]['by_launch']['on']['logs']) == sorted('%s@1.30' % x for x in (log2, log4, log5, log6))
    assert not any('midrun' in r for r in jo['launches'])


def test_rec3_5_collect_settings_and_mixed_options_note(tmp_path):
    """朝に見張り ON → 3-3 の前に再起動で 0 に戻った形。行の設定に出て、組に「混ざっている」の注意"""
    D = Day(tmp_path)
    D.add(12, dict(PRM, rec_back_guard=1, rec_hold_keep=0, launch_no_kick=0))
    D.add(13, dict(ON, rec_back_guard=1, rec_hold_keep=0, launch_no_kick=0))
    D.add(14, dict(PRM, rec_back_guard=0, rec_hold_keep=0, launch_no_kick=0))
    D.add(15, dict(ON, rec_back_guard=0, rec_hold_keep=0, launch_no_kick=0))
    j, out, _, _ = D.run()
    assert [r['settings']['rec_back_guard'] for r in j['launches']] == [1, 1, 0, 0]
    g = j['summary'][0]
    assert g['mixed_options'] == {'後ろの見張り': {'on': 2, 'off': 2}}
    assert '注意: 試験の選択肢の ON/OFF が混ざっている(後ろの見張り ON 2・OFF 2)' in out
    assert g['compare']['all_ok'] is True                 # 比べそのものは変えない(注意だけ)


def _same_day(tmp, opts=None, zero=True):
    """fix2 の試験の日に近い形(OFF・ON・ON なのに出ない・0 バイト・dry run)。R1 に届かない発進・途中で変えた本は無い。
    zero = False: 0 バイトの log.json の本を作らない(2026-10-27 直す 3 回目・1 組目 r3-r3-rec-1 で、⑤ はその本を「読めない記録」として
    出し ON の組を満たさないにした。「直す前と同じ」はその本の無い日で見る。ある日は tests/test_fix3_r3_1_tools.py が見る)"""
    D = Day(tmp)
    o = opts or {}
    D.add(12, dict(PRM, **o), hits=[(10.0, 3.0)])
    D.add(13, dict(ON, **o))
    D.add(14, dict(PRM, **o))
    _, dn = D.add(15, dict(ON, **o), t_ai=1.0, t_move=1.5, launch=None)        # ON なのに出なかった
    (dn / 'events.jsonl').write_text(json.dumps({'kind': 'launch_boost', 'event': 'skip', 'why': 'no_switch',
                                                 'reason': '最初から AI 側', 'cam_t_us': T0_US + 1000}) + '\n', encoding='utf-8')
    for name, meta in (('20261103-150000', None), ('20261103-150500', {'drive': {'run': 'runB', 'dry_run': True}})):
        if meta is None and not zero:
            continue
        (D.raw / name).mkdir()
        (D.raw / name / 'log.json').write_text('' if meta is None else json.dumps(meta), encoding='utf-8')
    return D


@pytest.mark.parametrize('args', [(), ('--first-only',)])
def test_collect_output_same_as_before_without_the_new_cases(tmp_path, args):
    """届かない発進・途中で変えた本・選択肢の欄が無い日: JSON と標準出力が直す前の写しとまるごと同じ(2026-10-27 r3-r3-rec-1 から
    0 バイトの log.json の本の無い日で)"""
    D = _same_day(tmp_path, zero=False)
    j_new, out_new, txt_new, _ = D.run(args=args, name='collect_r12_same')
    j_old, out_old, txt_old, _ = D.run(path=BEFORE / 'collect.py.orig', name='collect_r12_before', args=args)
    assert txt_new == txt_old.replace(str(tmp_path / 'collect_r12_before.json'), str(tmp_path / 'collect_r12_same.json'))
    assert out_new.replace('collect_r12_same', 'X') == out_old.replace('collect_r12_before', 'X')
    assert j_new['summary'] and 'compare' in j_new['summary'][0]


def test_collect_options_zero_only_adds_settings_fields(tmp_path):
    """選択肢の欄が 0 で入っている日(再起動のあとのサーバー): 行の settings に欄が足されるだけで、まとめ・文は前と同じ"""
    D = _same_day(tmp_path, OPT_0, zero=False)          # 2026-10-27 r3-r3-rec-1 から 0 バイトの log.json の本の無い日で
    j_new, out_new, _, _ = D.run(name='collect_r12_z')
    j_old, out_old, _, _ = D.run(path=BEFORE / 'collect.py.orig', name='collect_r12_zb')
    assert j_new['summary'] == j_old['summary']
    strip = lambda r: dict(r, settings={k: v for k, v in r['settings'].items() if k not in OPT_0})      # noqa: E731
    assert [strip(r) for r in j_new['launches']] == j_old['launches']
    assert all(r['settings']['rec_hold_keep'] == 0 for r in j_new['launches'])
    assert out_new.replace('collect_r12_z.', 'X.') == out_old.replace('collect_r12_zb.', 'X.')


def test_rec3_3_midrun_on_collision_still_counts_for_the_first_lap_check():
    """途中で変えた ON の発進は比べ(中央値)に入れないが、その最初の周の衝突は「最初の周の衝突 0」に数える(衝突は止める条件)。
    途中で変えた OFF の発進の衝突は数えない"""
    col = load_mod('collect_r12_mc', COLLECT)
    mid = [{'key': 'speed_ceiling', 'from': 1, 'to': 0, 't_s': 1.1, 'src': 'event'}]
    rows = [_row('20261103-133000', 'off', 3.0), _row('20261103-134000', 'off', 3.0),
            _row('20261103-133500', 'on', 2.7), _row('20261103-134500', 'on', 2.7),
            dict(_row('20261103-135000', 'on', 2.7, hits=[(5.0, 4.5, 'LH')]), midrun=mid),
            dict(_row('20261103-135500', 'off', 3.0, hits=[(5.0, 4.5, 'LH')]), midrun=mid)]
    g = col.summarize(rows)[0]
    assert g['by_launch']['on']['n'] == 2 and g['by_launch']['off']['n'] == 2 and g['midrun_on_collisions'] == 1
    assert g['compare']['checks']['first_lap_collisions_0'] is False and g['compare']['all_ok'] is False
    assert 'うち ON の発進の最初の周の衝突 1(比べの「最初の周の衝突 0」には数える)' in '\n'.join(col.summary_lines([g]))
    g2 = col.summarize(rows[:4] + rows[5:])[0]               # 途中で変えた OFF の衝突だけ → 比べは満たす
    assert 'midrun_on_collisions' not in g2 and g2['compare']['all_ok'] is True and len(g2['midrun_excluded']) == 1
