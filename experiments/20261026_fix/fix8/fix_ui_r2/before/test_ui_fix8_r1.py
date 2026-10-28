"""画面の係の直し 8-1 回目(2026-10-28 fix8。反証 8-1 回目・係 ui の指摘 3 件。
~/jetformer_reports/fix_20261026/fix8/fix_ui_r1.md)の試験。

  runs-opt-mark-on-the-on-runs(② experiments/20261026_fix/plan/runs_1103.py): 試験の選択肢の印「回 N の試験の選択肢が … と違う
    (再起動で OFF に戻った? …)」と「計画と違う N 本」を、回の最初の本と比べて付けたので、回の最初の本が OFF の日(回の切れ目で起き直し・
    入れ忘れのまま 1 本走り、2 本目から入れ直した)は ON で走った正しい本に付き、OFF の本に付かなかった(表の下の「回の本は X から」とも
    食い違った)。直し: その回の計画の選択肢 = 回の中で ON の選択肢がいちばん多い本(同じ数なら先の本)と比べる(起き直しも手順書 3 章の
    「止めて OFF」も ON → OFF の向き)。回の最初の本がいちばん多い日(手順書どおり・回の途中から OFF・わざと OFF・全部 OFF)は前と
    バイト単位で同じ。
  stale-status-after-drive-tab-return(app.js loadDrive・pollDrive): 走行タブを離れている間は表示の問い合わせの輪が止まるのに
    paramsLoaded は true のままで、走行タブに戻った直後(モデルの一覧の GET を待つ間)の ▶・基準の設定・ブロックが、離れたときの古い
    status で動いた(確認の文は「見張り: ON」「左旋回中 4.1」、車は curl・別の画面で変えたサーバーの値で走った)。直し(表示と断る時だけ):
    止まっていれば(drIdle)loadDrive の頭で paramsLoaded を下ろし、戻ったあとに出した表示の問い合わせが通るまで drNotSynced が「サーバーの
    設定を読み込んでいます」で断る(画面を開いた直後と同じ)。戻る前に出した問い合わせの返事は「読み直した」に数えない(drTabGen)。
    問い合わせ・送る物(POST の body・走行開始の params)・版(9)は前と同じ。走行中・token のある画面は前と同じ。
  toast-half-width-on-phone(style.css .toast): left: 50%・幅 auto のため使える幅が画面の半分になり、スマホ(420 px)では長い断りの文が
    幅 210 px の細い列で「停止中」のバッジと走行タブの文に重なった。直し: width: max-content・max-width: 90%(min(90%, 960px))。

直す前の写し tests/fixtures/fix8_ui_r1_before/(app.js 2b211259…・style.css 5842642f…・runs_1103.py 33bc7fca…。SHA256SUMS)。
画面は node の vm で(tests/fixtures/app_sim_fix7_r2.js = 1 つの画面。tests/fixtures/app_sim_fix8.js = 反証 8-1 回目の台本 sim2.js の
写し = 2 つの画面が 1 つの偽のサーバーを共有)。409 の文は本物の server._boot_refusal(import だけ。起動しない)。トーストの幅は
JF_UI_FIREFOX=1 のときだけ headless の Firefox で(file:// の頁。app.js を読まない = 通信なし)。車・jetformer-web には触れない。
② は作り物の log.json(tmp)だけ。data/bench_candidates.json は読むだけ。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix8_r1.py
    # Firefox の 1 本(--basetemp は /tmp・隠しフォルダの外 = snap の Firefox が読める場所):
    JF_UI_FIREFOX=1 /home/jetson/bin/jf-heavy --lock big --mem 3000M -- ... -m pytest ... tests/test_ui_fix8_r1.py -k firefox

JF_UI81_APP・JF_UI81_RUNS・JF_UI81_CSS=<写し> でその写しを「今の版」として流す(わざと壊す台本 experiments/20261026_fix/fix8/fix_ui_r1/mut/)。
"""
import ast
import contextlib
import difflib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

from jetformer import drive

import test_ui_fix6 as F6
import test_ui_fix7 as T7
import test_ui_fix7_r2 as T72

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
FIXT = ROOT / 'tests' / 'fixtures'
BEFORE = FIXT / 'fix8_ui_r1_before'                     # 直す前の app.js・style.css・runs_1103.py(SHA256SUMS)
APP = Path(os.environ.get('JF_UI81_APP') or STATIC / 'app.js')
CSS = Path(os.environ.get('JF_UI81_CSS') or STATIC / 'style.css')
RUNS = Path(os.environ.get('JF_UI81_RUNS') or ROOT / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py')
SIM1 = FIXT / 'app_sim_fix7_r2.js'                     # 1 つの画面(7-2 回目の台本。変えていない)
SIM2 = FIXT / 'app_sim_fix8.js'                        # 2 つの画面(反証 8-1 回目の sim2.js の写し)
REVIEW_SIM2 = ROOT / 'experiments' / '20261026_fix' / 'fix8' / 'review_ui_r1' / 'sim2.js'
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
MD = '/api/drive/models'
LOADING = 'サーバーの設定を読み込んでいます。少し待ってから押してください'     # drNotSynced(画面を開いた直後と同じ文)
GUARD_LINE, KEEP_LINE = '\nリカバリーの後ろの見張り: ON', '\n中立止めのあともリカバリーを残す: ON'
MARK_NOTE = '(再起動で OFF に戻った? 3 章で止めて OFF にした本 X から先なら数える。手順書 3・7 章)'
cands = F6.cands
blk = F6.blk
server_params = F6.server_params
MORNING_ON = server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sums(d):
    return dict(reversed(x.split()) for x in (d / 'SHA256SUMS').read_text().splitlines() if x.strip())


def _diff(a_path, b_path):
    a = Path(a_path).read_text(encoding='utf-8').splitlines()
    b = Path(b_path).read_text(encoding='utf-8').splitlines()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
    return [a[i] for o in ops for i in range(o[1], o[2])], [b[j] for o in ops for j in range(o[3], o[4])]


# ================================================================== 形(直す前の写し・変えた所・変えていないファイル)
def test_before_copies_are_the_versions_before_this_fix():
    """直す前の写しは 7-2 回目のあとの app.js(2b211259…)・10/26 からの style.css(5842642f…)・fix8 tools-decided のあとの
    runs_1103.py(33bc7fca…)"""
    sums = _sums(BEFORE)
    assert set(sums) == {'app.js', 'style.css', 'runs_1103.py'}
    for name, h in sums.items():
        assert _sha(BEFORE / name) == h, name
    assert sums['app.js'].startswith('2b211259db71') and sums['style.css'].startswith('5842642f6e85')
    assert sums['runs_1103.py'].startswith('33bc7fca9921')


def test_app_js_only_added_lines_and_no_sending():
    """app.js は行を足しただけ(消した行なし)。足した行に送る物(POST・body・client)・問い合わせ(api・fetch)・タイマー・確認の文は無い。
    足したコードは loadDrive の頭の 1 行と pollDrive の 2 行だけ。版は 9 のまま。server.py・index.html・bench.py は変えていない"""
    removed, added = _diff(BEFORE / 'app.js', APP)
    assert removed == []
    code = [x.split('//')[0].strip() for x in added if x.split('//')[0].strip()]
    assert code == [
        'if (drIdle()) { state.drive.paramsLoaded = false; state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }',
        'const gen = state.drive.drTabGen;',
        'if (st && gen !== state.drive.drTabGen && drIdle()) state.drive.paramsLoaded = false;']
    assert not any(re.search(r"'POST'|body|client|fetch\(|api\(|setTimeout|confirm\(|toast\(", x) for x in code)
    js = APP.read_text(encoding='utf-8')
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9'] and re.findall(r'const body = \{ client: (\d+),', js) == ['9']
    assert _sha(ROOT / 'jetformer' / 'web' / 'server.py').startswith('ad7a7d05f7a9')
    assert _sha(STATIC / 'index.html').startswith('ef81bb6017d6')
    assert _sha(ROOT / 'jetformer' / 'web' / 'bench.py').startswith('a633bec341a3')


def test_style_css_only_the_toast_rule():
    """style.css は .toast の 1 行だけ変えた(width: max-content と max-width: min(90%, 960px)を足した。ほかの宣言は同じ)と注の行"""
    removed, added = _diff(BEFORE / 'style.css', CSS)
    old = [x for x in removed if x.startswith('.toast {')]
    assert removed == old and len(old) == 1
    new = [x for x in added if x.startswith('.toast {')]
    assert len(new) == 1 and all(x.startswith('/*') or x.startswith('   ') for x in added if x not in new)
    assert new[0] == old[0].replace('max-width: 90%;', 'width: max-content; max-width: 90%; max-width: min(90%, 960px);')
    assert 'left: 50%; transform: translateX(-50%);' in new[0]


def test_runs_1103_only_round_options_changed():
    """runs_1103.py で変えたのは round_options の中身(と説明の文・注)だけ。ほかの関数・定数・options_line・round_form・round_rev は
    AST で同じ"""
    def body(p):
        mod = ast.parse(Path(p).read_text(encoding='utf-8'))
        out = {}
        for n in mod.body[1:]:                                  # [0] は説明の文
            key = getattr(n, 'name', None) or ast.dump(n)[:60]
            if isinstance(n, ast.FunctionDef) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                n = ast.FunctionDef(**dict(n.__dict__, body=n.body[1:]))     # 関数の説明の文は比べない
            out[key] = ast.dump(n)
        return out
    a, b = body(BEFORE / 'runs_1103.py'), body(RUNS)
    assert set(a) == set(b)
    assert [k for k in a if a[k] != b[k]] == ['round_options']


def test_sim_copy_is_the_review_harness():
    """台本 app_sim_fix8.js は反証 8-1 回目の sim2.js に 2 行の頭書きを足しただけ"""
    head, body = SIM2.read_text(encoding='utf-8').split('\n', 2)[1], SIM2.read_text(encoding='utf-8').split('\n', 2)[2]
    if REVIEW_SIM2.exists():
        assert body == REVIEW_SIM2.read_text(encoding='utf-8')
    assert hashlib.sha256(body.encode('utf-8')).hexdigest()[:12] in head


# ================================================================== runs-opt-mark-on-the-on-runs(② runs_1103.py)
BENCH = json.loads((ROOT / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
DAY = [d for d in BENCH.get('days') or [] if d.get('date') == '2026-11-03']
EXPECT = {int(b['block']): {k: v for k, v in (b.get('expect') or {}).items() if v is not None} for d in DAY for b in d.get('blocks') or []}
BLOCK_OF = {b.get('run_id'): int(b['block']) for d in DAY for b in d.get('blocks') or []}
RBASE = {'throttle_gain': 0.9, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'launch_boost': 0, 'launch_us': 250.0,
         'steer_trim_us': 0.0, 'zone_caps': 0, 'rev_s': 0.4, 'rev_us': 300.0, 'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}
G = {'rec_back_guard': 1, 'rec_hold_keep': 1}            # 11/3 の朝の形(見張り・残す ON)
OFF = {}
NK = {'launch_no_kick': 1}
ROUNDS = {1: 5, 2: 6, 3: 8, 4: 4}                        # 11/3 の回と本の数(1-1〜1-5・2-1〜2-6・3-1〜3-8・4-1〜4-4)


@pytest.fixture(scope='module')
def day1103():
    if [BLOCK_OF.get('%d-%d' % (r, i)) for r in ROUNDS for i in range(1, ROUNDS[r] + 1)] != list(range(1, 24)):
        pytest.skip('data/bench_candidates.json の 11/3 のブロックが前提(1〜23 = 1-1〜4-4)と違う')


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_day(raw, recs):
    """recs = [(本 '1-1' か None(本「—」= 朝の確かめ・memo に試し), 試験の選択肢)]。記録の名前は 08:00 から 10 分ずつ、ai_no は 1 から。
    試験の選択肢のほかは計画(expect の null でない値)どおり = 印は試験の選択肢だけ"""
    for i, (rid, opts) in enumerate(recs):
        log = '20261103-%02d%02d00' % (8 + i // 6, (i % 6) * 10)
        d = raw / log
        d.mkdir(parents=True)
        if rid is None:
            desc, plan = 'ai_no=%d memo=試し' % (i + 1), {}
        else:
            b = BLOCK_OF[rid]
            desc, plan = 'bench=real_bench_20261023 day=2026-11-03 official=1 battery=#A block=%d ai_no=%d' % (b, i + 1), EXPECT.get(b, {})
        params = dict(RBASE, **plan, **opts)
        (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
            'run': 'clean_driving_20261011-nocmd-s3-20261011-024626', 'params': params, 'dry_run': False}},
            ensure_ascii=False), encoding='utf-8')
        if params.get('launch_boost') == 1:          # 計画が ON の本(回 3 の偶数)は発進の前進が始まった記録(その印を出さない)
            (d / 'events.jsonl').write_text(json.dumps({'kind': 'launch_boost', 'event': 'start'}) + '\n', encoding='utf-8')


def _runs(path, raw, name):
    """runs_1103 の標準出力(表と下の行)と JSON の行"""
    m = _load(name, path)
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


def both_runs(tmp_path, recs, tag):
    raw = tmp_path / ('raw_%s' % tag)
    _write_day(raw, recs)
    return _runs(RUNS, raw, 'runs81_n_%s' % tag), _runs(BEFORE / 'runs_1103.py', raw, 'runs81_o_%s' % tag)


def opt_marks(rows):
    return {(r['run_id'] or r['log']): [x for x in r['diffs'] if '試験の選択肢が' in x] for r in rows}


def count_line(table):
    return [x for x in table if ' 本・計画と違う ' in x]


def options_line(table):
    return [x for x in table if x.startswith('試験の選択肢(')]


def test_runs_first_run_of_round_off_marks_the_off_run(tmp_path, day1103):
    """反証の E: 回 2 の前に起き直し、2-1 を OFF のまま走り、2-2 の前に入れ直した。今は OFF の 2-1 に印(比べる相手は ON の 2-2)、
    「計画と違う 1 本」。直す前は ON の 2-2・2-3 に印・「計画と違う 2 本」(指摘)。表の下の試験の選択肢の行は前と同じ"""
    recs = [('1-1', G), ('1-2', G), ('2-1', OFF), ('2-2', G), ('2-3', G)]
    (tn, rn), (to, ro) = both_runs(tmp_path, recs, 'e')
    mn, mo = opt_marks(rn), opt_marks(ro)
    want = ('回 2 の試験の選択肢が 2-2(No.4)と違う: 後ろの見張り OFF(2-2 は ON)・リカバリーを残す OFF(2-2 は ON)' + MARK_NOTE)
    assert mn == {'1-1': [], '1-2': [], '2-1': [want], '2-2': [], '2-3': []}
    assert [k for k, v in mo.items() if v] == ['2-2', '2-3']                    # 直す前: ON の本に印
    assert count_line(tn) == ['5 本・計画と違う 1 本'] and count_line(to) == ['5 本・計画と違う 2 本']
    assert options_line(tn) == options_line(to) and '2-1(No.3)から なし(前の本と違う)' in options_line(tn)[0]
    assert [r['diffs'] for r in rn if r['run_id'] != '2-1'] == [[]] * 4


def test_runs_dash_then_first_round_run_off_agrees_with_the_line(tmp_path, day1103):
    """反証の K: 朝の確かめ(本「—」)のあと選択肢を入れる前に 1-1 を走り、1-2 から ON。表の下の行は「…(No.1)から なし(回の本は
    1-1(No.2)から) → 1-2(No.3)から 後ろの見張り・リカバリーを残す」= 1-1 は OFF。今は印もその 1-1 に付く(行と食い違わない)。
    直す前は ON の 1-2・1-3 に印"""
    recs = [(None, OFF), ('1-1', OFF), ('1-2', G), ('1-3', G)]
    (tn, rn), (to, ro) = both_runs(tmp_path, recs, 'k')
    line = options_line(tn)
    assert line == options_line(to) and line[0].endswith(
        '20261103-080000(No.1)から なし(回の本は 1-1(No.2)から) → 1-2(No.3)から 後ろの見張り・リカバリーを残す(前の本と違う)')
    mn, mo = opt_marks(rn), opt_marks(ro)
    assert mn['1-1'] == ['回 1 の試験の選択肢が 1-2(No.3)と違う: 後ろの見張り OFF(1-2 は ON)・リカバリーを残す OFF(1-2 は ON)' + MARK_NOTE]
    assert not mn['1-2'] and not mn['1-3'] and not mn['20261103-080000']
    assert mo['1-1'] == [] and len(mo['1-2']) == len(mo['1-3']) == 1
    assert count_line(tn) == ['4 本・計画と違う 1 本'] and count_line(to) == ['4 本・計画と違う 2 本']
    # 行が「回の本は X から」で名指しした本(なしの区切りの回の本)= 印の付いた本
    named = re.search(r'回の本は (\S+?)\(No', line[0]).group(1)
    assert [k for k, v in mn.items() if v] == [named]


def test_runs_first_off_with_memo_trial_and_round4(tmp_path, day1103):
    """OFF の本の memo に「試し」と書いた日(手順書 7 章「気づいたら」)・回 4 の切れ目でも、印は OFF の本(反証 B の E1 と同じ形)"""
    recs = [('3-7', G), ('3-8', G), ('4-1', OFF), ('4-2', G), ('4-3', G), ('4-4', G)]
    raw = tmp_path / 'raw_r4'
    _write_day(raw, recs)
    lj = raw / '20261103-082000' / 'log.json'
    j = json.loads(lj.read_text(encoding='utf-8'))
    j['description'] += ' memo=試し(起き直しのまま)'
    lj.write_text(json.dumps(j, ensure_ascii=False), encoding='utf-8')
    tn, rn = _runs(RUNS, raw, 'runs81_r4')
    mn = opt_marks(rn)
    assert [k for k, v in mn.items() if v] == ['4-1'] and mn['4-1'][0].startswith('回 4 の試験の選択肢が 4-2(No.4)と違う: ')
    assert count_line(tn) == ['6 本・計画と違う 1 本']


SAME_AS_BEFORE = {
    'all_on': [('1-1', G), ('1-2', G), ('1-3', G), ('2-1', G), ('2-2', G)],
    'reboot_mid_round': [('1-1', G), ('1-2', G), ('1-3', OFF), ('1-4', OFF)],                    # 反証の C(回の途中で起き直し)
    'reboot_and_reentered': [('1-1', G), ('1-2', OFF), ('1-3', G), ('1-4', G)],
    'deliberate_off_from_2_3': [('2-1', G), ('2-2', G), ('2-3', OFF), ('2-4', OFF), ('2-5', OFF), ('2-6', OFF), ('3-1', OFF), ('3-2', OFF)],
    'round2_all_off': [('1-1', G), ('1-2', G), ('2-1', OFF), ('2-2', OFF), ('2-3', OFF)],          # 反証 A の付記 F(前からの形)
    'no_options': [('1-1', OFF), ('1-2', OFF), ('2-1', OFF)],
    'dash_morning_then_on': [(None, OFF), ('1-1', G), ('1-2', G), (None, OFF), ('1-3', G)],
    'nokick_ties_first': [('1-1', NK), ('1-2', {'rec_back_guard': 1}), ('1-3', NK)],              # 同じ数の別の形(先の本を相手に)
    'guard_only_forgot_keep_mid': [('1-1', G), ('1-2', {'rec_back_guard': 1}), ('1-3', G)],
}


@pytest.mark.parametrize('name', list(SAME_AS_BEFORE))
def test_runs_same_as_before_when_the_first_run_has_the_most(tmp_path, day1103, name):
    """回の最初の本の ON がいちばん多い日(手順書どおり・回の途中から OFF・入れ直し・わざと OFF・回の全部が OFF・どれも OFF・本「—」・
    同じ数の別の形): 標準出力・JSON が直す前とバイト単位で同じ"""
    (tn, rn), (to, ro) = both_runs(tmp_path, SAME_AS_BEFORE[name], name)
    assert tn == to and rn == ro


def _ref(rs):
    return max(rs, key=lambda r: sum(1 for k in ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick') if r['options'].get(k) == 1))


def test_runs_random_days_mark_the_runs_short_of_the_round_plan(tmp_path, day1103):
    """でたらめな日 120 本(種 81。回ごとに ON の形・途中の起き直し・入れ忘れ・入れ直し・本「—」・補助なし): 試験の選択肢の印が付くのは
    「その回で ON がいちばん多い最初の本」と形が違う本だけで、その本を名指しする。印の数 =「計画と違う N 本」の N(ほかの印は無い)。
    回の最初の本がいちばん多い日は直す前とバイト単位で同じ。ON がいちばん多い本(と同じ形)には印が付かない"""
    rng = random.Random(81)
    shapes = [G, G, G, OFF, {'rec_back_guard': 1}, dict(G, **NK)]
    n_same = n_moved = 0
    for i in range(120):
        recs = []
        for r in rng.sample(sorted(ROUNDS), rng.randint(1, 3)):
            plan = rng.choice([G, G, OFF, dict(G, **NK)])
            for k in range(1, rng.randint(2, ROUNDS[r]) + 1):
                if rng.random() < 0.15:
                    recs.append((None, rng.choice([OFF, G])))
                recs.append(('%d-%d' % (r, k), plan if rng.random() < 0.7 else rng.choice(shapes)))
        (tn, rn), (to, ro) = both_runs(tmp_path, recs, 'f%d' % i)
        rounds = {}
        for r in rn:
            if r['run_id']:
                rounds.setdefault(r['run_id'].split('-')[0], []).append(r)
        firsts_most = True
        for rnd, rs in rounds.items():
            ref = _ref(rs)
            firsts_most &= ref is rs[0]
            on = lambda x: tuple(k for k in ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick') if x['options'].get(k) == 1)  # noqa: E731
            for r in rs:
                marks = [x for x in r['diffs'] if '試験の選択肢が' in x]
                if on(r) == on(ref):
                    assert not marks, (i, r['run_id'])
                else:
                    assert len(marks) == 1 and marks[0].startswith('回 %s の試験の選択肢が %s(No.%s)と違う: ' % (
                        rnd, ref['run_id'], ref['ai_no'])) and marks[0].endswith(MARK_NOTE), (i, r['run_id'], marks)
        assert all(r['diffs'] == [x for x in r['diffs'] if '試験の選択肢が' in x] for r in rn)      # 計画どおりの値なので印はこれだけ
        assert count_line(tn) == ['%d 本・計画と違う %d 本' % (len(rn), sum(1 for r in rn if r['diffs']))]
        assert options_line(tn) == options_line(to)
        if firsts_most:
            assert tn == to and rn == ro, i
            n_same += 1
        else:
            n_moved += 1
            assert opt_marks(rn) != opt_marks(ro), i
    assert n_same >= 30 and n_moved >= 30                                   # どちらの形も十分に含む


# ================================================================== stale-status-after-drive-tab-return(app.js)
def sim1(tmp_path, c, steps, app=None, name='in.json', server=None, **extra):
    """1 つの画面(app_sim_fix7_r2.js)。test_ui_fix7_r2.run_sim と同じ入力"""
    return T72.run_sim(tmp_path, c, steps, app=app or APP, name=name, server=server if server is not None else MORNING_ON, **extra)


def tagged(res):
    return [x['value'] for x in res if x['op'] in ('el', 'eval')]


def starts(res):
    return [x for x in res if x['op'] == 'start']


def vals(res, op):
    return [x.get('value') for x in res if x['op'] == op]


PRE = [['tab', 'drive'], ['tick']]                     # 走行タブで同期した
AWAY = [['tab', 'live'], ['tick']]                     # ライブのタブへ(走行タブの輪は止まる)
ST_EVAL = ['eval', 'JSON.stringify([state.drive.paramsLoaded, !!state.drive.lostAt])']


@needs_node
def test_return_with_slow_list_refuses_until_status_is_reread(tmp_path, cands):
    """反証 A(1 つの画面): 見張り・残す ON → ライブのタブ → curl で見張り OFF(手順書 3 章の API)→ 一覧の GET が返らないうちに走行タブ
    → すぐブロック 1-2 → ▶。今は表示の問い合わせを読み直すまでブロックも ▶ も「サーバーの設定を読み込んでいます」で断り、走行開始を
    送らない(通信は一覧の GET だけ)。一覧が返って同期したら、確認の文に見張り・残すの行は無く(サーバーは OFF)、走った値と同じ。
    直す前は古い status のまま「見張り: ON」「残す: ON」の確認の文で走った(サーバーは 0・0)"""
    c, di = cands
    steps = PRE + AWAY + [['api', {'rec_back_guard': 0}], ['hold', MD, True], ['calls'], ['tab', 'drive'], ST_EVAL,
                          ['pick', di, blk(c, di, 2)], ['el', '#toast', 'text'], ['start'], ['el', '#toast', 'text'], ['calls'],
                          ['hold', MD, False], ['release', MD], ST_EVAL, ['pick', di, blk(c, di, 2)], ['start'], ['read', 'server']]
    new, tn = sim1(tmp_path, c, steps, name='n.json')
    old, to = sim1(tmp_path, c, steps, app=BEFORE / 'app.js', name='o.json')
    n = tagged(new)
    assert n[:3] == ['[false,false]', LOADING, LOADING]
    s0, s1 = starts(new)
    assert s0['sent'] is None and s0['confirm'] == '' and vals(new, 'calls')[1] == [['GET', MD, None]]
    assert n[3] == '[true,false]' and s1['status'] == 200 and GUARD_LINE not in s1['confirm'] and KEEP_LINE not in s1['confirm']
    srv_new = vals(new, 'read')[-1]
    assert srv_new['rec_back_guard'] == 0 and srv_new['rec_hold_keep'] == 0
    o0 = starts(old)[0]                                                   # 直す前: 指摘が起きる
    assert tagged(old)[0] == '[true,false]' and o0['status'] == 200 and GUARD_LINE in o0['confirm'] and KEEP_LINE in o0['confirm']


@needs_node
def test_return_after_curl_left_42_confirm_reads_the_server(tmp_path, cands):
    """反証 B: ライブのタブの間に curl で左 4.2・launch_us 200(手順書 5 章)→ 一覧が返らないうちに走行タブ → ブロック 2-5 → 天井 ON → ▶。
    今は ▶ を断り、同期したあとの ▶ の確認の文は「左旋回中 4.2 m/s」(走った値)。直す前は「4.1」で走った(走った値は 4.2)"""
    c, di = cands
    flow = [['pick', di, blk(c, di, 10)], ['check', '#dr-speed-ceiling', True], ['start']]
    steps = PRE + AWAY + [['api', {'ceil_left_mps': 4.2, 'launch_us': 200}], ['hold', MD, True], ['tab', 'drive']] + flow + [
        ['el', '#toast', 'text'], ['hold', MD, False], ['release', MD]] + flow + [['read', 'server']]
    new, _ = sim1(tmp_path, c, steps, server=server_params(), name='n.json')
    old, _ = sim1(tmp_path, c, steps, server=server_params(), app=BEFORE / 'app.js', name='o.json')
    s0, s1 = starts(new)
    assert s0['sent'] is None and tagged(new)[0] == LOADING
    assert '速さの天井: ON(全体 4.5・左旋回中 4.2 m/s' in s1['confirm'] and s1['status'] == 200
    assert vals(new, 'read')[-1]['ceil_left_mps'] == 4.2 and vals(new, 'read')[-1]['speed_ceiling'] == 1
    o0 = starts(old)[0]
    assert o0['status'] == 200 and '左旋回中 4.1 m/s' in o0['confirm']                     # 直す前: 指摘が起きる


@needs_node
def test_reply_to_a_status_sent_before_leaving_is_not_a_reread(tmp_path, cands):
    """走行タブを離れる前に出した表示の問い合わせの返事が、戻った直後(一覧の GET を待つ間)に届いた: 表示はするが「読み直した」に数え
    ない(▶ は断ったまま)。その返事はサーバーが変わる前の値(見張り ON)。戻ったあとの問い合わせが通ってから ▶ が通り、確認の文は
    今の値。直す前はその古い返事のまま走った"""
    c, di = cands
    steps = PRE + [['gstart'], ['tab', 'live'], ['api', {'rec_back_guard': 0}], ['hold', MD, True], ['tab', 'drive'], ['gend'], ST_EVAL,
                   ['el', '#dr-rec-back-guard', 'checked'], ['start'], ['el', '#toast', 'text'], ['hold', MD, False], ['release', MD],
                   ST_EVAL, ['start'], ['read', 'server']]
    new, _ = sim1(tmp_path, c, steps, name='n.json')
    old, _ = sim1(tmp_path, c, steps, app=BEFORE / 'app.js', name='o.json')
    n = tagged(new)
    assert n[0] == '[false,false]' and n[1] is True and n[2] == LOADING                    # 古い返事は表示だけ(スイッチは ON のまま)
    s0, s1 = starts(new)
    assert s0['sent'] is None and n[3] == '[true,false]' and s1['status'] == 200 and GUARD_LINE not in s1['confirm']
    o0 = starts(old)[0]
    assert tagged(old)[0] == '[true,false]' and o0['status'] == 200 and GUARD_LINE in o0['confirm']


SAME_FLOWS = {
    # 反証 A の比べ: 一覧がすぐ返る・返事のあと ▶・5 秒たって(drLost → 同期)から ▶・走行タブのまま
    'A0_list_at_once': lambda c, di: (PRE + AWAY + [['api', {'rec_back_guard': 0}], ['tab', 'drive'], ST_EVAL, ['pick', di, blk(c, di, 2)],
                                                   ['start'], ['read', 'server']], MORNING_ON),
    'A5_after_5s': lambda c, di: (PRE + AWAY + [['api', {'rec_back_guard': 0}], ['hold', MD, True], ['tab', 'drive'], ['tick'], ST_EVAL,
                                                ['pick', di, blk(c, di, 2)], ['start'], ['read', 'server']], MORNING_ON),
    'B0_ceil_at_once': lambda c, di: (PRE + AWAY + [['api', {'ceil_left_mps': 4.2, 'launch_us': 200}], ['tab', 'drive'],
                                                    ['pick', di, blk(c, di, 10)], ['check', '#dr-speed-ceiling', True], ['start'],
                                                    ['read', 'server']], server_params()),
    'C_stay_on_drive': lambda c, di: (PRE + [['api', {'rec_back_guard': 0}], ['tick'], ST_EVAL, ['start'], ['read', 'server']], MORNING_ON),
    'reclick_drive_tab': lambda c, di: (PRE + [['tab', 'drive'], ST_EVAL, ['start'], ['tick'], ['read', 'server']], MORNING_ON),
    'models_fail_on_return': lambda c, di: (PRE + AWAY + [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ST_EVAL,
                                                          ['start'], ['tick'], ['tick'], ['read', 'server']], MORNING_ON),
    'down_on_return': lambda c, di: (PRE + AWAY + [['down_all', True], ['tab', 'drive'], ['tick'], ST_EVAL, ['start'],
                                                   ['el', '#toast', 'text'], ['down_all', False], ['tick'], ST_EVAL, ['start']], MORNING_ON),
    'first_open_no_init': lambda c, di: ([['tab', 'drive'], ST_EVAL, ['start'], ['tick'], ['start']], MORNING_ON),
}


@needs_node
@pytest.mark.parametrize('name', list(SAME_FLOWS))
def test_flows_without_the_window_exactly_as_before(tmp_path, cands, name):
    """戻った直後に押さない・一覧がすぐ返る・5 秒たってから・走行タブのまま・走行タブをもう一度押す・戻ったときに一覧が失敗する・
    落ちている・初めて開いた画面: 通信の全部・タイマー・トースト・文・送った物・確認の文が直す前とまったく同じ"""
    c, di = cands
    steps, sp = SAME_FLOWS[name](c, di)
    extra = {'no_init': True} if name == 'first_open_no_init' else {}
    steps = steps + [['calls'], ['ntimers'], ['posts']]
    new, tn = sim1(tmp_path, c, steps, server=sp, name='n.json', **extra)
    old, to = sim1(tmp_path, c, steps, server=sp, app=BEFORE / 'app.js', name='o.json', **extra)
    assert tn == to and tagged(new) == tagged(old)
    assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers') and vals(new, 'posts') == vals(old, 'posts')
    assert [(s['sent'], s['confirm'], s['boot_at'], s['status']) for s in starts(new)] == [
        (s['sent'], s['confirm'], s['boot_at'], s['status']) for s in starts(old)]


@needs_node
@pytest.mark.parametrize('who', ['other_page', 'this_page'])
def test_running_page_returns_to_the_drive_tab_as_before(tmp_path, cands, who):
    """走行中(ほかの画面が走らせている・この画面が走らせた = token)に走行タブへ戻る: 輪は走行タブの外でも回っているので paramsLoaded は
    下ろさない(前と同じ)。止める操作・通信・タイマーも直す前とまったく同じ"""
    c, di = cands
    run = [['running', True]] if who == 'other_page' else [['start']]
    steps = PRE + run + [['tab', 'live'], ['tick'], ['hold', MD, True], ['tab', 'drive'], ST_EVAL, ['el', '#btn-dr-start', 'disabled'],
                         ['click', '#btn-dr-stop'], ['calls'], ['ntimers'], ['posts']]
    new, tn = sim1(tmp_path, c, steps, name='n.json')
    old, to = sim1(tmp_path, c, steps, app=BEFORE / 'app.js', name='o.json')
    assert tagged(new)[0] == '[true,false]' and tagged(new) == tagged(old) and tn == to
    assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers') and vals(new, 'posts') == vals(old, 'posts')


@needs_node
def test_all_earlier_flows_exactly_as_before(tmp_path, cands):
    """前からの流れ(10/26〜27 の流れ・6-1・6-2・7-1 回目の場面。test_ui_fix7_r2 の 30 通り): 通信の全部・タイマー・トースト・文・送った物・
    確認の文・boot_at が直す前の写しとまったく同じ = スイッチが全部 OFF のとき(と ON のときも)送る物はビット単位で同じ"""
    c, di = cands
    for i, (key, (st, extra, sp)) in enumerate(T72._earlier_flows(c, di).items()):
        st = st + [['calls'], ['ntimers']]
        new, tn = sim1(tmp_path, c, st, name='n%d.json' % i, server=sp, **extra)
        old, to = sim1(tmp_path, c, st, app=BEFORE / 'app.js', name='o%d.json' % i, server=sp, **extra)
        assert tn == to and tagged(new) == tagged(old), key
        assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers'), key
        assert vals(new, 'posts') == vals(old, 'posts') and T72.sent(new) == T72.sent(old), key


@needs_node
def test_random_mornings_exactly_as_before(tmp_path, cands):
    """でたらめな朝の流れ 30 本(test_ui_fix7_r2 と同じ種 72。起き直し・落ちる・途切れ・返事だけ届かない・ライブのタブ → 一覧の失敗 →
    走行タブ を混ぜる。一覧の GET は止めない): 送った物・スイッチを押したか・トースト・通信が直す前とまったく同じ"""
    c, di = cands
    rng = random.Random(72)
    for i in range(30):
        st = T72._fuzz_flow(rng, c, di) + [['calls']]
        new, tn = sim1(tmp_path, c, st, server=server_params(throttle_gain=0.9), name='fn.json')
        old, to = sim1(tmp_path, c, st, server=server_params(throttle_gain=0.9), app=BEFORE / 'app.js', name='fo.json')
        assert vals(new, 'posts') == vals(old, 'posts') and T72.sent(new) == T72.sent(old), i
        assert [x for x in new if x['op'] == 'press_if_off'] == [x for x in old if x['op'] == 'press_if_off'], i
        assert tn == to and vals(new, 'calls') == vals(old, 'calls'), i


# ------------------------------------------------------------------ 2 つの画面(app_sim_fix8.js)
def sim2(tmp_path, c, steps, pages, app=None, name='in2.json', server=None):
    inp = tmp_path / name
    d = {'bench': c, 'server_params': server if server is not None else server_params(), 'steps': steps, 'server_client_version': 9,
         'confirm': True, 'boot_at': 1000.0, 'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}),
         'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True, 'boot_refusal_texts': T7.REAL, 'pages': pages}
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM2), str(app or APP), str(STATIC / 'index.html'), str(inp)], capture_output=True, text=True,
                       timeout=180, env=dict(os.environ, TZ='Asia/Tokyo'))
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def two_page_case(mid, hold=True):
    """反証の p_two_pages.py の流れ: A・B が走行タブ → A が基準の設定・見張り・残す ON → B がライブのタブ → mid → (一覧の GET を止めて)
    B が走行タブに戻ってすぐ ▶ → (一覧を返す)→ 1 秒 → B がもう一度 ▶"""
    return ([['@', 'A', ['tab', 'drive']], ['@', 'B', ['tab', 'drive']], ['tickall'], ['tickall'], ['@', 'A', ['baseline']],
             ['@', 'A', ['check', '#dr-rec-back-guard', True]], ['@', 'A', ['check', '#dr-rec-hold-keep', True]], ['tickall'],
             ['@', 'B', ['tab', 'live']], ['tickall'], ['tickall']] + mid + ([['hold', MD, True]] if hold else []) +
            [['@', 'B', ['tab', 'drive']], ['@', 'B', ['start']], ['server']] + ([['release', MD]] if hold else []) +
            [['tickall'], ['@', 'B', ['start']], ['server'], ['@', 'B', ['toasts']]])


def confirm_matches_server(st):
    """走り出した ▶ の確認の文の見張り・残す・天井(と左)が、サーバーが走った値と同じか(反証の p_fuzz_invariant.py の照らしの一部)"""
    c, ss = st['confirm'], st.get('server_at_start') or {}
    m = re.search(r'速さの天井: ON\(全体 ([\d.]+)・左旋回中 ([\d.]+) m/s', c)
    return ((GUARD_LINE in c) == (ss.get('rec_back_guard') == 1) and (KEEP_LINE in c) == (ss.get('rec_hold_keep') == 1)
            and bool(m) == (ss.get('speed_ceiling') == 1) and (not m or abs(float(m.group(2)) - float(ss.get('ceil_left_mps'))) < 1e-6))


TWO = {
    # 反証の 1: A が 3 章の「止めて OFF」(見張りを OFF = 残すもサーバーが OFF)をした間に、走行タブを離れていた B が戻ってすぐ ▶
    'guard_off_by_A': ([['@', 'A', ['check', '#dr-rec-back-guard', False]], ['tickall']], {'rec_back_guard': 0, 'rec_hold_keep': 0}),
    # 反証の 3: curl で左 4.2・launch_us 200(手順書 5 章)のあと、B が戻ってすぐ天井 ON の ▶(スイッチの表示だけ ON にして押す)
    'curl_left_42': ([['api', {'ceil_left_mps': 4.2, 'launch_us': 200}],
                      ['@', 'B', ['eval', "document.querySelector('#dr-speed-ceiling').checked = true; 1"]]], {'ceil_left_mps': 4.2}),
}


@needs_node
@pytest.mark.parametrize('name', list(TWO))
def test_two_pages_return_refuses_until_reread(tmp_path, cands, name):
    """2 つの画面(反証の 1・3): B が戻ってすぐの ▶ は「サーバーの設定を読み込んでいます」で断る(走行開始を送らない)。一覧が返って 1 秒
    のあとの ▶ は、確認の文が走った値と同じ。直す前は戻ってすぐの ▶ が古い確認の文で走った"""
    c, di = cands
    mid, srv_now = TWO[name]
    steps = two_page_case(mid)
    new = sim2(tmp_path, c, steps, ['A', 'B'], name='n.json')
    old = sim2(tmp_path, c, steps, ['A', 'B'], app=BEFORE / 'app.js', name='o.json')
    n0, n1 = [s for s in new['steps'] if s['op'] == 'start']
    assert n0['sent'] is None and n0['confirm'] == '' and LOADING in new['toasts']['B']
    assert n1['status'] == 200 and confirm_matches_server(n1)
    assert all(n1['server_at_start'][k] == v for k, v in srv_now.items())
    o0 = [s for s in old['steps'] if s['op'] == 'start'][0]
    assert o0['status'] == 200 and not confirm_matches_server(o0)                         # 直す前: 指摘が起きる


@needs_node
def test_two_pages_list_at_once_exactly_as_before(tmp_path, cands):
    """2 つの画面で一覧がすぐ返る(反証の 2): B のトースト・送った物・確認の文が直す前と同じ"""
    c, di = cands
    steps = two_page_case(TWO['guard_off_by_A'][0], hold=False)
    new = sim2(tmp_path, c, steps, ['A', 'B'], name='n.json')
    old = sim2(tmp_path, c, steps, ['A', 'B'], app=BEFORE / 'app.js', name='o.json')
    assert new['toasts'] == old['toasts']
    assert [(s.get('sent'), s.get('confirm'), s.get('status')) for s in new['steps'] if s['op'] == 'start'] == [
        (s.get('sent'), s.get('confirm'), s.get('status')) for s in old['steps'] if s['op'] == 'start']


def _fuzz_two(rng, c, di, two):
    """反証の p_fuzz_invariant.py の混ぜ方に「走行タブを離れている間にサーバーが変わり、一覧の GET が返らないうちに戻ってすぐ ▶」を足した"""
    pages = ['A', 'B'] if two else ['A']
    steps = [['@', p, ['tab', 'drive']] for p in pages] + [['tickall'], ['tickall']]
    boot = 1000.0
    for _ in range(rng.randint(12, 24)):
        p = rng.choice(pages)
        k = rng.random()
        if k < 0.06:
            boot += 1000.0
            steps += [['reboot', boot, T72.RESET], ['tickall']]
        elif k < 0.12:
            steps += [['api', rng.choice([{'ceil_left_mps': 4.2}, {'ceil_left_mps': 4.1}, {'rec_back_guard': 1},
                                          {'rec_back_guard': 1, 'rec_hold_keep': 1}, {'rec_back_guard': 0}])], ['tickall']]
        elif k < 0.30:                                     # 戻ってすぐ ▶(この直しの場面)
            other = [x for x in pages if x != p]
            change = rng.choice([[['api', {'rec_back_guard': 0}]], [['api', {'rec_back_guard': 1, 'rec_hold_keep': 1}]],
                                 [['api', {'ceil_left_mps': 4.2}]]] + ([[['@', other[0], ['check', '#dr-rec-back-guard', False]]]] if other else []))
            steps += [['@', p, ['tab', 'live']], ['tickall']] + change + [['hold', MD, True], ['@', p, ['tab', 'drive']]]
            steps += rng.choice([[['@', p, ['start']]], [['@', p, ['pick', di, blk(c, di, rng.randint(1, 23))]], ['@', p, ['start']]],
                                 [['@', p, ['check', '#dr-speed-ceiling', True]], ['@', p, ['start']]]])
            steps += [['hold', MD, False], ['release', MD], ['tickall']]
        elif k < 0.42:
            sel = rng.choice(['#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-speed-ceiling'])
            steps += [['@', p, ['check', sel, rng.random() < 0.7]], ['tickall']]
        elif k < 0.50:
            steps += [['@', p, ['baseline']], ['tickall']]
        elif k < 0.60:
            steps += [['@', p, ['pick', di, blk(c, di, rng.randint(1, 23))]], ['tickall']]
        elif k < 0.85:
            steps += [['@', p, ['start']], ['tickall']]
        else:
            steps += [['@', p, ['stop']], ['tickall']]
    return steps


@needs_node
def test_random_mornings_with_tab_returns_confirm_always_matches(tmp_path, cands):
    """でたらめな朝 16 本(種 81。1 画面と 2 画面。起き直し・curl・スイッチ・基準の設定・ブロック・▶・■ に「離れている間にサーバーが
    変わり、一覧の GET が返らないうちに戻ってすぐ ▶」を混ぜる): 今の版は走り出した ▶ の全部で、確認の文の見張り・残す・天井と左が
    サーバーの走った値と同じ。直す前の写しは同じ流れで違う ▶ がある(指摘)"""
    c, di = cands
    rng = random.Random(81)
    n_ok = n_bad_old = 0
    for i in range(16):
        two = i % 2 == 1
        steps = _fuzz_two(rng, c, di, two)
        pages = ['A', 'B'] if two else ['A']
        new = sim2(tmp_path, c, steps, pages, name='fn.json', server=server_params(throttle_gain=0.9))
        old = sim2(tmp_path, c, steps, pages, app=BEFORE / 'app.js', name='fo.json', server=server_params(throttle_gain=0.9))
        for s in new['steps']:
            if s['op'] == 'start' and s.get('status') == 200:
                assert confirm_matches_server(s), (i, s['confirm'], s['server_at_start'])
                n_ok += 1
        n_bad_old += sum(1 for s in old['steps'] if s['op'] == 'start' and s.get('status') == 200 and not confirm_matches_server(s))
    assert n_ok >= 20 and n_bad_old >= 1


# ================================================================== toast-half-width-on-phone(style.css)
def test_toast_rule_takes_its_text_width_up_to_90_percent():
    """.toast は文の長さの幅(max-content)を画面の 90%(大きな画面は 960 px)まで。left 50%・translateX(-50%)・下 20 px・
    pointer-events none・走行中の bottom 96 px は前と同じ"""
    css = CSS.read_text(encoding='utf-8')
    rule = re.search(r'^\.toast \{([^}]*)\}', css, re.M).group(1)
    decl = [x.strip() for x in rule.split(';') if x.strip()]
    assert 'width: max-content' in decl and decl.index('max-width: 90%') < decl.index('max-width: min(90%, 960px)')
    assert {'position: fixed', 'bottom: 20px', 'left: 50%', 'transform: translateX(-50%)', 'pointer-events: none'} <= set(decl)
    assert 'right' not in rule and 'body.driving .toast { bottom: 96px; }' in css


FIREFOX = os.environ.get('JF_UI_FIREFOX') == '1'
GECKO = shutil.which('geckodriver') or ('/snap/bin/geckodriver' if os.path.exists('/snap/bin/geckodriver') else None)
REFUSAL = None


def _refusal():
    from jetformer.web import server as srv
    os.environ['TZ'] = 'Asia/Tokyo'
    time.tzset()
    return '走行は始まっていません(停止中)。' + srv._boot_refusal({'boot_at': 1793116404.711}, 1793116404.711 + 3600)


MEASURE = r'''
const [txt] = arguments;
document.documentElement.style.scrollbarWidth = 'none';     // スマホの重ねるスクロールバー(幅を取らない)と同じに
document.querySelectorAll('.tab-page').forEach((p) => p.classList.toggle('active', p.id === 'tab-drive'));
document.querySelector('#btn-dr-start').scrollIntoView({block: 'center'});
const t = document.querySelector('#toast');
t.textContent = txt; t.className = 'toast error';
const r = (e) => { const b = e.getBoundingClientRect(); return [Math.round(b.left), Math.round(b.top), Math.round(b.width), Math.round(b.height)]; };
return {vw: document.documentElement.clientWidth, toast: r(t), phase: r(document.querySelector('#dr-phase')), msg: r(document.querySelector('#dr-message'))};
'''


def _overlap(t, e):
    return t[1] < e[1] + max(e[3], 1) and e[1] < t[1] + t[3] and t[0] < e[0] + e[2] and e[0] < t[0] + t[2]


@pytest.mark.skipif(not (FIREFOX and GECKO), reason='JF_UI_FIREFOX=1 と geckodriver のときだけ(jf-heavy --lock big --mem 3000M)')
def test_firefox_toast_width_on_phone(tmp_path):
    """headless の Firefox(本物の index.html に style.css を file:// で。app.js は読まない = 通信なし。見える高さ 860 px = 反証の係の
    測り方。スクロールバーは幅を取らない = スマホと同じ)で、▶ を画面の真ん中にして、
    ▶ の断りの出し直し(本物の server._boot_refusal の文)と短い文のトーストを測る。直す前の style.css は幅 420 で箱の幅が画面の半分・
    「停止中」のバッジか走行タブの文に重なる(指摘)。今の style.css は 360・390・420 で幅が画面の 85% 以上・画面の中・重ならない。
    短い文と 1920 px の画面は直す前と同じ大きさ"""
    import test_vbat_ui as V
    if str(tmp_path).startswith('/tmp') or '/.' in str(tmp_path):
        pytest.skip('snap の Firefox が読めない場所(--basetemp を /tmp・隠しフォルダの外に)')
    text = _refusal()
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    html = re.sub(r'<script src="/static/app.js"></script>', '', html)
    pages = {}
    for tag, css in (('before', BEFORE / 'style.css'), ('now', CSS)):
        cp = tmp_path / ('%s.css' % tag)
        cp.write_bytes(css.read_bytes())
        (tmp_path / ('%s.html' % tag)).write_text(html.replace('href="/static/style.css"', 'href="%s"' % cp.name), encoding='utf-8')
        for w in (360, 390, 420, 1920):
            p = tmp_path / ('%s_%d.html' % (tag, w))
            p.write_text('<!doctype html><meta charset="utf-8"><body style="margin:0"><iframe src="%s.html" style="width:%dpx;height:860px;'
                         'border:0;display:block"></iframe>' % (tag, w), encoding='utf-8')
            pages[(tag, w)] = p
    port = V._free_port()
    gd = subprocess.Popen([GECKO, '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wd = V._WD(port)
    snap_common = os.path.expanduser('~/snap/firefox/common')
    profile = tempfile.mkdtemp(prefix='ui-fix8-profile-', dir=snap_common if os.path.isdir(snap_common) else str(tmp_path))
    res = {}
    try:
        assert V._wait(lambda: wd.req('GET', '/status', timeout=3))
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless', '-profile', profile]}}}}
        wd.sid = wd.req('POST', '/session', caps, timeout=180)['sessionId']
        wd.req('POST', '/session/%s/window/rect' % wd.sid, {'width': 2000, 'height': 900})
        for (tag, w), p in pages.items():
            wd.req('POST', '/session/%s/url' % wd.sid, {'url': 'file://%s' % p})
            wd.req('POST', '/session/%s/frame' % wd.sid, {'id': 0})
            for name, txt in (('long', text), ('short', '停止しました')):
                res[(tag, w, name)] = wd.req('POST', '/session/%s/execute/sync' % wd.sid, {'script': MEASURE, 'args': [txt]})
            wd.req('POST', '/session/%s/frame' % wd.sid, {'id': None})
    finally:
        try:
            wd.req('DELETE', '/session/%s' % wd.sid)
        except Exception:      # noqa: BLE001
            pass
        gd.terminate()
        gd.wait(timeout=30)
        shutil.rmtree(profile, ignore_errors=True)
    b = res[('before', 420, 'long')]
    assert abs(b['toast'][2] - b['vw'] / 2) <= 6 and (_overlap(b['toast'], b['phase']) or _overlap(b['toast'], b['msg']))     # 指摘
    for w in (360, 390, 420):
        m = res[('now', w, 'long')]
        t = m['toast']
        assert t[2] >= 0.85 * m['vw'] and t[0] >= 0 and t[0] + t[2] <= m['vw'], (w, m)
        assert not _overlap(t, m['phase']) and not _overlap(t, m['msg']), (w, m)
        assert t[3] <= 0.6 * res[('before', w, 'long')]['toast'][3], (w, m)          # 高さは直す前の 6 割以下(幅が倍近い)
        assert res[('now', w, 'short')]['toast'] == res[('before', w, 'short')]['toast']
    for name in ('long', 'short'):
        assert res[('now', 1920, name)]['toast'] == res[('before', 1920, name)]['toast']
