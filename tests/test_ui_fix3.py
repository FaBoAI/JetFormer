"""画面(app.js・index.html・server.py)の直し 3 回目(2026-10-26 fix3 係 ui)の試験。

- state-1: 天井を ON にしても、ブロック・基準の設定で OFF にした自動合わせが黙って ON に戻らない
- state-2: 速度ゲインのスライダーは 0.01 刻み(0.92・0.97・1.02・±0.03 が入る)・±0.01 のボタン
- state-5: 画面の版 7(DRIVE_CLIENT_VERSION と app.js の client)。版 6 以下の画面からは走行を始めない
- 試験の選択肢のスイッチ: rec_back_guard・rec_hold_keep・launch_no_kick(既定 OFF・走行中は ON にできない・基準の設定と
  ブロックでは変えない・走行開始では送らない・確認の文に ON の行・古いサーバーでは出さない)
- state2-1(fix2): トリムがブロック・基準の設定・取り消し・スイッチで 0 に戻らない
- 走行開始で送る値が直す前と同じ流れ・違う流れ(直す前の写し tests/fixtures/ui_fix3_before/ と比べる)

本物の app.js をまるごと node の vm で読み込む(tests/fixtures/app_sim_fix3.js。DOM・サーバーは偽物。<input type=range> の値の丸めは
本物の index.html の min・max・step で HTML の決まりどおりに真似る)。ブロックは本物の data/bench_candidates.json(読むだけ)の 11/3。
サーバー(jetformer-web)・車・ブラウザは使わない。node が無ければ飛ばす。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix3.py
"""
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jetformer import drive
from jetformer.web import bench

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
# JF_UI_FIX3_APP・JF_UI_FIX3_INDEX: わざと壊した写しで試験が落ちるかを見るとき(experiments/20261026_fix/fix3/ui/mutate_ui.py)だけ
APP_JS = Path(os.environ.get('JF_UI_FIX3_APP') or STATIC / 'app.js')
INDEX = Path(os.environ.get('JF_UI_FIX3_INDEX') or STATIC / 'index.html')
BEFORE = ROOT / 'tests' / 'fixtures' / 'ui_fix3_before'        # 直す前の写し(2026-10-26 19:47。fix2 の直しのあと)
APP_OLD, INDEX_OLD = BEFORE / 'app.js', BEFORE / 'index.html'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix3.js'
NODE = shutil.which('node')
OPT = ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick')
OPT_SEL = {'rec_back_guard': '#dr-rec-back-guard', 'rec_hold_keep': '#dr-rec-hold-keep', 'launch_no_kick': '#dr-launch-no-kick'}
OPT_LINE = {'rec_back_guard': 'リカバリーの後ろの見張り: ON', 'rec_hold_keep': '中立止めのあともリカバリーを残す: ON',
            'launch_no_kick': '発進の前進を出さない区間は発進の補助も出さない: ON'}
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')


@pytest.fixture(scope='module')
def cands():
    c = bench.load_candidates()
    days = [i for i, d in enumerate(c.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks')]
    if not days:
        pytest.skip('data/bench_candidates.json に 2026-11-03 のブロックがありません')
    return dict(c, today='2026-11-03'), days[0]


def blk(c, di, no):
    """ブロックの番号(1〜23)→ 選ぶ欄の番号"""
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def server_params(**over):
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    sp.update(over)
    return sp


def run_sim(tmp_path, c, steps, server=None, old=False, client_version=7, confirm=True, name='input.json', status_fail_first=0,
            missing_ids=()):
    app, html = (APP_OLD, INDEX_OLD) if old else (APP_JS, INDEX)
    inp = tmp_path / name
    inp.write_text(json.dumps({'bench': c, 'server_params': server if server is not None else server_params(), 'steps': steps,
                               'server_client_version': client_version, 'confirm': confirm, 'status_fail_first': status_fail_first,
                               'missing_ids': list(missing_ids)},
                              ensure_ascii=False, default=str),
                   encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app), str(html), str(inp)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def starts(res):
    return [x for x in res if x['op'] == 'start']


def reads(res, what):
    return [x['value'] for x in res if x['op'] == 'read' and x['what'] == what]


def els(res):
    return [x['value'] for x in res if x['op'] == 'el']


def posts(res):
    return [x['value'] for x in res if x['op'] == 'posts']


def params_posts(p):
    return [b for path, b, *_ in p if path == '/api/drive/params']


def test_before_copies_are_the_files_before_fix3():
    """比べる相手(直す前の写し)は 19:47 の app.js・index.html(fix_round2.md の後の sha256 と同じ)"""
    sums = dict(reversed(line.split()) for line in (BEFORE / 'SHA256SUMS').read_text().splitlines())
    for name in ('app.js', 'index.html'):
        assert hashlib.sha256((BEFORE / name).read_bytes()).hexdigest() == sums[name], name
    assert sums['app.js'].startswith('0b0057bd551d3192')            # fix_round2.md 1 章の app.js の後


def test_node_check_and_html_ids():
    if NODE:
        r = subprocess.run([NODE, '--check', str(APP_JS)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    html = INDEX.read_text(encoding='utf-8')
    ids = re.findall(r'\bid="([^"]+)"', html)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert dup == [], dup


# ---------------------------------------------------------------- state-1 天井を ON にしても自動合わせが戻らない
@needs_node
def test_state1_ceiling_on_keeps_block_auto_off(tmp_path, cands):
    """前の本(2-4)が天井 + 自動合わせ ON → 2-6(ブロック 11。天井だけ)を選ぶ → 天井 ON → 表示の同期 2 回 → 走行開始"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 11)], ['read', 'pending'], ['posts'], ['check', '#dr-speed-ceiling', True], ['posts'],
             ['poll'], ['read', 'auto'], ['poll'], ['read', 'auto'], ['read', 'server'], ['start']]
    res, _ = run_sim(tmp_path, c, steps, server=server_params(speed_ceiling=1, speed_auto=1))
    assert {'speed_ceiling', 'speed_auto'} <= set(reads(res, 'pending')[0])
    assert params_posts(posts(res)[1]) == [{'speed_ceiling': 1, 'speed_auto': 0}]     # 天井といっしょに自動合わせ OFF を送る
    assert reads(res, 'auto') == [False, False]                                     # 約 1 秒後も OFF のまま
    assert reads(res, 'server')[0]['speed_auto'] == 0
    s = starts(res)[0]
    assert (s['sent']['speed_ceiling'], s['sent']['speed_auto']) == (1, 0)
    assert '\n速さの天井: ON' in s['confirm'] and '自動合わせ: ON' not in s['confirm']
    # 対照: 直す前の app.js では、表示の同期で自動合わせが ON に戻り、走行開始で 1 を送る(反証 1 回目 state-1 の再現)
    res0, _ = run_sim(tmp_path, c, steps, server=server_params(speed_ceiling=1, speed_auto=1), old=True, client_version=6, name='old.json')
    assert params_posts(posts(res0)[1]) == [{'speed_ceiling': 1}]
    assert reads(res0, 'auto') == [True, True]
    s0 = starts(res0)[0]
    assert (s0['sent']['speed_ceiling'], s0['sent']['speed_auto']) == (1, 1) and '自動合わせ: ON' in s0['confirm']


@needs_node
def test_state1_auto_blocks_still_turn_auto_on(tmp_path, cands):
    """2-3(ブロック 8。天井 + 自動合わせ): 天井 ON → 自動合わせ ON は今まで通り(別の POST)。前の本の状態に関係なく同じ"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 8)], ['check', '#dr-speed-ceiling', True], ['check', '#dr-speed-auto', True], ['poll'],
             ['read', 'ceil'], ['read', 'auto'], ['posts'], ['start']]
    for srv in (server_params(), server_params(speed_ceiling=1, speed_auto=1), server_params(speed_ceiling=1, speed_auto=0)):
        res, _ = run_sim(tmp_path, c, steps, server=srv)
        assert reads(res, 'ceil') == [True] and reads(res, 'auto') == [True]
        assert params_posts(posts(res)[0]) == [{'speed_ceiling': 1, 'speed_auto': 0}, {'speed_auto': 1}]
        s = starts(res)[0]
        assert (s['sent']['speed_ceiling'], s['sent']['speed_auto']) == (1, 1)
        assert '\n速さの天井: ON' in s['confirm'] and '\n速さのゲインの自動合わせ: ON' in s['confirm']


@needs_node
def test_state1_ceiling_off_and_cancel_unchanged(tmp_path, cands):
    """天井を OFF にする(自動合わせも OFF)・天井の確認でキャンセル は前と同じ"""
    c, di = cands
    res, _ = run_sim(tmp_path, c, [['poll'], ['read', 'ceil'], ['check', '#dr-speed-ceiling', False], ['posts'], ['start']],
                     server=server_params(speed_ceiling=1, speed_auto=1))
    assert reads(res, 'ceil') == [True]
    assert params_posts(posts(res)[0]) == [{'speed_ceiling': 0, 'speed_auto': 0}]
    assert (starts(res)[0]['sent']['speed_ceiling'], starts(res)[0]['sent']['speed_auto']) == (0, 0)
    res, _ = run_sim(tmp_path, c, [['pick', di, blk(c, di, 11)], ['posts'], ['check', '#dr-speed-ceiling', True], ['posts'],
                                   ['read', 'ceil'], ['start']], server=server_params(speed_ceiling=1, speed_auto=1), confirm=False)
    assert params_posts(posts(res)[1]) == [] and reads(res, 'ceil') == [False]
    assert starts(res) == [] or starts(res)[0]['sent'] is None                    # 確認でキャンセルしたので走行も始まらない


# ---------------------------------------------------------------- state-2 速度ゲイン 0.01 刻み
def test_state2_slider_attributes_and_defaults():
    html = INDEX.read_text(encoding='utf-8')
    tag = re.search(r'<input type="range" id="dr-throttle-gain"[^>]*>', html).group(0)
    assert 'min="0"' in tag and 'max="2"' in tag and 'step="0.01"' in tag and 'value="1"' in tag    # 範囲と既定 1 はそのまま
    assert drive.PARAM_LIMITS['throttle_gain'] == (0.0, 2.0)
    old = INDEX_OLD.read_text(encoding='utf-8')
    assert 'step="0.05"' in re.search(r'<input type="range" id="dr-throttle-gain"[^>]*>', old).group(0)
    # 速度ゲインのほかのスライダー(ステア・トリムなど)は変えていない
    ranges = lambda h: {m.group(1): m.group(0) for m in re.finditer(r'<input type="range" id="([^"]+)"[^>]*>', h)}
    now, before = ranges(html), ranges(old)
    assert set(now) == set(before)
    assert {k for k in now if now[k] != before[k]} == {'dr-throttle-gain'}
    for el in ('btn-dr-tg-down', 'btn-dr-tg-up'):
        assert 'id="%s"' % el in html
    js = APP_JS.read_text(encoding='utf-8')
    assert "['#dr-throttle-gain', 'throttle_gain', (v) => (+v).toFixed(2)]" in js        # 表示は 2 桁(0.92)
    assert 'GAIN_STEP = 0.01, GAIN_MIN = 0, GAIN_MAX = 2' in js
    assert re.search(r"throttle_gain: 1\.0\b", re.search(r'^const BENCH_FALLBACK = \{.*?^};\n', js, re.S | re.M).group(0))


@needs_node
def test_state2_gain_step_function():
    js = APP_JS.read_text(encoding='utf-8')
    code = re.search(r'^const GAIN_STEP = .*?\n', js, re.M).group(0) + re.search(r'^function gainStep\(.*?^}\n', js, re.S | re.M).group(0)
    cases = [['0.9', 0.01], ['0.92', 0.01], ['0.92', -0.01], ['1.02', -0.01], ['2', 0.01], ['0', -0.01], ['0.07', 0.01], ['x', 0.01],
             ['1.995', 0.01]]
    r = subprocess.run([NODE, '-e', code + 'process.stdout.write(JSON.stringify(%s.map(([v, d]) => gainStep(v, d))));' % json.dumps(cases)],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == [0.91, 0.93, 0.91, 1.01, 2, 0, 0.08, 1.01, 2]


@needs_node
@pytest.mark.parametrize('v', [0.92, 0.97, 1.02, 0.87, 0.93, 0.85, 1.0])
def test_state2_api_value_survives_start(tmp_path, cands, v):
    """手順書の値を API(curl)で入れる → 表示の同期 → 走行開始で同じ値を送る。直す前は 0.05 刻みに丸めた値を送っていた"""
    c, _ = cands
    steps = [['server_set', {'throttle_gain': v}], ['poll'], ['read', 'gain'], ['read', 'gain_text'], ['start']]
    res, _ = run_sim(tmp_path, c, steps)
    assert reads(res, 'gain') == [v] and reads(res, 'gain_text') == ['%.2f' % v]
    s = starts(res)[0]
    assert s['sent']['throttle_gain'] == v and '速度ゲイン %.2f' % v in s['confirm']
    res0, _ = run_sim(tmp_path, c, steps, old=True, client_version=6, name='old.json')
    snapped = round(round(v / 0.05) * 0.05, 2) if abs(v / 0.05 - round(v / 0.05)) != 0.5 else None
    s0 = starts(res0)[0]
    if snapped is not None:
        assert s0['sent']['throttle_gain'] == snapped                     # 直す前: 0.92 → 0.9 など
    assert reads(res0, 'gain_text') == ['%.2f' % v]                      # 直す前は表示だけ 0.92 で、送るのは丸めた値だった


@needs_node
def test_state2_buttons_move_by_001_and_send(tmp_path, cands):
    """ブロック 1(0.90)→ +0.01 を 2 回 → 0.92 を送る(スライダーを動かしたのと同じ: すぐ送る・pending を消す)→ 走行開始も 0.92。
    −0.01 を 5 回 → 0.87。−0.03 = 3 回"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 1)], ['read', 'gain'], ['posts'], ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'],
             ['posts'], ['read', 'gain'], ['read', 'gain_text'], ['read', 'pending'], ['start'], ['running', False],
             ['click', '#btn-dr-tg-down'], ['click', '#btn-dr-tg-down'], ['click', '#btn-dr-tg-down'], ['click', '#btn-dr-tg-down'],
             ['click', '#btn-dr-tg-down'], ['read', 'gain'], ['start']]
    res, _ = run_sim(tmp_path, c, steps)
    assert reads(res, 'gain')[:2] == [0.9, 0.92] and reads(res, 'gain_text') == ['0.92']
    assert params_posts(posts(res)[1]) == [{'throttle_gain': 0.91}, {'throttle_gain': 0.92}]
    assert 'throttle_gain' not in reads(res, 'pending')[0]
    s1, s2 = starts(res)
    assert s1['sent']['throttle_gain'] == 0.92 and '速度ゲイン 0.92' in s1['confirm']
    assert reads(res, 'gain')[2] == 0.87 and s2['sent']['throttle_gain'] == 0.87


@needs_node
def test_state2_buttons_while_running_send_now(tmp_path, cands):
    """走行中のボタン = 走行中のスライダーと同じ(すぐ送る。「ゲインは走行中に動かしても、すぐ反映されます」)"""
    c, _ = cands
    res, _ = run_sim(tmp_path, c, [['server_set', {'throttle_gain': 0.9}], ['running', True], ['posts'], ['click', '#btn-dr-tg-up'],
                                   ['posts'], ['read', 'server']], server=server_params())
    assert params_posts(posts(res)[1]) == [{'throttle_gain': 0.91}] and reads(res, 'server')[0]['throttle_gain'] == 0.91


@needs_node
def test_state2_block_values_and_3_digit_api(tmp_path, cands):
    """ブロックの速度(0.90・0.85・0.95・1.00)はそのまま入る。ブロックに 0.92 を書いても入る(直す前は 0.9)。
    API で 3 桁(0.923)を入れると、画面は 0.92 を出し、走行開始で 0.92 を送る(確認の文の値 = 送る値)"""
    c, di = cands
    c2 = copy.deepcopy(c)
    c2['days'][di]['blocks'][blk(c, di, 1)]['params'] = {'throttle_gain': 0.92}
    res, _ = run_sim(tmp_path, c2, [['pick', di, blk(c, di, 1)], ['read', 'gain'], ['start']])
    assert reads(res, 'gain') == [0.92] and starts(res)[0]['sent']['throttle_gain'] == 0.92
    res0, _ = run_sim(tmp_path, c2, [['pick', di, blk(c, di, 1)], ['read', 'gain'], ['start']], old=True, client_version=6, name='o.json')
    assert reads(res0, 'gain') == [0.9]
    res, _ = run_sim(tmp_path, c, [['server_set', {'throttle_gain': 0.923}], ['poll'], ['read', 'gain'], ['read', 'gain_text'], ['start']])
    assert reads(res, 'gain') == [0.92] and reads(res, 'gain_text') == ['0.92']
    assert starts(res)[0]['sent']['throttle_gain'] == 0.92 and '速度ゲイン 0.92' in starts(res)[0]['confirm']
    for no in (1, 13, 20):              # 11/3 のブロックは全部 0.90
        r2, _ = run_sim(tmp_path, c, [['pick', di, blk(c, di, no)], ['read', 'gain'], ['start']], name='b%d.json' % no)
        assert reads(r2, 'gain') == [0.9] and starts(r2)[0]['sent']['throttle_gain'] == 0.9


# ---------------------------------------------------------------- state-5 画面の版 7
def test_state5_versions():
    src = (ROOT / 'jetformer' / 'web' / 'server.py').read_text(encoding='utf-8')
    ver = int(re.search(r'^DRIVE_CLIENT_VERSION = (\d+)', src, re.M).group(1))
    js = APP_JS.read_text(encoding='utf-8')
    cl = [int(x) for x in re.findall(r'const body = \{ client: (\d+),', js)]
    assert ver == 9 and cl == [9]                  # 2026-10-27 fix5(t1・t2・t3)で 8 → 9(r2-r2-state-3 で 7 → 8、2026-10-26 state-5 で 6 → 7)
    assert re.findall(r'const body = \{ client: (\d+),', APP_OLD.read_text(encoding='utf-8')) == ['6']
    assert 'if _client_version(body) < DRIVE_CLIENT_VERSION:' in src and 'raise HTTPException(409' in src
    # 版を見るのは走行の開始だけ(止める・設定・表示は古い画面からも断らない)
    assert src.count('DRIVE_CLIENT_VERSION') == 3 and src.count('if _client_version(body) <') == 1


@needs_node
def test_state5_old_page_refused_new_page_starts(tmp_path, cands):
    """再起動のあとのサーバー(版 7): 直す前の画面(版 6)は 409 で始まらない → 読み直した画面は始まる。
    今のサーバー(版 6。10/26 07:41 起動)でも、新しい画面(版 7)は始まる(app.js は読み直しで効き、再起動の前でも走れる)"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 1)], ['start'], ['read', 'toasts']]
    res0, _ = run_sim(tmp_path, c, steps, old=True, client_version=7, name='o.json')
    s0 = starts(res0)[0]
    assert s0['client'] == 6 and s0['refused'] is True
    assert any('管理画面が古いまま' in t for t in reads(res0, 'toasts')[0])
    res, _ = run_sim(tmp_path, c, steps, client_version=7)
    assert starts(res)[0]['client'] == 9 and starts(res)[0]['refused'] is False     # 2026-10-27 fix5 で版 9
    res6, _ = run_sim(tmp_path, c, steps, client_version=6, name='v6.json')
    assert starts(res6)[0]['refused'] is False


# ---------------------------------------------------------------- 試験の選択肢のスイッチ
def test_opt_switches_static():
    html = INDEX.read_text(encoding='utf-8')
    for key, sel in OPT_SEL.items():
        tag = re.search(r'<input type="checkbox" id="%s"[^>]*>' % sel[1:], html).group(0)
        assert 'checked' not in tag and 'autocomplete="off"' in tag, key            # 既定 OFF・再読み込みで前の ON を戻さない
        assert re.search(r'<div class="opt-line hidden" id="dr-opt-line-%s">' % key, html), key   # サーバーに無いうちは出さない
        assert 'id="%s-note"' % sel[1:] in html
    for words in ('既定は OFF', '起動するたびに OFF', '走行中は ON にできません', '基準の設定・ブロックでは変わりません', 'ユーザーの了承',
                  '実車の ESC ではまだ', '机上の試験だけ'):
        assert words in html, words
    assert html.index('id="dr-launch-no-kick"') > html.index('id="dr-launch-boost"')       # 発進の前進の中
    assert html.index('id="dr-rec-back-guard"') > html.index('id="dr-recovery"')            # リカバリーの下
    js = APP_JS.read_text(encoding='utf-8')
    keys = re.findall(r"\{ key: '([a-z_]+)', sel: '(#[a-z-]+)'", js)
    assert dict(keys) == OPT_SEL
    for k in OPT:                                                                  # drive.py の項目と決まり(既定 0・起動で 0)
        assert drive.PARAM_LIMITS[k] == (0, 1) and drive.DEFAULT_PARAMS[k] == 0 and k in drive.RESET_ON_BOOT, k
    rd = re.search(r'^function readDriveParams\(\) \{.*?^}\n', js, re.S | re.M).group(0)
    assert not any(k in rd for k in OPT)                                           # 走行開始では送らない(サーバーの値のまま)
    fn = re.search(r'^function benchApply\(.*?^}\n', js, re.S | re.M).group(0)
    assert '/api/drive/' not in fn and 'api(' not in fn


@needs_node
@pytest.mark.parametrize('key', OPT)
def test_opt_switch_on_off(tmp_path, cands, key):
    """スイッチを ON(確認の文に答える)→ すぐ送る → 走行開始の確認の文に ON の行・送る params には入れない → OFF もすぐ送る"""
    c, di = cands
    sel = OPT_SEL[key]
    steps = [['el', '#dr-opt-line-%s' % key, 'hidden'], ['el', sel, 'checked'], ['el', sel, 'disabled'], ['posts'],
             ['check', sel, True], ['posts'], ['poll'], ['el', sel, 'checked'], ['el', sel + '-note', 'html'], ['read', 'check_base'],
             ['pick', di, blk(c, di, 1)], ['el', sel, 'checked'], ['read', 'note'], ['start'], ['read', 'server'], ['running', False],
             ['check', sel, False], ['posts'], ['poll'], ['el', sel, 'checked'], ['start']]
    res, _ = run_sim(tmp_path, c, steps, server=server_params(rec_back_guard=1 if key == 'rec_hold_keep' else 0))
    e = els(res)
    assert e[:3] == [False, False, False]                                           # 出ている・OFF・押せる
    ch = [x for x in res if x['op'] == 'check']
    assert 'を ON にします' in ch[0]['confirm'] and 'ユーザーの了承は出ていますか' in ch[0]['confirm'] and '走行中は ON にできません' in ch[0]['confirm']
    assert ch[1]['confirm'] == ''                                                   # OFF は確認なし
    p = posts(res)
    assert params_posts(p[1]) == [{key: 1}] and params_posts(p[2])[-1] == {key: 0}
    assert e[3] is True and '<b class="warn-text">ON:' in e[4]
    assert ': ON(基準の設定・ブロックでは変えない)' in reads(res, 'check_base')[0]
    assert e[5] is True                                                             # ブロックを選んでも ON のまま
    assert 'は ON のままです' in reads(res, 'note')[0]
    s1, s2 = starts(res)
    assert key not in s1['sent'] and key not in s2['sent']
    assert '\n' + OPT_LINE[key] in s1['confirm'] and OPT_LINE[key] not in s2['confirm']
    assert reads(res, 'server')[0][key] == 1 and e[6] is False


@needs_node
def test_opt_cancel_and_running(tmp_path, cands):
    """確認でキャンセルすると送らない。走行中は ON にできない(押せない・押しても送らない)。走行中に ON のものは OFF にできる"""
    c, _ = cands
    res, _ = run_sim(tmp_path, c, [['posts'], ['check', '#dr-rec-back-guard', True], ['posts'], ['el', '#dr-rec-back-guard', 'checked']],
                     confirm=False)
    assert params_posts(posts(res)[1]) == [] and els(res) == [False]
    res, toasts = run_sim(tmp_path, c, [['running', True], ['el', '#dr-rec-hold-keep', 'disabled'], ['posts'],
                                        ['check', '#dr-rec-hold-keep', True], ['posts'], ['el', '#dr-rec-hold-keep', 'checked'],
                                        ['el', '#dr-rec-back-guard', 'disabled'], ['check', '#dr-rec-back-guard', False], ['posts'],
                                        ['read', 'server']], server=server_params(rec_back_guard=1))
    e = els(res)
    assert e[0] is True and e[1] is False and e[2] is False                         # OFF のものは押せない・ON のものは押せる
    assert params_posts(posts(res)[1]) == [] and any('走行中は中立止めのあともリカバリーを残すを ON にできません' in t for t in toasts)
    assert params_posts(posts(res)[2]) == [{'rec_back_guard': 0}] and reads(res, 'server')[0]['rec_back_guard'] == 0


@needs_node
def test_opt_block_and_baseline_never_turn_on_or_off(tmp_path, cands):
    """ブロックの params に 1 を書いても ON にしない(注を出す)。基準の設定・取り消しも変えない。API(curl)で入れた ON は画面に出る"""
    c, di = cands
    c2 = copy.deepcopy(c)
    b = blk(c, di, 1)
    c2['days'][di]['blocks'][b]['params'] = dict(c2['days'][di]['blocks'][b].get('params') or {}, rec_back_guard=1, rec_hold_keep=1,
                                                 launch_no_kick=1)
    res, _ = run_sim(tmp_path, c2, [['pick', di, b], ['read', 'note'], ['poll'], ['el', '#dr-rec-back-guard', 'checked'], ['baseline'],
                                    ['undo'], ['poll'], ['read', 'server'], ['start']])
    note = reads(res, 'note')[0]
    for name in ('リカバリーの後ろの見張り', '中立止めのあともリカバリーを残す', '発進の前進を出さない区間は発進の補助も出さない'):
        assert '%sはブロックでは ON にしません' % name in note
    assert els(res) == [False] and all(reads(res, 'server')[0][k] == 0 for k in OPT)
    assert not any(OPT_LINE[k] in starts(res)[0]['confirm'] for k in OPT)
    res, _ = run_sim(tmp_path, c, [['server_set', {'rec_back_guard': 1, 'rec_hold_keep': 1}], ['poll'], ['el', '#dr-rec-hold-keep', 'checked'],
                                   ['baseline'], ['poll'], ['el', '#dr-rec-hold-keep', 'checked'], ['start']])
    assert els(res) == [True, True] and '\n' + OPT_LINE['rec_hold_keep'] in starts(res)[0]['confirm']


@needs_node
def test_opt_old_server_hides_switches(tmp_path, cands):
    """今の古いサーバー(params に 3 つが無い)では、スイッチを出さず「今のサーバーには無い」の 1 行。送らない・確認の文の行も無い"""
    c, di = cands
    old = {k: v for k, v in server_params().items() if k not in OPT}
    steps = [['el', '#dr-opt-line-rec_back_guard', 'hidden'], ['el', '#dr-opt-line-rec_hold_keep', 'hidden'],
             ['el', '#dr-opt-line-launch_no_kick', 'hidden'], ['el', '#dr-rec-back-guard', 'disabled'],
             ['el', '#dr-opt-rec-missing', 'text'], ['el', '#dr-opt-launch-missing', 'text'], ['pick', di, blk(c, di, 1)], ['start']]
    res, _ = run_sim(tmp_path, c, steps, server=old)
    e = els(res)
    assert e[:4] == [True, True, True, True]
    assert '後ろの見張り' in e[4] and '中立止めのあともリカバリーを残す' in e[4] and '再起動' in e[4]
    assert '発進の補助も出さない' in e[5]
    assert not any(k in starts(res)[0]['sent'] for k in OPT)
    # 新しいサーバーのときは「無い」の行を出さない
    res, _ = run_sim(tmp_path, c, steps, name='new.json')
    assert els(res)[:6] == [False, False, False, False, '', '']


# ---------------------------------------------------------------- state2-1(fix2)トリム
@needs_node
def test_trim_survives_every_screen_path(tmp_path, cands):
    """トリム +15 は、ブロック・基準の設定・取り消し・天井・発進の前進・試験の選択肢のスイッチ・走行のあとの同期で 0 に戻らない"""
    c, di = cands
    steps = [['slide', '#dr-trim', 15], ['poll'], ['pick', di, blk(c, di, 2)], ['read', 'trim'], ['undo'], ['read', 'trim'],
             ['baseline'], ['read', 'trim'], ['pick', di, blk(c, di, 13)], ['check', '#dr-launch-boost', True],
             ['check', '#dr-speed-ceiling', True], ['check', '#dr-rec-back-guard', True], ['poll'], ['read', 'trim'], ['start'],
             ['poll'], ['running', False], ['pick', di, blk(c, di, 14)], ['read', 'trim'], ['start'], ['read', 'server']]
    res, _ = run_sim(tmp_path, c, steps)
    assert reads(res, 'trim') == [15, 15, 15, 15, 15]
    assert [s['sent']['steer_trim_us'] for s in starts(res)] == [15, 15]
    assert all('\nトリム: +15 µs' in s['confirm'] for s in starts(res))
    assert reads(res, 'server')[0]['steer_trim_us'] == 15


# ---------------------------------------------------------------- 走行開始で送る値: 直す前と同じ流れ
def _flows(c, di):
    b = lambda no: blk(c, di, no)
    return {
        'ブロック 1 → 走行開始': ([['pick', di, b(1)], ['start']], {}),
        'トリム +15 → ブロック 2・3': ([['slide', '#dr-trim', 15], ['poll'], ['pick', di, b(2)], ['start'], ['poll'], ['running', False],
                                       ['pick', di, b(3)], ['start']], {}),
        '2-3: 天井 + 自動合わせ': ([['pick', di, b(8)], ['check', '#dr-speed-ceiling', True], ['check', '#dr-speed-auto', True],
                                   ['poll'], ['start']], {}),
        '2-6: 前の本が天井だけ': ([['pick', di, b(11)], ['check', '#dr-speed-ceiling', True], ['poll'], ['poll'], ['start']],
                                  {'speed_ceiling': 1, 'speed_auto': 0}),
        '3-2: 発進の前進 ON': ([['pick', di, b(13)], ['check', '#dr-launch-boost', True], ['poll'], ['start']], {}),
        '3-3: 前の本が発進の前進 ON': ([['pick', di, b(14)], ['poll'], ['start']], {'launch_boost': 1}),
        '基準の設定 → 走行開始': ([['baseline'], ['start']], {'speed_ceiling': 1, 'speed_auto': 1, 'launch_boost': 1}),
        '速度 1.10(0.05 の倍数)': ([['slide', '#dr-throttle-gain', 1.1], ['start']], {}),
        '後ろの見張りを API で ON → ブロック 4': ([['pick', di, b(4)], ['poll'], ['start']], {'rec_back_guard': 1}),
        '取り消し': ([['pick', di, b(5)], ['undo'], ['poll'], ['start']], {'speed_ceiling': 1, 'speed_auto': 1}),
    }


@needs_node
def test_sent_values_same_as_before_except_listed(tmp_path, cands):
    """state-1 の場面・0.05 の倍数でない速度・画面の版のほかは、走行開始で送る params と確認の文が直す前と同じ"""
    c, di = cands
    for i, (name, (steps, srv)) in enumerate(_flows(c, di).items()):
        new, _ = run_sim(tmp_path, c, steps, server=server_params(**srv), name='n%d.json' % i)
        old, _ = run_sim(tmp_path, c, steps, server=server_params(**srv), old=True, client_version=6, name='o%d.json' % i)
        sn, so = starts(new), starts(old)
        assert len(sn) == len(so) > 0, name
        for a, b in zip(sn, so):
            assert a['sent'] == b['sent'], (name, {k: (a['sent'].get(k), b['sent'].get(k)) for k in set(a['sent']) | set(b['sent'])
                                                   if a['sent'].get(k) != b['sent'].get(k)})
            assert a['confirm'] == b['confirm'], name
            assert (a['client'], b['client']) == (9, 6)             # 2026-10-27 fix5 で今の画面は版 9


# ---------------------------------------------------------------- 画面を開いた直後(サーバーの設定をまだ読んでいない)
@needs_node
def test_start_waits_for_first_sync(tmp_path, cands):
    """サーバーにトリム +15・速度 0.92 が入っていても、画面を開いて最初の同期の前は欄が index.html の初めの値(トリム 0・速度 1.00)。
    その間に走行開始を押しても送らない(前は 0・1.00 を送った)。同期のあとは 15・0.92 を送る"""
    c, _ = cands
    srv = server_params(steer_trim_us=15.0, throttle_gain=0.92)
    steps = [['read', 'trim'], ['start'], ['read', 'toasts'], ['poll'], ['read', 'trim'], ['start']]
    res, _ = run_sim(tmp_path, c, steps, server=srv, status_fail_first=2)
    s1, s2 = starts(res)
    assert reads(res, 'trim') == [0, 15]
    assert s1['sent'] is None and any('サーバーの設定を読み込んでいます' in t for t in reads(res, 'toasts')[0])
    assert s2['sent']['steer_trim_us'] == 15 and s2['sent']['throttle_gain'] == 0.92
    res0, _ = run_sim(tmp_path, c, steps, server=srv, status_fail_first=2, old=True, client_version=6, name='old.json')
    assert starts(res0)[0]['sent']['steer_trim_us'] == 0 and starts(res0)[0]['sent']['throttle_gain'] == 1.0   # 直す前


@needs_node
def test_new_app_with_old_index_html_still_works(tmp_path, cands):
    """新しい app.js と古い index.html(キャッシュ)の組み合わせ: 新しい欄(±0.01・試験の選択肢)が無くても画面は動き、
    走行開始で送る値は直す前と同じ(天井だけの場面を除く)"""
    c, di = cands
    ids = lambda p: set(re.findall(r'\bid="([^"]+)"', p.read_text(encoding='utf-8')))
    missing = sorted(ids(INDEX) - ids(INDEX_OLD))
    assert {'btn-dr-tg-up', 'btn-dr-tg-down', 'dr-rec-back-guard', 'dr-rec-hold-keep', 'dr-launch-no-kick'} <= set(missing)
    steps = [['pick', di, blk(c, di, 13)], ['check', '#dr-launch-boost', True], ['poll'], ['start']]
    res, _ = run_sim(tmp_path, c, steps, server=server_params(rec_back_guard=1), missing_ids=missing)
    res0, _ = run_sim(tmp_path, c, steps, server=server_params(rec_back_guard=1), old=True, client_version=6, name='o.json')
    assert starts(res)[0]['sent'] == starts(res0)[0]['sent'] and starts(res)[0]['sent']['launch_boost'] == 1
    assert '\nリカバリーの後ろの見張り: ON' in starts(res)[0]['confirm']


@needs_node
def test_opt_hold_keep_needs_back_guard(tmp_path, cands):
    """中立止めのあとも残す(rec_hold_keep)は後ろの見張りが ON のときだけ(drive._opt_rules と同じ)。見張りが OFF のあいだは押せず、
    押しても送らない。見張りを ON → 残すを ON → 見張りを OFF にすると、サーバーが残すも OFF にし、画面も OFF に戻る"""
    c, _ = cands
    steps = [['el', '#dr-rec-hold-keep', 'disabled'], ['el', '#dr-rec-hold-keep-note', 'html'], ['posts'],
             ['check', '#dr-rec-hold-keep', True], ['posts'], ['el', '#dr-rec-hold-keep', 'checked'],
             ['check', '#dr-rec-back-guard', True], ['poll'], ['el', '#dr-rec-hold-keep', 'disabled'],
             ['check', '#dr-rec-hold-keep', True], ['poll'], ['el', '#dr-rec-hold-keep', 'checked'], ['start'], ['running', False],
             ['check', '#dr-rec-back-guard', False], ['poll'], ['el', '#dr-rec-hold-keep', 'checked'], ['read', 'server'], ['read', 'toasts'],
             ['start']]
    res, toasts = run_sim(tmp_path, c, steps)
    e = els(res)
    assert e[0] is True and 'リカバリーの後ろの見張りが OFF のあいだは押せません' in e[1]
    assert params_posts(posts(res)[1]) == [] and e[2] is False
    assert any('後ろの見張りが ON のときだけ使えます' in t for t in toasts)
    assert e[3] is False and e[4] is True                                          # 見張りのあとは押せる・ON になった
    s1, s2 = starts(res)
    assert '\nリカバリーの後ろの見張り: ON' in s1['confirm'] and '\n中立止めのあともリカバリーを残す: ON' in s1['confirm']
    assert e[5] is False and reads(res, 'server')[0]['rec_hold_keep'] == 0
    assert any('(中立止めのあともリカバリーを残すも OFF)' in t for t in reads(res, 'toasts')[0])
    assert '後ろの見張り' not in s2['confirm'] and '中立止め' not in s2['confirm']


# ---------------------------------------------------------------- 偽のサーバーの決まり = 本物の drive.Driver の決まり
@needs_node
def test_ui_posts_replayed_on_real_driver(tmp_path, cands, monkeypatch):
    """画面が送った POST(params・走行開始)を、本物の drive.Driver(装置なし・設定ファイルは tmp・走行ループは動かさない)に
    同じ順で入れる。断る・受け付けるが偽のサーバーと同じで、終わりの値も同じ(試験の偽のサーバーが本物の決まりどおり)"""
    c, di = cands
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    monkeypatch.setattr(drive, 'model_info', lambda name: {'trained_modes': [5], 'drivable': True, 'postprocess': None})
    scen = [
        ([['pick', di, blk(c, di, 11)], ['check', '#dr-speed-ceiling', True], ['poll'], ['poll'], ['start']],
         dict(speed_ceiling=1, speed_auto=1)),
        ([['pick', di, blk(c, di, 8)], ['check', '#dr-speed-ceiling', True], ['check', '#dr-speed-auto', True], ['start']], {}),
        ([['check', '#dr-rec-hold-keep', True], ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True],
          ['check', '#dr-launch-no-kick', True], ['pick', di, blk(c, di, 13)], ['check', '#dr-launch-boost', True], ['start']], {}),
        ([['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True], ['check', '#dr-rec-back-guard', False],
          ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'], ['start']], {}),
        ([['server_set', {'throttle_gain': 0.97}], ['poll'], ['baseline'], ['click', '#btn-dr-tg-down'], ['start']],
         dict(speed_ceiling=1, launch_boost=1)),
    ]
    for i, (steps, srv) in enumerate(scen):
        sp = server_params(**srv)
        res, _ = run_sim(tmp_path, c, steps + [['posts'], ['read', 'server']], server=sp, name='s%d.json' % i)
        d = drive.Driver(hub=None, recorder=None)
        d._run = lambda *a, **k: None                                   # 走行ループは動かさない(装置に触れない)
        d.params = dict(sp)
        for path, body, status in posts(res)[0]:
            if path == '/api/drive/params':
                try:
                    d.set_params(body)
                    got = 200
                except ValueError:
                    got = 400
            else:
                d.start(body['run'], body['mode_id'], params=body['params'])
                d.thread.join(1)
                got = 200
            assert got == status, (i, path, body, status)
        want = reads(res, 'server')[0]
        assert {k: d.params[k] for k in want} == {k: (type(d.params[k])(v)) for k, v in want.items()}, i


@needs_node
def test_launch_no_kick_line_says_no_effect_without_launch(tmp_path, cands):
    """launch_no_kick は発進の前進が ON の本でだけ効く。OFF の本の確認の文には「何も変わりません」を足す"""
    c, di = cands
    srv = server_params(launch_no_kick=1)
    res, _ = run_sim(tmp_path, c, [['pick', di, blk(c, di, 12)], ['start'], ['running', False], ['pick', di, blk(c, di, 13)],
                                   ['check', '#dr-launch-boost', True], ['start']], server=srv)
    off, on = starts(res)
    assert '\n発進の前進を出さない区間は発進の補助も出さない: ON(この本は発進の前進が OFF なので、何も変わりません)' in off['confirm']
    line = [x for x in on['confirm'].split('\n') if x.startswith('発進の前進を出さない区間')]
    assert line == ['発進の前進を出さない区間は発進の補助も出さない: ON'] and '\n発進の前進: ON' in on['confirm']


def test_state2_server_side_keeps_001_values(tmp_path, monkeypatch):
    """サーバー側(web/bench.py・drive.py)は 0.05 に丸めない: ブロックの上書き 0.92 はそのまま・説明欄の札は throttle_gain=0.92・
    set_params の 0.92 もそのまま(画面の刻みだけが 0.05 だった)"""
    w = []
    ov, _ = bench._block_overrides({'params': {'throttle_gain': 0.92}, 'tune': '速さ0.92'}, w, 'B1')
    assert ov['params']['throttle_gain'] == 0.92 and w == []
    doc = {'version': 1, 'bench': 'b', 'candidates': [{'id': 'B', 'label': 'B', 'run': bench.B_RUN, 'hypothesis': '-'}],
           'days': [{'date': '2026-11-03', 'blocks': [{'block': 1, 'id': 'B', 'params': {'throttle_gain': 0.97}, 'tune': '速さ0.97'}]}],
           'description_template': 'bench=b day={date} block={block} cand={id} model={run} memo='}
    p = tmp_path / 'bench_candidates.json'
    p.write_text(json.dumps(doc), encoding='utf-8')
    out = bench.load_candidates(p)
    b = out['days'][0]['blocks'][0]
    assert b['params']['throttle_gain'] == 0.97 and 'throttle_gain=0.97' in b['description'] and 'tune=速さ0.97' in b['description']
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    d = drive.Driver(hub=None, recorder=None)
    assert d.set_params({'throttle_gain': 0.92})['throttle_gain'] == 0.92
