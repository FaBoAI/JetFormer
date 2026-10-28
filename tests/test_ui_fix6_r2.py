"""画面の係の直し 6-2 回目(2026-10-27 fix6。反証の指摘 w2-ui-w1-ui-r62-1。~/jetformer_reports/fix_20261026/fix6/fix_ui_r2.md)の試験。

  w2-ui-w1-ui-r62-1: 走行タブを押したときにモデルの一覧の GET(/api/drive/models)が通らないと、loadDrive は toast を出して return
    するだけで、表示の問い合わせの輪(pollDrive)を始めなかった。pollDrive は走行タブか走行中のときしか次を掛けないので、ほかのタブから
    戻ったときに輪が止まったまま・drLost も呼ばれず paramsLoaded は true のまま・▶ も押せて、天井の横の文・スイッチと ▶ の確認の文が
    前の表示のまま(curl で入れた左 4.2 を読まず 4.1、起き直したあとも起き直す前の 4.2・200 µs)だった。status の GET が返らないまま
    止まったとき(fetch に期限が無い)も同じに輪が止まった。
  直し(app.js。表示と問い合わせだけ。走行開始で送る物・断る決まりは同じ):
    (a) 一覧の GET が失敗したら、止まっていれば(drIdle)drLost() と pollDrive()。
    (b) 一覧の GET が 5 秒(DR_IDLE_WAIT_MS)返らなければ、そのとき止まっていれば同じ(GET は打ち切らない。返事が来たら前と同じに続ける)。
    (c) pollDrive の status の GET は、止まっているときだけ 5 秒で打ち切って失敗(drLost・次の問い合わせ)。本文の途中の打ち切りも失敗。
    走行中・走行を始めた画面(token あり)は、問い合わせの回数・間隔・打ち切らないことが前とまったく同じ(生存確認を変えない)。

本物の app.js(今の版と、直す前の写し tests/fixtures/fix6_ui_r2_before/app.js = 直す 6-1 回目のあとの版 e41c2f3e…。sha256 を照らす)を
node の vm で(tests/fixtures/app_sim_fix6_r2.js = app_sim_fix6.js + 一覧だけの失敗・全部の失敗・止まった GET・タイマーを鳴らす・
通信の全部)。車・ブラウザ・jetformer-web には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix6_r2.py

JF_UI6R2_APP=<app.js の写し> を付けると、その写しを「今の版」として流す(わざと壊す台本 experiments/20261026_fix/fix6/ui_r2/mut/ が使う)。
そのあと直す 7-1 回目(w1-ui-r7-ui-1。tests/test_ui_fix7.py)が app.js を直した。動きの試験は今の app.js で流す(7-1 回目の直しは起き直しの
知らせの文だけで、既定に戻ったときの文は前と同じ)。「違いは 5 か所だけ」は 6-2 回目のあとの版の写し(tests/fixtures/fix7_ui_before/)で見る。
さらに直す 7-2 回目(w2-ui-w2-ui-r72-ui-1・-2。tests/test_ui_fix7_r2.py)が app.js を直した。一覧を読めなかったあとは輪の 2 回目の同期で
一覧を 1 回読み直し、読めていない間の ▶ の文が変わる(test_first_open_with_models_fail の 1 行だけ合わせた。ほかは今の app.js で前と同じに通る)。
"""
import difflib
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

from test_ui_fix3 import _flows
import test_ui_fix6 as F6

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
APP = Path(os.environ.get('JF_UI6R2_APP') or STATIC / 'app.js')
BEFORE = ROOT / 'tests' / 'fixtures' / 'fix6_ui_r2_before'
# 2026-10-28 fix8 tools-decided のあとの版の sha256(このあとの直しはその係がここに足す)
LATER_FIX8_TOOLS_DECIDED = {'experiments/20261026_fix/plan/runs_1103.py': '33bc7fca99216cc2d4704e8fa5de9741626231fd4a0d165cb981183801acba00'}
# 2026-10-28 fix8 画面の係 直す 8-1 回目(tests/test_ui_fix8_r1.py)のあとの版の sha256。その版なら直す前の写し tests/fixtures/fix8_ui_r1_before/
# (= 上の版)へたどる(中身の違いは tests/test_ui_fix8_r1.py)
FIX8_UI_R1_BEFORE = ROOT / 'tests' / 'fixtures' / 'fix8_ui_r1_before'
LATER_FIX8_UI_R1 = {'experiments/20261026_fix/plan/runs_1103.py': 'c2d616c2f72b00935868691236f94e4733c3185a24d200c524b6049d615fd9fc',
                    'jetformer/web/static/style.css': 'dbc8b79ad935ed914c7b8a2cc02f8c4ddaf0c453af33ca3fb85fa0ea1103df56',
                    'jetformer/web/static/app.js': 'aa275bb5da0bbd740f38c217dd0ae9ee19240f061dafd8ca70f6916448837285'}
# この組(6-2 回目)のあとの版の写し(908934c6…)。そのあと 7-1 回目(w1-ui-r7-ui-1。tests/test_ui_fix7.py)が app.js を直したので、
# 「違いは 5 か所だけ」はこの写しで見る(今の app.js との違いは test_ui_fix7.py が見る)。JF_UI6R2_APP を付けたときはその写し
R2_AFTER = ROOT / 'tests' / 'fixtures' / 'fix7_ui_before'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix6_r2.js'
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
ST = '/api/drive/status'
MD = '/api/drive/models'
LOST = 'サーバーにつながっていません'
WAIT = 5000                                   # DR_IDLE_WAIT_MS

OLD_API = 'async function api(method, url, body) {'
OLD_MODELS = "  try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) { toast(e.message, true); return; }"
OLD_POLL = "  try { st = await api('GET', '/api/drive/status' + tok); } catch (e) { st = null; drLost(); }"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_before_copy_and_only_the_loop_places_changed():
    """直す前の写しは 6-1 回目のあとの版(e41c2f3e…)。この組のあとの版(908934c6…。写し tests/fixtures/fix7_ui_before/。7-1 回目の
    直す前)との違いは api の打ち切りの印・DR_IDLE_WAIT_MS と drIdle・loadDrive の一覧の GET の失敗と期限・pollDrive の打ち切りの
    4 か所だけ。走行開始の body・版(9)・ほかの送信は変えていない。server.py・index.html・runs_1103.py は変えていない"""
    sums = dict(reversed(x.split()) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert set(sums) == {'app.js'} and _sha(BEFORE / 'app.js') == sums['app.js']
    assert sums['app.js'].startswith('e41c2f3ea796')
    after = APP if os.environ.get('JF_UI6R2_APP') else R2_AFTER / 'app.js'     # 7-1 回目の直す前の写し(fix7)
    if not os.environ.get('JF_UI6R2_APP'):
        r2 = dict(reversed(x.split()) for x in (R2_AFTER / 'SHA256SUMS').read_text().splitlines() if x.strip())
        assert r2 == {'app.js': _sha(after)} and r2['app.js'].startswith('908934c6297b')
    a = (BEFORE / 'app.js').read_text(encoding='utf-8').splitlines()
    b = after.read_text(encoding='utf-8').splitlines()
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
    removed = [a[i] for o in ops for i in range(o[1], o[2])]
    added = [b[j] for o in ops for j in range(o[3], o[4])]
    assert removed == [OLD_API, OLD_MODELS, OLD_POLL], removed
    assert len(ops) == 5, ops                       # api の頭・signal の 1 行・定数と drIdle・loadDrive・pollDrive
    code = [x.split('//')[0] for x in added]
    assert not any(re.search(r"'POST'|body\.|const body|client:", x) for x in code)      # 送る物は足していない
    assert sum('pollDrive()' in x for x in code) == 1 and sum('drLost()' in x for x in code) == 2
    assert added[0] == 'async function api(method, url, body, signal) {' and added[1].strip().startswith('if (signal) opt.signal = signal;')
    assert [x.strip() for x in code if 'await api(' in x] == [
        "try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) { clearTimeout(late); toast(e.message, true); lost(); return; }",
        "st = await api('GET', '/api/drive/status' + tok, undefined, ac ? ac.signal : undefined);"]
    js = after.read_text(encoding='utf-8')
    assert re.findall(r'^const DR_IDLE_WAIT_MS = (\d+);', js, re.M) == [str(WAIT)]
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9'] and re.findall(r'const body = \{ client: (\d+),', js) == ['9']
    assert _sha(ROOT / 'jetformer' / 'web' / 'server.py').startswith('ad7a7d05f7a9')
    assert _sha(STATIC / 'index.html').startswith('ef81bb6017d6')
    # 2026-10-28 fix8 tools-decided が runs_1103.py を変えた(試験の選択肢の印の後ろの文と説明だけ)。その版なら直す前の写し
    # (tests/fixtures/fix8_tools_decided_before/runs_1103.py.orig)がこの組の版と同じかを見る(中身の違いは tests/test_fix8_tools_decided.py)
    rp = ROOT / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'
    if _sha(rp) == LATER_FIX8_UI_R1['experiments/20261026_fix/plan/runs_1103.py']:      # fix8 ui r1 の版 → その直す前の写し(33bc7fca…)
        rp = FIX8_UI_R1_BEFORE / 'runs_1103.py'
    if _sha(rp) == LATER_FIX8_TOOLS_DECIDED['experiments/20261026_fix/plan/runs_1103.py']:
        rp = ROOT / 'tests' / 'fixtures' / 'fix8_tools_decided_before' / 'runs_1103.py.orig'
    assert _sha(rp).startswith('302296586af9')


# ------------------------------------------------------------------ 画面を node で流す
cands = F6.cands
blk = F6.blk
server_params = F6.server_params
PRE_B = server_params(rec_back_guard=1, ceil_left_mps=4.1)                               # 指摘の B(起き直しなし)
PRE_A = server_params(rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)  # 指摘の A(朝の設定のあと起き直す)


def run_sim(tmp_path, c, steps, app=None, name='input.json', server=None, **extra):
    inp = tmp_path / name
    d = {'bench': c, 'server_params': server if server is not None else PRE_B, 'steps': steps, 'server_client_version': 9,
         'confirm': True, 'boot_at': 1000.0, 'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}),
         'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True, 'boot_refusal_texts': F6.REAL}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app or APP), str(STATIC / 'index.html'), str(inp)], capture_output=True, text=True,
                       timeout=120, env=dict(os.environ, TZ='Asia/Tokyo'))
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def both(tmp_path, c, steps, **kw):
    """今の版と直す前の写しで同じ操作を流す"""
    new = run_sim(tmp_path, c, steps, name='n.json', **kw)
    old = run_sim(tmp_path, c, steps, app=BEFORE / 'app.js', name='o.json', **kw)
    return new, old


def ops(res, op):
    return [x for x in res if x['op'] == op]


def vals(res, op):
    return [x.get('value') for x in res if x['op'] == op]


def tagged(res):
    """['eval', 式] と ['el', sel, prop] の値を、印の付いた順に"""
    return [x['value'] for x in res if x['op'] in ('el', 'eval')]


def starts(res):
    return ops(res, 'start')


def left_lines(text):
    return [l for l in text.split('\n') if '左旋回中' in l]


BASE = [['realpoll'], ['tab', 'live'], ['tick'], ['tick'], ['calls']]          # 走行タブを一度読んだあと、ライブのタブへ
LOOK = [['eval', "'LOOK'"], ['tick'], ['tick'], ['ntimers'], ['eval', 'driveTimer !== null'], ['eval', 'state.drive.paramsLoaded'], ['read', 'st_params'],
        ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-ceil-spec', 'text'], ['el', '#dr-message', 'html']]


def look(res):
    """LOOK の結果(印 'LOOK' のすぐあと): (タイマー, 輪がある, paramsLoaded, 左, ▶ を押せない, 天井の横の文, つながっていませんの文)"""
    i = next(k for k, x in enumerate(res) if x['op'] == 'eval' and x.get('value') == 'LOOK')
    r = res[i + 1:i + 1 + len(LOOK) - 1]
    t = vals(r, 'ntimers')[0]
    ev = tagged(r)
    left = vals(r, 'read')[0]['ceil_left_mps']
    return t, ev[0], ev[1], left, ev[2], ev[3], LOST in (ev[4] or '')


@needs_node
def test_models_fail_then_curl_left_is_read(tmp_path, cands):
    """指摘の B: ライブ → 走行タブ(一覧の GET だけ失敗)→ 手順書 5 章の curl で左 4.2 → ブロック 10 → 天井 ON → ▶。
    直す前の写しは輪が止まり、天井の横の文・スイッチと ▶ の確認の文が 4.1 のまま(本物の Driver は 4.2 で走る)。今の版は輪が回り、
    どれも 4.2。送った POST(curl・スイッチ・走行開始の body と返事)は直す前と同じ"""
    c, di = cands
    steps = BASE + [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['api', {'ceil_left_mps': 4.2}]] + LOOK + [
        ['pick', di, blk(c, di, 10)], ['check', '#dr-speed-ceiling', True], ['tick'], ['start'], ['read', 'server'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    assert look(old) == ([], False, True, 4.1, False, '(全体 4.5・左旋回中 4.1 m/s)', False)          # 指摘が起きる
    assert look(new) == ([1000], True, True, 4.2, False, '(全体 4.5・左旋回中 4.2 m/s)', False)
    for res, want in ((old, '4.1'), (new, '4.2')):
        sw = [x for x in ops(res, 'check')][-1]['confirm']
        s = starts(res)[-1]
        assert left_lines(sw) == ['速さの天井を ON にします(全体 4.5・左旋回中 %s m/s)。' % want]
        assert left_lines(s['confirm']) == ['速さの天井: ON(全体 4.5・左旋回中 %s m/s。前進を弱めるだけ)' % want]
        assert s['status'] == 200 and s['boot_at'] == 1000.0
    assert vals(new, 'posts') == vals(old, 'posts')                 # 送った物は同じ(違うのは表示だけ)
    assert vals(new, 'read')[-1]['ceil_left_mps'] == 4.2 and vals(new, 'read')[-1]['speed_ceiling'] == 1
    assert 'Failed to fetch (sim6r2: モデルの一覧だけ通らない)' in tn and 'Failed to fetch (sim6r2: モデルの一覧だけ通らない)' in to


@needs_node
def test_models_fail_server_down_then_reboot(tmp_path, cands):
    """指摘の A: 朝の設定(見張り・残す ON・左 4.2・launch_us 200)→ ライブ → サーバーが落ちる → 走行タブ(一覧の GET が失敗)→
    既定の値で起き直す。直す前の写しは輪が止まり、▶ を押せて「つながっていません」も起動し直しの知らせも出ない(▶ は起き直す前の
    4.2 の文で 409)。今の版は、落ちている間は ▶ を押せず「つながっていません」、起き直すと 1 秒の問い合わせで知らせが出て
    4.1 を読み、▶ は新しい起動の時刻で通る(確認の文は今のサーバーの値。見張り・残すの行なし)"""
    c, di = cands
    steps = BASE + [['down_all', True], ['tab', 'drive'], ['eval', 'state.drive.paramsLoaded'], ['el', '#btn-dr-start', 'disabled'],
                    ['el', '#dr-message', 'html'], ['tick'], ['reboot', 2000.0, RESET], ['down_all', False]] + LOOK + [
        ['el', '#dr-launch-spec', 'text'], ['pick', di, blk(c, di, 10)], ['check', '#dr-speed-ceiling', True], ['start'],
        ['read', 'server'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps, server=PRE_A)
    on, oo = tagged(new)[:3], tagged(old)[:3]
    assert oo[0] is True and oo[1] is False and LOST not in oo[2]                  # 直す前: 同期したまま・押せる・文なし
    assert on[0] is False and on[1] is True and LOST in on[2]                     # 今: 落ちている間は断る
    assert look(old)[:5] == ([], False, True, 4.2, False)
    assert look(new) == ([1000], True, True, 4.1, False, '(全体 4.5・左旋回中 4.1 m/s)', False)
    so, sn = starts(old)[-1], starts(new)[-1]
    assert so['status'] == 409 and so['boot_at'] == 1000.0 and left_lines(so['confirm']) == ['速さの天井: ON(全体 4.5・左旋回中 4.2 m/s。前進を弱めるだけ)']
    assert sn['status'] == 200 and sn['boot_at'] == 2000.0 and left_lines(sn['confirm']) == ['速さの天井: ON(全体 4.5・左旋回中 4.1 m/s。前進を弱めるだけ)']
    assert '後ろの見張り' not in sn['confirm'] and '後ろの見張り' not in so['confirm']     # 前は見張りの行が消えて 4.2 が残る混ざった文
    assert tagged(old)[-1] == '(前進 200 µs 以上・2.5 m/s まで・最大 1.2 秒)' and tagged(new)[-1] == '(前進 250 µs 以上・2.5 m/s まで・最大 1.2 秒)'
    assert to[-1] == F6.PREFIX + F6.REAL['1000']                 # 前: ▶ が起き直しで断られて初めて知った(6-1 回目の出し直し)
    assert tn.count(F6.BOOT_TOAST) == 1 and tn.index(F6.BOOT_TOAST) < tn.index('AI走行 No.1 を開始しました')   # 今: ▶ の前に知らせ
    assert not any(t.startswith(F6.PREFIX) for t in tn)
    sent = [p for x in vals(new, 'posts') for p in x if p[0] == '/api/drive/start']
    assert [(p[1]['boot_at'], p[2]) for p in sent] == [(2000.0, 200)]
    assert vals(new, 'read')[-1]['ceil_left_mps'] == 4.1 and vals(new, 'read')[-1]['rec_back_guard'] == 0


@needs_node
def test_models_fail_and_status_hangs_refuses_start(tmp_path, cands):
    """一覧の GET が失敗し、そのあと status の GET も返らない(通信の途切れ): 今の版はすぐ ▶ を断る(「つながっていません」。走行開始は
    送らない)。5 秒で status の GET を打ち切り、次の問い合わせで同期すると押せる。直す前の写しは輪が止まり、▶ がそのまま通った"""
    c, di = cands
    steps = BASE + [['models_fail', True], ['hold', ST, True], ['tab', 'drive'], ['models_fail', False], ['ntimers'],
                    ['eval', 'state.drive.paramsLoaded'], ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-message', 'html'],
                    ['start'], ['el', '#toast', 'text'], ['tick'], ['hold', ST, False], ['tick'], ['ntimers'],
                    ['eval', 'state.drive.paramsLoaded'], ['el', '#btn-dr-start', 'disabled'], ['calls']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    n, o = tagged(new), tagged(old)
    assert vals(new, 'ntimers')[0] == [6000, WAIT] and vals(old, 'ntimers')[0] == [6000]   # トーストの 6 秒のほか、今: 打ち切りの 5 秒 / 前: 何も無い
    assert n[:2] == [False, True] and LOST in n[2]
    assert o[:2] == [True, False] and LOST not in o[2]
    sn, so = starts(new)[0], starts(old)[0]
    assert sn.get('status') is None and sn['confirm'] == '' and n[3].startswith(LOST)        # 断った(確認の文も出ない)
    assert so['status'] == 200                                                          # 前は通った
    assert vals(new, 'ntimers')[-1] == [1000] and n[4:] == [True, False]
    st_calls = [x for x in vals(new, 'calls')[-1] if x[1] == ST]
    assert st_calls[0][2] is None and all(x[2] == 200 for x in st_calls[1:]) and len(st_calls) >= 2   # 1 本目は打ち切り


@needs_node
def test_models_hang_starts_the_loop(tmp_path, cands):
    """一覧の GET が返らない(止まった GET): 今の版は 5 秒で輪を始め(drLost → 同期)、curl の 4.2 を読む。一覧の返事が来たら前と同じに
    続ける(選ぶ欄・輪)。直す前の写しは一覧の返事が来るまで輪が止まったまま 4.1"""
    c, di = cands
    steps = BASE + [['hold', MD, True], ['tab', 'drive'], ['api', {'ceil_left_mps': 4.2}], ['ntimers']] + LOOK + [
        ['pick', di, blk(c, di, 10)], ['check', '#dr-speed-ceiling', True], ['start'],
        ['hold', MD, False], ['release', MD], ['tick'], ['ntimers'], ['read', 'st_params'], ['el', '#dr-model', 'value'], ['calls']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    assert vals(new, 'ntimers')[0] == [WAIT] and vals(old, 'ntimers')[0] == []
    assert look(old)[:4] == ([], False, True, 4.1)
    assert look(new) == ([1000], True, True, 4.2, False, '(全体 4.5・左旋回中 4.2 m/s)', False)
    assert left_lines(starts(new)[0]['confirm']) == ['速さの天井: ON(全体 4.5・左旋回中 4.2 m/s。前進を弱めるだけ)']
    assert left_lines(starts(old)[0]['confirm']) == ['速さの天井: ON(全体 4.5・左旋回中 4.1 m/s。前進を弱めるだけ)']
    rel_n, rel_o = [x for x in ops(new, 'release')], [x for x in ops(old, 'release')]
    assert rel_n[0]['n'] == rel_o[0]['n'] == 1                     # 一覧の GET は打ち切っていない
    assert vals(new, 'read')[-1]['ceil_left_mps'] == vals(old, 'read')[-1]['ceil_left_mps'] == 4.2   # 返事のあとは前も戻る
    assert tagged(new)[-1] == tagged(old)[-1] and tagged(new)[-1]
    assert 'aborted' not in ' '.join(tn).lower()


@needs_node
def test_status_hang_while_idle_is_cut(tmp_path, cands):
    """走行タブで止まっているとき status の GET が返らない: 今の版は 5 秒で打ち切って「つながっていません」・▶ を断る・1 秒後に
    また問い合わせる(通れば同期して 4.2)。直す前の写しは輪が止まり、▶ が押せて 4.1 の文のまま(返事が来るまで)"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['calls'], ['hold', ST, True], ['tick'], ['ntimers'], ['api', {'ceil_left_mps': 4.2}],
             ['tick'], ['ntimers'], ['eval', 'state.drive.paramsLoaded'], ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-message', 'html'],
             ['confirm', False], ['start'], ['hold', ST, False], ['tick'], ['ntimers'], ['read', 'st_params'],
             ['el', '#dr-ceil-spec', 'text'], ['el', '#btn-dr-start', 'disabled'], ['release', ST], ['calls']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    tn_, to_ = vals(new, 'ntimers'), vals(old, 'ntimers')
    assert tn_[0] == [WAIT] and to_[0] == []                     # 今: GET を待つ間は打ち切りの 5 秒だけ
    assert tn_[1] == [1000] and to_[1] == []                     # 打ち切ったあと 1 秒後にまた / 前: 止まったまま
    n, o = tagged(new), tagged(old)
    assert n[:2] == [False, True] and LOST in n[2]
    assert o[:2] == [True, False] and LOST not in o[2]
    assert starts(new)[0]['confirm'] == '' and starts(old)[0]['confirm'].startswith('AI走行を開始します。')   # 今は確認の文の前に断る
    assert tn[-1].startswith(LOST) and all(x['sent'] is None for x in starts(new) + starts(old))   # どちらも送っていない(前はキャンセル)
    assert tn_[2] == [1000] and vals(new, 'read')[-1]['ceil_left_mps'] == 4.2 and n[3:5] == ['(全体 4.5・左旋回中 4.2 m/s)', False]
    assert to_[2] == [] and vals(old, 'read')[-1]['ceil_left_mps'] == 4.1 and o[3] == '(全体 4.5・左旋回中 4.1 m/s)'
    assert ops(new, 'release')[0]['n'] == 0 and ops(old, 'release')[0]['n'] == 1   # 今は打ち切り済み / 前は返事を待っていた
    st_n = [x for x in vals(new, 'calls')[-1] if x[1] == ST]
    assert st_n[0][2] is None and st_n[1][2] == 200


@needs_node
def test_status_body_stall_is_cut(tmp_path, cands):
    """status の返事の頭は来たが本文が来ない: 今の版は 5 秒で打ち切って失敗(api は本文なしの null を返すので、それも失敗にする)"""
    c, di = cands
    steps = [['tab', 'drive'], ['tick'], ['hold_body', ST, True], ['tick'], ['ntimers'], ['tick'], ['ntimers'],
             ['eval', 'state.drive.paramsLoaded'], ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-message', 'html'],
             ['hold_body', ST, False], ['tick'], ['eval', 'state.drive.paramsLoaded'], ['el', '#btn-dr-start', 'disabled']]
    (new, tn), (old, to) = both(tmp_path, c, steps)
    assert vals(new, 'ntimers') == [[WAIT], [1000]] and vals(old, 'ntimers') == [[], []]
    n, o = tagged(new), tagged(old)
    assert n[:2] == [False, True] and LOST in n[2] and n[3:] == [True, False]
    assert o[:2] == [True, False] and LOST not in o[2]


# 走行中・走行を始めた画面(token あり)は前とまったく同じ(通信の全部・タイマー・トースト・文)
RUN = [['tab', 'drive'], ['tick'], ['start'], ['tick'], ['tick'], ['calls']]
RUNNING = {
    'GET が止まる(走行中・token)': RUN + [['hold', ST, True], ['tick'], ['ntimers'], ['tick'], ['tick'], ['ntimers'], ['calls'],
                                       ['hold', ST, False], ['release', ST], ['tick'], ['ntimers'], ['calls']],
    '一覧の GET が失敗(走行中)': RUN + [['tab', 'live'], ['tick'], ['models_fail', True], ['tab', 'drive'], ['models_fail', False],
                                     ['tick'], ['tick'], ['ntimers'], ['calls']],
    '一覧の GET が止まる(走行中)': RUN + [['tab', 'live'], ['tick'], ['hold', MD, True], ['tab', 'drive'], ['tick'], ['tick'],
                                       ['calls'], ['hold', MD, False], ['release', MD], ['tick'], ['ntimers'], ['calls']],
    '▶ のあとの GET が止まる(token・まだ止まっている表示)': [['tab', 'drive'], ['tick'], ['calls'], ['hold', ST, True], ['start'],
                                                         ['ntimers'], ['tick'], ['tick'], ['ntimers'], ['calls'], ['hold', ST, False],
                                                         ['release', ST], ['tick'], ['ntimers'], ['calls']],
    'ほかの画面が走行中に GET が止まる(token なし)': [['tab', 'drive'], ['tick'], ['running', True], ['tick'], ['tick'], ['calls'],
                                                 ['hold', ST, True], ['tick'], ['ntimers'], ['tick'], ['tick'], ['ntimers'], ['calls'],
                                                 ['hold', ST, False], ['release', ST], ['tick'], ['ntimers'], ['calls']],
    'サーバーが落ちる(走行中)': RUN + [['down_all', True], ['tick'], ['tick'], ['ntimers'], ['tab', 'live'], ['tab', 'drive'], ['tick'],
                                    ['ntimers'], ['down_all', False], ['tick'], ['ntimers'], ['calls']],
}


RUNNING_IDS = ['hold_token', 'models_fail_running', 'models_hang_running', 'hold_after_start', 'hold_other_page', 'down_running']


@needs_node
@pytest.mark.parametrize('name', list(RUNNING), ids=RUNNING_IDS)
def test_running_page_polls_exactly_as_before(tmp_path, cands, name):
    c, di = cands
    tail = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, RUNNING[name] + tail)
    assert vals(new, 'calls') == vals(old, 'calls'), name           # 通信の全部(GET の回数・token・順)が同じ
    assert vals(new, 'posts') == vals(old, 'posts') and tn == to and tagged(new) == tagged(old)
    if name.startswith('一覧の GET が止まる'):
        assert vals(new, 'ntimers')[-1] == vals(old, 'ntimers')[-1]    # 一覧の期限のタイマーは走行中は何もしない(鳴っても同じ)
    else:
        assert vals(new, 'ntimers') == vals(old, 'ntimers'), name   # 打ち切りのタイマーも掛けない
    if name.startswith('ほかの画面'):
        assert not starts(new) and not any('?token=' in x[1] for c in vals(new, 'calls') for x in c)
        assert any(x == '/api/drive/status' for c in vals(new, 'calls') for _, x, _ in c)
    else:
        assert starts(new)[0]['status'] == 200 and any('?token=' in x[1] for x in vals(new, 'calls')[-1])


@needs_node
def test_first_open_with_models_fail(tmp_path, cands):
    """走行タブを初めて開いたときに一覧の GET が失敗: 直す前は輪が始まらず「読み込んでいます」のまま(タブを替えるまで)。今の版は
    status を読み(1 秒ごと)、モデルが無いので ▶ は押せない(タブを替えると一覧を読み直す。前と同じ)。
    そのあと直す 7-2 回目(w2-ui-w2-ui-r72-ui-2。tests/test_ui_fix7_r2.py)からは、一覧を読めていない間に ▶ を押したときの文が
    「モデルを選んでください」ではなく「モデルの一覧を読めていません…」(JF_UI6R2_APP で 6-2 回目のあとの版の写しを流すときは前の文)"""
    c, di = cands
    steps = [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['eval', 'driveTimer !== null'],
             ['eval', 'state.drive.paramsLoaded'], ['el', '#btn-dr-start', 'disabled'], ['start'], ['el', '#toast', 'text'],
             ['tab', 'live'], ['tab', 'drive'], ['eval', 'driveTimer !== null'], ['eval', 'state.drive.paramsLoaded'],
             ['el', '#dr-model', 'value']]
    (new, tn), (old, to) = both(tmp_path, c, steps, no_init=True)
    n, o = tagged(new), tagged(old)
    assert n[:3] == [True, True, True] and o[:2] == [False, False]
    from test_ui_fix7_r2 import MISS                    # 7-2 回目の文(読み込みの順の都合でここで)
    assert o[3] == 'モデルを選んでください' and n[3] == ('モデルを選んでください' if os.environ.get('JF_UI6R2_APP') else MISS)
    assert n[4:6] == o[4:6] == [True, True] and n[6] == o[6] and n[6]
    assert all(x['sent'] is None for x in starts(new) + starts(old))


# ------------------------------------------------------------------ 一覧が通るときは前とまったく同じ
NORMAL = {
    'ライブ ⇄ 走行タブ・curl・ブロック・天井・▶': BASE + [['tab', 'drive'], ['tick'], ['tick'], ['api', {'ceil_left_mps': 4.2}], ['tick'],
                                                ['ntimers'], ['calls'], ['pick', 0, 0], ['check', '#dr-speed-ceiling', True],
                                                ['start'], ['tick'], ['tick'], ['ntimers'], ['calls'], ['click', '#btn-dr-stop'],
                                                ['tick'], ['tab', 'live'], ['tick'], ['ntimers'], ['tab', 'drive'], ['tick'], ['calls']],
    '起き直し(1 秒の問い合わせで知らせ)': BASE + [['tab', 'drive'], ['tick'], ['reboot', 2000.0, RESET], ['tick'], ['tick'], ['ntimers'],
                                            ['start'], ['tick'], ['calls']],
    '問い合わせの失敗(fix5 t1)': BASE + [['tab', 'drive'], ['tick'], ['down', True], ['tick'], ['ntimers'], ['start'], ['down', False],
                                     ['tick'], ['ntimers'], ['start'], ['calls']],
}


@needs_node
@pytest.mark.parametrize('name', list(NORMAL), ids=['tabs_curl_block_start', 'reboot_noticed', 'poll_fail_fix5'])
def test_normal_paths_exactly_as_before(tmp_path, cands, name):
    """一覧も status も通るとき(と、fix5 の「表示の問い合わせが失敗」)は、通信の全部・タイマー・トースト・文・送った物・確認の文が
    直す前の写しとまったく同じ(一覧の期限・打ち切りのタイマーは返事で消える)"""
    c, di = cands
    steps = [s if s[0] != 'pick' else ['pick', di, blk(c, di, 10)] for s in NORMAL[name]]
    tail = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-phase', 'text'], ['posts']]
    (new, tn), (old, to) = both(tmp_path, c, steps + tail, server=PRE_A)
    assert vals(new, 'calls') == vals(old, 'calls') and vals(new, 'ntimers') == vals(old, 'ntimers'), name
    assert tn == to and tagged(new) == tagged(old) and vals(new, 'posts') == vals(old, 'posts')
    assert [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in starts(new)] == \
           [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in starts(old)]


@needs_node
def test_fix6_r1_scenes_and_flows_exactly_as_before(tmp_path, cands):
    """直す 6-1 回目の起き直しの場面(A〜I)と、10/26〜27 の画面の流れ(test_ui_fix3 の _flows)と版で断られる流れは、
    通信の全部・トースト・文・送った物・確認の文が直す前の写しとまったく同じ"""
    c, di = cands
    flows = {('scene', k): (F6.setup_steps(c, di) + v[0] + F6.TAIL, v[1], F6.MORNING) for k, v in F6.SCENES.items()}
    for k, (st, sp) in dict(_flows(c, di)).items():
        flows[('flow', k)] = ([['poll']] + st + [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['posts']], {}, server_params(**sp))
    flows[('flow', '版で断られる')] = ([['poll'], ['start'], ['start'], ['el', '#toast', 'text'], ['posts']], {'server_client_version': 10},
                                  server_params())
    for i, (key, (st, extra, sp)) in enumerate(flows.items()):
        kw = dict({'server_client_version': 8}, **extra)
        new, tn = run_sim(tmp_path, c, st + [['calls'], ['ntimers']], name='n%d.json' % i, server=sp, **kw)
        old, to = run_sim(tmp_path, c, st + [['calls'], ['ntimers']], app=BEFORE / 'app.js', name='o%d.json' % i, server=sp, **kw)
        assert tn == to and tagged(new) == tagged(old) and vals(new, 'posts') == vals(old, 'posts'), key
        assert vals(new, 'calls') == vals(old, 'calls'), key
        assert vals(new, 'ntimers') == vals(old, 'ntimers'), (key, vals(new, 'ntimers'), vals(old, 'ntimers'))
        assert [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in starts(new)] == \
               [(s['sent'], s['confirm'], s['boot_at'], s.get('status')) for s in starts(old)], key
