"""画面の係の直し(2026-10-27 fix5。反証 4 回目の指摘 t1・t2・t3。~/jetformer_reports/fix_20261026/fix4/findings_round4.md)の試験。

  t1: サーバーが起き直した直後、画面が同期する前に ▶ を押すと、確認の文は起き直す前の値(見張り ON・左 4.2・200 µs)のまま、
    車は既定の値(見張り OFF・左 4.1・250 µs)で走った(確認の文を読んでいる間に起き直したときも)。直し:
    (a) app.js: 表示の問い合わせ(pollDrive)が失敗したら paramsLoaded を戻し、次に同期するまで ▶・基準の設定・ブロックを断る
        (「サーバーにつながっていません」。▶ も押せなくする)。画面を開いた直後の文は前と同じ。
    (b) app.js: 走行開始の body に画面が最後に読んだ status の boot_at。server.py(_boot_refusal): 今の driver.boot_at と違う・
        無いなら 409。断られたら画面はすぐ同期する。画面の版 8 → 9(DRIVE_CLIENT_VERSION と app.js の client・DRIVE_CLIENT)。
  t2: スイッチ(postDriveParams)の送信の途中に古い問い合わせの返事が来ると、スイッチが前の本の値に約 1 秒戻り、その間の ▶ で
    天井だけの本が自動合わせ ON で・発進の前進 ON の本が OFF で走った。直し: postDriveParams に項目ごとの送信中の数(drPostN)。
    setDriveParams はその項目を戻さない。
  t3: approvals 10 の (b)(リカバリーの下がる量を小さくする)をスライダーで入れると、ブロックのたびに 0.4 秒・300 µs に戻った。
    直し: 入れ方は bench_candidates.json の baseline(approvals 10 に書いた。画面の動きは変えない)。▶ の確認の文に、0.4・300 で
    ないときだけ「リカバリーの下がる量」の行。② runs_1103 の KEYS に rev_s・rev_us、表の下の行と回の中の違いの印。

本物の app.js(今の版と直す前の写し tests/fixtures/fix5_ui_before/。sha256 を照らす)を node の vm で(tests/fixtures/app_sim_fix5.js =
反証 4 回目の sim4.js + 偽のサーバーの起動の照らし)。本物の drive.Driver(装置なし・走行ループは動かさない)と server.py の関数
(import だけ。起動しない・driver.start は偽物)。車・ブラウザ・jetformer-web には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix5.py
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from jetformer import drive
from jetformer.web import bench
from jetformer.web import server as srv

from test_ui_fix3 import _flows
from test_vbat_desc import fake_start, tracker, feed      # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
BEFORE = ROOT / 'tests' / 'fixtures' / 'fix5_ui_before'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix5.js'
RUNS = ROOT / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py'
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
RESET = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
UP = '#btn-dr-tg-up'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_before_copies():
    sums = dict(reversed(x.split()) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    assert set(sums) == {'app.js', 'index.html', 'server.py', 'runs_1103.py'}
    for name, h in sums.items():
        assert _sha(BEFORE / name) == h, name
    assert sums['app.js'].startswith('2294bfd96ea8') and sums['server.py'].startswith('6330f0853bbc')   # 10/27 04:26・01:23 の版
    old = (BEFORE / 'app.js').read_text(encoding='utf-8')
    assert 'drPostN' not in old and 'boot_at:' not in old and 'const body = { client: 8,' in old
    assert '_boot_refusal' not in (BEFORE / 'server.py').read_text(encoding='utf-8')
    assert (STATIC / 'index.html').read_bytes() == (BEFORE / 'index.html').read_bytes()      # index.html は変えていない


@pytest.fixture(scope='module')
def cands():
    c = bench.load_candidates()
    days = [i for i, d in enumerate(c.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks')]
    if not days:
        pytest.skip('data/bench_candidates.json に 2026-11-03 のブロックがありません')
    return dict(c, today='2026-11-03'), days[0]


def blk(c, di, no):
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def server_params(**over):
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    sp.update(over)
    return sp


def run_sim(tmp_path, c, steps, server=None, app=None, name='input.json', **extra):
    inp = tmp_path / name
    models = sorted({cd['run'] for cd in c['candidates'] if cd.get('run')})
    d = {'bench': c, 'server_params': server if server is not None else server_params(), 'steps': steps,
         'server_client_version': 8, 'confirm': True, 'boot_at': 1000.0, 'models': models,
         'script_src': 'http://x/static/app.js?v=1793667600'}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app or STATIC / 'app.js'), str(STATIC / 'index.html'), str(inp)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def starts(res):
    return [x for x in res if x['op'] == 'start']


def reads(res, what):
    return [x['value'] for x in res if x['op'] == 'read' and x['what'] == what]


def els(res):
    return [x['value'] for x in res if x['op'] == 'el']


def evals(res):
    return [x['value'] for x in res if x['op'] == 'eval']


def posts(res):
    return [x['value'] for x in res if x['op'] == 'posts']


def no_ver(body):
    return {k: v for k, v in (body or {}).items() if k not in ('client', 'build', 'boot_at')}


class _Rec:
    recording = False

    def add_event(self, *a, **k):
        return True


@pytest.fixture()
def real_driver(monkeypatch, tmp_path):
    """本物の drive.Driver(装置なし・設定ファイルは tmp・走行ループは動かさない)を作る関数"""
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    monkeypatch.setattr(drive, 'model_info', lambda name: {'trained_modes': [5], 'drivable': True, 'postprocess': None, 'name': name})

    def make(params=None):
        d = drive.Driver(hub=None, recorder=_Rec())
        d._run = lambda *a, **k: None
        if params is not None:
            d.params = dict(params)
        return d
    return make


def replay(d, plist, check_boot=True):
    """sim の POST を本物の Driver に同じ順で入れる。戻り値 [(path, 本物, sim)]。走行開始は本物の server._boot_refusal を通す
    (check_boot False = 直す前のサーバー。照らさない。直す前の写しの画面は boot_at を送らない)"""
    out = []
    for path, body, status in plist:
        if path == '/api/drive/params':
            try:
                d.set_params(body)
                got = 200
            except ValueError:
                got = 400
        elif path == '/api/drive/start':
            if check_boot and srv._boot_refusal(body, d.boot_at):
                got = 409
            else:
                d.start(body['run'], body['mode_id'], params=body['params'], description=body.get('description', ''),
                        dry_run=body.get('dry_run', False))
                d.thread.join(1)
                got = 200
        else:
            continue
        out.append((path, got, status))
    return out


# ================================================================ t1 起き直した直後・確認の文を読んでいる間の起き直し
MORNING = server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, launch_no_kick=1, ceil_left_mps=4.2, launch_us=200.0)
STALE = ('後ろの見張り: ON', '残す: ON', '補助も出さない: ON', '左旋回中 4.2', '前進 200 µs')


def setup(c, di):
    """朝: 見張り・残す・補助なし・左 4.2・200 µs(サーバー)。3-4(ブロック 15)を選び、ゲイン 0.92・天井・発進の前進 ON。▶ はキャンセル"""
    return [['pick', di, blk(c, di, 15)], ['click', UP], ['click', UP], ['check', '#dr-speed-ceiling', True],
            ['check', '#dr-launch-boost', True], ['poll'], ['confirm', False], ['start'], ['confirm', True], ['posts']]


def stale(confirm):
    return [w for w in STALE if w in confirm]


@needs_node
@pytest.mark.parametrize('case', ['no_sync', 'dialog_spans_restart'])
def test_t1_before_copy_runs_with_other_values_than_the_confirm(tmp_path, cands, case):
    """直す前の写し(反証 4 回目 S1 の再現): 走行開始が通り、確認の文には起き直す前の 5 行、サーバー(起き直したあと)は既定の値"""
    c, di = cands
    tail = {'no_sync': [['reboot', 2000.0, RESET]],
            'dialog_spans_restart': [['down', True], ['realpoll'], ['realpoll'], ['reboot_on_confirm', 2000.0, RESET]]}[case]
    res, _ = run_sim(tmp_path, c, setup(c, di) + tail + [['start'], ['read', 'server']], server=MORNING, app=BEFORE / 'app.js')
    s = starts(res)[1]
    assert s['status'] == 200 and s['refused'] is False and s['boot_at'] is None and s['client'] == 8
    assert stale(s['confirm']) == list(STALE)
    sv = reads(res, 'server')[0]
    assert (sv['rec_back_guard'], sv['rec_hold_keep'], sv['launch_no_kick'], sv['ceil_left_mps'], sv['launch_us']) == (0, 0, 0, 4.1, 250.0)


@needs_node
def test_t1a_poll_failure_refuses_start_bench_and_pick_until_next_sync(tmp_path, cands):
    """(a) 表示の問い合わせが失敗したら、次に同期するまで ▶・基準の設定・ブロックを断る(何も送らない)。▶ は押せなくなり、走行タブに
    「つながっていません」。つながって同期したら(起き直したサーバー)、確認の文は今のサーバーの値で、そのまま走る"""
    c, di = cands
    steps = setup(c, di) + [
        ['down', True], ['realpoll'], ['el', '#btn-dr-start', 'disabled'], ['el', '#dr-message', 'html'],
        ['start'], ['baseline'], ['pick', di, blk(c, di, 16)], ['read', 'applied'], ['read', 'toasts'], ['posts'],
        ['down', False], ['reboot', 2000.0, RESET], ['realpoll'], ['el', '#btn-dr-start', 'disabled'],
        ['start'], ['read', 'server'], ['posts']]
    res, toasts = run_sim(tmp_path, c, steps, server=MORNING, server_boot_check=True)
    dis0, msg, dis1 = els(res)
    assert dis0 is True and 'サーバーにつながっていません' in msg and dis1 is False
    s_lost, s_ok = starts(res)[1:]
    assert s_lost['sent'] is None and s_lost['confirm'] == ''                       # 確認の文も出さない(送らない)
    assert reads(res, 'applied')[0] == {'day': di, 'i': blk(c, di, 15)}            # 基準の設定・ブロック 16 は入れなかった
    lost = [t for t in reads(res, 'toasts')[0] if 'サーバーにつながっていません' in t]
    assert len(lost) == 3                                                          # ▶・基準の設定・ブロック
    assert posts(res)[1] == []                                                     # 落ちている間は何も送っていない
    assert s_ok['status'] == 200 and s_ok['boot_at'] == 2000.0 and stale(s_ok['confirm']) == []
    sv = reads(res, 'server')[0]
    assert sv['rec_back_guard'] == 0 and sv['speed_ceiling'] == s_ok['sent']['speed_ceiling'] == 0
    assert any('起動し直しました' in t for t in toasts)


@needs_node
def test_t1a_first_open_keeps_the_old_message(tmp_path, cands):
    """画面を開いた直後(一度も同期していない)に問い合わせが失敗しているときの文は前と同じ「読み込んでいます」。同期したら始まる"""
    c, di = cands
    steps = [['start'], ['read', 'toasts'], ['poll'], ['pick', di, blk(c, di, 1)], ['start']]
    res, _ = run_sim(tmp_path, c, steps, status_fail_first=2, server_boot_check=True)
    s0, s1 = starts(res)
    assert s0['sent'] is None and any('サーバーの設定を読み込んでいます' in t for t in reads(res, 'toasts')[0])
    assert not any('つながっていません' in t for t in reads(res, 'toasts')[0])
    assert s1['status'] == 200 and s1['boot_at'] == 1000.0


@needs_node
@pytest.mark.parametrize('case', ['no_sync', 'dialog_only'])
def test_t1b_stale_boot_is_refused_and_page_resyncs(tmp_path, cands, case, real_driver):
    """(b) 起き直したあと同期せずに ▶(no_sync)・確認の文を読んでいる間に起き直した(dialog_only。問い合わせは失敗していないので (a) では
    断れない): 画面は見た boot_at(1000)を送り、サーバーは 409。画面はすぐ同期し(boot_at 2000)、もう一度の ▶ は今の値の確認の文で走る。
    本物の server._boot_refusal と drive.Driver に同じ POST を入れても同じ(確認の文 = 走る値)"""
    c, di = cands
    tail = [['reboot', 2000.0, RESET]] if case == 'no_sync' else [['reboot_on_confirm', 2000.0, RESET]]
    steps = setup(c, di) + tail + [['start'], ['eval', '__T.state.drive.bootAt'], ['start'], ['read', 'server'], ['posts']]
    res, toasts = run_sim(tmp_path, c, steps, server=MORNING, server_boot_check=True)
    s1, s2 = starts(res)[1:]
    assert s1['boot_at'] == 1000.0 and s1['status'] == 409 and s1['refused'] is True and stale(s1['confirm']) == list(STALE)
    assert evals(res) == [2000.0]
    assert s2['boot_at'] == 2000.0 and s2['status'] == 200 and stale(s2['confirm']) == []
    assert any('起動し直しました' in t for t in toasts)
    # 本物: 起き直したあとの Driver(boot_at は本物の時刻)。画面の 1000 → 前の起動、2000 → 今の起動
    d2 = real_driver()
    plist = [(p, dict(b, boot_at={1000.0: d2.boot_at - 60.0, 2000.0: d2.boot_at}[b['boot_at']]) if p == '/api/drive/start' else b, st)
             for p, b, st in posts(res)[1]]
    assert [x for x in replay(d2, plist) if x[0] == '/api/drive/start'] == [('/api/drive/start', 409, 409), ('/api/drive/start', 200, 200)]
    assert (d2.params['rec_back_guard'], d2.params['ceil_left_mps'], d2.params['launch_us']) == (0, 4.1, 250.0)
    assert ('後ろの見張り: ON' in s2['confirm']) is False and d2.params['speed_ceiling'] == s2['sent']['speed_ceiling']


@needs_node
def test_t1_old_server_without_boot_at_still_starts(tmp_path, cands):
    """再起動の前のサーバー(10/26 07:41 起動。status に boot_at が無く、走行開始で照らさない): 今の画面は boot_at null を送り、
    今まで通り始まる。問い合わせが失敗している間に断るのは同じ"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 1)], ['start'], ['running', False], ['down', True], ['realpoll'], ['start'],
             ['down', False], ['realpoll'], ['start']]
    res, _ = run_sim(tmp_path, c, steps, boot_at=None, server_boot_check=True)
    s0, s1, s2 = starts(res)
    assert s0['status'] == 200 and s0['boot_at'] is None and s0['client'] == 9
    assert s1['sent'] is None and s2['status'] == 200


def test_t1b_boot_refusal_rules():
    b = 1793667600.123
    assert srv._boot_refusal({}, None) is None and srv._boot_refusal({'boot_at': 5}, None) is None     # 作り物の driver は見ない
    assert srv._boot_refusal({'boot_at': b}, b) is None and srv._boot_refusal({'boot_at': b + 0.0004}, b) is None
    assert srv._boot_refusal({'boot_at': repr(b)}, b) is None                                          # 文字の数でもよい
    for seen in (b - 3600, b + 5, b + 0.001, True):
        why = srv._boot_refusal({'boot_at': seen}, b)
        assert why.startswith('管理画面のサーバーが ') and 'に起動し直しました(この画面が読んだのは ' in why and '手順書 7 章' in why
    for seen in (None, 'x', [], {}):
        assert 'この画面は今のサーバーの設定をまだ読んでいません' in srv._boot_refusal({'boot_at': seen}, b)
    for seen in (float('nan'), float('inf'), 1e300):                                                 # 落ちない
        assert 'に起動し直しました(この画面が読んだのは ? に' in srv._boot_refusal({'boot_at': seen}, b)


def test_t1b_api_drive_start_checks_boot_after_version(tracker, fake_start):
    """本物の api_drive_start(driver.start は偽物): boot_at が違う・無い → 409(driver.start を呼ばない)。版の照らしが先"""
    feed(tracker, 97.0, 100.0, 8.3)
    body = {'client': 9, 'boot_at': srv.driver.boot_at, 'run': bench.B_RUN, 'mode_id': 5, 'description': 'AI走行 B', 'dry_run': True,
            'record': False, 'params': {}}
    for bad in ({'boot_at': srv.driver.boot_at - 30.0}, {'boot_at': None}, {'boot_at': 'x'}):
        with pytest.raises(HTTPException) as e:
            srv.api_drive_start(dict(body, **bad))
        assert e.value.status_code == 409 and '起動し直しました' in e.value.detail, bad
    b2 = dict(body)
    del b2['boot_at']
    with pytest.raises(HTTPException) as e:
        srv.api_drive_start(b2)
    assert 'まだ読んでいません' in e.value.detail
    with pytest.raises(HTTPException) as e:
        srv.api_drive_start(dict(body, client=8))
    assert '版 8・走行を始められるのは版 9 から' in e.value.detail
    assert fake_start == []
    assert srv.api_drive_start(body)['ok'] is True and len(fake_start) == 1


def test_versions_match_and_page_sends_boot_at():
    src = (ROOT / 'jetformer' / 'web' / 'server.py').read_text(encoding='utf-8')
    js = (STATIC / 'app.js').read_text(encoding='utf-8')
    assert int(re.search(r'^DRIVE_CLIENT_VERSION = (\d+)', src, re.M).group(1)) == 9 == srv.DRIVE_CLIENT_VERSION
    assert re.findall(r'const body = \{ client: (\d+),', js) == ['9'] and re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['9']
    assert 'boot_at: state.drive.bootAt ?? null' in js and js.count("'/api/drive/start'") == 1
    fn = src[src.index('def api_drive_start'):src.index('if trainer.running:', src.index('def api_drive_start'))]
    assert fn.index('_client_version(body) < DRIVE_CLIENT_VERSION') < fn.index('_page_build(body)') < fn.index('_boot_refusal(body')
    assert "getattr(driver, 'boot_at', None)" in fn and 'drive.py' not in fn            # drive.py は読むだけ(Driver の boot_at)


# ================================================================ t2 スイッチの送信中に古い問い合わせの返事
def _race(sel, on):
    """クリックの前に表示の問い合わせが 1 本飛んでいて、その返事がスイッチの送信の返事より先に来る(反証 4 回目 S2・S4b)"""
    return [['gstart'], ['defer', True], ['checkn', sel, on], ['gend'], ['el', sel, 'checked'], ['resolve'], ['el', sel, 'checked'],
            ['defer', False], ['start'], ['posts']]


@needs_node
@pytest.mark.parametrize('apply_on_resolve', [False, True])
def test_t2_ceiling_switch_while_old_poll_returns(tmp_path, cands, apply_on_resolve, real_driver):
    """2-4(天井 + 自動合わせ)のあと 2-5(天井だけ。ブロック 11)で天井を ON。直す前: 自動合わせのスイッチが ON に戻り、▶ で 1 を送る。
    今: OFF のまま 0 を送り、確認の文に自動合わせの行が無い。本物の Driver も 0"""
    c, di = cands
    pre = server_params(throttle_gain=0.95, speed_ceiling=1, speed_auto=1)
    steps = [['pick', di, blk(c, di, 11)]] + _race('#dr-speed-ceiling', True)
    steps = [x if x[1:2] != ['#dr-speed-ceiling'] or x[0] != 'el' else ['el', '#dr-speed-auto', 'checked'] for x in steps]
    for app, want in ((BEFORE / 'app.js', 1), (None, 0)):
        d = real_driver(pre)
        res, _ = run_sim(tmp_path, c, steps, server=pre, app=app, defer_posts=False, apply_on_resolve=apply_on_resolve,
                         boot_at=d.boot_at, name='%s.json' % bool(app))
        s = starts(res)[0]
        assert els(res) == [bool(want)] * 2 and s['sent']['speed_auto'] == want and s['sent']['speed_ceiling'] == 1
        assert ('自動合わせ: ON' in s['confirm']) is bool(want)
        assert [x[1] for x in replay(d, posts(res)[0], check_boot=app is None)] == [200, 200]
        assert d.params['speed_auto'] == want


@needs_node
def test_t2_launch_switch_while_old_poll_returns(tmp_path, cands, real_driver):
    """3-3 OFF のあとの 3-4(ブロック 15)で発進の前進を ON。直す前: スイッチが OFF に戻り、▶ で 0 を送る(ON の本が OFF で走る)。今: 1"""
    c, di = cands
    pre = server_params(throttle_gain=1.0, speed_ceiling=1, launch_boost=0)
    steps = [['pick', di, blk(c, di, 15)], ['check', '#dr-speed-ceiling', True]] + _race('#dr-launch-boost', True)
    for app, want in ((BEFORE / 'app.js', 0), (None, 1)):
        d = real_driver(pre)
        res, _ = run_sim(tmp_path, c, steps, server=pre, app=app, boot_at=d.boot_at, name='%s.json' % bool(app))
        s = starts(res)[0]
        assert els(res) == [bool(want)] * 2 and s['sent']['launch_boost'] == want
        assert any(x.startswith('発進の前進: ON') for x in s['confirm'].split('\n')) is bool(want)
        assert all(x[1] == 200 for x in replay(d, [p for ps in posts(res) for p in ps], check_boot=app is None))
        assert d.params['launch_boost'] == want


@needs_node
def test_t2_flag_switch_while_old_poll_returns(tmp_path, cands):
    """補助のスイッチ(DR_FLAGS。発進の補助を OFF)も同じ: 直す前は ON に戻って 1 を送る。今は 0"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 1)], ['poll']] + _race('#dr-start-kick', False)
    for app, want in ((BEFORE / 'app.js', 1), (None, 0)):
        res, _ = run_sim(tmp_path, c, steps, app=app, name='%s.json' % bool(app))
        assert els(res) == [bool(want)] * 2 and starts(res)[0]['sent']['start_kick'] == want


@needs_node
def test_t2_failed_send_still_reverts_and_counter_returns(tmp_path, cands):
    """送れなかったスイッチは今まで通り押す前に戻る(印も 0 に戻り、次の同期でサーバーの値に合う)。もう一度押せば送る"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 15)], ['fail_next', 1], ['check', '#dr-launch-boost', True], ['el', '#dr-launch-boost', 'checked'],
             ['eval', 'JSON.stringify(drPostN)'], ['poll'], ['el', '#dr-launch-boost', 'checked'],
             ['check', '#dr-launch-boost', True], ['poll'], ['start']]
    res, toasts = run_sim(tmp_path, c, steps)
    assert els(res) == [False, False] and json.loads(evals(res)[0]) == {'launch_boost': 0}
    assert starts(res)[0]['sent']['launch_boost'] == 1 and any('Failed to fetch' in t for t in toasts)


# ================================================================ t3 リカバリーの下がる量(approvals 10 の (b))
REV_LINE = 'リカバリーの下がる量: 0.30 秒・中立+250 µs(10/23・24 は 0.40 秒・300 µs)'


@needs_node
def test_t3_confirm_has_rev_line_only_when_not_usual(tmp_path, cands):
    """スライダーで 0.30 秒・250 µs にして(基準の設定・ブロックを押さずに)▶: 確認の文に行が出る(送る値は前と同じ)。0.4・300 なら行は無い"""
    c, di = cands
    steps = [['slide', '#dr-rev-s', 0.3], ['slide', '#dr-rev-us', 250], ['poll'], ['start']]
    (n, _), (o, _) = (run_sim(tmp_path, c, steps, name='n.json'), run_sim(tmp_path, c, steps, app=BEFORE / 'app.js', name='o.json'))
    a, b = starts(n)[0], starts(o)[0]
    assert (a['sent']['rev_s'], a['sent']['rev_us']) == (0.3, 250) and a['sent'] == b['sent']
    assert REV_LINE in a['confirm'].split('\n') and 'リカバリーの下がる量' not in b['confirm']
    assert [x for x in a['confirm'].split('\n') if x != REV_LINE] == b['confirm'].split('\n')
    n2, _ = run_sim(tmp_path, c, [['pick', di, blk(c, di, 1)], ['start']], name='n2.json')
    assert 'リカバリーの下がる量' not in starts(n2)[0]['confirm']


@needs_node
def test_t3_slider_then_block_goes_back_to_baseline_as_before(tmp_path, cands):
    """反証 4 回目 S4a(スライダーで入れてから 基準の設定 → ブロック → ▶): 今まで通り 0.4・300 に戻して送る(画面の動きは変えない。
    approvals 10 に「スライダーでは入れない」と書いた)。戻したことは基準の設定の「変えた所」に出る"""
    c, di = cands
    steps = [['slide', '#dr-rev-s', 0.3], ['slide', '#dr-rev-us', 250], ['poll'], ['baseline'], ['read', 'note'],
             ['pick', di, blk(c, di, 1)], ['start']]
    for app in (None, BEFORE / 'app.js'):
        res, _ = run_sim(tmp_path, c, steps, app=app, name='%s.json' % bool(app))
        s = starts(res)[0]
        assert (s['sent']['rev_s'], s['sent']['rev_us']) == (0.4, 300) and 'リカバリーの下がる量' not in s['confirm']
        assert '後退の強さ 250 → 300、下がる時間 0.30 → 0.40' in reads(res, 'note')[0]


@needs_node
def test_t3_baseline_route_of_approvals_10b(tmp_path, cands, real_driver):
    """approvals 10 の (b) の入れ方: bench_candidates.json の baseline の rev_s・rev_us を書き換えた(ここでは作り物の一覧)日。朝の
    基準の設定・どのブロックも 0.30・250 を送り、確認の文に行。本物の Driver はその値で走る。計画の照らし合わせ(plan_check)の違いに
    rev は出ない(PLAN_KEYS に無い。ゲインなどの違いは前と同じに出る = approvals 11 の cand=EXP)"""
    c0, di = cands
    c = copy.deepcopy(c0)
    c['baseline']['params'].update(rev_s=0.3, rev_us=250.0)
    steps = [['baseline'], ['read', 'note'], ['check', '#dr-dry', True], ['start'], ['running', False], ['check', '#dr-dry', False]]
    for no, ceil in ((1, False), (6, True), (13, True)):
        steps += [['baseline'], ['pick', di, blk(c, di, no)]] + ([['check', '#dr-speed-ceiling', True]] if ceil else [])
        steps += [['poll'], ['start'], ['running', False]]
    d = real_driver(server_params())
    res, _ = run_sim(tmp_path, c, steps + [['posts']], boot_at=d.boot_at, server_boot_check=True)
    assert '後退の強さ 300 → 250、下がる時間 0.40 → 0.30' in reads(res, 'note')[0]
    sts = starts(res)
    assert len(sts) == 4 and all((s['sent']['rev_s'], s['sent']['rev_us']) == (0.3, 250) for s in sts)
    assert all(REV_LINE in s['confirm'].split('\n') for s in sts[1:])           # dry(推論の確認)は確認の文を出さない
    assert all(x[1] == x[2] == 200 for x in replay(d, posts(res)[0]))
    assert (d.params['rev_s'], d.params['rev_us']) == (0.3, 250.0)
    for s in sts[1:]:
        _, diffs = bench.plan_check(s['desc'], s['run'], 5, s['sent'], data=c)
        _, diffs0 = bench.plan_check(s['desc'], s['run'], 5, dict(s['sent'], rev_s=0.4, rev_us=300), data=c0)
        assert not any('rev_' in x for x in diffs) and diffs == diffs0, (diffs, diffs0)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _runs_main(path, raw, name):
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
    return m, out[:-1], json.loads(out[-1])


def _rec(raw, name, ai_no, block, tune, params):
    base = {'steer_gain': 1.0, 'throttle_gain': 0.9, 'lookahead': 3, 'tof_guard': 1, 'zone_caps': 0, 'start_kick': 1,
            'speed_auto': 0, 'speed_ceiling': 0, 'ceil_left_mps': 4.1, 'launch_boost': 0, 'launch_us': 250}
    d = raw / name
    (d / ('%s-0001' % name)).mkdir(parents=True)
    desc = 'ai_no=%d bench=x day=2026-11-03 block=%d cand=EXP tune=%s battery=#A V_before=8.30 memo=' % (ai_no, block, tune)
    (d / 'log.json').write_text(json.dumps({'description': desc, 'drive': {
        'run': bench.B_RUN, 'dry_run': False, 'params': dict(base, **params)}}, ensure_ascii=False), encoding='utf-8')


@pytest.fixture()
def round1_blocks():
    b = json.loads((ROOT / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
    blocks = {x['block']: x for d in b['days'] if d['date'] == '2026-11-03' for x in d['blocks']}
    if not all(blocks.get(k, {}).get('run_id') == '1-%d' % k for k in (1, 2, 3)):
        pytest.skip('11/3 の回 1 のブロックが前提と違う')


def test_t3_runs_1103_rev_line_and_round_mark(tmp_path, round1_blocks):
    raw = tmp_path / 'raw'
    _rec(raw, '20261103-090000', 1, 1, '1-1', {'rev_s': 0.3, 'rev_us': 250.0})
    _rec(raw, '20261103-091000', 2, 2, '1-2', {'rev_s': 0.3, 'rev_us': 250.0})
    _rec(raw, '20261103-092000', 3, 3, '1-3', {'rev_s': 0.4, 'rev_us': 300.0})
    m, table, rows = _runs_main(RUNS, raw, 'runs_fix5_a')
    assert m.KEYS[-2:] == ('rev_s', 'rev_us') and m.REV_USUAL == (0.4, 300.0)
    by = {r['run_id']: r for r in rows}
    assert by['1-1']['params']['rev_s'] == 0.3 and by['1-3']['params']['rev_us'] == 300.0
    assert by['1-1']['diffs'] == [] and by['1-2']['diffs'] == []
    assert by['1-3']['diffs'] == ['回 1 の下がる量が 1-1(No.1)と違う: rev_s 0.4(1-1 は 0.3)・rev_us 300(1-1 は 250)'
                                  '(基準の設定・ブロックで基準の値に戻った? approvals 10)']
    line = [x for x in table if x.startswith('リカバリーの下がる量')]
    assert line == ['リカバリーの下がる量(rev_s・rev_us。log.json の走行開始の値。10/23・24 は 0.40 秒・300 µs。approvals 10): '
                    '1-1(No.1)から 0.30 秒・250 µs → 1-3(No.3)から 0.40 秒・300 µs(前の本と違う)']
    assert table[-1] == '3 本・計画と違う 1 本'


@pytest.mark.parametrize('rev', [{}, {'rev_s': 0.4, 'rev_us': 300.0}])
def test_t3_runs_1103_same_as_before_when_usual(tmp_path, round1_blocks, rev):
    """どれも 0.4・300(か欄が無い古い記録)なら、表は直す前の写しと同じ。JSON の行は params の rev_s・rev_us のほか同じ"""
    raw = tmp_path / 'raw'
    _rec(raw, '20261103-090000', 1, 1, '1-1', dict(rev))
    _rec(raw, '20261103-091000', 2, 2, '1-2', dict(rev, speed_ceiling=1))
    _rec(raw, '20261103-092000', 3, 3, '1-3', dict(rev, throttle_gain=0.95))
    _, t_new, j_new = _runs_main(RUNS, raw, 'runs_fix5_b')
    _, t_old, j_old = _runs_main(BEFORE / 'runs_1103.py', raw, 'runs_fix5_before')
    assert t_new == t_old
    strip = lambda r: dict(r, params={k: v for k, v in r['params'].items() if k not in ('rev_s', 'rev_us')})     # noqa: E731
    assert [strip(r) for r in j_new] == j_old
    assert all(r['params']['rev_s'] == rev.get('rev_s') for r in j_new)


# ================================================================ ほかの流れは前と同じ・11/3 の 1 日
@needs_node
@pytest.mark.parametrize('boot_check', [False, True])
def test_other_flows_send_the_same(tmp_path, cands, boot_check):
    """直す 1 回目の 10 の流れ(tests/test_ui_fix3._flows): 走行開始で送る params・確認の文・説明欄・収集の印・送った POST・トーストは
    直す前の写しと同じ(版 9・boot_at のほか)。サーバーが起動の時刻を照らしても(boot_check)同じ"""
    c, di = cands
    for i, (name, (steps, sp)) in enumerate(_flows(c, di).items()):
        st = steps + [['posts']]
        new, tn = run_sim(tmp_path, c, st, server=server_params(**sp), server_boot_check=boot_check, name='n%d.json' % i)
        old, to = run_sim(tmp_path, c, st, server=server_params(**sp), app=BEFORE / 'app.js', name='o%d.json' % i)
        sn, so = starts(new), starts(old)
        assert len(sn) == len(so) > 0, name
        for a, b in zip(sn, so):
            for k in ('sent', 'confirm', 'confirms', 'refused', 'desc', 'record', 'dry', 'build', 'run', 'status'):
                assert a[k] == b[k], (name, k, a[k], b[k])
            assert (a['client'], b['client']) in ((9, 8), (None, None)) and (a['boot_at'], b['boot_at']) in ((1000.0, None), (None, None))
        norm = lambda r: [[[p, no_ver(bd) if p == '/api/drive/start' else bd, s] for p, bd, s in x] for x in posts(r)]   # noqa: E731
        assert norm(new) == norm(old), name
        assert tn == to, name


# 11/3 の 1 日(手順書 3・4・5 章と bench_candidates.json の after_press。反証 4 回目 review_state/s3_day_flow.py と同じ計画)
# (ブロック, ゲイン, 天井, 自動合わせ, 発進の前進, 押す前の curl, 押したあとの curl, 左, 強さ)
DAY = [(1, 0.92, 0, 0, 0, [], [], 4.1, 250.0), (2, 0.93, 0, 0, 0, [], [], 4.1, 250.0), (3, 0.95, 0, 0, 0, [], [], 4.1, 250.0),
       (4, 0.95, 1, 0, 0, [], [], 4.1, 250.0), (5, 0.95, 0, 0, 0, [], [], 4.1, 250.0),
       (6, 0.97, 1, 0, 0, [], [], 4.1, 250.0), (7, 0.97, 0, 0, 0, [], [], 4.1, 250.0), (8, 1.00, 1, 1, 0, [], [], 4.1, 250.0),
       (9, 1.00, 1, 1, 0, [], [], 4.1, 250.0), (10, 1.00, 1, 0, 0, [], [{'ceil_left_mps': 4.2}], 4.2, 250.0),
       (11, 1.00, 1, 0, 0, [], [{'ceil_left_mps': 4.1}], 4.1, 250.0),
       (12, 1.00, 1, 0, 0, [], [], 4.1, 250.0), (13, 1.00, 1, 0, 1, [], [], 4.1, 250.0),
       (14, 1.01, 1, 0, 0, [{'launch_us': 200}], [], 4.1, 200.0), (15, 1.01, 1, 0, 1, [], [], 4.1, 200.0),
       (16, 1.02, 1, 0, 0, [], [], 4.1, 200.0), (17, 1.02, 1, 0, 1, [], [], 4.1, 200.0),
       (18, 1.02, 1, 0, 0, [], [], 4.1, 200.0), (19, 1.02, 1, 0, 1, [], [], 4.1, 200.0),
       (20, 1.00, 1, 0, 1, [], [], 4.1, 200.0), (21, 1.00, 1, 0, 1, [], [], 4.1, 200.0), (22, 1.02, 1, 0, 1, [], [], 4.1, 200.0)]


def _day_steps(c, di):
    steps = [['poll'], ['baseline'], ['check', '#dr-dry', True], ['start'], ['running', False], ['check', '#dr-dry', False],
             ['baseline'], ['check', '#dr-record', False], ['check', '#dr-launch-boost', True], ['start'], ['running', False],
             ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True], ['poll']]
    plan = [None, None]
    for (no, g, ceil, auto, launch, api_before, api_after, left, us) in DAY:
        n = int(round((g - 0.90) / 0.01))
        s = [['api', b] for b in api_before] + [['baseline'], ['poll'], ['pick', di, blk(c, di, no)], ['poll']]
        s += [['click', UP if n > 0 else '#btn-dr-tg-down']] * abs(n) + [['api', b] for b in api_after]
        s += [['check', sel, True] for sel, on in (('#dr-speed-ceiling', ceil), ('#dr-speed-auto', auto), ('#dr-launch-boost', launch)) if on]
        if no == 1:
            s.append(['slide', '#dr-trim', 15])
        steps += s + [['poll'], ['start'], ['running', False]]
        plan.append({'throttle_gain': g, 'speed_ceiling': ceil, 'speed_auto': auto, 'launch_boost': launch, 'ceil_left_mps': left,
                     'launch_us': us, 'rec_back_guard': 1, 'rec_hold_keep': 1, 'launch_no_kick': 0, 'zone_caps': 0, 'start_kick': 1,
                     'steer_trim_us': 15.0, 'rev_s': 0.4, 'rev_us': 300.0})
    return steps + [['posts']], plan


@needs_node
def test_day_flow_1103_all_24_starts_as_planned(tmp_path, cands, real_driver):
    """手順書の 11/3 の流れ(朝の dry run・発進の前進の確かめ・ブロック 1〜22)を、今の画面・起動の時刻を照らすサーバーで。本物の
    server._boot_refusal と drive.Driver に同じ POST を入れ、走行開始 24 回とも断られず、計画どおりの値で走り、確認の文の行と合う"""
    c, di = cands
    d = real_driver(server_params())
    steps, plan = _day_steps(c, di)
    res, toasts = run_sim(tmp_path, c, steps, boot_at=d.boot_at, server_boot_check=True)
    sts = starts(res)
    assert len(sts) == 24 and all(s['status'] == 200 and s['boot_at'] == d.boot_at and s['client'] == 9 for s in sts)
    k = 0
    for path, body, status in posts(res)[0]:
        if path == '/api/drive/params':
            d.set_params(body)
            continue
        if path != '/api/drive/start':
            continue
        assert srv._boot_refusal(body, d.boot_at) is None
        d.start(body['run'], body['mode_id'], params=body['params'], description=body.get('description', ''), dry_run=body['dry_run'])
        d.thread.join(1)
        want, conf = plan[k], sts[k]['confirm']
        if want is None:
            assert (body['dry_run'], body['record']) == ((True, False) if k == 0 else (False, False))
        else:
            diff = {kk: (vv, d.params[kk]) for kk, vv in want.items() if abs(float(d.params[kk]) - float(vv)) > 1e-9}
            assert diff == {}, (k, diff)
            lines = conf.split('\n')
            assert ('速さの天井: ON(全体 4.5・左旋回中 %.1f m/s。前進を弱めるだけ)' % want['ceil_left_mps'] in lines) is bool(want['speed_ceiling'])
            assert any('自動合わせ: ON' in x for x in lines) is bool(want['speed_auto'])
            assert any(x.startswith('発進の前進: ON(前進 %.0f µs' % want['launch_us']) for x in lines) is bool(want['launch_boost'])
            assert any('後ろの見張り: ON' in x for x in lines) and any('残す: ON' in x for x in lines)
            assert '速度ゲイン %.2f' % want['throttle_gain'] in conf and 'リカバリーの下がる量' not in conf
            assert 'block=%d ' % DAY[k - 2][0] in body['description'] + ' '
        k += 1
    assert k == 24
    assert not any(w in t for t in toasts for w in ('つながっていません', '起動し直しました', '失敗', '古いまま'))
