"""画面とサーバーの直す 2 回目・1 組目(2026-10-27 fix3。反証 2 回目の指摘 r2-r2-state-1・2・3)の試験。

  r2-r2-state-1: 手順書 3 章の朝の順(ベンチマークを 1 つ選ぶ → 基準の設定 → 発進の前進の確かめ ▶)で、基準の設定は説明欄・
    収集の印に触れないので、確かめの走行が公式のブロックの本(block=N cand=B official=1・AI No.1)として記録され、夜の道具 ②・⑤
    が回 1(回 3)を「計画と違う」「満たさない」と読んだ。直し(app.js の benchApply): 基準の設定で、説明欄がベンチマークの雛形
    (bench= と block= がある)なら消す(走行開始は「AI走行 モデル名」。取り消すで戻る)。収集が ON なら注を出す。手順書 3 章も。
  r2-r2-state-2: 試験の選択肢のスイッチ(OPT_SWITCHES)は drAcked を呼ばず、送信の返事より前に出た問い合わせの返事が
    state.drive.st.params を古い値にして、スイッチと走行開始の確認の文を約 1 秒、前の値に戻した。直し: 送信のあと drAcked
    (その項目とそれを要る項目)、renderDrive は since より後に返事が来た試験の選択肢を前の値のままにする。
  r2-r2-state-3: 版 7 のあと app.js を 2 回直したのに版を上げていなかった。直し: 版 8(server.py の DRIVE_CLIENT_VERSION と
    app.js の client)。走行開始で build(その画面が読み込んだ app.js の更新時刻。index が付ける ?v=)も送り、サーバーは今のファイル
    より古ければ断る(版を上げ忘れても古い画面で走らない)。▶ の横に「画面の版 8(app.js MM/DD HH:MM)」(手順書 3 章の印)。

本物の app.js をまるごと node の vm で読み込む(tests/fixtures/app_sim_fix3_r2_1.js。app_sim_fix3_r1_2.js の写しに、走行開始の
説明欄・収集の印・build と script_src を足したもの)。直す前の写しは tests/fixtures/fix3_r2_1_before/(sha256 を照らす)。
サーバーの関数(api_drive_start・plan_check)は本物を import して呼ぶ(起動しない・driver.start は偽物)。車・ブラウザは使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix3_r2_1.py
"""
import hashlib
import json
import os
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
BEFORE = ROOT / 'tests' / 'fixtures' / 'fix3_r2_1_before'
OLD_2025 = ROOT / 'tests' / 'fixtures' / 'fix3_r1_before' / 'app.js'          # 10/26 20:25 の画面(版 7。反証の s3)
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix3_r2_1.js'
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')
G, H, K = '#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-launch-no-kick'


def test_before_copies():
    sums = dict(reversed(x.split()) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    for name, h in sums.items():
        assert hashlib.sha256((BEFORE / name).read_bytes()).hexdigest() == h, name
    assert re.findall(r'const body = \{ client: (\d+),', (BEFORE / 'app.js').read_text(encoding='utf-8')) == ['7']


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
    d = {'bench': c, 'server_params': server if server is not None else server_params(), 'steps': steps,
         'server_client_version': 7, 'confirm': True}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app or STATIC / 'app.js'), str(STATIC / 'index.html'), str(inp)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def old(tmp_path, c, steps, **kw):
    return run_sim(tmp_path, c, steps, app=BEFORE / 'app.js', name='old_' + kw.pop('name', 'input.json'), **kw)


def starts(res):
    return [x for x in res if x['op'] == 'start']


def reads(res, what):
    return [x['value'] for x in res if x['op'] == 'read' and x['what'] == what]


def els(res):
    return [x['value'] for x in res if x['op'] == 'el']


BATTERY_LINE = '説明欄の電池の番号がまだ空欄です(走ったあと、シーンの説明でも直せます)'


def same_start(a, b, desc=True):
    assert a['sent'] == b['sent'] and a['record'] == b['record'] and a['dry'] == b['dry']
    if desc:
        assert a['desc'] == b['desc'] and a['confirm'] == b['confirm']
    else:                               # 説明欄を消した場面: 確認の文は「電池の番号がまだ空欄」の行(雛形の battery=#__)だけ違う
        lines = lambda t: [x for x in t.splitlines() if x.strip() and x != BATTERY_LINE]     # noqa: E731
        assert lines(a['confirm']) == lines(b['confirm'])
    if a['client'] is None:             # 確認でキャンセル(送らない)
        assert b['client'] is None
        return
    assert (a['client'], b['client']) == (8, 7) and a.get('build') == 0 and b.get('build') is None


def same_posts(new, o):
    """送った POST(走行開始は版・build のほか)が同じ"""
    def norm(res):
        out = []
        for x in res:
            if x['op'] != 'posts':
                continue
            out.append([[path, {k: v for k, v in (body or {}).items() if k not in ('client', 'build')} if path == '/api/drive/start'
                         else body, st] for path, body, st in x['value']])
        return out
    assert norm(new) == norm(o)


# ================================================================ r2-r2-state-1 基準の設定で雛形を消す
def morning(c, di, pick_no):
    """手順書 3 章の朝の順(前の版): 読み直し → ベンチマークを 1 つ選ぶ → 基準の設定 → 発進の前進 ON → ▶(CH3 は人側のまま)"""
    return [['read', 'record'], ['read', 'desc'], ['pick', di, blk(c, di, pick_no)], ['read', 'desc'], ['baseline'],
            ['read', 'record'], ['read', 'desc'], ['read', 'note'], ['check', '#dr-launch-boost', True], ['start']]


@needs_node
@pytest.mark.parametrize('pick_no', [1, 13, 19])
def test_state1_morning_check_not_recorded_as_block(tmp_path, cands, pick_no):
    """指摘の場面: 直す前の写しは block=N cand=B official=1 の説明欄で送り、本物の plan_check は違いなし(公式の本のまま)。今は
    基準の設定で雛形を消し、説明欄は「AI走行 モデル名」(plan_check は手を付けない = ベンチマークの本ではない)。送る params・
    確認の文・収集の印は直す前と同じ"""
    c, di = cands
    st = morning(c, di, pick_no)
    new, _ = run_sim(tmp_path, c, st, name='n.json')
    o, _ = old(tmp_path, c, st, name='o.json')
    sn, so = starts(new)[0], starts(o)[0]
    same_start(sn, so, desc=False)
    assert BATTERY_LINE in so['confirm'] and BATTERY_LINE not in sn['confirm']
    assert so['desc'].startswith('bench=') and ('block=%d ' % pick_no) in so['desc'] and 'official=1' in so['desc']   # 再現
    d_old, diffs_old = bench.plan_check(so['desc'], c['baseline']['run'], 5, so['sent'])
    assert diffs_old == [] and bench.get_field(d_old, 'official') == '1' and bench.get_field(d_old, 'cand') == 'B'
    run_b = c['baseline']['run']
    assert sn['desc'] == 'AI走行 %s' % run_b and sn['record'] is True and sn['sent']['launch_boost'] == 1
    assert bench.plan_check(sn['desc'], run_b, 5, sn['sent']) == (sn['desc'], [])
    assert bench.get_field(sn['desc'], 'block') is None and bench.get_field(sn['desc'], 'official') is None
    descs, recs, notes = reads(new, 'desc'), reads(new, 'record'), reads(new, 'note')
    assert 'official=1' in descs[1] and descs[2] == '' and recs == reads(o, 'record') and recs[-1] is True
    assert reads(o, 'desc')[2] == descs[1]                                     # 直す前は雛形が残った
    assert ('説明欄のベンチマークの雛形(ブロック %d・official=1)を消した' % pick_no) in notes[0]
    assert '「走行中のデータを収集する」は ON のままです' in notes[0]


@needs_node
def test_state1_night_tools_do_not_take_it_as_a_round(cands):
    """消したあとの説明欄(AI No. が先頭に付く)は、夜の道具 ② runs_1103・⑤ collect.py の回(block)にならない。前の説明欄は回 1"""
    c, di = cands
    sys.path.insert(0, str(ROOT / 'experiments' / '20261026_fix' / 'diag_launch'))
    sys.path.insert(0, str(ROOT / 'experiments' / '20261026_fix' / 'plan'))
    import collect as co            # noqa: E402
    import runs_1103 as rn          # noqa: E402
    blocks, _ = co.bench_blocks()
    new = bench.with_ai_no('AI走行 %s' % c['baseline']['run'], 1)
    old_d = bench.with_ai_no(c['days'][di]['blocks'][blk(c, di, 1)]['description'] + ' official=1', 1)
    assert co.session_new('20261103-090000', new, blocks) == '11/03 ?'
    assert co.session_new('20261103-090000', old_d, blocks) == '11/03 回1'
    assert rn.kv(new, 'block') is None and rn.kv(old_d, 'block') == '1'


@needs_node
def test_state1_other_flows_send_the_same(tmp_path, cands):
    """直す 1 回目の 10 の流れ(tests/test_ui_fix3._flows)・1 本の手順(基準の設定 → ブロック → ▶)・取り消し・基準の設定だけ:
    走行開始で送る params・確認の文・説明欄・収集の印は直す前と同じ(版のほか)。変わるのは「ブロックのあとに基準の設定を押して、
    ブロックを選ばずに ▶」の説明欄だけ(上の試験)"""
    c, di = cands
    flows = dict(_flows(c, di))
    flows['1 本の手順を 2 本'] = ([['baseline'], ['pick', di, blk(c, di, 1)], ['start'], ['running', False], ['poll'],
                              ['baseline'], ['pick', di, blk(c, di, 2)], ['start']], {})
    flows['ブロック → 基準の設定 → 取り消す'] = ([['pick', di, blk(c, di, 3)], ['baseline'], ['undo'], ['poll'], ['start']], {})
    flows['手で書いた説明欄 → 基準の設定'] = ([['set', '#dr-desc', 'block=9 手で書いた'], ['baseline'], ['start']], {})
    for i, (name, (steps, sp)) in enumerate(flows.items()):
        new, _ = run_sim(tmp_path, c, steps, server=server_params(**sp), name='n%d.json' % i)
        o, _ = old(tmp_path, c, steps, server=server_params(**sp), name='o%d.json' % i)
        sn, so = starts(new), starts(o)
        assert len(sn) == len(so) > 0, name
        for a, b in zip(sn, so):
            same_start(a, b)


# ================================================================ r2-r2-state-2 試験の選択肢と古い問い合わせの返事
@needs_node
def test_state2_guard_off_not_reverted_by_stale_poll(tmp_path, cands):
    """指摘の場面: 見張り ON のサーバー → 問い合わせ(GET)が出る → 見張りを OFF(すぐ送る)→ GET の返事(古い ON)。直す前の写しは
    スイッチが ON に戻り、確認の文に「リカバリーの後ろの見張り: ON」。今は戻らない。次の問い合わせのあとは同じ"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 7)], ['gstart'], ['check', G, False], ['el', G, 'checked'], ['gend'], ['el', G, 'checked'],
             ['read', 'st_params'], ['start'], ['read', 'server'], ['running', False], ['poll'], ['el', G, 'checked'],
             ['read', 'st_params']]
    sp = server_params(rec_back_guard=1)
    new, _ = run_sim(tmp_path, c, steps, server=sp, name='n.json')
    o, _ = old(tmp_path, c, steps, server=sp, name='o.json')
    assert els(o)[:2] == [False, True] and reads(o, 'st_params')[0]['rec_back_guard'] == 1          # 指摘の再現
    assert 'リカバリーの後ろの見張り: ON' in starts(o)[0]['confirm']
    assert els(new) == [False, False, False] and reads(new, 'st_params')[0]['rec_back_guard'] == 0
    assert 'リカバリーの後ろの見張り' not in starts(new)[0]['confirm']
    assert reads(new, 'server')[0]['rec_back_guard'] == 0 == reads(o, 'server')[0]['rec_back_guard']   # サーバーは前も今も 0
    assert starts(new)[0]['sent'] == starts(o)[0]['sent']                                             # 送る値は同じ
    assert reads(new, 'st_params')[1]['rec_back_guard'] == reads(o, 'st_params')[1]['rec_back_guard'] == 0   # 次の同期で同じ
    assert els(o)[2] is False


@needs_node
def test_state2_guard_on_then_keep_not_refused(tmp_path, cands):
    """逆向き: 見張り ON → 古い返事 → 続けて「中立止めのあとも残す」。直す前の写しは古い st.params を見て断った(トースト)。今は入る"""
    c, di = cands
    steps = [['gstart'], ['check', G, True], ['gend'], ['el', G, 'checked'], ['check', H, True], ['el', H, 'checked'],
             ['read', 'server'], ['start']]
    new, tn = run_sim(tmp_path, c, steps, name='n.json')
    o, to = old(tmp_path, c, steps, name='o.json')
    assert els(o) == [False, False] and reads(o, 'server')[0]['rec_hold_keep'] == 0                  # 指摘の再現
    assert any('リカバリーの後ろの見張りが ON のときだけ' in t for t in to)
    assert els(new) == [True, True] and reads(new, 'server')[0]['rec_hold_keep'] == 1
    assert '中立止めのあともリカバリーを残す: ON' in starts(new)[0]['confirm']
    assert starts(new)[0]['sent'] == starts(o)[0]['sent']


@needs_node
@pytest.mark.parametrize('sp,steps', [
    ({}, [['check', G, True], ['poll'], ['check', H, True], ['poll'], ['check', K, True], ['poll'], ['start'],
          ['check', G, False], ['poll'], ['start']]),
    ({'rec_back_guard': 1, 'rec_hold_keep': 1}, [['poll'], ['check', G, False], ['poll'], ['start']]),
    ({}, [['gstart'], ['gend'], ['check', G, True], ['poll'], ['start']]),                         # 返事が先(入れ替わりなし)
    ({}, [['check', G, True], ['gstart'], ['gend'], ['el', G, 'checked'], ['start']]),             # 送信の返事のあとに出た問い合わせ
    ({}, [['confirm', False], ['check', G, True], ['poll'], ['start']]),                           # ON の確認でキャンセル
    ({'rec_back_guard': 1}, [['fail_next', 1], ['check', G, False], ['poll'], ['start']]),         # 送信の失敗(スイッチを戻す)
])
def test_state2_no_race_same_as_before(tmp_path, cands, sp, steps):
    """入れ替わりの無い流れは、スイッチ・確認の文・送った POST・送る値が直す前と同じ"""
    c, di = cands
    st = steps + [['el', G, 'checked'], ['el', H, 'checked'], ['el', K, 'checked'], ['read', 'server'], ['posts']]
    new, tn = run_sim(tmp_path, c, st, server=server_params(**sp), name='n.json')
    o, to = old(tmp_path, c, st, server=server_params(**sp), name='o.json')
    assert els(new) == els(o) and reads(new, 'server') == reads(o, 'server') and tn == to
    same_posts(new, o)
    for a, b in zip(starts(new), starts(o)):
        same_start(a, b)


# ================================================================ r2-r2-state-3 画面の版 8・app.js の更新時刻
def test_state3_versions_match():
    src = (ROOT / 'jetformer' / 'web' / 'server.py').read_text(encoding='utf-8')
    js = (STATIC / 'app.js').read_text(encoding='utf-8')
    ver = int(re.search(r'^DRIVE_CLIENT_VERSION = (\d+)', src, re.M).group(1))
    assert ver == 8 and re.findall(r'const body = \{ client: (\d+),', js) == ['8']
    assert re.findall(r'^const DRIVE_CLIENT = (\d+);', js, re.M) == ['8']             # ▶ の横に出す版も同じ数
    assert 'build: PAGE_BUILD' in js and 'id="dr-client-ver"' in (STATIC / 'index.html').read_text(encoding='utf-8')


def _body(**kw):
    b = {'client': 8, 'run': bench.B_RUN, 'mode_id': 5, 'description': 'AI走行 B', 'dry_run': True, 'record': False, 'params': {}}
    b.update(kw)
    return b


def test_state3_server_refuses_old_pages(tracker, fake_start, monkeypatch, tmp_path):
    """再起動のあとのサーバー: 版 7(10/26 20:25〜23:11 の画面)は 409。版 8 でも、画面の app.js(build)が今のファイルより古ければ
    409(画面を開いたあとに app.js を直した)。build が同じ・新しい・無い(試験・道具)なら版だけを見る(前と同じ)"""
    feed(tracker, 97.0, 100.0, 8.3)
    with pytest.raises(HTTPException) as e:
        srv.api_drive_start(_body(client=7))
    assert e.value.status_code == 409 and '版 7・走行を始められるのは版 8 から' in e.value.detail
    js = tmp_path / 'app.js'
    js.write_text('x')
    os.utime(js, (1793667600, 1793667600))                     # 2026-11-03 10:00(JST)に直した app.js
    monkeypatch.setattr(srv, 'STATIC', tmp_path)
    with pytest.raises(HTTPException) as e:
        srv.api_drive_start(_body(build=1793667600 - 60))
    assert e.value.status_code == 409 and '管理画面が古いまま' in e.value.detail and '11/03 09:59' in e.value.detail
    assert '11/03 10:00 に直したもの' in e.value.detail and fake_start == []
    for build in (1793667600, 1793667600 + 5, 0, None, 'x'):
        b = _body() if build is None else _body(build=build)
        assert srv.api_drive_start(b)['ok'] is True, build
    assert len(fake_start) == 5
    monkeypatch.setattr(srv, 'STATIC', tmp_path / 'none')       # app.js が読めない: 版だけ
    assert srv.api_drive_start(_body(build=1))['ok'] is True


def test_state3_index_gives_the_build_the_page_reads():
    """index(本物)が app.js に付ける ?v= は、今の app.js の更新時刻(画面はそれを build として送る)"""
    html = srv.index().body.decode('utf-8')
    v = re.search(r'/static/app\.js\?v=(\d+)"', html).group(1)
    assert int(v) == srv._app_js_mtime() == int((STATIC / 'app.js').stat().st_mtime)


@needs_node
def test_state3_page_reads_build_and_shows_version(tmp_path, cands):
    c, di = cands
    steps = [['read', 'client_ver'], ['pick', di, blk(c, di, 1)], ['start']]
    res, _ = run_sim(tmp_path, c, steps, script_src='/static/app.js?v=1793667600', name='a.json')
    assert reads(res, 'client_ver') == ['画面の版 8(app.js 11/03 10:00)']
    assert starts(res)[0]['build'] == 1793667600 and starts(res)[0]['client'] == 8
    res2, _ = run_sim(tmp_path, c, steps, name='b.json')                                  # script の src が読めない
    assert reads(res2, 'client_ver') == ['画面の版 8'] and starts(res2)[0]['build'] == 0
    o, _ = old(tmp_path, c, steps, script_src='/static/app.js?v=1793667600', name='c.json')
    assert reads(o, 'client_ver') == [''] and starts(o)[0].get('build') is None            # 直す前の画面には版の文が無い


@needs_node
def test_state3_old_page_from_2025_is_refused(tmp_path, cands):
    """反証の場面(s3): 20:25 の画面(版 7)で ±0.01 を 2 回 → 再起動のあとのサーバー(版 8)は断る。今の画面は始まる"""
    c, di = cands
    up = '#btn-dr-tg-up'
    steps = [['pick', di, blk(c, di, 1)], ['tap', up], ['timers'], ['tap', up], ['resolve1'], ['poll'], ['timers'], ['resolve'],
             ['poll'], ['start'], ['read', 'toasts']]
    o, _ = run_sim(tmp_path, c, steps, app=OLD_2025, server_client_version=8, defer_posts=True, name='o.json')
    s0 = starts(o)[0]
    assert s0['client'] == 7 and s0['refused'] is True and any('管理画面が古いまま' in t for t in reads(o, 'toasts')[0])
    n, _ = run_sim(tmp_path, c, steps, server_client_version=8, defer_posts=True, name='n.json')
    s1 = starts(n)[0]
    assert s1['client'] == 8 and s1['refused'] is False and s1['sent']['throttle_gain'] == 0.92
