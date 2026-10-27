"""画面の直す 3 回目・1 組目(2026-10-27 fix3。反証 3 回目の指摘 r3-r3-state-1・r3-r3-state-2・r3-r3-state-3)の試験。

  r3-r3-state-1: サーバーの日付(/api/bench/candidates の today)が 11/03 でない(Jetson の RTC は電源を切ると戻り、ネットが無いと
    NTP で合わない)と、選ぶ欄は黙って「ブロックのある最初の日」(10/23)に落ち、「ベンチマーク 2」を選ぶと 10/23 の SLOW-SEC1・
    ゲイン 1.00・説明欄 day=2026-10-23 が入った。直し(app.js): 今日のブロックが無いときは選ぶ欄を注意の 1 行だけにして選べなく
    する(ほかの日に落とさない)。モデルの一覧にほかの日のベンチマークの番号を付けない。カードに注意の行。カードのボタンでほかの日の
    ブロックを押すと確認の文(日付つき)。手順書 3 章に日付の確かめ。
  r3-r3-state-2: 朝の dry run の「モーターに出力しない」が残ったまま 4 章どおり 基準の設定 → ブロック → ▶ とすると、確認の文が
    出ず推論の確認(出力は中立)で始まり、1-1 の 1 本を失った。直し: ブロックを選ぶと「モーターに出力しない」を外す(変えた所の文に
    出す。取り消すで戻る)。▶ で dry のまま説明欄がブロックの本(block=・official=1)なら確かめる。手順書 3 章・4 章 6。
  r3-r3-state-3: 最初の同期の前(状態の問い合わせが失敗している間)に押したブロック・基準の設定は、サーバーの値が分からず pending に
    ならず、同期で前の本の天井・自動合わせ・発進の前進・ゲインに黙って戻った。直し: benchApply も ▶ と同じく、サーバーの設定を
    読むまで断る(同じトースト)。選ぶ欄で断ったときは説明欄に official=1 を足さない。

本物の app.js をまるごと node の vm で読み込む(tests/fixtures/app_sim_fix3_r3_1.js。反証 3 回目の sim3.js と同じ足し + カードの
ボタン tapday・read の dry など)。直す前の写しは tests/fixtures/fix3_r3_1_before/app.js(sha256 を照らす)。サーバー・車は偽物。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix3_r3_1.py
"""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jetformer import drive
from jetformer.web import bench

from test_ui_fix3 import _flows

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
BEFORE = ROOT / 'tests' / 'fixtures' / 'fix3_r3_1_before'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix3_r3_1.js'
NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='node がありません')


def test_before_copies():
    sums = dict(reversed(x.split()) for x in (BEFORE / 'SHA256SUMS').read_text().splitlines() if x.strip())
    for name in ('app.js', 'index.html'):
        assert hashlib.sha256((BEFORE / name).read_bytes()).hexdigest() == sums[name], name
    old = (BEFORE / 'app.js').read_text(encoding='utf-8')
    assert 'if (di < 0) di = days.findIndex((x) => x.blocks.length);' in old          # 直す前(ほかの日に落ちる)
    assert (STATIC / 'index.html').read_bytes() == (BEFORE / 'index.html').read_bytes()   # この組は index.html を変えていない


@pytest.fixture(scope='module')
def cands():
    c = bench.load_candidates()
    days = [i for i, d in enumerate(c.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks')]
    if not days:
        pytest.skip('data/bench_candidates.json に 2026-11-03 のブロックがありません')
    return dict(c, today='2026-11-03'), days[0]


def blk(c, di, no):
    return next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)


def day_index(c, date):
    return next(i for i, d in enumerate(c['days']) if d['date'] == date)


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
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def both(tmp_path, c, steps, name, **kw):
    return run_sim(tmp_path, c, steps, name='n_' + name, **kw), run_sim(tmp_path, c, steps, app=BEFORE / 'app.js', name='o_' + name, **kw)


def starts(res):
    return [x for x in res if x['op'] == 'start']


def reads(res, what):
    return [x['value'] for x in res if x['op'] == 'read' and x['what'] == what]


def els(res):
    return [x['value'] for x in res if x['op'] == 'el']


def same_starts(a, b):
    sa, sb = starts(a), starts(b)
    assert len(sa) == len(sb) > 0
    for x, y in zip(sa, sb):
        for k in ('sent', 'confirm', 'confirms', 'refused', 'desc', 'record', 'dry', 'build', 'run'):
            assert x[k] == y[k], (k, x[k], y[k])
        # 画面の版: 今の画面は 9(2026-10-27 fix5 t1・t2・t3。直す前の写しは 8)。送らなかった(キャンセル)ならどちらも None
        assert (x['client'], y['client']) in ((9, 8), (None, None)), (x['client'], y['client'])


def opts(html):
    return re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)', html)


# ================================================================ r3-r3-state-1 サーバーの日付が 11/03 でない
@pytest.mark.parametrize('today', ['2026-11-02', '2026-10-27', '2026-11-04'])
def test_state1_no_block_today_select_is_not_filled_from_another_day(tmp_path, cands, today):
    """今日のブロックが無い日: 選ぶ欄は注意の 1 行だけ(サーバーの日付・ブロックのある日)で選べない。カードに注意の行。
    モデルの一覧にベンチマークの番号を付けない。直す前の写しは 10/23 の 8 本を並べ、「ベンチマーク 2」で SLOW-SEC1・ゲイン 1.00・
    説明欄 day=2026-10-23 を送った(反証の s7 の再現)"""
    c, _ = cands
    c = dict(c, today=today)
    steps = [['el', '#dr-bench-pick', 'html'], ['el', '#dr-bench-pick', 'disabled'], ['el', '#dr-bench-pick-wrap', 'hidden'],
             ['read', 'bench_error'], ['eval', "[...document.querySelector('#dr-model').options].map((o) => o.textContent).join('|')"],
             ['baseline'], ['pick', 0, 1], ['read', 'desc'], ['read', 'gain'], ['start']]
    (n, _), (o, _) = both(tmp_path, c, steps, 's1_%s' % today)
    hn, ho = els(n)[0], els(o)[0]
    assert len(opts(hn)) == 1 and opts(hn)[0][0] == ''
    assert 'サーバーの日付 %s' % today in hn and 'Jetson の時刻' in hn and '2026-11-03' in hn
    assert els(n)[1] is True and els(n)[2] is False
    err = reads(n, 'bench_error')[0]
    assert '今日(サーバーの日付 %s)のブロックはありません' % today in err and 'sudo date -s' in err and '手順書 3 章' in err
    assert 'ベンチマーク' not in [x for x in n if x['op'] == 'eval'][0]['value']
    sn = starts(n)[0]
    assert sn['run'] == bench.B_RUN and 'day=' not in (sn['desc'] or '') and sn['sent']['throttle_gain'] == 1.0   # 基準の設定のまま
    # 直す前(再現): 10/23 の 8 本・ベンチマーク 2 = SLOW-SEC1・説明欄 day=2026-10-23 block=2
    assert [x[1].split(':')[0] for x in opts(ho)[1:]] == ['ベンチマーク%d' % i for i in range(1, 9)]
    so = starts(o)[0]
    assert so['run'] == 'bench_20261022-slow-sec-s1' and 'day=2026-10-23 block=2' in so['desc'] and 'official=1' in so['desc']
    assert reads(o, 'bench_error')[0] == ''


def test_state1_today_has_blocks_same_as_before(tmp_path, cands):
    """サーバーの日付が 11/03(ブロックがある): 選ぶ欄・モデルの一覧・カードの注意・送る値が直す前とまるごと同じ"""
    c, di = cands
    steps = [['el', '#dr-bench-pick', 'html'], ['el', '#dr-bench-pick', 'disabled'], ['read', 'bench_error'],
             ['eval', "[...document.querySelector('#dr-model').options].map((o) => o.textContent).join('|')"],
             ['pick', di, blk(c, di, 2)], ['el', '#dr-bench-pick', 'html'], ['read', 'desc'], ['start']]
    (n, tn), (o, to) = both(tmp_path, c, steps, 's1_today')
    assert els(n) == els(o) and reads(n, 'bench_error') == reads(o, 'bench_error') == ['']
    assert [x for x in n if x['op'] == 'eval'] == [x for x in o if x['op'] == 'eval']
    assert reads(n, 'desc') == reads(o, 'desc')
    same_starts(n, o)
    assert tn == to


@pytest.mark.parametrize('answer', [False, True])
def test_state1_card_button_of_another_day_asks(tmp_path, cands, answer):
    """カードのボタンでほかの日(サーバーの日付と違う日)のブロックを押すと、日付つきの確認の文。キャンセルなら何も入れない。
    OK なら直す前と同じに入る。今日のブロックのボタンは確認なし(前と同じ)"""
    c, di = cands
    c2 = dict(c, today='2026-11-02')
    d23 = day_index(c, '2026-10-23')
    steps = [['baseline'], ['confirm', answer], ['tapday', di, blk(c, di, 2)], ['read', 'applied'], ['read', 'desc'], ['read', 'gain'],
             ['confirm', True], ['start']]
    (n, _), (o, _) = both(tmp_path, c2, steps, 's1_card_%d' % answer)
    tap = [x for x in n if x['op'] == 'tapday'][0]
    assert len(tap['confirm']) == 1 and 'このブロックは 2026-11-03 の段です(今日 = サーバーの日付 2026-11-02)' in tap['confirm'][0]
    assert [x for x in o if x['op'] == 'tapday'][0]['confirm'] == []                               # 直す前は黙って入れた
    if answer:
        assert reads(n, 'applied') == reads(o, 'applied') == [{'day': di, 'i': blk(c, di, 2)}]
        assert reads(n, 'desc') == reads(o, 'desc') and 'day=2026-11-03 block=2' in reads(n, 'desc')[0]
        assert starts(n)[0]['sent'] == starts(o)[0]['sent']
    else:
        assert reads(n, 'applied') == [{'baseline': True}] and reads(n, 'desc') == [''] and starts(n)[0]['run'] == bench.B_RUN
    # 今日のブロック(11/03 の日に 11/03 のボタン)・10/23 の日に 10/23 のボタン: 確認なしで前と同じ
    for cc, dd in ((c, di), (dict(c, today='2026-10-23'), d23)):
        steps2 = [['tapday', dd, 1], ['read', 'applied'], ['read', 'desc'], ['start']]
        (n2, _), (o2, _) = both(tmp_path, cc, steps2, 's1_card_today_%d_%d' % (answer, dd))
        assert [x for x in n2 if x['op'] == 'tapday'][0]['confirm'] == []
        assert reads(n2, 'desc') == reads(o2, 'desc') and reads(n2, 'applied') == reads(o2, 'applied')
        same_starts(n2, o2)


# ================================================================ r3-r3-state-2 「モーターに出力しない」が残る
def test_state2_block_clears_dry_left_from_morning(tmp_path, cands):
    """反証の s1_dry_left_on: 基準の設定 → dry ON → ▶ → 停止 → 4 章どおり 基準の設定 → ベンチマーク 1 → +0.01 ×2 → ▶。
    今: ブロックを選ぶと「モーターに出力しない」が外れ、2 本目は dry = False・確認の文が出る・ゲイン 0.92・説明欄 block=1 official=1。
    直す前の写しは dry = True のまま確認の文なし(1-1 の 1 本を失った)"""
    c, di = cands
    steps = [['baseline'], ['check', '#dr-dry', True], ['start'], ['running', False],
             ['baseline'], ['read', 'dry'], ['pick', di, blk(c, di, 1)], ['read', 'dry'], ['read', 'note'],
             ['click', '#btn-dr-tg-up'], ['click', '#btn-dr-tg-up'], ['start']]
    (n, tn), (o, to) = both(tmp_path, c, steps, 's2')
    a1, a2 = starts(n)
    b1, b2 = starts(o)
    assert a1['dry'] is True and b1['dry'] is True and a1['confirms'] == b1['confirms'] == []          # 朝の dry run は前と同じ
    assert reads(n, 'dry') == [True, False] and reads(o, 'dry') == [True, True]
    assert '「モーターに出力しない」ON → OFF' in reads(n, 'note')[0]
    assert a2['dry'] is False and a2['record'] is True and a2['confirm'].startswith('AI走行を開始します。')
    assert a2['sent']['throttle_gain'] == 0.92 and 'block=1' in a2['desc'] and 'official=1' in a2['desc']
    assert b2['dry'] is True and b2['confirms'] == [] and 'block=1' in b2['desc']                        # 指摘の再現
    assert a2['sent'] == b2['sent'] and a2['desc'] == b2['desc'] and a2['record'] == b2['record']    # 送る値は dry のほか同じ
    assert any('AI走行 No.1 を開始しました' in t for t in tn) and not any('AI走行 No.' in t for t in to)


def test_state2_undo_restores_dry_and_no_dry_flow_unchanged(tmp_path, cands):
    """ブロックで外した「モーターに出力しない」は「取り消す」で戻る。dry を使わない流れ(ブロック → ▶)は前と同じ"""
    c, di = cands
    steps = [['check', '#dr-dry', True], ['pick', di, blk(c, di, 1)], ['read', 'dry'], ['undo'], ['read', 'dry']]
    (n, _), (o, _) = both(tmp_path, c, steps, 's2u')
    assert reads(n, 'dry') == [False, True] and reads(o, 'dry') == [True, True]
    steps2 = [['pick', di, blk(c, di, 1)], ['read', 'dry'], ['start']]
    (n2, t2), (o2, u2) = both(tmp_path, c, steps2, 's2n')
    assert reads(n2, 'dry') == reads(o2, 'dry') == [False]
    same_starts(n2, o2)
    assert t2 == u2


@pytest.mark.parametrize('answer', [False, True])
def test_state2_dry_with_block_desc_asks_at_start(tmp_path, cands, answer):
    """ブロックを選んだあとに「モーターに出力しない」を入れ直して ▶: 「推論の確認で始めますか?」を聞く。キャンセルなら送らない。
    OK なら直す前と同じ dry の走行開始(説明欄・params も同じ)。説明欄にブロックの無い dry(朝の dry run)は聞かない(前と同じ)"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 3)], ['check', '#dr-dry', True], ['confirm', answer], ['start']]
    (n, _), (o, _) = both(tmp_path, c, steps, 's2c_%d' % answer)
    a, b = starts(n)[0], starts(o)[0]
    assert len(a['confirms']) == 1 and '「モーターに出力しない」が ON です' in a['confirms'][0] and 'block=3' in a['confirms'][0]
    assert b['confirms'] == [] and b['dry'] is True
    if answer:
        assert a['dry'] is True and a['sent'] == b['sent'] and a['desc'] == b['desc'] and a['record'] == b['record']
    else:
        assert a['sent'] is None and a['client'] is None
    steps2 = [['baseline'], ['check', '#dr-dry', True], ['start']]
    (n2, _), (o2, _) = both(tmp_path, c, steps2, 's2m_%d' % answer)
    same_starts(n2, o2)
    assert starts(n2)[0]['confirms'] == []


# ================================================================ r3-r3-state-3 最初の同期の前のブロック
def test_state3_block_before_first_sync_is_refused(tmp_path, cands):
    """反証の s5: 前の本(2-4)のサーバーの値 天井 1・自動合わせ 1・発進の前進 1・ゲイン 0.97。状態の問い合わせが初めの 2 回失敗して
    いる間にブロック 5(天井 OFF・0.90)を押す。今: 断る(▶ と同じトースト)・説明欄に足さない・選ぶ欄は「選ぶ」の行。同期のあとに
    もう一度選ぶと 0・0・0・0.90 を送る。直す前の写しは pending が空のまま、同期で前の本の値に戻り 1・1・1・0.97 を送った"""
    c, di = cands
    sp = server_params(speed_ceiling=1, speed_auto=1, launch_boost=1, throttle_gain=0.97)
    b5 = blk(c, di, 5)
    steps = [['pick', di, b5], ['read', 'pending'], ['read', 'desc'], ['read', 'applied'], ['read', 'pick'], ['realpoll'],
             ['read', 'gain'], ['el', '#dr-speed-ceiling', 'checked'], ['pick', di, b5], ['read', 'pending'], ['start']]
    (n, tn), (o, to) = both(tmp_path, c, steps, 's3', server=sp, status_fail_first=2)
    assert any('サーバーの設定を読み込んでいます' in t for t in tn)
    assert reads(n, 'pending')[0] == [] and reads(n, 'desc')[0] == '' and reads(n, 'applied')[0] is None and reads(n, 'pick')[0] == ''
    assert reads(n, 'gain')[0] == 0.97 and els(n)[0] is True                                   # 同期でサーバーの値(前の本)を出す
    assert reads(n, 'pending')[1] == ['launch_boost', 'speed_auto', 'speed_ceiling', 'throttle_gain']
    s = starts(n)[0]['sent']
    assert (s['speed_ceiling'], s['speed_auto'], s['launch_boost'], s['throttle_gain']) == (0, 0, 0, 0.9)
    assert 'block=5' in starts(n)[0]['desc'] and 'official=1' in starts(n)[0]['desc']
    # 直す前(再現): 最初に押したブロックが入ったまま pending が空で、同期で前の本の値に戻る(2 回目の選び直しの前の ▶ と同じ)
    so = run_sim(tmp_path, c, [['pick', di, b5], ['read', 'pending'], ['realpoll'], ['start']], server=sp, status_fail_first=2,
                 app=BEFORE / 'app.js', name='o_s3b')[0]
    assert reads(so, 'pending') == [[]]
    x = starts(so)[0]['sent']
    assert (x['speed_ceiling'], x['speed_auto'], x['launch_boost'], x['throttle_gain']) == (1, 1, 1, 0.97)


def test_state3_baseline_before_first_sync_is_refused(tmp_path, cands):
    """基準の設定も同じ(同期の前は断る)。同期のあとは前と同じ"""
    c, _ = cands
    sp = server_params(speed_ceiling=1, throttle_gain=0.97)
    steps = [['baseline'], ['read', 'pending'], ['read', 'applied'], ['realpoll'], ['baseline'], ['read', 'pending'], ['start']]
    (n, tn), (o, to) = both(tmp_path, c, steps, 's3b', server=sp, status_fail_first=2)
    assert reads(n, 'pending')[0] == [] and reads(n, 'applied')[0] is None and any('サーバーの設定を読み込んでいます' in t for t in tn)
    assert reads(n, 'pending')[1] == reads(o, 'pending')[1] and starts(n)[0]['sent'] == starts(o)[0]['sent']


def test_state3_refused_pick_does_not_touch_desc(tmp_path, cands):
    """選ぶ欄で断ったとき(モデルが走行の一覧に無い)は説明欄に official=1 を足さない(直す前は今の説明欄に足した)。
    選ぶ欄は「選ぶ」の行に戻る(同じ番号をもう一度選べる)"""
    c, di = cands
    steps = [['set', '#dr-desc', 'memo=x'], ['pick', di, blk(c, di, 1)], ['read', 'desc'], ['read', 'pick']]
    (n, _), (o, _) = both(tmp_path, c, steps, 's3m', models=['other-model'])
    assert reads(n, 'desc') == ['memo=x'] and reads(n, 'pick') == ['']
    assert reads(o, 'desc') == [' official=1 memo=x']                                               # 直す前


# ================================================================ そのほかの流れは前と同じ
def test_other_flows_send_the_same(tmp_path, cands):
    """直す 1 回目の 10 の流れ(tests/test_ui_fix3._flows)・1 本の手順(基準の設定 → ブロック → ±0.01 → ▶)・同じブロックの選び直し:
    走行開始で送る値・確認の文・説明欄・収集・dry・版・build・送った POST・トーストが直す前と同じ"""
    c, di = cands
    flows = dict(_flows(c, di))
    flows['1 本の手順'] = ([['baseline'], ['pick', di, blk(c, di, 13)], ['click', '#btn-dr-tg-up'], ['check', '#dr-launch-boost', True],
                         ['start'], ['running', False], ['pick', di, blk(c, di, 14)], ['pick', di, blk(c, di, 14)], ['start']], {})
    for i, (name, (steps, srv)) in enumerate(flows.items()):
        steps = steps + [['posts']]
        (n, tn), (o, to) = both(tmp_path, c, steps, 'f%d' % i, server=server_params(**srv))
        same_starts(n, o)
        # 走行開始の body の版(client)と boot_at(2026-10-27 fix5 t1 で足した)のほかは同じ
        nv = lambda r: [[[p, {k: v for k, v in bd.items() if k not in ('client', 'boot_at')} if p == '/api/drive/start' else bd, st]   # noqa: E731
                         for p, bd, st in x['value']] for x in r if x['op'] == 'posts']
        assert nv(n) == nv(o), name
        assert tn == to, name
