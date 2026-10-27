"""画面(app.js)の直す 1 回目・1 組目(2026-10-26 fix3。反証 1 回目の指摘 r1-state3-1・r1-state3-2・r1-rs1-2 の画面の側)の試験。

  r1-state3-1: 速度ゲインの ±0.01 を 2 回押すと 1 回分が消えることがあった(送信中の 1 ビットを、前の送信の返事が消す。送信より
    前に出た表示の問い合わせの返事が送信の返事より後に来ても、スライダーが古い値に戻る)。直し: 送信中の印を数で持つ・同じ項目の
    送信は 1 本ずつ・送信の返事より前に出た問い合わせの返事では戻さない。
  r1-state3-2: 天井 ON の送信が失敗すると pending が消えたままで、同期でサーバーに残った前の本の天井・自動合わせ ON が画面に戻り、
    走行開始で送った。直し: 送れなかったら pending を元に戻す(天井・自動合わせ・発進の前進・区間の上限・補助のスイッチ・ToF・
    TCP・スライダー。チェックボックスは押す前に戻す)。
  r1-rs1-2(画面): 発進の補助(start_kick)は走行中は OFF にだけできる(サーバーも 0 → 1 を断る)。

本物の app.js をまるごと node の vm で読み込む(tests/fixtures/app_sim_fix3_r1.js。tests/fixtures/app_sim_fix3.js の写しに、返事の
順番の入れ替わり・通信の失敗・ブラウザと同じ clearTimeout を足したもの)。直す前の写しは tests/fixtures/fix3_r1_before/。
ブロックは本物の data/bench_candidates.json(読むだけ)の 11/3。サーバー・車・ブラウザは使わない。node が無ければ飛ばす。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_ui_fix3_r1.py
"""
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from jetformer import drive
from jetformer.web import bench

from test_ui_fix3 import _flows

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'jetformer' / 'web' / 'static'
FIX = ROOT / 'tests' / 'fixtures' / 'fix3_r1_before'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_fix3_r1.js'
NODE = shutil.which('node')
UP, DOWN = '#btn-dr-tg-up', '#btn-dr-tg-down'
needs_node = pytest.mark.skipif(NODE is None, reason='node がありません')


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


def run_sim(tmp_path, c, steps, server=None, old=False, name='input.json', **extra):
    app, html = (FIX / 'app.js', FIX / 'index.html') if old else (STATIC / 'app.js', STATIC / 'index.html')
    inp = tmp_path / name
    d = {'bench': c, 'server_params': server if server is not None else server_params(), 'steps': steps,
         'server_client_version': 7, 'confirm': True}
    d.update(extra)
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app), str(html), str(inp)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    return out['steps'], out['toasts']


def reads(res, what):
    return [x['value'] for x in res if x['op'] == 'read' and x['what'] == what]


def starts(res):
    return [x for x in res if x['op'] == 'start']


def posts(res):
    return [x['value'] for x in res if x['op'] == 'posts']


def posts_no_ver(res):
    """送った POST(走行開始の body の画面の版 client・build を外す。2026-10-27 fix3 r2-r2-state-3 で版 8 と build を足した。
    2026-10-27 fix5 t1 で足した boot_at(画面が読んだサーバーの起動の時刻)も外す)"""
    return [[[path, {k: v for k, v in body.items() if k not in ('client', 'build', 'boot_at')} if path == '/api/drive/start' and body else body]
             + list(rest) for path, body, *rest in x] for x in posts(res)]


def gain_line(confirm):
    return [x for x in confirm.split('/') if '速度ゲイン' in x][0].split('\n')[0].strip()


def test_before_copies():
    sums = dict(reversed(line.split()) for line in (FIX / 'SHA256SUMS').read_text().splitlines())
    for name in ('app.js', 'index.html'):
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == sums[name], name
    assert sums['app.js'].startswith('07c93783ef40')                       # ui.md 9 章の「後」


# ================================================================ r1-state3-1 ±0.01 の 2 回押し
RACE = [['tap', UP], ['timers'], ['tap', UP], ['resolve'], ['poll'], ['timers'], ['resolve'], ['poll'], ['read', 'gain'],
        ['read', 'server'], ['start']]
CONTROL = [['tap', UP], ['timers'], ['tap', UP], ['resolve'], ['timers'], ['resolve'], ['poll'], ['read', 'gain'],
           ['read', 'server'], ['start']]


@needs_node
@pytest.mark.parametrize('steps,name', [(RACE, 'race'), (CONTROL, 'control')])
def test_state3_1_double_tap(tmp_path, cands, steps, name):
    """指摘の場面: ブロック 1(0.90)→ +0.01 → 1 回目の送信 → +0.01 → 1 回目の返事 → 表示の同期 → 2 回目のタイマー。
    直す前は表示・サーバー・走行開始・確認の文が 0.91(反証 s2_gain_tap_race.py の再現)。今は 0.92(同期が来ない対照も 0.92)"""
    c, di = cands
    pre = [['pick', di, blk(c, di, 1)]]
    out = {}
    for old in (False, True):
        res, _ = run_sim(tmp_path, c, pre + steps, old=old, name='%s%d.json' % (name, old), defer_posts=True)
        s = starts(res)[0]
        out[old] = (reads(res, 'gain')[0], reads(res, 'server')[0]['throttle_gain'], s['sent']['throttle_gain'],
                    gain_line(s['confirm']))
    assert out[False] == (0.92, 0.92, 0.92, '速度ゲイン 0.92')
    assert out[True] == ((0.91, 0.91, 0.91, '速度ゲイン 0.91') if name == 'race' else (0.92, 0.92, 0.92, '速度ゲイン 0.92'))


@needs_node
@pytest.mark.parametrize('apply_on_resolve', [False, True])
def test_state3_1_stale_status_reply(tmp_path, cands, apply_on_resolve):
    """送信より前にサーバーに届いた表示の問い合わせの返事が、送信の返事より後に来る(指摘の 2 つ目の原因)。直す前はスライダーが
    0.90 に戻り、次の +0.01 が 0.91 を送った。今はその返事では戻さず、0.92"""
    c, di = cands
    if apply_on_resolve:     # 送信は出たが、サーバーが値を変える前に問い合わせを読んだ
        steps = [['tap', UP], ['timers'], ['gstart'], ['resolve'], ['gend'], ['read', 'gain']]
    else:                    # 問い合わせが先に出た
        steps = [['gstart'], ['tap', UP], ['timers'], ['resolve'], ['gend'], ['read', 'gain']]
    steps += [['tap', UP], ['timers'], ['resolve'], ['poll'], ['read', 'gain'], ['read', 'server'], ['start']]
    out = {}
    for old in (False, True):
        res, _ = run_sim(tmp_path, c, steps, server=server_params(throttle_gain=0.9), old=old, name='st%d.json' % old,
                         defer_posts=True, apply_on_resolve=apply_on_resolve)
        out[old] = (reads(res, 'gain'), reads(res, 'server')[0]['throttle_gain'], starts(res)[0]['sent']['throttle_gain'])
    assert out[False] == ([0.91, 0.92], 0.92, 0.92)
    assert out[True] == ([0.9, 0.91], 0.91, 0.91)


def _random_taps(rng):
    ops, ups, downs, gets = [], 0, 0, 0
    for _ in range(rng.randint(6, 14)):
        r = rng.random()
        if r < 0.32:
            ops.append(['tap', UP]); ups += 1
        elif r < 0.40:
            ops.append(['tap', DOWN]); downs += 1
        elif r < 0.55:
            ops.append(['timers'])
        elif r < 0.68:
            ops.append(['resolve1'])
        elif r < 0.75:
            ops.append(['resolve'])
        elif r < 0.85:
            ops.append(['gstart']); gets += 1
        elif r < 0.93:
            ops.append(['gend'])
        else:
            ops.append(['poll'])
    drain = [['timers'], ['resolve']] * 6 + [['gend']] * (gets + 1) + [['timers'], ['resolve']] * 3 + [['poll']]
    return ops + drain + [['read', 'gain'], ['read', 'server'], ['start']], round(0.90 + 0.01 * (ups - downs), 2)


@needs_node
def test_state3_1_random_interleavings(tmp_path, cands):
    """±0.01 のでたらめな押し方 40 通り(タイマー・送信の返事・問い合わせの返事の順番をでたらめに入れ替える。サーバーが値を変えるのが
    送信のときか返事のときかも混ぜる)。最後は全部の返事を返して同期する。今の app.js は 40 通りとも、表示・サーバー・走行開始で送る値・
    確認の文が「0.90 + 押した数 × 0.01」。直す前の写しは、どれかで 1 回分が消える"""
    c, di = cands
    rng = random.Random(20261026)
    bad_new, bad_old = [], 0
    for i in range(40):
        steps, want = _random_taps(rng)
        aor = rng.random() < 0.5
        for old in (False, True):
            res, _ = run_sim(tmp_path, c, steps, server=server_params(throttle_gain=0.9), old=old, name='r%d_%d.json' % (i, old),
                             defer_posts=True, apply_on_resolve=aor)
            s = starts(res)[0]
            got = (reads(res, 'gain')[0], reads(res, 'server')[0]['throttle_gain'], s['sent']['throttle_gain'],
                   gain_line(s['confirm']))
            ok = got == (want, want, want, '速度ゲイン %.2f' % want)
            if old:
                bad_old += not ok
            elif not ok:
                bad_new.append((i, steps[:-3], want, got))
    assert bad_new == []
    assert bad_old >= 1


# ================================================================ r1-state3-2 送れなかったら pending を戻す
@needs_node
def test_state3_2_ceiling_post_fail(tmp_path, cands):
    """指摘の場面: 前の本(2-4)が天井 + 自動合わせ ON → 2-6(ブロック 11。天井だけ)→ 天井 ON の送信が通信の途切れで失敗 → 同期。
    直す前は天井・自動合わせが ON に戻り、走行開始で speed_ceiling 1・speed_auto 1 を送った。今は押す前(OFF・OFF・pending)のまま"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 11)], ['fail_next', 1], ['check', '#dr-speed-ceiling', True], ['read', 'ceil'], ['read', 'auto'],
             ['poll'], ['read', 'ceil'], ['read', 'auto'], ['read', 'pending'], ['start']]
    srv = server_params(speed_ceiling=1, speed_auto=1)
    out = {}
    for old in (False, True):
        res, toasts = run_sim(tmp_path, c, steps, server=srv, old=old, name='c%d.json' % old)
        s = starts(res)[0]
        out[old] = {'ceil': reads(res, 'ceil'), 'auto': reads(res, 'auto'), 'pending': reads(res, 'pending')[0],
                    'sent': (s['sent']['speed_ceiling'], s['sent']['speed_auto']),
                    'lines': [x for x in s['confirm'].splitlines() if '天井' in x or '自動合わせ' in x],
                    'toast': any('Failed to fetch' in t for t in toasts)}
    new, old = out[False], out[True]
    assert old['ceil'] == [False, True] and old['auto'] == [False, True] and old['sent'] == (1, 1) and old['lines']   # 再現
    assert new['ceil'] == [False, False] and new['auto'] == [False, False]
    assert {'speed_ceiling', 'speed_auto'} <= set(new['pending'])
    assert new['sent'] == (0, 0) and new['lines'] == [] and new['toast']


@needs_node
def test_state3_2_ceiling_post_ok_unchanged(tmp_path, cands):
    """対照(失敗しない): 天井 ON で {speed_ceiling: 1, speed_auto: 0}、そのあとは今まで通り(直す前と同じ送る値・確認の文)"""
    c, di = cands
    steps = [['pick', di, blk(c, di, 11)], ['check', '#dr-speed-ceiling', True], ['poll'], ['poll'], ['read', 'ceil'],
             ['read', 'auto'], ['start']]
    srv = server_params(speed_ceiling=1, speed_auto=1)
    a, _ = run_sim(tmp_path, c, steps, server=srv, name='n.json')
    b, _ = run_sim(tmp_path, c, steps, server=srv, old=True, name='o.json')
    assert reads(a, 'ceil') == reads(b, 'ceil') == [True] and reads(a, 'auto') == reads(b, 'auto') == [False]
    assert starts(a)[0]['sent'] == starts(b)[0]['sent'] and starts(a)[0]['confirm'] == starts(b)[0]['confirm']


def _fail_case(c, di, what):
    """(手順, サーバーの値, 走行開始で送る値の鍵)"""
    b14 = blk(c, di, 14)
    if what == 'launch':
        return ([['pick', di, b14], ['fail_next', 1], ['check', '#dr-launch-boost', True], ['poll'], ['read', 'launch'],
                 ['start']], server_params(launch_boost=1), 'launch_boost')
    if what == 'zone':
        return ([['pick', di, b14], ['fail_next', 1], ['check', '#dr-zone-caps', True], ['poll'], ['start']],
                server_params(zone_caps=1), 'zone_caps')
    if what == 'flag':
        return ([['baseline'], ['fail_next', 1], ['check', '#dr-start-kick', False], ['poll'], ['start']],
                server_params(start_kick=0), 'start_kick')
    if what == 'gain':
        return ([['pick', di, blk(c, di, 1)], ['fail_next', 1], ['tap', UP], ['timers'], ['poll'], ['read', 'gain'], ['start']],
                server_params(throttle_gain=1.0), 'throttle_gain')
    raise KeyError(what)


@needs_node
@pytest.mark.parametrize('what,want_new,want_old', [('launch', 0, 1), ('zone', 0, 1), ('flag', 1, 0), ('gain', 0.91, 1.0)])
def test_state3_2_other_controls_post_fail(tmp_path, cands, what, want_new, want_old):
    """同じ形の欄(発進の前進・区間の上限・補助のスイッチ・速度ゲイン): ブロック・基準の設定で入れた値(サーバーと違う = pending)の
    あとに押した送信が失敗。直す前は同期でサーバーに残った前の値が画面に戻り、走行開始でそれを送った。今は押す前の画面の値
    (スイッチは押す前・スライダーは動かした値)のまま、それを送る"""
    c, di = cands
    steps, srv, key = _fail_case(c, di, what)
    got = {}
    for old in (False, True):
        res, toasts = run_sim(tmp_path, c, steps, server=srv, old=old, name='%s%d.json' % (what, old))
        got[old] = starts(res)[0]['sent'][key]
        assert any('Failed to fetch' in t for t in toasts)
    assert (got[False], got[True]) == (want_new, want_old)


# ================================================================ r1-rs1-2 発進の補助は走行中は OFF にだけできる
@needs_node
def test_rs1_2_start_kick_switch_while_running(tmp_path, cands):
    c, di = cands
    steps = [['running', True], ['el', '#dr-start-kick', 'disabled'], ['posts'], ['check', '#dr-start-kick', True],
             ['posts'], ['el', '#dr-start-kick', 'checked']]
    res, toasts = run_sim(tmp_path, c, steps, server=server_params(start_kick=0))
    els = [x['value'] for x in res if x['op'] == 'el']
    assert els == [True, False]                                             # 押せない・押しても OFF のまま
    assert [b for p, b, *_ in posts(res)[1] if p == '/api/drive/params'] == []      # 送らない
    assert any('発進の補助' in t and 'ON にできません' in t for t in toasts)
    # ON の走行中は OFF にできる(すぐ送る)・止まっていれば ON にできる(今まで通り)
    steps = [['running', True], ['el', '#dr-start-kick', 'disabled'], ['posts'], ['check', '#dr-start-kick', False], ['posts'],
             ['running', False], ['check', '#dr-start-kick', True], ['posts'], ['el', '#dr-start-kick', 'disabled']]
    res, _ = run_sim(tmp_path, c, steps, server=server_params(start_kick=1), name='b.json')
    pp = posts(res)
    assert [x['value'] for x in res if x['op'] == 'el'] == [False, False]
    assert [b for p, b, *_ in pp[1] if p == '/api/drive/params'] == [{'start_kick': 0}]
    assert [b for p, b, *_ in pp[2] if p == '/api/drive/params'] == [{'start_kick': 1}]


# ================================================================ ほかの流れは直す前と同じ
@needs_node
def test_sent_values_same_as_before_other_flows(tmp_path, cands):
    """直した場面(送信の失敗・返事の順番の入れ替わり・走行中の発進の補助の ON)のほかは、走行開始で送る params・確認の文・送った
    POST が直す前の写しと同じ(test_ui_fix3 の 10 の流れ + ±0.01・スイッチ・天井・ToF の流れ。返事はすぐ返す)"""
    c, di = cands
    flows = dict(_flows(c, di))
    flows.update({
        '±0.01 を 3 回(返事を待つ)→ 走行開始': ([['pick', di, blk(c, di, 1)], ['click', UP], ['click', UP], ['click', DOWN], ['start']], {}),
        '補助のスイッチを止まっているとき OFF → ON': ([['check', '#dr-start-kick', False], ['poll'], ['check', '#dr-start-kick', True],
                                                     ['check', '#dr-recovery', False], ['poll'], ['start']], {}),
        '天井 ON → 自動合わせ ON → 天井 OFF': ([['pick', di, blk(c, di, 8)], ['check', '#dr-speed-ceiling', True],
                                              ['check', '#dr-speed-auto', True], ['check', '#dr-speed-ceiling', False], ['poll'],
                                              ['start']], {'speed_ceiling': 1, 'speed_auto': 1}),
        '走行中の ±0.01・補助を OFF': ([['pick', di, blk(c, di, 1)], ['start'], ['click', UP], ['check', '#dr-start-kick', False],
                                      ['poll'], ['running', False], ['poll'], ['pick', di, blk(c, di, 2)], ['start']], {}),
    })
    for i, (name, (steps, srv)) in enumerate(flows.items()):
        st = steps + [['posts']]
        new, _ = run_sim(tmp_path, c, st, server=server_params(**srv), name='n%d.json' % i)
        old, _ = run_sim(tmp_path, c, st, server=server_params(**srv), old=True, name='o%d.json' % i)
        sn, so = starts(new), starts(old)
        assert len(sn) == len(so) > 0, name
        for a, b in zip(sn, so):
            assert a['sent'] == b['sent'], (name, {k: (a['sent'].get(k), b['sent'].get(k)) for k in set(a['sent']) | set(b['sent'])
                                                   if a['sent'].get(k) != b['sent'].get(k)})
            assert a['confirm'] == b['confirm'] and (a['client'], b['client']) == (9, 7), name    # 2026-10-27 fix5 で今の画面は版 9
        assert posts_no_ver(new) == posts_no_ver(old), name               # 版のほか(r2-r2-state-3)
