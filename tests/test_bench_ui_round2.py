"""画面(app.js)の直す 2 回目(2026-10-26 fix2): トリムがブロック・基準の設定で 0 に戻らない(state2-1)、
同じベンチマークを選ぶ欄で選び直すと入れ直す(state2-3)、確認の文のトリム・後ろの見張りの行。

本物の app.js をまるごと node の vm で読み込み(tests/fixtures/app_sim_round2.js。DOM・サーバーは偽物)、画面の操作を流す。
ブロックは本物の data/bench_candidates.json(web/bench.load_candidates。読むだけ)の 2026-11-03。node が無ければ飛ばす。
サーバー・車・ブラウザは使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider --basetemp=<消せる場所> tests/test_bench_ui_round2.py
"""
import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jetformer import drive
from jetformer.web import bench

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / 'jetformer' / 'web' / 'static' / 'app.js'
SIM = ROOT / 'tests' / 'fixtures' / 'app_sim_round2.js'
NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='node がありません')


@pytest.fixture(scope='module')
def cands():
    c = bench.load_candidates()
    days = [i for i, d in enumerate(c.get('days') or []) if d.get('date') == '2026-11-03' and d.get('blocks')]
    if not days:
        pytest.skip('data/bench_candidates.json に 2026-11-03 のブロックがありません')
    c = dict(c, today='2026-11-03')            # 11/3 の画面(選ぶ欄は今日のブロックを並べる)
    return c, days[0]


def run_sim(tmp_path, cands_, steps, server=None, app=APP_JS):
    c, _ = cands_
    sp = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    sp.update(server or {})
    inp = tmp_path / 'input.json'
    inp.write_text(json.dumps({'bench': c, 'server_params': sp, 'steps': steps}, ensure_ascii=False, default=str),
                   encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(app), str(inp)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])['steps']


def starts(res):
    return [x for x in res if x['op'] == 'start']


def reads(res, what):
    return [x['value'] for x in res if x['op'] == 'read' and x['what'] == what]


# ---------------------------------------------------------------- state2-1 トリム
def test_trim_survives_block_and_is_sent(tmp_path, cands):
    """指摘の順番: 1-1 のあとトリム +15 → 「ベンチマーク 2」(1-2)→ 走行開始 → 15 を送る。次のブロック(1-3)も 15"""
    _, di = cands
    res = run_sim(tmp_path, cands, [['slide', '#dr-trim', 15], ['poll'], ['pick', di, 1], ['read', 'trim'], ['read', 'pending'],
                                    ['read', 'note'], ['poll'], ['read', 'trim'], ['read', 'check_base'], ['start'], ['poll'],
                                    ['pick', di, 2], ['start'], ['read', 'server']])
    assert res[0]['server'] == 15
    assert reads(res, 'trim') == [15, 15]                         # 選んだあと・表示の同期のあと
    assert 'steer_trim_us' not in reads(res, 'pending')[0]
    assert 'トリム +15 µs はそのまま' in reads(res, 'note')[0]
    assert 'トリム +15 µs' in reads(res, 'check_base')[0]
    s1, s2 = starts(res)
    assert s1['sent']['steer_trim_us'] == 15 and s2['sent']['steer_trim_us'] == 15
    assert s1['sent']['throttle_gain'] == 0.9                      # ブロックの上書きはそのまま入る
    assert '\nトリム: +15 µs' in s1['confirm']                     # 確認の文にトリムの行
    assert reads(res, 'server')[0]['steer_trim_us'] == 15


def test_trim_survives_baseline_button(tmp_path, cands):
    """手順書 4 章 1 の「基準の設定」のあとも残る(前は 0 に戻した)"""
    _, di = cands
    res = run_sim(tmp_path, cands, [['slide', '#dr-trim', 15], ['poll'], ['baseline'], ['read', 'trim'], ['pick', di, 1], ['start']])
    assert reads(res, 'trim') == [15]
    assert starts(res)[0]['sent']['steer_trim_us'] == 15


def test_trim_zero_has_no_confirm_line(tmp_path, cands):
    """トリム 0・見張り OFF なら確認の文は前と同じ(行を足さない)"""
    _, di = cands
    res = run_sim(tmp_path, cands, [['pick', di, 1], ['start']])
    c = starts(res)[0]['confirm']
    assert 'トリム' not in c and '後ろの見張り' not in c
    assert starts(res)[0]['sent']['steer_trim_us'] == 0


def test_block_that_sets_trim_still_sets_it(tmp_path, cands):
    """ブロックの params に steer_trim_us を書いたときだけ、ブロックの値が入る(手順書 5 章の「使える」項目のまま)"""
    c, di = cands
    c2 = copy.deepcopy(c)
    c2['days'][di]['blocks'][1]['params'] = dict(c2['days'][di]['blocks'][1].get('params') or {}, steer_trim_us=12.0)
    res = run_sim(tmp_path, (c2, di), [['slide', '#dr-trim', 15], ['poll'], ['pick', di, 1], ['read', 'trim'], ['start']])
    assert reads(res, 'trim') == [12] and starts(res)[0]['sent']['steer_trim_us'] == 12


def test_rec_back_guard_line_when_server_on(tmp_path, cands):
    _, di = cands
    res = run_sim(tmp_path, cands, [['pick', di, 1], ['start']], server={'rec_back_guard': 1})
    assert '\nリカバリーの後ろの見張り: ON' in starts(res)[0]['confirm']


def test_old_app_resets_trim(tmp_path, cands):
    """対照: 直す前の app.js(before_round2 の写し)では 0 を送っていた(指摘の再現)"""
    before = ROOT / 'experiments' / '20261026_fix' / 'fix2' / 'before_round2' / 'jetformer' / 'web' / 'static' / 'app.js'
    if not before.exists():
        pytest.skip('直す前の写しがありません')
    _, di = cands
    res = run_sim(tmp_path, cands, [['slide', '#dr-trim', 15], ['poll'], ['pick', di, 1], ['start']], app=before)
    assert starts(res)[0]['sent']['steer_trim_us'] == 0 and 'トリム' not in starts(res)[0]['confirm']


# ---------------------------------------------------------------- state2-3 同じブロックを選び直す
def test_repick_same_block_reapplies(tmp_path, cands):
    """1-4 を走ったあと速度を 1.10 に動かし、同じ「ベンチマーク 4」を選び直す → change が出て 0.90 に戻る"""
    _, di = cands
    res = run_sim(tmp_path, cands, [['pick', di, 3], ['read', 'pick'], ['read', 'pick_first'], ['start'], ['poll'],
                                    ['slide', '#dr-throttle-gain', 1.1], ['poll'], ['read', 'gain'],
                                    ['pick', di, 3], ['read', 'gain'], ['read', 'pick'], ['start']])
    picks = [x for x in res if x['op'] == 'pick']
    assert picks[0]['fired'] and picks[1]['fired']                 # 2 回目も change が出る
    assert reads(res, 'pick') == ['', '']                          # 選んだあとは「選ぶ」の行に戻る
    assert re.search(r'今: ベンチマーク4: ', reads(res, 'pick_first')[0])
    assert reads(res, 'gain') == [1.1, 0.9]
    assert [s['sent']['throttle_gain'] for s in starts(res)] == [0.9, 0.9]


def test_old_app_repick_does_nothing(tmp_path, cands):
    """対照: 直す前の app.js では、選ばれたままなので change が出ない(指摘の再現)"""
    before = ROOT / 'experiments' / '20261026_fix' / 'fix2' / 'before_round2' / 'jetformer' / 'web' / 'static' / 'app.js'
    if not before.exists():
        pytest.skip('直す前の写しがありません')
    _, di = cands
    res = run_sim(tmp_path, cands, [['pick', di, 3], ['start'], ['poll'], ['slide', '#dr-throttle-gain', 1.1], ['poll'],
                                    ['pick', di, 3], ['read', 'gain'], ['start']], app=before)
    picks = [x for x in res if x['op'] == 'pick']
    assert picks[0]['fired'] and not picks[1]['fired']
    assert reads(res, 'gain') == [1.1] and starts(res)[1]['sent']['throttle_gain'] == 1.1


def test_js_field_values_drop_trim_only():
    """benchFieldValues は基準のトリム・後ろの見張りを入れない(ほかの項目は前と同じ)"""
    js = APP_JS.read_text(encoding='utf-8')
    parts = [re.search(r'^const BENCH_FALLBACK = \{.*?^};\n', js, re.S | re.M).group(0),
             re.search(r'^function benchFieldValues\(.*?^}\n', js, re.S | re.M).group(0)]
    fb = bench.fallback_baseline()
    code = '\n'.join(parts) + '\nprocess.stdout.write(JSON.stringify(benchFieldValues(%s, "")));' % json.dumps(fb)
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert 'steer_trim_us' not in got['params'] and 'rec_back_guard' not in got['params']
    assert got['params'] == {k: v for k, v in fb['params'].items() if k not in ('steer_trim_us', 'rec_back_guard')}


def test_old_server_without_new_keys(tmp_path, cands):
    """今動いている古いサーバー(発進の前進・後ろの見張りの項目が無い params)でも、新しい画面は同じに動く
    (app.js は再起動なしで読み直すと効く: web/server.py は static を毎回ディスクから出す)"""
    _, di = cands
    old = {k: v for k, v in drive.DEFAULT_PARAMS.items() if k not in drive.LAUNCH_KEYS + ('rec_back_guard',)}
    old.update(steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
    c, _ = cands
    sp = json.dumps({'bench': c, 'server_params': old, 'steps': [
        ['slide', '#dr-trim', 15], ['poll'], ['pick', di, 1], ['start'], ['poll'], ['pick', di, 1], ['read', 'pick'], ['start']]},
        ensure_ascii=False, default=str)
    inp = tmp_path / 'input.json'
    inp.write_text(sp, encoding='utf-8')
    r = subprocess.run([NODE, str(SIM), str(APP_JS), str(inp)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])['steps']
    s1, s2 = starts(res)
    assert s1['sent']['steer_trim_us'] == 15 and s2['sent']['steer_trim_us'] == 15
    assert 'rec_back_guard' not in s1['sent'] and '後ろの見張り' not in s1['confirm']
    assert [x['fired'] for x in res if x['op'] == 'pick'] == [True, True] and reads(res, 'pick') == ['']
