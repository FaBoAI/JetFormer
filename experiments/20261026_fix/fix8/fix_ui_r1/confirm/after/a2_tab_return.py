#!/usr/bin/env python3
"""反証 8-1 回目 ui の指摘 stale-status-after-drive-tab-return を打ち消す側で確かめる(票 A)。
本物の app.js・index.html を、sim_copy.js(tests/fixtures/app_sim_fix7_r2.js の写し。変えていない)で node の vm に読み込んで流す。
偽のサーバーだけ(本物の jetformer-web・車には触れない)。1 つの画面で: 走行タブで同期 → ライブのタブ → その間に curl(= 別の画面)で
値を変える → 走行タブに戻る → すぐ ▶。一覧の GET を止める(遅い Wi-Fi)/止めない、5 秒たつ、を比べる。"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive              # noqa: E402  既定値を読むだけ
from jetformer.web import bench          # noqa: E402  候補を読むだけ

STATIC = JF / 'jetformer' / 'web' / 'static'
NODE = shutil.which('node')
MD = '/api/drive/models'
c = bench.load_candidates()
di = next(i for i, d in enumerate(c['days']) if d.get('date') == '2026-11-03' and d.get('blocks'))
c = dict(c, today='2026-11-03')
blk = lambda no: next(i for i, b in enumerate(c['days'][di]['blocks']) if int(b['block']) == no)   # noqa: E731


def sp(**o):
    p = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=0.9, lookahead=3, start_kick=1)
    p.update(o)
    return p


def run(name, server, steps):
    inp = HERE / 'out' / ('a2_%s.input.json' % name)
    d = {'bench': c, 'server_params': server, 'steps': steps, 'server_client_version': 9, 'confirm': True, 'boot_at': 1000.0,
         'models': sorted({cd['run'] for cd in c['candidates'] if cd.get('run')}),
         'script_src': 'http://x/static/app.js?v=1793667600', 'server_boot_check': True}
    inp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding='utf-8')
    r = subprocess.run([NODE, str(HERE / 'sim_copy.js'), str(STATIC / 'app.js'), str(STATIC / 'index.html'), str(inp)],
                       capture_output=True, text=True, timeout=120, env=dict(os.environ, TZ='Asia/Tokyo'))
    inp.unlink()
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def lines_of(confirm):
    keep = ('見張り', '残す', '天井', '発進の前進', '速度ゲイン')
    return [x for x in confirm.splitlines() if any(k in x for k in keep)]


# 台本の初めの読み込みは state.tab = 'live'(app.js の showTab('live'))のまま loadDrive を呼ぶので、まず本物の showTab('drive') で走行タブを開く
PRE = [['tab', 'drive'], ['dtimers']]
AWAY = [['tab', 'live'], ['tick'], ['dtimers']]           # ライブのタブへ。残っていた走行タブの輪を 1 回鳴らす(タブが live なので次を掛けない)
OUT = []


def show(name, res, note):
    OUT.append('===== %s: %s' % (name, note))
    for s in res['steps']:
        if s['op'] == 'start':
            OUT.append('  ▶ status=%s 確認の文の行=%s' % (s['status'], json.dumps([x[:40] for x in lines_of(s['confirm'])], ensure_ascii=False)))
        elif s['op'] == 'read' and s['what'] == 'server':
            v = s['value']
            OUT.append('  サーバー(走った値): 見張り=%s 残す=%s 天井=%s 左=%s launch_us=%s' % (
                v['rec_back_guard'], v['rec_hold_keep'], v['speed_ceiling'], v['ceil_left_mps'], v['launch_us']))
        elif s['op'] == 'calls':
            OUT.append('  通信(戻ってから ▶ まで): %s' % json.dumps([x[:2] + [x[2]] for x in s['value']], ensure_ascii=False))
        elif s['op'] == 'dtimers':
            OUT.append('  ライブのタブで残る走行タブのタイマー: %s' % s['value'])
        elif s['op'] == 'eval':
            OUT.append('  %s' % json.dumps(s['value'], ensure_ascii=False))
        elif s['op'] == 'el':
            OUT.append('  天井のスイッチ checked=%s' % s['value'])
        elif s['op'] == 'check':
            OUT.append('  天井のスイッチを押した checked=%s 確認=%s' % (s['checked'], (s['confirm'] or '')[:24]))
        elif s['op'] == 'api':
            OUT.append('  curl %s' % json.dumps({k: s['body'].get(k) for k in ('rec_back_guard', 'rec_hold_keep', 'ceil_left_mps', 'launch_us')}))
    OUT.append('  トースト(最後の 2): %s' % json.dumps(res['toasts'][-2:], ensure_ascii=False))


CEIL = ['el', '#dr-speed-ceiling', 'checked']
ST_EVAL = ['eval', 'JSON.stringify({paramsLoaded: state.drive.paramsLoaded, st_guard: state.drive.st.params.rec_back_guard, '
                   'sw_guard: document.querySelector("#dr-rec-back-guard").checked})']
MORNING_ON = sp(rec_back_guard=1, rec_hold_keep=1)

# A: 見張り・残す ON → ライブ → curl で見張り OFF(手順書 3 章の止めて OFF)→ 一覧の GET を止めたまま走行タブ → ブロック 1-2 → ▶
steps_a = PRE + AWAY + [['api', {'rec_back_guard': 0}], ['hold', MD, True], ['calls'], ['tab', 'drive'], ST_EVAL,
                  ['pick', di, blk(2)], ['start'], ['calls'], ['read', 'server'], ['release', MD], ['read', 'server']]
show('A_hold', run('A', MORNING_ON, steps_a), '一覧の GET を止めたまま戻ってすぐ ▶')
# A0(比べ): 同じで一覧の GET を止めない
steps_a0 = PRE + AWAY + [['api', {'rec_back_guard': 0}], ['calls'], ['tab', 'drive'], ST_EVAL, ['pick', di, blk(2)], ['start'], ['calls'], ['read', 'server']]
show('A0_nohold', run('A0', MORNING_ON, steps_a0), '一覧がすぐ返る')
# A1(比べ): 止めたまま、一覧の返事が来てから ▶
steps_a1 = PRE + AWAY + [['api', {'rec_back_guard': 0}], ['hold', MD, True], ['calls'], ['tab', 'drive'], ['release', MD], ST_EVAL,
                   ['pick', di, blk(2)], ['start'], ['calls'], ['read', 'server']]
show('A1_released', run('A1', MORNING_ON, steps_a1), '一覧の返事が来てから ▶')
# A5(比べ): 止めたまま 5 秒たつ(タイマーを全部鳴らす)→ ▶
steps_a5 = PRE + AWAY + [['api', {'rec_back_guard': 0}], ['hold', MD, True], ['calls'], ['tab', 'drive'], ['tick'], ST_EVAL,
                   ['pick', di, blk(2)], ['start'], ['calls'], ['read', 'server']]
show('A5_after5s', run('A5', MORNING_ON, steps_a5), '止めたまま 5 秒(DR_IDLE_WAIT_MS)たってから ▶')
# B: ライブ → curl で左 4.2・launch_us 200(手順書 5 章。ブロック 10 の after_press「API で ceil_left_mps 4.2 → ゲイン → 天井 ON」)→
#    一覧の GET を止めたまま走行タブ → ブロック 2-5(10)→ 天井のスイッチ ON → ▶
steps_b = PRE + AWAY + [['api', {'ceil_left_mps': 4.2, 'launch_us': 200}], ['hold', MD, True], ['calls'], ['tab', 'drive'],
                        ['pick', di, blk(10)], ['check', '#dr-speed-ceiling', True], CEIL, ['start'], ['calls'], ['read', 'server']]
show('B_hold_ceil', run('B', sp(), steps_b), '天井・curl 左 4.2 のあと、止めたまま戻ってすぐ ブロック 2-5 → 天井 ON → ▶')
steps_b0 = [s for s in steps_b if s != ['hold', MD, True]]
show('B0_nohold_ceil', run('B0', sp(), steps_b0), '同じで一覧がすぐ返る')
# C: ライブのタブにいない(走行タブのまま)で curl → 1 秒の輪のあと ▶(比べ: タブを離れなければ 1 秒で合う)
steps_c = PRE + [['api', {'rec_back_guard': 0}], ['tick'], ST_EVAL, ['start'], ['read', 'server']]
show('C_stay_drive', run('C', MORNING_ON, steps_c), '走行タブのまま(輪 1 回)')

txt = '\n'.join(OUT)
(HERE / 'out' / 'a2_tab_return_after.txt').write_text(txt + '\n', encoding='utf-8')
print(txt)
