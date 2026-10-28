"""2 つの画面(A = スマホ・B = もう 1 つの画面)を同じ作り物のサーバーで(sim2.js)。B が走行タブを離れている間に A が設定を変え、
B が走行タブに戻った直後(モデルの一覧の GET を待っている間)に ▶ を押すと、確認の文は何を読み、サーバーは何で走るか。
読むだけ(本物の app.js・index.html。jetformer-web・車には触れない)。出力 out/two_pages.txt"""
import json
import sys
from run import run, blk, DI, RESET, HERE

SIM2 = HERE / 'sim2.js'
lines = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    lines.append(s)
    print(s)


def confirm_lines(c):
    return [x for x in c.split('\n') if x.startswith(('リカバリーの後ろの見張り', '中立止めのあとも', '速さの天井', '発進の前進', 'モデル', 'ジャンル'))]


def case(name, mid, hold_models=True):
    steps = [['@', 'A', ['tab', 'drive']], ['@', 'B', ['tab', 'drive']], ['tickall'], ['tickall'],
             ['@', 'A', ['baseline']], ['@', 'A', ['check', '#dr-rec-back-guard', True]], ['@', 'A', ['check', '#dr-rec-hold-keep', True]],
             ['tickall'], ['@', 'B', ['tab', 'live']], ['tickall'], ['tickall']] + mid + (
            [['hold', '/api/drive/models', True]] if hold_models else []) + [
             ['@', 'B', ['tab', 'drive']],
             ['@', 'B', ['start']], ['server']] + ([['release', '/api/drive/models']] if hold_models else []) + [['tickall'],
             ['@', 'B', ['el', '#dr-rec-back-guard', 'checked']], ['@', 'B', ['toasts']]]
    out = run(steps, sim=SIM2, pages=['A', 'B'], name='two_%d.json' % len(lines))
    st = [s for s in out['steps'] if s.get('op') == 'start'][0]
    srv = [s for s in out['steps'] if s.get('op') == 'server'][0]
    log('## ' + name)
    log('  B の ▶ の確認の文(抜き出し):', ' / '.join(confirm_lines(st['confirm'])))
    log('  送った body の params(抜き出し):', json.dumps({k: (st['sent'] or {}).get(k) for k in ('throttle_gain', 'speed_ceiling', 'launch_boost')}))
    ss = st['server_at_start'] or {}
    log('  サーバーが走った値: 見張り %s・残す %s・左 %s・launch_us %s・天井 %s(返事 %s・running %s)' % (
        ss.get('rec_back_guard'), ss.get('rec_hold_keep'), ss.get('ceil_left_mps'), ss.get('launch_us'), ss.get('speed_ceiling'), st['status'], srv['running']))
    log('  そのあとの B のスイッチ(見張り):', [s['value'] for s in out['steps'] if s.get('op') == 'el'])
    return out


# 1. A が 3 章の「止めて OFF」(見張りを OFF = 残すもサーバーが OFF)にしたあと、B が戻って ▶
case('1 A が見張りを OFF(3 章で止めて OFF)→ B が走行タブに戻ってすぐ ▶(一覧の GET を待っている間)',
     [['@', 'A', ['check', '#dr-rec-back-guard', False]], ['tickall']])
# 2. 比べ: 一覧の GET がすぐ返る(B の表示が先に今の値になる)
case('2 比べ: 同じ流れで一覧がすぐ返る', [['@', 'A', ['check', '#dr-rec-back-guard', False]], ['tickall']], hold_models=False)
# 3. curl で左 4.2・launch_us 200(手順書 5 章)を入れたあと、B が戻って ▶(天井 ON の本)
case('3 curl で左 4.2(手順書 5 章)→ B が戻ってすぐ ▶',
     [['api', {'ceil_left_mps': 4.2, 'launch_us': 200}], ['@', 'B', ['eval', "document.querySelector('#dr-speed-ceiling').checked = true; 1"]]])
(HERE / 'out' / 'two_pages.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# 4. 1 つの画面: A がライブのタブにいる間に、手順書 3 章の API(curl の 1 を 0 = 見張り OFF)で OFF → A が走行タブに戻ってすぐ ▶
def case1(name, mid, hold_models=True):
    steps = [['@', 'A', ['tab', 'drive']], ['tickall'], ['tickall'],
             ['@', 'A', ['baseline']], ['@', 'A', ['check', '#dr-rec-back-guard', True]], ['@', 'A', ['check', '#dr-rec-hold-keep', True]],
             ['tickall'], ['@', 'A', ['tab', 'live']], ['tickall'], ['tickall']] + mid + (
            [['hold', '/api/drive/models', True]] if hold_models else []) + [
             ['@', 'A', ['tab', 'drive']], ['@', 'A', ['pick', DI, blk(2)]], ['@', 'A', ['start']], ['server']] + (
            [['release', '/api/drive/models']] if hold_models else []) + [['tickall'], ['@', 'A', ['el', '#dr-rec-back-guard', 'checked']]]
    out = run(steps, sim=SIM2, pages=['A'], name='one_%d.json' % len(lines))
    st = [s for s in out['steps'] if s.get('op') == 'start'][0]
    srv = [s for s in out['steps'] if s.get('op') == 'server'][0]
    log('## ' + name)
    log('  A の ▶ の確認の文(抜き出し):', ' / '.join(confirm_lines(st['confirm'])))
    ss = st['server_at_start'] or {}
    log('  サーバーが走った値: 見張り %s・残す %s(返事 %s・running %s)。説明欄 %s' % (ss.get('rec_back_guard'), ss.get('rec_hold_keep'), st['status'], srv['running'], (st['desc'] or '')[:60]))
    log('  そのあとの A のスイッチ(見張り):', [s['value'] for s in out['steps'] if s.get('op') == 'el'])


case1('4 1 つの画面: ライブのタブの間に curl で見張り OFF(手順書 3 章の API)→ 走行タブに戻ってすぐブロック 1-2 → ▶',
      [['api', {'rec_back_guard': 0}]])
case1('5 比べ: 同じ流れで一覧がすぐ返る', [['api', {'rec_back_guard': 0}]], hold_models=False)
(HERE / 'out' / 'two_pages.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
