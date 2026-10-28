"""古い画面(app.js 10/27 18:45 = fix7 7-1 回目のあとの版の写し tests/fixtures/fix7_ui_r2_before/app.js)と、読み直した画面(今の app.js)を
同じサーバーで(sim2.js。サーバーは本物の server.py と同じに build が今のファイルより古ければ 409)。朝の流れの ▶ の結果(この係)"""
from run import run, blk, DI, RESET, HERE, APP
from pathlib import Path
OLD = Path('/home/jetson/jetformer/tests/fixtures/fix7_ui_r2_before/app.js')
SIM2 = HERE / 'sim2.js'
now = int(APP.stat().st_mtime)
old_build = 1793092000          # 10/27 18:45 ごろ(app.js 18:45 の版の ?v=)
steps = [['@', 'O', ['tab', 'drive']], ['@', 'N', ['tab', 'drive']], ['tickall'], ['tickall'],
         ['@', 'O', ['baseline']], ['@', 'O', ['check', '#dr-rec-back-guard', True]], ['@', 'O', ['check', '#dr-rec-hold-keep', True]],
         ['@', 'O', ['pick', DI, blk(1)]], ['@', 'O', ['start']], ['tickall'], ['@', 'O', ['toasts']],
         ['@', 'N', ['pick', DI, blk(1)]], ['@', 'N', ['start']], ['server']]
out = run(steps, sim=SIM2, pages=['O', 'N'], name='old.json', page_app={'O': str(OLD)},
          page_src={'O': 'http://x/static/app.js?v=%d' % old_build}, server_build=now)
lines = []
for s in out['steps']:
    if s.get('op') == 'start':
        lines.append('画面 %s の ▶: 返事 %s・確認の文の見張り %s' % (s['page'], s['status'], 'ON' if '後ろの見張り: ON' in s['confirm'] else '行なし'))
lines.append('古い画面の最後のトースト: %s' % out['toasts']['O'][-1])
lines.append('読み直した画面の最後のトースト: %s' % out['toasts']['N'][-1])
(HERE / 'out' / 'old_page.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
