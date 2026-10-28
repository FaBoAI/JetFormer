"""11/3 の朝の流れ(読み直し → 基準の設定 → 見張り → 残す → ブロック 1 → ▶)の各すき間で起き直す(1 秒の輪が 1 回回る)。▶ の確認の文の
見張り・残すの行と、サーバーが走った値・最後のトースト(この係。sim2.js・1 つの画面)。出力 out/morning_reboot_gaps.txt"""
import json
from run import run, blk, DI, RESET, HERE

SIM2 = HERE / 'sim2.js'
FLOW = [['@', 'A', ['baseline']], ['@', 'A', ['check', '#dr-rec-back-guard', True]], ['@', 'A', ['check', '#dr-rec-hold-keep', True]],
        ['@', 'A', ['pick', DI, blk(1)]]]
NAMES = ['読み直しのあと', '基準の設定のあと', '見張りのあと', '残すのあと', 'ブロックのあと', '確認の文の間']
lines = []
for gap in range(6):
    steps = [['@', 'A', ['tab', 'drive']], ['tickall']]
    for i, s in enumerate(FLOW):
        if i == gap:
            steps += [['reboot', 2000.0, RESET], ['tickall']]
        steps += [s, ['tickall']]
    if gap == 4:
        steps += [['reboot', 2000.0, RESET], ['tickall']]
    if gap == 5:
        steps += [['reboot_on_confirm', 2000.0, RESET]]
    steps += [['@', 'A', ['start']], ['server'], ['tickall'], ['@', 'A', ['toasts']]]
    out = run(steps, sim=SIM2, pages=['A'], name='gap_%d.json' % gap)
    st = [s for s in out['steps'] if s.get('op') == 'start'][0]
    ss = st['server_at_start'] if st['status'] == 200 else None
    c = st['confirm']
    lines.append('%s: ▶ の返事 %s・確認の文の見張り %s・残す %s・サーバーが走った見張り %s・残す %s・最後のトースト「%s」' % (
        NAMES[gap], st['status'], 'ON' if '後ろの見張り: ON' in c else '行なし', 'ON' if 'リカバリーを残す: ON' in c else '行なし',
        ss and ss.get('rec_back_guard'), ss and ss.get('rec_hold_keep'), out['toasts']['A'][-1][:90]))
(HERE / 'out' / 'morning_reboot_gaps.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
