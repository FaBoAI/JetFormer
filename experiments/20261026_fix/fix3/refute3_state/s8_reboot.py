#!/usr/bin/env python3
"""r1-state3-4 が直ったか: 朝に試験の選択肢 3 つ ON → 昼にサーバーが起動し直す(RESET_ON_BOOT)→ 画面の知らせ・スイッチ・確認の文"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import cands, blk, run, server_params   # noqa: E402
from jetformer import drive   # noqa: E402

c, di = cands()
G, H, NK = '#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-launch-no-kick'
reset = {k: drive.DEFAULT_PARAMS[k] for k in drive.RESET_ON_BOOT}
res = run([['check', G, True], ['check', H, True], ['check', NK, True], ['realpoll'],
           ['reboot', 2000.0, reset], ['realpoll'], ['el', '#dr-message', 'html'], ['el', G, 'checked'],
           ['baseline'], ['pick', di, blk(c, di, 3)], ['start']], server_params())
msg = [x['value'] for x in res['steps'] if x['op'] == 'el'][0]
print('知らせ:', '起動し直しました' in msg, '/ スイッチ(見張り):', [x['value'] for x in res['steps'] if x['op'] == 'el'][1])
s = [x for x in res['steps'] if x['op'] == 'start'][0]
print('確認の文に見張りの行:', 'リカバリーの後ろの見張り: ON' in s['confirm'])
print('トースト:', [t for t in res['toasts'] if '起動し直し' in t][:1])
