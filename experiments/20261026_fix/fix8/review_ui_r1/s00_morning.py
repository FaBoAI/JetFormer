"""朝の流れ(画面の読み直し → 基準の設定 → 見張り → 残す → ブロック 1 → ▶)を今の app.js で。確かめの土台"""
import json
from run import run, blk, RESET
steps = [['tab', 'drive'], ['tick'], ['baseline'], ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True],
         ['pick', 0, 0]]
import run as R
steps = [['tab', 'drive'], ['tick'], ['baseline'], ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True],
         ['pick', R.DI, blk(1)], ['read', 'desc'], ['start'], ['posts'], ['read', 'server']]
out = run(steps, no_init=True)
for s in out['steps']:
    if s['op'] == 'start':
        print('confirm:', s['confirm'])
        print('sent:', json.dumps(s['sent'], ensure_ascii=False))
        print('status', s['status'], 'boot_at', s['boot_at'])
    elif s['op'] in ('read', 'posts', 'check'):
        print(s['op'], json.dumps(s, ensure_ascii=False)[:600])
print('toasts', out['toasts'])
