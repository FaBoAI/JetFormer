#!/usr/bin/env python3
"""画面を開いた直後に状態の問い合わせが 2 回失敗(モデル・ベンチマークは読めた)→ ブロック 5(天井 OFF)を押す → 問い合わせが
戻る → ▶。前の本(2-4)のサーバーの値: 天井 1・自動合わせ 1・発進の前進 1・ゲイン 0.97"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import cands, blk, run, server_params   # noqa: E402

c, di = cands()
sp = server_params(speed_ceiling=1, speed_auto=1, launch_boost=1, throttle_gain=0.97)
for fail in (0, 2):
    res = run([['pick', di, blk(c, di, 5)], ['read', 'pending'], ['read', 'gain'], ['realpoll'], ['read', 'gain'],
               ['el', '#dr-speed-ceiling', 'checked'], ['start']], sp, extra={'status_fail_first': fail})
    s = [x for x in res['steps'] if x['op'] == 'start'][0]
    print('status_fail_first=%d' % fail, [x.get('value') for x in res['steps'] if x['op'] in ('read', 'el')],
          'sent ceil=%s auto=%s launch=%s gain=%s' % (s['sent']['speed_ceiling'], s['sent']['speed_auto'], s['sent']['launch_boost'], s['sent']['throttle_gain']))
    print('   toasts:', [t for t in res['toasts'] if 'ブロック' in t or '読み込' in t])
