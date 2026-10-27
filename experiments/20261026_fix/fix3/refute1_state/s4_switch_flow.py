#!/usr/bin/env python3
"""再起動のあとの朝: 画面のスイッチで見張り → 残す → 発進の補助を出さない を ON(API ではなく)→ トリム +15(スライダー)→
回 3 の 3-8(発進 ON)→ 回 4 の 4-1(発進 ON・天井)→ 4-2 で見張りを OFF(残すも OFF になる)→ 4-3"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from flow_1103 import cands, blk, run, server_params
work = sys.argv[1]
c, di = cands()
UP = '#btn-dr-tg-up'
CE, AU, LB = '#dr-speed-ceiling', '#dr-speed-auto', '#dr-launch-boost'
G, H, K = '#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-launch-no-kick'
steps = [['check', H, True], ['read', 'server'], ['check', G, True], ['check', H, True], ['check', K, True], ['read', 'server'],
         ['slide', '#dr-trim', 15], ['baseline'],
         ['pick', di, blk(c, di, 19)], ['click', UP], ['click', UP], ['check', CE, True], ['check', LB, True], ['start'], ['running', False],
         ['pick', di, blk(c, di, 20)], ['click', UP], ['click', UP], ['check', CE, True], ['check', LB, True], ['start'], ['running', False],
         ['pick', di, blk(c, di, 21)], ['check', G, False], ['read', 'server'], ['click', UP], ['click', UP], ['check', CE, True], ['check', LB, True],
         ['start'], ['running', False],
         ['pick', di, blk(c, di, 22)], ['click', UP], ['click', UP], ['check', CE, True], ['start']]
res = run(work, steps, server_params())
for x in res['steps']:
    if x['op'] == 'read' and x['what'] == 'server':
        print('server:', {k: x['value'][k] for k in ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick')})
    if x['op'] == 'check':
        print('check', x['checked'], (x['confirm'] or '')[:30].replace('\n', ' '))
    if x['op'] == 'start':
        s = x['sent']
        print('start', {k: s.get(k) for k in ('throttle_gain', 'steer_trim_us', 'speed_ceiling', 'speed_auto', 'launch_boost')},
              [l for l in x['confirm'].splitlines() if any(w in l for w in ('トリム', '見張り', '残す', '補助も', '発進の前進:'))])
print([t for t in res['toasts'] if '見張り' in t or '残す' in t or '補助' in t])
