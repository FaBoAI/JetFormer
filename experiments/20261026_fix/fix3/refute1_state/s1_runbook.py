#!/usr/bin/env python3
"""手順書の 11/3 の流れ(朝に見張り + 残すを API で ON → 基準 → ブロック 1・4・8・9・10・11・12・13・20)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from flow_1103 import cands, blk, run, server_params

work = sys.argv[1]
c, di = cands()
UP, DN = '#btn-dr-tg-up', '#btn-dr-tg-down'
CE, AU, LB = '#dr-speed-ceiling', '#dr-speed-auto', '#dr-launch-boost'
steps = [['server_set', {'rec_back_guard': 1, 'rec_hold_keep': 1}], ['poll'], ['baseline'], ['read', 'check_base']]
plan = [
    (1, 0.92, []), (4, 0.92, [CE]), (8, 0.97, [CE, AU]), (9, 0.97, [CE, AU]),
    ('L42', None, None), (10, 0.97, [CE]), ('L41', None, None), (11, 0.97, [CE]),
    (12, 0.97, [CE, AU]), (13, 0.97, [CE, AU, LB]), (14, 0.97, [CE, AU]), (20, 0.97, [CE, AU, LB]), (21, 0.97, [CE, AU, LB]),
]
for no, g, sw in plan:
    if no == 'L42':
        steps.append(['server_set', {'ceil_left_mps': 4.2}]); steps.append(['poll']); continue
    if no == 'L41':
        steps.append(['server_set', {'ceil_left_mps': 4.1}]); steps.append(['poll']); continue
    steps.append(['pick', di, blk(c, di, no)])
    n = round((g - 0.90) / 0.01)
    steps += [['click', UP]] * n
    for s in sw:
        steps.append(['check', s, True])
    steps += [['poll'], ['poll'], ['start'], ['running', False], ['poll']]
res = run(work, steps, server_params())
out = []
k = 0
for x in res['steps']:
    if x['op'] == 'start':
        s = x['sent'] or {}
        conf = x['confirm']
        out.append({'block': [p for p in plan if p[0] not in ('L42', 'L41')][k][0],
                    'sent': {kk: s.get(kk) for kk in ('throttle_gain', 'steer_trim_us', 'speed_ceiling', 'speed_auto', 'launch_boost', 'zone_caps', 'start_kick')},
                    'opt_in_sent': [kk for kk in ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick') if kk in s],
                    'confirm_lines': [l for l in conf.splitlines() if any(w in l for w in ('速度ゲイン', '天井', '自動合わせ', '発進の前進', '見張り', '残す', 'トリム'))],
                    'client': x['client']})
        k += 1
    elif x['op'] == 'read':
        out.append({'read': x['what'], 'value': x['value']})
print(json.dumps(out, ensure_ascii=False, indent=1))
print('toasts:', json.dumps(res['toasts'][-40:], ensure_ascii=False))
