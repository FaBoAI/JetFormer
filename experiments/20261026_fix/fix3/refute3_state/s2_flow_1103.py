#!/usr/bin/env python3
"""手順書 11/3(3 章・4 章・2 章の表・5 章の API)どおりに、本物の app.js で 1 日を流す。各 ▶ で送った値と、そのときのサーバーの
試験の選択肢・左の値を、計画(bench_candidates.json の expect と手順書)と比べる。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import cands, blk, run, server_params   # noqa: E402

c, di = cands()
G, H, NK = '#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-launch-no-kick'
CE, AU, LB = '#dr-speed-ceiling', '#dr-speed-auto', '#dr-launch-boost'
UP, DN = '#btn-dr-tg-up', '#btn-dr-tg-down'


def gain_to(cur, want):
    n = round((want - cur) / 0.01)
    return [['click', UP if n > 0 else DN]] * abs(n)


def one(no, gain, switches, desc_extra='battery=#A V_before=8.31', api=None):
    st = [['baseline'], ['pick', di, blk(c, di, no)]] + gain_to(0.90, gain)
    for a in api or []:
        st += [['server_set', a], ['realpoll']]
    for sw in switches:
        st.append(['check', sw, True])
    st += [['read', 'desc'], ['start'], ['read', 'server'], ['running', False]]
    return st


steps = [['baseline'],
         ['check', G, True], ['check', H, True], ['check', NK, True],                 # 3 章(使うと決めた日)
         ['baseline'], ['check', '#dr-record', False], ['check', LB, True], ['start'], ['read', 'server'], ['running', False]]  # 3 章 発進の前進の確かめ
plan = [
    (1, 0.92, [], {}), (2, 0.92, [], {}), (3, 0.92, [], {}), (4, 0.92, [CE], {'speed_ceiling': 1}), (5, 0.92, [], {}),
    (6, 0.97, [CE], {'speed_ceiling': 1}), (7, 0.95, [], {}), (8, 0.97, [CE, AU], {'speed_ceiling': 1, 'speed_auto': 1}),
    (9, 0.97, [CE, AU], {'speed_ceiling': 1, 'speed_auto': 1}),
    (10, 0.97, [CE, AU], {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.2}),
    (11, 0.97, [CE, AU], {'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.1}),
] + [(n, 0.97, [CE, AU] + ([LB] if n % 2 == 1 else []), {'speed_ceiling': 1, 'speed_auto': 1, 'launch_boost': 1 if n % 2 == 1 else 0})
     for n in range(12, 20)] + [(n, 0.97, [CE, AU, LB], {'speed_ceiling': 1, 'speed_auto': 1, 'launch_boost': 1}) for n in (20, 21, 22)]
exp = []
for no, g, sws, want in plan:
    api = [{'ceil_left_mps': 4.2}] if no == 10 else ([{'ceil_left_mps': 4.1}] if no == 11 else None)
    steps += one(no, g, sws, api=api)
    exp.append((no, g, want))
res = run(steps, server_params())
starts = [x for x in res['steps'] if x['op'] == 'start']
servers = [x['value'] for x in res['steps'] if x['op'] == 'read' and x['what'] == 'server']
print('morning check: sent launch=%s desc=%r record=%s' % (starts[0]['sent']['launch_boost'], starts[0]['desc'], starts[0]['record']))
bad = 0
for (no, g, want), s, srv in zip(exp, starts[1:], servers[1:]):
    sent = s['sent']
    probs = []
    if abs(sent['throttle_gain'] - g) > 1e-9:
        probs.append('gain %s' % sent['throttle_gain'])
    for k in ('speed_ceiling', 'speed_auto', 'launch_boost', 'zone_caps'):
        if int(sent[k]) != int(want.get(k, 0)):
            probs.append('%s=%s' % (k, sent[k]))
    if 'ceil_left_mps' in want and abs(srv['ceil_left_mps'] - want['ceil_left_mps']) > 1e-9:
        probs.append('left %s' % srv['ceil_left_mps'])
    for k in ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick'):
        if int(srv[k]) != 1:
            probs.append('%s=%s' % (k, srv[k]))
    lines = [l for l in s['confirm'].splitlines() if ': ON' in l or 'トリム' in l or '速度ゲイン' in l]
    print('block %2d %s %s' % (no, 'OK' if not probs else 'NG ' + ','.join(probs), ' | '.join(lines)[:260]))
    bad += bool(probs)
print('NG', bad)
