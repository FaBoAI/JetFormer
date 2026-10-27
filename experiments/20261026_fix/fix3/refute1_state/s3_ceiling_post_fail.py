#!/usr/bin/env python3
"""前の本(2-4)が天井 + 自動合わせ ON → 2-6(ブロック 11。天井だけ)を選ぶ → 天井 ON の送信が通信の途切れで失敗 → 表示の同期 →
走行開始。天井を押した時点で「OFF にした印」(pending)を消すので、同期でサーバーに残った前の本の ON(天井・自動合わせ)が画面に戻る"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from flow_1103 import cands, blk, run, server_params
work = sys.argv[1]
c, di = cands()
SIM = Path(__file__).with_name('sim_race.js')
CE = '#dr-speed-ceiling'
for name, fail in (('通信が途切れた', True), ('対照(途切れない)', False)):
    steps = [['pick', di, blk(c, di, 11)], ['read', 'ceil'], ['read', 'auto'], ['read', 'pending']]
    if fail:
        steps.append(['fail_next', 1])
    steps += [['check', CE, True], ['read', 'ceil'], ['read', 'auto'], ['poll'], ['read', 'ceil'], ['read', 'auto'],
              ['read', 'pending'], ['read', 'server'], ['start']]
    res = run(work, steps, server_params(speed_ceiling=1, speed_auto=1), sim=SIM)
    st = res['steps']
    rd = [(x['what'], x['value']) for x in st if x['op'] == 'read' and x['what'] in ('ceil', 'auto', 'pending')]
    s = [x for x in st if x['op'] == 'start'][0]
    print(name)
    print('  表示(押す前・押した直後・同期のあと):', rd)
    print('  走行開始で送る: speed_ceiling', s['sent']['speed_ceiling'], 'speed_auto', s['sent']['speed_auto'])
    print('  確認の文:', [l for l in s['confirm'].splitlines() if '天井' in l or '自動合わせ' in l])
    print('  トースト:', [t for t in res['toasts'] if '天井' in t or 'fetch' in t or '途切れ' in t])
