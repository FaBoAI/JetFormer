#!/usr/bin/env python3
"""±0.01 のボタンを 2 回押す。1 回目の送信の返事が戻ってから 2 回目の 120 ms のタイマーが鳴るまでの間に、表示の同期(1 秒ごと)が
来ると、2 回目が消える(本物の app.js。サーバーは偽物。POST の返事だけ遅らせる)。対照: 同期が来ないとき"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from flow_1103 import cands, blk, run, server_params
work = sys.argv[1]
c, di = cands()
SIM = Path(__file__).with_name('sim_race.js')
UP = '#btn-dr-tg-up'
base = [['pick', di, blk(c, di, 1)], ['read', 'gain'], ['read', 'pending']]
race = base + [['tap', UP], ['timers'], ['tap', UP], ['resolve'], ['poll'], ['timers'], ['resolve'], ['poll'], ['read', 'server'], ['start']]
ctrl = base + [['tap', UP], ['timers'], ['tap', UP], ['resolve'], ['timers'], ['resolve'], ['poll'], ['read', 'server'], ['start']]
for name, steps in (('race', race), ('control', ctrl)):
    res = run(work, steps, server_params(), sim=SIM, extra={'defer_posts': True})
    st = res['steps']
    s = [x for x in st if x['op'] == 'start'][0]
    srv = [x['value'] for x in st if x['op'] == 'read' and x['what'] == 'server'][0]
    print(name, '| 押した後の表示:', [x.get('value') for x in st if x['op'] in ('tap',)],
          '| 同期のあと:', [(x.get('gain'), x.get('text')) for x in st if x['op'] == 'poll'],
          '| サーバー:', srv['throttle_gain'], '| 走行開始で送る:', s['sent']['throttle_gain'],
          '| 確認の文:', [l for l in s['confirm'].splitlines() if '速度ゲイン' in l])
