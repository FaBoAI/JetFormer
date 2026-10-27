#!/usr/bin/env python3
"""r1-state3-2 が直ったか: ブロックのあとのスイッチ・±0.01 の送信が失敗したとき、走行開始で送る値と確認の文"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import cands, blk, run, server_params   # noqa: E402

c, di = cands()
CE, AU, LB, UP = '#dr-speed-ceiling', '#dr-speed-auto', '#dr-launch-boost', '#btn-dr-tg-up'
prev = server_params(speed_ceiling=1, speed_auto=1, launch_boost=1, throttle_gain=0.97)
cases = {
    'ceil ON fail': [['baseline'], ['pick', di, blk(c, di, 12)], ['fail_next', 1], ['check', CE, True], ['poll'], ['poll'], ['start']],
    'launch ON fail': [['baseline'], ['pick', di, blk(c, di, 13)], ['fail_next', 1], ['check', LB, True], ['poll'], ['poll'], ['start']],
    'auto ON fail': [['baseline'], ['pick', di, blk(c, di, 8)], ['check', CE, True], ['fail_next', 1], ['check', AU, True], ['poll'], ['poll'], ['start']],
    'gain fail': [['baseline'], ['pick', di, blk(c, di, 5)], ['fail_next', 1], ['click', UP], ['poll'], ['poll'], ['start']],
    'gain fail x2 then ok': [['baseline'], ['pick', di, blk(c, di, 5)], ['fail_next', 1], ['click', UP], ['click', UP], ['poll'], ['start']],
    'ceil ON then OFF fail': [['baseline'], ['pick', di, blk(c, di, 5)], ['check', CE, True], ['fail_next', 1], ['check', CE, False], ['poll'], ['poll'], ['start']],
}
for name, st in cases.items():
    res = run(st, prev)
    s = [x for x in res['steps'] if x['op'] == 'start'][0]
    p = s['sent']
    lines = [l for l in s['confirm'].splitlines() if ('天井' in l or '自動' in l or '発進の前進:' in l or '速度ゲイン' in l)]
    print('%-22s sent ceil=%s auto=%s launch=%s gain=%s | %s' % (name, p['speed_ceiling'], p['speed_auto'], p['launch_boost'], p['throttle_gain'], ' / '.join(lines)[:200]))
