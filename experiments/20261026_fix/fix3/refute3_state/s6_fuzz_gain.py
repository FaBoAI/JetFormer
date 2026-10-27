#!/usr/bin/env python3
"""r1-state3-1 が直ったか: ブロック → ±0.01 を n 回(送信の返事・問い合わせの返事の遅れ・失敗をでたらめに)→ 全部戻ったあと
画面 = サーバー = 0.90 + 押した分(失敗した送信のあと、次の押しで送り直される分も)・確認の文の速度ゲイン = 画面"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import cands, blk, run, server_params   # noqa: E402

c, di = cands()
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
rng = random.Random(seed)
bad = 0
for i in range(N):
    steps, ng, net = [['baseline'], ['pick', di, blk(c, di, rng.choice([1, 5, 12, 13]))], ['defer', True]], 0, 0
    for _ in range(rng.randint(2, 12)):
        r = rng.random()
        if r < 0.45:
            up = rng.random() < 0.75
            steps.append(['tap', '#btn-dr-tg-up' if up else '#btn-dr-tg-down']); net += 1 if up else -1
            if rng.random() < 0.5:
                steps.append(['timers'])
        elif r < 0.6:
            steps.append(['gstart']); ng += 1
        elif r < 0.7 and ng:
            steps.append(['gend']); ng -= 1
        elif r < 0.85:
            steps.append(['resolve1'])
        elif r < 0.9:
            steps.append(['timers'])
        else:
            steps.append(['realpoll'])
    settle = [['timers'], ['resolve'], ['timers'], ['resolve'], ['timers'], ['resolve']] + [['gend']] * (ng + 1) + [['defer', False], ['realpoll'], ['realpoll']]
    sp = server_params(throttle_gain=rng.choice([0.9, 0.97, 1.0]))
    res = run(steps + settle + [['read', 'server'], ['read', 'gain'], ['start']], sp, extra={'apply_on_resolve': rng.random() < 0.5})
    out = res['steps']
    srv = [x['value'] for x in out if x['op'] == 'read' and x['what'] == 'server'][0]
    g = [x['value'] for x in out if x['op'] == 'read' and x['what'] == 'gain'][0]
    s = [x for x in out if x['op'] == 'start'][0]
    want = round(0.90 + 0.01 * net, 2)
    probs = []
    if abs(g - want) > 1e-9: probs.append('screen %s want %s' % (g, want))
    if abs(s['sent']['throttle_gain'] - want) > 1e-9: probs.append('sent %s' % s['sent']['throttle_gain'])
    if '速度ゲイン %.2f' % want not in s['confirm']: probs.append('confirm')
    if net != 0 and abs(srv['throttle_gain'] - want) > 1e-9: probs.append('server(before start) %s' % srv['throttle_gain'])
    if probs:
        bad += 1; print('CASE', i, probs, steps)
print('seed', seed, 'bad', bad, '/', N)
