#!/usr/bin/env python3
"""スイッチ(試験の選択肢 3 つ・天井・自動合わせ・発進の前進)の送信・問い合わせの入れ替わり・失敗をでたらめに混ぜ、全部の返事が
戻って表示の同期が 2 回済んだあと: スイッチ = サーバーの値・画面が持つサーバーの値 = サーバー・確認の文の行 = サーバーの値か"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run3 import run, server_params   # noqa: E402

SW = {'rec_back_guard': '#dr-rec-back-guard', 'rec_hold_keep': '#dr-rec-hold-keep', 'launch_no_kick': '#dr-launch-no-kick',
      'speed_ceiling': '#dr-speed-ceiling', 'speed_auto': '#dr-speed-auto', 'launch_boost': '#dr-launch-boost'}
LINE = {'rec_back_guard': 'リカバリーの後ろの見張り: ON', 'rec_hold_keep': '中立止めのあともリカバリーを残す: ON',
        'launch_no_kick': '発進の前進を出さない区間は発進の補助も出さない: ON', 'speed_ceiling': '速さの天井: ON',
        'speed_auto': '速さのゲインの自動合わせ: ON', 'launch_boost': '発進の前進: ON'}
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
rng = random.Random(seed)
bad = 0
for i in range(N):
    steps, ng = [], 0
    for _ in range(rng.randint(3, 14)):
        r = rng.random()
        if r < 0.45:
            k = rng.choice(list(SW))
            steps.append(['checkn', SW[k], rng.random() < 0.6])
        elif r < 0.6:
            steps.append(['gstart']); ng += 1
        elif r < 0.72 and ng:
            steps.append(['gend']); ng -= 1
        elif r < 0.84:
            steps.append(['resolve1'])
        elif r < 0.9:
            steps.append(['fail_next', 1])
        else:
            steps.append(['realpoll'])
    settle = [['resolve']] * 6 + [['gend']] * (ng + 2) + [['resolve']] * 3 + [['realpoll'], ['realpoll']]
    reads = [['read', 'server'], ['read', 'st_params']] + [['el', s, 'checked'] for s in SW.values()] + [['start']]
    init = {k: rng.random() < 0.3 for k in SW}
    if init['rec_hold_keep']:
        init['rec_back_guard'] = True
    if init['speed_auto']:
        init['speed_ceiling'] = True
    sp = server_params(**{k: int(v) for k, v in init.items()})
    aor = rng.random() < 0.5
    res = run(steps + settle + reads, sp, extra={'defer_posts': True, 'apply_on_resolve': aor})
    out = res['steps']
    srv = [x['value'] for x in out if x['op'] == 'read' and x['what'] == 'server'][0]
    stp = [x['value'] for x in out if x['op'] == 'read' and x['what'] == 'st_params'][0]
    elv = [x['value'] for x in out if x['op'] == 'el']
    stt = [x for x in out if x['op'] == 'start'][0]
    probs = []
    for (k, sel), ch in zip(SW.items(), elv):
        if bool(ch) != bool(srv[k]):
            probs.append('%s screen=%s server=%s' % (k, ch, srv[k]))
        if int(stp.get(k, -1)) != int(srv[k]):
            probs.append('%s st.params=%s server=%s' % (k, stp.get(k), srv[k]))
        has = any(l.startswith(LINE[k]) for l in stt['confirm'].splitlines())
        if has != bool(srv[k]):
            probs.append('%s confirm-line=%s server=%s' % (k, has, srv[k]))
    if probs:
        bad += 1
        print('CASE', i, 'aor', aor, 'init', {k: int(v) for k, v in init.items()})
        print('  steps', steps)
        print('  ', probs)
print('seed', seed, 'bad', bad, '/', N)
