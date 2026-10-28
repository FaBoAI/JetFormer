"""でたらめな朝(1 つの画面か 2 つの画面。起き直し・Wi-Fi の切れ・表示の問い合わせだけ途切れ・curl・タブの行き来・スイッチ・基準の設定・
ブロック・ゲイン・▶・■)を sim2.js で流し、走り出した ▶ ごとに「確認の文の行 = サーバーが走った値」を照らす(この係)。
サーバー側が変わったあとは必ず 1 回以上 tickall(1 秒たった)を挟む(1 秒の輪の遅れは数えない)。モデルの一覧の GET は止めない。
出力 out/fuzz_invariant.txt"""
import json
import random
import re
import sys
from run import run, blk, DI, RESET, HERE

SIM2 = HERE / 'sim2.js'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def gen(rng, two):
    pages = ['A', 'B'] if two else ['A']
    steps = [['@', p, ['tab', 'drive']] for p in pages] + [['tickall'], ['tickall']]
    boot = 1000.0
    for _ in range(rng.randint(12, 30)):
        p = rng.choice(pages)
        k = rng.random()
        if k < 0.08:
            boot += 1000.0
            steps += [['reboot', boot, RESET], ['tickall']]
        elif k < 0.13:
            steps += [['down_all', True], ['tickall'], ['tickall'], ['down_all', False], ['tickall']]
        elif k < 0.17:
            steps += [['status_fail_next', rng.randint(1, 3)], ['tickall'], ['tickall']]
        elif k < 0.25:
            steps += [['api', rng.choice([{'ceil_left_mps': 4.2}, {'ceil_left_mps': 4.1}, {'launch_us': 200}, {'launch_us': 250},
                                          {'rec_back_guard': 1}, {'rec_back_guard': 1, 'rec_hold_keep': 1}, {'rec_back_guard': 0}])], ['tickall']]
        elif k < 0.33:
            steps += [['@', p, ['tab', rng.choice(['live', 'drive', 'drive'])]], ['tickall']]
        elif k < 0.45:
            sel = rng.choice(['#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-speed-ceiling', '#dr-launch-boost'])
            steps += [['@', p, ['check', sel, rng.random() < 0.7]], ['tickall']]
        elif k < 0.52:
            steps += [['@', p, ['baseline']], ['tickall']]
        elif k < 0.62:
            steps += [['@', p, ['pick', DI, blk(rng.randint(1, 23))]], ['tickall']]
        elif k < 0.68:
            steps += [['@', p, ['click', rng.choice(['#btn-dr-tg-up', '#btn-dr-tg-down'])]], ['tickall']]
        elif k < 0.70:
            steps += [['fail_next', 1], ['@', p, ['check', '#dr-rec-back-guard', True]], ['tickall']]
        elif k < 0.88:
            steps += [['@', p, ['start']], ['server'], ['tickall']]
        else:
            steps += [['@', p, ['stop']], ['tickall']]
    return steps


def check(st, srv):
    c = st['confirm']
    ss = st.get('server_at_start') or {}
    bad = []
    if st.get('status') != 200 or not c:
        return bad
    g = '\nリカバリーの後ろの見張り: ON' in c
    kp = '\n中立止めのあともリカバリーを残す: ON' in c
    if g != (ss.get('rec_back_guard') == 1):
        bad.append('見張り: 確認の文 %s・走った %s' % ('ON' if g else '行なし', ss.get('rec_back_guard')))
    if kp != (ss.get('rec_hold_keep') == 1):
        bad.append('残す: 確認の文 %s・走った %s' % ('ON' if kp else '行なし', ss.get('rec_hold_keep')))
    m = re.search(r'速さの天井: ON\(全体 ([\d.]+)・左旋回中 ([\d.]+) m/s', c)
    if m and abs(float(m.group(2)) - float(ss.get('ceil_left_mps'))) > 1e-6:
        bad.append('左: 確認の文 %s・走った %s' % (m.group(2), ss.get('ceil_left_mps')))
    if bool(m) != (ss.get('speed_ceiling') == 1):
        bad.append('天井: 確認の文 %s・走った %s' % ('ON' if m else '行なし', ss.get('speed_ceiling')))
    m = re.search(r'発進の前進: ON\(前進 (\d+) µs 以上', c)
    if m and abs(float(m.group(1)) - float(ss.get('launch_us'))) > 1e-6:
        bad.append('強さ: 確認の文 %s・走った %s' % (m.group(1), ss.get('launch_us')))
    if bool(m) != (ss.get('launch_boost') == 1):
        bad.append('発進の前進: 確認の文 %s・走った %s' % ('ON' if m else '行なし', ss.get('launch_boost')))
    m = re.search(r'速度ゲイン ([\d.]+)', c)
    if m and abs(float(m.group(1)) - float(ss.get('throttle_gain'))) > 0.005:
        bad.append('ゲイン: 確認の文 %s・走った %s' % (m.group(1), ss.get('throttle_gain')))
    return bad


rng = random.Random(SEED)
lines, nstart, nbad = [], 0, 0
for i in range(N):
    two = i % 2 == 1
    steps = gen(rng, two)
    out = run(steps, sim=SIM2, pages=['A', 'B'] if two else ['A'], name='fz_%d.json' % i)
    for j, s in enumerate(x for x in out['steps'] if x.get('op') == 'start'):
        if s.get('status') == 200:
            nstart += 1
        b = check(s, None)
        if b:
            nbad += 1
            lines.append('朝 %d(%s)の ▶ %d(画面 %s): %s' % (i, '2 画面' if two else '1 画面', j, s.get('page'), ' / '.join(b)))
    # 走り出したら止める(次の ▶ のため)は台本の ■ に任せる
summary = 'でたらめな朝 %d 本(種 %d)・走り出した ▶ %d・確認の文と走った値が違う ▶ %d' % (N, SEED, nstart, nbad)
lines.insert(0, summary)
(HERE / 'out' / ('fuzz_invariant_%d.txt' % SEED)).write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines[:40]))
