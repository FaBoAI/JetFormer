"""本物の app.js に、朝の操作・ブロック・スイッチ・curl・起き直し・Wi-Fi の切れ・タブの行き来をでたらめな順で入れ、
走行開始が通った(200)ときに ▶ の確認の文とサーバーが実際に走る値が合うか、断られたときに最後に見える文が「始まっていない」と
分かるかを見る(読むだけ)。  python3 fuzz_confirm.py <一時の場所> [回数] [種]"""
import json
import random
import sys

from common import RESET, blk, cands, ops, run_sim, server_params
from common import check

tmp, N, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40, int(sys.argv[3]) if len(sys.argv) > 3 else 1
c, di = cands()
blocks = [int(b['block']) for b in c['days'][di]['blocks'] if b.get('drivable')]
rng = random.Random(seed)
boot = [1000.0]


def op():
    r = rng.random()
    if r < 0.10:
        return [['baseline']]
    if r < 0.30:
        k = rng.choice(['#dr-rec-back-guard', '#dr-rec-hold-keep', '#dr-speed-ceiling', '#dr-speed-auto', '#dr-launch-boost'])
        return [['check', k, rng.random() < 0.75]]
    if r < 0.40:
        return [['pick', di, blk(c, di, rng.choice(blocks))]]
    if r < 0.48:
        return [['click', rng.choice(['#btn-dr-tg-up', '#btn-dr-tg-down'])]]
    if r < 0.53:
        return [['server_set', rng.choice([{'ceil_left_mps': 4.2}, {'ceil_left_mps': 4.1}, {'launch_us': 200.0}, {'launch_us': 250.0}])], ['tick']]
    if r < 0.57:
        boot[0] += 1000.0
        return [['reboot', boot[0], RESET]]
    if r < 0.60:
        boot[0] += 1000.0
        return [['reboot_on_confirm', boot[0], RESET]]
    if r < 0.66:
        return [['down_all', True], ['tick']] + ([['tick']] if rng.random() < 0.5 else []) + [['down_all', False]]
    if r < 0.70:
        return [['tab', 'live'], ['tick'], ['models_fail', rng.random() < 0.5], ['tab', 'drive'], ['models_fail', False]]
    if r < 0.74:
        return [['hold', '/api/drive/status', True], ['tick']] + ([['tick']] if rng.random() < 0.3 else []) + [['release', '/api/drive/status'], ['hold', '/api/drive/status', False]]
    if r < 0.77:
        return [['start_fail']]
    if r < 0.79:
        return [['undo']]
    if r < 0.85:
        return [['tick']]
    return [['start'], ['read', 'server'], ['el', '#toast', 'text'], ['running', False], ['tick']]


bad = []
for t in range(N):
    boot[0] = 1000.0
    steps = [['tab', 'drive'], ['tick']]
    for _ in range(rng.randint(8, 30)):
        steps += op()
    steps += [['tick'], ['start'], ['read', 'server'], ['el', '#toast', 'text']]
    try:
        res, toasts = run_sim(tmp, c, steps, server_params(throttle_gain=0.9), name='f%d.json' % t)
    except RuntimeError as e:
        bad.append({'t': t, 'err': str(e)[-500:]})
        continue
    sts = ops(res, 'start')
    reads = [x['value'] for x in res if x['op'] == 'read']
    els = [x['value'] for x in res if x['op'] == 'el']
    for k, s in enumerate(sts):
        if s.get('status') == 200:
            m = check(s['confirm'], reads[k], s['sent'])
            if m:
                bad.append({'t': t, 'start': k, 'mismatch': m, 'steps': steps})
        elif s.get('status') == 409:
            if '始まっていません' not in els[k] and '古いまま' not in els[k] and '始めません' not in els[k]:
                bad.append({'t': t, 'start': k, 'last': els[k], 'steps': steps})
print(json.dumps({'runs': N, 'seed': seed, 'bad': bad}, ensure_ascii=False)[:20000])
