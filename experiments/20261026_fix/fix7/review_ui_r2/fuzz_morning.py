"""11/3 の朝の流れ(走行タブ → 基準の設定 → 見張り → 残す → ブロック → ゲイン → 天井・発進の前進 → ▶)に、起き直し・Wi-Fi の切れ
(返事だけ届かない POST・表示の問い合わせだけ途切れる・確認の文の間に落ちる・一覧の失敗)をでたらめに混ぜ、人は画面の文を読んで
(起き直しの知らせ・スイッチが OFF なら)見張り → 残す を入れ直す。次を見る(読むだけ):
  I1: ▶ が通ったら、確認の文の行と、サーバーが実際に走る値が同じ(common.confirm_vs_server)
  I2: 輪の問い合わせのあと(送っている物が無いとき)、走行タブの起き直しの文の「見張り・残すは ON / OFF」「今 X」がサーバーと同じ
  I3: 起き直しの知らせのトーストが出たら、その「今は …」の項目(無ければ全部既定)が、そのときのサーバーと同じ

    python3 fuzz_morning.py <一時の場所> <seed> <回数>
"""
import json
import random
import re
import sys

from common import RESET, blk, cands, confirm_vs_server, note_of, run_sim, server_params

tmp, seed, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = random.Random(seed)
c, di = cands()
BLOCKS = [blk(c, di, n) for n in (1, 5, 10, 15)]
GUARD, KEEP = '#dr-rec-back-guard', '#dr-rec-hold-keep'
NAMES = {'rec_back_guard': 'リカバリーの後ろの見張り', 'rec_hold_keep': '中立止めのあともリカバリーを残す'}


def user_flow():
    b = rng.choice(BLOCKS)
    return [['baseline'], ['check', GUARD, True], ['check', KEEP, True], ['pick', di, b], ['click', '#btn-dr-tg-up'],
            ['check', '#dr-speed-ceiling', True], ['check', '#dr-launch-boost', True], ['start']]


DIST = [
    lambda boot: [['reboot', boot, RESET]],
    lambda boot: [['reboot_on_confirm', boot, RESET]],
    lambda boot: [['down_on_confirm']],
    lambda boot: [['post_lost', 1]],
    lambda boot: [['status_fail_next', 1]],
    lambda boot: [['status_fail_next', 3]],
    lambda boot: [['down_all', True], ['tick'], ['down_all', False]],
    lambda boot: [['down_all', True], ['tick'], ['reboot', boot, RESET], ['down_all', False]],
    lambda boot: [['tab', 'live'], ['models_fail', True], ['tab', 'drive'], ['models_fail', False]],
]
OBS = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['read', 'server'], ['el', GUARD, 'checked'], ['el', KEEP, 'checked']]

bad = []
nstart = 0
for run in range(runs):
    boot = 1000.0
    steps = [['tab', 'drive'], ['tick']]
    flow = user_flow() + user_flow()
    for s in flow:
        if rng.random() < 0.25:
            if any(x[0] in ('reboot', 'reboot_on_confirm') for x in steps):
                pass
            boot2 = boot + 1000.0
            d = rng.choice(DIST)
            seq = d(boot2)
            if any(x[0] in ('reboot', 'reboot_on_confirm') for x in seq):
                boot = boot2
            steps += seq
        steps.append(s)
        if s[0] == 'start':
            steps += [['read', 'server'], ['tick']] + OBS + [['running', False], ['tick']] + OBS
            # 人: 起き直しを知った・スイッチが OFF なら入れ直す(この台本では、知らせが出たかに関わらず、入れ直しを毎回足す)
            steps += [['check', GUARD, True], ['tick'], ['check', KEEP, True], ['tick']] + OBS
        elif rng.random() < 0.7:
            steps += [['tick']] + OBS
        if s[0] == 'start':
            pass
    steps += [['start'], ['read', 'server'], ['tick']] + OBS
    fixed = [['press_if_off', x[1]] if x[0] == 'check' and x[2] is True else x for x in steps]
    try:
        res, toasts = run_sim(tmp, c, fixed, server=server_params(throttle_gain=0.9), name=f'fz{seed}.json')
    except RuntimeError as e:
        bad.append({'run': run, 'err': str(e)[-400:]})
        continue
    # I1
    for i, x in enumerate(res):
        if x['op'] == 'start' and x.get('status') == 200:
            nstart += 1
            srvp = res[i + 1]['value']
            mm = confirm_vs_server(x['confirm'], srvp)
            if mm:
                bad.append({'run': run, 'I1': mm, 'confirm': x['confirm'], 'steps': fixed})
    # I2: 輪の問い合わせのあとの文
    for i, x in enumerate(res):
        if x['op'] == 'el' and x['sel'] == '#dr-message':
            srvp = res[i + 1]['value']
            note = note_of(x['value'])
            if not note:
                continue
            for k, name in NAMES.items():
                says_on = re.search(r'([^、:]*)は ON', note)
                on_names = [n.strip() for n in says_on.group(1).split('・')] if says_on else []
                if (name in on_names) != (srvp.get(k) == 1):
                    bad.append({'run': run, 'I2': k, 'note': note, 'srv': {kk: srvp.get(kk) for kk in NAMES}})
print(json.dumps({'seed': seed, 'runs': runs, 'starts_ok': nstart, 'bad': bad[:5], 'n_bad': len(bad)}, ensure_ascii=False))
