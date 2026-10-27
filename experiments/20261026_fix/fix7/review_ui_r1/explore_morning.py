"""11/3 の朝の流れ(読み直し → 基準の設定 → 見張り → 残す → ブロック → ▶)の間に、起き直し・Wi-Fi の切れを入れて、
▶ の確認の文と、サーバーが実際に走る値(走行開始のあとのサーバーの値)が合うかを総当たりで見る(読むだけ)。
    python3 explore_morning.py <一時の場所>
"""
import json
import re
import sys

from common import RESET, blk, cands, check, ops, run_sim, server_params

tmp = sys.argv[1]
c, di = cands()
B15 = blk(c, di, 15)
MORNING = [['tab', 'drive'], ['tick'],
           ['baseline'], ['tick'],
           ['check', '#dr-rec-back-guard', True], ['tick'],
           ['check', '#dr-rec-hold-keep', True], ['tick'],
           ['pick', di, B15], ['tick'],
           ['click', '#btn-dr-tg-up'],
           ['check', '#dr-speed-ceiling', True],
           ['check', '#dr-launch-boost', True], ['tick']]
DIS = {
    'R': [['reboot', 2000.0, RESET]],
    'RT': [['reboot', 2000.0, RESET], ['tick']],
    'RC': [['reboot_on_confirm', 2000.0, RESET]],
    'DN': [['down_all', True], ['tick'], ['tick'], ['reboot', 2000.0, RESET], ['down_all', False], ['tick']],
    'W': [['down_all', True], ['tick'], ['down_all', False]],
    'WT': [['down_all', True], ['tick'], ['tick'], ['down_all', False], ['tick']],
    'MF': [['tab', 'live'], ['tick'], ['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['tick']],
}
TAIL = [['start'], ['read', 'server'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html']]
# 断られたあと、手順書 7 章どおり入れ直してもう一度 ▶(見張り → 残す → 天井 → 発進の前進 は画面の今の値を見て OFF なら ON)
pos = [i for i, s in enumerate(MORNING) if s[0] != 'tick'] + [len(MORNING)]


rows = []
for dname, dis in DIS.items():
    for p in pos:
        steps = MORNING[:p] + dis + MORNING[p:] + TAIL
        res, toasts = run_sim(tmp, c, steps, server_params(throttle_gain=0.9))
        st = ops(res, 'start')[-1]
        srvp = [x['value'] for x in res if x['op'] == 'read'][-1]
        els = [x['value'] for x in res if x['op'] == 'el']
        where = MORNING[p][0] + (':' + MORNING[p][1] if p < len(MORNING) and len(MORNING[p]) > 1 and isinstance(MORNING[p][1], str) else '') if p < len(MORNING) else '▶ の直前'
        bad = check(st['confirm'], srvp, st['sent']) if st['status'] == 200 else []
        rows.append({'dis': dname, 'before': where, 'status': st['status'], 'confirm_shown': bool(st['confirm']),
                     'srv': {k: srvp[k] for k in ('rec_back_guard', 'rec_hold_keep', 'speed_ceiling', 'launch_boost', 'ceil_left_mps', 'launch_us', 'throttle_gain')},
                     'mismatch': bad, 'last_toast': els[0][:90], 'boot_note': '起動し直しました' in els[1]})
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
