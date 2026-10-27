"""確認の文を読んでいる間にサーバーが落ち(systemd が起こし直すまで約 8 秒。10/27 17:16 の再起動は止めてから 4 秒で起こし、
boot_at はさらに 4 秒後)、起き上がる前に OK を押した場面(▶ の POST は届かない)。fix6 r1 の直し(w1-ui-ui61-2)は「同期が通る」
場面 F しか見ていない。同期も通らないと、断った文は出し直されず、あとで輪の問い合わせの知らせ「…既定に戻っています」が最後に残る。

    python3 p_start_crash.py <一時の場所>
"""
import json
import sys

from common import FIXT, RESET, cands, run_sim

tmp = sys.argv[1]
c, di = cands()
LOOK = [['el', '#toast', 'text'], ['el', '#dr-phase', 'text'], ['eval', 'state.drive.paramsLoaded']]
SCENES = {
    # S1: 確認の文の間に落ちる → OK(届かない)→ catch の同期も落ちたまま → 起き上がる → 輪の問い合わせで知らせ
    'S1 落ちたまま OK → あとで起き上がる': [['down_on_confirm'], ['start']] + LOOK + [['tick']] + LOOK
        + [['reboot', 2000.0, RESET], ['down_all', False], ['tick']] + LOOK + [['el', '#dr-message', 'html']],
    # 比べ: fix6 r1 の場面 F(確認の文の間に起き直し、POST は届かない、同期は通る)
    'F 同期は通る(fix6 r1 の場面)': [['reboot_on_confirm', 2000.0, RESET], ['start_fail'], ['start']] + LOOK + [['tick']] + LOOK,
    # S1c: 起き上がったあとに OK(409)、ただし catch の同期だけ通信の途切れ → 次の輪で知らせ
    'S1c 409・同期だけ落ちる': [['reboot_on_confirm', 2000.0, RESET], ['status_fail_next', 1], ['start']] + LOOK + [['tick']] + LOOK,
}
apps = {'今の app.js': None, 'fix5 のあと(fix6 r1 の直す前)': FIXT / 'fix6_ui_before' / 'app.js'}
for name, steps in SCENES.items():
    steps = [['tab', 'drive'], ['tick']] + steps                # 走行タブを開いて輪を回す(本物の画面と同じ)
    for an, app in apps.items():
        res, toasts = run_sim(tmp, c, steps, app=app)
        st = [x for x in res if x['op'] == 'start'][0]
        looks = [x['value'] for x in res if x['op'] in ('el', 'eval')]
        print(json.dumps({'場面': name, '版': an, '▶ の返事': st.get('status', '届かない'), 'POST': st['sent'] is not None,
                          '見えるもの(トースト・バッジ・同期済み)の移り': [looks[i:i + 3] for i in range(0, len(looks) - (len(looks) % 3), 3)],
                          'トーストの列': toasts}, ensure_ascii=False))
