"""起き直しの知らせ(noteServerBoot のトーストと走行タブの文)が、知らせを出すときのサーバーの値を見ずに「既定に戻っています」
「…は OFF」と出す場面を、本物の app.js で示す(読むだけ)。
  朝: 見張り・残す ON、左 4.2、launch_us 200(boot_at 1000)。画面はライブのタブ(走行タブの問い合わせは止まっている)。
  jetformer-web が起き直す(2000)→ 画面が走行タブに戻る前に、端末の curl で朝の設定を入れ直した(手順書 3 章・5 章の API)。
  → 画面が走行タブを開くと、トーストは「朝に入れた設定(後ろの見張り・左の値・…)は既定に戻っています」、走行タブの文は
    「起動のたびに既定に戻るもの: …見張り・…残す・…は OFF、天井の左旋回中 4.2 m/s、発進の前進の強さ 200 µs」。
    スイッチ・確認の文・サーバーは見張り ON・残す ON・4.2・200(正しい)。
    python3 p_boot_note.py <一時の場所>
"""
import json
import sys

from common import RESET, cands, ops, run_sim, server_params

tmp = sys.argv[1]
c, di = cands()
MORNING = server_params(throttle_gain=0.9, rec_back_guard=1, rec_hold_keep=1, ceil_left_mps=4.2, launch_us=200.0)
REENTER = {'rec_back_guard': 1, 'rec_hold_keep': 1, 'ceil_left_mps': 4.2, 'launch_us': 200.0}
for name, after_reboot in (('curl で入れ直したあとに走行タブ', [['server_set', REENTER]]), ('入れ直す前に走行タブ(比べ)', [])):
    steps = ([['tab', 'drive'], ['tick'], ['tab', 'live'], ['tick'], ['reboot', 2000.0, RESET]] + after_reboot
             + [['tab', 'drive'], ['el', '#toast', 'text'], ['el', '#dr-message', 'html'],
                ['el', '#dr-rec-back-guard', 'checked'], ['el', '#dr-rec-hold-keep', 'checked'], ['el', '#dr-ceil-spec', 'text'],
                ['el', '#dr-launch-spec', 'text'], ['read', 'server']]
             + ([] if not after_reboot else [['tick'], ['check', '#dr-rec-back-guard', False], ['el', '#toast', 'text'], ['read', 'server']]))
    res, toasts = run_sim(tmp, c, steps, MORNING, name='bn.json')
    els = [x['value'] for x in res if x['op'] == 'el']
    srv = [x['value'] for x in res if x['op'] == 'read']
    print('==', name)
    print(' トースト:', els[0])
    note = els[1].split('<div class="warn-text">')[-1].replace('</div>', '')
    print(' 走行タブの文:', note)
    print(' スイッチ 見張り/残す:', els[2], els[3], ' 天井の横:', els[4], ' 発進の前進の横:', els[5])
    print(' サーバー:', {k: srv[0][k] for k in REENTER})
    if after_reboot:
        print(' 知らせどおり「見張りを入れ直す」つもりでスイッチを押すと → トースト:', els[6], ' サーバー:',
              {k: srv[1][k] for k in ('rec_back_guard', 'rec_hold_keep')})
