"""反証 w2-ui-w2-ui-r72-ui-1 を打ち消せるか確かめる(読むだけ。本物の app.js・index.html を review_ui_r2/sim_r2.js で動かす)。
車・jetformer-web には触れない。本物のファイルは変えない。
    python p_check.py <一時の場所>"""
import json
import sys
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/fix7/review_ui_r2')
from common import RESET, cands, run_sim  # noqa: E402

tmp = sys.argv[1]
c, di = cands()
LOOK = [['el', '#toast', 'text'], ['el', '#toast', 'hidden'], ['el', '#dr-phase', 'text'], ['el', '#btn-dr-start', 'disabled'],
        ['el', '#dr-message', 'html']]
SCENES = {
    # S1 を長めに: 落ちている間に輪が 4 回失敗(約 4 秒)→ 起き上がる → 2 回同期
    'S1 長め': [['down_on_confirm'], ['start']] + LOOK + [['tick']] + LOOK + [['tick']] + [['tick']] + [['tick']] + LOOK
        + [['reboot', 2000.0, RESET], ['down_all', False], ['tick']] + LOOK + [['tick']] + LOOK,
    # 比べ: 落ちたまま OK、でも起き直さない(Wi-Fi の途切れだけ。boot_at は同じ)
    'S1w 途切れだけ': [['down_on_confirm'], ['start']] + LOOK + [['tick']] + LOOK + [['down_all', False], ['tick']] + LOOK,
    # S1c
    'S1c 409・同期だけ落ちる': [['reboot_on_confirm', 2000.0, RESET], ['status_fail_next', 1], ['start']] + LOOK + [['tick']] + LOOK,
}
for name, steps in SCENES.items():
    steps = [['tab', 'drive'], ['tick']] + steps
    res, toasts = run_sim(tmp, c, steps)
    st = [x for x in res if x['op'] == 'start'][0]
    looks = [x['value'] for x in res if x['op'] == 'el']
    print(json.dumps({'場面': name, '▶ の返事': st.get('status', '届かない'), 'POST': st['sent'] is not None,
                      '移り(トースト・隠れ・バッジ・▶押せない・走行タブの文)': [looks[i:i + 5] for i in range(0, len(looks), 5)],
                      'トーストの列': toasts[2:]}, ensure_ascii=False, indent=1))
