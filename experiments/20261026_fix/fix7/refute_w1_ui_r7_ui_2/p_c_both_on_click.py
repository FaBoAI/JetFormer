"""反証用(w1-ui-r7-ui-2。読むだけ): 11/3 の決め (c) 両方 ON の朝の流れに click・slide(と、輪が先に鳴る tap+tick)を入れ、
手順書 5 章の curl で左 4.2 にしてから ▶。もとの台本と「長いタイマーを捨てない」写しで、▶ の確認の文とサーバーが実際に走る値が合うかを見る。
    python3 p_c_both_on_click.py <一時の場所>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/fix7/review_ui_r1')
import common  # noqa: E402
from common import blk, cands, check, ops, run_sim, server_params  # noqa: E402

tmp = sys.argv[1]
c, di = cands()
B10 = blk(c, di, 10)
HEAD = [['tab', 'drive'], ['tick'], ['baseline'], ['tick'], ['check', '#dr-rec-back-guard', True], ['tick'],
        ['check', '#dr-rec-hold-keep', True], ['tick'], ['pick', di, B10], ['tick']]
VAR = {
    'click': [['click', '#btn-dr-tg-up']],
    'slide': [['slide', '#dr-throttle-gain', 0.95]],
    'click+slide': [['click', '#btn-dr-tg-up'], ['slide', '#dr-trim', 10]],
    'tap+tick(輪が先)': [['tap', '#btn-dr-tg-up'], ['tick']],
}
TAIL = [['api', {'ceil_left_mps': 4.2}], ['tick'], ['tick'], ['check', '#dr-speed-ceiling', True], ['tick'], ['start'], ['read', 'server'],
        ['read', 'gain'], ['el', '#toast', 'text']]
HERE = Path(__file__).resolve().parent
for label, sim in (('もと', Path('/home/jetson/jetformer/tests/fixtures/app_sim_fix6_r2.js')), ('捨てない', HERE / 'keep_app_sim_fix6_r2.js')):
    common.SIM = sim
    for vname, v in VAR.items():
        res, toasts = run_sim(tmp, c, HEAD + v + TAIL, server_params(throttle_gain=0.9), name='c.json')
        st = ops(res, 'start')[-1]
        reads = [x['value'] for x in res if x['op'] == 'read']
        srvp, gain = reads[-2], reads[-1]
        bad = check(st['confirm'], srvp, st['sent']) if st['status'] == 200 else ['status %s' % st['status']]
        print(json.dumps({'台本': label, '操作': vname, 'status': st['status'], 'ずれ': bad, '画面のゲイン': gain,
                          'サーバー': {k: srvp[k] for k in ('rec_back_guard', 'rec_hold_keep', 'speed_ceiling', 'ceil_left_mps', 'throttle_gain', 'steer_trim_us')},
                          '確認の文の天井の行': [x for x in st['confirm'].split('\n') if '天井' in x or '見張り' in x or '残す' in x]}, ensure_ascii=False))
