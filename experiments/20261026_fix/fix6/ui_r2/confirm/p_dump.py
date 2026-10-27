"""指摘 w2-ui-w1-ui-r62-1 の A・B を、直す前の写し(6-1 回目のあと)と今の app.js で流して、確認の文・トースト・輪を出す(読むだけ)。
    cd ~/jetformer && jf-heavy --lock none --mem 1500M -- python experiments/20261026_fix/fix6/ui_r2/confirm/p_dump.py <一時の場所>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tests')
sys.path.insert(0, '/home/jetson/jetformer')
import test_ui_fix6_r2 as T  # noqa: E402

tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)
c = T.bench.load_candidates()
di = next(i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03')
c = dict(c, today='2026-11-03')
b10 = T.blk(c, di, 10)
SC = {
    'B(一覧の GET だけ失敗 → curl 4.2)': (T.BASE + [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['api', {'ceil_left_mps': 4.2}]]
                                     + T.LOOK + [['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['tick'], ['start'], ['read', 'server']], T.PRE_B),
    'A(落ちる → 走行タブ → 既定で起き直す)': (T.BASE + [['down_all', True], ['tab', 'drive'], ['tick'], ['reboot', 2000.0, T.RESET], ['down_all', False]]
                                       + T.LOOK + [['pick', di, b10], ['check', '#dr-speed-ceiling', True], ['start'], ['read', 'server']], T.PRE_A),
}
out = {}
for name, (steps, sp) in SC.items():
    for label, app in (('before', T.BEFORE / 'app.js'), ('after', None)):
        res, toasts = T.run_sim(tmp, c, steps, app=app, name='%s.json' % label, server=sp)
        lk = T.look(res)
        s = T.starts(res)[-1]
        out.setdefault(name, {})[label] = {
            'LOOK(タイマー・輪・paramsLoaded・左・▶押せない・天井の横の文・つながっていません)': lk,
            'スイッチの確認の文の左': T.left_lines(T.ops(res, 'check')[-1]['confirm']),
            '▶ の確認の文': s['confirm'].split('\n'), '▶ の返事': s.get('status'), 'boot_at': s['boot_at'],
            'サーバー': {k: T.vals(res, 'read')[-1][k] for k in ('ceil_left_mps', 'speed_ceiling', 'launch_us', 'rec_back_guard', 'rec_hold_keep')},
            'トースト': toasts}
print(json.dumps(out, ensure_ascii=False, indent=1))
