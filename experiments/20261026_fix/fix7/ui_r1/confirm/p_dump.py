"""fix7 画面の係 7-1 回目: 起き直しの知らせの文を、直す前の写し(tests/fixtures/fix7_ui_before/app.js)と今の app.js で並べて出す(読むだけ)。
台本は直した tests/fixtures/app_sim_fix7.js。車・jetformer-web には触れない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \\
        experiments/20261026_fix/fix7/ui_r1/confirm/p_dump.py <一時の場所> > experiments/20261026_fix/fix7/ui_r1/confirm/out_dump.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tests')
sys.path.insert(0, '/home/jetson/jetformer')
import test_ui_fix7 as T  # noqa: E402

tmp = Path(sys.argv[1])
tmp.mkdir(parents=True, exist_ok=True)
c = dict(T.F6.bench.load_candidates(), today='2026-11-03')
di = next(i for i, d in enumerate(c['days']) if d['date'] == '2026-11-03' and d['blocks'])
LOOK = [['el', '#toast', 'text'], ['el', '#dr-message', 'html'], ['el', '#dr-rec-back-guard', 'checked'], ['el', '#dr-rec-hold-keep', 'checked'],
        ['el', '#dr-ceil-spec', 'text'], ['el', '#dr-launch-spec', 'text'], ['read', 'server']]
SCENES = {
    '指摘の場面: ライブで起き直し → 走行タブの前に curl で朝の設定': [['tab', 'drive'], ['tick'], ['tab', 'live'], ['tick'], ['reboot', 2000.0, T.RESET],
                                                      ['api', T.REENTER], ['tab', 'drive']] + LOOK,
    '比べ: 入れ直す前に走行タブ(全部既定)': [['tab', 'drive'], ['tick'], ['tab', 'live'], ['tick'], ['reboot', 2000.0, T.RESET], ['tab', 'drive']] + LOOK,
    '知らせのあとにスイッチと curl で入れ直す(手順書 7 章の形)': [['tab', 'drive'], ['tick'], ['reboot', 2000.0, T.RESET], ['tick'],
                                                   ['check', '#dr-rec-back-guard', True], ['check', '#dr-rec-hold-keep', True],
                                                   ['api', {'ceil_left_mps': 4.2, 'launch_us': 200.0}], ['tick']] + LOOK,
    '一部とほかの項目を入れ直してから走行タブ': [['tab', 'drive'], ['tick'], ['tab', 'live'], ['tick'], ['reboot', 2000.0, T.RESET], ['api', T.OTHER],
                                ['tab', 'drive']] + LOOK,
    '▶ が通信の途切れで断られ、同期で起き直しを知る(入れ直し済み)': [['tab', 'drive'], ['tick'], ['reboot', 2000.0, T.RESET], ['api', T.REENTER],
                                                     ['start_fail'], ['start']] + LOOK,
}
for name, steps in SCENES.items():
    print('==', name)
    for label, app in (('直す前', T.BEFORE / 'app.js'), ('今', T.APP)):
        res, toasts = T.run_sim(tmp, c, steps, app=app, name='d.json')
        v = T.tagged(res)
        srv = T.vals(res, 'read')[-1]
        print(' [%s] トースト: %s' % (label, v[0]))
        print(' [%s] 走行タブの文: %s' % (label, T.note_of(v[1])))
        print(' [%s] スイッチ 見張り/残す: %s %s  天井の横: %s  発進の前進の横: %s' % (label, v[2], v[3], v[4], v[5]))
        print(' [%s] サーバー: %s' % (label, {k: srv[k] for k in ('rec_back_guard', 'rec_hold_keep', 'ceil_left_mps', 'launch_us', 'speed_ceiling')}))
