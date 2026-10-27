#!/usr/bin/env python3
"""試験の選択肢のスイッチ(OPT_SWITCHES)は、r1-state3-1 の「送信の返事より前に出た問い合わせの返事では戻さない」(drAckAt・since)
を使っていない(optBusy だけ)。問い合わせ(GET)がスイッチの送信より先に出て、返事が送信の返事より後に来ると、古い値で
スイッチ・state.drive.st.params・走行開始の確認の文が戻る。比べ: 天井のスイッチ(postDriveParams → drAckAt)では戻らない"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_sim import cands, blk, run, server_params   # noqa: E402

c, di = cands()
G, H = '#dr-rec-back-guard', '#dr-rec-hold-keep'
CE = '#dr-speed-ceiling'
cases = {
    # 朝に見張りを ON にしていた → やめると決めて OFF にした(OFF は確認の文なしですぐ送る)
    'guard OFF': ([['pick', di, blk(c, di, 7)], ['gstart'], ['check', G, False], ['el', G, 'checked'], ['read', 'server'],
                   ['gend'], ['el', G, 'checked'], ['read', 'st_params'], ['start']], server_params(rec_back_guard=1)),
    # 見張りを ON → 続けて「残す」を ON(手順書 3 章「見張りを入れたあと」)
    'guard ON → keep ON': ([['gstart'], ['check', G, True], ['gend'], ['el', G, 'checked'], ['check', H, True], ['el', H, 'checked'],
                            ['read', 'server']], server_params()),
    # 対照: 天井(postDriveParams・drAckAt あり)
    'ceiling OFF (対照)': ([['gstart'], ['check', CE, False], ['gend'], ['el', CE, 'checked'], ['start']],
                           server_params(speed_ceiling=1)),
}
for name, (steps, sp) in cases.items():
    res = run(steps, sp)
    print('---', name)
    for x in res['steps']:
        if x['op'] == 'el':
            print('  画面 %s checked=%s' % (x['sel'], x['value']))
        elif x['op'] == 'read' and x['what'] == 'server':
            print('  サーバー rec_back_guard=%s rec_hold_keep=%s speed_ceiling=%s' % (x['value']['rec_back_guard'], x['value']['rec_hold_keep'], x['value']['speed_ceiling']))
        elif x['op'] == 'read' and x['what'] == 'st_params':
            print('  画面が持つサーバーの値(state.drive.st.params) rec_back_guard=%s' % x['value'].get('rec_back_guard'))
        elif x['op'] == 'check':
            print('  check -> %s' % x['checked'])
        elif x['op'] == 'start':
            lines = [l for l in x['confirm'].splitlines() if '見張り' in l or '天井' in l]
            print('  走行開始: 確認の文の行 %s / 送った speed_ceiling=%s' % (lines, x['sent'] and x['sent'].get('speed_ceiling')))
    print('  トースト:', [t for t in res['toasts'] if '見張り' in t or '残す' in t or '天井' in t])
