"""11/3 の朝の流れ(画面を読み直す → 走行タブ → 基準の設定 → 見張り → 残す → ブロック → ▶)で、読み直したあと初めて走行タブを
開いたときにモデルの一覧の GET だけ通らない(Wi-Fi の一瞬の途切れ)場面。fix6 r2(w2-ui-w1-ui-r62-1)の直しで、表示の問い合わせの輪は
回り、▶・基準の設定・ブロックは「読み込んでいます」で断られなくなった。一覧は読み直さない(タブを替えるまで)。そのまま押すと何が出るか。

    python3 p_first_open_models_fail.py <一時の場所>
"""
import json
import sys

from common import FIXT, MORNING, blk, cands, run_sim

tmp = sys.argv[1]
c, di = cands()
B1 = blk(c, di, 1)
LOOK = [['el', '#toast', 'text'], ['el', '#dr-model', 'value'], ['eval', 'state.drive.models.length'], ['eval', 'state.drive.paramsLoaded'],
        ['el', '#dr-bench-pick-wrap', 'hidden'], ['eval', 'JSON.stringify(state.bench.data && state.bench.data.today)']]
steps = ([['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['tick'], ['tick']] + LOOK
         + [['baseline']] + LOOK
         + [['tapday', di, B1]] + LOOK
         + [['start']] + LOOK
         + [['check', '#dr-rec-back-guard', True]] + [['read', 'server']]
         + [['tick'], ['tab', 'live'], ['tab', 'drive'], ['tick'], ['baseline']] + LOOK)
apps = {'今の app.js': None, '6-1 回目のあと(fix6 r2 の直す前)': FIXT / 'fix6_ui_r2_before' / 'app.js'}
for an, app in apps.items():
    res, toasts = run_sim(tmp, c, steps, server=MORNING, app=app, no_init=True, name='fo.json')
    looks = [x['value'] for x in res if x['op'] in ('el', 'eval')]
    rows = [looks[i:i + len(LOOK)] for i in range(0, len(looks), len(LOOK))]
    st = [x for x in res if x['op'] == 'start'][0]
    print(json.dumps({'版': an, '見え方の移り(トースト・選んだモデル・一覧の数・同期済み・ベンチマークの欄が隠れている・今日)': rows,
                      '▶': {'POST': st['sent'] is not None, 'confirm': st['confirm'][:60]}, 'トーストの列': toasts}, ensure_ascii=False))
