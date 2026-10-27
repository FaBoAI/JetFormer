"""反証 w2-ui-w2-ui-r72-ui-2 の確かめ(読むだけ)。review_ui_r2 の common.py・sim_r2.js をそのまま使い、本物の app.js を node で動かす。
A = 指摘の手順(読み直す → 走行タブ(一覧だけ失敗)→ 待つ → 基準の設定)の再現。
B = 手順書 3 章の並び(読み直す → 走行タブで版を見る(一覧だけ失敗)→ 待つ → センサーを開始(ライブのタブ)→ 走行タブ → 基準の設定)。
C = B で、ライブから戻った走行タブでも一覧がもう一度失敗した(2 回続けて途切れた)とき。
D = 一覧だけ失敗したまま待つ間に、画面に何が出ているか(モデルの欄・ベンチマークの欄・天井の横の文・dr-message)。
    python3 p_check.py <一時の場所>
"""
import json
import sys
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/fix7/review_ui_r2')
from common import MORNING, cands, run_sim  # noqa: E402

tmp = sys.argv[1]
c, di = cands()
LOOK = [['el', '#toast', 'text'], ['el', '#dr-model', 'value'], ['eval', 'state.drive.models.length'], ['eval', 'state.drive.paramsLoaded'],
        ['el', '#dr-bench-pick-wrap', 'hidden'], ['eval', 'JSON.stringify(state.bench.data && state.bench.data.today)'],
        ['eval', 'document.querySelector("#dr-model").options.length'], ['eval', 'document.querySelector("#dr-model-info").textContent'],
        ['eval', 'document.querySelector("#dr-message").textContent.slice(0,120)'],
        ['eval', 'document.querySelector("#btn-dr-start").disabled']]
plans = {
    'A 指摘の手順': [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['tick'], ['tick']] + LOOK + [['baseline']] + LOOK,
    'B 手順書 3 章の並び': [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['tick'], ['tick']] + LOOK
        + [['tab', 'live'], ['tab', 'drive'], ['tick']] + LOOK + [['baseline']] + LOOK,
    'C ライブから戻ったときも一覧が失敗': [['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['tick'], ['tick'],
        ['tab', 'live'], ['models_fail', True], ['tab', 'drive'], ['models_fail', False], ['tick'], ['tick']] + LOOK + [['baseline']] + LOOK,
    'D 待つ(tick 10 回)': [['models_fail', True], ['tab', 'drive'], ['models_fail', False]] + [['tick']] * 10 + LOOK,
}
for name, steps in plans.items():
    res, toasts = run_sim(tmp, c, steps, server=MORNING, no_init=True, name='rf.json')
    looks = [x.get('value') for x in res if x['op'] in ('el', 'eval')]
    rows = [looks[i:i + len(LOOK)] for i in range(0, len(looks), len(LOOK))]
    print(json.dumps({'手順': name, '見え方(トースト・選んだモデル・一覧の数・同期済み・ベンチの欄が隠れている・今日・モデルの選択肢の数・モデルの説明・dr-message・▶ disabled)': rows,
                      'トーストの列': toasts}, ensure_ascii=False))
