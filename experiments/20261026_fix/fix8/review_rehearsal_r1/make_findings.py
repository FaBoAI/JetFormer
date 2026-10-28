"""反証 8-1 回目・係 rehearsal: findings.json を書く(ctx_review8.md の形)。数と文は out/ の台本の出力から写した。"""
import json
from pathlib import Path

RB = '/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md'
D = 'experiments/20261026_fix/fix8/review_rehearsal_r1'
SUMMARY = (
    "反証 8-1 回目・係 rehearsal(通し稽古。直していない)。両方 ON の日の形で、手順書を 11/2 の準備(空きの測り直し・main session のもう一度の再起動の todo 4 章 1〜5・"
    "11/1 の 24〜27 の足し)→ 朝(日付・画面の版 9・基準の設定・dry run と外し忘れ・発進の前進の確かめ・試験の選択肢のスイッチと curl・最初のリカバリーで OFF)→ "
    "3.1 の表(GET は本物・systemctl/journalctl は作り物)→ 回 1〜4 の 22 本(本物の app.js を sim_rehearsal.js で 1 日通し・トリム・左 4.2・launch_us 200・選び直し)→ "
    "止める条件と 7 章(起こし直しと入れ直し・わざと OFF の本 X(回 1 と回 3)・入れ直し忘れ・launch_us の混ざり・印の無い走り直しと --skip LOG@T・電源断の 0 バイトと "
    "car_check --latest/名前・試しの書き方)→ 休みと夜の道具(手順書 8.1・8.2・4 章 10 の塊を文から抜き出し、HOME を作り物の家に替えてそのまま。作り物の 11/3 の 1 日 10 形(normal・trial_check・guard_off・guard_off3・reboot3・us200・resume・crash・scene_trial・v1))→ "
    "9 章の今、の 47 手順をたどった。コマンド・画面の文・道具の出力・判定は、下の 5 つのほかは手順書のとおりだった(確認の文の行・409/起き直しの文・curl の 400 の文・"
    "3.1 の 1〜5 行・② の区切りと印・⑤ の「ON − OFF … → 使ってよい/満たさない」と注意の行・car_check の行・--skip の形・bench_nightly の -- --logs)。"
    "指摘 5 件(どれも minor。走りの安全・判定の向きは変えない。手順書の文の直し)。台本と出力は " + D + "/(out/)。作り物の家 tmp_rehearsal/ は消した。")

F = [
    {'id': 'rehearsal-auto-vsrc', 'severity': 'minor', 'file': RB, 'line': 161,
     'title': "4 章の順(4 スイッチ → 5 説明欄)だと 2-3・2-4 の自動合わせのスイッチの確認の文が必ず「電圧の源: なし」と出て、4 章 6 の読み上げにも電圧の源が無い"
              "(V_before を書き忘れても ▶ で気づかない)",
     'scenario': (
         "ui_rehearsal.py day・auto_vb(本物の app.js・index.html を sim_rehearsal.js = tests/fixtures/app_sim_fix7_r2.js の写しで。偽のサーバーは本物の GET の params から・"
         "ファーム無し = vbat なし): 4 章の順で 2-3 → 自動合わせのスイッチの確認の文「電圧の源: なし(1.00 倍。周ごとの直しだけ)」(out/ui_rehearsal_day.txt の [171]・[196]。"
         "説明欄に V_before を書く前)。5 で V_before=8.31 を書いて ▶ → 確認の文「速さのゲインの自動合わせ: ON(目標 1 周 8.12 秒・電圧の源 説明欄の V_before(走行のあいだ一定))」。"
         "2-4 で V_before を書かずに ▶ → 確認の文「…電圧の源 なし(1.00 倍。周ごとの直しだけ)」で、ほかの警告なし(prepStartLines は battery=#__ だけ)のまま始まる"
         "(out/ui_rehearsal_auto_vb.txt)。手順書 4 章 6 の読み上げは「「速さのゲインの自動合わせ: ON」の行」だけで電圧の源を読まず、気づくのは走ったあとの car_check の"
         "「電圧の源 … なし: 6 章の「+ 自動合わせ ON」に入れない」(4 章 10 の表)= 回 2 の 1 本が 6 章の比べに使えない。2 章 8・9 は「ファームが無ければ V_before= 必須」と"
         "書くが、スイッチの確認の文はいつも「なし」なので、読んでも「なし」に慣れてしまう"),
     'fix': (
         "手順書 4 章 6 の読み上げに「自動合わせ ON の本は、行の「電圧の源」が 基板 か 説明欄の V_before(「なし」ならキャンセル → 説明欄に V_before= → もう一度 ▶)」を足す。"
         "4 章 4 に「自動合わせのスイッチの確認の文の電圧の源は、説明欄に V_before を書く前は「なし」と出る(読むのは ▶ の確認の文)」。"
         "または 2-3・2-4 は 5(説明欄の V_before)を 4 のスイッチより先にする")},
    {'id': 'rehearsal-after-memo', 'severity': 'minor', 'file': RB, 'line': 106,
     'title': "「走り終わりのメモの memo に「試し」」が、説明欄に bench= の無い本(3 章 5)・前に走った本・メモを書いたあと・起こし直しのあと(7 章)にはできない"
              "(パネルが出ない)。手順書に別の道が無い",
     'scenario': (
         "ui_rehearsal.py after(本物の renderDrive に、止まったあとの status = record_log・description を足して渡す): ① 3 章 5 の朝の確かめを収集したまま走った本"
         "(基準の設定で説明欄が空 → サーバーの説明は「AI走行 …」)→ #dr-after は hidden ② ブロックの本 → 出る(20261103-083000) ③ ② のメモを書いた(afterDone)あと → 出ない "
         "④ 次の本が止まる → 次の本(083400)のメモだけ ⑤ 起こし直し(status に record_log が無い)→ 出ない(out/ui_rehearsal_after.txt。app.js の renderAfter は "
         "bench= のある record_log の本だけ出す)。手順書 3 章 5「収集したままにした日は、走り終わりのメモの memo に「試し」」、7 章「起こし直した」の「気づいたら、入れ直す前に"
         "走った本の走り終わりのメモに「試し」」、7 章「走った値が計画と違った」(休みの ② で気づく本は前の本)は、書いてあるとおりにはできない。データのタブのシーンの説明"
         "(PATCH /api/scenes)に memo=試し(…) を書けば夜の道具は読む(作り物の家で 3-4 の 1 つのシーンの scene.json に書いて 8.1 の塊 → ⑤「試しの 1 本 … "
         "20261103-092600@1.00(on・試し(…))」・④「…・試し(数えない)」・③「試し(…)」。out/night_scene_trial_8.1.txt)が、手順書にその道が書いていない"),
     'fix': (
         "手順書 3 章 5・4 章 9・7 章(起こし直した・走った値が計画と違った)に「走り終わりのメモが出ないとき(説明欄に bench= が無い本・メモを書いた/閉じたあと・次の本を"
         "走ったあと・起こし直したあと)は、データのタブでその記録のシーンを 1 つ開き、説明の memo= の後ろに「試し(理由)」を足して保存する(夜の道具はどのシーンの直しも読む)。"
         "できなければ紙に書いて main session が --skip」を足す")},
    {'id': 'rehearsal-v1-count', 'severity': 'minor', 'file': RB, 'line': 92,
     'title': "11/1 に 24〜27 を足すと(todo 1 章のおすすめ)、朝の確かめ「選ぶ欄に 11/03 の 1〜23 が並ぶか」(手順書の頭 4 行目と 3 章)が 27 本になって合わない",
     'scenario': (
         "作り物の家(tmp の写し。本物の data/bench_candidates.json は変えていない = sha256 353ec306… のまま)で experiments/20261026_fix/fix3/add_v1_block.py "
         "--run v3_20261026-v3d-aug2-e60-s1 --label V1 --allow-not-drivable --write → 2026-11-03 の blocks 27(out/v1_bench_blocks.txt)。その bench で本物の app.js"
         "(ui_rehearsal.py v1・REH_BENCH)→ 選ぶ欄のベンチマーク 27 本・24 を押すとモデル v3_20261026-v3d-aug2-e60-s1・速度 0.85(out/ui_rehearsal_v1.txt)。"
         "手順書 4 行目「2026-11-03 の 1〜23 が入っている」・92 行目「選ぶ欄に 11/03 の 1〜23 が並ぶか」は、その日の朝の確かめで違って見える"
         "(README 6.2 の 10 は 24〜27 の時間と試験 2 つを直すと書くが、この 2 か所は無い)"),
     'fix': "手順書の頭と 3 章を「1〜23(11/1 に 24〜27 を足した日は 1〜27。2 章の 24〜27 の行)」に。README 6.2 の 10 の「直す」一覧にこの 2 か所を足す"},
    {'id': 'rehearsal-fw-cwd', 'severity': 'minor', 'file': RB, 'line': 99,
     'title': "3 章の電圧の確かめ `python3 tools/fw_vbat_check.py --meter …` に cd が無く、朝の端末(~)のままだと「can't open file」で動かない",
     'scenario': (
         "手順書 3 章の文を抜き出し、作り物の家の ~ でそのまま流す → 「python3: can't open file '…/home/tools/fw_vbat_check.py'」終了コード 2。~/jetformer で流すと動く"
         "(GET だけ。センサーが止まっているので「読めない・足りない(終わりの番号 3)」)(out/fw_vbat_cwd.txt)。3 章でその前に使うコマンド(date・timedatectl・"
         "hostname -I・3.1 の systemctl・journalctl・絶対パスの python3 -I -c)はどこでも動くので、端末は ~ のまま。todo_user_1026.md 1 章「ファームが来たら」の行も同じ文"),
     'fix': "手順書 3 章と todo_user_1026.md 1 章を `cd ~/jetformer && python3 tools/fw_vbat_check.py --meter <LiPo チェッカーの値>` に(8.2 の ⑦ は塊の頭の cd ~/jetformer で動く)"},
    {'id': 'rehearsal-s3-silent', 'severity': 'minor', 'file': RB, 'line': 302,
     'title': "8.1 の塊の ③ standing_3lap は --md・--json を付けているので画面に何も出さない。休みに読む「③ の「動いてから」」・6 章の 25 秒の判定の ③ を見るコマンドが手順書に無い",
     'scenario': (
         "night_rehearsal.sh normal 8.1(手順書 8.1 の塊を文から抜き出し、作り物の 11/3 の 22 本でそのまま)→ 画面に出るのは ① ② ④ ⑤ だけで、② の「22 本・計画と違う 0 本」"
         "のすぐあとが ④ の「# 実際の設定で分けた早い集計」(out/night_normal_8.1.txt)。③ を単独で同じ引数で流しても標準出力 0 行(standing_3lap.py の --md は「無ければ標準出力」)。"
         "表は $R/standing_3lap.md にだけ入る。手順書 1 章 0:40 の休み「③ standing_3lap の「動いてから」で床」・4 章 7「紙に書いた手の時間は、休みに 8.1 の ③ … の値と並べる」・"
         "6 章の 25 秒の判定は ③ を読むと書くが、開くコマンドも「④ の 3 章が同じ表(tools/standing_3lap.py と同じ表)を出す」ことも書いていない"),
     'fix': (
         "手順書 8.1 の塊の ③ の行のあとに `sed -n '/^## 1/,/^## 2/p' $R/standing_3lap.md`(か cat)を足す。または 8.1 の表の ③ の行に「画面では ④ の 3 章"
         "「止まってから 3 周(…tools/standing_3lap.py と同じ表)」で読める。注の全部は standing_3lap.md」と書く")},
]


def main():
    out = Path(__file__).resolve().parent / 'findings.json'
    out.write_text(json.dumps({'summary': SUMMARY, 'findings': F}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    json.loads(out.read_text(encoding='utf-8'))
    print(out, len(F), 'findings')


if __name__ == '__main__':
    main()
