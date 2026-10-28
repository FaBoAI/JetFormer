#!/usr/bin/env python3
"""fix8 係 docs 直す 8-1 回目(2026-10-28): 反証 8-1 回目で確かめて残った文書の指摘 9 件(係 docs 5・係 rehearsal 4)を、
手順書・todo・README・fix3/approvals.md・fix7/README.md に入れる。コードは変えない。

- 係 docs の指摘(experiments/20261026_fix/fix8/review_docs_r1/findings.json・票 refute_docs_r1_A・_B):
  f8r1-docs-residual2-reversed(approvals 14 の 2 行目・fix7/README 3 章の s4-c の残りの言い方を確かめた事実に)・
  f8r1-docs-readme-commit-filecount(README 6.1 の 5 の commit ごとのファイルの数)・f8r1-docs-readme-proposal-running(README 6 章の頭・
  6.2 の 32・7 章の安全の案)・f8r1-docs-fix7readme-ud-rows(fix7/README 7 章 4 の印)・f8r1-docs-stale-headers(todo の題・手順書 9 章の
  表の頭・approvals のいちばん上の「今」)。
- 係 rehearsal の指摘(review_rehearsal_r1/findings.json・票 refute_rehearsal_r1_A・_B): rehearsal-auto-vsrc(4 章 6 の「電圧の源」)・
  rehearsal-after-memo(走り終わりのメモの欄が出ない本の「試し」の道)・rehearsal-fw-cwd(fw_vbat_check の前の cd ~/jetformer)・
  rehearsal-s3-silent(8.1 の ③ は画面に出さない)。
- 打ち消された指摘(approvals3-old-row・todo-trial-wording・rehearsal-v1-count)は入れない。
- 置き換えの決まりは前の台本と同じ: old が文書の中でちょうど 1 回・new が 0 回のときだけ入れる。2 回目に流すと全部「済」。
- 前の係の文(前の台本の new)の中を書き換えるのは SUPERSEDE の 5 か所だけ(approvals 14 の 2 行目の中身とおすすめ = fix7 docs の仕上げの
  approvals 4 番目、README 6 章の頭と 6.2 の 32 = fix8 docs-decided の README 4・10 番目、fix7/README 7 章 4 = fix8 docs-decided の
  fix7/README 4 番目)。ほかは前の new の外(前・後ろ)に足すか、試験の無い台本(fix7 docs の 17:3x)の文の書き換え。
  前の係の試験(tests/test_fix7_docs_final.py・tests/test_fix8_docs_decided.py)は、revert() で書き換えを戻した文書で前の台本の直しが
  「済」かを照らし、effective() で書き換えたあとの形の new を使う(2026-10-28 fix8 docs r1 で合わせた)。
- 数は自分で流した台本の出力から写した(この場所の out/: c_dir_loop_rerun.txt = 票 A の c_dir_loop.py の写し、
  demo_residual2_rerun.txt = 反証の係の demo_residual2.py、push_stats.txt、check_scene_memo.txt)と fix8/proposal_safety_after1103.md。
- --apply: 控え <名前>.bak-20261028-fix8-docs-r1 を作ってから書く(控えがあれば上書きしない)。

使い方: python3 -I -B docs_fix8_r1.py [--apply]
"""
import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
JF = Path('/home/jetson/jetformer')
E = JF / 'experiments' / '20261026_fix'
DECIDED = E / 'fix8' / 'docs_decided' / 'docs_fix8_decided.py'
FINAL = E / 'fix7' / 'docs_final' / 'docs_fix7_final.py'
TESTS = {'r1': JF / 'tests' / 'test_fix7_docs_r1.py', 'r2': JF / 'tests' / 'test_fix7_docs_r2.py',
         'final': JF / 'tests' / 'test_fix7_docs_final.py', 'decided': JF / 'tests' / 'test_fix8_docs_decided.py',
         'r81': JF / 'tests' / 'test_fix8_docs_r1.py'}
BAK = '.bak-20261028-fix8-docs-r1'
STAMP = '10/28 05:2x'
RB, TODO, RD, AP, F7 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'
FILES = (RB, TODO, RD, AP, F7)
DEC_FILES = (RB, TODO, RD, AP, 'fix3/disk.md', F7, 'fix6/README.md')

# よく使う文
SCENE_WAY = ('画面の「シーン」のタブで、名前がその本の記録(20261103-HHMMSS)で始まるシーンを 1 つ開き、「走行の説明」の `memo=` の'
             'すぐ後ろに `試し(理由)。` を足して(`memo=` が無ければ終わりに ` memo=試し(理由)`)「保存」')
FW_CMD = '`cd ~/jetformer && python3 tools/fw_vbat_check.py --meter <LiPo チェッカーの値>`'

# ==================================================================================================== 手順書
RB_LOG_OLD = '\n- **10/28 03:3x fix8 係 docs-decided**'
RB_LOG = ('\n- **' + STAMP + ' fix8 係 docs 直す 8-1 回目**(下の 03:3x の docs-decided のあと。`fix8/fix_docs_r1.md`。控え '
          '`runbook_20261103.md' + BAK + '`。手順書だけ。コードは変えていない。反証 8-1 回目の係 rehearsal・docs の指摘): 3 章の電圧の確かめに '
          '`cd ~/jetformer &&`(朝の端末 ~ のままだと「can\'t open file」)/ 3 章 5・4 章 9・7 章「起こし直した」に、走り終わりのメモの欄が'
          '出ない本(説明欄に `bench=` が無い・メモを書いた・閉じた・次の本を走った・起こし直した)を「試し」にする道(「シーン」のタブの'
          '「走行の説明」に `試し(理由)`。できなければ紙 → main session の `--skip`)/ 4 章 6 に、自動合わせ ON の本の確認の文の「電圧の源」の'
          '読み方(「なし」ならキャンセル → 説明欄に `V_before=`)/ 8.1 の ③ は画面に何も出さないこと(④ の画面の 3 章か '
          '`cat $R/standing_3lap.md` で読む)/ 9 章の表の頭の時刻。' + RB_LOG_OLD)
RB_FW_OLD = '- [ ] 電圧: ファームがあれば `python3 tools/fw_vbat_check.py --meter <LiPo チェッカーの値>`(8.2 の ⑦)。'
RB_FW = ('- [ ] 電圧: ファームがあれば ' + FW_CMD + '(8.2 の ⑦。`cd ~/jetformer` が要る: 朝の端末(~)のままだと「python3: can\'t open '
         'file \'/home/jetson/tools/fw_vbat_check.py\'」で動かない。10/28 fix8 docs r1)。')
RB_35_OLD = 'この「なし」の区切りは起こし直しではない = 8.1 ②。10/27 fix7 docs r2)。\n'
RB_35 = ('この「なし」の区切りは起こし直しではない = 8.1 ②。10/27 fix7 docs r2)。この本は説明欄に `bench=` が無いので、止めても走り終わりの'
         'メモの欄は出ない(欄はベンチマークの本だけ。10/28 fix8 docs r1)。block が無いので書かなくても夜の道具はどの回の本にも入れない。'
         '書くなら画面の「シーン」のタブで、名前がこの記録(20261103-HHMMSS)で始まるシーンを 1 つ開き、「走行の説明」の終わりに '
         '` memo=試し(朝の確かめ)` を足して(もう `memo=` があれば、そのすぐ後ろに `試し(朝の確かめ)。`)「保存」(7 章「起こし直した」の'
         '行の終わりと同じ道)。\n')
RB_46_OLD = '\n   - 朝に入れた見張り・残す(補助なしは使う日だけ)の行が**消えている**'
RB_46 = ('\n   - **自動合わせ ON の本(2-3・2-4)は「電圧の源」も読む**(10/28 fix8 docs r1): ▶ の確認の文の「速さのゲインの自動合わせ: ON'
         '(目標 1 周 … 秒・電圧の源 …)」の行の「電圧の源」が「基板の電圧(走り始めから)」か「説明欄の V_before(走行のあいだ一定)」なら '
         'OK。「なし(1.00 倍。周ごとの直しだけ)」ならキャンセル → 5 の説明欄に `V_before=`(LiPo チェッカーの値。ファームが無い日は必須 = '
         '2 章)を書いて → もう一度 ▶(書かずに走ると、10 の car_check の「電圧の源」が「なし」になり、6 章の「+ 自動合わせ ON」に入らない)。'
         '4 で自動合わせのスイッチを ON にしたときの確認の文の「電圧の源: なし(1.00 倍。周ごとの直しだけ)」は、まだ 5 の説明欄に V_before を'
         '書く前なのでそう出る(決めは ▶ の確認の文で読む)。' + RB_46_OLD)
RB_49_OLD = '前は ⑤ ③ ④ が memo を読まず、試しの本 1 本で回 3 の答えが逆になった)。\n'
RB_49 = ('前は ⑤ ③ ④ が memo を読まず、試しの本 1 本で回 3 の答えが逆になった)。走り終わりのメモの欄が出ない本(説明欄に `bench=` が'
         '無い・メモを書いた・閉じた・次の本を走った・起こし直した)は、7 章「起こし直した」の行の終わりの道(「シーン」のタブの「走行の説明」に '
         '`試し(理由)`。10/28 fix8 docs r1)。\n')
RB_7R_OLD = '→ 気づいたら、入れ直す前に走った本の走り終わりのメモに「試し」(4 章 9) |'
RB_7R = ('→ 気づいたら、入れ直す前に走った本の走り終わりのメモに「試し」(4 章 9)。**その本の走り終わりのメモの欄は、たいていもう出ていない**'
         '(欄が出るのは、説明欄に `bench=` のある いちばん新しい本が止まっていて、まだメモを書いていない・「書かずに閉じる」を押していない'
         'ときだけ。メモを書いた・閉じた・次の本を走った・jetformer-web か Jetson が起き直したあとは出ない。10/28 fix8 docs r1)→ そのときは'
         + SCENE_WAY + '。夜の道具(③④⑤)はシーンの説明の直しも記録の説明欄に重ねて読むので、その本を「試し」として数えない'
         '(`tools/section_scorecard.py` の `effective_description` → `tools/standing_3lap.py` の `trial_memo`)。それもできなければ紙に'
         '「試し: 本・時刻・理由」→ main session が `--skip 記録名`(本ごと外す。7 章「走った値が計画と違った」の行と同じ) |')
RB_3O_OLD = '| `standing_3lap.md`・`.json` | 回 1・回 3・回 4 のあと'
RB_3O = ('| `standing_3lap.md`・`.json`(**画面には何も出さない**。止まってから 3 周は ④ の画面の「## 3. 止まってから 3 周(最初の R1 の起点'
         'まで行った発進。tools/standing_3lap.py と同じ表)」で読める。R1 に届かなかった発進・走りながらの 3 周の窓・注の全部は '
         '`cat $R/standing_3lap.md`。10/28 fix8 docs r1) | 回 1・回 3・回 4 のあと')
RB_3B_OLD = 'python3 -I tools/standing_3lap.py --day 20261103 --md $R/standing_3lap.md --json $R/standing_3lap.json\n'
RB_3B = (RB_3B_OLD + '#   ③ は画面に何も出さない(表は $R/standing_3lap.md にだけ書く)。止まってから 3 周は次の ④ の画面の「## 3. 止まってから 3 周'
         '(…)」で読める。全部は cat $R/standing_3lap.md(10/28 fix8 docs r1)\n')
RB_9_OLD = '| もの | 今(10/27 17:3x。前の版は 10/27 05:4x) |'
RB_9 = '| もの | 今(' + STAMP + ' に見直した。前の版は 10/27 17:3x・10/27 05:4x) |'

# ==================================================================================================== todo
TD_TITLE_OLD = '# ユーザーが 11/3 までにすること(10/27 17:3x 現在)'
TD_TITLE = '# ユーザーが 11/3 までにすること(' + STAMP + ' 現在)'
TD_LOG_OLD = '\n- **10/28 03:3x fix8 係 docs-decided**'
TD_LOG = ('\n- **' + STAMP + ' fix8 係 docs 直す 8-1 回目**(下の 03:3x の docs-decided のあと。`fix8/fix_docs_r1.md`。控え '
          '`todo_user_1026.md' + BAK + '`): 題の時刻を今に / 1 章「ファームが来たら」の電圧の確かめに `cd ~/jetformer &&` / 3 章の「試し」の'
          '行に、走り終わりのメモの欄が出ないときの道(手順書 7 章「起こし直した」)。' + TD_LOG_OLD)
TD_FW_OLD = '止まっている車で `python3 tools/fw_vbat_check.py --meter <LiPo チェッカーの値>`'
TD_FW = ('止まっている車で ' + FW_CMD + '(`cd ~/jetformer` が要る。端末を開いたまま(~)だと「can\'t open file」で動かない。'
         '10/28 fix8 docs r1)')
TD_TRIAL_OLD = 'サーバーが自動で書く「計画と違うので試し(元 cand=B)…」は気にしなくてよい(数える)。\n'
TD_TRIAL = ('サーバーが自動で書く「計画と違うので試し(元 cand=B)…」は気にしなくてよい(数える)。走り終わりのメモの欄が出ない(メモを書いた・'
            '閉じた・次の本を走った・起こし直した)ときは、' + SCENE_WAY.replace('名前がその本の記録(20261103-HHMMSS)で始まるシーン', 'その記録のシーン')
            + '(手順書 7 章「起こし直した」の行の終わり。できなければ紙 → main session の `--skip`。10/28 fix8 docs r1)。\n')

# ==================================================================================================== README
RD_LOG_OLD = '\n- **10/28 03:3x 追記**(fix8 係 docs-decided。'
RD_LOG = ('\n- **' + STAMP + ' 追記**(fix8 係 docs 直す 8-1 回目。下の 03:3x のあと。控え `README.md' + BAK + '`。報告 '
          '`fix8/fix_docs_r1.md`): 5 章の電圧の確かめに `cd ~/jetformer &&`・6 章の頭と 6.2 の 32(11/3 のあとに向けた安全の案は済)・'
          '6.1 の 5(commit ごとのファイルの数)・7 章に `fix8/proposal_safety_after1103.md`。' + RD_LOG_OLD)
RD_FW_OLD = 'ファームが来たら書き込み → `python3 tools/fw_vbat_check.py --meter <LiPo チェッカーの値>`。'
RD_FW = 'ファームが来たら書き込み → ' + FW_CMD + '(10/28 fix8 docs r1: `cd ~/jetformer` が要る)。'
# SUPERSEDE(fix8 docs-decided の README #4): 6 章の頭
RD_6_X = '11/3 のあとに向けた安全の案の係はまだ動いている。'
RD_6_Y = ('11/3 のあとに向けた安全の案は**済**(10/28 03:25。`fix8/proposal_safety_after1103.md`: approvals 14 の 1・2 行目(安全の残り 1・2)の'
          '直し方の案 pA・pB・pC を写しで比べ、おすすめは pB を見張り ON の日だけ・入れるのは 11/3 のあと。残り 2 の「やり直して多く下がる」は'
          '閉じたループでは逆向きが多い(短く下がる側が多い)→ approvals 14 の 2 行目を直した。10/28 fix8 docs r1)。')
RD_65_OLD = '**9da6d0d** を 10/28 00:56 に push。1635 ファイル。'
RD_65 = '**9da6d0d** を 10/28 00:56 に push。ファイルは fce5d47 が 1635・9da6d0d が 802(`git show --stat`。10/28 fix8 docs r1)。'
# SUPERSEDE(fix8 docs-decided の README #10): 6.2 の 32
RD_632_X = '| 32 | 反証 8 回目の残り(11/3 のあとに向けた安全の案の係。`fix8/proposal_safety_after1103.md`)と、'
RD_632_Y = ('| 32 | 反証 8 回目の残り(反証と直しの続き。11/3 のあとに向けた安全の案 `fix8/proposal_safety_after1103.md` は 10/28 03:25 に'
            '済 = 6 章の頭)と、')
RD_7C_OLD = 'approvals 3 の (a) の前提の確かめと drive.py の説明の文・文書を今に |\n'
RD_7C = (RD_7C_OLD + '| `fix8/proposal_safety_after1103.md` | 11/3 のあとに向けた安全の残り(approvals 14 の 1・2 行目)の直し方の案(10/28 03:25。'
         'safety-proposal の係)。案 pA・pB・pC を写しで作り、閉じたループと本物の記録で比べた。おすすめ pB を見張り ON の日だけ・入れるのは '
         '11/3 のあと |\n')

# ==================================================================================================== approvals
AP_09_OLD = '- 今(10/27 05:3x): 学習 v3 の段 1 は'
AP_09 = ('- 前の今(10/27 05:3x。**今は下の「今(10/28 03:3x)」の行**。この行は 10/27 05:3x の記録のまま。印は 10/28 fix8 docs r1): '
         '学習 v3 の段 1 は')
AP_LOG_OLD = '\n- **今(10/28 03:3x。fix8 係 docs-decided が今に合わせた。'
AP_LOG = ('\n- **' + STAMP + ' fix8 係 docs 直す 8-1 回目**(控え `approvals.md' + BAK + '`。`fix8/fix_docs_r1.md`): 14 の 2 行目(s4-c の'
          '残り)の言い方を確かめた事実に(やり直した組は短く下がる側が多い・中立のあと後退に入り直すのが遅い ESC では escape が前の壁を'
          '短く押す組がある・当たり・引っかかりが無くても起きうる。守りに手順書 7 章「当たって止まった車が前へ押し続けたら人側」)・'
          'いちばん上の「今(10/27 05:3x)」に「前の今」の印。答え・おすすめは変えていない(今の形は下の「今(10/28 03:3x)」の行のまま)。'
          + AP_LOG_OLD)
# SUPERSEDE(fix7 docs の仕上げの approvals #4 = 14 章): 2 行目の中身
AP_14C_X = ('やり直して下がった時間を捨て、最大 0.2 秒の後退ぶん多く下がる。起きるのは、下がっている途中で後ろに当たる・引っかかるときなど'
            '(強い当たりは後ろの当たりとして先に前へ出る)。')
AP_14C_Y = ('やり直して下がった時間を捨てる。**下がる距離は、短くなる側が多い**(10/28 fix8 docs r1 で直した文。前に書いた「最大 0.2 秒の後退'
            'ぶん多く下がる」は確かめると逆向きが多かった): 閉じたループ(偽物の ESC・車)で、この途切れでやり直しの決めが途切れの無い同じ場面と'
            '変わった組は、案の係の 336 組では 88 組とも短く下がり(−0.11〜−0.95 m・中央 −0.50 m。後ろの当たりは増えない)、票 A の別の ESC の'
            'モデル 3 つ(62 組ずつ)では短く 38〜42 組(最大 −0.16〜−0.27 m)・長く 14 組(最大 +0.14 m)。**中立のあと後退に入り直すのが'
            '遅い ESC では、短く下がったあと escape が前の壁を短く押す組がある**(案の係の閉じたループで ESC の下がり始めが本物の最大 0.27 秒の組: '
            '0.22 m まで短くなり 6 組が最大 0.16 秒押した。本物より遅い 0.35 秒では 22 組・最大 0.37 秒。票 A の「中立のあともう一度待つ ESC」: '
            '28 組・最大 0.13 秒。途切れの無い同じ場面は 0 秒)。前に書いた「起きるのは、下がっている途中で後ろに当たる・引っかかるときなど」も'
            '違い、当たり・引っかかりが無くても、ゆっくり下がり始める・始めに 0 が入る後退の eRPM で起きうる(本物の後退の並びを置くと 132 並びの'
            'うち 9。票 A: 例の 2 並びは人の弱い後退・途中で離した後退で、リカバリーの後退(中立 + 300 µs)を表すとは言えない)。後ろに強く'
            '当たったときは、後ろの当たりとして先に前へ出る。'
            '途切れそのもので下がる時間がのびる形は、やり直しと関係なく、ふつうの後退でも ON・OFF とも同じ(前から)。')
# 同じ行のおすすめ
AP_14R_X = '| **このまま**(手順書 7 章「1 m 以上下がる・後ろの物に近づく → すぐ人側」) |'
AP_14R_Y = ('| **このまま**(11/3 は直さない。手順書 3 章「その日の最初のリカバリーを人が見る」・7 章「リカバリーで車が 1 m 以上下がる・後ろの物に'
            '近づく」「当たって止まった車が、AI 側のまま前へ押し続けている」の行(どちらも → すぐ CH3 を人側)。直すなら 11/3 のあと = '
            '`fix8/proposal_safety_after1103.md` の pB) |')

# ==================================================================================================== fix7/README.md(記録。印と足しだけ)
F7_LOG_OLD = '= 出切っていない**(7 章の 1)。\n- **jetformer-web の再起動が要る**'
F7_LOG = ('= 出切っていない**(7 章の 1)。\n- (' + STAMP + ' fix8 係 docs 直す 8-1 回目。控え `README.md' + BAK + '`。`../fix8/fix_docs_r1.md`)'
          '3 章の安全の 2 行目(s4-c の残り)に「確かめると逆向きが多い」の印と守りの足し・7 章 4 の印を今に(`../user_decisions.md` の 10/28 '
          '03:4x の 2 行)。ほかは記録のまま。\n- **jetformer-web の再起動が要る**')
F7_3_OLD = ('やり直して最大 0.2 秒の後退ぶん多く下がる | 本物の記録 235 本では起きていない。直す前は逆に後退側を出し続けた。approvals 14・'
            '手順書 7 章「1 m 以上下がったらすぐ人側」 |')
F7_3 = ('やり直して最大 0.2 秒の後退ぶん多く下がる(→ **10/28 fix8 docs r1: 確かめると逆向きが多い**。やり直した組は短く下がる側が多く'
        '(閉じたループで −0.11〜−0.95 m。別の ESC のモデルでは長く下がる組もあり最大 +0.14 m)、中立のあと後退に入り直すのが遅い ESC では '
        'escape が前の壁を短く押す組がある(本物の最大 0.27 秒までの ESC で 1 回 最大 0.13〜0.16 秒)。当たり・引っかかりが無くても起きうる。'
        '`../fix3/approvals.md` 14 の 2 行目) | 本物の記録 235 本では起きていない。直す前は逆に後退側を出し続けた。approvals 14・手順書 7 章'
        '「1 m 以上下がったらすぐ人側」(10/28 fix8 docs r1: と 7 章「当たって止まった車が前へ押し続けたらすぐ人側」・3 章「最初のリカバリーを'
        '人が見る」) |')
# SUPERSEDE(fix8 docs-decided の fix7/README #4): 7 章 4
F7_74_X = '⑤ は再起動の行だけ(approvals 6・「ON の形」の注の行はまだ))。'
F7_74_Y = ('⑤ は再起動の行・approvals 6・「ON の形」の注の行とも `user_decisions.md` に入った(approvals 6 と注の行は 10/28 03:4x に main '
           'session。印は 10/28 fix8 docs r1))。')

# ==================================================================================================== 一覧
# (文書, [(old, new)])。足す形(old が new の中にある)・試験の無い台本の文の書き換え・SUPERSEDE の書き換え(old = 前の係の new の中の文)
ALL = [
    (RB, [(RB_LOG_OLD, RB_LOG), (RB_FW_OLD, RB_FW), (RB_35_OLD, RB_35), (RB_46_OLD, RB_46), (RB_49_OLD, RB_49), (RB_7R_OLD, RB_7R),
          (RB_3O_OLD, RB_3O), (RB_3B_OLD, RB_3B), (RB_9_OLD, RB_9)]),
    (TODO, [(TD_TITLE_OLD, TD_TITLE), (TD_LOG_OLD, TD_LOG), (TD_FW_OLD, TD_FW), (TD_TRIAL_OLD, TD_TRIAL)]),
    (RD, [(RD_LOG_OLD, RD_LOG), (RD_FW_OLD, RD_FW), (RD_6_X, RD_6_Y), (RD_65_OLD, RD_65), (RD_632_X, RD_632_Y), (RD_7C_OLD, RD_7C)]),
    (AP, [(AP_09_OLD, AP_09), (AP_LOG_OLD, AP_LOG), (AP_14C_X, AP_14C_Y), (AP_14R_X, AP_14R_Y)]),
    (F7, [(F7_LOG_OLD, F7_LOG), (F7_3_OLD, F7_3), (F7_74_X, F7_74_Y)]),
]
# 9 件の指摘 → その直し(文書, ALL の何番目)。頭の記録の行(各文書の 0・1 番目の LOG)はどの指摘にも入れない
FINDINGS = {
    'f8r1-docs-residual2-reversed': [(AP, 2), (AP, 3), (F7, 1)],
    'f8r1-docs-readme-commit-filecount': [(RD, 3)],
    'f8r1-docs-readme-proposal-running': [(RD, 2), (RD, 4), (RD, 5)],
    'f8r1-docs-fix7readme-ud-rows': [(F7, 2)],
    'f8r1-docs-stale-headers': [(TODO, 0), (RB, 8), (AP, 0)],
    'rehearsal-auto-vsrc': [(RB, 3)],
    'rehearsal-after-memo': [(RB, 2), (RB, 4), (RB, 5), (TODO, 3)],
    'rehearsal-fw-cwd': [(RB, 1), (TODO, 2), (RD, 1)],
    'rehearsal-s3-silent': [(RB, 6), (RB, 7)],
}
LOGS = [(RB, 0), (TODO, 1), (RD, 0), (AP, 1), (F7, 0)]

# 前の係の文(その台本の ALL の (文書, 何番目) の new。fix7 の台本は fix8 docs-decided の書き換えのあとの形)の中を意図して書き換えた所:
# (台本, 文書, 何番目, 前の文 X, 今の文 Y)。台本は 'final'(fix7 docs の仕上げ)・'decided'(fix8 docs-decided)
SUPERSEDE = [
    ('final', AP, 4, AP_14C_X, AP_14C_Y),       # approvals 14 の 2 行目の中身: 「多く下がる」→ 確かめた事実(f8r1-docs-residual2-reversed)
    ('final', AP, 4, AP_14R_X, AP_14R_Y),       # 同じ行のおすすめ: 守りに 7 章「押し続けたら人側」・3 章「最初のリカバリーを人が見る」
    ('decided', RD, 4, RD_6_X, RD_6_Y),         # README 6 章の頭: 案の係はまだ動いている → 済(f8r1-docs-readme-proposal-running)
    ('decided', RD, 10, RD_632_X, RD_632_Y),    # README 6.2 の 32: 同上
    ('decided', F7, 4, F7_74_X, F7_74_Y),       # fix7/README 7 章 4: 注の行はまだ → 入った(f8r1-docs-fix7readme-ud-rows)
]
# 前の文を消す書き換え(X の文が Y に残らない)
REMOVES = {(AP, AP_14C_X), (AP, AP_14R_X), (RD, RD_6_X), (RD, RD_632_X), (F7, F7_74_X)}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prev_new(script, f, i):
    """前の台本 script の (文書 f, 何番目 i) の new の今の形(fix7 の台本は fix8 docs-decided の書き換えのあと)"""
    dec = _load('docs_fix8_decided_from_r81', DECIDED)
    if script == 'decided':
        return dict(dec.ALL)[f][i][1]
    fin = _load('docs_fix7_final_from_r81', FINAL)
    return dec.effective(script, f, i, dict(fin.ALL)[f][i][1])


def effective(script, f, i, new):
    """前の台本 script の (文書 f, 何番目 i) の new(fix7 の台本なら fix8 docs-decided の書き換えのあとの形)の、fix8 docs r1 が
    書き換えたあとの形(書き換えていなければ new のまま)"""
    for s, ff, ii, x, y in SUPERSEDE:
        if (s, ff, ii) == (script, f, i):
            assert new.count(x) == 1, (s, ff, ii, x[:30])
            new = new.replace(x, y, 1)
    return new


def revert(texts):
    """fix8 docs r1 の SUPERSEDE の書き換えを戻した文書(texts にある文書だけ。足しただけの直しはそのまま)"""
    out = dict(texts)
    for _, f, _, x, y in reversed(SUPERSEDE):
        if f in out and out[f].count(y) == 1:
            out[f] = out[f].replace(y, x, 1)
    return out


def superseded_present(texts):
    """SUPERSEDE の書き換えが今の文書に入っている(Y がちょうど 1 回・消した X が 0 回)か。戻り値は問題の一覧"""
    bad = []
    for s, f, i, x, y in SUPERSEDE:
        if f not in texts:
            continue
        if texts[f].count(y) != 1:
            bad.append(('Y が 1 回でない', s, f, i, texts[f].count(y)))
        if (f, x) in REMOVES and texts[f].count(x) != 0:
            bad.append(('消した文が残る', s, f, i))
    return bad


def apply_text(texts, files=ALL):
    """(直した文, [(文書, old, 'apply' | 'done' | 'missing' | 'many(N)')])。前の台本と同じ決まり"""
    out, rep = dict(texts), []
    for f, edits in files:
        t = out[f]
        for old, new in edits:
            n_old, n_new = t.count(old), t.count(new)
            if n_new == 0 and n_old == 1:
                t = t.replace(old, new)
                rep.append((f, old, 'apply'))
            elif n_new == 1 and (n_old == 0 or (old in new and n_old == new.count(old))):
                rep.append((f, old, 'done'))
            else:
                rep.append((f, old, 'missing' if n_old == 0 else 'many(%d)' % n_old))
        out[f] = t
    return out, rep


def read_all(root=REPORTS, files=DEC_FILES):
    return {f: (Path(root) / f).read_text(encoding='utf-8') for f in files}


def earlier_problems(texts, tmp_root=None):
    """前の係の直しと試験の確かめが崩れていないか(texts = 直したあとの 7 本。戻り値は問題の一覧):
    - SUPERSEDE の X が、その前の台本の new(今の形)の中にちょうど 1 回(= 前の係の文の中だけを書き換えた)
    - 書き換えを戻した文書で、fix8 docs-decided の 68 個の直しが「済」で new がちょうど 1 回・その台本の earlier_problems
      (fix7 docs r1・r2・仕上げの直しと試験の確かめ f1〜f5・g1〜g6・h1〜h5、docs-decided の試験の i1〜i6、SUPERSEDE 21 か所)が空
    - 今の文書で、前の係の試験の確かめ(f1〜f5・g1〜g6・h1〜h5・i1〜i6)と、この組の試験の確かめ(試験のファイルがあれば)が空"""
    bad = []
    for s, f, i, x, y in SUPERSEDE:
        if prev_new(s, f, i).count(x) != 1:
            bad.append(('SUPERSEDE の X が前の new の中に 1 回でない', s, f, i))
    dec = _load('docs_fix8_decided_from_r81b', DECIDED)
    old = revert(texts)
    _, rep = dec.apply_text(old)
    bad += [('decided', f, o[:30], st) for f, o, st in rep if st != 'done']
    for f, edits in dec.ALL:
        for _, new in edits:
            if old[f].count(new) != 1:
                bad.append(('decided count', f, new[:30], old[f].count(new)))
    bad += [('decided.earlier_problems(戻した文書)',) + tuple(x) for x in dec.earlier_problems(old)]
    sys.path.insert(0, str(JF))
    sys.path.insert(0, str(JF / 'tools'))
    sys.path.insert(0, str(JF / 'tests'))
    t1 = _load('t_fix7_docs_r1_from_r81', TESTS['r1'])
    t2 = _load('t_fix7_docs_r2_from_r81', TESTS['r2'])
    t3 = _load('t_fix7_docs_final_from_r81', TESTS['final'])
    t4 = _load('t_fix8_docs_decided_from_r81', TESTS['decided'])
    for k, fn in sorted(t1.CHECKS.items()):
        bad += [('t1 ' + k, p) for p in fn(texts)]
    with tempfile.TemporaryDirectory(prefix='docs_fix8_r1_', dir=tmp_root) as tmp:
        for k, fn in sorted(t2.CHECKS.items()):
            bad += [('t2 ' + k, p) for p in fn(texts, Path(tmp) / k)]
    for k, fn in sorted(t3.CHECKS.items()):
        bad += [('t3 ' + k, p) for p in fn(texts)]
    bad += [('t3 h5', p) for p in t3.h5_problems(texts[F7])]
    for k, fn in sorted(t4.CHECKS.items()):
        bad += [('t4 ' + k, p) for p in fn(texts)]
    if TESTS['r81'].exists():
        t5 = _load('t_fix8_docs_r1_from_r81', TESTS['r81'])
        for k, fn in sorted(t5.CHECKS.items()):
            bad += [('t5 ' + k, p) for p in fn(texts)]
    bad += [('superseded',) + tuple(x) for x in superseded_present(texts)]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args(argv)
    tmp_root = E / 'fix8' / 'tmp_docs_r1'
    tmp_root.mkdir(exist_ok=True)
    texts = read_all()
    new, rep = apply_text(texts)
    for f, old, s in rep:
        print('%-8s %s: %s' % (s, f, old.strip()[:60].replace('\n', '⏎')))
    if [x for x in rep if x[2] not in ('apply', 'done')]:
        print('当たらない直しがある → 書かない')
        return 2
    for f, edits in ALL:
        for old, nw in edits:
            if new[f].count(nw) != 1:
                print('new が 1 回でない:', f, nw[:40])
                return 4
    prob = earlier_problems(new, str(tmp_root))
    for x in prob:
        print('崩れる:', x)
    if prob:
        print('→ 書かない')
        return 3
    n_apply = sum(1 for _, _, s in rep if s == 'apply')
    print('入れる %d・済 %d・前の係の直しと確かめ・この組の確かめ: 崩れない(SUPERSEDE %d か所)' % (n_apply, len(rep) - n_apply,
                                                                                   len(SUPERSEDE)))
    if not a.apply:
        print('(--apply なしなので書かない)')
        return 0
    for f in FILES:
        if new[f] == texts[f]:
            continue
        b = REPORTS / (f + BAK)
        if not b.exists():
            shutil.copy2(REPORTS / f, b)
        (REPORTS / f).write_text(new[f], encoding='utf-8')
        print('書いた:', f, '(控え %s)' % b.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
