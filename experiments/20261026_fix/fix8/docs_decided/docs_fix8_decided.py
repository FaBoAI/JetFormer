#!/usr/bin/env python3
"""fix8 係 docs-decided(2026-10-28): 反証 8 回目の最初の係(fix8 tools-decided・safety-decided)と main session の記録
(user_decisions.md 10/28 00:5x)を、ユーザーが読む文書(手順書・todo・README・fix3/approvals.md)と fix3/disk.md・fix6/README.md・
fix7/README.md に入れる。コードは変えない。

- 入れるもの: ① jetformer-web のもう一度の再起動(10/28 00:53:23。fix7 の直しが入った)と、そのあとの drive.py の説明の文(10/28 02:33。
  コメントだけ)のため反証 8 回目のあとにもう一度再起動すること ② approvals 3「途中で OFF にした日」= (a)(10/28 main session)と
  fix8 safety-decided の確かめ ③ approvals 8「fix7 で残した読み方」①② は変えた(③ は今のまま)と fix8 tools-decided の道具の言葉
  ④ pytest 一式の作業場所(10/28: -o tmp_path_retention_policy=failed で約 0.6 GB・付けないと 7.1 GB)⑤ 「次に」(commit 9da6d0d・
  反証 8 回目の途中)。
- 置き換えの決まりは前の台本と同じ: old が文書の中でちょうど 1 回・new が 0 回のときだけ入れる。2 回目に流すと全部「済」。
- 前の係(fix7 docs r1・r2・仕上げ)が入れた文はそのまま残す(足す形)。ただし SUPERSEDE の 21 か所だけは、前の係の文(その台本の new)の
  中を意図して書き換える(決まった読み方・もう一度の再起動の記録・PYTEST_GB の値。ほとんどは「 |」の前に足すだけ。文を消すのは
  README 3.2 の 4 の「`PYTEST_GB` は 3.9 のまま …」・手順書 3 章と 8.1 ⑤ の「… main session が決める」の 3 つ)。前の係の試験は、
  revert() で書き換えを戻した文書で前の台本の直しが「済」かを照らし、effective() で書き換えたあとの形の new を使う
  (tests/test_fix7_docs_r1.py・test_fix7_docs_r2.py・test_fix7_docs_final.py。2026-10-28 fix8 docs-decided で合わせた)。
- 入れる前に: 前の 3 つの台本の直しが(書き換えを戻した文書で)全部「済」で new がちょうど 1 回・SUPERSEDE の old がその前の台本の
  new の中にちょうど 1 回・画面の版 9 の直し(docs_ban9)が「済」で古い版の数 0・前の係の試験の確かめ(f1〜f5・g1〜g6・h1〜h5)と
  この組の試験の確かめ(tests/test_fix8_docs_decided.py の i1〜i6)が空、を照らし、1 つでも崩れたら書かない。
- --apply: 控え <名前>.bak-20261028-fix8-decided を作ってから書く(控えがあれば上書きしない)。

使い方: python3 -I docs_fix8_decided.py [--apply]
"""
import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
JF = Path('/home/jetson/jetformer')
FIX7 = JF / 'experiments' / '20261026_fix' / 'fix7'
SCRIPTS = {'r1': FIX7 / 'docs_r1' / 'docs_fix7_r1.py', 'r2': FIX7 / 'docs_r2' / 'docs_fix7_r2.py',
           'final': FIX7 / 'docs_final' / 'docs_fix7_final.py'}
TESTS = {'r1': JF / 'tests' / 'test_fix7_docs_r1.py', 'r2': JF / 'tests' / 'test_fix7_docs_r2.py',
         'final': JF / 'tests' / 'test_fix7_docs_final.py', 'fix8': JF / 'tests' / 'test_fix8_docs_decided.py'}
BAK = '.bak-20261028-fix8-decided'
STAMP = '10/28 03:3x'
RB, TODO, RD, AP = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md'
DK, F7, F6 = 'fix3/disk.md', 'fix7/README.md', 'fix6/README.md'
FILES = (RB, TODO, RD, AP, DK, F7, F6)

# よく使う文
RESTART = '10/28 00:53:23'
DOC_FIX = ('10/28 02:33 に `jetformer/drive.py` の説明の文(コメント)だけを直した**(fix8 safety-decided。docstring を除いた AST は '
           '00:53:23 の起動のコードと同じ = 動きは同じ')
NEWER = '「起動より新しい: /home/jetson/jetformer/jetformer/drive.py」'

# ==================================================================================================== 手順書
RB_LOG_OLD = '\n- **10/27 23:5x fix7 係 docs の仕上げ**'
RB_LOG = ('\n- **' + STAMP + ' fix8 係 docs-decided**(下の 23:5x の仕上げ・22:xx の 7-2 回目のあと。`fix8/decided_docs.md`。控え '
          '`runbook_20261103.md' + BAK + '`。手順書だけ。コードは変えていない): **jetformer-web は ' + RESTART + ' に main session が'
          'もう一度再起動した**(fix7 の安全の直しが入った。1 章・3.1・8 章の頭・9 章)。10/28 02:33 に `jetformer/drive.py` の説明の文'
          '(コメントだけ。動きは同じ)を直したので、3.1 の「起動のあとに直したコード」は反証 8 回目のあとのもう一度の再起動まで drive.py の'
          '名前を出す / 3 章「わざと OFF にした日」・8.1 ⑤: 回 3 の途中で OFF にした日も ⑤ の行のまま読む(10/28 main session の決め (a)。'
          '`fix3/approvals.md` 3)/ 7 章・8.1 ②〜⑤: リカバリーの途中で CH3 を人側にしたあとの走り直しにも印(「リカバリーの途中 X 秒から '
          'Y 秒」)・試しの本と途中で変えた本の ON の空転と最初の周の記録の欠けも ⑤ が数える・② の「回の本は X(No.Y)から」と印の後ろの文'
          '(10/28 main session の決め・fix8 tools)。')
RB_1_OLD = '実装と試験は 10/26 に済み): 12・14・16・18(OFF)だけ走る'
RB_1 = ('実装と試験は 10/26 に済み。' + RESTART + ' に main session がもう一度再起動して、fix7 の安全の直しも入った = 3.1): '
        '12・14・16・18(OFF)だけ走る')
RB_31_OLD = '\n\n| 見ること | 正しい形 | 違ったら |'
RB_31 = ('\n- **' + RESTART + ' に main session がもう一度再起動した**(10/27 15:4x の了承と同じ目的 = 直しのコードを 11/3 の前に入れる、'
         'とみた。`fix3/approvals.md` 2 の「誰が」・`user_decisions.md` 10/28 00:5x)。反証 7 回目の直しと試験一式(3415 passed)のあと、'
         '走行・学習・収集・録画が無いのを見てから `todo_user_1026.md` 4 章の 1〜5: 3 は active・NRestarts 0・「エラーの行なし」、4 は新しい'
         '項目が全部既定(0・4.1・250)・`boot_at` あり(00:53:24)、5 は「起動より新しいコードなし」= **上の行の fix7 安全の直し(assist.py '
         '22:02・drive.py 19:36)は今のサーバーに入った**。本物の画面を headless の Firefox で読むだけで「画面の版 9(app.js 10/27 21:58)」・'
         '見張りは OFF・「残す」は見張りが OFF の間は押せない・± 0.01 を見た。**そのあと ' + DOC_FIX + '。`fix8/decided_safety.md` 2 章)。'
         'そのため下の表の「起動のあとに直したコード」は、次の再起動まで ' + NEWER + ' を出す(10/28 03:0x に読むだけで見た)。'
         'これは動きの違いではない。**反証 8 回目のあとに main session がもう一度再起動する**(走行・収集・学習・GPU の予測が無いときに '
         '`todo_user_1026.md` 4 章の 1〜5)か、11/3 の朝に Jetson を入れたときに消える。drive.py のほかの名前が出たら、今までどおり表の'
         '「違ったら」。11/3 の朝にもう一度この表を全部見る・車輪を浮かせて実際の出力で 1 回(3 章)がまだなのは同じ。' + RB_31_OLD)
# SUPERSEDE(r1 の手順書 #2): 3 章「わざと OFF にした日」の「… main session が決める(」
RB_OFF_X = '回 3 の途中で OFF にしたときの ⑤ の「ON − OFF」の読み方は main session が決める('
RB_OFF_Y = ('回 3 の途中で OFF にしたときも、⑤ の「ON − OFF」はそのまま ⑤ の行で読む(**10/28 main session の決め (a)**。見張り・残すが'
            '変えるのは、当たった・止まったあとのリカバリーと中立止めのあとの扱いだけで、AI 側の中立止めもリカバリーも無い発進は、ON と OFF で'
            '書く値がビット単位で同じ(10/28 fix8 safety-decided が閉じたループと本物の 10/23・10/24 の発進で確かめた)。違うのは、0.5 秒より'
            '長い AI 側の中立止めの間に走ったあとの車が止まった発進だけで、ON は明けて中止(人を待つ)になり、⑤ はそれを「R1 に届かなかった'
            '発進」・その場で戻した発進を走り直しにする =「使ってよい」の側には動かない。')
RB_HOLD_OLD = '紙に理由。10/27 fix7 docs r2)。OFF の日(既定)は前と同じ'
RB_HOLD = ('紙に理由。10/27 fix7 docs r2)。**リカバリーの途中(当たった・止まったあとのブレーキ・後退・前進など)で CH3 を人側にした AI の'
           '区間のあとの発進にも、同じ走り直しの印が付く**(③⑤ の形の文は「リカバリーの途中 X 秒から Y 秒」。リカバリーが終わってから人側に'
           'したときは出ない。扱いは上と同じ = 数えない。ON の最初の周の衝突・空転・記録の欠けは ⑤ が数える。10/28 main session の決め・'
           'fix8 tools)。OFF の日(既定)は前と同じ')
RB_HIT_OLD = '■ で止めてもう 1 本。10/27 fix7 docs r2)。人側で HS のいつもの場所に戻してから。'
RB_HIT = ('■ で止めてもう 1 本。10/27 fix7 docs r2)。**リカバリーの途中(下がっている・止まっている)で CH3 を人側にしたあとも同じ**'
          '(同じ本の中で戻した発進は ⑤ の比べ・③ の止まってから 3 周・④ の「動く → A1」に数えない。③⑤ に「リカバリーの途中 X 秒から Y 秒」。'
          '10/28 fix8 tools)。人側で HS のいつもの場所に戻してから。')
RB_TRIAL_OLD = 'ただし試しの ON の最初の周の衝突は「最初の周の衝突 0」に数える。'
RB_TRIAL = ('ただし試しの ON の最初の周の衝突は「最初の周の衝突 0」に数える。空転(比 > 1.3・比が出ない)・最初の周の記録の欠けも数える'
            '(どれかがあれば回 3 は「満たさない」。8.1 ⑤。10/28 main session の決め・fix8 tools)。')
RB_8_OLD = '\n\n### 8.1 休み(パックの替えの間。'
RB_8 = ('\n- (10/28 fix8 docs)上の `back_no_esc`・`early_none` を出す fix7 安全の直しは、' + RESTART + ' のもう一度の再起動で今のサーバーに'
        '入った(3.1)。' + RB_8_OLD)
# SUPERSEDE(r2 の手順書 #13): 8.1 ② の「…からが朝の形。10/27 fix7 docs r2) | 標準出力の表 |」の前に足す
RB_2_X = 'からが朝の形。10/27 fix7 docs r2) | 標準出力の表 |'
RB_2_Y = ('からが朝の形。10/27 fix7 docs r2)。区切りの後ろに「(… 回の本は X(No.Y)から)」が付いたら、本「—」から始まった区切りのまま'
          '回の本 X が続いた = 本 X から先もその区切りの選択肢で走った(「なし」なら起こし直しのまま → 7 章で入れ直す。3 章で止めて OFF に'
          'した日の紙の本 X から先なら数える)。朝の手順どおり(確かめのあとに選択肢を入れた)の行には付かない。印の後ろの文は「(再起動で '
          'OFF に戻った? 3 章で止めて OFF にした本 X から先なら数える。手順書 3・7 章)」(10/28 fix8 tools) | 標準出力の表 |')
RB_3_OLD = '(`fix2/tools.md` 5 章 1) |\n'
RB_3 = (RB_3_OLD + '| ③ | (10/28 fix8 tools)「数えない(リカバリー中止のあとの走り直し。リカバリーの途中 X 秒から Y 秒・人側 Z 秒)」= 同じ本の'
        '中で、リカバリーの途中(当たった・止まったあとのブレーキ・後退・前進など)で CH3 を人側 → AI 側にした発進(表の下に注「- 「リカバリー'
        'の途中 X 秒から Y 秒」のリカバリー中止のあとの走り直し = …」)。扱いは中止のあとの走り直しと同じ(最速・④ の 1 章・2 章に入れない。'
        '10/28 main session の決め = `fix3/approvals.md` 8 の「fix7 で残した読み方」①) |\n')
RB_4_OLD = '\n| ④ | 1 章の印「数えない発進 N(リカバリー中止のあとの走り直し)」='
RB_4 = ('\n| ④ | (10/28 fix8 tools)2 章の下の「- リカバリー中止のあとの走り直し(…)」の行の終わりに「リカバリー(当たった・止まったあとの'
        'ブレーキ・後退・前進など)の途中で人が CH3 を人側にした区間のあとの発進も同じ(③ の表の「リカバリーの途中 X 秒から Y 秒」)。」が'
        '出たら、③ の「リカバリーの途中 X 秒から Y 秒」の発進(止まってから 3 周の列・「動く → A1 中央値」に入れない。1 章の「数えない発進 N」'
        'にも入る) |' + RB_4_OLD)
RB_5_OLD = '(7 章の中立止めの行。10/27 fix7 tools)。'
RB_5 = (RB_5_OLD + '**(10/28 fix8 tools)** 一つ前の区間がリカバリーの途中(当たった・止まったあとのブレーキ・後退・前進など)で人が CH3 を'
        '人側にして終わった発進も「リカバリー中止のあとの走り直し」に入る(形の文「リカバリーの途中 X 秒から Y 秒」。組の行の説明に'
        '「リカバリー(…)の途中で人が CH3 を人側にした区間のあとも同じ(「リカバリーの途中 X 秒から Y 秒」)」、発進の行の後ろに「 中止の'
        'あとの走り直し(リカバリーの途中 X 秒から Y 秒・人側 Z 秒。比べに入れない / R1 に届かなかった発進に数える)」)。**試しの本・途中で'
        '変えた本の ON の発進も、最初の周の衝突のほか、空転(比 > 1.3・比が出ない)・最初の周の記録の欠けを比べの条件に数える**(空転は R1 に'
        '届いた発進だけ): 「  → 試しの本の ON の発進が空転した N 本 …」= checks の slip で満たさない、「  → 試しの本の ON の発進で空転の比が'
        '出ない N 本: …」= slip_all_measured で満たさない、「最初の周の記録が欠けた発進(on)」の名前の後ろの「。試し」= on_all_readable で'
        '満たさない(途中で変えた本は「  → 途中で変えた本の ON の発進が …」・「。途中で変えた」。比べ(起点まで・失う・動く・LH)には今まで'
        '通り入れない。10/28 main session の決め = `fix3/approvals.md` 8 の「fix7 で残した読み方」①②)。')
RB_MID_OLD = '「途中で変えた N 本」= 走行中に設定を変えた本(比べに入れない。ただし ON の発進の最初の周の衝突は「最初の周の衝突 0」に数える)'
RB_MID = ('「途中で変えた N 本」= 走行中に設定を変えた本(比べに入れない。ただし ON の発進の最初の周の衝突は「最初の周の衝突 0」に数える。'
          '空転・最初の周の記録の欠けも試しの本と同じに数える。10/28 fix8 tools)')
# SUPERSEDE(r1 の手順書 #15): 8.1 ⑤ の「回 3 の途中で OFF にしたときの読み方は main session が決める = `fix3/approvals.md` 3)」
RB_MIX_X = '回 3 の途中で OFF にしたときの読み方は main session が決める = `fix3/approvals.md` 3)'
RB_MIX_Y = ('回 3 の途中で OFF にしたときも ⑤ の行のまま読む = 10/28 main session の決め (a)。`fix3/approvals.md` 3。行の後ろに「(3 章で止めて '
            'OFF にした本 X から先の OFF なら起こし直しではない = 数える。手順書 3・7 章)」が付く = 10/28 fix8 tools)')
# SUPERSEDE(r2 の手順書 #19): 9 章の rec_hold_keep の行
RB_9_X = '(3.1 の「起動のあとに直したコード」))。OFF の日は 7 章「壁を押し続けたらすぐ人側」で守る |'
RB_9_Y = ('(3.1 の「起動のあとに直したコード」)→ ' + RESTART + ' のもう一度の再起動で入った)。OFF の日は 7 章「壁を押し続けたらすぐ人側」で'
          '守る |')

# ==================================================================================================== todo
TD_LOG_OLD = '\n- **10/27 23:5x fix7 係 docs の仕上げ**'
TD_LOG = ('\n- **' + STAMP + ' fix8 係 docs-decided**(下の 23:5x・22:xx のあと。`fix8/decided_docs.md`。控え `todo_user_1026.md' + BAK +
          '`): jetformer-web は ' + RESTART + ' に main session がもう一度再起動した(fix7 の直しが入った。1 章・4 章)。drive.py の説明の文'
          '(10/28 02:33。コメントだけ)のため反証 8 回目のあとにもう一度 / commit 9da6d0d(2 章 6)/ approvals 3「途中で OFF にした日」は '
          '(a)・approvals 8 の「fix7 で残した読み方」①② は変えた(1 章・2 章 10・13・16、3 章)。')
TD_26A_OLD = '| 再起動は**済**(10/27 17:16:28。main session)。'
TD_26A = '| 再起動は**済**(10/27 17:16:28。**もう一度 ' + RESTART + '**。main session)。'
TD_26B_OLD = '**11/3 の朝にもう一度 3.1 の表を見る** |'
TD_26B = ('**11/3 の朝にもう一度 3.1 の表を見る**。10/28 02:33 の drive.py の説明の文(コメントだけ)のため、反証 8 回目のあとに main session が'
          'もう一度再起動する(4 章) |')
# SUPERSEDE(r2 の todo #1)
TD_26C_X = '(4 章の 5・手順書 3.1 の「起動のあとに直したコード」の行で見る) |'
TD_26C_Y = ('(4 章の 5・手順書 3.1 の「起動のあとに直したコード」の行で見る)→ **' + RESTART + ' に main session がもう一度再起動して入った**'
            '(4 章の 5 は「起動より新しいコードなし」だった) |')
TD_20_OLD = 'どれも 11/3 の手順に効く。答えは番号と選んだものを一言 |'
TD_20 = ('どれも 11/3 の手順に効く。答えは番号と選んだものを一言。3 の「途中で OFF にした日」は 10/28 に main session が (a) に決めた。'
         '8 に足した読み方の ①② は 10/28 に main session が変えた(2 章 13) |')
TD_GIT_OLD = 'と fce5d47(10/27 17:19。10/26〜10/27 の直しと反証 2〜6 回目)を push 済み。このあとの変更も main session がまとめて |'
TD_GIT = ('と fce5d47(10/27 17:19。10/26〜10/27 の直しと反証 2〜6 回目)・**9da6d0d**(10/28 00:56。反証 7 回目の直し)を push 済み。'
          'このあとの変更(反証 8 回目)も main session がまとめて |')
TD_10_OLD = '10/27 fix6 で両方 ON の日の形をまた直した(`fix6/README.md`) |'
TD_10 = ('10/27 fix6 で両方 ON の日の形をまた直した(`fix6/README.md`)。回 3 の途中で OFF にした日の ⑤ の「ON − OFF」はそのまま ⑤ の行で'
         '読む(**10/28 main session の決め (a)**。`fix3/approvals.md` 3) |')
TD_13_OLD = '`fix3/approvals.md` 8。おすすめ このまま(回 4 が OFF になりやすい安全側) |'
TD_13 = ('`fix3/approvals.md` 8。おすすめ このまま(回 4 が OFF になりやすい安全側)。8 に足した「fix7 で残した読み方」の ①② は 10/28 に '
         'main session が変えた(リカバリーの途中で人側にしたあとの走り直しも比べに入れない・試しの本と途中で変えた本の ON の空転と最初の'
         '周の記録の欠けも数える。夜の道具に入った)。③ は今のまま |')
# SUPERSEDE(final の todo #2): 2 章 16
TD_16_X = '`fix3/approvals.md` 14・8。おすすめ このまま(11/3 は今のコード。返事が無ければこのまま) |'
TD_16_Y = ('`fix3/approvals.md` 14・8。おすすめ このまま(11/3 は今のコード。返事が無ければこのまま。8 の 3 つのうち ①② は 10/28 に main '
           'session が変えた = 2 章 13) |')
TD_RES_OLD = '\n- (10/27 直す 3 回目・2 組目)回 3・回 4 の発進の前進 ON の本は'
TD_RES = ('\n- (10/28 fix8)**リカバリーの途中(下がっている・止まっている)で CH3 を人側にして、その場で AI 側に戻した発進も同じ**(夜の道具は'
          '走り直しとして比べ・止まってから 3 周に数えない。当たり・空転・記録の欠けは数える。③⑤ に「リカバリーの途中 X 秒から Y 秒」。'
          '手順書 7 章)。止まってから 3 周を数えるなら ■ で止めて HS に置き、もう 1 本。' + TD_RES_OLD)
TD_TRIAL_OLD = '⑤ ③ ④ がその本を数えない。'
TD_TRIAL = ('⑤ ③ ④ がその本を数えない(ただし ON の発進の最初の周の衝突・空転・記録の欠けは ⑤ が数える = 回 3 は「満たさない」になる。'
            '10/28 main session の決め・fix8 tools)。')
TD_OFF_OLD = '\n\n## 4. jetformer-web の再起動の手順と確かめ'
TD_OFF = ('\n- (10/28)最初のリカバリーで見張り・残すを OFF にしたのが回 3 の途中でも、夜の ⑤ の「ON − OFF」はそのまま読む(10/28 main '
          'session の決め (a)。手順書 3 章)。' + TD_OFF_OLD)
TD_4_OLD = '\n- 10/26 夜に、変えた Python 17 本の文法'
TD_4 = ('\n- **' + RESTART + ' に main session がもう一度再起動した**(10/27 15:4x の了承と同じ目的 = 直しのコードを 11/3 の前に入れる、'
        'とみた。`fix3/approvals.md` 2 の「誰が」・`user_decisions.md` 10/28 00:5x)。反証 7 回目の直しと試験一式(3415 passed・2 failed・'
        '4 skipped。落ちた 2 件は試験が git の HEAD の台本を読んでいたためで、試験を 81a4d34 に固定して 15 passed。'
        '`fix7/pytest_full_20261028.txt`)のあと、走行・学習・収集・録画が無いのを見てから下の 1〜5 をした。3: active・0・「エラーの行なし」。'
        '4: 新しい項目は全部既定(0・4.1・250)・boot_at(00:53:24)。5: 「起動より新しいコードなし」= **上の行の fix7 安全の直し(assist.py '
        '22:02・drive.py 19:36)が入った**。本物の画面を headless の Firefox で読むだけで「画面の版 9(app.js 10/27 21:58)」。**そのあと ' +
        DOC_FIX + ')。そのため下の 5 は、次の再起動まで ' + NEWER + ' を出す(10/28 03:0x に読むだけで見た)。**反証 8 回目のあとに '
        'main session がもう一度、下の 1〜5 をする**(同じ目的)か、11/3 の朝に Jetson を入れたときに消える。' + TD_4_OLD)

# ==================================================================================================== README
RD_LOG_OLD = '\n- **10/27 23:5x 追記**'
RD_LOG = ('\n- **' + STAMP + ' 追記**(fix8 係 docs-decided。下の 23:5x・22:xx のあと。控え `README.md' + BAK + '`。報告 '
          '`fix8/decided_docs.md`): 0 章 7・8・3.2 の 4・6 章の頭・6.1 の 3・5・6.2 の 8・26・29・32・6.3 の 31・6.4・7 章を今に'
          '(jetformer-web は ' + RESTART + ' にもう一度再起動・commit 9da6d0d・反証 8 回目の途中・drive.py の説明の文のため反証 8 回目の'
          'あとにもう一度再起動・pytest 一式は `-o tmp_path_retention_policy=failed` で約 0.6 GB)。')
RD_7_OLD = 'と **fce5d47**(10/27 17:19。10/26〜10/27 の直しと反証 2〜6 回目。1635 ファイル)を push **済**'
RD_7 = RD_7_OLD + '・**9da6d0d**(10/28 00:56。反証 7 回目の直し)も push **済**'
# SUPERSEDE(r2 の README #1): 0 章 8
RD_8_X = '次の再起動か 11/3 の朝の Jetson の起動で入る。6.2 の 8) |'
RD_8_Y = '次の再起動か 11/3 の朝の Jetson の起動で入る → **' + RESTART + ' のもう一度の再起動で入った**。6.2 の 8) |'
# SUPERSEDE(r1 の README #2): 3.2 の 4(「`PYTEST_GB` は 3.9 のまま …」を 10/28 の実測に)
RD_PY_X = ('`fix3/disk/disk_check.py` の `PYTEST_GB` は 3.9 のまま = 「pytest 一式の最中」の空きを約 3.2 GB 多く出す(作業場所の差)。'
           '10/27 fix7 docs r1)')
RD_PY_Y = ('**10/28 の一式は `-o tmp_path_retention_policy=failed`(通った試験の一時ファイルをすぐ消す)を付けて約 0.6 GB**(10/28 '
           '00:08〜00:52: 空き 10493M → いちばん少なくて 9913M(00:44)。終わりの作業場所 55M。`fix7/pytest_full_20261028.txt`)。空き 8 GB を'
           '切らずに流すには、始めに約 8.6 GB(上の約 15 GB と 7.1 GB は付けない形)。**一式はこのオプションを付けて流す**。'
           '`fix3/disk/disk_check.py` の `PYTEST_GB` は 0.6(10/28 fix8 tools。付けないときの 7.1 GB は `PYTEST_GB_KEEP`)。10/27 fix7 docs r1・'
           '10/28 fix8 docs)')
RD_6_OLD = '\n\n### 6.1 段 1 のあと(10/27〜28)'
RD_6 = ('\n- (10/28 fix8 係 docs-decided)fix7 の走りの直しは **' + RESTART + ' のもう一度の再起動で入った**(6.2 の 8)。commit **9da6d0d**'
        '(10/28 00:56)を push 済み。試験一式は 10/28 にも 3415 passed(6.1 の 3)。**反証 8 回目の途中**: 最初の係が決めを入れた'
        '(`fix8/decided_tools.md` = approvals 8 の「fix7 で残した読み方」①② を夜の道具に・`fix8/decided_safety.md` = approvals 3 の (a) の'
        '前提の確かめと drive.py の説明の文・`fix8/decided_docs.md` = 文書)。11/3 のあとに向けた安全の案の係はまだ動いている。drive.py の'
        '説明の文(10/28 02:33。コメントだけ)のため、**反証 8 回目のあとにもう一度再起動**(6.2 の 8・32)。' + RD_6_OLD)
RD_61_OLD = '| 3 | pytest 一式は**済**(10/27 16:28〜17:12、2734 passed。`fix6/pytest_full_20261027.txt`)。'
RD_61 = ('| 3 | pytest 一式は**済**(10/27 16:28〜17:12、2734 passed。`fix6/pytest_full_20261027.txt`。10/28 00:08〜00:52 にもう一度 '
         '**3415 passed**・2 failed・4 skipped。落ちた 2 件は試験を 81a4d34 に固定して 15 passed。`fix7/pytest_full_20261028.txt`。作業場所は '
         '3.2 の 4)。')
RD_65_OLD = '| 5 | ~~commit~~ **済**(fce5d47 を 10/27 17:19 に push。'
RD_65 = RD_65_OLD + '反証 7 回目の直しは **9da6d0d** を 10/28 00:56 に push。'
# SUPERSEDE(r2 の README #3): 6.2 の 8
RD_628_X = '手順書 3.1 の「起動のあとに直したコード」の行・`todo_user_1026.md` 4 章の 5) | ユーザー・main session |'
RD_628_Y = ('手順書 3.1 の「起動のあとに直したコード」の行・`todo_user_1026.md` 4 章の 5)→ **' + RESTART + ' に main session がもう一度'
            '再起動して入った**(3.1 の「起動より新しいコードなし」)。10/28 02:33 の drive.py の説明の文(コメントだけ。動きは同じ)のため、'
            '**反証 8 回目のあとにもう一度再起動**(それまで 3.1 の「起動のあとに直したコード」は drive.py の名前を出す) | ユーザー・main '
            'session |')
# SUPERSEDE(r2 の README #4): 6.2 の 26
RD_626_X = '**出た**: 10/27 19:36 からの fix7 安全の直し → 6.2 の 8) |'
RD_626_Y = '**出た**: 10/27 19:36 からの fix7 安全の直し → 6.2 の 8。' + RESTART + ' に入った) |'
# SUPERSEDE(final の README #2): 6.2 の 29
RD_629_X = 'なるべく早く(11/3 の前のもう一度の再起動は今も要る = 6.2 の 8) |'
RD_629_Y = ('なるべく早く(11/3 の前のもう一度の再起動は今も要る = 6.2 の 8 → ' + RESTART + ' に済。**反証 8 回目を始めた** = 6 章の頭・'
            '6.2 の 32) |')
RD_632_OLD = '\n\n### 6.3 11/3 のあとでもよい(小さいもの)'
RD_632 = ('\n| 32 | 反証 8 回目の残り(11/3 のあとに向けた安全の案の係。`fix8/proposal_safety_after1103.md`)と、そのあとの jetformer-web の'
          '再起動(10/28 02:33 の drive.py の説明の文を入れる。コメントだけ。手順は `todo_user_1026.md` 4 章の 1〜5)・試験一式・commit | '
          'main session | 11/3 の前 |' + RD_632_OLD)
# SUPERSEDE(final の README #3): 6.3 の 31
RD_631_X = '`fix7/README.md` 3 章。安全の形と読み方の知らせは `fix3/approvals.md` 14・8 |'
RD_631_Y = ('`fix7/README.md` 3 章。安全の形と読み方の知らせは `fix3/approvals.md` 14・8。10/28: `drive.py` の説明の文・`runs_1103` の印の'
            '文と本「—」の区切り・`PYTEST_GB` は fix8 の係が直した(`fix8/decided_safety.md`・`fix8/decided_tools.md`)。夜の道具の印の'
            '付かない走り直しのうち、リカバリーの途中で人側にしたあとは印が付くようになった(approvals 8 の ①) |')
# SUPERSEDE(r2 の README #5): 6.4
RD_64_X = '10/27 19:36 からの fix7 安全の直しは次の再起動のあと(6.2 の 8) |'
RD_64_Y = ('10/27 19:36 からの fix7 安全の直しは次の再起動のあと(6.2 の 8)→ ' + RESTART + ' のもう一度の再起動のあとも、GET・journal・'
           '「起動より新しいコードなし」・本物の画面を読むだけで見た(走行はまだ) |')
RD_7C_OLD = '残すこと・了承・再起動が要るか |\n'
RD_7C = (RD_7C_OLD + '| `fix8/decided_tools.md`・`fix8/decided_safety.md`・`fix8/decided_docs.md` | 反証 8 回目の最初の係(10/28): main session '
         'の決め(approvals 8 の ①②)を夜の道具に・approvals 3 の (a) の前提の確かめと drive.py の説明の文・文書を今に |\n')

# ==================================================================================================== approvals
AP_NOW_OLD = '\n\n## 0. 一覧(急ぐ順)'
AP_NOW = ('\n- **今(' + STAMP + '。fix8 係 docs-decided が今に合わせた。控え `approvals.md' + BAK + '`)**: **2 は済み**(10/27 17:16:28 と、'
          '**もう一度 ' + RESTART + '**(main session。10/27 15:4x の了承が当たるとみた = 2 の「誰が」)。fix7 の安全の直しが入った。'
          '10/28 02:33 の drive.py の説明の文(コメントだけ。動きは同じ)のため、反証 8 回目のあとにもう一度 = 2 の「今」)、**3 は決まり**'
          '((c) 両方 ON。「途中で OFF にした日」の ⑤ の読み方は **(a)**(10/28 main session。fix8 safety-decided が前提を確かめた)= 3 の表)、'
          '**8 の「fix7 で残した読み方」①② は変えた**(10/28 main session の決め。fix8 tools-decided で夜の道具に入った。③ は今のまま = 8 章)。'
          'commit 9da6d0d を push 済み。答えがまだなのは 4・7〜11・13 のまま(14 は知らせ = 返事が無ければこのまま)。' + AP_NOW_OLD)
# SUPERSEDE(r2 の approvals #2): 0 章 2
AP_02_X = ('**10/27 19:36 からの fix7 安全の直しのため、もう一度**(2 の「今」) | 済(車輪を浮かせた確かめはまだ)。もう一度は 11/3 の前 |')
AP_02_Y = ('**10/27 19:36 からの fix7 安全の直しのため、もう一度**(2 の「今」)→ **' + RESTART + ' にもう一度した**(main session)。10/28 '
           '02:33 の drive.py の説明の文(コメントだけ)のため、反証 8 回目のあとにもう一度 | 済(車輪を浮かせた確かめはまだ)。もう一度は '
           '11/3 の前(10/28 00:53 に済。説明の文のぶんは反証 8 回目のあと) |')
AP_03_OLD = '| **決まり(10/27): (c) 両方 ON**(朝から全部の回で) | 決まり |'
AP_03 = '| **決まり(10/27): (c) 両方 ON**(朝から全部の回で)。「途中で OFF にした日」の ⑤ は **(a)**(10/28 main session) | 決まり |'
AP_08_OLD = '| 8 | 回 3 の判定の読み方(厳しめに直した) | **このまま** | 11/2 |'
AP_08 = ('| 8 | 回 3 の判定の読み方(厳しめに直した) | **このまま**(「fix7 で残した読み方」の ①② は 10/28 に main session が変えた = 夜の'
         '道具に入った。③ はこのまま) | 11/2 |')
AP_1_OLD = '11/3 の朝の見込みは作業場所を消す前提なので変わらない。10/27 fix7 docs r1)。'
AP_1 = (AP_1_OLD + '**10/28 追記**: 10/28 の一式は `-o tmp_path_retention_policy=failed` を付けて約 0.6 GB(空きのいちばん少ないとき '
        '9.9 GB)。`fix3/disk/disk_check.py` の `PYTEST_GB` も 0.6 に(付けないときの 7.1 GB は `PYTEST_GB_KEEP`。10/28 fix8 tools)。')
# SUPERSEDE(r2 の approvals #3): 2 の「今」
AP_2N_X = '`fix7/fix_safety_r1.md` 7 章 1・10/27 fix7 docs r2) |'
AP_2N_Y = ('`fix7/fix_safety_r1.md` 7 章 1・10/27 fix7 docs r2)。**→ ' + RESTART + ' に main session がもう一度再起動した**(反証 7 回目の'
           '直し・試験一式 3415 passed のあと、走行・学習・収集・録画が無いのを見てから。`todo_user_1026.md` 4 章の 3〜5: active・'
           'NRestarts 0・エラーの行なし・新しい項目は既定・「起動より新しいコードなし」= fix7 安全の直しが入った。本物の画面を読むだけで'
           '「画面の版 9(app.js 10/27 21:58)」。`user_decisions.md` 10/28 00:5x)。**そのあと ' + DOC_FIX + ')ので、手順書 3.1 の'
           '「起動のあとに直したコード」は次の再起動まで drive.py の名前を出す → **反証 8 回目のあとに main session がもう一度再起動する**'
           '(11/3 の朝に Jetson を入れても消える) |')
# SUPERSEDE(final の approvals #2): 2 の「誰が」
AP_2W_X = '再起動が要るわけは `fix7/README.md` 5 章。10/27 23:5x fix7 docs の仕上げ) |'
AP_2W_Y = ('再起動が要るわけは `fix7/README.md` 5 章。10/27 23:5x fix7 docs の仕上げ)。**→ 10/28 00:5x main session の判断: 当たる**'
           '(もう一度の再起動は 15:4x の了承と同じ目的 = 直しのコードを 11/3 の前に入れる、とみて ' + RESTART + ' に行った。'
           '`user_decisions.md` 10/28 00:5x。反証 8 回目のあとの再起動(drive.py の説明の文を入れる)も同じ目的で main session がする) |')
AP_3D_OLD = '・4 章 10・5 章・7 章・8.1 をこの形にした(10/27 fix7) |'
AP_3D = ('・4 章 10・5 章・7 章・8.1 をこの形にした(10/27 fix7)。途中で OFF にした日の ⑤ の読み方は (a)(10/28 main session。下の'
         '「「途中で OFF にした日」の決まり」の行) |')
AP_3C_OLD = '10/23・10/24 の 45,844 回のフレームの間で、0.5 秒より長いのは 1 回 |'
AP_3C = ('10/23・10/24 の 45,844 回のフレームの間で、0.5 秒より長いのは 1 回(10/28 fix8: 閉じたループでは、発進の 0.17〜2.6 秒あとに'
         '始まる 0.5〜1.5 秒のカメラの止まりの間に車が止まった形。発進の前進 ON の方が発進が速いぶん起きにくい。⑤ では起点まで・失う・LH の'
         '入口の速さが出ない発進になる。動く・空転は同じ。`fix8/decided_safety.md` 1.2・1.6 節) |')
AP_3R_OLD = '\n\n## 4. 「発進の前進を出さない区間は発進の補助も出さない」(`launch_no_kick`)'
AP_3R = ('\n| 「途中で OFF にした日」の決まり(10/28) | **決まり(10/28 00:5x main session): (a)** そのまま ⑤ の行で決める(上の行の案 (a)。'
         '`user_decisions.md` 10/28 00:5x)。**前提は 10/28 fix8 safety-decided が確かめた**(`fix8/decided_safety.md`): AI 側の中立止めが'
         '無く、リカバリーが始まらない発進は、両方 ON と両方 OFF で PCA9685 に書く値・AI_CMD の全部の欄・出来事がビット単位で同じ(閉じた'
         'ループ 1,438 組、中立止めがあっても中止にならなかった 2,970 組、本物の 10/23・10/24 の発進 53 × 発進の前進 OFF/ON の 106 組)。'
         '違うのは前提の外の 1 つの形だけ: 0.5 秒より長い AI 側の中立止め(カメラの止まりなど)の間に、その区間で 0.5 m/s(1000 eRPM)より'
         '速く走った車が止まった発進で、両方 ON は明けたフレームで中止(出来事 hold_long。人を待つ)、OFF は走り続ける(閉じたループで 92 組。'
         '当たりは ON・OFF とも 0)。⑤ はその発進を「R1 に届かなかった発進」(ON の組なら満たさない)、その場で戻した発進を「リカバリー中止の'
         'あとの走り直し」(比べから外す)にする =「使ってよい」の側には動かない。10/23・10/24 の AI 側の処理したフレームの間 45,918 のうち、'
         'この形は 0(0.5 秒より長いのは 1 つで、走る前の止まっている車 = 中止にならない)。閉じたループの偽物の ESC・車での確かめで、本物の '
         'ESC・車では確かめていない。前提が崩れたら (c)(その日の回 3 は「満たさない」)に戻す(main session の決めの条件) |' + AP_3R_OLD)
AP_8T_OLD = '| 試しの本・途中で変えた本 | 比べに入れない(ON の最初の周の衝突だけは数える) |'
AP_8T = ('| 試しの本・途中で変えた本 | 比べに入れない(ON の最初の周の衝突・空転(比 > 1.3・比が出ない)・最初の周の記録の欠けは数える = '
         '満たさない。空転は R1 に届いた発進だけ。10/28 main session の決め・fix8 tools。前は衝突だけ) |')
AP_8R_OLD = '\n\n- 車のせいでない理由(人・物がコースに入った)で止めた本は'
AP_8R = ('\n| リカバリーの途中で CH3 を人側にしたあとの走り直し(10/28) | 上の行と同じ(比べに入れない。ON の最初の周の衝突・空転・最初の周の'
         '記録の欠けは数える。③⑤ の形の文「リカバリーの途中 X 秒から Y 秒」。10/28 main session の決め・fix8 tools。本物の 10/10〜10/24 で'
         '印が付いた発進は 3 つで、どれも前から数えない発進 = 判定は変わらない) |' + AP_8R_OLD)
# SUPERSEDE(final の approvals #3): 8 の「fix7 で残した読み方」①②③ に印
AP_81_X = 'で守る)② 試しの本・途中で変えた本の ON の空転'
AP_81_Y = ('で守る)→ **変えた**(10/28 main session の決め。fix8 tools-decided で夜の道具に入った: ③⑤ の「リカバリーの途中 X 秒から Y 秒」= '
           '走り直しとして ⑤ の比べ・③ の止まってから 3 周・④ の「動く → A1」に入れない。本物の 10/10〜10/24 で印の付いた発進 3 つは、どれも'
           '前から数えない発進で、判定は変わらない)② 試しの本・途中で変えた本の ON の空転')
AP_82_X = '走り直しの ON は数える)③ 0.5 秒より短い中立止め'
AP_82_Y = ('走り直しの ON は数える)→ **変えた**(10/28 main session の決め。fix8 tools: ⑤ が試し・途中で変えた本の ON の空転と最初の周の'
           '記録の欠けも数える =「  → 試しの本 / 途中で変えた本の ON の発進が空転した N 本 …」などで満たさない)③ 0.5 秒より短い中立止め')
AP_83_X = '→ main session が `--skip 記録名@AI側にした秒`)。\n'
AP_83_Y = '→ main session が `--skip 記録名@AI側にした秒`)→ **今のまま**(10/28 main session の決め)。\n'
# SUPERSEDE(final の approvals #4): 14 の頭
AP_14_X = '17:16 の起動には入っていない(2 の「今」・`fix7/README.md` 5 章)。'
AP_14_Y = '17:16 の起動には入っていない(2 の「今」・`fix7/README.md` 5 章)→ ' + RESTART + ' のもう一度の再起動で入った。'

# ==================================================================================================== fix3/disk.md
DK_LOG_OLD = '\n\n## 0. 要点'
DK_LOG = ('\n- **' + STAMP + ' 追記**(fix8 係 docs-decided。控え `disk.md' + BAK + '`): 0 章・2 章の pytest 一式の作業場所を 10/28 の実測に'
          '(`-o tmp_path_retention_policy=failed` を付けて約 0.6 GB・付けないと約 7.1 GB。`disk_check.py` の `PYTEST_GB` も 0.6・'
          '`PYTEST_GB_KEEP` 7.1 = fix8 tools)。ほかの数は 10/26 20:55 の見積もりのまま(片付けは 10/27 に A1・A2・B1。今の空きは '
          '`disk_check.py` で)。' + DK_LOG_OLD)
DK_18_OLD = '| pytest 一式の最中 | **3.0 GB**(作業場所 3.9 GB)。終わって消せば 6.9 GB に戻る |'
DK_18 = ('| pytest 一式の最中 | **3.0 GB**(作業場所 3.9 GB = 10/26 の見積もり)。終わって消せば 6.9 GB に戻る。**10/28 の実測**: 作業場所は '
         '`-o tmp_path_retention_policy=failed` を付けて約 0.6 GB(10/28 の一式: 空き 10493M → いちばん少なくて 9913M。終わりの作業場所 55M。'
         '`../fix7/pytest_full_20261028.txt`)・付けないと約 7.1 GB(10/27 の一式: 空き 11 GB → 3.4 GB。`../fix6/pytest_full_20261027.txt`)。'
         '今の Jetson で `disk_check.py` を流すと「pytest 一式の最中: X GB(作業場所 0.6 GB。-o tmp_path_retention_policy=failed を付けた'
         'とき。付けないと作業場所 7.1 GB で Y GB)。終わって消せば Z GB に戻る」の形で出す(10/28 fix8 tools) |')
DK_47_OLD = '| pytest 一式 | 3.9 GB(作業場所) | verify.md。既定の場所だと `/tmp/pytest-of-jetson` に **3 回分残る**(2 回流すと 7.8 GB) |'
DK_47 = ('| pytest 一式 | **約 0.6 GB**(作業場所。`-o tmp_path_retention_policy=failed` を付けたとき。10/28 の実測)・付けないと **約 7.1 GB**'
         '(10/27 の実測)。前に書いた 3.9 GB は 10/26 の小さい一式 | 0.6 GB: `../fix7/pytest_full_20261028.txt`(10/28 00:08〜00:52。空き '
         '10493M → いちばん少なくて 9913M(00:44)・終わりの作業場所 55M)。7.1 GB: `../fix6/pytest_full_20261027.txt`(空き 11 GB → 3.4 GB)。'
         '3.9 GB: verify.md。`disk_check.py` の `PYTEST_GB` 0.6・`PYTEST_GB_KEEP` 7.1(10/28 fix8 tools)。既定の場所だと '
         '`/tmp/pytest-of-jetson` に **3 回分残る**(10/26 の大きさで 2 回流すと 7.8 GB) |')

# ==================================================================================================== fix6/README.md
F6_LOG_OLD = '- **2 巡目にも新しい指摘が 5 件出た = 出切っていない**(7 章)。\n'
F6_LOG = ('- ' + STAMP + ' fix8 係 docs-decided(控え `README.md' + BAK + '`): 7 章 1 に「' + RESTART + ' のもう一度の再起動で fix7 の直しが'
          '入った・一式は `-o tmp_path_retention_policy=failed` で約 0.6 GB」。報告 `../fix8/decided_docs.md`。\n' + F6_LOG_OLD)
# SUPERSEDE(r1 の fix6/README #2): 7 章 1
F6_71_X = '`../README.md` 3.2 の 4) | main session |'
F6_71_Y = ('`../README.md` 3.2 の 4)。→ 10/28: fix7 の直しは ' + RESTART + ' のもう一度の再起動で入った・一式は '
           '`-o tmp_path_retention_policy=failed` を付けて約 0.6 GB(`../fix7/README.md` 7 章・`../README.md` 3.2 の 4) | main session |')

# ==================================================================================================== fix7/README.md(7 章の「次に」に印だけ)
F7_LOG_OLD = '(5 章)。\n\n## 0. ひと目で'
F7_LOG = ('(5 章)。\n- (' + STAMP + ' fix8 係 docs-decided。控え `README.md' + BAK + '`)7 章の「次に」の済んだ行に印だけ付けた(再起動は '
          '10/28 00:53:23・試験一式 3415 passed と commit 9da6d0d・反証 8 回目を始めた・approvals 3 は (a))。ほかは 10/27 23:5x の記録のまま。'
          '\n\n## 0. ひと目で')
F7_1_OLD = '| 1 | **2 巡目にも新しい指摘が 10 件出た = 出切っていない**。'
F7_1 = '| 1 | **済**(10/28: 反証 8 回目を始めた)。**2 巡目にも新しい指摘が 10 件出た = 出切っていない**。'
F7_2_OLD = '| 2 | **jetformer-web の再起動**(5 章。'
F7_2 = '| 2 | **済**(10/28 00:53:23。drive.py の説明の文のぶんは反証 8 回目のあと)。**jetformer-web の再起動**(5 章。'
F7_3_OLD = '| 3 | 試験一式と commit('
F7_3 = '| 3 | **済**(10/28: 3415 passed・commit 9da6d0d を push)。試験一式と commit('
F7_4_OLD = '| 4 | approvals 3「途中で OFF にした日」の決めと、3 章の「文書・コードの文」①〜⑤ |'
F7_4 = ('| 4 | **済**(10/28: approvals 3 は (a)・①〜④ は fix8 の係が直した・⑤ は再起動の行だけ(approvals 6・「ON の形」の注の行はまだ))。'
        'approvals 3「途中で OFF にした日」の決めと、3 章の「文書・コードの文」①〜⑤ |')

# ==================================================================================================== 一覧
# (文書, old, new)。足す形(old が new の中にある)と、SUPERSEDE の書き換え(old = 前の係の new の中の文)
ALL = [
    (RB, [(RB_LOG_OLD, RB_LOG + RB_LOG_OLD), (RB_1_OLD, RB_1), (RB_31_OLD, RB_31), (RB_OFF_X, RB_OFF_Y), (RB_HOLD_OLD, RB_HOLD),
          (RB_HIT_OLD, RB_HIT), (RB_TRIAL_OLD, RB_TRIAL), (RB_8_OLD, RB_8), (RB_2_X, RB_2_Y), (RB_3_OLD, RB_3), (RB_4_OLD, RB_4),
          (RB_5_OLD, RB_5), (RB_MID_OLD, RB_MID), (RB_MIX_X, RB_MIX_Y), (RB_9_X, RB_9_Y)]),
    (TODO, [(TD_LOG_OLD, TD_LOG + TD_LOG_OLD), (TD_26A_OLD, TD_26A), (TD_26B_OLD, TD_26B), (TD_26C_X, TD_26C_Y), (TD_20_OLD, TD_20),
            (TD_GIT_OLD, TD_GIT), (TD_10_OLD, TD_10), (TD_13_OLD, TD_13), (TD_16_X, TD_16_Y), (TD_RES_OLD, TD_RES),
            (TD_TRIAL_OLD, TD_TRIAL), (TD_OFF_OLD, TD_OFF), (TD_4_OLD, TD_4)]),
    (RD, [(RD_LOG_OLD, RD_LOG + RD_LOG_OLD), (RD_7_OLD, RD_7), (RD_8_X, RD_8_Y), (RD_PY_X, RD_PY_Y), (RD_6_OLD, RD_6),
          (RD_61_OLD, RD_61), (RD_65_OLD, RD_65), (RD_628_X, RD_628_Y), (RD_626_X, RD_626_Y), (RD_629_X, RD_629_Y),
          (RD_632_OLD, RD_632), (RD_631_X, RD_631_Y), (RD_64_X, RD_64_Y), (RD_7C_OLD, RD_7C)]),
    (AP, [(AP_NOW_OLD, AP_NOW), (AP_02_X, AP_02_Y), (AP_03_OLD, AP_03), (AP_08_OLD, AP_08), (AP_1_OLD, AP_1), (AP_2N_X, AP_2N_Y),
          (AP_2W_X, AP_2W_Y), (AP_3D_OLD, AP_3D), (AP_3C_OLD, AP_3C), (AP_3R_OLD, AP_3R), (AP_8T_OLD, AP_8T), (AP_8R_OLD, AP_8R),
          (AP_81_X, AP_81_Y), (AP_82_X, AP_82_Y), (AP_83_X, AP_83_Y), (AP_14_X, AP_14_Y)]),
    (DK, [(DK_LOG_OLD, DK_LOG), (DK_18_OLD, DK_18), (DK_47_OLD, DK_47)]),
    (F6, [(F6_LOG_OLD, F6_LOG), (F6_71_X, F6_71_Y)]),
    (F7, [(F7_LOG_OLD, F7_LOG), (F7_1_OLD, F7_1), (F7_2_OLD, F7_2), (F7_3_OLD, F7_3), (F7_4_OLD, F7_4)]),
]

# 前の係の文(その台本の ALL の (文書, 何番目) の new)の中を意図して書き換えた所: (台本, 文書, 何番目, 前の文 X, 今の文 Y)。
# 前の係の試験は revert() で X に戻した文書で前の台本の直しが「済」かを照らし、effective() で X → Y にした new を使う
SUPERSEDE = [
    ('r1', RB, 2, RB_OFF_X, RB_OFF_Y),          # 3 章「わざと OFF にした日」: main session が決める → (a)(10/28 main session)
    ('r1', RB, 15, RB_MIX_X, RB_MIX_Y),         # 8.1 ⑤: 同上・混ざった注意の行の後ろの文(fix8 tools)
    ('r1', RD, 2, RD_PY_X, RD_PY_Y),            # README 3.2 の 4: PYTEST_GB 3.9 のまま → 0.6(fix8 tools)・10/28 の実測
    ('r1', F6, 2, F6_71_X, F6_71_Y),            # fix6/README 7 章 1: 10/28 の再起動と一式(足すだけ)
    ('r2', RB, 13, RB_2_X, RB_2_Y),             # 8.1 ②: 「回の本は X(No.Y)から」・印の後ろの文(足すだけ)
    ('r2', RB, 19, RB_9_X, RB_9_Y),             # 9 章 rec_hold_keep: 10/28 の再起動で入った(足すだけ)
    ('r2', TODO, 1, TD_26C_X, TD_26C_Y),        # todo 1 章: 10/28 の再起動で入った(足すだけ)
    ('r2', RD, 1, RD_8_X, RD_8_Y),              # README 0 章 8(足すだけ)
    ('r2', RD, 3, RD_628_X, RD_628_Y),          # README 6.2 の 8(足すだけ)
    ('r2', RD, 4, RD_626_X, RD_626_Y),          # README 6.2 の 26(足すだけ)
    ('r2', RD, 5, RD_64_X, RD_64_Y),            # README 6.4(足すだけ)
    ('r2', AP, 2, AP_02_X, AP_02_Y),            # approvals 0 章 2(足すだけ)
    ('r2', AP, 3, AP_2N_X, AP_2N_Y),            # approvals 2 の「今」(足すだけ)
    ('final', TODO, 2, TD_16_X, TD_16_Y),       # todo 2 章 16(足すだけ)
    ('final', RD, 2, RD_629_X, RD_629_Y),       # README 6.2 の 29(足すだけ)
    ('final', RD, 3, RD_631_X, RD_631_Y),       # README 6.3 の 31(足すだけ)
    ('final', AP, 2, AP_2W_X, AP_2W_Y),         # approvals 2 の「誰が」: main session の判断(足すだけ)
    ('final', AP, 3, AP_81_X, AP_81_Y),         # approvals 8「fix7 で残した読み方」①: 変えた
    ('final', AP, 3, AP_82_X, AP_82_Y),         # 同 ②: 変えた
    ('final', AP, 3, AP_83_X, AP_83_Y),         # 同 ③: 今のまま
    ('final', AP, 4, AP_14_X, AP_14_Y),         # approvals 14 の頭: 10/28 の再起動で入った(足すだけ)
]
# 文を消す書き換え(ほかは X の文がそのまま Y に残る = 足すだけ)
REMOVES = {(RB, RB_OFF_X), (RB, RB_MIX_X), (RD, RD_PY_X)}


def effective(script, f, i, new):
    """前の台本 script の (文書 f, 何番目 i) の new の、fix8 が書き換えたあとの形(書き換えていなければ new のまま)"""
    for s, ff, ii, x, y in SUPERSEDE:
        if (s, ff, ii) == (script, f, i):
            assert new.count(x) == 1, (s, ff, ii, x[:30])
            new = new.replace(x, y, 1)
    return new


def revert(texts):
    """fix8 の SUPERSEDE の書き換えを戻した文書(texts にある文書だけ。足しただけの直しはそのまま)"""
    out = dict(texts)
    for _, f, _, x, y in SUPERSEDE:
        if f in out and out[f].count(y) == 1:
            out[f] = out[f].replace(y, x, 1)
    return out


def superseded_present(texts):
    """SUPERSEDE の書き換えが今の文書に入っている(Y がちょうど 1 回。文を消す所は X が 0 回)か。戻り値は問題の一覧"""
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def read_all(root=REPORTS):
    return {f: (root / f).read_text(encoding='utf-8') for f in FILES}


def earlier_problems(texts):
    """前の係の直しと試験の確かめが崩れていないか(戻り値は問題の一覧):
    - SUPERSEDE の X が、その前の台本の new の中にちょうど 1 回(= 前の係の文の中だけを書き換えた)
    - 書き換えを戻した文書で、前の 3 つの台本の直しが全部「済」で new がちょうど 1 回・画面の版 9 の直しも「済」で古い版の数 0
    - 今の文書で、前の係の試験の確かめ(f1〜f5・g1〜g6・h1〜h5)と、この組の試験の確かめ(i1〜i6。試験のファイルがあれば)が空"""
    bad = []
    mods = {k: _load('fix8_docs_prev_%s' % k, p) for k, p in SCRIPTS.items()}
    for s, f, i, x, y in SUPERSEDE:
        edits = dict(mods[s].ALL)[f]
        if edits[i][1].count(x) != 1:
            bad.append(('SUPERSEDE の X が前の new の中に 1 回でない', s, f, i))
    old = revert(texts)
    for k, m in mods.items():
        _, rep = m.apply_text(old)
        bad += [(k, f, o[:30], st) for f, o, st in rep if st != 'done']
        for f, edits in m.ALL:
            for _, new in edits:
                if old[f].count(new) != 1:
                    bad.append((k + ' count', f, new[:30], old[f].count(new)))
    bad += [('r1 (r2.r1_intact)',) + tuple(x) for x in mods['r2'].r1_intact(old)]
    sys.path.insert(0, str(JF))
    sys.path.insert(0, str(JF / 'tools'))
    sys.path.insert(0, str(JF / 'tests'))
    t1 = _load('t_fix7_docs_r1_from_fix8', TESTS['r1'])
    t2 = _load('t_fix7_docs_r2_from_fix8', TESTS['r2'])
    t3 = _load('t_fix7_docs_final_from_fix8', TESTS['final'])
    for k, fn in sorted(t1.CHECKS.items()):
        bad += [('t1 ' + k, p) for p in fn(texts)]
    with tempfile.TemporaryDirectory(prefix='docs_fix8_') as tmp:
        for k, fn in sorted(t2.CHECKS.items()):
            bad += [('t2 ' + k, p) for p in fn(texts, Path(tmp) / k)]
    for k, fn in sorted(t3.CHECKS.items()):
        bad += [('t3 ' + k, p) for p in fn(texts)]
    bad += [('t3 h5', p) for p in t3.h5_problems(texts[F7])]
    if TESTS['fix8'].exists():
        t4 = _load('t_fix8_docs_decided_from_fix8', TESTS['fix8'])
        for k, fn in sorted(t4.CHECKS.items()):
            bad += [('t4 ' + k, p) for p in fn(texts)]
    bad += [('superseded',) + tuple(x) for x in superseded_present(texts)]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args(argv)
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
    prob = earlier_problems(new)
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
