#!/usr/bin/env python3
"""fix8 係 docs 仕上げ(反証 8 回目の終わり。2026-10-28): 反証 8-2 回目の直しの係の「手順書に足す文」と、安全 8-2 回目の確かめて残った
指摘・食い違い 1 つを、ユーザーが読む文書(手順書・todo・README・fix3/approvals.md)に入れる。コードは変えない。

- 画面(fix8/fix_ui_r2.md 11 章): 手順書 3.1 の app.js の行(app.js 10/28 11:02・style.css 10:15)・4 章 6 の電圧の源・4 章 8 の ■ と
  トースト・7 章「Failed to fetch」の app.js の時刻、todo 3 章の画面の行、README 6.2 の 11。
- 夜の道具(fix8/fix_tools_r2.md 10 章): 手順書 6 章「発進の前進 ON」に off_all_compared(と、R1 に届かなかった OFF の扱いの変更 = 10/26
  夜 fix3 r1-rec3-1 の決めを変えた。main session が了承)・8.1 ⑤ に短い止まりの続き・外れた OFF・注意の行・3 章「わざと OFF にした日」、
  approvals 3「途中で OFF にした日」の決まり・8 の表(行の書き換えと新しい行)。
- 安全(反証 8-2 回目 rs8-2-reverse-end-stall-overrun。11/3 の安全のコードは変えない = main session の決め): approvals 3「困ること(ON)」・
  手順書 7 章「1 m 以上下がる」に 1 文、approvals 14 に新しい行、todo 1 章・2 章 10・2 章 16、README 6.2 に新しい 37(ユーザーに伝えること)、
  approvals 0 章の 3・14 に印。直し方の案は fix8/proposal_safety_after1103.md に新しい 10 章(この台本の外。PROPOSAL_CH10)。
- 食い違い(main session の決め: 11/3 は手順書を正とする): 手順書 8.1 ⑤ に「⑤ の出力の --skip の文は、当たった・空転した・記録が欠けた
  ON の本には 11/3 は使わない」。
- fix8/fix_docs_r2.md の「残したこと」5: approvals 14「後ろ 1.2・1.35 m の物」の行に今の後退の幅。README 6 章の頭に反証 8 回目のまとめ・
  6.3 に新しい 38・7 章に fix8/README.md と 8-2 回目の報告。
- 頭の記録の 1 行(4 本)・todo の題の時刻。

置き換えの決まりは前の台本と同じ: old が文書の中でちょうど 1 回・new が 0 回のときだけ入れる。2 回目に流すと全部「済」。
前の係の文(前の台本の new。あとの台本が書き換えたあとの今の形)の中を書き換えるのは SUPERSEDE の所だけ。前の係の試験は revert() で
この組の書き換えを戻した文書で前の台本の直しが「済」かを照らし、effective() で書き換えたあとの形の new を使う。
--apply: 控え <名前>.bak-20261028-fix8-final を作ってから書く(控えがあれば上書きしない)。案の報告の 10 章も同じとき(PROPOSAL_CH10 を
末尾に足すだけ。1〜9 章は変えない)。

使い方: python3 -I -B docs_fix8_final.py [--apply]
"""
import argparse
import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
JF = Path('/home/jetson/jetformer')
E = JF / 'experiments' / '20261026_fix'
HERE = E / 'fix8' / 'docs_final'
PREV = {'r1': E / 'fix7' / 'docs_r1' / 'docs_fix7_r1.py', 'r2': E / 'fix7' / 'docs_r2' / 'docs_fix7_r2.py',
        'final': E / 'fix7' / 'docs_final' / 'docs_fix7_final.py', 'decided': E / 'fix8' / 'docs_decided' / 'docs_fix8_decided.py',
        'r81': E / 'fix8' / 'fix_docs_r1' / 'docs_fix8_r1.py', 's1': E / 'fix8' / 'docs_sync1' / 'docs_fix8_sync1.py',
        'r82': E / 'fix8' / 'fix_docs_r2' / 'docs_fix8_r2.py'}
TESTS = {'r1': JF / 'tests' / 'test_fix7_docs_r1.py', 'r2': JF / 'tests' / 'test_fix7_docs_r2.py',
         'final': JF / 'tests' / 'test_fix7_docs_final.py', 'decided': JF / 'tests' / 'test_fix8_docs_decided.py',
         'r81': JF / 'tests' / 'test_fix8_docs_r1.py', 's1': JF / 'tests' / 'test_fix8_docs_sync1.py',
         'r82': JF / 'tests' / 'test_fix8_docs_r2.py', 'fin': JF / 'tests' / 'test_fix8_docs_final.py'}
BAK = '.bak-20261028-fix8-final'
STAMP = '10/28 12:4x'
RB, TODO, RD, AP, F7 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'
FILES = (RB, TODO, RD, AP)
ALL7 = (RB, TODO, RD, AP, 'fix3/disk.md', F7, 'fix6/README.md')
PROPOSAL = 'fix8/proposal_safety_after1103.md'
PROPOSAL_BEFORE_SHA = '6773816649fb7ecaa1c43e5a6c323a5d1f24985425f7a51c1e3dd5c1264bf1cb'      # 10/28 07:46 の版(1〜9 章。53,507 バイト)
PROPOSAL_BEFORE_LEN = 53507

DEC = '10/28 main session の決め・fix8 docs 仕上げ'
SAFE_ID = 'rs8-2-reverse-end-stall-overrun'
# 安全の数(指摘 review_safety_r2/findings.json・票 refute_safety_r2_A・_B の verdicts.json と out/ から。試験が照らす)
SAFE_STALL = '後退の終わりにカメラが 0.1〜0.3 秒止まる'
SAFE_MORE = 'ON は OFF より最大 +0.6 m(乱れの無いときより最大 +0.55 m)多く下がることがある'
SAFE_REAR = '後ろ 1.3〜1.7 m の物に ON だけ当たる場面がある'
SAFE_REAL = '本物の 10/10〜10/24 の後退の終わり 11 回でこの止まりは 0 回・1 回のリカバリーあたり約 0.4% の見込み'
SAFE_REV = 'モデルによっては逆に OFF のほうが多く下がる'
SAFE_GUARD = '人の守りは手順書 7 章の「1 m 以上下がる → すぐ人側」'

# ==================================================================================================== 手順書
RB_LOG_OLD = '\n- **10/28 11:4x fix8 係 docs 直す 8-2 回目**'
RB_LOG = ('\n- **' + STAMP + ' fix8 係 docs 仕上げ(反証 8 回目の終わり)**(下の 11:4x の 8-2 回目のあと。`fix8/docs_final.md`・`fix8/README.md`。'
          '控え `runbook_20261103.md' + BAK + '`。手順書だけ。コードは変えていない): 反証 8-2 回目の直しの係の「手順書に足す文」(画面 = '
          '`fix8/fix_ui_r2.md` 11 章・夜の道具 = `fix8/fix_tools_r2.md` 10 章)を 3.1(app.js 10/28 11:02・style.css 10:15)・3 章「わざと OFF に'
          'した日」・4 章 6(電圧の源)・4 章 8(■ とトースト)・6 章「発進の前進 ON」・7 章「Failed to fetch」・8.1 ⑤ に / 7 章「1 m 以上下がる」に、'
          '両方 ON の日の後退の終わりのカメラの止まりの形(反証 8-2 回目 ' + SAFE_ID + '。11/3 のコードは変えない)/ 8.1 ⑤ に「⑤ の出力の '
          '`--skip` の文は、当たった・空転した・記録が欠けた ON の本には 11/3 は使わない = 手順書を正とする」(10/28 main session の決め)。'
          + RB_LOG_OLD)
# 3.1 の app.js の行(SUPERSEDE s1 #3・decided #2)
RB_31_X = ('読み直した画面の ▶ の横は「画面の版 9(app.js 10/28 04:54)」(このあと誰かが app.js を直せばその時刻)。**04:54 より前に開いた画面**'
           '(スマホで開いたまま・前の日の画面)は ▶ で 409「管理画面が古いままです(この画面の app.js は …、今のファイルは 10/28 04:54 に'
           '直したもの)」→ 読み直す')
RB_31_Y = ('**app.js は 10/28 11:02 にもう一度直した**(fix8 ui r2。▶ とスイッチの確認の文の「電圧の源」の、説明欄の V_before の読み方をサーバーと'
           '同じにした。表示だけ・画面の版は 9 のまま・送る物は同じ)。style.css も 10/28 10:15 に直した(走行中に下に浮いている ■「■ AI走行を停止'
           '(スペースキー)」がスマホで画面の半分の幅・3〜4 行になり、▶ のすぐあとのトースト(3〜6 秒)に上の行を隠されていた → 文の幅の 1〜2 行に'
           'し、幅 600 px 以下の画面では走行中のトーストを ■ の上に出す)。どちらも静的なファイルなので再起動は要らず、画面を読み直せば入る'
           '(`fix8/fix_ui_r2.md`)。読み直した画面の ▶ の横は「画面の版 9(app.js 10/28 11:02)」(このあと誰かが app.js を直せばその時刻)。'
           '**11:02 より前に開いた画面**(スマホで開いたまま・前の日の画面。10/28 04:54〜11:02 に開いた画面も)は ▶ で 409「管理画面が古いままです'
           '(この画面の app.js は …、今のファイルは 10/28 11:02 に直したもの)」→ 読み直す')
# 4 章 6 の電圧の源(SUPERSEDE r81 #3)
RB_46_X = '「説明欄の V_before(走行のあいだ一定)」なら OK。'
RB_46_Y = (RB_46_X + '(10/28 fix8 ui r2 の画面(app.js 11:02)では、「説明欄」と出るのはサーバーがその値を走行に使うときだけ = 説明欄の '
           '`V_before=` が 8.31 のような 6.0〜9.0 の数のとき。831(点の打ち忘れ)・8,31・0.83・9.5・「8.31,」・セルの値 4.15 は「なし(1.00 倍。'
           '周ごとの直しだけ)」と出る → 下のとおりキャンセル → 説明欄の V_before を直す → もう一度 ▶。前の画面は V_before= のあとが数字で始まれば'
           '「説明欄」と出て、サーバーは使わず「なし」で走った。`fix8/fix_ui_r2.md` 4 章)')
# 4 章 8(前の係の new の外)
RB_48_OLD = '8. 決めた周を走ったら、次の直線で CH3 を人側 → ■ 停止。\n'
RB_48 = (RB_48_OLD + '   - 読み直した画面(style.css 10/28 10:15)では、▶ のすぐあとに出るトースト(「AI走行 No.N を開始しました」3 秒・11/3 のブロックの'
         '「試し」の赤い文 6 秒)は下の ■ の上に出て、■ の字は隠れない(前は 3〜6 秒、■ の上の行が隠れた。押せば止まるのは前も今も同じ。'
         'fix8 ui r2)。■ の字が隠れて見えたら画面を読み直す。\n')
# 6 章「発進の前進 ON(回 4)」(前の係の new の外)
RB_6_OLD = '10/26 夜 fix3 r1-rec3-1。前はその本を黙って外して「使ってよい」と出た)。'
RB_6 = (RB_6_OLD + '**「→ OFF の発進が比べから外れた N 本」の行が出たら「満たさない」**(checks の off_all_compared。OFF の発進がカメラの止まりで'
        '比べから外れたか、OFF の本(回の計画の 1 本 = 説明欄の block)の発進が R1 に届かず比べに 1 本も残らない = 残りの OFF の中央値で比べると、'
        '速い OFF が外れたとき「使ってよい」の側に動く。10/28 fix8 tools r2。前は外れた OFF が 2 本になると残りの平均で「使ってよい」が出た = '
        '反証 8-2 回目 w2-tools-r8-2-off-few-left。**R1 に届かなかった OFF の発進は「名前を出すだけで、組の答えは変えない」(10/26 夜 fix3 '
        'r1-rec3-1 の決め)だったのを、10/28 に変えた**(main session が了承。`fix8/fix_tools_r2.md` 8 章))。同じ本の中で置き直して走り直した'
        '発進や、同じ block で別の記録にやり直した本(同じベンチマーク N でもう 1 本)が比べに入っていれば数えない。車のせいでない理由(人・物が'
        'コースに入った・試し)で止めた本だけ、main session が `--skip 記録名` で外して読み直す(外すと残りの OFF で比べる = 速い OFF を外すと'
        '「使ってよい」の側に動きうる。決めは main session)。')
# 3 章「わざと OFF にした日」(SUPERSEDE decided #3・r1 #2 と r2 #4)
RB_3X_X = '⑤ はそれを「R1 に届かなかった発進」・その場で戻した発進を走り直しにする =「使ってよい」の側には動かない。'
RB_3X_Y = ('⑤ はそれを「R1 に届かなかった発進」・その場で戻した発進を走り直しにし、**発進の前進 OFF の本なら、その本の発進が比べに 1 本も'
           '残らなければ「→ OFF の発進が比べから外れた N 本」で満たさない**(10/28 fix8 tools r2。前は外れた OFF が 2 本になると残りの平均で'
           '「使ってよい」が出た = 反証 8-2 回目 w2-tools-r8-2-off-few-left)=「使ってよい」の側には動かない。')
RB_3E_X = '= この形でも「使ってよい」の側には動かない(8.1 ⑤・`fix3/approvals.md` 3)。'
RB_3E_Y = RB_3E_X + '外した OFF の本も「→ OFF の発進が比べから外れた N 本」で満たさない(10/28 fix8 tools r2)。'
# 7 章「1 m 以上下がる」(SUPERSEDE s1 #4。fix8 docs r2 の書き換えの前の所)
RB_7_X = '3 章の「10/23 の形の後ろの近い物」・`fix3/approvals.md` 3。10/28 fix8 safety r1)。'
RB_7_Y = (RB_7_X + '**両方 ON の日、' + SAFE_STALL + 'と、' + SAFE_MORE + '**(止まりの間は前に書いた後退側が中立止めまで出続け(ON・OFF とも)、'
          '明けたあと ON は惰性を中立止めの始めから 0.25 秒続けてから前へ出る。OFF は手順を戻し、モデルの前進が後ろへの惰性を止める。'
          + SAFE_REAR + ' = 上の 10/23 の形の「約 1 m より遠い物には ON のほうが遅く当たるか当たらない」とは別の形。' + SAFE_REAL + '。'
          + SAFE_REV + '。11/3 のコードは変えない(10/28 main session の決め)。人の守りはこの行の「1 m 以上下がる → すぐ人側」(上乗せの 0.3 秒'
          'ほどには人は間に合わないので、そのあとの動きを止める)。反証 8-2 回目 ' + SAFE_ID + '。`fix3/approvals.md` 3 の「困ること(ON)」・14。'
          'fix8 docs 仕上げ)。')
# 7 章「Failed to fetch」(SUPERSEDE s1 #7)
RB_7F_X = '(10/28 04:54 の app.js を読み直した画面。'
RB_7F_Y = '(10/28 04:54 からの app.js(今は 10/28 11:02 の版)を読み直した画面。'
# 8.1 ⑤ の夜の道具(SUPERSEDE s1 #11 を 2 か所)
RB_8O_X = 'OFF が全部外れたら「→ OFF の発進が全部カメラの止まりで比べから外れた」(off_no_stall = 満たさない)。'
RB_8O_Y = RB_8O_X + '1 本でも外れたら「→ OFF の発進が比べから外れた N 本」(off_all_compared = 満たさない。下の文。10/28 fix8 tools r2)。'
RB_8N_X = '(どちらの行も出ない。`fix8/fix_tools_r1.md` 3 章)'
RB_8N_Y = (RB_8N_X + '。**「→ うち短い止まりが続いた発進 N 本」** = 1 回のフレームの間は 0.22 秒以下でも、drive が中立を書いた時間の見積もり(処理した'
           'フレームの間ごとに、前のフレームの 0.1 秒あとか間の最初の中立止めの行から次のフレームまで)の合計が 0.22 秒より長い発進。「発進から '
           'R1 の起点までにカメラが止まった発進」と同じに比べから外す(ON の本なら on_no_stall、OFF の本なら off_all_compared で満たさない)。'
           '名前の後ろが「短い止まりが続いた: 中立の見積もりの合計 X 秒(中立 N 回・いちばん長いフレームの間 Y 秒(Z 秒から))」、発進の行の'
           '後ろが「カメラの止まりが続いた(…。比べに入れない)」(10/28 fix8 tools r2)。**「→ OFF の発進が比べから外れた N 本(…)」** = 6 章。'
           '名前の後ろが「(カメラの止まり・起点まで X 秒)」か「(R1 に届かなかった・理由)」。OFF が全部 R1 に届かない日も「ON − OFF: 起点まで — 秒 '
           '… → 満たさない」の行が出る(前は「ON − OFF」の行が出なかった。10/28 fix8 tools r2)。**「注意: 発進から R1 の起点までに短い止まりが'
           '続いた発進 N 本」** = 中立が 2 回以上・合計 0.05〜0.22 秒の、比べに入れた発進(外さない)。発進の始めの加速の途中なら起点までが中立の'
           '合計くらいまで伸びうる(OFF の本なら「使ってよい」の側)。OFF の本にこの行があって「ON − OFF」が「使ってよい」なら、main session が'
           'その本の動画で発進を確かめて決める(`--skip 記録名@T` で外して読み直せる)。本物の 10/10〜10/24 の比べに入る発進では出ない(中立は 1 回・'
           '0.033 秒まで。`fix8/fix_tools_r2.md` 3 章。10/28 fix8 tools r2)')
# 8.1 ⑤ の --skip の文のあと(前の係の new の外): ⑤ の出力の --skip の文は 11/3 は使わない
RB_8S_OLD = '`collect.py` の `--skip` の説明「当たった本は外さない」と同じ向きで、空転・欠けも(10/28 main session の決め・fix8 docs r2)。'
SKIP_FIN = ('**⑤ の出力の `--skip` の文(「動画・紙で確かめ、当たっていない・車のせいでないと分かった本だけ、main session が --skip 記録名 で'
            '外して読み直せる」(読めない・欠けた ON の本の行)・「--skip 記録名 で外せる」(「記録が欠けている本」「読めない記録」の行)・「決めは '
            'main session(--skip 記録名 で外すと、起点までの時間の比べからも外れる)」(空転の比が出ない ON の発進の行)など)は、当たった・空転'
            'した・記録が欠けた ON の本には 11/3 は使わない = この手順書を正とする**(⑤ の文は道具の形のまま。読むと「動画で当たっていないと'
            '分かれば外せる」ととれるが、外すとその本の当たり・空転・欠けまで比べの条件から消える。' + DEC + ')。')
RB_8S = RB_8S_OLD + SKIP_FIN

# ==================================================================================================== todo
TD_TITLE_X = '# ユーザーが 11/3 までにすること(10/28 11:4x 現在)'
TD_TITLE_Y = '# ユーザーが 11/3 までにすること(' + STAMP + ' 現在)'
TD_LOG_OLD = '- **10/28 11:4x fix8 係 docs 直す 8-2 回目**(下の 09:1x の同期 1 のあと。`fix8/fix_docs_r2.md`。'
TD_LOG = ('- **' + STAMP + ' fix8 係 docs 仕上げ(反証 8 回目の終わり)**(下の 11:4x の 8-2 回目のあと。`fix8/docs_final.md`・`fix8/README.md`。'
          '控え `todo_user_1026.md' + BAK + '`): 題の時刻を今に / 1 章「11/2 まで」・2 章 10 に approvals 3 のもう 1 つの知らせ(両方 ON の日の'
          '後退の終わりのカメラの止まり)・2 章 16 に approvals 14 の新しい知らせ / 3 章の画面の行を今の app.js 10/28 11:02・style.css 10:15 に'
          '(反証 8-2 回目の画面の係の直し)。\n' + TD_LOG_OLD)
NOTICE2 = ('**もう 1 つ(10/28 fix8 docs 仕上げ)**: 両方 ON の日、' + SAFE_STALL + 'と、ON は OFF より最大約 0.6 m 多く下がることがある'
           '(' + SAFE_REAR + '。本物の記録では 0 回・1 回のリカバリーあたり約 0.4% の見込み)。11/3 のコードは変えない。守りは手順書 7 章の'
           '「1 m 以上下がる → すぐ人側」(`fix3/approvals.md` 3 の「困ること(ON)」・14・`README.md` 6.2 の 37)')
# 1 章「11/2 まで」(SUPERSEDE r82 #2・r1 #2。fix8 docs r2 の知らせのすぐあと)
TD_23_X = '`README.md` 6.2 の 33。10/28 fix8 docs r2))。'
TD_23_Y = '`README.md` 6.2 の 33。10/28 fix8 docs r2)。' + NOTICE2 + ')。'
# 2 章 10(SUPERSEDE r82 #3)
TD_48_X = '`README.md` 6.2 の 33。10/28 fix8 docs r2)。11/3 は朝の準備で見張り → 残すの順に ON'
TD_48_Y = '`README.md` 6.2 の 33。10/28 fix8 docs r2)。' + NOTICE2 + '。11/3 は朝の準備で見張り → 残すの順に ON'
# 2 章 16(SUPERSEDE final #2・decided #8)
TD_216_X = '8 の 3 つのうち ①② は 10/28 に main session が変えた = 2 章 13) |'
TD_216_Y = ('8 の 3 つのうち ①② は 10/28 に main session が変えた = 2 章 13。14 には 10/28 に表の下へ知らせを 1 つ足した = 両方 ON の日の後退の終わりの'
            'カメラの止まり(上の 10 の「もう 1 つ」。返事が無ければこのまま)) |')
# 3 章の画面の行(SUPERSEDE s1 #2)
TD_3S_X = '今の app.js は 10/28 04:54(▶ の横「画面の版 9(app.js 10/28 04:54)」。このあと app.js を直せばその時刻)。'
TD_3S_Y = ('今の app.js は 10/28 11:02(▶ の横「画面の版 9(app.js 10/28 11:02)」。このあと app.js を直せばその時刻。style.css は 10/28 10:15 = '
           '走行中の ■ とトーストの重なりの直し。どちらも読み直しで入る。確認の文の「電圧の源: 説明欄の V_before」はサーバーが使う値(6.0〜9.0)'
           'のときだけ出る = 831・8,31 などは「なし」→ 説明欄を直す。10/28 fix8 ui r2)。')

# ==================================================================================================== README
RD_LOG_OLD = '\n- **10/28 11:4x 追記**(fix8 係 docs 直す 8-2 回目。'
RD_LOG = ('\n- **' + STAMP + ' 追記**(fix8 係 docs 仕上げ(反証 8 回目の終わり)。下の 11:4x の 8-2 回目のあと。控え `README.md' + BAK + '`。'
          '報告 `fix8/docs_final.md`): 6 章の頭に反証 8 回目のまとめ(**`fix8/README.md`**)・6.2 の 11(app.js 10/28 11:02・style.css 10:15)・'
          '6.2 に新しい 37(ユーザーに伝えること: 両方 ON の日の後退の終わりのカメラの止まり・回 3 の外れた OFF・⑤ の `--skip` の文)・6.3 に'
          '新しい 38(反証 8-2 回目で残したこと)・7 章に `fix8/README.md` と 8-2 回目の報告。' + RD_LOG_OLD)
# 6 章の頭(SUPERSEDE decided #4)
RD_6H_X = 'ユーザーに伝えることは 6.2 の 33。'
RD_6H_Y = (RD_6H_X + '\n- (10/28 fix8 docs 仕上げ)**反証 8 回目はここで終える**(main session の決め 10/28 11:3x。まとめは **`fix8/README.md`**): '
           '8-1 は新しい指摘 20 件 → 打ち消し 4 件・残す 16 件(16 件とも直した)。8-2 は新しい指摘 10 件(どれも minor)→ 打ち消し 0 件・残す 10 件と、'
           '8-1 から渡された試験の古い期待 1 件 = 11 件とも直した(安全 1 は文書だけ = 11/3 の安全のコードは変えない・画面 2 = style.css 10/28 10:15・'
           'app.js 11:02(`fix8/fix_ui_r2.md`)・夜の道具 2 = collect.py(`fix8/fix_tools_r2.md`)・文書 5(`fix8/fix_docs_r2.md`)・この仕上げ'
           '(`fix8/docs_final.md`))。**巡ごとに細かい場面になったが、2 巡目にも新しい指摘が出た = 出切っていない**。8-2 の直しはどれも '
           'jetformer-web の再起動が要らない(assist.py・drive.py・web/server.py・web/bench.py は 10/28 08:40:33 の起動より前のまま。app.js・'
           'style.css は画面を読み直せば入る)。ユーザーに伝えることは 6.2 の 37。')
# 6.2 の 11(SUPERSEDE s1 #2)
RD_11_X = '画面の版 9(app.js 10/28 04:54)に読み直してから)'
RD_11_Y = ('画面の版 9(app.js 10/28 11:02)に読み直してから。10/28 fix8 ui r2 で style.css の浮いている ■ を直した: 幅 390 px の画面で ■ が '
           '351×90 px の 2 行(前は 195×120 の 3 行)、走行開始のトーストは ■ の 20 px 上(前は ■ の上の行に 40 px 重なった)。スマホで ▶ のすぐ'
           'あとに ■ の字が全部読めるかを見る。app.js も 11:02 に直した = 確認の文の「電圧の源」が「説明欄」なのはサーバーが V_before を使うときだけ)')
# 6.2 に新しい 37(SUPERSEDE r82 #1・decided #10)
RD_37_X = '\n\n### 6.3 11/3 のあとでもよい(小さいもの)'
RD_37_Y = ('\n| 37 | **ユーザーに伝えること**(10/28 fix8 docs 仕上げ。返事が無ければこのまま): ① approvals 3 の (c) にもう 1 つの知らせ: 両方 ON の日、'
           + SAFE_STALL + 'と、' + SAFE_MORE + '(' + SAFE_REAR + '。' + SAFE_REAL + '。' + SAFE_REV + ')。11/3 のコードは変えない'
           '(main session の決め)。守りは手順書 7 章の「1 m 以上下がる → すぐ人側」。直し方の案は 11/3 のあと(`fix8/proposal_safety_after1103.md` '
           '10 章)。`fix3/approvals.md` 3 の「困ること(ON)」・14 ② 回 3 の判定: OFF の発進がカメラの止まりで比べから外れたか、OFF の本の発進が '
           'R1 に届かず比べに 1 本も残らないと「満たさない」にした(10/28 fix8 tools r2。10/26 夜 fix3 の「R1 に届かなかった OFF は名前を出す'
           'だけ」の決めを変えた = main session が了承。回 4 が OFF になりやすい安全側。`fix3/approvals.md` 8)③ ⑤ の出力の「`--skip` で外して'
           '読み直せる」の文は、当たった・空転した・記録が欠けた ON の本には 11/3 は使わない(手順書を正とする = main session の決め。手順書 8.1 ⑤) '
           '| main session → ユーザー | 11/2(返事が無ければこのまま) |' + RD_37_X)
# 6.3 に新しい 38(SUPERSEDE s1 #4)
RD_38_X = '`fix8/docs_sync_r1.md`。どれも 11/3 のあとでよいと係がみた |\n'
RD_38_Y = (RD_38_X + '| 38 | 反証 8-2 回目で残したこと(安全: 両方 ON の日の後退の終わりのカメラの止まりの形は 11/3 のあとに直す = '
           '`fix8/proposal_safety_after1103.md` 10 章(案の文だけ。写しでは比べていない)。最初のリカバリーの「おかしな動き」の ① 3 秒・② 2.5 m は'
           '偽物の ESC と本物の記録 6 回で決めた = 11/3 の最初のリカバリーの記録で決め直す。画面: 実機のスマホのフォントで ■ を見ていない・'
           '`\\p{Nd}` の読めない Firefox 72〜77(2020 年のパソコン版)では app.js が読めない。夜の道具: 合計 0.05〜0.22 秒の短い止まりの続きは注意の'
           '行だけ・`--skip` した OFF の本は「外れた OFF」に数えない。⑤ の出力の `--skip` の文は道具の形のまま(11/3 は手順書が正)。文書: 反証 8 回目'
           'は出切っていない) | `fix8/README.md` 3 章・`fix8/fix_ui_r2.md` 10 章・`fix8/fix_tools_r2.md` 7 章・`fix8/fix_docs_r2.md` 6 章・'
           '`fix8/docs_final.md`。どれも 11/3 のあとでよいと係がみた |\n')
# 7 章の終わり(前の係の new の外)
RD_7_OLD = '(0〜6 章の「本物の記録 235 本」は 235 の場面 = 9 章 9.7) |\n'
RD_7 = (RD_7_OLD + '| `fix8/fix_ui_r2.md`・`fix8/fix_tools_r2.md`・`fix8/fix_docs_r2.md`・`fix8/docs_final.md` | 反証 8-2 回目の直し(10/28。画面 2 と 8-1 から'
        '渡された試験の古い期待 1・夜の道具 2・文書 5。安全 1 は文書だけ = この仕上げ)と、その文書の仕上げ(手順書に足す文・安全の知らせ・'
        '⑤ の `--skip` の文との食い違い)。`fix8/proposal_safety_after1103.md` には 10/28 に 10 章(後退の終わりのカメラの止まりの案。案の文だけ)が'
        '足された |\n| **`fix8/README.md`** | **反証 8 回目(2 巡・8-1 20 件・8-2 10 件)の直しのまとめ**(10/28、fix8 係 docs の仕上げ)。main session の'
        '決めの実装(`decided_*.md`)・巡ごとの指摘と打ち消された数・残すこと・了承・ユーザーに伝えること・11/3 のあとの安全の案・再起動が要るか・'
        '今のファイルの sha256 |\n')

# ==================================================================================================== approvals
AP_LOG_OLD = '- **10/28 11:4x fix8 係 docs 直す 8-2 回目**(控え `approvals.md.bak-20261028-fix8-docs-r2`。'
AP_LOG = ('- **' + STAMP + ' fix8 係 docs 仕上げ(反証 8 回目の終わり)**(控え `approvals.md' + BAK + '`。`fix8/docs_final.md`・`fix8/README.md`): 3 の'
          '「困ること(ON)」と 14 に、両方 ON の日の後退の終わりのカメラの止まりの形(反証 8-2 回目 ' + SAFE_ID + '。11/3 のコードは変えない = 知らせ)'
          '/ 3 の「途中で OFF にした日」の決まり・8 の表に、夜の道具の 8-2 回目の直し(外れた OFF は満たさない・短い止まりの続き。R1 に届かなかった'
          ' OFF の扱いは 10/26 夜 fix3 の決めを変えた = main session が了承)/ 14 の「後ろ 1.2・1.35 m の物」の行に今の後退の幅 / 0 章の 3・8・14 に'
          '印。答え・おすすめは変えていない。\n' + AP_LOG_OLD)
# 0 章の 3(SUPERSEDE decided #2・s1 #1)
AP_03_X = '11/3 のコードは変えない。返事が無ければ (c) のまま) | 決まり |'
AP_03_Y = ('11/3 のコードは変えない。返事が無ければ (c) のまま)。**もう 1 つ(10/28 fix8 docs 仕上げ)**: 両方 ON の日の後退の終わりのカメラの止まり'
           '(3 の「困ること(ON)」の終わり・14 の表の下の新しい知らせ。11/3 のコードは変えない。返事が無ければ (c) のまま) | 決まり |')
# 0 章の 8(SUPERSEDE decided #3)
AP_08_X = '③ はこのまま) | 11/2 |'
AP_08_Y = ('③ はこのまま。10/28 fix8 tools r2 で、比べから外れた OFF があれば満たさない = R1 に届かなかった OFF の 10/26 夜 fix3 の決めを変えた'
           '(main session が了承。8 の表の新しい行)) | 11/2 |')
# 0 章の 14(SUPERSEDE final #1)
AP_014_X = '**このまま**(11/3 は今のコード。返事が無ければこのまま) | 11/2 |'
AP_014_Y = ('**このまま**(11/3 は今のコード。返事が無ければこのまま。10/28 に表の下へ知らせを 1 つ足した = 両方 ON の日の後退の終わりのカメラの'
            '止まり。反証 8-2 回目) | 11/2 |')
# 3「困ること(ON)」(SUPERSEDE decided #8)
AP_3C_X = '動く・空転は同じ。`fix8/decided_safety.md` 1.2・1.6 節) |'
AP_3C_Y = ('動く・空転は同じ。`fix8/decided_safety.md` 1.2・1.6 節)。**(10/28 知らせ。反証 8-2 回目 ' + SAFE_ID + ')両方 ON の日、'
           + SAFE_STALL + 'と、' + SAFE_MORE + '**(止まりの間は前に書いた後退側が中立止めの中立まで出続け(ON・OFF とも)、明けたあと ON は'
           '惰性(\'coast\')を中立止めの始めから 0.25 秒続けてから前へ出る。OFF は手順を戻し、モデルの前進が後ろへの惰性を止める。閉じたループの'
           '偽物の ESC の模型 24 通り × カメラの位相 2 × 処理の時間 2 = 96 設定で、止まりが後退の終わりのフレームか 1 つ前から 3〜8 フレームのとき、'
           '96 設定とも ON が OFF と乱れの無い ON の両方より多く下がる。**' + SAFE_REAR + '**(1.0 m では ON のほうが約 +0.5 m/s 強い)= 下の 10/23 の'
           '形の「約 1 m より遠い物には ON のほうが遅く当たるか当たらない」とは別の形。**' + SAFE_REAL.replace('本物の 10/10〜10/24 の', '本物の '
           '10/10〜10/24(59 本)の') + '**(AI 側のフレームの間が 0.12〜0.31 秒になるのは 10/23・10/24 で 0.21%)。**' + SAFE_REV + '**(後退の間に'
           ' 0.8 秒中立を出すモデル)。止まりが 0.37 秒(10 フレーム)より長いと ON = OFF(知られている「中立止めの間の惰性」の形)。見張りだけ ON'
           '(残す OFF)は OFF と同じ値。11/3 のコードは変えない(10/28 main session の決め)。' + SAFE_GUARD + '(この上乗せの 0.3 秒ほどには人は'
           '間に合わない)。直し方の案は 11/3 のあと(`fix8/proposal_safety_after1103.md` 10 章)。14 の表の下の知らせ。(c) を変えたいときは一言(返事が無ければ'
           ' (c) のまま)) |')
# 3「途中で OFF にした日」の決まり(SUPERSEDE decided #9・s1 #3 を 2 か所)
AP_3D_X = '両方 ON でも止まらない形も同じに外す)'
AP_3D_Y = (AP_3D_X + '。ただし、外した OFF が 2 本以上・いちばん速い OFF の 1 本だと、残りの OFF の中央値で「使ってよい」の側に動いた(反証 8-2 回目 '
           'w2-tools-r8-2-off-few-left。8-1 の前から R1 に届かなかった OFF でも)ので、**OFF の発進がカメラの止まりで外れたか、OFF の本の発進が '
           'R1 に届かず比べに 1 本も残らなければ「→ OFF の発進が比べから外れた N 本」= 満たさない(checks の off_all_compared)**(10/28 fix8 tools r2。'
           'R1 に届かなかった OFF の扱いは 10/26 夜 fix3 r1-rec3-1 の決め(名前を出すだけで答えは変えない)を変えた = main session が了承)')
AP_3S_X = '「使ってよい」が境目から 0.1 秒以内なら main session が動画で確かめて決める)'
AP_3S_Y = (AP_3S_X + '。1 回の間は短くても止まりが続き drive の中立の合計が 0.22 秒より長い発進も同じに外す(「→ うち短い止まりが続いた発進 N 本」。'
           '中立 2 回以上・合計 0.05〜0.22 秒は「注意: 発進から R1 の起点までに短い止まりが続いた発進 N 本」。本物の 10/10〜10/24 の AI 側の 3 秒の'
           '窓 55,838 で合計の最大 0.200 秒・R1 に届いた発進 38 で 0.100 秒。10/28 fix8 tools r2)')
# 8 の表の「カメラが止まった発進」の行(SUPERSEDE s1 #5 を 2 か所)
AP_8R_X = 'OFF が全部外れたら満たさない(off_no_stall)。'
AP_8R_Y = 'OFF の発進が 1 本でも外れたら満たさない(off_all_compared。全部なら off_no_stall も。10/28 fix8 tools r2)。'
AP_8E_X = '0.15〜0.22 秒の短い止まりは比べに入れて注意の行に名前 |'
AP_8E_Y = ('0.15〜0.22 秒の短い止まりは比べに入れて注意の行に名前。短い止まりが続いて drive の中立の合計が 0.22 秒より長い発進も外す(合計 '
           '0.05〜0.22 秒は注意の行。10/28 fix8 tools r2) |')
# 8 の表に新しい行(SUPERSEDE decided #11)
AP_8N_X = 'どれも前から数えない発進 = 判定は変わらない) |'
AP_8N_Y = (AP_8N_X + '\n| OFF の本の発進が R1 に届かず比べに 1 本も残らない(10/28 fix8 tools r2) | 満たさない(off_all_compared)。同じ本の走り直し・'
           '同じ block の別の記録が比べに入っていれば数えない。OFF が全部届かない日も満たさない(「ON − OFF: 起点まで — 秒 … → 満たさない」)。'
           '**10/26 夜 fix3 r1-rec3-1 の決め(R1 に届かなかった OFF は名前を出すだけで答えは変えない)を変えた**(main session が了承。回 4 が OFF に'
           'なりやすい安全側) |')
# 14 の「後ろ 1.2・1.35 m の物」と新しい行(SUPERSEDE final #4)
AP_14_X = ('(最大 1.83 m/s。直す前の同じ場面の当たりの最大と同じ) | 10 のおすすめ (a)(11/3 は変えない。11/3 のリカバリーの記録で決め直す)'
           'のまま |')
AP_14_Y = ('(最大 1.83 m/s。直す前の同じ場面の当たりの最大と同じ)。(10/28 fix8 docs 仕上げ: 今のコードの閉じたループ(偽物の ESC の模型 24 通り × '
           '24 場面)では、見張り ON の最初のリカバリーは 1.02〜1.97 m 下がる(同じ場面の OFF は 1.17〜2.91 m)=「1.1〜1.4 m」は fix7 の閉じた'
           'ループの幅。3 の「なぜ」・`fix8/fix_docs_r2.md` 1.1) | 10 のおすすめ (a)(11/3 は変えない。11/3 のリカバリーの記録で決め直す)のまま |\n'
           '\n- **(10/28 知らせ。反証 8-2 回目 ' + SAFE_ID + '。fix8 docs 仕上げ。上の表は fix7 で残した 4 つのまま)両方 ON の日の、後退の終わりの'
           'カメラの止まり**: 両方 ON の日、後退(\'reverse\')を終えるはずのフレームからカメラが 3〜8 フレーム(0.1〜0.3 秒。0.5 秒より短い)'
           '止まると、明けたあと ON は惰性を中立止めの始めから 0.25 秒続けるので、OFF より最大 +0.6 m(乱れの無いときより最大 +0.55 m)多く下がる。'
           + SAFE_REAR + '。見張りだけ ON(残す OFF)は OFF と同じ値 = 残す(`rec_hold_keep`)の形。本物の後退の終わり 11 回で 0 回・1 回の'
           'リカバリーあたり約 0.4%。' + SAFE_REV + '。3 の「困ること(ON)」の終わり。直し方の案は `fix8/proposal_safety_after1103.md` 10 章'
           '(11/3 のあと)。**おすすめ: このまま**(11/3 は今のコード。守りは手順書 7 章の「1 m 以上下がる → すぐ人側」。返事が無ければこのまま)')

ALL = [
    (RB, [(RB_LOG_OLD, RB_LOG), (RB_31_X, RB_31_Y), (RB_46_X, RB_46_Y), (RB_48_OLD, RB_48), (RB_6_OLD, RB_6), (RB_3X_X, RB_3X_Y),
          (RB_3E_X, RB_3E_Y), (RB_7_X, RB_7_Y), (RB_7F_X, RB_7F_Y), (RB_8O_X, RB_8O_Y), (RB_8N_X, RB_8N_Y), (RB_8S_OLD, RB_8S)]),
    (TODO, [(TD_TITLE_X, TD_TITLE_Y), (TD_LOG_OLD, TD_LOG), (TD_23_X, TD_23_Y), (TD_48_X, TD_48_Y), (TD_216_X, TD_216_Y),
            (TD_3S_X, TD_3S_Y)]),
    (RD, [(RD_LOG_OLD, RD_LOG), (RD_6H_X, RD_6H_Y), (RD_11_X, RD_11_Y), (RD_37_X, RD_37_Y), (RD_38_X, RD_38_Y), (RD_7_OLD, RD_7)]),
    (AP, [(AP_LOG_OLD, AP_LOG), (AP_03_X, AP_03_Y), (AP_08_X, AP_08_Y), (AP_014_X, AP_014_Y), (AP_3C_X, AP_3C_Y), (AP_3D_X, AP_3D_Y),
          (AP_3S_X, AP_3S_Y), (AP_8R_X, AP_8R_Y), (AP_8E_X, AP_8E_Y), (AP_8N_X, AP_8N_Y), (AP_14_X, AP_14_Y)]),
]
# 元(報告の文・指摘)→ 直し(文書, ALL の何番目)
SOURCES = {
    'ui r2 11 章(app.js 11:02・style.css 10:15・電圧の源・■ とトースト)': [(RB, 1), (RB, 2), (RB, 3), (RB, 8), (TODO, 5), (RD, 2)],
    'tools r2 10 章(off_all_compared・launch_neutral・R1 に届かない OFF)': [(RB, 4), (RB, 5), (RB, 6), (RB, 9), (RB, 10), (AP, 2), (AP, 5),
                                                                       (AP, 6), (AP, 7), (AP, 8), (AP, 9)],
    SAFE_ID: [(RB, 7), (TODO, 2), (TODO, 3), (TODO, 4), (RD, 3), (AP, 1), (AP, 3), (AP, 4), (AP, 10)],
    'skip-contradiction(⑤ の --skip の文と手順書。11/3 は手順書が正)': [(RB, 11)],
    'docs r2 の残したこと・反証 8 回目のまとめ(README 6・7 章)': [(RD, 1), (RD, 4), (RD, 5)],
}
LOGS = [(RB, 0), (TODO, 0), (TODO, 1), (RD, 0), (AP, 0)]

# 前の係の文(その台本の ALL の (文書, 何番目) の new の今の形)の中を書き換えた所: (台本, 文書, 何番目, 前の文 X, 今の文 Y)
SUPERSEDE = [
    ('s1', RB, 3, RB_31_X, RB_31_Y),            # 3.1 の app.js の行(同期 1 の 3.1 の足しの中)
    ('decided', RB, 2, RB_31_X, RB_31_Y),       # 同じ所(docs-decided の 3.1 の足しの中でもある)
    ('r81', RB, 3, RB_46_X, RB_46_Y),           # 4 章 6 の電圧の源
    ('decided', RB, 3, RB_3X_X, RB_3X_Y),       # 3 章「わざと OFF にした日」の (a) の文
    ('r1', RB, 2, RB_3X_X, RB_3X_Y),            # 同じ所(fix7 docs r1 の足しの中でもある)
    ('r2', RB, 4, RB_3E_X, RB_3E_Y),            # 3 章「わざと OFF にした日」の終わり
    ('s1', RB, 4, RB_7_X, RB_7_Y),              # 7 章「1 m 以上下がる」
    ('s1', RB, 7, RB_7F_X, RB_7F_Y),            # 7 章「Failed to fetch」の app.js の時刻
    ('s1', RB, 11, RB_8O_X, RB_8O_Y),           # 8.1 ⑤ の OFF が全部外れたら
    ('s1', RB, 11, RB_8N_X, RB_8N_Y),           # 8.1 ⑤ の注意の行のあと
    ('r82', TODO, 0, TD_TITLE_X, TD_TITLE_Y),   # todo の題(fix8 docs r2 の書き換え = r82 の ALL の 0 番目)
    ('r81', TODO, 0, TD_TITLE_X, TD_TITLE_Y),   # 同じ題(fix8 docs r1 の ALL の 0 番目の今の形)
    ('s1', TODO, 0, TD_TITLE_X, TD_TITLE_Y),    # 同じ題(同期 1 の ALL の 0 番目の今の形)
    ('r82', TODO, 2, TD_23_X, TD_23_Y),         # 1 章「11/2 まで」の知らせのあと
    ('r1', TODO, 2, TD_23_X, TD_23_Y),          # 同じ所(fix7 docs r1 の足しの中でもある)
    ('r82', TODO, 3, TD_48_X, TD_48_Y),         # 2 章 10 の知らせのあと
    ('final', TODO, 2, TD_216_X, TD_216_Y),     # 2 章 16
    ('decided', TODO, 8, TD_216_X, TD_216_Y),   # 同じ所
    ('s1', TODO, 2, TD_3S_X, TD_3S_Y),          # 3 章の画面の行
    ('decided', RD, 4, RD_6H_X, RD_6H_Y),       # README 6 章の頭
    ('s1', RD, 2, RD_11_X, RD_11_Y),            # README 6.2 の 11
    ('r82', RD, 1, RD_37_X, RD_37_Y),           # README 6.2 の新しい 37(36 のあと)
    ('decided', RD, 10, RD_37_X, RD_37_Y),      # 同じ所
    ('s1', RD, 4, RD_38_X, RD_38_Y),            # README 6.3 の新しい 38(35 のあと)
    ('decided', AP, 2, AP_03_X, AP_03_Y),       # approvals 0 章の 3
    ('s1', AP, 1, AP_03_X, AP_03_Y),            # 同じ所
    ('decided', AP, 3, AP_08_X, AP_08_Y),       # approvals 0 章の 8
    ('final', AP, 1, AP_014_X, AP_014_Y),       # approvals 0 章の 14
    ('decided', AP, 8, AP_3C_X, AP_3C_Y),       # approvals 3「困ること(ON)」
    ('decided', AP, 9, AP_3D_X, AP_3D_Y),       # approvals 3「途中で OFF にした日」の決まり: 外れた OFF
    ('s1', AP, 3, AP_3D_X, AP_3D_Y),
    ('decided', AP, 9, AP_3S_X, AP_3S_Y),       # 同じ行: 短い止まりの続き
    ('s1', AP, 3, AP_3S_X, AP_3S_Y),
    ('s1', AP, 5, AP_8R_X, AP_8R_Y),            # approvals 8 の表の「カメラが止まった発進」の行
    ('s1', AP, 5, AP_8E_X, AP_8E_Y),
    ('decided', AP, 11, AP_8N_X, AP_8N_Y),      # approvals 8 の表の新しい行
    ('final', AP, 4, AP_14_X, AP_14_Y),         # approvals 14
]
# 前の文を消す書き換え(X の文が Y に残らない)
REMOVES = {(RB, RB_31_X), (RB, RB_3X_X), (RB, RB_7F_X), (TODO, TD_TITLE_X), (TODO, TD_3S_X), (RD, RD_11_X), (AP, AP_8R_X)}

# ==================================================================================================== 案の報告の 10 章(末尾に足すだけ)
PROPOSAL_CH10 = r"""
## 10. 追記(10/28 fix8 係 docs 仕上げ): 両方 ON の日の後退の終わりのカメラの止まりの案(指摘 rs8-2-reverse-end-stall-overrun。案の文だけ)

- この章は、反証 8-2 回目の指摘 rs8-2-reverse-end-stall-overrun(打ち消しの票 A・B とも real・minor)を受けて、係 docs の仕上げが足した(1〜9 章は変えていない)。報告 `fix8/docs_final.md`。
- **main session の決め(10/28 05:xx・11:3x): 11/3 の安全のコードは変えない**。この章は直し方の案の文だけ(**写しは作っていない・閉じたループで比べていない**)。入れるかは 11/3 のあと、係 safety が写しで比べてから main session とユーザーが決める(入れるにはコード・試験・jetformer-web の再起動が要る)。どの案も見張り・残すが両方 ON(`rec_back_guard` = 1・`rec_hold_keep` = 1)の日だけ効く形にする(OFF・見張りだけ ON は今とビット単位で同じ)。
- 指摘と票(台本と出力): `experiments/20261026_fix/fix8/review_safety_r2/findings.json`(`h12.py`・`trace12.py`・`exp12.py`・`ana12ov.py`・`ovcheck12.py`・`realov12.py` と `out/`)・`refute_safety_r2_A/verdicts.json`(`rA.py`・`sweep_rA.py`・`ana2_rA.py` → `out/ana2_s1.txt`〜`ana2_s4.txt`)・`refute_safety_r2_B/verdicts.json`(`retrace_B.py`・`real_cam_B.py`・`real_B.py` と `out/`)。

### 10.1 今の形(残り)

- 両方 ON で、後退('reverse')を終えるはずのフレームからカメラが 3〜8 フレーム(フレームの間 0.13〜0.3 秒。`HOLD_KEEP_MAX_S` 0.5 秒より短い)止まると: drive はフレームが 0.1 秒来ないと中立を書くので、その間は前に書いた後退側が中立止めの中立まで最大 0.1 秒出続ける(ON・OFF とも)。明けたあと、ON(残した手順)は `Recovery._after_hold`(`jetformer/assist.py` の 'reverse' の枝。`if self.backed_s >= rev_s: self._go('coast', now); self.t_phase = now - hold_s`)で「後退は中立止めの始めで出し終えていた」とみて 'coast' に移り、惰性を中立止めの始めから 0.25 秒続けてから escape(前進)。OFF は中立止めで手順を戻し、明けたフレームからモデルの前進が後ろへの惰性を止める。
- 指摘の例(`trace12.py`。回 1 の形・場面 rev(前の壁 2.0 m・モデルはずっと前進 120 µs)・ESC の模型 human・後退の終わりのフレームから 3 フレーム): 下がった距離 ON 1.678 m・最高 2.65 m/s / OFF 1.320 m / 乱れなし 1.320 m・2.30 m/s。止まった壁から後ろ 1.2 m の物: ON 2.35・OFF 0.29・乱れなし 1.42 m/s、1.4 m: ON 2.09・OFF と乱れなしは当たらない、1.6 m: ON だけ 1.13 m/s(票 B の `retrace_B.py` も同じ数)。
- 広さ(票 A。ESC の模型 24 × カメラの位相 2 × 処理の時間 2 = 96 設定・前進 120 µs・`out/ana2_s1.txt`): 止まりが後退の終わりのフレーム(k 0)か 1 つ前(k −1)から 3〜8 フレームのとき、96 / 96 設定で ON が「乱れの無い ON」と「同じ止まりの OFF」の両方より多く下がる。ON − 乱れの無い ON は k 0 で +0.127〜+0.547 m(中央 +0.238)・k −1 で +0.060〜+0.293 m、ON − OFF は最大 +0.605 m(k 0・3 フレーム)。10 フレーム(0.37 秒)では ON = OFF(知られている「中立止めの間の惰性」の形。`fix6/README.md` 3 章)。2 フレーム(中立止めにならない 48 / 96)は ON = OFF。止まりの始めを手順の全部のフレームにずらすと、ON が悪いのは k −1・0 だけ(`ana2_s2.txt`)。
- 逆の形(票 A・`out/ana2_s4.txt`): 後退の間に 0.8 秒中立を出すモデル(lag0.8)では OFF のほうが多く下がる(OFF − 乱れなし +0.27〜+0.47 m・ON − 乱れなし +0.12〜+0.39 m)。ずっと中立なら OFF は +1.1〜+5.3 m。0.5 秒で前進するモデル(lag0.5)は前進 120 µs と同じ向き(ON が悪い)。本物の 10/23・10/24 のリカバリーの後退の終わりのモデルの値は前進側(No.6 約 −123 µs・160847 −100〜−135 µs)= 指摘の向き(OFF のほうが短い)に近い。
- 本物(指摘の `out/realov12.json`・票 A・B): 10/10〜10/24 の 59 本で後退の終わり 11 回にカメラの止まりは 0 回(後退側の行のあとの 0.1 秒より長い間 4 つは、どれも人の取り戻し 2.4〜11 秒)。AI 側の処理したフレームの間が 0.12〜0.31 秒になるのは 10/23・10/24 で 98 / 45,861 = 0.21%、カメラが来なかった間は 10/10〜10/24 で 121 / 67,620 = 0.18%(どれも前へ 1 m/s より速く走っている間)。効くのは後退の最後の 1〜2 フレームなので、1 回のリカバリーあたり約 0.4% の見込み(票 B は 11/3 の 1 日あたり約 0.2〜0.5%)。ON だけ当たるのは後ろ 1.3〜1.7 m の物(票 B。0.8 m では OFF も同じに当たる・1.0 m では ON が約 +0.5 m/s 強い)。
- どのスイッチ・どの版か(指摘の `out/ovcheck12.json`): 見張りだけ ON(`rec_hold_keep` 0)は OFF と同じ値 = 残す(`rec_hold_keep`)で手順を残す形。fix7 r2・fix7 の前の両方 ON も今と同じ値、fix6・fix5 の前は 1.68〜1.85 m でもっと多かった(fix6 s3-b で減って残った)。fix6 s3-b の「'coast' の途中の中立止め … keep 0 と同じ(+0.001 m 以内)」は、止まりが後退の終わりのフレームに重なる場面を見ていなかった。

### 10.2 案(写しは作っていない)

| 案 | 中身(両方 ON の日だけ) | 良い所 | 気をつけること |
|---|---|---|---|
| **a1 超えた分だけ惰性を短くする**(比べる順の 1 番目) | `_after_hold` の 'reverse' の枝で、明けて `backed_s` が `rev_s` を超えていたら、超えた分(止まりの間の後退側・惰性で下がった秒)だけ惰性を短くする: `self.t_phase = now - hold_s - (self.backed_s - rev_s)` | 変える所が 1 行。変わるのは 'reverse' で明けて rev_s を超えていた形だけ。惰性の長さの決め方(0.25 秒)はそのまま | 超えた分は「後ろへ動いていたサンプル」の秒(車両の値が来ない間の見積もり `hold_rev_s` を含む)。超えた分が 0.25 秒より長いと惰性 0 = すぐ escape |
| a2 escape に進める | 超えた分が 1 フレーム(1/30 秒)より長ければ 'coast' を飛ばして escape(前進が後ろへの惰性を止める) | 指摘の場面の惰性をまるごと消す(OFF のモデルの前進と同じ向き) | 後ろへ速く動いている車にすぐ前進が掛かる(ESC がブレーキとみるか・escape の間の前の壁まで)。9 章の qC と同じ数え方で escape の前の壁までの最短を見る |
| b drive で中立をすぐ書く | 残した手順が 'reverse' のときのカメラの止まりは、`cam_evt.wait(0.1)` を待たずに中立を書く | 止まりの間の後退側(ON・OFF 共通の最大 0.1 秒)も短くなる | `jetformer/drive.py` を変える(中立止めの決まり・ほかの段に触れる)。OFF をビット単位で同じに保つには `rec_hold_keep` = 1 の枝だけにする |

- 指摘の案(`findings.json` の fix)のとおり。どれも OFF はビット単位で同じにする形。効き目・新しく悪くなる所は比べていないので分からない(9 章の qC・6 章の pB と同じ台で比べる)。

### 10.3 比べ方(11/3 のあとに係 safety が。指摘の案の試験)

- ESC の模型 18(と票 A の 24)× 場面 rev / react × 止まりの始め = 後退の終わりの 1 フレーム前〜後退の終わり × 3〜8 フレームで、ON の下がった距離 ≤ 乱れなし + 0.05 m・後ろ 1.0〜1.6 m の物の当たり ≤ 乱れなし + 0.1 m/s。乱れなし・止まりが 0.5 秒より長いときは今と同じ値。全部 OFF・見張りだけ ON はビット単位で同じ。
- 逆の形(lag0.8・ずっと中立のモデル)で ON が今より悪くならないか。escape の間の前の壁までの最短(9.4 と同じ数え方)。
- 9 章の qC・6 章の pB と一緒に入れるなら、入れた写しで 9.3 の台(反証の係の場面の組 4,542・本物の記録 59 本)と前からの試験(Recovery を見る 11 ファイル + `tests/test_rec_fix8_safety_r1.py`)を流し直す。

### 10.4 本物の ESC で確かめること(11/3 の記録で)

- リカバリーの後退の終わりのフレームのまわりに、AI 側のフレームの間が 0.1 秒より長い所があるか・そのときの下がった距離(止まった所から)。
- リカバリーの後退の終わりのモデルの値(前進側か中立か = 逆の形の見込み)。

### 10.5 おすすめ(仮。比べる前)

- **11/3 は今のまま**(main session の決め)。守りは手順書 7 章の「1 m 以上下がる → すぐ人側」(この上乗せの 0.3 秒ほどには人は間に合わない)。
- 比べるなら **a1** から(変える所がいちばん小さい・惰性の長さの決め方を変えない)。a1 で足りなければ a2、b は drive.py に触るので最後。どれも両方 ON の日だけ。比べるまでは、どれもすすめとは言えない。
"""


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prev_mods():
    return {k: _load('docs_prev_%s_from_fin' % k, p) for k, p in PREV.items()}


def prev_new(mods, k, f, i):
    """前の台本 k の (文書 f, 何番目 i) の new の、この組の前の今の形(fix8 docs r2 の prev_new のあと、fix8 docs r2 の書き換え)"""
    new = dict(mods[k].ALL)[f][i][1]
    if k in ('r1', 'r2', 'final'):
        new = mods['decided'].effective(k, f, i, new)
    if k in ('final', 'decided'):
        new = mods['r81'].effective(k, f, i, new)
    if k not in ('s1', 'r82'):
        new = mods['s1'].effective(k, f, i, new)
    if k != 'r82':
        new = mods['r82'].effective(k, f, i, new)
    return new


def effective(script, f, i, new):
    """前の台本 script の (文書 f, 何番目 i) の new(あとの台本の書き換えのあとの形)の、この組が書き換えたあとの形"""
    for s, ff, ii, x, y in SUPERSEDE:
        if (s, ff, ii) == (script, f, i):
            assert new.count(x) == 1, (s, ff, ii, x[:30])
            new = new.replace(x, y, 1)
    return new


def revert(texts):
    """この組の SUPERSEDE の書き換えを戻した文書(texts にある文書だけ。足しただけの直しはそのまま)"""
    out, seen = dict(texts), set()
    for _, f, _, x, y in reversed(SUPERSEDE):
        if (f, x, y) in seen:
            continue
        seen.add((f, x, y))
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


def proposal_state(raw):
    """案の報告: 'before'(10/28 07:46 の版のまま)/ 'done'(その版 + 10 章)/ 'other'"""
    if hashlib.sha256(raw).hexdigest() == PROPOSAL_BEFORE_SHA:
        return 'before'
    if (len(raw) > PROPOSAL_BEFORE_LEN and hashlib.sha256(raw[:PROPOSAL_BEFORE_LEN]).hexdigest() == PROPOSAL_BEFORE_SHA
            and raw[PROPOSAL_BEFORE_LEN:] == PROPOSAL_CH10.encode('utf-8')):
        return 'done'
    return 'other'


def read_all(root=REPORTS, files=ALL7):
    return {f: (Path(root) / f).read_text(encoding='utf-8') for f in files}


def prior_news_problems(texts, mods=None):
    """書き換えを戻した文書(revert)で、前の台本の new(今の形)がどれもちょうど 1 回か・今の文書で書き換えたあとの new がちょうど 1 回か・
    SUPERSEDE の X が前の new の中にちょうど 1 回か。戻り値は問題の一覧"""
    mods = mods or prev_mods()
    bad = []
    for s, f, i, x, y in SUPERSEDE:
        if prev_new(mods, s, f, i).count(x) != 1:
            bad.append(('SUPERSEDE の X が前の new の中に 1 回でない', s, f, i))
    old = revert(texts)
    for k, m in mods.items():
        for f, edits in m.ALL:
            if f not in old:
                continue
            for i in range(len(edits)):
                n = prev_new(mods, k, f, i)
                if old[f].count(n) != 1:
                    bad.append(('前の new が 1 回でない(戻した文書)', k, f, i, old[f].count(n)))
                ne = effective(k, f, i, n)
                if texts[f].count(ne) != 1:
                    bad.append(('前の new(書き換えたあと)が 1 回でない(今の文書)', k, f, i, texts[f].count(ne)))
    return bad


def live_checks_problems(texts, tmp_root=None):
    """前の 7 組の試験の確かめ(test_live_documents と同じ CHECKS)を今の文書で。戻り値は問題の一覧"""
    sys.path.insert(0, str(JF))
    sys.path.insert(0, str(JF / 'tools'))
    sys.path.insert(0, str(JF / 'tests'))
    bad = []
    ts = {k: _load('t_%s_from_fin' % k, TESTS[k]) for k in ('r1', 'r2', 'final', 'decided', 'r81', 's1', 'r82')}
    with tempfile.TemporaryDirectory(prefix='docs_fix8_final_', dir=tmp_root) as tmp:
        for k, t in ts.items():
            for name, fn in sorted(t.CHECKS.items()):
                ps = fn(texts, Path(tmp) / name) if k == 'r2' else fn(texts)
                bad += [('%s %s(今の文書)' % (k, name), p) for p in ps]
    bad += [('final h5(今の文書)', p) for p in ts['final'].h5_problems(texts[F7])]
    return bad


def earlier_problems(texts, tmp_root=None):
    """前の係の直しと試験の確かめが崩れていないか(texts = 直したあとの 7 本。戻り値は問題の一覧):
    - prior_news_problems(戻した文書で前の 7 つの台本の new がちょうど 1 回・今の文書で書き換えたあとの new がちょうど 1 回)
    - この組の書き換えを戻した文書で、fix8 docs r2 の 40 個の直しが「済」・SUPERSEDE が入っている・その台本の earlier_problems(同期 1 →
      fix8 docs r1 → docs-decided → fix7 docs の仕上げ・r2・r1 の直しと、前の係の試験の確かめ f〜m)が空
    - 今の文書で、前の 7 組の試験の確かめ(f・g・h・i・j・k・m)と、この組の試験の確かめ(試験のファイルがあれば)が空・SUPERSEDE が入っている"""
    mods = prev_mods()
    bad = [('news',) + tuple(x) for x in prior_news_problems(texts, mods)]
    r82 = mods['r82']
    base = revert(texts)
    _, rep = r82.apply_text(base)
    bad += [('r82', f, o[:30], st) for f, o, st in rep if st != 'done']
    bad += [('r82 superseded(戻した文書)',) + tuple(x) for x in r82.superseded_present(base)]
    bad += [('r82.earlier_problems(戻した文書)',) + tuple(x) for x in r82.earlier_problems(base, tmp_root)]
    bad += live_checks_problems(texts, tmp_root)
    if TESTS['fin'].exists():
        t8 = _load('t_fin_from_fin', TESTS['fin'])
        for name, fn in sorted(t8.CHECKS.items()):
            bad += [('fin ' + name, p) for p in fn(texts)]
    bad += [('superseded',) + tuple(x) for x in superseded_present(texts)]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--quick', action='store_true', help='前の係の earlier_problems の鎖を流さない(new と今の文書の確かめだけ)')
    a = ap.parse_args(argv)
    tmp_root = E / 'fix8' / 'tmp_docs_final'
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
    praw = (REPORTS / PROPOSAL).read_bytes()
    pst = proposal_state(praw)
    print('案の報告:', pst)
    if pst == 'other':
        print('案の報告が 07:46 の版でも、その版 + 10 章でもない → 書かない')
        return 5
    if a.quick:
        prob = [('news',) + tuple(x) for x in prior_news_problems(new)] + live_checks_problems(new, str(tmp_root))
    else:
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
    if a.quick:
        print('--quick では書かない')
        return 6
    for f in FILES:
        if new[f] == texts[f]:
            continue
        b = REPORTS / (f + BAK)
        if not b.exists():
            shutil.copy2(REPORTS / f, b)
        (REPORTS / f).write_text(new[f], encoding='utf-8')
        print('書いた:', f, '(控え %s)' % b.name)
    if pst == 'before':
        b = REPORTS / (PROPOSAL + BAK)
        if not b.exists():
            shutil.copy2(REPORTS / PROPOSAL, b)
        with open(REPORTS / PROPOSAL, 'ab') as fh:
            fh.write(PROPOSAL_CH10.encode('utf-8'))
        print('書いた:', PROPOSAL, '(10 章を末尾に。控え %s)' % b.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
