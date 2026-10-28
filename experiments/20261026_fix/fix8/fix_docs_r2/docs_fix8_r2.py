#!/usr/bin/env python3
"""fix8 係 docs 直す 8-2 回目(2026-10-28): 反証 8-2 回目で確かめて残った文書の指摘 5 件(係 docs 4・係 rehearsal 1)を、ユーザーが
読む文書(手順書・todo・README・fix3/approvals.md)に入れる。コードは変えない。

- f8r2-docs-paper-trial-skip-drops-faults・rehearsal-r2-unmarked-rerun-skip-drops-hit(同じ根): **当たった・空転した・記録が欠けた
  ON の本は --skip しない**(試し・印の無い走り直し・落ちた本でも。例外なし = 10/28 main session の決め)を手順書 6 章・7 章
  (1 m 以上下がる・中立止め・走った値が計画と違った・起こし直した・いつでも)・8.1 ⑤・todo 3 章・approvals 8・12 に。確かめは
  out/skip_check_r2.txt(本物の collect.py と car_check を作り物の回 3 で)。
- f8r2-docs-1023-first-recovery-off: main session の決め(10/28 11:xx)= その場の守り(1 m 以上下がる・後ろの物に近づく・後ろ約 1 m
  以内に物 → 人側)は残し、人側にしただけでは OFF にしない。その日を OFF にするのは、ふつうのリカバリーでは起きない形だけ
  (① 止まってから NOBACK_S 秒たっても下がらない ② 1 回で BACK_M m より長く下がる)。数は out/ana_first_rec_r2.txt
  (閉じたループ first_rec_r2.py)と out/real_rec_r2.txt(本物の記録)から。手順書 3・7 章・approvals 3・todo 3 章(と 2 章 10)をそろえた。
- f8r2-docs-todo-no-approvals3-notice: todo 1 章「11/2 まで」・2 章 10 に approvals 3 の 10/28 の知らせ。
- f8r2-docs-approvals-restart-stale: approvals 0 章の 2 といちばん上の 2 つ目の「今(10/27 17:3x)」に 08:40 の起動し直しの印。

置き換えの決まりは前の台本と同じ: old が文書の中でちょうど 1 回・new が 0 回のときだけ入れる。2 回目に流すと全部「済」。
前の係の文(前の台本の new。あとの台本が書き換えたあとの今の形)の中を書き換えるのは SUPERSEDE の所だけ。前の係の試験は revert() で
この組の書き換えを戻した文書で前の台本の直しが「済」かを照らし、effective() で書き換えたあとの形の new を使う。
--apply: 控え <名前>.bak-20261028-fix8-docs-r2 を作ってから書く(控えがあれば上書きしない)。

使い方: python3 -I -B docs_fix8_r2.py [--apply]
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
PREV = {'r1': E / 'fix7' / 'docs_r1' / 'docs_fix7_r1.py', 'r2': E / 'fix7' / 'docs_r2' / 'docs_fix7_r2.py',
        'final': E / 'fix7' / 'docs_final' / 'docs_fix7_final.py', 'decided': E / 'fix8' / 'docs_decided' / 'docs_fix8_decided.py',
        'r81': E / 'fix8' / 'fix_docs_r1' / 'docs_fix8_r1.py', 's1': E / 'fix8' / 'docs_sync1' / 'docs_fix8_sync1.py'}
TESTS = {'r1': JF / 'tests' / 'test_fix7_docs_r1.py', 'r2': JF / 'tests' / 'test_fix7_docs_r2.py',
         'final': JF / 'tests' / 'test_fix7_docs_final.py', 'decided': JF / 'tests' / 'test_fix8_docs_decided.py',
         'r81': JF / 'tests' / 'test_fix8_docs_r1.py', 's1': JF / 'tests' / 'test_fix8_docs_sync1.py',
         'r82': JF / 'tests' / 'test_fix8_docs_r2.py'}
BAK = '.bak-20261028-fix8-docs-r2'
STAMP = '10/28 11:4x'
RB, TODO, RD, AP, F7 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'
FILES = (RB, TODO, RD, AP)
ALL7 = (RB, TODO, RD, AP, 'fix3/disk.md', F7, 'fix6/README.md')

# ---------------------------------------------------------------------------------------------- 数(台本の出力から写した)
NOBACK_S = 3            # ① 止まってから この秒たっても下がらない
BACK_M = 2.5            # ② 1 回のリカバリーで この m より長く下がる
N_ON = '1.02〜1.97 m'   # ふつうの ON の最初のリカバリーの下がる距離(閉じたループ。normal18 + gapslow の 576 場面)
N_NB = '0.64〜2.22 秒'  # ふつうの ON の、止まってから下がり始めるまで(同じ 576 場面)
R_NB = '0.83〜1.48 秒'  # 本物(10/23・10/24 の rev_s 0.4・rev_us 300 のリカバリーの始め 6 回)
K2 = '3.63〜6.18 m'      # 止まらない後退の模型(車輪の値が後ろへの動きを出さない。k2_runaway の 72 場面)の下がる距離
CL = ('閉じたループの偽物の ESC の模型 24 通り(fix8 safety r1 の 18 通りと、最初の \'gap\' 0.15 秒では下がれない 6 通り)× 24 場面'
      '(10/23 の形 16・前の壁に当たって止まるふつうの形 8)')
SRC2 = '`experiments/20261026_fix/fix8/fix_docs_r2/out/ana_first_rec_r2.txt`・`real_rec_r2.txt`'
SRC1 = '`experiments/20261026_fix/fix8/fix_docs_r2/out/skip_check_r2.txt`'

# ---------------------------------------------------------------------------------------------- よく使う文
SKIP_NO = '当たった・空転した・記録が欠けた ON の本は `--skip` しない'
SKIP_SEE = ('car_check の「衝突」の行が「0 回」でない・「発進の空転の比」の行の頭が「!」か下に「→ 比が出ない発進は…」・「! 記録が欠けている」'
            'の行が出た本、⑤ の「最初の周の記録が欠けた発進(on)」「読めない記録」に名前が出た ON の本')
SKIP_WHY = ('`--skip` はその本(`記録名@秒` ならその発進)の当たり・空転・欠けまで ⑤ の比べの条件から消すので、「満たさない」が「使ってよい」に'
            '変わる')
SKIP_THEN = 'そのまま ⑤ に数えさせる = 最初の周の当たり・空転・欠けなら回 3 は「満たさない」= 回 4 は発進の前進 OFF'
DEC = '10/28 main session の決め・fix8 docs r2'
OFF_ONLY = ('① 止まってから %d 秒たっても下がらない ② 1 回で %.1f m より長く下がる' % (NOBACK_S, BACK_M))

# 「おかしな動き」の文(手順書 3 章「最初のリカバリー」の行。r2 #3 の頭の「1 m 以上下がる・後ろの物に近づく・」を、その場の守りと
# OFF にする形に分けた。後ろの「(前の壁に … 押す(約 0.5 秒)のも、おかしな動き。…) → すぐ CH3 を人側 → 止めて OFF …」は前の文のまま)
FIRST = ('**その場の守り**(7 章。人側にするだけ): 1 m 以上下がる・後ろの物に近づく・止まった所の後ろ約 1 m 以内に物がある 10/23 の形'
         '(上の行。リカバリーを待たずに)・当たって止まった車が前へ押し続けている → すぐ CH3 を人側。**人側にしただけ・1 m 以上下がっただけでは'
         '見張り・残すを OFF にしない**(見張り ON でもふつうのリカバリーは約 1〜2 m 下がる: ' + CL + 'で、ON の最初のリカバリーは 576 場面とも '
         + N_ON + ' 下がった。本物の 10/23 の AI 側のまま終わったリカバリー(`rev_s` 0.4・`rev_us` 300)は 0.94 m。前に書いた「1 m 以上下がる'
         ' → 止めて OFF」だと、ふつうの最初のリカバリーでその日が OFF になった)。**その日を OFF にするのは、ふつうのリカバリーでは起きない'
         '「おかしな動き」を見たときだけ**(' + DEC + '。どちらも数秒かけて分かる形 = 0.6 秒の内に 10/23 の形かを見分けなくてよい): '
         '**① 下がらない**(0 m の壁押し): 前の壁・物で止まってから **%d 秒** たっても 1 度も下がらない(ブレーキ・中立のあと下がらずにまた前へ'
         '押す・同じ所でまたリカバリーが始まる・そのまま「リカバリーをやめて中立」で止まって AI 側に戻らない、も同じ形。ふつうは止まってから '
         % NOBACK_S + N_NB + 'で下がり始める(本物 ' + R_NB + ')。ESC が後退に入らない模型で 1 度も下がらなかった 96 場面は 96 とも当たる)/ '
         '**② 長く下がる**: 1 回のリカバリーで、止まった所から **%.1f m** より長く下がる(ふつうの ON は 1.97 m まで。車輪の値が後ろへの動きを'
         '出さない模型では 72 場面とも ' % BACK_M + K2 + ')。下がって前へ出たあとの 2 回目のリカバリー・「中立で停止中」のあとの「リカバリーを'
         'やめて中立」(残すの決まりどおり)は ① ではない。人側にして形を見届けなかったときは OFF にせず、次のリカバリーを同じように見る'
         '(' + SRC2 + ')。① ② のおかしな動き(① の例: ')

# ==================================================================================================== 手順書
RB_LOG_OLD = '\n- **10/28 09:1x fix8 係 docs 同期 1**'
RB_LOG = ('\n- **' + STAMP + ' fix8 係 docs 直す 8-2 回目**(下の 09:1x の同期 1 のあと。`fix8/fix_docs_r2.md`。控え `runbook_20261103.md' + BAK
          + '`。手順書だけ。コードは変えていない。反証 8-2 回目の係 docs・rehearsal の指摘): 6 章・7 章(「1 m 以上下がる」「中立で停止中」'
          '「走った値が計画と違った」「起こし直した」「いつでも」)・8.1 ⑤ に **' + SKIP_NO + '**(試し・印の無い走り直し・落ちた本でも。'
          + SKIP_THEN + '。10/28 main session の決め)/ 3 章「10/23 の形の後ろの近い物」「最初のリカバリー」・7 章「1 m 以上下がる」に、'
          '**人側にしただけ・1 m 以上下がっただけでは OFF にしない**・その日を OFF にするのはふつうのリカバリーでは起きない形(' + OFF_ONLY
          + ')だけ(10/28 main session の決め。数は閉じたループと本物の記録から)。' + RB_LOG_OLD)
# 3 章「10/23 の形の後ろの近い物」の行の終わり(SUPERSEDE s1 #1)
RB_NEAR_X = '先に人側にしただけなら見張り・残すは OFF にしない(OFF にするかは下の「最初のリカバリー」の行で決める)。'
RB_NEAR_Y = ('先に人側にしただけなら見張り・残すは OFF にしない(**OFF にするのは、下の「最初のリカバリー」の行の「おかしな動き」' + OFF_ONLY
             + ' を見たときだけ**。10/23 の形かどうかは見分けなくてよい。' + DEC + ')。')
# 3 章「最初のリカバリー」(SUPERSEDE r2 #3)
RB_FIRST_X = '1 m 以上下がる・後ろの物に近づく・おかしな動き(前の壁に当たって止まったあと、'
RB_FIRST_Y = FIRST + '前の壁に当たって止まったあと、'
# 6 章「発進の前進 ON(回 4)」の行(どれも前の係の new の外)
RB_6A_OLD = '車のせいでない理由(人・物がコースに入った・試し)で止めた本だけ、main session が `--skip 記録名` で外して読み直す。'
RB_6A = (RB_6A_OLD + '**ただし' + SKIP_NO + '**(車のせいでない理由・試し・走り直しでも。' + SKIP_THEN + '。見分け方は 8.1 ⑤ の「`--skip`」の文。'
         + DEC + ')。')
RB_6B_OLD = '→ 動画・紙で確かめ、当たっていない・車のせいでないと分かった本だけ `--skip 記録名`。'
RB_6B = ('→ 動画・紙で確かめる(**記録が欠けた ON の本なので `--skip` しない** =「満たさない」のまま・回 4 は発進の前進 OFF。' + DEC
         + '。前は、動画・紙で当たっていないと分かった本を外していた)。')
RB_6C_OLD = '→ 紙でどの本か確かめ、ON でない・当たっていないと分かれば `--skip 記録名`。'
RB_6C = '→ 紙でどの本か確かめ、ON でないと分かれば `--skip 記録名`(ON の本なら記録が欠けた ON の本 = `--skip` しない。' + DEC + ')。'
# 7 章「1 m 以上下がる・後ろの物に近づく」の行の終わり(SUPERSEDE s1 #4)
RB_7B_X = '紙に「走り直し: 本・時刻」)。先に人側にしただけなら見張り・残すは OFF にしない(3 章の「最初のリカバリー」の行で決める) |'
RB_7B_Y = ('紙に「走り直し: 本・時刻」。**その本が当たった・空転した・記録が欠けたら main session は `--skip` しない** = ⑤ に数えさせる(走り直しの'
           '最初の周なら回 3 は「満たさない」。下の中立止めの行・8.1 ⑤。fix8 docs r2)。先に人側にしただけなら見張り・残すは OFF にしない'
           '(**1 m 以上下がった・後ろの物に近づいて人側にしただけでは OFF にしない**。見張り ON でもふつうのリカバリーは約 1〜2 m 下がる。'
           'OFF にするのは 3 章の「最初のリカバリー」の行の「おかしな動き」' + OFF_ONLY + ' のときだけ。' + DEC + ') |')
# 7 章「中立で停止中」の行の「印が無ければ …--skip」(SUPERSEDE r2 #8)
RB_7H_X = '(印が無ければ、⑤ は main session が `--skip 記録名@AI側にした秒` で外す。'
RB_7H_Y = (RB_7H_X + '**ただしその本(ON)が当たった・空転した・記録が欠けた(' + SKIP_SEE + ')なら `--skip` しない** = ' + SKIP_THEN + '(' + SKIP_WHY
           + '。8.1 ⑤ の「`--skip`」の文。' + DEC + ')。')
# 7 章「走った値が計画と違った」(前の係の new のさかい目に足す。後ろの「⑤ の行に名前が無ければ … --skip」は r1 #10 の文のまま)
RB_7P_OLD = '10/28 fix8 tools r1)。⑤ の行に名前が無ければ、main session が `--skip 記録名` で外す)'
RB_7P = ('10/28 fix8 tools r1)。**下の `--skip` は、当たった・空転した・記録が欠けた ON の本には使わない**(紙の「試し」でも。'
         + SKIP_THEN + '。8.1 ⑤ の「`--skip`」の文。' + DEC + ')。⑤ の行に名前が無ければ、main session が `--skip 記録名` で外す)')
# 7 章「起こし直した」の行の終わり(SUPERSEDE r81 #5)
RB_7R_X = '→ main session が `--skip 記録名`(本ごと外す。7 章「走った値が計画と違った」の行と同じ) |'
RB_7R_Y = ('→ main session が `--skip 記録名`(本ごと外す。7 章「走った値が計画と違った」の行と同じ)。**ただし' + SKIP_NO + '**(紙の「試し」でも。'
           + SKIP_THEN + '。' + SKIP_WHY + '。8.1 ⑤。' + DEC + ') |')
# 7 章「いつでも」(前の係の new の外)
RB_7A_OLD = 'どの本か紙で分かり、ON でない・当たっていないと分かれば main session が `--skip 記録名`。'
RB_7A = ('どの本か紙で分かり、ON でないと分かれば main session が `--skip 記録名`(**ON の本なら記録が欠けた ON の本 = `--skip` しない**。'
         + DEC + '。前は「当たっていない」と分かった ON の本も外していた)。')
RB_7C_OLD = ('動画・紙で、欠けた区間に発進が無かった・当たっていないと確かめた本だけ main session が `--skip 記録名`(本ごと外れる。読めた発進も'
             '比べから外れる)。')
RB_7C = ('この本は記録が欠けた ON の本なので、**動画・紙で当たっていないと分かっても `--skip` しない**(「満たさない」のまま = 回 4 は発進の前進 OFF。'
         + DEC + '。前は、欠けた区間に発進が無かった・当たっていないと確かめた本を main session が本ごと外していた)。')
# 8.1 ⑤: 印の無い走り直し(SUPERSEDE r2 #18)
RB_8R_X = '印の無い走り直し(紙に「走り直し」と書いた本)は main session が `--skip 記録名@AI側にした秒` で外す'
RB_8R_Y = RB_8R_X + '(**' + SKIP_NO + '**。下の「`--skip`」の文。fix8 docs r2)'
# 8.1 ⑤: --skip の文(前の係の new の外)
RB_8S_OLD = '`--skip 記録名`(か `記録名@AI側の秒`)はまとめから外す(main session が、車のせいでない理由で止めた本だけ。外した名前が出る)。'
RB_8S = (RB_8S_OLD + '**' + SKIP_NO + '**(試し・印の無い走り直し・落ちた本でも。例外なし。見分け方: その本の ' + SKIP_SEE + '。' + SKIP_WHY
         + '(作り物の回 3 で、紙の「試し」の ON の本と印の無い走り直しの、最初の周の当たり・空転・欠けの 6 通りとも、外さなければ「満たさない」・'
         '外すと「使ってよい」。不具合の無い本は外しても「使ってよい」のまま。' + SRC1 + ')。' + SKIP_THEN + '。`collect.py` の `--skip` の'
         '説明「当たった本は外さない」と同じ向きで、空転・欠けも(' + DEC + ')。')
# 8.1 ⑤: 形・ゲインの違う試しの --skip(SUPERSEDE s1 #10)
RB_8T_X = '`--skip` で外すと不具合も数えない = 当たった本は `--skip` しない。10/28 fix8 tools r1)'
RB_8T_Y = ('`--skip` で外すと不具合も数えない = 当たった本は `--skip` しない(空転した・記録が欠けた ON の本も = 上の「`--skip`」の文・'
           'fix8 docs r2)。10/28 fix8 tools r1)')
# 8.1 ⑤: カメラが止まった ON の発進の --skip(SUPERSEDE s1 #11)
RB_8C_X = '当たっていない・車のせいでないと動画で確かめたら main session が `--skip 記録名@T`)'
RB_8C_Y = ('当たっていない・車のせいでないと動画で確かめたら main session が `--skip 記録名@T`。当たった・空転した・記録が欠けた ON の本は外さない'
           ' = 上の「`--skip`」の文。fix8 docs r2)')

# ==================================================================================================== todo
TD_TITLE_X = '# ユーザーが 11/3 までにすること(10/28 09:1x 現在)'
TD_TITLE_Y = '# ユーザーが 11/3 までにすること(' + STAMP + ' 現在)'
TD_LOG_OLD = '- **10/28 09:1x fix8 係 docs 同期 1**(反証 8-1 回目の直しの文書の同期。下の 05:2x のあと。'
TD_LOG = ('- **' + STAMP + ' fix8 係 docs 直す 8-2 回目**(下の 09:1x の同期 1 のあと。`fix8/fix_docs_r2.md`。控え `todo_user_1026.md' + BAK
          + '`): 題の時刻を今に / 1 章「11/2 まで」・2 章 10 に approvals 3 の 10/28 の知らせ(10/23 の形の後ろの近い物。(c) を変えたいときは'
          '一言)/ 2 章 10・3 章の「最初のリカバリー」「10/23 の形」に、人側にしただけ・1 m 以上下がっただけでは OFF にしない・その日を OFF に'
          'するのは ' + OFF_ONLY + ' のときだけ / 3 章の「走り直し」「試し」「R1 に届かなかった」「落ちた」に、' + SKIP_NO + '(10/28 main '
          'session の決め)。\n' + TD_LOG_OLD)
NOTICE = ('**10/28 に 3 へ知らせ 1 つ**: 10/23 の形(前の物の手前でブレーキのあと止まり、すぐ下がる形)では、見張り ON は止まった所から約 1 m より'
          '近い後ろの物に OFF より強く当たることがある(約 1 m より遠い物には ON のほうが遅く当たるか当たらない)。11/3 のコードは変えない。守りは'
          ' 3 章の「リカバリーを待たずに人側にしてよい」。(c) を変えたいときは一言・返事が無ければ (c) のまま(`fix3/approvals.md` 3 の'
          '「困ること(ON)・10/23 の形」・`README.md` 6.2 の 33。10/28 fix8 docs r2)')
# 1 章「11/2 まで」(SUPERSEDE r1 #2)
TD_23_X = '3 は決まり((c) 両方 ON)。'
TD_23_Y = '3 は決まり((c) 両方 ON。' + NOTICE + ')。'
# 2 章 10(前の係の new の外に知らせ・SUPERSEDE r1 #5 に「おかしければ」)
TD_48N_OLD = '**(c) 両方 ON**(10/27 15:4x。`user_decisions.md`・`fix3/approvals.md` 3)。11/3 は朝の準備で見張り → 残すの順に ON'
TD_48N = ('**(c) 両方 ON**(10/27 15:4x。`user_decisions.md`・`fix3/approvals.md` 3)。' + NOTICE + '。11/3 は朝の準備で見張り → 残すの順に ON')
TD_48_X = '最初のリカバリーは人が見る。おかしければ止めて OFF('
TD_48_Y = ('最初のリカバリーは人が見る。おかしければ止めて OFF(「おかしければ」は、ふつうのリカバリーでは起きない形 = 手順書 3 章の ' + OFF_ONLY
           + ' のときだけ。1 m 以上下がった・人側にしただけでは OFF にしない。10/28 main session の決め。')
# 3 章「その日の最初のリカバリーを見る」(SUPERSEDE r1 #6)
TD_67_X = '(1 m 以上下がる・後ろの物に近づく → すぐ人側 → 止めて OFF。'
TD_67_Y = ('(1 m 以上下がる・後ろの物に近づく → すぐ人側。**人側にしただけ・1 m 以上下がっただけでは OFF にしない**(見張り ON でもふつうの'
           'リカバリーは約 1〜2 m 下がる)。**その日を OFF にするのは、ふつうのリカバリーでは起きない形だけ**: ① 前の壁・物で止まってから '
           '%d 秒たっても 1 度も下がらない(下がらずにまた前へ押す・同じ所でまたリカバリーが始まる も)② 1 回のリカバリーで止まった所から '
           '%.1f m より長く下がる → すぐ人側 → 止めて OFF(10/28 main session の決め。手順書 3 章)。' % (NOBACK_S, BACK_M))
# 3 章 10/23 の形(SUPERSEDE s1 #3)
TD_71_X = ('先に人側にしただけなら見張り・残すは OFF にしない。そのあとは「当たって止まったあと」と同じ = その場で AI 側に戻さず HS に置き直す・'
           '紙に「走り直し: 本・時刻」。')
TD_71_Y = ('先に人側にしただけなら見張り・残すは OFF にしない(1 m 以上下がっただけでも OFF にしない。OFF にするのは上の「その日の最初の'
           'リカバリーを見る」の行の ① ② のときだけ。10/28 main session の決め)。そのあとは「当たって止まったあと」と同じ = その場で AI 側に'
           '戻さず HS に置き直す・紙に「走り直し: 本・時刻」(その本が当たった・空転した・記録が欠けたら main session は `--skip` しない = '
           '回 3 は「満たさない」。10/28 fix8 docs r2)。')
# 3 章「当たって止まった車が…」の「戻したら紙に」(SUPERSEDE r2 #5・仕上げ #4 の重なり)
TD_72_X = '戻したら紙に「走り直し: 本・時刻」。'
TD_72_Y = '戻したら紙に「走り直し: 本・時刻」(その本が当たった・空転した・記録が欠けたら main session は `--skip` しない。10/28 fix8 docs r2)。'
# 3 章「試し」(SUPERSEDE r81 #3)
TD_75_X = 'できなければ紙 → main session の `--skip`。10/28 fix8 docs r1)。'
TD_75_Y = ('できなければ紙 → main session の `--skip`。10/28 fix8 docs r1。**' + SKIP_NO.replace('ON の本は', 'ON の本は main session も')
           + '** = そのまま数える = 回 3 は「満たさない」。10/28 main session の決め・fix8 docs r2)。')
# 3 章「R1 に届かなかった」・「落ちた」(前の係の new の外)
TD_76_OLD = '車のせいでない理由で止めた本だけ main session が外して読み直す)。'
TD_76 = '車のせいでない理由で止めた本だけ main session が外して読み直す。当たった・空転した・記録が欠けた ON の本は外さない。' + DEC + ')。'
TD_77A_OLD = '落ちた ON の本は紙・動画で確かめて main session が外すまで ⑤ が「満たさない」。10/27 fix6-2。'
TD_77A = ('落ちた ON の本は ⑤ が「満たさない」(記録が欠けた ON の本なので main session も外さない = 回 4 は発進の前進 OFF。' + DEC
          + ')。10/27 fix6-2。')
TD_77B_OLD = '紙・動画でどの本か・当たっていないと分かれば main session が外して読み直す。'
TD_77B = '紙・動画でどの本か確かめ、ON でないと分かれば main session が外して読み直す(ON の本なら外さない。' + DEC + ')。'

# ==================================================================================================== README
RD_LOG_OLD = '\n- **10/28 09:1x 追記**(fix8 係 docs 同期 1。'
RD_LOG = ('\n- **' + STAMP + ' 追記**(fix8 係 docs 直す 8-2 回目。下の 09:1x の同期 1 のあと。控え `README.md' + BAK + '`。報告 '
          '`fix8/fix_docs_r2.md`): 6.2 に新しい 36(ユーザーに伝えること: 10/28 main session の決め 2 つ = その日を OFF にする「おかしな動き」の'
          '決まりと、' + SKIP_NO + ')・6.3 の 35 の「`--skip` は不具合も外す」に手順書の今。' + RD_LOG_OLD)
RD_636_X = '\n\n### 6.3 11/3 のあとでもよい(小さいもの)'
RD_636_Y = ('\n| 36 | **ユーザーに伝えること**(10/28 main session の決め・fix8 docs r2。返事が無ければこのまま): ① approvals 3 の (c) の「最初の'
            'リカバリーは人が見る。おかしければ止めて OFF」の「おかしければ」は、ふつうのリカバリーでは起きない形(' + OFF_ONLY + ')だけ。'
            '1 m 以上下がった・後ろの物に近づいて人側にしただけでは OFF にしない(見張り ON でもふつうのリカバリーは約 1〜2 m 下がる。'
            '前の手順書の「1 m 以上下がる → 止めて OFF」だと、ふつうの最初のリカバリーでその日が OFF になった。手順書 3・7 章・`fix3/approvals.md`'
            ' 3)② ' + SKIP_NO + '(試し・印の無い走り直し・落ちた本でも。そのまま ⑤ に数えさせる = 回 3 は「満たさない」= 回 4 は発進の前進'
            ' OFF。`fix3/approvals.md` 12 の「動画・紙で確かめて `--skip` するまで回 4 は OFF」は、落ちた本が ON ならその日は OFF のまま。手順書'
            ' 8.1 ⑤) | main session → ユーザー | 11/2(返事が無ければこのまま) |' + RD_636_X)
RD_635_X = '`--skip` は不具合も外す。'
RD_635_Y = ('`--skip` は不具合も外す(→ 10/28 fix8 docs r2: 手順書・todo・approvals は ' + SKIP_NO + ' 形にした = 道具の形はそのまま。'
            '6.2 の 36)。')

# ==================================================================================================== approvals
AP_LOG_OLD = '- **10/28 09:1x fix8 係 docs 同期 1**(反証 8-1 回目の直しの文書の同期。控え'
AP_LOG = ('- **' + STAMP + ' fix8 係 docs 直す 8-2 回目**(控え `approvals.md' + BAK + '`。`fix8/fix_docs_r2.md`): 0 章の 2・いちばん上の 2 つ目の'
          '「今(10/27 17:3x)」に **10/28 08:40 の起動し直しで再起動は要らなくなった**印 / 3 の「決まり」「なぜ」「困ること(ON)・10/23 の形」に、'
          'その日を OFF にする「おかしな動き」はふつうのリカバリーでは起きない形(' + OFF_ONLY + ')だけ・人側にしただけでは OFF にしない'
          '(10/28 main session の決め)と、ふつうの後退の幅 / 8・12 に ' + SKIP_NO + '(10/28 main session の決め)。答え・おすすめは変えていない。'
          '\n' + AP_LOG_OLD)
AP_10_OLD = '- **今(10/27 17:3x。fix7 係 docs が今に合わせた。控え `approvals.md.bak-20261027-fix7-docs`)**: '
AP_10 = ('- 前の今(10/27 17:3x。**今は下の「今(10/28 03:3x)」の行**。この行は 10/27 17:3x の記録のまま: この行の「もう一度の再起動が要る」は'
         ' 10/28 00:53:23 にした・そのあとの drive.py の説明の文(10/28 02:33)のぶんも **10/28 08:40 の Jetson の起動し直しで入った**'
         '(jetformer-web 08:40:33。「起動より新しいコードなし」= 2 の「今(10/28 08:40)」の行)= 再起動は要らない。印は 10/28 fix8 docs r2。'
         'fix7 係 docs が今に合わせた。控え `approvals.md.bak-20261027-fix7-docs`): ')
# 0 章の 2(SUPERSEDE r2 #2。fix8 docs-decided の書き換えの中)
AP_02_X = ('反証 8 回目のあとにもう一度 | 済(車輪を浮かせた確かめはまだ)。もう一度は 11/3 の前(10/28 00:53 に済。説明の文のぶんは反証 8 回目の'
           'あと) |')
AP_02_Y = ('反証 8 回目のあとにもう一度 → **10/28 08:40 の Jetson の起動し直しで入った**(jetformer-web 08:40:33。「起動より新しいコードなし」= '
           '2 の「今(10/28 08:40)」の行。この再起動は要らなくなった。10/28 fix8 docs r2) | 済(車輪を浮かせた確かめはまだ)。もう一度は 11/3 の前'
           '(10/28 00:53 に済。説明の文のぶんは反証 8 回目のあと → **10/28 08:40 の起動し直しで入った** = 要らなくなった) |')
# 3「決まり」(SUPERSEDE r1 #3)
AP_3K_X = '最初のリカバリーは人が見る。おかしければ止めて OFF('
AP_3K_Y = ('最初のリカバリーは人が見る。おかしければ止めて OFF(**「おかしければ」は、ふつうのリカバリーでは起きない形だけ** = 手順書 3 章の'
           '「おかしな動き」' + OFF_ONLY + '。1 m 以上下がった・後ろの物に近づいて人側にしただけでは OFF にしない。' + DEC + '。')
# 3「なぜ」(前の係の new の外)
AP_3W_OLD = '見張り ON の再生では、後退をやめた時点で 0.75 m(その先の惰性は記録から分からない)。'
AP_3W = (AP_3W_OLD + '(10/28 fix8 docs r2: 惰性まで入れると、見張り ON でもふつうに約 1〜2 m 下がる = ' + CL + 'で、ON の最初のリカバリーは '
         + N_ON + '(No.6 に合わせた模型 1.28〜1.40 m。同じ場面の OFF は 10/23 の形で 1.17〜2.91 m・No.6 に合わせた模型 2.30〜2.42 m)。本物の'
         ' 10/23 の AI 側のまま終わったふつうのリカバリー(`rev_s` 0.4・`rev_us` 300)は 0.94 m。' + SRC2 + ')')
# 3「困ること(ON)・10/23 の形」(SUPERSEDE s1 #2)
AP_3N_X = 'リカバリーを待たずに CH3 を人側にしてよい(3 章にも)。'
AP_3N_Y = ('リカバリーを待たずに CH3 を人側にしてよい(3 章にも)。人側にしただけ・1 m 以上下がっただけでは見張り・残すを OFF にしない(OFF に'
           'するのは手順書 3 章の「最初のリカバリー」の行の「おかしな動き」' + OFF_ONLY + ' のときだけ。' + DEC + ')。')
# 8 の表の下の --skip(前の係の new の外)・③ の新しい行・12 の落ちた ON の本(前の係の new の外)
AP_8S_OLD = '紙・動画で確かめてから main session が `--skip 記録名` で外して読み直す。'
AP_8S = (AP_8S_OLD + '**' + SKIP_NO + '**(試し・印の無い走り直し・落ちた本でも。' + SKIP_THEN + '。手順書 8.1 ⑤。' + DEC + ')。')
AP_8T_OLD = '**今のまま**(10/28 main session の決め)。\n\n## 9. 左 4.2'
AP_8T = ('**今のまま**(10/28 main session の決め)。\n- (' + DEC + ')③ の紙の「走り直し」→ `--skip 記録名@AI側にした秒` は、**その本(ON)が'
         '当たった・空転した・記録が欠けたときは使わない**(' + SKIP_THEN + '。' + SKIP_WHY + '。反証 8-2 回目で、リカバリーを待たずに人側に'
         'したあとの走り直し(手順書 3・7 章の 10/23 の形の行)はいつも印が付かない形と分かった。試し・落ちた本の `--skip` も同じ。手順書 7 章の'
         '中立止めの行・8.1 ⑤)。\n\n## 9. 左 4.2')
AP_12_OLD = '回 3 で 1 本落ちると、動画・紙で確かめて main session が `--skip 記録名` で外すまで回 4 は OFF。'
AP_12 = (AP_12_OLD + '**→ 10/28 main session の決め: 落ちた ON の本は記録が欠けた ON の本なので `--skip` しない = その日の回 4 は発進の前進 OFF**'
         '(' + SKIP_NO + '。手順書 8.1 ⑤。fix8 docs r2)。')

ALL = [
    (RB, [(RB_LOG_OLD, RB_LOG), (RB_NEAR_X, RB_NEAR_Y), (RB_FIRST_X, RB_FIRST_Y), (RB_6A_OLD, RB_6A), (RB_6B_OLD, RB_6B),
          (RB_6C_OLD, RB_6C), (RB_7B_X, RB_7B_Y), (RB_7H_X, RB_7H_Y), (RB_7P_OLD, RB_7P), (RB_7R_X, RB_7R_Y), (RB_7A_OLD, RB_7A),
          (RB_7C_OLD, RB_7C), (RB_8R_X, RB_8R_Y), (RB_8S_OLD, RB_8S), (RB_8T_X, RB_8T_Y), (RB_8C_X, RB_8C_Y)]),
    (TODO, [(TD_TITLE_X, TD_TITLE_Y), (TD_LOG_OLD, TD_LOG), (TD_23_X, TD_23_Y), (TD_48N_OLD, TD_48N), (TD_48_X, TD_48_Y),
            (TD_67_X, TD_67_Y), (TD_71_X, TD_71_Y), (TD_72_X, TD_72_Y), (TD_75_X, TD_75_Y), (TD_76_OLD, TD_76), (TD_77A_OLD, TD_77A),
            (TD_77B_OLD, TD_77B)]),
    (RD, [(RD_LOG_OLD, RD_LOG), (RD_636_X, RD_636_Y), (RD_635_X, RD_635_Y)]),
    (AP, [(AP_LOG_OLD, AP_LOG), (AP_10_OLD, AP_10), (AP_02_X, AP_02_Y), (AP_3K_X, AP_3K_Y), (AP_3W_OLD, AP_3W), (AP_3N_X, AP_3N_Y),
          (AP_8S_OLD, AP_8S), (AP_8T_OLD, AP_8T), (AP_12_OLD, AP_12)]),
]
# 指摘(と頭の記録)→ 直し(文書, ALL の何番目)
FINDINGS = {
    'f8r2-docs-paper-trial-skip-drops-faults': [(RB, 3), (RB, 4), (RB, 5), (RB, 8), (RB, 9), (RB, 10), (RB, 11), (RB, 13), (RB, 14),
                                                (RB, 15), (TODO, 8), (TODO, 9), (TODO, 10), (TODO, 11), (AP, 6), (AP, 8), (RD, 2)],
    'rehearsal-r2-unmarked-rerun-skip-drops-hit': [(RB, 6), (RB, 7), (RB, 12), (TODO, 6), (TODO, 7), (AP, 7)],
    'f8r2-docs-1023-first-recovery-off': [(RB, 1), (RB, 2), (TODO, 4), (TODO, 5), (AP, 3), (AP, 4), (AP, 5), (RD, 1)],
    'f8r2-docs-todo-no-approvals3-notice': [(TODO, 2), (TODO, 3)],
    'f8r2-docs-approvals-restart-stale': [(AP, 1), (AP, 2)],
}
LOGS = [(RB, 0), (TODO, 0), (TODO, 1), (RD, 0), (AP, 0)]

# 前の係の文(その台本の ALL の (文書, 何番目) の new の今の形)の中を書き換えた所: (台本, 文書, 何番目, 前の文 X, 今の文 Y)
SUPERSEDE = [
    ('s1', RB, 1, RB_NEAR_X, RB_NEAR_Y),        # 3 章 10/23 の形: OFF にするのは ① ② だけ
    ('r2', RB, 3, RB_FIRST_X, RB_FIRST_Y),      # 3 章 最初のリカバリー: その場の守り / OFF にする形 ① ②
    ('s1', RB, 4, RB_7B_X, RB_7B_Y),            # 7 章 1 m 以上下がる: OFF にしない・走り直しの --skip
    ('r2', RB, 8, RB_7H_X, RB_7H_Y),            # 7 章 中立止め: 印の無い走り直しの --skip
    ('r81', RB, 5, RB_7R_X, RB_7R_Y),           # 7 章 起こし直した: 紙の試しの --skip
    ('r2', RB, 18, RB_8R_X, RB_8R_Y),           # 8.1 ⑤ 印の無い走り直し
    ('s1', RB, 10, RB_8T_X, RB_8T_Y),           # 8.1 ⑤ 形・ゲインの違う試しの --skip
    ('s1', RB, 11, RB_8C_X, RB_8C_Y),           # 8.1 ⑤ カメラが止まった発進の --skip
    ('r81', TODO, 0, TD_TITLE_X, TD_TITLE_Y),   # todo の題の時刻(同期 1 の書き換えのあと)
    ('s1', TODO, 0, TD_TITLE_X, TD_TITLE_Y),    # 同じ題(同期 1 の ALL の 0 番目でもある)
    ('r1', TODO, 2, TD_23_X, TD_23_Y),          # 1 章 11/2 まで: approvals 3 の知らせ
    ('r1', TODO, 5, TD_48_X, TD_48_Y),          # 2 章 10: 「おかしければ」
    ('r1', TODO, 6, TD_67_X, TD_67_Y),          # 3 章 最初のリカバリー
    ('s1', TODO, 3, TD_71_X, TD_71_Y),          # 3 章 10/23 の形
    ('r2', TODO, 5, TD_72_X, TD_72_Y),          # 3 章 戻したら紙に(r2 #5 と仕上げ #4 の重なり)
    ('final', TODO, 4, TD_72_X, TD_72_Y),
    ('r81', TODO, 3, TD_75_X, TD_75_Y),         # 3 章 試し: 紙 → --skip
    ('decided', RD, 10, RD_636_X, RD_636_Y),    # README 6.2: 新しい 36(足すだけ)
    ('s1', RD, 4, RD_635_X, RD_635_Y),          # README 6.3 の 35: 手順書の今(足すだけ)
    ('r2', AP, 2, AP_02_X, AP_02_Y),            # approvals 0 章の 2(fix8 docs-decided の書き換えの中 = decided #1 の new でもある)
    ('decided', AP, 1, AP_02_X, AP_02_Y),
    ('r1', AP, 3, AP_3K_X, AP_3K_Y),            # approvals 3 決まり: 「おかしければ」
    ('s1', AP, 2, AP_3N_X, AP_3N_Y),            # approvals 3 困ること(ON)・10/23 の形
]
# 前の文を消す書き換え(X の文が Y に残らない)
REMOVES = {(TODO, TD_TITLE_X), (RB, RB_FIRST_X), (RB, RB_7B_X), (TODO, TD_67_X), (TODO, TD_75_X), (AP, AP_02_X)}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prev_mods():
    return {k: _load('docs_prev_%s_from_r82' % k, p) for k, p in PREV.items()}


def prev_new(mods, k, f, i):
    """前の台本 k の (文書 f, 何番目 i) の new の、この組の前の今の形(同期 1 の prev_new のあと、同期 1 の書き換え)"""
    new = dict(mods[k].ALL)[f][i][1]
    if k in ('r1', 'r2', 'final'):
        new = mods['decided'].effective(k, f, i, new)
    if k in ('final', 'decided'):
        new = mods['r81'].effective(k, f, i, new)
    if k != 's1':
        new = mods['s1'].effective(k, f, i, new)
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


def earlier_problems(texts, tmp_root=None):
    """前の係の直しと試験の確かめが崩れていないか(texts = 直したあとの 7 本。戻り値は問題の一覧):
    - prior_news_problems(戻した文書で前の 6 つの台本の new がちょうど 1 回・今の文書で書き換えたあとの new がちょうど 1 回)
    - この組の書き換えを戻した文書で、同期 1 の 42 個の直しが「済」・SUPERSEDE が入っている・その台本の earlier_problems(fix8 docs r1・
      docs-decided・fix7 docs r1・r2・仕上げの直しと、前の係の試験の確かめ f・g・h・i・j・k)が空
    - 今の文書で、この組の試験の確かめ(試験のファイルがあれば)が空・SUPERSEDE が入っている"""
    mods = prev_mods()
    bad = [('news',) + tuple(x) for x in prior_news_problems(texts, mods)]
    s1 = mods['s1']
    base = revert(texts)
    _, rep = s1.apply_text(base)
    bad += [('s1', f, o[:30], st) for f, o, st in rep if st != 'done']
    bad += [('s1 superseded(戻した文書)',) + tuple(x) for x in s1.superseded_present(base)]
    bad += [('s1.earlier_problems(戻した文書)',) + tuple(x) for x in s1.earlier_problems(base, tmp_root)]
    if TESTS['r82'].exists():
        sys.path.insert(0, str(JF / 'tests'))
        t7 = _load('t_r82_from_r82', TESTS['r82'])
        for name, fn in sorted(t7.CHECKS.items()):
            bad += [('r82 ' + name, p) for p in fn(texts)]
    bad += [('superseded',) + tuple(x) for x in superseded_present(texts)]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args(argv)
    tmp_root = E / 'fix8' / 'tmp_docs_r2'
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
