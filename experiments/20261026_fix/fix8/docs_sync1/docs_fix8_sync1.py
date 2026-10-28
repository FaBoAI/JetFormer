#!/usr/bin/env python3
"""fix8 係 docs 同期 1(2026-10-28): 反証 8-1 回目の直しの報告(fix8/fix_ui_r1.md 10 章・fix_tools_r1.md 10 章・fix_safety_r1.md 3 章 A〜K)の
「手順書に足す文」と fix8/fix_docs_r1.md の「残したこと」を、ユーザーが読む文書(手順書・todo・README・fix3/approvals.md・fix7/README.md)に
入れる。コードは変えない。画面・道具の文はコードの文字列と同じ(tests/test_fix8_docs_sync1.py が照らす)。数は係の台本の出力から写した
(fix8/proposal_r1/out/ana_r1.json・fix8/fix_safety_r1/out/facts12.json・rec_concat12_*.json・fix8/fix_tools_r1/out/scan_real_gaps.txt・
fix8/fix_ui_r1/css/out/heights.txt・toast_*.txt)。

- 置き換えの決まりは前の台本と同じ: old が文書の中でちょうど 1 回・new が 0 回のときだけ入れる。2 回目に流すと全部「済」。
- 前の係の文(前の台本の new。あとの台本が書き換えたあとの形)の中を書き換えるのは SUPERSEDE の所だけ。前の係の試験は、revert() で
  この組の書き換えを戻した文書で前の台本の直しが「済」かを照らし、effective() で書き換えたあとの形の new を使う
  (tests/test_fix7_docs_r1.py・test_fix7_docs_r2.py・test_fix7_docs_final.py・test_fix8_docs_decided.py・test_fix8_docs_r1.py を合わせた)。
- 前の係の SUPERSEDE の Y(fix8 docs-decided・fix8 docs r1 が書き換えた文)の中は変えない(earlier_problems が見る)。
- 10/28 08:40 に Jetson が起動し直した(main session の知らせ。jetformer-web は 08:40:33 に今のコードで起きた)。読むだけで確かめた
  (out/restart_check_0840.txt)ので、文書の「反証 8 回目のあとにもう一度再起動」に「10/28 08:40 の起動し直しで入った」の印を足した
  (RESTART の直し。前の係の SUPERSEDE の Y の中は変えないので、Y の後ろか別の行に足した)。
- --apply: 控え <名前>.bak-20261028-fix8-sync1 を作ってから書く(控えがあれば上書きしない)。

使い方: python3 -I -B docs_fix8_sync1.py [--apply]
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
        'r81': E / 'fix8' / 'fix_docs_r1' / 'docs_fix8_r1.py'}
TESTS = {'r1': JF / 'tests' / 'test_fix7_docs_r1.py', 'r2': JF / 'tests' / 'test_fix7_docs_r2.py',
         'final': JF / 'tests' / 'test_fix7_docs_final.py', 'decided': JF / 'tests' / 'test_fix8_docs_decided.py',
         'r81': JF / 'tests' / 'test_fix8_docs_r1.py', 's1': JF / 'tests' / 'test_fix8_docs_sync1.py'}
BAK = '.bak-20261028-fix8-sync1'
STAMP = '10/28 09:1x'
RB, TODO, RD, AP, F7 = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md', 'fix7/README.md'
FILES = (RB, TODO, RD, AP, F7)
ALL7 = (RB, TODO, RD, AP, 'fix3/disk.md', F7, 'fix6/README.md')

# よく使う文
NEAR = ('10/23 の形(前の物の手前でモデルがブレーキ側を出して止まり、止まっている間 ESC が後退を待っていて、リカバリーの最初の後退側'
        '(\'brake\')ですぐ下がる形)')
BAND = ('閉じたループの偽物の ESC の模型 18 通り × 16 場面 × 距離 0.1〜2.0 m の 39 点 = 11,232 組で、1.0 m 未満の 5,184 組のうち ON が '
        '0.1 m/s より速い 2,922 組・OFF が速い 622 組(ON が速いときの差は平均 +0.38 m/s・最大 +1.0 m/s。10/23 No.6 の記録に合わせた模型では'
        ' 0.30〜0.65 m だけ)、1.0 m 以上の 6,048 組では OFF が速い 2,020・OFF だけ当たる 3,226 に対し ON が速い 34・ON だけ当たる 8')
REAL6 = ('本物の記録(10/10〜10/24 の場面をつないだ 59 本・リカバリーの始め 16 回)でこの形は 10/23 No.6 の 1 回だけで、そのときの後ろの物は'
         '止まった所から約 1.16〜1.22 m(ON のほうが良い距離)')
HUMAN_EARLY = ('**止まった所の後ろ約 1 m 以内に物・人・壁があるときは、リカバリー(止まって約 0.8 秒で始まる)を待たずに CH3 を人側に'
               'してよい**')
STALL = '発進から R1 の起点までにカメラが止まった発進'
OTHER = '注意: 同じ回・同じモデルの別の組(…)の、比べから外した本(試し・途中で変えた・走り直し)の ON の発進に …'
REFUSE = 'サーバーの設定を読み込んでいます。少し待ってから押してください'
BOOT = ('jetformer-web は 08:40:33 に今のディスクのコードで起きた(10/28 09:06 に読むだけで見た: `todo_user_1026.md` 4 章の 3 は active・'
        'NRestarts 0・「エラーの行なし」(`-b`)、4 は新しい項目が全部既定(0・4.1・250)・`boot_at` あり(08:40:54)・走行中でない、5 は'
        '「起動より新しいコードなし」。`fix8/docs_sync_r1.md`)')
REC235 = ('本物の記録(235 の場面を場面ごとに、と 1 本の記録の場面をつないだ 59 本。235 は記録の本数ではない)では起きていない'
          '(出す値が直す前と全部同じ。10/28 fix8 safety r1 がつないで照らした)')

# ==================================================================================================== 手順書
RB_LOG_OLD = '\n- **10/28 05:2x fix8 係 docs 直す 8-1 回目**'
RB_LOG = ('\n- **' + STAMP + ' fix8 係 docs 同期 1**(反証 8-1 回目の直しの文書の同期。下の 05:2x の 8-1 回目のあと。`fix8/docs_sync_r1.md`。'
          '控え `runbook_20261103.md' + BAK + '`。手順書だけ。コードは変えていない): 3 章に、見張り ON の日の 10/23 の形(前の物の手前で'
          'ブレーキのあと止まり、すぐ下がる形)の後ろの近い物の行と、7 章「1 m 以上下がる・後ろの物に近づく」の行に「リカバリーを待たずに'
          '人側にしてよい」(fix8 safety r1)/ 3 章「わざと OFF にした日」・7 章「走った値が計画と違った」「起こし直した」・8.1 ⑤ に、OFF の本'
          'のカメラの止まり・形とゲインの違う試しの ON の不具合(fix8 tools r1)/ 3.1 に今の app.js の時刻 10/28 04:54・7 章「Failed to '
          'fetch」の行に走行タブに戻った直後の「' + REFUSE + '」・8.1 の読み方の注意に ② の試験の選択肢の印(fix8 ui r1)/ 3.1 に **10/28 08:40 の'
          ' Jetson の起動し直し**(jetformer-web 08:40:33。drive.py の説明の文も入った = 反証 8 回目のあとのもう一度の再起動は要らなくなった)。'
          + RB_LOG_OLD)
# 3 章: 10/23 の形の後ろの近い物(新しい行。上の行の「見張りが OFF のとき …10/23 No.6…→ 7 章。」のすぐ下)
RB_NEAR_OLD = '(10/23 No.6・20261023-205458 の 2 本目のシーン: 1.24 m 下がって後ろに 5.8G)→ 7 章。'
RB_NEAR = (RB_NEAR_OLD + '\n  - **見張り ON の日の、10/23 の形の後ろの近い物**(10/28 fix8 safety r1。`fix3/approvals.md` 3 の「困ること(ON)・'
           '10/23 の形」): ' + NEAR + 'では、見張り ON は \'gap\' の中立をはさまずに後退側を出し続ける(fix2 の見張りからの形)。そのため'
           '**止まった所から約 1 m より近い後ろの物には ON のほうが OFF より強く当たることがあり、約 1 m より遠い物には ON のほうが遅く当たるか'
           '当たらない**(' + BAND + '。' + REAL6 + ')。' + HUMAN_EARLY + '(7 章「リカバリーで車が 1 m 以上下がる・後ろの物に近づく」の'
           '行)。先に人側にしただけなら見張り・残すは OFF にしない(OFF にするかは下の「最初のリカバリー」の行で決める)。')
# 3 章「わざと OFF にした日」(fix7 docs r2 の文の後ろ。SUPERSEDE r2)
RB_OFF3_X = '② の「…(No.1)から なし」の区切りも起こし直しではない。8.1 ②。10/27 fix7 docs r2)。'
RB_OFF3_Y = (RB_OFF3_X + '本 X から先の OFF の本で、発進から R1 の起点までに AI 側の処理したフレームの間が 0.22 秒より長い(カメラの止まり。'
             '明けて走り続けた)発進は、⑤ が「' + STALL + '」として比べから外す(10/28 fix8 tools r1。前は起点までが伸びた OFF が比べに入り'
             '「使ってよい」の側に動いた)= この形でも「使ってよい」の側には動かない(8.1 ⑤・`fix3/approvals.md` 3)。')
# 3.1: 今の app.js の時刻(10/28 00:53:23 の行の下。SUPERSEDE decided)
RB_APP_X = '11/3 の朝にもう一度この表を全部見る・車輪を浮かせて実際の出力で 1 回(3 章)がまだなのは同じ。'
RB_APP_Y = (RB_APP_X + '\n- **10/28 04:54 に `jetformer/web/static/app.js` を直した**(fix8 ui r1。走行タブに戻った直後の同期と、▶ を押せる時・'
            '表示だけ。画面の版は 9 のまま・送る物は同じ。style.css も 04:59 にトーストの幅を直した。どちらも静的なファイルなので再起動は'
            '要らず、画面を読み直せば入る。`fix8/fix_ui_r1.md`)。読み直した画面の ▶ の横は「画面の版 9(app.js 10/28 04:54)」(このあと誰かが '
            'app.js を直せばその時刻)。**04:54 より前に開いた画面**(スマホで開いたまま・前の日の画面)は ▶ で 409「管理画面が古いままです(この'
            '画面の app.js は …、今のファイルは 10/28 04:54 に直したもの)」→ 読み直す(止める・設定・表示は断らない。7 章の「古いまま」の行)。'
            '上の表の「起動のあとに直したコード」は .py だけを見るので、app.js・style.css の名前は出ない。'
            '\n- **10/28 08:40 に Jetson が起動し直した**(電源断かリセット。main session の知らせ。前の起動の journal は止める記録なしで終わる)。'
            '**' + BOOT[:BOOT.index('(')] + '** = 上の 00:53:23 の行の drive.py の説明の文(02:33)も入った' + BOOT[BOOT.index('('):] + '。そのため'
            '**「反証 8 回目のあとに main session がもう一度再起動する」は要らなくなった**(このあと `jetformer/assist.py`・`drive.py`・'
            '`web/server.py`・`web/bench.py` を直したら、また要る)。起動で既定に戻る値(RESET_ON_BOOT)は既定のまま = 走る日は 3 章の朝の設定を'
            '今までどおり入れる。画面は読み直すだけ(上の行)。11/3 の朝にもう一度この表を全部見る・車輪を浮かせて実際の出力で 1 回(3 章)が'
            'まだなのは同じ。')
RB_BOOT_X = 'か、11/3 の朝に Jetson を入れたときに消える。'
RB_BOOT_Y = RB_BOOT_X + '(→ **10/28 08:40 の Jetson の起動し直しで消えた** = 下の「10/28 08:40」の行)'
# 7 章「1 m 以上下がる・後ろの物に近づく」(前の文はそのまま頭に残す)
RB_7B_OLD = '| リカバリーで車が 1 m 以上下がる・後ろの物に近づく | すぐ CH3 を人側(後ろの見張りが OFF だと、前の本のブレーキのあとに長く下がることがある。3 章) |'
RB_7B = ('| リカバリーで車が 1 m 以上下がる・後ろの物に近づく | すぐ CH3 を人側(後ろの見張りが OFF だと、前の本のブレーキのあとに長く下がることが'
         'ある。3 章)。**前の物の手前でブレーキのあと止まり(10/23 の形)、止まった所の後ろ約 1 m 以内に物・人・壁があるときは、リカバリー'
         '(止まって約 0.8 秒で始まる)を待たずに CH3 を人側にしてよい**(この形のリカバリーは始まるとすぐ(約 0.1 秒で)下がり始め、見張り ON は'
         ' \'gap\' の中立をはさまないので、約 1 m より近い物には OFF より強く当たることがある。約 1 m より遠い物には ON のほうが遅く当たるか'
         '当たらない。1 m より近くても遠くても、OFF でも当たる距離。3 章の「10/23 の形の後ろの近い物」・`fix3/approvals.md` 3。10/28 fix8 safety '
         'r1)。人側にしたあとは下の「当たって止まったあと、その場で AI 側に戻したくなった」の行と同じ(その場で AI 側に戻さない・人側で HS に'
         '置き直す・止まってから 3 周を数えるなら ■ で止めてもう 1 本・紙に「走り直し: 本・時刻」)。先に人側にしただけなら見張り・残すは OFF に'
         'しない(3 章の「最初のリカバリー」の行で決める) |')
# 7 章「走った値が計画と違った」(SUPERSEDE decided)
RB_7P_X = '(どれかがあれば回 3 は「満たさない」。8.1 ⑤。10/28 main session の決め・fix8 tools)'
RB_7P_Y = (RB_7P_X + '。天井・左・自動合わせ・目標・ゲインが計画と違って試しの本が別の組になっても、同じ回・同じモデルの比べのある組(計画の組)の'
           '条件に数える(⑤ の計画の組に「' + OTHER + '」。10/28 fix8 tools r1)')
# 7 章「起こし直した」(SUPERSEDE r81)
RB_7R_X = '→ 気づいたら、入れ直す前に走った本の走り終わりのメモに「試し」(4 章 9)。**その本の'
RB_7R_Y = ('→ 気づいたら、入れ直す前に走った本の走り終わりのメモに「試し」(4 章 9。その本の ON の発進の最初の周の衝突・空転・欠けは、形が'
           '違っても回 3 の計画の組の条件に数える = 満たさない。10/28 fix8 tools r1)。**その本の')
# 7 章「Failed to fetch」の行の終わり(走行タブに戻った直後)
RB_7F_OLD = '数十秒たっても戻らなければ、Jetson と Wi-Fi を確かめ、画面を読み直す |'
RB_7F = ('数十秒たっても戻らなければ、Jetson と Wi-Fi を確かめ、画面を読み直す。**走行タブに戻った直後**(ライブのタブ・ほかのタブから)に ▶・'
         '基準の設定・ブロックを押すと「' + REFUSE + '」が出る(10/28 04:54 の app.js を読み直した画面。戻ったあとにサーバーの今の値を読むまで'
         '押せない = 離れている間に curl・別の画面で変えた値を確認の文に出すため。10/28 fix8 ui r1)→ 1〜2 秒待って押し直す。何秒たっても'
         '出るなら、この行の「サーバーにつながっていません」と同じ(Wi-Fi・サーバーを確かめる) |')
# 8.1 の読み方の注意: ② の試験の選択肢の印(新しい行)
RB_2N_OLD = '| 道具 | 読み方の注意 |\n|---|---|\n| ③ | 「確かめる: A4 が無い」'
RB_2N = ('| 道具 | 読み方の注意 |\n|---|---|\n| ② の印 | (10/28 fix8 ui r1)試験の選択肢の印「回 N の試験の選択肢が X(No.Y)と違う: …」は、**その回で'
         ' ON の選択肢がいちばん多い本(同じ数なら先の本)と比べて、ON の少ない本 = OFF で走った本に付く**(回の最初の本が起こし直しのまま OFF'
         ' で、2 本目から入れ直した日も、印と「計画と違う N 本」は OFF の本。表の下の「(… 回の本は X(No.Y)から)」と同じ本を指す。前は回の'
         '最初の本と比べたので ON で走った本に付いた)。上の表の ② の「回の中で回の最初の本と違う本」は、試験の選択肢ではこの本(ON が'
         'いちばん多い本)と違う本。回の形(天井・左など)・下がる量の印は今も回の最初の本と比べる |\n| ③ | 「確かめる: A4 が無い」')
# 8.1 ⑤: 形・ゲインの違う試し(decided の文の後ろ)
RB_5A_OLD = '10/28 main session の決め = `fix3/approvals.md` 8 の「fix7 で残した読み方」①②)。「空転がいちばんひどい'
RB_5A = ('10/28 main session の決め = `fix3/approvals.md` 8 の「fix7 で残した読み方」①②)。**形(天井・左・自動合わせ・目標)・ゲインが違って'
         '試しの本・途中で変えた本・走り直しが別の組になっても、同じ回・同じモデルの比べのある組の条件に数える**(その組に「' + OTHER + '」= '
         'checks の first_lap_collisions_0・slip・slip_all_measured・on_all_readable のどれか。不具合の出た組に「→ この組の比べから外した本の ON '
         'の発進の不具合(衝突・空転・欠け)は、同じ回・同じモデルの比べのある組(…)の条件にも数えた」。10/28 fix8 tools r1)。「空転がいちばんひどい')
RB_5B_OLD = '計画と違う形の本を「試し」か `--skip` で外すと 1 つの形に戻る。'
RB_5B = (RB_5B_OLD + '(「試し」にした本の ON の不具合は計画の組の条件に残る。`--skip` で外すと不具合も数えない = 当たった本は `--skip` しない。'
         '10/28 fix8 tools r1)')
RB_5C_OLD = '発進の前進 OFF の本の、欠けたシーンの発進は数えられない(動画・紙で)。10/27 fix5 c1・fix6・fix6-2) |'
RB_5C = ('発進の前進 OFF の本の、欠けたシーンの発進は数えられない(動画・紙で)。10/27 fix5 c1・fix6・fix6-2)。**(10/28 fix8 tools r1)「' + STALL
         + ' N 本(AI 側の処理したフレーム(中立止めの行を除く)の間が 0.22 秒より長い。…)」** = 発進(AI 側にした時刻)から R1 の起点までの、'
         'AI 側の処理したフレーム(中立止めの行を除く)の間が 0.22 秒より長い発進(drive はフレームが 0.1 秒来ないと中立を書くので、止まった'
         '間は惰性で起点までが伸びる)。比べ(起点まで・失う・動く・LH)に入れない。ON の組にも OFF の組にも入れない(3 章で途中から OFF にした'
         '日の OFF の本も同じ)。ON の本があれば「→ ON の発進で、発進から R1 までにカメラが止まった本がある = 満たさない」(checks の '
         'on_no_stall。当たっていない・車のせいでないと動画で確かめたら main session が `--skip 記録名@T`)。OFF が全部外れたら「→ OFF の発進が'
         '全部カメラの止まりで比べから外れた」(off_no_stall = 満たさない)。その ON の発進の最初の周の衝突・空転・最初の周の記録の欠けは数える。'
         '発進の行の後ろに「カメラの止まり(フレームの間 X 秒・Y 秒から。比べに入れない)」。「注意: 発進から R1 の起点までにフレームの間が '
         '0.15〜0.22 秒の発進 N 本」= 比べには入れた短い止まり(中立は約 0.05〜0.12 秒)。発進の始めの加速の途中なら起点までが約 0.1 秒まで'
         '伸びうる(OFF の本なら「使ってよい」の側)。「ON − OFF」が「使ってよい」で境目(−0.15 秒)から 0.1 秒以内なら、main session がその本の'
         '動画で発進を確かめて決める(`--skip 記録名@T` で外して読み直せる)。本物の 10/10〜10/24 の比べに入る発進では、発進 → R1 のフレームの'
         '間は 0.133 秒まで(どちらの行も出ない。`fix8/fix_tools_r1.md` 3 章) |')

# ==================================================================================================== todo
TD_TITLE_X = '# ユーザーが 11/3 までにすること(10/28 05:2x 現在)'
TD_TITLE_Y = '# ユーザーが 11/3 までにすること(' + STAMP + ' 現在)'
TD_LOG_OLD = '- **10/28 05:2x fix8 係 docs 直す 8-1 回目**(下の 03:3x'
TD_LOG = ('- **' + STAMP + ' fix8 係 docs 同期 1**(反証 8-1 回目の直しの文書の同期。下の 05:2x のあと。`fix8/docs_sync_r1.md`。控え '
          '`todo_user_1026.md' + BAK + '`): 題の時刻を今に / 3 章に 10/23 の形の後ろの近い物(リカバリーを待たずに人側にしてよい。fix8 '
          'safety r1)・画面の読み直しの行に今の app.js の時刻と走行タブに戻った直後の「読み込んでいます」(fix8 ui r1)/ 1 章・4 章の頭に '
          '**10/28 08:40 の Jetson の起動し直し**(jetformer-web 08:40:33。drive.py の説明の文も入った = もう一度の再起動は要らなくなった)。\n'
          + TD_LOG_OLD)
TD_BOOT1_X = '反証 8 回目のあとに main session がもう一度再起動する(4 章)'
TD_BOOT1_Y = (TD_BOOT1_X + ' → **10/28 08:40 の Jetson の起動し直しで入った**(jetformer-web 08:40:33。4 章の 5 は「起動より新しいコードなし」。'
              'もう一度の再起動は要らなくなった。10/28 fix8 docs 同期 1)')
TD_BOOT4_X = '**反証 8 回目のあとに main session がもう一度、下の 1〜5 をする**(同じ目的)か、11/3 の朝に Jetson を入れたときに消える。'
TD_BOOT4_Y = (TD_BOOT4_X + '→ **10/28 08:40 の Jetson の起動し直し(電源断かリセット)で消えた**: ' + BOOT + '。もう一度の 1〜5 は要らなくなった'
              '(このあと assist.py・drive.py・server.py・bench.py を直したら、また要る。10/28 fix8 docs 同期 1)。')
TD_SCR_OLD = '印を見るためにベンチマークは選ばない)。発進の前進の確かめ(車輪を浮かせて)は「走行中のデータを収集する」を外して走る(手順書 3 章。r2-r2-state-1)。'
TD_SCR = (TD_SCR_OLD + '今の app.js は 10/28 04:54(▶ の横「画面の版 9(app.js 10/28 04:54)」。このあと app.js を直せばその時刻)。それより前に開いた'
          '画面は ▶ で 409「管理画面が古いままです」→ 読み直す。走行タブに戻った直後の ▶ は「' + REFUSE + '」→ 1〜2 秒待って押し直す'
          '(手順書 3.1・7 章。10/28 fix8 ui r1)。')
TD_NEAR_OLD = '- 画面の「空き X GB」が 2 GB 未満なら走らない(記録が書けない。手順書 3・7 章)。\n'
TD_NEAR = (TD_NEAR_OLD + '- (10/28 fix8 safety r1)**前の物の手前でブレーキのあと止まり(10/23 の形)、止まった所の後ろ約 1 m 以内に物・人・壁があるときは、'
           'リカバリー(止まって約 0.8 秒で始まる)を待たずに CH3 を人側にしてよい**(この形では、見張り ON は約 1 m より近い後ろの物に OFF より'
           '強く当たることがある。約 1 m より遠い物には ON のほうが遅く当たるか当たらない。11/3 のコードは変えない。先に人側にしただけなら'
           '見張り・残すは OFF にしない。そのあとは「当たって止まったあと」と同じ = その場で AI 側に戻さず HS に置き直す・紙に「走り直し: 本・'
           '時刻」。手順書 7 章「リカバリーで車が 1 m 以上下がる・後ろの物に近づく」の行・`fix3/approvals.md` 3)。\n')

# ==================================================================================================== README
RD_LOG_OLD = '\n- **10/28 05:2x 追記**(fix8 係 docs 直す 8-1 回目。'
RD_LOG = ('\n- **' + STAMP + ' 追記**(fix8 係 docs 同期 1。反証 8-1 回目の直しの文書の同期。下の 05:2x のあと。控え `README.md' + BAK + '`。'
          '報告 `fix8/docs_sync_r1.md`): 6 章の頭に反証 8-1 回目の直し(20 件 → 16 件・全部直した)・6.2 の 11(トーストの幅と残りの重なり)・'
          '6.2 に新しい 33(ユーザーに伝えること: 10/23 の形の後ろの近い物)・6.3 に新しい 34(11/3 のあとの安全の案 qC)・35(反証 8-1 回目で'
          '残したこと)・7 章に反証 8-1 回目の直しの報告・6 章の頭と 6.2 の 8・32 に **10/28 08:40 の Jetson の起動し直し**(jetformer-web '
          '08:40:33。drive.py の説明の文も入った = 反証 8 回目のあとのもう一度の再起動は要らなくなった)。' + RD_LOG_OLD)
RD_6H_X = '反証 8 回目のあとにもう一度再起動**(6.2 の 8・32)。'
RD_6H_Y = (RD_6H_X + '\n- (10/28 fix8 docs 同期 1)**反証 8-1 回目**(5 つの係の反証: 新しい指摘 20 件 → 打ち消しの 2 票で 16 件が残り、**16 件とも直した**):'
         ' 画面 3(app.js 10/28 04:54・style.css・runs_1103.py。`fix8/fix_ui_r1.md`)・夜の道具 2(collect.py。`fix8/fix_tools_r1.md`)・安全 2'
         '(**11/3 の安全のコード(assist.py・drive.py)は変えない** = main session の決め。本物の記録をつないで流す試験と台本・文書の文・'
         '11/3 のあとの案 `fix8/proposal_safety_after1103.md` 9 章。`fix8/fix_safety_r1.md`)・文書 9(`fix8/fix_docs_r1.md`)。この 16 件の直しは'
         'どれも jetformer-web の再起動が要らない。**drive.py の説明の文(02:33)は 10/28 08:40 の Jetson の起動し直しで入った**(電源断か'
         'リセット。jetformer-web は 08:40:33 に今のコードで起き、「起動より新しいコードなし」= 上の行の「反証 8 回目のあとにもう一度再起動」は'
         '要らなくなった。6.2 の 8・32)。文書の同期は '
         '`fix8/docs_sync_r1.md`。ユーザーに伝えることは 6.2 の 33。')
RD_BOOT8_X = '| ユーザー・main session | ユーザーが車のそばにいる日 |'
RD_BOOT8_Y = ('| ユーザー・main session | ユーザーが車のそばにいる日(drive.py の説明の文(02:33)のぶんは **10/28 08:40 の Jetson の起動し直しで入った**'
              '(jetformer-web 08:40:33。3.1 の「起動より新しいコードなし」)= もう一度の再起動は要らなくなった。6 章の頭。10/28 fix8 docs 同期 1) |')
RD_BOOT32_X = '手順は `todo_user_1026.md` 4 章の 1〜5)'
RD_BOOT32_Y = ('手順は `todo_user_1026.md` 4 章の 1〜5 → **10/28 08:40 の Jetson の起動し直しで入った** = この再起動は要らなくなった。10/28 fix8 '
               'docs 同期 1)')
RD_611_OLD = '| 11 | スマホで: ±0.01 を速く押して数字が飛ばない・「画面の版 9」・起動し直しの文が読める・「今日」の印・ブロックで dry が外れる |'
RD_611 = ('| 11 | スマホで: ±0.01 を速く押して数字が飛ばない・「画面の版 9」・起動し直しの文が読める・「今日」の印・ブロックで dry が外れる'
          '(10/28 fix8 ui r1 で style.css のトーストの幅を直した: 幅 420 px の画面で ▶ の断りの出し直し(約 200 字)が幅 378・高さ 176 px(前は'
          ' 210×313)。見える高さが 760 px 以下の画面で ▶ を真ん中にすると、出ている 6 秒の間は「停止中」のバッジの行にまだ重なりうる(文の頭が'
          '「走行は始まっていません(停止中)」。消えればバッジが見える)。スマホでは長い文が画面の幅いっぱいで読めるかを見る。画面の版 9'
          '(app.js 10/28 04:54)に読み直してから) |')
RD_633_X = '・試験一式・commit | main session | 11/3 の前 |'
RD_633_Y = (RD_633_X + '\n| 33 | **ユーザーに伝えること**(10/28 fix8 safety r1。`fix3/approvals.md` 3 の「困ること(ON)・10/23 の形」・0 章の 3): ユーザーが '
          'approvals 3 の (c) を決めたとき(10/27 15:4x)の「困ること(ON)」に無かった形。10/23 の形(前の物の手前でブレーキのあと止まり、'
          'すぐ下がる形)では、見張り ON は止まった所から約 1 m より近い後ろの物に OFF より強く当たることがあり、約 1 m より遠い物には ON の'
          'ほうが遅く当たるか当たらない(閉じたループの偽物の ESC の模型 18 通り・11,232 組。本物の 10/10〜10/24 では 10/23 No.6 の 1 回だけで、'
          '後ろの物は約 1.16〜1.22 m = ON のほうが良い距離)。11/3 のコードは変えない(main session の決め)。守りは手順書 7 章・3 章(止まった'
          '所の後ろ約 1 m 以内に物があるときは、リカバリーを待たずに人側にしてよい)。直す案は 11/3 のあと(6.3 の 34)。(c) を変えたいときは'
          '一言 | main session → ユーザー | 11/2(返事が無ければ (c) のまま) |')
RD_634_OLD = '夜の道具の印の付かない走り直しのうち、リカバリーの途中で人側にしたあとは印が付くようになった(approvals 8 の ①) |\n'
RD_634 = (RD_634_OLD + '| 34 | 10/23 の形の後ろの近い物の安全の案 **qC**(見張り ON の日だけ。段を飛ばさず数えるだけ・back_early のあとの \'gap\' の惰性は'
          '数えない = 書く値が惰性に移るまで OFF と同じ。閉じたループ 11,232 組のどれでも OFF より 0.1 m/s を超えて速くない・下がる距離の中央 '
          '1.38 m。1.0 m 以上の後ろの物に今の ON より速い組がある・fix6 の値の止まりの窓の下限より短く下がる) | '
          '`fix8/proposal_safety_after1103.md` 9 章(9.6: 11/3 は今のまま。11/3 の記録で 9.5 の ①②(10/23 の形のリカバリーで、後退側で下がり'
          '始めるまでの秒と、OFF の \'gap\' の中立の間に後ろへの速さが落ちるか・後ろの物までの距離)を見てから、main session とユーザーが入れる'
          'かを決める。入れるにはコード・試験・jetformer-web の再起動) |\n'
          '| 35 | 反証 8-1 回目で残したこと(画面: 見える高さ 760 px 以下ではトーストが 6 秒「停止中」のバッジに重なる・走行タブに戻った直後の'
          '窓の間はスイッチの確認の文が古い左の値を出しうる(▶ の確認の文と走る値は正しい)・② の回の形(天井・左など)の印は回の最初の本と'
          '比べる(⑤ は形が分かれたら満たさない = 安全側)・② の「ON がいちばん多い本」の外の形(1 本だけ余分に ON)・`tests/test_vbat_ui.py` の '
          'Firefox の 1 本の古い期待(4.0 m/s。いつもは飛ばす)。夜の道具: 0.15〜0.22 秒の短いカメラの止まりは注意の行だけ・`--skip` は不具合も'
          '外す。安全: 案の比べは偽物の ESC と開いたループの記録の再生だけ。文書: 8.1 の塊に `cat $R/standing_3lap.md` の行は無い(表の欄と'
          '注の行に書いた)) | `fix8/fix_ui_r1.md` 11 章・`fix8/fix_tools_r1.md` 7 章・`fix8/fix_safety_r1.md` 6 章・`fix8/fix_docs_r1.md` '
          '6 章・`fix8/docs_sync_r1.md`。どれも 11/3 のあとでよいと係がみた |\n')
RD_7_OLD = ('| `fix8/proposal_safety_after1103.md` | 11/3 のあとに向けた安全の残り(approvals 14 の 1・2 行目)の直し方の案(10/28 03:25。'
            'safety-proposal の係)。案 pA・pB・pC を写しで作り、閉じたループと本物の記録で比べた。おすすめ pB を見張り ON の日だけ・入れるのは '
            '11/3 のあと |\n')
RD_7 = (RD_7_OLD + '| `fix8/fix_ui_r1.md`・`fix8/fix_tools_r1.md`・`fix8/fix_safety_r1.md`・`fix8/fix_docs_r1.md`・`fix8/docs_sync_r1.md` | '
        '反証 8-1 回目の直し(10/28。画面 3・夜の道具 2・安全 2・文書 9 = 16 件)と、その文書の同期(同期 1)。`fix8/proposal_safety_after1103.md`'
        ' には 10/28 に 9 章(10/23 の形の案 qC・qA。すすめは qC を見張り ON の日だけ・入れるのは 11/3 のあと)が足された(0〜6 章の「本物の'
        '記録 235 本」は 235 の場面 = 9 章 9.7) |\n')

# ==================================================================================================== approvals
AP_LOG_OLD = '- **10/28 05:2x fix8 係 docs 直す 8-1 回目**(控え `approvals.md'
AP_LOG = ('- **' + STAMP + ' fix8 係 docs 同期 1**(反証 8-1 回目の直しの文書の同期。控え `approvals.md' + BAK + '`。`fix8/docs_sync_r1.md`): '
          '**3 に知らせ 1 つ**(新しい行「困ること(ON)・10/23 の形」と 0 章の 3。(c) を決めたときの「困ること(ON)」に無かった形: 10/23 の形では'
          '見張り ON は止まった所から約 1 m より近い後ろの物に OFF より強く当たることがある。11/3 のコードは変えない = main session の決め。'
          '返事が無ければ (c) のまま)・3 の「途中で OFF にした日」の決まりの行に OFF の本のカメラの止まり(fix8 tools r1)・8 の表に形と'
          'ゲインの違う試しと、カメラが止まった発進(fix8 tools r1)・14 の 2・3 行目の「本物の記録 235 本」を 235 の場面と、場面をつないだ'
          '再生の 1 回(fix8 safety r1)・**10/28 08:40 の Jetson の起動し直し**(2 の表に「今(10/28 08:40)」の行・下の「今」の行に印。'
          'drive.py の説明の文も入った = 反証 8 回目のあとのもう一度の再起動は要らなくなった)。答えがまだなのは 4・7〜11・13 のまま(下の'
          '「今(10/28 03:3x)」の行)。\n' + AP_LOG_OLD)
AP_BOOT0_X = 'のため、反証 8 回目のあとにもう一度 = 2 の「今」)'
AP_BOOT0_Y = ('のため、反証 8 回目のあとにもう一度 = 2 の「今」→ **10/28 08:40 の Jetson の起動し直しで入った**(jetformer-web 08:40:33。'
              '「起動より新しいコードなし」= 2 の「今(10/28 08:40)」の行。10/28 fix8 docs 同期 1))')
AP_BOOT2_OLD = '| 何を | `sudo systemctl restart jetformer-web`'
AP_BOOT2 = ('| 今(10/28 08:40) | **10/28 08:40 に Jetson が起動し直した**(電源断かリセット。main session の知らせ)。**' + BOOT[:BOOT.index('(')]
            + '** = 上の行の drive.py の説明の文(02:33)も入った' + BOOT[BOOT.index('('):] + '。**上の行・0 章の 2・下の「誰が」の「反証 8 回目の'
            'あとにもう一度」の再起動は要らなくなった**(このあと assist.py・drive.py・server.py・bench.py を直したら、また要る = そのときは'
            '「誰が」と同じ考えで main session)。車輪を浮かせて実際の出力で 1 回・スマホでの見え方・11/3 の朝の手順書 3.1 はまだ(10/28 fix8 docs'
            ' 同期 1) |\n' + AP_BOOT2_OLD)
# 0 章の 3(SUPERSEDE decided)
AP_03_X = '「途中で OFF にした日」の ⑤ は **(a)**(10/28 main session) | 決まり |'
AP_03_Y = ('「途中で OFF にした日」の ⑤ は **(a)**(10/28 main session)。**10/28 知らせ**: (c) を決めたときの「困ること(ON)」に無かった形'
           '(10/23 の形の後ろの近い物。3 の「困ること(ON)・10/23 の形」の行。11/3 のコードは変えない。返事が無ければ (c) のまま) | 決まり |')
# 3 の表: 困ること(ON)・10/23 の形(新しい行。困ること(ON) の行のすぐ下)
AP_3N_OLD = '| 選び方 | (a) 両方 OFF(今まで通り。'
AP_3N = ('| 困ること(ON)・10/23 の形(10/28 知らせ。fix8 safety r1) | ' + NEAR + 'では、見張り ON は \'gap\' の中立をはさまずに後退側を出し続ける'
         '(fix2 の見張りからの形。fix7 の前も同じ)。そのため、**止まった所から約 1 m より近い後ろの物には、ON のほうが強く当たることがある**。'
         '**約 1 m より遠い物には、ON のほうが遅く当たるか当たらない**(' + BAND + ')。' + REAL6 + '。**ユーザーが (c) を決めたとき'
         '(10/27 15:4x)の上の「困ること(ON)」には無かった形**。11/3 のコードは変えない(main session の決め: 近い物で悪く・遠い物で良いの'
         '入れかわりで、本物の ESC で確かめていない動きを増やさない・再起動を増やさない。`fix8/fix_safety_r1.md`)。守りは人: 止まった所の'
         'すぐ後ろ(約 1 m 以内)に物があるときは、手順書 7 章の「リカバリーで車が 1 m 以上下がる・後ろの物に近づく」の行のとおり、リカバリーを'
         '待たずに CH3 を人側にしてよい(3 章にも)。直す案は 11/3 のあと(`fix8/proposal_safety_after1103.md` 9 章の qC。見張り ON の日だけ・'
         '書く値が惰性に移るまで OFF と同じ = 11,232 組のどれでも OFF より 0.1 m/s を超えて速くない)。(c) を変えたいときは一言(返事が無ければ'
         ' (c) のまま) |\n' + AP_3N_OLD)
# 3 の「途中で OFF にした日」の決まり(SUPERSEDE decided)
AP_3D_X = ('⑤ はその発進を「R1 に届かなかった発進」(ON の組なら満たさない)、その場で戻した発進を「リカバリー中止のあとの走り直し」(比べから'
           '外す)にする =「使ってよい」の側には動かない。')
AP_3D_Y = ('⑤ はその発進を、両方 ON(明けて中止)なら「R1 に届かなかった発進」(ON の組なら満たさない)、その場で戻した発進を「リカバリー中止の'
           'あとの走り直し」(比べから外す)、**OFF(明けて走り続ける)なら「' + STALL + '」(AI 側の処理したフレームの間が 0.22 秒より長い。'
           '比べから外す。ON の本なら満たさない)**にする =「使ってよい」の側には動かない(10/28 fix8 tools r1。前は OFF の本で起きると起点まで'
           'が伸びた OFF が比べに入り「使ってよい」の側に動いた = 反証 8-1 回目 w1-tools-r8-1-stall-inflates-off。両方 ON でも止まらない形も'
           '同じに外す)。0.15〜0.22 秒の短い止まり(中立は約 0.05〜0.12 秒)は外さずに「注意: 発進から R1 の起点までにフレームの間が 0.15〜0.22 '
           '秒の発進 N 本 …」に名前が出る(発進の始めにあると境目から約 0.04 秒以内の日の答えを変えうる →「使ってよい」が境目から 0.1 秒以内なら'
           ' main session が動画で確かめて決める)。0.15 秒以下は前からの測りのぶれとして残る(本物の 10/10〜10/24 の R1 に届いた発進 38 の発進 →'
           ' R1 のいちばん長いフレームの間は 0.200 秒、比べに入る 33 は 0.133 秒。`fix8/fix_tools_r1.md` 3 章)。')
# 8 の表: 試しの本・途中で変えた本(SUPERSEDE decided)と、カメラが止まった発進(新しい行)
AP_8T_X = '空転は R1 に届いた発進だけ。10/28 main session の決め・fix8 tools。前は衝突だけ)'
AP_8T_Y = (AP_8T_X + '。形(天井・左・自動合わせ・目標)・ゲインが違って別の組になっても、同じ回・同じモデルの比べのある組の条件に数える(10/28 fix8 '
           'tools r1。反証 8-1 回目 w1-tools-r8-1-trial-form)')
AP_8S_OLD = '| ON に `launch_us` 250 と 200 が混ざった |'
AP_8S = ('| ' + STALL + '(発進から R1 の起点までの AI 側の処理したフレームの間が 0.22 秒より長い。10/28 fix8 tools r1) | ON・OFF とも比べに'
         '入れない。ON の本があれば満たさない(on_no_stall)。OFF が全部外れたら満たさない(off_no_stall)。その ON の最初の周の衝突・空転・記録の'
         '欠けは数える。0.15〜0.22 秒の短い止まりは比べに入れて注意の行に名前 |\n' + AP_8S_OLD)
# 14 の 2 行目・3 行目(SUPERSEDE final)
AP_14I_X = '本物の記録 235 本では起きていない(出す値が直す前と全部同じ)。'
AP_14I_Y = REC235 + '。'
AP_14J_X = '**本物の ESC では確かめていない**(閉じたループの偽物の ESC と、10/10〜10/24 の記録の再生だけ)'
AP_14J_Y = (AP_14J_X + '。見立てが外れる形(ESC の後退でない後ろへの動き)は本物の記録に 1 回ある: 10/24 20261024-160847 の 40.07 秒、前の物を'
            '押して止まったあとのリカバリーの \'brake\' の間に、車が受け身で後ろへ動いた(eRPM 200〜1400 = 0.1〜0.7 m/s・hall_count が一方向に'
            '進む・AI 側の間に約 0.13 m、人が CH3 を人側にした 40.45 秒のあともスロットル中立のまま動き、計 約 0.24 m)。今の ON はそこで後ろへの'
            '動きを見張りに使わず(back_no_esc)、**見張り OFF と同じ値を書いた**(7-1 回目の前の ON は back_early で後退側を出し続けた)。場面を'
            'つないだ再生で見つけた(場面ごとの再生では場面のさかい目で見落としていた。10/28 fix8 safety r1)')

# ==================================================================================================== fix7/README.md(記録。印と直しだけ)
F7_LOG_X = 'ほかは 10/27 23:5x の記録のまま。'
F7_LOG = (F7_LOG_X + '\n- (' + STAMP + ' fix8 係 docs 同期 1。控え `README.md' + BAK + '`。`../fix8/docs_sync_r1.md`)1・2 章の安全の行と 3 章の安全の 2 行目の'
          '「本物の記録 235 本」を 235 の場面に(1 本の記録の場面をつないで流すと 10/24 160847 の 10 フレームだけ違い、そこで今の ON = 見張り'
          ' OFF。`../fix8/fix_safety_r1.md` 1 章)・7 章 2 に 10/28 08:40 の Jetson の起動し直し(drive.py の説明の文も入った)。ほかは記録のまま。')
F7_BOOT_X = 'drive.py の説明の文のぶんは反証 8 回目のあと)'
F7_BOOT_Y = ('drive.py の説明の文のぶんは反証 8 回目のあと → **10/28 08:40 の Jetson の起動し直しで入った**(jetformer-web 08:40:33。「起動より新しい'
             'コードなし」。`../fix8/docs_sync_r1.md`))')
F7_1_OLD = '本物の記録 235 本で出す値が 1 フレームも変わらない。'
F7_1 = ('本物の記録の 235 の場面(238 のうち空の 3 つを除く。記録の本数ではない)を場面ごとに作り直して流すと、出す値は 1 フレームも変わらない。'
        '**ただし 1 本の記録の場面をつないで流すと(59 本・同じ 122,767 フレーム)、10/24 20261024-160847 の 40.07〜40.37 秒の 10 フレームだけ'
        '違う**: 直す前は back_early で \'gap\' を飛ばして後退側(舵 1291・後退側 1796 µs)を出し続け、今は back_no_esc で \'brake\'(1441・1796)→'
        ' \'gap\'(1291・1506)= **見張り OFF と同じ値**。前の物を押して止まったあとの \'brake\' の間に、車は受け身で本当に後ろへ動いていた'
        '(hall_count −62909 → −62883・AI 側の間に約 0.13 m。ESC の後退ではない)。今の ON は OFF と同じなので悪くはなっていない(10/28 fix8 '
        'safety r1。`tests/test_rec_fix8_safety_r1.py` が固定。`fix_safety_r1.md` 0・3 章の「235 本」も場面の数 = 報告は記録のまま)。')
F7_2_OLD = '本物の記録 235 本で出す値が 1 フレームも違わない。'
F7_2 = ('本物の記録の 235 の場面(場面ごと)でも、場面をつないだ 59 本でも、出す値が 1 フレームも違わない(10/28 fix8 safety r1 がつないで'
        '照らした。235 は場面の数。`fix_safety_r2.md` 0・3・6 章の「235 本」も同じ)。')
# 3 章の安全の 2 行目の右の欄(SUPERSEDE r81)
F7_3_X = '| 本物の記録 235 本では起きていない。直す前は逆に後退側を出し続けた。'
F7_3_Y = '| 本物の記録(235 の場面を場面ごとに、と場面をつないだ 59 本)では起きていない(10/28 fix8 safety r1)。直す前は逆に後退側を出し続けた。'

# ==================================================================================================== 一覧
ALL = [
    (RB, [(RB_LOG_OLD, RB_LOG), (RB_NEAR_OLD, RB_NEAR), (RB_OFF3_X, RB_OFF3_Y), (RB_APP_X, RB_APP_Y), (RB_7B_OLD, RB_7B),
          (RB_7P_X, RB_7P_Y), (RB_7R_X, RB_7R_Y), (RB_7F_OLD, RB_7F), (RB_2N_OLD, RB_2N), (RB_5A_OLD, RB_5A), (RB_5B_OLD, RB_5B),
          (RB_5C_OLD, RB_5C), (RB_BOOT_X, RB_BOOT_Y)]),
    (TODO, [(TD_TITLE_X, TD_TITLE_Y), (TD_LOG_OLD, TD_LOG), (TD_SCR_OLD, TD_SCR), (TD_NEAR_OLD, TD_NEAR), (TD_BOOT1_X, TD_BOOT1_Y),
            (TD_BOOT4_X, TD_BOOT4_Y)]),
    (RD, [(RD_LOG_OLD, RD_LOG), (RD_6H_X, RD_6H_Y), (RD_611_OLD, RD_611), (RD_633_X, RD_633_Y), (RD_634_OLD, RD_634), (RD_7_OLD, RD_7),
          (RD_BOOT8_X, RD_BOOT8_Y), (RD_BOOT32_X, RD_BOOT32_Y)]),
    (AP, [(AP_LOG_OLD, AP_LOG), (AP_03_X, AP_03_Y), (AP_3N_OLD, AP_3N), (AP_3D_X, AP_3D_Y), (AP_8T_X, AP_8T_Y), (AP_8S_OLD, AP_8S),
          (AP_14I_X, AP_14I_Y), (AP_14J_X, AP_14J_Y), (AP_BOOT0_X, AP_BOOT0_Y), (AP_BOOT2_OLD, AP_BOOT2)]),
    (F7, [(F7_LOG_X, F7_LOG), (F7_1_OLD, F7_1), (F7_2_OLD, F7_2), (F7_3_X, F7_3_Y), (F7_BOOT_X, F7_BOOT_Y)]),
]
# 報告の文(入れ先)→ 直し(文書, ALL の何番目)。頭の記録の行は LOGS
SOURCES = {
    'safety A(approvals 3 困ること(ON)・10/23 の形)': [(AP, 2), (AP, 1)],
    'safety B(手順書 7 章 1 m 以上下がる)': [(RB, 4)],
    'safety C(手順書 3 章 10/23 の形)': [(RB, 1), (TODO, 3)],
    'safety README 6.2 の伝えること': [(RD, 3)],
    'safety D(fix7/README 1 章)': [(F7, 1)],
    'safety E(fix7/README 2 章)': [(F7, 2)],
    'safety I(approvals 14 の 2 行目)': [(AP, 6), (F7, 3)],
    'safety J(approvals 14 の 3 行目)': [(AP, 7)],
    'tools 1(approvals 3 途中で OFF にした日)': [(AP, 3)],
    'tools 2(手順書 3 章 わざと OFF にした日)': [(RB, 2)],
    'tools 3(手順書 7 章 走った値が計画と違った)': [(RB, 5)],
    'tools 4(手順書 7 章 起こし直した)': [(RB, 6)],
    'tools 5(手順書 8.1 ⑤)': [(RB, 9), (RB, 10), (RB, 11)],
    'tools 6(approvals 8 の表)': [(AP, 4), (AP, 5)],
    'ui 1(手順書 8.1 ②)': [(RB, 8)],
    'ui 2(手順書 7 章 Failed to fetch)': [(RB, 7)],
    'ui 3・11 章 1(app.js の時刻 10/28 04:54)': [(RB, 3), (TODO, 2)],
    'ui 4(README 6.2 の 11)': [(RD, 2)],
    'proposal 9 章 qC(README 6.3)': [(RD, 4)],
    '8-1 の まとめ・残したこと(README 6 章の頭・6.3 の 35・7 章)': [(RD, 1), (RD, 5)],
    'todo の題の時刻': [(TODO, 0)],
    'main session の知らせ: 10/28 08:40 の Jetson の起動し直し(読むだけで確かめた)': [(RB, 12), (TODO, 4), (TODO, 5), (RD, 6), (RD, 7),
                                                                              (AP, 8), (AP, 9), (F7, 4)],
}
LOGS = [(RB, 0), (TODO, 1), (RD, 0), (AP, 0), (F7, 0)]

# 前の係の文(その台本の ALL の (文書, 何番目) の new の今の形)の中を書き換えた所: (台本, 文書, 何番目, 前の文 X, 今の文 Y)
SUPERSEDE = [
    ('r2', RB, 4, RB_OFF3_X, RB_OFF3_Y),        # 3 章「わざと OFF にした日」の終わり: OFF の本のカメラの止まり(fix8 tools r1)
    ('decided', RB, 2, RB_APP_X, RB_APP_Y),     # 3.1: 今の app.js の時刻(fix8 ui r1)
    ('decided', RB, 6, RB_7P_X, RB_7P_Y),       # 7 章「走った値が計画と違った」: 形・ゲインの違う試し(fix8 tools r1)
    ('r81', RB, 5, RB_7R_X, RB_7R_Y),           # 7 章「起こし直した」: 同上
    ('r81', TODO, 0, TD_TITLE_X, TD_TITLE_Y),   # todo の題の時刻
    ('decided', RD, 4, RD_6H_X, RD_6H_Y),       # README 6 章の頭: 反証 8-1 回目の直し(足すだけ)
    ('decided', RD, 10, RD_633_X, RD_633_Y),    # README 6.2: 新しい 33(ユーザーに伝えること。足すだけ)
    ('decided', F7, 0, F7_LOG_X, F7_LOG),       # fix7/README の頭の記録(足すだけ)
    ('decided', AP, 2, AP_03_X, AP_03_Y),       # approvals 0 章の 3: 知らせ
    ('decided', AP, 9, AP_3D_X, AP_3D_Y),       # approvals 3「途中で OFF にした日」の決まり: OFF の本のカメラの止まり(fix8 tools r1)
    ('decided', AP, 10, AP_8T_X, AP_8T_Y),      # approvals 8 の表「試しの本・途中で変えた本」: 形・ゲインの違う試し(fix8 tools r1)
    ('final', AP, 4, AP_14I_X, AP_14I_Y),       # approvals 14 の 2 行目: 235 本 → 235 の場面・つないだ 59 本(fix8 safety r1)
    ('final', AP, 4, AP_14J_X, AP_14J_Y),       # approvals 14 の 3 行目: 見立てが外れる形が本物に 1 回(fix8 safety r1)
    ('r81', F7, 1, F7_3_X, F7_3_Y),             # fix7/README 3 章の安全の 2 行目: 235 本 → 235 の場面
    # 10/28 08:40 の Jetson の起動し直し(「反証 8 回目のあとにもう一度再起動」に「入った」の印。足すだけ)
    ('decided', RB, 2, RB_BOOT_X, RB_BOOT_Y),   # 手順書 3.1 の 00:53:23 の行
    ('decided', TODO, 2, TD_BOOT1_X, TD_BOOT1_Y),     # todo 1 章の再起動の行
    ('decided', TODO, 12, TD_BOOT4_X, TD_BOOT4_Y),    # todo 4 章の頭
    ('decided', RD, 10, RD_BOOT32_X, RD_BOOT32_Y),    # README 6.2 の 32
    ('decided', AP, 0, AP_BOOT0_X, AP_BOOT0_Y),       # approvals の頭の「今(10/28 03:3x)」の行
    ('decided', F7, 2, F7_BOOT_X, F7_BOOT_Y),         # fix7/README 7 章の 2
]
# 前の文を消す書き換え(X の文が Y に残らない)
REMOVES = {(TODO, TD_TITLE_X), (AP, AP_3D_X), (AP, AP_14I_X), (F7, F7_3_X)}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prev_mods():
    return {k: _load('docs_prev_%s_from_s1' % k, p) for k, p in PREV.items()}


def prev_new(mods, k, f, i):
    """前の台本 k の (文書 f, 何番目 i) の new の、この組の前の今の形(r1・r2・仕上げは fix8 docs-decided、仕上げ・docs-decided は
    fix8 docs r1 の書き換えのあと)"""
    new = dict(mods[k].ALL)[f][i][1]
    if k in ('r1', 'r2', 'final'):
        new = mods['decided'].effective(k, f, i, new)
    if k in ('final', 'decided'):
        new = mods['r81'].effective(k, f, i, new)
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
    """書き換えを戻した文書(revert)で、前の台本の new(今の形)がどれもちょうど 1 回か。戻り値は問題の一覧(この組の足しだけの直しが
    前の new を崩していないか・SUPERSEDE の X が前の new の中にちょうど 1 回か)"""
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
    - prior_news_problems(戻した文書で前の 5 つの台本の new がちょうど 1 回・今の文書で書き換えたあとの new がちょうど 1 回)
    - この組の書き換えを戻した文書で、fix8 docs r1 の 26 個の直しが「済」・その台本の earlier_problems(fix8 docs-decided の 68 個・
      fix7 docs r1・r2・仕上げの直しと、試験の確かめ f1〜f5・g1〜g6・h1〜h5・i1〜i6・j0〜j9)が空
    - 今の文書で、前の係の試験の確かめ(f・g・h・i・j)と、この組の試験の確かめ(試験のファイルがあれば)が空・SUPERSEDE が入っている"""
    mods = prev_mods()
    bad = [('news',) + tuple(x) for x in prior_news_problems(texts, mods)]
    r81 = mods['r81']
    base = revert(texts)
    _, rep = r81.apply_text(base)
    bad += [('r81', f, o[:30], st) for f, o, st in rep if st != 'done']
    bad += [('r81 superseded(戻した文書)',) + tuple(x) for x in r81.superseded_present(base)]
    bad += [('r81.earlier_problems(戻した文書)',) + tuple(x) for x in r81.earlier_problems(base, tmp_root)]
    sys.path.insert(0, str(JF))
    sys.path.insert(0, str(JF / 'tools'))
    sys.path.insert(0, str(JF / 'tests'))
    ts = {k: _load('t_%s_from_s1' % k, TESTS[k]) for k in ('r1', 'r2', 'final', 'decided', 'r81')}
    with tempfile.TemporaryDirectory(prefix='docs_fix8_sync1_', dir=tmp_root) as tmp:
        for k, t in ts.items():
            for name, fn in sorted(t.CHECKS.items()):
                ps = fn(texts, Path(tmp) / name) if k == 'r2' else fn(texts)
                bad += [('%s %s(今の文書)' % (k, name), p) for p in ps]
    bad += [('final h5(今の文書)', p) for p in ts['final'].h5_problems(texts[F7])]
    if TESTS['s1'].exists():
        t6 = _load('t_s1_from_s1', TESTS['s1'])
        for name, fn in sorted(t6.CHECKS.items()):
            bad += [('s1 ' + name, p) for p in fn(texts)]
    bad += [('superseded',) + tuple(x) for x in superseded_present(texts)]
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args(argv)
    tmp_root = E / 'fix8' / 'tmp_docs_sync1'
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
