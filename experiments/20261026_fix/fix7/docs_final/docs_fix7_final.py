#!/usr/bin/env python3
"""fix7 係 docs の仕上げ(2026-10-27 23:5x): 反証 7 回目の 2 巡の直しの報告(fix7/fix_*_r1.md・fix_*_r2.md)の「手順書に足す文」と
「main session へ」を、ユーザーが読む文書 4 本(手順書・todo・README・fix3/approvals.md)に入れる。コードは変えない。

- 前の係(fix7 docs r1・r2)が入れた文は二重にしない。入っていなかったものだけ足す(どれも「足す形」= old を残して後ろに足す)。
- 置き換えの決まりは前の台本と同じ: old が文書の中でちょうど 1 回・new が 0 回のときだけ入れる。2 回目に流すと全部「済」。
- 入れる前に、7-1 回目(docs_fix7_r1.py)・7-2 回目(docs_fix7_r2.py)の直しが全部「済」で new の文がちょうど 1 回のまま
  (= 前の係の試験が通る形)か、前の係の試験の確かめ(f1〜f5・g1〜g6)が空のままかを照らし、1 つでも崩れたら書かない。
- --apply: 控え <名前>.bak-20261027-fix7-final を作ってから書く(控えがあれば上書きしない)。

使い方: python3 -I docs_fix7_final.py [--apply]
"""
import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPORTS = Path('/home/jetson/jetformer_reports/fix_20261026')
JF = Path('/home/jetson/jetformer')
R2 = JF / 'experiments' / '20261026_fix' / 'fix7' / 'docs_r2' / 'docs_fix7_r2.py'
T1 = JF / 'tests' / 'test_fix7_docs_r1.py'
T2 = JF / 'tests' / 'test_fix7_docs_r2.py'
BAK = '.bak-20261027-fix7-final'
STAMP = '10/27 23:5x'
RB, TODO, RD, AP = 'runbook_20261103.md', 'todo_user_1026.md', 'README.md', 'fix3/approvals.md'
FILES = (RB, TODO, RD, AP)

# ---------------------------------------------------------------------------------------------------- 手順書
RB_LOG_OLD = '\n- **10/27 22:xx fix7 係 docs 直す 7-2 回目**(`fix7/fix_docs_r2.md`。'
RB_LOG = ('\n- **' + STAMP + ' fix7 係 docs の仕上げ**(下の 7-2 回目のあと。`fix7/README.md`。控え `runbook_20261103.md' + BAK + '`。'
          '手順書だけ。コードは変えていない): 反証 7 回目の 2 巡の直しの報告(`fix7/fix_*_r1.md`・`fix_*_r2.md`)の「手順書に足す文」が'
          'どれも入っているかをコードと記録で照らし、入っていなかった 3 つを足した: 3 章の最初のリカバリーに、7-2 回目の形(ブレーキの'
          '途中にカメラが一瞬止まった直後)も直ったことと、見張りが OFF でも同じ形になるまれな道が残ること(`fix3/approvals.md` 14)・'
          '3.1 に jetformer-web だけを再起動した日の journal(todo 4 章の 3)・4 章 6 に、中止のあとに ■ で新しい本にするときも HS に'
          '置いてから ▶(置かずに走った本は記録から分からない)。反証 7 回目のまとめ(残すこと・了承・再起動が要るか)は `fix7/README.md`。')
# 3 章の最初のリカバリーの行・わざと OFF にした日の行は、7-1 回目・7-2 回目の new の文が行の終わり(と次の見出し)まで続くので、
# 最初のリカバリーの行の上に 1 行足す
RB_FIRST_OLD = '\n  - 見張り・残すが ON の日(11/3)は、**その日の最初のリカバリーを人が見る**'
RB_FIRST_ADD = ('\n  - **最初のリカバリーの「おかしな動き」の今**(すぐ下の行。10/27 fix7 docs の仕上げ): 前の壁に当たって止まったあと下がらずに'
                '前進で壁を押す形は、反証 7-1 回目の形に加えて 7-2 回目の形(ブレーキの途中にカメラが一瞬止まって中立止めが入った直後)も、'
                'fix7 安全の 7-2 回目の直しが入ったサーバー(3.1)では起きないはず。ただし、\'gap\'(中立 0.15 秒)のあとすぐには後退に入れない '
                'ESC で、最初の後退の 0.35 秒の間に車輪の値(eRPM)が一瞬跳ねると、**見張りが OFF でも同じ形になる道が残る**(リカバリーの'
                '約 0.3% の見込み。fix7 の前からある形で、直していない。`fix3/approvals.md` 14)。どちらか分からなくても、出たらすぐ下の行の'
                'とおり(すぐ人側 → 止めて OFF → 紙に)。')
RB_J_OLD = 'どれも止まっている車で、CH3 は人側のまま。\n'
RB_J_ADD = ('**jetformer-web だけを再起動した日**(Jetson を入れ直さない日。10/27 19:36 からの fix7 安全の直しを入れる、もう一度の再起動'
            'など)は、`-b` でなく `todo_user_1026.md` 4 章の 3(起動の時刻 ActiveEnterTimestamp から)で見る(`-b` は Jetson を入れて'
            'からの全部 = 前のサーバーの行も出す。10/27 fix7 docs の仕上げ)。')
RB_HS_OLD = '6. 車を HS のいつもの場所に止めて、まっすぐ置く。'
RB_HS = ('6. 車を HS のいつもの場所に止めて、まっすぐ置く(リカバリー中止・「中立で停止中」のあとに ■ で止めて新しい本にするときも同じ。'
         '置かずに ▶ すると、その本の発進は夜の道具がふつうに数え、置いていないことは記録から分からない。7 章の中立止めの行。'
         '10/27 fix7 tools r1)。')

# ---------------------------------------------------------------------------------------------------- todo
TD_LOG_OLD = '\n- **10/27 22:xx fix7 係 docs 直す 7-2 回目**(`fix7/fix_docs_r2.md`。'
TD_LOG = ('\n- **' + STAMP + ' fix7 係 docs の仕上げ**(下の 7-2 回目のあと。`fix7/README.md`。控え `todo_user_1026.md' + BAK + '`): '
          '1 章・2 章に `fix3/approvals.md` 14(反証 7 回目で残した安全の形の知らせ。返事が無ければこのまま)と 8 に足した読み方の残り / '
          '3 章に、最初のリカバリーの「おかしな動き」が見張り OFF でもまれに出る道が残ることと、■ で新しい本にするときも HS に置いてから ▶。')
TD_DUE_OLD = '5(走行中は ON にできない決まり)は再起動で入った(戻すなら言ってください) |'
TD_DUE = ('5(走行中は ON にできない決まり)は再起動で入った(戻すなら言ってください)。14(10/27 fix7 で残した安全の形の知らせ)と 8 に'
          '足した「fix7 で残した読み方」は、返事が無ければこのまま |')
TD_ROW_OLD = '知らせ(⑤ の落ちた ON の本・④ の人側にした直後の当たり・電源断の events.jsonl)は同 12 |\n'
TD_ROW = ('| 16 | 10/27 fix7(反証 7 回目)で残した安全の形(\'gap\' のあとすぐ下がれない ESC の最初の後退の跳ね = 見張り OFF でも同じ・'
          '本当に下がり始めた車のやり直し・ESC の見立て・後ろ 1.2〜1.35 m の物)と、回 3 の判定の読み方の残り 3 つ | '
          '`fix3/approvals.md` 14・8。おすすめ このまま(11/3 は今のコード。返事が無ければこのまま) |\n')
# 3 章の最初のリカバリーの行は 7-2 回目の new が次の行の頭まで続くので、3 章の終わりに 1 行足す
TD_ODD_OLD = '最初のリカバリーで見張り・残すを OFF にした日は、その 2 つは入れ直さない(手順書 3 章)。\n'
TD_ODD = ('- (10/27 fix7 docs の仕上げ)最初のリカバリーの「おかしな動き」(前の壁に当たって止まったあと、下がらずに前進で壁を押す)は、'
          'fix7 安全の直しが入ったサーバーでは主な道(反証 7-1・7-2 回目の形)が無くなるが、\'gap\' のあとすぐには後退に入れない ESC で、'
          '見張りが OFF でもまれに出る道が残る(リカバリーの約 0.3%。直していない。`fix3/approvals.md` 14)。出たら同じく人側 → 止めて '
          'OFF → 紙に(手順書 3 章)。\n')
TD_HS_OLD = '戻したら紙に「走り直し: 本・時刻」。手順書 7 章。10/27 fix7 tools r1・r2)。\n'
TD_HS_ADD = ' ■ で止めて新しい本にするときも、HS に置いてから ▶(置かずに走った本は、記録からは分からず夜の道具が数える。手順書 4 章 6)。'

# ---------------------------------------------------------------------------------------------------- README
RD_LOG_OLD = '\n- **10/27 22:xx 追記**(fix7 係 docs 直す 7-2 回目。'
RD_LOG = ('\n- **' + STAMP + ' 追記**(fix7 係 docs の仕上げ。下の 7-2 回目のあと。控え `README.md' + BAK + '`): 6 章の頭に反証 7 回目'
          '(2 巡・19 件。2 巡目にも 10 件 = 出切っていない)・6.2 の 29・30・6.3 の 31・7 章に **`fix7/README.md`**(反証 7 回目のまとめ・'
          '残すこと・了承・再起動が要るか)。')
RD_HEAD_OLD = '**6 回目の 2 巡目にも新しい指摘が 5 件出た = 出切っていない**(6.2 の 26)。\n'
RD_HEAD = ('- 反証 7 回目(2 巡・指摘 19 件 = 1 巡目 9・2 巡目 10。19 件とも直した → fix7。まとめは **`fix7/README.md`**)。'
           '**2 巡目にも新しい指摘が 10 件出た = まだ出切っていない**(6.2 の 29)。fix7 の走りの直し(`assist.py`・`drive.py`。後ろの見張り'
           ' ON の日だけ効く)は 17:16 の起動に入っていない(6.2 の 8)。残したことの知らせは `fix3/approvals.md` 14・8(6.2 の 30)。\n')
RD_62_OLD = ('| 27 | fix6 で係が決めた値・試験の決まりを緩めた所の了承と、夜の道具の知らせ(`fix3/approvals.md` 13・12) | ユーザー | '
             '11/2 |\n')
RD_62 = ('| 29 | 反証 7 回目の 2 巡目にも新しい指摘が 10 件出た(出切っていない)。もう 1 巡するか決める。走りの直し(assist.py・drive.py・'
         'server.py・bench.py)が出たら、入れたあとに再起動(`fix7/README.md` 5・7 章) | main session | なるべく早く(11/3 の前の'
         'もう一度の再起動は今も要る = 6.2 の 8) |\n'
         '| 30 | 反証 7 回目で残した安全の形の知らせと、回 3 の判定の読み方の残り 3 つ(`fix3/approvals.md` 14・8) | ユーザー | '
         '11/2(返事が無ければこのまま) |\n')
RD_63_OLD = '`fix6/README.md` 3 章。どれも 11/3 のあとでよいと係がみた |\n'
RD_63 = ('| 31 | 反証 7 回目で残したこと(\'gap\' のあとすぐ下がれない ESC の最初の後退の跳ね(見張り OFF でも同じ)・本当に下がり始めた車の'
         'やり直し・ESC の見立て・画面の一覧の読み直しは 1 回だけ・夜の道具の印の付かない走り直し・`drive.py` の説明の文・`runs_1103` の'
         '本「—」の区切りと印の文・`disk_check.py` の `PYTEST_GB` など) | `fix7/README.md` 3 章。安全の形と読み方の知らせは '
         '`fix3/approvals.md` 14・8 |\n')
RD_7_OLD = '| `fix7/docs.md` | 文書の取りまとめ(10/27 17:3x。変えた所・確かめた元・入れなかった案) |\n'
RD_7 = ('| **`fix7/README.md`** | **反証 7 回目(2 巡・19 件)の直しのまとめ**(' + STAMP + '、fix7 係 docs の仕上げ)。係ごとの報告 '
        '`fix7/fix_{safety,ui,tools,docs}_r1.md`・`_r2.md` への入口・残すこと・了承・再起動が要るか |\n')

# ---------------------------------------------------------------------------------------------------- approvals
AP_NOW_OLD = '・8 の表に 2 行(形の分かれ・中止のあとの走り直し)。\n'
AP_NOW = ('- **' + STAMP + ' fix7 docs の仕上げ**(控え `approvals.md' + BAK + '`): **新しい 14**(反証 7 回目で残した安全の形の知らせ。'
          '返事が無ければこのまま)・8 に「fix7 で残した読み方」3 つ・2 の「誰が」にもう一度の再起動の了承のこと。まとめは `fix7/README.md`。\n')
AP_T_OLD = '| 13 | 10/27 fix6 で係が決めた値(後退の見積もり)と、試験の決まりを緩めた所 | **了承** | 11/2 |\n'
AP_T = '| 14 | 10/27 fix7(反証 7 回目)で残した安全の形(知らせ) | **このまま**(11/3 は今のコード。返事が無ければこのまま) | 11/2 |\n'
AP_WHO_OLD = '| 誰が | ユーザーか main session。手順は `todo_user_1026.md` 4 章 |'
AP_WHO = ('| 誰が | ユーザーか main session。手順は `todo_user_1026.md` 4 章。**もう一度の再起動(「今」の行。fix7 安全の直しを入れる)に、'
          '10/27 15:4x の了承(「jetformer-web を再起動してもいいですよ」)が当たるかは main session が決める**(当たらないとみるなら、'
          'ユーザーに一言聞いてから。再起動が要るわけは `fix7/README.md` 5 章。' + STAMP + ' fix7 docs の仕上げ) |')
AP_8_OLD = '「このまま」はこの 2 行も含む(どちらも回 4 が OFF になりやすい側・比べから外す側)。\n'
AP_8 = ('- (' + STAMP + ' fix7 docs の仕上げ。**fix7 で残した読み方**。今の道具の形のまま =「このまま」に入る。変えるなら main session が'
        '決め、11/3 の前に夜の道具を直す。`fix7/fix_tools_r1.md` 10 章 2・`fix7/fix_tools_r2.md` 10 章 2・3)① リカバリーの途中(後退など)'
        'で人が CH3 を取った区間のあとの発進には走り直しの印が付かず、⑤ の比べに入る(前と同じ。本物の 10/10〜10/24 で 7 区間。多くは'
        '当たったあと = ③ は「前に衝突」で数えない。手順書 7 章の「当たって止まったあと」の行 = 人側で HS のいつもの場所に戻してから、で'
        '守る)② 試しの本・途中で変えた本の ON の空転・最初の周の記録の欠けは比べの条件に入らない(衝突だけ数える。fix3 からの形。'
        '走り直しの ON は数える)③ 0.5 秒より短い中立止めのうちに人側にしたあと・人側にしたとき止まっていなかった区間のあとの走り直し'
        'には印が付かない(手順書 7 章のとおり紙に「走り直し」→ main session が `--skip 記録名@AI側にした秒`)。\n')
AP_END_OLD = 'fix5 の報告そのものは変えていない(`fix6/fix_safety_r1.md` 7 章) | — |\n'
AP_14 = (
    '\n## 14. 10/27 fix7(反証 7 回目)で残した安全の形(知らせ。返事が無ければこのまま)\n\n'
    '- fix7 の安全の直し(`fix7/fix_safety_r1.md` の s4-a・s4-b、`fix7/fix_safety_r2.md` の s4-c)は、どれも後ろの見張り'
    '(`rec_back_guard`)が ON の日だけ効く。OFF なら PCA9685 に書く値・記録は直す前とビット単位で同じ。17:16 の起動には入っていない'
    '(2 の「今」・`fix7/README.md` 5 章)。下は係が直さずに残したもの。**どれも 11/3 は今のコードのまま**(直すにはコード・試験・'
    '再起動が要るので、直すなら 11/3 のあと)。元: `fix7/fix_safety_r1.md` 6 章・`fix7/fix_safety_r2.md` 6 章。\n'
    '- 答え方: 「14 はこのまま」か、変えたい行を一言。返事が無い間はこのまま。\n\n'
    '| こと | 中身 | おすすめ |\n'
    '|---|---|---|\n'
    '| \'gap\' のあとすぐには下がれない ESC の、最初の後退の中の跳ね | リカバリーは前の壁に当たったあと、ブレーキ → 中立(\'gap\' '
    '0.15 秒)→ 後退。\'gap\' ではまだ後退に入れない ESC(10/10 のログで、最初の \'gap\' では下がれないのは 11%)で、最初の後退の '
    '0.35 秒の間に車輪の値(eRPM)の 1 サンプルの跳ねがフレームに入ると、「下がった」とみて「下がらなければやり直す」が止まり、0 m の'
    'まま後退側を出したあと escape が前の壁を押す(手順書 3 章の「おかしな動き」と同じ形)。閉じたループの偽物の ESC で 41 組のうち '
    '22 組。**見張り ON・見張り OFF・fix7 の前のどれも同じ 22 組**(前からある形。OFF の形をビット単位で保つ決まりなので変えていない)。'
    '起きる割合はリカバリーの約 0.3% の見込み(下がれない ESC 11% × その 0.35 秒に跳ねがフレームに入る割合)。ON の日だけ直す案'
    '(後退の中の後ろへの動きが 1 フレームだけで、次のフレームで止まっていれば下がったとみない)はあるが、前からの決まり(1 フレームの'
    '後退でやり直さない)を変える | **このまま**(11/3 は直さない。手順書 3 章の「最初のリカバリーを人が見る」・7 章の「押し続けたら'
    'すぐ人側」で守る。直すかは 11/3 の記録を見て決める) |\n'
    '| \'gap\' を飛ばした後退で、本当に下がり始めた車のやり直し(fix7 r2 の直しの残り) | fix7 r2 の直しは、\'gap\' を飛ばして入った後退で、'
    '0.35 秒のあいだ「本当に下がっている(0.2 秒続けて後ろへ動く)」を見なければ中立からやり直す。本当に後ろへ動き始めたのに 0.2 秒'
    '続く前に一度でも車輪の値が 300 以下になり、それが 0.35 秒のあとだと、やり直して下がった時間を捨て、最大 0.2 秒の後退ぶん多く'
    '下がる。起きるのは、下がっている途中で後ろに当たる・引っかかるときなど(強い当たりは後ろの当たりとして先に前へ出る)。本物の'
    '記録 235 本では起きていない(出す値が直す前と全部同じ)。直す前は逆に、この形でやり直さず後退側を出し続けた(後ろの物を押す) | '
    '**このまま**(手順書 7 章「1 m 以上下がる・後ろの物に近づく → すぐ人側」) |\n'
    '| ESC の見立て(`EscMayReverse`) | fix7 r1 の直しは「この ESC は前へ走った直後に後退側を出しても下がらない(ブレーキになる)。'
    '下がるのはブレーキ → 中立 → 後退側の順で、止まっているか後ろへ転がっているときだけ」(10/10 のログ: 前進の直後の後退側は人の '
    '29 回とも下がらなかった)とみて、その手順でまだ ESC が後退で走らせていない間(\'brake\'・\'wait\'・\'gap\')の後ろへの動き(車輪の'
    '値の跳ね・たわみの戻り)を見張りに使わない(出来事 `back_no_esc`)。見立てが外れて、その間に本当に下がっていたら、その後ろへの'
    '動きと後ろの当たりを見張りが数えない(見張り OFF と同じ形)。人が後ろへ押した・坂で下がったときも同じ。**本物の ESC では確かめて'
    'いない**(閉じたループの偽物の ESC と、10/10〜10/24 の記録の再生だけ) | **このまま**(3 の「最初のリカバリーは人が見る」) |\n'
    '| 後ろ 1.2・1.35 m の物 | `rev_s` 0.4・`rev_us` 300 の後退は 1.1〜1.4 m 下がるので、後ろ 1.2・1.35 m の物には見張り ON でも OFF '
    'でも当たる(直しの前からある形)。fix7 r2 の直しで「0 m のまま壁を押す」だった組が下がるようになったぶん、中立 0.05 秒で下がれる'
    '偽物の ESC の 3 組で当たりが増えた(最大 1.83 m/s。直す前の同じ場面の当たりの最大と同じ) | 10 のおすすめ (a)(11/3 は変えない。'
    '11/3 のリカバリーの記録で決め直す)のまま |\n')

# (old, new) はどれも「足す形」(new の中に old が残る)
ALL = [
    (RB, [(RB_LOG_OLD, RB_LOG + RB_LOG_OLD),
          (RB_FIRST_OLD, RB_FIRST_ADD + RB_FIRST_OLD),
          (RB_J_OLD, RB_J_OLD[:-1] + RB_J_ADD + '\n'),
          (RB_HS_OLD, RB_HS)]),
    (TODO, [(TD_LOG_OLD, TD_LOG + TD_LOG_OLD),
            (TD_DUE_OLD, TD_DUE),
            (TD_ROW_OLD, TD_ROW_OLD + TD_ROW),
            (TD_ODD_OLD, TD_ODD_OLD + TD_ODD),
            (TD_HS_OLD, TD_HS_OLD[:-1] + TD_HS_ADD + '\n')]),
    (RD, [(RD_LOG_OLD, RD_LOG + RD_LOG_OLD),
          (RD_HEAD_OLD, RD_HEAD_OLD + RD_HEAD),
          (RD_62_OLD, RD_62_OLD + RD_62),
          (RD_63_OLD, RD_63_OLD + RD_63),
          (RD_7_OLD, RD_7_OLD + RD_7)]),
    (AP, [(AP_NOW_OLD, AP_NOW_OLD + AP_NOW),
          (AP_T_OLD, AP_T_OLD + AP_T),
          (AP_WHO_OLD, AP_WHO),
          (AP_8_OLD, AP_8_OLD + AP_8),
          (AP_END_OLD, AP_END_OLD + AP_14)]),
]


def apply_text(texts, files=ALL):
    """(直した文, [(文書, old, 'apply' | 'done' | 'missing' | 'many(N)')])。前の台本と同じ決まり。
    TD_DUE・AP_WHO・RB_HS は old の後ろに足す形でないので、done は「old が new の頭で、new が 1 回」で見る"""
    out, rep = {}, []
    for f, edits in files:
        t = texts[f]
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


def earlier_intact(texts):
    """前の係の直しが崩れていないか: 7-2 回目の 45 個が「済」で new がちょうど 1 回・7-1 回目の 44 個と画面の版 9 も(r1_intact)。
    前の係の試験の確かめ(7-1 回目の f1〜f5・7-2 回目の g1〜g6)が空。戻り値は問題の一覧"""
    r2 = _load('docs_fix7_r2_from_final', R2)
    full = dict(texts)
    full['fix6/README.md'] = (REPORTS / 'fix6' / 'README.md').read_text(encoding='utf-8')
    bad = []
    _, rep = r2.apply_text(full)
    bad += [('r2', f, o[:30], s) for f, o, s in rep if s != 'done']
    for f, edits in r2.ALL:
        for old, new in edits:
            if full[f].count(new) != 1:
                bad.append(('r2 count', f, new[:30], full[f].count(new)))
    bad += [('r1',) + tuple(x) for x in r2.r1_intact(full)]
    sys.path.insert(0, str(JF))
    sys.path.insert(0, str(JF / 'tools'))
    sys.path.insert(0, str(JF / 'tests'))
    t1 = _load('t_fix7_docs_r1_from_final', T1)
    t2 = _load('t_fix7_docs_r2_from_final', T2)
    for k, fn in sorted(t1.CHECKS.items()):
        bad += [('t1 ' + k, p) for p in fn(full)]
    with tempfile.TemporaryDirectory(prefix='docs_final_') as tmp:
        for k, fn in sorted(t2.CHECKS.items()):
            bad += [('t2 ' + k, p) for p in fn(full, Path(tmp) / k)]
    return bad


def read_all(root=REPORTS):
    return {f: (root / f).read_text(encoding='utf-8') for f in FILES}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args(argv)
    texts = read_all()
    new, rep = apply_text(texts)
    for f, old, s in rep:
        print('%-7s %s: %s' % (s, f, old.strip()[:60].replace('\n', ' ')))
    bad = [x for x in rep if x[2] not in ('apply', 'done')]
    if bad:
        print('当たらない直しがある → 書かない')
        return 2
    prob = earlier_intact(new)
    for x in prob:
        print('前の係の直し・確かめが崩れる:', x)
    if prob:
        print('→ 書かない')
        return 3
    for f, edits in ALL:
        for old, nw in edits:
            if new[f].count(nw) != 1:
                print('new が 1 回でない:', f, nw[:40])
                return 4
    n_apply = sum(1 for _, _, s in rep if s == 'apply')
    print('入れる %d・済 %d・前の係の直しと確かめ: 崩れない' % (n_apply, len(rep) - n_apply))
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
