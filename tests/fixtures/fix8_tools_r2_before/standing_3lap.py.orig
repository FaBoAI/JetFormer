#!/usr/bin/env python3
"""止まってから 3 周の時間(2026-10-26 fix2・係 tools): 記録(data/raw/<日>-*/ の AI_CMD・VEHICLE・IMU・log.json)から、
止まった車を AI 側にしてから 3 周を走り終えるまでの時間・走りながらの 3 周の窓の最速・衝突・設定を出す

    python3 tools/standing_3lap.py --day 20261103                   # その日の AI 走行(dry run でない)全部 → 表(標準出力)
    python3 tools/standing_3lap.py --day 20261103 --since 1300      # 13:00 以後の記録だけ
    python3 tools/standing_3lap.py 20261024-162033 20261024-183123  # 記録を名指し
    python3 tools/standing_3lap.py --day 20261103 --json OUT.json --md OUT.md
      --line-ahead-m X   スタートの線が「車を置いた場所」の X m 先(既定 0 = 置いた場所が線)
      --raw DIR --labels DIR --bench FILE(既定 data/raw・data/labels・data/bench_candidates.json)

読むだけ(numpy。GPU なし)。data/raw・data/labels は変えない。ラベルの無い記録はその場で作る(書き出さない)。
0 バイト・壊れた npz は飛ばして problems に書く(label_events.load_log と同じ。zero_byte に名前を並べる。無い npz は missing_npz)。
log.json が読めない(0 バイトなど)記録は飛ばす(skipped)。

定義(どれも tools/ の今の決まりをそのまま使う):
  発進     AI の区間(ラベルの ai。dry run を除く)の最初のフレームの時刻 t_ai で |eRPM| < 400(止まっている)。
           t_move = t_ai のあと、前へ 400 eRPM(0.2 m/s)に初めて届いた車両データの時刻(最初に車が動いた時刻)。
  起点     R1 の起点 = section_scorecard.score_log の起点(label_events.checked_anchors。偽の起点を外したもの)。
           A1 = 発進のあと最初の起点(t_ai から 15 秒以内・AI の区間の中)、A2・A3・A4 = その次の起点。
           起点どうしが 5〜16 秒でなくなったところで切る(止まった・人が運転した)。
  線       スタートの線 = 車を置いた場所(+ --line-ahead-m)。d_pre = 線 → A1 の車輪の前進の距離。
           k 周目の終わり = 起点 A_(k+1) の d_pre m 手前を通った時刻(起点ごとに車輪の距離を合わせ直すので、空転の
           ずれが 3 周でたまらない)。3 周目の終わり = 「3 周目の終わりの起点」A4 から d_pre 戻った時刻(method = anchor)。
           A4 が無いとき(3 周目の線のあとで止めた・記録の終わり)だけ A3 + (1 周の距離の中央値 − d_pre)(method = A3+dist)。
  止まってから 3 周   from_move = 3 周目の終わり − t_move(主。車が動いた = 線を出た)。from_ai = 3 周目の終わり − t_ai。
           --line-ahead-m を付けたときは from_line = 3 周目の終わり − 線を越えた時刻 も。
           valid = 1 周ずつ(線 → 線と起点 → 起点)5〜16 秒・起点 → 起点の向きの変化 300〜420°(1 周の距離の中央値と同じ。
           偽の起点の決まりで外れなかった半周を除く)・t_ai から 3 周目の終わりまでひとつの AI の区間・
           その間のフレームと車両データの途切れが 0.2 秒以下。外れたら valid = False と理由(why)。
  距離の 3 周(参考)  dist3_from_ai = t_ai から車輪の前進の距離で「1 周の距離の中央値 × 3」を走り終えるまで。
           1 周の距離の中央値 = 起点 → 次の起点(16 秒未満・向きの変化 300〜420°)の距離の中央値。
           bench_20261024_report/analysis_tt.md 5.2 の「発進から実際に測った 3 周」(No.9 の 26.35 秒)と同じ作り方。
  走りながらの 3 周   score_log の周(起点 → 次の起点・5〜16 秒・ずっと AI・0.2 秒を超える途切れなし)の、続いた 3 周
           (前の周の終わり = 次の周の始まり、0.02 秒以内)の合計。最速・きれい(3 周とも衝突なし)の最速・中央値・窓の数。
           analysis_tt.md 3 章と同じ(窓は 1 周ずつずらすので重なる)。
  衝突     score_log の collision_events(label の決まり: 水平 2.5G・0.6 秒でまとめる・AI 側。立て直し中も入れる)と、
           人側にした直後(AI の区間の終わりから 0.5 秒以内)の人側の衝突(下の 2026-10-27 fix6-2。走りながらの 3 周の窓は AI 側だけ)。
           強い = 4G 以上。周の種類: きれい / 軽い接触(4G 未満だけ)/ 強い衝突 / リカバリーなど(衝突なしで clean でない)。
  設定     log.json の drive.params(走行開始の値): throttle_gain・steer_gain・speed_ceiling・ceil_all_mps・ceil_left_mps・
           speed_auto・auto_target_mps・launch_boost・launch_us・launch_mps・launch_max_s・zone_caps・start_kick、
           モデル(drive.run)と候補の名前(bench_candidates.json の candidates の run が同じもの)。
           欄が無い = その機能の無いコード(表では「—」。OFF と同じ)。フレームの速さのゲイン(AI のフレームの中央値)。
           steer_trim_us・rec_back_guard(critic)と、欄があるときだけ rec_hold_keep・launch_no_kick(2026-10-26 fix3 直す 1 回目
           r1-rec3-5。1 なら文に「リカバリーを残す」「見送りで補助なし」)。
  発進の前進の記録   AI_CMD の launch_on・launch_end・assist = 5(欄が無い記録は None)と events.jsonl の launch_boost。
           出来事の時刻は cam_epoch_us(AI_CMD と同じ UNIX µs。2026-10-26 fix2 から drive.py が書く)。無ければ cam_t_us が
           UNIX µs のときだけ使う(drive.py の cam_t_us は単調時計で、記録の時刻とは比べられない → 時刻なし)。
  何回目の発進(2026-10-26 fix2 R2-rec-3)  start_no = その記録の中で何回目の「止まった状態からの発進」か(1 から)。
           coll_before = その発進より前の、その記録の衝突の数。counted = valid かつ coll_before = 0(ぶつかった本の、人が
           止めて戻したあとの発進を「衝突 0 の止まってから 3 周」に見せない)。best_standing・quick_eval_by_params の 1 章・
           2 章(衝突なし・25 秒以下)は counted だけを数える。2 回目以降の発進は表の「確か」に印。
  走行中に変えた設定(2026-10-26 fix2 state2-2)  midrun = events.jsonl の params(drive.set_params が走行中に書く)・
           log.json の drive.params_end・AI_CMD の天井・自動合わせが途中から OFF(section_scorecard の ceil_off_midrun・
           auto_off_midrun)・発進の前進の終わり 10(OFF にした)・AI のフレームのゲイン・トリムが途中で変わった。
           settings_text に「途中で変えた(…)」を足す。quick_eval_by_params はその本を ON の組にも OFF の組にも入れない。
  衝突の場所(2026-10-26 fix2 R2-rec-1)  collisions に最大 G・8G 以上の数・場所ごとの数(section_scorecard.collision_summary)。
  印(note。valid は変えない)   起点 → 起点の周が 7.0 秒より速い・1 周の車輪の距離が中央値から 2 割以上ずれた
           (10/23 No.17 の 5.27 秒の周など。偽の起点の決まりでは外れない)。記録・動画で確かめる。
  空転の比(参考)    車輪の速さの増え ÷ IMU の前の加速の積分(動いてから 0.6 秒。diag_launch/collect.py と同じ)。
           1.3 を超えたら note に印(d_pre が長く出て、止まってから 3 周が短く出ているかもしれない)。
           IMU で距離を直すのはやめた(10/24 の発進で IMU の距離が車輪より 0〜29% 長く出た = 加速の沈み込みの傾きとみられる。
           直しの誤差が直したい量と同じくらい)。
           2026-10-27 fix3 直す 3 回目・2 組目(反証 3 回目 r3-r3-rec-4): 比は IMU の積分が 0.2 m/s 以下だと出ない(None)。前はそのとき
           印が付かず、いちばんひどい空転(車輪 +1.8 m/s・IMU +0.09 m/s)が印なしで抜けた。今は slip_kind() で読み、
           車輪の増えが 0.5 m/s より大きいのに IMU の積分が 0.2 m/s 以下(extreme)= 「> 1.3 とみなす」の印、動いてから 0.6 秒の窓
           (0.15 秒以上)に IMU が 5 サンプルも無い(IMU が抜けた)= 「確かめる: 空転の比が出ない」の印。どちらも note
           (quick_eval_by_params の「確かめる」)。車輪も IMU もほとんど増えない発進(車輪 ≤ 0.5 m/s。空転していない)・窓が短い
           (A1 が近い)・IMU の無い呼び方には印を付けない(前と同じ)。⑤ collect.py は、比が出ない ON を全部「満たさない」にする
           (条件の比が無いので)。比が出る発進は前と同じ(10/23・10/24 の 33 発進は全部比が出る)。
  試し(2026-10-27 fix3 直す 3 回目・2 組目、反証 3 回目 r3-r3-rec-3)  説明欄(log.json と走り終わりのメモ = scene.json を
           section_scorecard.effective_description でまとめたもの)の memo(と key=値 の外の文)に、人が「試し」と書いた本
           (手順書 7 章「その本は数えない(memo に「試し」)」)は、trial に memo の文を入れ、発進を数えない(counted = False・
           表の「確か」に「数えない(試し)」・設定の文に「試し」)。走行開始でサーバーが足す「計画と違うので試し(元 cand=…): …」
           (web/bench.plan_check。11/3 はゲインが 1.00 でない本の全部に付く)は試しに数えない。試しの無い本は前と同じ。
  途中で変えた発進(2026-10-27 fix5、反証 4 回目 c2)  midrun の時刻(t_s)が、その発進の 3 周目の終わり(end_s)より前か、分からない
           (log.json の params_end だけ)なら、その発進は midrun = True・counted = False(表の「確か」に「数えない(途中で変えた)」・
           注の 1 行)。前は設定の文に「途中で変えた」と書くだけで counted = True のまま、best_standing・④ の止まってから 3 周に入った
           (25 秒の判定に本番の形でない本が混ざりうる)。3 周目の終わりのあとに変えた本・変えていない本は前と同じ。
  無い npz(2026-10-27 fix5、反証 4 回目 c1)  シーンの VEHICLE・IMU・CAM_FRONT の npz が無い(書いている途中で落ちたシーン・車両の値が
           来なかったシーン)を missing_npz に並べ(あるときだけ)、表の下に「無いので飛ばした npz」と「記録が欠けている」の行。
           前は 0 バイトの npz(zero_byte)しか出ず、落ちた本の欠けが ③ にも ④ の印にも出なかった。
  人側にした直後の衝突(2026-10-27 fix6-2、反証 6-2 回目 w2-tools-r62-2)  AI の区間の終わりから 0.5 秒以内の人側の衝突(car_check の
           after_ai_collisions と同じ。2026-10-27 fix3 r2-rec2-4 で car_check にだけ足した)も、本の衝突(collisions: all・strong・
           max_g・n_ge8・by_place。表 2 章の衝突(強)・quick_eval_by_params の 1 章の衝突(強)・最大 G・8G 以上・場所・2 章の衝突/通過・
           強い衝突の元)と、発進の coll_before(前に衝突 = 数えない)・その発進から 3 周目の終わりまでの衝突に入れる。collisions に
           after_ai([秒, G, 場所, AI の終わりからの秒])・ai_all・ai_strong(AI 側だけの数)を足し、text と表に「うち人側にした直後 N」。
           前は AI 側だけで、人が当たる直前に CH3 を人側にした当たり(本物の 10/23・10/24 で 6 本)が ④ の 1 章に出ず、手順書 6 章の
           「左 4.2 … SW の出口・LH の当たり 0」が car_check(SW 1)と食い違った。走りながらの 3 周の窓(ずっと AI 側の周)は AI 側だけの
           まま。そういう衝突の無い本は表・JSON とも前と同じ
  リカバリー中止のあとの走り直し(2026-10-27 fix7 係 tools、反証 7-1 回目 w1-tools-r7-1-tools-1)  同じ本の中で、一つ前の AI の
           区間がリカバリー中止(assist 3。中立で人を待つ)で終わり、CH3 を人側 → AI 側にした発進(abort_resumes)は、after_abort
           (中止の始め・中止で終わった区間の終わり・人側の秒)を付け、数えない(counted = False・表の「確か」に「数えない(リカバリー
           中止のあとの走り直し)」・注の 1 行)。中止で止まった所(コースの途中)から出たかもしれず、手で HS に置き直しても車輪の距離は
           0 なので、記録からは置き直したか分からない(一律に数えない = 安全側)。前は「ok・2 回目の発進」で、前に衝突が無ければ counted
           になり、R1 の手前から出た 3 周が best_standing・quick_eval_by_params の止まってから 3 周に入った。中止は rec_hold_keep = 1
           (11/3 は全部の回)の日に CH3 を人側にするまで解けないので、この形が出る。区間の最後のフレームは中立止め(hold)・AI_CMD が
           人側のフレームを除いて見る(どちらも assist 0 と書かれる)。走り直しの区間で車輪の前進が RESUME_MOVE_M(0.5 m)より少ない
           うちにまた人側にしたら、その次の区間も同じ印(carried)。中止の無い本(assist 3 のフレームが無い)は表・JSON とも前と同じ。
           ⑤ diag_launch/collect.py も同じ関数で、その発進を ON/OFF の比べから外す
  中立止めのまま人側にしたあとの走り直し(2026-10-27 fix7 r2 係 tools、反証 7-2 回目 w2-tools-r7-2-tools-1)  手順書 7 章の
           「中立で停止中 → 車に近づく前に CH3 を人側」のとおり、中立止めの途中で人側にすると、リカバリー中止(assist 3)は明けた
           最初のフレームでしか決まらないので 1 フレームも書かれず、上の印が付かなかった(11/3 は両方 ON なので、中立止めが先に明ければ
           中止 = 印あり、人が先に CH3 を倒せば印なし = 人の反応の速さで決まった)。今は abort_resumes が、区間の最後(AI_CMD が人側の
           フレームを除く)まで中立止めが HOLD_KEEP_MAX_S(0.5 秒)以上続いた区間(kind 'hold')と、次のフレームまで le.GAP_S を超えて
           途切れたまま終わった区間(カメラが止まったまま人側。kind 'stall')も「止まって終わった区間」とし(どちらも終わりで止まっていて、
           その区間で車輪の前進が 0.5 m 以上あったときだけ)、その次の区間に同じ after_abort(kind・hold_s / gap_s を足す。中止は
           kind 'abort')。表の「確か」は「数えない(リカバリー中止のあとの走り直し。中立止め X 秒から Y 秒 / カメラの途切れ X 秒から
           Y 秒・人側 Z 秒)」と注 HOLD_NOTE(名前と中止の文は fix7 r1 のまま = 手順書が引く言葉。中止だけの本は表が前と同じ)。本物の
           20261010-160306 の 41.17 秒の発進(中立止め 0.73 秒のまま人側 → 止まった所から AI 側)にこの印が付く(R1 に届かない OFF の発進で、
           前に衝突もあるので数は変わらない)。そういう区間の無い本は表・JSON とも前と同じ
  リカバリーの途中で人側にしたあとの走り直し(2026-10-28 fix8 係 tools-decided。main session の決め 10/28 00:5x = fix3/approvals.md 8
           の「fix7 で残した読み方」①)  一つ前の AI の区間が、リカバリーの途中(assist 2。当たった・止まったあとのブレーキ・中立・後退・
           前進・'halt')で人が CH3 を人側にして終わった発進(区間の最後のフレームは中立止め・AI_CMD が人側のフレームを除いて見る)も、
           中止のあとの走り直しと同じ印(after_abort。kind 'recovery'・abort_s = その区間の最後のリカバリーの始め・rec_s = そこから区間の
           終わりまで)を付けて数えない(counted = False)。どちらも HS からの発進か記録から分からない(同じ理由)。表の「確か」は
           「数えない(リカバリー中止のあとの走り直し。リカバリーの途中 X 秒から Y 秒・人側 Z 秒)」と注 RECOVERY_NOTE(名前は fix7 r1 の
           まま = 手順書が引く言葉)。前に衝突のある発進はもとから数えない(「前に衝突」)。中止・中立止め・カメラの途切れの印が先(fix7 の
           印は前と同じ)。本物の 10/10・10/23・10/24 では、リカバリーの途中で終わった区間 7 つのあとの AI の区間 8 つ(続き 2 を含む)に
           印が付き、うち止まった状態からの発進は 3 つ(20261023-211828@154.35・20261024-135608@27.35・20261024-160847@46.50。どれも前から
           数えない発進 = 前に衝突・途中で変えた / R1 に届かない)で、止まってから 3 周の数・最速は同じ(fix8/decided_tools.md)。そういう
           区間の無い本は表・JSON とも前と同じ
  距離の 3 周と線の方法の違い  1 周の車輪の距離は同じ走行でも 24.0〜28.3 m とばらつく(10/24。衝突・線の違い)。
           距離の 3 周はそれを 3 周分そのまま受ける(10/24 No.11 で −0.57 秒・No.12 で +0.58 秒)。線の方法は起点ごとに
           合わせ直すので、ずれは最後の d_pre(5〜7 m)の分だけ。
"""

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from jetformer import config  # noqa: E402
import label_events as le  # noqa: E402
import section_scorecard as ss  # noqa: E402

try:
    from jetformer.assist import LAUNCH_END_NAMES, ASSIST_LAUNCH, ASSIST_START  # noqa: E402
except Exception:       # noqa: BLE001  古いコード(発進の前進が無い)
    LAUNCH_END_NAMES, ASSIST_LAUNCH, ASSIST_START = {}, 5, 4
try:
    from jetformer.assist import ASSIST_GIVEUP  # noqa: E402
except Exception:       # noqa: BLE001
    ASSIST_GIVEUP = 3
try:        # リカバリーの途中(ブレーキ・中立・後退・前進など。2026-10-28 fix8 tools-decided)
    from jetformer.assist import ASSIST_RECOVERY  # noqa: E402
except Exception:       # noqa: BLE001
    ASSIST_RECOVERY = 2
try:        # 残した中立止めのうち、続きから進めてよい長さ(drive の Recovery と同じ値。より長ければ明けて中止。2026-10-27 fix7 r2)
    from jetformer.assist import Recovery as _Recovery  # noqa: E402
    HOLD_KEEP_MAX_S = float(_Recovery.HOLD_KEEP_MAX_S)
except Exception:       # noqa: BLE001
    HOLD_KEEP_MAX_S = 0.5
RESUME_MOVE_M = 0.5          # 中止のあとの走り直しの区間で、車輪の前進がこれより少なければ次の区間も中止の場所から(2026-10-27 fix7)
# 走り直しの前の AI の区間が止まって終わった形(after_abort の kind。2026-10-27 fix7 r2): 文の頭(中止は fix7 r1 の文のまま)
STOP_KIND_TEXT = {'abort': '中止', 'hold': '中立止め', 'stall': 'カメラの途切れ'}
# 'recovery' = リカバリーの途中で人が CH3 を人側にして終わった区間(2026-10-28 fix8 tools-decided。main session の決め =
# fix3/approvals.md 8 の「fix7 で残した読み方」①)。上の行は tests/test_fix7_docs_r2.py の g1 が文のまま引くので別の行で足す
STOP_KIND_TEXT['recovery'] = 'リカバリーの途中'
# ③ の表・④ の印・⑤ の行の名前。fix7 r2 で中立止め・カメラの途切れのあとも入れたが、名前は fix7 r1 のまま(手順書 7 章・8.1 と
# tests/test_fix7_docs_r2.py が引く言葉。中立止め・途切れは形の文「中立止め X 秒から Y 秒」と注 HOLD_NOTE で分ける)
RESUME_NAME = 'リカバリー中止のあとの走り直し'

SCHEMA = 'jf-standing-3lap/v1'
STOP_ERPM = 400.0            # |eRPM| < これ = 止まっている(analysis_tt・diag_launch/collect.py と同じ)
MOVE_ERPM = 400.0            # 前へこれに届いた = 動いた(0.2 m/s。collect.py の move_0p2_s と同じ)
SLIP_S, SLIP_WARN = 0.6, 1.3   # 空転の比を見る長さ(動いてから)と印のしきい値(diag_launch/collect.py・手順書 7 章と同じ)
SLIP_IMU_MIN = 0.2           # IMU の前の加速の積分がこれ以下なら比を出さない(前からの値。collect.py の slip_window と同じ)
SLIP_EXTREME_DV_WHEEL = 0.5  # 比が出ないのに車輪の速さがこれより増えた = いちばんひどい空転(> 1.3 とみなす)(2026-10-27 r3-r3-rec-4)
# 試し(2026-10-27 fix3 直す 3 回目・2 組目 r3-r3-rec-3): 人が memo に書く字と、走行開始でサーバーが足す自動の文
# (web/bench.plan_check: '計画と違うので試し(元 cand=%s): %s' % (cand, '・'.join(違い))。違いの文に「。」は入らない)
TRIAL_WORD = '試し'
TRIAL_AUTO = re.compile(r'計画と違うので試し\(元 cand=[^)]*\)(?::(?:(?! / )[^。])*)?')
MOVE2_ERPM = 800.0           # 参考: 0.4 m/s(analysis_tt の move_after_s・StartKick.MOVE_ERPM と同じ)
MAX_TO_R1_S = 15.0           # 発進 → 最初の起点の上限(analysis_tt と同じ)
LAP_MIN_S, LAP_MAX_S = ss.LAP_MIN_S, ss.LAP_MAX_S
CONT_S = 0.02                # 続いた周: 前の周の終わりと次の周の始まりの差
DIST_DPSI = (300.0, 420.0)   # 1 周の距離の中央値に使う周の向きの変化(analysis_tt と同じ)
FAST_SUSPECT_S = 7.0         # 印: これより速い周(10/24 の最速 8.25 秒・トップ 7.67 秒/周)。数えるのはそのまま
DIST_SUSPECT = 0.20          # 印: 1 周の車輪の距離が中央値からこれ以上ずれた周
# 印: A4 が無く 3 周目の終わりを A3 + 距離で見積もった(2026-10-26 critic、反証 1 回目 R2)。10/23・10/24 の 11 回で
# A4 を隠すと −0.54〜+0.07 秒ずれた(多くは速く出る。fix2/refute_tools_F2/out/a3dist_vs_anchor.md)。数えるのはそのまま
A3DIST_NOTE = '確かめる: A4 が無い(線のあと R1 の前で人側)。3 周目の終わりは距離の見積もりで −0.5〜+0.1 秒ずれうる(速く出がち)→ ストップウォッチ・動画で読む'
STRONG_G = le.STRONG_G
GOAL_S, PASS_S = 25.0, 27.0
FPS = le.FPS
EPOCH_MIN_US = 10 ** 15         # これより小さい時刻は UNIX µs ではない(単調時計。2001 年 9 月 = 1e15 µs)
SETTING_KEYS = ('throttle_gain', 'steer_gain', 'speed_ceiling', 'ceil_all_mps', 'ceil_left_mps', 'speed_auto',
                'auto_target_mps', 'launch_boost', 'launch_us', 'launch_mps', 'launch_max_s', 'zone_caps', 'start_kick',
                # 2026-10-26 critic(反証 1 回目 tools:F8): car_check の「直す」で入れるトリム(ブロックを選んでも残る)と
                # リカバリーの後ろの見張り(API で入れる)。どちらも 0 なら文・鍵は今までと同じ
                'steer_trim_us', 'rec_back_guard')
# 2026-10-26 fix3 直す 1 回目 r1-rec3-5: fix3 で足した試験の選択肢(既定 0・起動のたびに 0)。log.json の params に欄があるときだけ
# 設定に入れる(欄の無い 10/23・10/24 の記録は設定・JSON・文・鍵が前とビット単位で同じ)。1 なら文に「リカバリーを残す」
# 「見送りで補助なし」・by_params の鍵に入る
OPT_KEYS = ('rec_hold_keep', 'launch_no_kick')
OPT_TEXT = {'rec_hold_keep': 'リカバリーを残す', 'launch_no_kick': '見送りで補助なし'}


def _r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ------------------------------------------------------------------ 記録の読み込み
def read_meta(log, raw_dir=None):
    """log.json(0 バイト・壊れていれば None)"""
    p = Path(raw_dir or config.RAW_DIR) / log / 'log.json'
    try:
        if p.stat().st_size == 0:
            return None
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def zero_byte_files(log, raw_dir=None):
    """記録の中の 0 バイトの npz(シーン/名前)。load_log が飛ばすもの"""
    root = Path(raw_dir or config.RAW_DIR) / log
    out = []
    for p in sorted(root.glob('*/*.npz')) if root.is_dir() else []:
        try:
            if p.stat().st_size == 0:
                out.append('%s/%s' % (p.parent.name, p.name))
        except OSError:
            continue
    return out


MISSING_SENSORS = ('VEHICLE', 'IMU', 'CAM_FRONT')   # 無ければ欠け(label_events.load_log と同じ。AI_CMD は AI のいないシーンに無い)


def missing_npz_files(log, raw_dir=None):
    """記録のシーンの中の、無い npz(シーン/名前。VEHICLE・IMU・CAM_FRONT だけ)。load_log が「ありません」で飛ばすもの。
    2026-10-27 fix5(反証 4 回目 c1): 電源断・jetformer-web の落ちで、書いている途中のシーン(CAM_FRONT/ の画像だけ)の npz が
    無い本と、車両の値が来なかったシーン(本物の 20261024-162305 の 0005・0006 は VEHICLE が無い)。前は zero_byte(0 バイト)にしか
    出ず、③ の 0 バイトの行・④ の印に出なかった"""
    root = Path(raw_dir or config.RAW_DIR) / log
    out = []
    if not root.is_dir():
        return out
    for s in sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(log + '-')):
        for n in MISSING_SENSORS:
            if not (root / s / (n + '.npz')).exists():
                out.append('%s/%s.npz' % (s, n))
    return out


def day_logs(day, raw_dir=None, since=None):
    """その日の AI 走行(log.json の drive があり dry run でない)。戻り値 (記録, 飛ばした [(記録, 理由)])"""
    day = str(day).replace('-', '')
    root = Path(raw_dir or config.RAW_DIR)
    logs, skipped = [], []
    for d in sorted(root.glob('%s-*' % day)):
        if not d.is_dir() or len(d.name) != 15:
            continue
        if since and d.name.split('-')[1][:4] < since:
            continue
        meta = read_meta(d.name, root)
        if meta is None:
            skipped.append((d.name, 'log.json が読めない(0 バイトか壊れている)'))
            continue
        dr = meta.get('drive') or {}
        if not dr:
            continue                                     # 人の収集
        if dr.get('dry_run'):
            skipped.append((d.name, 'dry run'))
            continue
        logs.append(d.name)
    return logs, skipped


def candidate_names(bench=None):
    """{run: 候補の id}(bench_candidates.json の candidates)。読めなければ空"""
    b = bench
    if b is None:
        b = ss.load_bench(config.DATA_DIR / 'bench_candidates.json')
    out = {}
    for c in (b or {}).get('candidates') or []:
        if c.get('run') and c.get('id'):
            out.setdefault(c['run'], c['id'])
    return out


def settings_of(meta, gains_frames=None, names=None):
    """log.json の実際の設定(走行開始の drive.params)。欄が無ければ None"""
    dr = (meta or {}).get('drive') or {}
    p = dr.get('params') or {}
    run = dr.get('run')
    s = {k: p.get(k) for k in SETTING_KEYS}
    s.update({k: p[k] for k in OPT_KEYS if k in p})          # 欄があるときだけ(r1-rec3-5)
    s.update(run=run, model=(names or {}).get(run) or run, mode_id=(meta or {}).get('mode_id'),
             throttle_gain_frames=_r(gains_frames, 3) if gains_frames is not None else None)
    return s


def _onoff(v):
    if v is None:
        return None
    try:
        return int(float(v)) == 1
    except (TypeError, ValueError):
        return None


def settings_text(s):
    """設定の短い文(表の 1 列)。例: 'B・0.90・天井 4.1・自動 4.27・発進 250'。走行中に変えた本は「・途中で変えた(…)」"""
    f = lambda x: '%g' % x if isinstance(x, (int, float)) else str(x)      # noqa: E731
    parts = [str(s.get('model') or '?')]
    parts.append('%.2f' % s['throttle_gain'] if isinstance(s.get('throttle_gain'), (int, float)) else '速さ?')
    if isinstance(s.get('steer_gain'), (int, float)) and abs(s['steer_gain'] - 1.0) > 1e-6:
        parts.append('ステア %.2f' % s['steer_gain'])
    if _onoff(s.get('speed_ceiling')):
        txt = '天井 %s' % f(s.get('ceil_left_mps'))
        if s.get('ceil_all_mps') is not None and abs(float(s['ceil_all_mps']) - 4.5) > 1e-6:
            txt += '/%s' % f(s['ceil_all_mps'])
        parts.append(txt)
    if _onoff(s.get('speed_auto')):
        parts.append('自動 %s' % f(s.get('auto_target_mps')))
    if _onoff(s.get('launch_boost')):
        txt = '発進 %s' % f(s.get('launch_us'))
        if (s.get('launch_mps') not in (None, 2.5)) or (s.get('launch_max_s') not in (None, 1.2)):
            txt += '/%s/%s' % (f(s.get('launch_mps')), f(s.get('launch_max_s')))
        parts.append(txt)
    if _onoff(s.get('zone_caps')):
        parts.append('区間の上限')
    if isinstance(s.get('steer_trim_us'), (int, float)) and abs(float(s['steer_trim_us'])) > 1e-6:
        parts.append('トリム %+g' % float(s['steer_trim_us']))
    if _onoff(s.get('rec_back_guard')):
        parts.append('後ろの見張り')
    for k in OPT_KEYS:                   # r1-rec3-5(0・無いときは出さない)
        if _onoff(s.get(k)):
            parts.append(OPT_TEXT[k])
    if s.get('mode_id') not in (None, 5):
        parts.append('ジャンル %s' % s['mode_id'])
    if s.get('midrun'):
        parts.append('途中で変えた(%s)' % midrun_text(s['midrun']))
    if s.get('trial'):                   # 2026-10-27 r3-r3-rec-3(あるときだけ)
        parts.append('試し(%s)' % s['trial'][:40])
    return '・'.join(parts)


MIDRUN_NAMES = {'speed_ceiling': '天井', 'speed_auto': '自動', 'launch_boost': '発進', 'throttle_gain': '速さ',
                'steer_gain': 'ステア', 'steer_trim_us': 'トリム', 'lookahead': '先読み', 'zone_caps': '区間の上限',
                'tof_guard': 'ToF ガード', 'rec_back_guard': '後ろの見張り', 'recovery': 'リカバリー',
                **OPT_TEXT}               # r1-rec3-5: 前は項目の名前のまま(rec_hold_keep 1→0)


def midrun_text(mid):
    """走行中に変えた設定の短い文。例: '天井 1→0 25.0 秒・自動 1→0 25.0 秒'"""
    f = lambda x: '?' if x is None else ('%g' % x if isinstance(x, (int, float)) else str(x))      # noqa: E731
    return '・'.join('%s %s→%s%s' % (MIDRUN_NAMES.get(m['key'], m['key']), f(m.get('from')), f(m.get('to')),
                                     ' %.1f 秒' % m['t_s'] if m.get('t_s') is not None else '') for m in mid)


def midrun_changes(meta, sc, lab, ai, standing, raw_dir, log, t0_us):
    """走行中に変えた設定 [{key, from, to, t_s, src}](項目ごとに最初の 1 回。src = event / params_end / frames)"""
    p = ((meta or {}).get('drive') or {}).get('params') or {}
    out, seen = [], set()

    def add(key, a, b, t_s, src):
        if key in seen:
            return
        seen.add(key)
        out.append({'key': key, 'from': a, 'to': b, 't_s': t_s, 'src': src})
    mp = ss.midrun_params(log, raw_dir, t0_us)
    for e in mp['events']:
        for k, (a, b) in sorted(e['changed'].items()):
            add(k, a, b, e['t_s'], 'event')
    for k, v in sorted((mp['params_end'] or {}).items()):
        a = p.get(k)
        try:
            same = a is not None and abs(float(a) - float(v)) <= 1e-9
        except (TypeError, ValueError):
            same = a == v
        if not same:
            add(k, a, v, None, 'params_end')
    spc = (sc or {}).get('speed_ctrl') or {}
    if 'ceil_off_midrun' in (spc.get('flags') or []):
        add('speed_ceiling', 1, 0, (spc.get('ceil') or {}).get('off_at_s'), 'frames')
    if 'auto_off_midrun' in (spc.get('flags') or []):
        add('speed_auto', 1, 0, (spc.get('auto') or {}).get('off_at_s'), 'frames')
    for srow in standing or []:
        if (srow.get('launch') or {}).get('end_code') == 10:
            add('launch_boost', 1, 0, (srow['launch'].get('end_s') or 0.0) + srow['t_ai'], 'frames')
    if lab is not None and ai is not None and ai.any():
        t = lab['t_s'].astype(np.float64)[ai]
        cols = [('steer_gain', lab['gains'][ai, 0]), ('steer_trim_us', lab['trim'][ai] if 'trim' in lab else None)]
        if not _onoff(p.get('speed_auto')):          # 自動合わせ ON はフレームのゲインが周ごとに変わる(変えたのではない)
            cols.append(('throttle_gain', lab['gains'][ai, 1]))
        for k, x in cols:
            if x is None or not len(x):
                continue
            x = np.asarray(x, np.float64)
            d = np.nonzero(np.abs(x - x[0]) > 0.005)[0]
            if len(d):
                add(k, _r(x[0], 3), _r(x[d[0]], 3), _r(t[d[0]], 3), 'frames')
    return out


def launch_events(log, raw_dir=None):
    """events.jsonl の launch_boost の出来事 [(時刻 µs(カメラ。無ければ記録の時刻), 出来事, 理由)]"""
    p = Path(raw_dir or config.RAW_DIR) / log / 'events.jsonl'
    out = []
    try:
        lines = p.read_text(encoding='utf-8').splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('kind') != 'launch_boost':
            continue
        tu = e.get('cam_epoch_us')
        if tu is None:                   # 2026-10-26 fix2 の前: cam_t_us(drive.py は単調時計)。UNIX µs のときだけ使う
            tu = e.get('cam_t_us', e.get('t_us'))
            if tu is not None and int(tu) < EPOCH_MIN_US:
                tu = None
        out.append((int(tu) if tu is not None else None, str(e.get('event') or ''),
                    str(e.get('reason') or e.get('code_reason') or '')))
    return out


# ------------------------------------------------------------------ 小道具
def _first_reach(vt, Df, target, lo_t, hi_t):
    """lo_t〜hi_t の間で前進の距離 Df が target に初めて届いた時刻(線形に補う)。届かなければ None"""
    m = (vt >= lo_t) & (vt <= hi_t)
    if not m.any():
        return None
    tt, DD = vt[m], Df[m]
    if DD[-1] < target:
        return None
    k = int(np.searchsorted(DD, target, side='left'))
    if k <= 0:
        return float(tt[0])
    x0, x1 = DD[k - 1], DD[k]
    fr = (target - x0) / (x1 - x0) if x1 > x0 else 0.0
    return float(tt[k - 1] + fr * (tt[k] - tt[k - 1]))


def _max_gap(ts, a, b):
    """a〜b の間の時刻の並び ts の最大の間(端の a・b も入れる)"""
    ts = np.asarray(ts, float)
    m = (ts > a) & (ts < b)
    pts = np.concatenate([[a], ts[m], [b]])
    return float(np.max(np.diff(pts))) if len(pts) > 1 else 0.0


def lap_dist_median(anchors, it, hd, vt, Df):
    """1 周の距離の中央値 [m](起点 → 次の起点で 16 秒未満・向きの変化 300〜420°。analysis_tt と同じ)"""
    d = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        dpsi = float(np.interp(b, it, hd) - np.interp(a, it, hd))
        if DIST_DPSI[0] <= dpsi <= DIST_DPSI[1] and b - a < LAP_MAX_S:
            d.append(float(np.interp(b, vt, Df) - np.interp(a, vt, Df)))
    return float(np.median(d)) if d else None


def _coll_rows(coll, a, b):
    return [c for c in coll if a <= c['t_s'] < b]


def coll_summary(rows):
    g = [float(c['g']) for c in rows]
    return {'n': len(rows), 'strong': sum(1 for x in g if x >= STRONG_G), 'gmax': _r(max(g), 2) if g else 0.0,
            'list': [[_r(c['t_s'], 3), _r(c['g'], 2), c.get('loc')] for c in rows]}


def lap_kind(cc, clean):
    """analysis_tt の周の種類"""
    if not cc:
        return 'きれい' if clean else 'リカバリーなど'
    return '軽い接触' if max(float(c['g']) for c in cc) < STRONG_G else '強い衝突'


def window_kind(kinds):
    if all(k == 'きれい' for k in kinds):
        return 'きれい'
    if any(k == '強い衝突' for k in kinds):
        return '強い衝突あり'
    if any(k == '軽い接触' for k in kinds):
        return '軽い接触だけ'
    return 'リカバリーなど'


# ------------------------------------------------------------------ 走りながらの 3 周
def flying_windows(laps, coll):
    """score_log の周 → 続いた 3 周の窓 [{t_start, laps_s, total_s, kind, gmax, collisions}]"""
    L = []
    for lp in laps:
        a = float(lp['t_start'])
        b = a + float(lp['lap_s'])
        cc = _coll_rows(coll, a, b)
        L.append({'a': a, 'b': b, 'lap_s': float(lp['lap_s']), 'kind': lap_kind(cc, lp.get('clean')), 'cc': cc})
    out = []
    for i in range(len(L) - 2):
        w = L[i:i + 3]
        if not all(abs(w[j]['b'] - w[j + 1]['a']) <= CONT_S for j in range(2)):
            continue
        cc = [c for x in w for c in x['cc']]
        out.append({'t_start': _r(w[0]['a'], 3), 'laps_s': [_r(x['lap_s'], 3) for x in w],
                    'total_s': _r(sum(x['lap_s'] for x in w), 3), 'kind': window_kind([x['kind'] for x in w]),
                    'gmax': _r(max([float(c['g']) for c in cc], default=0.0), 2), 'collisions': coll_summary(cc),
                    'suspect': any(x['lap_s'] < FAST_SUSPECT_S for x in w)})
    return out


def flying_summary(windows):
    tot = [w['total_s'] for w in windows]
    cl = [w for w in windows if w['kind'] == 'きれい']
    best = min(windows, key=lambda w: w['total_s']) if windows else None
    bcl = min(cl, key=lambda w: w['total_s']) if cl else None
    return {'n': len(windows), 'n_clean': len(cl), 'best_s': best['total_s'] if best else None,
            'best': best, 'best_clean_s': bcl['total_s'] if bcl else None, 'best_clean': bcl,
            'median_s': _r(float(np.median(tot)), 3) if tot else None,
            'n_le25': sum(1 for x in tot if x <= GOAL_S), 'n_le27': sum(1 for x in tot if x <= PASS_S)}


# ------------------------------------------------------------------ 止まってから 3 周
def launch_record(A, t0_us, t_a, t_b, events=()):
    """AI_CMD の発進の前進の記録(t_a〜t_b のフレーム)。launch_on の欄が無い記録は recorded = False"""
    out = {'recorded': False, 'assist5_frames': None, 'on_frames': None, 'on_s': None, 'end_code': None,
           'end_reason': None, 'end_s': None, 'kick_frames': None, 'events': []}
    if A is not None and 'assist' in A:
        at = (np.asarray(A['timestamps_us'], np.int64) - int(t0_us)) / 1e6
        m = (at >= t_a - 1e-6) & (at <= t_b + 1e-6)
        asg = np.asarray(A['assist'])[m]
        out['kick_frames'] = int(np.sum(asg == ASSIST_START))
        if 'launch_on' in A:
            on = np.asarray(A['launch_on'], bool)[m]
            end = np.asarray(A['launch_end'], np.int64)[m] if 'launch_end' in A else np.zeros(int(m.sum()), np.int64)
            tm = at[m]
            out.update(recorded=True, assist5_frames=int(np.sum(asg == ASSIST_LAUNCH)), on_frames=int(on.sum()))
            if on.any():
                out['on_s'] = _r(float(tm[on][-1] - t_a) + 1.0 / FPS, 3)
            k = np.nonzero(end > 0)[0]
            if len(k):
                out.update(end_code=int(end[k[0]]), end_reason=LAUNCH_END_NAMES.get(int(end[k[0]]), str(int(end[k[0]]))),
                           end_s=_r(float(tm[k[0]] - t_a), 3))
    for tu, ev, why in events or ():
        if tu is None:
            continue
        ts = (int(tu) - int(t0_us)) / 1e6
        if t_a - 0.2 <= ts <= t_b:
            out['events'].append([_r(ts, 3), ev, why])
    return out


def _line_ends(vt, Df, A_, d_pre, t_line0, t1, Lrun):
    """k 周目(1〜3)の終わり = 起点 A_(k+1) の d_pre m 手前を通った時刻。A4 が無ければ 3 周目だけ A3 + (Lrun − d_pre)"""
    D = lambda x: float(np.interp(x, vt, Df))      # noqa: E731
    ends, methods = [], []
    prev = t_line0
    for k in (1, 2, 3):
        te, meth = None, None
        if k < len(A_):
            te = _first_reach(vt, Df, D(A_[k]) - d_pre, prev, A_[k])
            meth = 'anchor'
        elif k == 3 and len(A_) == 3 and Lrun:
            te = _first_reach(vt, Df, D(A_[2]) + Lrun - d_pre, A_[2], t1 + le.GAP_S)
            meth = 'A3+dist'
        if te is None:
            break
        ends.append(te)
        methods.append(meth)
        prev = te
    return ends, methods


def launch_slip(t_move, t_end, vt, verpm, it, ay):
    """発進の空転の目安(diag_launch/collect.py の slip_window と同じ): 車輪の速さの増え ÷ IMU の前の加速の積分
    (t_move → min(t_move + 0.6 秒, A1))。1 より大きい = 車輪が先に回った。手順書の「空転の比 > 1.3」と同じ物差し。
    ay = 前の加速 [m/s²](止まっているときの中央値を引いたもの)"""
    if it is None or ay is None or t_end - t_move < 0.15:
        return None
    v = -np.asarray(verpm, np.float64) * le.KV
    dv_w = float(np.interp(t_end, vt, v) - np.interp(t_move, vt, v))
    m = (it >= t_move) & (it <= t_end)
    if m.sum() < 5:
        return None
    x, y = it[m], ay[m]
    dv_i = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    return {'t_a': _r(t_move, 3), 't_b': _r(t_end, 3), 'dv_wheel': _r(dv_w, 3), 'dv_imu': _r(dv_i, 3),
            'ratio': _r(dv_w / dv_i, 3) if dv_i > SLIP_IMU_MIN else None}


def slip_kind(sl):
    """空転の比の読み(2026-10-27 fix3 直す 3 回目・2 組目 r3-r3-rec-4。launch_slip・collect.py の slip_window の戻り値):
    'ok' = 比 ≤ 1.3 / 'high' = 比 > 1.3 / 'extreme' = 比が出ない(IMU の積分 ≤ 0.2 m/s)のに車輪の速さが 0.5 m/s より増えた
    (いちばんひどい空転。> 1.3 とみなす)/ 'unknown' = 比が出ない(窓が短い・IMU が抜けた・車輪も IMU もほとんど増えない)"""
    if not sl:
        return 'unknown'
    r = sl.get('ratio')
    if r is not None:
        return 'high' if r > SLIP_WARN else 'ok'
    dw, di = sl.get('dv_wheel'), sl.get('dv_imu')
    if dw is not None and di is not None and dw > SLIP_EXTREME_DV_WHEEL and di <= SLIP_IMU_MIN:
        return 'extreme'
    return 'unknown'


def slip_note(sl, imu_gap=False):
    """比が出ない発進の印(r3-r3-rec-4)。いちばんひどい空転(extreme)と、IMU が抜けて比が出ない(imu_gap)だけ。ほかは None
    (比が出る発進の印は前と同じ「空転の比 X(> 1.3)…」。車輪も IMU もほとんど増えない = 空転していない発進には付けない)"""
    k = slip_kind(sl)
    if k == 'extreme':
        return ('空転の比が出ない(車輪 +%.2f m/s・IMU +%.2f m/s。IMU がほとんど進まない = いちばんひどい空転とみなす > %.1f): '
                'd_pre が長く出て、止まってから 3 周が短く出ているかもしれない' % (sl['dv_wheel'], sl['dv_imu'], SLIP_WARN))
    if sl is None and imu_gap:
        return '確かめる: 空転の比が出ない(動いてから %.1f 秒の間に IMU が抜けた)→ 動画で空転を見る' % SLIP_S
    return None


def trial_memo(desc):
    """人が説明欄に「試し」と書いた本なら、その文(自動の「計画と違うので試し(元 cand=…): …」を除いたもの)。でなければ None
    (2026-10-27 fix3 直す 3 回目・2 組目 r3-r3-rec-3。手順書 7 章「その本は数えない(memo に「試し」)」)。
    desc = section_scorecard.effective_description の文(走り終わりのメモは scene.json にだけ入る)。memo と key=値 の外の文を見る"""
    s = str(desc or '')
    m = re.search(r'(?:^|\s)memo=(.*)$', s)
    text = ' / '.join(x for x in ((m.group(1).strip() if m else ''), ss._free_text(s)) if x)
    rest = TRIAL_AUTO.sub('', text)
    if TRIAL_WORD not in rest:
        return None
    parts = [x.strip(' ・') for y in rest.split(' / ') for x in y.split('。')]
    return '。'.join(x for x in parts if x) or TRIAL_WORD


def abort_resumes(t, ai, assist, hold=None, sel=None, vt=None, Df=None, verpm=None):
    """中止・中立止めのあとの走り直し(2026-10-27 fix7 係 tools、反証 7-1 回目 w1-tools-r7-1-tools-1・7-2 回目 w2-tools-r7-2-tools-1)。
    {AI の区間の始めのフレームの番号 i0: {'kind': 一つ前の AI の区間が止まって終わった形('abort' / 'hold' / 'stall'), 'abort_s':
    止まり始め [秒], 'ai_end_s': その AI の区間の終わり, 'human_s': 人側の秒(一つ前の AI の区間の終わり → この区間の始め)
    [, 'hold_s': 中立止めの長さ(hold)| 'gap_s': フレームの途切れの長さ(stall)][, 'carried': 間にはさまった走り直しの区間の数
    (あるときだけ)]}}。
    t・ai = フレームの時刻とラベルの ai(呼び手が le.runs(ai) で区間を作るのと同じもの)。assist・hold・sel = ラベルの assist・hold・
    ai_cmd_select(AI_CMD の select)。vt・Df = 車両データの時刻と車輪の前進の距離(le.cumulative_distance の 2 つ目)。verpm = 車両
    データの eRPM(vt と同じ長さ。止まっているかを見る)。
    止まって終わった区間(kind):
      'abort'  区間の最後のフレームが、中立止め(hold)・AI_CMD が人側(sel = False)のフレームを除いて、リカバリー中止(assist 3)
               (fix7 r1)。drive.py は中立止めのフレームを assist 0 で書き、CH3 を人側にした瞬間のフレーム(ラベルの ai は車両の CH3 の
               直前の値なのでまだ AI 側)も assist 0 なので、それを除かないと中止を見落とす。abort_s = 中止の始め
      'hold'   (fix7 r2)区間の最後のフレーム(AI_CMD が人側のフレームを除く)まで中立止めが HOLD_KEEP_MAX_S(0.5 秒)以上続いた
               (前に処理したフレームから。drive の「見ていない時間」と同じ数え方)。手順書 7 章「中立で停止中 → 車に近づく前に CH3 を
               人側」のとおり中立止めの途中で人側にすると、中止(assist 3)は明けた最初のフレームでしか決まらないので 1 フレームも
               書かれない(drive は中立止めの間 helper.step を呼ばず、人側のフレームでは Assist.step が reset する)。abort_s = 中立止めの
               始め、hold_s = その長さ
      'stall'  (fix7 r2)区間の最後のフレームから次のフレームまで le.GAP_S より長く途切れた(カメラが止まったまま人側。drive はカメラが
               止まった中立止めのフレームを書かない)。abort_s = 区間の最後のフレーム、gap_s = 途切れの長さ
      hold・stall は、終わりで止まっていて(hold: 区間の最後のフレームの時刻で |eRPM| < STOP_ERPM。stall: 途切れの間に一度でも
      |eRPM| < STOP_ERPM)、その区間で車輪の前進が RESUME_MOVE_M 以上あった区間だけ(置いた場所から一度も動かなかった区間 =
      本物の 20261024-183123 の 40.10〜43.36 秒を外す)。verpm・vt・Df が無ければ hold・stall は見ない(abort だけ = fix7 r1 と同じ)。
      'recovery' (2026-10-28 fix8 tools-decided。main session の決め = fix3/approvals.md 8 の「fix7 で残した読み方」①)区間の最後の
               フレームが、中立止め・AI_CMD が人側のフレームを除いて、リカバリーの途中(assist 2 = ASSIST_RECOVERY。当たった・止まった
               あとのブレーキ・中立・後退・前進・惰性を止める 'halt')= 人がリカバリーの途中で CH3 を人側にした区間。中止と同じく、
               止まった所・当たった所(コースの途中)から出たかもしれず、HS に置き直したかは記録から分からない(同じ理由)ので、
               ほかの決まり(止まっている・進んだ距離)は付けない。abort・hold・stall のどれでもないときだけ(その印が先 = fix7 の印は
               前と同じ)。abort_s = その区間の最後のリカバリー(続いた assist 2)の始め、rec_s = そこから区間の終わりまでの秒。
               verpm が無い呼び方でも見る(assist だけで決まる)
    その次の AI の区間を走り直しとする(どこに置いたかは記録から分からない: 手で運ぶと車輪の距離は 0)。走り直しの区間で車輪の前進が
    RESUME_MOVE_M より少ないうちにまた人側にしたら、その次の区間も(まだ止まった場所の近く。vt・Df が無ければ続けない)。
    assist 3・2 のフレームが無く verpm も無い呼び方は {}(呼び手は前と同じ)"""
    t = np.asarray(t, np.float64)
    ai = np.asarray(ai, bool)
    a = None if assist is None else np.asarray(assist)
    has_abort = a is not None and len(a) > 0 and bool((a == ASSIST_GIVEUP).any())
    has_rec = a is not None and len(a) > 0 and bool((a == ASSIST_RECOVERY).any())     # fix8 tools-decided
    vt_ = None if vt is None else np.asarray(vt, np.float64)
    can_stop = verpm is not None and vt_ is not None and Df is not None and len(vt_) > 0
    if not has_abort and not has_rec and not can_stop:
        return {}
    h = np.zeros(len(t), bool) if hold is None else np.asarray(hold, bool)
    s = np.ones(len(t), bool) if sel is None else np.asarray(sel, bool)
    ve = np.asarray(verpm, np.float64) if can_stop else None

    def moved(i0, i1):
        if vt is None or Df is None or not len(vt):
            return float('inf')
        return float(np.interp(t[i1], vt, Df) - np.interp(t[i0], vt, Df))

    def stop_end(i0, i1):
        """hold・stall で止まって終わった区間なら {kind, abort_s, hold_s | gap_s}、でなければ None(fix7 r2)"""
        if not can_stop or moved(i0, i1) < RESUME_MOVE_M:
            return None
        k = np.arange(i0, i1 + 1)
        k = k[s[i0:i1 + 1]]                      # AI_CMD が人側のフレームを除く
        if len(k) and h[k[-1]]:
            j = len(k) - 1
            while j > 0 and h[k[j - 1]]:
                j -= 1
            tp = float(t[k[j - 1]]) if j > 0 else float(t[k[j]])     # 前に処理したフレーム(中立止めの前)
            hs = float(t[k[-1]]) - tp
            if hs >= HOLD_KEEP_MAX_S - 1e-6 and abs(float(np.interp(t[i1], vt_, ve))) < STOP_ERPM:
                return {'kind': 'hold', 'abort_s': _r(float(t[k[j]]), 3), 'hold_s': _r(hs, 2)}
        if i1 + 1 < len(t):
            ta, tb = float(t[i1]), float(t[i1 + 1])
            if tb - ta > le.GAP_S:
                m = (vt_ >= ta) & (vt_ <= tb)
                e = np.abs(np.concatenate([ve[m], [float(np.interp(tb, vt_, ve))]]))
                if float(e.min()) < STOP_ERPM:
                    return {'kind': 'stall', 'abort_s': _r(ta, 3), 'gap_s': _r(tb - ta, 2)}
        return None
    out, pend, prev_end = {}, None, None
    for i0, i1 in le.runs(ai):
        if pend is not None:
            out[i0] = dict(pend, human_s=_r(float(t[i0]) - prev_end, 2))
            if pend.get('carried') == 0:
                del out[i0]['carried']
        k = np.arange(i0, i1 + 1)
        k = k[~h[i0:i1 + 1] & s[i0:i1 + 1]]
        if has_abort and len(k) and int(a[k[-1]]) == ASSIST_GIVEUP:
            j = len(k) - 1
            while j > 0 and int(a[k[j - 1]]) == ASSIST_GIVEUP:
                j -= 1
            pend = {'kind': 'abort', 'abort_s': _r(float(t[k[j]]), 3), 'ai_end_s': _r(float(t[i1]), 3), 'carried': 0}
        else:
            se = stop_end(i0, i1)        # fix7 r2: 中立止め・カメラの途切れのまま止まって終わった区間
            if se is not None:
                pend = dict(se, ai_end_s=_r(float(t[i1]), 3), carried=0)
            elif has_rec and len(k) and int(a[k[-1]]) == ASSIST_RECOVERY:
                # fix8 tools-decided: リカバリーの途中で人が CH3 を人側にした区間(中立止め・AI_CMD が人側のフレームを除いて最後が assist 2)
                j = len(k) - 1
                while j > 0 and int(a[k[j - 1]]) == ASSIST_RECOVERY:
                    j -= 1
                t_r = float(t[k[j]])
                pend = {'kind': 'recovery', 'abort_s': _r(t_r, 3), 'rec_s': _r(float(t[i1]) - t_r, 2),
                        'ai_end_s': _r(float(t[i1]), 3), 'carried': 0}
            elif pend is not None and moved(i0, i1) < RESUME_MOVE_M:
                pend = dict(pend, carried=pend['carried'] + 1)
            else:
                pend = None
        prev_end = float(t[i1])
    return out


def abort_text(ab, fmt='%.2f'):
    """after_abort の止まった形の文(③ の表。2026-10-27 fix7 r2): 「中止 X 秒」(fix7 r1 の文のまま)/「中立止め X 秒から Y 秒」/
    「カメラの途切れ X 秒から Y 秒」/「リカバリーの途中 X 秒から Y 秒」(2026-10-28 fix8 tools-decided)(kind の無い古い印は中止)"""
    ab = ab or {}
    kind = ab.get('kind') or 'abort'
    x = (fmt % ab['abort_s']) if ab.get('abort_s') is not None else '?'
    if kind == 'hold':
        return '%s %s 秒から %s 秒' % (STOP_KIND_TEXT['hold'], x, _f(ab.get('hold_s'), '%.1f'))
    if kind == 'stall':
        return '%s %s 秒から %s 秒' % (STOP_KIND_TEXT['stall'], x, _f(ab.get('gap_s'), '%.1f'))
    if kind == 'recovery':
        return '%s %s 秒から %s 秒' % (STOP_KIND_TEXT['recovery'], x, _f(ab.get('rec_s'), '%.1f'))
    return '%s %s 秒' % (STOP_KIND_TEXT['abort'], x)


def standing_starts(t, ai, it, hd, vt, verpm, anchors, coll, A=None, t0_us=0, events=(), line_ahead_m=0.0, Lrun=None,
                    ay=None):
    """止まった状態からの発進ごとの「止まってから 3 周」。t・ai はフレーム(ai は dry run を除く)"""
    _, Df = le.cumulative_distance(vt, verpm)
    D = lambda x: float(np.interp(x, vt, Df))      # noqa: E731
    anchors = np.asarray(anchors, float)
    rows = []
    for i0, i1 in le.runs(ai):
        t_ai, t1 = float(t[i0]), float(t[i1])
        e0 = float(np.interp(t_ai, vt, verpm)) if len(vt) else float('nan')
        if not np.isfinite(e0) or abs(e0) >= STOP_ERPM or t1 - t_ai < 0.1:
            continue
        row = {'t_ai': _r(t_ai, 3), 't_ai_end': _r(t1, 3), 'erpm_at_ai': _r(e0, 0), 'ai_seg_s': _r(t1 - t_ai, 2),
               't_move': None, 'move_after_s': None, 'valid': False, 'why': None, 'method': None,
               'from_move_s': None, 'from_ai_s': None, 'from_line_s': None, 'dist3_from_ai_s': None,
               'dist3_from_move_s': None,
               # 何回目の発進か・それより前の衝突(2026-10-26 fix2 R2-rec-3)。counted は下で決める
               'start_no': len(rows) + 1, 'coll_before': len([c for c in coll if c['t_s'] < t_ai]), 'counted': False}
        row['launch'] = launch_record(A, t0_us, t_ai, t1, events)
        mv = np.nonzero((vt >= t_ai) & (vt <= t1) & (-verpm >= MOVE_ERPM))[0]
        if not len(mv):
            row['why'] = 'AI の区間で動いていない'
            rows.append(row)
            continue
        t_move = float(vt[mv[0]])
        mv2 = np.nonzero((vt >= t_ai) & (vt <= t1) & (-verpm >= MOVE2_ERPM))[0]
        row.update(t_move=_r(t_move, 3), move_after_s=_r(t_move - t_ai, 3),
                   move_0p4_after_s=_r(float(vt[mv2[0]]) - t_ai, 3) if len(mv2) else None)
        nxt = anchors[anchors > t_ai]
        if not len(nxt) or nxt[0] > t1 or nxt[0] - t_ai > MAX_TO_R1_S:
            row['why'] = '15 秒以内(AI の区間の中)に R1 の起点が無い'
            rows.append(row)
            continue
        # 続いた起点(5〜16 秒ごと)を 4 つまで
        A_ = [float(nxt[0])]
        for x in nxt[1:]:
            if len(A_) == 4 or not (LAP_MIN_S <= x - A_[-1] <= LAP_MAX_S):
                break
            A_.append(float(x))
        row['anchors'] = [_r(x, 3) for x in A_]
        row['to_r1_s'] = _r(A_[0] - t_ai, 3)
        row['d_from_ai_m'] = _r(D(A_[0]) - D(t_ai), 2)                 # analysis_tt の dist_to_r1_m と同じ
        # 参考: 発進の空転(空転すると d_pre が長く出て、止まってから 3 周が短く出る)
        t_sl = min(t_move + SLIP_S, A_[0])
        row['launch_slip'] = launch_slip(t_move, t_sl, vt, verpm, it, ay)
        rs = (row['launch_slip'] or {}).get('ratio')
        if rs is not None and rs > SLIP_WARN:
            row['note'] = '空転の比 %.2f(> %.1f): d_pre が長く出て、止まってから 3 周が短く出ているかもしれない' % (rs, SLIP_WARN)
        elif rs is None:                 # 2026-10-27 r3-r3-rec-4: いちばんひどい空転・IMU が抜けて比が出ないにも印(あるときだけ)
            note = slip_note(row['launch_slip'], imu_gap=(it is not None and ay is not None and t_sl - t_move >= 0.15))
            if note:
                row['note'] = note
        # 距離の 3 周(参考。analysis_tt と同じ: t_ai から 1 周の距離の中央値 × 3)
        if Lrun:
            m = (vt >= t_ai) & (vt <= t1)
            D0 = D(t_ai)
            if m.any() and Df[m][-1] - D0 >= 3 * Lrun:
                t3 = float(np.interp(D0 + 3 * Lrun, Df[m], vt[m]))
                row.update(dist3_from_ai_s=_r(t3 - t_ai, 3), dist3_from_move_s=_r(t3 - t_move, 3), dist3_end=_r(t3, 3))
        # 線
        x_line = D(t_move) + float(line_ahead_m)
        t_line0 = t_move if line_ahead_m <= 0 else _first_reach(vt, Df, x_line, t_move, A_[0])
        d_pre = D(A_[0]) - x_line
        row.update(d_pre_m=_r(d_pre, 2), lap_dist_median_m=_r(Lrun, 2) if Lrun else None,
                   start_frac_of_lap=_r(1 - d_pre / Lrun, 3) if Lrun else None)
        if t_line0 is None or d_pre <= 0:
            row['why'] = 'スタートの線が最初の起点より先'
            rows.append(row)
            continue
        ends, methods = _line_ends(vt, Df, A_, d_pre, t_line0, t1, Lrun)
        row['line_ends'] = [_r(x, 3) for x in ends]
        row['laps_line_s'] = [_r(b - a, 3) for a, b in zip([t_line0] + ends[:-1], ends)]
        row['laps_anchor_s'] = [_r(b - a, 3) for a, b in zip(A_[:-1], A_[1:])]
        if len(ends) < 3:
            row['why'] = '3 周に届かない(続いた起点 %d・線 %d 回)' % (len(A_), len(ends))
            rows.append(row)
            continue
        e3 = ends[2]
        row['method'] = methods[2]
        row['end_s'] = _r(e3, 3)
        if len(A_) == 4:
            row['to_a4_from_move_s'] = _r(A_[3] - t_move, 3)          # 参考: 3 周目の終わりの起点まで
        row.update(from_move_s=_r(e3 - t_move, 3), from_ai_s=_r(e3 - t_ai, 3))
        if line_ahead_m > 0:
            row['t_line0'] = _r(t_line0, 3)
            row['from_line_s'] = _r(e3 - t_line0, 3)
        cc = _coll_rows(coll, t_ai, e3)
        row['collisions'] = coll_summary(cc)
        row['collisions_by_lap'] = [len(_coll_rows(coll, a, b)) for a, b in zip([t_ai] + ends[:2], ends)]
        why = []
        if not all(LAP_MIN_S <= x <= LAP_MAX_S for x in row['laps_line_s']):
            why.append('線 → 線の周が 5〜16 秒でない')
        dps = [float(np.interp(b, it, hd) - np.interp(a, it, hd)) for a, b in zip(A_[:-1], A_[1:])]
        row['dpsi_anchor_deg'] = [_r(x, 1) for x in dps]
        bad = [x for x in dps if not (DIST_DPSI[0] <= x <= DIST_DPSI[1])]
        if bad:                          # 偽の起点を外す決まりで外れなかった半周など(1 周 = 左回り 323〜379°)
            why.append('起点 → 起点の向きの変化 %s°(300〜420° でない)' % '・'.join('%.0f' % x for x in bad))
        if e3 > t1 + le.GAP_S:
            why.append('3 周目の終わりの前に人側')
        if _max_gap(t, t_ai, e3) > le.GAP_S:
            why.append('フレームの途切れ %.2f 秒' % _max_gap(t, t_ai, e3))
        if _max_gap(vt, t_move, e3) > le.GAP_S:
            why.append('車両データの途切れ %.2f 秒' % _max_gap(vt, t_move, e3))
        row['valid'] = not why
        row['why'] = '・'.join(why) or None
        row['counted'] = row['valid'] and row['coll_before'] == 0
        # 印(数えるのはそのまま。記録・動画で確かめる): 速すぎる周・1 周の距離が中央値から 2 割以上ずれた周
        row['dist_anchor_m'] = [_r(D(b) - D(a), 2) for a, b in zip(A_[:-1], A_[1:])]
        odd = [x for x in row['laps_anchor_s'] if x < FAST_SUSPECT_S]
        if Lrun:
            odd += ['%.1f m' % x for x in row['dist_anchor_m'] if abs(x / Lrun - 1.0) > DIST_SUSPECT]
        if odd:
            note = '確かめる: ' + '・'.join(('%.2f 秒' % x) if isinstance(x, float) else x for x in odd)
            row['note'] = (row['note'] + '・' + note) if row.get('note') else note
        if row['method'] == 'A3+dist':   # 2026-10-26 critic(反証 1 回目 R2・tools:F2。数えるのはそのまま)
            row['note'] = ((row['note'] + '・') if row.get('note') else '') + A3DIST_NOTE
        rows.append(row)
    return rows


# ------------------------------------------------------------------ 1 本の記録
def analyze_log(log, raw_dir=None, labels_dir=None, names=None, line_ahead_m=0.0):
    """1 本の記録の「止まってから 3 周」・走りながらの 3 周・設定・周の要約"""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    labels_dir = Path(labels_dir or config.DATA_DIR / 'labels')
    meta = read_meta(log, raw_dir)
    res = {'log': log, 'zero_byte': zero_byte_files(log, raw_dir)}
    mz = missing_npz_files(log, raw_dir)
    if mz:                               # 2026-10-27 fix5 c1(あるときだけ。無ければ前と同じ)
        res['missing_npz'] = mz
    if meta is None:
        res['error'] = 'log.json が読めない'
        return res
    L = le.load_log(log, raw_dir)
    labels = ss._labels(log, labels_dir, raw_dir)
    info, lab, _ = labels
    ex = {}
    sc = ss.score_log(log, labels_dir, raw_dir, labels=labels, loaded=L, extras=ex)
    desc = sc.get('desc') or {}
    res.update(ai_no=desc.get('ai_no'), cand=sc.get('cand'), block=desc.get('block'), tune=desc.get('tune'),
               day=desc.get('day'), labels=sc.get('labels'), problems=sc.get('problems') or [])
    if lab is None or 'error' in sc:
        res['error'] = sc.get('error') or info.get('error') or 'ラベルがありません'
        res['settings'] = settings_of(meta, None, names)
        res['settings_text'] = settings_text(res['settings'])
        return res
    t = lab['t_s'].astype(np.float64)
    ai = lab['ai'].astype(bool) & ~lab['dry_run'].astype(bool)
    g = lab['gains'][ai, 1] if ai.any() else np.zeros(0)
    res['settings'] = settings_of(meta, float(np.median(g)) if len(g) else None, names)
    res['settings_text'] = settings_text(res['settings'])
    anchors = np.asarray(ex['anchors'], float)
    _, Df = le.cumulative_distance(ex['vt'], ex['verpm'])
    Lrun = lap_dist_median(anchors, ex['it'], ex['hd'], ex['vt'], Df)
    coll = sc.get('collision_events') or []
    # 2026-10-27 fix6-2(反証 6-2 回目 w2-tools-r62-2): 人側にした直後(AI の区間の終わりから 0.5 秒以内)の人側の衝突(car_check の
    # after_ai_collisions と同じ。r2-rec2-4)も、本の衝突と発進の「前に衝突」・その 3 周の衝突に入れる(coll_b)。前は AI 側だけで、
    # 人が当たる直前に CH3 を人側にした当たりが ④ の 1 章(衝突・最大 G・8G 以上・場所)に出ず、car_check と食い違った。
    # 走りながらの 3 周の窓(ずっと AI 側の周)は coll のまま。そういう衝突の無い本は前と同じ(coll_b is coll)
    import car_check as cc  # noqa: E402  (car_check は standing_3lap を中で読むので、ここで読む)
    after = cc.after_ai_collisions(ex.get('ev') or info.get('events') or {}, lab)
    coll_b = sorted(list(coll) + after, key=lambda c: float(c['t_s'])) if after else coll
    res['lap_dist_median_m'] = _r(Lrun, 2) if Lrun else None
    res['r1_anchors'] = int(len(anchors))
    res['r1_anchors_rejected'] = [x.get('t_s') for x in sc.get('r1_anchors_rejected') or []]
    I = L.get('IMU')
    ay = None
    if I is not None and len(I['accel']) == len(ex['it']):          # 前の加速(collect.py と同じ: 止まっているときのずれを引く)
        still = (np.abs(np.interp(ex['it'], ex['vt'], ex['verpm'])) < 50) & (np.abs(I['gyro'][:, 2]) < 5)
        _, bias = le.horizontal_g(I['accel'], still)
        ay = np.asarray(I['accel'][:, 1], np.float64) - bias[1]
    res['standing'] = standing_starts(t, ai, ex['it'], ex['hd'], ex['vt'], ex['verpm'], anchors, coll_b,
                                      L.get('AI_CMD'), info['t0_us'], launch_events(log, raw_dir), line_ahead_m, Lrun, ay)
    # 2026-10-27 fix7(反証 7-1 回目 w1-tools-r7-1-tools-1): リカバリー中止のあと CH3 を人側 → AI 側にした発進は数えない(あるときだけ。
    # 中止の無い本は前と同じ)。standing_starts と同じ t・ai の区間。2026-10-27 fix7 r2(反証 7-2 回目 w2-tools-r7-2-tools-1): 中立止め・
    # カメラの途切れのまま人側にして止まって終わった区間のあとも(verpm で止まっているかを見る)。2026-10-28 fix8 tools-decided: リカバリー
    # の途中(assist 2)で人が CH3 を人側にした区間のあとも(main session の決め = approvals 8「fix7 で残した読み方」①)
    ab = abort_resumes(t, ai, lab.get('assist'), lab.get('hold'), lab.get('ai_cmd_select'), ex['vt'], Df, ex['verpm'])
    if ab:
        by_t = {_r(float(t[i0]), 3): v for i0, v in ab.items()}
        for s in res['standing']:
            v = by_t.get(s['t_ai'])
            if v is not None:
                s['after_abort'] = v
                s['counted'] = False
    wins = flying_windows(sc.get('laps') or [], coll)
    res['flying'] = flying_summary(wins)
    res['windows'] = wins
    # 周の要約(quick_eval と同じ数え方)
    ps = [x for x in sc.get('passes') or [] if x.get('ai')]
    clean = [x['lap_s'] for x in sc.get('laps') or [] if x.get('clean')]
    ent = [x['entry_erpm'] for x in ps if x.get('entry_erpm')]
    res['laps'] = {'passes': len(ps), 'laps': len(sc.get('laps') or []), 'clean': len(clean), 'clean_laps_s': clean,
                   'best_clean_s': _r(min(clean), 3) if clean else None,
                   'median_clean_s': _r(st.median(clean), 3) if clean else None,
                   'r1_entry_median_erpm': float(st.median(ent)) if ent else None, 'entries_erpm': ent}
    cs = ss.collision_summary(coll_b)
    res['collisions'] = {'all': len(coll_b), 'strong': sum(1 for c in coll_b if c.get('strong')), 'max_g': cs['max_g'],
                         'n_ge8': cs['n_ge8'], 'by_place': cs['by_place'], 'text': ss.collision_text(cs, with_list=False)}
    if after:                            # 2026-10-27 fix6-2(あるときだけ。無ければ前と同じ): 人側にした直後の衝突の一覧と AI 側だけの数
        res['collisions']['after_ai'] = [[_r(float(c['t_s']), 2), _r(float(c['g']), 2), ss.place_of(c), c['after_ai_s']]
                                         for c in after]
        res['collisions'].update(ai_all=len(coll), ai_strong=sum(1 for c in coll if c.get('strong')))
        res['collisions']['text'] += '。うち人側にした直後(AI の区間の終わりから %.1f 秒以内)%d: %s' % (
            cc.AFTER_AI_S, len(after), '、'.join('%.1f 秒 %s %.2fG(AI の終わりから %.2f 秒)' % (x[0], x[2], x[1], x[3])
                                                for x in res['collisions']['after_ai']))
    res['midrun'] = midrun_changes(meta, sc, lab, ai, res['standing'], raw_dir, log, info['t0_us'])
    if res['midrun']:
        res['settings']['midrun'] = res['midrun']
        res['settings_text'] = settings_text(res['settings'])
        # 2026-10-27 fix5(反証 4 回目 c2): 3 周目の終わりより前(か時刻が分からない = params_end だけ)に設定を変えた発進は数えない
        # (手順書 7 章「走行中に OFF にした本は数えない」。前は設定の文に「途中で変えた」と書くだけで counted = True のまま、
        # best_standing・quick_eval_by_params の 1 章・2 章に入った)。変えたのが 3 周目の終わりのあとなら、その 3 周は走行開始の設定
        # なので数える(前と同じ)
        for s in res['standing']:
            e3 = s.get('end_s')
            if e3 is None:
                continue
            if any(m.get('t_s') is None or float(m['t_s']) < float(e3) for m in res['midrun']):
                s['midrun'] = True
                s['counted'] = False
    trial = trial_memo(sc.get('description'))
    if trial:                            # 2026-10-27 r3-r3-rec-3: 人が「試し」と書いた本(あるときだけ。無ければ前と同じ)
        res['trial'] = trial
        res['settings']['trial'] = trial
        res['settings_text'] = settings_text(res['settings'])
        for s in res['standing']:        # 数えない(best_standing・quick_eval_by_params の 1 章・2 章に入れない)
            s['trial'] = True
            s['counted'] = False
    res['takeovers_failure'] = (sc.get('takeovers') or {}).get('failure')
    res['ai_s'] = sc.get('ai_s')
    res['vbat'] = sc.get('vbat')
    res['speed_ctrl_flags'] = (sc.get('speed_ctrl') or {}).get('flags') or []
    return res


# ------------------------------------------------------------------ 表
def _f(x, fmt='%.2f'):
    return '—' if x is None else fmt % x


def counted(s):
    """数える発進: valid で、その記録の中でそれより前に衝突が無い(counted の欄が無い古い結果は valid)"""
    return bool(s.get('valid')) and s.get('counted', True) is not False and not s.get('coll_before')


def best_standing(res):
    """その記録のいちばん速い「止まってから 3 周」(動いてから)。valid で、それより前に衝突の無い発進だけ
    (2026-10-26 fix2 R2-rec-3。前はぶつかったあとに人が止めて戻した発進も入れていた)"""
    v = [s for s in res.get('standing') or [] if counted(s)]
    return min(v, key=lambda s: s['from_move_s']) if v else None


def launch_text(lr):
    if not lr or not lr.get('recorded'):
        return '—'
    if not lr.get('on_frames'):
        why = [e[2] for e in lr.get('events') or [] if e[1] == 'skip']
        return '出ず' + ('(%s)' % why[0] if why else '')
    return '%.2f 秒・%s' % (lr['on_s'] or 0.0, lr.get('end_reason') or '?')


def standing_table(results, only_r1=False):
    """止まってから 3 周の表(発進ごと)の行。only_r1 = 最初の R1 の起点まで行った発進だけ"""
    out = ['| No. | 記録 | 設定 | AI 側 [秒] | 動くまで | 動く → A1 | 動いてから 3 周 | AI 側から | 線から線の周 | '
           '距離の 3 周(AI 側から) | 衝突(強) | 発進の前進 | 確か |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    n = 0
    for r in results:
        for s in r.get('standing') or []:
            if s.get('to_r1_s') is None and (only_r1 or s.get('t_move') is None):
                continue
            n += 1
            col = s.get('collisions') or {}
            laps = '・'.join('%.2f' % x for x in s.get('laps_line_s') or [] if x is not None) or '—'
            ok = (('ok' + ('' if s.get('method') == 'anchor' else '(%s)' % s.get('method'))) if s.get('valid')
                  else (s.get('why') or '—'))
            if s.get('valid') and s.get('coll_before'):
                ok += '・数えない(この本の前の衝突 %d)' % s['coll_before']
            if s.get('midrun'):          # 2026-10-27 fix5 c2(あるときだけ)
                ok += '・数えない(途中で変えた)'
            if s.get('trial'):           # 2026-10-27 r3-r3-rec-3(あるときだけ)
                ok += '・数えない(試し)'
            if s.get('after_abort'):     # 2026-10-27 fix7(あるときだけ)。r2: 中立止め・カメラの途切れも(形の文は abort_text)
                ok += '・数えない(%s。%s・人側 %s 秒)' % (
                    RESUME_NAME, abort_text(s['after_abort']), _f(s['after_abort'].get('human_s'), '%.1f'))
            if (s.get('start_no') or 1) >= 2:
                ok += '・%d 回目の発進' % s['start_no']
            if s.get('note'):
                ok += '・' + s['note']
            out.append('| %s | %s | %s | %.2f | %s | %s | **%s** | %s | %s | %s | %s | %s | %s |' % (
                r.get('ai_no') or '—', r['log'], r.get('settings_text', ''), s['t_ai'], _f(s.get('move_after_s')),
                _f(s.get('to_r1_s') - s['move_after_s'] if s.get('to_r1_s') is not None and s.get('move_after_s') is not None
                   else None),
                _f(s.get('from_move_s')), _f(s.get('from_ai_s')), laps, _f(s.get('dist3_from_ai_s')),
                '%d(%d)' % (col.get('n', 0), col.get('strong', 0)) if col else '—', launch_text(s.get('launch')), ok))
    if not n:
        out.append('| — | (止まった状態からの発進なし) | | | | | | | | | | | |')
    return out


def flying_table(results):
    """走りながらの 3 周の窓の表(記録ごと)の行"""
    out = ['| No. | 記録 | 設定 | 窓 | 最速 | その窓の周 | 種類 | きれいな最速 | 中央値 | 25 秒以下 | 27 秒以下 | きれいな周 | '
           'ベスト | 中央値 | 衝突(強) |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        if 'error' in r:
            out.append('| %s | %s | %s | (%s) | | | | | | | | | | | |' % (r.get('ai_no') or '—', r['log'],
                                                                   r.get('settings_text', ''), r['error']))
            continue
        F, Lp, C = r['flying'], r['laps'], r['collisions']
        b = F.get('best') or {}
        out.append('| %s | %s | %s | %d | %s | %s | %s | %s | %s | %d | %d | %d | %s | %s | %d(%d)%s |' % (
            r.get('ai_no') or '—', r['log'], r['settings_text'], F['n'],
            _f(F['best_s']) + ('(確かめる: 7 秒より速い周)' if b.get('suspect') else ''),
            '・'.join('%.2f' % x for x in b.get('laps_s') or []) or '—', b.get('kind') or '—', _f(F['best_clean_s']),
            _f(F['median_s']), F['n_le25'], F['n_le27'], Lp['clean'], _f(Lp['best_clean_s']), _f(Lp['median_clean_s']),
            C['all'], C['strong'], after_ai_mark(C)))
    return out


NOTES = ['- 止まってから 3 周 = 3 周目の終わり(「3 周目の終わりの起点」A4 から、発進の場所 → A1 の車輪の距離だけ戻った時刻)'
         ' − 車が動いた時刻(前へ 0.2 m/s)。AI 側から = − AI 側にした時刻。距離の 3 周 = analysis_tt.md 5.2 の作り方(参考)。',
         '- 起点は section_scorecard の R1 の起点(偽の起点を外したもの)。衝突は label の決まり(水平 2.5G、強い = 4G 以上)。',
         '- 衝突(強)はその発進から 3 周目の終わりまで。同じ本の中で、それより前に衝突がある発進は「数えない」'
         '(ぶつかったあとに人が止めて戻した発進。quick_eval_by_params の 1 章・2 章にも入れない)。']


AFTER_AI_NOTE = ('- 衝突(人側にした直後も)= 2 章の衝突(強)は、AI 側の衝突と、人側にした直後(AI の区間の終わりから 0.5 秒以内)の人側の'
                 '衝突(car_check の「衝突」の行と同じ数え方。人が当たる直前に CH3 を人側にした当たり。手順書 7 章の「8G 以上 1 回」「同じ場所で'
                 ' 2 本続けて」「左 4.2 で SW の出口・LH」に数える)。「(うち人側にした直後 N)」がその数。1 章の「前の衝突」(数えない)と、その'
                 '発進から 3 周目の終わりまでの衝突(強)も同じ数え方。走りながらの 3 周の窓(ずっと AI 側の周)は AI 側だけ。')
ABORT_NOTE = ('- リカバリー中止のあとの走り直し = 同じ本の中で、一つ前の AI の区間がリカバリー中止(中立で人を待つ)で終わり、CH3 を人側 → AI 側'
              'にした発進。中止で止まった所(コースの途中)から出たかもしれない(手で HS に置き直しても車輪の距離は 0 なので、記録からは分から'
              'ない)ので、止まってから 3 周を数えない(最速・quick_eval_by_params の 1 章・2 章に入れない。⑤ collect.py の ON/OFF の比べにも'
              '入れない)。その 3 周の衝突は 1 章・2 章の衝突に入る。')
HOLD_NOTE = ('- 「中立止め X 秒から Y 秒」「カメラの途切れ X 秒から Y 秒」のリカバリー中止のあとの走り直し = 一つ前の AI の区間が、中立で停止中'
             '(中立止めが 0.5 秒以上・カメラの途切れ = フレームの無い中立止め)のまま人側にして止まって終わった発進(終わりで止まっていて、'
             'その区間で 0.5 m 以上進んだときだけ)。手順書 7 章「中立で停止中 → 車に近づく前に CH3 を人側」のとおりだと、中止が決まる前に'
             '人側になるので中止のフレームは書かれないが、止まった所から出たかもしれないのは同じ → 上と同じに数えない(2026-10-27 fix7 r2)。')
RECOVERY_NOTE = ('- 「リカバリーの途中 X 秒から Y 秒」のリカバリー中止のあとの走り直し = 一つ前の AI の区間が、リカバリー(当たった・止まった'
                 'あとのブレーキ・後退・前進など)の途中で人が CH3 を人側にして終わった発進(X = その区間の最後のリカバリーの始め・Y = 人側に'
                 'するまでの秒)。当たった所・止まった所(コースの途中)から出たかもしれず、HS に置き直したかは記録から分からないので、上と同じに'
                 '数えない(2026-10-28 fix8。main session の決め = approvals 8「fix7 で残した読み方」①)。')


def after_ai_mark(C):
    """本の衝突の欄の後ろの印(人側にした直後の衝突があるときだけ。2026-10-27 fix6-2)"""
    aa = (C or {}).get('after_ai')
    return '(うち人側にした直後 %d: %s)' % (len(aa), '・'.join('%s %.2fG' % (x[2], x[1]) for x in aa)) if aa else ''


def zero_byte_line(results):
    zb = [(r['log'], z) for r in results for z in r.get('zero_byte') or []]
    out = ['', '- 0 バイトで飛ばした npz: ' + '、'.join('%s/%s' % x for x in zb)] if zb else []
    # 2026-10-27 fix5(反証 4 回目 c1): 無い npz(書いている途中で落ちたシーン・車両の値が来なかったシーン)も(あるときだけ)
    mz = [(r['log'], z) for r in results for z in r.get('missing_npz') or []]
    if mz:
        out += ([] if out else ['']) + [
            '- 無いので飛ばした npz(VEHICLE・IMU・CAM_FRONT): ' + '、'.join('%s/%s' % x for x in mz),
            '- 上の npz のある本(%s)は記録が欠けている: 欠けた区間の衝突・8G 以上・周は表に入っていない → car_check の'
            '「記録が欠けている」の行・動画・紙で(手順書 7 章「いつでも」)' % '・'.join(sorted({x[0] for x in zb + mz}))]
    return out


def to_markdown(results, line_ahead_m=0.0):
    out = ['# 止まってから 3 周(tools/standing_3lap.py)', ''] + NOTES
    if any(r.get('trial') for r in results):     # 2026-10-27 r3-r3-rec-3(あるときだけ)
        out.append('- 試し = 説明欄の memo に人が「試し」と書いた本(手順書 7 章: 数えない)。止まってから 3 周を数えない(最速・'
                   'quick_eval_by_params の 1 章・2 章に入れない)。走行開始でサーバーが足す「計画と違うので試し(元 cand=…)」は数える。')
    if any(s.get('midrun') for r in results for s in r.get('standing') or []):     # 2026-10-27 fix5 c2(あるときだけ)
        out.append('- 途中で変えた = 走行中に設定を変えた本(events.jsonl の params・log.json の params_end・天井と自動合わせの途中 OFF・'
                   '発進の前進の終わり 10・フレームのゲインやトリム)。3 周目の終わりより前(か時刻が分からない)に変えた発進は数えない'
                   '(手順書 7 章「走行中に OFF にした本は数えない」。最速・quick_eval_by_params の 1 章・2 章に入れない)。')
    if any((r.get('collisions') or {}).get('after_ai') for r in results):     # 2026-10-27 fix6-2(あるときだけ)
        out.append(AFTER_AI_NOTE)
    if any(s.get('after_abort') for r in results for s in r.get('standing') or []):     # 2026-10-27 fix7(あるときだけ)
        out.append(ABORT_NOTE)
    if any((s.get('after_abort') or {}).get('kind') in ('hold', 'stall') for r in results for s in r.get('standing') or []):
        out.append(HOLD_NOTE)            # 2026-10-27 fix7 r2(中立止め・カメラの途切れのあとの走り直しがあるときだけ)
    if any((s.get('after_abort') or {}).get('kind') == 'recovery' for r in results for s in r.get('standing') or []):
        out.append(RECOVERY_NOTE)        # 2026-10-28 fix8 tools-decided(リカバリーの途中で人側にしたあとの走り直しがあるときだけ)
    if line_ahead_m > 0:
        out.append('- スタートの線 = 置いた場所の %.2f m 先(線から = 線を越えてから)。' % line_ahead_m)
    out += ['', '## 1. 止まってから 3 周(発進ごと)', ''] + standing_table(results)
    out += ['', '## 2. 走りながらの 3 周の窓(記録ごと)', ''] + flying_table(results)
    out += zero_byte_line(results)
    return '\n'.join(out) + '\n'


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run(logs, raw_dir=None, labels_dir=None, bench_path=None, line_ahead_m=0.0):
    names = candidate_names((ss.load_bench(bench_path) or {}) if bench_path else None)
    out = []
    for lg in logs:
        try:
            out.append(analyze_log(lg, raw_dir, labels_dir, names, line_ahead_m))
        except Exception as e:      # noqa: BLE001  1 本が読めなくてもほかは出す
            out.append({'log': lg, 'error': '%s: %s' % (type(e).__name__, e), 'zero_byte': zero_byte_files(lg, raw_dir)})
            mz = missing_npz_files(lg, raw_dir)
            if mz:                  # 2026-10-27 fix5 c1(あるときだけ)
                out[-1]['missing_npz'] = mz
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='止まってから 3 周の時間(記録から)')
    ap.add_argument('logs', nargs='*', help='記録の名前(YYYYMMDD-HHMMSS)')
    ap.add_argument('--day', help='その日の AI 走行の全部(YYYYMMDD)')
    ap.add_argument('--since', help='HHMM 以後の記録だけ(--day と一緒に)')
    ap.add_argument('--raw', default=None)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--bench', default=str(config.DATA_DIR / 'bench_candidates.json'))
    ap.add_argument('--line-ahead-m', type=float, default=0.0)
    ap.add_argument('--json', help='結果の JSON')
    ap.add_argument('--md', help='表の Markdown(無ければ標準出力)')
    a = ap.parse_args(argv)
    logs, skipped = list(a.logs), []
    if a.day:
        dl, skipped = day_logs(a.day, a.raw, a.since)
        logs += [x for x in dl if x not in logs]
    if not logs:
        print('記録がありません(--day か記録の名前)', file=sys.stderr)
        for lg, why in skipped:
            print('飛ばした: %s(%s)' % (lg, why), file=sys.stderr)
        return 1
    res = run(logs, a.raw, a.labels, a.bench, a.line_ahead_m)
    md = to_markdown(res, a.line_ahead_m)
    if skipped:
        md += '- 飛ばした記録: ' + '、'.join('%s(%s)' % x for x in skipped) + '\n'
    if a.json:
        Path(a.json).write_text(json.dumps({'schema': SCHEMA, 'line_ahead_m': a.line_ahead_m, 'skipped': skipped,
                                            'results': res}, ensure_ascii=False, indent=1, default=_json_default),
                                encoding='utf-8')
    if a.md:
        Path(a.md).write_text(md, encoding='utf-8')
    else:
        print(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
