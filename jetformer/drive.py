"""AI走行: 学習結果(TensorRTエンジン)で車を操作する

カメラの新しいフレーム(30fps)ごとに:
  1. その撮像時刻までのToF・IMU(直近10サンプル)・車両の値を集める(データセットの aligned と同じ規則)
  2. normalize_inputs(学習と共通)→ TensorRT推論 → 未来10フレームの [ステアリング, スロットル] µs
  3. 「先読み」のフレームを採用 → ゲイン・トリム → PCA9685
プロポのCH3がAI側(SELECT=1、LED紫)のときだけ出力が車に届く。CH3を戻せば常に人の操縦に戻る。

モデルへの「現在の操作値」は、車に届いている操作に合わせる:
  人が操縦中 → プロポの値 / AI側 → 前のフレームの予測のうち「このフレームの時刻」に当たる値(ゲイン前)
  / 中立を出している間 → 中立
ゲイン後の値を戻すと、ゲイン>1のとき自分の出力を真似て際限なく大きくなる。また先読みで採用した値
(数フレーム先)を戻すと、毎フレーム先へ進んだことになって動きが速回しになる。どちらも学習時と違うので避ける。

安全のための動作(どれも中立を出す): センサーの値が途切れた・古い / 推論結果が異常 / AI側なのにプロポのCH3が
届いていない(送信機の電源断・圏外では基板のSELECTがAI側のまま残るため)/ PCA9685への書き込み失敗。
走行を始めたページからの問い合わせが3秒途絶えたら走行を止める。

モデルの種類(方策、jetformer/policies)はどれも出力 actions [1,10,2] を持つので、先読み・ゲイン・ToFガード・補助・ラップ・
CH3 の交代・センサーの途切れで中立はそのまま効く。推論の直後・先読みの前に、方策の後処理(model_config.json の
postprocess。tcp の制御器)と予測の重ね合わせ(ensemble_j > 0)を通せる(policies.post.Stage。評価の道具と同じ手順)。
エンジンには基本の5つの入力を全部渡す(エンジンが使わない入力は読まれないだけ)。

区間の速度上限(safety_caps v1.1: R1 2.5・GM 2.2・DE 3.0 m/s。2026-10-21): パラメータ zone_caps(既定 0 = OFF、
起動のたびに OFF)。ON/OFF に関係なく、毎フレーム区間・上限・「上限ならこう出した」スロットル・ブレーキを計算して
AI の記録(AI_CMD の cap_*)に残す(影の判定)。車への出力を変えるのは zone_caps = 1 のときだけ。上限を掛けるのは
車に出す直前のスロットル(ゲイン・ToF ガード・補助のあと。上限は速さ [m/s] の決まりで、ゲインの前に掛けるとゲインで
上限の速さが変わるため。G1 の再生も記録の out = この値で判定した)。上限は前進を弱めるだけ(PWM は入力以上)。
計算が例外になったら、その走行のあいだは上限を外す(OFF と同じ出力)・記録とメッセージに残す。
v1.1 は位相が不確か(unknown: 最初の R1 の前・R1 から 16 秒超・1 秒以上止まったあと)でも 3.0 m/s で切る(4 つめの上限。
ユーザーの了承は 3 区間だけなので、画面の ON の確認で示す。ZONE_CAPS_SPEC['unknown_note'])。
走行中に OFF → ON にはできない(set_params が断る。1 走行 = 1 設定)。ON → OFF は走行中でもできる。

発進の前進(assist.LaunchBoost、2026-10-26、jetformer_reports/fix_20261026/protocol_v3.md 14 章): パラメータ launch_boost
(既定 0、起動のたびに 0・値も既定へ)。CH3 を AI 側にした最初のフレームで止まっていて前が空いていれば(ToF の TTC の
直近 5 フレームの中央値 ≥ 3.3 秒)、スロットルを min(モデル, 中立 − launch_us) にする(前進だけ・舵はモデルのまま)。
順番は ゲイン → ToF ガード → 発進の前進(StartKick の場所)→ リカバリー → 区間の上限 → 速さの天井。走行中は 0 → 1 に
できない(1 → 0 はできる)・値は変えられない。OFF のとき、PCA9685 に書く値と記録の今までの欄は手を入れる前と同じ
(新しい欄 launch_on・launch_end は False・0)。記録: AI_CMD の assist = 5・launch_on・launch_end、stats の launch_boosts。

リカバリーの後ろの見張り(2026-10-26 fix2 safety2-1): パラメータ rec_back_guard(既定 0・起動のたびに 0)。走行タブの
「リカバリーの試験の選択肢」のスイッチ(2026-10-26 fix3 から。app.js の OPT_SWITCHES。fix2 のときは画面に欄が無く API だけ)か
API(POST /api/drive/params)で入れる。走行開始では送らない(サーバーの値のまま)。基準の設定・ブロックでは変えない(app.js の
benchApply が入れない。ブロックの params に書いても web/bench.py の SWITCH_KEYS が捨てて警告。2026-10-27 fix4)。走行中は
0 → 1 にできない(下の _opt_rules)。走行開始の確認の文に ON の行。1 のとき
assist.Recovery が、どの段でも後ろへ動いたことを数え(下がった時間が rev_s に届いたら惰性)、後ろへ動いている・直前まで
動いていたときの衝撃ですぐ前進に移る。0 のとき PCA9685 に書く値と記録は手を入れる前と同じ。見張りが手順を変えたら
events.jsonl に recovery_guard(what = 何・phase = 段・backed_s = 下がった秒・run・cam_epoch_us。hold_long は hold_s も)。

中立止めでリカバリーを残す(2026-10-26 fix3 safety1-1): パラメータ rec_hold_keep(既定 0・起動のたびに 0。画面の試験の
選択肢のスイッチか API で入れる。走行開始では送らない。基準の設定・ブロックでは変えない)。0 のとき今まで通り、中立止め(hold)の最後に補助の手順を全部戻す(リカバリーの中止・途中・armed も消える。
中止が解けるとモデルの弱い前進が壁を押し続け、止まった車ではリカバリーが二度と始まらない)。1 のとき、AI 側のままの
中立止め(最新の車両の値が AI 側・dry run でない)では assist.Recovery を戻さない(中止は中止のまま・途中の段は続きから・
armed・衝撃・あきらめの数も残す)。戻すのは CH3 を人側にしたとき・人側の中立止め・走行の開始(Assist を作り直す)。
後退の受け付け(リカバリー OFF のとき)は今まで通り戻す。0 のとき PCA9685 に書く値と記録は手を入れる前と同じ。
残した中立止めの間の衝撃(2026-10-26 fix3 r1-rs1-1): 中立止めの間は Recovery に IMU が届かない(impact_g は処理したフレームの
直近 0.15 秒だけ)ので、残した中立止めのあとの最初のフレームで、前に処理したフレームのあとに来た IMU の水平 G の最大を
hold_g として渡す。後退側・下がる惰性の段で 2.5G 以上ならすぐ前進(出来事 recovery_guard の what = rear_impact_hold)。
前は、後退の途中に 0.1 秒より長い中立止めをはさんで後ろへ当たると見落とし、明けたあと 'reverse' が後ろの物を最大 約 0.8 秒押した。
残した中立止めの間の後退と長さ(2026-10-27 fix3 直す 2 回目 r2-r2-rs-1・rs-2): 残した中立止めの間、前に見た車両の値のあとに来た
サンプルで後ろへ動いていた秒(eRPM > assist.BACKING_ERPM。隣り合う 2 つのどちらかが超えていたら、その間を全部。値が来なかった
間も)を数え、明けた最初のフレームで hold_since(中立を書き始めた時刻)・hold_back_s として渡す(assist.Recovery._after_hold:
下がった時間に足す・段の時間は中立を出していた間は進めない・前に処理したフレームから 0.5 秒より長い中立止めのあとは後退側の段と止まっている時間の
途中を中止(giveup)にして人を待つ。出来事 recovery_guard の what = back_hold / hold_long(hold_long は見ていない秒 hold_s も))。
長い中立止めのあとの中止は、2026-10-27 fix3 直す 3 回目 r3-r3-safety-1・2 で assist.Recovery の中で広げた(段なし・'escape' の
途中でも、armed で、明けたフレームで止まっているか中立止めの間の衝撃が IMPACT_G 以上なら中止。中止にするとき後ろへ動いていれば、
先に前進で止める段 'halt'(最大 0.3 秒)。drive.py が渡すもの・書くものは同じ。くわしくは assist.py の 1.「自動リカバリー」)。
前は明けたフレームの dt(0.2 秒まで)しか数えず、段の時間も中立止めの間に進んだので、'reverse' の始めの中立止めのあと
やり直し(+50 µs)で中立止めなしより後ろへ多く下がり、2〜5 秒の中立止めのあとは明けて 0.02〜0.35 秒で後退を出した。
rec_hold_keep = 0 は今まで通り(hold_since は None のまま)。
反証 4 回目の直し(2026-10-27 fix5 係 safety。rec_hold_keep = 1・rec_back_guard = 1 の日だけ効く。どちらかが 0 なら PCA9685 に
書く値・記録・出来事は直す前とビット単位で同じ): 残した中立止めの間の後退の秒のうち、車両の値が来なかっただけの間(前後の
サンプルはどちらも後ろへ動いていない)の秒を hold_unseen_s として別に渡す(s3。assist.Recovery は下がり始めたあとだけ下がった
時間に数える)。helper.step に rec_keep(両方 1)を渡し、発進の前進・発進の補助が出したフレームでも Recovery の前に見た
フレームの時刻を進める(s2)。衝撃だけの中止を走っている車にしないのは assist.py の中(s1)。
反証 6-1 回目の直し(2026-10-27 fix6 係 safety。同じく両方 1 の日だけ効く。どちらかが 0 ならビット単位で同じ): 両方 1 の日は
PCA9685 に書けたスロットル(人側は「分からない」)を assist.EscTrack に入れ、残した中立止めの間に車両の値が HOLD_VEH_GAP_S より
来なかった間は、前に見た値が HOLD_VEH_FAST_ERPM より速く後ろへ動いていたなら全部、そうでなければ EscTrack が見積もった
「ESC が後退で走らせていたかもしれない秒」を数える(前に見た値が後ろへ動いていたなら、その値で Recovery が数えたフレームの
分は除く)。見積もりは hold_back_s とは別の hold_rev_s で渡し(hold_unseen_s は渡さない)、assist.Recovery が後退を 0.2 秒残す
ところまで下がった時間に足す(s3-a)。前は、値が古いまま出した後退側で車が下がり始めても数えず、明けてから rev_s をまるごと
下がり直す(後ろへ多く)か、値が来なかった間を全部二重に数えて後退を早く切る(前の壁へ escape)かが、明けたサンプルの速さで
分かれた。'coast' の続きの直し(s3-b)は assist.py の中。

発進の前進が出さない区間の発進の補助(2026-10-26 fix3 safety1-2): パラメータ launch_no_kick(既定 0・起動のたびに 0。
画面の試験の選択肢のスイッチか API で入れる。走行開始では送らない。基準の設定・ブロックでは変えない)。launch_boost = 1 で発進の前進が「出さない」と決めた区間(前が近い・ToF が無い・TTC が
そろわない・車両の値が古い・始めのフレームで終わり・中立止めのあと など)では、1 のとき StartKick も出さない
(0 は protocol_v3 14 章どおり今まで通り StartKick)。記録: 出来事 launch_boost の skip に no_kick = True。
launch_boost = 0 のときは何も変えない。「出さない」になった区間は、途中で 0 にしても StartKick を出さない(CH3 を人側にするまで。
fix3 r1-rs1-2: 前は 1 → 0 にした最初のフレームで StartKick の 2 秒の窓が新しく開き、リカバリーの中止・後退を上書きした)。
発進の補助 start_kick も、走行中は 0 → 1 にできない(1 → 0 はできる。set_params。同じ理由で、区間の途中に新しい窓が開くため。
fix3 r1-rs1-2)。
試験の選択肢の 3 つ(rec_back_guard・rec_hold_keep・launch_no_kick。OPT_NAMES)は、走行中は 0 → 1 にできない(1 → 0 は
できる。1 走行 = 1 設定。画面の試験の選択肢のスイッチと同じ決まり。_opt_rules。断るときサーバーは 400)。rec_hold_keep は
rec_back_guard が ON のときだけ使える(set_params も走行開始 start も。見張りを OFF にすると rec_hold_keep も OFF。
走行ループも両方 1 のときだけ残す): 残したリカバリーは、見張りが無いと後退の途中の中立止めのあとに後ろへ当たっても
やり直しで後ろを押す(今までは中立止めの reset が手順を偶然切っていた)。

走行中に変えた設定(2026-10-26 fix2 state2-2): 走行中の set_params で値が変わった項目は、events.jsonl に params
(changed = {項目: [前, 後]}・epoch_us = 変えた時刻(AI_CMD と同じ UNIX µs))を書く。止めるとき、変えた項目が 1 つでも
あれば log.json(と残りの scene.json)の drive に params_end(走り終わりの値)と params_changes(変えた順の一覧)を足す。
変えなかった走行の log.json は今までと同じ(drive.params = 走行開始の値)。

AI の時間(stats['ai_s']、2026-10-21): CH3 が AI 側で dry run でない時間の和。道具(label_events の dt_s)と同じ決め方で、
ベンチマークの「AI の時間が数える時間 + 1 分を過ぎたら止める」を画面で見るため。数えるだけで出力は変えない。
"""

import collections
import json
import math
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import assist as assist_mod
from . import clock, config, modes, policies
from . import safety_caps
from . import speed_auto
from .laps import LapCounter
from .policies import post as post_mod

SETTINGS_PATH = config.DATA_DIR / 'drive_settings.json'
RUNS_DIR = config.DATA_DIR / 'runs'
ENGINE_FILE = 'jetformer_fp16.engine'
IMU_STEPS = 10                     # model.IMU_STEPS(torchを読み込まずに使うため複製)
PWM_CENTER, PWM_SCALE = 1500.0, 500.0
AI_GAP_S = 0.2                     # AI の時間: これより空いたフレームの間は 1 フレーム分だけ数える(tools/label_events.GAP_S と同じ)

PARAM_LIMITS = {
    'steer_gain': (0.0, 2.0),      # ステアリングのゲイン(中心からの切れ角を何倍にするか)
    'throttle_gain': (0.0, 2.0),   # 速度のゲイン(中立からのスロットル量を何倍にするか)
    'steer_trim_us': (-200.0, 200.0),
    'lookahead': (1, 10),          # 何フレーム先の予測を使うか(1 = 約33ms先)
    'osc_percent': (-10.0, 10.0),  # PCA9685の時計補正 [%]
    'esc_reverse': (0, 1),         # 1 = 後退の指令の前に中立を挟む(このESCは中立を挟まないと後退しない)
    'recovery': (0, 1),            # 1 = ぶつかって止まったら自動で少し下がってからモデルへ戻す
    'rev_us': (150.0, 500.0),      # リカバリーで下がる強さ(中立からのµs)
    'rev_s': (0.2, 1.2),           # リカバリーで後退を出し続ける時間(下がり始めてから)
    'rec_back_guard': (0, 1),      # 1 = リカバリーの後ろの見張り(どの段でも下がった時間を数える・後ろの衝撃ですぐ前進。既定 0・起動で 0)
    'rec_hold_keep': (0, 1),       # 1 = AI 側のままの中立止めでリカバリーを戻さない(中止・途中・armed を残す。既定 0・起動で 0。fix3)
    'tof_guard': (0, 3),           # ToFガード 0=OFF 1=記録だけ(動きは変えない) 2=減速 3=減速+ハンドル
    # 方策の後処理(tcp の制御器だけが読む)・予測の重ね合わせ(どの方策にも)
    'speed_cap': (0.0, 5.0),       # 速度の上限 [m/s](0 = なし)。tcp の制御器が前進を弱めるだけ(速くはしない)
    'speed_scale': (0.5, 1.5),     # 予測した目標速度に掛ける倍率(tcp_mode 1〜3 だけで効く)
    'tcp_mode': (0, 3),            # tcp の出し方 0=操作の枝だけ(既定・比較で評価した出力) 1=ハイブリッド 2=混ぜる 3=通過点を追う
    'ensemble_j': (0, 3),          # 予測の重ね合わせ: 何フレーム前の予測まで混ぜるか(0 = オフ)
    'ensemble_m': (0.0, 2.0),      # 予測の重ね合わせ: 古い予測を軽くする強さ(重み exp(-m·j))
    'zone_caps': (0, 1),           # 区間の速度上限 v1.1(R1 2.5・GM 2.2・DE 3.0 m/s)1 = 出力に掛ける 0 = 記録だけ(影の判定)
    'start_kick': (0, 1),          # 1 = 発進の補助(AI 側にした直後、止まっていてモデルが進まないとき弱い前進を出す。assist.StartKick)
    # 速さの天井・速さのゲインの自動合わせ(2026-10-25、speed_control_20261025/spec.md)。どちらも既定 OFF・起動のたびに OFF・
    # 走行中は ON にできない(OFF はできる)・値は走行中に変えられない。OFF でも影の判定を記録する(AI_CMD の ceil_*・sa_*)
    'speed_ceiling': (0, 1),       # 1 = 速さの天井(全体 ceil_all_mps・左旋回中 ceil_left_mps)。前進を弱めるだけ。ブレーキなし・舵は変えない
    'ceil_all_mps': (4.0, 5.0),    # 全体の天井 [m/s]
    'ceil_left_mps': (3.5, 4.5),   # 左旋回中の天井 [m/s](全体より大きければ全体)
    'speed_auto': (0, 1),          # 1 = 速さのゲインの自動合わせ(speed_ceiling = 1 のときだけ)
    'auto_target_mps': (3.8, 4.4), # 周の速さ p90 の目標 [m/s]
    'auto_vref_v': (7.0, 9.0),     # throttle_gain を決めた電圧 [V](較正の値。起動しても残す)
    # 発進の前進(2026-10-26、protocol_v3 14 章)。既定 OFF・起動のたびに 0(値も既定へ)・走行中は ON にできない・値は変えられない
    'launch_boost': (0, 1),        # 1 = 発進の前進(AI 側にした最初のフレームから前進を少なくとも launch_us。assist.LaunchBoost)
    'launch_us': (150.0, 300.0),   # 出す前進の下限 [µs](モデルの中立から。速さのゲインは掛けない)
    'launch_mps': (2.0, 3.0),      # 0.1 秒先の速さの見込みがこれに届いたら終わり [m/s]
    'launch_max_s': (0.5, 2.0),    # これより長く出さない [秒]
    'launch_no_kick': (0, 1),      # 1 = 発進の前進が「出さない」と決めた区間では発進の補助(StartKick)も出さない(既定 0・起動で 0。fix3)
}
START_KICK_TTC_S = 1.0             # 発進の補助を出さない ToF の衝突までの時間 [秒](これより近い)
HOLD_VEH_FAST_ERPM = 1000.0       # 残した中立止めの間に車両の値が来なかったとき、前に見た値がこれより速く後ろへ動いていたら、その間は全部
                                  # 下がっていたとして数える('coast' が続く速さ。中立の惰性ではすぐ止まらない。2026-10-27 fix6 safety r1)
HOLD_VEH_GAP_S = 0.06              # 残した中立止めの間、車両の値がこれより空いたら後ろへ動いていたとして数える(2026-10-27 fix3 r2-r2-rs-1)
# 起動のたびに既定へ戻す項目(drive_settings.json から読まない)。区間の速度上限は ON のまま別の日に持ち越さない
RESET_ON_BOOT = ('tcp_mode', 'speed_scale', 'zone_caps',
                 'speed_ceiling', 'ceil_all_mps', 'ceil_left_mps', 'speed_auto', 'auto_target_mps',
                 'launch_boost', 'launch_us', 'launch_mps', 'launch_max_s',
                 'rec_back_guard',      # 走行の制御に足すものは既定 OFF・起動で OFF(protocol_v3 の決まり。2026-10-26 fix2)
                 'rec_hold_keep', 'launch_no_kick')     # 2026-10-26 fix3 safety1-1・1-2
SPEED_CTRL_SWITCHES = ('speed_ceiling', 'speed_auto')                               # 走行中は 0 → 1 にできない
SPEED_CTRL_VALUES = ('ceil_all_mps', 'ceil_left_mps', 'auto_target_mps', 'auto_vref_v')   # 走行中は変えられない
LAUNCH_VALUES = ('launch_us', 'launch_mps', 'launch_max_s')                         # 発進の前進の値(走行中は変えられない)
LAUNCH_KEYS = ('launch_boost',) + LAUNCH_VALUES
LAUNCH_FRESH_NS = 100_000_000       # 発進の前進: ToF・車両の値がこれより古ければ始めない・終わる(0.1 秒)
LAUNCH_VERSION = '2026-10-26'
ENGINE_CHECK_FILE = 'engine_check.json'          # エンジンの確認(精度・速さ)の結果。あれば走行タブに出す(精度の不合格は走らせない)
# 走行タブで最初に選ぶ「推奨のモデル」(方策の比較などの評価で決めたもの。tools/policy_zoo.py recommend で書く)。
# 無ければ今まで通り(前回のモデル → 一番新しいモデル)
RECOMMENDED_PATH = config.DATA_DIR / 'drive_recommended.json'
GUARD_NAMES = {0: '', 1: '減速', 2: 'ブレーキ'}
FLOOR_REF_PATH = config.DATA_DIR / 'tof_floor_ref.npy'      # ToFガードの床の基準(自由走行から作る)
DEFAULT_PARAMS = {'steer_gain': 1.0, 'throttle_gain': 0.5, 'steer_trim_us': 0.0, 'lookahead': 1, 'osc_percent': 0.0,
                  'esc_reverse': 1, 'recovery': 1, 'rev_us': assist_mod.DEFAULT_REV_US, 'rev_s': assist_mod.DEFAULT_REV_S,
                  'tof_guard': 1, 'speed_cap': 0.0, 'speed_scale': 1.0, 'tcp_mode': 0, 'ensemble_j': 0, 'ensemble_m': 0.5,
                  'zone_caps': 0, 'start_kick': 0,
                  'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'auto_target_mps': 4.27,
                  'auto_vref_v': 8.35,
                  'launch_boost': 0, 'launch_us': 250.0, 'launch_mps': 2.5, 'launch_max_s': 1.2,
                  'rec_back_guard': 0, 'rec_hold_keep': 0, 'launch_no_kick': 0}


def params_changed(prev, new):
    """set_params の前と後で値が変わった項目 {項目: [前, 後]}(2026-10-26 fix2 state2-2。数は 1e-9 より大きい差だけ)"""
    out = {}
    for k, v in new.items():
        a = prev.get(k)
        try:
            same = a is not None and abs(float(a) - float(v)) <= 1e-9
        except (TypeError, ValueError):
            same = a == v
        if not same:
            out[k] = [a, v]
    return out


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _trained_modes_from_dataset(dataset, stride):
    """古い学習結果用: データセットの学習用シーンに含まれていたジャンル(torchを使わずに数える)"""
    root = config.DATASET_DIR / (dataset or '')
    splits = _read_json(root / 'splits.json') if dataset else None
    if not splits:
        return []
    found = set()
    for scene in splits.get('train', []):
        try:
            a = np.load(root / 'aligned' / (scene + '.npz'))
            ok = a['future_valid'] & a['vehicle_valid']
            mode = a['mode'] if 'mode' in a else np.zeros(len(ok), np.int16)
            ok &= (mode != modes.UNSET) & ((np.arange(len(ok)) % max(int(stride or 1), 1)) == 0)
            found.update(int(m) for m in np.unique(mode[ok]))
        except (OSError, KeyError, ValueError):
            continue
    return sorted(found)


def model_info(name):
    """走行に使える学習結果の情報。エンジンが無ければ None"""
    d = RUNS_DIR / name
    if '/' in name or '..' in name or not (d / ENGINE_FILE).exists():
        return None
    cfg = _read_json(d / 'model_config.json')
    if cfg is None:
        return None
    res = _read_json(d / 'result.json') or {}
    run = _read_json(d / 'run.json') or {}
    trained = cfg.get('trained_modes')
    if trained is None:            # 古い学習結果: 学習用シーンに含まれていたジャンル(分からなければ走らせない)
        trained = _trained_modes_from_dataset(cfg.get('dataset'), cfg.get('stride'))
    trained = [m for m in trained if 0 <= m < cfg['num_modes']]
    policy = policies.policy_of(cfg)            # 'policy' の無い古い学習結果は jetformer_v0
    ok, why = policies.drivable(cfg)
    check = engine_check_of(d)
    if ok and check and check.get('accuracy_ok') is False:      # 今のエンジンが精度の確認に落ちている → 走らせない
        ok, why = False, engine_gate_reason(check)
    return {
        'name': name, 'dataset': cfg.get('dataset'), 'created_at': run.get('created_at'),
        'state': res.get('state'), 'best_val_us': res.get('best_val_us'),
        'best_val_per_mode_us': res.get('best_val_per_mode_us', {}),
        'num_modes': cfg['num_modes'], 'modes': cfg.get('modes', []), 'trained_modes': trained,
        'steer_center_us': cfg.get('steer_center_us'), 'throttle_neutral_us': cfg.get('throttle_neutral_us'),
        'stride': cfg.get('stride'), 'future_steps': cfg.get('future_steps', 10),
        'command_input': cfg.get('command_input', True),
        'policy': policy, 'policy_title': policies.title_of(policy),
        'policy_cfg': cfg.get('policy_cfg') if 'policy' in cfg else dict(policies.V0_FROZEN_CFG),
        'inputs': policies.inputs_of(cfg), 'outputs': policies.outputs_of(cfg),
        'postprocess': cfg.get('postprocess'), 'drivable': ok, 'why_not': why,
        'engine_precision': (cfg.get('engine') or {}).get('precision', 'fp16'),
        'engine_check': check,
    }


def engine_check_of(run_dir):
    """<学習結果>/engine_check.json(エンジンの確認: 実フレームで TensorRT と PyTorch FP32 の差・速さ)→ 表示用の dict。
    無い・今のエンジンの確認でない(エンジンの大きさ・更新時刻が違う)なら None。
    精度の確認(accuracy_ok)に落ちたエンジンは model_info が走行できない(drivable False)にする。速さだけの不合格は警告だけ"""
    c = _read_json(Path(run_dir) / ENGINE_CHECK_FILE)
    if not isinstance(c, dict):
        return None
    try:
        stt = (Path(run_dir) / ENGINE_FILE).stat()
        eng = c.get('engine') or {}
        if int(eng.get('size', -1)) != stt.st_size or int(eng.get('mtime_ns', -1)) != stt.st_mtime_ns:
            return None
    except (OSError, TypeError, ValueError):
        return None
    keys = ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'gate_us', 'run_numpy_median_ms', 'checked_at', 'note')
    return {k: c.get(k) for k in keys}


def engine_gate_reason(check):
    """精度の確認に落ちたエンジンの「走れない理由」(走行タブに出す)"""
    try:
        detail = '(TensorRT と PyTorch FP32 の差 最大 %.2f µs > %.1f µs)' % (float(check['actions_max_us']),
                                                                         float(check.get('gate_us') or 2.5))
    except (KeyError, TypeError, ValueError):
        detail = ''
    return ('エンジンが精度の確認に落ちています%s。エンジンを作り直して確認に合格するまで走行に使えません'
            '(学習タブで作り直す・tools/build_engine.py)' % detail)


def recommended():
    """推奨のモデル(data/drive_recommended.json)→ {'run', 'reason', 'source', 'decided_at'}。無い・壊れていれば None"""
    r = _read_json(RECOMMENDED_PATH)
    if not isinstance(r, dict) or not isinstance(r.get('run'), str) or not r['run']:
        return None
    return {k: r.get(k) for k in ('run', 'reason', 'source', 'decided_at')}


def save_recommended(run, reason='', source='', decided_at=None):
    """推奨のモデルを書く(走行タブが最初に選び、一覧の先頭に出す)。run が走行に使えなくても書く(画面では選べないまま)"""
    if not run or '/' in run or '..' in run:
        raise ValueError('学習結果の名前が正しくありません: %r' % run)
    rec = {'run': run, 'reason': reason or '', 'source': source or '',
           'decided_at': decided_at or time.strftime('%Y-%m-%dT%H:%M:%S')}
    RECOMMENDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(RECOMMENDED_PATH) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(tmp, RECOMMENDED_PATH)
    return rec


def list_models():
    """推奨のモデル(recommended())を先頭に、残りは新しい順(作成日時、無ければ名前)。
    推奨のモデルには recommended True と推奨の理由(recommended_reason)を付ける"""
    if not RUNS_DIR.exists():
        return []
    models = [m for m in (model_info(d.name) for d in RUNS_DIR.iterdir()) if m]
    models.sort(key=lambda m: (m['created_at'] or '', m['name']), reverse=True)
    rec = recommended()
    for m in models:
        m['recommended'] = bool(rec) and m['name'] == rec['run']
        if m['recommended']:
            m['recommended_reason'] = rec.get('reason') or ''
    return [m for m in models if m['recommended']] + [m for m in models if not m['recommended']]


def _reference_from_dataset(dataset):
    """古い学習結果用: データセットから(ステアリング中心, スロットル中立)を求める"""
    from .train_data import JetFormerDataset
    root = config.DATASET_DIR / (dataset or '')
    if not dataset or not (root / 'splits.json').exists():
        return PWM_CENTER, PWM_CENTER
    return JetFormerDataset(str(root), 'train').action_reference()


def assemble_inputs(cam_t, tof_buf, imu_buf, veh_buf, max_age_ns, imu_max_age_ns):
    """撮像時刻 cam_t 以前の最新のToF・直近10個のIMU・最新の車両値を選ぶ(dataset._aligned_arrays と同じ規則)。

    tof_buf: [(t_ns, depth_raw)]、imu_buf: [(t_ns, [ax,ay,az m/s^2, gx,gy,gz rad/s])]、veh_buf: [(t_ns, dict)]
    戻り値 (tof_raw, imu[10,6], vehicle_dict, None) または (None, None, None, 止める理由)"""
    tof = [x for x in tof_buf if x[0] <= cam_t]
    if not tof:
        return None, None, None, 'ToFの値が来ていません'
    if cam_t - tof[-1][0] > max_age_ns:
        return None, None, None, 'ToFの値が古い(%dms)' % ((cam_t - tof[-1][0]) / 1e6)
    imu = [x for x in imu_buf if x[0] <= cam_t]
    if len(imu) < IMU_STEPS:
        return None, None, None, 'IMUの値が足りません'
    if cam_t - imu[-1][0] > imu_max_age_ns:
        return None, None, None, 'IMUの値が古い(%dms)' % ((cam_t - imu[-1][0]) / 1e6)
    veh = [x for x in veh_buf if x[0] <= cam_t]
    if not veh:
        return None, None, None, '車両(PWM)の値が来ていません'
    if cam_t - veh[-1][0] > max_age_ns:
        return None, None, None, '車両(PWM)の値が古い(%dms)' % ((cam_t - veh[-1][0]) / 1e6)
    hist = np.stack([x[1] for x in imu[-IMU_STEPS:]]).astype(np.float32)
    return tof[-1][1], hist, veh[-1][1], None


def vehicle_state(veh, prev_cmd, ref, applied):
    """モデルに渡す [steering_us, throttle_us, mode_us, erpm](学習時は人の操作値)。

    prev_cmd: 前のフレームの予測のうち、このフレームの時刻に当たる値(ゲイン前)
    applied: 前のフレームでAIの出力を出したか(中立で止めていたら False)"""
    if veh['select']:
        steer, throttle = prev_cmd if (applied and prev_cmd is not None) else ref
    elif veh['valid'][0] and veh['valid'][1]:
        steer, throttle = veh['pwm'][0], veh['pwm'][1]
    else:
        steer, throttle = ref
    return np.array([steer, throttle, veh['pwm'][2], veh['erpm']], dtype=np.float32)


def predicted_now(prev_pred, prev_cam_t, cam_t, fps):
    """前のフレームの予測 [10,2] から、撮像時刻 cam_t に当たる値を取り出す(取りこぼしたフレーム分も進める)"""
    gap = int(round((cam_t - prev_cam_t) * fps / 1e9))
    k = min(max(gap, 1), len(prev_pred)) - 1
    return float(prev_pred[k, 0]), float(prev_pred[k, 1])


def _num(x, nd=2):
    """表示用の数(数値でなければ None。JSON に NaN を入れない)"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def post_view(pinfo, v_now):
    """後処理の情報 → 画面の表示用(st['now']['post'])"""
    if not pinfo:
        return None
    return {'mode': int(pinfo.get('mode', -1)), 'branch': str(pinfo.get('branch', '')),
            'v_hat': _num(pinfo.get('v_hat')), 'v_cmd': _num(pinfo.get('v_cmd')), 'v_now': _num(v_now),
            'v_curv': _num(pinfo.get('v_curv')), 'cap': _num(pinfo.get('cap'))}


def uses_post(p):
    """走行のパラメータが後処理(tcp の制御器)を要るか: 速度の上限 > 0 または tcp の出し方 ≥ 1"""
    return float(p.get('speed_cap', 0) or 0) > 0 or int(p.get('tcp_mode', 0) or 0) >= 1


def apply_gains(raw, ref, p, lo, hi):
    """採用した予測 raw=(steer, throttle) µs → 出力 µs。ゲインは中心/中立の周りに掛ける。
    速度のゲインは前進側だけ(ブレーキ・後退側はモデルのまま。ゲイン2.0でブレーキが2000µsに張り付いたため)"""
    steer = ref[0] + p['steer_trim_us'] + p['steer_gain'] * (raw[0] - ref[0])
    dt = raw[1] - ref[1]
    throttle = ref[1] + (p['throttle_gain'] * dt if dt < 0 else dt)
    return float(np.clip(steer, lo, hi)), float(np.clip(throttle, lo, hi))


class _Output(object):
    """PCA9685への出力。I2Cの一時的な失敗は読み直して続け、続けて失敗したら走行を止める"""
    MAX_FAILURES = 5          # 約150ms

    def __init__(self, pca, cfg, ref, params):
        self.pca, self.cfg, self.ref, self.params = pca, cfg, ref, params
        self.failures = 0
        self.errors = 0

    def neutral_values(self):
        return self.ref[0] + self.params()['steer_trim_us'], self.ref[1]

    def write(self, steer_us, throttle_us):
        """書けたら True。失敗したら中立を試み、続けて失敗したら RuntimeError"""
        if self.pca is None:
            return True
        try:
            self.pca.set_pair_us(self.cfg['steering_ch'], steer_us, self.cfg['throttle_ch'], throttle_us)
            self.failures = 0
            return True
        except OSError as e:
            self.errors += 1
            self.failures += 1
            self.neutral_safe(tries=2)
            if self.failures >= self.MAX_FAILURES:
                raise RuntimeError('PCA9685への書き込みが続けて失敗しました: %s' % e)
            return False

    def neutral(self):
        return self.write(*self.neutral_values())

    def neutral_safe(self, tries=6):
        """止めるとき用: スロットルを先に、チャネルごとに何度か試す。全部書けたら True"""
        if self.pca is None:
            return True
        steer, throttle = self.neutral_values()
        ok = True
        for ch, us in ((self.cfg['throttle_ch'], throttle), (self.cfg['steering_ch'], steer)):
            for _ in range(tries):
                try:
                    self.pca.set_us(ch, us)
                    break
                except OSError:
                    time.sleep(0.03)
            else:
                ok = False
        return ok


# 区間の速度上限の説明(画面・道具の表示用。値は走行ループが使う safety_caps.ZONE_CAPS_V1_1)
# v1.1 は 3 区間(R1・GM・DE)のほかに「位相が不確か(unknown)」でも上限を掛ける(4 つめ)。ユーザーの了承(2026-10-21)は
# 3 区間だけなので、画面の ON の確認でこれも示す(Fix-drive。値は変えない: G1 はこの形で判定した)
_ZC_CAP, _ZC_PHASE = safety_caps.ZONE_CAPS_V1_1.cap, safety_caps.ZONE_CAPS_V1_1.phase
ZONE_CAPS_SPEC = {'version': safety_caps.ZONE_CAPS_VERSION, 'zones': list(safety_caps.ZONES),
                  'caps_mps': {'R1': _ZC_CAP.cap_R1_app, 'GM': _ZC_CAP.cap_GM,
                               'DE': _ZC_CAP.cap_DE, 'unknown': _ZC_CAP.cap_unknown},
                  'approved_zones': ['R1', 'GM', 'DE'],      # ユーザーが了承した区間(2026-10-21)。unknown は含まない
                  'unknown_rule': {'cap_mps': _ZC_CAP.cap_unknown, 'lost_s': _ZC_PHASE.lost_s,
                                   'stop_reset_s': _ZC_PHASE.stop_reset_s},
                  'unknown_note': ('4 つめの上限: 位相が不確か(unknown)のときも %.1f m/s で切ります。走行の始めから最初の R1 を'
                                   '見つけるまで(1 周目のほぼ全部)・R1 から %g 秒を超えたとき・%g 秒以上止まったあと。'
                                   'ユーザーの了承は 3 区間(R1・GM・DE)だけです。ON の了承をもらうときに、これも含めて確かめてください'
                                   % (_ZC_CAP.cap_unknown, _ZC_PHASE.lost_s, _ZC_PHASE.stop_reset_s)
                                   if np.isfinite(_ZC_CAP.cap_unknown) else ''),
                  'brake_zones': list(_ZC_CAP.brake_zones), 'brake_us': _ZC_CAP.brake_us,
                  'modes': {'-2': '計算の失敗', '-1': '動かしていない', '0': 'そのまま', '1': '上限で切った',
                            '2': '中立にした', '3': 'ブレーキ'}}
ZONE_CODES = {z: i for i, z in enumerate(safety_caps.ZONES)}       # 記録の cap_zone(-1 = 計算していない・失敗)
CAP_MODE_CODES = dict({m: i for i, m in enumerate(safety_caps.CAP_MODES)}, off=-1)   # 記録の cap_mode(-2 = 失敗)


class _ZoneCaps(object):
    """区間の速度上限(safety_caps v1.1)の走行ループ側のつなぎ: センサーのバッファの値を数にして、純粋な包み
    (safety_caps.zone_imu・zone_frame)へ渡すだけ。状態は ZoneState 1 つ。計算が例外になったら failed に理由を残し、
    その走行のあいだは何もしない(呼ぶ側は出力を変えない = OFF と同じ)。時刻は最初のサンプルを原点にした秒"""

    def __init__(self, cfg=safety_caps.ZONE_CAPS_V1_1):
        self.cfg = cfg
        self.state = safety_caps.ZoneState()
        self.imu_done = None            # 位相へ入れた最後の IMU の時刻 [ns]
        self.origin = None
        self.failed = None              # 失敗の説明(None = 動いている)
        self.ms = collections.deque(maxlen=30)
        self.ms_max = 0.0

    def _sec(self, t_ns):
        if self.origin is None:
            self.origin = int(t_ns)
        return (int(t_ns) - self.origin) / 1e9

    @staticmethod
    def _speed_at(veh, t_ns):
        """t_ns までの最新の車両の速さ [m/s](無ければ最初の値。G1 の再生の prev_idx と同じ)"""
        if not veh:
            return 0.0
        e = veh[0][1]['erpm']
        for tv, d in reversed(veh):
            if tv <= t_ns:
                e = d['erpm']
                break
        return -float(e) * safety_caps.KV

    def _feed(self, imu_buf, veh_buf, upto_ns):
        """前回より後・upto_ns 以前(None なら全部)の IMU のサンプルを位相へ"""
        veh = list(veh_buf)
        st = self.state
        for t_ns, x in list(imu_buf):
            if self.imu_done is not None and t_ns <= self.imu_done:
                continue
            if upto_ns is not None and t_ns > upto_ns:
                break
            st = safety_caps.zone_imu(st, self._sec(t_ns), float(np.degrees(x[5])), self._speed_at(veh, t_ns), self.cfg)
            self.imu_done = t_ns
        self.state = st

    def _fail(self, e):
        if not self.failed:             # 最初の失敗を残す
            traceback.print_exc()
            self.failed = '%s: %s' % (type(e).__name__, e)

    def frame(self, cam_t_ns, out_us, veh, pred, active, imu_buf, veh_buf):
        """AI のフレーム(推論して出力を決めたフレーム)→ ZoneOut(失敗していれば None)。出力は変えない"""
        if self.failed:
            return None
        t0 = time.perf_counter()
        try:
            self._feed(imu_buf, veh_buf, cam_t_ns)
            veh_t = None
            for tv, d in reversed(list(veh_buf)):
                if d is veh:
                    veh_t = tv
                    break
            fresh = veh_t is not None and self._sec(cam_t_ns) - self._sec(veh_t) < self.cfg.fresh_s
            v = -float(veh['erpm']) * safety_caps.KV
            thr_in = float(out_us[1])
            self.state, z = safety_caps.zone_frame(self.state, self._sec(cam_t_ns), thr_in, float(out_us[0]), v,
                                                   np.asarray(pred)[:, 0].tolist(), bool(active) and fresh, self.cfg)
            if not (np.isfinite(z.thr_us) and z.thr_us >= thr_in):       # 前進を強める値は出さない(起きないはず)
                raise ValueError('上限の出力 %r が入力 %r より前進側です' % (z.thr_us, thr_in))
        except Exception as e:      # noqa: BLE001  上限が無くても走れる(OFF と同じ)
            self._fail(e)
            return None
        ms = (time.perf_counter() - t0) * 1000.0
        self.ms.append(ms)
        self.ms_max = max(self.ms_max, ms)
        return z

    def hold(self, cam_t_ns, imu_buf, veh_buf):
        """中立で止めたフレーム: 位相は進め、層の状態だけ戻す"""
        if self.failed:
            return
        try:
            self._feed(imu_buf, veh_buf, cam_t_ns)
            self.state = safety_caps.zone_inactive(self.state)
        except Exception as e:      # noqa: BLE001
            self._fail(e)

    def record(self, z, on, applied, thr_in):
        """AI の記録(AI_CMD)の cap_* の欄。z = frame() の戻り値(None = 中立で止めた / 失敗)。例外は出さない"""
        if z is not None:
            try:
                return {'cap_on': bool(on), 'cap_zone': ZONE_CODES.get(z.zone, -1),
                        'cap_zone_gyro': ZONE_CODES.get(z.zone_gyro, -1), 'cap_pred_sig': bool(z.pred_signal),
                        'cap_rel_psi': np.nan if z.rel_psi is None else float(z.rel_psi),
                        'cap_mode': CAP_MODE_CODES.get(z.mode, -1), 'cap_limit': float(z.cap_eff), 'cap_in_us': thr_in,
                        'cap_thr_us': float(z.thr_us), 'cap_brake_start': bool(z.brake_start), 'cap_applied': bool(applied)}
            except Exception as e:      # noqa: BLE001
                self._fail(e)
        return {'cap_on': bool(on), 'cap_zone': -1, 'cap_zone_gyro': -1, 'cap_pred_sig': False, 'cap_rel_psi': np.nan,
                'cap_mode': -2 if self.failed else -1, 'cap_limit': np.nan, 'cap_in_us': thr_in, 'cap_thr_us': np.nan,
                'cap_brake_start': False, 'cap_applied': False}

    def view(self, z, on, applied):
        """画面の表示用(st['now']['zone'])。例外は出さない"""
        if z is not None:
            try:
                return {'on': bool(on), 'error': None, 'zone': str(z.zone), 'zone_gyro': str(z.zone_gyro), 'mode': str(z.mode),
                        'limit': _num(z.cap_eff), 'thr_us': _num(z.thr_us, 1), 'applied': bool(applied),
                        'brake': z.mode == 'brake'}
            except Exception as e:      # noqa: BLE001
                self._fail(e)
        return {'on': bool(on), 'error': self.failed, 'zone': None, 'zone_gyro': None, 'mode': None, 'limit': None,
                'thr_us': None, 'applied': False, 'brake': False}


# ------------------------------------------------------------------ 速さの天井・速さのゲインの自動合わせ(2026-10-25)
# 仕様: jetformer_reports/speed_control_20261025/spec.md 2・3 章(記録 build_drive.md)。どちらも既定 OFF。
# OFF でも AI のフレームごとに計算して記録する(AI_CMD の ceil_*・sa_*、影の判定)。車への出力を変えるのは ON・AI のフレームだけ。
# 計算が例外になったら、その走行のあいだは外す(OFF と同じ出力)。天井が失敗したら自動合わせも外す
SPEED_CTRL_VERSION = '2026-10-25'
G0 = 9.80665
VEH_FRESH_NS = 100_000_000          # 車両の値が撮像からこれより古いフレームは天井を動かさない(0.1 秒)
ACCEL_LAG_NS = 100_000_000          # 加速 a = (v − 0.1 秒前の v) ÷ 0.1


def speed_ctrl_spec(params=None):
    """画面・道具の表示用(st['speed_ctrl_spec'])。値は走行ループが使うものから作る(文と値をずらさない)"""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    cp = safety_caps.ceil_params(p['ceil_all_mps'], p['ceil_left_mps'])
    ap = speed_auto.auto_params(p['auto_target_mps'], p['auto_vref_v'])
    return {
        'version': SPEED_CTRL_VERSION,
        'ceiling': {
            'all_mps': cp.cap_all, 'left_mps': cp.cap_left_eff(),
            'left_rule': {'steer_us': cp.left_steer_us, 'gz_dps': cp.left_gz_dps, 'hold_s': cp.left_hold_s},
            'release_mps2': cp.release_accel, 'neutral_over_mps': cp.neutral_margin, 'pred_s': cp.pred_s,
            'kp': cp.kp, 'ki': cp.ki, 'brake': False, 'changes_steer': False, 'per_run': True,
            'start_from_hold': cp.start_from_hold, 'release_bumpless': cp.release_bumpless,
            'title': '速さの天井(全体 %.1f・左旋回中 %.1f m/s)' % (cp.cap_all, cp.cap_left_eff()),
            'rule_text': ('左旋回 = 車に出す舵 +%.0f µs 以上 か ヨー +%.0f deg/s 以上(どちらも切れてから %.2f 秒は左旋回のまま。'
                          '終わったら %.1f m/s² で全体の天井へ戻す)'
                          % (cp.left_steer_us, cp.left_gz_dps, cp.left_hold_s, cp.release_accel)),
            # 「この走行のあいだだけ」は誤りだった: ON は OFF にするまで次の走行にも残る(区間の速度上限と同じ)。fix.md
            'note': ('前進を弱めるだけ。ブレーキなし・舵は変えない。天井 + %.1f m/s を超えたら中立(惰性)。'
                     '速すぎる当たり(奥の左スイーパーの出口・LH の手前)を減らすもので、当たりを 0 にはしない(DE の当たりには効かない)。'
                     'OFF にするまで続く(管理画面の再起動で OFF)' % cp.neutral_margin)},
        'auto': {
            'target_mps': ap.target_mps, 'vref_v': ap.vref_v, 'm_range': [ap.m_lo, ap.m_hi],
            'raise_max_gain': ap.gain_raise_max, 'ff_range': [ap.ff_lo, ap.ff_hi], 'fb_range': [ap.fb_lo, ap.fb_hi],
            'step_max': ap.step_max, 'deadband_mps': ap.deadband, 'requires_ceiling': True, 'per_run': True,
            # 周の直しの物差し(fix.md): 周の時間。目標 T* = 14.52 − 1.50 × target_mps(4.27 → 8.12 秒。2026-10-25 ユーザー: 25.0 秒ちょうど = 8.12 秒)・不感帯 ±0.10 秒
            'metric': ap.metric, 'target_lap_s': round(ap.target_lap_s(), 3), 'deadband_s': ap.deadband_s,
            'ceil_frac_max': ap.ceil_frac_max_lap if ap.metric == speed_auto.METRIC_LAP else ap.ceil_frac_max,
            'v_range': [ap.vboard_lo, ap.vboard_hi],
            'sources': {'0': 'なし(1.00)', '1': '基板の電圧', '2': '説明欄の V_before'},
            'title': ('速さのゲインの自動合わせ(目標 1 周 %.2f 秒 = 周の速さ p90 %.2f m/s 相当)'
                      % (ap.target_lap_s(), ap.target_mps)),
            'note': ('速さのゲインに %.2f〜%.2f 倍を掛けます(上げるのは %.2f まで)。電圧の前送り(%.2f V が 1.00 倍。'
                     '%.1f〜%.1f V の外の値は使わない)と、R1 の起点ごとの周の時間の直し(1 周 %.0f%% まで・%.2f±%.2f 秒は直さない)。'
                     '天井が ON のときだけ。OFF にするまで続く(管理画面の再起動で OFF)。走り終わりに次の走行の速度ゲインの目安を出す'
                     % (ap.m_lo, ap.m_hi, ap.gain_raise_max, ap.vref_v, ap.vboard_lo, ap.vboard_hi, ap.step_max * 100,
                        ap.target_lap_s(), ap.deadband_s))},
        'ceil_modes': {str(k): v for k, v in safety_caps.CEIL_MODE_NAMES.items()},
        'auto_events': {str(k): v for k, v in speed_auto.EVENT_NAMES.items()},
        'auto_skip_reasons': dict(speed_auto.SKIP_NAMES), 'auto_warn_text': dict(speed_auto.WARN_TEXT),
    }


def _median3_speed(veh, t_ns):
    """t_ns 以前の最新 3 つの車両のサンプルの速さ [m/s] の中央値と、最新のサンプルの時刻。3 つ無ければ古い方で埋める
    (下見の ceiling_shadow.py と同じ)。無ければ (nan, None)"""
    xs, t_last = [], None
    for tv, d in reversed(veh):
        if tv <= t_ns:
            if t_last is None:
                t_last = tv
            xs.append(-float(d['erpm']) * safety_caps.KV)
            if len(xs) == 3:
                break
    if not xs:
        return np.nan, None
    while len(xs) < 3:
        xs.append(xs[-1])
    xs.sort()
    return xs[1], t_last


def _gz5(imu, t_ns):
    """t_ns 以前の最新 5 つの IMU のヨー [deg/s] の平均(5 つ無ければ古い方で埋める。無ければ nan)"""
    xs = []
    for ti, x in reversed(imu):
        if ti <= t_ns:
            xs.append(float(np.degrees(x[5])))
            if len(xs) == 5:
                break
    if not xs:
        return np.nan
    while len(xs) < 5:
        xs.append(xs[-1])
    xs.reverse()                    # 古い順に足す(試験・再生と同じ順)
    return sum(xs) / 5.0


def speed_inputs(veh, imu, t_ns):
    """天井の入力 (v, a, gz5, 最新の車両のサンプルの時刻) を、撮像時刻 t_ns までのバッファから(spec 1.1 節)"""
    v, t_last = _median3_speed(veh, t_ns)
    v_old, t_old = _median3_speed(veh, t_ns - ACCEL_LAG_NS)
    a = (v - v_old) / 0.1 if t_old is not None else np.nan
    return v, a, _gz5(imu, t_ns), t_last


def launch_inputs(p, t_ns, tof_buf, imu_buf, veh_buf, ref, base_us, ttc, ttc_hist, guard_on, impact_g, guard_changed):
    """発進の前進の 1 フレームの入力 assist.LaunchIn(launch_boost = 1 のときだけ呼ぶ)。速さ・加速・ヨーは速さの天井と同じ
    (speed_inputs)。ToF・車両の値は撮像時刻 t_ns から 0.1 秒以内なら使える"""
    v, a, gz5, veh_t = speed_inputs(list(veh_buf), list(imu_buf), t_ns)
    tof_t = max((x[0] for x in list(tof_buf) if x[0] <= t_ns), default=None)
    tof_ok = bool(guard_on) and tof_t is not None and t_ns - tof_t < LAUNCH_FRESH_NS and bool(np.isfinite(ttc))
    return assist_mod.LaunchIn(
        us=float(p.get('launch_us', DEFAULT_PARAMS['launch_us'])), mps=float(p.get('launch_mps', DEFAULT_PARAMS['launch_mps'])),
        max_s=float(p.get('launch_max_s', DEFAULT_PARAMS['launch_max_s'])), v=float(v), a=float(a), gz=float(gz5),
        steer_rel=float(base_us[0]) - float(ref[0]) - float(p.get('steer_trim_us', 0.0)), ttc=float(ttc),
        ttc5=tuple(float(x) for x in ttc_hist), tof_ok=tof_ok,
        veh_ok=veh_t is not None and t_ns - veh_t < LAUNCH_FRESH_NS, impact_g=float(impact_g),
        guard_changed=bool(guard_changed))


def launch_spec(params=None):
    """画面・道具の表示用(st['launch_spec'])。値は走行ループ・assist.LaunchBoost が使うものから作る"""
    pp = dict(DEFAULT_PARAMS, **(params or {}))
    B = assist_mod.LaunchBoost
    return {
        'version': LAUNCH_VERSION, 'us': float(pp['launch_us']), 'mps': float(pp['launch_mps']),
        'max_s': float(pp['launch_max_s']),
        'limits': {k: list(PARAM_LIMITS[k]) for k in LAUNCH_KEYS},
        'start_erpm': B.START_ERPM, 'ttc_start_s': B.TTC_START_S, 'ttc_n': B.TTC_N,
        'end_yaw_dps': B.END_YAW_DPS, 'end_steer_us': B.END_STEER_US, 'end_steer_mps': B.END_STEER_MPS,
        'end_impact_g': B.END_IMPACT_G, 'end_ttc_s': B.END_TTC_S, 'end_ttc_n': B.END_TTC_N,
        'end_names': {str(k): v for k, v in assist_mod.LAUNCH_END_NAMES.items()},
        'title': '発進の前進(前進 %.0f µs 以上・%.1f m/s まで・最大 %.1f 秒)' % (pp['launch_us'], pp['launch_mps'],
                                                                   pp['launch_max_s']),
        'note': ('CH3 を AI 側にした最初のフレームで、止まっていて前が %.1f m 以上空いている(TTC の中央値 %.1f 秒以上)ときだけ、'
                 'スロットルを 中立 − %.0f µs より前に出します(モデルが強ければモデル。前進だけ・舵はモデルのまま)。'
                 '%.1f m/s・%.1f 秒・ヨー %.0f deg/s・舵 %.0f µs・衝撃 %.1fG・前が近い・CH3 を人側・中立で止めた のどれかで終わり、'
                 '同じ AI の区間ではもう出しません(発進の補助も出しません)。既定は OFF で、管理画面を起動するたびに OFF。'
                 '走行中は ON にできません(OFF にはできます)'
                 % (B.TTC_START_S * 0.3, B.TTC_START_S, pp['launch_us'], pp['launch_mps'], pp['launch_max_s'],
                    B.END_YAW_DPS, B.END_STEER_US, B.END_IMPACT_G)),
    }


class _SpeedCeil(object):
    """速さの天井(safety_caps.ceil_step)の走行ループ側のつなぎ。バッファの値を数にして渡すだけ。状態は CeilState 1 つ。
    例外は failed に残し、その走行のあいだは何もしない(呼ぶ側は出力を変えない = OFF と同じ)。時刻は最初のフレームが原点の秒"""

    def __init__(self, params):
        self.cfg = safety_caps.ceil_params(params.get('ceil_all_mps', DEFAULT_PARAMS['ceil_all_mps']),
                                           params.get('ceil_left_mps', DEFAULT_PARAMS['ceil_left_mps']))
        self.state = safety_caps.CeilState()
        self.origin = None
        self.failed = None
        self.ms = collections.deque(maxlen=30)
        self.ms_max = 0.0
        self.inputs = None              # 最後のフレームの (v, a, steer_rel, gz5)

    def _sec(self, t_ns):
        if self.origin is None:
            self.origin = int(t_ns)
        return (int(t_ns) - self.origin) / 1e9

    def _fail(self, e):
        if not self.failed:
            traceback.print_exc()
            self.failed = '%s: %s' % (type(e).__name__, e)

    def frame(self, cam_t_ns, out_us, steer_center_us, neutral_us, active, imu_buf, veh_buf):
        """推論して出力を決めたフレーム → CeilOut(失敗していれば None)。出力は変えない"""
        if self.failed:
            return None
        t0 = time.perf_counter()
        try:
            v, a, gz5, veh_t = speed_inputs(list(veh_buf), list(imu_buf), cam_t_ns)
            steer_rel = float(out_us[0]) - float(steer_center_us)
            fresh = veh_t is not None and cam_t_ns - veh_t < VEH_FRESH_NS
            thr_in = float(out_us[1])
            self.state, c = safety_caps.ceil_step(self.state, self._sec(cam_t_ns), v, a, steer_rel, gz5, thr_in,
                                                  float(neutral_us), bool(active) and fresh, self.cfg)
            if not (c.thr_us >= thr_in and (c.thr_us <= neutral_us or c.thr_us == thr_in)):
                raise ValueError('天井の出力 %r が入力 %r より前進側か、中立 %r より後ろです' % (c.thr_us, thr_in, neutral_us))
            self.inputs = (v, a, steer_rel, gz5)
        except Exception as e:      # noqa: BLE001  天井が無くても走れる(OFF と同じ)
            self._fail(e)
            return None
        ms = (time.perf_counter() - t0) * 1000.0
        self.ms.append(ms)
        self.ms_max = max(self.ms_max, ms)
        return c

    def hold(self):
        """中立で止めたフレーム: 状態を戻す"""
        self.state = safety_caps.CeilState()
        self.inputs = None

    def record(self, c, on, applied, thr_in):
        """AI_CMD の ceil_* の欄(spec 2.6 節)。c = frame() の戻り値(None = 中立で止めた / 失敗)。例外は出さない"""
        nan = np.nan
        if c is not None and self.inputs is not None:
            try:
                v, a, steer_rel, gz5 = self.inputs
                return {'ceil_on': bool(on), 'ceil_left': bool(c.left), 'ceil_cap': float(c.cap) if c.active else nan,
                        'ceil_v': float(v), 'ceil_vpred': float(c.v_pred) if c.active else nan,
                        'ceil_steer': float(steer_rel), 'ceil_gz': float(gz5), 'ceil_mode': int(c.mode),
                        'ceil_in_us': float(thr_in), 'ceil_thr_us': float(c.thr_us) if c.active else nan,
                        'ceil_applied': bool(applied)}
            except Exception as e:      # noqa: BLE001
                self._fail(e)
        return {'ceil_on': bool(on), 'ceil_left': False, 'ceil_cap': nan, 'ceil_v': nan, 'ceil_vpred': nan,
                'ceil_steer': nan, 'ceil_gz': nan, 'ceil_mode': -2 if self.failed else -1, 'ceil_in_us': float(thr_in),
                'ceil_thr_us': nan, 'ceil_applied': False}

    def view(self, c, on, applied):
        """画面の表示用(st['now']['ceil'])。例外は出さない"""
        if c is not None and self.inputs is not None:
            try:
                return {'on': bool(on), 'error': None, 'active': bool(c.active), 'left': bool(c.left),
                        'cap': _num(c.cap) if c.active else None, 'v': _num(self.inputs[0]),
                        'mode': safety_caps.CEIL_MODE_NAMES.get(int(c.mode), ''), 'applied': bool(applied),
                        'thr_us': _num(c.thr_us, 1) if c.active else None}
            except Exception as e:      # noqa: BLE001
                self._fail(e)
        return {'on': bool(on), 'error': self.failed, 'active': False, 'left': False, 'cap': None, 'v': None,
                'mode': None, 'applied': False, 'thr_us': None}


class _SpeedAuto(object):
    """速さのゲインの自動合わせ(jetformer.speed_auto)の走行ループ側のつなぎ。
    gain(): 推論のあと・apply_gains の前に、電圧の前送り m_ff と周ごとの直し m_fb から m を作る(OFF でも影で毎フレーム)。
    after()/hold(): フレームの値を周のために残し、R1 の起点(区間の判定のジャイロの位相。失敗したら LapCounter の周の区切り)
    ごとに周を閉じて m_fb を直す。時刻は最初のフレームが原点の秒"""
    EFF_N_MIN = 30                  # 直線の効き(記録だけ)を出し始めるフレームの数
    EFF_TAU_S = 5.0

    def __init__(self, params, description=''):
        self.cfg = speed_auto.auto_params(params.get('auto_target_mps', DEFAULT_PARAMS['auto_target_mps']),
                                          params.get('auto_vref_v', DEFAULT_PARAMS['auto_vref_v']))
        self.v_desc = speed_auto.v_before_of(description or '', self.cfg)
        self.origin = None
        self.failed = None
        self.ms = collections.deque(maxlen=30)
        self.ms_max = 0.0
        # 電圧
        self.vb = collections.deque()           # 直近 3 秒の基板の電圧 (t, V)
        self.vb_done = None                     # 取り込んだ最後の車両のサンプルの時刻 [ns]
        self.vb_last_t = None                   # 最後に来た基板の電圧の時刻 [秒]
        self.v_level = np.nan                   # 基板の電圧の上の包絡(最後に出せた値)
        self.src, self.v_used = speed_auto.SRC_NONE, np.nan
        self.m_ff = self.m = self.t_prev = None
        self.moved = False
        self.fb = speed_auto.FbState()
        # 周
        self.frames = collections.deque(maxlen=1500)     # LapFrame(50 秒分)
        self.anchor = None                      # 最後に使った起点 (t, ψ)
        self.anchor_src = None                  # 'phase' / 'laps'
        self.seen_anchor = None                 # 位相の起点(zcaps の秒)で最後に見たもの
        self.marks_done = 0                     # LapCounter の周の区切りのうち見たもの
        self.imu_done = None
        self.imu_hg = self.imu_a3 = 0.0
        self.event, self.lap_p90, self.lap_s = speed_auto.EV_NONE, np.nan, np.nan
        # 直線の効き(記録だけ、spec 3.3 節)
        self.eff, self.eff_n = np.nan, 0
        self.cmd_hist = collections.deque()     # (t, 前進の指令 µs)
        self.erpm_hist = collections.deque()    # (t, eRPM)
        self.last_gain = None
        self._frame_ms = 0.0

    def _sec(self, t_ns):
        if self.origin is None:
            self.origin = int(t_ns)
        return (int(t_ns) - self.origin) / 1e9

    def _fail(self, e):
        if not self.failed:
            traceback.print_exc()
            self.failed = '%s: %s' % (type(e).__name__, e)

    def _timed(self, t0, close):
        """gain() と after() の時間を足して、1 フレームの時間として残す"""
        self._frame_ms += (time.perf_counter() - t0) * 1000.0
        if close:
            self.ms.append(self._frame_ms)
            self.ms_max = max(self.ms_max, self._frame_ms)
            self._frame_ms = 0.0

    # ---- 電圧
    def _voltage(self, t, veh):
        cfg = self.cfg
        for tv, d in veh:
            if self.vb_done is not None and tv <= self.vb_done:
                continue
            self.vb_done = tv
            vb = d.get('vbat_v', np.nan)
            # 6.0〜9.0 V の外(分圧の外れ・張り付き)は来なかったのと同じ(解読も外すが、古い解読・偽物でも守る。fix.md)
            if d.get('vbat_ok', False) and vb is not None and speed_auto.board_v_ok(float(vb), cfg):
                ts = self._sec(tv)
                self.vb.append((ts, float(vb)))
                self.vb_last_t = ts
        while self.vb and self.vb[0][0] < t - cfg.vbat_window_s:
            self.vb.popleft()
        # 走り出す前は約 0.2 秒(20 サンプル)で出してよい(止まっていれば加速の下がりは無い)。走り出したら 100 サンプル
        level = speed_auto.vbat_level([x[1] for x in self.vb], cfg,
                                      None if self.moved else cfg.vbat_min_n_still)
        if np.isfinite(level):
            self.v_level = level
        if np.isfinite(self.v_level):
            stale = self.vb_last_t is None or t - self.vb_last_t > cfg.vbat_stale_s
            self.src, self.v_used = speed_auto.SRC_BOARD, self.v_level
            return None if stale else speed_auto.ff_of_voltage(self.v_level, cfg)      # 途切れたら最後の m_ff を保つ
        if np.isfinite(self.v_desc):
            self.src, self.v_used = speed_auto.SRC_DESC, self.v_desc
        else:
            self.src, self.v_used = speed_auto.SRC_NONE, np.nan
        return speed_auto.ff_of_voltage(self.v_used, cfg)

    def gain(self, cam_t_ns, throttle_gain, v_now, veh_buf):
        """このフレームに掛けるはずの速さのゲイン g(失敗していれば None)"""
        if self.failed:
            return None
        t0 = time.perf_counter()
        try:
            cfg = self.cfg
            t = self._sec(cam_t_ns)
            dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
            self.t_prev = t
            if v_now >= cfg.moving_v:
                self.moved = True
            was_board = self.src == speed_auto.SRC_BOARD
            ff = self._voltage(t, list(veh_buf))
            if ff is None:
                ff = self.m_ff if self.m_ff is not None else 1.0
            # そのまま入れる: 走り出す前(止まっている間の電圧がいちばん正しい。基板の電圧は 20 サンプルで出す)。
            # 走り出したあとに基板の電圧の最初の値が来たら、止まっているときか、今の m_ff(説明欄の V_before・なし)との差が
            # 0.02 以下のときだけそのまま入れる。ほかは 1 秒 0.01・0.03 まで(走っている最中に 1 フレームで跳ばない。fix.md)
            first_board = self.src == speed_auto.SRC_BOARD and not was_board
            direct = not self.moved or (first_board and (
                v_now < cfg.moving_v or self.m_ff is None or abs(ff - self.m_ff) <= cfg.ff_jump_max + 1e-12))
            self.m_ff = ff if direct else speed_auto.rate_limit(self.m_ff, ff, cfg.ff_rate * dt)
            m_raw = speed_auto.m_of(self.m_ff, self.fb.m_fb, cfg)
            self.m = m_raw if direct else speed_auto.rate_limit(self.m, m_raw, cfg.m_rate * dt)
            g = speed_auto.gain_of(throttle_gain, self.m, cfg)
            if not np.isfinite(g) or g < 0:
                raise ValueError('自動合わせのゲイン %r が正しくありません' % g)
            self.last_gain = g
        except Exception as e:      # noqa: BLE001  自動合わせが無くても走れる(throttle_gain のまま)
            self._fail(e)
            return None
        self._frame_ms = 0.0
        self._timed(t0, False)
        return g

    # ---- 周
    def _imu(self, cam_t_ns, imu_buf):
        hg = a3 = 0.0
        for ti, x in list(imu_buf):
            if self.imu_done is not None and ti <= self.imu_done:
                continue
            if ti > cam_t_ns:
                break
            self.imu_done = ti
            ax, ay, az = float(x[0]), float(x[1]), float(x[2])
            hg = max(hg, math.hypot(ax, ay) / G0)
            a3 = max(a3, abs(math.sqrt(ax * ax + ay * ay + az * az) / G0 - 1.0))
        return hg, a3

    def _new_anchors(self, zcaps, lap_marks):
        """新しく確かめた R1 の起点 [(t, ψ or None)]。位相が不確かに戻ったら、前の起点を忘れる"""
        if zcaps is not None and not zcaps.failed:
            if zcaps.origin is None:            # まだ IMU が来ていない
                return []
            self.anchor_src = 'phase'
            ph = zcaps.state.phase
            if ph.anchor_t is None:
                if self.seen_anchor is not None:
                    self._forget()              # 止まった・引き返した(v1.1 (B)): 次の起点は「前の起点なし」
                self.seen_anchor = None
                return []
            if ph.anchor_t == self.seen_anchor:
                return []
            self.seen_anchor = ph.anchor_t
            t_ns = zcaps.origin + ph.anchor_t * 1e9
            return [((t_ns - self.origin) / 1e9, float(ph.anchor_psi))]
        if self.anchor_src != 'laps':           # 位相が使えない: LapCounter の周の区切り(向きの決まりは使わない)
            self._forget()
            self.anchor_src, self.marks_done = 'laps', len(lap_marks)
            return []
        new = lap_marks[self.marks_done:]
        self.marks_done = len(lap_marks)
        return [((float(x) * 1e9 - self.origin) / 1e9, None) for x in new]

    def _forget(self):
        """前の起点を忘れる(その周は閉じられない)。前の起点からのフレームに衝突(水平 2.5G 以上)があれば、使わない周
        (衝突)と数えて次の 1 周は上げない(衝突 → 人が持って止める、のいちばん多い形で「衝突の次」を守る。fix.md)"""
        if self.anchor is not None:
            t_p, lim = self.anchor[0], self.cfg.impact_g
            hit = any(f.impact_g >= lim for f in self.frames if f.t >= t_p)
            self.fb, ev = speed_auto.fb_abandon(self.fb, hit)
            if ev != speed_auto.EV_NONE:
                self.event = ev
        self.anchor = None

    def _close(self, t_a, psi):
        cfg = self.cfg
        if self.anchor is None:
            self.anchor = (t_a, psi)
            while self.frames and self.frames[0].t < t_a:     # 周に入らないフレーム(周を閉じるフレームで見ない)
                self.frames.popleft()
            return
        t_p, psi_p = self.anchor
        if (psi is not None and psi_p is not None and t_a - t_p <= cfg.lap_max_s
                and psi - psi_p < cfg.false_anchor_deg):
            self.event = speed_auto.EV_FALSE_ANCHOR         # 衝突で右へ回ったもの。前に使った起点はそのまま
            return
        lap = speed_auto.lap_stats(self.frames, t_p, t_a, cfg)
        self.fb, self.event, _ = speed_auto.fb_lap(self.fb, lap, self.m_ff if self.m_ff is not None else 1.0, cfg)
        self.lap_p90, self.lap_s = lap.v90, lap.dur
        self.anchor = (t_a, psi)
        while self.frames and self.frames[0].t < t_a:
            self.frames.popleft()

    def _lap_frame(self, t, cam_t_ns, v, ai, assist, guard, ceil_lim, imu_buf, zcaps, lap_marks, hold=False,
                   gain=np.nan):
        hg, a3 = self._imu(cam_t_ns, imu_buf)
        self.frames.append(speed_auto.LapFrame(t, v, ai, int(assist), bool(guard), hg, a3, bool(ceil_lim), bool(hold),
                                               float(gain)))
        self.event, self.lap_p90, self.lap_s = speed_auto.EV_NONE, np.nan, np.nan
        for t_a, psi in self._new_anchors(zcaps, lap_marks):
            self._close(t_a, psi)

    def _eff(self, t, f_us, erpm, steer_rel, gz5):
        """直線の効き(記録だけ): eRPM ÷ (26.314 × 直前 0.5 秒の指令の平均 + 1343.5)の 5 秒の指数平均"""
        self.cmd_hist.append((t, f_us))
        self.erpm_hist.append((t, erpm))
        while self.cmd_hist and self.cmd_hist[0][0] < t - 0.5:
            self.cmd_hist.popleft()
        while len(self.erpm_hist) > 1 and self.erpm_hist[1][0] <= t - 0.3:
            self.erpm_hist.popleft()
        cmds = [x[1] for x in self.cmd_hist]
        mean = sum(cmds) / len(cmds)
        sd = math.sqrt(max(sum((x - mean) ** 2 for x in cmds) / len(cmds), 0.0))
        fwd = -erpm
        if (abs(steer_rel) <= 60 and abs(gz5) <= 20 and mean > 60 and sd < 30 and fwd > 1500
                and self.erpm_hist[0][0] <= t - 0.3 + 1e-9 and abs(erpm - self.erpm_hist[0][1]) < 500):
            x = fwd / (safety_caps.CeilParams().hold_slope * mean + safety_caps.CeilParams().hold_offset)
            self.eff = x if not np.isfinite(self.eff) else self.eff + (x - self.eff) * min(1.0, (1.0 / 30) / self.EFF_TAU_S)
            self.eff_n += 1

    def after(self, cam_t_ns, v, ai, assist, guard, ceil_lim, f_out_us, erpm, steer_rel, gz5, imu_buf, zcaps, lap_marks,
              gain=np.nan):
        """出力を決めたフレームのあと(周の数・直し)。gain = このフレームで実際に掛けた速さのゲイン(走り終わりの目安)。
        例外は failed に残す"""
        if self.failed:
            return
        t0 = time.perf_counter()
        try:
            t = self._sec(cam_t_ns)
            self._lap_frame(t, cam_t_ns, v, ai, assist, guard, ceil_lim, imu_buf, zcaps, lap_marks,
                            gain=gain if ai else np.nan)
            if ai and np.isfinite(steer_rel) and np.isfinite(gz5):
                self._eff(t, f_out_us, float(erpm), steer_rel, gz5)
        except Exception as e:      # noqa: BLE001
            self._fail(e)
            return
        self._timed(t0, True)

    def hold(self, cam_t_ns, imu_buf, zcaps, lap_marks):
        """中立で止めたフレーム(周の中に 4 つ = 約 0.13 秒までなら周を使う。それより多ければ使わない。fix.md)"""
        if self.failed:
            return
        try:
            self._lap_frame(self._sec(cam_t_ns), cam_t_ns, np.nan, False, 0, False, False, imu_buf, zcaps, lap_marks,
                            hold=True)
        except Exception as e:      # noqa: BLE001
            self._fail(e)

    def record(self, on, g_applied, throttle_gain):
        """AI_CMD の sa_* の欄(spec 3.9 節)。例外は出さない"""
        nan = np.nan
        if not self.failed and self.m is not None:
            try:
                return {'sa_on': bool(on), 'sa_m': float(self.m), 'sa_m_ff': float(self.m_ff), 'sa_m_fb': float(self.fb.m_fb),
                        'sa_gain': float(self.last_gain), 'sa_src': int(self.src), 'sa_v': float(self.v_used),
                        'sa_eff': float(self.eff) if self.eff_n >= self.EFF_N_MIN else nan, 'sa_event': int(self.event),
                        'sa_lap_p90': float(self.lap_p90), 'sa_lap_s': float(self.lap_s),
                        'sa_applied': bool(g_applied != throttle_gain)}
            except Exception as e:      # noqa: BLE001
                self._fail(e)
        return {'sa_on': bool(on), 'sa_m': nan, 'sa_m_ff': nan, 'sa_m_fb': nan, 'sa_gain': nan, 'sa_src': -1, 'sa_v': nan,
                'sa_eff': nan, 'sa_event': -1, 'sa_lap_p90': nan, 'sa_lap_s': nan, 'sa_applied': False}

    def record_hold(self, on):
        """中立で止めた行: sa_on・sa_m と周の値(m_fb・出来事・閉じた周の p90 と時間。周はこの行で閉じることもある)。
        ほかは「計算していない」"""
        r = self.record(on, 0.0, 0.0)
        r.update(sa_gain=np.nan, sa_src=-1, sa_v=np.nan, sa_eff=np.nan, sa_m_ff=np.nan, sa_applied=False)
        return r

    def view(self, on, g_applied):
        """画面の表示用(st['now']['auto'])。例外は出さない"""
        try:
            fb = self.fb
            return {'on': bool(on), 'error': self.failed, 'm': _num(self.m, 3), 'm_ff': _num(self.m_ff, 3),
                    'm_fb': _num(fb.m_fb, 3), 'gain': _num(g_applied, 3), 'v': _num(self.v_used),
                    'src': int(self.src), 'lap_p90': _num(fb.last_v90), 'lap_s': _num(fb.last_lap_s),
                    'laps_used': int(fb.laps_used), 'warn': fb.warn, 'warn_text': speed_auto.WARN_TEXT.get(fb.warn, '')}
        except Exception as e:      # noqa: BLE001
            self._fail(e)
            return {'on': bool(on), 'error': self.failed}

    def end(self, on, throttle_gain):
        """走り終わりの目安(st['auto_end'])"""
        try:
            out = speed_auto.end_summary(self.fb, on, throttle_gain, self.cfg)
        except Exception as e:      # noqa: BLE001
            self._fail(e)
            out = {'on': bool(on)}
        out['error'] = self.failed
        return out


class Driver(object):
    def __init__(self, hub, recorder):
        self.hub = hub
        self.recorder = recorder
        self.lock = threading.Lock()
        self.thread = None
        self._stop = threading.Event()
        saved = _read_json(SETTINGS_PATH) or {}
        self.params = dict(DEFAULT_PARAMS)
        self.params.update({k: v for k, v in saved.get('params', {}).items() if k in DEFAULT_PARAMS})
        # tcp の出し方・目標速度の倍率は、次に起動したときに持ち越さない(試験的な出し方 2・3 が別の日・別のモデルで
        # そのまま使われないように。速度の上限は前進を弱めるだけなので残す)。レビュー 2026-10-12
        for k in RESET_ON_BOOT:
            self.params[k] = DEFAULT_PARAMS[k]
        self.last = {'run': saved.get('run'), 'mode_id': saved.get('mode_id')}
        self.heartbeat_t = time.monotonic()
        self.hb_token = None
        self.hb_count = 0
        self.history = collections.deque(maxlen=300)
        self.st = {'phase': 'idle', 'message': '停止中', 'error': None}
        self.listeners = []        # 出来事の通知先 cb(event, **info)(音声案内など)。走行ループから呼ぶので待たせないこと
        self.last_takeover_ns = None
        self.takeover_reasons = []
        self.params_changes = []        # この走行の中で set_params が変えた設定(_note_params)
        # 起動の時刻(status の boot_at)。画面が「サーバーが起動し直した = RESET_ON_BOOT の項目が既定に戻った」を知らせるのに使う
        # (2026-10-26 fix3 直す 1 回目 r1-state3-4。表示だけ。走行・params には使わない)
        self.boot_at = round(time.time(), 3)

    def _emit(self, event, **info):
        for cb in list(self.listeners):
            try:
                cb(event, **info)
            except Exception:
                traceback.print_exc()

    # ------------------------------------------------------------ 操作
    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def heartbeat(self, token):
        """走行を始めたページだけが生存確認になる(ほかのページが開いていても、操作しているページが
        閉じた・通信が切れたら止まるように)"""
        if token and token == self.hb_token:
            self.heartbeat_t = time.monotonic()
            self.hb_count += 1
            return True
        return False

    def _clean_params(self, values):
        p = dict(self.params)
        for k, (lo, hi) in PARAM_LIMITS.items():
            if values.get(k) is not None:
                v = float(values[k])
                if not np.isfinite(v):
                    raise ValueError('%s が数値ではありません' % k)
                p[k] = min(max(v, lo), hi)
        p['lookahead'] = int(round(p['lookahead']))
        for k in ('esc_reverse', 'recovery', 'zone_caps', 'start_kick', 'speed_ceiling', 'speed_auto',
                  'launch_boost', 'rec_back_guard', 'rec_hold_keep', 'launch_no_kick'):     # 古い設定ファイル・古い画面には無い項目
            p[k] = 1 if float(p.get(k, DEFAULT_PARAMS[k])) >= 0.5 else 0
        p['tof_guard'] = int(round(float(p.get('tof_guard', DEFAULT_PARAMS['tof_guard']))))
        for k in ('tcp_mode', 'ensemble_j'):           # 整数の項目(古い設定ファイルには無い)
            p[k] = int(round(float(p.get(k, DEFAULT_PARAMS[k]))))
        for k in SPEED_CTRL_VALUES + LAUNCH_VALUES:    # 古い設定ファイルには無い
            p[k] = float(p.get(k, DEFAULT_PARAMS[k]))
        return p

    def _launch_rules(self, p, running):
        """発進の前進の決まり(protocol_v3 14 章)。走行中は 0 → 1 にできない(1 → 0 はできる)・値は変えられない
        (1 走行 = 1 設定)。断るときは ValueError(サーバーは 400)"""
        if running:
            prev = self.params
            if p['launch_boost'] == 1 and int(prev.get('launch_boost', 0) or 0) == 0:
                raise ValueError('走行中は発進の前進を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)')
            for k in LAUNCH_VALUES:
                if p[k] != float(prev.get(k, DEFAULT_PARAMS[k])):
                    raise ValueError('走行中は %s を変えられません。止めてから変えてください(1 走行 = 1 設定)' % k)
        return p

    OPT_NAMES = {'rec_back_guard': 'リカバリーの後ろの見張り', 'rec_hold_keep': '中立止めのあともリカバリーを残す',
                 'launch_no_kick': '発進の前進を出さない区間は発進の補助も出さない'}

    def _opt_rules(self, p, running):
        """試験の選択肢(後ろの見張り・fix3 の 2 つ)の決まり(2026-10-26 fix3)。断るときは ValueError(サーバーは 400)。
        ・走行中は 0 → 1 にできない(1 → 0 はできる。1 走行 = 1 設定。画面 app.js の OPT_SWITCHES と同じ決まりをサーバーでも守る)
        ・rec_hold_keep は rec_back_guard が ON のときだけ(後ろの見張りを OFF にしたら rec_hold_keep も OFF)。中立止めで残した
          リカバリーは、見張りが無いと、後退の途中の中立止めのあとに後ろへ当たってもやり直しで後ろを押し続ける
          (今までは中立止めの reset が手順を偶然切っていた。safety.md 3 章)"""
        prev = self.params
        if int(prev.get('rec_back_guard', 0) or 0) == 1 and p['rec_back_guard'] == 0:
            p['rec_hold_keep'] = 0
        if running:
            for k, name in self.OPT_NAMES.items():
                if p[k] == 1 and int(prev.get(k, 0) or 0) == 0:
                    raise ValueError('走行中は「%s」を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)' % name)
        if p['rec_hold_keep'] == 1 and p['rec_back_guard'] == 0:
            raise ValueError('「%s」は、%sが ON のときだけ使えます(先に%sを ON にしてください)'
                             % (self.OPT_NAMES['rec_hold_keep'], self.OPT_NAMES['rec_back_guard'], self.OPT_NAMES['rec_back_guard']))
        return p

    def _speed_ctrl_rules(self, p, running):
        """速さの天井・自動合わせの決まり(spec 1.2 節。画面と同じ決まりをサーバーでも守る)。p は _clean_params のあと。
        天井を OFF にしたら自動合わせも OFF。走行中は 0 → 1 にできない・値は変えられない(1 走行 = 1 設定)。
        自動合わせは天井が ON のときだけ。断るときは ValueError(サーバーは 400)"""
        prev = self.params
        if int(prev.get('speed_ceiling', 0) or 0) == 1 and p['speed_ceiling'] == 0:
            p['speed_auto'] = 0
        if running:
            names = {'speed_ceiling': '速さの天井', 'speed_auto': '速さのゲインの自動合わせ'}
            for k in SPEED_CTRL_SWITCHES:
                if p[k] == 1 and int(prev.get(k, 0) or 0) == 0:
                    raise ValueError('走行中は%sを ON にできません。止めてから ON にしてください(1 走行 = 1 設定)' % names[k])
            for k in SPEED_CTRL_VALUES:
                if p[k] != float(prev.get(k, DEFAULT_PARAMS[k])):
                    raise ValueError('走行中は %s を変えられません。止めてから変えてください(1 走行 = 1 設定)' % k)
        if p['speed_auto'] == 1 and p['speed_ceiling'] == 0:
            raise ValueError('速さのゲインの自動合わせは、速さの天井が ON のときだけ使えます(先に天井を ON にしてください)')
        return p

    def _save_settings(self):
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(SETTINGS_PATH) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'params': self.params, 'run': self.last['run'], 'mode_id': self.last['mode_id']}, f,
                          ensure_ascii=False, indent=1)
            os.replace(tmp, SETTINGS_PATH)
        except OSError:
            pass

    def set_params(self, values):
        """走行中でも反映する(ゲイン・トリム・先読み・時計補正・ジャンル)。送られた項目だけ変える"""
        with self.lock:
            p = self._clean_params(values)
            if self.running and self.st.get('post_failed') and uses_post(p) and not uses_post(self.params):
                raise ValueError('後処理を読み込めなかったため、速度の上限・TCP の出し方は使えません(操作の枝だけで走っています)')
            if self.running and p['zone_caps'] == 1 and int(self.params.get('zone_caps', 0) or 0) == 0:
                # 区間の速度上限は 1 走行 = 1 設定(画面と同じ決まりをサーバーでも守る)。OFF にするのは走行中でもできる
                raise ValueError('走行中は区間の速度上限を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)')
            if self.running and p['start_kick'] == 1 and int(self.params.get('start_kick', 0) or 0) == 0:
                # 発進の補助も 1 走行 = 1 設定(2026-10-26 fix3 r1-rs1-2。画面 app.js の DR_FLAGS と同じ決まり)。走行中に 0 → 1 に
                # すると、StartKick の 2 秒の窓が AI の区間の途中で新しく開き、リカバリーより先に見るので、リカバリーの中止・後退の
                # 途中を上書きして前進 80 µs を最大 2 秒出していた。OFF にするのは走行中でもできる(StartKick を見なくなるだけ)
                raise ValueError('走行中は発進の補助(AI側にしたら走り出す)を ON にできません。止めてから ON にしてください'
                                 '(1 走行 = 1 設定)')
            p = self._speed_ctrl_rules(p, self.running)
            p = self._launch_rules(p, self.running)
            p = self._opt_rules(p, self.running)
            if values.get('mode_id') is not None:
                mode_id = int(values['mode_id'])
                info = self.st.get('model')
                if self.running and info and mode_id not in info['trained_modes']:
                    raise ValueError('このモデルはジャンル %d を学習していません' % mode_id)
                if self.running:
                    self.st['mode_id'] = mode_id
                    if self.recorder.recording:
                        try:
                            self.recorder.set_mode(mode_id)
                        except (RuntimeError, ValueError):
                            pass
                self.last['mode_id'] = mode_id
            changed = params_changed(self.params, p) if self.running else {}
            self.params = p
            self._save_settings()
            if changed:
                self._note_params(changed)
            return dict(p)

    def _params_end_to_log(self, log):
        """止めるとき: 走行中に設定を変えていたら、log.json の drive に params_end・params_changes を足す。戻り値 書いたか"""
        if not self.params_changes:
            return False
        try:
            upd = getattr(self.recorder, 'update_drive', None)
            return bool(upd(log, params_end=dict(self.params), params_changes=list(self.params_changes))) if upd else False
        except Exception:       # noqa: BLE001  書けなくても止める処理は続ける
            traceback.print_exc()
            return False

    def _note_params(self, changed):
        """走行中に変わった設定を events.jsonl(params)と走り終わりの log.json の欄に残す(2026-10-26 fix2 state2-2)"""
        ev = {'changed': changed, 'run': self.st.get('run'), 'epoch_us': int(clock.to_epoch_us(clock.now_ns()))}
        self.params_changes.append(ev)
        try:
            self.recorder.add_event('params', ev)
        except Exception:       # noqa: BLE001  記録できなくても走行は続ける
            traceback.print_exc()

    def start(self, run, mode_id, record=False, description='', tags=None, dry_run=False, params=None):
        """走行を始める。戻り値 (run, token)。token は生存確認(heartbeat)に使う"""
        with self.lock:
            if self.running:
                raise RuntimeError('すでに走行中です')
            info = model_info(run or '')
            if info is None:
                raise ValueError('走行に使える学習結果がありません(TensorRTエンジンが必要): %s' % run)
            if not info.get('drivable', True):
                raise ValueError('このモデルでは走行できません: %s' % info.get('why_not'))
            mode_id = int(mode_id)
            if mode_id not in info['trained_modes']:
                raise ValueError('このモデルはジャンル %d(%s)を学習していません' % (mode_id, modes.name_of(mode_id)))
            new_params = self._opt_rules(self._launch_rules(self._speed_ctrl_rules(self._clean_params(params or {}), False), False),
                                         False)
            if info.get('postprocess') and uses_post(new_params):
                ref0 = (float(info.get('steer_center_us') or PWM_CENTER), float(info.get('throttle_neutral_us') or PWM_CENTER))
                obj, note = post_mod.make(info, ref0)
                if obj is None:
                    raise ValueError('%s。速度の上限・TCP の出し方は使えません(上限を 0、出し方を「操作の枝だけ」にすると走れます)' % note)
            self.params = new_params
            self.last = {'run': run, 'mode_id': mode_id}
            self._save_settings()
            self._stop.clear()
            self.history.clear()
            self.hb_token = uuid.uuid4().hex
            self.heartbeat_t = time.monotonic()
            self.hb_count = 0
            self.last_takeover_ns = None
            self.takeover_reasons = []
            self.params_changes = []
            self.st = {'phase': 'starting', 'message': 'モデルとセンサーを準備中…', 'error': None, 'run': run,
                       'model': info, 'mode_id': mode_id, 'dry_run': bool(dry_run), 'record': bool(record),
                       'recording_note': '', 'ref': None, 'started': time.time(), 'stop_reason': '',
                       'stats': {}, 'now': None, 'description': description or ''}
            self.thread = threading.Thread(target=self._run, args=(info, bool(record), description or '',
                                                                   list(tags or []), bool(dry_run)),
                                           name='drive', daemon=True)
            self.thread.start()
            return run, self.hb_token

    TAKEOVER_REASONS = {'collision': '衝突した', 'stuck': '止まった', 'preemptive': '危なかった(予防)', 'other': 'その他・終了'}

    def takeover_reason(self, reason):
        """人が交代した理由を記録する(ロードマップ S3)。直前(60秒以内)の交代の時刻と結びつける"""
        if reason not in self.TAKEOVER_REASONS:
            raise ValueError('理由は %s のどれか' % ', '.join(self.TAKEOVER_REASONS))
        now = clock.now_ns()
        tk = self.last_takeover_ns if self.last_takeover_ns and now - self.last_takeover_ns < 60e9 else None
        ev = {'reason': reason, 'label': self.TAKEOVER_REASONS[reason],
              'takeover_us': int(tk // 1000) if tk else None,
              'since_takeover_s': round((now - tk) / 1e9, 2) if tk else None, 'run': self.st.get('run')}
        ev['recorded'] = self.recorder.add_event('takeover_reason', ev)
        self.takeover_reasons.append(ev)
        self.st['takeover_reasons'] = self.takeover_reasons[-20:]
        return ev

    def stop(self, reason='停止ボタン'):
        if not self.running:
            raise RuntimeError('走行していません')
        self.st['stop_reason'] = reason
        self.st['message'] = '停止中…'
        self._stop.set()

    def shutdown(self, timeout=3.0):
        """サーバー終了時: 止めて、中立を出し終わるまで待つ"""
        if self.running:
            try:
                self.stop('サーバー終了')
            except RuntimeError:
                pass
            self.thread.join(timeout)

    # ------------------------------------------------------------ 走行ループ
    def _run(self, info, record, description, tags, dry_run):
        st = self.st
        pca = out = runner = listeners = None
        started_log = None
        ref = (PWM_CENTER, PWM_CENTER)
        try:
            self.hub.start()
            from .actuator import PCA9685
            if info['steer_center_us'] is not None and info['throttle_neutral_us'] is not None:
                ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
            else:
                ref = _reference_from_dataset(info['dataset'])
            st['ref'] = {'steer_center_us': ref[0], 'throttle_neutral_us': ref[1]}

            # モデルを読む前に中立を出す(前のプロセスが残した出力を、準備の数秒間も車に出し続けないように)
            pcfg = config.PCA9685
            osc_pct = self.params['osc_percent']
            try:
                pca = PCA9685(pcfg, 1.0 + osc_pct / 100.0)
            except Exception as e:
                if not dry_run:
                    raise RuntimeError('PCA9685(I2C 0x%02x)に出力できません: %s' % (pcfg['address'], e))
                st['recording_note'] = 'PCA9685に接続できないため、出力なしで推論だけ行います'
            out = _Output(pca, pcfg, ref, lambda: self.params)
            out.neutral()

            import torch
            from .train_data import normalize_inputs
            from .trt_runtime import TRTRunner
            torch.zeros(1, device='cuda')
            torch.cuda.synchronize()        # このスレッドでCUDAを初期化しておく
            runner = TRTRunner(str(RUNS_DIR / info['name'] / ENGINE_FILE))
            num_modes = info['num_modes']
            # エンジンの入出力が走行の約束(基本の5つの入力・actions [1,10,2])に合うか、走る前に確かめる
            ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                        {n: list(runner.shapes[n]) for n in runner.outputs}, num_modes)
            if not ok:
                raise RuntimeError('エンジンの入出力が走行に合いません: %s' % why)
            zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}     # エンジンの形から作る
            for _ in range(3):              # 初回の推論は遅いので先に済ませる
                runner.run_numpy(**zeros)

            tof_buf = collections.deque(maxlen=10)
            imu_buf = collections.deque(maxlen=60)
            veh_buf = collections.deque(maxlen=50)
            cam_slot = [None]
            cam_evt = threading.Event()

            def on_sample(channel, s):
                if channel == 'CAM_FRONT':
                    cam_slot[0] = s
                    cam_evt.set()
                elif channel == 'TOF_FRONT':
                    tof_buf.append((s.t_ns, s.data['depth']))
                elif channel == 'IMU':
                    imu_buf.append((s.t_ns, np.concatenate([s.data['accel'], np.deg2rad(s.data['gyro'])])))
                elif channel == 'VEHICLE':
                    veh_buf.append((s.t_ns, s.data))
            listeners = on_sample
            for ch in ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE'):
                self.hub[ch].subscribe(on_sample)

            if record:
                st['message'] = '収集を開始しています…'
                pp = info.get('postprocess') or None
                drive_info = {'run': info['name'], 'dataset': info['dataset'], 'dry_run': dry_run,
                              'params': dict(self.params), 'steer_center_us': ref[0], 'throttle_neutral_us': ref[1],
                              'policy': info.get('policy', policies.DEFAULT), 'policy_cfg': info.get('policy_cfg'),
                              'postprocess': ({'kind': pp.get('kind'), 'version': pp.get('version')}
                                              if isinstance(pp, dict) else None)}
                try:
                    started_log = self.recorder.start(description=description, tags=tags, mode_id=st['mode_id'],
                                                      drive=drive_info)
                    st['record_log'] = started_log            # 走り終わりのメモを書く記録(走行タブ)
                except RuntimeError:
                    st['recording_note'] = 'すでに収集中のため、その収集にAIの指令も記録します'
            self._loop(st, runner, normalize_inputs, pca, out, ref, num_modes, osc_pct,
                       cam_slot, cam_evt, tof_buf, imu_buf, veh_buf, dry_run)
            st['phase'] = 'idle'
            st['message'] = '停止しました(%s)' % (st['stop_reason'] or '停止')
        except Exception as e:
            traceback.print_exc()
            st['phase'] = 'error'
            st['error'] = '%s: %s' % (type(e).__name__, e)
            st['message'] = '走行を開始できませんでした' if not st.get('stats') else '走行を止めました(エラー)'
        finally:
            # 後始末は1つずつ。どれかが失敗しても、ほかを必ず行う(中立が最優先)
            if out is not None and not out.neutral_safe():
                st['error'] = (st.get('error') or '') + ' / 停止時に中立を出せませんでした(CH3を人側へ戻してください)'
            if pca is not None:
                try:
                    pca.close()
                except Exception:
                    pass
            if listeners is not None:
                for ch in ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE'):
                    try:
                        self.hub[ch].unsubscribe(listeners)
                    except Exception:
                        pass
            if started_log:
                self._params_end_to_log(started_log)     # 走行中に変えた設定(変えなかった走行の log.json は今までと同じ)
            if started_log:
                try:
                    st['message'] += '・収集したシーンを保存中…'
                    stopped = self.recorder.stop(only_log=started_log)     # 途中で人が始め直した収集は止めない
                    st['message'] = st['message'].replace('・収集したシーンを保存中…',
                                                          '・収集したシーンを保存しました' if stopped else '')
                except Exception:
                    traceback.print_exc()
            if st.get('announced'):
                self._emit('drive_stop', reason=st.get('stop_reason', ''), error=st['phase'] == 'error')
            if runner is not None:
                try:
                    runner = None              # GPUメモリを返す(学習に必要な空きを残す)
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    traceback.print_exc()
            st['finished'] = time.time()

    def _loop(self, st, runner, normalize_inputs, pca, out, ref, num_modes, osc_pct,
              cam_slot, cam_evt, tof_buf, imu_buf, veh_buf, dry_run):
        dcfg = config.DRIVE
        fps = config.CAMERA['fps']
        max_age = int(dcfg['max_sensor_age_ms'] * 1e6)
        imu_age = int(dcfg['imu_max_age_ms'] * 1e6)
        jpeg = [cv2.IMWRITE_JPEG_QUALITY, config.CAMERA['jpeg_quality']]
        lo, hi = dcfg['min_us'], dcfg['max_us']
        stats = {'frames': 0, 'skipped': 0, 'holds': 0, 'fps': 0.0, 'latency_ms': 0.0, 'infer_ms': 0.0, 'i2c_errors': 0,
                 'recoveries': 0, 'esc_assists': 0, 'giveups': 0, 'start_kicks': 0, 'launch_boosts': 0}
        helper = assist_mod.Assist(ref)
        last_kind = assist_mod.ASSIST_NONE
        # 発進の前進(launch_boost。既定 OFF): 直近 5 フレームの ToF ガードの TTC(中立で止めたら空にする)。OFF でも数を
        # 残すだけで出力は変えない
        ttc_hist = collections.deque(maxlen=assist_mod.LaunchBoost.TTC_N)
        st['launch'] = {'on': int(self.params.get('launch_boost', 0) or 0) == 1, 'active': False, 'count': 0,
                        'last_end': '', 'last_skip': ''}
        # 方策の後処理(tcp の制御器)と予測の重ね合わせ。後処理の無い方策・ensemble_j = 0 なら actions のまま
        stage = post_mod.Stage(st.get('model') or {}, ref, fps=fps)
        if stage.note:
            st['recording_note'] = ((st.get('recording_note') or '') + ' ' + stage.note).strip()
        st['post_kind'] = post_mod.kind_of(st.get('model') or {}) if stage.post is not None else None
        st['post_failed'] = stage.failed
        guard = None                        # ToFガード(床の基準があるときだけ)
        try:
            from .tof_guard import ToFGuard
            guard = ToFGuard(np.load(FLOOR_REF_PATH), center=ref[0], neutral=ref[1])
        except (OSError, ValueError) as e:
            st['recording_note'] = (st.get('recording_note') or '') + ' ToFガードは使えません(床の基準 %s: %s)' % (
                FLOOR_REF_PATH.name, e)
        stats.update(guard_slow=0, guard_brake=0)
        last_guard = 0
        laps = LapCounter()                 # ジャイロで周回を数え、1周のタイムを測る
        st['laps'] = laps.summary()
        imu_done = [0]
        zcaps = _ZoneCaps()                 # 区間の速度上限(OFF でも影の判定を記録する)
        st['zone_caps_error'] = None
        stats.update(zone_ms=0.0, zone_ms_max=0.0, zone_brakes=0, zone_applied=0)
        # 速さの天井・速さのゲインの自動合わせ(OFF でも影の判定を記録する。値はこの走行のあいだ変わらない)
        sceil = _SpeedCeil(self.params)
        sauto = _SpeedAuto(self.params, st.get('description') or '')
        lap_marks = []                      # LapCounter の周の区切りの時刻(位相が失敗したときの R1 の起点の代わり)
        st['speed_ctrl_error'] = None
        st['auto_end'] = None
        stats.update(ceil_applied=0, ceil_neutral=0, ceil_ms=0.0, ceil_ms_max=0.0, auto_ms=0.0, auto_ms_max=0.0,
                     auto_laps=0)
        # AI の時間(画面の「AI の時間」。ベンチマークの「数える 240 秒 + 1 分」で止める目安。2026-10-21 Fix-all)
        stats['ai_s'] = 0.0
        ai_sum, ai_t_prev, ai_was = 0.0, None, False

        def zone_failed():
            """上限の計算が例外になった(この走行のあいだは上限を外す)ことを、1 回だけ記録とメッセージに残す"""
            if not zcaps.failed or st.get('zone_caps_error'):
                return
            st['zone_caps_error'] = zcaps.failed
            zc_on = int(self.params.get('zone_caps', 0) or 0)
            if zc_on == 1:
                note = '区間の速度上限の計算が失敗したため、この走行では上限を外しました(%s)' % zcaps.failed
            else:                   # OFF の日: 出力はもとから変えていない。止まったのは記録だけの影の判定
                note = ('区間の速度上限(OFF・記録だけ)の影の判定が計算の失敗で止まりました。車への出力は変わりません(%s)'
                        % zcaps.failed)
            st['recording_note'] = ((st.get('recording_note') or '') + ' ' + note).strip()
            st['message'] = (st.get('message') or '') + '・' + note
            try:
                self.recorder.add_event('zone_caps_error', {'error': zcaps.failed, 'run': st.get('run'),
                                                            'zone_caps': zc_on})
            except Exception:       # noqa: BLE001  記録できなくても走行は続ける
                traceback.print_exc()

        def speed_failed():
            """天井・自動合わせの計算が例外になった(この走行のあいだは外す)ことを、それぞれ 1 回だけ記録とメッセージに残す。
            天井が失敗したら自動合わせも外す(自動合わせは天井がある前提)"""
            if sceil.failed and not sauto.failed:
                sauto.failed = '速さの天井の失敗のため外しました(%s)' % sceil.failed
            errs = st.get('speed_ctrl_error') or {}
            for part, obj, name, key in (('ceil', sceil, '速さの天井', 'speed_ceiling'),
                                         ('auto', sauto, '速さのゲインの自動合わせ', 'speed_auto')):
                if not obj.failed or part in errs:
                    continue
                if part == 'auto' and 'ceil' in errs:
                    errs[part] = obj.failed
                    continue                # 天井の文に含めた
                on = int(self.params.get(key, 0) or 0)
                if on == 1:
                    note = '%sの計算が失敗したため、この走行では%sを外しました(%s)' % (
                        name, '天井と自動合わせ' if part == 'ceil' else '自動合わせ', obj.failed)
                else:
                    note = '%s(OFF・記録だけ)の影の判定が計算の失敗で止まりました。車への出力は変わりません(%s)' % (name, obj.failed)
                errs[part] = obj.failed
                st['recording_note'] = ((st.get('recording_note') or '') + ' ' + note).strip()
                st['message'] = (st.get('message') or '') + '・' + note
                try:
                    self.recorder.add_event('speed_ctrl_error', {'part': part, 'error': obj.failed, 'run': st.get('run'),
                                                                 'speed_ceiling': int(self.params.get('speed_ceiling', 0) or 0),
                                                                 'speed_auto': int(self.params.get('speed_auto', 0) or 0)})
                except Exception:       # noqa: BLE001  記録できなくても走行は続ける
                    traceback.print_exc()
            if errs:
                st['speed_ctrl_error'] = errs

        def launch_note(cam_t_ns=None):
            """発進の前進の出来事(始めた・始めなかった・終わった)を記録の出来事と画面の状態へ。OFF のときは何もしない"""
            b = helper.boost
            ev, b.event = b.event, None
            lst = st['launch']
            lst.update(on=int(self.params.get('launch_boost', 0) or 0) == 1, active=bool(b.active), count=b.count,
                       last_end=assist_mod.LAUNCH_END_NAMES.get(b.last_end, ''),
                       last_skip=assist_mod.LAUNCH_SKIP_NAMES.get(b.last_skip, ''))
            stats['launch_boosts'] = b.count
            if ev is None:
                return
            info = dict(ev[1], event=ev[0], run=st.get('run'))
            if cam_t_ns is not None:
                info['cam_t_us'] = int(cam_t_ns // 1000)                       # 単調時計(CLOCK_MONOTONIC)の µs
                info['cam_epoch_us'] = int(clock.to_epoch_us(cam_t_ns))        # AI_CMD の timestamps_us と同じ時計(2026-10-26 fix2)
            try:
                self.recorder.add_event('launch_boost', info)
            except Exception:       # noqa: BLE001  記録できなくても走行は続ける
                traceback.print_exc()

        def count_laps(veh_now):
            """前のフレームから届いたIMUのサンプルを周回の計測へ"""
            if veh_now is None:
                return
            for t_ns, x in list(imu_buf):
                if t_ns <= imu_done[0]:
                    continue
                imu_done[0] = t_ns
                lap = laps.add(t_ns / 1e9, float(np.degrees(x[5])), float(veh_now['erpm']), bool(veh_now['select']),
                               float(np.hypot(x[0], x[1])) / 9.80665)
                if lap:
                    st['laps'] = laps.summary()
                    lap_marks.append(lap['t_end'])
                    self._emit('lap', **lap)
        st['stats'] = stats
        st['phase'] = 'driving'
        st['message'] = '走行中' if not dry_run else '推論の確認中(出力は中立のまま)'
        st['announced'] = True
        self._emit('drive_start', dry_run=dry_run)
        last_select, rc_lost = None, False
        prev_pred, prev_cam_t, applied = None, None, False
        use_command = (st.get('model') or {}).get('command_input', True)   # 学習と同じく、今の操作値を入れるか
        last_seq = None
        mode_cache = {}
        times = collections.deque(maxlen=30)
        lat_hist = collections.deque(maxlen=30)
        inf_hist = collections.deque(maxlen=30)
        # 中立止めの間の IMU(2026-10-26 fix3 r1-rs1-1)。rec_hold_keep でリカバリーを残した中立止めの間は helper.step を呼ばず、
        # impact_g は処理したフレームの直近 0.15 秒の IMU しか見ないので、その間に後ろへ当たっても Recovery が見落とし、明けたあと
        # 'reverse' が後ろの物を押し続けた。残した中立止めのあとの最初のフレームで、前に処理したフレームのあとに来た IMU の
        # 水平 G の最大を hold_g として渡す(assist.Recovery.step)。[見た IMU の最後の時刻 ns, 残した中立止めの間の G の最大,
        # 残した中立止めがあった]。出力に効くのは残した中立止めのあとだけ(rec_hold_keep = 0 なら hold_g = 0 = 今までと同じ)
        hold_imu = [None, 0.0, False]
        # 残した中立止めの間の後退(2026-10-27 fix3 r2-r2-rs-1)。[前に見た車両の値の時刻 ns(None = まだ), その eRPM, 数えた秒]。
        # 前に helper.step を呼んだフレームの撮像時刻(proc_cam)の車両の値から数え始める。出力に効くのは残した中立止めのあとだけ
        # [3] = その中立止めの前に処理したフレームの撮像時刻 ns・[4] = 車両の値が来なかった間に ESC が後退で走らせていたかもしれない
        # 秒(esc_track の見積もり。Recovery へ hold_rev_s で渡す)(2026-10-27 fix6 safety r1。fix5 の [3] = 値が来なかっただけの秒は
        # やめた)。出力に効くのは残した中立止めのあとだけ
        hold_veh = [None, 0.0, 0.0, None, 0.0]
        proc_cam = [None]
        hold_t0 = [None]            # 残した中立止めが中立を書き始めた時刻(time.monotonic()。helper.step の now と同じ時計)
        # PCA9685 に書けたスロットルから、ESC が後退で走らせていたかもしれない区間(2026-10-27 fix6 safety r1。assist.EscTrack)。
        # rec_hold_keep = 1・rec_back_guard = 1 の日だけ値を入れ、残した中立止めのあと車両の値が来なかった間の数えにだけ使う
        esc_track = assist_mod.EscTrack(ref[1])

        def keep_on():
            return int(p.get('rec_hold_keep', 0) or 0) == 1 and int(p.get('rec_back_guard', 0) or 0) == 1

        def veh_unseen_back(upto_ns=None):
            """前に見た車両の値のあとに来たサンプル(upto_ns まで)で、後ろへ動いていた秒(見た時刻を進める。veh_buf は 50 個 =
            100 Hz で約 0.5 秒。中立止めの hold() は 0.1 秒以内ごとに呼ばれる)。隣り合う 2 つのサンプルのどちらかが
            BACKING_ERPM を超えていたら、その間を全部数える。間が HOLD_VEH_GAP_S より空いた(車両の値が来なかった)ら
            (2026-10-27 fix6 safety r1): 前に見た値が HOLD_VEH_FAST_ERPM より速く後ろへ動いていたなら全部数える(中立の惰性でも
            下がっていた)。そうでなければ、書いた値から見積もった「ESC が後退で走らせていたかもしれない秒」(esc_track)を
            hold_veh[4] に足す(戻り値には入れない)。どちらも、前に見た値が後ろへ動いていた(> BACKING_ERPM)なら、そのあと
            処理したフレームは Recovery がその値で dt を数えたので、前に処理したフレームの撮像時刻(hold_veh[3])から"""
            s = 0.0
            for t_ns, x in list(veh_buf):
                if upto_ns is not None and t_ns > upto_ns:
                    break
                if hold_veh[0] is not None and t_ns <= hold_veh[0]:
                    continue
                e = float(x['erpm'])
                if hold_veh[0] is not None:
                    if t_ns - hold_veh[0] > HOLD_VEH_GAP_S * 1e9:
                        a = hold_veh[0]
                        if hold_veh[1] > assist_mod.BACKING_ERPM and hold_veh[3] is not None:
                            a = max(a, hold_veh[3])
                        if hold_veh[1] > HOLD_VEH_FAST_ERPM:
                            s += max(t_ns - a, 0) / 1e9
                        else:
                            hold_veh[4] += esc_track.rev_s(a, t_ns)
                    elif e > assist_mod.BACKING_ERPM or hold_veh[1] > assist_mod.BACKING_ERPM:
                        s += (t_ns - hold_veh[0]) / 1e9
                hold_veh[0], hold_veh[1] = t_ns, e
            return s

        def veh_seen_from(cam_ns):
            """残した中立止めの始め: 前に処理したフレームが使った車両の値(撮像時刻より前の最新)から数える"""
            hold_veh[0], hold_veh[1], hold_veh[2], hold_veh[3], hold_veh[4] = None, 0.0, 0.0, cam_ns, 0.0
            if cam_ns is None:
                return
            for t_ns, x in list(veh_buf):
                if t_ns <= cam_ns:
                    hold_veh[0], hold_veh[1] = t_ns, float(x['erpm'])

        def imu_unseen_g():
            """前に見た IMU のあとに来たサンプルの水平 G の最大 [G](見た時刻を進める。imu_buf は 60 個 = 約 0.6 秒)"""
            g, t_seen = 0.0, hold_imu[0]
            for t_ns, x in list(imu_buf):
                if t_seen is None or t_ns > t_seen:
                    g = max(g, float(np.hypot(x[0], x[1])) / 9.80665)
                    hold_imu[0] = t_ns if hold_imu[0] is None else max(hold_imu[0], t_ns)
            return g

        def hold(reason, cam=None, veh_select=None, infer_ms=np.nan):
            if veh_select is None:          # 最新の車両値から(AI側のまま止めているのか分かるように)
                veh_select = bool(veh_buf[-1][1]['select']) if veh_buf else False
            lb_end = helper.launch_hold()   # 発進の前進は中立で止めたら終わり(⑧)
            ttc_hist.clear()
            launch_note(cam.t_ns if cam is not None else None)
            n_ok = out.neutral()
            if keep_on():                   # ESC に届いた値(人側・dry run は分からない。fix6 safety r1)
                if not veh_select or dry_run:
                    esc_track.unknown(clock.now_ns())
                elif n_ok:
                    esc_track.write(clock.now_ns(), out.neutral_values()[1])
            stats['holds'] += 1
            stats['i2c_errors'] = out.errors
            n_s, n_t = out.neutral_values()
            st['hold_reason'] = reason
            st['hold_since'] = st.get('hold_since') or time.time()
            st['now'] = {'raw': None, 'out': [round(n_s, 1), round(n_t, 1)], 'select': veh_select, 'hold': True, 'pred': None}
            zcaps.hold(cam.t_ns if cam is not None else None, imu_buf, veh_buf)
            zone_failed()
            sceil.hold()
            if cam is not None:
                sauto.hold(cam.t_ns, imu_buf, zcaps, lap_marks)
            speed_failed()
            if cam is not None:
                self.recorder.add_ai(cam.t_ns, {
                    'pred': np.full((10, 2), np.nan, np.float32), 'raw': (np.nan, np.nan), 'out': (n_s, n_t),
                    'gains': (p['steer_gain'], p['throttle_gain']), 'trim': p['steer_trim_us'], 'lookahead': p['lookahead'],
                    'mode_id': st['mode_id'], 'hold': True, 'select': veh_select, 'latency_ms': np.nan,
                    'infer_ms': infer_ms, 'dry_run': dry_run, 'assist': assist_mod.ASSIST_NONE,
                    'model_out': (np.nan, np.nan), 'ttc': np.nan, 'guard': 0, 'guard_applied': False,
                    'model_pred': np.full((10, 2), np.nan, np.float32), 'post_mode': -1, 'v_hat': np.nan,
                    'v_cmd': np.nan, 'ens_j': int(p.get('ensemble_j', 0)),
                    **zcaps.record(None, int(p.get('zone_caps', 0) or 0) == 1, False, n_t),
                    **sceil.record(None, int(p.get('speed_ceiling', 0) or 0) == 1, False, n_t),
                    **sauto.record_hold(int(p.get('speed_auto', 0) or 0) == 1),
                    'launch_on': False, 'launch_end': int(lb_end)})
            if (int(p.get('rec_hold_keep', 0) or 0) == 1 and int(p.get('rec_back_guard', 0) or 0) == 1
                    and veh_select and not dry_run):
                helper.reset(keep_recovery=True)    # AI 側のままの中立止め: リカバリーは残す(rec_hold_keep。見張りが ON のときだけ)
                hold_imu[1] = max(hold_imu[1], imu_unseen_g())     # この間の衝撃を、次に処理したフレームで Recovery に渡す
                if not hold_imu[2]:
                    veh_seen_from(proc_cam[0])      # この中立止めの始め(r2-r2-rs-1)
                    hold_t0[0] = time.monotonic()
                hold_veh[2] += veh_unseen_back()    # この間に後ろへ動いていた秒も、次に処理したフレームで渡す
                hold_imu[2] = True
            else:
                helper.reset()              # 中立で止めた後は、補助の手順を最初から
                hold_imu[1], hold_imu[2] = 0.0, False
                hold_veh[2] = hold_veh[4] = 0.0
            stage.reset()                   # 後処理・重ね合わせも最初から(止めていた間の予測を混ぜない)

        while not self._stop.is_set():
            p = self.params
            if time.monotonic() - self.heartbeat_t > dcfg['heartbeat_s']:
                st['stop_reason'] = ('走行を始めた画面からの応答が一度もありませんでした。画面を再読み込みしてから開始してください'
                                     if self.hb_count == 0 else
                                     '走行を始めた画面との通信が %.0f 秒途絶えたため' % dcfg['heartbeat_s'])
                break
            if pca is not None and p['osc_percent'] != osc_pct:
                osc_pct = p['osc_percent']
                pca.set_osc_scale(1.0 + osc_pct / 100.0)
            if not cam_evt.wait(0.1):
                if self._stop.is_set():
                    break
                hold('カメラの映像が来ていません')
                applied = False
                continue
            cam_evt.clear()
            cam = cam_slot[0]
            seq = cam.data['seq']
            if seq == last_seq:
                continue
            if last_seq is not None and seq > last_seq + 1:
                stats['skipped'] += seq - last_seq - 1
            last_seq = seq
            # AI の時間: 道具(tools/label_events の dt_s・ai)と同じ決め方。前のフレームが AI 側(CH3 が AI・dry run でない)なら、
            # そのフレームから今のフレームまでの時間を足す(0.2 秒を超えて空いたら 1 フレーム分)。数えるだけで出力は変えない
            if ai_t_prev is not None and ai_was:
                gap_s = (cam.t_ns - ai_t_prev) / 1e9
                ai_sum += gap_s if 0.0 <= gap_s <= AI_GAP_S else 1.0 / fps
                stats['ai_s'] = round(ai_sum, 1)
            ai_t_prev = cam.t_ns
            ai_was = bool(veh_buf) and not dry_run and bool(veh_buf[-1][1]['select'])
            if clock.now_ns() - cam.t_ns > max_age:
                hold('処理が遅れています(画像が%dms前)' % ((clock.now_ns() - cam.t_ns) / 1e6), cam)
                applied = False
                continue
            tof_raw, imu, veh, reason = assemble_inputs(cam.t_ns, list(tof_buf), list(imu_buf), list(veh_buf),
                                                        max_age, imu_age)
            if reason:
                hold(reason, cam)
                applied = False
                continue
            count_laps(veh)
            if last_select is not None and bool(veh['select']) != last_select:
                self._emit('handover', select=bool(veh['select']))       # CH3で操縦が切り替わった
                if not veh['select']:
                    self.last_takeover_ns = cam.t_ns                        # 人が取り返した(交代の理由のボタンと結びつける)
                    st['takeovers'] = st.get('takeovers', 0) + 1
            last_select = bool(veh['select'])
            if veh['select'] and not veh['valid'][2]:
                # 送信機の電源断・圏外: 基板はSELECTをAI側のまま保つので、人が取り戻せない。中立で止める
                if not rc_lost:
                    self._emit('rc_lost')
                rc_lost = True
                hold('AI側のままプロポの信号(CH3)が届いていません', cam, True)
                applied = False
                continue
            rc_lost = False
            mode_id = st['mode_id']
            prev_cmd = predicted_now(prev_pred, prev_cam_t, cam.t_ns, fps) if prev_pred is not None else None
            vehicle = vehicle_state(veh, prev_cmd, ref, applied)
            ok, enc = cv2.imencode('.jpg', cam.data['image'], jpeg)     # 学習データ(JPEG保存)と同じ画質にそろえる
            image = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else cam.data['image']
            img, tof, imu_n, state = normalize_inputs(image, tof_raw, imu, vehicle, use_command)
            if mode_id not in mode_cache:
                oh = np.zeros((1, num_modes), np.float32)
                oh[0, mode_id] = 1.0
                mode_cache[mode_id] = oh
            t0 = time.perf_counter()
            # 基本の5つを必ず渡す(エンジンが使わない入力は読まれない)
            feed = {'image': img[None], 'tof': tof[None], 'imu': imu_n[None], 'state': state[None],
                    'mode': mode_cache[mode_id]}
            res = runner.run_numpy(**feed)
            model_pred = stage.model_pred(res)     # [10,2] µs(後処理・重ね合わせの前)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            if not np.all(np.isfinite(model_pred)):
                hold('推論結果が数値ではありません', cam, bool(veh['select']), infer_ms)
                applied = False
                continue
            # 後処理(tcp の制御器。無い方策はそのまま)→ 予測の重ね合わせ(ensemble_j > 0 のとき)
            v_now = post_mod.speed_of(veh['erpm'])
            try:
                pred, pinfo = stage.apply(res, model_pred, v_now, p, cam.t_ns)
            except post_mod.PostError as e:
                hold(str(e), cam, bool(veh['select']), infer_ms)          # 推論の異常と同じく中立
                applied = False
                continue
            k = min(max(p['lookahead'], 1), len(pred)) - 1
            raw = (float(pred[k, 0]), float(pred[k, 1]))
            # 速さのゲインの自動合わせ: g = throttle_gain × m(OFF でも影で計算する)。掛けるのは ON(天井も ON)・AI のフレームだけ。
            # 前進側だけに掛かるのは apply_gains のまま。モデルに戻す「今の操作値」はゲインの前の予測のまま
            ai_frame = bool(veh['select']) and not dry_run
            ceil_on = int(p.get('speed_ceiling', 0) or 0) == 1
            auto_on = ceil_on and int(p.get('speed_auto', 0) or 0) == 1
            g_auto = sauto.gain(cam.t_ns, p['throttle_gain'], -float(veh['erpm']) * safety_caps.KV, veh_buf)
            speed_failed()
            g_used = p['throttle_gain']
            if auto_on and ai_frame and g_auto is not None and not sceil.failed:
                g_used = g_auto
            model_us = apply_gains(raw, ref, p if g_used == p['throttle_gain'] else dict(p, throttle_gain=g_used), lo, hi)
            # ToFガード: 衝突までの時間(TTC)が短ければ減速(・近い側から離すハンドル)。1 = 記録だけ
            g_mode, g_code, g_ttc, g_applied, base_us = int(p.get('tof_guard', 1)), 0, np.nan, False, model_us
            if guard is not None and g_mode >= 1:
                g_steer, g_thr, g_info = guard.step(model_us[0], model_us[1], tof_raw, float(veh['erpm']),
                                                    float(np.degrees(imu[-1, 5])))
                g_ttc = float(g_info['ttc'])
                g_code = {'slow': 1, 'brake': 2}.get(g_info['action'], 0)
                if g_code and g_mode >= 2 and veh['select'] and not dry_run:
                    base_us = (g_steer if g_mode >= 3 else model_us[0], g_thr)
                    g_applied = True
                if g_code and not last_guard:
                    stats['guard_brake' if g_code == 2 else 'guard_slow'] += 1
                last_guard = g_code
            ttc_hist.append(g_ttc)
            # 補助: 後退の受け付け(中立を挟む)・ぶつかって止まったときの自動リカバリー
            recent = [x[1] for x in list(imu_buf)[-10:] if x[0] > cam.t_ns - 150_000_000]
            impact_g = max((float(np.hypot(a[0], a[1])) / 9.80665 for a in recent), default=0.0)
            # リカバリーを残した中立止めのあとの最初のフレーム: その間(前に処理したフレームのあと)の IMU の衝撃(r1-rs1-1)
            held = hold_imu[2]
            hold_since = hold_t0[0] if held else None
            hold_g = max(hold_imu[1], imu_unseen_g()) if held else 0.0
            # その間(前に処理したフレームからこのフレームまで)に後ろへ動いていた秒(r2-r2-rs-1)
            hold_back_s = (hold_veh[2] + veh_unseen_back(cam.t_ns)) if held else 0.0
            hold_rev_s = hold_veh[4] if held else 0.0          # 値が来なかった間の後退の見積もり(fix6 safety r1)
            hold_imu[1], hold_imu[2] = 0.0, False
            hold_veh[2] = hold_veh[4] = 0.0
            proc_cam[0] = cam.t_ns
            if imu_buf:
                t_new = imu_buf[-1][0]
                hold_imu[0] = t_new if hold_imu[0] is None else max(hold_imu[0], t_new)
            # 発進の前進の入力(launch_boost = 1 のときだけ作る。OFF は None = 今までと同じ)
            launch_in = None
            if int(p.get('launch_boost', 0) or 0) == 1:
                launch_in = launch_inputs(p, cam.t_ns, tof_buf, imu_buf, veh_buf, ref, base_us, g_ttc, ttc_hist,
                                          guard is not None and g_mode >= 1, impact_g, g_applied)
            out_us, kind = helper.step(base_us, float(veh['erpm']), bool(veh['select']) and not dry_run,
                                       time.monotonic(), esc_on=p.get('esc_reverse', 1) == 1,
                                       rec_on=p.get('recovery', 1) == 1,
                                       impact_g=impact_g, rev_us=float(p.get('rev_us', assist_mod.DEFAULT_REV_US)),
                                       rev_s=float(p.get('rev_s', assist_mod.DEFAULT_REV_S)),
                                       kick_on=int(p.get('start_kick', 0) or 0) == 1,
                                       blocked=bool(np.isfinite(g_ttc) and g_ttc < START_KICK_TTC_S),
                                       launch=launch_in, rec_guard=int(p.get('rec_back_guard', 0) or 0) == 1,
                                       launch_no_kick=int(p.get('launch_no_kick', 0) or 0) == 1, hold_g=hold_g,
                                       hold_since=hold_since, hold_back_s=hold_back_s, hold_rev_s=hold_rev_s,
                                       rec_keep=(int(p.get('rec_hold_keep', 0) or 0) == 1
                                                 and int(p.get('rec_back_guard', 0) or 0) == 1))
            launch_on, launch_end = helper.launch_on, helper.launch_end
            launch_note(cam.t_ns)
            if helper.rec.guard_event is not None:      # リカバリーの後ろの見張りが手順を変えた(rec_back_guard = 1 のときだけ)
                rg, helper.rec.guard_event = helper.rec.guard_event, None
                try:
                    info = {'what': rg[0], 'phase': rg[1], 'backed_s': rg[2], 'run': st.get('run'),
                            'cam_epoch_us': int(clock.to_epoch_us(cam.t_ns))}
                    if len(rg) > 3:                     # hold_long: 見ていない秒(前に処理したフレームから。2026-10-27 fix3 r2-r2-rs-2)
                        info['hold_s'] = rg[3]
                    self.recorder.add_event('recovery_guard', info)
                except Exception:       # noqa: BLE001  記録できなくても走行は続ける
                    traceback.print_exc()
            out_us = (float(np.clip(out_us[0], lo, hi)), float(np.clip(out_us[1], lo, hi)))
            if kind != last_kind:
                if kind == assist_mod.ASSIST_RECOVERY:
                    laps.note('recovery')
                    self._emit('recovery')
                elif kind == assist_mod.ASSIST_GIVEUP:
                    stats['giveups'] += 1
                    self._emit('recovery_giveup')
                last_kind = kind
            stats['recoveries'] = helper.recoveries
            if stats['frames'] % 15 == 0:
                st['laps'] = laps.summary()
            stats['esc_assists'] = helper.esc_count
            stats['start_kicks'] = helper.kick_count
            # 区間の速度上限: 車に出す直前のスロットルに(ゲイン・ToF ガード・補助のあと)。OFF でも計算して記録する(影の判定)。
            # 出力を変えるのは zone_caps = 1 で、AI 側のフレームで、上限が前進を弱めたときだけ(計算の失敗後は外す)
            zone_on = int(p.get('zone_caps', 0) or 0) == 1
            cap_in = out_us[1]
            zout = zcaps.frame(cam.t_ns, out_us, veh, pred, bool(veh['select']) and not dry_run, imu_buf, veh_buf)
            zone_failed()
            zone_applied = False
            if zone_on and zout is not None and zout.active and zout.thr_us != out_us[1]:
                out_us = (out_us[0], float(np.clip(zout.thr_us, lo, hi)))
                zone_applied = True
            if zcaps.ms:
                stats['zone_ms'] = round(float(np.mean(zcaps.ms)), 3)
                stats['zone_ms_max'] = round(zcaps.ms_max, 3)
            # 速さの天井: 車に出す直前(ゲイン・ToF ガード・補助・区間の上限のあと)。OFF でも計算して記録する(影の判定)。
            # 出力を変えるのは speed_ceiling = 1 で、AI のフレームで、天井が前進を弱めたときだけ(計算の失敗後は外す)
            ceil_in = out_us[1]
            cout = sceil.frame(cam.t_ns, out_us, ref[0] + p['steer_trim_us'], ref[1], ai_frame, imu_buf, veh_buf)
            speed_failed()
            ceil_applied = False
            if ceil_on and cout is not None and cout.active and cout.thr_us != out_us[1]:
                out_us = (out_us[0], float(np.clip(cout.thr_us, lo, hi)))
                ceil_applied = True
            if sceil.ms:
                stats['ceil_ms'] = round(float(np.mean(sceil.ms)), 3)
                stats['ceil_ms_max'] = round(sceil.ms_max, 3)
            written = out.neutral() if dry_run else out.write(*out_us)
            if written and keep_on():       # ESC に届いた値(人側・dry run は分からない。fix6 safety r1)
                if ai_frame:
                    esc_track.write(clock.now_ns(), out_us[1])
                else:
                    esc_track.unknown(clock.now_ns())
            stats['i2c_errors'] = out.errors
            if not written:
                hold('PCA9685への書き込みに失敗しました(読み直します)', cam, bool(veh['select']), infer_ms)
                applied = False
                continue
            # 画面の数(出力を変えたフレーム・ブレーキの始まり)は、書けたフレームだけ数える(記録の cap_applied・
            # cap_brake_start と同じ。書けなかったフレームは中立で止めた行として記録される)
            if zone_applied:
                stats['zone_applied'] += 1
            if zout is not None and zout.brake_start:
                stats['zone_brakes'] += 1
            if ceil_applied:
                stats['ceil_applied'] += 1
                stats['ceil_neutral'] += int(cout.mode == safety_caps.CEIL_NEUTRAL)
            # 自動合わせ: このフレームを周の数へ(R1 の起点で周を閉じて m_fb を直す)。天井の失敗後は外す
            ci = sceil.inputs if cout is not None else None
            sauto.after(cam.t_ns, ci[0] if ci else np.nan, ai_frame, kind, g_applied,
                        bool(cout is not None and cout.active and cout.mode >= safety_caps.CEIL_LIMIT),
                        ref[1] - out_us[1], veh['erpm'], ci[2] if ci else np.nan, ci[3] if ci else np.nan,
                        imu_buf, zcaps, lap_marks, gain=g_used)
            speed_failed()
            if sauto.ms:
                stats['auto_ms'] = round(float(np.mean(sauto.ms)), 3)
                stats['auto_ms_max'] = round(sauto.ms_max, 3)
            if sauto.event != speed_auto.EV_NONE or stats['auto_laps'] != sauto.fb.laps_used:
                stats['auto_laps'] = sauto.fb.laps_used
                st['auto_end'] = sauto.end(auto_on, p['throttle_gain'])
            # リカバリー中はモデルの出力を車に出していないので、次のフレームの「今の操作値」は中立にする
            prev_pred, prev_cam_t = pred, cam.t_ns
            applied = not dry_run and kind in (assist_mod.ASSIST_NONE, assist_mod.ASSIST_ESC)
            latency = (clock.now_ns() - cam.t_ns) / 1e6
            now_m = time.monotonic()
            times.append(now_m)
            lat_hist.append(latency)
            inf_hist.append(infer_ms)
            stats['frames'] += 1
            stats['fps'] = round((len(times) - 1) / max(times[-1] - times[0], 1e-6), 1) if len(times) > 1 else 0.0
            stats['latency_ms'] = round(float(np.mean(lat_hist)), 1)
            stats['latency_ms_max'] = round(float(np.max(lat_hist)), 1)
            stats['infer_ms'] = round(float(np.mean(inf_hist)), 2)
            st['hold_reason'] = None
            st['hold_since'] = None
            st['now'] = {'raw': [round(raw[0], 1), round(raw[1], 1)], 'out': [round(out_us[0], 1), round(out_us[1], 1)],
                         'select': bool(veh['select']), 'hold': False, 'erpm': int(veh['erpm']),
                         'human': [int(veh['pwm'][0]), int(veh['pwm'][1])] if veh['valid'][0] and veh['valid'][1] else None,
                         'pred': pred.round(1).tolist(), 'assist': assist_mod.ASSIST_NAMES[kind],
                         'ttc': None if not np.isfinite(g_ttc) else round(min(g_ttc, 9.9), 2),
                         'guard': GUARD_NAMES[g_code] + ('' if g_applied or not g_code else '(記録だけ)'),
                         'model_out': [round(model_us[0], 1), round(model_us[1], 1)],
                         'post': post_view(pinfo, v_now), 'ens_j': int(p.get('ensemble_j', 0)),
                         'zone': zcaps.view(zout, zone_on, zone_applied),
                         'ceil': sceil.view(cout, ceil_on, ceil_applied),
                         # OFF のときは「ON なら掛けたはず」のゲイン(影)。ON は実際に掛けたゲイン(fix.md)
                         'auto': sauto.view(auto_on, g_used if auto_on else sauto.last_gain)}
            self.history.append((now_m, raw[0], raw[1], out_us[0], out_us[1], bool(veh['select'])))
            self.recorder.add_ai(cam.t_ns, {
                'pred': pred.astype(np.float32), 'raw': raw, 'out': out_us,
                'gains': (p['steer_gain'], g_used), 'trim': p['steer_trim_us'], 'lookahead': p['lookahead'],
                'mode_id': mode_id, 'hold': False, 'select': bool(veh['select']), 'latency_ms': latency,
                'infer_ms': infer_ms, 'dry_run': dry_run, 'assist': kind, 'model_out': model_us,
                'ttc': g_ttc, 'guard': g_code, 'guard_applied': g_applied,
                'model_pred': np.asarray(model_pred, np.float32), 'post_mode': int(pinfo.get('mode', -1)) if pinfo else -1,
                'v_hat': float(pinfo.get('v_hat', np.nan)) if pinfo else np.nan,
                'v_cmd': float(pinfo.get('v_cmd', np.nan)) if pinfo else np.nan,
                'ens_j': int(p.get('ensemble_j', 0)), **zcaps.record(zout, zone_on, zone_applied, cap_in),
                **sceil.record(cout, ceil_on, ceil_applied, ceil_in),
                **sauto.record(int(p.get('speed_auto', 0) or 0) == 1, g_used, p['throttle_gain']),
                'launch_on': bool(launch_on), 'launch_end': int(launch_end)})
        # 発進の前進の途中で走行を止めた(⑧)
        helper.boost.stop()
        launch_note()
        # 走り終わりの自動合わせの目安(throttle_gain を変えるのは人)
        pe = self.params
        st['auto_end'] = sauto.end(int(pe.get('speed_ceiling', 0) or 0) == 1 and int(pe.get('speed_auto', 0) or 0) == 1,
                                   pe['throttle_gain'])

    # ------------------------------------------------------------ 状態
    def status(self):
        st = dict(self.st)
        if st['phase'] in ('starting', 'driving') and not self.running:
            st['phase'] = 'error'
            st['error'] = st.get('error') or '走行スレッドが止まりました'
        st['running'] = self.running
        st['params'] = dict(self.params)
        st['last'] = dict(self.last)
        st['boot_at'] = getattr(self, 'boot_at', None)      # 起動の時刻(r1-state3-4。画面の知らせだけ)
        st['recording'] = self.recorder.recording
        st['elapsed_s'] = int((st.get('finished') or time.time()) - st['started']) if st.get('started') else 0
        st['heartbeat_age_s'] = round(time.monotonic() - self.heartbeat_t, 1) if self.running else None
        now = time.monotonic()
        h = list(self.history)[::3]
        st['history'] = {'t': [round(r[0] - now, 2) for r in h], 'raw_steer': [round(r[1]) for r in h],
                         'raw_throttle': [round(r[2]) for r in h], 'out_steer': [round(r[3]) for r in h],
                         'out_throttle': [round(r[4]) for r in h], 'select': [int(r[5]) for r in h]}
        st['limits'] = {k: list(v) for k, v in PARAM_LIMITS.items()}
        st['zone_caps_spec'] = ZONE_CAPS_SPEC
        st['speed_ctrl_spec'] = speed_ctrl_spec(self.params)
        st['launch_spec'] = launch_spec(self.params)
        st['pca9685'] = {'address': config.PCA9685['address'], 'freq_hz': config.PCA9685['freq_hz'],
                         'steering_ch': config.PCA9685['steering_ch'], 'throttle_ch': config.PCA9685['throttle_ch']}
        return st
