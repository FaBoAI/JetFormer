"""protocol_line_r4.json(経路いろいろ第 4 回の追補の決まり)を書いて固める(2026-10-14、line r4 の段。学習・検証・試験の前)。
    python3 r3/line4/tools/write_protocol_line_r4.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
E = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
L4 = E / "r3/line4"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731

SHIELD_PARAMS = {
    "H": 24, "c_nominal": 0.04, "c_pessimistic": -1,
    "pessimistic": {"steerRate": 0.8, "delay": 1, "brakeDeceleration": 0.7, "frictionMultiplier": 0.85},
    "identify": {"enabled": False}, "governor": {"enabled": False}, "worst": {"enabled": False}, "unstick": {"enabled": False},
    "backups": [{"la": 0.6, "k": 0.2}, {"la": 0.6, "k": 0}, {"hold": True}], "keep_nominal": True,
    "pess_fc": False, "pess_speedGain": 1, "steer_set": [-0.52, -0.26, 0, 0.26, 0.52],
    "stop": {"enabled": True, "E": True, "Q": False, "H_max": 240, "v_stop": 0.02},
    "select": "v5", "cont_candidates": True,
    "escape": {"enabled": True, "after": 3, "v_still": 0.1, "fwd_steers": [-0.52, -0.39, -0.26, -0.13, 0, 0.13, 0.26, 0.39, 0.52], "fwd_speed": 0.5,
               "fwd_frames": 30, "turn_frames": [0, 8], "rev_steers": [-0.52, -0.26, 0, 0.26, 0.52], "rev_speed": -0.4, "rev_frames": [12, 24],
               "min_progress": 0.1, "retry_every": 15},
}
code_files = ["eval_line_r4.mjs", "episode_ln4.mjs", "gate_miss.mjs", "variation_metrics.mjs"] + \
    [f"rt4/{p.name}" for p in sorted((L4 / "rt4").iterdir())] + [f"train/{p.name}" for p in sorted((L4 / "train").iterdir()) if p.is_file()]
P = {
    "title": "経路いろいろ(line)第 4 回の追補の決まり — protocol.json(sha256 21dba877…09f7e)・protocol_line_r2.json(82de57e2…3c61)・protocol_line_r3.json(25c853d7…b170)を変えずに、line だけに足す・置き換える項目",
    "version": "line-r4-1",
    "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "written_by": "line r4 の段(sim_rl_modes_20261013。modes の作業 wf_7e500cf6-ccd とは別。学習・検証・試験の前に固定。シールドの値は下調べの組(確認 r3 の崩れた 5 回)だけを見て決めた)",
    "base_protocol_sha256": sha(REP / "protocol.json"),
    "base_protocol_line_r2_sha256": sha(REP / "protocol_line_r2.json"),
    "protocol_line_r3_sha256": sha(REP / "r3/protocol_line_r3.json"),
    "base_modes_json_sha256": sha(REP / "modes.json"),
    "request": "シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども作成できるように。強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意(ユーザー、2026-10-13 21:20)。2026-10-14 08:30「作業を引き続き実施」",
    "why": [
        "確認 line r3(check_line_r3.md)の不合格: 試験の組(186 回)は全部合格だったが、開始の分布の角(ゲート 16・17、左右 −0.2 m・向き +12〜+15°、各ゲートの開始の 1%)で、(1) シールドがあっても衝突(GR-2、2.78 秒に柵へ 3.14 m/s。13 フレームで合格する候補が無く、方策の指令をそのまま通した = proposal_best)、(2) 有効な回で柱の足から 2.13 cm(C129)・格子で 2.12〜3.00 cm、(3) シールドが速さ 0 だけを通して止まったまま(stopped_turn_to_backup 約 500 フレーム、ゲート 17 のちょうど角で 16/16、箱で 15/80)。一様に引いた開始の約 0.0072%、大きく横を向いた開始の約 0.10%。シールドを外すとゲート 17 は柵へ衝突、ゲート 16 は 3 cm 未満(方策の弱さ)",
        "確認 line r3 の昇格の段への指摘: line のシールドを共有の mode_runtime.mjs に mode.json の runtime.shield・shield_params で入れる。教師の metadata にシールドの値が入っていない(1,275 回・F-ID 2 つとも)。記録の前の確かめ(valid_episode_r3・3 cm・シールド ≤ 10%)を保つ",
        "r4 の下調べ(学習しない。r3/line4/probe/feas_corner.mjs と、確認 r3 の崩れた 5 回 DEV-FAIL5): (a) ゲート 17 の角(−0.2 m・+15°)の開始から、前へだけで 50 cm 以上空けたまま 1.8 m 進む操作の列がある(止まったのは方策が逆へ回り込むため = 方策の弱さ)。(b) r3 のシールド(v4 の動き = v5 のファイルに {stop 切・select v4・cont_candidates 切・escape 切})で 5 回とも確認 r3 と同じ崩れ(衝突 1・2.12 cm・止まり 2)。(c) シールド v5(止まれることを合格の条件・合格する候補が無いときはブレーキを先に)で衝突と 3 cm 未満は消えた(5 回の最小 4.01 cm)が、止まり 2 は残った。(d) v5 + 抜け出し(escape)で止まりは消えたが、後ろへ下がるフレームが入り、1 回は有効な周が無かった。→ シールド v5 を安全網に、学び直しで角の開始そのものを直す",
        "r3 の試験の組(LN-P60-R3・LN-HEAD60-R3・LN-EXT60-R3)は使ったので、r4 は全部新しい組で試験する(開始・種のどれも r1〜r3 の試験・検証・下調べ・報告、確認 r1〜r3、審査の段、no_wall r4 の組と重ならない)。角の組(LN-CORNER-R4)と崩れた箱の組(LN-BOX120-R4)を判定に足す",
    ],
    "unchanged": [
        "判定のしきい値と式: L1 ≥ 57/60(P60・HEAD60・EXT60)、L2 ≥ 0.03 m、L3 中央値 ≤ 0.05 m・95% 点 ≤ 0.12 m(開始から 3 m 以降)、L4(左右 ≥ 0.20・イン/アウト ≥ 0.10・どの 2 つも 1 m の移動平均の最大 ≥ 0.08 m)、L5 VM3 ≥ 0.08 m、L6 LN-FAM6 の c0 の 1 周目 ≤ 10.0 秒、L7 シールドの作動(P60 全体 ≤ 2%・P60 と FAM6 の回ごと ≤ 10%)、L8 衝突 0",
        "目標の線 v2・経路の形の家族と強さ・family_plan_15・走行中の c_v U(0.9, 1.0)・速さの上限 4.5·c_v・観測 cond97",
        "開始の分布(ゲート U{0..32}・左右 ±0.2 m・向き ±15°・開始のすき間 ≥ 0.12 m)。角も大きく横を向いた開始も除かない",
        "定義(valid_episode・clearance・lap_from_stop・tracking_error、protocol_line_r3 の stall・valid_episode_r3・large_heading_start)と判定のコア(元の sim-core.js、sha256 aaeb2379…549e)、LN-FAM6",
        "シールドが替えた指令は正解ラベル(l = 1、sh = 1)",
    ],
    "definitions_added": {
        "corner_box": "開始の分布の角の箱: 左右の符号 so ∈ {−, +}・向きの符号 sy ∈ {−, +} の角で、|左右| ∈ [0.16, 0.20] m・|向き| ∈ [12, 15]°(各ゲートの開始の面積の 1%)。開始のすき間 ≥ 0.12 m になるまで同じ箱の中で引き直す(上限 60 回。引けない角は飛ばして記録)",
        "failed_box_16_17": "確認 r3 で崩れた箱: ゲート 16・17 の左右 U(−0.20, −0.16) m・向き U(+12, +15)°(角の箱の 1 つ、so = −・sy = +)。開始のすき間 ≥ 0.12 m まで引き直す",
        "reverse_frames": "sim の速さ < −0.01 m/s のフレーム(シールドの抜け出しで後ろへ動いた)",
        "escape": "シールドの fallback が escape_fwd・escape_rev のフレーム",
    },
    "runtime": {
        "shield": {
            "version": "jf-shield/v5",
            "file": "experiments/20261013_sim_rl_modes/r3/line4/rt4/shield.mjs(no_wall r4 の r3/nw4/runtime_v5/shield.mjs の写し。sha256 が protocol_no_wall_r4.json runtime.shield.file_sha256_at_freeze と同じ)",
            "file_sha256_at_freeze": sha(L4 / "rt4/shield.mjs"),
            "mode_runtime_file": "experiments/20261013_sim_rl_modes/r3/line4/rt4/mode_runtime.mjs(共有の web/rl/mode_runtime.mjs 50de50e7… に、mode.json の runtime でシールドを切り替える modeRuntime と、metadata.shield = 実際に使った値の全部 を足したもの)",
            "mode_runtime_sha256_at_freeze": sha(L4 / "rt4/mode_runtime.mjs"),
            "mode_runtime": {"shield": True, "shield_params": SHIELD_PARAMS},
            "what": [
                "標準の物理の模型だけ(悲観の模型はしきい値 −1 で常に合格、見積もり・速さの見張り・模型 Q・すくみをほどくは切)で、すき間 4 cm を守る(r3 と同じ)",
                "stop(v5): 提案・候補の先読みを、0.8 秒(24 フレーム)のあとも予備の操縦(最大すき間の線の pure pursuit + ブレーキ)で止まるまで続け(上限 240 フレーム)、止まるまで全部ですき間 ≥ 4 cm のときだけ合格。r3(v4)は 0.8 秒先でまだ速く走っていても合格にしていたので、角の開始で合格する候補が無くなり(GR-2)提案を通していた",
                "select 'v5': 提案が合格しないとき、合格する候補(提案に近い順)。どれも合格しなければ、予想のよい順・同じならブレーキ(max_e)。「同じ点数なら提案を残す」(proposal_best)をやめる",
                "cont_candidates: 予備の操縦の今の舵 + ブレーキ・前のフレームの舵を保ってブレーキも候補に",
                "escape: 止まっていて(|v| < 0.1 m/s)シールドが速さ 0 を返し、方策は進みたいフレームが 3 つ続いたら、決まった操縦の表(その場で舵を合わせて前へ・後ろへ下がって前へ)を止まるまで先読みし、4 cm を守って R の上で 0.10 m 以上進むもののうち一番進むものを、1 フレームずつ確かめ直しながら行う(前へだけの操縦があればそれを先に)。後ろへ下がるフレームは速さの指令が負(sh = 1)",
            ],
            "labels": "シールドが替えた指令は正解ラベル(l = 1、sh = 1)。ただし後ろへ動いたフレームのあるシーンは記録の前の確かめで替える(recording.prefilter)",
            "teacher_metadata": "教師の metadata.shield に、shield.mjs が既定と合わせた値の全部(version を含む)を入れる",
            "browser": "Node とブラウザで同じファイル。共有の rl/runtime・web/rl への取り込み(shield.mjs を v5 に・mode_runtime.mjs に modeRuntime・teacher.js が modeRuntime: entry.runtime を渡す)、F-ID、1 フレームの時間は昇格の段",
        },
        "start_alignment": "使わない(r3 の下調べで悪化)",
    },
    "sets": {
        "test": {
            "name": "LN-P60-R4 + LN-HEAD60-R4 + LN-EXT60-R4 + LN-CORNER-R4 + LN-BOX120-R4 + LN-FAM6(合計 422 回)",
            "LN-P60-R4": {"halton_from": 90001, "count": 60, "seed_base": 8430000, "rule": "Halton 90,001 番から(protocol seeds.halton の写し方。これまでの組の 1〜29,999・40,001〜・50,001〜・60,001〜・70,001〜 の外)、開始のすき間 ≥ 0.12 m を満たさない番号は飛ばして 60 個(固める前に数えた: 飛ばし 2、最後の番号 90,062)。回 i の種 8,430,000 + i。形は family_plan_15 の順を 4 回、強さ・c0・random の種・c_v は createRng(mix32(種, 0x11e))(r2・r3 と同じ作り方)"},
            "LN-HEAD60-R4": {"rng_seed": "0x4ead0011", "shape_seed": "0x4ead0012", "count": 60, "seed_base": 8431000, "rule": "mulberry32(0x4ead0011) で ゲート ⌊33u⌋・左右 −0.2 + 0.4u・向き −15 + 30u の順に引き、開始のすき間 ≥ 0.12 m かつ large_heading_start のものだけ 60 個。形は family_plan_15 の順、強さ・c_v U(0.9, 1.0) は mulberry32(0x4ead0012)"},
            "LN-EXT60-R4": {"rng_seed": "0xe7e70011", "shape_seed": "0xe7e70012", "count": 60, "seed_base": 8432000, "rule": "mulberry32(0xe7e70011) の新しい開始、形の強さを上限(const c0 ±0.9・inside/outside/late_apex/early_apex a 0.9)の 6 つ + random(種は mulberry32(0xe7e70012))を i mod 10 で、c_v は i が奇数なら 1.0・偶数なら 0.9(r3 と同じ割り当て)"},
            "LN-CORNER-R4": {"rng_seed": "0xc0e40001", "shape_seed": "0xc0e40002", "count": 116, "seed_base": 8433000, "rule": "33 ゲート × 4 つの角(順に (−,+)・(−,−)・(+,+)・(+,−))の corner_box を mulberry32(0xc0e40001) で 1 つずつ(|左右|・|向き| の順に引く)。開始のすき間 ≥ 0.12 m の点が 60 回で引けない角は飛ばす(固める前に数えた: 16 の角を飛ばして 116 回)。形は family_plan_15 の順、強さ・c_v U(0.9, 1.0) は mulberry32(0xc0e40002)"},
            "LN-BOX120-R4": {"rng_seed": "0xb0c40001", "shape_seed": "0xb0c40002", "count": 120, "seed_base": 8434000, "rule": "failed_box_16_17 をゲート 16 から 60 回・ゲート 17 から 60 回、mulberry32(0xb0c40001) で(左右・向きの順)。形は family_plan_15 の順、強さ・c_v は mulberry32(0xb0c40002)。確認 r3 の BOX16・BOX17(0x6c6e3312・0x6c6e3313)とは別の種"},
            "LN-FAM6": "protocol.json のまま(STD の開始・c_v 1.0・連続 20 秒の 6 つの形。定義の組)",
            "episode": "連続 20 秒(600 フレーム)、元の sim-core.js、標準の物理、シールドあり(runtime.shield.mode_runtime を教師の modeRuntime に)",
            "code": "r3/line4/eval_line_r4.mjs --set test(buildJobs、固定のときの sha256 は files.code_sha256_at_freeze)",
        },
        "validation": {
            "name": "LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 + LN-VBOX40-R4 + LN-FAM6(合計 203 回)",
            "rule": "LN-V60-R4: Halton 95,001 番から 60 個(飛ばし 4)、種 8,530,000 + i、形・c_v は試験と同じ作り方。LN-VHEAD40-R4: mulberry32(0x4ead0013)・形 0x4ead0014、large_heading_start だけ 40 個、種 8,531,000 + i。LN-VCORNER-R4: 33 ゲートごとに 4 つの角から mulberry32(0xc0e40003) で 2 つを選び corner_box を 1 つずつ(引けない角は飛ばす: 57 回)、形 0xc0e40004、種 8,533,000 + i。LN-VBOX40-R4: failed_box_16_17 をゲート 16・17 から 20 回ずつ、mulberry32(0xb0c40003)・形 0xb0c40004、種 8,534,000 + i。高速版(train/fast_core_modes.mjs)でよい。チェックポイントを選ぶためだけに使う",
            "code": "r3/line4/eval_line_r4.mjs --set validation",
        },
        "fresh_report": {"name": "LN-F300-R4(報告だけ)", "rule": "mulberry32(0x5eed4c01) の一様な新しい開始 300 回(開始のすき間 ≥ 0.12 m)、形 0x5eed4c02、種 8,630,000 + i。選んだ方策を試験の組のあとに 1 回だけ走らせ、60 回ずつ 5 つの L1・L2・L3・L5 と止まり・衝突・後ろへ動いた回の数を報告する"},
        "regression_reported": "確認 r3 の崩れた 5 回(GR-2・C129・C137・GR-7・BOX17-44)・ちょうど角の 32 回(NBC)・BOX16 と BOX17 の 160 回(確認 r3 と同じ作り方)を、選んだ方策 + シールドで走らせて報告する(下調べで一部を見たので判定には使わない)",
        "fastest_reported": "最速の方策(bundle/fastest、シールドなし)で LN-P60-R4 の 60 の開始(線の違いの報告)",
        "dev": "下調べの組(DEV-FAIL5・DEV-NBC32・DEV-BOX160 = 確認 r3 の組と、新しい種の DEV-H60(Halton 98,001〜、種 8,930,000 +)・DEV-HEAD40(0xde740001)・DEV-CORNER(0xde740003))。シールドの値と学び直しの要否を見るのに使う。選び・試験には使わない",
    },
    "acceptance": {
        "L1_r4": "LN-P60-R4・LN-HEAD60-R4・LN-EXT60-R4 のそれぞれ 57/60 以上、LN-CORNER-R4 の ⌈0.95·116⌉ = 111 以上、LN-BOX120-R4 の ⌈0.95·120⌉ = 114 以上が valid_episode_r3(検証の組も同じ割合: VHEAD40 は 38、VCORNER 57 は 55、VBOX40 は 38)",
        "L2_r4": "全部の組の valid_episode_r3 の回と LN-FAM6 の全部で、最小すき間 ≥ 0.03 m",
        "L3_L5": "式・しきい値は protocol.json のまま。組は LN-P60-R4 の valid_episode_r3 の回(目標の線 v2)",
        "L4_L6": "protocol.json のまま(LN-FAM6、全部 valid_episode_r3 であること)",
        "L7_shield": "LN-P60-R4 の shield_fraction ≤ 0.02、かつ LN-P60-R4・LN-FAM6 のどの回も shield_fraction ≤ 0.10(r3 と同じ)",
        "L8_no_collision": "全部の組(422 回)のどの回も衝突 0",
        "L9_no_stall_new": "全部の組のどの回も止まり(protocol_line_r3 definitions_added.stall)0。確認 r3 の must-fix「止まりを学び直しで消す」を判定にした(きびしくする向き)",
        "L10_no_reverse_new": "LN-P60-R4 と LN-FAM6 のどの回も reverse_frames 0(抜け出しが記録の分布の普通の回で働かないこと。ほかの組は報告)",
        "reported": [
            "組ごとの有効・衝突・止まり・最小すき間・3 cm 未満・シールドの作動(> 5%・> 10% の回)・後ろへ動いた回・抜け出しの回",
            "形ごとの追従の誤差・1 周の時間、12 cm を越える追従の誤差のフレームが R の上のどこにあるか",
            "large_heading_start の回の最小すき間、シールドの替え方の内訳",
            "VM1・VM2・VM4・VM7(参考)、LN-F300-R4 の 60 回ずつ 5 つの値、確認 r3 の崩れた組(regression_reported)",
            "最速との線の違い(LN-P60-R4 の同じ 60 の開始)",
            "シールドなしの同じ試験の組(比べる値。判定のあと)",
        ],
        "thresholds_relaxed": "無し(全部そのまま。足した組・L9・L10 はきびしくする向き)",
    },
    "training": {
        "run": "line_r4(r3/line4/train/train_line_r4.py --run line_r4。観測 cond97・目標の線 v2・報酬は r3 のまま)",
        "init": "line r3 の選んだ方策 r3/line_r3/train/main/checkpoints/step_005013504.pt(方策 9f10c7b7…de52d)",
        "seed": 46,
        "steps_max": 5000000,
        "checkpoints": "100 万ステップごと(policies/step_N.json)",
        "early_stop": "200 万ステップ以降、続けて 2 つのチェックポイントが検証の組で L1_r4〜L10 を全部満たしたら止めてよい",
        "ppo": "lr 4e-5 → 2e-5、4 ワーカー × 16 環境、rollout 256、log σ は初期化元のまま",
        "start_mix": [
            {"p": 0.35, "name": "record", "gate": "U{0..32}", "left_offset_m": [-0.2, 0.2], "left_yaw_deg": [-15, 15], "min_initial_clearance_m": 0.12},
            {"p": 0.20, "name": "large_heading", "what": "同じ分布から棄却法、|Δψ| ≥ 0.8 rad", "min_initial_clearance_m": 0.12},
            {"p": 0.15, "name": "corner_box", "gate": "U{0..32}", "corner": "4 つの角を等しく、|左右| U(0.13, 0.22) m・|向き| U(9, 17)°", "min_initial_clearance_m": 0.10},
            {"p": 0.15, "name": "hot_corner", "gate": "U{15..18}", "left_offset_m": [-0.23, -0.10], "left_yaw_deg": [6, 18], "min_initial_clearance_m": 0.10},
            {"p": 0.15, "name": "wide", "gate": "U{0..32}", "left_offset_m": [-0.25, 0.25], "left_yaw_deg": [-20, 20], "min_initial_clearance_m": 0.10},
        ],
        "start_mix_note": "角の成分は開始のすき間が足りなければ同じ成分の中で引き直す(上限 40 回)。試験の角の箱(|左右| 0.16〜0.20・|向き| 12〜15°)を含むが、試験の開始の点そのものは使わない(学習の乱数は種 46 の環境の乱数)",
        "conditioning": "protocol_line_r2.json training.conditioning のまま(c_v は 80% U(0.9, 1.0)・20% U(0.6, 0.9))",
        "reward": "protocol_line_r3.json training.reward のまま(すき間の壁 7 cm・重み 0.3、止まりの見張り)",
        "shield_in_training": "使わない(計算が重い)。検証・試験・記録はシールドあり",
        "physics": "標準の物理だけ",
        "lock": "jf-heavy --mem 2G --lock rl -- nice -n 15(チェックポイント・続きから再開できること)",
    },
    "selection": {
        "rule": "検証の組(シールドあり)で L1_r4〜L10 を全部満たすチェックポイントのうち、むずかしい組(LN-VHEAD40-R4・LN-VCORNER-R4・LN-VBOX40-R4)の最小すき間(衝突・止まりの回は 0)が一番大きいもの。0.1 mm に丸めて同じなら L3 の 95% 点が小さいもの。全部を満たすものが無ければ、満たす数が一番多く、同じならむずかしい組の失敗(衝突・止まり・有効な回の 3 cm 未満)の数が少ないもの",
        "candidates": "学習の 100 万ごとのチェックポイントと、初期化元(step_000000000 = r3 の方策 9f10c7b7… + シールド v5)",
        "test_runs": "試験の組は選んだ 1 つにだけ 1 回使う。試験の結果を見て選び直さない",
    },
    "fallback": "選んだ方策が不合格なら、そのまま不合格として報告する(candidates/line_r4/ に status not_published)。代わりの方法(ベストラップの 7 本の線の混ぜ合わせ)は r1〜r3 とも VM3 で不合格なので走らせない",
    "recording": {
        "prefilter": "記録の前の予想(rl/predict_scenes.mjs の r3 版)で、valid_episode_r3 でない・最小すき間 < 0.03 m・shield_fraction > 0.10・reverse_frames > 0 のシーンを次の候補に替える(r3 に「後ろへ動いた」を足した)",
        "mode_json_overrides": "合格したときだけ: runtime.shield = true・runtime.shield_params = runtime.shield.mode_runtime.shield_params(全部の鍵)・recording.prefilter_min_clearance_m 0.03・prefilter_min_valid_laps 1・prefilter_no_stall true・prefilter_max_shield_fraction 0.10・prefilter_no_reverse true",
    },
    "publication": {
        "r4": "合格なら candidates/line_r4/(status accepted。昇格の段が bundle/line/ に置く)、不合格なら candidates/line_r4/(status not_published)。r1〜r3 の候補はそのまま。bundle/ はこの段では変えない",
        "promote_step": "昇格の段は、modes の作業(wf_7e500cf6-ccd)が終わってから共有のファイル(rl/runtime・web/rl・bundle・teacher.js)を変える",
    },
    "deviations": [
        "第 4 回の試験そのもの: protocol selection.test_runs(1 モード 2 回まで)と allowed_tuning.retry(1 回)の外(r3 と同じ扱い)。r3 の試験の組は使ったので、r4 は全部新しい組で 1 回だけ試験する",
        "RL の予算: 合計は protocol budget.hard_cap をすでに越えている(r3 までの記録のとおり)。r4 は最大 500 万ステップ",
        "シールドを v5(no_wall r4 と同じファイル)にし、値を line 用(標準の物理の模型だけ、4 cm)にした。止まり・後ろへ動いた回の扱いを判定(L9・L10)と記録の前の確かめに足した",
    ],
    "files": {
        "code_dir": "experiments/20261013_sim_rl_modes/r3/line4/",
        "sources_sha256": "r3/line4/sources_sha256_line_r4.txt",
        "code_sha256_at_freeze": {f: sha(L4 / f) for f in code_files},
        "dev_results": "jetformer_reports/sim_rl_modes_20261013/r3/line_r4/dev/",
        "report": "jetformer_reports/sim_rl_modes_20261013/r3/train_line_r4.md",
    },
}
out = REP / "r3/protocol_line_r4.json"
if out.exists():
    raise SystemExit("protocol_line_r4.json はもう固めてあります(書き直さない)")
out.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n")
(REP / "r3/protocol_line_r4.sha256").write_text(f"{sha(out)}  protocol_line_r4.json\n")
print(sha(out))
