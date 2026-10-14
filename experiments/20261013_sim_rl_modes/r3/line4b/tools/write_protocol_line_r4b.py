"""protocol_line_r4b.json(経路いろいろ第 4 回の続き r4b の追補の決まり)を書いて固める(2026-10-14 夜、line r4 の段のやり直し。
r4b の学習・検証・試験の前)。protocol_line_r4.json(15:52 に固定)は変えない。
    python3 r3/line4b/tools/write_protocol_line_r4b.py
下調べ(dev4b)の結果のうち、固める時点でそろっているものの集計を dev_seen_before_freeze に書く(見たものを残すため)。
"""
import glob
import hashlib
import json
from datetime import datetime
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
E = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
L = E / "r3/line4b"
DEV = REP / "r3/line_r4b/dev"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731

P4 = json.loads((REP / "r3/protocol_line_r4.json").read_text())
SHIELD_PARAMS = json.loads(json.dumps(P4["runtime"]["shield"]["mode_runtime"]["shield_params"]))
assert SHIELD_PARAMS["select"] == "v5"
SHIELD_PARAMS["select"] = "v5b"
R4_POL = REP / "r3/line_r4/train/main/policies"
R4_CK = REP / "r3/line_r4/train/main/checkpoints"

code_files = ["eval_line_r4b.mjs", "episode_ln4.mjs", "gate_miss.mjs", "variation_metrics.mjs", "r3b_fail_cases.json", "select_line_r4b.py", "test_shield_v5b.mjs", "dev/run_dev4b.sh"] + \
    [f"rt4b/{p.name}" for p in sorted((L / "rt4b").iterdir())] + [f"train/{p.name}" for p in sorted((L / "train").iterdir()) if p.is_file()]


def dev_seen():
    out = {}
    for f in sorted(glob.glob(str(DEV / "*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        if "reported" not in d:
            continue
        sh = (d.get("shield") or {}).get("mode_runtime", {}) or {}
        out[Path(f).stem] = {"policy_sha256": d["policy"]["sha256"][:12], "policy_step": (d["policy"].get("meta") or {}).get("steps"),
                             "shield_select": (sh.get("shield_params") or {}).get("select") if d.get("shield") else "なし",
                             "by_set": {k: {x: v[x] for x in ("n", "valid_r3", "collisions", "stalls", "min_all", "below_3cm_valid", "escape_runs", "reverse_runs")}
                                        for k, v in d["reported"]["by_set"].items() if v["n"]}}
    return out


P = {
    "title": "経路いろいろ(line)第 4 回の続き(r4b)の追補の決まり — protocol.json(21dba877…)・protocol_line_r2.json(82de57e2…)・protocol_line_r3.json(25c853d7…)・protocol_line_r4.json(f77e844a…)を変えずに、line だけに足す・置き換える項目",
    "version": "line-r4b-1",
    "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "written_by": "line r4 の段のやり直し(2026-10-14 夜。r4 の段は 16:28 の電源断で学習の途中に止まった)。r4b の学習・検証・試験の前に固定",
    "base_protocol_sha256": sha(REP / "protocol.json"),
    "base_protocol_line_r2_sha256": sha(REP / "protocol_line_r2.json"),
    "protocol_line_r3_sha256": sha(REP / "r3/protocol_line_r3.json"),
    "protocol_line_r4_sha256": sha(REP / "r3/protocol_line_r4.json"),
    "base_modes_json_sha256": sha(REP / "modes.json"),
    "request": "シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども作成できるように。強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意(ユーザー、2026-10-13 21:20)。2026-10-14 08:30「作業を引き続き実施」",
    "why": [
        "確認 line r3 の再確認(check_line_r3.md、20:57、pass = false)の must-fix: (1) ゲート 30/31 の角(左右 +0.17〜+0.19 m・向き +13.8〜+14.8°)× 右寄りの形の端(const c0 −0.85〜−0.9・c_v 0.95〜1.0)で、シールドがあっても衝突(G30R-2・EX213-4・EX213-5、柱の足 disk-88)と 1.33 cm(CB264-213)。どれもシールドが合格する候補を見つけられず、方策の指令を通した(proposal_best)。シールドなしでは G30WIDE の 40 回で衝突 10・3 cm 未満 8。学び直しの開始にこの組み合わせを入れ、試験に「ゲート 29〜32 の (+, +) の角の箱 × const c0 −0.9〜−0.6 × c_v 0.9〜1.0」の組を足すこと。r4 の LN-CORNER-R4 は family_plan_15 の形なので約 1% しか当たらない。(2) 合格する候補が無いときに方策の指令を通さず、ブレーキを先に。(3) ゲート 16/17 の箱の止まり(新しい種で 3/80・8/80)を学び直しで消す。(4) 昇格の段の前提: 共有の mode_runtime に line のシールドを mode.json の runtime.shield・shield_params で入れ、試したシールドのファイルを配り、教師の metadata にシールドの値を入れ、F-ID 2 つが Node とブラウザで同じことを確かめる",
        "r4 の状態: protocol_line_r4.json(sha256 f77e844a…、15:52 に固定)。学習(種 46、r3 の方策から)は 16:28 の電源断で 4,014,080 ステップで止まった(チェックポイントは 100 万ごと、検証は 300 万まで)。r4 の試験の組は一度も走らせていない(結果のファイルが無い)。確認の段は r4 の組の定義(protocol_line_r4.json)だけを見た",
        "r4b の下調べ(学習しない。dev4b の組 = 新しい種の DEV-G30R60・DEV-G30W40・DEV-BOX80 と、確認の段の崩れた 16 回 DEV-R3B-FAIL16。選び・試験には使わない): dev_seen_before_freeze を見よ。r3 の方策 + r3 のシールドの動きでは確認の段の 16 回がそのまま崩れる(衝突 3・1.33 cm 2・止まり 11)。r3 の方策 + シールド v5(r4 の値)では衝突と 3 cm 未満は 0(最小 4.00 cm)、ただしゲート 16/17 の 11 回のうち抜け出し(後ろへ動く)11・止まり 1・周なし 2(方策の弱さをシールドの抜け出しが隠している)",
        "→ r4 の決まりを土台に、(a) シールドを v5b(合格する候補が無いときはブレーキの候補だけから選ぶ。v5 の max_e は E の予想が一番よければ提案を通しうる)、(b) 学び直しの開始にゲート 28〜32 の (+, +) 寄り × 右寄りの形と、角の箱 × 形の端を足し、r4 の 4,014,080 から続ける、(c) 試験・検証に G30R の組を足し、(d) 抜け出しに頼る回を判定(L11)で 1% までに限る",
    ],
    "unchanged": [
        "protocol_line_r4.json の判定 L1〜L10 のしきい値と式・組の定義・開始の分布・目標の線 v2・形の家族と強さ・family_plan_15・c_v U(0.9, 1.0)・観測 cond97・判定のコア(元の sim-core.js)・LN-FAM6",
        "r4 の試験の組(LN-P60-R4・LN-HEAD60-R4・LN-EXT60-R4・LN-CORNER-R4・LN-BOX120-R4・LN-FAM6、422 回)はそのまま試験に使う(まだ一度も走らせていない)",
        "シールドの値は select 以外 r4 と同じ(標準の物理の模型だけ・4 cm・止まれることを合格の条件・抜け出しあり)",
        "シールドが替えた指令は正解ラベル(l = 1、sh = 1)。後ろへ動いたシーンは記録の前の確かめで替える",
    ],
    "definitions_added": {
        "g30r_box": "ゲート 29・30・31・32 の (+, +) の角の箱: 左右 U(+0.16, +0.20) m・向き U(+12, +15)°(開始のすき間 ≥ 0.12 m まで同じ箱の中で引き直す、上限 200 回)、形は const c0 U(−0.9, −0.6)(4 桁)、c_v U(0.9, 1.0)(4 桁)。eval_line_r4b.mjs g30rJobs",
        "g30w_box": "g30r_box の広め(報告だけ): 左右 U(+0.10, +0.20) m・向き U(+7.5, +15)°",
        "escape_run": "シールドの fallback が escape_fwd・escape_rev のフレームが 1 つ以上ある回",
        "brake_first": "シールド v5b で、E(見積もった模型 = 標準の物理)に合格する候補が 1 つも無いフレーム。速さ 0 の候補だけから選ぶ",
    },
    "runtime": {
        "shield": {
            "version": "jf-shield/v5(select 'v5b')",
            "file": "experiments/20261013_sim_rl_modes/r3/line4b/rt4b/shield.mjs(r3/line4/rt4/shield.mjs = no_wall r4 の r3/nw4/runtime_v5/shield.mjs 8ece9be4… に select 'v5b' を足したもの。select を渡さなければ v5 と同じ動き)",
            "file_sha256_at_freeze": sha(L / "rt4b/shield.mjs"),
            "mode_runtime_file": "experiments/20261013_sim_rl_modes/r3/line4b/rt4b/mode_runtime.mjs(r4 と同じ中身。共有の web/rl/mode_runtime.mjs 50de50e7… に modeRuntime と metadata.shield を足したもの)",
            "mode_runtime_sha256_at_freeze": sha(L / "rt4b/mode_runtime.mjs"),
            "mode_runtime": {"shield": True, "shield_params": SHIELD_PARAMS},
            "what": [
                "r4(v5)と同じ: 標準の物理の模型だけで 4 cm を守る・止まれることを合格の条件(stop)・続きの候補・抜け出し(escape)",
                "v5b(新): E に合格する候補が 1 つも無いフレームでは、ブレーキ(速さ 0)の候補だけから、E の予想(しきい値を割るまでが長い・最小すき間が大きい)、予備の操縦の舵に近い順に選ぶ('brake_first')。速さが正の方策の提案は通さない(確認の指摘 2)",
            ],
            "labels": "シールドが替えた指令は正解ラベル(l = 1、sh = 1)。後ろへ動いたフレームのあるシーンは記録の前の確かめで替える",
            "teacher_metadata": "教師の metadata.shield に、shield.mjs が既定と合わせた値の全部(version・select を含む)を入れる",
            "browser": "この段で: 候補の走行中の仕組み(rt4b)を、ユーザーと同じ非ループバックのページ(http://192.168.55.1:8080 の /sim/c/default/)の中で読み込み、F-ID 2 つを Firefox で 1 フレームずつ再生して Node と差 0 を確かめる(共有のファイルは変えない)。共有の rl/runtime・web/rl・teacher.js への取り込みと、取り込んだあとの F-ID の確かめは昇格の段",
        },
    },
    "sets": {
        "test": {
            "name": "LN-P60-R4 + LN-HEAD60-R4 + LN-EXT60-R4 + LN-CORNER-R4 + LN-BOX120-R4 + LN-G30R120-R4B + LN-FAM6(合計 542 回)",
            "r4_sets": "protocol_line_r4.json sets.test のまま(422 回)",
            "LN-G30R120-R4B": {"rng_seed": "0x4b320001", "shape_seed": "0x4b320002", "per_gate": 30, "gates": [29, 30, 31, 32], "seed_base": 9300000,
                               "rule": "g30r_box をゲートごとに 30 回(ゲート 29 → 32 の順、mulberry32(0x4b320001) で左右・向きの順)。形・c_v は mulberry32(0x4b320002) で c0・c_v の順。回の種 9,300,000 + i"},
            "episode": "連続 20 秒(600 フレーム)、元の sim-core.js、標準の物理、シールド v5b(runtime.shield.mode_runtime を教師の modeRuntime に)",
            "code": "r3/line4b/eval_line_r4b.mjs --set test",
        },
        "validation": {
            "name": "LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 + LN-VBOX40-R4 + LN-VG30R40-R4B + LN-FAM6(合計 243 回)",
            "rule": "r4 の検証の組(protocol_line_r4.json sets.validation)+ LN-VG30R40-R4B: g30r_box をゲートごとに 10 回、mulberry32(0x4b310001)・形 0x4b310002、種 9,200,000 + i。高速版(train/fast_core_modes.mjs)・シールド v5b。チェックポイントを選ぶためだけに使う",
            "code": "r3/line4b/eval_line_r4b.mjs --set validation --core fast",
        },
        "fresh_report": {
            "name": "LN-F300-R4 + LN-FG30W60-R4B(報告だけ)",
            "rule": "LN-F300-R4 は protocol_line_r4.json のまま。LN-FG30W60-R4B: g30w_box をゲートごとに 15 回、mulberry32(0x4b330001)・形 0x4b330002、種 9,400,000 + i。選んだ方策を試験のあとに 1 回だけ",
        },
        "regression_reported": "protocol_line_r4.json の regression(REG-FAIL5・REG-NBC32・REG-BOX160)+ 確認の段の崩れた 16 回(REG-R3B-FAIL16、下調べで見たので判定には使わない)",
        "no_shield_reported": "選んだ方策をシールドなしで試験の組の全部(542 回)と LN-FG30W60-R4B に(方策そのものの弱さの報告。判定のあと)。r3 の方策のシールドなしも LN-G30R120-R4B・LN-BOX120-R4 で比べる",
        "fastest_reported": "最速の方策(bundle/fastest、シールドなし)で LN-P60-R4 の 60 の開始(線の違いの報告)",
        "dev": "dev4b(DEV-G30R60: g30r_box × 15/ゲート、0x4b300001/2、種 9,100,000 +・DEV-G30W40: g30w_box × 10/ゲート、0x4b300003/4、9,101,000 +・DEV-BOX80: failed_box_16_17 × 40/ゲート、0x4b300005/6、9,102,000 +・DEV-R3B-FAIL16)。シールドの値と学び直しの開始の混ぜ方を決めるのに使う。選び・試験には使わない",
        "seeds_note": "r4b の種(回 9,100,000〜9,400,059・乱数 0x4b30xxxx〜0x4b33xxxx)は、r1〜r4 の試験・検証・下調べ・報告、確認 r1〜r3(再確認を含む)、審査の段の種と重ならない",
    },
    "acceptance": {
        "L1_r4b": "LN-P60-R4・LN-HEAD60-R4・LN-EXT60-R4 のそれぞれ 57/60 以上、LN-CORNER-R4 の 111/116 以上、LN-BOX120-R4 の 114/120 以上、LN-G30R120-R4B の 114/120 以上が valid_episode_r3(検証の組も同じ割合: VHEAD40 38・VCORNER 55/57・VBOX40 38・VG30R40 38)",
        "L2_L10": "protocol_line_r4.json のまま。L2・L8・L9 は新しい組を含む全部の組に(L2: 全部の組の valid_episode_r3 の回と LN-FAM6 で最小すき間 ≥ 0.03 m、L8: 衝突 0、L9: 止まり 0)",
        "L11_escape_new": "全部の組(試験 542 回・検証 243 回)で escape_run が ⌊0.01·回数⌋ 以下(試験 5・検証 2)。止まりをシールドの抜け出し(後ろへ動く)で隠さず、方策の学び直しで消したことを確かめる(確認の指摘 3。きびしくする向き)",
        "reported": [
            "protocol_line_r4.json acceptance.reported の全部 + 新しい組の値",
            "brake_first のフレーム数と、その回の最小すき間",
            "シールドなしの同じ試験の組(衝突・3 cm 未満・止まり。方策そのものの弱さ)と、r3 の方策のシールドなしとの比べ(LN-G30R120-R4B・LN-BOX120-R4)",
        ],
        "thresholds_relaxed": "無し(足した組・L11 はきびしくする向き)",
    },
    "training": {
        "run": "line_r4b(r3/line4b/train/train_line_r4b.py --run line_r4b。観測 cond97・目標の線 v2・報酬は r3・r4 のまま)",
        "init": "line r4 の 4,014,080 ステップ(r3/line_r4/train/main/checkpoints/step_004014080.pt、sha256 " + sha(R4_CK / "step_004014080.pt") + "。方策 " + sha(R4_POL / "step_004014080.json") + ")",
        "seed": 47,
        "steps_max": 4000000,
        "checkpoints": "50 万ステップごと(policies/step_N.json)",
        "early_stop": "使わない(400 万まで)",
        "ppo": "lr 3e-5 → 1.5e-5、4 ワーカー × 16 環境、rollout 256、log σ は初期化元のまま",
        "start_mix": [
            {"p": 0.30, "name": "record", "rule": "ゲート U{0..32}・左右 ±0.2 m・向き ±15°・開始のすき間 ≥ 0.12 m。形・c_v はいつもの混ぜ方"},
            {"p": 0.15, "name": "large_heading", "rule": "同じ分布から棄却法、|Δψ| ≥ 0.8 rad、≥ 0.12 m"},
            {"p": 0.12, "name": "corner_box", "rule": "ゲート U{0..32}・4 つの角を等しく、|左右| U(0.13, 0.22) m・|向き| U(9, 17)°、≥ 0.10 m"},
            {"p": 0.13, "name": "hot_corner", "rule": "ゲート U{15..18}・左右 U(−0.23, −0.10) m・向き U(+6, +18)°、≥ 0.10 m"},
            {"p": 0.15, "name": "g30_right", "rule": "ゲート U{28..32}・左右 U(+0.06, +0.23) m・向き U(+4, +18)°、≥ 0.10 m。形 const c0(半分 U(−0.9, −0.8)・半分 U(−0.8, −0.5))、c_v U(0.9, 1.0)"},
            {"p": 0.07, "name": "corner_ext", "rule": "corner_box と同じ開始。形は範囲の端 6 つ(const c0 ±0.9・inside/outside/late_apex/early_apex a 0.9)と random から等しく、c_v U(0.9, 1.0)"},
            {"p": 0.08, "name": "wide", "rule": "ゲート U{0..32}・±0.25 m・±20°、≥ 0.10 m"},
        ],
        "start_mix_note": "角の成分は開始のすき間が足りなければ同じ成分の中で引き直す(上限 40 回)。試験・検証の開始の点そのものは使わない(学習の乱数は種 47 の環境の乱数)。g30_right・corner_ext は引いた形・c_v を元の reset の extra に渡す(元の reset が引く乱数の数は同じ)",
        "conditioning": "g30_right・corner_ext 以外は protocol_line_r2.json training.conditioning のまま(c_v は 80% U(0.9, 1.0)・20% U(0.6, 0.9))",
        "reward": "protocol_line_r3.json training.reward のまま(すき間の壁 7 cm・重み 0.3、止まりの見張り)",
        "shield_in_training": "使わない。検証・試験・記録はシールド v5b あり",
        "physics": "標準の物理だけ",
        "lock": "jf-heavy --mem 2G --lock rl --need-mb 1200 -- nice -n 15(チェックポイント・続きから再開できること)",
        "code": "r3/line4b/train/run_line_r4b.sh",
    },
    "selection": {
        "rule": "検証の組(シールド v5b)で L1_r4b・L2〜L11 を全部満たすチェックポイントのうち、むずかしい組(LN-VHEAD40-R4・LN-VCORNER-R4・LN-VBOX40-R4・LN-VG30R40-R4B)の最小すき間(衝突・止まりの回は 0)が一番大きいもの。0.1 mm に丸めて同じなら escape_run の数が少ないもの、それも同じなら L3 の 95% 点が小さいもの。全部を満たすものが無ければ、満たす数が一番多く、同じならむずかしい組の失敗(衝突・止まり・有効な回の 3 cm 未満)の数が少ないもの",
        "candidates": "r4b の 50 万ごとのチェックポイント(8 つ)と、r4 の 3,014,656・4,014,080(r4b の検証の組・シールド v5b で評価し直す)",
        "test_runs": "試験の組は選んだ 1 つにだけ 1 回使う。試験の結果を見て選び直さない",
        "code": "r3/line4b/select_line_r4b.py",
    },
    "fallback": "選んだ方策が不合格なら、そのまま不合格として報告する(candidates/line_r4/ に status not_published)",
    "recording": {
        "prefilter": "記録の前の予想で、valid_episode_r3 でない・最小すき間 < 0.03 m・shield_fraction > 0.10・reverse_frames > 0 のシーンを次の候補に替える(r4 と同じ)",
        "mode_json_overrides": "合格したときだけ: runtime.shield = true・runtime.shield_params = runtime.shield.mode_runtime.shield_params(全部の鍵)・recording.prefilter_min_clearance_m 0.03・prefilter_min_valid_laps 1・prefilter_no_stall true・prefilter_max_shield_fraction 0.10・prefilter_no_reverse true",
    },
    "publication": {
        "r4b": "合格なら candidates/line_r4/(status accepted。昇格の段が bundle/line/ に置く)、不合格なら candidates/line_r4/(status not_published)。r1〜r3 の候補はそのまま。bundle/ はこの段では変えない",
        "promote_step": "昇格の段が共有のファイル(rl/runtime・web/rl・bundle・teacher.js・sim-rl.js)を変える。候補には、試したシールド・mode_runtime の写しと差分、teacher.js・sim-rl.js の差分の案、F-ID 2 つと Node・ブラウザの確かめの結果を置く",
    },
    "deviations": [
        "第 4 回の続き(r4b)そのもの: protocol selection.test_runs・allowed_tuning.retry の外(r3・r4 と同じ扱い)。試験はこの決まりで 1 回だけ",
        "r4 の学習(protocol_line_r4.json training、500 万まで)は電源断で 4,014,080 で止まった。そこから学び直しの開始の混ぜ方を替えて続ける(r4b、種 47、最大 400 万)。r4 の 300 万・400 万も選びの候補に入れる",
        "シールドを v5b に(select だけ。値は r4 と同じ)",
        "RL の予算: 合計は protocol budget.hard_cap をすでに越えている。r4b は最大 400 万ステップ(r4 の 401 万と合わせて約 801 万)",
    ],
    "dev_seen_before_freeze": None,
    "files": {
        "code_dir": "experiments/20261013_sim_rl_modes/r3/line4b/",
        "sources_sha256": "r3/line4b/sources_sha256_line_r4b.txt",
        "code_sha256_at_freeze": None,
        "dev_results": "jetformer_reports/sim_rl_modes_20261013/r3/line_r4b/dev/",
        "report": "jetformer_reports/sim_rl_modes_20261013/r3/train_line_r4b.md(依頼の名前は train_line_r4.md。同じ中身を両方に置く)",
    },
}
P["dev_seen_before_freeze"] = dev_seen()
P["files"]["code_sha256_at_freeze"] = {f: sha(L / f) for f in code_files}
out = REP / "r3/protocol_line_r4b.json"
if out.exists():
    raise SystemExit("protocol_line_r4b.json はもう固めてあります(書き直さない)")
out.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n")
(REP / "r3/protocol_line_r4b.sha256").write_text(f"{sha(out)}  protocol_line_r4b.json\n")
print(sha(out))
