"""no_wall r4 の追補の決まり protocol_no_wall_r4.json を書き、sha256 を r3/protocol_r4.sha256 に固める(2026-10-14、r4 の段。学習・検証・試験の前に 1 回だけ)。
シールドの値は r3/nw4/runtime_v5/shield.mjs の SHIELD_DEFAULTS を node で読んでそのまま入れる(既定の値 = 決まりの値。test_shield_v5.mjs で照合)。
    python3 write_protocol_nw4.py   (既にあれば止まる)
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
R3 = REP / "r3"
OUT = R3 / "protocol_no_wall_r4.json"
SHA = R3 / "protocol_r4.sha256"
E = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/nw4")
if OUT.exists() or SHA.exists():
    raise SystemExit("protocol_no_wall_r4.json はもう固めてあります(書き直さない)")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
defaults = json.loads(subprocess.run(["node", "-e", "import('%s/runtime_v5/shield.mjs').then(m=>console.log(JSON.stringify(m.SHIELD_DEFAULTS)))" % E],
                                     capture_output=True, text=True, check=True).stdout)
params = {k: v for k, v in defaults.items() if k != "version"}
lowc = json.loads((E / "train/starts_lowc_nw4.json").read_text())
P = {
    "title": "絶対壁にぶつからない(no_wall)第 4 回の追補の決まり — protocol.json(sha256 21dba877…09f7e)と protocol_no_wall_r3.json(73a5a719…)を変えずに、no_wall だけに足す・置き換える項目",
    "version": "no_wall-r4-1",
    "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "written_by": "no_wall r4 の段(sim_rl_modes_20261013。modes の作業 wf_7e500cf6-ccd とは別。学習・検証・試験の前に固定。シールド v5 の値は下調べの組 dev4・dev4e だけを見て決めた)",
    "base_protocol_sha256": sha(REP / "protocol.json"),
    "base_modes_json_sha256": sha(REP / "modes.json"),
    "protocol_no_wall_r3_sha256": sha(R3 / "protocol_no_wall_r3.json"),
    "request": "シミュレーターの走行は、ほかにももっといろいろなパターン、ぶつかるのを避けたり、いろいろな経路を通るパターンなども作成できるように。強化学習モード(最速)・(絶対壁にぶつからない)・(離脱復活)など、データセットのバリエーションが出る構成のモードを用意(ユーザー、2026-10-13 21:20)。2026-10-14 08:30「作業を引き続き実施」",
    "why": [
        "確認 no_wall r3(check_no_wall_r3.md)の不合格: (1) N5_r3 の止まり — ゲート 23 の柱の足へ 11〜15° 向いた開始(開始のすき間 12〜13 cm)で 571 フレーム止まったまま(12〜15 cm の層 300 回で 7、分布全体で約 0.05%)。(2) 標準の物理でシールド v4.1 が衝突を起こした(ゲート 20・左 −0.13 m・−13°、6.12 秒にカーテン。外すと衝突しない)・隣の開始で 5.63 cm。原因は先読み(E 24・Q 45 フレーム)が止まる距離より短いこと、どの候補も Q に合格しないとき提案をそのまま通すこと(proposal_best)。(3) bundle/no_wall/status.json(審査の段が 13:55 に not_accepted に直した)。(4) 画面の「最小すき間 8.5 cm(標準の物理)」は試験の組の最小で、新しい開始 1,000 回では 8.01 cm、まれな開始では 5.63 cm・衝突 1",
        "r3 の試験の組(protocol_no_wall_r3.json sets.test)は r3 で使ったので、r4 は全部新しい組で試験する(開始・物理・種のどれも r3 の試験・検証・下調べ、確認 r2・r3 の組、審査の段の新しい種と重ならない)",
    ],
    "unchanged": [
        "判定のしきい値: N2 ≥ 0.080 m(標準の物理)・N3 ≥ 0.050 m(物理のずれ)・N4 ≤ 12.0 秒・N6 ≤ 0.02・N7 ≤ 0.05(回ごと ≤ 0.25)・DR の組の valid_episode_r3 ≥ 95%",
        "記録の開始の分布(modes.json no_wall.recording.start_distribution: ゲート U{0..32}・左右 ±0.2 m・向き ±15°・開始のすき間 ≥ 0.12 m)",
        "DR-T の範囲・定義(protocol.json definitions と protocol_no_wall_r3.json definitions_added の stall・valid_episode_r3・dr_vertex・dr_hard_corner)・判定のコア(元の sim-core.js、摩擦円の回は高速版の写し)",
        "シールドが替えた指令を正解ラベルにする決まり(modes.json common.shield.labels)・記録は標準の物理",
    ],
    "runtime": {
        "shield": {
            "version": defaults["version"],
            "file": "experiments/20261013_sim_rl_modes/r3/nw4/runtime_v5/shield.mjs(既定の値 = 下の params。ブラウザの teacher.js は値を渡さないので既定で走る)",
            "file_sha256_at_freeze": sha(E / "runtime_v5/shield.mjs"),
            "params": params,
            "what": [
                "v4.1(r3 の候補)の全部の値と動きはそのまま(見積もり・模型 Q・速さの見張り・ブレーキの探り)。v4.1 と同じ動きにする鍵の組 {stop: {enabled: false}, select: 'v4', cont_candidates: false, escape: {enabled: false}} は試験(test_shield_v5.mjs)で v4.1 と操作が全く同じことを確かめる",
                "stop(確認 r3 の指摘の原因 1): E(見積もった模型)と Q の先読みを、決まった長さ(E 24・Q 45 フレーム)のあとも予備の操縦(ブレーキ)で止まるまで続け(上限 240 フレーム)、止まるまで全部ですき間 ≥ しきい値のときだけ合格。標準の物理では E = sim なので、予備の操縦を続けることがいつも候補に残り(cont_candidates)、一度合格すればあとのフレームにも合格する候補がある",
                "select 'v5'(原因 2): 提案が合格しないとき、全部の模型に合格 → E と Q に合格(提案に近い順、v4 と同じ)。どれも Q に合格しなければ E に合格する候補からブレーキ(速さ 0)を先に・Q の予想のよい順('e_brake')。E にも合格しなければ E の予想のよい順・同じならブレーキ('max_e')。「同じ点数なら提案を残す」をやめる",
                "c_nominal を 0.08 → %s(判定の 8 cm は変えない。判定の自前のすき間は小区間の間も測るので、小区間の端だけで 8 cm を守るシールドに 1 mm の余裕)" % params["c_nominal"],
                "escape(止まりの直し): 止まっていて(|v| < 0.1 m/s)シールドが速さ 0 を返し、方策は進みたいフレームが 3 つ続いたら、決まった操縦の表(前へ: 舵 9 通り × その場で 0・8 フレーム舵を合わせて 0.5 m/s を 30 フレーム、後ろへ: 舵 5 通り × −0.4 m/s を 12・24 フレーム → 舵を 9 通りに切りながら止まる → 0.5 m/s を 30 フレーム)を E・Q・P で止まるまで先読みし、全部のしきい値(%s・0.07・0.05 m。ゆるめない)を満たし R の上で 0.10 m 以上進むもののうち一番進むものを、1 フレームずつ確かめ直しながら行う。前へだけの操縦があればそれを先に。後ろへ下がるフレームの指令は速さが負(sh = 1)" % params["c_nominal"],
                "近くの障害物の形を先読みの途中で作り直す(結果は全部の障害物で測るのと同じ値。試験で確かめる)。v4.1 は Q の 45 フレームの先読みに 24 フレームぶんの半径の形を使っていた",
            ],
            "browser": "Node とブラウザで同じファイル。web/rl/ の作り直し・F-ID・1 フレームの時間(抜け出しの計画のフレームは Node で約 0.1〜0.2 秒)は昇格の段",
        }
    },
    "definitions_added": {
        "reverse_frames": "sim の速さ < −0.01 m/s のフレーム(抜け出しで後ろへ動いた)",
        "escape": "シールドの fallback が escape_fwd・escape_rev のフレーム(抜け出しの計画を行っている)",
    },
    "sets": {
        "test": {
            "start_stream": "Halton 60,001〜64,999(protocol seeds.halton と同じ基数 2・3・5 の写し方。protocol の範囲 1〜29,999・審査の段の 40,001〜/50,001〜 の外)を no_wall の記録の開始の分布に写し、開始のすき間 < 0.12 m の番号を飛ばす。種 6,200,000 + i",
            "NW-N300-R4": {"physics": "標準", "count": 300, "starts": "Halton の流れの最初の 300", "seed_base": 6200000, "episode": "連続 20 秒、シールドあり。最初の 150 は軌跡を残す(最速との比べ)"},
            "NW-FS1000-R4": {"physics": "標準", "count": 1000, "starts": "記録の開始の分布から mulberry32(0x4e4a7e01) の一様乱数(開始のすき間 ≥ 0.12 m まで引き直す)", "seed_base": 6201000,
                             "why": "画面の「記録の開始から一様に引いた N 回の最小すき間」(確認 r3 の指摘 4)"},
            "NW-LOWC-R4": {"physics": "標準", "count": 400, "starts": "同じ分布から mulberry32(0x4e4a7e02)、開始のすき間 0.12〜0.15 m の層だけ(層の重み約 2%%。学習の一覧 starts_lowc_nw4.json は %d 回引いて %d 個)" % (lowc["draws"], lowc["count"]),
                           "seed_base": 6202000, "why": "確認 r3 の止まり・5.6 cm・衝突はこの層(柱・柵へ向いた開始)に集まる"},
            "NW-EDGE-R4": {"physics": "標準", "rule": "33 ゲート × 左右 {−0.19, −0.16, +0.16, +0.19} m × 向き {−14.5, −11.5, +11.5, +14.5}°(開始のすき間 ≥ 0.12 m のもの、約 460 回)。確認 r2・r3 の格子・r3 の格子と同じ点は無い",
                           "seed_base": 6202500, "why": "分布の端(ゲート 17 の右 0.2 m・左向き 15° の逆走など)"},
            "NW-STD": "protocol.json のまま(STD の単周と連続 20 秒)",
            "NW-G": "protocol.json のまま(ベストラップの r2 の条件の表、範囲の中 68 回。範囲の外 12 回は報告)",
            "NW-R200-R4": {"physics": "DR-T 一様(遅延 0/1 半々・摩擦円 30%)", "count": 200, "rng": "mulberry32(0x4e4a7e03)(開始も同じ乱数で記録の開始の分布から)", "seed_base": 6203000},
            "NW-VX-R4": {"physics": "dr_vertex", "count": 300, "rng": "同じ乱数の続き", "seed_base": 6204000},
            "NW-HC-R4": {"physics": "dr_hard_corner の全部の組 64 通り(舵の速さ・時定数・加速の上限・舵のゲイン・前進のゲインの端 × 遅延 0/1)× 開始 4", "count": 256, "rng": "同じ乱数の続き", "seed_base": 6205000,
                         "why": "確認 r3 の指摘: r3 の KX は乱数で 12 通りを引いたので、舵 1.8・前進 1.04 の組が無かった"},
            "NW-FC30-R4": {"physics": "標準の値 + 摩擦円", "count": 30, "seed_base": 6206000},
            "NW-LONG-R4": {"rule": "120 秒の連続周回: 標準 10・DR-T 一様 10", "count": 20, "seed_base": 6207000},
            "reported_only": {"NW-OFF-R4": "NW-N300-R4・NW-R200-R4・NW-VX-R4 をシールドなしで", "FAST": "最速の方策(bundle/fastest、web/rl、シールドなし)で NW-N300-R4 の最初の 150",
                              "LOWMU": "μ × 0.4(範囲の外)10 回", "NW-G-OUT": "NW-G の範囲の外 12 回"},
            "code": "r3/nw4/nw4_sets.mjs buildTest4Jobs(固定のときの sha256 は files.code_sha256_at_freeze)",
        },
        "validation": {
            "rule": "NW-V-R4 = V4-N 標準 150(Halton 70,001〜、種 6,100,000 + i)+ V4-LOWC 標準 150(開始のすき間 12〜15 cm、mulberry32(0x4e4a5601)、種 6,101,000 + i)+ V4-EDGE 標準(33 ゲート × 左右 ±0.18 m × 向き ±13°、種 6,101,500 +)+ V4-R DR-T 一様 100・V4-VX 頂点 100・V4-HC 止まりにくい隅の全部の組 64(mulberry32(0x4e4a5602)、種 6,102,000〜6,104,063)+ V4-STD。チェックポイントを選ぶためだけに使う",
            "code": "r3/nw4/nw4_sets.mjs buildValidation4Jobs",
        },
        "dev": "下調べの組 dev4(標準 300・12〜15 cm の層 300・DR-T 一様 100・頂点 100・隅 64、mulberry32(0x4e4a0d01)、種 6,900,000 +)と dev4e(33 ゲート × 左右 {±0.2, ±0.15} × 向き {±15, ±12}°、種 6,950,000 +)。シールド v5 の値と学び直しの要否を決めるのに使った。選び・試験には使わない",
        "regression_reported": "確認 r3 の組(NBR 54・LOWC 300・HCALL 384・EDGE2 44・REG 142)を確認 r3 の走らせ方 r3/ck_nw3/run_ck3.mjs(変えない)で候補に走らせる。下調べで一部を見たので判定には使わない(報告だけ。確認 r3 の must-fix の「NBR・LOWC・HCALL を run_ck3.mjs で走らせ直す」)",
    },
    "acceptance": {
        "N1_r4": "試験の判定の組(NW-N300-R4・NW-FS1000-R4・NW-LOWC-R4・NW-EDGE-R4・NW-STD・NW-G の範囲の中・NW-R200-R4・NW-VX-R4・NW-HC-R4・NW-FC30-R4・NW-LONG-R4)の全部の回で衝突 0(シールドあり)",
        "N2_r4": "標準の物理の全部の回(NW-N300-R4・NW-FS1000-R4・NW-LOWC-R4・NW-EDGE-R4・NW-STD・NW-LONG-R4 の標準 10)で最小すき間 ≥ 0.080 m",
        "N3_r4": "物理をずらした全部の回(NW-G の範囲の中・NW-R200-R4・NW-VX-R4・NW-HC-R4・NW-FC30-R4・NW-LONG-R4 の DR 10)で最小すき間 ≥ 0.050 m",
        "N4": "protocol.json のまま(NW-STD の単周、停止から 1 周 ≤ 12.0 秒)",
        "N5_r4": "標準の物理の全部の回(N300・FS1000・LOWC・EDGE・STD 連続・LONG の標準)と NW-FC30-R4 が valid_episode_r3(止まり無し・有効な周 ≥ 1、120 秒は ≥ 5)。NW-R200-R4 ≥ 190/200・NW-VX-R4 ≥ 285/300・NW-HC-R4 ≥ 244/256",
        "N6": "protocol.json のまま(組は NW-N300-R4)",
        "N7_r4": "NW-N300-R4 の shield_fraction ≤ 0.05、かつ NW-N300-R4 のどの回も ≤ 0.25(抜け出しのフレームも数える)",
        "reported": ["NW-OFF-R4 の衝突・すき間とシールドあり / なしの対(救った・起こした)", "組ごとの後ろへ動いた回・フレーム、抜け出しの回数、1 フレームの計算の時間(中央値・99% 点・最大)",
                     "物理をずらした組のシールドの作動と 1 周の時間、隅の組ごとの結果", "NW-G の範囲の外・LOWMU", "最速との比べ(線・速さ・すき間、モードの中のばらつき)", "標準の物理で 8〜9 cm の回の数と場所", "ρ の分布",
                     "確認 r3 の組の走らせ直し(regression_reported)"],
        "ui_numbers": "「N 回の試験で衝突 C・最小すき間 X cm(標準の物理 M 回: 記録の開始から一様 1000 回では Xu cm、開始のすき間 12〜15 cm・分布の端を含む)/ Y cm(物理のずれ)」。X = N2_r4 の組の最小、Xu = NW-FS1000-R4 の最小、Y = N3_r4 の組の最小。不合格なら「・不合格(参考)」を足す。説明(mode.json の dataset_card・status の note)には確認 r3 の数字(一様 1,000 回で 8.01 cm、まれな開始で 5.63 cm・衝突 1 は v4.1 のもの)と r4 の数字を並べる",
        "thresholds_relaxed": "無し",
    },
    "training": {
        "run": "no_wall_r4(r3/nw4/train/train_nw4.py・env_nw4.mjs。r3 の no_wall_r3 と同じ報酬・物理の内訳・速さの見張り・止まりの見張りで、開始の分布だけ替える)",
        "init": "r3 の選んだ点 r3/no_wall_r3/train/checkpoints/step_006012928.pt(方策 3b099f9b…)",
        "seed": 24,
        "seed_note": "protocol allowed_tuning.retry は r2 で使った。r3(種 23)・r4(種 24)は protocol の外の作り直し(この追補と train_no_wall_r4.md に書く)",
        "steps_max": 3000000,
        "lr": 3e-05,
        "ckpt_every": 500000,
        "start_mix": [
            {"p": 0.40, "what": "記録の開始の分布(開始のすき間 ≥ 0.12 m)"},
            {"p": 0.20, "what": "開始のすき間 12〜15 cm の開始の一覧 starts_lowc_nw4.json(記録の開始の分布から mulberry32(0x4e4a7a01) で一様に引いた %d 個。ゲート 20〜25 の柱・柵へ向いた開始が多い)" % lowc["count"]},
            {"p": 0.15, "what": "分布の端: 全ゲート、|左右| 0.13〜0.2 m・|向き| 10〜15°(開始のすき間 ≥ 0.12 m)"},
            {"p": 0.10, "what": "r3 のむずかしい開始(ゲート 7〜10・16・17・20〜23、|左右| 0.1〜0.2 m・|向き| 7.5〜15°)"},
            {"p": 0.15, "what": "広め: 左右 ±0.25 m・向き ±20°(開始のすき間 ≥ 0.10 m)"},
        ],
        "lowc_list_sha256": sha(E / "train/starts_lowc_nw4.json"),
        "dr_reward_governor_stall": "r3(protocol_no_wall_r3.json training)のまま",
        "shield_in_training": "使わない(重い)。検証・試験はシールド v5 あり",
        "lock": "jf-heavy --mem 2G --lock rl -- nice -n 15(r3/nw4/run_train_nw4.sh。チェックポイント・続きから再開)",
    },
    "selection": {
        "candidates": "学習の始まり step_000000000(= r3 の選んだ方策 3b099f9b…。r4 の学び直しが役に立たなかったときのため)と、学習の 50 万ステップごとのチェックポイント",
        "rule": "検証の組(シールド v5)で N1_r4〜N7_r4 の形を全部満たし止まり 0 の点のうち一番後のもの。無ければ満たす数が一番多く、同じなら標準の物理の最小すき間が大きいもの",
        "test_runs": "試験の組は選んだ 1 つにだけ 1 回使う。試験の結果を見て選び直さない・シールドを変えない",
    },
    "recording": {
        "prefilter": "記録の前の予想(r3 版 predict_scenes_nw3.mjs の写しを v5 で)で、valid_episode_r3 でない・最小すき間 < 0.08 m・shield_fraction > 0.25・後ろへ動いたフレームがある シーンを次の候補に替える(後ろへ下がる正解ラベルをデータセットに入れない)",
        "mode_json_overrides": "runtime.shield_version jf-shield/v5・runtime.shield_params(上の値)・recording.prefilter_min_clearance_m 0.08・prefilter_min_valid_laps 1・prefilter_no_stall true・prefilter_max_shield_fraction 0.25・prefilter_no_reverse true・scenario_blocklist ['low-friction']",
    },
    "status_fix": "bundle/no_wall/status.json は審査の段(2026-10-14 13:55)が not_accepted「不合格(参考)」(failed: check_r2_corner_collisions・check_r2_N2_fresh・check_r2_N5_stall)に直した。r4 の候補の status は試験の結果のまま。昇格の段が r3 の候補を「不合格(参考)」で出すなら failed に N5_r3・check_r3_nominal_collision・check_r3_N2_5p6cm を足す(candidates/no_wall_r4/bundle_status_fix.json)",
    "publication": {
        "r4": "合格なら昇格の段が bundle/no_wall/ を r4 の一式で置き換える(status accepted、web/rl に shield.mjs v5)。不合格なら r4 の一式は candidates/no_wall_r4/ のまま(「不合格(参考)」で出すかは昇格の段)",
        "promote_step": "昇格の段は、modes の作業(wf_7e500cf6-ccd)が終わってから共有のファイルを変える",
    },
    "files": {
        "code_sha256_at_freeze": {p.name: sha(p) for p in sorted(list(E.glob("*.mjs")) + list(E.glob("*.py")) + list(E.glob("*.sh")) + list((E / "runtime_v5").glob("*.mjs")) + list((E / "train").glob("*")))
                                  if p.is_file()},
        "dev_results": "jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r4/dev/",
    },
}
txt = json.dumps(P, ensure_ascii=False, indent=1) + "\n"
OUT.write_text(txt)
SHA.write_text(f"{sha(OUT)}  protocol_no_wall_r4.json\n")
print("書きました", OUT, sha(OUT))
