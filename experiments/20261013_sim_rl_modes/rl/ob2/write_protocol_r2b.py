"""障害物よけ r2 の後半(obstacle_r2b)の追記の決まり protocol_obstacle_r2b.json を書く(2026-10-14 学習の段 obstacle r2)。後半を始める前に 1 回だけ固める。"""
import hashlib
import json
from pathlib import Path

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
OUT = REP / "protocol_obstacle_r2b.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
if OUT.exists():
    raise SystemExit("protocol_obstacle_r2b.json はもう固めてあります")
INIT = REP / "train/obstacle/r2/main/checkpoints/step_007012352.pt"
P = {
    "title": "障害物よけ(obstacle)第 2 回の後半の追記 — protocol_obstacle_r2.json(9b591db7…a011d)を変えずに、学習の後半の環境だけを足す",
    "version": "obstacle-r2b-1",
    "written_at": "2026-10-14T07:50:00+09:00",
    "written_by": "学習の段 obstacle r2",
    "base_protocol_obstacle_r2_sha256": sha(REP / "protocol_obstacle_r2.json"),
    "base_protocol_sha256": sha(REP / "protocol.json"),
    "why": [
        "r2 の前半(obstacle_r2、種 62)を 701 万で止めた。選びの組 OB-V100-R2 で有効 14 → 24 → 36 → 39 → 53 → 52(100 万〜600 万)と上がったが、",
        "(1) 決定的な方策が障害物の手前で止まる・はう回がある(500 万で有効 53 のうち 16 回は 1 周も回っていない、600 万で 3 回)。protocol の valid_episode は止まっていても有効に数えるので、O1 が見かけだけ上がり、見本にもならない。学習の『止まる』の終了(0.2 m/s 未満が 2 秒)は、0.2 m/s を少し越えてはう走りを止めない",
        "(2) 衝突の多くは 3.3〜4.2 m/s のまま、よける幅が足りない・通り道の無い側へ行く(600 万の選びの組: 通り道の無い側・狭い側への失敗 19、広い側でも幅が足りない 22。下調べ rl/ob2/passage_sides.mjs)",
        "(3) O4(障害物なし 1 周)が 8.73 → 9.22〜9.51 秒に遅くなった(R の s 6〜11 m、黒シートの先の曲がりで 1.0 m/s・外側 56 cm を通る)。障害物なしの回は 10% だけ",
    ],
    "unchanged": "protocol_obstacle_r2.json の判定・試験の組・選びの組 OB-V100-R2・選び方・試験の使い方・記録の決まり・観測・走行中の仕組み・報酬の前半の項(障害物の壁・全部の障害物の壁・近づく速さの罰・3 cm の終了・ゲートの外し)・開始の分布・配置の作り方",
    "training": {
        "run": "obstacle_r2b(rl/env_modes.mjs trainConfig('obstacle_r2b')・OBSTACLE_R2B_DEFAULTS、rl/train_modes.py --run obstacle_r2b)",
        "init": "r2 の前半の 701 万のチェックポイント(" + str(INIT) + "、sha256 " + sha(INIT) + ")。重み・観測の正規化・log σ をそのまま引き継ぐ(Adam の状態は新しく)",
        "seed": 62,
        "steps": 8000000,
        "steps_note": "前半 701 万 + 後半 800 万 = 1,501 万(この段の依頼の 1,500 万)",
        "lr": "1.75e-4 → 3e-5(前半の 3e-4 → 3e-5 の 701 万の所の値から続ける)",
        "added": {
            "slow_progress_terminal": "1.5 秒以降、直近 60 フレーム(2 秒)の R の進みが 0.6 m 未満なら −10 で終える(学習だけ)",
            "passage_guidance": "rl/ob2/guidance.mjs(sha256 " + sha(EXP / "rl/ob2/guidance.mjs") + "、学習だけ): 障害物ごとに、障害物の s ±0.2 m で全部の障害物とのすき間 ≥ 8 cm の横の区間(ゲートの窓 ±0.65 m の中)の共通部分のうち一番広いものを選び、目標 e_t = 障害物に近い端から min(0.12 m, 幅/2) 内側。次の障害物の手前 2.5 m から 0.3 m 過ぎるまで −0.05·min(1, ((e_y − e_t)/0.15)²)·ramp(1.0 m より近いと 1、2.5 m で 0)",
            "none_p": "障害物なしの回 10% → 20%",
        },
        "early_stop": "後半 100 万(通しで 800 万)以降、続けて 3 つのチェックポイントが OB-V100-R2 で O1〜O4 を全部満たし有効 ≥ 98/100 なら止めてよい",
    },
    "selection_addition": {
        "candidates": "前半(r2/main、100 万〜700 万)と後半(r2b/main、100 万ごと)の全部のチェックポイント。選び方は protocol_obstacle_r2.json selection.rule のまま",
        "strict_valid": "前半で止まる・はう回が見つかったので、試験の組を使う条件に足す: 選んだチェックポイントが OB-V100-R2 で、有効な回のうち 1 周も回っていない回を無効に数えても O1(95/100)を満たすこと(厳しくする向きだけ。判定の O1 の式は protocol のまま)。試験・報告でも『有効だが周 0』の回の数を出す",
    },
    "files": {"env_modes.mjs": sha(EXP / "rl/env_modes.mjs"), "train_modes.py": sha(EXP / "rl/train_modes.py"), "guidance.mjs": sha(EXP / "rl/ob2/guidance.mjs"),
              "early_stop_r2.py": sha(EXP / "rl/ob2/early_stop_r2.py")},
}
OUT.write_text(json.dumps(P, ensure_ascii=False, indent=1) + "\n")
(REP / "protocol_obstacle_r2b.sha256").write_text(f"{sha(OUT)}  protocol_obstacle_r2b.json\n")
print(sha(OUT))
