"""r3(確認 r2 のあと。手順の外の参考の追加実験)の結果をまとめ、事前登録の判定を出す。2026-10-13

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python -I rl/r3/r3_summary.py

読むもの:
  rl/r3/indep/<名前>.json        確認 r2 の道具(元の sim-core.js・ユーザーの遅延の待ち行列)で測った値(rl/r3/indep_check_r3.mjs)
  rl/r3/robust_eval_r3_s2N.json  r3 の条件表 39・ランダムな物理・REC60・摩擦円・荒さ(rl/r2/robust_eval.mjs、高速版のコア)
  rl/r2/robust_eval.json         同じものの選んだ方策・r2・教師(r2 の段の値)
  rl/final/final_eval_main.json  選んだ方策と教師の最終の評価(元のコア)
  rl/r3/<run>/final_eval.json    r2 の 500 万・r3 の 2 本の同じ評価(rl/final_eval.mjs)
  rl/<run>/evals/step_*.json     チェックポイントごとの標準の評価(学習の中の自動の評価、高速版のコア)
書くもの: rl/r3/r3_summary.json(表と判定)。標準出力に要点
判定(deviations.md 2026-10-13 22:40 の事前登録): 角が直った = 角の 20 マスすべてで REC60 の開始 ≥ 57/60、
  すき間の約束が効いた = 標準の物理の REC60 の最小すき間 ≥ 3 cm、速さを保った = 標準の物理の STD の単周 ≤ 8.831 秒。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze_rl as A  # noqa: E402

RL = A.RL
WHO = [("teacher", "RC 教師"), ("selected", "選んだ方策(r1)"), ("r2_5M", "r2 500 万"), ("r3_s21", "r3 種 21"), ("r3_s22", "r3 種 22")]
FINAL = {"teacher": (RL / "final" / "final_eval_main.json", "teacher"), "selected": (RL / "final" / "final_eval_main.json", "rl"),
         "r2_5M": (RL / "r3" / "r2_5M" / "final_eval.json", "rl"), "r3_s21": (RL / "r3" / "r3-s21" / "final_eval.json", "rl"),
         "r3_s22": (RL / "r3" / "r3-s22" / "final_eval.json", "rl")}
ROBUST = {"teacher": (RL / "r2" / "robust_eval.json", "RC 教師"), "selected": (RL / "r2" / "robust_eval.json", "選んだ方策"),
          "r2_5M": (RL / "r2" / "robust_eval.json", "DR 5.0M"), "r3_s21": (RL / "r3" / "robust_eval_r3_s21.json", "r3 種 21"),
          "r3_s22": (RL / "r3" / "robust_eval_r3_s22.json", "r3 種 22")}
H1_SINGLE = 8.831


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    out = {"schema": "rl-bestlap-r3-summary/v1", "preregistration": "deviations.md 2026-10-13 22:40", "controllers": {}, "judgement": {}, "checkpoints": {}}
    for key, label in WHO:
        C = {"label": label}
        ind = jload(RL / "r3" / "indep" / f"{key}.json")
        I = (ind or {}).get("controllers", {}).get(key, {})
        if I:
            C["indep"] = {k: I[k] for k in ("sha256", "rec60", "samecond", "corner", "grid2d", "rec60x", "allbad", "starts") if k in I}
            for k in ("corner", "grid2d"):
                if k in C["indep"]:
                    C["indep"][k] = {kk: vv for kk, vv in C["indep"][k].items()}
        fp, part = FINAL[key]
        fe = jload(fp)
        if fe and part in fe:
            r = fe[part]
            std = r["STD"]
            C["final"] = {
                "single_lap_s": std["single"].get("single_lap_s"), "single_verification_pass": (std["single"].get("verification") or {}).get("pass"),
                "flying_lap_s": (std["continuous"].get("laps") or [{}, {}])[1].get("elapsed") if len(std["continuous"].get("laps") or []) > 1 else None,
                "single_metrics": std["single"].get("metrics"),
                "E": [{"name": e["name"], "lap": e.get("single_lap_s"), "valid": e.get("lap_complete") and e.get("collisions") == 0,
                       "min_clearance": e.get("min_clearance"), "verification": (e.get("verification") or {}).get("pass"),
                       "metrics": {k: (e.get("metrics") or {}).get(k) for k in ("max_speed", "lat_acc_max", "lat_acc_p95", "steer_cmd_rate_p95", "steer_edge_frac")}}
                      for e in (r.get("single") or [])],
                "realism": (r.get("REC60_realism") or {}).get("realism"), "rec60": (r.get("REC60_realism") or {}).get("summary"),
                "delay1_single_s": (r.get("STD_delay1_reference") or {}).get("single_lap_s") or (r.get("STD_delay1_reference") or {}).get("sim_time")}
            if part == "rl":
                C["final"]["selection_sets"] = {k: fe.get("selection_sets", {}).get(k) for k in ("eligible", "score")}
                C["final"]["rob24"] = (fe.get("selection_sets", {}).get("ROB24") or {})
                C["final"]["fast_vs_original"] = fe.get("fast_vs_original")
        rp, name = ROBUST[key]
        rob = jload(rp)
        R = (rob or {}).get("controllers", {}).get(name)
        if R:
            C["robust"] = {"grid_summary": R.get("grid_summary"), "grid_failed": [g["name"] for g in R.get("grid", [])[1:] if not (g["single"].get("valid") and g["continuous"].get("valid"))],
                           "random": {k: {"valid": v["valid"], "n": v["n"], "lap1_mean": v.get("lap1_mean"), "lap2_mean": v.get("lap2_mean")} for k, v in (R.get("random") or {}).items()},
                           "rec60": {k: {kk: v.get(kk) for kk in ("valid", "n", "lap1_mean", "lap2_mean", "min_clearance")} for k, v in (R.get("rec60") or {}).items()},
                           "friction_circle": R.get("friction_circle"), "rough": R.get("rough")}
        out["controllers"][key] = C
    # チェックポイントごとの標準の評価(学習の中の自動の評価)
    for run in ("r3-s21", "r3-s22"):
        rows = []
        for f in sorted((RL / run / "evals").glob("step_*.json")):
            e = json.loads(f.read_text())
            s = e["REC60"]["summary"]
            rows.append({"step": int(f.stem.split("_")[1]), "eligible": e.get("eligible"), "rec60_valid": s["valid_two_laps"],
                         "lap1_mean": s["lap1"].get("mean") if s["lap1"].get("n") == 60 else None, "lap2_mean": s["lap2"].get("mean") if s["lap2"].get("n") == 60 else None,
                         "min_clearance": s["min_clearance"], "std_single": e["STD"]["single_lap_from_stop"], "std_flying": e["STD"]["flying_lap"],
                         "rob24": e["ROB24"]["summary"]["valid_two_laps"]})
        out["checkpoints"][run] = rows
    # 事前登録の判定
    for key in ("r3_s21", "r3_s22"):
        C = out["controllers"].get(key, {})
        I = C.get("indep", {})
        corner = I.get("corner")
        rec = (I.get("rec60") or {}).get("delay0")
        lap = ((I.get("samecond") or {}).get("delay0") or {}).get("lap")
        j = {"corner_all_ge57": corner is not None and all(c["rec60_valid"] >= 57 for c in corner["cells"]),
             "corner_cells_ge57": corner and corner["cells_ge57"], "rec60_min_clearance_m": rec and rec["clearance_frame_min"]["d"],
             "clearance_ge_3cm": rec is not None and rec["clearance_frame_min"]["d"] >= .03 and rec["valid"] == rec["n"],
             "std_single_lap_s": lap, "speed_kept": lap is not None and lap <= H1_SINGLE + 1e-12,
             "eligible_procedure": (C.get("final", {}).get("selection_sets") or {}).get("eligible")}
        j["all_three"] = j["corner_all_ge57"] and j["clearance_ge_3cm"] and j["speed_kept"]
        out["judgement"][key] = j
    n_ok = sum(1 for k in ("r3_s21", "r3_s22") if out["judgement"].get(k, {}).get("all_three"))
    out["judgement"]["verdict"] = {2: "有望(種によらない)", 1: "種しだい", 0: "この設定では足りない"}[n_ok]
    (RL / "r3" / "r3_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for key, label in WHO:
        C = out["controllers"].get(key, {})
        I = C.get("indep", {})
        sc = I.get("samecond", {})
        rec = I.get("rec60", {})
        print(label, json.dumps({
            "単周": (sc.get("delay0") or {}).get("lap") or (C.get("final") or {}).get("single_lap_s"),
            "フライング": (C.get("final") or {}).get("flying_lap_s"),
            "REC60": rec.get("delay0") and f"{rec['delay0']['valid']}/{rec['delay0']['n']} {rec['delay0']['lap1_mean']:.4f}/{rec['delay0']['lap2_mean']:.4f} "
                                          f"すき間 {100 * rec['delay0']['clearance_frame_min']['d']:.2f} cm <3cm {rec['delay0']['clearance_frame_below_3cm_runs']}",
            "遅延1": rec.get("delay1_applied") and f"{rec['delay1_applied']['valid']}/{rec['delay1_fresh']['valid']}",
            "ONNX条件": sc.get("delay1_gain101_applied") and f"{sc['delay1_gain101_applied']['lap']}/{sc['delay1_gain101_fresh']['lap']}",
            "角": I.get("corner") and f"{I['corner']['cells_60']}/20 (≥57: {I['corner']['cells_ge57']}) 合計 {I['corner']['total_valid']}/{I['corner']['total_n']}",
            "25マス": I.get("grid2d") and f"{I['grid2d']['cells_60']}/25 (≥57: {I['grid2d']['cells_ge57']}) 合計 {I['grid2d']['total_valid']}/{I['grid2d']['total_n']}",
            "rec60x": I.get("rec60x") and {k: f"{v['valid']}/{v['n']} 角 {v['corner_valid']}/{v['corner_draws']}" for k, v in I["rec60x"].items()},
            "全部悪い": I.get("allbad") and f"{I['allbad']['std_single']['valid']}/{I['allbad']['std_cont']['valid']} REC60 {I['allbad']['rec60']['valid']}/{I['allbad']['rec60']['n']}",
            "開始": I.get("starts") and f"{I['starts']['grid_valid']}/{I['starts']['grid_n']} {I['starts']['gates_valid']}/{I['starts']['gates_n']}",
            "条件表": (C.get("robust") or {}).get("grid_summary", {}).get("both_valid"),
            "ランダム": {k: v["valid"] for k, v in ((C.get("robust") or {}).get("random") or {}).items()},
            "摩擦円": [(r["mode"], r["reason"], r["laps"]) for r in ((C.get("robust") or {}).get("friction_circle") or [])],
        }, ensure_ascii=False))
    print("判定", json.dumps(out["judgement"], ensure_ascii=False))


if __name__ == "__main__":
    main()
