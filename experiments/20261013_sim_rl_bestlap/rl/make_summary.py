"""RL の段のまとめ: H1 の判定・種ごとの表・周回の比べ・現実味の表・周回時間の比べの図(2026-10-13、RL の学習の段)

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python rl/make_summary.py            # → rl/summary.json、fig/rl_lap_comparison.png、表(Markdown)を標準出力へ

読むもの: rl/selection.json(analyze_rl.py select)、rl/final/final_eval_main.json・final_eval_open.json(final_eval.mjs、元のコア)、
          各 run の evals/step_*.json(高速版のコア)、rl/realism_realcar.json、env/teacher_baseline.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_rl as A  # noqa: E402

RL, FIG = A.RL, A.FIG
FINAL = RL / "final"


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f3(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def run_table() -> list[dict]:
    rows = []
    for n in A.runs():
        for e in A.load_evals(n):
            rows.append({"run": n, "step": e["step"], "eligible": e["eligible"], "rec60_valid": e["rec60_valid"],
                         "lap1_mean": e["rec60_lap1_mean"] if e["rec60_lap1_n"] == 60 else None,
                         "lap2_mean": e["rec60_lap2_mean"] if e["rec60_lap2_n"] == 60 else None,
                         "min_clearance": e["rec60_min_clearance"], "std_single": e["std_single"], "std_flying": e["std_flying"],
                         "rob24": e["rob24_valid"], "score": e["score"]})
    return rows


def h1(sel: dict, fe_main: dict) -> dict:
    T1, T2, TS = A.T_LAP1, A.T_LAP2, A.T_SINGLE
    if not fe_main or "selection_sets" not in fe_main:
        return {"judged": False}
    s = fe_main["selection_sets"]
    c1 = s["STD_single_lap_from_stop"] is not None and s["STD_single_lap_from_stop"] <= A.H1_SINGLE
    c2 = (s["REC60"]["lap1"].get("n") == 60 and s["REC60"]["lap2"].get("n") == 60
          and s["REC60"]["lap1"]["mean"] <= T1 - 0.1 and s["REC60"]["lap2"]["mean"] <= T2 - 0.1)
    seeds = []
    for n, r in sel["runs"].items():
        if r["variant"] != "main":
            continue
        b = r["best"]
        seeds.append({"run": n, "has_eligible": b is not None, "best_step": b["step"] if b else None,
                      "std_single": b["std_single"] if b else None, "faster_than_teacher_single": bool(b and b["std_single"] is not None and b["std_single"] < TS)})
    c3 = sum(1 for x in seeds if x["has_eligible"] and x["faster_than_teacher_single"]) >= 2
    return {"judged": True, "pass": bool(c1 and c2 and c3),
            "c1_std_single_le_8.831": {"value": s["STD_single_lap_from_stop"], "threshold": A.H1_SINGLE, "ok": bool(c1)},
            "c2_rec60_means_0.1_faster": {"lap1_mean": s["REC60"]["lap1"].get("mean"), "lap1_threshold": T1 - 0.1,
                                          "lap2_mean": s["REC60"]["lap2"].get("mean"), "lap2_threshold": T2 - 0.1, "ok": bool(c2)},
            "c3_two_of_three_seeds": {"seeds": seeds, "ok": bool(c3)}}


def realism_rows(fe_main, fe_open, teacher_fe, car) -> list[dict]:
    rows = []
    if fe_main:
        rows.append({"who": "RL main(選んだ方策、REC60)", **fe_main["rl"]["REC60_realism"]["realism"], "min_clearance": fe_main["rl"]["REC60_realism"]["summary"]["min_clearance"]})
    if fe_open:
        rows.append({"who": "RL open(参考、REC60)", **fe_open["rl"]["REC60_realism"]["realism"], "min_clearance": fe_open["rl"]["REC60_realism"]["summary"]["min_clearance"]})
    if teacher_fe:
        rows.append({"who": "RC 教師(REC60)", **teacher_fe["teacher"]["REC60_realism"]["realism"], "min_clearance": teacher_fe["teacher"]["REC60_realism"]["summary"]["min_clearance"]})
        lo = teacher_fe.get("learned_onnx")
        if lo:
            rows.append({"who": "学習済み ONNX 8.964 秒の周(E0・遅延 1・前進ゲイン 1.01)", **lo["metrics"], "min_clearance": lo["min_clearance"]})
    if car:
        rows.append({"who": "実車 clean_driving_20261011(目安)", **{k: car.get(k) for k in ("max_speed", "speed_p99", "frac_over_4mps", "lat_acc_max", "lat_acc_p95",
                                                                                  "steer_cmd_rate_p95", "steer_edge_frac", "throttle_pwm_p01")},
                     "speed_median": car.get("speed_median"), "rho_gt1_frac": None, "steer_rate_p95": None, "min_clearance": None})
    return rows


def lap_fig(fe_main, fe_open, teacher_fe) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A._jp_font()
    starts = ["E0", "E1", "E2", "E3", "E4", "E5"]
    series = []
    if teacher_fe:
        t = {r["name"]: r["single_lap_s"] for r in teacher_fe["teacher"]["single"]}
        series.append(("RC 教師(E0〜E2 だけ。約束の範囲)", "#eb6834", [t.get(s) for s in starts]))
        # 学習済み ONNX の 8.964 秒は遅延 1・前進ゲイン 1.01 の条件なので、遅延 0 の列には置かない(右の図で同じ条件どうしを比べる。確認 r1 の指摘)
    if fe_main:
        m = {r["name"]: r["single_lap_s"] for r in fe_main["rl"]["single"]}
        series.append(("RL main(選んだ方策、上限 4.5 m/s)", "#2a78d6", [m.get(s) for s in starts]))
    if fe_open:
        o = {r["name"]: r["single_lap_s"] for r in fe_open["rl"]["single"]}
        series.append(("RL open(参考、上限 7.24 m/s)", "#eda100", [o.get(s) for s in starts]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), facecolor=A.SURFACE, gridspec_kw={"width_ratios": [2.3, 1]})
    ax.set_facecolor(A.SURFACE)
    same_condition_panel(ax2, teacher_fe)
    w = 0.6 / max(1, len(series))
    allv = []
    marks = ["o", "s", "^", "v"]
    for k, (lab, col, vals) in enumerate(series):
        xs = np.arange(len(starts)) + (k - (len(series) - 1) / 2) * w
        pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
        if not pts:
            continue
        ax.plot(*zip(*pts), ls="none", marker=marks[k % 4], ms=8, color=col, label=lab)
        for x, v in pts:
            ax.text(x, v + 0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=A.INK2)
            allv.append(v)
    ax.set_xticks(np.arange(len(starts)))
    ax.set_xticklabels(["E0 標準", "E1 +3 cm・+3°", "E2 −3 cm・−3°", "E3 +8 cm・−5°", "E4 −8 cm・+5°", "E5 ゲート 20"], fontsize=9, color=A.INK2)
    if allv:
        ax.set_ylim(min(allv) - 0.15, max(allv) + 0.2)
    ax.axhline(A.H1_SINGLE, color=A.INK2, lw=1.0, ls=":")
    ax.text(len(starts) - 0.5, A.H1_SINGLE, f"H1 の条件 {A.H1_SINGLE:.3f}", color=A.INK2, fontsize=8, ha="right", va="bottom")
    ax.set_ylabel("停止から 1 周の時間 [秒](元の sim-core.js)", color=A.INK2, fontsize=9)
    ax.set_title("停止から 1 周(単周)・制御遅延 0 — 開始 E0〜E5\n(点が無い = 走らせていない。教師は約束の外の開始を走れない)", color=A.INK, fontsize=10.5, loc="left")
    ax.grid(True, axis="y", color=A.GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=A.INK2, labelsize=8)
    ax.legend(fontsize=8, frameon=False, labelcolor=A.INK2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.tight_layout()
    fig.savefig(FIG / "rl_lap_comparison.png", dpi=130, facecolor=A.SURFACE, bbox_inches="tight")
    plt.close(fig)


def same_condition_panel(ax, teacher_fe) -> None:
    """右の図: 学習済み ONNX(遅延 1・前進ゲイン 1.01)と同じ条件で比べる(rl/r2/same_condition_vs_onnx.json。確認 r1 の must-fix)。"""
    sc = jload(RL / "r2" / "same_condition_vs_onnx.json")
    ax.set_facecolor(A.SURFACE)
    if not sc:
        ax.set_axis_off()
        return
    c = sc["comparison"]
    cats = ["遅延 0", "遅延 1", "遅延 1 +\n前進ゲイン 1.01\n(ONNX の条件)"]
    rl_app = [c["rl_delay0_s"], c["rl_delay1_applied_s"], c["rl_delay1_gain101_applied_s"]]
    rl_fr = [c["rl_delay0_s"], c["rl_delay1_fresh_s"], c["rl_delay1_gain101_fresh_s"]]
    x = np.arange(3)
    for i in range(3):
        lo, hi = sorted([rl_app[i], rl_fr[i]])
        ax.plot([x[i] - 0.12] * 2, [lo, hi], color="#2a78d6", lw=3, solid_capstyle="round", alpha=0.5)
    ax.plot(x - 0.12, rl_app, ls="none", marker="s", ms=8, color="#2a78d6", label="RL main(選んだ方策。線 = 観測の前回の行動の約束 2 通りの幅)")
    for i in range(3):
        lab = f"{rl_app[i]:.3f}" if abs(rl_app[i] - rl_fr[i]) < 1e-9 else f"{min(rl_app[i], rl_fr[i]):.3f}〜{max(rl_app[i], rl_fr[i]):.3f}"
        ax.text(x[i] - 0.12, max(rl_app[i], rl_fr[i]) + 0.03, lab, ha="center", va="bottom", fontsize=7.5, color=A.INK2)
    ax.plot([2 + 0.12], [c["onnx_from_stop_s"]], ls="none", marker="D", ms=8, color="#1baf7a", label="学習済み ONNX(記録の再生)")
    ax.text(2 + 0.12, c["onnx_from_stop_s"] + 0.03, f"{c['onnx_from_stop_s']:.3f}", ha="center", va="bottom", fontsize=7.5, color=A.INK2)
    t = sc.get("teacher", {})
    t0 = (t.get("delay0_single") or {}).get("laps") or [None]
    ax.plot([0 + 0.12], [t0[0]], ls="none", marker="o", ms=8, color="#eb6834", label="RC 教師(遅延 1 では衝突)")
    ax.text(0 + 0.12, t0[0] + 0.03, f"{t0[0]:.3f}", ha="center", va="bottom", fontsize=7.5, color=A.INK2)
    for i in (1, 2):
        ax.text(x[i] + 0.12, 8.2, "教師は\n衝突", ha="center", va="center", fontsize=7.5, color="#eb6834")
    d = c["diff_same_condition_s"]
    ax.annotate("", xy=(2 - 0.02, max(rl_app[2], rl_fr[2])), xytext=(2 - 0.02, c["onnx_from_stop_s"]),
                arrowprops=dict(arrowstyle="<->", color=A.INK2, lw=1))
    ax.text(1.55, (c["onnx_from_stop_s"] + rl_app[2]) / 2, f"同じ条件で\n{min(d):.2f}〜{max(d):.2f} 秒速い", fontsize=8, color=A.INK, ha="right", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8.5, color=A.INK2)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(7.95, 9.15)
    ax.set_title("同じ条件で比べる(E0 標準の開始・停止から 1 周)", color=A.INK, fontsize=10.5, loc="left")
    ax.grid(True, axis="y", color=A.GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=A.INK2, labelsize=8)
    ax.legend(fontsize=7.2, frameon=False, labelcolor=A.INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)


def main() -> None:
    sel = jload(RL / "selection.json")
    fe_main = jload(FINAL / "final_eval_main.json")
    fe_open = jload(FINAL / "final_eval_open.json")
    teacher_fe = fe_main if fe_main and "teacher" in fe_main else jload(FINAL / "final_eval_teacher.json")
    car = jload(RL / "realism_realcar.json")
    out = {"schema": "rl-bestlap-summary/v1", "teacher_rec60_lap1_mean_s": A.T_LAP1, "teacher_rec60_lap2_mean_s": A.T_LAP2,
           "teacher_single_lap_from_stop_s": A.T_SINGLE, "learned_onnx_s": A.LEARNED, "runs": run_table(),
           "H1": h1(sel, fe_main) if sel else {"judged": False}, "realism": realism_rows(fe_main, fe_open, teacher_fe, car)}
    sc = jload(RL / "r2" / "same_condition_vs_onnx.json")
    if sc:
        # 確認 r1 の must-fix: 学習済み ONNX の 8.964 秒は遅延 1・前進ゲイン 1.01 の値。同じ条件の RL と比べる
        out["same_condition_vs_onnx"] = sc["comparison"]
        out["learned_onnx_condition"] = sc["comparison"]["onnx_condition"]
    if fe_main:
        out["fast_vs_original_main"] = fe_main.get("fast_vs_original")
    if fe_open:
        out["fast_vs_original_open"] = fe_open.get("fast_vs_original")
    (RL / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    lap_fig(fe_main, fe_open, teacher_fe)
    # 表(Markdown)
    print("| run | ステップ | 選べる | REC60 有効 | 1 周目平均 | 2 周目平均 | 最小すき間 [cm] | STD 単周 | STD フライング | ROB24 |")
    print("|---|---:|:-:|---:|---:|---:|---:|---:|---:|---:|")
    for r in out["runs"]:
        print(f"| {r['run']} | {r['step'] / 1e6:.1f}M | {'○' if r['eligible'] else '×'} | {r['rec60_valid']}/60 | {f3(r['lap1_mean'])} | {f3(r['lap2_mean'])} | "
              f"{f3(None if r['min_clearance'] is None else r['min_clearance'] * 100, 2)} | {f3(r['std_single'])} | {f3(r['std_flying'])} | {r['rob24']}/24 |")
    print()
    print(json.dumps(out["H1"], ensure_ascii=False, indent=1))
    print()
    keys = [("max_speed", "最高速", 2), ("speed_p99", "速さ 99%", 2), ("frac_over_4mps", "4 m/s 超", 3), ("lat_acc_max", "横加速度 最大", 2),
            ("lat_acc_p95", "横加速度 95%", 2), ("rho_gt1_frac", "ρ>1", 3), ("steer_cmd_rate_p95", "舵指令の速さ 95%", 2),
            ("steer_rate_p95", "舵角の速さ 95%", 2), ("steer_edge_frac", "舵が端", 3), ("throttle_pwm_p01", "スロットル 1%", 1), ("min_clearance", "最小すき間", 3)]
    print("| | " + " | ".join(k[1] for k in keys) + " |")
    print("|---|" + "---:|" * len(keys))
    for r in out["realism"]:
        print(f"| {r['who']} | " + " | ".join(f3(r.get(k), nd) for k, _, nd in keys) + " |")
    print("図:", FIG / "rl_lap_comparison.png")


if __name__ == "__main__":
    main()
