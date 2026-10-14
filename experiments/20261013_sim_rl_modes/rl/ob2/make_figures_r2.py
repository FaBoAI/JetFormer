"""障害物よけ r2 の図(2026-10-14 学習の段 obstacle r2)。

    python rl/ob2/make_figures_r2.py --run <train/obstacle/r2/main> --r1 <train/obstacle/main> --probe <train/obstacle/probe_explore> \\
        [--selection <train/obstacle/r2/selection.json>] [--traces <traces.json>] [--variation <variation_pred.json>] --out <fig/>

図:
  obstacle_r2_learning_curve.png  上の段 = 学習中(r2 の青、r1 の本番の灰の破線、r1 の下調べ σ 0.37 の灰の点線): 走り切り・終わり方(衝突・3 cm・ゲートの外し)・
                                  足した障害物との最小すき間・σ。下の段 = チェックポイントごとの選びの組 OB-V100-R2(青)と protocol の検証の組(灰): O1〜O4
  obstacle_r2_topdown.png         上から見た線(rl/ob/dump_traces.mjs の軌跡。rl/ob/make_figures.py の topdown を使う)
  obstacle_r2_lines_s.png         記録のプランの予想の e_y を s ごとに重ねたもの(rl/ob/make_figures.py の lines_s)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ob"))
import make_figures as F  # noqa: E402  (r1 の図の道具: フォント・色・コースの描き方)
import numpy as np  # noqa: E402

plt = F.plt
BLUE, ORANGE, GRAY, TEXT2, RED, AQUA = F.BLUE, F.ORANGE, F.GRAY, F.TEXT2, F.RED, F.AQUA


def prog(run: Path, offset: int = 0, run2: Path | None = None, offset2: int = 0):
    rows = [json.loads(x) for x in (run / "progress.jsonl").read_text().splitlines() if x.strip()]
    for r in rows:
        r["steps"] += offset
    if run2 is not None and (run2 / "progress.jsonl").exists():
        lim = offset2
        rows = [r for r in rows if r["steps"] <= lim]
        r2 = [json.loads(x) for x in (run2 / "progress.jsonl").read_text().splitlines() if x.strip()]
        for r in r2:
            r["steps"] += offset2
        rows += r2
    st = np.array([r["steps"] for r in rows]) / 1e6
    ep = [r["episodes"] for r in rows]
    frac = lambda k: np.array([e.get("reasons", {}).get(k, 0) / e["n"] if e.get("n") else np.nan for e in ep])  # noqa: E731
    g = lambda k: np.array([e.get(k) if e.get(k) is not None else np.nan for e in ep], float)  # noqa: E731
    sig = np.exp(np.array([r["log_std"] for r in rows]))
    return {"m": st, "surv": g("survived_20s_frac"), "col": frac("collision"), "clr": frac("clearance"), "gm": frac("gate_miss"), "slow": frac("slow"), "stall": frac("stall"),
            "obst_cl": g("min_obstacle_clearance_mean") * 100, "lap1": g("lap1_mean"), "sig": sig}


def evals(run: Path, sub: str, offset: int = 0):
    out = []
    for p in sorted((run / sub).glob("step_*.json")):
        d = json.loads(p.read_text())
        ok = [r for r in d.get("rows", []) if r.get("test") == "OB-V100-R2" and r.get("valid_episode")]
        nolap = sum(1 for r in ok if len(r["laps"]) < 1)
        j = d["judged"]
        o1 = str(j["O1_valid"]["value"]).split("/")
        out.append({"m": (int(p.stem.split("_")[1]) + offset) / 1e6, "pct": 100 * int(o1[0]) / int(o1[1]),
                    "pct_strict": 100 * (int(o1[0]) - nolap) / int(o1[1]) if d.get("rows") else None, "o2": j["O2_clearance"]["value"], "o3": j["O3_lap1_median"]["value"],
                    "o4": j["O4_empty"]["value"], "var6": (d.get("reported", {}).get("var6") or {}).get("max"), "all": d["all_judged_pass"]})
    return out


def learning_curve(run: Path, r1: Path | None, probe: Path | None, sel: dict | None, out: Path, run2: Path | None = None, offset2: int = 0) -> None:
    A = prog(run, 0, run2, offset2)
    B = prog(r1) if r1 and (r1 / "progress.jsonl").exists() else None
    C = prog(probe) if probe and (probe / "progress.jsonl").exists() else None
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.2))
    R = F.rolling
    ax = axs[0, 0]
    for X, ls, lab in ((B, "--", "r1 本番(σ 0.16・lr 1e-4)"), (C, ":", "r1 下調べ(σ 0.37・lr 3e-4)")):
        if X is not None:
            ax.plot(X["m"], R(X["surv"]), color=GRAY, linestyle=ls, linewidth=1.3, label=lab)
    ax.plot(A["m"], A["surv"], color=BLUE, alpha=.2, linewidth=1)
    ax.plot(A["m"], R(A["surv"]), color=BLUE, linewidth=2.2, label="r2")
    ax.set_ylim(0, 1.02)
    ax.set_title("20 秒を走り切った回の割合(学習中・雑音つき)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axs[0, 1]
    for k, c, lab in (("col", ORANGE, "衝突"), ("clr", RED, "すき間 3 cm 未満"), ("gm", GRAY, "ゲートの外し"), ("slow", AQUA, "進みの見張り(r2b だけ)"),
                      ("stall", "#9b6bd6", "止まる")):
        ax.plot(A["m"], R(A[k]), color=c, linewidth=2, label=lab)
    ax.set_ylim(0, 1.02)
    ax.set_title("r2: 回の終わり方の割合(学習中)", loc="left", fontsize=10)
    ax.text(0.99, 0.6, "残り = 20 秒を走り切り", transform=ax.transAxes, ha="right", fontsize=7.5, color=TEXT2)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[0, 2]
    if B is not None:
        ax.plot(B["m"], R(B["obst_cl"]), color=GRAY, linestyle="--", linewidth=1.3)
    if C is not None:
        ax.plot(C["m"], R(C["obst_cl"]), color=GRAY, linestyle=":", linewidth=1.3)
    ax.plot(A["m"], R(A["obst_cl"]), color=BLUE, linewidth=2)
    ax.axhline(3, color=GRAY, linewidth=1, linestyle=":")
    ax.set_title("足した障害物との最小すき間の平均 [cm](学習中、回ごと)", loc="left", fontsize=10)
    ax = axs[0, 3]
    ax.plot(A["m"], A["sig"][:, 0], color=BLUE, linewidth=2, label="舵 σ")
    ax.plot(A["m"], A["sig"][:, 1], color=AQUA, linewidth=2, label="速さ σ")
    if B is not None:
        ax.plot(B["m"], B["sig"][:, 0], color=GRAY, linestyle="--", linewidth=1.2, label="r1 本番の舵 σ")
    ax.set_title("方策の雑音 σ(行動 [−1, 1] の単位)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    for a in axs[0]:
        a.set_xlabel("学習のステップ [百万]")
    E2 = evals(run, "evals_r2")
    EP = evals(run, "evals/obstacle")
    if run2 is not None:
        E2 = [e for e in E2 if e["m"] * 1e6 <= offset2] + evals(run2, "evals_r2", offset2)
        EP = [e for e in EP if e["m"] * 1e6 <= offset2] + evals(run2, "evals/obstacle", offset2)
        for a in axs.flat:
            a.axvline(offset2 / 1e6, color=TEXT2, linewidth=0.8, linestyle="-.", alpha=0.6)
        axs[0, 0].text(offset2 / 1e6, 0.95, " 後半 r2b(進みの見張り・通り道の目標)", fontsize=7.5, color=TEXT2, va="top")
    sel_m = next((c["step"] for c in sel["candidates"] if c["checkpoint"] == sel["selected"]), None) / 1e6 if sel else None
    specs = [(axs[1, 0], "pct", "有効な回 [%](O1 ≥ 95%)", 95, 1), (axs[1, 1], "o2", "有効な回の最小すき間 [cm](O2 ≥ 3)", 3, 100),
             (axs[1, 2], "o3", "1 周目の中央値 [秒](O3 ≤ 10.5)", 10.5, 1), (axs[1, 3], "o4", "障害物なし 停止から 1 周 [秒](O4 ≤ 9.0)", 9.0, 1)]
    for ax, k, title, ref, mul in specs:
        for E, c, lab, mk in ((EP, GRAY, "protocol の検証の組(26 + 1 回)", "s"), (E2, BLUE, "選びの組 OB-V100-R2(100 + 1 回)", "o")):
            xs = [e["m"] for e in E if e[k] is not None]
            ys = [e[k] * mul for e in E if e[k] is not None]
            if xs:
                ax.plot(xs, ys, color=c, marker=mk, linewidth=2 if c == BLUE else 1.2, markersize=5, label=lab)
        if k == "o4":
            xs = [e["m"] for e in E2 if e["var6"] is not None and np.isfinite(e["var6"])]
            ax.plot(xs, [e["var6"] for e in E2 if e["var6"] is not None and np.isfinite(e["var6"])], color=AQUA, marker="^", linewidth=1.2,
                    markersize=5, label="±3 cm・±3° の 6 つの最大")
            ax.axhline(8.092, color=ORANGE, linewidth=1, linestyle=":")
        ax.axhline(ref, color=GRAY, linewidth=1, linestyle="--")
        if sel_m is not None:
            ax.axvline(sel_m, color=GRAY, linewidth=1, linestyle=":")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("チェックポイント [百万ステップ]")
    xs = [e["m"] for e in E2 if e.get("pct_strict") is not None]
    axs[1, 0].plot(xs, [e["pct_strict"] for e in E2 if e.get("pct_strict") is not None], color=BLUE, linestyle=":", marker=".", linewidth=1.2,
                   label="選びの組・1 周も回らない回を除く")
    axs[1, 0].set_ylim(0, 102)
    axs[1, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
    axs[1, 3].legend(frameon=False, fontsize=7.5)
    if sel_m is not None:
        axs[1, 0].text(sel_m, 8, "選んだ方策 ", ha="right", fontsize=8, color=TEXT2)
    fig.suptitle(f"強化学習モード「障害物よけ」r2 の学習(最速の方策から、種 62、正規化の直し・σ 0.37・lr 3e-4・近づく速さの罰・3 cm の終了"
                 + (f" → {offset2 / 1e6:.2f} 百万から後半 r2b" if run2 is not None else "") + f"。通しで {A['m'][-1]:.2f} 百万ステップ)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "obstacle_r2_learning_curve.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--run2", type=Path, default=None, help="後半(r2b)の学習の出力")
    ap.add_argument("--offset2", type=int, default=0, help="後半の通しのステップのずれ")
    ap.add_argument("--r1", type=Path, default=None)
    ap.add_argument("--probe", type=Path, default=None)
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--traces", type=Path, default=None)
    ap.add_argument("--variation", type=Path, default=None)
    ap.add_argument("--nominal", type=Path, default=HERE.parent / "nominal_line_fastest.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads(a.selection.read_text()) if a.selection and a.selection.exists() else None
    learning_curve(a.run, a.r1, a.probe, sel, a.out, a.run2, a.offset2)
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        if a.traces and a.traces.exists():
            F.topdown(json.loads(a.traces.read_text()), t)
            shutil.move(t / "obstacle_topdown.png", a.out / "obstacle_r2_topdown.png")
        if a.variation and a.variation.exists():
            F.lines_s(json.loads(a.variation.read_text()), json.loads(a.nominal.read_text()), t)
            if (t / "obstacle_lines_s.png").exists():
                shutil.move(t / "obstacle_lines_s.png", a.out / "obstacle_r2_lines_s.png")
    print("書いた:", sorted(p.name for p in a.out.glob("obstacle_r2_*.png")))


if __name__ == "__main__":
    main()
