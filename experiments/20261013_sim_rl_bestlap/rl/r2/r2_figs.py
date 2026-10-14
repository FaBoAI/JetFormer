"""r2(手順の外の参考の追加実験)の図: 物理のずれのランダム化で学習を続けた方策の学習曲線と、ずれへの強さの表。2026-10-13

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python -I rl/r2/r2_figs.py

読むもの: rl/r2-dr-s11/progress.jsonl・evals/step_*.json(高速版のコアの標準の評価)、rl/r2/robust_eval.json(robust_eval.mjs)
書くもの: fig/rl_r2_dr_curves.png、fig/rl_r2_robustness.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze_rl as A  # noqa: E402

RL, FIG = A.RL, A.FIG
RUN = RL / "r2-dr-s11"
ROB = RL / "r2" / "robust_eval.json"
C_TEACHER, C_BASE, C_DR = "#eb6834", "#2a78d6", "#4a3aa7"
GOOD, CRIT = "#0ca30c", "#d03b3b"


def rolling(y, k=8):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def style(ax, title, ylabel, xlabel="学習を続けたステップ数 [百万](0 = 選んだ方策)"):
    ax.set_facecolor(A.SURFACE)
    ax.set_title(title, color=A.INK, fontsize=10.5, loc="left")
    ax.set_ylabel(ylabel, color=A.INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=A.INK2, fontsize=9)
    ax.grid(True, color=A.GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(A.GRID)
    ax.tick_params(colors=A.INK2, labelsize=8)


def curves(rob: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A._jp_font()
    P = [json.loads(l) for l in (RUN / "progress.jsonl").read_text().splitlines() if l.strip()]
    x = np.array([p["steps"] for p in P]) / 1e6
    nom = [p.get("episodes_nominal", {}).get("two_valid_laps_frac", np.nan) if p.get("episodes_nominal", {}).get("n") else np.nan for p in P]
    per = [p.get("episodes_perturbed", {}).get("two_valid_laps_frac", np.nan) if p.get("episodes_perturbed", {}).get("n") else np.nan for p in P]
    ret = [p["episodes"].get("return_mean", np.nan) for p in P]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=A.SURFACE)
    a0, a1, a2 = axes
    style(a0, "学習中の回: 20 秒で 2 周とも有効の割合(確率的な方策・8 更新の移動平均)", "割合")
    a0.plot(x, rolling(nom), color=C_BASE, lw=2, label="標準の物理の回(2 割)")
    a0.plot(x, rolling(per), color=C_DR, lw=2, label="物理をずらした回(8 割)")
    a0.set_ylim(0, 1.02)
    a0.legend(fontsize=8, frameon=False, labelcolor=A.INK2, loc="lower right")
    # 決定的な方策の標準の評価(高速版のコア、REC60)
    E = []
    base = json.loads((RL / "best" / "main" / "eval_fast.json").read_text())
    E.append((0.0, base))
    for f in sorted((RUN / "evals").glob("step_*.json")):
        E.append((int(f.stem.split("_")[1]) / 1e6, json.loads(f.read_text())))
    xs = [e[0] for e in E]
    l1 = [e[1]["REC60"]["summary"]["lap1"].get("mean") if e[1]["REC60"]["summary"]["lap1"].get("n") == 60 else None for e in E]
    l2 = [e[1]["REC60"]["summary"]["lap2"].get("mean") if e[1]["REC60"]["summary"]["lap2"].get("n") == 60 else None for e in E]
    style(a1, "標準の物理での周回時間(決定的な方策・REC60 の平均)", "秒")
    a1.plot(xs, l1, color=C_DR, lw=2, marker="o", ms=6, label="1 周目(停止から)")
    a1.plot(xs, l2, color=C_DR, lw=2, ls="--", marker="s", ms=6, mfc=A.SURFACE, label="2 周目(フライング)")
    for ref, txt in ((A.T_LAP1, f"教師 1 周目 {A.T_LAP1:.3f}"), (A.T_LAP2, f"教師 2 周目 {A.T_LAP2:.3f}")):
        a1.axhline(ref, color=C_TEACHER, lw=1.1, ls="-.")
        a1.text(0.01, ref, " " + txt, color=A.INK2, fontsize=8, va="bottom", transform=a1.get_yaxis_transform())
    for xv, v in zip(xs, l1):
        if v is not None:
            a1.text(xv, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=A.INK2)
    a1.set_ylim(7.6, 8.85)
    a1.legend(fontsize=8, frameon=False, labelcolor=A.INK2, loc="lower right")
    # ずれへの強さ(robust_eval の条件表とランダムな物理)
    style(a2, "物理のずれへの強さ(決定的な方策・STD の開始)", "2 周とも有効の割合")
    if rob:
        pts = []
        for name, C in rob["controllers"].items():
            if name == "RC 教師":
                continue
            step = 0.0 if name == "選んだ方策" else float(name.replace("DR ", "").replace("M", ""))
            g = C["grid_summary"]
            pts.append((step, g["both_valid"] / g["conditions"], C["random"]["in_range_r2"]["valid"] / 100, C["random"]["wide"]["valid"] / 100))
        pts.sort()
        px = [p[0] for p in pts]
        a2.plot(px, [p[1] for p in pts], color=C_DR, lw=2, marker="o", ms=6, label="条件の表 39 条件(単周と連続の両方)")
        a2.plot(px, [p[2] for p in pts], color=C_DR, lw=2, ls="--", marker="s", ms=6, mfc=A.SURFACE, label="ランダムな物理 100(学習の範囲)")
        a2.plot(px, [p[3] for p in pts], color=C_DR, lw=1.4, ls=":", marker="^", ms=6, mfc=A.SURFACE, label="ランダムな物理 100(範囲の外を含む広い範囲)")
        t = rob["controllers"].get("RC 教師")
        if t:
            g = t["grid_summary"]
            a2.axhline(g["both_valid"] / g["conditions"], color=C_TEACHER, lw=1.1, ls="-.")
            a2.text(0.01, g["both_valid"] / g["conditions"], f" 教師(条件の表){g['both_valid']}/{g['conditions']}", color=A.INK2, fontsize=8, va="bottom",
                    transform=a2.get_yaxis_transform())
        a2.set_ylim(0, 1.05)
        a2.legend(fontsize=7.5, frameon=False, labelcolor=A.INK2, loc="lower right")
    fig.suptitle("参考の追加実験 r2: 物理のずれのランダム化で、選んだ方策(main 種 1・1,000 万)から 500 万ステップ学習を続けた", color=A.INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "rl_r2_dr_curves.png", dpi=130, facecolor=A.SURFACE)
    plt.close(fig)


def robustness(rob: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A._jp_font()
    who = [("RC 教師", "RC 教師"), ("選んだ方策", "RL 選んだ方策"), (rob.get("final_name", "DR 5.0M"), "RL r2(ずれで学習)")]
    who = [w for w in who if w[0] in rob["controllers"]]
    grid = rob["controllers"][who[0][0]]["grid"]
    names = [g["name"] + ("" if g["in_range"] in (None, True) else "  *範囲の外") for g in grid]
    nrow, ncol = len(names), 2 * len(who)
    fig, ax = plt.subplots(figsize=(13.5, 0.30 * nrow + 1.8), facecolor=A.SURFACE)
    ax.set_facecolor(A.SURFACE)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -1.45)
    for j, (key, label) in enumerate(who):
        G = rob["controllers"][key]["grid"]
        for k, mode in enumerate(("single", "continuous")):
            col = 2 * j + k
            for i, g in enumerate(G):
                r = g[mode]
                ok = r.get("valid")
                if ok:
                    txt = f"✓ {r['laps'][0]:.2f}" if mode == "single" else f"✓ {r['laps'][0]:.2f} / {r['laps'][1]:.2f}"
                    fc, tc = "#e8f5e8", A.INK
                else:
                    n_ok = len(r.get("laps") or [])
                    txt = "× 衝突" + (f"({n_ok} 周のあと)" if n_ok else "") if r.get("reason") == "collision" else f"× {r.get('reason')}"
                    fc, tc = "#f9e3e3", CRIT
                ax.add_patch(plt.Rectangle((col - 0.48, i - 0.45), 0.96, 0.9, facecolor=fc, edgecolor=A.SURFACE, lw=2))
                ax.text(col, i, txt, ha="center", va="center", fontsize=7.3, color=tc)
            ax.text(col, -0.85, ("停止から 1 周 [秒]" if mode == "single" else "連続 20 秒: 1 周目 / 2 周目"), ha="center", va="center", fontsize=7.8, color=A.INK2)
        ax.text(2 * j + 0.5, -1.25, label, ha="center", va="center", fontsize=10, color=A.INK)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(names, fontsize=7.8, color=A.INK2)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    summ = "  ".join(f"{label}: 両方完走 {rob['controllers'][k]['grid_summary']['both_valid']}/{rob['controllers'][k]['grid_summary']['conditions']}" for k, label in who)
    ax.set_title("物理のずれの条件ごとの結果(STD の開始・高速版のコア。✓ = 33 ゲート・黒シート 1 回・衝突 0、× = 衝突など)\n" + summ
                 + "\n*範囲の外 = r2 の学習で引かなかった値", color=A.INK, fontsize=10, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "rl_r2_robustness.png", dpi=130, facecolor=A.SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rob = json.loads(ROB.read_text()) if ROB.exists() else None
    curves(rob)
    if rob:
        robustness(rob)
    print("図:", FIG / "rl_r2_dr_curves.png", FIG / "rl_r2_robustness.png")


if __name__ == "__main__":
    main()
