"""r3(確認 r2 のあと。手順の外の参考の追加実験)の図。2026-10-13

    cd ~/jetformer/experiments/20261013_sim_rl_bestlap
    python -I rl/r3/r3_figs.py

読むもの: rl/r2-dr-s11・rl/r3-s21・rl/r3-s22 の progress.jsonl と evals/step_*.json(高速版のコアの標準の評価)、
          rl/r3/indep/*.json(確認 r2 の道具で測った角・2 次元の格子など)
書くもの: fig/rl_r3_curves.png(学習曲線)、fig/rl_r3_corner.png(角と 2 次元の格子)、fig/rl_r3_tradeoff.png(速さと強さ)
色: 系列の色は役割で固定(教師 = 橙、選んだ方策 = 青、r2 = 紫、r3 = 緑青)。r3 の 2 本は同じ色で、種は線の形で分ける。
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
C_TEACHER, C_BASE, C_R2, C_R3 = "#eb6834", "#2a78d6", "#4a3aa7", "#1baf7a"
RUNS = [("r2-dr-s11", "r2(一様なランダム化)", C_R2, "-"), ("r3-s21", "r3 種 21(角を多めに + すき間の約束)", C_R3, "-"),
        ("r3-s22", "r3 種 22(同じ設定)", C_R3, "--")]
INDEP = RL / "r3" / "indep"


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


def load_progress(run: str) -> list[dict]:
    p = RL / run / "progress.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pooled(P: list[dict], key: str, k: int = 12) -> np.ndarray:
    """回の集計 key(episodes_*)の『2 周とも有効』を、直近 k 更新の回を合わせて割合にする(回の少ない集計でも揺れにくい)。"""
    ok = np.array([(p.get(key) or {}).get("two_valid_laps_frac", 0) * (p.get(key) or {}).get("n", 0) for p in P], float)
    n = np.array([(p.get(key) or {}).get("n", 0) for p in P], float)
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        s, m = ok[max(0, i - k + 1):i + 1].sum(), n[max(0, i - k + 1):i + 1].sum()
        out[i] = s / m if m >= 5 else np.nan
    return out


def rolling(y, k=12):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def det_evals(run: str) -> list[tuple[float, dict]]:
    E = [(0.0, json.loads((RL / "best" / "main" / "eval_fast.json").read_text()))]
    for f in sorted((RL / run / "evals").glob("step_*.json")):
        E.append((int(f.stem.split("_")[1]) / 1e6, json.loads(f.read_text())))
    return E


def curves() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A._jp_font()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.0), facecolor=A.SURFACE)
    (a_nom, a_cor, a_ret), (a_lap, a_clr, a_end) = axes
    style(a_nom, "学習中の回: 標準の物理の回で 20 秒・2 周とも有効の割合", "割合")
    style(a_cor, "学習中の回: 遅い舵(≤ 1.0)× 弱いブレーキ(≤ 0.9)の回で 2 周とも有効", "割合")
    style(a_ret, "学習中の回: 回の報酬の平均(全部の回)", "報酬")
    style(a_lap, "決定的な方策・標準の物理: REC60 の 1 周目の平均", "秒")
    style(a_clr, "決定的な方策・標準の物理: REC60 の最小すき間", "cm")
    style(a_end, "学習中の回の終わり方(r3 の 2 本の平均)", "割合")
    for run, label, color, ls in RUNS:
        P = load_progress(run)
        if not P:
            continue
        x = np.array([p["steps"] for p in P]) / 1e6
        a_nom.plot(x, pooled(P, "episodes_nominal"), color=color, ls=ls, lw=2, label=label)
        if run.startswith("r3"):
            a_cor.plot(x, pooled(P, "episodes_slow_steer_weak_brake", k=40), color=color, ls=ls, lw=2, label=label)
        a_ret.plot(x, rolling([p["episodes"].get("return_mean", np.nan) for p in P]), color=color, ls=ls, lw=2, label=label)
        E = det_evals(run)
        xs = [e[0] for e in E]
        l1 = [e[1]["REC60"]["summary"]["lap1"].get("mean") if e[1]["REC60"]["summary"]["lap1"].get("n") == 60 else np.nan for e in E]
        cl = [100 * e[1]["REC60"]["summary"]["min_clearance"] for e in E]
        a_lap.plot(xs, l1, color=color, ls=ls, lw=2, marker="o", ms=6, mfc=color if ls == "-" else A.SURFACE, label=label)
        a_clr.plot(xs, cl, color=color, ls=ls, lw=2, marker="o", ms=6, mfc=color if ls == "-" else A.SURFACE, label=label)
    a_cor.text(0.02, 0.95, "r2 は回ごとの物理を記録していないので線が無い。\nr2 の学習では、この角の回は全体の約 1%(r3 は約 5〜6%)", transform=a_cor.transAxes,
               fontsize=8, color=A.INK2, va="top")
    for ax in (a_nom, a_cor):
        ax.set_ylim(0, 1.02)
    a_lap.axhline(A.T_LAP1, color=C_TEACHER, lw=1.1, ls="-.")
    a_lap.text(0.01, A.T_LAP1, f" 教師 {A.T_LAP1:.3f}", color=A.INK2, fontsize=8, va="bottom", transform=a_lap.get_yaxis_transform())
    a_clr.axhline(3.0, color=A.INK2, lw=1.1, ls=":")
    a_clr.text(0.01, 3.0, " 手順の条件 3 cm", color=A.INK2, fontsize=8, va="bottom", transform=a_clr.get_yaxis_transform())
    a_clr.set_ylim(0, None)
    # 終わり方(r3 の 2 本を合わせる)
    reasons = ["collection_duration_complete", "clearance", "collision", "gate_miss", "stall", "wrong_way"]
    names = {"collection_duration_complete": "20 秒を走り切った", "clearance": "すき間の約束(3 cm 未満)", "collision": "衝突", "gate_miss": "ゲート逃し",
             "stall": "停止", "wrong_way": "逆走"}
    cols = {"collection_duration_complete": C_R3, "clearance": "#eda100", "collision": "#e34948", "gate_miss": "#4a3aa7", "stall": "#9b9a96", "wrong_way": "#e87ba4"}
    PP = [load_progress(r) for r in ("r3-s21", "r3-s22")]
    PP = [P for P in PP if P]
    if PP:
        m = min(len(P) for P in PP)
        x = np.array([p["steps"] for p in PP[0][:m]]) / 1e6
        tot = np.zeros(m)
        cnt = {r: np.zeros(m) for r in reasons}
        for P in PP:
            for i, p in enumerate(P[:m]):
                rs = p["episodes"].get("reasons", {})
                tot[i] += sum(rs.values())
                for r in reasons:
                    cnt[r][i] += rs.get(r, 0)
        for r in reasons:
            y = rolling(cnt[r] / np.maximum(tot, 1), k=20)
            if np.nanmax(y) > 0.005:
                a_end.plot(x, y, color=cols[r], lw=2, label=names[r])
        a_end.set_ylim(0, 1.0)
        a_end.legend(fontsize=8, frameon=False, labelcolor=A.INK2, loc="center right")
    for ax in (a_nom, a_ret, a_lap, a_clr):
        ax.legend(fontsize=8, frameon=False, labelcolor=A.INK2, loc="lower right")
    a_cor.legend(fontsize=8, frameon=False, labelcolor=A.INK2, loc="lower right")
    fig.suptitle("参考の追加実験 r3: 選んだ方策(main 種 1・1,000 万)から 500 万ステップ学習を続けた(r2 と r3 の比べ。確率的な方策の値は 12 更新の移動平均)",
                 color=A.INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "rl_r3_curves.png", dpi=130, facecolor=A.SURFACE)
    plt.close(fig)


def load_indep() -> dict:
    out = {}
    for f in sorted(INDEP.glob("*.json")):
        if f.name.startswith("sanity"):
            continue
        d = json.loads(f.read_text())
        for k, v in d.get("controllers", {}).items():
            out.setdefault(k, {}).update(v)
    return out


LABELS = [("teacher", "RC 教師", C_TEACHER), ("selected", "選んだ方策(r1)", C_BASE), ("r2_5M", "r2 500 万", C_R2), ("r3_s21", "r3 種 21", C_R3), ("r3_s22", "r3 種 22", C_R3)]


def corner(ind: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    A._jp_font()
    who = [w for w in LABELS if w[0] in ind and "corner" in ind[w[0]]]
    if not who:
        return
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#cfe0f6", "#7fb0ec", "#2a78d6", "#0d3f86"])
    fig, axes = plt.subplots(2, len(who), figsize=(3.5 * len(who) + 1, 8.4), facecolor=A.SURFACE, squeeze=False)
    for j, (key, label, _) in enumerate(who):
        for i, (setname, title) in enumerate((("corner", "角の 20 マス(遅延 0)"), ("grid2d", "舵×ブレーキの 25 マス(遅延 0)"))):
            ax = axes[i][j]
            ax.set_facecolor(A.SURFACE)
            G = ind[key].get(setname)
            if not G:
                ax.axis("off")
                continue
            S, B = G["steer"], G["brake"]
            M = np.zeros((len(S), len(B)))
            for c in G["cells"]:
                M[S.index(c["steerRate"]), B.index(c["brake"])] = c["rec60_valid"] / c["rec60_n"]
            ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for a in range(len(S)):
                for b in range(len(B)):
                    c = next(c for c in G["cells"] if c["steerRate"] == S[a] and c["brake"] == B[b])
                    v = c["rec60_valid"]
                    ax.text(b, a, f"{v}", ha="center", va="center", fontsize=9, color="#ffffff" if M[a, b] > .6 else A.INK)
            # 学習の範囲の角(舵 ≤ 0.95 × ブレーキ 0.8)を枠で示す
            ax.set_xticks(range(len(B)))
            ax.set_xticklabels([f"{b:g}" for b in B], fontsize=8, color=A.INK2)
            ax.set_yticks(range(len(S)))
            ax.set_yticklabels([f"{s:g}" for s in S], fontsize=8, color=A.INK2)
            ax.set_xlabel("ブレーキ [m/s²]", fontsize=8.5, color=A.INK2)
            if j == 0:
                ax.set_ylabel("舵の速さ [rad/s]", fontsize=8.5, color=A.INK2)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{label} — {title}\n60/60 のマス {G['cells_60']}/{len(G['cells'])}・合計 {G['total_valid']}/{G['total_n']}", fontsize=9.5, color=A.INK, loc="left")
            ax.set_xticks(np.arange(-.5, len(B), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(S), 1), minor=True)
            ax.grid(which="minor", color=A.SURFACE, linewidth=2)
            ax.tick_params(which="minor", length=0)
    fig.suptitle("物理のずれの 2 次元の格子: 各マスは REC60 の 60 の開始のうち、連続 20 秒の 2 周とも有効だった本数(元の sim-core.js・確認 r2 の道具。ほかの物理は標準・遅延 0)\n"
                 "上: 確認 r2 と同じ角の 20 マス。下: 舵 × ブレーキの平面(舵の速さ 0.85〜1.7・ブレーキ 0.8〜2.0、ほかは標準・遅延 0)の 25 マス。色が濃いほど多い(0〜60)。"
                 "遅延 1 では角は直っていない(確認 r3: 舵 0.85 × ブレーキ 0.8 で r3 種 21 は 23/60、種 22 は 29/60)",
                 color=A.INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "rl_r3_corner.png", dpi=130, facecolor=A.SURFACE)
    plt.close(fig)


def tradeoff(ind: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A._jp_font()
    pts = []
    for key, label, color in LABELS:
        R = ind.get(key)
        if not R or "grid2d" not in R:
            continue
        lap = R.get("samecond", {}).get("delay0", {}).get("lap")
        if key == "teacher":
            lap = A.T_SINGLE
        if lap is None:
            continue
        g = R["grid2d"]
        pts.append((key, label, color, lap, g["total_valid"] / g["total_n"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=A.SURFACE)
    style(ax, "速さと、物理のずれへの強さ(決定的な方策)", "舵 × ブレーキの平面の 25 マス(ほかは標準・遅延 0)で有効だった割合(REC60 の開始 × 25 マス)",
          xlabel="標準の物理の STD の停止から 1 周 [秒](右ほど速い)")
    for key, label, color, lap, rob in pts:
        ax.scatter([lap], [rob], s=90, color=color, edgecolor=A.SURFACE, linewidth=2, zorder=3,
                   marker="s" if key == "r3_s22" else "o")
        left = key == "r3_s21"   # r3 の 2 本は近いので、種 21 の文字は点の左に置く
        ax.annotate(f"{label}\n{lap:.3f} 秒・{100 * rob:.1f}%", (lap, rob), textcoords="offset points", xytext=(-10, -26) if left else (8, 6),
                    ha="right" if left else "left", fontsize=8.5, color=A.INK)
    ax.axvline(8.831, color=A.INK2, lw=1, ls=":")
    ax.text(8.831, 0.02, " H1 の閾値 8.831 秒", color=A.INK2, fontsize=8, rotation=90, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG / "rl_r3_tradeoff.png", dpi=130, facecolor=A.SURFACE)
    plt.close(fig)


def trajectory() -> None:
    """上から見た軌跡(STD の停止から 1 周・標準の物理・元の sim-core.js): 教師・選んだ方策・r2 の 500 万・r3 の 2 本。右は黒シートの入口の拡大。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    A._jp_font()
    srcs = [("teacher", "RC 教師", C_TEACHER, "-", RL / "final" / "final_eval_main.json", "teacher"),
            ("selected", "選んだ方策(r1)", C_BASE, "-", RL / "final" / "final_eval_main.json", "rl"),
            ("r2", "r2 500 万", C_R2, "-", RL / "r3" / "r2_5M" / "final_eval.json", "rl"),
            ("r3_s21", "r3 種 21", C_R3, "-", RL / "r3" / "r3-s21" / "final_eval.json", "rl"),
            ("r3_s22", "r3 種 22", C_R3, "--", RL / "r3" / "r3-s22" / "final_eval.json", "rl")]
    lines = []
    for key, label, color, ls, fe_path, part in srcs:
        if not fe_path.exists():
            continue
        fe = json.loads(fe_path.read_text())
        single = fe[part]["STD"]["single"]
        tr = json.loads(Path(single["record"]).read_text())["trace"]
        lines.append((key, label, color, ls, single.get("single_lap_s"), tr))
    if not lines:
        return
    world = json.loads((A.EXP / "sim" / "world-v3.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), facecolor=A.SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
    for ax in axes:
        ax.set_facecolor(A.SURFACE)
        for sf in world["surfaces"]:
            if sf["id"] == "carpet":
                continue
            col = {"shortcut": "#2b2b2b", "blue-mat": "#b7d3f6", "turf": "#cfe8cf", "ptfe": "#ece6f5"}.get(sf["id"], "#e4e3df")
            ax.add_patch(Rectangle((sf["x"] - sf["w"] / 2, sf["z"] - sf["d"] / 2), sf["w"], sf["d"], facecolor=col, edgecolor="none", alpha=0.45, zorder=0))
        for g in world["segments"]:
            ax.plot([g["a"][0], g["b"][0]], [g["a"][1], g["b"][1]], color="#6b6a66", lw=max(0.8, g["radius"] * 120), solid_capstyle="round", zorder=2)
        for d in world["disks"]:
            if d.get("kind") == "post-foot":   # 脚は sim と同じ 0.16 × 0.15 m の長方形
                ax.add_patch(Rectangle((d["x"] - .08, d["z"] - .075), .16, .15, facecolor="#52514e", edgecolor="none", alpha=0.75, zorder=2))
            else:
                ax.add_patch(Circle((d["x"], d["z"]), d["radius"], facecolor="#9b9a95", edgecolor="none", alpha=0.7, zorder=2))
        for i, pnt in enumerate(world["route"]):
            ax.plot(pnt[0], pnt[1], marker="o", ms=3, color=A.INK2, zorder=3)
            if ax is axes[0]:
                ax.text(pnt[0], pnt[1] + 0.08, str(i), fontsize=6, color=A.INK2, ha="center", zorder=3)
        for key, label, color, ls, lap, tr in lines:
            ax.plot([t["x"] for t in tr], [t["z"] for t in tr], color=color, ls=ls, lw=1.6, zorder=4,
                    label=f"{label} {lap:.3f} 秒" if lap else label)
        ax.set_aspect("equal")
        ax.tick_params(colors=A.INK2, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(A.GRID)
    b = world["bounds"]
    axes[0].set_xlim(b["minX"], b["maxX"]); axes[0].set_ylim(b["maxZ"], b["minZ"])
    axes[0].set_title("停止から 1 周(標準の開始・標準の物理・元の sim-core.js)。数字はゲートの番号", color=A.INK, fontsize=11, loc="left")
    axes[0].legend(fontsize=9, frameon=False, labelcolor=A.INK2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    axes[0].set_xlabel("x [m]", color=A.INK2); axes[0].set_ylabel("z [m](上が −z)", color=A.INK2)
    axes[1].set_xlim(1.2, 3.2); axes[1].set_ylim(1.0, -0.9)
    axes[1].annotate("脚 disk-70 (2.36, 0)。r2 の 500 万は REC60 の 16/60 本で\n車の中心 (2.18, 0.14) 付近でこの脚に 3 cm 未満まで寄った", (2.36, 0.0), xytext=(1.25, 0.8), fontsize=8, color=A.INK,
                     arrowprops=dict(arrowstyle="->", color=A.INK2, lw=0.8))
    axes[1].set_title("拡大: 黒シートの入口(ゲート 9〜11 付近)", color=A.INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "rl_r3_trajectory.png", dpi=130, facecolor=A.SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # 2026-10-14(報告の段): 確認 r3 の must-fix 1 で、「学習の範囲全体の 25 マス」を「舵 × ブレーキの平面・ほかは標準・遅延 0」と分かる名前にした。
    # --only-corner で角と速さの図だけを作り直す(数字は変えていない)
    if "--only-corner" not in sys.argv:
        curves()
    ind = load_indep()
    corner(ind)
    tradeoff(ind)
    if "--only-corner" not in sys.argv:
        trajectory()
    print("図:", FIG / "rl_r3_curves.png", FIG / "rl_r3_corner.png", FIG / "rl_r3_tradeoff.png", FIG / "rl_r3_trajectory.png")


if __name__ == "__main__":
    main()
