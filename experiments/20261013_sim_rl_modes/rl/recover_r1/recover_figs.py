"""離脱復活(recover)の学習の段 r1 の図(2026-10-13)。

    cd ~/jetformer/experiments/20261013_sim_rl_modes
    python -I rl/recover_r1/recover_figs.py --run main [--selected <policy.json の名前の step>]

読むもの(~/jetformer_reports/sim_rl_modes_20261013/train/recover/ の下):
  <run>/progress.jsonl・pilot/progress.jsonl(学習中の回の集計)、<run>/evals/recover/step_*.json(検証の組、高速版)、
  eval_baseline_val.json(最速の方策 = 初期化元の検証の組。x = 0 の点)、figdata/traces_*.json(rl/recover_r1/dump_traces.mjs)、
  eval_test*.json(試験の組の結果。乱しの始まりの位置と成否)
書くもの(~/jetformer_reports/sim_rl_modes_20261013/fig/):
  recover_learning_curve.png(学習曲線と検証の組の判定の値)、recover_trajectories.png(上から見た走り。乱しの区間・衝突)、
  recover_event_map.png(乱しの始まりの位置と成否。試験の組)
色: 系列の色は役割で固定(最速の方策 = 橙、離脱復活の方策 = 青、パイロット = 灰)。乱しの区間 = 紫、衝突 = 赤の ×(状態の色、文字を添える)、戻った = 緑の ○。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
TR = REPORTS / "train" / "recover"
FIG = REPORTS / "fig"
EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
WORLD = json.loads((EXP / "snapshot/sim/world-v3.json").read_text())
REFLOOP = json.loads((EXP / "rl/runtime/reference_loop.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_REC, C_FAST, C_PILOT = "#2a78d6", "#eb6834", "#8a8984"
C_PERT, C_BAD, C_GOOD = "#4a3aa7", "#d03b3b", "#0ca30c"
LOOP_L = REFLOOP["length_m"]


def jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "VL Gothic", "Droid Sans Fallback"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            return name
    return None


def style(ax, title, ylabel, xlabel="環境のステップ数 [百万](0 = 初期化元の最速の方策)"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def load_progress(run: str) -> list[dict]:
    p = TR / run / "progress.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pooled(P: list[dict], key: str, k: int = 10) -> np.ndarray:
    """回の集計の割合 key を、直近 k 更新の回を合わせて割合にする(回が少なくても揺れにくい)。"""
    num = np.array([(p["episodes"].get(key) or 0) * p["episodes"].get("n", 0) for p in P], float)
    den = np.array([p["episodes"].get("n", 0) for p in P], float)
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        a, b = num[max(0, i - k + 1):i + 1].sum(), den[max(0, i - k + 1):i + 1].sum()
        out[i] = a / b if b >= 5 else np.nan
    return out


def rolling(y, k=10):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def frac(v: str) -> float:
    a, b = v.split("/")
    return int(a) / max(1, int(b))


def evals(run: str) -> list[tuple[float, dict]]:
    E = []
    b = TR / "eval_baseline_val.json"
    if b.exists():
        E.append((0.0, json.loads(b.read_text())))
    for f in sorted((TR / run / "evals" / "recover").glob("step_*.json")):
        E.append((int(f.stem.split("_")[1]) / 1e6, json.loads(f.read_text())))
    return E


def learning_curve(run: str, selected_step: float | None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jp_font()
    P, PP = load_progress(run), load_progress("pilot")
    E = evals(run)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2), facecolor=SURFACE)
    (a_surv, a_ret, a_ev), (a_c12, a_c5, a_c6) = axes
    x = np.array([p["steps"] for p in P]) / 1e6
    xp = np.array([p["steps"] for p in PP]) / 1e6
    if len(PP):
        a_surv.plot(xp, pooled(PP, "survived_20s_frac"), color=C_PILOT, lw=1.6, ls="--", label="パイロット(既定の値・100 万)")
        a_ret.plot(xp, rolling([p["episodes"].get("return_mean", np.nan) or np.nan for p in PP]), color=C_PILOT, lw=1.6, ls="--", label="パイロット")
    if len(P):
        a_surv.plot(x, pooled(P, "survived_20s_frac"), color=C_REC, lw=2, label="本番(調整後・種 31)")
        a_ret.plot(x, rolling([p["episodes"].get("return_mean", np.nan) or np.nan for p in P]), color=C_REC, lw=2, label="本番")
        a_ev.plot(x, rolling([p["episodes"].get("events_mean", np.nan) or np.nan for p in P]), color=C_REC, lw=2, label="本番")
        a_ev.axvspan(0, 3.0, color=C_PERT, alpha=0.07, lw=0)
        a_ev.text(1.5, 0.02, "乱しの強さ 0.5 → 1.0 倍\n(カリキュラム)", color=INK2, fontsize=8, ha="center", va="bottom", transform=a_ev.get_xaxis_transform())
        a_ev.set_ylim(0, None)
    style(a_surv, "学習中: 20 秒を走り切った回の割合(直近 10 更新)", "割合")
    a_surv.set_ylim(-0.02, 1.02)
    style(a_ret, "学習中: 1 回の報酬の平均(10 更新の移動平均)", "報酬")
    style(a_ev, "学習中: 1 回の乱しの出来事の数(終わった回)", "回数")
    # 検証の組(高速版): C1・C2
    if E:
        xe = np.array([e[0] for e in E])
        c1 = [frac(e[1]["judged"]["C1_valid"]["value"]) for e in E]
        c2 = [e[1]["judged"]["C2_success"]["value"] for e in E]
        a_c12.plot(xe, c1, color=C_REC, lw=2, marker="o", ms=5, label="C1 有効な回の割合")
        a_c12.plot(xe, c2, color=C_REC, lw=1.6, ls="--", marker="s", ms=5, mfc=SURFACE, label="C2 出来事の成功率")
        for e, c in ((0.90, "C1 の基準 0.90"), (0.95, "C2 の基準 0.95")):
            a_c12.axhline(e, color=INK2, lw=1, ls="-." if e == 0.9 else ":")
            a_c12.text(0.01, e, " " + c, color=INK2, fontsize=8, va="bottom", transform=a_c12.get_yaxis_transform())
        ky = [e[1]["judged"]["C5_kick"]["value"]["yaw_rate"] for e in E]
        kl = [e[1]["judged"]["C5_kick"]["value"]["lateral_rate"] for e in E]
        a_c5.plot(xe, ky, color=C_REC, lw=2, marker="o", ms=5, label="向きのキック 0.3 rad")
        a_c5.plot(xe, kl, color=C_REC, lw=1.6, ls="--", marker="s", ms=5, mfc=SURFACE, label="横のキック 0.15 m")
        a_c5.axhline(0.9, color=INK2, lw=1, ls="-.")
        a_c5.text(0.01, 0.9, " C5 の基準 0.90", color=INK2, fontsize=8, va="bottom", transform=a_c5.get_yaxis_transform())
        lap = [e[1]["judged"]["C6_unperturbed"]["value"]["lap_from_stop"] for e in E]
        a_c6.plot(xe, [v if v is not None else np.nan for v in lap], color=C_REC, lw=2, marker="o", ms=5, label="停止から 1 周(乱しなし)")
        a_c6.axhline(9.0, color=INK2, lw=1, ls="-.")
        a_c6.text(0.01, 9.0, " C6 の基準 9.0 秒", color=INK2, fontsize=8, va="bottom", transform=a_c6.get_yaxis_transform())
        a_c6.axhline(8.092, color=C_FAST, lw=1, ls=":")
        a_c6.text(0.99, 8.092, "最速の方策 8.092 秒 ", color=INK2, fontsize=8, va="bottom", ha="right", transform=a_c6.get_yaxis_transform())
        for ax in (a_c12, a_c5, a_c6):
            ys = [ln.get_ydata()[0] for ln in list(ax.lines) if len(ln.get_xdata()) == len(xe)]
            ax.plot([0] * len(ys), ys, marker="o", ms=9, color=C_FAST, zorder=5, linestyle="none", label="x = 0: 最速の方策(初期化元)")
            if selected_step is not None:
                ax.axvline(selected_step, color=C_REC, lw=1, alpha=0.5)
    style(a_c12, "検証の組(25 回・高速版): 有効な回と出来事の成功", "割合")
    a_c12.set_ylim(-0.02, 1.02)
    style(a_c5, "検証の組: 状態のキックのあとの生き残り(始めてよい位置の回)", "割合")
    a_c5.set_ylim(-0.02, 1.02)
    style(a_c6, "検証の組: 乱しなしの停止から 1 周(STD)", "秒")
    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="center right" if ax is a_c6 else "lower right")
    sel = f"(縦線 = 選んだチェックポイント {selected_step:.1f} M)" if selected_step is not None else ""
    fig.suptitle(f"強化学習モード(離脱復活)の学習 — PPO・初期化 main 種 1・1,000 万ステップ。上段 = 学習中の回(確率的な方策・学習の乱し)、下段 = 検証の組{sel}", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "recover_learning_curve.png"
    fig.savefig(out, dpi=125, facecolor=SURFACE)
    plt.close(fig)
    return out


def draw_track(ax, labels=False):
    from matplotlib.patches import Circle, Rectangle
    ax.set_facecolor(SURFACE)
    for s in WORLD["surfaces"]:
        if s["id"] == "carpet":
            continue
        col = {"shortcut": "#2b2b2b", "blue-mat": "#b7d3f6", "turf": "#cfe8cf", "ptfe": "#ece6f5"}.get(s["id"], "#e4e3df")
        ax.add_patch(Rectangle((s["x"] - s["w"] / 2, s["z"] - s["d"] / 2), s["w"], s["d"], facecolor=col, edgecolor="none", alpha=0.55, zorder=0))
        if labels:
            ax.text(s["x"], s["z"], {"shortcut": "黒シート", "blue-mat": "青マット", "turf": "芝", "ptfe": "PTFE"}.get(s["id"], s["id"]),
                    fontsize=6.5, color=INK2 if s["id"] != "shortcut" else SURFACE, ha="center", va="center", zorder=1)
    for g in WORLD["segments"]:
        ax.plot([g["a"][0], g["b"][0]], [g["a"][1], g["b"][1]], color="#6b6a66", lw=max(0.8, g["radius"] * 110), solid_capstyle="round", zorder=2)
    for d in WORLD["disks"]:
        ax.add_patch(Circle((d["x"], d["z"]), d["radius"], facecolor="#9b9a95" if d["radius"] > 0.05 else "#52514e", edgecolor="none", alpha=0.7, zorder=2))
    b = WORLD["bounds"]
    ax.set_xlim(b["minX"], b["maxX"])
    ax.set_ylim(b["maxZ"], b["minZ"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def draw_episode(ax, ep, color):
    tr = np.array(ep["trace"], float)
    if not len(tr):
        return
    ax.plot(tr[:, 0], tr[:, 1], color=color, lw=1.4, zorder=4, alpha=0.95)
    learn = tr[:, 3]
    # 乱しの区間(info.jf.l = 0 のフレーム)を太い紫で
    idx = np.nonzero(learn == 0)[0]
    if len(idx):
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for r in runs:
            j = np.arange(r[0], min(len(tr), r[-1] + 2))
            ax.plot(tr[j, 0], tr[j, 1], color=C_PERT, lw=4.2, solid_capstyle="round", zorder=5, alpha=0.85)
    for e in ep["events"]:
        ps = e["ps"]
        if ps < len(tr):
            ax.plot(tr[ps, 0], tr[ps, 1], marker="o", ms=6, color=C_PERT, mec=SURFACE, mew=1.2, zorder=6)
            ax.text(tr[ps, 0], tr[ps, 1] - 0.16, f"{e['n']}", fontsize=6.5, color=INK2, ha="center", va="bottom", zorder=7)
    ax.plot(tr[0, 0], tr[0, 1], marker=">", ms=7, color=INK, zorder=6)
    if ep["collision_xz"]:
        cx, cz = ep["collision_xz"]
        ax.plot(cx, cz, marker="X", ms=12, color=C_BAD, mec=SURFACE, mew=1.2, zorder=8)
        ax.text(cx, cz + 0.22, "衝突", fontsize=8, color=C_BAD, ha="center", va="top", zorder=8, fontweight="bold")


EV_JA = {"steer_pulse": "舵のパルス", "excursion": "わざと外す", "speed_pulse": "ブレーキ/踏み込み"}


def ep_title(ep):
    t = ep["frames"] / 30
    ev = "・".join(f"{e['n']}:{EV_JA.get(e['type'], e['type'])}({e['cls']})" for e in ep["events"])
    res = "20 秒走り切り(有効)" if ep["valid"] else f"{t:.1f} 秒で衝突"
    return f"{ep['id']}: {res}\n{ev}"


def trajectories(rec_json: Path, fast_json: Path, ids: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jp_font()
    R = json.loads(rec_json.read_text())
    F = json.loads(fast_json.read_text())
    byid = lambda D: {e["id"]: e for e in D["episodes"]}  # noqa: E731
    r, f = byid(R), byid(F)
    fig, axes = plt.subplots(2, len(ids), figsize=(5.6 * len(ids), 9.0), facecolor=SURFACE)
    for k, i in enumerate(ids):
        for row, (D, col, name) in enumerate(((f, C_FAST, "最速の方策(初期化元)"), (r, C_REC, "離脱復活の方策"))):
            ax = axes[row][k]
            draw_track(ax, labels=(k == 0))
            if i in D:
                draw_episode(ax, D[i], col)
                ax.set_title(f"{name} — " + ep_title(D[i]), color=INK, fontsize=8.5, loc="left")
    fig.suptitle("試験の組 RC-P100 の同じ開始・同じ乱しの予定(元の sim-core.js)。紫の太線 = 乱しの区間(学習から外すフレーム)、紫の点と番号 = 乱しの始まり、▶ = 開始",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "recover_trajectories.png"
    fig.savefig(out, dpi=115, facecolor=SURFACE)
    plt.close(fig)
    return out


def loop_xy(s: float):
    xs, zs, ss = REFLOOP["x"], REFLOOP["z"], REFLOOP["s"]
    s = s % LOOP_L
    i = int(np.searchsorted(ss, s, side="right") - 1)
    j = (i + 1) % len(xs)
    s1 = ss[j] if j > 0 else LOOP_L
    u = (s - ss[i]) / max(1e-9, s1 - ss[i])
    return xs[i] + (xs[j] - xs[i]) * u, zs[i] + (zs[j] - zs[i]) * u


def event_map(tests: list[tuple[str, Path, str]]) -> Path:
    """試験の組 RC-P100 の全出来事: 乱しの始まりの位置(R 上の点)と成否。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jp_font()
    fig, axes = plt.subplots(1, len(tests), figsize=(8.2 * len(tests), 6.0), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, path, col) in zip(axes, tests):
        d = json.loads(path.read_text())
        rows = d["tests"]["RC-P100"]
        draw_track(ax, labels=True)
        xs, zs = REFLOOP["x"], REFLOOP["z"]
        ax.plot(list(xs) + [xs[0]], list(zs) + [zs[0]], color=GRID, lw=1, zorder=3)
        ok = bad = 0
        rng = np.random.default_rng(0)
        for r in rows:
            for ev, rc in zip(r.get("events", []), r.get("recovery", [])):
                x, z = loop_xy(ev["s"])
                jx, jz = rng.normal(0, 0.05, 2)
                if rc["success"]:
                    ok += 1
                    ax.plot(x + jx, z + jz, marker="o", ms=5, color=C_GOOD, alpha=0.55, mec="none", zorder=5)
                else:
                    bad += 1
                    ax.plot(x + jx, z + jz, marker="X", ms=8, color=C_BAD, alpha=0.85, mec=SURFACE, mew=0.6, zorder=6)
        J = d["judged"]
        ax.set_title(f"{name}: 出来事 {ok + bad}・成功 {ok}(緑の ○)・失敗 {bad}(赤の ×)\nC1 有効な回 {J['C1_valid']['value']}・C2 成功率 {J['C2_success']['value']:.3f}",
                     color=INK, fontsize=10, loc="left")
    fig.suptitle("試験の組 RC-P100(100 回・元の sim-core.js): 乱しの始まりの位置(参照ループ R 上の点。重なりを避けて少しずらして描く)",
                 color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "recover_event_map.png"
    fig.savefig(out, dpi=115, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--selected", type=float, default=None, help="選んだチェックポイントのステップ数 [百万]")
    ap.add_argument("--traces-rec", default=str(TR / "figdata" / "traces_selected.json"))
    ap.add_argument("--traces-fast", default=str(TR / "figdata" / "traces_fastest.json"))
    ap.add_argument("--ids", default="P0,P1,P2")
    ap.add_argument("--test-rec", default=str(TR / "eval_test.json"))
    ap.add_argument("--test-fast", default=str(REPORTS / "eval" / "recover" / "baseline_fastest_test.json"))
    a = ap.parse_args()
    print(learning_curve(a.run, a.selected))
    if Path(a.traces_rec).exists() and Path(a.traces_fast).exists():
        print(trajectories(Path(a.traces_rec), Path(a.traces_fast), a.ids.split(",")))
    tests = [("最速の方策(初期化元)", Path(a.test_fast), C_FAST)]
    if Path(a.test_rec).exists():
        tests.append(("離脱復活の方策", Path(a.test_rec), C_REC))
    print(event_map(tests))


if __name__ == "__main__":
    main()
