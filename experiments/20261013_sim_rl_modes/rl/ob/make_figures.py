"""障害物よけ(obstacle、強化学習モード(障害物よけ))の図(2026-10-14、obstacle の学習の段 r1)。

    python rl/ob/make_figures.py --run <train/obstacle/main> [--pilot <train/obstacle/pilot>] --traces <traces.json> \\
        [--variation <variation_pred.json>] [--nominal rl/nominal_line_fastest.json] --out <fig/>

図:
  obstacle_learning_curve.png  学習の曲線(学習中のエピソードの集計)と、チェックポイントごとの検証の組(27 回)の判定の値
  obstacle_topdown.png         上から見た線: recording 組の配置 5 つ + 障害物なし。障害物よけの方策(速さで色分け)と最速の方策(衝突に ×)
  obstacle_lines_s.png         記録のプランの予想(15 シーン)の R からの横ずれ e_y を s ごとに重ねたもの。橙の印 = そのシーンの障害物
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 11, "axes.titlecolor": "#0b0b0b",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb"})
BLUE, ORANGE, AQUA, GRAY, TEXT2, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e", "#e34948"
OBST = "#f08a24"
L = 26.12


def rolling(y, n=9):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - n + 1): i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def load_progress(run: Path):
    rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line.strip()]
    steps = np.array([r["steps"] for r in rows]) / 1e6
    ep = [r["episodes"] for r in rows]
    g = lambda k: np.array([e.get(k) if e.get(k) is not None else np.nan for e in ep], float)  # noqa: E731
    col = np.array([e.get("reasons", {}).get("collision", 0) / e["n"] if e.get("n") else np.nan for e in ep])
    obst = np.array([e.get("reward_parts_mean", {}).get("obstacle", np.nan) if e.get("n") else np.nan for e in ep])
    return steps, g, col, obst


def learning_curve(run: Path, pilot: Path | None, out: Path, pilot_label: str = "パイロット(100 万、結果に数えない)") -> None:
    steps, g, col, obst = load_progress(run)
    pil = load_progress(pilot) if pilot and (pilot / "progress.jsonl").exists() else None
    evals = []
    for p in sorted((run / "evals" / "obstacle").glob("step_*.json")):
        d = json.loads(p.read_text())
        j = d["judged"]
        o1 = j["O1_valid"]["value"].split("/")
        evals.append({"m": int(p.stem.split("_")[1]) / 1e6, "valid": int(o1[0]), "runs": int(o1[1]), "o2": j["O2_clearance"]["value"],
                      "o3": j["O3_lap1_median"]["value"], "o4": j["O4_empty"]["value"], "all": d["all_judged_pass"],
                      "n_pass": sum(1 for v in j.values() if v.get("pass") is True), "n_j": len(j)})
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.6))
    panels = [
        (axs[0, 0], g("survived_20s_frac"), "20 秒を走り切った回の割合(学習)", "割合", None, "survived_20s_frac"),
        (axs[0, 1], g("lap1_mean"), "1 周目の時間 [秒](学習、走り切った周)", "秒", 10.5, "lap1_mean"),
        (axs[0, 2], g("min_obstacle_clearance_mean") * 100, "足した障害物との最小すき間の平均 [cm](学習)", "cm", 8.0, "min_obstacle_clearance_mean"),
        (axs[0, 3], obst, "障害物の罰の和 / 回(学習)", "報酬", None, None),
    ]
    for ax, y, title, yl, ref, key in panels:
        if pil is not None and key is not None:
            ps, pg, _, _ = pil
            yy = pg(key) * (100 if key == "min_obstacle_clearance_mean" else 1)
            ax.plot(ps, rolling(yy), color=GRAY, linewidth=1.2, linestyle="--", label=pilot_label)
        ax.plot(steps, y, color=BLUE, alpha=0.25, linewidth=1)
        ax.plot(steps, rolling(y), color=BLUE, linewidth=2, label="本番")
        if ref is not None:
            ax.axhline(ref, color=GRAY, linewidth=1, linestyle=":")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("学習のステップ [百万]")
        ax.set_ylabel(yl)
    ax = axs[0, 0]
    ax.plot(steps, rolling(col), color=ORANGE, linewidth=2)
    ax.text(steps[-1], rolling(col)[-1], " 衝突で終了", color=TEXT2, fontsize=8, va="center")
    ax.text(steps[-1], rolling(g("survived_20s_frac"))[-1], " 走り切り", color=TEXT2, fontsize=8, va="center")
    ax.set_ylim(0, 1.02)
    if pil is not None:
        axs[0, 0].legend(frameon=False, fontsize=7, loc="center right")
    if evals:
        m = [e["m"] for e in evals]
        ax = axs[1, 0]
        ax.plot(m, [100 * e["valid"] / e["runs"] for e in evals], color=BLUE, marker="o", linewidth=2, markersize=6)
        ax.axhline(95, color=GRAY, linewidth=1, linestyle="--")
        ax.set_ylim(0, 102)
        ax.set_title(f"検証: 有効な回 [%]({evals[0]['runs']} 回、O1 ≥ 95%)", loc="left", fontsize=10)
        for e in evals:
            ax.annotate(f"{e['n_pass']}/{e['n_j']}", (e["m"], 100 * e["valid"] / e["runs"]), textcoords="offset points", xytext=(0, -14), ha="center",
                        fontsize=7, color=TEXT2)
        ax.text(0.02, 0.04, "点の下の数字 = 合格した判定の数", transform=ax.transAxes, fontsize=7, color=TEXT2)
        ax = axs[1, 1]
        ax.plot([e["m"] for e in evals if e["o2"] is not None], [e["o2"] * 100 for e in evals if e["o2"] is not None], color=BLUE, marker="o",
                linewidth=2, markersize=6)
        ax.axhline(3, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title("検証: 有効な回の最小すき間 [cm](O2 ≥ 3 cm)", loc="left", fontsize=10)
        ax = axs[1, 2]
        ax.plot([e["m"] for e in evals if e["o3"] is not None], [e["o3"] for e in evals if e["o3"] is not None], color=BLUE, marker="o",
                linewidth=2, markersize=6)
        ax.axhline(10.5, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title("検証: 1 周目の中央値 [秒](O3 ≤ 10.5 秒)", loc="left", fontsize=10)
        ax = axs[1, 3]
        ax.plot([e["m"] for e in evals if e["o4"] is not None], [e["o4"] for e in evals if e["o4"] is not None], color=BLUE, marker="o",
                linewidth=2, markersize=6)
        ax.axhline(9.0, color=GRAY, linewidth=1, linestyle="--")
        ax.axhline(8.092, color=ORANGE, linewidth=1, linestyle=":")
        ax.text(m[0], 8.10, "最速 8.09 秒", fontsize=7, color=ORANGE, va="bottom")
        ax.set_title("検証: 障害物なし 停止から 1 周 [秒](O4 ≤ 9 秒)", loc="left", fontsize=10)
        sel_p = run.parent / "selection.json"
        sel_m = int(json.loads(sel_p.read_text())["selected"].split("_")[1]) / 1e6 if sel_p.exists() else None
        for a in axs[1]:
            a.set_xlabel("チェックポイント [百万ステップ]")
            if sel_m is not None:
                a.axvline(sel_m, color=GRAY, linewidth=1, linestyle=":")
        if sel_m is not None:
            axs[1, 0].text(sel_m, 50, "選んだ方策 ", ha="right", fontsize=8, color=TEXT2)
    fig.suptitle(f"強化学習モード「障害物よけ」(obstacle)の学習(最速の方策 main 種 1 の 1,000 万から、種 61。ここまで {steps[-1]:.2f} 百万ステップ / 予定 15)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "obstacle_learning_curve.png", dpi=130)
    plt.close(fig)


def draw_course(ax, course) -> None:
    for s in course["surfaces"]:
        if s["id"] == "carpet":
            continue
        ax.add_patch(plt.Rectangle((s["x"] - s["w"] / 2, s["z"] - s["d"] / 2), s["w"], s["d"], color="#efeee9", zorder=0))
    for g in course["segments"]:
        ax.plot([g["a"][0], g["b"][0]], [g["a"][1], g["b"][1]], color="#3b3a37" if g["kind"] != "curtain" else "#8a8984",
                linewidth=1.3 if g["kind"] != "gate-foot" else 0.8, solid_capstyle="round", zorder=2)
    for d in course["disks"]:
        if d["kind"] in ("post-foot", "gate-post"):
            ax.add_patch(plt.Circle((d["x"], d["z"]), d["r"], color="#b9b8b2" if d["kind"] == "post-foot" else "#3b3a37", zorder=2, linewidth=0))
    ref = course.get("reference")
    if ref:
        ax.plot(ref["x"], ref["z"], color=GRAY, linewidth=0.5, linestyle=":", zorder=1)
    b = course["bounds"]
    ax.set_xlim(b["minX"], b["maxX"])
    ax.set_ylim(b["maxZ"], b["minZ"])  # z を下向きに(アプリの上から見た向き)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_obstacles(ax, obstacles) -> None:
    for o in obstacles:
        # 小さいので、見つけやすいように薄い輪を足す(大きさは実物ではない)
        ax.add_patch(plt.Circle((o["x"], o["z"]), 0.28, facecolor="none", edgecolor=OBST, linewidth=0.9, alpha=0.7, zorder=5))
        if o["shape"] == "post":
            ax.add_patch(plt.Circle((o["x"], o["z"]), o["r"], color=OBST, zorder=5, linewidth=0))
        else:
            h, ux, uz = o["length"] / 2, math.cos(o["yaw"]), -math.sin(o["yaw"])
            a = (o["x"] - h * ux, o["z"] - h * uz)
            b = (o["x"] + h * ux, o["z"] + h * uz)
            nx, nz = -uz * o["r"], ux * o["r"]
            ax.add_patch(plt.Polygon([(a[0] + nx, a[1] + nz), (b[0] + nx, b[1] + nz), (b[0] - nx, b[1] - nz), (a[0] - nx, a[1] - nz)], color=OBST,
                                     zorder=5, linewidth=0))
            for p in (a, b):
                ax.add_patch(plt.Circle(p, o["r"], color=OBST, zorder=5, linewidth=0))


def speed_line(ax, trace, cmap, norm, lw=2.0, frames=None):
    t = trace if frames is None else trace[:frames]
    pts = np.array([[q["x"], q["z"]] for q in t])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw, zorder=4)
    lc.set_array(np.array([q["v"] for q in t[1:]]))
    ax.add_collection(lc)
    return lc


def topdown(traces, out: Path) -> None:
    runs = {r["id"]: r for r in traces["runs"]}
    lay_ids = [r["id"][3:] for r in traces["runs"] if r["id"].startswith("ob_") and r["layout_id"]][:5]
    fig, axs = plt.subplots(2, 3, figsize=(20, 10.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#a9c9f0", "#2a78d6", "#0b3a75"])
    norm = plt.Normalize(0, 4.5)
    lc = None
    panels = [(f"ob_{lid}", f"fastest_{lid}") for lid in lay_ids] + [("ob_empty", "fastest_empty")]
    for ax, (oid, fid) in zip(axs.flat, panels):
        draw_course(ax, traces["course"])
        r, f = runs.get(oid), runs.get(fid)
        if r is None:
            ax.set_visible(False)
            continue
        draw_obstacles(ax, r["obstacles"])
        if f:
            ft = f["trace"]
            ax.plot([q["x"] for q in ft], [q["z"] for q in ft], color=ORANGE, linewidth=1.0, linestyle="--", zorder=3, alpha=0.9)
            if f["collisions"]:
                ax.scatter([ft[-1]["x"]], [ft[-1]["z"]], marker="x", s=70, color=RED, linewidths=2, zorder=7)
        lc = speed_line(ax, r["trace"], cmap, norm, lw=1.8)
        t0 = r["trace"][0]
        ax.scatter([t0["x"]], [t0["z"]], s=28, color="#0b0b0b", zorder=7)
        if r["collisions"]:
            ax.scatter([r["trace"][-1]["x"]], [r["trace"][-1]["z"]], marker="x", s=70, color="#0b0b0b", linewidths=2, zorder=7)
        ps = [p for p in r.get("passes", []) if p.get("completed")]
        side = "".join("左" if p["side"] == "left" else "右" for p in ps)
        if r["layout_id"]:
            pc = min((p["min_clearance_m"] for p in ps), default=None)
            sub = (f"{oid[3:]}(障害物 {len(r['obstacles'])} 個)  {'有効' if r['valid'] else r['reason']}  周 "
                   f"{', '.join(f'{x:.2f}' for x in r['laps']) or '—'} 秒\n通過 {len(ps)} 回(通った側 {side or '—'})"
                   + (f"  障害物との最小 {pc * 100:.1f} cm" if pc is not None else "")
                   + f"  全体の最小すき間 {(r['min_clearance'] or 0) * 100:.1f} cm")
            ftxt = f"最速: {f['reason']}(フレーム {f['frames']})" if f else ""
        else:
            sub = (f"障害物なし(STD の開始)  {'有効' if r['valid'] else r['reason']}  周 {', '.join(f'{x:.2f}' for x in r['laps'])} 秒\n"
                   f"最小すき間 {(r['min_clearance'] or 0) * 100:.1f} cm")
            ftxt = f"最速: 周 {', '.join(f'{x:.2f}' for x in f['laps'])} 秒" if f else ""
        ax.set_title(sub, loc="left", fontsize=8.5)
        ax.text(0.01, 0.01, ftxt, transform=ax.transAxes, fontsize=7.5, color=ORANGE)
    if lc is not None:
        cb = fig.colorbar(lc, ax=axs, shrink=0.5, pad=0.01)
        cb.set_label("障害物よけの方策の速さ [m/s]", color=TEXT2)
    fig.suptitle("上から見た線(元の sim-core.js・標準の物理・連続 20 秒)。橙 = 足した障害物、色の線 = 障害物よけ、橙の破線 = 最速の方策(× = 衝突)、● = 開始、橙の輪 = 障害物の場所の目印",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(out / "obstacle_topdown.png", dpi=125, bbox_inches="tight")
    plt.close(fig)


def lines_s(var, nominal, out: Path) -> None:
    sc = var.get("scene_traces", [])
    if not sc:
        return
    fig, axs = plt.subplots(2, 1, figsize=(15, 7.6), gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axs[0]
    pal = plt.get_cmap("tab20")
    for i, s in enumerate(sc):
        f = np.array(s["f"])
        s0 = f[0, 0]
        keep = f[:, 0] - s0 >= 0
        sm = np.mod(f[keep, 0], L)
        ey = f[keep, 1]
        brk = np.where(np.diff(sm) < 0)[0] + 1
        for seg in np.split(np.arange(len(sm)), brk):
            ax.plot(sm[seg], ey[seg] * 100, color=pal(i % 20), linewidth=0.9, alpha=0.85)
        for o in s["obstacles"]:
            w = (o["r"] + (o["length"] / 2 if o["length"] else 0)) * 100
            ax.errorbar([o["s"]], [o["ey"] * 100], yerr=[[w], [w]], fmt="s", color=pal(i % 20), markersize=5, markeredgecolor=OBST, markeredgewidth=1.2,
                        elinewidth=2.5, capsize=0, zorder=6)
    if nominal:
        ey = np.array([v if v is not None else np.nan for v in nominal["ey_mean"]])
        ss = np.arange(len(ey)) * nominal.get("ds_m", 0.05)
        ax.plot(ss, ey * 100, color="#0b0b0b", linewidth=1.6, linestyle="--", label="最速の方策のふだんの線(障害物を置く基準)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlim(0, L)
    ax.set_ylabel("R からの横ずれ e_y [cm](左が正)")
    ax.set_title(f"記録のプランの予想 {len(sc)} シーン(配置ごとに色を変えた)。■ と縦の棒 = そのシーンの障害物の位置と大きさ。"
                 f"VM3 = {var['policy_scenes']['VM3_m']['mean_abs_dev'] * 100:.1f} cm(判定 ≥ 4 cm)", loc="left", fontsize=10)
    ax = axs[1]
    for i, s in enumerate(sc):
        f = np.array(s["f"])
        sm = np.mod(f[:, 0], L)
        brk = np.where(np.diff(sm) < 0)[0] + 1
        for seg in np.split(np.arange(len(sm)), brk):
            ax.plot(sm[seg], f[seg, 2], color=pal(i % 20), linewidth=0.8, alpha=0.8)
    ax.set_xlim(0, L)
    ax.set_xlabel("R に沿った位置 s [m](mod 26.12 m、全部の周)")
    ax.set_ylabel("速さ [m/s]")
    fig.tight_layout()
    fig.savefig(out / "obstacle_lines_s.png", dpi=125)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, default=None)
    ap.add_argument("--pilot-label", default="パイロット(100 万、結果に数えない)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--traces", type=Path, default=None)
    ap.add_argument("--variation", type=Path, default=None)
    ap.add_argument("--nominal", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    learning_curve(a.run, a.pilot, a.out, a.pilot_label)
    if a.traces and a.traces.exists():
        topdown(json.loads(a.traces.read_text()), a.out)
    if a.variation and a.variation.exists():
        nominal = json.loads(a.nominal.read_text()) if a.nominal and a.nominal.exists() else None
        lines_s(json.loads(a.variation.read_text()), nominal, a.out)
    print("書いた:", sorted(p.name for p in a.out.glob("obstacle_*.png")))


if __name__ == "__main__":
    main()
