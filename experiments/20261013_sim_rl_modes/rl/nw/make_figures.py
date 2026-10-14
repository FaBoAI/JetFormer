"""no_wall(強化学習モード「絶対壁にぶつからない」)の図(2026-10-14、no_wall の学習の段)。

    python rl/nw/make_figures.py --run <train/no_wall/main> --traces <traces.json> --out <fig/>

図:
  no_wall_learning_curve.png   学習の曲線(学習中のエピソードの集計)と、チェックポイントごとの検証の組(シールド v2)の判定の値
  no_wall_topdown.png          上から見た線: no_wall(標準の物理、速さで色分け)と最速の線。シールドが替えたフレームに印
  no_wall_topdown_variation.png 物理をずらした回・摩擦円・いろいろな開始の線を重ねたもの
  no_wall_clearance_s.png      R に沿った位置 s ごとのすき間(1 周目): no_wall と最速、8 cm の線
"""
from __future__ import annotations

import argparse
import json
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
BLUE, ORANGE, AQUA, GRAY, TEXT2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"


def rolling(y, n=9):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - n + 1): i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def learning_curve(run: Path, out: Path) -> None:
    rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line.strip()]
    steps = np.array([r["steps"] for r in rows]) / 1e6
    ep = [r["episodes"] for r in rows]
    g = lambda k: np.array([e.get(k) if e.get(k) is not None else np.nan for e in ep], float)  # noqa: E731
    clear_term = np.array([e.get("reasons", {}).get("clearance", 0) / e["n"] if e.get("n") else np.nan for e in ep])
    evals = []
    for p in sorted((run / "evals" / "no_wall").glob("step_*.json")):
        d = json.loads(p.read_text())
        j = d["judged"]
        n1 = j["N1_no_collision"]["value"].split("/")
        evals.append({"m": int(p.stem.split("_")[1]) / 1e6, "nocol": int(n1[0]) / int(n1[1]), "n2": j["N2_clearance_nominal"]["value"],
                      "n3": j["N3_clearance_randomised"]["value"], "rq": j["N6_friction"]["value"]["rho_req_gt1"],
                      "rc": j["N6_friction"]["value"]["rho_comb_gt1"], "n7": j["N7_shield"]["value"], "all": d["all_judged_pass"],
                      "n_pass": sum(1 for v in j.values() if v.get("pass") is True), "n_j": len(j)})
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.6))
    panels = [
        (axs[0, 0], g("survived_20s_frac"), "20 秒を走り切った回の割合(学習)", "割合", None),
        (axs[0, 1], g("lap1_mean"), "1 周目の時間 [秒](学習)", "秒", None),
        (axs[0, 2], g("min_clearance_mean") * 100, "回の最小すき間の平均 [cm](学習)", "cm", 8.0),
        (axs[0, 3], g("rho_comb_gt1_frac") * 100, "ρ_comb > 1 のフレーム [%](学習)", "%", 2.0),
    ]
    for ax, y, title, yl, ref in panels:
        ax.plot(steps, y, color=BLUE, alpha=0.25, linewidth=1)
        ax.plot(steps, rolling(y), color=BLUE, linewidth=2)
        if ref is not None:
            ax.axhline(ref, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel("学習のステップ [百万]")
        ax.set_ylabel(yl)
    ax = axs[0, 0]
    ax.plot(steps, rolling(clear_term), color=ORANGE, linewidth=2)
    ax.text(steps[-1], rolling(clear_term)[-1], " すき間の約束で終了", color=TEXT2, fontsize=8, va="center")
    ax.text(steps[-1], rolling(g("survived_20s_frac"))[-1], " 走り切り", color=TEXT2, fontsize=8, va="center")
    if evals:
        m = [e["m"] for e in evals]
        ax = axs[1, 0]
        ax.plot(m, [e["nocol"] * 100 for e in evals], color=BLUE, marker="o", linewidth=2, markersize=6)
        ax.set_ylim(min(80, min(e["nocol"] * 100 for e in evals) - 2), 101)
        ax.set_title("検証: 衝突なしの回 [%](88 回、N1)", loc="left")
        ax = axs[1, 1]
        ax.plot(m, [e["n2"] * 100 for e in evals], color=BLUE, marker="o", linewidth=2, markersize=6, label="標準の物理(N2、≥ 8 cm)")
        ax.plot(m, [e["n3"] * 100 for e in evals], color=ORANGE, marker="s", linewidth=2, markersize=6, label="ずらした物理(N3、≥ 5 cm)")
        ax.axhline(8, color=BLUE, linewidth=1, linestyle="--")
        ax.axhline(5, color=ORANGE, linewidth=1, linestyle="--")
        ax.set_title("検証: 最小すき間 [cm](N2・N3)", loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax = axs[1, 2]
        ax.plot(m, [e["rq"] * 100 for e in evals], color=BLUE, marker="o", linewidth=2, markersize=6, label="ρ_req > 1")
        ax.plot(m, [e["rc"] * 100 for e in evals], color=ORANGE, marker="s", linewidth=2, markersize=6, label="ρ_comb > 1")
        ax.axhline(2, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title("検証: 摩擦の限界を超えたフレーム [%](N6)", loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax = axs[1, 3]
        ax.plot(m, [e["n7"] * 100 for e in evals], color=BLUE, marker="o", linewidth=2, markersize=6)
        ax.axhline(5, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title("検証: シールドの置き換え [%](N7)", loc="left")
        for e in evals:
            axs[1, 0].annotate(f"{e['n_pass']}/{e['n_j']}", (e["m"], e["nocol"] * 100), textcoords="offset points", xytext=(0, -14), ha="center",
                               fontsize=7, color=TEXT2)
        sel_p = run.parent / "selection.json"
        sel_m = int(json.loads(sel_p.read_text())["selected"].split("_")[1]) / 1e6 if sel_p.exists() else None
        for a in axs[1]:
            a.set_xlabel("チェックポイント [百万ステップ]")
            if sel_m is not None:
                a.axvline(sel_m, color=GRAY, linewidth=1, linestyle=":")
        if sel_m is not None:
            axs[1, 3].text(sel_m, 4.6, "選んだ方策 ", ha="right", fontsize=8, color=TEXT2)
        axs[1, 0].text(0.02, 0.04, "点の下の数字 = 合格した判定の数", transform=axs[1, 0].transAxes, fontsize=7, color=TEXT2)
    fig.suptitle("強化学習モード「絶対壁にぶつからない」(no_wall)の学習(r2 の方策から 1,000 万ステップ、種 21)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "no_wall_learning_curve.png", dpi=130)
    plt.close(fig)


def draw_course(ax, course) -> None:
    for s in course["surfaces"]:
        if s["id"] == "carpet":
            continue
        ax.add_patch(plt.Rectangle((s["x"] - s["w"] / 2, s["z"] - s["d"] / 2), s["w"], s["d"], color="#efeee9", zorder=0))
        ax.text(s["x"], s["z"] + s["d"] / 2 - 0.12, {"blue-mat": "青マット", "turf": "芝", "ptfe": "PTFE μ0.22", "shortcut": "黒シート"}.get(s["id"], s["id"]),
                fontsize=6, color=GRAY, ha="center")
    for g in course["segments"]:
        ax.plot([g["a"][0], g["b"][0]], [g["a"][1], g["b"][1]], color="#3b3a37" if g["kind"] != "curtain" else "#8a8984",
                linewidth=1.6 if g["kind"] != "gate-foot" else 1.0, solid_capstyle="round", zorder=2)
    for d in course["disks"]:
        if d["kind"] in ("post-foot", "gate-post"):
            ax.add_patch(plt.Circle((d["x"], d["z"]), d["r"], color="#b9b8b2" if d["kind"] == "post-foot" else "#3b3a37", zorder=2, linewidth=0))
    ref = course.get("reference")
    if ref:
        ax.plot(ref["x"], ref["z"], color=GRAY, linewidth=0.6, linestyle=":", zorder=1)
    b = course["bounds"]
    ax.set_xlim(b["minX"], b["maxX"])
    ax.set_ylim(b["maxZ"], b["minZ"])  # z を下向きに(アプリの上から見た向き)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def first_lap(tr, frames=None):
    return tr if frames is None else tr[:frames]


def topdown(traces, out: Path) -> None:
    runs = {r["id"]: r for r in traces["runs"]}
    fig, ax = plt.subplots(figsize=(12, 7.4))
    draw_course(ax, traces["course"])
    f = runs.get("fastest_std")
    if f:
        t = f["trace"][:260]
        ax.plot([q["x"] for q in t], [q["z"] for q in t], color=ORANGE, linewidth=1.6, zorder=3, label=f"最速(1 周目、最小すき間 {f['min_clearance'] * 100:.1f} cm)")
    r = runs["nw_std"]
    t = r["trace"][:330]
    pts = np.array([[q["x"], q["z"]] for q in t])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    v = np.array([q["v"] for q in t[1:]])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#a9c9f0", "#2a78d6", "#0b3a75"])
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 4.5), linewidth=2.6, zorder=4)
    lc.set_array(v)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("no_wall の速さ [m/s]", color=TEXT2)
    sh = [q for q in r["trace"] if q.get("sh")]
    if sh:
        ax.scatter([q["x"] for q in sh], [q["z"] for q in sh], s=36, marker="x", color="#e34948", zorder=5, label=f"シールドが替えたフレーム({len(sh)})")
    dmin = min(r["trace"], key=lambda q: q["d"])
    ax.scatter([dmin["x"]], [dmin["z"]], s=60, facecolors="none", edgecolors="#0b0b0b", linewidths=1.2, zorder=6)
    ax.annotate(f"no_wall の最小すき間 {r['min_clearance'] * 100:.1f} cm", (dmin["x"], dmin["z"]), textcoords="offset points", xytext=(10, 10), fontsize=8,
                color="#0b0b0b")
    fd = min(f["trace"][:260], key=lambda q: q["d"]) if f else None
    if fd:
        ax.scatter([fd["x"]], [fd["z"]], s=60, facecolors="none", edgecolors=ORANGE, linewidths=1.2, zorder=6)
    ax.plot([], [], color=BLUE, linewidth=2.6, label=f"no_wall(1 周目 {r['laps'][0]:.2f} 秒、最小すき間 {r['min_clearance'] * 100:.1f} cm)" if r["laps"] else "no_wall")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=9)
    ax.set_title("上から見た線(標準の物理・STD の開始・連続 20 秒の 1 周目)。点線 = 参照の線 R", loc="left")
    fig.tight_layout()
    fig.savefig(out / "no_wall_topdown.png", dpi=130)
    plt.close(fig)


def topdown_variation(traces, out: Path) -> None:
    runs = traces["runs"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4))
    ax = axs[0]
    draw_course(ax, traces["course"])
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    sel = [r for r in runs if r["id"].startswith("nw_grid") or r["id"] == "nw_fc"]
    for i, r in enumerate(sel):
        t = r["trace"]
        ax.plot([q["x"] for q in t], [q["z"] for q in t], color=pal[i % len(pal)], linewidth=1.1, alpha=0.9,
                label=f"{r['label'].replace('no_wall・', '')}: すき間 {r['min_clearance'] * 100:.1f} cm・衝突 {r['collisions']}")
        sh = [q for q in t if q.get("sh")]
        if sh:
            ax.scatter([q["x"] for q in sh], [q["z"] for q in sh], s=20, marker="x", color="#e34948", zorder=5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=7)
    ax.set_title("物理をずらした回(STD の開始、20 秒)。× = シールドが替えたフレーム", loc="left")
    ax = axs[1]
    draw_course(ax, traces["course"])
    sel = [r for r in runs if r["id"].startswith("nw_start")]
    for i, r in enumerate(sel):
        t = r["trace"]
        ax.plot([q["x"] for q in t], [q["z"] for q in t], color=pal[i % len(pal)], linewidth=1.1, alpha=0.9,
                label=f"{r['label'].replace('no_wall・', '')}: すき間 {r['min_clearance'] * 100:.1f} cm")
        ax.scatter([t[0]["x"]], [t[0]["z"]], s=30, color=pal[i % len(pal)], zorder=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=7)
    ax.set_title("いろいろな開始(ゲート・左右 ±0.2 m・向き ±15°、標準の物理)。● = 開始", loc="left")
    fig.tight_layout()
    fig.savefig(out / "no_wall_topdown_variation.png", dpi=130)
    plt.close(fig)


def clearance_s(traces, out: Path) -> None:
    runs = {r["id"]: r for r in traces["runs"]}
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for rid, col, lab in (("nw_std", BLUE, "no_wall"), ("fastest_std", ORANGE, "最速")):
        r = runs.get(rid)
        if not r:
            continue
        t = [q for q in r["trace"] if q["s"] is not None and q["s"] < 26.12]
        ax.plot([q["s"] for q in t], [q["d"] * 100 for q in t], color=col, linewidth=1.8, label=f"{lab}(フレームの最小 {min(q['d'] for q in t) * 100:.1f} cm)")
    ax.axhline(8, color=GRAY, linewidth=1, linestyle="--")
    ax.text(0.1, 8.3, "8 cm(N2)", fontsize=8, color=TEXT2)
    ax.set_ylim(0, 30)
    ax.set_xlabel("R に沿った位置 s [m](STD の開始から 1 周)")
    ax.set_ylabel("車体と壁・柱のすき間 [cm]")
    ax.set_title("1 周目のすき間(標準の物理、フレームの値、30 cm で切る)", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "no_wall_clearance_s.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    learning_curve(a.run, a.out)
    tr = json.loads(a.traces.read_text())
    topdown(tr, a.out)
    topdown_variation(tr, a.out)
    clearance_s(tr, a.out)
    print("書いた:", sorted(p.name for p in a.out.glob("no_wall_*.png")))


if __name__ == "__main__":
    main()
