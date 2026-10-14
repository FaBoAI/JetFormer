"""no_wall r2(強化学習モード「絶対壁にぶつからない」の作り直し)の図(2026-10-14、学習の段 no_wall r2)。

    python rl/nw2/make_figures_r2.py --r2 <train/no_wall/r2> --sets <sets.json> --traces <fig_traces/> --out <fig/>

図:
  no_wall_r2_learning_curve.png  r2 の学習の曲線(学習中の集計: 20 秒の走り切り・1 周目・最小すき間・遅い舵 + 遅延の回の走り切り)と、
                                 チェックポイントごとの検証の組(シールド v3)の値
  no_wall_r2_probe_sets.png      試験でない組(G16・ABL・CORNER)の衝突の数: r1 + シールド v2 / r1 + v3 / r2 + v3、シールドなしと比べた「救った・起こした」
  no_wall_r2_topdown.png         上から見た線: ゲート 16・舵 0.8・遅延 1(確認 r1 の衝突の形)と、標準の物理の STD。方策 × シールドの版ごと
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
BLUE, ORANGE, AQUA, GRAY, TEXT2, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e", "#e34948"
COURSE = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/no_wall/traces_selected.json")


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
    evals = []
    for p in sorted((run / "evals" / "no_wall").glob("step_*.json")):
        d = json.loads(p.read_text())
        j = d["judged"]
        n1 = j["N1_no_collision"]["value"].split("/")
        evals.append({"m": int(p.stem.split("_")[1]) / 1e6, "col": int(n1[1]) - int(n1[0]), "n2": j["N2_clearance_nominal"]["value"],
                      "n3": j["N3_clearance_randomised"]["value"], "n4": j["N4_lap_from_stop"]["value"], "rc": j["N6_friction"]["value"]["rho_comb_gt1"],
                      "n7": j["N7_shield"]["value"], "r200sh": d.get("reported", {}).get("R200_shield_fraction"), "all": d["all_judged_pass"]})
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.6))
    panels = [
        (axs[0, 0], g("survived_20s_frac"), "20 秒を走り切った回の割合(学習)", "割合", None),
        (axs[0, 1], g("lap1_mean"), "1 周目の時間 [秒](学習、物理のずれ込み)", "秒", None),
        (axs[0, 2], g("min_clearance_mean") * 100, "回の最小すき間の平均 [cm](学習)", "cm", 8.0),
        (axs[0, 3], g("rho_comb_gt1_frac") * 100, "ρ_comb > 1 のフレーム [%](学習)", "%", 2.0),
    ]
    for ax, y, title, yl, ref in panels:
        ax.plot(steps, y, color=BLUE, alpha=0.25, linewidth=1)
        ax.plot(steps, rolling(y), color=BLUE, linewidth=2)
        if ref is not None:
            ax.axhline(ref, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel("r2 の学習のステップ [百万](r1 の 1,000 万から続き)")
        ax.set_ylabel(yl)
    sel = json.loads((run / "selection.json").read_text()) if (run / "selection.json").exists() else None
    if evals:
        m = [e["m"] for e in evals]
        ax = axs[1, 0]
        extra = {}
        if sel:
            for c in sel["candidates"]:
                if c.get("selection_extra"):
                    extra[c["steps"] / 1e6] = sum(v["collisions"] for v in c["selection_extra"].values())
        ax.bar([x - 0.18 for x in m], [e["col"] for e in evals], width=0.34, color=RED, label="検証の組 88 回")
        xs = [x for x in m if any(abs(x - k) < 0.05 for k in extra)]
        ys = [next(v for k, v in extra.items() if abs(x - k) < 0.05) for x in xs]
        ax.bar([x + 0.18 for x in xs], ys, width=0.34, color=ORANGE, label="選びの追加の組 300 回")
        ax.set_ylim(0, max(3, max([e["col"] for e in evals] + ys + [0]) + 1))
        for e in evals:
            ax.text(e["m"] - 0.18, e["col"] + 0.05, f"{e['col']}" + (" ✓" if e["all"] else ""), ha="center", fontsize=8, color=TEXT2)
        for x, y in zip(xs, ys):
            ax.text(x + 0.18, y + 0.05, f"{y}", ha="center", fontsize=8, color=TEXT2)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_title("衝突の数(✓ = 検証の組で 7 つ全部合格)", loc="left")
        if sel:
            sm = int(sel["selected"].split("_")[1]) / 1e6
            for a2 in axs[1]:
                a2.axvline(sm, color=GRAY, linestyle=":", linewidth=1.2)
            axs[1, 0].text(sm, axs[1, 0].get_ylim()[1] * 0.92, " 選んだ点", fontsize=8, color=TEXT2)
        ax = axs[1, 1]
        ax.plot(m, [e["n2"] * 100 for e in evals], "o-", color=BLUE, label="N2 標準の物理")
        ax.plot(m, [e["n3"] * 100 for e in evals], "o-", color=ORANGE, label="N3 物理のずれ")
        ax.axhline(8, color=BLUE, linestyle="--", linewidth=1)
        ax.axhline(5, color=ORANGE, linestyle="--", linewidth=1)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("最小すき間 [cm](検証の組)", loc="left")
        ax = axs[1, 2]
        ax.plot(m, [e["n4"] for e in evals], "o-", color=AQUA)
        ax.axhline(12, color=GRAY, linestyle="--", linewidth=1)
        ax.set_title("N4 停止から 1 周 [秒](検証の組)", loc="left")
        ax = axs[1, 3]
        ax.plot(m, [e["n7"] * 100 for e in evals], "o-", color=BLUE, label="N7 標準の物理(N60)")
        ax.plot(m, [(e["r200sh"] or 0) * 100 for e in evals], "o-", color=ORANGE, label="R200(物理のずれ)")
        ax.axhline(5, color=GRAY, linestyle="--", linewidth=1)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("シールドが替えたフレーム [%]", loc="left")
        for ax in axs[1]:
            ax.set_xlabel("チェックポイント [百万ステップ]")
    fig.suptitle("no_wall r2(種 22、r1 から 600 万ステップ、遅い舵 + 遅延の回を増やし、速さの見張りつき)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "no_wall_r2_learning_curve.png", dpi=130)
    plt.close(fig)


def probe_sets(sets: dict, out: Path) -> None:
    names = [("G16", "ゲート 16 の格子 40 回\n(舵 0.8〜1.0・遅延 1)"), ("ABL", "原因分け 330 回\n(C2 から 1 つずつ)"), ("CORNER", "DR-T の隅 460 回\n(4 隅 × ゲート × 開始の隅)")]
    variants = [(k, lab, c) for k, lab, c in [("r1_v2", "r1 + シールド v2", GRAY), ("r1_v3", "r1 + シールド v3", BLUE), ("r2_v3", "r2 + シールド v3", AQUA)] if k in sets]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    ax = axs[0]
    x = np.arange(len(names))
    w = 0.8 / max(1, len(variants))
    for i, (k, lab, c) in enumerate(variants):
        vals = [sets[k].get(n, {}).get("on", {}).get("collisions", np.nan) for n, _ in names]
        bars = ax.bar(x + (i - (len(variants) - 1) / 2) * w, vals, width=w, color=c, label=lab)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{int(v)}", ha="center", fontsize=8, color=TEXT2)
    offs = [sets.get("r1_off", {}).get(n, {}).get("collisions", np.nan) for n, _ in names]
    ax.scatter(x, offs, marker="_", s=900, color=RED, zorder=5, label="r1 シールドなし")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in names], fontsize=9)
    ax.set_title("試験でない組の衝突の数(シールドあり)。範囲の中の悪い物理", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    rows = []
    for k, lab, c in variants:
        cs = sets[k].get("CORNER", {})
        if "saved" in cs:
            rows.append((lab, cs["saved"], len(cs["caused"]), len(cs.get("collided_zero_shield", [])), c))
    if rows:
        y = np.arange(len(rows))
        ax.barh(y + 0.2, [r[1] for r in rows], height=0.35, color=[r[4] for r in rows], label="救った(なしで衝突 → ありで衝突なし)")
        ax.barh(y - 0.2, [r[2] for r in rows], height=0.35, color=RED, label="起こした(なしで衝突なし → ありで衝突)")
        for i, r in enumerate(rows):
            ax.text(r[1] + 1, i + 0.2, str(r[1]), va="center", fontsize=8, color=TEXT2)
            ax.text(r[2] + 1, i - 0.2, f"{r[2]}(シールドが 1 フレームも替えずに衝突 {r[3]})", va="center", fontsize=8, color=TEXT2)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_title("CORNER: シールドなしの同じ回と比べて", loc="left")
    fig.tight_layout()
    fig.savefig(out / "no_wall_r2_probe_sets.png", dpi=130)
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
    b = course["bounds"]
    ax.set_xlim(b["minX"], b["maxX"])
    ax.set_ylim(b["maxZ"], b["minZ"])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def speed_line(ax, fig, tr, cmap, lw=2.4, z=4):
    pts = np.array([[q["x"], q["z"]] for q in tr])
    if len(pts) < 2:
        return None
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 4.5), linewidth=lw, zorder=z)
    lc.set_array(np.array([q["v"] for q in tr[1:]]))
    ax.add_collection(lc)
    return lc


def topdown(traces_dir: Path, out: Path) -> None:
    course = json.loads(COURSE.read_text())["course"]
    variants = [(f, json.loads((traces_dir / f"{f}.json").read_text())) for f in ("r1_v2", "r1_v3", "r2_v3") if (traces_dir / f"{f}.json").exists()]
    if not variants:
        return
    fig, axs = plt.subplots(len(variants), 2, figsize=(15, 4.6 * len(variants)))
    axs = np.atleast_2d(axs)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#a9c9f0", "#2a78d6", "#0b3a75"])
    lc = None
    for i, (key, d) in enumerate(variants):
        runs = {r["id"]: r for r in d["runs"]}
        for j, (rid, title) in enumerate([("g16_s08_d1", "ゲート 16 から・舵の速さ 0.8・遅延 1(ほかは標準)"), ("std_nominal", "標準の物理・STD の開始(記録と同じ)")]):
            ax = axs[i, j]
            draw_course(ax, course)
            r = runs.get(rid)
            if not r:
                continue
            tr = r["trace"]
            lc = speed_line(ax, fig, tr, cmap) or lc
            gv = [q for q in tr if q.get("sh") and (q.get("fb") or "") == "governor"]
            sh = [q for q in tr if q.get("sh") and (q.get("fb") or "") != "governor"]
            if gv:
                ax.scatter([q["x"] for q in gv], [q["z"] for q in gv], s=26, color="#f6c3ad", zorder=3, linewidths=0,
                           label=f"速さの見張りだけで替えたフレーム {len(gv)}")
            if sh:
                ax.scatter([q["x"] for q in sh], [q["z"] for q in sh], s=14, marker="x", color=ORANGE, zorder=5, linewidths=1.0,
                           label=f"シールドが指令を替えたフレーム {len(sh)}")
            ax.scatter([tr[0]["x"]], [tr[0]["z"]], s=40, marker="o", color="#0b0b0b", zorder=6)
            if r["collisions"]:
                c = r["collision"]
                ax.scatter([c["x"]], [c["z"]], s=140, marker="X", color=RED, zorder=7, label=f"衝突 {c['time']:.2f} 秒")
            res = "衝突" if r["collisions"] else f"衝突なし・最小すき間 {r['min_clearance'] * 100:.1f} cm"
            ax.set_title(f"{d['label']}: {title}\n{res}" + (f"・1 周目 {r['laps'][0]:.2f} 秒" if r.get("laps") else ""), loc="left", fontsize=10)
            if sh or gv or r["collisions"]:
                ax.legend(loc="lower left", frameon=False, fontsize=8)
    if lc is not None:
        cb = fig.colorbar(lc, ax=axs, shrink=0.5, pad=0.01)
        cb.set_label("速さ [m/s]", color=TEXT2)
    fig.savefig(out / "no_wall_r2_topdown.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", type=Path, required=True)
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.r2 / "progress.jsonl").exists():
        learning_curve(a.r2, a.out)
    if a.sets.exists():
        probe_sets(json.loads(a.sets.read_text()), a.out)
    topdown(a.traces, a.out)


if __name__ == "__main__":
    main()
