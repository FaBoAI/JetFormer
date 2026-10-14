"""離脱復活 r3 の図(2026-10-14、学習の段 recover r3)。jetformer_reports/sim_rl_modes_20261013/r3/fig/ に PNG を書く。

    python3 -I r3/rc/figs_rc3.py table      # 立て直せる表(使えるマスの数・速さごとの立て直せる割合)
    python3 -I r3/rc/figs_rc3.py curve      # 学習曲線と検証の組
    python3 -I r3/rc/figs_rc3.py events     # 試験の組の乱しの始まりの位置と成否
    python3 -I r3/rc/figs_rc3.py compare    # r1(元の乱し)と r3 の判定の値

色はカテゴリの固定の順(青・橙・水色・黄・桃)、量は青の一色の濃淡。成否は色と形(○ と ×)の両方で見分ける。
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

R3 = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3")
FIG = R3 / "fig"
RC3 = R3 / "rc3"
EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
LOOP = 26.119648724030924
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD, CRIT = "#2a78d6", "#d03b3b"


def setup():
    for f in ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic", "VL Gothic"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10, "lines.linewidth": 2})
    FIG.mkdir(parents=True, exist_ok=True)


def fig_table():
    tab = json.loads((EXP / "r3/rc/recoverable_table_v1.json").read_text())
    cells, combos = tab["cells"], tab["combos"]
    nS, nV = int(math.ceil(LOOP / tab["s_bin_m"])), 10
    usable = np.zeros((nV, nS))
    seen = np.zeros((nV, nS), bool)
    for k, (n, r) in cells.items():
        sb, vb, _ = k.split("|")
        sb, vb = int(sb), int(vb)
        if vb >= nV:
            continue
        seen[vb, sb] = True
        if n >= 3 and r / n >= .95:
            usable[vb, sb] += 1
    fig, axs = plt.subplots(2, 1, figsize=(10, 7.2), gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axs[0]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#f0efec"] + BLUE_RAMP)
    bounds = [-.5, .5, 3, 6, 9, 12, 15, 17.5, 18.5]
    m = np.where(seen, usable, np.nan)
    im = ax.imshow(m, origin="lower", aspect="auto", cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), extent=[0, nS * tab["s_bin_m"], 0, nV * tab["v_bin_mps"]],
                   interpolation="nearest")
    for z0, z1, name in [(5.9, 8.0, "黒シートの区間"), (14.6, 19.3, "奥の直線")]:
        ax.axvspan(z0, z1, color=MUTED, alpha=.08, lw=0)
        ax.text((z0 + z1) / 2, nV * tab["v_bin_mps"] - .25, name, ha="center", va="top", color=MUTED, fontsize=9)
    ax.set_xlabel("R 上の位置 s [m]"); ax.set_ylabel("乱しの始まりの速さ [m/s]")
    ax.set_title("立て直せる表: マスごとに「使える」組の数(18 通りのうち、立て直せる割合 ≥ 0.95・サンプル ≥ 3)", loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, pad=.01, ticks=[0, 1.75, 4.5, 7.5, 10.5, 13.5, 16.25, 18])
    cb.ax.set_yticklabels(["0", "1-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18"]); cb.set_label("使える組の数")
    ax.grid(False)
    # 速さごとの立て直せる割合(組の種類ごと、全部のサンプル)
    ax = axs[1]
    groups = {"舵のパルス 左": lambda c: c.startswith("steer_pulse") and c.endswith(":L"), "舵のパルス 右": lambda c: c.startswith("steer_pulse") and c.endswith(":R"),
              "わざと外す": lambda c: c.startswith("excursion"), "ブレーキ": lambda c: c.startswith("brake"), "踏み込み": lambda c: c.startswith("burst")}
    for gi, (g, f) in enumerate(groups.items()):
        n = np.zeros(nV); r = np.zeros(nV)
        for k, (cn, cr) in cells.items():
            sb, vb, c = k.split("|")
            if int(vb) < nV and f(c):
                n[int(vb)] += cn; r[int(vb)] += cr
        x = (np.arange(nV) + .5) * tab["v_bin_mps"]
        ok = n >= 20
        ax.plot(x[ok], (r / np.maximum(n, 1))[ok], color=SERIES[gi], marker="o", ms=5, label=g)
        last = np.nonzero(ok)[0][-1]
        ax.annotate(g, (x[last], r[last] / n[last]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.axhline(.95, color=MUTED, lw=1, ls="--"); ax.text(4.75, .955, "0.95(記録・試験で使えるしきい値)", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_ylim(0, 1.05); ax.set_xlim(.8, 5.6)
    ax.set_xlabel("乱しの始まりの速さ [m/s]"); ax.set_ylabel("立て直せる割合")
    ax.set_title("強さ 3 つをまとめた、速さごとの立て直せる割合(4 つの参照の走り × 8 開始の状態)", loc="left", fontsize=11)
    ax.legend(loc="lower left", frameon=False, ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "recover_r3_table.png", dpi=130)
    print("書いた", FIG / "recover_r3_table.png")


def fig_curve(run="main"):
    rd = RC3 / "train" / run
    prog = []
    for l in open(rd / "progress.jsonl"):
        p = json.loads(l)
        # last.pt から続けたとき、保存のあとに進んで捨てた分(前の回の行)を除く(同じステップの行は続けたあとの行を使う)
        while prog and prog[-1]["steps"] >= p["steps"]:
            prog.pop()
        prog.append(p)
    st = np.array([p["steps"] for p in prog]) / 1e6
    surv = np.array([p["episodes"].get("survived_20s_frac") or np.nan for p in prog])
    ret = np.array([p["episodes"].get("return_mean") or np.nan for p in prog])
    w = 10
    def sm(a):
        # 端でも欠けない移動平均(窓の中の値だけで割る)
        v = np.nan_to_num(a, nan=0.0); m = (~np.isnan(a)).astype(float)
        k = np.ones(w)
        return np.convolve(v, k, mode="same") / np.maximum(np.convolve(m, k, mode="same"), 1)
    evs = sorted((rd / "evals" / "recover").glob("step_*.json"))
    base = RC3 / "val_r1_init.json"
    rows = []
    if base.exists():
        rows.append((0, json.loads(base.read_text())))
    rows += [(int(p.stem.split("_")[1]), json.loads(p.read_text())) for p in evs]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axs[0]
    ax.plot(st, sm(surv), color=SERIES[0], label="20 秒の生き残り(学習中・確率的)")
    for rs in json.loads((rd / "run.json").read_text()).get("resumes", []):
        ax.axvline(rs["steps"] / 1e6, color=MUTED, lw=1, ls=":")
        ax.annotate("last.pt から続けた", (rs["steps"] / 1e6, .05), xytext=(3, 0), textcoords="offset points", fontsize=7, color=MUTED)
    ax.set_ylim(0, 1); ax.set_xlabel("学習のステップ [百万]"); ax.set_title("学習中の 20 秒の生き残り(10 更新の移動平均)", loc="left", fontsize=10)
    ax2 = axs[1]
    if rows:
        x = np.array([r[0] for r in rows]) / 1e6
        c1 = [int(r[1]["judged"]["C1_r3"]["value"].split("/")[0]) / int(r[1]["judged"]["C1_r3"]["value"].split("/")[1]) for r in rows]
        c2 = [r[1]["judged"]["C2"]["value"] for r in rows]
        c5 = [min(r[1]["judged"]["C5_r3"]["value"]["yaw_rate"], r[1]["judged"]["C5_r3"]["value"]["lateral_rate"]) for r in rows]
        for i, (y, name) in enumerate([(c1, "C1 有効な回の割合"), (c2, "C2 出来事の成功"), (c5, "C5 キック(向き・横の低い方)")]):
            ax2.plot(x, y, color=SERIES[i], marker="o", ms=5, label=name)
        ax2.axhline(.95, color=MUTED, lw=1, ls="--"); ax2.axhline(.90, color=MUTED, lw=1, ls=":")
        ax2.set_ylim(min(.5, min(c1 + c2 + c5) - .05), 1.01)
        ax2.set_xlabel("ステップ [百万](0 = 初期化元 r1)"); ax2.set_title("検証の組(シールドあり): C1・C2・C5", loc="left", fontsize=10)
        ax2.legend(frameon=False, fontsize=8, loc="lower right")
        ax3 = axs[2]
        c4 = [r[1]["judged"]["C4"]["value"]["events_per_episode"] for r in rows]
        lap = [r[1]["judged"]["C6_r3"]["value"]["lap_from_stop"] or np.nan for r in rows]
        ax3.plot(x, c4, color=SERIES[0], marker="o", ms=5, label="C4 出来事 / 回")
        ax3.axhline(3.5, color=MUTED, lw=1, ls="--")
        ax3.set_ylabel("出来事 / 回"); ax3.set_xlabel("ステップ [百万]")
        ax3.set_title("C4 出来事の数(左)・C6 停止から 1 周(右)", loc="left", fontsize=10)
        ax4 = ax3.twinx() if False else None  # 2 つの軸は使わない(下に数字で書く)
        for xi, li, ci in zip(x, lap, c4):
            ax3.annotate(f"{li:.2f} 秒" if li == li else "—", (xi, ci), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        ok = [r for r in rows if r[1]["all_judged_pass"]]
        for r in ok:
            for a in (ax2, ax3):
                a.axvline(r[0] / 1e6, color=SERIES[2], lw=1, alpha=.6)
    sel = rd / "selection.json"
    if sel.exists():
        s = json.loads(sel.read_text())["selected"]["step"] / 1e6
        for a in axs[1:]:
            a.axvline(s, color=TEXT, lw=1.2, ls="-.")
    fig.tight_layout()
    fig.savefig(FIG / "recover_r3_learning_curve.png", dpi=130)
    print("書いた", FIG / "recover_r3_learning_curve.png")


def loop_xy():
    ref = json.loads((EXP / "r3/rc/rt/reference_loop.json").read_text())
    return np.array(ref["s"]), np.array(ref["x"]), np.array(ref["z"])


def fig_events(eval_path=None, out_name="recover_r3_events.png"):
    ev = json.loads(Path(eval_path or (RC3 / "eval_test.json")).read_text())
    s_ref, x_ref, z_ref = loop_xy()
    tname = next(k for k in ev["tests"] if k.startswith(("RC-P100", "RC-V100", "RC-FRESH")))
    rows = ev["tests"][tname]
    pts = [(e["s"] % LOOP, e["success"], e["type"]) for r in rows for e in (r.get("recovery") or [])]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    ax.plot(x_ref, z_ref, color=GRID, lw=6, solid_capstyle="round", zorder=1)
    ax.plot(x_ref, z_ref, color=MUTED, lw=.8, zorder=2)
    xs = lambda s: np.interp(s, s_ref, x_ref)  # noqa: E731
    zs = lambda s: np.interp(s, s_ref, z_ref)  # noqa: E731
    ok = [p for p in pts if p[1]]; ng = [p for p in pts if not p[1]]
    ax.scatter([xs(p[0]) for p in ok], [zs(p[0]) for p in ok], s=22, facecolors="none", edgecolors=GOOD, linewidths=1.2, label=f"立て直せた {len(ok)}", zorder=3)
    ax.scatter([xs(p[0]) for p in ng], [zs(p[0]) for p in ng], s=40, marker="x", color=CRIT, linewidths=1.6, label=f"立て直せなかった {len(ng)}", zorder=4)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(f"{tname}: 乱しの始まりの位置と成否(上から)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="center")
    ax = axs[1]
    names = {"steer_pulse": "舵のパルス", "excursion": "わざと外す", "speed_pulse": "ブレーキ/踏み込み"}
    allev = [e for r in rows for e in (r.get("recovery") or [])]
    labels, ns, ss = [], [], []
    for t in ["steer_pulse", "excursion", "speed_pulse"]:
        for c in ["弱", "中", "強"]:
            g = [e for e in allev if e["type"] == t and e["cls"] == c]
            labels.append(f"{names[t]} {c}"); ns.append(len(g)); ss.append(sum(e["success"] for e in g))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, ns, color=GRID, height=.7, label="出来事の数")
    ax.barh(y, ss, color=SERIES[0], height=.7, label="立て直せた数")
    for yi, n, s in zip(y, ns, ss):
        ax.text(n + .5, yi, f"{s}/{n}" if n else "0", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel("出来事の数(当てた組)")
    ax.set_xlim(0, max(ns + [1]) * 1.35)
    ax.set_title("当てた組ごとの出来事と成功", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / out_name, dpi=130)
    print("書いた", FIG / out_name)


def fig_compare(items):
    """items: [(名前, eval.json のパス, 判定の鍵の写像)]。C1・C2・C4・C5 を小さな図を並べて比べる"""
    mets = [("C1 有効な回の割合", .90), ("C2 出来事の成功", .95), ("C4 出来事 / 回", 3.5), ("C5 キックの生き残り(低い方)", .90)]
    vals = {m: [] for m, _ in mets}
    for name, v in items:
        for (m, _), x in zip(mets, v):
            vals[m].append((name, x))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (m, thr) in zip(axs, mets):
        names = [n for n, _ in vals[m]]; xs = [x for _, x in vals[m]]
        y = np.arange(len(names))[::-1]
        ax.barh(y, [x if x is not None else 0 for x in xs], color=[SERIES[i] for i in range(len(names))], height=.6)
        for yi, x in zip(y, xs):
            ax.text((x or 0) * 1.01, yi, "—" if x is None else (f"{x:.3f}" if x < 1.5 else f"{x:.2f}"), va="center", fontsize=8)
        ax.axvline(thr, color=MUTED, lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(names if ax is axs[0] else [""] * len(names), fontsize=8)
        ax.set_title(m + f"(基準 {thr})", loc="left", fontsize=9); ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
        ax.set_xlim(0, max(thr, *(x or 0 for x in xs)) * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "recover_r3_compare.png", dpi=130)
    print("書いた", FIG / "recover_r3_compare.png")


if __name__ == "__main__":
    setup()
    what = sys.argv[1]
    if what == "table":
        fig_table()
    elif what == "curve":
        fig_curve(*(sys.argv[2:3] or ["main"]))
    elif what == "events":
        fig_events(*(sys.argv[2:4] or [None]))
    elif what == "compare":
        fig_compare(json.loads(Path(sys.argv[2]).read_text()))
