"""速さいろいろ r2 の図(2026-10-14、学習の段 speed r2)。

    python rl/speed_r2/figs_r2.py curves --run <train/speed/r2/main> --baseline <val_init.json> --selection <selection.json>
    python rl/speed_r2/figs_r2.py traj --figdata <figdata_r2.json>

出力: /home/jetson/jetformer_reports/sim_rl_modes_20261013/fig/speed_r2_learning_curve.png・speed_r2_trajectories.png・speed_r2_clearance_s.png
色・字・コースの描き方は第 1 回の rl/speed/speed_figs.py を読むだけで使う(dataviz の参照パレット。c_v の 5 段は青の順序の段、比べる線はオレンジ)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "speed"))
from speed_figs import (CV_COLOR, CV_LEVELS, EXP, FIG, GRID, INK, INK2, ORANGE, SURFACE, _draw_course, _jp_font, _style,  # noqa: E402
                        load_progress, smooth)

L_LOOP = 26.119648724030924
POST_S = (6.6, 7.2)   # 黒シートの入口の柱の横(確認 speed r1)


def load_evals(run: Path):
    out = []
    for p in sorted((run / "evals" / "speed").glob("step_*.json")):
        out.append((int(p.stem.split("_")[1]), json.loads(p.read_text())))
    return out


def curves(run: Path, baseline: Path | None, selection: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _jp_font()
    P = load_progress(run)
    run_json = json.loads((run / "run.json").read_text())
    init_steps = 0
    E = load_evals(run)
    B = json.loads(baseline.read_text()) if baseline and baseline.exists() else None
    sel = json.loads(selection.read_text())["selected"]["step"] if selection and selection.exists() else None
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.8), facecolor=SURFACE)
    (a_term, a_cl, a_npass), (a_s2, a_s3, a_s5) = axes
    x = (np.array([r["steps"] for r in P]) + init_steps) / 1e6
    n = np.array([r["episodes"]["n"] for r in P], float)
    rs = [r["episodes"]["reasons"] for r in P]
    def wsmooth(cnt, k=9):   # 回の数で重みをつけた移動平均(終わった回が少ない更新で跳ねないように)
        cnt = np.asarray(cnt, float)
        return np.array([cnt[max(0, i - k + 1):i + 1].sum() / max(1, n[max(0, i - k + 1):i + 1].sum()) for i in range(len(cnt))])
    surv = [q.get("collection_duration_complete", 0) for q in rs]
    clr = [q.get("clearance", 0) for q in rs]
    col = [q.get("collision", 0) for q in rs]
    gm = [q.get("gate_miss", 0) + q.get("stall", 0) for q in rs]
    k0 = int(np.argmax(np.cumsum(n) >= 20))   # 終わった回が 20 を越えるまでの最初の更新は描かない(1 回だけの更新で 0/1 に跳ねる)
    x, n = x[k0:], n[k0:]
    surv, clr, col, gm = surv[k0:], clr[k0:], col[k0:], gm[k0:]
    a_term.plot(x, wsmooth(surv), color=CV_COLOR[1.0], lw=1.8, label="20 秒を走り切った")
    a_term.plot(x, wsmooth(clr), color=ORANGE, lw=1.5, label="すき間 3 cm 未満で終了")
    a_term.plot(x, wsmooth(col), color="#8a8984", lw=1.2, ls=":", label="衝突で終了")
    a_term.plot(x, wsmooth(gm), color="#1baf7a", lw=1.1, ls="--", label="ゲートの外し・停止で終了")
    a_term.set_ylim(0, 1.05)
    _style(a_term, "学習の回の終わり方(9 更新の移動平均、回の数で重みづけ)", "割合")
    a_term.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="center right")
    mc = [r["episodes"].get("min_clearance_mean") for r in P][k0:]
    bar = [r["episodes"]["reward_parts_mean"].get("barrier") for r in P][k0:]
    a_cl.plot(x, smooth(np.array([v if v is not None else np.nan for v in mc]) * 100), color=CV_COLOR[1.0], lw=1.8, label="回の最小すき間の平均 [cm]")
    a_cl2 = a_cl.twinx()
    a_cl2.plot(x, smooth(bar), color=ORANGE, lw=1.3, ls="--", label="すき間の壁の罰(1 回の和、右)")
    a_cl2.tick_params(colors=INK2, labelsize=8)
    for s in ("top",):
        a_cl2.spines[s].set_visible(False)
    _style(a_cl, "学習の回: 最小すき間と、すき間の壁の罰", "すき間 [cm]")
    h1, l1 = a_cl.get_legend_handles_labels(); h2, l2 = a_cl2.get_legend_handles_labels()
    a_cl.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, labelcolor=INK2, loc="lower right")
    pts = ([(0.0, B)] if B else []) + [(s / 1e6, d) for s, d in E]
    if pts:
        xe = np.array([p[0] for p in pts])
        npass = [sum(1 for k in ("S1_valid", "S2_clearance", "S3_lap_order", "S4_speed_order", "S5_vm4") if d["judged"][k]["pass"] is True) for _, d in pts]
        a_npass.plot(xe, npass, color=CV_COLOR[1.0], lw=1.6, marker="o", ms=5)
        a_npass.axhline(5, color=INK2, lw=1, ls="--")
        for xi, (_, d), k in zip(xe, pts, npass):
            fails = [q.split("_")[0] for q, v in d["judged"].items() if v.get("pass") is False]
            if fails:
                a_npass.text(xi, k - 0.1, "・".join(fails), fontsize=7, color=INK2, ha="center", va="top")
        a_npass.set_ylim(2.5, 5.4)
        _style(a_npass, "検証の組(SP-V120-R2 + STD3、高速版): 満たした判定の数", "項目の数 / 5", "学習のステップ数 [百万](0 = 初期化元の 401 万)")
        s2 = np.array([d["judged"]["S2_clearance"]["value"] if d["judged"]["S2_clearance"]["value"] is not None else 0 for _, d in pts]) * 100
        b5 = [len(d["judged"]["S2_clearance"].get("below_5cm") or []) for _, d in pts]
        a_s2.plot(xe, s2, color=CV_COLOR[1.0], lw=1.8, marker="o", ms=5, label="S2 = 120 回の最小すき間(衝突は 0)")
        for xi, y, k in zip(xe, s2, b5):
            a_s2.text(xi, y + 0.25, f"{k}", fontsize=7.5, color=INK2, ha="center", va="bottom")
        a_s2.axhline(3, color=INK2, lw=1, ls="--")
        a_s2.text(xe[0], 3.1, "合格の線 3 cm(点の上の数 = 5 cm 未満の回)", fontsize=8, color=INK2, ha="left", va="bottom")
        a_s2.set_ylim(0, max(s2) + 1.5)
        _style(a_s2, "検証の組: 最小すき間(S2、第 2 回の数え方)", "すき間 [cm]", "学習のステップ数 [百万](0 = 初期化元)")
        a_s2.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower right", bbox_to_anchor=(1.0, 0.12))
        for cv, key in ((1.0, "t1"), (0.75, "t75"), (0.5, "t5")):
            ys = [d["judged"]["S3_lap_order"]["value"].get(key) for _, d in pts]
            ys = [y if y is not None else np.nan for y in ys]
            a_s3.plot(xe, ys, color=CV_COLOR[cv], lw=1.8, marker="o", ms=5, label=f"c_v {cv:g}(上限 {4.5 * cv:.2f} m/s)")
        _style(a_s3, "検証の組: 停止から 1 周(STD、単周)= S3", "1 周 [秒]", "学習のステップ数 [百万](0 = 初期化元)")
        a_s3.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="center right")
        vm4 = [d["judged"]["S5_vm4"]["value"] if d["judged"]["S5_vm4"]["value"] is not None else np.nan for _, d in pts]
        if B and B["judged"]["S5_vm4"]["value"] is None:
            a_s5.text(0.02, 0.30, "0 = 初期化元(点なし): 止まったままの有効な回(P36)が 1 つあり\nVM4 を数えられない(S5 不合格)", fontsize=7.5, color=INK2, va="bottom", transform=a_s5.transAxes)
        a_s5.set_xlim(-0.2, xe[-1] + 0.2)
        v1 = [next(q["mean_speed"] for q in d["judged"]["S4_speed_order"]["value"] if q["c_v"] == 1) for _, d in pts]
        a_s5.plot(xe, vm4, color=CV_COLOR[1.0], lw=1.8, marker="o", ms=5, label="S5 = VM4(s でそろえた速さのばらつき)")
        a_s5.axhline(0.4, color=INK2, lw=1, ls="--")
        a_s5.text(xe[-1], 0.403, "合格の線 0.4 m/s", fontsize=8, color=INK2, ha="right", va="bottom")
        a_s5.text(0.02, 0.06, f"c_v 1.0 の平均の速さ: {v1[0]:.2f} → {v1[-1]:.2f} m/s", transform=a_s5.transAxes, fontsize=8, color=INK2)
        _style(a_s5, "検証の組: 速さのばらつき VM4", "VM4 [m/s]", "学習のステップ数 [百万](0 = 初期化元)")
        a_s5.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper right")
    if sel is not None:
        for ax in axes.flat:
            ax.axvline(sel / 1e6, color=ORANGE, lw=1.1, ls="--")
        a_npass.text(sel / 1e6, 2.6, f" 選んだ {sel / 1e6:.2f} 百万", color=INK2, fontsize=8)
    fig.suptitle(f"速さいろいろ r2 の学習(初期化 = cond の 401 万、種 {run_json['seed']}、400 万ステップ。すき間の壁 9 cm から・3 cm で終了)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "speed_r2_learning_curve.png", dpi=125, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / "speed_r2_learning_curve.png")


def _car(ax, x, z, yaw, color, lw=1.4):
    from matplotlib.patches import Polygon
    hl, hw = 0.21, 0.095   # 車体 0.42 × 0.19 m(前 = (cos yaw, −sin yaw)、左 = (−sin yaw, −cos yaw))
    fx, fz, lx, lz = np.cos(yaw), -np.sin(yaw), -np.sin(yaw), -np.cos(yaw)
    pts = [(x + sx * hl * fx + sy * hw * lx, z + sx * hl * fz + sy * hw * lz) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor=color, lw=lw, zorder=7))


def traj(figdata: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _jp_font()
    D = json.loads(figdata.read_text())
    world = json.loads((EXP / "snapshot" / "sim" / "world-v3.json").read_text())
    # 1) 上から見た軌跡(STD の単周、5 段)+ 黒シートの入口の拡大(試験の組の c_v 0.5・0.625 の回、r2 と r1)
    fig = plt.figure(figsize=(17, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1])
    a_map = fig.add_subplot(gs[:, 0])
    a_z2, a_z1 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    a_v, a_d = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    a_map.set_facecolor(SURFACE)
    _draw_course(a_map, world, labels=True)
    for cv in CV_LEVELS:
        r = D["single"][f"{cv:g}"] if f"{cv:g}" in D["single"] else D["single"][str(cv)]
        tr = np.array(r["trace"])
        a_map.plot(tr[:, 0], tr[:, 1], color=CV_COLOR[cv], lw=1.6, zorder=4,
                   label=f"c_v {cv:g}: 停止から 1 周 {r['lap_from_stop']:.2f} 秒・すき間 {r['min_clearance'] * 100:.1f} cm")
    b = world["bounds"]
    a_map.set_xlim(b["minX"], b["maxX"]); a_map.set_ylim(b["maxZ"], b["minZ"])
    a_map.set_aspect("equal")
    a_map.tick_params(colors=INK2, labelsize=7)
    for s in a_map.spines.values():
        s.set_color(GRID)
    a_map.set_title("r2 の方策: 標準の開始から 1 周(元の sim-core.js)", color=INK, fontsize=10.5, loc="left")
    a_map.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1)
    # 黒シートの入口のまわり(s 5.6〜7.8 のフレーム)
    T = D["test"]
    zoom = None
    for ax, key, name in ((a_z2, "r2", "r2(選んだ方策)"), (a_z1, "r1", "r1(第 1 回の方策、比べるだけ)")):
        ax.set_facecolor(SURFACE)
        _draw_course(ax, world)
        lows = []
        for row in T:
            fr = np.array(row[key]["frames"])
            sm = np.mod(fr[:, 0], L_LOOP)
            m = (sm > 5.6) & (sm < 7.8)
            if not m.any():
                continue
            ax.plot(fr[m, 1], fr[m, 2], ".", ms=1.2, color=CV_COLOR[min(CV_COLOR, key=lambda c: abs(c - row["c_v"]))], zorder=4)
            k = np.argmin(np.where(m, fr[:, 3], np.inf))
            lows.append((fr[k, 3], fr[k, 1], fr[k, 2], row))
        lows.sort(key=lambda q: q[0])
        for d, xx, zz, row in lows[:3]:
            ax.plot(xx, zz, "o", ms=5, mfc="none", mec=ORANGE, mew=1.5, zorder=6)
            ax.text(xx + 0.03, zz - 0.03, f"{row['id']}(c_v {row['c_v']:g}){d * 100:.1f} cm", fontsize=7.5, color=INK, zorder=8,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=1))
            fm = row[key]["frame_min"]
            if 5.6 < np.mod(row[key]["frames"][fm][0], L_LOOP) < 7.8:   # 回の一番近い所がこの区間なら、その時の車体を描く
                _car(ax, *row[key]["at_min"], ORANGE)
        if zoom is None and lows:
            cx, cz = np.median([q[1] for q in lows]), np.median([q[2] for q in lows])
            zoom = (cx, cz)
        if zoom:
            ax.set_xlim(zoom[0] - 0.9, zoom[0] + 0.9); ax.set_ylim(zoom[1] + 0.9, zoom[1] - 0.9)
        ax.set_aspect("equal")
        ax.tick_params(colors=INK2, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        mins = [q[0] for q in lows]
        ax.set_title(f"黒シートの入口のまわり(s 5.6〜7.8)・試験の組 60 回: {name}\nこの区間の最小すき間 {min(mins) * 100:.1f} cm・中央値 {np.median(mins) * 100:.1f} cm(○ = 近い 3 回)",
                     color=INK, fontsize=9, loc="left")
    # s ごとの速さ(STD の連続 20 秒、5 段)と、試験の組の s ごとのすき間の下側(c_v ごとの最小)
    for cv in CV_LEVELS:
        r = D["continuous"][f"{cv:g}"] if f"{cv:g}" in D["continuous"] else D["continuous"][str(cv)]
        fr = np.array(r["frames"])
        s = fr[:, 0] - fr[0, 0]
        m = s <= L_LOOP
        a_v.plot(s[m], fr[m, 4], color=CV_COLOR[cv], lw=1.4, label=f"c_v {cv:g}(上限 {4.5 * cv:.2f} m/s)")
    _style(a_v, "r2: s ごとの実際の速さ(STD の連続 20 秒の 1 周目)", "速さ [m/s]", "標準の開始からの進み s [m]")
    a_v.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower right")
    bins = np.arange(0, L_LOOP + 0.25, 0.25)
    for key, ls, name in (("r2", "-", "r2"), ("r1", ":", "r1")):
        for cv in (0.5, 1.0):
            low = np.full(len(bins) - 1, np.nan)
            for row in T:
                if abs(row["c_v"] - cv) > 1e-9:
                    continue
                fr = np.array(row[key]["frames"])
                idx = np.clip(np.digitize(np.mod(fr[:, 0], L_LOOP), bins) - 1, 0, len(bins) - 2)
                for i, d in zip(idx, fr[:, 3]):
                    low[i] = d if np.isnan(low[i]) else min(low[i], d)
            a_d.plot((bins[:-1] + 0.125), low * 100, color=CV_COLOR[cv], lw=1.4 if key == "r2" else 1.1, ls=ls, label=f"{name}・c_v {cv:g}")
    a_d.axhline(3, color=INK2, lw=1, ls="--")
    a_d.axvspan(*POST_S, color=ORANGE, alpha=0.12, lw=0)
    a_d.text(POST_S[1] + 0.1, 1.0, "黒シートの入口の柱", fontsize=7.5, color=INK2)
    a_d.set_ylim(0, 25)
    _style(a_d, "試験の組 12 回ずつ: R 上の位置 s ごとのすき間の最小(0.25 m のマス)", "すき間 [cm]", "R 上の位置 s mod L [m]")
    a_d.legend(fontsize=7.5, frameon=True, facecolor=SURFACE, edgecolor="none", labelcolor=INK2, loc="lower right", ncol=2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "speed_r2_trajectories.png", dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / "speed_r2_trajectories.png")
    # 2) 回ごとの最小すき間(試験の組、r2 と r1、c_v ごと)
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.6), facecolor=SURFACE)
    for j, key in enumerate(("r1", "r2")):
        for k, cv in enumerate(CV_LEVELS):
            v = [row[key]["min_clearance"] * 100 if row[key]["collisions"] == 0 else 0.0 for row in T if abs(row["c_v"] - cv) < 1e-9]
            xs = k * 3 + j + np.linspace(-0.25, 0.25, len(v))
            ax.plot(xs, v, "o", ms=4.5, color=CV_COLOR[cv] if key == "r2" else "#b9b8b3", zorder=4)
    ax.axhline(3, color=INK2, lw=1, ls="--")
    ax.set_xticks([k * 3 + 0.5 for k in range(5)])
    ax.set_xticklabels([f"c_v {cv:g}\n(灰 r1・青 r2)" for cv in CV_LEVELS])
    _style(ax, "試験の組 SP-P60-R2 の回ごとの最小すき間(同じ 60 の開始。衝突は 0 として描く)", "回の最小すき間 [cm]", "")
    fig.tight_layout()
    fig.savefig(FIG / "speed_r2_clearance_by_run.png", dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / "speed_r2_clearance_by_run.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curves", "traj"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--figdata", type=Path)
    a = ap.parse_args()
    if a.cmd == "curves":
        curves(a.run, a.baseline, a.selection)
    else:
        traj(a.figdata)


if __name__ == "__main__":
    main()
