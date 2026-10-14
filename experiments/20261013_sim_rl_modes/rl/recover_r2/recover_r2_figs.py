"""離脱復活(recover)の学習の段 r2 の図(2026-10-14)。確認 recover r1 の走り(check_recover_r1/runs/rows_*.jsonl)と
r2 の数え(train/recover/r2/*.json)を読むだけ(走らせ直さない)。

    cd ~/jetformer/experiments/20261013_sim_rl_modes
    python -I rl/recover_r2/recover_r2_figs.py

書くもの(~/jetformer_reports/sim_rl_modes_20261013/fig/):
  recover_r2_black_sheet_exit.png  黒シートの出口(s 8.00〜8.25 m)で始まった舵のパルスの走り(左 = 全部戻る、右 = ほとんどカーテンに衝突)と、
                                    乱しの始まりの位置の分布(0.25 m のマス)と成否
  recover_r2_gate_miss.png          本当のゲートの外しの例(試験の組の P37・P59): sim がゲートを数えなくなった所
色: 左のパルス = 青、右のパルス = 橙(系列。役割で固定)、乱しの区間 = 紫の太線、衝突 = 赤の ×(文字を添える)、成功 = 緑。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
sys.path.insert(0, str(EXP / "rl" / "recover_r1"))
import recover_figs as F  # noqa: E402  (r1 の線の色・コースの描き方をそのまま使う)

REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
RUNS = REPORTS / "check_recover_r1" / "runs"
R2 = REPORTS / "train" / "recover" / "r2"
FIG = REPORTS / "fig"
L = F.LOOP_L
C_LEFT, C_RIGHT = "#2a78d6", "#eb6834"


def load_rows(sets):
    out = []
    for fn in sorted(RUNS.glob("rows_*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            if r["set"] in sets:
                out.append(r)
    return out


def black_sheet_exit(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F.jp_font()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.5, 7.0), facecolor=F.SURFACE, gridspec_kw={"width_ratios": [1.05, 1.0]})
    F.draw_track(ax, labels=False)
    xs, zs = F.REFLOOP["x"], F.REFLOOP["z"]
    ax.plot(list(xs) + [xs[0]], list(zs) + [zs[0]], color=F.GRID, lw=1, zorder=3)
    n = {1: [0, 0], -1: [0, 0]}
    for r in rows:
        pf, T = r["pf"], r["frames"]
        for k, e in enumerate(r["events"]):
            sm = e["s"] % L
            if e["type"] != "steer_pulse" or not (8.0 <= sm < 8.25):
                continue
            ps = e["ps"]
            nxt = r["events"][k + 1]["ps"] if k + 1 < len(r["events"]) else T
            coll = bool(r["collisions"]) and T <= nxt
            end = min(len(pf["x"]), ps + 45)
            sg = e["sign"]
            n[sg][0] += 1
            n[sg][1] += coll
            col = C_LEFT if sg > 0 else C_RIGHT
            ax.plot(pf["x"][ps:end], pf["z"][ps:end], color=col, lw=0.9, alpha=0.55, zorder=4)
            pe = min(e["pe"] if e["pe"] is not None else ps, end)
            ax.plot(pf["x"][ps:pe + 1], pf["z"][ps:pe + 1], color=F.C_PERT, lw=2.6, alpha=0.6, zorder=5, solid_capstyle="round")
            if coll and T - 1 < len(pf["x"]):
                ax.plot(pf["x"][T - 1], pf["z"][T - 1], marker="X", ms=7, color=F.C_BAD, mec=F.SURFACE, mew=0.6, zorder=7)
    # 拡大(s 7〜10.5 の辺り)
    pts = np.array([F.loop_xy(s) for s in np.linspace(7.0, 10.6, 40)])
    ax.set_adjustable("datalim")
    ax.set_xlim(pts[:, 0].min() - 0.9, pts[:, 0].max() + 0.9)
    ax.set_ylim(pts[:, 1].max() + 0.9, pts[:, 1].min() - 0.9)
    for srf in F.WORLD["surfaces"]:
        name = {"shortcut": "黒シート", "turf": "芝", "ptfe": "PTFE", "blue-mat": "青マット"}.get(srf["id"])
        if name:
            ax.text(srf["x"], srf["z"], name, fontsize=7.5, color=F.INK2 if srf["id"] != "shortcut" else F.SURFACE, ha="center", va="center", zorder=1, clip_on=True)
    for s, lab in ((8.0, "s 8.00"), (9.0, "s 9.0"), (10.0, "s 10.0")):
        x, z = F.loop_xy(s)
        ax.plot(x, z, marker="|", ms=12, color=F.INK, zorder=8)
        ax.text(x + 0.05, z - 0.08, lab, fontsize=8, color=F.INK, zorder=8)
    ax.plot([], [], color=C_LEFT, lw=2, label=f"左へのパルス {n[1][0]} 回(衝突 {n[1][1]})")
    ax.plot([], [], color=C_RIGHT, lw=2, label=f"右へのパルス {n[-1][0]} 回(衝突 {n[-1][1]})")
    ax.plot([], [], color=F.C_PERT, lw=3, label="乱しの区間(学習から外す)")
    ax.plot([], [], marker="X", ls="none", color=F.C_BAD, label="衝突(カーテン)")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor=F.SURFACE, edgecolor=F.GRID)
    ax.set_title("黒シートの出口(s 8.00〜8.25 m)で始まった舵のパルス 123 回の、その後 1.5 秒\n"
                 "離脱復活の方策。確認 r1 の走り(試験の組の作り直し 100 回 + 新しい開始 200 回)\n"
                 "乱しの始まりの速さは全部 2.9 m/s 未満(中央値 2.5 m/s)", color=F.INK, fontsize=9.5, loc="left")
    # 右: 乱しの始まりの位置の分布と成否
    bins = np.arange(0, L + 0.25, 0.25)
    ok, bad = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for r in rows:
        T = r["frames"]
        for k, e in enumerate(r["events"]):
            nxt = r["events"][k + 1]["ps"] if k + 1 < len(r["events"]) else T
            coll = bool(r["collisions"]) and T <= nxt
            i = min(len(ok) - 1, int((e["s"] % L) / 0.25))
            (bad if coll else ok)[i] += 1
    tot = ok.sum() + bad.sum()
    F.style(bx, f"乱しの始まりの位置(R 上の s、0.25 m のマス)。出来事 {int(tot)}", "出来事の数", "R 上の s [m]")
    bx.bar(bins[:-1], ok, width=0.23, align="edge", color=F.C_GOOD, alpha=0.75, label="そのあと衝突なし")
    bx.bar(bins[:-1], bad, width=0.23, align="edge", bottom=ok, color=F.C_BAD, alpha=0.85, label="そのあと衝突")
    for a, b, name in ((5.9, 8.0, "黒シートの区間\n(始めない)"), (14.6, 19.3, "奥の直線\n(始めない)")):
        bx.axvspan(a, b, color=F.GRID, alpha=0.6, zorder=0)
        bx.text((a + b) / 2, bx.get_ylim()[1] * 0.40, name, ha="center", fontsize=8, color=F.INK2)
    i8, i19 = int(8.0 / 0.25), int(19.25 / 0.25)
    for i, lab in ((i8, "8.00〜8.25"), (i19, "19.25〜19.50")):
        h = ok[i] + bad[i]
        bx.annotate(f"s {lab} m: {h / tot:.0%}\n(衝突 {int(bad[i])}/{int(h)})", (bins[i] + 0.12, h), xytext=(bins[i] + (1.2 if i == i8 else -6.5), h * 0.92),
                    fontsize=8.5, color=F.INK, arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.8))
    bx.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.suptitle("離脱復活 r2: 乱しは「始めてよい区間に入るまで遅らせる」決まりのため黒シートの出口と奥の直線の出口に集まり、"
                 "黒シートの出口では右へのパルスがほぼ必ずカーテンに当たる", color=F.INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "recover_r2_black_sheet_exit.png"
    fig.savefig(out, dpi=110, facecolor=F.SURFACE)
    plt.close(fig)
    return out


def gate_miss(rows, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F.jp_font()
    gate_s = sorted(F.REFLOOP["gate_s"])
    thr = max((gate_s[(i + 1) % 33] - gate_s[i]) % L for i in range(33)) + 0.5
    byid = {r["id"]: r for r in rows}
    fig, axes = plt.subplots(1, len(ids), figsize=(7.6 * len(ids), 6.4), facecolor=F.SURFACE)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, ids):
        r = byid[i]
        pf = r["pf"]
        F.draw_track(ax, labels=True)
        x, z, s, g = np.array(pf["x"]), np.array(pf["z"]), np.array(pf["s"]), pf["g"]
        ax.plot(x, z, color=F.C_REC, lw=1.3, zorder=4)
        learn = np.array(pf["l"])
        idx = np.nonzero(learn == 0)[0]
        if len(idx):
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                j = np.arange(run[0], min(len(x), run[-1] + 2))
                ax.plot(x[j], z[j], color=F.C_PERT, lw=4, alpha=0.8, zorder=5, solid_capstyle="round")
        # 数えが止まった所: 最後に g が変わったフレームから s がしきい値を越えたフレームまで
        last, f0 = 0, None
        for f in range(1, len(s)):
            if g[f] != g[f - 1]:
                last = f
            if s[f] - s[last] > thr:
                f0 = last
                break
        if f0 is not None:
            sm = s[f0] % L
            nxt = min((gs for gs in gate_s if gs > sm + 1e-6), default=gate_s[0] + L)
            gx, gz = F.loop_xy(nxt)
            ax.add_patch(matplotlib.patches.Circle((gx, gz), 0.65, fill=False, ec=F.C_BAD, lw=1.6, ls="--", zorder=6))
            ax.text(gx, gz + 0.95, "外したゲート(破線の円 = ゲートの中心から 0.65 m)", fontsize=8.5, color=F.C_BAD, ha="center", va="top", zorder=8,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=F.SURFACE, ec="none", alpha=0.85))
            ax.plot(x[f0], z[f0], marker="o", ms=7, color=F.INK, zorder=8)
        ax.plot(x[0], z[0], marker=">", ms=8, color=F.INK, zorder=8)
        laps = "・".join(f"{l['elapsed']:.2f} 秒" for l in r["laps"]) or "なし"
        ax.set_title(f"試験の組 {i}: 20 秒走り切り・衝突 0(有効な回)だが sim の周は {len(r['laps'])}({laps})\n"
                     f"紫の太線 = 乱しの区間、● = sim の数えが止まった所", color=F.INK, fontsize=9.5, loc="left")
    fig.suptitle("本当のゲートの外し: 有効な 38 回のうち 7 回(r1 の報告の「0」は数え方の誤り。r2 で直した)", color=F.INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "recover_r2_gate_miss.png"
    fig.savefig(out, dpi=110, facecolor=F.SURFACE)
    plt.close(fig)
    return out


def main():
    import matplotlib.patches  # noqa: F401
    rows = load_rows({"REPLAY-P100", "FRESH-P"})
    print(black_sheet_exit(rows))
    print(gate_miss([r for r in rows if r["set"] == "REPLAY-P100"], ["P37", "P59"]))


if __name__ == "__main__":
    main()
