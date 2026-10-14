"""離脱復活 r4 の図(2026-10-14、学習の段 recover r4)。jetformer_reports/sim_rl_modes_20261013/r3/fig/ に PNG を書く。

    python3 -I r3/rc4/figs_rc4.py table      # 立て直せる表 v1(r3)と v2(r4、ゲートの条件つき)の使えるマスの数(組ごと)
    python3 -I r3/rc4/figs_rc4.py curve      # 学習の経過と検証の組(C1・C2・C5 の 4 つ・C8・停止から 1 周)
    python3 -I r3/rc4/figs_rc4.py kicks      # 試験の組のキックの生き残り(スタートのゲートの区間の右・左・区間の外)、r3 の候補と r4
    python3 -I r3/rc4/figs_rc4.py band       # スタートのゲートの区間の右の余地(乱しなし)、r3・最速・r4

色はカテゴリの固定の順(青・橙・水色・黄)。どの図も凡例か直接の数字のラベルを付ける(色だけで見分けない)。r3 の図と同じ見た目。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

R3 = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3")
FIG = R3 / "fig"
RC4 = R3 / "rc4"
EXP = Path("/home/jetson/jetformer/experiments/20261013_sim_rl_modes")
LOOP = 26.119648724030924
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXT, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# 色は方策(もの)に付ける: r4 = 青、r3 の候補 = 橙、最速の方策 = 水色(どの図でも同じ)
ENTITY = {"r4": SERIES[0], "r3": SERIES[1], "fastest": SERIES[2]}


def setup():
    for f in ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic", "VL Gothic"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10, "lines.linewidth": 2})
    FIG.mkdir(parents=True, exist_ok=True)


def usable_by_combo(path: Path, rate=.95):
    tab = json.loads(path.read_text())
    out = {c: 0 for c in tab["combos"]}
    for k, (n, r) in tab["cells"].items():
        c = k.split("|", 2)[2]
        if n >= 3 and r / n >= rate - 1e-12:
            out[c] += 1
    return tab["combos"], out


def fig_table():
    combos, u1 = usable_by_combo(EXP / "r3/rc/recoverable_table_v1.json")
    _, u2 = usable_by_combo(EXP / "r3/rc4/recoverable_table_v2.json")
    name = {"steer_pulse": "舵のパルス", "excursion": "わざと外す", "brake": "ブレーキ", "burst": "踏み込み"}
    lab = []
    for c in combos:
        p = c.split(":")
        lab.append(f"{name[p[0]]} {p[1]}" + (f" {'左' if p[2] == 'L' else '右'}" if len(p) > 2 else ""))
    y = np.arange(len(combos))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    h = .38
    ax.barh(y - h / 2 - .01, [u1[c] for c in combos], height=h, color=SERIES[1], label="表 v1(r3: 立て直せる)")
    ax.barh(y + h / 2 + .01, [u2[c] for c in combos], height=h, color=SERIES[0], label="表 v2(r4: + ゲートを外さない)")
    for i, c in enumerate(combos):
        ax.text(u2[c] + 2, y[i] + h / 2, str(u2[c]), va="center", fontsize=8, color=MUTED)
        ax.text(u1[c] + 2, y[i] - h / 2, str(u1[c]), va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y, lab)
    ax.invert_yaxis()
    ax.set_xlabel("使えるマスの数(サンプル 3 以上・立て直せる割合 ≥ 0.95)")
    ax.set_title(f"立て直せる表: 使えるマス v1 {sum(u1.values())} → v2 {sum(u2.values())}", loc="left")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "recover_r4_table.png", dpi=130)


def fig_curve(run="main"):
    d = RC4 / "train" / run
    rows = []
    for p in sorted((d / "evals" / "recover").glob("step_*.json")):
        e = json.loads(p.read_text())
        J = e["judged"]
        rows.append({"step": int(p.stem.split("_")[1]) / 1e6, "C1": int(J["C1_r3"]["value"].split("/")[0]), "C2": J["C2"]["value"],
                     "c5": J["C5_r4"]["rates"], "C8": J["C8"]["value"]["rate"], "lap": J["C6_r4"]["value"]["lap_from_stop"],
                     "minc": J["C6_r4"]["value"]["min_clearance"]})
    r3v = json.loads((R3 / "rc3/train/main/evals/recover/step_005013504.json").read_text())
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
    x = [r["step"] for r in rows]

    def panel(ax, ys, title, thr=None, lo=None, hi=None, labels=None):
        for i, yv in enumerate(ys):
            ax.plot(x, yv, marker="o", ms=5, color=SERIES[i], label=None if labels is None else labels[i])
        if thr is not None:
            ax.axhline(thr, color=MUTED, lw=1, ls="--")
            ax.text(x[0] if x else 0, thr, " 基準", va="bottom", fontsize=8, color=MUTED)
        if lo is not None:
            ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("学習のステップ(百万、r3 の 5,013,504 から)")
        if labels:
            ax.legend(frameon=False, fontsize=8)
    panel(axs[0, 0], [[r["C1"] for r in rows]], "C1_r3 有効な回(/100)", 90, 80, 101)
    panel(axs[0, 1], [[r["C2"] for r in rows]], "C2 出来事の成功", .95, .9, 1.0)
    panel(axs[0, 2], [[r["c5"][k] for r in rows] for k in ("grid_yaw", "grid_lateral", "random_yaw", "random_lateral")],
          "C5_r4 キックの生き残り", .9, .6, 1.02, ["STD 向き", "STD 横", "新しい開始 向き", "新しい開始 横"])
    panel(axs[1, 0], [[r["C8"] for r in rows]], "C8 ゲートの外しのある有効な回の割合", .10, 0, max([.45] + [r["C8"] or 0 for r in rows]))
    panel(axs[1, 1], [[r["lap"] for r in rows]], "C6_r4 停止から 1 周(秒)", 9.0, 8.0, 9.4)
    panel(axs[1, 2], [[r["minc"] * 100 for r in rows]], "C6_r4 乱しなしの最小すき間(cm)", 3.0, 0, 8)
    g = r3v["judged"]
    axs[0, 1].text(.02, .05, f"r3 の候補(r3 の検証の組): C2 {g['C2']['value']:.3f}", transform=axs[0, 1].transAxes, fontsize=8, color=MUTED)
    fig.suptitle("離脱復活 r4: 検証の組(シールドあり)", x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "recover_r4_learning_curve.png", dpi=130)


def kick_groups(path: Path):
    e = json.loads(path.read_text())
    out = {}
    for t in ("RC-KICK-R4", "RC-KICKR-R4"):
        rows = [r for r in e["tests"].get(t, []) if r.get("kick_allowed")]
        for r in rows:
            m = ((r["kick_s"] % LOOP) + LOOP) % LOOP
            band = .30 <= m <= 3.70
            sign = (r["kick"].get("yaw") or r["kick"].get("lateral"))
            g = "区間・右" if band and sign < 0 else "区間・左" if band else "区間の外"
            k = (t, g)
            n, s = out.get(k, (0, 0))
            out[k] = (n + 1, s + int(bool(r.get("kick_survival_r3"))))
    return out


def fig_kicks():
    items = [("r4(選んだ方策)", RC4 / "eval_test.json", "r4"), ("r3 の候補", RC4 / "eval_r3_test.json", "r3"), ("最速の方策", RC4 / "eval_fastest_test.json", "fastest")]
    items = [(n, p, k) for n, p, k in items if p.exists()]
    groups = [(t, g) for t in ("RC-KICK-R4", "RC-KICKR-R4") for g in ("区間・右", "区間・左", "区間の外")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = .8 / max(1, len(items))
    for i, (name, p, ent) in enumerate(items):
        kg = kick_groups(p)
        vals = [kg.get(k, (0, 0)) for k in groups]
        xs = np.arange(len(groups)) + (i - (len(items) - 1) / 2) * w
        ax.bar(xs, [s / n if n else 0 for n, s in vals], width=w - .03, color=ENTITY[ent], label=name)
        for xx, (n, s) in zip(xs, vals):
            ax.text(xx, (s / n if n else 0) + .01, f"{s}/{n}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.axhline(.9, color=MUTED, lw=1, ls="--")
    ax.set_xticks(np.arange(len(groups)), [f"{'STD・時刻の格子' if t == 'RC-KICK-R4' else '新しい開始'}\n{g}" for t, g in groups], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("生き残り(判定に使う位置のキック)")
    ax.set_title("試験の組のキック: スタートのゲートの区間(s 0.30〜3.70 m)の右・左と区間の外", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "recover_r4_kicks.png", dpi=130)


def fig_band():
    files = [RC4 / "diag/band_line_r3.json", RC4 / "diag/band_line_r4.json"]
    pols = {}
    for f in files:
        if f.exists():
            pols.update(json.loads(f.read_text())["policies"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (name, v) in enumerate(pols.items()):
        pts = np.array(v["points"])
        if not len(pts):
            continue
        bins = np.arange(0, 4.55, .1)
        idx = np.digitize(pts[:, 0], bins) - 1
        for j, (col, ax) in enumerate(((2, axs[0]), (3, axs[1]))):
            mean = [pts[idx == b, col].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)]
            mn = [pts[idx == b, col].min() if (idx == b).any() else np.nan for b in range(len(bins) - 1)]
            xc = bins[:-1] + .05
            nm = {"r3": "r3 の候補", "fastest": "最速の方策", "r4": "r4(選んだ方策)"}.get(name, name)
            ax.plot(xc, np.array(mean) * 100, color=ENTITY.get(name, SERIES[i]), label=f"{nm}(平均)")
            ax.plot(xc, np.array(mn) * 100, color=ENTITY.get(name, SERIES[i]), lw=1, ls=":", label=f"{nm}(最小)")
    for ax in axs:
        ax.axvspan(.30, 3.70, color=GRID, alpha=.5, lw=0)
        ax.set_xlabel("R 上の s(m、スタートのゲートのあと)")
        ax.legend(frameon=False, fontsize=8)
    axs[0].axhline(30, color=MUTED, lw=1, ls="--")
    axs[0].set_ylabel("右の余地 e_y + ρR(s)(cm)")
    axs[0].set_title("右の余地(学習の壁の目標 30 cm、灰色が区間)", loc="left", fontsize=10)
    axs[1].axhline(15, color=MUTED, lw=1, ls="--")
    axs[1].set_ylabel("すき間 d_min(cm)")
    axs[1].set_title("すき間(破線 = 横のキック 15 cm)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "recover_r4_band_line.png", dpi=130)


if __name__ == "__main__":
    setup()
    what = sys.argv[1]
    {"table": fig_table, "curve": fig_curve, "kicks": fig_kicks, "band": fig_band}[what](*sys.argv[2:])
