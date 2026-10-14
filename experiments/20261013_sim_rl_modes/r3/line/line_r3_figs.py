"""経路いろいろ r3 の図(2026-10-14、r3 の学習の段 line)。判定には使わない(判定は r3/line/eval_line_r3.mjs の結果)。

    python r3/line/line_r3_figs.py all        # 下の 4 枚(データがあるものだけ)
    python r3/line/line_r3_figs.py curves     # r3/fig/line_r3_learning_curve.png(学習の値と、検証の組の判定の値)
    python r3/line/line_r3_figs.py heading    # r3/fig/line_r3_heading_clearance.png(開始の向きのずれと最小すき間。試験の組、r2 の方策と r3)
    python r3/line/line_r3_figs.py topdown    # r3/fig/line_r3_head_topdown.png(大きく横を向いた開始の最初の 3 秒の軌跡)
    python r3/line/line_r3_figs.py lines      # r3/fig/line_r3_fam6_lines.png(LN-FAM6 の線: r3・r2・最速)
色: r3 = 青(系列 1)、r2 の方策 + シールド = 水色(系列 3)、r2 の方策(シールドなし)= 橙(系列 2)。最速 = 灰。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl")
import line_figs as LF  # noqa: E402  (コースの絵・日本語の字体。読むだけ)

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
O = REP / "r3/line_r3"
FIG = REP / "r3/fig"
SURFACE, INK, INK2, GRID, MUTED = LF.SURFACE, LF.INK, LF.INK2, LF.GRID, LF.MUTED
C_R3, C_R2N, C_R2S = LF.S1, LF.S2, LF.S3


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LF._jp_font()
    return plt


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _style(ax, title, ylabel, xlabel="r3 の学習のステップ数 [百万]"):
    LF._style(ax, title, ylabel, xlabel)


def val_row(d: dict) -> dict:
    J, R = d["judged"], d["reported"]
    l1 = J["L1_r3"]["value"]
    l3 = J["L3_tracking"]["value"] or {}
    l4 = J["L4_distinct"].get("value") or {}
    return {"P": int(l1["P"].split("/")[0]), "H": int(l1["HEAD"].split("/")[0]), "L2": J["L2_r3"]["value"], "head_min": R.get("head_min_clearance_all"),
            "L3_med": l3.get("median"), "L3_p95": l3.get("p95"), "L4_pair": l4.get("min_pair_max_1m_avg"), "L4_lr": l4.get("left_right_mean"),
            "L4_io": l4.get("inside_outside_mean"), "L5": J["L5_vm3"]["value"], "shP": (R.get("shield_fraction") or {}).get("P"),
            "shH": (R.get("shield_fraction") or {}).get("HEAD"), "stalls": len(R.get("stalls") or []), "all": d["all_judged_pass"],
            "n_pass": sum(1 for v in J.values() if v.get("pass") is True)}


def curves():
    plt = _plt()
    run = O / "train/main"
    prog = []
    if (run / "progress.jsonl").exists():
        for line in (run / "progress.jsonl").read_text().splitlines():
            r = json.loads(line)
            e = r.get("episodes", {})
            if e.get("n"):
                rs = e.get("reasons", {})
                n = e["n"]
                prog.append({"x": r["steps"] / 1e6, "track": e.get("tracking_abs_mean"), "surv": e.get("survived_20s_frac"), "clear": e.get("min_clearance_mean"),
                             "r_clear": rs.get("clearance", 0) / n, "r_slow": rs.get("slow", 0) / n, "r_coll": rs.get("collision", 0) / n})
    evals = []
    base = _load(O / "analysis/val_r2policy.json")
    if base:
        evals.append({"x": 0.0, **val_row(base)})
    for p in sorted((run / "evals/line").glob("step_*.json")):
        evals.append({"x": int(p.stem.split("_")[1]) / 1e6, **val_row(json.loads(p.read_text()))})
    sel = _load(O / "selection.json")
    sel_x = sel["selected"]["step"] / 1e6 if sel else None
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5), facecolor=SURFACE)

    def tpanel(ax, keys, title, ylabel, labels=None, colors=None):
        _style(ax, title, ylabel)
        if not prog:
            return
        x = np.array([q["x"] for q in prog])
        for i, k in enumerate(keys):
            y = LF.smooth([q[k] if q[k] is not None else np.nan for q in prog], 15)
            ax.plot(x, y, color=(colors or [C_R3, C_R2N, C_R2S])[i], lw=2, label=(labels or [None])[i] if labels else None)
        if labels:
            ax.legend(fontsize=8, frameon=False)

    def vpanel(ax, series, title, ylabel, hlines=(), ylim=None):
        _style(ax, title, ylabel)
        if not evals:
            return
        x = np.array([q["x"] for q in evals])
        for i, (k, lab, col) in enumerate(series):
            y = np.array([q[k] if q[k] is not None else np.nan for q in evals], float)
            ax.plot(x, y, color=col, lw=2, marker="os"[i % 2], ms=7 if i == 0 else 4, label=lab, zorder=3 + i)
        for yv, lab in hlines:
            ax.axhline(yv, color=INK2, lw=1, ls="--")
            ax.text(ax.get_xlim()[1] if False else x.max(), yv, f" {lab}", color=INK2, fontsize=8, va="bottom", ha="right")
        if sel_x is not None:
            ax.axvline(sel_x, color=MUTED, lw=1, ls=":")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, frameon=False)

    tpanel(axes[0, 0], ["track"], "学習: 追従の誤差の平均(開始から 3 m 以降)", "m")
    tpanel(axes[0, 1], ["surv"], "学習: 20 秒を走り切った回の割合", "割合")
    tpanel(axes[0, 2], ["r_clear", "r_slow", "r_coll"], "学習: 終わり方の割合", "割合",
           labels=["すき間 3 cm 未満", "止まりの見張り(進み 0.6 m 未満)", "衝突"], colors=[C_R3, C_R2N, C_R2S])
    vpanel(axes[1, 0], [("head_min", "LN-VHEAD40-R3 の最小(選び方)", C_R3), ("L2", "L2(有効な回 + FAM6)", C_R2S)],
           "検証: 最小すき間(0 = r2 の方策 + シールド)", "m", hlines=[(.03, "L2 3 cm"), (.04, "シールド 4 cm")])
    vpanel(axes[1, 1], [("L3_p95", "95% 点", C_R3), ("L3_med", "中央値", C_R2S)], "検証: L3 追従の誤差", "m", hlines=[(.12, "12 cm"), (.05, "5 cm")])
    vpanel(axes[1, 2], [("L4_pair", "L4 どの 2 つも(1 m の最大)", C_R3), ("L5", "L5 VM3", C_R2S)], "検証: 線の違い", "m", hlines=[(.08, "0.08 m")])
    vpanel(axes[2, 0], [("shP", "LN-V60-R3", C_R3), ("shH", "LN-VHEAD40-R3", C_R2N)], "検証: シールドが替えたフレームの割合", "割合", hlines=[(.02, "L7 2%")])
    vpanel(axes[2, 1], [("P", "LN-V60-R3(/60)", C_R3), ("H", "LN-VHEAD40-R3(/40)", C_R2N)], "検証: valid_episode_r3 の数", "回")
    vpanel(axes[2, 2], [("n_pass", "満たした項目の数(/8)", C_R3), ("stalls", "止まりの回", C_R2N)], "検証: 判定の数", "数")
    fig.suptitle("経路いろいろ r3: 学習(大きく横を向いた開始 35%・すき間の壁 7 cm・止まりの見張り)と検証の組(シールドあり)。点線 = 選んだチェックポイント",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "line_r3_learning_curve.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


def _test_sets():
    return [("r3 の方策 + シールド(判定)", O / "eval_test_r3.json", C_R3), ("r2 の方策 + シールド(比べる値)", O / "compare/eval_test_r2policy_shield.json", C_R2S),
            ("r2 の方策・シールドなし(比べる値)", O / "compare/eval_test_r2policy_noshield.json", C_R2N)]


def heading():
    plt = _plt()
    sets = [(lab, _load(p), c) for lab, p, c in _test_sets()]
    sets = [s for s in sets if s[1]]
    if not sets:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    _style(ax, "試験の組(P60・HEAD60・EXT60)の回ごと: 開始の向きのずれ |Δψ| と最小すき間(衝突 = 0)", "最小すき間 [m]", "開始の車の向きと R の向きのずれ |Δψ| [rad]")
    for k, (lab, d, col) in enumerate(sets):
        rows = [r for t in ("LN-P60-R3", "LN-HEAD60-R3", "LN-EXT60-R3") for r in d["tests"].get(t, [])]
        x = np.array([abs(r["start"]["dpsi"]) for r in rows])
        y = np.array([0.0 if r.get("collisions") else (r.get("min_clearance") or 0.0) for r in rows])
        ax.scatter(x + (k - 1) * 0.008, y, s=22, color=col, alpha=0.85, edgecolors=SURFACE, linewidths=1.0, label=lab, zorder=3 + k)
    ax.axhline(.03, color=INK2, lw=1, ls="--"); ax.text(0.02, .031, "L2 3 cm", color=INK2, fontsize=8)
    ax.axhline(.04, color=MUTED, lw=1, ls=":"); ax.text(0.02, .041, "シールド 4 cm", color=INK2, fontsize=8)
    ax.axvline(1.0, color=MUTED, lw=1, ls=":"); ax.text(1.01, .2, "大きく横を向いた開始(≥ 1 rad)", color=INK2, fontsize=8)
    ax.set_ylim(-0.005, 0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax = axes[1]
    _style(ax, "大きく横を向いた開始(|Δψ| ≥ 1 rad)の最小すき間の分布", "回の数", "最小すき間 [cm]")
    bins = np.arange(0, 26, 1.0)
    for lab, d, col in sets:
        rows = [r for t in ("LN-P60-R3", "LN-HEAD60-R3", "LN-EXT60-R3") for r in d["tests"].get(t, []) if abs(r["start"]["dpsi"]) >= 1.0]
        y = [0.0 if r.get("collisions") else 100 * (r.get("min_clearance") or 0.0) for r in rows]
        ax.hist(np.clip(y, 0, 25), bins=bins, histtype="step", lw=2, color=col, label=f"{lab.split('(')[0]}(n = {len(y)})")
    ax.axvline(3, color=INK2, lw=1, ls="--")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "line_r3_heading_clearance.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


def topdown():
    plt = _plt()
    sets = [(lab, _load(p), c) for lab, p, c in _test_sets()]
    sets = [s for s in sets if s[1] and any("path" in r for r in s[1]["tests"].get("LN-HEAD60-R3", []))]
    if not sets:
        return
    ref = sets[0][1]["tests"]["LN-HEAD60-R3"]
    gates = sorted({r["start"]["gate"] for r in ref}, key=lambda g: -sum(1 for r in ref if r["start"]["gate"] == g))[:3]
    fig, axes = plt.subplots(len(sets), len(gates), figsize=(5.2 * len(gates), 4.6 * len(sets)), facecolor=SURFACE, squeeze=False)
    for i, (lab, d, col) in enumerate(sets):
        rows = {r["id"]: r for r in d["tests"]["LN-HEAD60-R3"]}
        for j, g in enumerate(gates):
            ax = axes[i, j]
            LF.draw_course(ax)
            xs, zs = [], []
            for r in rows.values():
                if r["start"]["gate"] != g or "path" not in r:
                    continue
                P = np.array(r["path"][:46])   # 2 フレームおき → 最初の 3 秒
                bad = bool(r.get("collisions")) or (r.get("min_clearance") or 0) < .03
                ax.plot(P[:, 0], P[:, 1], color=INK if bad else col, lw=2.6 if bad else 1.2, alpha=0.95 if bad else 0.7, zorder=4 + bad)
                ax.plot(P[0, 0], P[0, 1], "o", ms=4, color=INK, zorder=6)
                xs += list(P[:, 0]); zs += list(P[:, 1])
            if xs:
                cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
                half = max(max(xs) - min(xs), max(zs) - min(zs)) / 2 + 0.4
                ax.set_xlim(cx - half, cx + half); ax.set_ylim(cz + half, cz - half)
            n = sum(1 for r in rows.values() if r["start"]["gate"] == g)
            nb = sum(1 for r in rows.values() if r["start"]["gate"] == g and (r.get("collisions") or (r.get("min_clearance") or 0) < .03))
            mn = min([(0 if r.get("collisions") else r.get("min_clearance") or 0) for r in rows.values() if r["start"]["gate"] == g] or [np.nan])
            ax.set_title(f"{lab}\nゲート {g} からの開始 {n} 回: 3 cm 未満・衝突 {nb} 回、最小 {100 * mn:.1f} cm", color=INK, fontsize=9, loc="left")
    fig.suptitle("LN-HEAD60-R3(R に対して 1 rad 以上横を向いた開始)の最初の 3 秒。黒点 = 開始、黒の太線 = 3 cm 未満か衝突した回(r2 の方策・シールドなしのゲート 16 の 1 回)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "line_r3_head_topdown.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


def lines():
    plt = _plt()
    srcs = [("r3", _load(O / "eval_test_r3.json"), C_R3), ("r2(シールドなし)", _load(O / "compare/eval_test_r2policy_noshield.json"), C_R2N)]
    srcs = [s for s in srcs if s[1]]
    if not srcs:
        return
    fast = _load(O / "compare/eval_fastest_p60.json")
    fams = ["c-0.8", "c0", "c+0.8", "inside", "outside", "late_apex"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), facecolor=SURFACE, sharex=True, sharey=True)
    for ax, fam in zip(axes.flat, fams):
        _style(ax, LF.FAM_JA.get(fam, fam), "R からの横ずれ e_y [m](左 +)", "R の上の位置 s [m](開始から 3 m 以降の最初の通過、5 cm のマス)")
        for lab, d, col in srcs:
            r = next((q for q in d["tests"]["LN-FAM6"] if q["id"] == fam), None)
            if not r or not r.get("prof"):
                continue
            y = np.array([v if v is not None else np.nan for v in r["prof"]["ey"]], float)
            ax.plot(np.arange(len(y)) * .05, y, color=col, lw=2, label=lab)
        if fast:
            P = [q for q in fast["tests"]["LN-P60-R3-FAST"] if q.get("prof")]
            if P:
                Y = np.array([[v if v is not None else np.nan for v in q["prof"]["ey"]] for q in P], float)
                with np.errstate(all="ignore"):
                    ax.plot(np.arange(Y.shape[1]) * .05, np.nanmedian(Y, 0), color=MUTED, lw=1.2, ls="--", label="最速(P60 の開始 60 回の中央値)")
        ax.axhline(0, color=GRID, lw=1)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    d0 = srcs[0][1]["judged"]["L4_distinct"].get("value") or {}
    fig.suptitle(f"LN-FAM6 の 6 つの形の線(元の sim-core.js、STD の開始)。r3: 左右 {d0.get('left_right_mean', 0):.3f} m・イン/アウト {d0.get('inside_outside_mean', 0):.3f} m・"
                 f"どの 2 つも {d0.get('min_pair_max_1m_avg', 0):.3f} m(L4 ≥ 0.20 / 0.10 / 0.08)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "line_r3_fam6_lines.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("curves", curves), ("heading", heading), ("topdown", topdown), ("lines", lines)]:
        if what in ("all", name):
            fn()
