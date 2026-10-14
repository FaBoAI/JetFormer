"""経路いろいろ r4b の図(2026-10-14 夜、line r4 の段のやり直し。r3/line4/line_r4_figs.py の写しを元に)。判定には使わない(判定は eval_line_r4b.mjs の結果)。

    python r3/line4b/line_r4b_figs.py all        # 下の 4 枚(データがあるものだけ)
    python r3/line4b/line_r4b_figs.py curves     # r3/fig/line_r4b_learning_curve.png(学習の値と、検証の組の判定の値)
    python r3/line4b/line_r4b_figs.py corner     # r3/fig/line_r4b_test_sets.png(試験の組ごとの失敗の数と、角の組の最小すき間)
    python r3/line4b/line_r4b_figs.py topdown    # r3/fig/line_r4b_policy_alone.png(G30R120・BOX120 の最初の 4 秒、シールドなし: r3 の方策と選んだ方策)
    python r3/line4b/line_r4b_figs.py lines      # r3/fig/line_r4b_fam6_lines.png(LN-FAM6 の線: 選んだ方策・最速)
色(検証済みの 3 色、validate_palette.js で合格): 選んだ方策 = 青、r3 の方策 + r3 のシールドの動き = 橙、r3 の方策 + シールド v5b = 緑。
シールドなしは同じ方策の色に斜線(色だけで分けない)。棒には数を直に書く(緑の明るさの注意 = 数を書いて補う)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl")
import line_figs as LF  # noqa: E402  (コースの絵・日本語の字体。読むだけ)

REP = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
O = REP / "r3/line_r4b"
FIG = REP / "r3/fig"
SURFACE, INK, INK2, GRID, MUTED = LF.SURFACE, LF.INK, LF.INK2, LF.GRID, LF.MUTED
C_R4, C_R3OLD, C_R3V5 = LF.S1, LF.S2, LF.S3
TEST_SETS = ["LN-P60-R4", "LN-HEAD60-R4", "LN-EXT60-R4", "LN-CORNER-R4", "LN-BOX120-R4", "LN-G30R120-R4B", "LN-FAM6"]
SET_JA = {"LN-P60-R4": "P60\n(Halton)", "LN-HEAD60-R4": "HEAD60\n(横向き)", "LN-EXT60-R4": "EXT60\n(形の端)", "LN-CORNER-R4": "CORNER116\n(角の箱)",
          "LN-BOX120-R4": "BOX120\n(ゲート16・17)", "LN-G30R120-R4B": "G30R120\n(ゲート29〜32×右寄り)", "LN-FAM6": "FAM6"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LF._jp_font()
    return plt


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _style(ax, title, ylabel, xlabel="r4b の学習のステップ数 [百万](0 = r4 の 401 万 = 初期化元、−1 = r4 の 301 万)"):
    LF._style(ax, title, ylabel, xlabel)


def val_row(d: dict) -> dict:
    J, R = d["judged"], d["reported"]
    l1 = J["L1_r4"]["value"]
    l3 = J["L3_tracking"]["value"] or {}
    l4 = J["L4_distinct"].get("value") or {}
    bs = R.get("by_set") or {}
    sf = R.get("shield_fraction") or {}
    n = lambda k: int(str(l1.get(k, "0/0")).split("/")[0])  # noqa: E731
    return {"P": n("LN-V60-R4"), "H": n("LN-VHEAD40-R4"), "C": n("LN-VCORNER-R4"), "B": n("LN-VBOX40-R4"), "G": n("LN-VG30R40-R4B"), "L2": J["L2_r4"]["value"],
            "shG": sf.get("LN-VG30R40-R4B"), "shGruns": (bs.get("LN-VG30R40-R4B") or {}).get("shield_runs"),
            "esc": sum((v or {}).get("escape_runs", 0) for v in bs.values()),
            "hard_min": R.get("hard_min_clearance"), "hard_fail": R.get("hard_fail"), "L3_med": l3.get("median"), "L3_p95": l3.get("p95"),
            "L4_pair": l4.get("min_pair_max_1m_avg"), "L5": J["L5_vm3"]["value"], "shP": sf.get("LN-V60-R4"), "shC": sf.get("LN-VCORNER-R4"), "shB": sf.get("LN-VBOX40-R4"),
            "stalls": sum((v or {}).get("stalls", 0) for v in bs.values()), "coll": sum((v or {}).get("collisions", 0) for v in bs.values()),
            "rev": sum((v or {}).get("reverse_runs", 0) for v in bs.values()), "all": d["all_judged_pass"],
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
                prog.append({"x": r["steps"] / 1e6, "track": e.get("tracking_abs_mean"), "surv": e.get("survived_20s_frac"),
                             "r_clear": rs.get("clearance", 0) / n, "r_slow": rs.get("slow", 0) / n, "r_coll": rs.get("collision", 0) / n})
    evals = [{"x": int(p.stem.split("_")[1]) / 1e6, **val_row(json.loads(p.read_text()))} for p in sorted((run / "evals/line").glob("step_*.json"))]
    r4x = {"003014656": -1.0, "004014080": 0.0}
    evals = sorted([{"x": r4x[p.stem.split("_")[-1]], **val_row(json.loads(p.read_text()))} for p in sorted((O / "train/r4_ckpt/evals").glob("r4_step_*.json"))
                    if p.stem.split("_")[-1] in r4x] + evals, key=lambda q: q["x"])
    sel = _load(O / "selection.json")
    sel_x = None
    if sel:
        s0 = sel["selected"]
        sel_x = s0["step"] / 1e6 if s0.get("origin") == "r4b" else r4x.get(f"{s0['step']:09d}")
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5), facecolor=SURFACE)

    def tpanel(ax, keys, title, ylabel, labels=None, colors=None):
        _style(ax, title, ylabel)
        if not prog:
            return
        x = np.array([q["x"] for q in prog])
        for i, k in enumerate(keys):
            y = LF.smooth([q[k] if q[k] is not None else np.nan for q in prog], 15)
            ax.plot(x, y, color=(colors or [C_R4, C_R3OLD, C_R3V5])[i], lw=2, label=(labels or [None])[i] if labels else None)
        if labels:
            ax.legend(fontsize=8, frameon=False)

    def vpanel(ax, series, title, ylabel, hlines=()):
        _style(ax, title, ylabel)
        if not evals:
            return
        x = np.array([q["x"] for q in evals])
        for i, (k, lab, col) in enumerate(series):
            y = np.array([q[k] if q[k] is not None else np.nan for q in evals], float)
            ax.plot(x, y, color=col, lw=2, marker="os^"[i % 3], ms=7 if i == 0 else 5, label=lab, zorder=3 + i)
        for yv, lab in hlines:
            ax.axhline(yv, color=INK2, lw=1, ls="--")
            ax.text(x.max(), yv, f" {lab}", color=INK2, fontsize=8, va="bottom", ha="right")
        if sel_x is not None:
            ax.axvline(sel_x, color=MUTED, lw=1, ls=":")
        ax.legend(fontsize=8, frameon=False)

    tpanel(axes[0, 0], ["track"], "学習: 追従の誤差の平均(開始から 3 m 以降)", "m")
    tpanel(axes[0, 1], ["surv"], "学習: 20 秒を走り切った回の割合", "割合")
    tpanel(axes[0, 2], ["r_clear", "r_slow", "r_coll"], "学習: 終わり方の割合", "割合", labels=["すき間 3 cm 未満", "止まりの見張り", "衝突"])
    vpanel(axes[1, 0], [("hard_min", "むずかしい組の最小(選び方)", C_R4), ("L2", "L2(有効な回 + FAM6)", C_R3V5)], "検証: 最小すき間", "m",
           hlines=[(.03, "L2 3 cm"), (.04, "シールド 4 cm")])
    vpanel(axes[1, 1], [("L3_p95", "95% 点", C_R4), ("L3_med", "中央値", C_R3V5)], "検証: L3 追従の誤差", "m", hlines=[(.12, "12 cm"), (.05, "5 cm")])
    vpanel(axes[1, 2], [("L4_pair", "L4 どの 2 つも(1 m の最大)", C_R4), ("L5", "L5 VM3", C_R3V5)], "検証: 線の違い", "m", hlines=[(.08, "0.08 m")])
    vpanel(axes[2, 0], [("shG", "LN-VG30R40-R4B(ゲート 29〜32 × 右寄り)", C_R4), ("shB", "LN-VBOX40-R4", C_R3OLD), ("shP", "LN-V60-R4", C_R3V5)],
           "検証: シールドが替えたフレームの割合(方策がシールドに頼る度合い)", "割合", hlines=[(.02, "L7 2%")])
    vpanel(axes[2, 1], [("hard_fail", "むずかしい組の失敗(衝突・止まり・3 cm 未満)", C_R4), ("esc", "抜け出し(後ろへ動く)の回", C_R3OLD), ("shGruns", "VG30R40 でシールドが働いた回(/40)", C_R3V5)],
           "検証: 失敗・抜け出し・シールドが働いた回の数", "回")
    vpanel(axes[2, 2], [("n_pass", "満たした項目の数(/11)", C_R4), ("coll", "衝突", C_R3OLD)], "検証: 判定の数", "数")
    fig.suptitle("経路いろいろ r4b: 学習(r4 の 401 万から。ゲート 28〜32 × 右寄りの形 15%・角 × 形の端 7% を足した)と検証の組(シールド v5b、高速版)。点線 = 選んだチェックポイント",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "line_r4b_learning_curve.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


def _variants():
    # (名前, 結果, 色, 斜線)。シールドなしは同じ方策の色に斜線
    return [("選んだ方策 + シールド v5b(判定)", O / "eval_test_r4b.json", C_R4, None), ("選んだ方策・シールドなし", O / "compare/eval_test_noshield.json", C_R4, "//"),
            ("r3 の方策 + r3 のシールドの動き(確認 r3 の候補)", O / "compare/eval_test_r3pol_r3shield.json", C_R3OLD, None),
            ("r3 の方策 + シールド v5b", O / "compare/eval_test_r3pol_v5b.json", C_R3V5, None)]


def _clr(r):
    if r.get("invalid_spawn") or r.get("collisions"):
        return 0.0
    return r.get("min_clearance") or 0.0


def corner():
    plt = _plt()
    vs = [(lab, _load(p), c, h) for lab, p, c, h in _variants()]
    vs = [v for v in vs if v[1]]
    if not vs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    _style(ax, "試験の組ごとの失敗の数(衝突 + 止まり + 有効な回の 3 cm 未満)", "回", "")
    w = 0.8 / len(vs)
    for k, (lab, d, col, hatch) in enumerate(vs):
        ys = []
        for t in TEST_SETS:
            rs = d["tests"].get(t, [])
            ys.append(sum(1 for r in rs if r.get("collisions") or r.get("stall") or (r.get("valid_r3") and (r.get("min_clearance") or 0) < .03)))
        xs = np.arange(len(TEST_SETS)) + (k - (len(vs) - 1) / 2) * w
        ax.bar(xs, ys, width=w * 0.92, color=SURFACE if hatch else col, edgecolor=col, hatch=hatch, lw=1.5, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.2, str(y), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(TEST_SETS)))
    ax.set_xticklabels([SET_JA[t] for t in TEST_SETS], fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axes[1]
    _style(ax, "角の組(CORNER + BOX + G30R)の回ごとの最小すき間の分布(衝突・止まり = 0)", "回の数", "最小すき間 [cm]")
    bins = np.arange(0, 21, 0.5)
    for lab, d, col, hatch in vs:
        rs = [r for t in ("LN-CORNER-R4", "LN-BOX120-R4", "LN-G30R120-R4B") for r in d["tests"].get(t, [])]
        if not rs:
            continue
        y = [0.0 if r.get("stall") else 100 * _clr(r) for r in rs]
        ax.hist(np.clip(y, 0, 20), bins=bins, histtype="step", lw=2, color=col, ls="--" if hatch else "-", label=f"{lab.split('(')[0]}(最小 {min(y):.2f} cm)")
    ax.axvline(3, color=INK2, lw=1, ls="--")
    ax.text(3.1, ax.get_ylim()[1] * 0.9, "L2 3 cm", color=INK2, fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "line_r4b_test_sets.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


def _bad(r):
    return bool(r.get("collisions")) or bool(r.get("invalid_spawn")) or bool(r.get("stall")) or (r.get("valid_r3") and (r.get("min_clearance") or 0) < .03)


def topdown():
    """方策そのもの(シールドなし)の走り: r3 の方策と選んだ方策を、G30R120(ゲート 29〜32 × 右寄り)と BOX120(ゲート 16・17)の最初の 4 秒で"""
    plt = _plt()
    rows_spec = [("r3 の方策・シールドなし", O / "compare/eval_r3pol_noshield_g30_box.json", C_R3OLD),
                 ("r4 の 401 万(r4b の初期化元)・シールドなし", O / "compare/eval_r4init_noshield_g30_box.json", C_R3V5),
                 ("選んだ方策・シールドなし", O / "compare/eval_test_noshield.json", C_R4)]
    vs = [(lab, _load(p), c) for lab, p, c in rows_spec]
    vs = [v for v in vs if v[1]]
    if not vs:
        return
    sets = [("LN-G30R120-R4B", "G30R120(ゲート 29〜32、左右 +0.16〜+0.20 m・向き +12〜+15°、const c0 −0.9〜−0.6)"),
            ("LN-BOX120-R4", "BOX120(ゲート 16・17、左右 −0.20〜−0.16 m・向き +12〜+15°)")]
    fig, axes = plt.subplots(len(vs), 2, figsize=(13, 5.2 * len(vs)), facecolor=SURFACE, squeeze=False)
    for i, (lab, d, col) in enumerate(vs):
        for j, (t, tj) in enumerate(sets):
            ax = axes[i, j]
            LF.draw_course(ax)
            rows = [r for r in d["tests"].get(t, []) if "path" in r]
            xs, zs = [], []
            for r in rows:
                P = np.array(r["path"][:61])   # 2 フレームおき → 最初の 4 秒
                b = _bad(r)
                ax.plot(P[:, 0], P[:, 1], color=INK if b else col, lw=2.4 if b else 0.9, alpha=0.95 if b else 0.5, zorder=4 + b)
                ax.plot(P[0, 0], P[0, 1], "o", ms=2.5, color=INK, zorder=6)
                xs += list(P[:, 0]); zs += list(P[:, 1])
            if xs:
                cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
                half = max(max(xs) - min(xs), max(zs) - min(zs)) / 2 + 0.3
                ax.set_xlim(cx - half, cx + half); ax.set_ylim(cz + half, cz - half)
            all_rows = d["tests"].get(t, [])
            nc = sum(1 for r in all_rows if r.get("collisions") or r.get("invalid_spawn"))
            ns = sum(1 for r in all_rows if r.get("stall"))
            n3 = sum(1 for r in all_rows if r.get("valid_r3") and (r.get("min_clearance") or 0) < .03)
            ax.set_title(f"{lab}\n{tj}\n{len(all_rows)} 回: 衝突 {nc}・止まり {ns}・有効で 3 cm 未満 {n3}", color=INK, fontsize=8.5, loc="left")
    fig.suptitle("方策そのもの(シールドなし)の最初の 4 秒。黒点 = 開始、黒の太線 = 衝突・止まり・3 cm 未満の回(判定はシールドありの組。これは方策の弱さの比べ)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "line_r4b_policy_alone.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


def lines():
    plt = _plt()
    srcs = [("選んだ方策(r4b)+ v5b", _load(O / "eval_test_r4b.json"), C_R4), ("r3 の方策 + v5b", _load(O / "compare/eval_test_r3pol_v5b.json"), C_R3V5)]
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
            P = [q for q in fast["tests"]["LN-P60-R4-FAST"] if q.get("prof")]
            if P:
                Y = np.array([[v if v is not None else np.nan for v in q["prof"]["ey"]] for q in P], float)
                with np.errstate(all="ignore"):
                    ax.plot(np.arange(Y.shape[1]) * .05, np.nanmedian(Y, 0), color=MUTED, lw=1.2, ls="--", label="最速(P60 の開始 60 回の中央値)")
        ax.axhline(0, color=GRID, lw=1)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    d0 = srcs[0][1]["judged"]["L4_distinct"].get("value") or {}
    fig.suptitle(f"LN-FAM6 の 6 つの形の線(元の sim-core.js、STD の開始)。r4b: 左右 {d0.get('left_right_mean', 0):.3f} m・イン/アウト {d0.get('inside_outside_mean', 0):.3f} m・"
                 f"どの 2 つも {d0.get('min_pair_max_1m_avg', 0):.3f} m(L4 ≥ 0.20 / 0.10 / 0.08)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "line_r4b_fam6_lines.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("curves", curves), ("corner", corner), ("topdown", topdown), ("lines", lines)]:
        if what in ("all", name):
            fn()
