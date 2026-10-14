"""経路いろいろ(line)の図(2026-10-13、学習の段 line)。判定には使わない(判定は rl/eval_modes.mjs の結果)。

    python rl/line_figs.py curves <run_dir>[@名前@始まりのステップ] ... [--selected 名前@ステップ]   # 学習の曲線 + 検証の組の判定の値(fig/line_learning_curve.png)
    python rl/line_figs.py traj <traj.json> [接尾辞]           # 上から見た軌跡(fig/line_fam6_topdown*.png・line_lr_overlay*.png・line_p60_topdown*.png・line_ey_profiles*.png)

traj.json は rl/line_traj.mjs の出力(元の sim-core.js で試験の組と同じ回を走らせた x・z・s・e_y・e*)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPORTS = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013")
FIG = REPORTS / "fig"
WORLD = json.loads((EXP / "snapshot" / "sim" / "world-v3.json").read_text())
REF = json.loads((HERE / "runtime" / "reference_loop.json").read_text())

# 図の色(dataviz の参照パレット。系列は固定の順番)
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
FAM_JA = {"c-0.8": "右寄り c0 −0.8", "c0": "中央 c0 0", "c+0.8": "左寄り c0 +0.8", "inside": "イン a 0.8", "outside": "アウト a 0.8",
          "late_apex": "レイトエイペックス a 0.8"}
FAMILY_JA = {"const_left": "左寄り", "const_right": "右寄り", "const_center": "中央", "inside": "イン", "outside": "アウト",
             "late_apex": "レイトエイペックス", "early_apex": "アーリーエイペックス", "random": "ランダム"}


def _jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            return


def _style(ax, title, ylabel, xlabel="環境のステップ数 [百万]"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def smooth(y, k=9):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[np.isfinite(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


# ---- 学習の曲線 ----
def load_run(run: Path):
    prog = []
    for line in (run / "progress.jsonl").read_text().splitlines():
        r = json.loads(line)
        e = r.get("episodes", {})
        if e.get("n"):
            prog.append({"steps": r["steps"], "track": e.get("tracking_abs_mean"), "lap1": e.get("lap1_mean"), "surv": e.get("survived_20s_frac"),
                         "clear": e.get("min_clearance_mean"), "ret": e.get("return_mean")})
    evals = []
    for p in sorted((run / "evals" / "line").glob("step_*.json")):
        d = json.loads(p.read_text())
        J = d["judged"]
        evals.append({"steps": int(p.stem.split("_")[1]), "L3_med": J["L3_tracking"]["value"]["median"], "L3_p95": J["L3_tracking"]["value"]["p95"],
                      "L4_lr": (J["L4_distinct"].get("value") or {}).get("left_right_mean"), "L4_io": (J["L4_distinct"].get("value") or {}).get("inside_outside_mean"),
                      "L4_pair": (J["L4_distinct"].get("value") or {}).get("min_pair_max_1m_avg"), "L5": J["L5_vm3"]["value"], "L2": J["L2_clearance"]["value"],
                      "L6": J["L6_lap"]["value"], "pass_n": sum(1 for v in J.values() if v.get("pass") is True), "all": d["all_judged_pass"]})
    return prog, evals


def curves(specs: list[str], out_name="line_learning_curve.png", selected: dict | None = None):
    """specs: "run_dir" か "run_dir@名前@ステップの始まり"(作り直しは初期化したチェックポイントのステップから描く)。系列 = run(固定の色の順)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _jp_font()
    runs = []
    for k, sp in enumerate(specs):
        parts = sp.split("@")
        run = Path(parts[0])
        name = parts[1] if len(parts) > 1 else run.name
        off = float(parts[2]) if len(parts) > 2 else 0.0
        prog, evals = load_run(run)
        for r in prog + evals:
            r["x"] = (r["steps"] + off) / 1e6
        runs.append({"name": name, "prog": prog, "evals": evals, "color": (S1, S2, S3)[k % 3]})
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5), facecolor=SURFACE)
    def train_panel(ax, key, title, ylabel, ylim=None):
        _style(ax, title, ylabel)
        for R in runs:
            x = np.array([r["x"] for r in R["prog"]])
            y = np.array([r[key] if r[key] is not None else np.nan for r in R["prog"]])
            ax.plot(x, y, color=R["color"], lw=0.8, alpha=0.25)
            ax.plot(x, smooth(y), color=R["color"], lw=2, label=R["name"])
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    def eval_panel(ax, key, title, ylabel, thr, thr_text):
        _style(ax, title, ylabel)
        for R in runs:
            ex = [r["x"] for r in R["evals"]]
            ey = [r[key] if r[key] is not None else np.nan for r in R["evals"]]
            ax.plot(ex, ey, color=R["color"], lw=2, marker="o", ms=5, label=R["name"])
        ax.axhline(thr, color=INK2, lw=1, ls=":")
        ax.text(ax.get_xlim()[0], thr, " " + thr_text, color=INK2, fontsize=8, va="bottom")
        if selected:
            for R in runs:
                for r in R["evals"]:
                    if R["name"] == selected.get("name") and r["steps"] == selected.get("step") and r[key] is not None:
                        ax.plot(r["x"], r[key], marker="o", ms=12, mfc="none", mec=INK, mew=1.5, zorder=6)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    train_panel(axes[0, 0], "track", "学習中: 線の追従の誤差(開始から 3 m 以降の平均)", "誤差 [m]", (0, None))
    train_panel(axes[0, 1], "lap1", "学習中: 1 周目の時間(停止から、有効な周)", "秒")
    train_panel(axes[0, 2], "surv", "学習中: 20 秒を走り切った回の割合", "割合", (0, 1.05))
    eval_panel(axes[1, 0], "L2", "検証の組 L2: 最小すき間", "m", 0.03, "しきい値 ≥ 0.03")
    eval_panel(axes[1, 1], "L3_med", "検証の組 L3: 追従の誤差の中央値", "m", 0.05, "しきい値 ≤ 0.05")
    eval_panel(axes[1, 2], "L3_p95", "検証の組 L3: 追従の誤差の 95% 点", "m", 0.12, "しきい値 ≤ 0.12")
    eval_panel(axes[2, 0], "L4_lr", "検証の組 L4: 左寄り・右寄りの差の平均", "m", 0.20, "しきい値 ≥ 0.20")
    eval_panel(axes[2, 1], "L4_pair", "検証の組 L4: 6 つの形のどの 2 つも違う所(最小)", "m", 0.08, "しきい値 ≥ 0.08")
    eval_panel(axes[2, 2], "L5", "検証の組 L5: 横ずれのばらつき VM3", "m", 0.08, "しきい値 ≥ 0.08")
    for ax in axes[1:].flat:
        ax.set_xlabel("環境のステップ数 [百万](作り直しは初期化したチェックポイントから)", color=INK2, fontsize=8)
    t = "強化学習モード(経路いろいろ): cond の学習。点線 = 合格のしきい値(検証の組 15 + 6 回・高速版のコア)"
    if selected:
        t += f"。黒い輪 = 選んだチェックポイント({selected.get('name')} {selected.get('step'):,})"
    fig.suptitle(t, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / out_name, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / out_name)


# ---- 軌跡 ----
RX, RZ, RS, L = np.array(REF["x"]), np.array(REF["z"]), np.array(REF["s"]), REF["length_m"]
# R の接線と左の法線(e_y の向きは軌跡から決める)
_tx, _tz = np.gradient(RX), np.gradient(RZ)
_n = np.hypot(_tx, _tz)
_tx, _tz = _tx / _n, _tz / _n


def ref_at(s):
    s = np.mod(np.asarray(s, float), L)
    xs = np.concatenate([RS, [L]])
    def interp(a):
        return np.interp(s, xs, np.concatenate([a, a[:1]]))
    tx, tz = interp(_tx), interp(_tz)
    n = np.hypot(tx, tz)
    return interp(RX), interp(RZ), tx / n, tz / n


def left_sign(runs):
    """e_y が正(左)の向きの法線 (−tz, tx)·sign を、軌跡と e_y から決める"""
    acc = 0.0
    for r in runs[:3]:
        s, ey = np.array(r["s"], float), np.array(r["ey"], float)
        x, z = np.array(r["x"], float), np.array(r["z"], float)
        # trace はステップ後なので log の 1 つ後と比べる
        n = min(len(s) - 1, len(x) - 1)
        px, pz, tx, tz = ref_at(s[1:n + 1])
        d = (x[:n] - px) * (-tz) + (z[:n] - pz) * tx
        ok = np.isfinite(d) & np.isfinite(ey[1:n + 1])
        acc += float(np.sum(d[ok] * ey[1:n + 1][ok]))
    return 1.0 if acc >= 0 else -1.0


def target_xy(r, sign):
    s, es = np.array(r["s"], float), np.array([v if v is not None else np.nan for v in r["es"]], float)
    px, pz, tx, tz = ref_at(s)
    return px + sign * es * (-tz), pz + sign * es * tx


def draw_course(ax, labels=False):
    from matplotlib.patches import Rectangle, Circle
    for s in WORLD["surfaces"]:
        if s["id"] == "carpet":
            continue
        col = {"shortcut": "#2b2b2b", "blue-mat": "#b7d3f6", "turf": "#cfe8cf", "ptfe": "#ece6f5"}.get(s["id"], "#e4e3df")
        ax.add_patch(Rectangle((s["x"] - s["w"] / 2, s["z"] - s["d"] / 2), s["w"], s["d"], facecolor=col, edgecolor="none", alpha=0.45, zorder=0))
    for g in WORLD["segments"]:
        ax.plot([g["a"][0], g["b"][0]], [g["a"][1], g["b"][1]], color="#6b6a66", lw=max(0.8, g["radius"] * 100), solid_capstyle="round", zorder=2)
    for d in WORLD["disks"]:
        ax.add_patch(Circle((d["x"], d["z"]), d["radius"], facecolor="#9b9a95" if d["radius"] > 0.05 else "#52514e", edgecolor="none", alpha=0.7, zorder=2))
    ax.plot(np.append(RX, RX[0]), np.append(RZ, RZ[0]), color=MUTED, lw=0.7, ls="-", alpha=0.6, zorder=2)
    b = WORLD["bounds"]
    ax.set_xlim(b["minX"], b["maxX"])
    ax.set_ylim(b["maxZ"], b["minZ"])
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=6)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def first_lap(r, s_extra=0.0):
    """開始から 1 周 + s_extra の範囲(フレームの番号)"""
    s = np.array(r["s"], float)
    idx = np.where(s - s[0] <= L + s_extra)[0]
    return idx


def stats(r):
    s, ey = np.array(r["s"], float), np.array(r["ey"], float)
    es = np.array([v if v is not None else np.nan for v in r["es"]], float)
    k = (s - s[0]) >= 3
    e = np.abs(ey[k] - es[k])
    e = e[np.isfinite(e)]
    return (float(np.median(e)) if len(e) else np.nan, float(np.quantile(e, 0.95)) if len(e) else np.nan)


def traj(path: Path, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _jp_font()
    d = json.loads(path.read_text())
    runs = d["runs"]
    sign = left_sign(runs)
    fam = [r for r in runs if r["test"] == "LN-FAM6"]
    p60 = [r for r in runs if r["test"] == "LN-P60"][:15]   # 図は最初の 15 回(family_plan_15 の 1 回分)
    FIG.mkdir(parents=True, exist_ok=True)
    pol = Path(d["policy"]["path"])
    tag = f"方策 {pol.parent.parent.name}/{pol.stem}(sha256 {d['policy']['sha256'][:12]})、" + ("元の sim-core.js" if d.get("core", "original") == "original" else "高速版のコア") + "、" + ("試験" if d.get("set") == "test" else "検証") + "の組と同じ回"

    # (1) LN-FAM6 の 6 つの形を 1 枚ずつ(実際の線 = 青、目標の線 e*(s) = 黒の点線)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.2), facecolor=SURFACE)
    for ax, r in zip(axes.flat, fam):
        draw_course(ax)
        idx = first_lap(r, 3.0)
        tx, tz = target_xy(r, sign)
        ax.plot(tx[idx], tz[idx], color=INK, lw=1.3, ls=(0, (3, 2)), zorder=4, label="目標の線 e*(s)")
        ax.plot(np.array(r["x"])[idx], np.array(r["z"])[idx], color=S1, lw=2, zorder=5, label="RL の走り")
        med, p95 = stats(r)
        lap = f"{r['laps'][0]:.2f} 秒" if r["laps"] else "-"
        ax.set_title(f"{FAM_JA.get(r['id'], r['id'])}: 1 周目 {lap}・誤差 中央値 {med * 100:.1f} cm・95% 点 {p95 * 100:.1f} cm"
                     f"{'' if r['valid_episode'] else '(無効)'}", color=INK, fontsize=9, loc="left")
    axes.flat[0].legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower left")
    fig.suptitle("LN-FAM6: 標準の開始・c_v 1.0 で 6 つの経路の形(開始から 1 周 + 3 m)。灰色の細い線 = 参照ループ R(c = 0)\n" + tag,
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / f"line_fam6_topdown{suffix}.png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # (2) 左寄り・中央・右寄り を重ねる(3 系列)
    fig, ax = plt.subplots(1, 1, figsize=(13, 8), facecolor=SURFACE)
    draw_course(ax)
    for rid, col in (("c+0.8", S1), ("c0", S2), ("c-0.8", S3)):
        r = next((q for q in fam if q["id"] == rid), None)
        if not r:
            continue
        idx = first_lap(r)
        ax.plot(np.array(r["x"])[idx], np.array(r["z"])[idx], color=col, lw=2, zorder=5, label=FAM_JA[rid] + (f"(1 周目 {r['laps'][0]:.2f} 秒)" if r["laps"] else ""))
    ax.legend(fontsize=9, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    ax.set_title("同じ方策・同じ開始で、条件(経路の形)だけを変えた 3 本の 1 周。" + tag, color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"line_lr_overlay{suffix}.png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # (3) e_y(s) の比べ(開始から 3 m 以降の最初の通過。左右中央 と イン・アウト・レイト の 2 枚)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.4), facecolor=SURFACE, sharex=True)
    for ax, group in zip(axes, (("c+0.8", "c0", "c-0.8"), ("inside", "outside", "late_apex"))):
        _style(ax, "", "横ずれ e_y [m](正 = 左)", xlabel="")
        for rid, col in zip(group, (S1, S2, S3)):
            r = next((q for q in fam if q["id"] == rid), None)
            if not r:
                continue
            s = np.array(r["s"], float)
            k = np.where((s - s[0] >= 3) & (s - s[0] <= 3 + L))[0]
            sm = np.mod(s[k], L)
            o = np.argsort(sm)
            es = np.array([v if v is not None else np.nan for v in r["es"]], float)
            ax.plot(sm[o], np.array(r["ey"], float)[k][o], color=col, lw=2, label=FAM_JA[rid] + " 実際")
            ax.plot(sm[o], es[k][o], color=col, lw=1, ls=(0, (3, 2)), label=FAM_JA[rid] + " 目標")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=3, loc="upper right")
    axes[1].set_xlabel("R の上の位置 s [m](1 周 26.12 m)", color=INK2, fontsize=9)
    axes[0].set_title("LN-FAM6 の横ずれ: 実線 = 実際、点線 = 目標 e*(s)。" + tag, color=INK, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"line_ey_profiles{suffix}.png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # (4) LN-P60 の最初の 15 回(family_plan_15 の 1 回分)
    if p60:
        n = len(p60)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows + 0.8), facecolor=SURFACE)
        for ax, r in zip(axes.flat, p60):
            draw_course(ax)
            idx = np.arange(len(r["s"]))
            tx, tz = target_xy(r, sign)
            ax.plot(tx[idx], tz[idx], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
            ax.plot(np.array(r["x"])[idx], np.array(r["z"])[idx], color=S1, lw=1.4, zorder=5)
            ax.plot(r["x"][0], r["z"][0], marker="o", ms=5, color=S2, zorder=6)
            med, _ = stats(r)
            pr = r["params"]["line"]
            extra = f" c0 {pr['c0']:+.2f}" if "c0" in pr else (f" a {pr['a']:.2f}" if "a" in pr else "")
            ax.set_title(f"{r['id']} {FAMILY_JA.get(r['family'], r['family'])}{extra} c_v {r['params'].get('c_v', 1):.2f}\n"
                         f"誤差 中央値 {med * 100:.1f} cm{'' if r['valid_episode'] else '・無効'}", color=INK, fontsize=8, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle("LN-P60 の最初の 15 回(line の記録の開始の分布の Halton・形は family_plan_15 の並び)。20 秒の全部。青 = RL、黒の点線 = 目標の線、橙の点 = 開始\n" + tag,
                     color=INK, fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(FIG / f"line_p60_topdown{suffix}.png", dpi=100, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
    for nm in ("line_fam6_topdown", "line_lr_overlay", "line_ey_profiles", "line_p60_topdown"):
        print("図:", FIG / f"{nm}{suffix}.png")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "curves":
        args = sys.argv[2:]
        sel = None
        if "--selected" in args:
            i = args.index("--selected")
            nm, st = args[i + 1].split("@")
            sel = {"name": nm, "step": int(st)}
            del args[i:i + 2]
        curves(args, selected=sel)
    elif cmd == "traj":
        traj(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        raise SystemExit("使い方: python rl/line_figs.py curves <run_dir>... | traj <traj.json> [接尾辞]")
