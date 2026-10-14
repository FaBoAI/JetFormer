"""経路いろいろ r2 の図(2026-10-14、学習の段 line r2)。判定には使わない(判定は rl/eval_modes.mjs --protocol line_r2 の結果)。

    python rl/line_r2/line_r2_figs.py curves <run_dir> --baseline <val.json> [--selected ステップ]   # fig/line_r2_learning_curve.png
    python rl/line_r2/line_r2_figs.py target                                                      # fig/line_r2_target_v1_v2.png(目標の線 v1 と v2)
    python rl/line_r2/line_r2_figs.py traj <traj.json>                                            # fig/line_*_r2.png(第 1 回と同じ 4 枚。rl/line_figs.py traj)
    python rl/line_r2/line_r2_figs.py start <test_r2.json> <test_r1policy.json>                   # fig/line_r2_start_clearance.png・line_r2_tracking_by_s.png
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import line_figs as LF  # noqa: E402  (第 1 回の図の部品: コースの絵・軌跡の 4 枚)

FIG = LF.FIG
SURFACE, INK, INK2, GRID, MUTED, S1, S2, S3 = LF.SURFACE, LF.INK, LF.INK2, LF.GRID, LF.MUTED, LF.S1, LF.S2, LF.S3
EXP = HERE.parent.parent


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LF._jp_font()
    return plt


def eval_row(d: dict) -> dict:
    J = d["judged"]
    v4 = J["L4_distinct"].get("value") or {}
    return {"L1": int(str(J["L1_valid"]["value"]).split("/")[0]), "L2": J["L2_clearance"]["value"], "L3_med": J["L3_tracking"]["value"]["median"],
            "L3_p95": J["L3_tracking"]["value"]["p95"], "L4_lr": v4.get("left_right_mean"), "L4_io": v4.get("inside_outside_mean"),
            "L4_pair": v4.get("min_pair_max_1m_avg"), "L5": J["L5_vm3"]["value"], "L6": J["L6_lap"]["value"], "all": d["all_judged_pass"],
            "n_pass": sum(1 for v in J.values() if v.get("pass") is True)}


def curves(run: Path, baseline: Path | None, selected: int | None):
    plt = _plt()
    prog = []
    for line in (run / "progress.jsonl").read_text().splitlines():
        r = json.loads(line)
        e = r.get("episodes", {})
        if e.get("n"):
            prog.append({"x": r["steps"] / 1e6, "track": e.get("tracking_abs_mean"), "lap1": e.get("lap1_mean"), "surv": e.get("survived_20s_frac"),
                         "clear": e.get("min_clearance_mean")})
    evals = []
    if baseline and baseline.exists():
        evals.append({"x": 0.0, "steps": 0, **eval_row(json.loads(baseline.read_text()))})
    for p in sorted((run / "evals" / "line").glob("step_*.json")):
        st = int(p.stem.split("_")[1])
        evals.append({"x": st / 1e6, "steps": st, **eval_row(json.loads(p.read_text()))})
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5), facecolor=SURFACE)

    def train_panel(ax, key, title, ylabel, ylim=None):
        LF._style(ax, title, ylabel)
        x = np.array([r["x"] for r in prog])
        y = np.array([r[key] if r[key] is not None else np.nan for r in prog], float)
        ax.plot(x, y, color=S1, lw=0.8, alpha=0.25)
        ax.plot(x, LF.smooth(y), color=S1, lw=2, label="r2 の学習(初期化 = 第 1 回の方策)")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    def eval_panel(ax, key, title, ylabel, thr, thr_text, upper=False):
        LF._style(ax, title, ylabel)
        ex = [r["x"] for r in evals]
        ey = [r[key] if r[key] is not None else np.nan for r in evals]
        ax.plot(ex, ey, color=S1, lw=2, marker="o", ms=5, label="検証の組 LN-V60-R2 + LN-FAM6")
        for r in evals:
            if r["all"]:
                ax.plot(r["x"], r[key], marker="o", ms=5, color=S3, zorder=5)
        ax.axhline(thr, color=INK2, lw=1, ls=":")
        ax.text(0, thr, " " + thr_text, color=INK2, fontsize=8, va="bottom")
        if selected is not None:
            for r in evals:
                if r["steps"] == selected and r[key] is not None:
                    ax.plot(r["x"], r[key], marker="o", ms=12, mfc="none", mec=INK, mew=1.5, zorder=6)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    train_panel(axes[0, 0], "track", "学習中: 線の追従の誤差(開始から 3 m 以降の平均)", "誤差 [m]", (0, None))
    train_panel(axes[0, 1], "lap1", "学習中: 1 周目の時間(停止から、有効な周)", "秒")
    train_panel(axes[0, 2], "surv", "学習中: 20 秒を走り切った回の割合", "割合", (0, 1.05))
    eval_panel(axes[1, 0], "L2", "検証 L2: 最小すき間(有効な回 + FAM6)", "m", 0.03, "しきい値 ≥ 0.03")
    eval_panel(axes[1, 1], "L3_med", "検証 L3: 追従の誤差の中央値", "m", 0.05, "しきい値 ≤ 0.05")
    eval_panel(axes[1, 2], "L3_p95", "検証 L3: 追従の誤差の 95% 点", "m", 0.12, "しきい値 ≤ 0.12")
    eval_panel(axes[2, 0], "L4_pair", "検証 L4: 6 つの形のどの 2 つも違う所(最小)", "m", 0.08, "しきい値 ≥ 0.08")
    eval_panel(axes[2, 1], "L5", "検証 L5: 横ずれのばらつき VM3", "m", 0.08, "しきい値 ≥ 0.08")
    eval_panel(axes[2, 2], "L6", "検証 L6: 中央の線の 1 周目(停止から)", "秒", 10.0, "しきい値 ≤ 10.0")
    t = ("強化学習モード(経路いろいろ)r2: 目標の線 v2・進みに追従の掛け算・すき間の壁。点線 = 合格のしきい値(検証の組 60 + 6 回・高速版のコア)。"
         "0 = 初期化元(第 1 回の方策)を v2 の線で。緑の点 = 6 項目すべて合格")
    if selected is not None:
        t += f"。黒い輪 = 選んだもの({selected:,})"
    fig.suptitle(t, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "line_r2_learning_curve.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", out)


def target():
    """目標の線 v1 と v2(左寄り c0 +0.8・イン a 0.8・右寄り c0 −0.8)を上から・e*(s) で。尖りの所(s 10〜11.5)を見せる"""
    plt = _plt()
    js = subprocess.run(["node", "-e", """
import('%s/rl/runtime/line_profiles.mjs').then(m => {
  const out = {};
  for (const [k, p] of Object.entries({'c+0.8': {family: 'const', c0: .8}, inside: {family: 'inside', a: .8}, 'c-0.8': {family: 'const', c0: -.8}, random: {family: 'random', seed: 4242}})) {
    const a = m.createLineProfile(p, 0), b = m.createLineProfile(p, 0, {version: 'v2'}); b.anchor(0);
    const s = [], e1 = [], e2 = [];
    for (let x = 0; x < 26.11; x += .05) { s.push(+x.toFixed(3)); e1.push(a.eStar(x)); e2.push(b.lineAt(x)); }
    out[k] = {s, e1, e2};
  }
  console.log(JSON.stringify(out));
});""" % EXP], capture_output=True, text=True, check=True).stdout
    D = json.loads(js)
    sign = 1.0   # e_y の正 = 左。R の左の法線 = (tz, −tx)(rl/build_room_table.mjs と同じ)
    fig = plt.figure(figsize=(17, 10), facecolor=SURFACE)
    ax0 = fig.add_axes([0.03, 0.40, 0.48, 0.52])
    ax1 = fig.add_axes([0.55, 0.40, 0.42, 0.50])
    ax2 = fig.add_axes([0.05, 0.06, 0.92, 0.27])
    for ax, ttl, zoom in ((ax0, "コース全体(左寄り c0 +0.8 とイン a 0.8)", False), (ax1, "R の左の急な曲がり s 9.5〜12.5 m を拡大", True)):
        LF.draw_course(ax)
        for k, col in (("c+0.8", S1), ("inside", S2)):
            s = np.array(D[k]["s"])
            px, pz, tx, tz = LF.ref_at(s)
            for e, ls, lab in ((np.array(D[k]["e1"]), (0, (2, 2)), "v1"), (np.array(D[k]["e2"]), "-", "v2")):
                x, z = px + sign * e * tz, pz - sign * e * tx
                ax.plot(x, z, color=col, lw=1.2 if lab == "v1" else 2.2, ls=ls, zorder=5, label=f"{'左寄り c0 +0.8' if k == 'c+0.8' else 'イン a 0.8'} {lab}")
        if zoom:
            s = np.array(D["c+0.8"]["s"])
            k = (s > 9.5) & (s < 12.5)
            px, pz, _, _ = LF.ref_at(s[k])
            ax.set_xlim(px.min() - 1.0, px.max() + 1.0)
            ax.set_ylim(pz.max() + 1.0, pz.min() - 1.0)
        ax.set_title(ttl, color=INK, fontsize=10, loc="left")
    ax0.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower left")
    LF._style(ax2, "目標の横ずれ e*(s)(点線 = v1、実線 = v2)。v1 は s 10〜11.5 m で R の曲率の中心より先(e·κ > 1)まで振れる", "e* [m](正 = 左)", "R の上の位置 s [m]")
    for k, col, nm in (("c+0.8", S1, "左寄り c0 +0.8"), ("inside", S2, "イン a 0.8"), ("c-0.8", S3, "右寄り c0 −0.8")):
        s = np.array(D[k]["s"])
        ax2.plot(s, D[k]["e1"], color=col, lw=1, ls=(0, (2, 2)))
        ax2.plot(s, D[k]["e2"], color=col, lw=2, label=nm)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=3, loc="upper right")
    fig.suptitle("目標の線 v1(第 1 回)と v2(第 2 回): v2 は R の曲がりの内側へ寄れる量を 1/|κ| − 0.7 m に、傾きを 0.25 に抑え、σ 0.3 m でなめらかにした(開始の 2 m のつなぎはこの図に無い)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    out = FIG / "line_r2_target_v1_v2.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print("図:", out)


def start_and_by_s(test_r2: Path, test_r1: Path):
    """(1) 回ごとの最小すき間と、それが起きた開始からの時間(r2 の方策と第 1 回の方策、同じ組)。(2) 追従の誤差の 95% 点を R の上の s ごとに"""
    plt = _plt()
    A = json.loads(test_r2.read_text())
    B = json.loads(test_r1.read_text())
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor=SURFACE)
    LF._style(ax, "", "回の最小すき間 [cm](衝突は 0)", "最小すき間になった時刻(開始から)[秒]")
    for d, col, nm in ((B, S2, "第 1 回の方策(線 v2 で)"), (A, S1, "r2 の方策")):
        rows = d["tests"]["LN-P60"] + d["tests"].get("LN-FAM6", [])
        x = [r.get("dmin_frame_min_t") for r in rows]
        y = [0 if (r.get("collisions") or 0) > 0 else (r.get("min_clearance") or 0) * 100 for r in rows]
        ax.scatter(x, y, s=22, color=col, label=f"{nm}: 3 cm 未満 {sum(1 for v in y if v < 3)} 回・衝突 {sum(1 for r in rows if (r.get('collisions') or 0) > 0)} 回", zorder=4)
    ax.axhline(3, color=INK2, lw=1, ls=":")
    ax.text(0, 3, " L2 のしきい値 3 cm", color=INK2, fontsize=8, va="bottom")
    ax.axvline(1.5, color=MUTED, lw=1, ls=(0, (2, 2)))
    ax.legend(fontsize=9, frameon=False, labelcolor=INK2)
    ax.set_title("試験の組 LN-P60-R2 + LN-FAM6(元の sim-core.js): 回ごとの最小すき間と、その時刻", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    out = FIG / "line_r2_start_clearance.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print("図:", out)


def by_s(traj_r2: Path, traj_r1: Path):
    """追従の誤差の 95% 点を R の上の s(0.25 m のマス)ごとに。両方とも試験の組の全部の回の軌跡(--all)"""
    plt = _plt()
    fig, ax = plt.subplots(1, 1, figsize=(14, 5.5), facecolor=SURFACE)
    LF._style(ax, "", "追従の誤差の 95% 点 [cm]", "R の上の位置 s [m](0.25 m のマス)")
    L = LF.L
    for path, col, nm in ((traj_r1, S2, "第 1 回の方策(線 v2 で)"), (traj_r2, S1, "r2 の方策")):
        d = json.loads(path.read_text())
        E, S = [], []
        for r in d["runs"]:
            if r["test"] != "LN-P60" or not r["valid_episode"]:
                continue
            s = np.array(r["s"], float)
            ey = np.array(r["ey"], float)
            es = np.array([v if v is not None else np.nan for v in r["es"]], float)
            k = (s - s[0]) >= 3
            E.append(np.abs(ey[k] - es[k]))
            S.append(np.mod(s[k], L))
        E, S = np.concatenate(E), np.concatenate(S)
        bins = np.arange(0, L + 0.25, 0.25)
        idx = np.digitize(S, bins) - 1
        p95 = [np.quantile(E[idx == i], 0.95) * 100 if np.sum(idx == i) >= 20 else np.nan for i in range(len(bins) - 1)]
        tot = np.quantile(E, 0.95) * 100
        ax.step(bins[:-1], p95, where="post", color=col, lw=2, label=f"{nm}(全体の 95% 点 {tot:.1f} cm)")
    ax.axhline(12, color=INK2, lw=1, ls=":")
    ax.text(0, 12, " L3 のしきい値 12 cm(全体の 95% 点)", color=INK2, fontsize=8, va="bottom")
    for a, b in ((2, 4.5), (8, 12.5)):
        ax.axvspan(a, b, color=GRID, alpha=0.5, zorder=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False, labelcolor=INK2)
    ax.set_title("試験の組 LN-P60-R2 の有効な回: 追従の誤差の 95% 点を R の上の位置ごとに(灰色 = S 字の区間 s 2〜4.5・8〜12.5 m)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    out = FIG / "line_r2_tracking_by_s.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print("図:", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "curves":
        a = sys.argv[2:]
        base = Path(a[a.index("--baseline") + 1]) if "--baseline" in a else None
        sel = int(a[a.index("--selected") + 1]) if "--selected" in a else None
        curves(Path(a[0]), base, sel)
    elif cmd == "target":
        target()
    elif cmd == "traj":
        LF.traj(Path(sys.argv[2]), "_r2")
    elif cmd == "start":
        start_and_by_s(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "by_s":
        by_s(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
