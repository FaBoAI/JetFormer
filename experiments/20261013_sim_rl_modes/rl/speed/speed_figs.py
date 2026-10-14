"""速さいろいろ(speed)の図(2026-10-13、sim_rl_modes の学習の段 speed)。

    python rl/speed/speed_figs.py curves --run <cond の main の出力> [--pilot <パイロットの出力>] [--selection selection.json]
    python rl/speed/speed_figs.py traj --figdata <speed_figdata.json>

出力: /home/jetson/jetformer_reports/sim_rl_modes_20261013/fig/speed_learning_curve.png・speed_trajectories.png
色は dataviz の参照パレット(c_v の 5 段は青の順序の段 250〜650、検証済み。比べる線はオレンジ)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
FIG = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/fig")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CV_LEVELS = [0.5, 0.625, 0.75, 0.875, 1.0]
CV_COLOR = {0.5: "#86b6ef", 0.625: "#5598e7", 0.75: "#2a78d6", 0.875: "#1c5cab", 1.0: "#104281"}   # 遅い = 明るい、速い = 濃い
ORANGE, AQUA, YELLOW = "#eb6834", "#1baf7a", "#eda100"


def _jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "VL Gothic", "Droid Sans Fallback"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            return name
    return None


def _style(ax, title, ylabel, xlabel="環境のステップ数 [百万]"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
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


def load_progress(run: Path):
    rows = [json.loads(l) for l in (run / "progress.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("episodes", {}).get("n", 0) > 0]


def load_evals(run: Path):
    out = []
    for p in sorted((run / "evals" / "speed").glob("step_*.json")):
        d = json.loads(p.read_text())
        out.append((int(p.stem.split("_")[1]), d))
    return out


def curves(run: Path, pilot: Path | None, selection: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _jp_font()
    P = load_progress(run)
    E = load_evals(run)
    sel = json.loads(selection.read_text())["selected"]["step"] if selection and selection.exists() else None
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.6), facecolor=SURFACE)
    (a_surv, a_lap, a_npass), (a_s2, a_s3, a_s5) = axes
    x = np.array([r["steps"] for r in P]) / 1e6
    surv = [r["episodes"].get("survived_20s_frac", np.nan) for r in P]
    lap1 = [r["episodes"].get("lap1_mean") if r["episodes"].get("lap1_mean") is not None else np.nan for r in P]
    trk = [r["episodes"].get("tracking_abs_mean") if r["episodes"].get("tracking_abs_mean") is not None else np.nan for r in P]
    a_surv.plot(x, smooth(surv), color=CV_COLOR[1.0], lw=1.8, label="本番(cond-main、2,000 万)")
    if pilot and (pilot / "progress.jsonl").exists():
        PP = load_progress(pilot)
        a_surv.plot(np.array([r["steps"] for r in PP]) / 1e6, smooth([r["episodes"].get("survived_20s_frac", np.nan) for r in PP]),
                    color="#8a8984", lw=1.2, ls=":", label="パイロット(100 万、結果に数えない)")
    a_surv.set_ylim(0, 1.05)
    _style(a_surv, "学習の回: 20 秒を走り切った割合", "割合")
    a_surv.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower right")
    a_lap.plot(x, smooth(lap1), color=CV_COLOR[1.0], lw=1.8)
    _style(a_lap, "学習の回: 1 周目の平均(c_v は 40% が 1.0、60% が 0.5〜1.0)", "1 周目 [秒]")
    a_lap.text(0.02, 0.04, f"線の追従の誤差 |e_y − e*| の平均: 最初 {np.nanmean(trk[:5]) * 100:.1f} cm → 最後 {np.nanmean(trk[-5:]) * 100:.1f} cm",
               transform=a_lap.transAxes, fontsize=8, color=INK2)
    if E:
        xe = np.array([s for s, _ in E]) / 1e6
        npass = [sum(1 for v in d["judged"].values() if v.get("pass") is True) for _, d in E]
        a_npass.step(xe, npass, where="mid", color=CV_COLOR[1.0], lw=1.8)
        a_npass.plot(xe, npass, ls="none", marker="o", ms=5, color=CV_COLOR[1.0])
        a_npass.axhline(5, color=INK2, lw=1, ls="--")
        a_npass.text(xe[0], 5.05, "全部(5 項目)", fontsize=8, color=INK2, va="bottom")
        for xi, (_, d), n in zip(xe, E, npass):
            fails = [k.split("_")[0] for k, v in d["judged"].items() if v.get("pass") is False]
            a_npass.text(xi, n - 0.12, "・".join(fails), fontsize=6.5, color=INK2, ha="center", va="top", rotation=90)
        a_npass.set_ylim(1.8, 5.4)
        _style(a_npass, "検証の組(高速版、18 回): 満たした判定の数(下の字 = 不合格の項目)", "項目の数 / 5")
        mc_all = [d["judged"]["S2_clearance"]["value"] for _, d in E]
        mc_ex = [min(r["min_clearance"] for r in d["tests"]["SP-P60"] if r["valid_episode"] and r["id"] != "P12") for _, d in E]
        a_s2.plot(xe, np.array(mc_all) * 100, color=CV_COLOR[1.0], lw=1.8, marker="o", ms=5, label="S2 = 有効な回の最小すき間")
        a_s2.plot(xe, np.array(mc_ex) * 100, color=CV_COLOR[0.625], lw=1.4, ls="--", marker="s", ms=4, label="P12(開始のすき間 4.5 cm)を除いた最小")
        a_s2.axhline(3, color=INK2, lw=1, ls="--")
        a_s2.text(xe[-1], 3.15, "合格の線 3 cm", fontsize=8, color=INK2, ha="right", va="bottom")
        _style(a_s2, "検証の組: 最小すき間(1〜400 万は P12 が衝突 = 無効、500 万〜は有効)", "すき間 [cm]")
        a_s2.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="center right")
        for cv, key in ((1.0, "t1"), (0.75, "t75"), (0.5, "t5")):
            ys = [d["judged"]["S3_lap_order"]["value"].get(key) for _, d in E]
            ys = [y if y is not None else np.nan for y in ys]
            a_s3.plot(xe, ys, color=CV_COLOR[cv], lw=1.8, marker="o", ms=5, label=f"c_v {cv:g}(上限 {4.5 * cv:.2f} m/s)")
            a_s3.text(xe[-1] + 0.3, ys[-1], f"{ys[-1]:.2f} 秒", fontsize=8, color=INK2, va="center")
        _style(a_s3, "検証の組: 停止から 1 周(STD の開始、単周)= S3", "1 周 [秒]")
        a_s3.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="center right")
        a_s3.set_xlim(right=xe[-1] * 1.1 + 0.3)
        vm4 = [d["judged"]["S5_vm4"]["value"] for _, d in E]
        a_s5.plot(xe, vm4, color=CV_COLOR[1.0], lw=1.8, marker="o", ms=5, label="S5 = VM4(s でそろえた速さのばらつき)")
        v1 = [next(x["mean_speed"] for x in d["judged"]["S4_speed_order"]["value"] if x["c_v"] == 1) for _, d in E]
        a_s5.axhline(0.4, color=INK2, lw=1, ls="--")
        a_s5.text(xe[-1], 0.405, "合格の線 0.4 m/s", fontsize=8, color=INK2, ha="right", va="bottom")
        a_s5.text(0.3, 0.72, f"c_v 1.0 の平均の速さ: {v1[0]:.2f} → {v1[-1]:.2f} m/s(線の追従を覚えるほど速い端が遅くなる)", transform=a_s5.transAxes, fontsize=8, color=INK2)
        _style(a_s5, "検証の組: 速さのばらつき VM4", "VM4 [m/s]")
        a_s5.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="upper right")
    for ax in axes.flat:
        if sel is not None:
            ax.axvline(sel / 1e6, color=ORANGE, lw=1.2, ls="--")
    if sel is not None:
        a_npass.text(sel / 1e6, 1.9, f" 選んだ {sel / 1e6:.2f} 百万", color=INK2, fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "speed_learning_curve.png", dpi=125, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / "speed_learning_curve.png")


def _draw_course(ax, world, labels=False):
    from matplotlib.patches import Rectangle, Circle
    for s in world["surfaces"]:
        if s["id"] == "carpet":
            continue
        col = {"shortcut": "#2b2b2b", "blue-mat": "#b7d3f6", "turf": "#cfe8cf", "ptfe": "#ece6f5"}.get(s["id"], "#e4e3df")
        ax.add_patch(Rectangle((s["x"] - s["w"] / 2, s["z"] - s["d"] / 2), s["w"], s["d"], facecolor=col, edgecolor="none", alpha=0.45, zorder=0))
        if labels:
            ax.text(s["x"], s["z"], {"shortcut": "黒シート", "blue-mat": "青マット", "turf": "芝", "ptfe": "PTFE"}.get(s["id"], s["id"]),
                    fontsize=7, color=INK2 if s["id"] != "shortcut" else "#fcfcfb", ha="center", va="center", zorder=1)
    for g in world["segments"]:
        ax.plot([g["a"][0], g["b"][0]], [g["a"][1], g["b"][1]], color="#6b6a66", lw=max(0.8, g["radius"] * 120), solid_capstyle="round", zorder=2)
    for d in world["disks"]:
        ax.add_patch(Circle((d["x"], d["z"]), d["radius"], facecolor="#9b9a95" if d["radius"] > 0.05 else "#52514e", edgecolor="none", alpha=0.7, zorder=2))


def traj(figdata: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _jp_font()
    D = json.loads(figdata.read_text())
    world = json.loads((EXP / "snapshot" / "sim" / "world-v3.json").read_text())
    fig = plt.figure(figsize=(16, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1])
    a_map = fig.add_subplot(gs[:, 0])
    a_v = fig.add_subplot(gs[0, 1])
    a_ey = fig.add_subplot(gs[1, 1])
    a_map.set_facecolor(SURFACE)
    _draw_course(a_map, world, labels=True)
    if D.get("fallback"):
        fb = np.array(D["fallback"]["trace"])
        a_map.plot(fb[:, 0], fb[:, 1], color=ORANGE, lw=1.3, ls="--", zorder=3,
                   label=f"比べる: 最速の方策 + 上限 2.5 m/s(すき間 {D['fallback']['min_clearance'] * 100:.1f} cm)")
    for cv in CV_LEVELS:
        r = D["single"][f"{cv:g}"]
        tr = np.array(r["trace"])
        a_map.plot(tr[:, 0], tr[:, 1], color=CV_COLOR[cv], lw=1.6, zorder=4,
                   label=f"c_v {cv:g}: 停止から 1 周 {r['lap_from_stop']:.2f} 秒・すき間 {r['min_clearance'] * 100:.1f} cm")
    b = world["bounds"]
    a_map.set_xlim(b["minX"], b["maxX"]); a_map.set_ylim(b["maxZ"], b["minZ"])
    a_map.set_aspect("equal")
    a_map.tick_params(colors=INK2, labelsize=7)
    for s in a_map.spines.values():
        s.set_color(GRID)
    a_map.set_title("上から見た軌跡: 標準の開始から 1 周(元の sim-core.js)。線の目標は教師の線 R(c0 = 0)", color=INK, fontsize=10.5, loc="left")
    a_map.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    a_map.set_xlabel("x [m]", color=INK2); a_map.set_ylabel("z [m](上が −z)", color=INK2)
    # s ごとの速さと横ずれ(連続 20 秒の 1 周目、開始からの進み)
    for cv in CV_LEVELS:
        r = D["continuous"][f"{cv:g}"]
        lg = np.array(r["log"], float)
        s = lg[:, 0] - lg[0, 0]
        m = s <= 26.12
        a_v.plot(s[m], lg[m, 2], color=CV_COLOR[cv], lw=1.5, label=f"c_v {cv:g}(上限 {4.5 * cv:.2f} m/s)")
        a_ey.plot(s[m], lg[m, 1] * 100, color=CV_COLOR[cv], lw=1.3)
    _style(a_v, "s ごとの実際の速さ(連続 20 秒の 1 周目)", "速さ [m/s]", "参照ループ R に沿った進み s [m]")
    a_v.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower right", ncol=1)
    _style(a_ey, "s ごとの横ずれ e_y(教師の線 R からの左右。目標は 0)", "e_y [cm](左が正)", "参照ループ R に沿った進み s [m]")
    a_ey.axhline(0, color=INK2, lw=0.8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "speed_trajectories.png", dpi=125, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / "speed_trajectories.png")


def lowclear(figdata: Path) -> None:
    """試験の組ですき間 3 cm 未満だった回(S2 の不合格の中身)の最初の 3 秒。どれも停止からの走り出しの 0.1〜1.7 m で起きる。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    _jp_font()
    D = json.loads(figdata.read_text())
    L = D.get("low_clearance") or []
    if not L:
        print("すき間 3 cm 未満の回はありません")
        return
    world = json.loads((EXP / "snapshot" / "sim" / "world-v3.json").read_text())
    fig, axes = plt.subplots(1, len(L), figsize=(3.6 * len(L), 4.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, L):
        ax.set_facecolor(SURFACE)
        _draw_course(ax, world)
        tr = np.array(e["trace"])
        ax.plot(tr[:, 0], tr[:, 1], color=CV_COLOR[min(CV_COLOR, key=lambda c: abs(c - e["c_v"]))], lw=1.8, zorder=4)
        ax.plot(tr[0, 0], tr[0, 1], marker="o", ms=6, color=INK2, zorder=5)
        x, z, yaw = e["at_min"]
        hl, hw = 0.21, 0.095   # 車体 0.42 × 0.19 m の長方形(sim-core の約束: 前 = (cos yaw, −sin yaw)、左 = (−sin yaw, −cos yaw))
        fx, fz, lx, lz = np.cos(yaw), -np.sin(yaw), -np.sin(yaw), -np.cos(yaw)
        pts = [(x + sx * hl * fx + sy * hw * lx, z + sx * hl * fz + sy * hw * lz) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor=ORANGE, lw=1.6, zorder=6))
        ax.set_xlim(x - 0.9, x + 0.9); ax.set_ylim(z + 0.9, z - 0.9)
        ax.set_aspect("equal")
        ax.tick_params(colors=INK2, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        st = e["start"]
        ax.set_title(f"{e['id']}: c_v {e['c_v']:g}・ゲート {st['gate']} から\n開始のすき間 {e['start_dmin'] * 100:.1f} cm → 最小 {e['min_clearance'] * 100:.2f} cm"
                     f"\n({e['frame_min']} フレーム目・{e['s_from_start_m']:.2f} m 進んだ所)", color=INK, fontsize=9, loc="left")
    fig.suptitle("試験の組(SP-P60)ですき間 3 cm 未満だった 5 回: 停止からの走り出し(最初の 3 秒の軌跡、● = 開始、四角 = 一番近づいた所の車体)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "speed_test_low_clearance.png", dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("図:", FIG / "speed_test_low_clearance.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curves", "traj"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--pilot", type=Path, default=None)
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--figdata", type=Path)
    a = ap.parse_args()
    if a.cmd == "curves":
        curves(a.run, a.pilot, a.selection)
    else:
        traj(a.figdata)
        lowclear(a.figdata)


if __name__ == "__main__":
    main()
