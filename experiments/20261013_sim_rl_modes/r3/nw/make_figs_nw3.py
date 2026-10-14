"""no_wall r3 の図(2026-10-14)。

    python3 make_figs_nw3.py <no_wall_r3 の出力> <図の置き場所>

  no_wall_r3_learning_curve.png  学習中の終わり方の割合・回の最小すき間の平均 / チェックポイントごとの検証の組(最小すき間・衝突・止まり)。点線 = 選んだ点
  no_wall_r3_shield_v4_vs_v41.png  下調べの組 dev(600 万の点)で、決まりのとおりのシールド v4 と v4.1 の 衝突・止まり・最小すき間
  no_wall_r3_test_sets.png  試験の組ごとの最小すき間(しきい値 8 cm / 5 cm)と衝突・止まりの数、シールドなしの衝突
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

O, F = Path(sys.argv[1]), Path(sys.argv[2])
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": .6, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
C1, C2, C3, RED, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984", "#0b0b0b"

sel = json.loads((O / "selection.json").read_text()) if (O / "selection.json").exists() else None
sel_steps = sel["selected"]["steps"] if sel else None

# ---- 学習曲線 ----
prog = [json.loads(l) for l in open(O / "train/progress.jsonl") if l.strip()] if (O / "train/progress.jsonl").exists() else []
def val_points(sub):
    out = []
    for ev in sorted(O.glob(f"{sub}/*/eval_validation.json")):
        e = json.loads(ev.read_text())
        J = e["judged"]
        st = 0 if ev.parent.name.startswith("r2_") else int(ev.parent.name.split("_")[1])
        out.append({"steps": st, "n2": J["N2_r3"]["value"], "n3": J["N3_r3"]["value"], "col": J["N1_r3"]["collisions"], "stall": J["stall_zero"]["value"]})
    return sorted(out, key=lambda p: p["steps"])


vals = sorted(val_points("val") + val_points("val_extra"), key=lambda p: p["steps"])   # シールド v4.1(選びに使った点 + 参考の r2 の方策)
vals4 = val_points("val_v4frozen")   # 決まりのとおりの v4(参考。0 = r2 の方策)
base = None
if prog:
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))
    # 学習中の終わり方(30 回の更新ごとにまとめる)
    win = 30
    xs, sh = [], {"clearance": [], "slow": [], "stall": [], "collision": [], "完走": []}
    xm, mc = [], []
    for i in range(0, len(prog), win):
        ch = prog[i:i + win]
        n = sum(p["episodes"].get("n", 0) for p in ch)
        if not n:
            continue
        rs = {}
        for p in ch:
            for k, v in (p["episodes"].get("reasons") or {}).items():
                rs[k] = rs.get(k, 0) + v
        xs.append(ch[-1]["steps"] / 1e6)
        sh["clearance"].append(rs.get("clearance", 0) / n)
        sh["slow"].append(rs.get("slow", 0) / n)
        sh["stall"].append(rs.get("stall", 0) / n)
        sh["collision"].append(rs.get("collision", 0) / n)
        sh["完走"].append(rs.get("collection_duration_complete", 0) / n)
        mcs = [p["episodes"].get("min_clearance_mean") for p in ch if p["episodes"].get("min_clearance_mean") is not None]
        xm.append(ch[-1]["steps"] / 1e6)
        mc.append(100 * sum(mcs) / len(mcs) if mcs else None)
    a = ax[0][0]
    for k, c, lab in (("完走", C1, "20 秒を走り切る"), ("clearance", C2, "すき間の約束(8 cm)で終了"), ("slow", C3, "止まりの見張り(進み < 0.6 m / 2 秒)"),
                      ("collision", RED, "衝突")):
        a.plot(xs, [100 * v for v in sh[k]], color=c, lw=2, label=lab)
    a.set_ylabel("学習中の回の割合 [%]")
    a.set_xlabel("学習のステップ [百万]")
    a.set_title("学習中の終わり方(30 回の更新ごと)", fontsize=10, color=INK, loc="left")
    a.legend(frameon=False, fontsize=8)
    a = ax[0][1]
    a.plot(xm, mc, color=C1, lw=2)
    a.set_ylabel("回の最小すき間の平均 [cm]")
    a.set_xlabel("学習のステップ [百万]")
    a.set_title("学習中の回の最小すき間(物理のずれ 85% を含む)", fontsize=10, color=INK, loc="left")
    if vals or vals4:
        a = ax[1][0]
        for pts, mk, fill, lab in ((vals4, "o", "none", "v4(決まりのとおり、参考)"), (vals, "D", None, "v4.1(選びに使った)")):
            if not pts:
                continue
            x = [p["steps"] / 1e6 for p in pts]
            kw = {"markerfacecolor": "none"} if fill == "none" else {}
            a.plot(x, [100 * max(-1, p["n2"]) for p in pts], color=C1, lw=0 if fill == "none" else 0, marker=mk, ms=7, ls="none", label=f"標準の物理の最小(N2)・{lab}", **kw)
            a.plot(x, [100 * max(-1, p["n3"]) for p in pts], color=C2, lw=0, marker=mk, ms=7, ls="none", label=f"物理のずれの最小(N3)・{lab}", **kw)
        a.axhline(8, color=C1, lw=1, ls=":")
        a.axhline(5, color=C2, lw=1, ls=":")
        a.axhline(0, color=GRAY, lw=.8)
        a.text(2.5, 8.15, "8 cm(N2)", color="#52514e", fontsize=8)
        a.text(2.5, 5.15, "5 cm(N3)", color="#52514e", fontsize=8)
        a.set_xlim(-.3, 6.4)
        a.set_ylabel("検証の組の最小すき間 [cm](衝突は負)")
        a.set_xlabel("学習のステップ [百万](0 = r2 の方策)")
        a.set_title("検証の組 NW-V-R3(494 回)", fontsize=10, color=INK, loc="left")
        a.legend(frameon=False, fontsize=7, loc="lower right")
        a = ax[1][1]
        w = .1
        for pts, off, alpha, lab in ((vals4, -.11, .45, "v4"), (vals, .11, 1.0, "v4.1")):
            if not pts:
                continue
            x = [p["steps"] / 1e6 for p in pts]
            a.bar([v + off - w / 2 for v in x], [p["col"] for p in pts], width=w, color=RED, alpha=alpha, label=f"衝突した回・{lab}")
            a.bar([v + off + w / 2 for v in x], [p["stall"] for p in pts], width=w, color=C3, alpha=alpha, label=f"止まった回・{lab}")
            for v, p in zip(x, pts):
                a.text(v + off, max(p["col"], p["stall"]) + .1, f"{p['col']}/{p['stall']}", ha="center", fontsize=7, color=INK)
        a.set_xlim(-.3, 6.4)
        a.set_ylabel("回の数(494 回のうち)")
        a.set_xlabel("学習のステップ [百万](0 = r2 の方策)")
        a.set_title("検証の組の衝突 / 止まり(薄い = v4、濃い = v4.1)", fontsize=10, color=INK, loc="left")
        a.legend(frameon=False, fontsize=7)
        if sel_steps:
            for aa in (ax[1][0], ax[1][1], ax[0][0], ax[0][1]):
                aa.axvline(sel_steps / 1e6, color=GRAY, lw=1, ls="--")
    fig.tight_layout()
    fig.savefig(F / "no_wall_r3_learning_curve.png", dpi=130)
    plt.close(fig)

# ---- 下調べの組 dev: シールド v4 と v4.1(600 万の点)----
cp = O / "dev/compare_6M.json"
if cp.exists():
    cmpd = json.loads(cp.read_text())
    eb, es = cmpd["v4"], cmpd["v4.1"]
    names = [("D-N", "標準 100"), ("D-R", "DR-T 一様 60"), ("D-K7", "隅 K7 × 33"), ("D-KX", "止まりにくい隅 264")]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for i, (key, lab, ttl) in enumerate((("collisions", "回", "衝突した回"), ("stall", "回", "止まった回(1 秒以上)"), ("min_clearance_m", "cm", "最小すき間"))):
        a = ax[i]
        y = list(range(len(names)))
        vb = [eb[k][key] * (100 if key == "min_clearance_m" else 1) for k, _ in names]
        vs = [es[k][key] * (100 if key == "min_clearance_m" else 1) for k, _ in names]
        h = .36
        a.barh([v + h / 2 for v in y], vb, height=h, color=GRAY, label="シールド v4(決まりのとおり)")
        a.barh([v - h / 2 for v in y], vs, height=h, color=C1, label="シールド v4.1")
        for v, b, s_ in zip(y, vb, vs):
            a.text(b, v + h / 2, f" {b:.1f}" if key == "min_clearance_m" else f" {b:.0f}", va="center", fontsize=8, color="#52514e")
            a.text(s_, v - h / 2, f" {s_:.1f}" if key == "min_clearance_m" else f" {s_:.0f}", va="center", fontsize=8, color=INK)
        a.set_yticks(y)
        a.set_yticklabels([n for _, n in names] if i == 0 else [""] * len(names))
        a.invert_yaxis()
        a.set_xlabel(lab)
        a.set_title(ttl, fontsize=10, color=INK, loc="left")
        if key == "min_clearance_m":
            a.axvline(8, color=C1, lw=1, ls=":")
            a.axvline(5, color=C2, lw=1, ls=":")
        else:
            a.set_xlim(0, max(2, max(vb + vs)) * 1.3)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("下調べの組 dev(学び直した 600 万の点、試験・検証と別の開始と種): シールド v4 と v4.1", fontsize=10, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(F / "no_wall_r3_shield_v4_vs_v41.png", dpi=130)
    plt.close(fig)

# ---- 試験の組 ----
tp = O / "test/eval_test.json"
if tp.exists():
    e = json.loads(tp.read_text())
    S = e["reported"]["sets"]
    order = [("NW-N300-R3", "N300 標準"), ("NW-GRID-R3", "GRID 標準"), ("NW-STD", "STD"), ("NW-G", "G(範囲の中)"), ("NW-R200-R3", "R200 DR-T"),
             ("NW-VX-R3", "VX 頂点"), ("NW-KX-R3", "KX 隅"), ("NW-FC30-R3", "FC30 摩擦円"), ("NW-LONG-R3", "LONG 120 秒")]
    order = [(k, n) for k, n in order if k in S]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    a = ax[0]
    y = list(range(len(order)))
    nominal = {"NW-N300-R3", "NW-GRID-R3", "NW-STD"}
    v = [100 * S[k]["min_clearance_m"] for k, _ in order]
    cols = [C1 if k in nominal else C2 for k, _ in order]
    a.barh(y, [max(0, x) for x in v], color=cols, height=.6)
    for yy, (k, _), x in zip(y, order, v):
        a.text(max(0, x), yy, f" {x:.2f} cm(n {S[k]['n']}・衝突 {S[k]['collisions']}・止まり {S[k]['stall']})", va="center", fontsize=8, color=INK)
    a.axvline(8, color=C1, lw=1, ls=":")
    a.axvline(5, color=C2, lw=1, ls=":")
    a.set_yticks(y)
    a.set_yticklabels([n for _, n in order])
    a.invert_yaxis()
    a.set_xlabel("組の最小すき間 [cm](青 = 標準の物理・しきい値 8 cm、橙 = 物理のずれ・5 cm)")
    a.set_xlim(0, max(v) * 1.9)
    failed = [k for k in e["judged_keys"] if not e["judged"][k]["pass"]]
    a.set_title(f"試験の組(シールド {e['shield']['version'].split('/')[-1]}・1 回だけ): " + ("全部合格" if not failed else "不合格 " + "・".join(failed)),
                fontsize=10, color=INK, loc="left")
    a = ax[1]
    off = e["reported"].get("NW_OFF_R3", {})
    ks = [("NW-N300-R3", "N300"), ("NW-R200-R3", "R200"), ("NW-VX-R3", "VX")]
    y = list(range(len(ks)))
    h = .36
    on_c = [S[k]["collisions"] for k, _ in ks]
    off_c = [off.get(k, {}).get("collisions", 0) for k, _ in ks]
    a.barh([v_ - h / 2 for v_ in y], on_c, height=h, color=C1, label="シールドあり")
    a.barh([v_ + h / 2 for v_ in y], off_c, height=h, color=GRAY, label="シールドなし(方策だけ、報告)")
    for yy, a_, b_ in zip(y, on_c, off_c):
        a.text(a_, yy - h / 2, f" {a_}", va="center", fontsize=8, color=INK)
        a.text(b_, yy + h / 2, f" {b_}", va="center", fontsize=8, color="#52514e")
    a.set_yticks(y)
    a.set_yticklabels([n for _, n in ks])
    a.invert_yaxis()
    a.set_xlabel("衝突した回")
    a.set_xlim(0, max(2, max(on_c + off_c)) * 1.4)
    a.set_title("シールドあり / なし", fontsize=10, color=INK, loc="left")
    a.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(F / "no_wall_r3_test_sets.png", dpi=130)
    plt.close(fig)
print("図を書きました:", sorted(p.name for p in F.glob("no_wall_r3_*.png")))
