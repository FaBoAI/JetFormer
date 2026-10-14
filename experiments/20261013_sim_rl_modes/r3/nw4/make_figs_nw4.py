"""no_wall r4 の図(2026-10-14)。

    python3 make_figs_nw4.py <no_wall_r4 の出力> <図の置き場所>

  no_wall_r4_test_sets.png    試験の組ごとの最小すき間(しきい値 8 cm / 5 cm)と衝突・止まり・後ろへ動いた回の数
  no_wall_r4_fixes.png        確認 r3 の崩れ(v4.1)と r4(v5)の比べ: 12〜15 cm の層の止まり・NBR/HCALL/EDGE2 の衝突と最小すき間(確認 r3 の走らせ方 run_ck3.mjs)
  no_wall_r4_learning_curve.png  学習中の終わり方の割合 / チェックポイントごとの検証の組(最小すき間・衝突・止まり)。点線 = 選んだ点
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

O, F = Path(sys.argv[1]), Path(sys.argv[2])
CK = Path("/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_no_wall_r3")
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": .6, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
C1, C2, C3, RED, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984", "#0b0b0b"


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


# ---- 試験の組 ----
T = rows(O / "test/rows_test.jsonl")
if T:
    order = ["NW-N300-R4", "NW-FS1000-R4", "NW-LOWC-R4", "NW-EDGE-R4", "NW-STD", "NW-G", "NW-R200-R4", "NW-VX-R4", "NW-HC-R4", "NW-FC30-R4", "NW-LONG-R4"]
    std = {"NW-N300-R4", "NW-FS1000-R4", "NW-LOWC-R4", "NW-EDGE-R4", "NW-STD"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    xs, mins, cols = [], [], []
    for i, k in enumerate(order):
        rs = [r for r in T if r["set"] == k]
        if not rs:
            continue
        mn = min(r["min_clearance"] for r in rs) * 100
        xs.append(k.replace("NW-", "").replace("-R4", "") + f"\n{len(rs)}")
        mins.append(mn)
        cols.append(C1 if (k in std or (k == "NW-LONG-R4")) else C2)
    ax[0].bar(range(len(xs)), mins, color=cols)
    ax[0].axhline(8, color=RED, lw=1, ls="--"); ax[0].axhline(5, color=GRAY, lw=1, ls=":")
    ax[0].set_xticks(range(len(xs))); ax[0].set_xticklabels(xs, fontsize=7)
    ax[0].set_ylabel("組の最小すき間 [cm]"); ax[0].set_title("試験の組ごとの最小すき間(青 = 標準の物理、橙 = 物理のずれ。赤線 8 cm・灰線 5 cm)", fontsize=9)
    for i, m in enumerate(mins):
        ax[0].text(i, m + .2, f"{m:.2f}", ha="center", fontsize=7, color=INK)
    lab, col_, st_, back_ = [], [], [], []
    for k in order:
        rs = [r for r in T if r["set"] == k]
        if not rs:
            continue
        lab.append(k.replace("NW-", "").replace("-R4", ""))
        col_.append(sum(1 for r in rs if r["collisions"])); st_.append(sum(1 for r in rs if r.get("stall"))); back_.append(sum(1 for r in rs if (r.get("back_frames") or 0) > 0))
    y = range(len(lab))
    ax[1].barh([v - .25 for v in y], col_, height=.25, color=RED, label="衝突")
    ax[1].barh(list(y), st_, height=.25, color=C2, label="止まり")
    ax[1].barh([v + .25 for v in y], back_, height=.25, color=C3, label="後ろへ動いた(抜け出し)")
    ax[1].set_yticks(list(y)); ax[1].set_yticklabels(lab, fontsize=7); ax[1].invert_yaxis(); ax[1].legend(fontsize=7); ax[1].set_xlabel("回の数")
    ax[1].set_title("衝突・止まり・抜け出しの回", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "no_wall_r4_test_sets.png", dpi=130); plt.close(fig)

# ---- 確認 r3 の崩れの比べ ----
REG = O / "regression"
if REG.exists():
    old = {}
    for p in list(CK.glob("runs/rows_*.jsonl")) + list(CK.glob("runs_x/rows_*.jsonl")) + list(CK.glob("runs_nbr/rows_*.jsonl")) + list(CK.glob("runs_reg/rows_*.jsonl")):
        for r in rows(p):
            old[r["id"]] = r
    new = {}
    for p in REG.glob("runs_*/rows_*.jsonl"):
        for r in rows(p):
            new[r["id"]] = r
    sets = ["LOWC", "NBR", "EDGE2", "HCALL", "REG"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, (ttl, f) in enumerate([("衝突の回", lambda rs: sum(1 for r in rs if r.get("collisions"))),
                                   ("止まりの回", lambda rs: sum(1 for r in rs if r.get("stall"))),
                                   ("組の最小すき間 [cm]", lambda rs: 100 * min((r["min_clearance"] for r in rs if not r.get("collisions")), default=0))]):
        a_, b_, labs = [], [], []
        for s in sets:
            ids = [i for i, r in new.items() if r.get("set", "").startswith(s)]
            if not ids:
                continue
            ro = [old[i] for i in ids if i in old]
            rn = [new[i] for i in ids]
            labs.append(f"{s}\n{len(rn)}")
            a_.append(f(ro) if ro else 0); b_.append(f(rn))
        xx = range(len(labs))
        ax[j].bar([x - .2 for x in xx], a_, width=.4, color=GRAY, label="確認 r3(v4.1・r3 の方策)")
        ax[j].bar([x + .2 for x in xx], b_, width=.4, color=C1, label="r4 の候補(v5)")
        ax[j].set_xticks(list(xx)); ax[j].set_xticklabels(labs, fontsize=7); ax[j].set_title(ttl, fontsize=9)
        if j == 2:
            ax[j].axhline(8, color=RED, lw=1, ls="--"); ax[j].axhline(5, color=GRAY, lw=1, ls=":")
        if j == 0:
            ax[j].legend(fontsize=7)
    fig.suptitle("確認 r3 の組を同じ走らせ方(r3/ck_nw3/run_ck3.mjs)で: r3 の候補 → r4 の候補(LOWC・NBR・EDGE2 は標準の物理、HCALL・REG は物理のずれ)", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "no_wall_r4_fixes.png", dpi=130); plt.close(fig)

# ---- 学習曲線と検証 ----
prog = rows(O / "train/progress.jsonl")
sel = json.loads((O / "selection.json").read_text()) if (O / "selection.json").exists() else None
vals = []
for ev in sorted(O.glob("val/step_*/eval_validation.json")):
    e = json.loads(ev.read_text()); J = e["judged"]
    vals.append({"steps": int(ev.parent.name.split("_")[1]), "n2": J["N2_r4"]["value"], "n3": J["N3_r4"]["value"], "col": J["N1_r4"]["collisions"],
                 "stall": J["stall_zero"]["value"], "all": all(J[k]["pass"] for k in ("N1_r4", "N2_r4", "N3_r4", "N4", "N5_r4", "N6", "N7_r4", "stall_zero"))})
if prog or vals:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    if prog:
        win = 20
        xs, ys = [], {}
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
            for k in ("clearance", "slow", "collision", "collection_duration_complete"):
                ys.setdefault(k, []).append(rs.get(k, 0) / n)
        names = {"clearance": "すき間 8 cm で終了", "slow": "止まりの見張り", "collision": "衝突", "collection_duration_complete": "20 秒を走り切る"}
        for k, c in zip(names, (C2, RED, INK, C1)):
            if k in ys:
                ax[0].plot(xs, ys[k], color=c, label=names[k], lw=1.2)
        ax[0].set_xlabel("学習のステップ [100 万](r3 の選んだ点から)"); ax[0].set_ylabel("回の割合"); ax[0].legend(fontsize=7); ax[0].set_title("学習中の回の終わり方", fontsize=9)
    if vals:
        xv = [v["steps"] / 1e6 for v in vals]
        ax[1].plot(xv, [100 * v["n2"] for v in vals], "o-", color=C1, label="標準の物理の最小すき間")
        ax[1].plot(xv, [100 * v["n3"] for v in vals], "s-", color=C2, label="物理のずれの最小すき間")
        for v in vals:
            if v["stall"] or v["col"]:
                ax[1].annotate(f"止まり {v['stall']}・衝突 {v['col']}", (v["steps"] / 1e6, 100 * v["n2"]), fontsize=7, color=RED, xytext=(0, 8), textcoords="offset points")
        ax[1].axhline(8, color=RED, lw=1, ls="--"); ax[1].axhline(5, color=GRAY, lw=1, ls=":")
        if sel:
            ax[1].axvline(sel["selected"]["steps"] / 1e6, color=GRAY, ls=":", lw=1)
        ax[1].set_xlabel("チェックポイント [100 万ステップ](0 = r3 の選んだ方策)"); ax[1].set_ylabel("cm"); ax[1].legend(fontsize=7)
        ax[1].set_title("検証の組 NW-V-R4(シールド v5)", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "no_wall_r4_learning_curve.png", dpi=130); plt.close(fig)
print("図を書きました", sorted(p.name for p in F.glob("no_wall_r4_*.png")))
