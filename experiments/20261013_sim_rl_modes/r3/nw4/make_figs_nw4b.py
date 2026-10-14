"""no_wall r4(r4b・r4c を含む)の図(2026-10-14 21 時、電源断のあとの再開で make_figs_nw4.py を写して直した)。
make_figs_nw4.py との違い: 学習曲線を r4(train_a)・r4b(train)・r4c(train_c)の 3 本に分け、検証の点に val/a_step_*・val/c_step_* を足した。
確認 r3 の組の走らせ直しは regression/runs_all/rows_*.jsonl(reg_jobs_nw4b.py + run_ck3.mjs)。下調べのゲート 17 の角(dev4f)の図を足した。

    python3 make_figs_nw4b.py <no_wall_r4 の出力> <図の置き場所>

  no_wall_r4_test_sets.png       試験の組ごとの最小すき間(しきい値 8 cm / 5 cm)と衝突・止まり・後ろへ動いた回の数
  no_wall_r4_fixes.png           確認 r3 の崩れ(r3 の候補 + v4.1)と r4 の候補(v5)の比べ(確認 r3 の走らせ方 run_ck3.mjs)
  no_wall_r4_learning_curve.png  学習中の回の終わり方(r4・r4b・r4c)/ 点ごとの検証の組(最小すき間・衝突・止まり)
  no_wall_r4_corner_probe.png    下調べ: ゲート 17 の右へずれて左を向いた角の格子(dev4f)で止まった点(学習の点ごと)
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
    out = []
    if not Path(p).exists():
        return out
    for l in open(p):
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


# ---- 試験の組 ----
T = rows(O / "test/rows_test.jsonl")
if T:
    order = ["NW-N300-R4", "NW-FS1000-R4", "NW-LOWC-R4", "NW-EDGE-R4", "NW-STD", "NW-G", "NW-R200-R4", "NW-VX-R4", "NW-HC-R4", "NW-FC30-R4", "NW-LONG-R4"]
    std = {"NW-N300-R4", "NW-FS1000-R4", "NW-LOWC-R4", "NW-EDGE-R4", "NW-STD"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    labs, mins, cols, cnt = [], [], [], []
    for k in order:
        rs = [r for r in T if r["set"] == k]
        if not rs:
            continue
        ok = [r for r in rs if not r.get("collisions")]
        labs.append(f"{k.replace('NW-', '').replace('-R4', '')}\n{len(rs)}")
        mins.append(100 * min((r["min_clearance"] for r in ok), default=0))
        cols.append(C1 if (k in std or (k == "NW-LONG-R4")) else C2)
        cnt.append((sum(1 for r in rs if r.get("collisions")), sum(1 for r in rs if r.get("stall")), sum(1 for r in rs if (r.get("back_frames") or 0) > 0)))
    xx = range(len(labs))
    ax[0].bar(xx, mins, color=cols)
    for x, m in zip(xx, mins):
        ax[0].text(x, m + .2, f"{m:.2f}", ha="center", fontsize=7)
    ax[0].axhline(8, color=RED, lw=1, ls="--", label="N2 8 cm(標準の物理)")
    ax[0].axhline(5, color=GRAY, lw=1, ls=":", label="N3 5 cm(物理のずれ)")
    ax[0].set_xticks(list(xx)); ax[0].set_xticklabels(labs, fontsize=7); ax[0].set_ylabel("組の最小すき間 [cm]")
    ax[0].set_title("試験の組ごとの最小すき間(青 = 標準の物理、橙 = 物理のずれ。LONG は両方)", fontsize=9); ax[0].legend(fontsize=7)
    w = .27
    ax[1].bar([x - w for x in xx], [c[0] for c in cnt], width=w, color=RED, label="衝突")
    ax[1].bar(list(xx), [c[1] for c in cnt], width=w, color=C2, label="止まり")
    ax[1].bar([x + w for x in xx], [c[2] for c in cnt], width=w, color=C3, label="後ろへ動いた(抜け出し)")
    ax[1].set_xticks(list(xx)); ax[1].set_xticklabels([l.split("\n")[0] for l in labs], fontsize=6, rotation=60); ax[1].set_ylabel("回")
    ax[1].set_title("組ごとの衝突・止まり・後ろへ動いた回", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(F / "no_wall_r4_test_sets.png", dpi=130); plt.close(fig)

# ---- 確認 r3 の崩れの比べ ----
REG = O / "regression/runs_all"
if REG.exists():
    old = {}
    for p in list(CK.glob("runs/rows_*.jsonl")) + list(CK.glob("runs_x/rows_*.jsonl")) + list(CK.glob("runs_nbr/rows_*.jsonl")):
        for r in rows(p):
            old[r["id"]] = r
    new = {}
    for p in REG.glob("rows_*.jsonl"):
        for r in rows(p):
            new[r["id"]] = r
    sets = ["LOWC", "NBR", "EDGE2", "HCALL", "REG"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, (ttl, f) in enumerate([("衝突の回", lambda rs: sum(1 for r in rs if r.get("collisions"))),
                                   ("止まりの回", lambda rs: sum(1 for r in rs if r.get("stall"))),
                                   ("組の最小すき間 [cm](衝突の回を除く)", lambda rs: 100 * min((r["min_clearance"] for r in rs if not r.get("collisions")), default=0))]):
        a_, b_, labs = [], [], []
        for s in sets:
            ids = [i for i, r in new.items() if r.get("set", "") == s or (s == "REG" and r.get("set", "").startswith("REG"))]
            if not ids:
                continue
            ro = [old[i] for i in ids if i in old]
            rn = [new[i] for i in ids]
            labs.append(f"{s}\n{len(rn)}")
            a_.append(f(ro) if ro else 0); b_.append(f(rn))
        xx = range(len(labs))
        ax[j].bar([x - .2 for x in xx], a_, width=.4, color=GRAY, label="確認 r3(r3 の候補 + v4.1)")
        ax[j].bar([x + .2 for x in xx], b_, width=.4, color=C1, label="r4 の候補(v5)")
        for x, a, b in zip(xx, a_, b_):
            ax[j].text(x - .2, a, f"{a:.2f}" if j == 2 else f"{a}", ha="center", va="bottom", fontsize=6)
            ax[j].text(x + .2, b, f"{b:.2f}" if j == 2 else f"{b}", ha="center", va="bottom", fontsize=6)
        ax[j].set_xticks(list(xx)); ax[j].set_xticklabels(labs, fontsize=7); ax[j].set_title(ttl, fontsize=9)
        if j == 2:
            ax[j].axhline(8, color=RED, lw=1, ls="--"); ax[j].axhline(5, color=GRAY, lw=1, ls=":")
        if j == 0:
            ax[j].legend(fontsize=7)
    fig.suptitle("確認 r3 の組を同じ走らせ方(r3/ck_nw3/run_ck3.mjs)で: r3 の候補 → r4 の候補(LOWC・NBR・EDGE2 は標準の物理、HCALL・REG は物理のずれ。シールドあり)", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "no_wall_r4_fixes.png", dpi=130); plt.close(fig)

# ---- 学習曲線と検証 ----
runs = [("r4(種 24)", O / "train_a/progress.jsonl", C2), ("r4b(種 25)", O / "train/progress.jsonl", C1), ("r4c(種 26、r4b の終わりから)", O / "train_c/progress.jsonl", C3)]
sel = json.loads((O / "selection.json").read_text()) if (O / "selection.json").exists() else None
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
off_c = 3014656
for name, p, c in runs:
    prog = rows(p)
    if not prog:
        continue
    off = off_c if "r4c" in name else 0
    win = 20
    xs, ys = [], {}
    for i in range(0, len(prog), win):
        ch = prog[i:i + win]
        n = sum(q["episodes"].get("n", 0) for q in ch)
        if not n:
            continue
        rs = {}
        for q in ch:
            for k, v in (q["episodes"].get("reasons") or {}).items():
                rs[k] = rs.get(k, 0) + v
        xs.append((off + ch[-1]["steps"]) / 1e6)
        bad = sum(v for k, v in rs.items() if k != "collection_duration_complete")
        ys.setdefault("bad", []).append(bad / n)
    ax[0].plot(xs, ys.get("bad", []), color=c, lw=1.2, label=name)
ax[0].set_xlabel("学習のステップ [100 万](r3 の選んだ点から。r4c は r4b の終わり = 3.01 から続く)"); ax[0].set_ylabel("20 秒を走り切らなかった回の割合")
ax[0].set_title("学習中の回の終わり方(すき間 8 cm・止まりの見張り・逆走・ゲート外し・衝突)", fontsize=9); ax[0].legend(fontsize=7)
vals = []
for ev in sorted(O.glob("val/*/eval_validation.json")):
    e = json.loads(ev.read_text()); J = e["judged"]; n = ev.parent.name
    if n.startswith("c_step_"):
        run, x = "r4c", (off_c + int(n.split("_")[2])) / 1e6
    elif n.startswith("a_step_"):
        run, x = "r4", int(n.split("_")[2]) / 1e6
    else:
        s = int(n.split("_")[1]); run, x = ("r4b" if s else "r3 の方策"), s / 1e6
    vals.append({"name": n, "run": run, "x": x, "n2": J["N2_r4"]["value"], "n3": J["N3_r4"]["value"], "col": J["N1_r4"]["collisions"], "stall": J["stall_zero"]["value"],
                 "all": all(J[k]["pass"] for k in ("N1_r4", "N2_r4", "N3_r4", "N4", "N5_r4", "N6", "N7_r4", "stall_zero"))})
mk = {"r3 の方策": ("D", GRAY), "r4": ("^", C2), "r4b": ("o", C1), "r4c": ("s", C3)}
for v in vals:
    m, c = mk[v["run"]]
    ax[1].plot(v["x"], 100 * v["n2"], m, color=c, ms=6, mfc=c if v["all"] else "white")
    ax[1].plot(v["x"], 100 * v["n3"], m, color=c, ms=4, alpha=.5)
    if v["stall"] or v["col"]:
        ax[1].annotate(f"止まり {v['stall']}" + (f"・衝突 {v['col']}" if v["col"] else ""), (v["x"], 100 * v["n2"]), fontsize=6, color=RED, xytext=(0, 8),
                       textcoords="offset points", ha="left", rotation=60)
for run, (m, c) in mk.items():
    ax[1].plot([], [], m, color=c, label=run)
ax[1].axhline(8, color=RED, lw=1, ls="--")
if sel:
    sn = sel["selected"]["checkpoint"]
    sv = next((v for v in vals if v["name"] == sn), None)
    if sv:
        ax[1].axvline(sv["x"], color=GRAY, ls=":", lw=1)
        ax[1].annotate("選んだ点", (sv["x"], 100 * sv["n2"]), fontsize=7, xytext=(4, -12), textcoords="offset points")
ax[1].set_xlabel("点 [100 万ステップ](0 = r3 の選んだ方策)"); ax[1].set_ylabel("cm(大 = 標準の物理、小 = 物理のずれ)")
ax[1].set_title("検証の組 NW-V-R4(シールド v5)。塗り = 全部の判定と止まり 0 を満たす(該当なし)", fontsize=9); ax[1].legend(fontsize=7, loc="lower left")
ax[1].set_ylim(5, 9.2)
fig.tight_layout(); fig.savefig(F / "no_wall_r4_learning_curve.png", dpi=130); plt.close(fig)

# ---- 下調べ: ゲート 17 の角 ----
PB = O / "dev/probe_box"
fines = sorted(PB.glob("fine_*.jsonl"))
if fines:
    fig, axs = plt.subplots(1, len(fines), figsize=(2.6 * len(fines), 2.8), squeeze=False)
    for a, p in zip(axs[0], fines):
        rs = rows(p)
        for r in rs:
            col = RED if r["stall"] or r.get("collision") else C1
            a.scatter(-r["off"], r["yaw"], color=col, s=40 if r["stall"] else 18, marker="x" if r["stall"] else "o")
        a.set_title(p.stem.replace("fine_", "").replace("r4b_", "r4b ").replace("r4c_", "r4c "), fontsize=8)
        a.set_xlabel("右へのずれ [m]", fontsize=7); a.set_ylabel("左向き [°]", fontsize=7)
        a.set_xlim(.15, .205); a.set_ylim(11.5, 15.5)
    fig.suptitle("下調べ dev4f: ゲート 17 の分布の角(× = 止まったまま・周 0、● = 2 周)。検証・試験の点を含まない格子", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "no_wall_r4_corner_probe.png", dpi=130); plt.close(fig)
print("図:", sorted(p.name for p in F.glob("no_wall_r4_*.png")))
