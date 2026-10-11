"""(4) 学習の結果: 検証誤差の推移(data/runs/*/result.json)と、3モデル比較(overnight_20261011/compare/compare.json)。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *

OUT = os.path.join(os.path.dirname(__file__), "..", "training_results.png")
RUNS = "/home/jetson/jetformer/data/runs"
CMP = "/home/jetson/jetformer_reports/overnight_20261011/compare/compare.json"

curves = [  # (フォルダ, ラベル, 色, 線幅)
    ("lowspeed_20261010-s1-20261010-141911", "lowspeed", tint(BLUE, 0.40), 2.4),
    ("lowspeed_20261010b-s1-20261010-144643", "lowspeed_b", tint(BLUE, 0.65), 2.4),
    ("first_trial_20261010-s3-20261010-153629", "first_trial", BLUE, 3.4),
    ("clean_driving_20261011-nocmd-s3-20261011-024626", "モデルB", VIOLET, 3.4),
    ("clean_driving_20261011-s3-20261011-021839", "モデルA", GREEN, 3.4),
]
cmp = json.load(open(CMP))["models"]
CM = {  # 比較の値(3フレーム先、clean 2,336 フレーム)
    "first_trial": cmp["first_trial_20261010-s3-20261010-153629"],
    "モデルA": cmp["clean_driving_20261011-s3-20261011-021839"],
    "モデルB": cmp["clean_driving_20261011-nocmd-s3-20261011-024626"],
}

fig = new_fig()
fig.text(0.010, 0.962, "学習の結果", fontsize=28, fontweight="bold", color=INK, va="center")
fig.text(0.010, 0.913, "誤差 = 人の操作との平均絶対誤差(µs、小さいほど人に近い)。モデルA/B は 10/11 未明に学習、実車ではまだ走っていない",
         fontsize=17, color=INK2, va="center")

# ---- ① 検証誤差の推移
ax = fig.add_axes([0.075, 0.335, 0.345, 0.455])
fig.text(0.010, 0.858, "① 学習ごとの検証誤差(人の操作値を入れる条件)", fontsize=19, fontweight="bold", color=INK, va="center")
end_lbl = {}
for d, lab, col, lw in curves:
    r = json.load(open(os.path.join(RUNS, d, "result.json")))
    ep = [h["epoch"] for h in r["history"]]
    va = [h["val_us"] for h in r["history"]]
    ax.plot(ep, va, color=col, lw=lw, zorder=3, solid_capstyle="round")
    ax.plot([r["best_epoch"]], [r["best_val_us"]], "o", ms=10, color=col, mec="white", mew=1.5, zorder=4)
    end_lbl[lab] = (va[-1], r["best_val_us"], col)
lab_y = {"lowspeed": 72.0, "lowspeed_b": 62.0, "first_trial": 53.6, "モデルB": 47.2, "モデルA": 35.7}
lab_txt = {"lowspeed": "lowspeed 70.5", "lowspeed_b": "lowspeed_b 60.4", "first_trial": "first_trial 51.0",
           "モデルB": "モデルB 49.0", "モデルA": "モデルA 35.7"}
for lab, (yend, best, col) in end_lbl.items():
    ax.text(31.0, lab_y[lab], lab_txt[lab], fontsize=16, fontweight="bold" if lab in ("first_trial", "モデルA", "モデルB") else "normal",
            color=INK, va="center")
    ax.plot([30.15, 30.85], [yend, lab_y[lab]], color=col, lw=1.4)
ax.set_xlim(0.5, 30.5)
ax.set_ylim(30, 96)
ax.set_xticks([1, 10, 20, 30])
ax.set_yticks([40, 60, 80])
ax.tick_params(labelsize=16)
ax.set_xlabel("エポック", fontsize=16, color=INK2, labelpad=2)
ax.set_ylabel("検証誤差 µs", fontsize=16, color=INK2)
ax.grid(axis="y", color=GRID, lw=1.0)
ax.set_axisbelow(True)
ax.spines["right"].set_visible(False)
ax.annotate("電池切れで停止\n→ 再開", xy=(26, 49.9), xytext=(15.5, 43.2), fontsize=16, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2), ha="left", va="center")
fig.text(0.010, 0.195,
         "学習シーン数: 2 → 4 → 18 → 41(A/B は衝突の前後を除外)\n"
         "● = 最良のエポック。検証シーンが違うので直接は比べられない",
         fontsize=16, color=INK2, va="center", linespacing=1.3)

# ---- ② 3モデル比較(教師あり → 自己入力)
fig.text(0.585, 0.858, "② AI走行と同じ条件での比較", fontsize=19, fontweight="bold", color=INK, va="center")
fig.text(0.585, 0.815, "3 フレーム先・検証 11 シーン・2,336 フレーム", fontsize=16, color=INK2, va="center")
order = ["first_trial", "モデルA", "モデルB"]
colors = {"first_trial": BLUE, "モデルA": GREEN, "モデルB": VIOLET}
for j, (key, title, ybox) in enumerate([("steer_mae_k3", "ステア 誤差 µs", 0.560), ("throttle_mae_k3", "スロットル 誤差 µs", 0.295)]):
    a2 = fig.add_axes([0.715, ybox, 0.265, 0.180])
    fig.text(0.585, ybox + 0.205, title, fontsize=18, fontweight="bold", color=INK, va="center")
    for i, m in enumerate(order):
        tf = CM[m]["tf"]["clean"][key]
        sf = CM[m]["sf"]["clean"][key]
        y = i
        col = colors[m]
        a2.plot([tf, sf], [y, y], color=col, lw=3, zorder=2, solid_capstyle="butt")
        a2.plot([tf], [y], "o", ms=13, mfc=SURF, mec=col, mew=2.6, zorder=3)
        a2.plot([sf], [y], "o", ms=13, color=col, mec="white", mew=1.2, zorder=4)
        if abs(sf - tf) < 1:
            a2.text(sf + 7, y, "%.1f(同じ)" % sf, fontsize=16, fontweight="bold", va="center", color=INK)
        else:
            a2.text(tf - 6, y, "%.1f" % tf, fontsize=16, va="center", ha="right", color=INK2)
            a2.text(sf + 6, y, "%.1f" % sf, fontsize=16, fontweight="bold", va="center", color=INK)
        fig.text(0.585, ybox + 0.180 * (2.6 - i) / 3.2, m, fontsize=16,
                 fontweight="bold", color=INK, va="center")
    a2.set_ylim(2.6, -0.6)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlim(0, 158)
    a2.set_xticks([0, 50, 100, 150])
    a2.tick_params(labelsize=16)
    a2.grid(axis="x", color=GRID, lw=1.0)
    a2.set_axisbelow(True)
fig.text(0.585, 0.186, "○ 人の操作値を入れる(学習と同じ)\n● 自分の予測を入れる(走行と同じ)\n"
         "B は操作値を見ないので両方同じ\nA のスロットルは人より約 99µs 前進側",
         fontsize=16, color=INK2, va="center", linespacing=1.25)

# ---- 下の帯: 発見の経緯
fig.patches.append(matplotlib.patches.FancyBboxPatch((0.010, 0.010), 0.978, 0.098, boxstyle="round,pad=0,rounding_size=0.01",
                                                     fc=tint(YELLOW, 0.16), ec=YELLOW, lw=1.8, transform=fig.transFigure, figure=fig))
fig.text(0.022, 0.059, "14:56 の発見: lowspeed_b のステア誤差(1 フレーム先)は 36.1µs → 自分の予測を入れると 64.7µs\n"
         "「今の操作値」をまねていた → 15:20 操作値なし版 → モデルB(走行の条件で ステア −36%・スロットル −57%)",
         fontsize=16, color=INK, va="center", linespacing=1.3)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
