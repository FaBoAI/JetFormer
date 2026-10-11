"""(iii) 見えた差の大きさと、学習のばらつき(チェックポイント)の比較。
数値: evaluation.md 0章の表・4.1節(evaluation.json の review.contrasts / review.ckpt_noise)。
差は保留した通過(R1 17回・DE 14回)の平均、区間は通過単位のブートストラップ 95%。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *

OUT = os.path.join(os.path.dirname(__file__), "..", "summary_noise.png")

fig = new_fig()
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 135); ov.set_ylim(0, 84.375); ov.axis("off")


def txt(x, y, s, size=16, weight="normal", color=INK, ha="left", va="center", **kw):
    return ov.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=5, **kw)


txt(1.2, 81.0, "見えた差は、学習のばらつきと同じくらいの大きさ", size=26, weight="bold")
txt(1.2, 76.8, "保留した R1 の通過 17回・DE の通過 14回の平均と 95% 区間(通過の選び方だけのばらつき)。灰色の帯 = ±10µs",
    size=17, color=INK2)
txt(1.2, 72.6, "●  最良のエポックどうし      ○  最後のエポックどうし      ◆  同じ学習で重みだけ替えた差(last − best)",
    size=17, color=INK)

LABELS = ["印だけの効果 (a) − (n)", "印つき (a) − 印なし (c)", "AWR (b) − (c)", "AI データを足す (c) − B",
          "同じ学習で重みだけ替える"]
# (best/best [平均, 下, 上], last/last [平均, 下, 上])。最後の行は last − best(1つだけ)
R1 = [((4.0, 1.1, 6.9), (1.8, -0.4, 3.8)),
      ((3.4, -2.8, 9.5), (0.4, -4.7, 5.4)),
      ((-9.4, -17.1, -2.6), (-7.4, -14.3, -1.8)),
      ((2.5, -8.4, 15.0), (12.7, 3.6, 23.0)),
      ((9.0, 5.4, 12.2), "(b)")]
DE = [((9.9, 6.4, 13.3), (4.1, 2.7, 5.4)),
      ((2.5, -6.0, 10.6), (-4.8, -10.6, 0.9)),
      ((5.0, -2.6, 12.2), (-0.4, -5.4, 4.6)),
      ((-11.1, -17.0, -6.1), (-4.4, -9.6, -0.4)),
      ((6.4, 2.3, 10.6), "(c)")]
COLS = [AQUA, GREEN, ORANGE, BLUE, INK2]

XL = (-25, 27)


def panel(ax, data, title, xlabel):
    n = len(data)
    ys = [n - 1 - i for i in range(n)]
    ax.axvspan(-10, 10, color="#ecebe7", zorder=0)
    ax.axvline(0, color=INK2, lw=1.4, zorder=1)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":", zorder=1)
    for y, (bb, ll), col in zip(ys, data, COLS):
        if isinstance(ll, str):  # 同じ学習の last − best
            m, lo, hi = bb
            ax.plot([lo, hi], [y, y], color=col, lw=3.0, zorder=2)
            ax.scatter([m], [y], marker="D", s=150, color=col, zorder=3)
            ax.text(hi + 0.8, y, f"{m:+.1f}  {ll}", fontsize=16, va="center", color=INK)
            continue
        for (m, lo, hi), dy, filled in ((bb, 0.17, True), (ll, -0.17, False)):
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=3.0 if filled else 2.0, zorder=2)
            ax.scatter([m], [y + dy], s=150 if filled else 130, zorder=3,
                       color=col if filled else SURF, edgecolors=col, linewidths=2.4)
            if m < 0:  # 負の値は区間の左に置き、0 の線と重ならないようにする
                ax.text(lo - 0.8, y + dy, f"{m:+.1f}", fontsize=16, va="center", ha="right", color=INK)
            else:
                ax.text(hi + 0.8, y + dy, f"{m:+.1f}", fontsize=16, va="center", color=INK)
    ax.set_xlim(*XL)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.tick_params(axis="x", labelsize=16)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=19, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, fontsize=16, color=INK2)
    return ys


ax1 = fig.add_axes([0.255, 0.215, 0.345, 0.565])
ax2 = fig.add_axes([0.640, 0.215, 0.345, 0.565])
ys = panel(ax1, R1, "R1 の手前 0.4秒: スロットル", "差 [µs](+ = 遅い)")
panel(ax2, DE, "DE の出口 0.8秒: ステア", "差 [µs](+ = 左)")
# 行の名前(左の軸に合わせる)
for y, lab, col in zip(ys, LABELS, COLS):
    ax1.text(XL[0] - 1.2, y, lab, fontsize=16, ha="right", va="center", color=INK,
             fontweight="bold" if y == ys[-1] else "normal")
ax1.text(XL[0] - 1.2, ys[-1] - 0.38, "(last − best)", fontsize=16, ha="right", va="center", color=INK2)

# まとめ
txt(1.2, 10.4, "・同じ学習でも、重みを最良→最後のエポックに替えるだけで、出力は最大 9µs 動く(検証の誤差の差は 0.4µs)",
    size=17)
txt(1.2, 6.6, "・印の効果は DE で左へ 4〜10µs と向きはそろうが、大きさは帯(±10µs)の内側。R1 は重みの組で向きが変わる", size=17)
txt(1.2, 2.8, "→ 10µs 以下の差は、1回の学習では見分けられない。次は種を2つ以上にして、ばらつきを区間に入れる",
    size=17, weight="bold", color=RED)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
