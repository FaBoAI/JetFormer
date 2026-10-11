"""(ii) 結果の全体: 価値の AUROC と基準、ゲートC の合否、人との一致、「精度は上がったか」への答え。
数値: data/rl/ai_mix_20261011/value_eval.json(AUROC)、evaluation.json(verdict・accuracy、レビュー後の最終値)。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "summary_results.png")

fig = new_fig()
ov = fig.add_axes([0, 0, 1, 1], zorder=0)
ov.set_xlim(0, 135); ov.set_ylim(0, 84.375); ov.axis("off")


def txt(x, y, s, size=16, weight="normal", color=INK, ha="left", va="center", **kw):
    return ov.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=5, **kw)


def box(x, y, w, h, fc, ec, lw=2.0, r=0.8):
    ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=1))


# ---- タイトル
txt(1.2, 81.0, "結果(机上): 走行の精度は上がっていない。実車に出せる方策は無い", size=26, weight="bold")
txt(1.2, 76.8, "学習に使っていないデータで評価。モデル B = 今の走行モデル(人の走行だけで学習)。数字はレビュー後の最終値",
    size=17, color=INK2)

# =====================================================================
# 左上: 価値の AUROC
txt(1.2, 72.4, "① 価値 V は衝突を予測できなかった", size=20, weight="bold")
txt(1.2, 69.3, "「1秒以内に衝突」を当てる AUROC(AI の 16,257枚)", size=16, color=INK2)
axv = fig.add_axes([0.205, 0.475, 0.255, 0.305])
rows = [  # ラベル, AUROC, 区間, 色
    ("価値 V(判定に使う)", 0.4065, (0.358, 0.466), RED),
    ("速度だけ", 0.5633, (0.512, 0.633), MUTED),
    ("場所ごとの衝突率", 0.6243, (0.541, 0.688), MUTED),
    ("速度+場所で直接", 0.7028, (0.634, 0.751), MUTED),
    ("罰だけの価値(診断)", 0.6932, (0.631, 0.747), BLUE),
    ("特徴で直接(参考)", 0.7632, (0.721, 0.795), BLUE),
]
ys = list(range(len(rows)))[::-1]
for y, (lab, v, (lo, hi), col) in zip(ys, rows):
    axv.barh(y, v - 0.5, left=0.5, height=0.62, color=col, zorder=2)
    axv.plot([lo, hi], [y, y], color=INK, lw=2.0, zorder=3)
    axv.text(0.835, y, f"{v:.2f}", ha="left", va="center", fontsize=17,
             fontweight="bold" if v < 0.5 else "normal", color=RED if v < 0.5 else INK)
axv.set_yticks(ys)
axv.set_yticklabels([r[0] for r in rows], fontsize=16)
axv.get_yticklabels()[0].set_color(RED)
axv.get_yticklabels()[0].set_fontweight("bold")
axv.axvline(0.5, color=INK2, lw=1.6, zorder=1)
axv.axvline(0.8, color=RED, lw=2.0, ls="--", zorder=1)
axv.text(0.795, len(rows) - 0.25, "基準 0.8", color=RED, fontsize=16, ha="right", va="center")
axv.text(0.495, len(rows) - 0.25, "当てずっぽう 0.5", color=INK2, fontsize=16, ha="right", va="center")
axv.set_xlim(0.28, 0.92)
axv.set_ylim(-0.6, len(rows) + 0.2)
axv.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axv.tick_params(axis="x", labelsize=16)
axv.tick_params(axis="y", length=0)
axv.grid(axis="x", color=GRID, lw=1.0, zorder=0)
axv.set_axisbelow(True)
axv.spines["left"].set_visible(False)

# =====================================================================
# 右上: ゲートC
GX = 70.0
txt(GX, 72.4, "② ゲートC(机上の関門): どの方策も不合格", size=20, weight="bold")
txt(GX, 69.3, "重み(best/last)の4通りの組でも同じ結果", size=16, color=INK2)
items = ["1 価値が衝突を予測", "2 R1 の手前で遅く", "3 DE の出口で左へ", "4 崩れていない", "5 報酬の悪用がない",
         "6 陰性対照と見分ける"]
cols = ["(a) 印つき", "(b) AWR", "(n) 陰性対照"]
res = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]]  # evaluation.json verdict.table(best/best)
CX0, CW, CH, CG = 92.0, 13.6, 3.7, 0.5
RY0 = 63.0  # 1行目の上端
for j, c in enumerate(cols):
    txt(CX0 + j * (CW + CG) + CW / 2, RY0 + 1.9, c, size=16, weight="bold", ha="center")
for i, it in enumerate(items):
    yt = RY0 - i * (CH + CG)
    txt(CX0 - 1.0, yt - CH / 2, it, size=16, ha="right")
    if i < 5:
        for j in range(3):
            ok = res[i][j] == 1
            x = CX0 + j * (CW + CG)
            ov.add_patch(Rectangle((x, yt - CH), CW, CH, fc=tint(GREEN if ok else RED, 0.18), ec="none", zorder=1))
            txt(x + CW / 2, yt - CH / 2, "○ 合格" if ok else "× 不合格", size=17, weight="bold",
                color=GREEN if ok else RED, ha="center")
    else:
        x = CX0
        w = 3 * CW + 2 * CG
        ov.add_patch(Rectangle((x, yt - CH), w, CH, fc=tint(RED, 0.18), ec="none", zorder=1))
        txt(x + w / 2, yt - CH / 2, "× (n) も (a)(b) と同じ項目で落ちた", size=17, weight="bold", color=RED, ha="center")
yb = RY0 - 6 * (CH + CG)
txt(GX, yb - 1.3, "(c) filtered BC(失敗の前後を外しただけ)も候補にしない:", size=16, color=INK)
txt(GX, yb - 4.0, "人との一致が B より悪く、ステアが人より右に偏る", size=16, color=INK)

# =====================================================================
# 左下: 人との一致
txt(1.2, 31.6, "③ 人の走りとの一致: 今のモデル B が最良", size=20, weight="bold")
txt(1.2, 28.5, "保留した人の走行 4,670枚との誤差(小さいほど良い)", size=16, color=INK2)
axa = fig.add_axes([0.205, 0.075, 0.255, 0.235])
acc = [  # ラベル, best の L1, 表示(括弧は B との差。evaluation.md 3.1 の表と同じ丸め), 色
    ("モデル B(今)", 49.14, "49.1", INK),
    ("(h) 人だけで微調整", 49.19, "49.2(差なし)", MUTED),
    ("(a) 印つき", 50.93, "50.9(+1.8)", AQUA),
    ("(c) filtered BC", 51.02, "51.0(+1.9)", BLUE),
    ("(n) 陰性対照", 51.29, "51.3(+2.1)", MAGENTA),
    ("(b) AWR", 51.55, "51.6(+2.4)", ORANGE),
]
ys = list(range(len(acc)))[::-1]
for y, (lab, v, d, col) in zip(ys, acc):
    axa.plot([48.0, v], [y, y], color=GRID, lw=2.0, zorder=1)
    axa.scatter([v], [y], s=170, color=col, zorder=3)
    axa.text(v + 0.24, y, d, ha="left", va="center", fontsize=17, color=INK,
             fontweight="bold" if y == ys[0] else "normal")
axa.set_yticks(ys)
axa.set_yticklabels([a[0] for a in acc], fontsize=16)
axa.get_yticklabels()[0].set_fontweight("bold")
axa.set_xlim(48.0, 53.6)
axa.set_ylim(-0.6, len(acc) - 0.4)
axa.set_xticks([48, 49, 50, 51, 52, 53])
axa.tick_params(axis="x", labelsize=16)
axa.tick_params(axis="y", length=0)
axa.spines["left"].set_visible(False)
axa.set_xlabel("L1 [µs](横軸は 48 から)", fontsize=16, color=INK2)

# =====================================================================
# 右下: 答え
BX, BY, BW, BH = 70.0, 1.2, 64.0, 30.6
box(BX, BY, BW, BH, "white", INK2, lw=2.2)
txt(BX + 1.5, BY + BH - 2.6, "「走行の精度は上がったか」への答え", size=20, weight="bold")
ans = [
    ("机上で言えること", 17, "bold", INK),
    ("× 人の走りとの一致は上がらなかった(+3〜5% 悪化)", 16, "normal", INK),
    ("× R1・DE の改善は示せない(差は 10µs 以下 =", 16, "normal", INK),
    ("    学習のばらつきの内。保留の R1 は 17回だけ)", 16, "normal", INK),
    ("△ 印は DE で出力を左へ 4〜10µs 動かした(小さい)", 16, "normal", INK),
    ("△ 交代後の人の修正には近い向き(区間は 0 を含む)", 16, "normal", INK),
    ("机上では言えないこと", 17, "bold", INK),
    ("? 実車で衝突/周が減るか(ゲートD は未実施)", 16, "normal", INK),
]
for k, (s, size, wt, col) in enumerate(ans):
    txt(BX + 1.5, BY + BH - 6.4 - k * 2.75, s, size=size, weight=wt, color=col)
txt(BX + 1.5, BY + 1.9, "→ 実車には出さない。走行はモデル B のまま", size=18, weight="bold", color=RED)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
