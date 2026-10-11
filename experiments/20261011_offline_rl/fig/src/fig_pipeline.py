"""(i) JetFormer でのオフライン強化学習の流れ(データ → 価値 → 印 → 方策 → 関門)。
数値は data_check.md・value.md・policies.md・evaluation.md(レビュー後の最終値)。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "summary_pipeline.png")

fig = new_fig()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 135); ax.set_ylim(0, 84.375); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=2.0, r=0.8, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def txt(x, y, s, size=16, weight="normal", color=INK, ha="center", va="center", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def lines(x, ytop, rows, ha="left", lh=2.9):
    """rows = [(text, size, weight, color)] を ytop から下へ並べる"""
    for i, (s, size, weight, color) in enumerate(rows):
        txt(x, ytop - i * lh, s, size=size, weight=weight, color=color, ha=ha)


def arrow(x0, y0, x1, y1, color=MUTED, lw=2.4, ms=18, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0, ls=ls))


# ---- タイトル
txt(1.2, 81.0, "オフライン強化学習の流れ(JetFormer の構造はそのまま)", size=28, weight="bold", ha="left")
txt(1.2, 76.6, "10/10 の全ログ(人+AI)から「予想より良かった行動」を見つけ、それをまねるように学び直す。机上だけ・実車は未走行",
    size=17, color=INK2, ha="left")

# ---- 上段: 4つの段階
TOP, BOT = 72.6, 42.6
X0, W, GAP = 1.0, 30.6, 3.53
stages = [
    ("① データ", BLUE, [
        ("10/10 の 23ログを全部使う", 16, "normal", INK),
        ("人のきれいな走行", 16, "normal", INK),
        ("+ AI の走行も全部", 16, "bold", INK),
        ("  (今まで捨てていた)", 16, "normal", INK2),
        ("学習に使える 38,434枚", 16, "normal", INK),
        ("(人 21,216・AI 17,218)", 16, "normal", INK2),
        ("行動 = 実際に出した指令", 16, "normal", INK),
        ("衝突・交代を自動で印付け", 16, "normal", INK),
    ]),
    ("② 報酬と価値 V", ORANGE, [
        ("報酬(1/30秒ごと、m 単位)", 16, "normal", INK),
        ("= 前進した距離", 16, "bold", INK),
        ("  − 1 × 衝突(2.5G)", 16, "normal", INK),
        ("  − 2 × 失敗による交代", 16, "normal", INK),
        ("  − 0.5 × リカバリー開始", 16, "normal", INK),
        ("V = この先の報酬の合計", 16, "bold", INK),
        ("モデル B の特徴 256次元", 16, "normal", INK),
        ("→ 小さな MLP で予測", 16, "normal", INK2),
    ]),
    ("③ アドバンテージ → 印", AQUA, [
        ("A = 1秒間の実際の報酬", 16, "normal", INK),
        ("    + V(1秒後) − V(今)", 16, "normal", INK),
        ("=「予想より良かったか」", 16, "bold", INK),
        ("上位 35% に「良い」の印", 16, "bold", INK),
        ("45,744枚に付けた", 16, "normal", INK),
        ("例: R1 の手前の「良い」", 16, "normal", INK2),
        ("  ぶつかった通過 0%", 16, "normal", INK2),
        ("  ぶつからない通過 56%", 16, "normal", INK2),
    ]),
    ("④ 方策を学び直す", VIOLET, [
        ("モデル B から 12エポック", 16, "normal", INK),
        ("(c) 失敗の前後を外すだけ", 16, "normal", INK),
        ("(b) A で重み付け(AWR)", 16, "normal", INK),
        ("(a) 印をジャンル 6・7 に", 16, "bold", INK),
        ("(n) 印を逆にした陰性対照", 16, "normal", INK),
        ("対照 (h): 人だけで微調整", 16, "normal", INK2),
        ("1本 12〜14分(Jetson)", 16, "normal", INK2),
        ("ONNX・TensorRT まで作成", 16, "normal", INK2),
    ]),
]
HDR = 4.6
for i, (title, col, rows) in enumerate(stages):
    x = X0 + i * (W + GAP)
    box(x, BOT, W, TOP - BOT, "white", col, lw=2.4)
    box(x, TOP - HDR, W, HDR, tint(col, 0.22), col, lw=2.4)
    txt(x + W / 2, TOP - HDR / 2, title, size=19, weight="bold")
    lines(x + 1.3, TOP - HDR - 2.5, rows, lh=2.98)
    if i < 3:
        arrow(x + W + 0.2, (TOP + BOT) / 2 + 2, x + W + GAP - 0.2, (TOP + BOT) / 2 + 2, color=INK2)

# ---- 下段左: JetFormer の構造(変えない)と、ジャンルの分岐
txt(1.2, 39.3, "構造は変えない。「良い」の印はジャンルの分岐で入れる",
    size=18, weight="bold", ha="left")

YT, YB = 33.8, 11.5
# 入力
inputs = [("カメラ", BLUE), ("ToF", ORANGE), ("IMU", AQUA), ("eRPM", YELLOW), ("ジャンル", MAGENTA)]
XI, WI = 1.0, 12.0
hh = (YT - YB - 4 * 1.0) / 5
for k, (s, col) in enumerate(inputs):
    yb = YT - (k + 1) * hh - k * 1.0
    box(XI, yb, WI, hh, tint(col, 0.16), col, lw=1.8)
    txt(XI + WI / 2, yb + hh / 2, s, size=16)
    arrow(XI + WI, yb + hh / 2, 16.5, yb + hh / 2, color=col, lw=1.8, ms=13)
# Transformer
XT, WT = 16.5, 17.0
box(XT, YB, WT, YT - YB, tint(VIOLET, 0.12), VIOLET, lw=2.4)
lines(XT + WT / 2, (YT + YB) / 2 + 5.8, [
    ("Transformer", 17, "bold", INK),
    ("4層・256次元", 16, "normal", INK),
    ("", 10, "normal", INK),
    ("構造は", 16, "normal", INK),
    ("変更なし", 16, "bold", INK),
], ha="center", lh=2.9)
# 出力トークン
XO, WO = 36.5, 14.5
YOc = (YT + YB) / 2 + 3.0
box(XO, YOc - 3.6, WO, 7.2, "white", GREEN, lw=2.2)
lines(XO + WO / 2, YOc + 1.45, [("出力トークン", 16, "bold", INK), ("256次元", 16, "normal", INK)], ha="center")
arrow(XT + WT, YOc, XO, YOc, color=VIOLET)
# 価値 V(学習時だけ)
XV, WV = 30.0, 23.0
box(XV, 1.6, WV, 7.0, tint(ORANGE, 0.12), ORANGE, lw=2.0, ls="--")
lines(XV + WV / 2, 6.55, [("価値 V(小さな MLP)", 16, "bold", INK), ("学習の時だけ使う", 16, "normal", INK2)],
      ha="center")
arrow(XO + WO / 2, YOc - 3.6, XO + WO / 2, 8.6, color=ORANGE, ls="--")
# ジャンル別ヘッド
XH, WH = 54.5, 19.0
heads = [("4 高速走行", False), ("5 低速走行", False), ("6 低速走行・良", True), ("7 高速走行・良", True)]
hh2 = 4.4
for k, (s, new) in enumerate(heads):
    yb = YT - (k + 1) * hh2 - k * 1.2
    col = GREEN if new else MUTED
    box(XH, yb, WH, hh2, tint(GREEN, 0.20) if new else "white", col, lw=2.4 if new else 1.8)
    txt(XH + 1.0, yb + hh2 / 2, s, size=16, weight="bold" if new else "normal", ha="left")
    arrow(XO + WO, YOc, XH, yb + hh2 / 2, color=GREEN, lw=1.6, ms=12)
ybot_heads = YT - 4 * hh2 - 3 * 1.2
lines(XH + WH / 2, ybot_heads - 2.6, [("6・7 は今回の追加", 16, "bold", GREEN),
                                      ("走行で 6・7 を選ぶ", 16, "normal", INK2)], ha="center")
txt(XH + WH / 2, YT + 1.8, "ジャンル別ヘッド", size=16, weight="bold", color=INK2)
# 出力
XF, WF = 77.0, 13.5
box(XF, 17.0, WF, 13.0, tint(GREEN, 0.10), GREEN, lw=2.2)
lines(XF + WF / 2, 27.3, [("未来 10", 16, "bold", INK), ("フレーム", 16, "bold", INK),
                         ("ステア・", 16, "normal", INK), ("スロットル", 16, "normal", INK)], ha="center", lh=2.75)
for k in range(4):
    yb = YT - (k + 1) * hh2 - k * 1.2
    ax.plot([XH + WH, XH + WH + 1.0], [yb + hh2 / 2] * 2, color=GREEN, lw=1.6, zorder=3)
ax.plot([XH + WH + 1.0, XH + WH + 1.0], [YT - hh2 / 2, YT - 4 * hh2 - 3 * 1.2 + hh2 / 2], color=GREEN, lw=1.6, zorder=3)
arrow(XH + WH + 1.0, 23.5, XF, 23.5, color=GREEN, lw=1.6, ms=12)

# ---- 下段右: 関門
XG, WG = 92.0, 42.0
txt(XG + WG / 2, 39.3, "実車の前の関門(ロードマップ 4.4)", size=18, weight="bold")
box(XG, 15.8, WG, 20.6, "white", RED, lw=2.4)
txt(XG + 1.3, 34.1, "ゲートC(机上・落とすための関門)", size=17, weight="bold", ha="left", color=RED)
lines(XG + 1.3, 31.0, [
    ("1 価値が衝突を予測(AUROC ≥ 0.8)", 16, "normal", INK),
    ("2 R1 の手前で (c) より遅い", 16, "normal", INK),
    ("3 DE の出口で (c) より左を向く", 16, "normal", INK),
    ("4 崩れていない(人との誤差など)", 16, "normal", INK),
    ("5 報酬の悪用がない(空転・後退)", 16, "normal", INK),
    ("6 陰性対照 (n) が落ちる", 16, "normal", INK),
], lh=2.62)
arrow(XG + WG / 2, 15.8, XG + WG / 2, 12.6, color=INK2)
box(XG, 1.6, WG, 11.0, tint(MUTED, 0.10), INK2, lw=2.0)
txt(XG + 1.3, 10.2, "ゲートD(実車 A/B/A)", size=17, weight="bold", ha="left")
lines(XG + 1.3, 7.2, [("B → 候補 → B、各条件 43〜60周", 16, "normal", INK),
                     ("主の指標 = 衝突/周", 16, "normal", INK)], lh=2.8)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
