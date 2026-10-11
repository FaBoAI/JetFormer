"""(7) JetFormer の Transformer のしくみ(エンコーダと、デコーダの役割)。出典: ~/jetformer/jetformer/model.py
    d=256・8ヘッド・4層・FFN 512・pre-norm(norm_first)・dropout 0.1、トークン 49+42+1+1+1+1=95、
    出力用トークン → LayerNorm → Linear+ReLU → ジャンル別ヘッド(8種×10ステップ×2)→ one-hot で選ぶ"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(__file__), "..", "transformer.png")

fig = new_fig()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 135); ax.set_ylim(0, 84.375); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=2.0, r=0.8, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))


def txt(x, y, s, size=16, weight="normal", color=INK, ha="center", va="center", z=6, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def arrow(x0, y0, x1, y1, color=MUTED, lw=2.6, ms=20):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))


def grid(x, y, rows, cols, color, s=0.78, gap=0.18):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x + c * (s + gap), y + r * (s + gap)), s, s, fc=color, ec="none", zorder=4))


# ---- タイトル
txt(1.2, 81.0, "JetFormer の Transformer のしくみ", size=29, weight="bold", ha="left")
txt(1.2, 76.7, "エンコーダ4層で全センサーのトークンを混ぜ、デコーダの代わりに「出力用トークン」から操作を読み出す",
    size=17.5, color=INK2, ha="left")

# ---- ① トークンにする(左)
txt(1.2, 71.6, "① センサーをトークンに", size=19, weight="bold", ha="left", color=VIOLET)
rows = [  # (y, 色, 名前, 作り方, 個数, grid)
    (63.2, BLUE, "カメラ", "ResNet-18 → 7×7", "49個", (7, 7)),
    (55.6, ORANGE, "距離センサー", "小さな CNN → 6×7", "42個", (6, 7)),
    (49.4, AQUA, "IMU", "0.1秒分→MLP", "1個", (1, 1)),
    (44.4, YELLOW, "操作値・回転数", "MLP", "1個", (1, 1)),
    (39.4, MAGENTA, "走行ジャンル", "埋め込み", "1個", (1, 1)),
    (34.4, VIOLET, "出力用トークン", "", "1個", (1, 1)),
]
for y, col, name, how, n, (gr, gc) in rows:
    h = 6.6 if gr > 1 else 4.2
    box(1.2, y - h / 2, 36.0, h, tint(col, 0.14), col, lw=1.8)
    if gr > 1:
        txt(2.4, y + 1.3, name, size=15.5, weight="bold", ha="left")
        txt(2.4, y - 1.6, how, size=12.8, color=INK2, ha="left")
    else:
        txt(2.4, y, name, size=15, weight="bold", ha="left")
        txt(26.6, y, how, size=12.5, color=INK2, ha="right")
    if gr > 1:
        grid(23.0, y - 3.0, gr, gc, col, s=0.66, gap=0.16)
    else:
        ax.add_patch(Rectangle((27.4, y - 0.62), 1.24, 1.24, fc=col, ec="none", zorder=4))
        if name == "出力用トークン":
            txt(28.02, y - 0.05, "★", size=11, color="#ffffff", z=7)
    txt(36.2, y, n, size=15.5, weight="bold", color=col if col != YELLOW else "#a07000", ha="right")
txt(19.2, 29.0, "合計 95 トークン(1つ = 256 個の数)", size=15.5, weight="bold")
txt(19.2, 25.9, "+ 位置と種類の埋め込みを足す", size=14, color=INK2)

arrow(38.0, 50.0, 42.6, 50.0, color=VIOLET, lw=3.2, ms=24)

# ---- ② エンコーダ(中央)
EX, EW = 43.2, 46.0
txt(EX, 71.6, "② エンコーダで混ぜる(×4層)", size=19, weight="bold", ha="left", color=VIOLET)
box(EX, 24.4, EW, 44.4, tint(VIOLET, 0.07), VIOLET, lw=2.6, r=1.2)
# 4層の重なり
for k in (3, 2, 1):
    box(EX + 0.7 * k, 24.4 - 0.7 * k, EW, 44.4, "none", tint(VIOLET, 0.35), lw=1.4, r=1.2, z=1)
ax.add_patch(Circle((EX + EW - 2.6, 66.2), 2.3, fc=VIOLET, ec="none", zorder=5))
txt(EX + EW - 2.6, 66.2, "×4", size=15, weight="bold", color="#ffffff", z=7)

cx = EX + EW / 2 - 1.0
def sub(y, h, title, note, fc, ec):
    box(EX + 2.2, y, EW - 8.4, h, fc, ec, lw=1.8)
    if note:
        nl = note.count("\n") + 1
        txt(cx, y + h - 2.2, title, size=15.5, weight="bold")
        txt(cx, y + (h - 3.6) / 2, note, size=12.8, color=INK2, linespacing=1.35)
    else:
        txt(cx, y + h / 2, title, size=14, color=INK2)

def plus(y):
    ax.add_patch(Circle((cx, y), 1.25, fc="#ffffff", ec=VIOLET, lw=1.8, zorder=5))
    txt(cx, y - 0.05, "+", size=15, weight="bold", color=VIOLET, z=7)

sub(63.0, 3.0, "LayerNorm(大きさをそろえる)", None, "#ffffff", MUTED)
arrow(cx, 63.0, cx, 61.6, color=MUTED, lw=1.8, ms=13)
sub(51.0, 10.6, "マルチヘッド Self-Attention(8ヘッド)".replace("マルチヘッド ", ""), "どのトークンも、95個すべてを見て\n必要な情報を集める(センサーどうしの融合)",
    tint(BLUE, 0.15), BLUE)
arrow(cx, 51.0, cx, 49.4, color=MUTED, lw=1.8, ms=13)
plus(48.1)
txt(cx + 2.4, 48.1, "残差(元の値を足す)", size=12.5, color=INK2, ha="left")
arrow(cx, 46.85, cx, 45.4, color=MUTED, lw=1.8, ms=13)
sub(42.4, 3.0, "LayerNorm", None, "#ffffff", MUTED)
arrow(cx, 42.4, cx, 41.0, color=MUTED, lw=1.8, ms=13)
sub(32.6, 8.4, "FFN(256 → 512 → 256)", "トークンごとに特徴を変換", tint(AQUA, 0.16), AQUA)
arrow(cx, 32.6, cx, 31.0, color=MUTED, lw=1.8, ms=13)
plus(29.7)
txt(cx + 2.4, 29.7, "残差", size=12.5, color=INK2, ha="left")
txt(cx, 26.4, "4層くり返した後、出力用トークンだけを取り出す", size=13, color=VIOLET, weight="bold")

arrow(EX + EW + 0.6, 46.0, 94.0, 46.0, color=VIOLET, lw=3.2, ms=24)

# ---- ③ 読み出し(右)
RX, RW = 94.6, 39.2
txt(RX, 71.6, "③ 読み出す(デコーダの役割)", size=19, weight="bold", ha="left", color=VIOLET)
box(RX, 61.8, RW, 6.8, tint(VIOLET, 0.14), VIOLET, lw=2.0)
ax.add_patch(Rectangle((RX + 1.6, 64.6), 1.4, 1.4, fc=VIOLET, ec="none", zorder=4))
txt(RX + 2.3, 65.25, "★", size=11, color="#ffffff", z=7)
txt(RX + 4.4, 66.6, "出力用トークン(256)", size=15.5, weight="bold", ha="left")
txt(RX + 4.4, 63.6, "全センサーの情報が集まっている", size=12.8, color=INK2, ha="left")
arrow(RX + RW / 2, 61.8, RX + RW / 2, 59.4, color=MUTED, lw=1.8, ms=13)
box(RX, 55.6, RW, 3.8, "#ffffff", MUTED, lw=1.6)
txt(RX + RW / 2, 57.5, "LayerNorm → MLP", size=14, color=INK2)
arrow(RX + RW / 2, 55.6, RX + RW / 2, 53.2, color=MUTED, lw=1.8, ms=13)
box(RX, 45.2, RW, 8.0, tint(MAGENTA, 0.14), MAGENTA, lw=1.8)
txt(RX + RW / 2, 50.6, "ジャンル別ヘッド(まとめて1回で計算)", size=15, weight="bold")
txt(RX + RW / 2, 47.4, "8種 × 10ステップ × [ステア, スロットル]", size=13, color=INK2)
arrow(RX + RW / 2, 45.2, RX + RW / 2, 42.8, color=MUTED, lw=1.8, ms=13)
txt(RX + RW / 2 - 1.2, 44.0, "走行ジャンルで1つ選ぶ", size=12.5, color=MAGENTA, ha="right")
box(RX, 24.4, RW, 18.4, tint(GREEN, 0.12), GREEN, lw=2.0)
txt(RX + RW / 2, 40.2, "0.33秒先までの操作(10ステップ)", size=15, weight="bold")
# 10ステップの例
import numpy as np
xs = RX + 4.0 + np.arange(10) * 3.3
st = 0.25 + 0.55 * np.sin(np.linspace(0.2, 1.6, 10))
th = 0.55 - 0.18 * np.linspace(0, 1, 10)
base = 28.6
for i, x in enumerate(xs):
    ax.add_patch(Rectangle((x - 1.05, base), 0.95, st[i] * 8, fc=BLUE, ec="none", zorder=4))
    ax.add_patch(Rectangle((x + 0.1, base), 0.95, th[i] * 8, fc=GREEN, ec="none", zorder=4))
ax.plot([RX + 2.0, RX + RW - 2.0], [base, base], color=MUTED, lw=1.2, zorder=4)
txt(xs[0], 26.6, "33ms", size=11.5, color=INK2)
txt(xs[-1], 26.6, "333ms", size=11.5, color=INK2)
ax.add_patch(Rectangle((RX + 9.0, 35.6), 1.1, 1.1, fc=BLUE, ec="none", zorder=4))
txt(RX + 10.6, 36.15, "ステア", size=12.5, ha="left", color=INK2)
ax.add_patch(Rectangle((RX + 19.0, 35.6), 1.1, 1.1, fc=GREEN, ec="none", zorder=4))
txt(RX + 20.6, 36.15, "スロットル", size=12.5, ha="left", color=INK2)

# ---- 比較(下)
cards = [
    (1.2, MUTED, "一般的なエンコーダ・デコーダ(翻訳など)",
     ["デコーダの「問い合わせ」が、エンコーダの", "出力を Cross-Attention で読み、", "答えを1つずつ順に作る"]),
    (46.0, ORANGE, "TransFuser(手本)",
     ["カメラと LiDAR を、複数の解像度で", "Transformer で融合 → GRU のデコーダが", "通過点を1つずつ出し、PID で操作"]),
    (90.8, VIOLET, "JetFormer",
     ["デコーダなし。出力用トークンを入力に混ぜ、", "同じ Self-Attention で読み出す", "→ 10ステップを一度に(推論 1.9ms)"]),
]
for x, col, title, body in cards:
    box(x, 1.6, 43.0, 18.6, tint(col, 0.10) if col != MUTED else "#f3f2ef", col, lw=2.0)
    txt(x + 1.4, 17.0, title, size=15.5, weight="bold", ha="left", color=col if col != MUTED else INK)
    for i, line in enumerate(body):
        txt(x + 1.4, 12.6 - i * 3.5, line, size=13.4, color=INK, ha="left")

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
