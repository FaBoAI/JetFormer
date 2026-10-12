"""JetFormer v0(= モデル B)の構成図。2026-10-11 の日報の fig_architecture.py の写し(出力先だけ変えた)。数値は 2026-10-11 の実測。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "arch_jetformer_v0.png")

fig = new_fig()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 135); ax.set_ylim(0, 84.375); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=2.0, r=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def txt(x, y, s, size=16, weight="normal", color=INK, ha="center", va="center", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def lines(x, yc, rows, ha="center", lh=2.95):
    """rows = [(text, size, weight, color)] を yc を中心に縦に並べる"""
    n = len(rows)
    y0 = yc + (n - 1) * lh / 2
    for i, (s, size, weight, color) in enumerate(rows):
        txt(x, y0 - i * lh, s, size=size, weight=weight, color=color, ha=ha)


def arrow(x0, y0, x1, y1, color=MUTED, lw=2.2, ms=16):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))


# ---- タイトル
txt(1.2, 81.2, "JetFormer の構成", size=28, weight="bold", ha="left")
txt(1.2, 77.0, "TransFuser(CVPR 2021、カメラ+LiDAR を Transformer で融合)を手本に、Orin Nano で 30fps 動くように絞った v0",
    size=17, color=INK2, ha="left")

# ---- 列の見出し
HY = 72.6
for x, s in [(12.5, "入力(毎フレーム)"), (37.5, "センサー別エンコーダ"), (62.5, "融合"), (87, "出力"), (118, "車への出力(走行時)")]:
    txt(x, HY, s, size=17, weight="bold", color=INK2)

# ---- 入力 → エンコーダ の行
XI, WI = 1, 23          # 入力
XE, WE = 27, 21         # エンコーダ
XT, WT = 53, 19         # Transformer
rows = [
    # (y上端, 高さ, 色, 入力の行, エンコーダの行, トークン数)
    (70.0, 9.0, BLUE, [("カメラ IMX219", 17, "bold"), ("224×224×3・30fps", 16, "normal")],
     [("ResNet-18", 17, "bold"), ("ImageNet 事前学習", 16, "normal"), ("→ 512×7×7", 16, "normal")], "49"),
    (60.4, 9.0, ORANGE, [("ToF VL53L9", 17, "bold"), ("42×54×2", 16, "normal"), ("距離+信頼度", 16, "normal")],
     [("小さな CNN 4段", 17, "bold"), ("+有効マスク", 16, "normal"), ("→ 256×6×7", 16, "normal")], "42"),
    (50.8, 4.3, AQUA, [("IMU 100ms(10×6)", 16, "normal")], [("MLP", 16, "normal")], "1"),
    (45.9, 4.3, YELLOW, [("今の操作値+eRPM", 16, "normal")], [("MLP", 16, "normal")], "1"),
    (41.0, 4.3, MAGENTA, [("走行ジャンル(6種)", 16, "normal")], [("埋め込み", 16, "normal")], "1"),
    (36.1, 4.3, VIOLET, [("出力用トークン", 16, "normal")], [("学習するベクトル", 16, "normal")], "1"),
]
TOP, BOT = 70.0, 31.8
for ytop, h, col, inl, enl, ntok in rows:
    yb = ytop - h; yc = ytop - h / 2
    box(XI, yb, WI, h, tint(col, 0.16), col)
    lines(XI + WI / 2, yc, [(s, sz, wt, INK) for s, sz, wt in inl], lh=2.85)
    box(XE, yb, WE, h, "white", col)
    lines(XE + WE / 2, yc, [(s, sz, wt, INK) for s, sz, wt in enl], lh=2.85)
    arrow(XI + WI, yc, XE, yc, color=col)
    arrow(XE + WE, yc, XT, yc, color=col)
    txt(XE + WE + 2.4, yc + 1.25, ntok, size=16, weight="bold", color=INK)

# トークン数の注記
txt(XE + WE + 2.5, BOT - 1.4, "トークン数", size=16, color=INK2)

# ---- Transformer
box(XT, BOT, WT, TOP - BOT, tint(VIOLET, 0.12), VIOLET, lw=2.6)
lines(XT + WT / 2, (TOP + BOT) / 2 - 0.2, [
    ("Transformer", 19, "bold", INK),
    ("Encoder", 19, "bold", INK),
    ("", 10, "normal", INK),
    ("95 トークン", 17, "bold", INK),
    ("× 256 次元", 16, "normal", INK),
    ("4層・8ヘッド", 16, "normal", INK),
    ("", 10, "normal", INK),
    ("自己注意で", 16, "normal", INK),
    ("全センサーを", 16, "normal", INK),
    ("相互参照", 16, "normal", INK),
    ("", 10, "normal", INK),
    ("融合は最終段で", 16, "normal", VIOLET),
    ("1回だけ", 16, "bold", VIOLET),
], lh=2.75)

# ---- 出力(ヘッド)
XH, WH = 76, 22
HB_TOP = 49.8
box(XH, BOT, WH, HB_TOP - BOT, "white", GREEN, lw=2.2)
lines(XH + WH / 2, (HB_TOP + BOT) / 2, [("出力トークン", 16, "normal", INK), ("→ LayerNorm", 16, "normal", INK),
                          ("→ 全結合", 16, "normal", INK), ("ジャンル別ヘッド", 17, "bold", INK),
                          ("(CIL の分岐型)", 16, "normal", INK2)], lh=2.95)
OB = 53.0
box(XH, OB, WH, TOP - OB, tint(GREEN, 0.14), GREEN, lw=2.6)
lines(XH + WH / 2, (TOP + OB) / 2, [("未来 10 フレーム", 17, "bold", INK), ("ステア・スロットル", 16, "normal", INK),
                                ("33〜333ms 先", 16, "normal", INK), ("(約 0.33 秒)", 16, "normal", INK2)], lh=2.95)
arrow(XT + WT, 40.8, XH, 40.8, color=VIOLET)
arrow(XH + WH / 2, HB_TOP, XH + WH / 2, OB, color=GREEN)

# ---- 車への出力(制御)
XC, WC = 102, 32
ctrl = [
    ("先読み k を1つ選ぶ", "当日は k=3(0.1秒先)", GREEN, False),
    ("ゲイン:ステア・速度", "速度は前進側だけ(18:33〜)", GREEN, False),
    ("補助:リカバリー v2", "ESC の後退ルールに合わせる", ORANGE, True),
    ("ToF ガード", "当日は記録だけ(制御しない)", ORANGE, True),
    ("PCA9685 → 基板 → 車", "プロポ CH3=AI 側のときだけ", INK2, False),
]
hb, gap = 6.35, 1.5
y = TOP
for i, (a, b, col, dashed) in enumerate(ctrl):
    yb = y - hb
    ax.add_patch(FancyBboxPatch((XC, yb), WC, hb, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="white" if i < 4 else tint(INK2, 0.10), ec=col, lw=2.0,
                                ls=(0, (4, 2)) if dashed else "-", zorder=2))
    lines(XC + WC / 2, yb + hb / 2, [(a, 17, "bold", INK), (b, 16, "normal", INK2)], lh=2.9)
    if i < 4:
        arrow(XC + WC / 2, yb, XC + WC / 2, yb - gap, color=MUTED, lw=1.8, ms=13)
    y = yb - gap
arrow(XH + WH, TOP - hb / 2, XC, TOP - hb / 2, color=GREEN)

# ---- 下段左: 大きさと速さ
txt(1.2, 26.8, "大きさと速さ(10/10 の実測)", size=19, weight="bold", ha="left")
tiles = [
    ("14.1M", "パラメータ数", "うち ResNet-18 が 11.2M"),
    ("1.9 ms", "TensorRT FP16 の推論", "p99 は 2.2ms"),
    ("5.6 ms", "走行ループ内の推論 p50", "入出力の転送を含む"),
    ("28〜32 ms", "撮像 → PCA9685 出力 p50", "ループは約 30fps"),
]
TW, TH = 29.3, 11.0
for i, (big, l1, l2) in enumerate(tiles):
    cx = 1 + (i % 2) * (TW + 1.4)
    cy = 13.6 - (i // 2) * (TH + 1.2)
    box(cx, cy, TW, TH, "white", GRID, lw=1.6)
    txt(cx + 1.4, cy + TH - 3.0, big, size=24, weight="bold", ha="left")
    txt(cx + 1.4, cy + 4.5, l1, size=16, ha="left")
    txt(cx + 1.4, cy + 1.75, l2, size=16, color=INK2, ha="left")

# ---- 下段右: TransFuser との比較
X0 = 63.5
txt(X0, 26.8, "TransFuser との違い", size=19, weight="bold", ha="left")
txt(X0 + 27.5, 26.6, "※TransFuser の細部は版で異なる", size=16, color=INK2, ha="left")
cols = [X0, X0 + 10.5, X0 + 36.0]       # 項目 / TransFuser / JetFormer
heads = ["", "TransFuser", "JetFormer v0"]
table = [
    ("センサー", "カメラ+LiDAR(BEV)", "カメラ+ToF 42×54"),
    ("融合", "4 解像度で多段", "最終段で 1 回"),
    ("出力", "ウェイポイント → PID", "行動を直接 10 ステップ"),
    ("条件", "目標地点", "走行ジャンル(分岐ヘッド)"),
    ("学習", "シミュレータの専門家", "実車の人の運転 7〜15分"),
]
ry = 23.0
box(X0 - 0.6, 3.6 + 0.2, 135 - X0 - 0.4, 21.2, "white", GRID, lw=1.4, z=1)
for j, hdr in enumerate(heads):
    txt(cols[j], ry, hdr, size=17, weight="bold", color=[INK2, INK2, VIOLET][j], ha="left")
for i, (k, a, b) in enumerate(table):
    yy = ry - 3.4 * (i + 1)
    ax.plot([X0, 134.0], [yy + 1.7, yy + 1.7], color=GRID, lw=1.2, zorder=1)
    txt(cols[0], yy, k, size=16, color=INK2, ha="left")
    txt(cols[1], yy, a, size=16, ha="left")
    txt(cols[2], yy, b, size=16, weight="bold", ha="left")
txt(X0, 2.0, "共通点:センサー別エンコーダ → トークン → 自己注意で融合",
    size=16, color=INK2, ha="left")

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
from zoolib import check_overlaps  # noqa: E402
print('重なり・はみ出し:', check_overlaps(fig))
