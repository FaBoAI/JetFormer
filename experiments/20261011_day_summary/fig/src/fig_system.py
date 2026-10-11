"""(6) 車とセンサーの構成図(型番つき)。出典: ~/jetformer/jetformer/config.py・sensors/*.py・docs/firmware_protocol.md"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "system_config.png")

fig = new_fig()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 135); ax.set_ylim(0, 84.375); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=2.2, r=0.9, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))


def txt(x, y, s, size=16, weight="normal", color=INK, ha="center", va="center", z=5):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z)


def arrow(x0, y0, x1, y1, color=MUTED, lw=2.4, label=None, lx=None, ly=None, both=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if both else "-|>", mutation_scale=18,
                                 color=color, lw=lw, zorder=3, shrinkA=0, shrinkB=0))
    if label:
        txt(lx if lx is not None else (x0 + x1) / 2, ly if ly is not None else (y0 + y1) / 2 + 1.4,
            label, size=13.5, color=color, weight="bold")


def device(x, y, w, h, color, title, model, spec):
    box(x, y, w, h, tint(color, 0.13), color)
    txt(x + 1.2, y + h - 2.5, title, size=15.5, color=INK2, ha="left")
    txt(x + 1.2, y + h - 6.0, model, size=21, weight="bold", ha="left")
    txt(x + 1.2, y + 2.6, spec, size=14.5, color=INK2, ha="left")


# ---- タイトル
txt(1.2, 81.0, "車とセンサーの構成", size=30, weight="bold", ha="left")
txt(1.2, 76.6, "タミヤ TT-02 SRX(タイヤ 65mm)に、センサー3種と Jetson を積んだ自作の AI カー",
    size=18, color=INK2, ha="left")

# ---- センサー(左)
SX, SW, SH = 1.2, 39, 12.6
device(SX, 58.6, SW, SH, BLUE, "カメラ(AI の目)", "Sony IMX219",
       "1920×1080 30fps → 224×224 で入力")
device(SX, 42.4, SW, SH, ORANGE, "距離センサー(ToF)", "ST VL53L9",
       "54×42 ゾーン・約32fps(ESP32-P4 経由)")
device(SX, 26.2, SW, SH, AQUA, "IMU(加速度・ジャイロ)", "TDK ICM-20948",
       "100Hz・±8G・±1000°/秒")
device(SX, 10.0, SW, SH, YELLOW, "車両データ(回転数・プロポ)", "XIAO ESP32-S3 基板",
       "ホールセンサー → 回転数・100Hz")

# ---- Jetson(中央)
JX, JY, JW, JH = 51.0, 22.0, 36, 43
box(JX, JY, JW, JH, tint(VIOLET, 0.10), VIOLET, lw=2.8)
txt(JX + JW / 2, JY + JH - 3.6, "NVIDIA Jetson Orin Nano", size=18.5, weight="bold")
txt(JX + JW / 2, JY + JH - 7.4, "8GB・電池で駆動", size=15, color=INK2)
box(JX + 2.2, JY + 12.5, JW - 4.4, 16.0, "#ffffff", VIOLET, lw=2.2)
txt(JX + JW / 2, JY + 25.2, "JetFormer", size=24, weight="bold", color=VIOLET)
txt(JX + JW / 2, JY + 21.2, "TensorRT FP16・推論 1.9ms", size=15.5)
txt(JX + JW / 2, JY + 17.6, "撮像 → 出力 約28ms・30fps", size=15.5)
txt(JX + JW / 2, JY + 14.5, "0.33秒先までの操作を出す", size=14.5, color=INK2)
txt(JX + JW / 2, JY + 8.0, "記録・学習データの収集も", size=14.5, color=INK2)
txt(JX + JW / 2, JY + 4.6, "この上で行う", size=14.5, color=INK2)

# センサー → Jetson
ax_in = JX
arrow(SX + SW, 64.9, ax_in, 58.0, color=BLUE, label="CSI", lx=46.6, ly=63.6)
arrow(SX + SW, 48.7, ax_in, 48.0, color=ORANGE, label="USB", lx=45.8, ly=50.2)
arrow(SX + SW, 32.5, ax_in, 38.0, color=AQUA, label="I2C", lx=46.6, ly=37.4)
arrow(SX + SW, 16.3, ax_in, 28.0, color=YELLOW, label="I2C", lx=44.6, ly=26.0)

# ---- 出力(右)
RX, RW = 95, 36.0
device(RX, 58.6, RW, SH, GREEN, "PWM 出力", "NXP PCA9685",
       "CH0 ステア・CH1 スロットル・60Hz")
arrow(JX + JW, 52.0, RX, 64.9, color=GREEN, label="I2C", lx=89.4, ly=61.4)

box(RX, 36.4, RW, 15.2, tint(YELLOW, 0.13), YELLOW)
txt(RX + 1.2, 48.6, "切り替え(同じ XIAO ESP32-S3 基板)", size=15.5, color=INK2, ha="left")
txt(RX + 1.2, 44.4, "プロポ CH3 で AI 側 / 人の側", size=19, weight="bold", ha="left")
txt(RX + 1.2, 40.5, "AI 側のときだけ PCA9685 が車を動かす", size=13.6, color=INK2, ha="left")
txt(RX + 1.2, 37.9, "人はいつでもプロポで交代できる", size=13.6, color=INK2, ha="left")
arrow(RX + RW / 2, 58.6, RX + RW / 2, 51.6, color=GREEN)

box(RX, 10.0, 17.0, 19.0, tint(MAGENTA, 0.13), MAGENTA)
txt(RX + 8.5, 25.0, "サーボ", size=19, weight="bold")
txt(RX + 8.5, 20.6, "ステアリング", size=15, color=INK2)
box(RX + 19.0, 10.0, 17.0, 19.0, tint(RED, 0.10), RED)
txt(RX + 27.5, 25.0, "ESC", size=19, weight="bold")
txt(RX + 27.5, 20.6, "ブラシレス", size=15, color=INK2)
txt(RX + 27.5, 17.2, "モーター", size=15, color=INK2)
txt(RX + 27.5, 13.0, "(ホール付き)", size=14, color=INK2)
arrow(RX + 8.5, 36.4, RX + 8.5, 29.0, color=MAGENTA)
arrow(RX + 27.5, 36.4, RX + 27.5, 29.0, color=RED)


# プロポ
box(RX + 4.7, 72.4, 26.6, 6.0, "#ffffff", MUTED, lw=1.8, ls="--")
txt(RX + 18.0, 75.4, "プロポ(送信機・受信機)", size=15, color=INK2)

ax.plot([RX + 31.3, RX + 38.0, RX + 38.0], [75.4, 75.4, 44.0], color=MUTED, lw=2.0, zorder=3, ls=(0, (5, 3)))
ax.add_patch(FancyArrowPatch((RX + 38.0, 44.0), (RX + RW, 44.0), arrowstyle="-|>", mutation_scale=15, color=MUTED,
                             lw=2.0, zorder=3, shrinkA=0, shrinkB=0))

# ホール → 車両基板(下を通す)
ax.plot([RX + 27.5, RX + 27.5, 20.7, 20.7], [10.0, 6.4, 6.4, 8.0], color=YELLOW, lw=2.2, zorder=3, ls=(0, (5, 3)))
ax.add_patch(FancyArrowPatch((20.7, 7.6), (20.7, 10.0), arrowstyle="-|>", mutation_scale=16, color=YELLOW,
                             lw=2.2, zorder=3, shrinkA=0, shrinkB=0))
txt(70.0, 3.9, "モーターの回転数(ホールセンサー)と、プロポの操作(CH1〜3)を基板が読む", size=13.5, color=INK2)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
