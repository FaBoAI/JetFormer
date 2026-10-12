"""図の共通設定: 16:10・2160x1350px(表示 約1350x844 を想定)。
13.5インチ幅なので 1pt ≈ 表示1.39px。本文16pt以上 = 表示22px以上。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "font.size": 17,
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

W, H = 13.5, 8.4375
DPI = 160
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURF = "#fcfcfb"
# 既定の分類パレット(固定順)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = C


def tint(hex_color, a):
    """白に混ぜた淡い色(a=色の割合)"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    s = (252, 252, 251)
    return "#%02x%02x%02x" % tuple(int(s[i] * (1 - a) + c * a) for i, c in enumerate((r, g, b)))


def new_fig():
    return plt.figure(figsize=(W, H), dpi=DPI)
