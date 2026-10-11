"""(3) AI走行 11本の結果。値は facts.md 第4章の表(src/runs_table.json の集計)。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *

OUT = os.path.join(os.path.dirname(__file__), "..", "runs_results.png")
RT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "runs_table.json")

# AIだけの周(最初の周を除く)のラップ: runs_table.json の lapcounter から
laps = {}
for r in json.load(open(RT)):
    lc = r.get("lapcounter")
    if lc:
        laps[r["log"][-6:]] = [(x["time_s"], x["clean"]) for x in lc["lap_times"] if x["ai_only"] and not x["first"]]

P1, P2, P3, P4 = MUTED, BLUE, ORANGE, VIOLET
phases = [
    (P1, "初期 15:58〜16:32 — 車両データが途切れて HOLD(基板の再起動とみられる)。16:39 電池切れ"),
    (P2, "ゲイン調整 17:16〜17:34 — 006 は AI 151秒連続・交代1回、007 は交代11回中10回が R1 で停止"),
    (P3, "リカバリー v1 18:15 — 後退できたのは0回、コントローラ基板の再起動 12回"),
    (P4, "夜 20:39〜20:47(v2・ToF ガードは記録だけ)— 衝突 2.86回/周、センサー下向きの疑い(暫定)"),
]
# (名前, 開始, 設定, ログ, AI時間s, 周回(方位), 衝突, 交代, 時期)
runs = [
    ("AIトライ", "15:58", "1 / 1 / 1", "155820", 35.5, 2.11, 6, 4, 0),
    ("run_001", "16:03", "1.6 / 1 / 3", "160306", 38.8, 3.42, 3, 5, 0),
    ("run002", "16:32", "1.6 / 1 / 3", "163247", 36.2, 2.95, 8, 8, 0),
    ("003", "17:16", "1 / 0.8 / 3", "171639", 59.7, 4.90, 7, 2, 1),
    ("004", "17:25", "1.5 / 0.9 / 3", "172505", 13.7, 1.48, 1, 1, 1),
    ("006", "17:29", "22通り・先読み5", "172910", 151.3, 15.94, 22, 1, 1),
    ("007", "17:34", "13通り・先読み3", "173445", 234.5, 22.65, 28, 11, 1),
    ("008", "18:15", "1 / 2 / 3", "181504", 79.7, 5.70, 8, 9, 2),
    ("009", "20:39", "1.7/1.35/3・先読み10", "203924", 23.2, 1.26, 6, 8, 3),
    ("010", "20:42", "1.4/1.25/3 ほか", "204208", 50.5, 3.88, 8, 4, 3),
    ("101", "20:46", "1.5/1.0/3・先読み5", "204605", 25.3, 2.18, 7, 3, 3),
]

# ---- 行の位置(時期の見出し行を挟む)
HDR = 1.0
ys, hdr_y, u = [], [], 0.0
cur = -1
for r in runs:
    if r[8] != cur:
        cur = r[8]; hdr_y.append(u + HDR / 2); u += HDR
    ys.append(u + 0.5); u += 1.0
TOTAL = u
TOP, BOT = 0.835, 0.135

fig = new_fig()
fig.text(0.010, 0.962, "AI走行 11本の結果", fontsize=28, fontweight="bold", color=INK, va="center")
fig.text(0.010, 0.913, "モデルはすべて first_trial(人の運転 約7分・ジャンル5)。006・007 で計 38.6 周、夜の3本は 1 周あたりの衝突が 2 倍以上に",
         fontsize=17, color=INK2, va="center")


def fy(uu):  # 行の単位 → 図の y
    return TOP - uu / TOTAL * (TOP - BOT)


# 左の表(走行・開始・設定)
XN, XT, XS = 0.010, 0.088, 0.142
for x, s in [(XN, "走行"), (XT, "開始"), (XS, "設定")]:
    fig.text(x, 0.862, s, fontsize=17, fontweight="bold", color=INK2, va="center")
for (name, t, setting, *_rest), y in zip(runs, ys):
    col = [P1, P2, P3, P4][_rest[-1]]
    fig.text(XN, fy(y), name, fontsize=17, fontweight="bold", color=INK, va="center")
    fig.text(XT, fy(y), t, fontsize=16, color=INK2, va="center")
    fig.text(XS, fy(y), setting, fontsize=16, color=INK, va="center")
    fig.patches.append(matplotlib.patches.Rectangle((0.002, fy(y) - 0.012), 0.004, 0.024, color=col,
                                                    transform=fig.transFigure, figure=fig))
for (col, s), y in zip(phases, hdr_y):
    fig.text(XN, fy(y) - 0.002, s, fontsize=16, fontweight="bold", color=INK if col == P1 else col, va="center",
             zorder=10, bbox=dict(fc=SURF, ec="none", pad=1.5))

# ---- 5つのパネル(数値は棒に直接書くので目盛りは省く。ラップだけ目盛りあり)
X0, GAP = 0.330, 0.011
WIDTHS = [0.125, 0.100, 0.120, 0.095, 0.160]
specs = [
    ("AI時間 秒", 4, 375, lambda v: "%.1f" % v),
    ("周回(方位)", 5, 33, lambda v: "%.1f" % v),
    ("衝突 / 周", None, 6.6, lambda v: "%.2f" % v),
    ("交代 回", 7, 15.5, lambda v: "%d" % v),
    ("ラップ 秒", None, None, None),
]
xs = [X0 + sum(WIDTHS[:i]) + GAP * i for i in range(5)]
for i, (title, idx, xmax, fmt) in enumerate(specs):
    ax = fig.add_axes([xs[i], BOT, WIDTHS[i], TOP - BOT])
    ax.set_ylim(TOTAL, 0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    fig.text(xs[i], 0.862, title, fontsize=18, fontweight="bold", color=INK, va="center")
    if i < 4:
        ax.set_xlim(0, xmax)
        ax.set_xticks([])
        for r, y in zip(runs, ys):
            col = [P1, P2, P3, P4][r[8]]
            v = r[idx] if i != 2 else r[6] / r[5]
            ax.barh(y, v, height=0.62, color=col, zorder=3)
            bold = (i == 0 and r[0] in ("006", "007")) or (i == 1 and r[0] in ("006", "007")) \
                or (i == 2 and r[8] == 3) or (i == 3 and r[0] == "007")
            ax.text(v + xmax * 0.03, y, fmt(v), fontsize=16, va="center", color=INK,
                    fontweight="bold" if bold else "normal", zorder=4,
                    bbox=dict(fc=SURF, ec="none", pad=0.6) if i == 2 else None)
        if i == 2:
            for v, ls in [(0.99, (0, (4, 2))), (0.59, (0, (1.2, 1.6)))]:
                for y in ys:
                    ax.plot([v, v], [y - 0.48, y + 0.48], color=INK, lw=1.6, ls=ls, zorder=2)
    else:
        ax.set_xlim(8.0, 12.7)
        ax.set_xticks([8, 9, 10, 11, 12])
        ax.tick_params(axis="x", labelsize=16, length=4, pad=3)
        ax.grid(axis="x", color=GRID, lw=1.0)
        ax.set_axisbelow(True)
        ax.axvspan(9.24, 9.64, color=tint(INK2, 0.22), zorder=1, lw=0)
        for r, y in zip(runs, ys):
            col = [P1, P2, P3, P4][r[8]]
            L = laps.get(r[3], [])
            if not L:
                ax.text(8.12, y, "周なし", fontsize=16, color=MUTED, va="center")
                continue
            for t, clean in L:
                if clean:
                    ax.plot([t], [y], "o", ms=12, color=col, mec="white", mew=1.2, zorder=5)
                else:
                    ax.plot([t], [y], "o", ms=8.5, mfc="none", mec=col, mew=1.7, zorder=4)
            cl = sorted(t for t, c in L if c)
            if cl:
                med = (cl[(len(cl) - 1) // 2] + cl[len(cl) // 2]) / 2
                ax.text(max(t for t, _ in L) + 0.28, y, "%.2f" % med, fontsize=16, fontweight="bold",
                        color=INK, va="center")

# 注記(下)
NOTE_Y = 0.068
fig.text(0.010, NOTE_Y, "設定 = ステアゲイン / 速度ゲイン / 先読み\n衝突 = 水平 2.5G 以上(0.6 秒でまとめ)\n"
         "15:50 の first_trial_001 はログ削除で数値なし", fontsize=16, color=INK2, va="center", linespacing=1.2)
fig.text(xs[1], NOTE_Y + 0.012, "ジャイロ方位\n÷ 360°", fontsize=16, color=INK2, va="center", linespacing=1.2)
fig.text(xs[2], NOTE_Y + 0.012, "破線 = 人 午後 0.99\n点線 = 人 夜 0.59", fontsize=16, color=INK2, va="center", linespacing=1.2)
fig.text(xs[4], NOTE_Y - 0.021, "●衝突なし ○AIだけ\n数字 = ●の中央値\n灰帯 = 人の中央値", fontsize=16,
         color=INK2, va="center", linespacing=1.15)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
