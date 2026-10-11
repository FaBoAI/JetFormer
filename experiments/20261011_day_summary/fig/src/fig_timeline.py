"""(2) 10/10 の一日のタイムライン。時刻は facts.md 第5章・git log・生ログ。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import Rectangle, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "day_timeline.png")


def T(hhmm, day=10):
    h, m = map(int, hhmm.split(":"))
    return (day - 10) * 1440 + h * 60 + m


# 時間軸を5つの区間に分けてつなぐ(何もない時間は省く)
SEGS = [  # (開始, 終了, x0, x1, 見出し)
    (T("01:45"), T("04:58"), 14.5, 29.5, "10/10 未明"),
    (T("08:45"), T("10:25"), 31.0, 40.5, "朝"),
    (T("13:40"), T("21:00"), 42.0, 101.5, "10/10 午後〜夜"),
    (T("02:05", 11), T("03:15", 11), 103.0, 112.5, "10/11 未明"),
    (T("05:55", 11), T("09:35", 11), 114.0, 134.5, "10/11 朝"),
]


def X(t):
    for a, b, x0, x1, _ in SEGS:
        if a <= t <= b:
            return x0 + (t - a) / (b - a) * (x1 - x0)
    raise ValueError(t)


fig = new_fig()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 135); ax.set_ylim(0, 84.375); ax.axis("off")
R = fig.canvas.get_renderer()


def txt(x, y, s, size=16, weight="normal", color=INK, ha="left", va="center", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def width_units(s, size, weight):
    t = txt(0, 0, s, size=size, weight=weight)
    bb = t.get_window_extent(R)
    t.remove()
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(bb.x0, 0), (bb.x1, 0)])
    return x1 - x0


# ---- タイトル
txt(1.0, 81.2, "10/10 の一日 — 作って、集めて、学び、走らせた", size=28, weight="bold")
txt(1.0, 77.0, "10/10 01:58 の着手 〜 10/11 09:26。git 31 コミット・AI走行 11本・電池切れの電源断 5回以上(何もない時間は省略)",
    size=17, color=INK2)

# ---- レーン
LANES = {  # 名前: (上端, 下端)
    "dev": (72.0, 59.0), "col": (59.0, 50.5), "trn": (50.5, 41.0),
    "ai": (41.0, 27.0), "ana": (27.0, 16.5), "pwr": (16.5, 8.5),
}
LANE_NAMES = {"dev": "開発・修正\n(git)", "col": "人の運転\nの収集", "trn": "学習", "ai": "AI走行",
              "ana": "解析・報告", "pwr": "電源断"}
for i, (k, (top, bot)) in enumerate(LANES.items()):
    if i % 2 == 0:
        ax.add_patch(Rectangle((0.5, bot), 134.3, top - bot, fc="#f2f1ed", ec="none", zorder=0))
    txt(1.0, (top + bot) / 2, LANE_NAMES[k], size=17, weight="bold", linespacing=1.15)

# ---- 区間の見出し・時刻の目盛り・区切り
LAST_TICK = [-99]
for a, b, x0, x1, name in SEGS:
    txt((x0 + x1) / 2, 74.0, name, size=17, weight="bold", color=INK2, ha="center")
    ax.plot([x0, x1], [8.0, 8.0], color=MUTED, lw=1.6, zorder=2)
    h = (a // 60) + 1
    while h * 60 <= b:
        x = X(h * 60)
        ax.plot([x, x], [8.5, 72.0], color=GRID, lw=1.0, zorder=0.5)
        ax.plot([x, x], [7.4, 8.0], color=MUTED, lw=1.4, zorder=2)
        if x - LAST_TICK[0] > 4.5:
            txt(x, 5.6, "%d時" % (h % 24), size=16, color=INK2, ha="center")
            LAST_TICK[0] = x
        h += 1
for (_, _, _, xa, _), (_, _, xb, _, _) in zip(SEGS[:-1], SEGS[1:]):
    xm = (xa + xb) / 2
    ax.plot([xm, xm], [8.5, 72.0], color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=0.6)
    txt(xm, 8.0, "≈", size=18, color=MUTED, ha="center", z=6)

# ---- ラベルの段組み(重ならないように自動で段を決める)
def place(lane, items, rows, pad=0.9):
    """items: (t, text, align, opts)。rows: 段の y。左から順に、入る最初の段へ。"""
    used = {i: [] for i in range(len(rows))}
    out = []
    for t, s, align, o in sorted(items, key=lambda it: it[0]):
        size = o.get("size", 16); wt = o.get("weight", "normal")
        w = width_units(s, size, wt)
        x = X(t) + o.get("dx", 0)
        x0 = {"left": x - 0.4, "right": x - w + 0.4, "center": x - w / 2}[align]
        x1 = x0 + w
        cand = o.get("rows", range(len(rows)))
        for r in cand:
            if all(x1 + pad < a or x0 - pad > b for a, b in used[r]):
                used[r].append((x0, x1))
                txt(x0, rows[r], s, size=size, weight=wt, color=o.get("color", INK))
                out.append((r, x, x0, x1))
                break
        else:
            print("!! no row for", lane, s)
    return out


def ticks(times, y0, y1, color=INK2, lw=2.0):
    for t in times:
        x = X(t)
        ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=3, solid_capstyle="butt")


def bar(t0, t1, y, h, color, minw=0.55, hatch=None, alpha=1.0, ec=None):
    x0, x1 = X(t0), X(t1)
    if x1 - x0 < minw:
        x1 = x0 + minw
    ax.add_patch(Rectangle((x0, y - h / 2), x1 - x0, h, fc=color, ec=ec or "none", lw=1.2, hatch=hatch,
                           alpha=alpha, zorder=3))


# 開発(git コミット 31 本の時刻)
commits = ["02:30", "02:41", "02:41", "02:53", "02:54", "04:49",
           "09:09", "09:22", "09:23", "09:34", "10:17", "10:18",
           "14:05", "14:09", "14:26", "14:57", "15:20", "15:58", "16:10", "17:54", "17:58",
           "18:33", "18:39", "18:40", "18:43", "19:36", "20:36"]
ticks([T(c) for c in commits] + [T("02:48", 11), T("08:26", 11), T("09:25", 11), T("09:26", 11)], 68.3, 71.0)
dy = LANES["dev"]
rows_dev = [66.4, 63.6, 60.8]
place("dev", [
    (T("02:30"), "モデル v0", "left", {}),
    (T("02:30"), "学習・TRT", "left", {"rows": [1]}),
    (T("02:30"), "AI走行画面", "left", {"rows": [2]}),
    (T("08:51"), "音声案内", "left", {}),
    (T("09:30"), "収集の改良", "left", {"rows": [1]}),
    (T("14:26"), "評価ツール", "left", {"rows": [2]}),
    (T("15:20"), "操作値なし版", "left", {"weight": "bold", "rows": [1]}),
    (T("15:58"), "停止・途切れ修正", "left", {"rows": [0]}),
    (T("17:54"), "リカバリー v1", "left", {"rows": [1]}),
    (T("18:43"), "v2・ToFガード", "right", {"rows": [2], "dx": 0.6}),
    (T("19:36"), "周回の定義", "left", {"rows": [2]}),
    (T("20:36"), "衝突を除外", "right", {"rows": [0], "dx": 0.6}),
    (T("02:18", 11), "比較ツール", "left", {"rows": [1], "dx": 0.6}),
    (T("08:26", 11), "途中保存・再開", "right", {"rows": [0]}),
    (T("09:25", 11), "センサー点検", "right", {"rows": [1]}),
], rows_dev)

# 人の運転の収集
yc = 56.4
bar(T("13:53"), T("15:29"), yc, 2.2, tint(GREEN, 0.25))
ticks([T(c) for c in ["13:53", "14:40", "15:04", "15:24", "15:27"]], yc - 1.1, yc + 1.1, color=GREEN, lw=3)
bar(T("19:21"), T("19:35"), yc, 2.2, tint(INK2, 0.30))
bar(T("19:36"), T("20:31"), yc, 2.2, tint(GREEN, 0.25))
ticks([T(c) for c in ["19:36", "20:09", "20:16", "20:26"]], yc - 1.1, yc + 1.1, color=GREEN, lw=3)
place("col", [
    (T("13:53"), "低速 22シーン 約7分", "left", {}),
    (T("19:21"), "駐車", "right", {"color": INK2, "dx": -0.3}),
    (T("19:36"), "綺麗な走行 30シーン", "left", {"weight": "bold"}),
], [52.6])

# 学習
yc = 48.0
bar(T("03:20"), T("04:47"), yc, 2.2, tint(INK2, 0.30))
bar(T("14:19"), T("14:28"), yc, 2.2, tint(BLUE, 0.45))
bar(T("14:46"), T("14:55"), yc, 2.2, tint(BLUE, 0.45))
bar(T("15:15"), T("15:16"), yc, 2.2, tint(INK2, 0.30))
bar(T("15:20"), T("15:22"), yc, 2.2, tint(INK2, 0.30))
bar(T("15:36"), T("15:48"), yc, 2.2, BLUE)
bar(T("02:18", 11), T("02:38", 11), yc, 2.2, GREEN)
bar(T("02:46", 11), T("03:01", 11), yc, 2.2, VIOLET)
bar(T("08:26", 11), T("08:31", 11), yc, 2.2, VIOLET)
place("trn", [
    (T("03:20"), "机上テスト", "left", {"color": INK2, "dx": -2.2}),
    (T("14:19"), "lowspeed a・b", "right", {"dx": 1.3}),
    (T("15:36"), "first_trial 51.0µs(本番)", "left", {"weight": "bold", "rows": [1]}),
    (T("02:18", 11), "A 35.7µs", "left", {"rows": [0], "dx": -0.6}),
    (T("02:46", 11), "B は途中停止→再開", "left", {"rows": [1]}),
    (T("08:31", 11), "B 49.0µs", "right", {"rows": [0], "dx": 0.6}),
], [44.6, 41.9])

# AI 走行(色 = 時期)
runs = [  # (開始, 長さ秒, 色)
    ("15:50", None, MUTED), ("15:58", 48.5, MUTED), ("16:03", 69.4, MUTED), ("16:32", 140.0, MUTED),
    ("17:16", 100.0, BLUE), ("17:25", 20.0, BLUE), ("17:29", 162.5, BLUE), ("17:34", 300.0, BLUE),
    ("18:15", 190.8, ORANGE),
    ("20:39", 91.4, VIOLET), ("20:42", 117.1, VIOLET), ("20:46", 100.5, VIOLET),
]
yc = 38.6
for t, dur, col in runs:
    x = X(T(t))
    if dur is None:
        ax.plot([x], [yc], marker="o", ms=10, mfc="white", mec=col, mew=2.2, zorder=4)
    else:
        w = max(dur / 60 * 0.1352, 0.6)
        ax.add_patch(Rectangle((x, yc - 1.3), w, 2.6, fc=col, ec="white", lw=0.8, zorder=4))
place("ai", [
    (T("16:35"), "初期4本・不安定", "right", {"rows": [0], "color": INK2}),
    (T("17:33"), "006 約16周・交代1回", "right", {"weight": "bold", "rows": [1]}),
    (T("17:36"), "007 R1で止まる", "left", {"rows": [2]}),
    (T("18:15"), "008 リカバリー v1", "left", {"rows": [0]}),
    (T("20:49"), "夜の3本 衝突増", "right", {"rows": [1]}),
], [34.7, 31.9, 29.1])

# 解析・報告
yc = 24.6
for t in ["04:51", "14:26", "14:56", "17:46", "19:39"]:
    ax.plot([X(T(t))], [yc], marker="D", ms=9, color=AQUA, zorder=4)
bar(T("18:07"), T("18:21"), yc, 2.2, AQUA)
for t in ["06:21", "07:12", "08:44", "08:47"]:
    ax.plot([X(T(t, 11))], [yc], marker="D", ms=9, color=AQUA, zorder=4)
place("ana", [
    (T("04:51"), "朝の報告動画", "right", {"dx": 0.8}),
    (T("14:26"), "「操作値をまねる」癖を発見", "left", {"weight": "bold"}),
    (T("17:46"), "ダイジェスト動画", "right", {"rows": [1], "dx": 0.6}),
    (T("18:07"), "解析4本", "left", {"rows": [1]}),
    (T("19:39"), "レポート動画", "left", {"rows": [0], "dx": -0.6}),
    (T("06:21", 11), "比較: B が最良", "left", {"rows": [0], "dx": -2.0}),
    (T("07:12", 11), "動画レポート 2本", "left", {"rows": [1], "dx": -4.0}),
], [21.3, 18.5])

# 電源
yc = 14.2
for t in ["16:39", "17:07", "19:43", "20:47"]:
    ax.plot([X(T(t))], [yc], marker="X", ms=15, color=RED, mec="white", mew=1.0, zorder=4)
ax.plot([X(T("03:01", 11))], [yc], marker="X", ms=15, color=RED, mec="white", mew=1.0, zorder=4)
for t in ["15:16", "15:20"]:
    ax.plot([X(T(t))], [yc], marker="X", ms=12, color=MUTED, mec="white", mew=1.0, zorder=4)
ax.plot([X(T("03:53"))], [yc], marker="X", ms=12, color=MUTED, mec="white", mew=1.0, zorder=4)
place("pwr", [
    (T("03:53"), "メモリ不足", "left", {"color": INK2, "dx": -6.0}),
    (T("15:16"), "再起動", "right", {"color": INK2, "dx": -0.8}),
    (T("16:39"), "電池切れ 16:39・17:07・19:4x・20:47", "left", {"color": INK, "dx": -0.6}),
    (T("03:01", 11), "電池切れ", "left", {"dx": -2.4}),
    (T("06:06", 11), "06:06 復活", "left", {"color": INK2, "dx": 4.2}),
], [10.8])
ax.plot([X(T("06:06", 11))], [yc], marker="^", ms=12, color=GREEN, mec="white", mew=1.0, zorder=4)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
