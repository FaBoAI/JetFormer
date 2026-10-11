"""(iv) 見えてきた課題(数字つき)。数値は data_check.md・value.md・evaluation.md(レビュー後)・
overnight_20261011/research/offline_rl.md 6章(基板の再起動 20回)。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "summary_issues.png")

fig = new_fig()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 135); ax.set_ylim(0, 84.375); ax.axis("off")


def txt(x, y, s, size=16, weight="normal", color=INK, ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=5, **kw)


def box(x, y, w, h, fc, ec, lw=2.0, r=0.8, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


txt(1.2, 81.0, "見えてきた課題(数字つき)", size=28, weight="bold")
txt(1.2, 76.8, "オフライン強化学習を実車で試す前に、直すところ。→ は次にやること", size=17, color=INK2)

cards = [
    ("価値: 危険を測れていない", RED, "0.41", "衝突の予測 AUROC(基準 0.8)",
     ["報酬の大半が前進距離なので、", "V は「進める量」の価値に(仮説)", "罰だけの価値でも 0.69"],
     "→ 進みと危険の価値に分ける"),
    ("報酬の悪用", ORANGE, "57%", "人の後退のフレームに「良い」",
     ["比べる集団は 37%(+20ポイント)", "芝の空転にも 48%(全体 34%)", "前進距離は車輪の回転で測る"],
     "→ 後退を外し、進みはコース図で"),
    ("データ量: 難しい場面が少ない", BLUE, "17回", "保留した R1 の通過(強い衝突 0回)",
     ["AI の強い R1 衝突 11回のうち", "9回が 1本のログ(173445)", "AI の使えるデータは約 10分"],
     "→ 固定設定の AI 走行を約 100周"),
    ("位置: どこにいるか粗い", VIOLET, "47〜84%", "ジャイロの場所とコース図の一致",
     ["R1・DE はジャイロの向きで判定", "コース図は AI のログ 5本だけ", "走行中は位置が分からない"],
     "→ コース図を全ログに広げて検証"),
    ("検出力: 差が小さすぎる", INK2, "±10µs", "学習のばらつき(重みを替えるだけ)",
     ["見えた差 2〜11µs と同じ大きさ", "種は1つで、区間に入っていない", "実車は各条件 43〜60周が要る"],
     "→ 種を2つ以上・差の差で判定"),
    ("電源: データと作業が途切れる", YELLOW, "4+2件", "基板の再起動で消えていた罰",
     ["衝突 4件・交代 2件の罰が抜けていた", "(見つけて直した)", "10/10 の AI 走行で再起動 20回"],
     "→ 電源の安定化(ハード)が先"),
]
X0, W, GX = 1.0, 42.7, 2.4
TOPS = [73.0, 37.0]
H = 34.0
HDR = 5.0
for k, (title, col, big, cap, rows, nxt) in enumerate(cards):
    r, c = divmod(k, 3)
    x = X0 + c * (W + GX)
    top = TOPS[r]
    box(x, top - H, W, H, "white", col, lw=2.4)
    box(x, top - HDR, W, HDR, tint(col, 0.22), col, lw=2.4)
    txt(x + 1.4, top - HDR / 2, title, size=19, weight="bold")
    txt(x + 1.4, top - 10.6, big, size=36, weight="bold", color=col if col != YELLOW else "#b07800")
    txt(x + 1.4, top - 15.6, cap, size=16, color=INK2)
    for i, s in enumerate(rows):
        txt(x + 1.4, top - 19.6 - i * 2.9, s, size=16)
    txt(x + 1.4, top - 30.6, nxt, size=17, weight="bold", color=GREEN)

fig.savefig(OUT, dpi=DPI)
print("saved", os.path.abspath(OUT))
