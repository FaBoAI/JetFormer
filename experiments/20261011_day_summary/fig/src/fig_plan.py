"""動画用: 「ぶつからずに速く走る車への道筋」のまとめ表(fast_and_safe_plan.md の 10章と同じ内容)。

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    cd ~/jetformer_reports/summary_20261010/fig/src && nice -n 15 python fig_plan.py
出力: ../plan_table.png(2160x1350)
"""
import os
from matplotlib.patches import FancyBboxPatch, Rectangle
from common import new_fig, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, tint

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plan_table.png")

COLS = [("時期", 0.145), ("やること", 0.385), ("衝突/周", 0.115), ("ラップ", 0.115), ("合格の目安", 0.24)]
ROWS = [
    ("今(10/10)", "first_trial の実績(006/007)", "1.17", "9.9秒", "—", "now"),
    ("〜10/14", "測れる状態にする\n(取り付け・電源・電池・v2 の確認)", "測り直す", "測り直す",
     "再起動 0\nバンパー 7±1行", "base"),
    ("〜10/25\n(2週間)", "安全層: 3か所の速度上限、DE 出口の整形\n「遅く入って速く出る」手本で学習し直す",
     "≤0.9", "≤10.2秒", "R1 の強い衝突/通過\n≤10%", "t1"),
    ("〜12月中旬\n(2か月)", "後半の直線を閉ループで速く\nToF ガード v2・ペースの入力", "≤0.7", "≤9.5秒",
     "1段ごとにラップ −0.3秒\n強い衝突を増やさない", "t2"),
    ("2027年春\n(6か月)", "オフライン → オンライン RL\n3DGS・複数コース", "≤0.3", "≤9.0秒",
     "v2 成功 80%・走行中の位置\n車外 GPU", "t3"),
]


def main():
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.935, "ぶつからずに速く走る車への道筋", fontsize=32, fontweight="bold", color=INK, va="center")
    ax.text(0.04, 0.875, "ヘアピンは遅く入り、直線で速く — まず測れるようにし、モデルの外の安全層で守り、手本で教え、後半で速さを取り返す",
            fontsize=15.5, color=INK2, va="center")

    x0, x1 = 0.04, 0.96
    top, head_h, row_h = 0.82, 0.07, 0.118
    widths = [w * (x1 - x0) for _, w in COLS]
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w)

    # 見出し
    ax.add_patch(Rectangle((x0, top - head_h), x1 - x0, head_h, facecolor=INK, edgecolor="none"))
    for i, (name, _) in enumerate(COLS):
        ax.text(xs[i] + 0.012, top - head_h / 2, name, fontsize=18, fontweight="bold", color="white", va="center")

    fills = {"now": tint(MUTED, 0.16), "base": tint(MUTED, 0.06), "t1": tint(BLUE, 0.10),
             "t2": tint(AQUA, 0.12), "t3": tint(ORANGE, 0.10)}
    target_color = {"now": INK, "base": INK2, "t1": BLUE, "t2": "#0f7d57", "t3": "#b84c1c"}

    y = top - head_h
    for when, what, col, lap, gate, kind in ROWS:
        y0 = y - row_h
        ax.add_patch(Rectangle((x0, y0), x1 - x0, row_h, facecolor=fills[kind], edgecolor="none"))
        ax.plot([x0, x1], [y0, y0], color=GRID, lw=1.5)
        cy = y0 + row_h / 2
        ax.text(xs[0] + 0.012, cy, when, fontsize=17, fontweight="bold", color=INK, va="center", linespacing=1.35)
        ax.text(xs[1] + 0.012, cy, what, fontsize=16.5, color=INK, va="center", linespacing=1.45)
        big = kind not in ("base",)
        for j, val in ((2, col), (3, lap)):
            ax.text(xs[j] + widths[j] / 2, cy, val, fontsize=24 if big else 17,
                    fontweight="bold" if big else "normal", color=target_color[kind], ha="center", va="center")
        ax.text(xs[4] + 0.012, cy, gate, fontsize=15, color=INK2, va="center", linespacing=1.45)
        y = y0

    # 人の基準
    ax.add_patch(FancyBboxPatch((x0, 0.08), x1 - x0, 0.056, boxstyle="round,pad=0.004,rounding_size=0.008",
                                facecolor="white", edgecolor=GRID, lw=1.5))
    ax.text(x0 + 0.015, 0.108, "人の基準: 衝突 0.59〜0.99回/周・ラップ 9.4〜9.6秒(衝突なしの周の中央値)・ベスト 7.30秒",
            fontsize=16, color=INK, va="center")
    ax.text(0.04, 0.052, "衝突/周 = AI 周相当・回復中を除く。ラップ = AI・衝突なしの周の中央値。"
            "数字は目標で、効果の見込みは仮説。実車の A/B/A で決める。",
            fontsize=12.5, color=MUTED, va="center")
    ax.text(0.04, 0.024, "2か月の ≤9.5秒は ToF ガードの減速を全区間には入れない場合(入れるなら ≤9.8秒)。"
            "1回の A/B/A(各43周)で見える差は衝突 −40% 以上。",
            fontsize=12.5, color=MUTED, va="center")

    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
