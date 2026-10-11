"""動画用: 「ぶつからずに速く走る車への道筋」のロードマップ図(fast_and_safe_plan.md のレビュー後の版)。
目標(2章)・時期ごとの施策(3章)・ゲート(4.3)・次にやる3つ(8章)を1枚にまとめる。

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    cd ~/jetformer_reports/summary_20261010/fig/src && nice -n 15 python fig_plan_roadmap.py
出力: ../plan.png(2160x1350)。文字が枠からはみ出すと WARN を出す。
"""
import os
from matplotlib.patches import FancyBboxPatch, Polygon, Circle
from common import new_fig, INK, INK2, BLUE, tint

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plan.png")
FW, FH = 2160, 1350          # 座標 = PNG の画素(y は下向き)
X0, X1, GAP = 80, 2080, 14   # 左右の余白と列の間
NCOL = 4
CW = (X1 - X0 - GAP * (NCOL - 1)) / NCOL

GRAY_D = "#3d3c39"
GREEN_D = "#0f7d57"
ORANGE_D = "#b84c1c"
BLUE_D = "#1f63b8"
PHASES = [  # 名前, 期間, 帯の色, 文字の色, 薄い色のもと
    ("今", "〜10/14", GRAY_D, INK, "#8a8984"),
    ("2週間", "〜10/25", BLUE, BLUE_D, BLUE),
    ("2か月", "〜12/11ごろ", GREEN_D, GREEN_D, "#1baf7a"),
    ("後で", "12月中旬〜", ORANGE_D, ORANGE_D, "#eb6834"),
]
TARGETS = [  # 札, 衝突/周, ラップ
    ("10/10 の実績", "1.17", "9.9秒"),
    ("目標 10/25", "≤0.9", "≤10.2秒"),
    ("目標 12/11ごろ", "≤0.7", "≤9.5秒*"),
    ("目標 2027年4月ごろ", "≤0.3", "≤9.0秒"),
]
LATER = {  # 列 → カード(題, 説明)
    2: [("後半の直線を速く", "閉ループ 3.0→3.3→3.6 m/s"),
        ("ToF ガードの減速", "修正データ(HG-DAgger)"),
        ("ペースの入力", "合成データ・オフライン RL")],
    3: [("進路+目標速度の出力", ["ペースで足りないとき"]),
        ("オンライン RL", ["3DGS・複数コース"])],
}
GATES = [  # 題, 条件
    ("G0 基盤 10/14 朝", ["再起動 0・バンパー 7±1行", "v2 の成功 ≥8/10"]),
    ("G4 安全層・G3 B2", ["地点の衝突/通過で判定", "ラップの悪化 ≤0.3秒"]),
    ("G5〜G7 1段ずつ", ["ラップ −0.3秒(速くする段)", "強い衝突を増やさない"]),
    ("G8 RL の前提", ["v2 の成功 ≥80%", "走行中の位置・車外 GPU"]),
]

checks = []  # (text, (x0, y0, x1, y1)) 枠に収まっているか後で確かめる


def colx(i):
    return X0 + i * (CW + GAP)


def box(ax, x, y, w, h, fc, ec="none", lw=0, r=14, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, lw=lw, zorder=z))


def txt(ax, x, y, s, frame, **kw):
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    t = ax.text(x, y, s, zorder=5, **kw)
    checks.append((t, frame))
    return t


def badge(ax, x, y, n):
    ax.add_patch(Circle((x, y), 25, facecolor=INK, edgecolor="none", zorder=4))
    ax.text(x, y + 1, str(n), fontsize=17, fontweight="bold", color="white", ha="center", va="center", zorder=5)


def card(ax, x, y, w, h, title, subs, phase, n=None):
    """n があれば「次にやる3つ」の強調カード(白地・黒枠・番号)"""
    frame = (x, y, x + w, y + h)
    if n:
        box(ax, x, y, w, h, "white", INK, 2.6)
        badge(ax, x + 42, y + 50, n)
        tx = x + 82
    else:
        box(ax, x, y, w, h, tint(PHASES[phase][4], 0.12), tint(PHASES[phase][4], 0.45), 1.4)
        tx = x + 24
    nl = 1 + len(subs)
    if n:
        ys = [y + 50] + [y + 50 + 52 * (k + 1) for k in range(len(subs))]
    else:
        ys = [y + h / 2 - 26 * (nl - 1) + 52 * k for k in range(nl)]
    txt(ax, tx, ys[0], title, frame, fontsize=19, fontweight="bold")
    for k, s in enumerate(subs):
        txt(ax, x + (82 if n else 24), ys[k + 1], s, frame, fontsize=16, color=INK2)


def main():
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(FH, 0)
    ax.axis("off")
    page = (0, 0, FW, FH)

    txt(ax, X0, 68, "ぶつからずに速く走る車への道筋", page, fontsize=30, fontweight="bold")
    txt(ax, X0, 128, "ヘアピンは遅く入り、直線で速く。数字は目標で予測ではない — 各ゲートで実測して決める",
        page, fontsize=17, color=INK2)

    # 時期の帯(矢羽根)
    top, hh, tip = 162, 68, 28
    for i, (name, span, fc, _, _) in enumerate(PHASES):
        x = colx(i)
        pts = [(x, top), (x + CW - tip, top), (x + CW + tip * 0.15, top + hh / 2), (x + CW - tip, top + hh), (x, top + hh)]
        if i > 0:
            pts.append((x + tip, top + hh / 2))
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor="none", zorder=2))
        fr = (x, top, x + CW, top + hh)
        t = txt(ax, x + (tip + 16 if i else 22), top + hh / 2, name, fr, fontsize=22, fontweight="bold", color="white")
        txt(ax, x + CW - tip - 14, top + hh / 2 + 2, span, fr, fontsize=18, color="white", ha="right")

    # 目標
    ty, th = 246, 200
    for i, (tag, col, lap) in enumerate(TARGETS):
        x = colx(i)
        light = PHASES[i][4]
        box(ax, x, ty, CW, th, tint(light, 0.10 if i else 0.14), tint(light, 0.5), 1.4)
        fr = (x, ty, x + CW, ty + th)
        txt(ax, x + 22, ty + 34, tag, fr, fontsize=16, fontweight="bold", color=INK2)
        for k, (lab, val) in enumerate((("衝突/周", col), ("ラップ", lap))):
            yy = ty + 94 + k * 64
            txt(ax, x + 22, yy, lab, fr, fontsize=17, color=INK2)
            txt(ax, x + CW - 22, yy, val, fr, fontsize=27, fontweight="bold", color=PHASES[i][3], ha="right")

    # 見出し: 次にやる3つ / その後
    ly = 492
    left_w = 2 * CW + GAP
    ax.plot([X0, X0 + left_w], [ly + 20, ly + 20], color=INK, lw=2.2, zorder=1)
    txt(ax, X0, ly - 4, "次にやる3つ", page, fontsize=19, fontweight="bold")
    ax.plot([colx(2), X1], [ly + 20, ly + 20], color="#8a8984", lw=1.6, zorder=1)
    txt(ax, colx(2), ly - 4, "その後(施策は見込み)", page, fontsize=18, fontweight="bold", color=INK2)

    # やること
    iy, ih = 528, 420
    rh = (ih - 14) / 2
    card(ax, colx(0), iy, CW, rh, "測れる状態にする", ["取り付けの固定・電源", "電池の持ち・v2 の検証"], 0, n=1)
    card(ax, colx(1), iy, CW, rh, "手本で学習し直す", ["遅く入って速く出る", "10/16 収集→10/22 比較"], 1, n=3)
    card(ax, colx(0), iy + rh + 14, left_w, rh, "安全層 v1(モデルの外で守る)",
         ["速度上限 R1 2.5・GM 2.2・DE 3.0 m/s + DE 出口の整形",
          "再生で確認 → 10/14 シャドー → 10/20 実車で比べる"], 1, n=2)
    for col, items in LATER.items():
        m = len(items)
        h = (ih - 14 * (m - 1)) / m
        for k, (title, sub) in enumerate(items):
            subs = sub if isinstance(sub, list) else [sub]
            card(ax, colx(col), iy + k * (h + 14), CW, h, title, subs, col)

    # ゲート
    gy, gh = 966, 156
    for i, (title, conds) in enumerate(GATES):
        x = colx(i)
        box(ax, x, gy, CW, gh, "white", tint(PHASES[i][4], 0.6), 1.6)
        fr = (x, gy, x + CW, gy + gh)
        dx, dy = x + 36, gy + 38
        ax.add_patch(Polygon([(dx, dy - 17), (dx + 17, dy), (dx, dy + 17), (dx - 17, dy)], closed=True,
                             facecolor=PHASES[i][2], edgecolor="none", zorder=4))
        txt(ax, x + 66, dy, title, fr, fontsize=18, fontweight="bold", color=PHASES[i][3])
        for k, c in enumerate(conds):
            txt(ax, x + 24, gy + 84 + k * 44, c, fr, fontsize=16, color=INK)
    txt(ax, X0, 1160, "ゲート不合格なら: その層は記録だけ(シャドー)に戻す・1段下げる・日程を1週ずらす",
        page, fontsize=17, color=INK)

    # 注
    ax.plot([X0, X1], [1208, 1208], color="#e4e3df", lw=1.5)
    txt(ax, X0, 1246, "衝突/周 = AI 周相当・回復中を除く。ラップ = 衝突なしの周の中央値。人: 0.59〜0.99回/周・9.4〜9.6秒",
        page, fontsize=16, color=INK2)
    txt(ax, X0, 1294, "* ToF ガードの減速を全区間に入れるなら ≤9.8秒。効果の数字はすべて見込みで、実車ではまだ試していない",
        page, fontsize=16, color=INK2)

    # はみ出しの確認
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t, (a, b, c, d) in checks:
        bb = t.get_window_extent(r)
        (px0, py1), (px1, py0) = ax.transData.transform([(a, b), (c, d)])
        if bb.x0 < px0 + 4 or bb.x1 > px1 - 4 or bb.y0 < py0 + 2 or bb.y1 > py1 - 2:
            bad += 1
            print("WARN", repr(t.get_text()), [round(v) for v in (bb.x0, bb.x1, bb.y0, bb.y1)],
                  [round(v) for v in (px0, px1, py0, py1)])
    fig.savefig(OUT, dpi=160)
    print("wrote", os.path.abspath(OUT), "warnings:", bad)


if __name__ == "__main__":
    main()
