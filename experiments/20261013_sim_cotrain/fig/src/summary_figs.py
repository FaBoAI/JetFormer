"""1 分の動画用のまとめの図 2 枚(2160x1350)を作る。CPU だけ。

- summary_method.png  … やり方の全体(なぜ・データ・2 つの段・比べ方)
- summary_results.png … 結果の全体(主指標・副指標・走行の推奨)

数字は ../../evaluation.json から読む(手で写さない)。体裁は
~/jetformer/experiments/20261011_day_summary/fig/src/common.py に合わせる。
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/jetson/jetformer/experiments/20261011_day_summary/fig/src")
from common import (AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, RED, SURF,  # noqa: E402
                    VIOLET, new_fig, tint)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REP, "fig")
JF = "/home/jetson/jetformer"
REAL_IMG = JF + "/data/raw/20261010-152707/20261010-152707-0002/CAM_FRONT/1791613663805526.jpg"
SIM_IMG = (JF + "/data/raw/ds-jetformer-simulator-id8-20min_20261012-150956_b55616cf/"
           "ds-jetformer-simulator-id8-20min_20261012-150956_b55616cf-0003/CAM_FRONT/1791816865563696.jpg")

UW, UH = 1600.0, 1000.0  # 図の座標(左上が 0,0)
GRAY = "#52514e"


def canvas():
    """図全体を 1600x1000 の座標にした軸を返す。"""
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, UW)
    ax.set_ylim(UH, 0)
    ax.axis("off")
    return fig, ax


def sub_axes(fig, x0, y0, x1, y1):
    """1600x1000 の座標の箱に軸を置く。"""
    return fig.add_axes([x0 / UW, 1 - y1 / UH, (x1 - x0) / UW, (y1 - y0) / UH])


def box(ax, x0, y0, x1, y1, ec, fc, lw=2.5, r=14):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, zorder=1))


def arrow(ax, p0, p1, color=GRAY, lw=3.5, ms=28):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, zorder=3))


def t(ax, x, y, s, size=17, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=5, **kw)


def thumb(fig, path, x0, y0, side):
    a = sub_axes(fig, x0, y0, x0 + side, y0 + side * UW / UH * (1350 / 2160))
    a.imshow(np.asarray(Image.open(path).convert("RGB")))
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
    return a


# ---------------------------------------------------------------- 図 1: やり方

def fig_method(ev):
    fig, ax = canvas()
    t(ax, 60, 52, "シミュレーター+実機を混ぜて学習 → 実機で仕上げる", size=30, weight="bold")
    t(ax, 60, 108, "TransFuser軽量型(JetFormer v0)・2026-10-13・判定の規則は学習の前に固定",
      size=18, color=INK2)

    # なぜ
    box(ax, 50, 140, 1555, 222, ec=tint(RED, 0.5), fc=tint(RED, 0.08))
    ax.add_patch(FancyBboxPatch((70, 157), 110, 48, boxstyle="round,pad=0,rounding_size=10",
                                ec="none", fc=RED, zorder=2))
    t(ax, 125, 181, "なぜ", size=20, color="white", weight="bold", ha="center")
    t(ax, 205, 181, "実機だけだと 学習の誤差は 14〜18 µs、検証は 49 µs で止まる → データ不足?", size=18)

    # 列の見出し
    t(ax, 55, 268, "データ", size=21, weight="bold")
    t(ax, 640, 268, "2 つの段", size=21, weight="bold")
    t(ax, 1160, 268, "比べ方", size=21, weight="bold")
    for x0, x1 in ((55, 590), (640, 1110), (1160, 1555)):
        ax.plot([x0, x1], [292, 292], color=INK, lw=2.5, zorder=2)

    # データの 2 枚
    cards = [
        (310, BLUE, REAL_IMG, "実機", ["人の運転 52 シーン", "ジャンル 4 高速・5 低速", "学習 41 / 検証 11 シーン"],
         "速さ 中央 2.5 m/s・画像はぼけて灰色"),
        (610, ORANGE, SIM_IMG, "シミュレーター", ["60 シーン × 20 秒", "ジャンル 8・合成の先生", "同じコース・ToF は似る"],
         "速さ 中央 2.9 m/s・画像はくっきり"),
    ]
    for y0, col, img, name, lines, foot in cards:
        box(ax, 55, y0, 590, y0 + 270, ec=col, fc=tint(col, 0.07))
        thumb(fig, img, 72, y0 + 18, 170)
        t(ax, 262, y0 + 42, name, size=22, color=col, weight="bold")
        for i, s in enumerate(lines):
            t(ax, 262, y0 + 92 + i * 44, s, size=17)
        t(ax, 72, y0 + 236, foot, size=17, color=INK2)

    # 段 1・段 2
    box(ax, 640, 310, 1110, 590, ec=VIOLET, fc=tint(VIOLET, 0.08), lw=3)
    t(ax, 660, 350, "段 1  混ぜて学習 = C", size=22, color=VIOLET, weight="bold")
    for i, s in enumerate(["実機 1 : シミュレーター 2", "(学習 17,732 サンプル)",
                           "30 エポック・lr 3e-4", "出力の頭はジャンル別", "→ 速さの癖を頭で分ける"]):
        t(ax, 660, 395 + i * 38, s, size=17, color=INK if i != 4 else INK2)
    arrow(ax, (875, 593), (875, 622), color=VIOLET, lw=4, ms=30)
    box(ax, 640, 625, 1110, 880, ec=AQUA, fc=tint(AQUA, 0.09), lw=3)
    t(ax, 660, 665, "段 2  実機で微調整 = C+FT", size=22, color="#0f7a55", weight="bold")
    for i, s in enumerate(["C の best から始める", "実機 5,912 サンプル(A と同じ)",
                           "12 エポック・lr 1e-4", "シミュレーターは外す"]):
        t(ax, 660, 712 + i * 40, s, size=17)

    # データ → 段 1
    arrow(ax, (592, 445), (636, 430))
    arrow(ax, (592, 745), (636, 470))

    # 比べ方
    box(ax, 1160, 310, 1555, 590, ec=MUTED, fc="#f3f2ef")
    t(ax, 1180, 350, "比べる相手", size=21, weight="bold")
    for i, s in enumerate(["A: 実機だけ(10/12)", "A+FT: A を同じく微調整", "    (微調整だけの効き)",
                           "どれも 2 つの種で学習"]):
        t(ax, 1180, 398 + i * 42, s, size=17, color=INK if i != 2 else INK2)
    box(ax, 1160, 625, 1555, 880, ec=MUTED, fc="#f3f2ef")
    t(ax, 1180, 665, "測り方", size=21, weight="bold")
    for i, s in enumerate(["保留した実機 11 シーン", "4,670 フレームの L1",
                           "「良い」= 両方の種で A より", "小さく、平均 0.63 µs 超"]):
        t(ax, 1180, 712 + i * 42, s, size=17)
    arrow(ax, (1112, 752), (1156, 752))

    t(ax, 55, 940, "µs = PWM のパルス幅の差。A・モデル B は 10/12 の学習結果をそのまま使い、10/12 と同じフレーム・道具で測った",
      size=16, color=INK2)
    path = os.path.join(OUT, "summary_method.png")
    fig.savefig(path, dpi=160)
    return path


# ---------------------------------------------------------------- 図 2: 結果

def fig_results(ev):
    fig, ax = canvas()
    t(ax, 60, 52, "結果: 実機の精度は上がらなかった(A と同等)", size=30, weight="bold")
    t(ax, 60, 108, "保留した実機の人の走行 4,670 フレームとの誤差 L1・2 つの種(小さいほど良い)",
      size=18, color=INK2)

    L = ev["primary"]["L_us"]["best"]
    mean = ev["primary"]["arm_mean_us"]
    thr = ev["primary"]["threshold_us"]["best"]
    rows = [  # (表示名, 種の鍵, 平均の鍵, 色)
        ("A\n実機だけ", ["A_s0", "A_s1"], "A", BLUE),
        ("C\n混ぜて学習", ["C_s0", "C_s1"], "C", VIOLET),
        ("C+FT\n混ぜて→微調整", ["CFT_s0", "CFT_s1"], "C+FT", "#0f7a55"),
        ("A+FT\n対照", ["AFT_s0", "AFT_s1"], "A+FT", MUTED),
        ("モデル B\n今の走行", ["B"], None, INK),
    ]
    pa = sub_axes(fig, 290, 160, 990, 715)
    a_mean = mean["A"]["best"]
    pa.axvspan(a_mean - thr, a_mean + thr, color=tint(BLUE, 0.13), zorder=0, lw=0)
    pa.axvline(a_mean, color=BLUE, lw=2, zorder=1)
    n = len(rows)
    for i, (lab, keys, mk, col) in enumerate(rows):
        y = n - 1 - i
        xs = [L[k] for k in keys]
        m = mean[mk]["best"] if mk else xs[0]
        pa.scatter(xs, [y] * len(xs), s=260, color=col, edgecolor="white", lw=1.5, zorder=4)
        if mk and len(xs) > 1:
            pa.plot([m, m], [y - 0.28, y + 0.28], color=INK, lw=3.5, zorder=3)
        if mk == "A":
            s = f"{m:.2f}(基準)"
        elif mk:
            s = f"{m:.2f}({m - a_mean:+.2f})同等"
        else:
            s = f"{m:.2f}(参考)"
        pa.text(50.22, y, s, fontsize=18, color=INK, va="center", ha="left",
                fontweight="bold" if mk in ("C", "C+FT") else "normal")
    pa.set_yticks(range(n))
    pa.set_yticklabels([r[0] for r in rows][::-1], fontsize=17, linespacing=1.15)
    pa.set_ylim(-0.75, n - 0.45)
    pa.set_xlim(48.75, 51.3)
    pa.set_xticks([49.0, 49.5, 50.0])
    pa.tick_params(axis="x", labelsize=17)
    pa.tick_params(axis="y", length=0)
    pa.grid(axis="x", color=GRID, lw=1.2, zorder=0)
    pa.set_xlabel("L1 [µs](● = 種、縦の棒 = 平均、best.pth)", fontsize=17, color=INK2)
    pa.text(48.79, -0.6, f"A ± {thr:.2f} µs = 区別できない幅", fontsize=16, color=BLUE,
            ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", fc=tint(BLUE, 0.13), ec="none"))

    # 右の列: 副指標
    rob = ev["robustness"]["per_arm"]
    sim = ev["sim"]["per_arm"]
    x0, x1 = 1030, 1555
    box(ax, x0, 160, x1, 470, ec=tint(VIOLET, 0.6), fc=tint(VIOLET, 0.06))
    t(ax, x0 + 20, 196, "小さなずれには少し強くなった", size=19, weight="bold")
    ba = sub_axes(fig, x0 + 130, 228, x1 - 90, 385)
    ba.set_facecolor("none")
    items = [("A", 1.0, BLUE), ("C", rob["C"]["ratio_to_ref"], VIOLET),
             ("C+FT", rob["C+FT"]["ratio_to_ref"], "#0f7a55"), ("A+FT", rob["A+FT"]["ratio_to_ref"], MUTED)]
    for j, (nm, v, col) in enumerate(items):
        yy = len(items) - 1 - j
        ba.barh(yy, v, color=col, height=0.62)
        ba.text(v + 0.03, yy, f"{v:.2f}", va="center", fontsize=17, color=INK)
    ba.set_yticks(range(len(items)))
    ba.set_yticklabels([it[0] for it in items][::-1], fontsize=17)
    ba.set_xlim(0, 1.32)
    ba.set_xticks([])
    ba.tick_params(axis="y", length=0)
    for s in ("bottom", "left"):
        ba.spines[s].set_visible(False)
    t(ax, x0 + 20, 410, "ずれで増える誤差の比(A = 1)", size=16, color=INK2)
    t(ax, x0 + 20, 443, "増える量の差は 0.3 µs ほどと小さい", size=16, color=INK2)

    box(ax, x0, 490, x1, 690, ec=tint(ORANGE, 0.6), fc=tint(ORANGE, 0.06))
    t(ax, x0 + 20, 526, "シミュレーターの場面の L1", size=19, weight="bold")
    t(ax, x0 + 20, 574, f"A(実機だけ) {sim['A']['h5']['l1_mean_us']:.0f} µs", size=20, color=BLUE, weight="bold")
    t(ax, x0 + 20, 618, f"C {sim['C']['h8']['l1_mean_us']:.1f} µs・C+FT {sim['C+FT']['h8']['l1_mean_us']:.1f} µs",
      size=20, color="#0f7a55", weight="bold")
    t(ax, x0 + 20, 660, "微調整のあとも残る(学習に入った場面)", size=16, color=INK2)

    lat = ev["latency"]["per_run"]
    new = [lat[k]["run_numpy_median_ms"] for k in ("C_s0", "C_s1", "CFT_s0", "CFT_s1", "AFT_s0", "AFT_s1")]
    box(ax, x0, 710, x1, 805, ec=MUTED, fc="#f3f2ef")
    t(ax, x0 + 20, 740, "速さ: 6 本ともゲート合格", size=19, weight="bold")
    t(ax, x0 + 20, 780, f"推論 {min(new):.1f}〜{max(new):.1f} ms(A と同じ)", size=17)

    # 結論の帯
    box(ax, 50, 830, 1555, 905, ec=INK, fc=tint(BLUE, 0.06), lw=3)
    t(ax, 75, 868, "走行はモデル B のまま。実車の A/B/A で試す候補はなし", size=21, weight="bold")
    t(ax, 55, 948, "次の手: シミュレーターの画像を実機に近づける・場面を増やす・実機のデータを増やす",
      size=17, color=INK2)
    path = os.path.join(OUT, "summary_results.png")
    fig.savefig(path, dpi=160)
    return path


def main():
    with open(os.path.join(REP, "evaluation.json")) as f:
        ev = json.load(f)
    for p in (fig_method(ev), fig_results(ev)):
        print(p, Image.open(p).size)


if __name__ == "__main__":
    main()
