"""反証(大 1・大 2)の直しの図。2160×1350(ほかの図と同じ大きさ・色)。
  fig/fix_r1_bump.png   R1 の入口の上下の衝撃(3 軸)と進入の速さ + カメラ 2 コマ(人の写っていないコマだけ)
  fig/qp1_patterns.png  3 つのパターンの見込み(P1 を「車を直せたら / 直せなければ」に分け、P2 を仮説として書き直し)
動かし方: /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I figures_fix.py
先に r1_bump.py を動かして fix_data/r1_bump.json を作る。"""
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, '..', 'verify'))
import rawload  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report')
FIG = OUT / 'fig'
FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FP)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FP).get_name()
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
GRAY = '#9a9993'
plt.rcParams.update({'font.size': 17, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False})
SIZE = dict(figsize=(16, 10), dpi=135)
TOP, GOAL, PASS = 23.0, 25.0, 27.0

# カメラのコマ: 人が写っていないことを目で確かめたコマだけ(R1 の起点の 0.30 秒あと)
CAM = [('20261024-162033', 39.37, 'No.9(B 0.90・2 回目)\nR1 の起点の 0.30 秒あと・3 軸 4.9G'),
       ('20261024-183437', 30.11, 'No.12(B 0.90・3 回目)\nR1 の起点の 0.30 秒あと・3 軸 2.3G')]


def frame(log, t):
    r = rawload.load(log)
    k = int(np.argmin(np.abs(r['c_t'] - t)))
    return Image.open(r['c_files'][k]).convert('RGB'), float(r['c_t'][k])


def fig_bump():
    d = json.load(open(OUT / 'fix_data' / 'r1_bump.json'))
    P = [p for p in d['passes'] if p['entry_erpm']]
    no9 = [p for p in P if p['log'] == '20261024-162033']
    s3 = [p for p in P if p['session'] == '10/24 3回目' and p['setting'] == 'B 速さ0.90 ステア1.00']
    oth = [p for p in P if p not in no9 and p not in s3]
    fig = plt.figure(**SIZE)
    ax = fig.add_axes([0.07, 0.2, 0.52, 0.62])
    for grp, c, mk, sz, lab in ((oth, GRAY, '^', 70, 'ほかの AI の通過(10/23・10/24)'),
                                (s3, C1, 's', 110, '3 回目 B 0.90(No.11・12・18)'),
                                (no9, C2, 'o', 150, 'No.9(B 0.90・2 回目)')):
        ax.scatter([p['entry_erpm'] for p in grp], [p['a3_max'] for p in grp], s=sz, marker=mk, color=c,
                   edgecolor=SURF, linewidth=1.5, zorder=3, label='%s %d' % (lab, len(grp)))
    ax.axhline(2.5, color=INK2, lw=1.4, ls='--', zorder=1)
    ax.text(3480, 2.58, '2.5G(3 軸の決まり cmap)', fontsize=14, color=INK2, va='bottom')
    for b in d['bands']:
        lo, hi = b['lo'], min(b['hi'], 7200)
        lo = max(lo, 3500)
        ax.plot([lo + 60, hi - 60], [b['a3_median']] * 2, color=INK, lw=2.2, zorder=4)
        ax.text((lo + hi) / 2, 6.95, '%d/%d\n(%d%%)' % (b['bump'], b['passes'], round(100 * b['bump'] / b['passes'])),
                ha='center', va='top', fontsize=13, color=INK2)
    for x in (5000, 5500, 6000):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    ax.set_xlim(3500, 7200)
    ax.set_ylim(0.8, 7.2)
    ax.set_xlabel('R1 の進入の速さ(eRPM)')
    ax.set_ylabel('R1 の起点から 1.2 秒の 3 軸 |‖a‖−1G| の最大(G)')
    ax.grid(axis='y', color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=13)
    for i, (log, t, cap) in enumerate(CAM):
        im, tt = frame(log, t)
        a = fig.add_axes([0.655, 0.43 - i * 0.39, 0.30, 0.30])
        a.imshow(im.resize((448, 448)))
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(cap, fontsize=13, color=INK, loc='left')
    fig.suptitle('R1 の入口の上下の衝撃: 進入が速いほど大きい', x=0.03, ha='left', fontsize=24, color=INK, y=0.965)
    fig.text(0.03, 0.905, '121 通過(10/23 公式 + 10/24 の AI、偽の起点を除く)。No.9 は 12 通過中 11 が 2.5G 以上(中央 3.9G・最大 6.4G)。'
             '起点の 0.35〜0.48 秒あと', fontsize=14, color=INK2)
    fig.text(0.03, 0.872, '上の数 = 進入の帯ごとの「2.5G 以上 / 通過」。黒い線 = 帯ごとの中央値', fontsize=14, color=INK2)
    fig.text(0.655, 0.83, 'カメラ: どちらも R1 の入口の黒い板の縁を越える\n(224×224 を拡大。人の写っていないコマだけ)',
             fontsize=12, color=INK2, va='top')
    p = FIG / 'fix_r1_bump.png'
    fig.savefig(p)
    print(p)


def vline(ax, x, label, ls):
    ax.axvline(x, color=INK2, lw=1.4, ls=ls, zorder=1)
    ax.text(x, 1.005, label, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=15, color=INK)


def fig_patterns():
    """plan/figures_plan.py の fig1 を、反証の大 2 に合わせて書き直したもの。"""
    d = json.load(open(OUT / 'tt_data.json'))
    no9 = sorted(w['total_s'] for w in d['windows'] if w['log'] == '20261024-162033')
    s3 = sorted(w['total_s'] for w in d['windows']
                if w['session'] == '10/24 3回目' and w['setting'] == 'B 速さ0.90 ステア1.00')
    # (名前, 走りながら(lo, hi, 中心), 止まってから(lo, hi, 中心), 仮説か)
    est = [('P1a No.9 の再現\n車を直せたら', (25.3, 26.7, 25.75), (25.9, 27.3, 26.4), False),
           ('P1b No.9 の再現\n車を直せなければ', (27.4, 29.0, 28.22), (28.0, 29.6, 28.85), False),
           ('P2 速さの天井と自動合わせ\n(仮説・了承が要る)', (24.7, 25.5, None), (25.3, 26.1, None), True),
           ('P3 B-FAST\n(速い周を手本に学習)', (24.9, 27.5, None), None, True)]
    fig, ax = plt.subplots(**SIZE)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.84, bottom=0.17)
    rows = ['10/24 No.9(実績)\nB 0.90・2 回目', '10/24 3 回目(実績)\nB 0.90・修理のあと'] + [e[0] for e in est]
    n = len(rows)
    ys = {r: n - 1 - i for i, r in enumerate(rows)}
    for vals, r in ((no9, rows[0]), (s3, rows[1])):
        y = ys[r]
        for j, v in enumerate(vals):
            dy = (j - (len(vals) - 1) / 2) * 0.04
            ax.scatter(v, y + dy, s=120, color=C1, edgecolor=SURF, linewidth=1.5, zorder=3)
        m = st.median(vals)
        ax.plot([m, m], [y - 0.3, y + 0.3], color=INK, lw=2.2, zorder=4)
        txt = '中央値 %.2f 秒・%d 窓・27 秒以下 %d' % (m, len(vals), sum(v <= PASS for v in vals))
        ax.text(max(vals) + 0.12, y, txt, va='center', fontsize=14, color=INK2)
    for name, fly, stand, hyp in est:
        y = ys[name]
        for rng, c, dy in ((fly, C1, 0.14), (stand, C2, -0.14)):
            if rng is None:
                continue
            lo, hi, mid = rng
            ax.add_patch(FancyBboxPatch((lo, y + dy - 0.1), hi - lo, 0.2, boxstyle='round,pad=0,rounding_size=0.05',
                                        facecolor=c, alpha=0.30 if hyp else 0.55, edgecolor=c, lw=1.5,
                                        hatch='//' if hyp else None, zorder=2))
            if mid is not None:
                ax.plot([mid, mid], [y + dy - 0.13, y + dy + 0.13], color=INK, lw=2.2, zorder=4)
            ax.text(hi + 0.12, y + dy, '%.1f〜%.1f' % (lo, hi) + ('' if mid is None else '(中心 %.2f)' % mid),
                    va='center', fontsize=14, color=INK2)
    ax.text(29.75, ys[est[0][0]] - 0.5, '↑ どちらになるかは 11/3 の最初の 1 本(左右の比・s0)で分かる',
            fontsize=13, color=INK2, ha='right', va='center')
    vline(ax, TOP, 'トップ 約 23 秒', ':')
    vline(ax, GOAL, '目標 25 秒', '-')
    vline(ax, PASS, '予選の目安 27 秒', '--')
    ax.set_yticks([ys[r] for r in rows])
    ax.set_yticklabels(rows, fontsize=15, color=INK)
    ax.set_xlim(22.5, 32.0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('3 周の合計(秒)')
    ax.grid(axis='x', color=GRID, lw=1)
    ax.set_axisbelow(True)
    fig.suptitle('3 周の合計: 10/24 の実績と、3 つのパターンの見込み(直した版)', x=0.03, ha='left', fontsize=24, color=INK,
                 y=0.965)
    fig.text(0.03, 0.905, '点 = 10/24 の 3 周の窓(走りながら)。帯 = 見込み(粗い)。P1 は No.9 の 1 本と 3 回目の実績から。'
             'P2・P3 は閉ループの根拠が無い仮説。止まってから = +0.63 秒', fontsize=14, color=INK2)
    h = [Line2D([], [], marker='o', ls='', color=C1, markersize=11, label='実績(走りながら)'),
         Patch(facecolor=C1, alpha=0.55, label='見込み 走りながら'),
         Patch(facecolor=C2, alpha=0.55, label='見込み 止まってから'),
         Patch(facecolor=C1, alpha=0.3, hatch='//', label='仮説(実車の根拠なし)'),
         Line2D([], [], color=INK, lw=2.2, label='中央値・中心')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.42, -0.1), ncol=5, frameon=False, fontsize=14)
    p = FIG / 'qp1_patterns.png'
    fig.savefig(p)
    print(p)


if __name__ == '__main__':
    fig_bump()
    fig_patterns()
