"""予選の計画の図(qp1: 3 つのパターンの 3 周の見込み、qp2: 11/3 までの日程)。2160×1350。
python3 -I figures_plan.py"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report')
FIG = OUT / 'fig'
FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FP)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FP).get_name()
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.size': 17, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False})
SIZE = dict(figsize=(16, 10), dpi=135)
TOP, GOAL, PASS = 23.0, 25.0, 27.0


def vline(ax, x, label, ls):
    ax.axvline(x, color=INK2, lw=1.4, ls=ls, zorder=1)
    ax.text(x, 1.005, label, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=15, color=INK)


def fig1():
    d = json.load(open(OUT / 'tt_data.json'))
    no9 = sorted(w['total_s'] for w in d['windows'] if w['log'] == '20261024-162033')
    s3 = sorted(w['total_s'] for w in d['windows']
                if w['session'] == '10/24 3回目' and w['setting'] == 'B 速さ0.90 ステア1.00')
    # (名前, 走りながら(lo, hi, 中心), 止まってから(lo, hi, 中心))。中心 None = 分からない
    est = [('P1 No.9 の再現\n(コードを変えない)', (25.3, 26.7, 25.75), (25.9, 27.3, 26.4)),
           ('P2 速さの天井と\n自動合わせ(了承が要る)', (24.7, 25.5, 25.1), (25.3, 26.1, 25.8)),
           ('P3 B-FAST\n(速い周を手本に学習)', (24.9, 27.5, None), None)]
    fig, ax = plt.subplots(**SIZE)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.84, bottom=0.17)
    rows = ['10/24 No.9(実績)\nB 0.90・2 回目', '10/24 3 回目(実績)\nB 0.90・修理のあと'] + [e[0] for e in est]
    n = len(rows)
    ys = {r: n - 1 - i for i, r in enumerate(rows)}
    for vals, r in ((no9, rows[0]), (s3, rows[1])):
        y = ys[r]
        for j, v in enumerate(vals):
            dy = (j - (len(vals) - 1) / 2) * 0.05
            ax.scatter(v, y + dy, s=140, color=C1, edgecolor=SURF, linewidth=1.5, zorder=3)
        m = st.median(vals)
        ax.plot([m, m], [y - 0.3, y + 0.3], color=INK, lw=2.2, zorder=4)
        txt = '中央値 %.2f 秒\n%d 窓・27 秒以下 %d' % (m, len(vals), sum(v <= PASS for v in vals))
        if max(vals) > 28:
            ax.text(min(vals) - 0.15, y, txt, va='center', ha='right', fontsize=15, color=INK2)
        else:
            ax.text(max(vals) + 0.12, y, txt, va='center', fontsize=15, color=INK2)
    for name, fly, stand in est:
        y = ys[name]
        for k, (rng, c, dy) in enumerate(((fly, C1, 0.14), (stand, C2, -0.14))):
            if rng is None:
                continue
            lo, hi, mid = rng
            ax.add_patch(FancyBboxPatch((lo, y + dy - 0.1), hi - lo, 0.2, boxstyle='round,pad=0,rounding_size=0.05',
                                        facecolor=c, alpha=0.30 if mid is None else 0.55, edgecolor=c, lw=1.5,
                                        hatch='//' if mid is None else None, zorder=2))
            if mid is not None:
                ax.plot([mid, mid], [y + dy - 0.13, y + dy + 0.13], color=INK, lw=2.2, zorder=4)
            ax.text(hi + 0.12, y + dy, '%.1f〜%.1f' % (lo, hi) + ('' if mid is None else '(中心 %.2f)' % mid),
                    va='center', fontsize=15, color=INK2)
    vline(ax, TOP, 'トップ 約 23 秒', ':')
    vline(ax, GOAL, '目標 25 秒', '-')
    vline(ax, PASS, '予選の目安 27 秒', '--')
    ax.set_yticks([ys[r] for r in rows])
    ax.set_yticklabels(rows, fontsize=16, color=INK)
    ax.set_xlim(22.5, 31.2)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('3 周の合計(秒)')
    ax.grid(axis='x', color=GRID, lw=1)
    ax.set_axisbelow(True)
    fig.suptitle('3 周の合計: 10/24 の実績と、3 つのパターンの見込み', x=0.03, ha='left', fontsize=24, color=INK, y=0.965)
    fig.text(0.03, 0.905, '点 = 10/24 の 3 周の窓(走りながら計測)。帯 = 見込み(粗い)。止まってから = 走りながら + 0.63 秒(発進の中央値)',
             fontsize=15, color=INK2)
    h = [Line2D([], [], marker='o', ls='', color=C1, markersize=11, label='実績(走りながら)'),
         Patch(facecolor=C1, alpha=0.55, label='見込み 走りながら'),
         Patch(facecolor=C2, alpha=0.55, label='見込み 止まってから'),
         Patch(facecolor=C1, alpha=0.3, hatch='//', label='見込み(幅が広い・中心が言えない)'),
         Line2D([], [], color=INK, lw=2.2, label='中央値・中心')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.42, -0.1), ncol=5, frameon=False, fontsize=14)
    p = FIG / 'qp1_patterns.png'
    fig.savefig(p)
    print(p)


def fig2():
    days = ['10/25\n日', '10/26\n月', '10/27\n火', '10/28\n水', '10/29\n木', '10/30\n金', '10/31\n土', '11/1\n日',
            '11/2\n月', '11/3\n火・祝']
    lanes = ['ユーザーの手', '机の上・集計', 'コード・試験\n(了承のあと)', 'GPU の学習', 'シミュレーター', '実車']
    col = {lanes[0]: C2, lanes[1]: C1, lanes[2]: '#6250d6', lanes[3]: C3, lanes[4]: C4, lanes[5]: '#e87ba4'}
    # (レーン, 始まりの日, 日数, 文字)
    items = [
        (0, 0, 2, '後輪・舵のリンクの点検\n左右に転がして半径を比べる'),
        (0, 2, 2, '電圧計・LiPo チェッカー\n(INA226 は決めてから)'),
        (0, 7, 1, '電池 3 本\nの段取り\n予備の部品'),
        (0, 8, 1, '再起動\ndry run\n点検'),
        (1, 0, 1, '了承の確認\n確認の道具'),
        (1, 1, 1, '周の数え方\nの直し・再集計'),
        (1, 2, 1, '天井の値を\n記録の再生で\n決める'),
        (1, 5, 1, '11/3 の並び\n画面の確認'),
        (1, 6, 1, '予備日\nレポート'),
        (1, 7, 1, '11/3 のカード'),
        (2, 2, 1, '速さの天井\n(drive.py)'),
        (2, 3, 1, '速度ゲインの\n自動合わせ'),
        (2, 4, 1, '電圧の欄\n発進の補助\npytest 一式'),
        (3, 1, 1, 'B-FAST の\nデータ・学習'),
        (3, 2, 1, '机上の評価\n(門)'),
        (3, 4, 1, 'エンジン\nの確認'),
        (4, 3, 1, '天井・自動\n合わせの動き'),
        (5, 9, 1, '25 秒の確認\n→ 上積み'),
    ]
    fig, ax = plt.subplots(**SIZE)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.83, bottom=0.05)
    nL = len(lanes)
    for i, (lane, s, w, txt) in enumerate(items):
        y = nL - 1 - lane
        c = col[lanes[lane]]
        ax.add_patch(FancyBboxPatch((s + 0.05, y - 0.38), w - 0.1, 0.76, boxstyle='round,pad=0,rounding_size=0.08',
                                    facecolor=c, alpha=0.18, edgecolor=c, lw=2, zorder=2))
        ax.text(s + w / 2, y, txt, ha='center', va='center', fontsize=12, color=INK, zorder=3, linespacing=1.2)
    ax.set_xlim(0, len(days))
    ax.set_ylim(-0.6, nL - 0.4)
    ax.set_xticks([i + 0.5 for i in range(len(days))])
    ax.set_xticklabels(days, fontsize=14, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([nL - 1 - i for i in range(nL)])
    ax.set_yticklabels(lanes, fontsize=15, color=INK)
    for x in range(1, len(days)):
        ax.axvline(x, color=GRID, lw=1, zorder=0)
    ax.axvspan(9, 10, color='#e87ba4', alpha=0.06, zorder=0)
    ax.tick_params(length=0)
    for s in ('left', 'bottom', 'top', 'right'):
        ax.spines[s].set_visible(False)
    fig.suptitle('11/3(次の実車)までの日程', x=0.02, ha='left', fontsize=24, color=INK, y=0.975)
    fig.text(0.02, 0.92, '車は 11/3 まで走らせない。走行の制御のコードは了承のあと。GPU・画面の確認は jf-heavy --lock big で 1 つずつ',
             fontsize=15, color=INK2)
    p = FIG / 'qp2_schedule.png'
    fig.savefig(p)
    print(p)


if __name__ == '__main__':
    fig1()
    fig2()
