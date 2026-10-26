"""解析 A の図(2160×1350 PNG)。analyze_tt.py の tt_data.json を読む。

    python3 -I figures_tt.py
    → /home/jetson/jetformer_reports/bench_20261024_report/fig/tt1_3lap_distribution.png
                                                             tt2_no9_laps.png
                                                             tt3_sections_no9_vs_s3.png
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report')
FIG = OUT / 'fig'
FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FP)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FP).get_name()
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 17, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False})
SIZE = dict(figsize=(16, 10), dpi=135)          # 2160×1350
KIND = {'きれい': (C1, 'o', 'きれい(3 周とも衝突なし)'),
        '軽い接触だけ': (C2, '^', '軽い接触だけ(2.5〜4G)'),
        '強い衝突あり': (C3, 'X', '強い衝突あり(4G 以上)')}
TOP, GOAL, PASS = 23.0, 25.0, 27.0


def vline(ax, x, label, ls):
    ax.axvline(x, color=INK2, lw=1.4, ls=ls, zorder=1)
    ax.text(x, 1.005, label, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=15, color=INK)


def fig1(d):
    order = ['10/23 公式', '10/24 1回目', '10/24 2回目', '10/24 3回目']
    rows = {}
    for w in d['windows']:
        rows.setdefault((w['session'], w['setting']), []).append(w)
    keys = sorted(rows, key=lambda k: (order.index(k[0]), k[1]))
    fig, ax = plt.subplots(**SIZE)
    fig.subplots_adjust(left=0.29, right=0.77, top=0.86, bottom=0.12)
    ylabels = []
    for i, k in enumerate(keys):
        y = len(keys) - 1 - i
        ws = rows[k]
        n = len(ws)
        for j, w in enumerate(sorted(ws, key=lambda w: w['total_s'])):
            c, m, _ = KIND[w['kind']]
            dy = (j - (n - 1) / 2) * min(0.2, 0.62 / max(n - 1, 1))
            ax.scatter(w['total_s'], y + dy, s=150, color=c, marker=m, edgecolor=SURF, linewidth=1.5, zorder=3)
        tot = [w['total_s'] for w in ws]
        med = st.median(tot)
        ax.plot([med, med], [y - 0.32, y + 0.32], color=INK, lw=2.2, zorder=4)
        le27 = sum(t <= PASS for t in tot)
        ax.text(1.015, y, '%d 窓  最小 %.2f  中央値 %.2f\n27 秒以下 %d / %d' % (n, min(tot), med, le27, n),
                transform=ax.get_yaxis_transform(), va='center', fontsize=14, color=INK2, linespacing=1.3)
        nos = sorted({w['no'] for w in ws}, key=int)
        ylabels.append((y, '%s  %s(No.%s)' % (k[0].replace('10/24 ', ''), k[1].replace(' ステア1.00', ''), '・'.join(nos))))
    ax.set_yticks([y for y, _ in ylabels])
    ax.set_yticklabels([s for _, s in ylabels], fontsize=15, color=INK)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(22.5, 31.2)
    ax.grid(axis='x', color=GRID, lw=1)
    vline(ax, TOP, '23 秒(予選トップ)', '--')
    vline(ax, GOAL, '25 秒(目標の軸)', '-')
    vline(ax, PASS, '27 秒(予選の目安)', ':')
    ax.set_xlabel('続けて走った 3 周の合計 [秒](走りながらの計測・1 周ずつずらした窓)', fontsize=16)
    hs = [Line2D([], [], color=c, marker=m, ls='', markersize=13, label=l) for c, m, l in KIND.values()]
    hs.append(Line2D([], [], color=INK, lw=2.2, label='中央値'))
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, 1.06), ncol=4, frameon=False, fontsize=14)
    fig.suptitle('3 周の合計: 27 秒以下が続いたのは No.9(B・速さ 0.90・満充電の 2 回目)だけ。3 回目は 23 窓中 1 窓', x=0.02, ha='left',
                 fontsize=21, color=INK, y=0.985)
    fig.text(0.02, 0.012, '出典: tt_data.json(scorecard.json の laps。偽の起点で切れた 5 周を除く)。'
             '窓は重なるので独立ではない。止まってからのスタートは +約 0.6 秒(本文)。', fontsize=13, color=INK2)
    p = FIG / 'tt1_3lap_distribution.png'
    fig.savefig(p)
    plt.close(fig)
    return p


def fig2(d):
    laps = [x for x in d['laps'] if x['log'] == '20261024-162033' and not x['false_end']]
    fig, (ax, bx) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.25, 1]}, **SIZE)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.86, bottom=0.11, hspace=0.42)
    xs = list(range(1, len(laps) + 1))
    ys = [x['lap_s'] for x in laps]
    ax.plot(xs, ys, color=C1, lw=2, zorder=2)
    for x, lp in zip(xs, laps):
        clean = lp['kind'] == 'きれい'
        ax.scatter(x, lp['lap_s'], s=170, marker='o' if clean else '^', color=C1 if clean else C2,
                   edgecolor=SURF, linewidth=2, zorder=3)
        lab = '%.2f' % lp['lap_s']
        if not clean:
            lab += '\n接触 %.2fG' % lp['gmax']
        ax.annotate(lab, (x, lp['lap_s']), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=13, color=INK)
    for yv, txt, ls in ((TOP / 3, '7.67 秒 = 23 秒のペース', '--'), (GOAL / 3, '8.33 秒 = 25 秒のペース', '-'),
                        (PASS / 3, '9.00 秒 = 27 秒のペース', ':')):
        ax.axhline(yv, color=INK2, lw=1.3, ls=ls, zorder=1)
        ax.text(1.005, yv, txt, transform=ax.get_yaxis_transform(), va='center', fontsize=13, color=INK2)
    # いちばん速い 3 周の窓
    ws = [w for w in d['windows'] if w['log'] == '20261024-162033']
    best = min(ws, key=lambda w: w['total_s'])
    bestc = min([w for w in ws if w['kind'] == 'きれい'], key=lambda w: w['total_s'])
    for w, c, txt in ((best, C2, '最速の 3 周 %.2f 秒(軽い接触あり)'), (bestc, C1, 'きれいな 3 周の最速 %.2f 秒')):
        k0 = [lp['t_start'] for lp in laps].index(w['t_start']) + 1
        ax.axvspan(k0 - 0.4, k0 + 2.4, color=c, alpha=0.08, zorder=0)
        ax.text(k0 + 1, 7.42, txt % w['total_s'], ha='center', fontsize=13, color=INK)
    ax.set_ylim(7.3, 9.35)
    ax.set_xlim(0.4, len(laps) + 0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(['%d' % x for x in xs])
    ax.set_ylabel('1 周 [秒]')
    ax.grid(axis='y', color=GRID, lw=1)
    hs = [Line2D([], [], color=C1, marker='o', ls='', markersize=12, label='きれいな周(9 周)'),
          Line2D([], [], color=C2, marker='^', ls='', markersize=12, label='軽い接触のあった周(2 周)')]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=14)
    # 下: 区間(場所で切った 3 つ)の時間
    grp = [('R1+GM', 'R1+GM(右ヘアピン+緑マット)', C1), ('DE+SW', 'DE+SW(バックストレート+奥のスイーパー)', C2),
           ('HS+LH', 'HS+LH(ホームストレート+左ヘアピン)', C3)]
    mk = {'R1+GM': 'o', 'DE+SW': 's', 'HS+LH': 'D'}
    for g, name, c in grp:
        v = [lp['groups_s'][g] for lp in laps]
        bx.plot(xs, v, color=c, lw=2, marker=mk[g], markersize=9, markeredgecolor=SURF)
        bx.text(1.01, v[-1], '%s\n最小 %.2f・最大 %.2f' % (name.replace('(', '\n('), min(v), max(v)), va='center',
                fontsize=12.5, color=INK, clip_on=False, transform=bx.get_yaxis_transform(), linespacing=1.2)
    bx.set_xlim(0.4, len(laps) + 0.6)
    bx.set_xticks(xs)
    bx.set_xlabel('No.9 の周の順番(16:20、B・速さ 0.90・ステア 1.00、AI で続けて 11 周)')
    bx.set_ylabel('区間の時間 [秒]')
    bx.set_ylim(1.9, 4.3)
    bx.grid(axis='y', color=GRID, lw=1)
    fig.suptitle('No.9: 11 周のうち 9 周がきれい。周ごとのばらつきは DE+SW(奥のスイーパー)がいちばん大きい',
                 x=0.02, ha='left', fontsize=20, color=INK, y=0.985)
    fig.text(0.02, 0.008, '出典: tt_data.json(scorecard.json の laps・sections_s、20261024-162033)。'
             'R1・DE の区間は定義で時間が固定に近いので、場所で切れる 3 つにまとめた。', fontsize=12.5, color=INK2)
    p = FIG / 'tt2_no9_laps.png'
    fig.savefig(p)
    plt.close(fig)
    return p


def fig3(d):
    laps = d['laps']
    a = [x for x in laps if x['log'] == '20261024-162033' and not x['false_end']]
    b = [x for x in laps if x['session'] == '10/24 3回目' and x['model'] == 'B' and x['tg'] == 0.9 and x['sg'] == 1.0
         and not x['false_end']]
    groups = [('R1+GM', 'R1+GM\n右ヘアピン\n+緑マット'), ('DE+SW', 'DE+SW\nバックストレート\n+奥のスイーパー'),
              ('HS+LH', 'HS+LH\nホームストレート\n+左ヘアピン')]
    fig, (ax, bx) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.15, 1]}, **SIZE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.19, wspace=0.28)
    w = 0.36
    for i, (g, lab) in enumerate(groups):
        va = st.mean([x['groups_s'][g] for x in a])
        vb = st.mean([x['groups_s'][g] for x in b])
        ax.bar(i - w / 2 - 0.01, va, w, color=C1, zorder=2)
        ax.bar(i + w / 2 + 0.01, vb, w, color=C2, zorder=2)
        ax.text(i - w / 2, va + 0.04, '%.2f' % va, ha='center', fontsize=14, color=INK)
        ax.text(i + w / 2, vb + 0.04, '%.2f' % vb, ha='center', fontsize=14, color=INK)
        ax.text(i, max(va, vb) + 0.32, '+%.2f 秒' % (vb - va), ha='center', fontsize=17, color=INK, weight='bold')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lab for _, lab in groups], fontsize=14, color=INK)
    ax.set_ylabel('1 周のなかの時間(平均)[秒]')
    ax.set_ylim(0, 4.6)
    ax.grid(axis='y', color=GRID, lw=1)
    la, lb = st.mean([x['lap_s'] for x in a]), st.mean([x['lap_s'] for x in b])
    ax.set_title('区間の時間: 1 周 %.2f → %.2f 秒(+%.2f)' % (la, lb, lb - la), fontsize=17, color=INK, loc='left')
    # 右: 区間の速さ
    sp = [('R1+GM', 'R1+GM'), ('DE+SW', 'DE+SW'), ('HS+LH', 'HS+LH'), ('p90', '周の速い方\n90% 点')]
    for i, (g, lab) in enumerate(sp):
        va = st.median([x['speed_mps'][g] for x in a])
        vb = st.median([x['speed_mps'][g] for x in b])
        bx.bar(i - w / 2 - 0.01, va, w, color=C1, zorder=2)
        bx.bar(i + w / 2 + 0.01, vb, w, color=C2, zorder=2)
        bx.text(i - w / 2, va + 0.04, '%.2f' % va, ha='center', fontsize=13.5, color=INK)
        bx.text(i + w / 2, vb + 0.04, '%.2f' % vb, ha='center', fontsize=13.5, color=INK)
        bx.text(i, max(va, vb) + 0.3, '%+.0f%%' % ((vb / va - 1) * 100), ha='center', fontsize=16, color=INK, weight='bold')
    bx.set_xticks(range(len(sp)))
    bx.set_xticklabels([lab for _, lab in sp], fontsize=14, color=INK)
    bx.set_ylabel('車輪の速さ(中央値)[m/s]')
    bx.set_ylim(0, 4.8)
    bx.grid(axis='y', color=GRID, lw=1)
    bx.set_title('区間の速さ', fontsize=17, color=INK, loc='left')
    hs = [plt.Rectangle((0, 0), 1, 1, color=C1, label='No.9(2 回目 16:20、%d 周)' % len(a)),
          plt.Rectangle((0, 0), 1, 1, color=C2, label='3 回目の B 0.90(No.%s、計 %d 周)' % (
              '・'.join(sorted({x['no'] for x in b}, key=int)), len(b)))]
    fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.07, 0.905), ncol=2, frameon=False, fontsize=15)
    fig.suptitle('同じ B・速さ 0.90 でも 3 回目は 1 周 +0.8 秒: 3 つの区間すべてで遅く、ホームストレート+左ヘアピンがいちばん大きい',
                 x=0.02, ha='left', fontsize=19, color=INK, y=0.985)
    fig.text(0.02, 0.03, '出典: tt_data.json(scorecard.json の sections_s・labels の speed_mps)。偽の起点の周を除く全部の周'
             '(接触のあった周も入る)。速さは車輪の回転から(滑りを含む)。', fontsize=12.5, color=INK2)
    p = FIG / 'tt3_sections_no9_vs_s3.png'
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    d = json.loads((OUT / 'tt_data.json').read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3):
        print(f(d))


if __name__ == '__main__':
    main()
