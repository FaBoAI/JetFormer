#!/usr/bin/env python3
"""反証 V-replay の図(2026-10-25)。rr_ceiling.json・rr_gainup.json・rr_cache の No.9 から。読むだけ。"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_cache  # noqa: E402
import rr_ceiling as RC  # noqa: E402
import rr_common as C  # noqa: E402

FIG = os.path.join(C.OUT_DIR, 'fig_verify_replay')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
RCJ = json.load(open(os.path.join(C.HERE, 'rr_ceiling.json')))
GU = json.load(open(os.path.join(C.HERE, 'rr_gainup.json')))


def fig_runs():
    runs = RCJ['open_loop']['runs']
    order = ['10/23 その前', '10/23 公式', '10/23 速さ上げ', '10/24 1回目', '10/24 2回目', '10/24 3回目']
    runs = sorted(runs, key=lambda r: (order.index(r['session']), int(r['no'])))
    lab = ['%s No.%s' % (r['session'].replace('10/23 ', '23 ').replace('10/24 ', '24 '), r['no']) for r in runs]
    over = [100 * r['impl']['over'] for r in runs]
    chg = [100 * r['impl']['changed'] for r in runs]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(runs))
    ax.bar(x - 0.2, over, 0.38, color=S1, label='天井を超えた(記録の速さ > C)')
    ax.bar(x + 0.2, chg, 0.38, color=S2, label='天井が出力を変えた(開ループ・実装)')
    ax.axhline(5, color=MUTED, lw=1, ls='--')
    ax.text(len(runs) - 0.5, 5.8, '門 G-d 5%', color=INK2, ha='right', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=70, fontsize=7.5)
    ax.set_ylabel('AI のフレームの割合 [%]')
    ax.set_title('走行ごとの天井の掛かり方(10/23・10/24 の AI 39 本、開ループの影)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'vr1_runs_open_loop.png'), dpi=150)
    plt.close(fig)


def fig_no9_trace():
    D = rr_cache.load()['runs']
    d = D[RC.NO9]
    lp = sorted(d['laps'], key=lambda x: x['t_start'])[0]
    ij = RC.idx(d, lp['t_start'], lp['t_start'] + lp['lap_s'])
    o = RC.virtual(d, C.P('impl'), ij[0], ij[1], RC.run_b(d), dead=0.05, keep=True)
    ol = C.open_loop(d, C.P('impl'))
    t = d['t'][ij[0]:ij[1]] - lp['t_start']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    a1.plot(t, RC.smooth_v(d)[ij[0]:ij[1]], color=S1, lw=2, label='記録(ならした速さ)')
    a1.plot(t, o['vp'], color=S2, lw=2, label='天井あり(仮想の閉ループ)')
    a1.plot(t, ol['cap'][ij[0]:ij[1]], color=MUTED, lw=1.2, ls='--', label='天井 C')
    locd = d['locd'][ij[0]:ij[1]]
    prev = None
    for k in range(len(t)):
        name = C.LOC_DETAIL[locd[k]]
        if name != prev:
            a1.axvline(t[k], color='#e6e5e0', lw=0.8)
            a1.text(t[k] + 0.05, 1.05, name, color=INK2, fontsize=8)
            prev = name
    a1.set_ylabel('速さ [m/s]')
    a1.set_ylim(1.0, 4.8)
    a1.legend(frameon=False, loc='lower left', ncol=3)
    a1.set_title('No.9 の 1 周目(%.2f 秒): 天井で弱める場所と失う時間(この周 %.3f 秒)' % (lp['lap_s'], o['loss'].sum()),
                 color=INK, fontsize=11, loc='left')
    a2.plot(t, o['f_in'], color=S1, lw=1.5, label='モデルの前進の量(記録)')
    a2.plot(t, o['f_out'], color=S2, lw=1.5, label='天井のあと')
    a2.set_ylabel('前進の量 [µs]')
    a2.set_xlabel('周の始まり(R1 の起点)からの時間 [秒]')
    a2.legend(frameon=False, loc='lower left', ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'vr2_no9_lap1.png'), dpi=150)
    plt.close(fig)


def fig_sw():
    rows = RCJ['sw_passes']['dead0.05']
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for hit, col, mk, name in ((False, S1, 'o', '当たらなかった通過'), (True, S2, 'D', '出っ張りに当たった通過')):
        R = [r for r in rows if r['hit'] == hit]
        ax.scatter([r['vmax_rec'] for r in R], [r['vmax_v'] for r in R], s=34, color=col, marker=mk, edgecolor='#fcfcfb',
                   linewidth=1.5, label=name, zorder=3)
    lo, hi = 2.6, 5.8
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=':')
    for b, txt in ((3.5, '0/47'), (4.0, '2/34'), (4.5, '2/11')):
        ax.axhline(b, color='#e6e5e0', lw=1)
    for y, txt in ((3.2, '当たり 0/47'), (3.75, '2/34'), (4.25, '2/11'), (4.8, '4/6')):
        ax.text(hi - 0.05, y, txt, color=INK2, ha='right', fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('SW の中の最高の速さ: 記録 [m/s]')
    ax.set_ylabel('同じ通過の天井あり(仮想) [m/s]')
    ax.set_title('10/24 の SW の通過 97 回(帯は analysis_collisions 4 章の当たる率)', color=INK, fontsize=10, loc='left')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'vr3_sw_passes.png'), dpi=150)
    plt.close(fig)


def fig_gainup():
    R = [g for g in GU if g['run'] == 'No.9' and g['kv'] == 0.0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ceil, col, name in ((False, S1, '天井なし'), (True, S2, '天井あり(4.5 / 左 4.0)')):
        x = [g['gain'] for g in R if g['ceil'] == ceil]
        a1.plot(x, [g['v90_mean'] for g in R if g['ceil'] == ceil], color=col, lw=2, marker='o', ms=6, label=name)
        a2.plot(x, [g['stand3_best'] for g in R if g['ceil'] == ceil], color=col, lw=2, marker='o', ms=6, label=name)
    a1.axhline(4.20, color=MUTED, ls='--', lw=1)
    a1.text(0.905, 4.215, '自動合わせの目標 p90 4.20', color=INK2, fontsize=8)
    a1.set_xlabel('速さのゲイン(No.9 は 0.90)')
    a1.set_ylabel('周の速さ p90 の平均 [m/s]')
    a1.set_title('天井の下では p90 は 4.0 で頭打ち', color=INK, fontsize=10.5, loc='left')
    a1.legend(frameon=False, loc='upper left')
    a2.axhline(25.0, color=MUTED, ls='--', lw=1)
    a2.text(0.905, 25.05, '目標 25 秒', color=INK2, fontsize=8)
    a2.set_xlabel('速さのゲイン(No.9 は 0.90)')
    a2.set_ylabel('止まってからの 3 周(最速の窓)[秒]')
    a2.set_title('3 周の見込み(記録 + 0.63 秒。線形の仮想・モデルの反応なし)', color=INK, fontsize=10.5, loc='left')
    a2.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'vr4_no9_gain_up.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    fig_runs()
    fig_no9_trace()
    fig_sw()
    fig_gainup()
    print(sorted(os.listdir(FIG)))
