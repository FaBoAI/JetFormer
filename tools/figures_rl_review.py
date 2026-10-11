#!/usr/bin/env python3
"""tools/rl_review.py の事後の分析(review.json)の図。日本語・大きい文字(動画でも読めるように)

    python tools/rl_review.py figures --fig-dir <dir>
"""

from pathlib import Path

import numpy as np

from figures_rl_evaluate import GRID, MUTED, TEXT, _save, _setup
from rl_evaluate import COLORS, LABELS

SHORT = {'B': 'B', 'c': '(c)', 'b': '(b)', 'a_good': '(a)良', 'a_base': '(a)元', 'n_good': '(n)良', 'n_base': '(n)元',
         'h': '(h)人だけ'}


def _name(key):
    x, ref = key.split('-')
    return '%s − %s' % (SHORT[x], SHORT[ref])


def _row(ax, y, ci, col, filled=True, dy=0.0, label_text=True):
    if not ci or ci[0] is None:
        return
    ax.plot([ci[1], ci[2]], [y + dy, y + dy], color=col, lw=3 if filled else 2, solid_capstyle='round',
            alpha=1.0 if filled else 0.75)
    ax.plot(ci[0], y + dy, 'o', ms=11 if filled else 10, color=col if filled else '#fcfcfb', mec=col, mew=2)
    if label_text:
        ax.text(max(ci[2], ci[0]), y + dy, '  %+.1f' % ci[0], fontsize=12, color=MUTED, va='center')


def fig_ckpt(rec, out):
    """モデルどうしの差を best/best(塗り)と last/last(白抜き)で。下に、同じ学習の last − best"""
    plt = _setup()
    con, noise = rec['contrasts'], rec['ckpt_noise']
    keys = [k for k in ('c-B', 'h-B', 'c-h', 'a_good-c', 'b-c', 'n_good-c', 'a_good-n_good') if k in con]
    nkeys = [n for n in ('B', 'c', 'h', 'a_good', 'b', 'n_good') if n in noise]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
    for ax, ck, nk, title, pos, neg in ((axes[0], 'r1_d_thr', 'r1_d_thr_last_minus_best', 'R1 の手前 0.4秒: スロットルの差',
                                         '遅い', '速い'),
                                        (axes[1], 'de_d_steer', 'de_d_steer_last_minus_best', '緑マット出口〜0.8秒: ステアの差',
                                         '左', '右')):
        rows = [(_name(k), con[k]) for k in keys]
        n_all = len(rows) + 1 + len(nkeys)
        y = n_all - 1
        ticks, labs = [], []
        for lab, row in rows:
            x = lab.split(' − ')[0]
            col = COLORS.get({'(c)': 'c', '(h)人だけ': 'h', '(a)良': 'a_good', '(b)': 'b', '(n)良': 'n_good'}.get(x, 'B'), MUTED)
            _row(ax, y, row.get('best/best', {}).get(ck, {}).get('ci'), col, True, 0.17)
            _row(ax, y, row.get('last/last', {}).get(ck, {}).get('ci'), col, False, -0.17)
            ticks.append(y)
            labs.append(lab)
            y -= 1
        ax.axhline(y + 0.5, color=GRID, lw=1.2)
        ax.text(0.0, y + 0.1, '同じ学習の last − best(チェックポイントだけの違い)', transform=ax.get_yaxis_transform(),
                fontsize=13, color=TEXT, weight='bold', va='bottom')
        y -= 1
        for n in nkeys:
            _row(ax, y, noise[n][nk], COLORS.get(n, MUTED), True)
            ticks.append(y)
            labs.append(SHORT[n])
            y -= 1
        ax.axvspan(-rec['training_noise_us'], rec['training_noise_us'], color='#eeeeea', zorder=0)
        ax.axvline(0, color=MUTED, lw=1.2)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labs, fontsize=14)
        ax.set_ylim(y + 0.4, n_all - 0.4)
        ax.set_title(title, loc='left')
        ax.set_xlabel('差 [µs](通過単位ブートストラップ 95%% 区間)\n← %s      %s →' % (neg, pos))
        ax.grid(axis='x', color=GRID, lw=0.8)
    fig.suptitle('チェックポイントで差はどれだけ変わるか(塗り = best/best、白抜き = last/last、灰色の帯 = ±%.0fµs)'
                 % rec['training_noise_us'], x=0.02, ha='left', fontsize=18)
    fig.tight_layout()
    _save(fig, out, 'review_ckpt.png')
    plt.close(fig)


def fig_offset(rec, out):
    """ステアの全体のずれ(場所に関係ないか)と、人の操作に対するステアの偏り"""
    plt = _setup()
    go, vl = rec['global_offset'], rec['vs_labels']
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    sets = [('all_val', '保留の全フレーム'), ('human_clean', '人のきれいな走行'), ('human_clean_genre4', '人・高速走行(4)'),
            ('human_clean_genre5', '人・低速走行(5)'), ('ai_val', 'AI のフレーム'), ('de_windows', 'DE 出口の窓'),
            ('de_windows_minus_scene_mean', 'DE 出口 − シーンの平均')]
    pairs = [('c-B', COLORS['c'], 0.22), ('h-B', COLORS['h'], 0.0), ('c-h', '#1baf7a', -0.22)]
    yy = np.arange(len(sets))[::-1]
    for (k, col, dy) in pairs:
        row = go.get('%s@best' % k)
        if not row:
            continue
        for y, (s, _) in zip(yy, sets):
            _row(ax, y, row['steer'][s], col, True, dy, label_text=False)
        ax.plot([], [], 'o-', color=col, lw=3, ms=10, label=_name(k))
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.set_yticks(yy)
    ax.set_yticklabels([lab for _, lab in sets], fontsize=14)
    ax.set_xlabel('出力のステアの差 [µs](+ = 左、− = 右)')
    ax.set_title('微調整で、ステアは場所に関係なく右へずれた(best.pth)', loc='left', fontsize=17)
    ax.legend(loc='lower left', frameon=False, fontsize=13)  # 上から (c)−B・(h)−B・(c)−(h)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax = axes[1]
    names = [n for n in ('B', 'h', 'c', 'b', 'a_good', 'n_good') if '%s@best' % n in vl['human_clean']]
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        _row(ax, yi, vl['human_clean']['%s@best' % n]['steer_bias_us'], COLORS.get(n, MUTED), True, 0.15)
        de = vl['de_passes']['human']['signed_err_us'].get('%s@best' % n)
        _row(ax, yi, de, COLORS.get(n, MUTED), False, -0.18)
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[n] for n in names], fontsize=13)
    ax.set_xlabel('予測 − 人の操作 [µs](ステア、+ = 人より左)')
    ax.set_title('人の操作に対するステアの偏り\n塗り = 人のきれいな走行(%d フレーム)\n白抜き = DE 出口の人の通過(%d回)'
                 % (vl['human_clean_frames'], vl['de_passes']['human']['passes']), loc='left', fontsize=15)
    ax.grid(axis='x', color=GRID, lw=0.8)
    fig.tight_layout()
    _save(fig, out, 'review_offset.png')
    plt.close(fig)


def make_all(rec, out):
    out = Path(out)
    fig_ckpt(rec, out)
    fig_offset(rec, out)
