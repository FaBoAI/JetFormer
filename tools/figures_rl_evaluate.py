#!/usr/bin/env python3
"""tools/rl_evaluate.py の評価(evaluation.json)の図。日本語・大きい文字(動画でも読めるように)

    python tools/rl_evaluate.py figures --fig-dir <dir>
"""

import os
from pathlib import Path

import numpy as np

from rl_evaluate import COLORS, LABELS

ORDER = ['B', 'c', 'b', 'a_good', 'a_base', 'n_good', 'n_base']
POL = ['a_good', 'b', 'n_good', 'a_base', 'n_base']
TEXT, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
GOOD, BAD = '#0ca30c', '#d03b3b'


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 17, 'axes.titlesize': 19, 'axes.labelsize': 17, 'xtick.labelsize': 15,
                         'ytick.labelsize': 15, 'legend.fontsize': 14, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': TEXT, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': TEXT,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    return plt


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=110, bbox_inches='tight')
    print('図:', out / name)


def _ci_rows(ax, rows, title, xlabel, note_pos, note_neg):
    """rows = [(表示名, 色, [平均, 下, 上], 通過数)]。0 の線と区間つきの点"""
    y = np.arange(len(rows))[::-1]
    for yi, (lab, col, ci, n) in zip(y, rows):
        if ci[0] is None:
            continue
        ax.plot([ci[1], ci[2]], [yi, yi], color=col, lw=3, solid_capstyle='round')
        ax.plot(ci[0], yi, 'o', ms=12, color=col, mec='#fcfcfb', mew=2)
        ax.text(ci[2], yi + 0.22, ' %+.1f [%+.1f, %+.1f]' % tuple(ci), fontsize=13, color=MUTED, va='bottom')
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_title(title, loc='left')
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color=GRID, lw=0.8)
    lo, hi = ax.get_xlim()
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.text(hi, -0.6, note_pos + ' →', ha='right', va='bottom', fontsize=13, color=MUTED)
    ax.text(lo, -0.6, '← ' + note_neg, ha='left', va='bottom', fontsize=13, color=MUTED)


def fig_r1_de(res, out):
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2))
    r1, de = res['item2_r1'], res['item3_de']
    rows = [(LABELS[x] + ' − (c)', COLORS[x], r1['vs']['%s-c' % x]['d_thr_us'], r1['passes']) for x in POL]
    rows.append(('(c) − モデル B', COLORS['c'], r1['vs']['c-B']['d_thr_us'], r1['passes']))
    _ci_rows(axes[0], rows, 'R1 の手前 0.4秒: 出力のスロットルの差\n(保留の通過 %d回、うち AI %d回)' % (r1['passes'], r1['ai_passes']),
             '差 [µs](通過単位ブートストラップ 95% 区間)', '遅い(合格の向き)', '速い')
    rows = [(LABELS[x] + ' − (c)', COLORS[x], de['vs']['%s-c' % x]['d_steer_us'], de['passes']) for x in POL]
    rows.append(('(c) − モデル B', COLORS['c'], de['vs']['c-B']['d_steer_us'], de['passes']))
    _ci_rows(axes[1], rows, '緑マット出口〜0.8秒: 出力のステアの差\n(保留の通過 %d回、うち AI %d回)' % (de['passes'], de['ai_passes']),
             '差 [µs](通過単位ブートストラップ 95% 区間)', '左(合格の向き)', '右')
    axes[1].set_yticklabels([])
    fig.suptitle('ゲートC 2・3: (a)(b)(n) を (c) filtered BC と比べる(走行の設定 ステア1.3・速度1.0・先読み3)',
                 x=0.02, ha='left', fontsize=19)
    fig.tight_layout()
    _save(fig, out, 'eval_r1_de.png')
    plt.close(fig)


def fig_entry(res, out):
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={'width_ratios': [1.25, 1]})
    r1 = res['item2_r1']
    names = ['actual'] + ORDER
    vals = [r1['actual']['share_lt5500']] + [r1['per_config'][n]['share_lt5500'] for n in ORDER]
    labs = ['実際の走行'] + [LABELS[n] for n in ORDER]
    cols = ['#b5b4ae'] + [COLORS[n] for n in ORDER]
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, [100 * v for v in vals], color=cols, width=0.7)
    for xi, v in zip(x, vals):
        ax.text(xi, 100 * v + 1.5, '%.0f%%' % (100 * v), ha='center', fontsize=14, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=35, ha='right', fontsize=13)
    ax.set_ylabel('進入 eRPM < 5500 の通過 [%]')
    ax.set_ylim(0, 110)
    ax.set_title('R1 の見積もった進入速度(保留 %d通過)' % r1['passes'], loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8)
    em = res['erpm_map']
    ax = axes[1]
    ax.plot(em['x_us'], em['y_erpm'], color=COLORS['c'], lw=2.5)
    ax.axhline(5500, color=BAD, lw=1.2, ls='--')
    ax.text(em['x_us'][-1], 5700, '5500 eRPM(これ未満なら\nR1 の強い衝突はまれ)', fontsize=13, color=BAD, ha='right',
            va='bottom')
    ax.set_xlabel('スロットルの指令 [µs](小さいほど前進)')
    ax.set_ylabel('定常の前進 eRPM')
    v = em['validation_heldout_r1']
    ax.set_title('指令 → 速さの対応表(定常 %d点)\n実際の進入との順位相関 %.2f・誤差 %d eRPM'
                 % (em['samples'], v['spearman_map_vs_entry'], v['mae_erpm']), loc='left', fontsize=16)
    ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    _save(fig, out, 'eval_r1_entry.png')
    plt.close(fig)


def fig_collapse(res, out):
    plt = _setup()
    it = res['item4_collapse']
    th = res['rules']['thresholds']
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    xs = ['c'] + POL
    y = np.arange(len(xs))[::-1]
    panels = [('straight_drop', '直線の前進の強さの低下\n(モデル B 比、%d フレーム)' % it['straight_frames'], 100.0,
               100 * th['straight_drop_max'], '低下 [%]'),
              ('l1_ratio', '人のきれいな走行の L1\n(モデル B 比、%d フレーム)' % it['human_clean_frames'], 1.0,
               th['l1_ratio_max'], 'L1 の比(1 = 同じ)')]
    for ax, (key, title, sc, thr, xl) in zip(axes[:2], panels):
        for yi, x in zip(y, xs):
            ci = it['vs']['%s-B' % x][key]
            ax.plot([sc * ci[1], sc * ci[2]], [yi, yi], color=COLORS[x], lw=3, solid_capstyle='round')
            ax.plot(sc * ci[0], yi, 'o', ms=12, color=COLORS[x], mec='#fcfcfb', mew=2)
        ax.axvline(thr, color=BAD, ls='--', lw=1.5)
        ax.text(thr, -0.55, ' 合格の上限', color=BAD, fontsize=13, va='bottom')
        ax.set_ylim(-0.7, len(xs) - 0.4)
        ax.axvline(sc * (1.0 if key == 'l1_ratio' else 0.0), color=MUTED, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[x] for x in xs] if ax is axes[0] else [])
        ax.set_title(title, loc='left')
        ax.set_xlabel(xl)
        ax.grid(axis='x', color=GRID, lw=0.8)
    ax = axes[2]
    xs2 = ['B'] + xs
    y2 = np.arange(len(xs2))[::-1]
    for yi, x in zip(y2, xs2):
        ci = it['per_config'][x]['fwd_bias_ci']
        ax.plot([ci[1], ci[2]], [yi, yi], color=COLORS[x], lw=3, solid_capstyle='round')
        ax.plot(ci[0], yi, 'o', ms=12, color=COLORS[x], mec='#fcfcfb', mew=2)
    ax.axvline(th['fwd_bias_min_us'], color=BAD, ls='--', lw=1.5)
    ax.text(th['fwd_bias_min_us'], -0.55, ' 合格の下限', color=BAD, fontsize=13, va='bottom')
    ax.set_ylim(-0.7, len(xs2) - 0.4)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y2)
    ax.set_yticklabels([LABELS[x] for x in xs2], fontsize=13)
    ax.set_title('前進の偏り(予測 − 人)\n負 = 人より速く走りがち', loc='left')
    ax.set_xlabel('[µs](3フレーム先、保留の人の走行)')
    ax.grid(axis='x', color=GRID, lw=0.8)
    fig.suptitle('ゲートC 4: 崩れていないか(点 = 推定、線 = シーン単位ブートストラップ 95% 区間、赤の破線 = 合格の境目)',
                 x=0.02, ha='left', fontsize=18)
    fig.tight_layout()
    _save(fig, out, 'eval_collapse.png')
    plt.close(fig)


def fig_gate(res, out):
    plt = _setup()
    v = res['verdict']['table']
    items = [('1_value', '1 価値が衝突を予測(AUROC ≥ 0.8)'), ('2_r1', '2 R1 の手前で (c) より遅い'),
             ('3_de', '3 DE の出口で (c) より左'), ('4_collapse', '4 崩れていない(B 比)'), ('5_labels', '5 報酬の悪用がない(印)')]
    cols = POL
    fig, ax = plt.subplots(figsize=(17, 6.4))
    for j, x in enumerate(cols):
        for i, (k, _) in enumerate(items):
            ok = v[x][k]
            ax.add_patch(plt.Rectangle((j + 0.04, len(items) - i - 1 + 0.06), 0.92, 0.88,
                                       color=GOOD if ok else BAD, alpha=0.18 if ok else 0.22, lw=0))
            ax.text(j + 0.5, len(items) - i - 0.5, '○ 合格' if ok else '× 不合格', ha='center', va='center',
                    fontsize=17, color=GOOD if ok else BAD, weight='bold')
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(items))
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels([LABELS[x] for x in cols], fontsize=15)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(items)) + 0.5)
    ax.set_yticklabels([t for _, t in items][::-1], fontsize=15)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    passed = [LABELS[x] for x in ('a_good', 'b') if v[x]['all']]
    same = all(v['n_good'][k] == v[x][k] for x in ('a_good', 'b') for k, _ in items)
    ax.set_title('ゲートC(机上)の判定 — %s%s'
                 % ('合格: ' + '・'.join(passed) if passed else '実車に出せる方策は無い',
                    '(陰性対照 (n) と (a)(b) の結果が同じで、見分けられていない)' if same else ''),
                 loc='left', fontsize=18, pad=48)
    fig.tight_layout()
    _save(fig, out, 'eval_gate_c.png')
    plt.close(fig)


def fig_accuracy(res, out):
    plt = _setup()
    acc = res['accuracy']['metrics']
    show = ['B', 'c', 'b', 'a_good', 'n_good']
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), gridspec_kw={'width_ratios': [1, 1, 1.15]})
    k = np.arange(1, 11)
    for ax, key, title in ((axes[0], 'steer_mae', 'ステア'), (axes[1], 'throttle_mae', 'スロットル')):
        for n in show:
            m = acc[n]['human_clean']
            ax.plot(k, m[key], color=COLORS[n], lw=2.2, marker='o', ms=7, label=LABELS[n])
        ax.set_title('%s: 保留の人の走行との誤差\n(%d フレーム、自己入力 = 教師あり)'
                     % (title, acc['B']['human_clean']['n']), loc='left', fontsize=17)
        ax.set_xlabel('何フレーム先か(1 = 33ms)')
        ax.set_ylabel('平均絶対誤差 [µs]')
        ax.grid(color=GRID, lw=0.8)
    axes[0].legend(loc='upper left', frameon=False)
    ax = axes[2]
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    d = res['accuracy']['diffs']
    n = res['accuracy']['sets']
    sets = [('human_clean', '人のきれいな走行(%d フレーム・11シーン)' % n['human_clean']),
            ('iv_all', '交代後の人の修正(全 29窓・%d フレーム)' % n['iv_all']),
            ('iv_log_heldout', '修正のうちログも保留の 3窓(区間は目安)')]
    xs = ['c', 'b', 'a_good', 'n_good']
    ypos, labels = [], []
    yy = 0.0
    for sk, sl in sets[::-1]:
        for x in xs[::-1]:
            ci = d['%s-B/%s' % (x, sk)]['l1']
            ax.plot([ci[1], ci[2]], [yy, yy], color=COLORS[x], lw=3, solid_capstyle='round')
            ax.plot(ci[0], yy, 'o', ms=11, color=COLORS[x], mec='#fcfcfb', mew=2)
            ypos.append(yy)
            labels.append(LABELS[x])
            yy += 1
        ax.text(0.01, yy - 0.3, sl, transform=tr, fontsize=13, color=TEXT, ha='left', va='bottom', weight='bold')
        yy += 1.0
    ax.set_ylim(-0.7, yy - 0.2)
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('L1 の差(方策 − モデル B)[µs]\n負 = 人に近い')
    ax.set_title('人との一致の差(95% 区間)', loc='left', fontsize=17)
    ax.grid(axis='x', color=GRID, lw=0.8)
    fig.tight_layout()
    _save(fig, out, 'eval_accuracy.png')
    plt.close(fig)


def make_all(res, out):
    out = Path(out)
    fig_gate(res, out)
    fig_r1_de(res, out)
    fig_entry(res, out)
    fig_collapse(res, out)
    fig_accuracy(res, out)
