"""シミュレーターと実機を混ぜた学習(2026-10-13)の評価の図(tools/sim_cotrain_eval.py figures から呼ぶ。CPU・matplotlib)

日本語(Noto Sans CJK JP)・文字は 16 pt 以上・2160×1350 ピクセル(14.4 × 9 インチ × 150 dpi)。
色は腕ごとに固定(sim_cotrain_eval.ARM_COLORS)。色の印のそばに名前や値を直接書く(色だけで見分けさせない)。
"""

import os
from pathlib import Path

import numpy as np

import sim_cotrain_eval as se

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BOX = dict(facecolor=SURF, edgecolor='none', pad=1.5, alpha=0.9)     # 文字の下の線を隠す
FIGSIZE, DPI = (14.4, 9.0), 150
ROWS = ['A', 'B', 'C', 'C+FT', 'A+FT']


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                         'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                         'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1.0,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def _label(arm):
    return se.ARM_LABELS[arm]


def _save(fig, out, name):
    p = Path(out) / name
    fig.savefig(p, dpi=DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


CLASS_JA = {'better': '良い', 'equivalent': '同等', 'worse': '悪い'}


def fig_primary(res, out):
    """主指標: 学習結果ごとの L(● best.pth・○ last.pth)、腕の平均(縦の棒)、A の平均 ± 2 s_A の帯"""
    plt = _setup()
    P = res['primary']
    L = P['L_us']
    thr = P['threshold_us']['best']
    ref = P['arm_mean_us']['A']['best']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axvspan(ref - thr, ref + thr, color=se.ARM_COLORS['A'], alpha=0.10, lw=0)
    ax.axvline(ref, color=se.ARM_COLORS['A'], lw=1.5)
    allv = [v for ck in ('best', 'last') for v in L[ck].values()]
    lo, hi = min(allv), max(allv)
    xmin, xmax = min(lo, ref - thr) - 0.25, max(hi, ref + thr) + 0.9
    for i, arm in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        names = se.arm_names(arm)
        c = se.ARM_COLORS[arm]
        for k, n in enumerate(names):
            off = 0.0 if arm == 'B' else (-0.13 if k == 0 else 0.13)
            ax.plot(L['best'][n], y + off, 'o', ms=13, color=c, mec=SURF, mew=2, zorder=3)
            ax.plot(L['last'][n], y + off, 'o', ms=13, mfc='none', mec=c, mew=2.2, zorder=3)
        m = np.mean([L['best'][n] for n in names])
        if arm != 'B':
            ax.plot([m, m], [y - 0.32, y + 0.32], color=INK, lw=2.4, zorder=2)
        right = max(L[ck][n] for ck in ('best', 'last') for n in names)
        txt = '%.2f µs' % m
        if arm in ('C', 'C+FT', 'A+FT'):
            txt += '(%s)' % CLASS_JA[P['vs_A'][arm]['final_class']]
        ax.text(right + 0.06, y, txt, va='center', color=INK, fontsize=17, bbox=BOX)
    ax.text(ref + thr - 0.02, -0.82, 'A の平均 ± %.3f µs(2 × s_A)= 区別できない幅' % thr, ha='right', va='center',
            color=INK2, fontsize=16)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([_label(a) for a in ROWS][::-1])
    ax.set_ylim(-1.15, len(ROWS) - 0.45)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('保留した実機の人のきれいな走行の L1 [µs](4,670 フレーム。小さいほど人に近い)')
    ax.set_title('主指標  ● best.pth  ○ last.pth  縦の棒 = 2 つの種の平均(best)', loc='left')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return _save(fig, out, 'eval_primary.png')


def fig_genre(res, out):
    """ジャンル別: L1 と前進の偏り(3 フレーム先のスロットル − 人。負 = 人より強く前進)"""
    plt = _setup()
    M = res['accuracy']['metrics']
    hab = res['speed_habit']
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharey=True)
    for col, g in enumerate((4, 5)):
        key = 'human_clean_genre%d' % g
        for row, (field, xl) in enumerate((('l1_us', 'L1 [µs]'), ('fwd_bias_us', '前進の偏り [µs](負 = 人より強く前進)'))):
            ax = axes[row][col]
            refm = np.mean([M[n][key][field] for n in se.arm_names('A')])
            if field == 'fwd_bias_us':
                ax.axvspan(refm - se.SPEED_HABIT_US, refm + se.SPEED_HABIT_US, color=se.ARM_COLORS['A'], alpha=0.08, lw=0)
            ax.axvline(refm, color=se.ARM_COLORS['A'], lw=1.3)
            vals = [M[n][key][field] for a in ROWS for n in se.arm_names(a)]
            span = max(vals) - min(vals)
            pad = max(span * 0.45, 0.8 if field == 'l1_us' else 6.0)
            lo, hi = min(vals) - pad * 0.4, max(vals) + pad
            if field == 'fwd_bias_us':
                lo, hi = min(lo, refm - se.SPEED_HABIT_US - 1), max(hi, refm + se.SPEED_HABIT_US + 5)
            ax.set_xlim(lo, hi)
            for i, arm in enumerate(ROWS):
                y = len(ROWS) - 1 - i
                names = se.arm_names(arm)
                v = [M[n][key][field] for n in names]
                for k, x in enumerate(v):
                    off = 0.0 if arm == 'B' else (-0.13 if k == 0 else 0.13)
                    ax.plot(x, y + off, 'o', ms=12, color=se.ARM_COLORS[arm], mec=SURF, mew=2, zorder=3)
                ax.text(max(v) + 0.04 * (hi - lo), y, '%.1f' % np.mean(v), va='center', fontsize=16, color=INK, bbox=BOX)
            ax.set_xlabel(xl)
            ax.grid(axis='y', visible=False)
            if row == 0:
                ax.set_title(se.GENRE_NAMES[g], loc='left')
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([_label(a) for a in ROWS][::-1])
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
    moved = [arm for arm in hab for g in hab[arm] if hab[arm][g]['moved']]
    fig.suptitle('ジャンル別(best.pth・主指標のフレーム)。下の帯 = A ± 10 µs(速さの癖の移りの目安)  移った腕: %s'
                 % ('なし' if not moved else '・'.join(sorted(set(moved)))), fontsize=18, color=INK, x=0.01, ha='left',
                 y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, 'eval_genre.png')


def fig_sim(res, out):
    """シミュレーターの 12 シーンの L1(参考): 腕 × 与えた頭(ジャンル)"""
    plt = _setup()
    S = res['sim']['per_run']
    marks = {4: ('s', '頭 4(高速)'), 5: ('^', '頭 5(低速)'), 8: ('o', '頭 8(シミュレーター)')}
    rows = ['A', 'B', 'A+FT', 'C', 'C+FT']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    allv = [S[n][k]['l1_us'] for a in rows for n in se.arm_names(a) for k in S[n] if k.startswith('h')]
    lo, hi = min(allv), max(allv)
    xl = (lo - 0.06 * (hi - lo) - 2, hi + 0.30 * (hi - lo) + 12)
    ax.set_xlim(*xl)
    for i, arm in enumerate(rows):
        y = len(rows) - 1 - i
        names = se.arm_names(arm)
        for j, h in enumerate(se.SIM_HEADS):
            key = 'h%d' % h
            if key not in S[names[0]]:
                continue
            v = [S[n][key]['l1_us'] for n in names]
            yy = y + (j - 1) * 0.24
            for x in v:
                ax.plot(x, yy, marks[h][0], ms=12, color=se.ARM_COLORS[arm], mec=SURF, mew=1.5, zorder=3,
                        alpha=1.0 if h == 8 or arm in ('A', 'B', 'A+FT') else 0.55)
            ax.text(max(v) + 0.025 * (xl[1] - xl[0]), yy, '頭 %d: %.1f' % (h, np.mean(v)), va='center', fontsize=16,
                    color=INK, bbox=BOX)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(a) for a in rows][::-1])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('シミュレーターの val 12 シーンの L1 [µs](%d フレーム)' % res['sim']['frames'])
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=marks[h][0], ls='none', ms=12, color=INK2, label=marks[h][1]) for h in se.SIM_HEADS]
    ax.legend(handles=hs, loc='upper left', frameon=False)
    ax.set_title('シミュレーターの L1(参考。C・C+FT では学習に入ったシーン)\nA・A+FT・B はジャンル数 6 で頭 8 が無いので頭 4・5',
                 loc='left', fontsize=18)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return _save(fig, out, 'eval_sim_l1.png')


def fig_robust(res, out):
    """ずれへの強さ: ずれごとの ΔL1(種の平均)と、5 つの平均"""
    import policy_zoo as pz
    plt = _setup()
    R = res['robustness']
    arms = ['A', 'C', 'C+FT', 'A+FT']
    keys = list(pz.PERTS) + ['mean']
    labels = [pz.PERT_LABELS[k] for k in pz.PERTS] + ['5 つの平均']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.2
    for j, arm in enumerate(arms):
        names = se.arm_names(arm)
        for i, k in enumerate(keys):
            if k == 'mean':
                v = [R['per_run'][n]['dL1_us'] for n in names]
            else:
                v = [R['per_run'][n]['per_pert_dL1_us'][k] for n in names]
            x = i + (j - 1.5) * w
            ax.bar(x, np.mean(v), width=w * 0.92, color=se.ARM_COLORS[arm], alpha=0.85,
                   label=_label(arm) if i == 0 else None)
            for vv in v:
                ax.plot(x, vv, 'o', ms=7, color=INK, zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(R['ref_dL1_us'] * pz.ROBUST_RATIO, color=se.ARM_COLORS['A'], lw=1.3, ls='--',
               label='「強い」の線(A の平均 × 0.8 = %.2f µs)' % (R['ref_dL1_us'] * pz.ROBUST_RATIO))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('ΔL1 [µs](ずれありの L1 − ずれなし。小さいほど強い)')
    ax.legend(loc='upper left', frameon=False, ncol=2)
    ymax = max(max(R['per_run'][n]['per_pert_dL1_us'].values()) for a in arms for n in se.arm_names(a))
    ymin = min(min(R['per_run'][n]['per_pert_dL1_us'].values()) for a in arms for n in se.arm_names(a))
    ax.set_ylim(min(0, ymin) - 0.3, ymax * 1.35 + 0.3)
    ft = R['ft_change']['CFT_minus_C_dL1_us']
    ax.set_title('ずれへの強さ(1,559 フレーム、best.pth。棒 = 種の平均、● = 種)  C+FT − C の平均 ΔL1: %+.2f / %+.2f µs'
                 % tuple(ft), loc='left', fontsize=18)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return _save(fig, out, 'eval_robust.png')


def fig_r1_de(res, out):
    """R1・DE: 腕 − A の d̄ と 99.375% 区間(太線)・95% 区間(細線)。点線 = ±10 µs"""
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    rows = [('C − A', 'per_arm', 'C', se.ARM_COLORS['C']), ('C+FT − A', 'per_arm', 'C+FT', se.ARM_COLORS['C+FT']),
            ('A+FT − A', 'per_arm', 'A+FT', se.ARM_COLORS['A+FT']), ('C+FT − C', 'CFT_vs_C', None, se.ARM_COLORS['C+FT']),
            ('C+FT − A+FT', 'CFT_vs_AFT', None, se.ARM_COLORS['C+FT']), ('モデル B − A', 'model_B_vs_A', None, se.ARM_COLORS['B']),
            ('陰性対照 A(10/12)', 'nc_1012', 'NC-A', INK2), ('陰性対照 B(10/12)', 'nc_1012', 'NC-B', INK2)]
    for ax, item, title in ((axes[0], 'r1', 'R1(進入前 0.4 秒のスロットル。+ = 遅い)'),
                            (axes[1], 'de', 'DE(緑マット出口〜0.8 秒のステア。+ = 左)')):
        it = res[item]
        for i, (lab, grp, key, c) in enumerate(rows):
            y = len(rows) - 1 - i
            j = it[grp] if key is None else it[grp][key]
            d, g, b = j['d_mean'], j['ci_good'], j['ci_bad']
            ax.plot([b[1], b[2]], [y, y], color=c, lw=2, alpha=0.6)
            ax.plot([g[1], g[2]], [y, y], color=c, lw=6, alpha=0.35)
            ax.plot(d, y, 'o', ms=12, color=c, mec=SURF, mew=2, zorder=3)
            flag = '良い' if j['good'] else ('悪い' if j['bad'] else '')
            ax.text(max(g[2], b[2]) + 1.0, y, '%+.1f %s' % (d, flag), va='center', fontsize=16, color=INK, bbox=BOX)
        ax.axvline(0, color=INK2, lw=1)
        for v in (-10, 10):
            ax.axvline(v, color=INK2, lw=1, ls=':')
        ax.set_title('%s  %d 通過' % (title, it['passes']), loc='left', fontsize=17)
        ax.set_xlabel('差の平均 [µs](太線 99.375% 区間・細線 95% 区間)')
        ax.grid(axis='y', visible=False)
        lo = min(min(it[g_][k_]['ci_good'][1] if k_ else it[g_]['ci_good'][1], 0) for _, g_, k_, _ in rows)
        hi = max(it[g_][k_]['ci_good'][2] if k_ else it[g_]['ci_good'][2] for _, g_, k_, _ in rows)
        ax.set_xlim(min(lo, -12) - 2, max(hi, 12) + 11)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows][::-1])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle('難しい場所の出力(参考。判定には使わない。同じ作り方の種どうしでも差が R1 7.7・DE 2.8 µs ある)', x=0.01,
                 ha='left', fontsize=18, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, 'eval_r1_de.png')


def make_all(res, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return [fig_primary(res, out), fig_genre(res, out), fig_sim(res, out), fig_robust(res, out), fig_r1_de(res, out)]
