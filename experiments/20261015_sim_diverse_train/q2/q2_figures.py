"""Q2 の図(q2_eval.py figures から呼ぶ。CPU・matplotlib)。10/13 の sim_cotrain_figures.py の写しを腕 A・B・C・C-DIV にしたもの

日本語(Noto Sans CJK JP)・文字は 16 pt 以上・2160×1350 ピクセル(14.4 × 9 インチ × 150 dpi)。
色は腕ごとに固定(q2_eval.ARM_COLORS。モデル B は基準なので灰色)。色の印のそばに名前や値を直接書く(色だけで見分けさせない)。
"""

import os
from pathlib import Path

import numpy as np

import q2_eval as qe

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BOX = dict(facecolor=SURF, edgecolor='none', pad=1.5, alpha=0.9)
FIGSIZE, DPI = (14.4, 9.0), 150
ROWS = ['A', 'B', 'C', 'C-DIV']
CLASS_JA = {'better': '良い', 'equivalent': '同等', 'worse': '悪い'}


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


def _names(res, arm):
    return [n for n in qe.arm_names(arm) if n in res['names']]


def _save(fig, out, name):
    p = Path(out) / name
    fig.savefig(p, dpi=DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


def _offsets(k, n):
    return 0.0 if n == 1 else (-0.13 if k == 0 else 0.13)


def _vs(ref, cls):
    return '%s と同等' % ref if cls == 'equivalent' else '%s より%s' % (ref, CLASS_JA[cls])


def fig_primary(res, out):
    """主指標: 学習結果ごとの L(● best・○ last)、腕の平均(縦の棒)、A の平均 ± 閾値の帯"""
    plt = _setup()
    P = res['primary']
    L = P['L_us']
    thr = P['threshold_vs_A_us']['best']
    ref = P['arm_mean_us']['A']['best']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axvspan(ref - thr, ref + thr, color=qe.ARM_COLORS['A'], alpha=0.10, lw=0)
    ax.axvline(ref, color=qe.ARM_COLORS['A'], lw=1.5)
    allv = [v for ck in ('best', 'last') for v in L[ck].values()]
    lo, hi = min(allv), max(allv)
    xmin, xmax = min(lo, ref - thr) - 0.25, max(hi, ref + thr) + 1.6
    for i, arm in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        names = _names(res, arm)
        c = qe.ARM_COLORS[arm]
        for k, n in enumerate(names):
            off = _offsets(k, len(names))
            ax.plot(L['best'][n], y + off, 'o', ms=13, color=c, mec=SURF, mew=2, zorder=3)
            ax.plot(L['last'][n], y + off, 'o', ms=13, mfc='none', mec=c, mew=2.2, zorder=3)
        m = np.mean([L['best'][n] for n in names])
        if arm != 'B':
            ax.plot([m, m], [y - 0.32, y + 0.32], color=INK, lw=2.4, zorder=2)
        right = max(L[ck][n] for ck in ('best', 'last') for n in names)
        txt = '%.2f µs' % m
        if arm == 'C':
            txt += '(%s)' % _vs('A', P['C_vs_A']['final_class'])
        if arm == 'C-DIV':
            txt += '(%s・%s%s)' % (_vs('A', P['CDIV_vs_A']['final_class']), _vs('C', P['CDIV_vs_C']['final_class']),
                                  '・暫定' if res['single_seed'] else '')
        ax.text(right + 0.06, y, txt, va='center', color=INK, fontsize=17, bbox=BOX)
    ax.text(ref + thr - 0.02, -0.82, 'A の平均 ± %.3f µs(2 × s_A)= 区別できない幅' % thr, ha='right', va='center',
            color=INK2, fontsize=16)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([qe.ARM_LABELS[a] for a in ROWS][::-1])
    ax.set_ylim(-1.15, len(ROWS) - 0.45)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('保留した実機の人のきれいな走行の L1 [µs](%s フレーム。小さいほど人に近い)' % format(res['frames']['human_clean'], ','))
    ax.set_title('主指標  ● best.pth  ○ last.pth  縦の棒 = 種の平均(best)', loc='left')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return _save(fig, out, 'q2_primary.png')


def fig_genre(res, out):
    """ジャンル別: L1 と前進の偏り(3 フレーム先のスロットル − 人。負 = 人より強く前進)"""
    plt = _setup()
    M = res['accuracy']['metrics']
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharey=True)
    gname = {4: '高速(ジャンル 4)', 5: '低速(ジャンル 5)'}
    for col, g in enumerate((4, 5)):
        key = 'human_clean_genre%d' % g
        for row, (field, xl) in enumerate((('l1_us', 'L1 [µs]'), ('fwd_bias_us', '前進の偏り [µs](負 = 人より強く前進)'))):
            ax = axes[row][col]
            refm = np.mean([M[n][key][field] for n in _names(res, 'A')])
            if field == 'fwd_bias_us':
                ax.axvspan(refm - qe.SPEED_HABIT_US, refm + qe.SPEED_HABIT_US, color=qe.ARM_COLORS['A'], alpha=0.08, lw=0)
            ax.axvline(refm, color=qe.ARM_COLORS['A'], lw=1.3)
            vals = [M[n][key][field] for a in ROWS for n in _names(res, a)]
            span = max(vals) - min(vals)
            pad = max(span * 0.45, 0.8 if field == 'l1_us' else 6.0)
            lo, hi = min(vals) - pad * 0.4, max(vals) + pad
            if field == 'fwd_bias_us':
                lo, hi = min(lo, refm - qe.SPEED_HABIT_US - 1), max(hi, refm + qe.SPEED_HABIT_US + 5)
            ax.set_xlim(lo, hi)
            for i, arm in enumerate(ROWS):
                y = len(ROWS) - 1 - i
                names = _names(res, arm)
                v = [M[n][key][field] for n in names]
                for k, x in enumerate(v):
                    ax.plot(x, y + _offsets(k, len(v)), 'o', ms=12, color=qe.ARM_COLORS[arm], mec=SURF, mew=2, zorder=3)
                ax.text(max(v) + 0.04 * (hi - lo), y, '%.1f' % np.mean(v), va='center', fontsize=16, color=INK, bbox=BOX)
            ax.set_xlabel(xl)
            ax.grid(axis='y', visible=False)
            if row == 0:
                ax.set_title(gname[g], loc='left')
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([qe.ARM_SHORT[a] for a in ROWS][::-1])
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
    hab = res['speed_habit']
    moved = [arm for arm in ('C', 'C-DIV') for g in ('4', '5') if hab[arm][g]['moved']]
    fig.suptitle('ジャンル別(best.pth・主指標のフレーム)。帯 = A ± 10 µs(速さの癖の移りの目安)  移った腕: %s'
                 % ('なし' if not moved else '・'.join(sorted(set(moved)))), fontsize=18, color=INK, x=0.01, ha='left', y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, 'q2_genre.png')


def fig_speed_hs(res, out):
    """低速の直線 HS(10/13 の確認で C が強く前進を出した所)の前進の偏り、種ごと"""
    plt = _setup()
    hs = res['speed_habit']['low_speed_straight_HS']
    if not hs:
        return None
    fb = hs['fwd_bias_per_run_us']
    a = hs['A_mean_us']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axvspan(a - qe.SPEED_HABIT_US, a + qe.SPEED_HABIT_US, color=qe.ARM_COLORS['A'], alpha=0.10, lw=0)
    ax.axvline(a, color=qe.ARM_COLORS['A'], lw=1.5)
    ax.axvline(0, color=INK2, lw=1, ls=':')
    vals = list(fb.values())
    lo, hi = min(min(vals), a - 12) - 4, max(max(vals), a + 12) + 9
    for i, arm in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        names = _names(res, arm)
        for k, n in enumerate(names):
            x = fb[n]
            ax.plot(x, y + _offsets(k, len(names)), 'o', ms=14, color=qe.ARM_COLORS[arm], mec=SURF, mew=2, zorder=3)
            ax.text(x, y + _offsets(k, len(names)) + 0.2, '%+.1f' % x, ha='center', fontsize=16, color=INK, bbox=BOX)
        ax.text(hi - 0.5, y, '平均 %+.1f µs' % np.mean([fb[n] for n in names]), ha='right', va='center', fontsize=17,
                color=INK, bbox=BOX)
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([qe.ARM_LABELS[r] for r in ROWS][::-1])
    ax.set_ylim(-0.6, len(ROWS) - 0.3)
    ax.set_xlabel('前進の偏り [µs](3 フレーム先のスロットル − 人。負 = 人より強く前進)\n帯 = A の平均 ± 10 µs(速さの癖の移りの目安)')
    ax.set_title('低速(ジャンル 5)の直線 HS・%d フレーム(参考)\n10/13 の確認で C が人より強く前進を出した所' % hs['n'], loc='left',
                 fontsize=18)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return _save(fig, out, 'q2_speed_hs.png')


def fig_sim(res, out):
    """シミュレーターの L1(参考): 左 = 前の教師の val 12 シーン、右 = 5 モードの val 15 シーン(どの腕にも保留)"""
    plt = _setup()
    marks = {4: ('s', '頭 4(高速)'), 5: ('^', '頭 5(低速)'), 8: ('o', '頭 8(シミュレーター)')}
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    for ax, key, title in ((axes[0], 'sim_old_val12', '前の教師のデータの val 12 シーン'),
                           (axes[1], 'sim_modes_val15', '5 モードの見本の val 15 シーン(保留)')):
        S = res[key]['per_run']
        allv = [S[n][k]['l1_us'] for a in ROWS for n in _names(res, a) if S.get(n) for k in S[n] if k.startswith('h')]
        lo, hi = min(allv), max(allv)
        xl = (max(0, lo - 0.08 * (hi - lo) - 2), hi + 0.42 * (hi - lo) + 12)
        ax.set_xlim(*xl)
        for i, arm in enumerate(ROWS):
            y = len(ROWS) - 1 - i
            names = [n for n in _names(res, arm) if S.get(n)]
            for j, h in enumerate(qe.SIM_HEADS):
                hk = 'h%d' % h
                if not names or hk not in S[names[0]]:
                    continue
                v = [S[n][hk]['l1_us'] for n in names]
                yy = y + (j - 1) * 0.26
                for x in v:
                    ax.plot(x, yy, marks[h][0], ms=11, color=qe.ARM_COLORS[arm], mec=SURF, mew=1.5, zorder=3,
                            alpha=1.0 if h == 8 or arm in ('A', 'B') else 0.55)
                ax.text(max(v) + 0.025 * (xl[1] - xl[0]), yy, '頭%d %.1f' % (h, np.mean(v)), va='center', fontsize=16,
                        color=INK, bbox=BOX)
        ax.set_xlabel('L1 [µs](%s フレーム)' % format(res[key]['frames'], ','))
        ax.set_title(title, loc='left', fontsize=18)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([qe.ARM_SHORT[a] for a in ROWS][::-1])
    axes[0].set_ylim(-0.7, len(ROWS) - 0.3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=marks[h][0], ls='none', ms=12, color=INK2, label=marks[h][1]) for h in qe.SIM_HEADS]
    axes[0].legend(handles=hs, loc='upper left', frameon=True, facecolor=SURF, edgecolor=GRID)
    fig.suptitle('シミュレーターの L1(参考)。A・B はジャンル数 6 で頭 8 が無いので頭 4・5。C の頭 8 は左の線そのもので学習している',
                 x=0.01, ha='left', fontsize=17, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, 'q2_sim_l1.png')


def fig_robust(res, out):
    """ずれへの強さ: ずれごとの ΔL1(種の平均)と、5 つの平均"""
    import policy_zoo as pz
    plt = _setup()
    R = res['robustness']
    arms = ['A', 'C', 'C-DIV']
    keys = list(pz.PERTS) + ['mean']
    labels = [pz.PERT_LABELS[k] for k in pz.PERTS] + ['5 つの平均']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.26
    for j, arm in enumerate(arms):
        names = _names(res, arm)
        for i, k in enumerate(keys):
            if k == 'mean':
                v = [R['per_run'][n]['dL1_us'] for n in names]
            else:
                v = [R['per_run'][n]['per_pert_dL1_us'][k] for n in names]
            x = i + (j - 1) * w
            ax.bar(x, np.mean(v), width=w * 0.9, color=qe.ARM_COLORS[arm], alpha=0.85,
                   label=qe.ARM_LABELS[arm] if i == 0 else None)
            for vv in v:
                ax.plot(x, vv, 'o', ms=8, color=INK, zorder=3)
            if k == 'mean':
                ax.text(x, max(v) + 0.08, '%.2f' % np.mean(v), ha='center', fontsize=16, color=INK, bbox=BOX)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(R['ref_dL1_us'] * pz.ROBUST_RATIO, color=qe.ARM_COLORS['A'], lw=1.3, ls='--',
               label='「強い」の線(A の平均 × 0.8 = %.2f µs)' % (R['ref_dL1_us'] * pz.ROBUST_RATIO))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('ΔL1 [µs](ずれありの L1 − ずれなし)')
    ax.legend(loc='upper right', frameon=False)
    ymax = max(max(R['per_run'][n]['per_pert_dL1_us'].values()) for a in arms for n in _names(res, a))
    ymin = min(min(R['per_run'][n]['per_pert_dL1_us'].values()) for a in arms for n in _names(res, a))
    ax.set_ylim(min(0, ymin) - 0.3, ymax * 1.5 + 0.3)
    ra = R['per_arm']
    ax.set_title('ずれへの強さ(1,559 フレーム、best.pth。棒 = 種の平均、● = 種)。A を 1 とすると C %.2f 倍・C-DIV %.2f 倍'
                 % (ra['C']['ratio_to_ref'], ra['C-DIV']['ratio_to_ref']), loc='left', fontsize=18)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return _save(fig, out, 'q2_robust.png')


def fig_r1_de(res, out):
    """R1・DE: 差の平均と 99.375% 区間(太線)・95% 区間(細線)。点線 = ±10 µs"""
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    rows = [('C-DIV − A', 'compare', 'C-DIV − A', qe.ARM_COLORS['C-DIV']), ('C − A', 'compare', 'C − A', qe.ARM_COLORS['C']),
            ('C-DIV − C', 'compare', 'C-DIV − C', qe.ARM_COLORS['C-DIV']), ('モデル B − A', 'model_B_vs_A', None, qe.ARM_COLORS['B']),
            ('陰性対照 A(10/12)', 'nc_1012', 'NC-A', INK2), ('陰性対照 B(10/12)', 'nc_1012', 'NC-B', INK2)]
    for ax, item, title in ((axes[0], 'r1', 'R1(進入前 0.4 秒のスロットル。+ = 遅い)'),
                            (axes[1], 'de', 'DE(緑マット出口〜0.8 秒のステア。+ = 左)')):
        it = res[item]
        lo, hi = 0.0, 0.0
        for i, (lab, grp, key, c) in enumerate(rows):
            y = len(rows) - 1 - i
            j = it[grp] if key is None else it[grp][key]
            d, g, b = j['d_mean'], j['ci_good'], j['ci_bad']
            ax.plot([b[1], b[2]], [y, y], color=c, lw=2, alpha=0.6)
            ax.plot([g[1], g[2]], [y, y], color=c, lw=6, alpha=0.35)
            ax.plot(d, y, 'o', ms=12, color=c, mec=SURF, mew=2, zorder=3)
            flag = '良い' if j['good'] else ('悪い' if j['bad'] else '')
            ax.text(max(g[2], b[2]) + 1.0, y, '%+.1f %s' % (d, flag), va='center', fontsize=16, color=INK, bbox=BOX)
            lo, hi = min(lo, g[1], b[1]), max(hi, g[2], b[2])
        ax.axvline(0, color=INK2, lw=1)
        for v in (-10, 10):
            ax.axvline(v, color=INK2, lw=1, ls=':')
        ax.set_title('%s  %d 通過' % (title, it['passes']), loc='left', fontsize=17)
        ax.set_xlabel('差の平均 [µs](太線 99.375% 区間・細線 95% 区間)')
        ax.grid(axis='y', visible=False)
        ax.set_xlim(min(lo, -12) - 2, max(hi, 12) + 12)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows][::-1])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.suptitle('難しい場所の出力(参考。判定には使わない。同じ作り方の種どうしでも差が R1 7.7・DE 2.8 µs ある)', x=0.01,
                 ha='left', fontsize=18, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, 'q2_r1_de.png')


def make_all(res, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    figs = [fig_primary(res, out), fig_genre(res, out), fig_speed_hs(res, out), fig_robust(res, out), fig_r1_de(res, out)]
    if all(res[k]['per_run'].get(n) for k in ('sim_old_val12', 'sim_modes_val15') for n in res['names']):
        figs.append(fig_sim(res, out))
    return [f for f in figs if f]
