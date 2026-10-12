"""方策の比較(policy zoo)の評価の図(tools/policy_zoo.py figures から呼ぶ。CPU・matplotlib・日本語・大きい文字)

色は方策ごとに固定(policy_zoo.COLORS。順位で塗り替えない)。モデル B は灰色。文字は黒・灰色のインクで書き、色の印のそばに
名前を直接書く(色だけで見分けさせない)。軸は1つ(2つの縦軸は使わない)。
"""

import os
from pathlib import Path

import numpy as np

import policy_zoo as pz

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                         'ytick.labelsize': 13, 'legend.fontsize': 12, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                         'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1.0,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def _label(p):
    return pz.LABELS.get(p, p)


def fig_primary(res, out):
    """主指標: 学習結果ごとの L(● best.pth・○ last.pth)、方策の平均(縦の棒)、v0 の平均 ± δ の帯"""
    plt = _setup()
    L = res['primary']['L_us']
    delta = res['primary']['delta_us']['best']
    rows = ['B'] + pz.POLICIES
    ref = res['primary']['per_policy']['jetformer_v0']['best']['ref_mean']
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axvspan(ref - delta, ref + delta, color='#2a78d6', alpha=0.07, lw=0)
    ax.axvline(ref, color='#2a78d6', lw=1.5)
    # 帯の説明は一番下の行のさらに下に書く(上に書くと題と重なる)
    ax.text(ref + delta - 0.2, -0.75, 'v0 の平均 ± δ(%.0f µs)= 区別できない幅' % delta, ha='right', va='center',
            color=INK2, fontsize=12)
    for i, p in enumerate(rows):
        y = len(rows) - 1 - i
        names = ['B'] if p == 'B' else pz.seed_names(p)
        c = pz.COLORS[p]
        for k, n in enumerate(names):
            off = 0.0 if p == 'B' else (-0.12 if k == 0 else 0.12)
            ax.plot(L['best'][n], y + off, 'o', ms=10, color=c, mec=SURF, mew=2, zorder=3)
            ax.plot(L['last'][n], y + off, 'o', ms=10, mfc='none', mec=c, mew=2, zorder=3)
        if p != 'B':
            m = np.mean([L['best'][n] for n in names])
            ax.plot([m, m], [y - 0.3, y + 0.3], color=INK, lw=2, zorder=2)
            # 値の文字は best・last の右端の印より右に置く(印と重ねない)
            ax.text(max(L[ck][n] for ck in ('best', 'last') for n in names) + 0.35, y, '%.1f µs' % m, va='center',
                    color=INK, fontsize=13)
        else:
            ax.text(max(L['best']['B'], L['last']['B']) + 0.35, y, '%.1f µs' % L['best']['B'], va='center', color=INK,
                    fontsize=13)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(p) for p in rows][::-1])
    ax.set_ylim(-1.1, len(rows) - 0.5)
    lo = min(min(v.values()) for v in L.values())
    hi = max(max(v.values()) for v in L.values())
    ax.set_xlim(min(lo - 1.5, ref - delta - 0.5), max(hi + 3.0, ref + delta + 0.5))
    ax.set_xlabel('保留した人のきれいな走行の L1 [µs](小さいほど人に近い)')
    ax.set_title('主指標(4,670 フレーム)  ● best.pth  ○ last.pth  縦の棒 = 2 つの種の平均')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(out / 'eval_primary.png', dpi=100)
    # 拡大: δ の帯を外し、0.1 µs の差が見える幅で
    fig, ax = plt.subplots(figsize=(13, 6.2))
    for i, p in enumerate(rows):
        y = len(rows) - 1 - i
        names = ['B'] if p == 'B' else pz.seed_names(p)
        c = pz.COLORS[p]
        for k, n in enumerate(names):
            off = 0.0 if p == 'B' else (-0.12 if k == 0 else 0.12)
            ax.plot(L['best'][n], y + off, 'o', ms=11, color=c, mec=SURF, mew=2, zorder=3)
            ax.plot(L['last'][n], y + off, 'o', ms=11, mfc='none', mec=c, mew=2, zorder=3)
            ax.text(max(L['best'][n], L['last'][n]) + 0.06, y + off, '種%d' % k if p != 'B' else 'B', va='center',
                    color=INK2, fontsize=11)
    ax.axvline(ref, color='#2a78d6', lw=1.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_label(p) for p in rows][::-1])
    ax.set_xlabel('L1 [µs](拡大。縦線 = v0 の 2 つの種の平均)')
    ax.set_title('主指標の拡大: 種どうし・best/last の揺れと、方策の差の大きさ')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(out / 'eval_primary_zoom.png', dpi=100)
    plt.close('all')


def fig_r1_de(res, out):
    """R1・DE: d̄(方策 − v0)の点と区間(太 = 99.375%、細 = 95%)。±10 µs = D1 の組の閾値の目安"""
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    rows = [('act', 'act'), ('tcp', 'tcp'), ('msfusion', 'msfusion'), ('dinov2', 'dinov2'), ('B', 'model_B_vs_v0'),
            ('NC-A', 'NC-A'), ('NC-B', 'NC-B')]
    names = {'NC-A': '陰性対照 A(v0 種1 − {種0, B})', 'NC-B': '陰性対照 B(v0 種0 − {種1, B})', 'B': 'モデル B − v0'}
    for ax, item, title, unit in ((axes[0], 'r1', 'R1 の進入前のスロットル(+ = 遅い)', 'スロットルの差 [µs]'),
                                  (axes[1], 'de', 'DE 出口のステア(+ = 左)', 'ステアの差 [µs]')):
        it = res[item]
        for i, (lab, key) in enumerate(rows):
            y = len(rows) - 1 - i
            if key in it['per_policy']:
                j = it['per_policy'][key]
            elif key == 'model_B_vs_v0':
                j = it['model_B_vs_v0']
            else:
                j = it['nc'][key]
            c = pz.COLORS.get(lab, INK2)
            ax.plot(j['ci_bad'][1:], [y, y], color=c, lw=2, alpha=0.6)
            ax.plot(j['ci_good'][1:], [y, y], color=c, lw=6, alpha=0.35, solid_capstyle='round')
            ax.plot(j['d_mean'], y, 'o', ms=10, color=c, mec=SURF, mew=2, zorder=3)
            tag = '良い' if j['good'] else ('悪い' if j['bad'] else '')
            if tag:
                ax.text(j['ci_bad'][2] + 1, y, tag, va='center', color=INK, fontsize=12)
        ax.axvline(0, color=INK2, lw=1.2)
        for v in (-10, 10):
            ax.axvline(v, color=INK2, lw=1, ls=(0, (1, 3)))
        ax.set_title('%s  %d 通過' % (title, it['passes']))
        ax.set_xlabel(unit)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([names.get(l, _label(l)) for l, _ in rows][::-1])
    fig.suptitle('通過ごとのブートストラップ 10,000 回: 太 = 99.375%(「良い」の区間)、細 = 95%(「悪い」の区間)、点線 = ±10 µs', fontsize=13,
                 color=INK2)
    fig.tight_layout()
    fig.savefig(out / 'eval_r1_de.png', dpi=100)
    plt.close('all')


def fig_speed(res, out):
    """速さ(run_numpy の中央値)と主指標(種の平均と範囲)の散布"""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    lat = res['latency']['per_run']
    L = res['primary']['L_us']['best']
    for p in pz.POLICIES:
        names = pz.seed_names(p)
        xs = [lat[n]['run_numpy_median_ms'] for n in names]
        ys = [L[n] for n in names]
        c = pz.COLORS[p]
        ax.plot([np.mean(xs)] * 2, [min(ys), max(ys)], color=c, lw=2)
        ax.plot([min(xs), max(xs)], [np.mean(ys)] * 2, color=c, lw=2)
        ax.plot(np.mean(xs), np.mean(ys), 'o', ms=12, color=c, mec=SURF, mew=2, zorder=3)
        ax.text(np.mean(xs) + 0.07, np.mean(ys) + 0.07, _label(p), color=INK, fontsize=13)
    ax.plot(lat['B']['run_numpy_median_ms'], L['B'], 's', ms=11, color=pz.COLORS['B'], mec=SURF, mew=2, zorder=3)
    ax.text(lat['B']['run_numpy_median_ms'] + 0.07, L['B'], 'モデル B', va='center', color=INK, fontsize=13)
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(min(lo_y, L['B'] - 0.08), hi_y)        # モデル B の印と名前が軸の目盛りと重ならないように
    ax.axvline(6.0, color=INK2, lw=1, ls=(0, (1, 3)))
    ax.text(5.95, ax.get_ylim()[1], '速さのゲート 6 ms', ha='right', va='top', color=INK2, fontsize=12)
    ax.set_xlabel('推論 1 回(run_numpy・走行と同じ呼び方)の中央値 [ms]')
    ax.set_ylabel('主指標 L1 [µs]')
    ax.set_title('速さと人との近さ(線 = 2 つの種の範囲。dinov2 は作り直したエンジン)')
    fig.tight_layout()
    fig.savefig(out / 'eval_speed_vs_l1.png', dpi=100)
    plt.close('all')


def fig_mae_k(res, out):
    """k = 1〜10 フレーム先の MAE(主指標のフレーム、種の平均)"""
    plt = _setup()
    met = res['accuracy']['metrics']
    k = np.arange(1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, key, title in ((axes[0], 'steer_mae', 'ステア'), (axes[1], 'throttle_mae', 'スロットル')):
        ax.plot(k, met['B']['human_clean'][key], color=pz.COLORS['B'], lw=2, ls=(0, (4, 2)), label='モデル B')
        for p in pz.POLICIES:
            v = np.mean([met[n]['human_clean'][key] for n in pz.seed_names(p)], axis=0)
            ax.plot(k, v, color=pz.COLORS[p], lw=2, marker='o', ms=6, mec=SURF, label=_label(p))
        ax.set_title('%s の誤差(何フレーム先か)' % title)
        ax.set_xlabel('何フレーム先か(1 = 33 ms)')
        ax.set_ylabel('平均絶対誤差 [µs]')
        ax.set_xticks(k)
    axes[0].legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'eval_mae_by_step.png', dpi=100)
    plt.close('all')


def fig_robust(res, out):
    """ずれへの強さ: ずれごとの L1 の増え方(種の平均)"""
    plt = _setup()
    rob = res['robustness']['per_run']
    fig, ax = plt.subplots(figsize=(13, 6.2))
    rows = list(pz.PERTS) + ['mean']
    labs = [pz.PERT_LABELS[p] for p in pz.PERTS] + ['5 つの平均']
    ents = ['B'] + pz.POLICIES
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for j, e in enumerate(ents):
            names = ['B'] if e == 'B' else pz.seed_names(e)
            if r == 'mean':
                v = np.mean([rob[n]['dL1_us'] for n in names])
            else:
                v = np.mean([rob[n]['per_pert_dL1_us'][r] for n in names])
            ax.plot(v, y + (j - 2.5) * 0.11, 'o' if e != 'B' else 's', ms=9, color=pz.COLORS[e], mec=SURF, mew=1.5,
                    label=(_label(e) if i == 0 else None))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labs[::-1])
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_xlabel('ずれを入れたときの L1 の増え方 ΔL1 [µs](小さいほど強い)')
    ax.set_title('ずれへの強さ(約 1,559 フレーム、best.pth、種の平均)')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(out / 'eval_robustness.png', dpi=100)
    plt.close('all')


def fig_tcp(res, out):
    """tcp の制御器の出し方ごとの L1 と、R1 の進入の見積もり(参考)"""
    plt = _setup()
    t = res['tcp']['per_run']
    if not t:
        return
    sets = list(pz.TCP_SETTINGS)
    labs = {'control': 'control\n(既定)', 'control_cap': 'control\n+ 上限 2.75', 'hybrid': 'hybrid', 'hybrid_cap': 'hybrid\n+ 上限 2.75',
            'trajectory': 'trajectory'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))
    w = 0.34
    x = np.arange(len(sets))
    for k, n in enumerate(sorted(t)):
        c = pz.COLORS['tcp']
        a = 1.0 if k == 0 else 0.55
        v = [t[n]['settings'][s]['L1_human_clean_us'] for s in sets]
        axes[0].bar(x + (k - 0.5) * w, v, width=w - 0.02, color=c, alpha=a, label='種 %d' % k)
        e = [t[n]['settings'][s]['r1_entry_est_median_erpm'] for s in sets]
        axes[1].bar(x + (k - 0.5) * w, e, width=w - 0.02, color=c, alpha=a, label='種 %d' % k)
    act = res['r1']['entry_estimate']['actual']['median_erpm']
    axes[1].axhline(act, color=INK2, lw=1.5)
    axes[1].text(len(sets) - 0.5, act, '実際の進入(中央値)%d' % act, ha='right', va='bottom', color=INK2, fontsize=12)
    axes[1].axhline(5500, color=INK2, lw=1, ls=(0, (1, 3)))
    axes[1].text(-0.4, 5500, '5,500 eRPM(上限 2.75 m/s)', va='bottom', color=INK2, fontsize=12)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([labs[s] for s in sets])
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('人のきれいな走行の L1 [µs]')
    axes[0].set_title('tcp の出し方ごとの人との近さ')
    axes[1].set_ylabel('見積もった R1 の進入 eRPM(中央値)')
    axes[1].set_title('R1 の進入の見積もり(17 通過、参考)')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'eval_tcp_modes.png', dpi=100)
    plt.close('all')


def make_all(res, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_primary, fig_r1_de, fig_speed, fig_mae_k, fig_robust, fig_tcp):
        f(res, out)
        print('図:', f.__name__)
