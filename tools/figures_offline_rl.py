#!/usr/bin/env python3
"""オフライン強化学習(tools/offline_rl.py)の結果の図。読み取りだけ・CPU だけ

    python tools/offline_rl.py figures --fig-dir ~/jetformer_reports/offline_rl_20261011/fig
    python tools/figures_offline_rl.py <data/rl/ai_mix_20261011> <図の置き場>

色: 特徴を使うもの = 青、特徴を使わない基準 = 灰、人 = 橙、AI = 青(dataviz の既定の並び)。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
GRAY, GRAY_LIGHT = '#8a8984', '#c9c8c2'
RED = '#e34948'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
LOCS = ['R1', 'GM', 'DE', 'other', 'HS', 'LH']
LOC_JA = {'R1': 'R1\n右ヘアピン', 'GM': 'GM\n芝ヘアピン', 'DE': 'DE\nバックストレート', 'other': 'SW\n奥の道',
          'HS': 'HS\nホームストレート', 'LH': 'LH\n暗幕ヘアピン', 'unknown': '不明'}


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 17, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 15,
                         'ytick.labelsize': 15, 'legend.fontsize': 15, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': SURFACE,
                         'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 1.0, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.axisbelow': True})
    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print('図:', path)


NAMES = {'V_feat': '価値 V(特徴・位置なし)= 判定', 'V_feat_loc': '価値 V(特徴+位置)', 'direct_feat': '参考: 特徴で衝突を直接当てる',
         'V_speed': '価値 V(速度だけ)', 'V_loc': '価値 V(速度+位置)', 'speed': '速度だけ', 'loc_rate': '場所ごとの衝突率',
         'direct_speed_loc': '参考: 速度+場所で直接当てる', 'ttc': '記録された ToF の TTC',
         'course_pos_rate': 'コース図 0.5m ごとの衝突率', 'V1_feat': '価値 V(打ち切りを外しただけ)',
         'V_feat_risk': '診断: 罰だけの価値(特徴)', 'V_feat_minus_speed': '診断: V(特徴)− V(速度だけ)'}
USES_FEAT = {'V_feat', 'V_feat_loc', 'direct_feat', 'V1_feat', 'V_feat_risk', 'V_feat_minus_speed'}


def fig_auroc(ev, path):
    plt = _setup()
    rows = [n for n in ('V_feat', 'V_feat_loc', 'V_feat_minus_speed', 'V_feat_risk', 'direct_feat', 'V_speed', 'V_loc',
                        'speed', 'loc_rate', 'direct_speed_loc') if n in ev['scores']]
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.2), gridspec_kw={'width_ratios': [1.7, 1]})
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    for yi, n in zip(y, rows):
        e = ev['scores'][n]
        col = BLUE if n in USES_FEAT else GRAY
        ax.barh(yi, e['auroc'] - 0.5, left=0.5, height=0.62, color=col)
        lo, hi = e['ci95_cluster']
        if lo is not None:
            ax.plot([lo, hi], [yi, yi], color=INK, lw=2)
        if e['auroc'] < 0.5:
            ax.text(min(e['auroc'], lo or 1) - 0.01, yi, '%.2f' % e['auroc'], va='center', ha='right', fontsize=16)
        else:
            ax.text(max(e['auroc'], hi or 0) + 0.01, yi, '%.2f' % e['auroc'], va='center', fontsize=16, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES.get(n, n) for n in rows])
    ax.axvline(0.8, color=RED, lw=2, ls='--')
    ax.axvline(0.5, color=INK2, lw=1.2)
    ax.text(0.805, -0.45, 'ゲートC の基準 0.8', color=RED, fontsize=14, va='center')
    ax.set_xlim(0.2, 0.95)
    ax.set_xlabel('AUROC(0.5 = 当てずっぽう)')
    ax.set_title('「1秒以内に衝突」をどれだけ当てられるか\nAI が前進中の %d フレーム(陽性 %d)、ログ単位の交差推定'
                 % (ev['frames'], ev['positives']), fontsize=18)
    ax.grid(axis='y', visible=False)
    # 右: 同じフレームで TTC と比べる
    ax = axes[1]
    sub = ev.get('ttc_logs') or {}
    names = [n for n in ('V_feat', 'V_feat_risk', 'speed', 'ttc') if n in sub.get('auroc', {})]
    yy = np.arange(len(names))[::-1]
    for yi, n in zip(yy, names):
        a = sub['auroc'][n]
        ax.barh(yi, a - 0.5, left=0.5, height=0.6, color=BLUE if n in USES_FEAT else (AQUA if n == 'ttc' else GRAY))
        ax.text(a - 0.01 if a < 0.5 else a + 0.01, yi, '%.2f' % a, va='center', ha='right' if a < 0.5 else 'left',
                fontsize=16)
    ax.set_yticks(yy)
    ax.set_yticklabels([NAMES.get(n, n).replace('= 判定', '') for n in names])
    ax.axvline(0.8, color=RED, lw=2, ls='--')
    ax.axvline(0.5, color=INK2, lw=1.2)
    ax.set_xlim(0.2, 0.95)
    ax.set_xlabel('AUROC')
    ax.set_title('TTC が記録された 009/010/101\n(%d フレーム、陽性 %d)' % (sub.get('frames', 0), sub.get('positives', 0)),
                 fontsize=18)
    ax.grid(axis='y', visible=False)
    _save(fig, path)


def fig_roc(sc, path):
    plt = _setup()
    y = sc['y'].astype(bool)
    ev = sc['eval_ai'].astype(bool)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for n, col, lab in (('score_V_feat', BLUE, NAMES['V_feat']), ('score_direct_feat', ORANGE, NAMES['direct_feat']),
                        ('score_loc_rate', GRAY, NAMES['loc_rate'])):
        if n not in sc:
            continue
        s = sc[n]
        m = ev & np.isfinite(s)
        o = np.argsort(-s[m])
        yy = y[m][o]
        tpr = np.concatenate([[0], np.cumsum(yy) / max(yy.sum(), 1)])
        fpr = np.concatenate([[0], np.cumsum(~yy) / max((~yy).sum(), 1)])
        ax.plot(fpr, tpr, color=col, lw=2.5, label=lab)
    ax.plot([0, 1], [0, 1], color=GRAY_LIGHT, lw=1.5, ls='--')
    ax.set_xlabel('誤警報の割合(衝突しないフレーム)')
    ax.set_ylabel('捉えた割合(1秒以内に衝突するフレーム)')
    ax.set_title('ROC(AI が前進中のフレーム)')
    ax.legend(loc='lower right', fontsize=14)
    _save(fig, path)


def fig_auroc_by_loc(ev, path):
    plt = _setup()
    fig, ax = plt.subplots(figsize=(15, 6.8))
    series = [('V_feat', BLUE), ('direct_feat', AQUA), ('speed', GRAY)]
    x = np.arange(len(LOCS))
    wdt = 0.26
    for j, (n, col) in enumerate(series):
        if n not in ev['scores']:
            continue
        pl = ev['scores'][n]['per_loc']
        vals = [pl.get(l, {}).get('auroc') for l in LOCS]
        ax.bar(x + (j - 1) * wdt, [v if v is not None else 0 for v in vals], width=wdt - 0.03, color=col,
               label=NAMES[n].replace(' = 判定', ''))
    pl = ev['scores']['V_feat']['per_loc']
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n陽性 %d / %d' % (LOC_JA[l], pl.get(l, {}).get('pos', 0), pl.get(l, {}).get('n', 0))
                        for l in LOCS], fontsize=13)
    ax.axhline(0.5, color=INK2, lw=1.2)
    ax.axhline(0.8, color=RED, lw=2, ls='--')
    ax.set_ylim(0.3, 0.95)
    ax.set_ylabel('AUROC')
    ax.set_title('場所ごとの AUROC(場所の中で、どのフレームが危ないかを当てられるか)')
    ax.legend(loc='upper right', ncol=3, fontsize=14)
    ax.grid(axis='x', visible=False)
    _save(fig, path)


def fig_calibration(ev, path):
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    cal = ev['value_calibration']['variants']
    for n, col, lab in (('feat', BLUE, '特徴・位置なし'), ('speed', GRAY, '速度だけ')):
        if n not in cal or 'all' not in cal[n]:
            continue
        b = cal[n]['all']['bins']
        ax.plot([q['pred'] for q in b], [q['actual'] for q in b], marker='o', ms=9, lw=2.5, color=col,
                label='%s(R² %.2f)' % (lab, cal[n]['all']['r2_vs_fold_const']))
    lim = [0, 11]
    ax.plot(lim, lim, color=GRAY_LIGHT, lw=1.5, ls='--')
    ax.set_xlabel('予測した価値 V [m 相当]')
    ax.set_ylabel('実際の割引リターン [m 相当]')
    ax.set_title('価値の較正(保留した fold、10 分位)')
    ax.legend(fontsize=14, loc='upper left')
    ax = axes[1]
    risk = ev['risk_calibration']
    for n, col, lab in (('V_feat', BLUE, '−V から(特徴・位置なし)'), ('V_speed', GRAY, '−V から(速度だけ)')):
        if n not in risk:
            continue
        b = risk[n]['bins']
        ax.plot([q['pred'] for q in b], [q['actual'] for q in b], marker='o', ms=9, lw=2.5, color=col,
                label='%s Brier %.3f' % (lab, risk[n]['brier']))
    ax.plot([0, 0.4], [0, 0.4], color=GRAY_LIGHT, lw=1.5, ls='--')
    ax.set_xlabel('予測した「1秒以内に衝突」の確率')
    ax.set_ylabel('実際の割合')
    base = risk.get('V_feat', {}).get('brier_base_rate')
    ax.set_title('衝突の確率の較正(基準の Brier %.3f)' % (base or float('nan')))
    ax.legend(fontsize=14, loc='upper left')
    _save(fig, path)


def fig_good_share(san, adv, path):
    plt = _setup()
    t = san['by_loc']['I']
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(LOCS))
    for j, (sn, col, lab) in enumerate((('human', ORANGE, '人の操作'), ('ai', BLUE, 'AI の操作'))):
        vals = [t.get(sn, {}).get(l, {}).get('good_share') for l in LOCS]
        ax.bar(x + (j - 0.5) * 0.38, [100 * (v or 0) for v in vals], width=0.35, color=col, label=lab)
        for xi, v in zip(x, vals):
            if v is not None:
                ax.text(xi + (j - 0.5) * 0.38, 100 * v + 1, '%.0f' % (100 * v), ha='center', fontsize=13)
    ax.axhline(35, color=INK2, lw=1.5, ls='--')
    ax.text(-0.5, 35.6, '全体の目標 35%', ha='left', fontsize=14, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([LOC_JA[l] for l in LOCS], fontsize=14)
    ax.set_ylabel('「良い」の印の割合 [%]')
    ax.set_title('場所ごとの「良い」の割合(方策に使えるフレーム)')
    ax.set_ylim(0, 55)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(axis='x', visible=False)
    _save(fig, path)


def fig_r1(san, path):
    plt = _setup()
    fig, ax = plt.subplots(figsize=(13, 7.8))
    for src, col, lab in ((0, ORANGE, '人'), (1, BLUE, 'AI')):
        ps = [p for p in san['passes'] if p.get('src_window') == src and p.get('r1_window', {}).get('good_share') is not None
              and p.get('entry_erpm') is not None]
        x = np.array([p['entry_erpm'] for p in ps])
        g = np.array([100 * p['r1_window']['good_share'] for p in ps])
        hit = np.array([p['r1_strong'] for p in ps], bool)
        ax.scatter(x[~hit], g[~hit], s=70, color=col, alpha=0.75, edgecolors=SURFACE, linewidths=1.5,
                   label='%s(強い衝突なし)' % lab)
        ax.scatter(x[hit], g[hit], s=170, color=col, marker='X', edgecolors=INK, linewidths=1.2,
                   label='%s(R1 で強い衝突)' % lab)
    for v in (5500, 6200):
        ax.axvline(v, color=INK2, lw=1.2, ls=':')
    ax.set_xlabel('R1 への進入速度 [eRPM](起点の手前 0.4秒の最大)')
    ax.set_ylabel('手前 0.4秒の「良い」の割合 [%]')
    r = san['r1']
    ax.set_title('R1: 速く入った通過ほど「良い」が少ないか(順位相関 AI %s / 人 %s)' % (
        r['ai'].get('spearman_entry_vs_good_share'), r['human'].get('spearman_entry_vs_good_share')), fontsize=17)
    for v, txt in ((5500, '5500'), (6200, '6200')):
        ax.text(v, 103, txt, ha='center', fontsize=13, color=INK2)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    _save(fig, path)


def fig_de(san, path):
    plt = _setup()
    fig, ax = plt.subplots(figsize=(13, 7.8))
    for src, col, lab in ((0, ORANGE, '人'), (1, BLUE, 'AI')):
        ps = [p for p in san['passes'] if p.get('src_window') == src and p.get('de_window')
              and p['de_window'].get('good_share') is not None and p.get('de_yaw_dps') is not None]
        x = np.array([p['de_yaw_dps'] for p in ps])
        g = np.array([100 * p['de_window']['good_share'] for p in ps])
        hit = np.array([p['de_collision'] for p in ps], bool)
        ax.scatter(x[~hit], g[~hit], s=70, color=col, alpha=0.75, edgecolors=SURFACE, linewidths=1.5,
                   label='%s(DE で衝突なし)' % lab)
        ax.scatter(x[hit], g[hit], s=170, color=col, marker='X', edgecolors=INK, linewidths=1.2,
                   label='%s(DE で衝突)' % lab)
    ax.axvline(0, color=INK2, lw=1.2, ls=':')
    ax.set_xlabel('緑マット出口のヨー [deg/s](0.2〜0.8秒後の平均。負 = 右へ回り続けている)')
    ax.set_ylabel('出口から 0.8秒の「良い」の割合 [%]')
    d = san['de']
    ax.set_title('DE: 右へ回り続けた通過ほど「良い」が少ないか(順位相関 AI %s / 人 %s)' % (
        d['ai'].get('spearman_yaw_vs_good_share'), d['human'].get('spearman_yaw_vs_good_share')), fontsize=17)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    _save(fig, path)


def fig_timeline(rl_dir, path, log='20261010-173445', t_from=30.0, t_to=62.0):
    """1本の AI 走行の一部: V と A、「良い」の印、衝突"""
    plt = _setup()
    with np.load(rl_dir / 'frames.npz') as z:
        f = {k: z[k] for k in ('log', 'timestamps_us', 'source', 'loc_code', 'segment_id')}
    with np.load(rl_dir / 'advantages.npz') as z:
        A, I, V = z['A'], z['I'], z['V']
    info = json.loads((Path(rl_dir).parents[1] / 'labels' / (log + '.json')).read_text(encoding='utf-8'))
    m = np.nonzero(f['log'] == log)[0]
    o = m[np.argsort(f['timestamps_us'][m])]
    t = (f['timestamps_us'][o] - info['t0_us']) / 1e6
    w = (t >= t_from) & (t <= t_to)
    o, t = o[w], t[w]
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    ax = axes[0]
    vv = V[o].astype(float)
    vv[f['segment_id'][o] < 0] = np.nan
    ax.plot(t, vv, color=BLUE, lw=2.5, label='価値 V(このログを見ていないモデル)')
    ax.set_ylabel('V [m 相当]')
    ai = f['source'][o] == 1
    ax.fill_between(t, ax.get_ylim()[0] if False else -1, 12, where=~ai, color=GRAY_LIGHT, alpha=0.5, step='mid',
                    label='人が操作')
    ax.set_ylim(-1, 12)
    for c in info['events']['collisions']:
        if t_from <= c['t_s'] <= t_to:
            for a_ in axes:
                a_.axvline(c['t_s'], color=RED, lw=2)
            ax.text(c['t_s'], 11.3, ' 衝突 %s' % c['loc'], color=RED, fontsize=14)
    ax.legend(loc='lower left', fontsize=13)
    ax.set_title('ai_run_007(173445)の一部: 衝突の手前で V が下がり、A が負になるか')
    ax = axes[1]
    a = A[o].astype(float)
    good = I[o] == 1
    ax.bar(t[good], a[good], width=1 / 30, color=AQUA, label='「良い」の印 (I=1)')
    ax.bar(t[~good], a[~good], width=1 / 30, color=GRAY, label='それ以外')
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel('A(1秒の N ステップ)')
    ax.set_xlabel('ログの時刻 [秒]')
    ax.legend(loc='lower left', fontsize=13, ncol=2)
    lo = np.nanpercentile(a, 1) if np.isfinite(a).any() else -3
    ax.set_ylim(min(lo, -1) - 0.2, max(np.nanpercentile(a, 99) if np.isfinite(a).any() else 1, 0.5) + 0.2)
    _save(fig, path)


def fig_event_average(rl_dir, path, before_s=3.0, after_s=0.5):
    """AI の衝突の前後の平均: 価値 V(位置なし)・速度・A。V は衝突の手前で下がるか"""
    plt = _setup()
    with np.load(rl_dir / 'frames.npz') as z:
        f = {k: z[k] for k in ('log', 'timestamps_us', 'source', 'erpm')}
    with np.load(rl_dir / 'advantages.npz') as z:
        A = z['A'].astype(float)
        Vn = z['V_noloc'].astype(float) if 'V_noloc' in z.files else None
        I = z['I']
    if Vn is None:
        return
    bins = np.arange(-before_s, after_s + 1e-9, 1 / 30.0)
    acc = {k: [[] for _ in bins] for k in ('V', 'speed', 'A', 'good')}
    n_ev = 0
    for log in sorted(set(f['log'].tolist())):
        info = json.loads((Path(rl_dir).parents[1] / 'labels' / (log + '.json')).read_text(encoding='utf-8'))
        m = np.nonzero((f['log'] == log) & (f['source'] == 1))[0]
        if not len(m):
            continue
        t = f['timestamps_us'][m]
        for c in info['events']['collisions']:
            if not c['ai']:
                continue
            dt = (t - c['t_us']) / 1e6
            w = (dt >= -before_s - 1 / 60) & (dt <= after_s + 1 / 60)
            if w.sum() < 30:
                continue
            n_ev += 1
            k = np.clip(np.round((dt[w] + before_s) * 30).astype(int), 0, len(bins) - 1)
            rows = m[w]
            for kk, r in zip(k, rows):
                acc['V'][kk].append(Vn[r])
                acc['speed'][kk].append(-f['erpm'][r] * 0.5e-3)
                acc['A'][kk].append(A[r])
                acc['good'][kk].append(float(I[r] == 1) if I[r] >= 0 else np.nan)
    mean = {k: np.array([np.nanmean(x) if len(x) else np.nan for x in v]) for k, v in acc.items()}
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, k, col, lab in ((axes[0], 'V', BLUE, '価値 V(位置なし)[m 相当]'), (axes[1], 'speed', GRAY, '速度 [m/s]'),
                            (axes[2], 'A', AQUA, 'A(1秒先までの実際 − V)')):
        ax.plot(bins, mean[k], color=col, lw=3)
        ax.set_ylabel(lab, fontsize=15)
        ax.axvline(0, color=RED, lw=2)
        ax.axvline(-1, color=INK2, lw=1.2, ls=':')
    axes[2].axhline(0, color=INK2, lw=1)
    axes[0].set_title('AI の衝突 %d 件の前後の平均(0 = 衝突。点線 = 1秒前)' % n_ev)
    axes[2].set_xlabel('衝突からの時間 [秒]')
    _save(fig, path)


def make_all(rl_dir, fig_dir):
    rl_dir, fig_dir = Path(rl_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ev = json.loads((rl_dir / 'value_eval.json').read_text(encoding='utf-8'))
    fig_auroc(ev, fig_dir / 'value_auroc.png')
    fig_auroc_by_loc(ev, fig_dir / 'value_auroc_by_loc.png')
    fig_calibration(ev, fig_dir / 'value_calibration.png')
    if (rl_dir / 'value_eval_scores.npz').exists():
        with np.load(rl_dir / 'value_eval_scores.npz') as z:
            fig_roc({k: z[k] for k in z.files}, fig_dir / 'value_roc.png')
    if (rl_dir / 'sanity.json').exists():
        san = json.loads((rl_dir / 'sanity.json').read_text(encoding='utf-8'))
        adv = json.loads((rl_dir / 'advantages.json').read_text(encoding='utf-8'))
        fig_good_share(san, adv, fig_dir / 'good_share_by_loc.png')
        fig_r1(san, fig_dir / 'r1_entry_vs_good.png')
        fig_de(san, fig_dir / 'de_yaw_vs_good.png')
    if (rl_dir / 'advantages.npz').exists():
        fig_timeline(rl_dir, fig_dir / 'timeline_173445.png')
        fig_event_average(rl_dir, fig_dir / 'collision_event_average.png')


if __name__ == '__main__':
    make_all(sys.argv[1], sys.argv[2])
