#!/usr/bin/env python3
"""実機の精度の試し(real_accuracy_20261015)の図(CPU・matplotlib)。evaluation.json → fig/*.png

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_real_accuracy/eval/realacc_figures.py

日本語(Noto Sans CJK JP)・文字は 16 pt 以上・2160×1350 ピクセル(14.4 × 9 インチ × 150 dpi)。
色の印のそばに名前や値を直接書く(色だけで見分けさせない)。
"""

import json
import os
from pathlib import Path

import numpy as np

REPORT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015')
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, GREEN, GRAY, PURPLE, RED = '#2a78d6', '#eb6834', '#1baf7a', '#8d8c88', '#8a6bd1', '#d23f31'
BOX = dict(facecolor=SURF, edgecolor='none', pad=1.5, alpha=0.9)
FIGSIZE, DPI = (14.4, 9.0), 150
CLS = {'better': '良い', 'equivalent': '同等', 'worse': '悪い', None: 'まだ'}


def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 21, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                         'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                         'savefig.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1.0,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def _save(fig, out, name):
    import matplotlib.pyplot as plt
    p = Path(out) / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def fig_primary(res, out):
    """腕ごとの主指標: ● best・○ last(2 つの種)、A の平均 ± 閾値の帯、最終の判定"""
    plt = _setup()
    Q, R = res['questions'], res['reference_questions']
    pv = res['per_variant']
    rows = [('A(基準・種 0・1)', ['A_s0', 'A_s1'], None, BLUE),
            ('モデル B(今の走行)', ['B'], None, GRAY),
            ('FRESH-A(v0 の新しい種 2・3)', ['v0-s2', 'v0-s3'], 'FRESH_A', BLUE),
            ('FRESH-MSF(msfusion 種 2・3)', ['msf-s2', 'msf-s3'], 'FRESH_MSF', PURPLE),
            ('SAME(同じ形 2 本の平均)', ['SAME_A', 'SAME_new'], 'SAME_vs_A', GREEN),
            ('ENS(v0 + msfusion の平均)', ['P2', 'P3'], 'ENS_vs_A', ORANGE),
            ('ROT(位相を回す)', ['rot-s0', 'rot-s1'], 'ROT_vs_A', PURPLE),
            ('GAIN(電池の補正)組 (i)', ['GAIN(A_s0)', 'GAIN(A_s1)'], 'GAIN_i', GREEN),
            ('GAIN 組 (ii)(新しい種)', ['GAIN(v0-s2)', 'GAIN(v0-s3)'], 'GAIN_ii', GREEN),
            ('TE(時間の平均)組 (i)', ['TE(A_s0)', 'TE(A_s1)'], 'TE_i', GRAY),
            ('TE 組 (ii)(新しい種)', ['TE(v0-s2)', 'TE(v0-s3)'], 'TE_ii', GRAY),
            ('GAINL + 補正(任意)', ['GAIN(gainl-s0)', 'GAIN(gainl-s1)'], 'GAINL_vs_Apost', GREEN)]
    rows = [r for r in rows if all(v in pv for v in r[1])]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    a_mean = np.mean([pv['A_s0']['best']['primary'], pv['A_s1']['best']['primary']])
    thr = 0.626
    ax.axvspan(a_mean - thr, a_mean + thr, color=BLUE, alpha=0.10, lw=0)
    ax.axvline(a_mean, color=BLUE, lw=1.5, ls='--')
    xs_all = []
    for k, (lab, vs, qid, col) in enumerate(rows):
        y = len(rows) - 1 - k
        b = [pv[v]['best']['primary'] for v in vs]
        l_ = [pv[v]['last']['primary'] for v in vs]
        xs_all += b + l_
        ax.scatter(b, [y] * len(b), s=130, color=col, zorder=3)
        ax.scatter(l_, [y - 0.22] * len(l_), s=110, facecolors='none', edgecolors=col, lw=2, zorder=3)
        verdict = ''
        if qid:
            q = Q.get(qid) or R.get(qid)
            if q and not q['pending']:
                verdict = '  → %s(%+.2f µs)' % (CLS[q['final_class']], -q['primary']['best']['gain_us'])
        ax.text(max(b + l_) + 0.25, y - 0.1, verdict, va='center', fontsize=16, color=INK, bbox=BOX)
    ax.set_yticks([len(rows) - 1 - k for k in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows])
    lo, hi = min(xs_all), max(xs_all)
    ax.set_xlim(lo - 1.0, hi + 6.5)
    ax.set_xlabel('保留の実機のきれいな走行 4,670 フレームの L1 [µs](小さいほど良い)')
    ax.set_title('腕ごとの主指標(● best・○ last。帯 = A の平均 ± 0.626 µs = 「良い」に要る改善)')
    fig.tight_layout()
    return _save(fig, out, 'eval_primary.png')


def fig_size(res, out):
    """データの量の曲線: 学習サンプル数(対数)と L(2 つの種と平均)"""
    s = res.get('size')
    if not s:
        return None
    plt = _setup()
    pv = res['per_variant']
    N = s['N']
    arms = [('Q', ['sizeQ-s0', 'sizeQ-s1']), ('H', ['sizeH-s0', 'sizeH-s1']), ('T', ['sizeT-s0', 'sizeT-s1']),
            ('full', ['A_s0', 'A_s1'])]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for metric, col, lab in (('primary', BLUE, '主指標(11 シーン)'), ('S1_no_battery_scene', GREEN, '電池のシーンを除く(10 シーン)')):
        xs, ms = [], []
        for k, vs in arms:
            if not all(v in pv for v in vs):
                continue
            vals = [pv[v]['best'][metric] for v in vs]
            ax.scatter([N[k]] * len(vals), vals, s=80, color=col, alpha=0.6, zorder=3)
            xs.append(N[k])
            ms.append(np.mean(vals))
        ax.plot(xs, ms, '-o', color=col, lw=2.5, ms=11, zorder=4)
        ax.text(xs[-1] * 1.04, ms[-1], lab, va='center', fontsize=16, color=col, bbox=BOX)
    eqa = [('Q', ['sizeQeq-s0', 'sizeQeq-s1']), ('H', ['sizeHeq-s0', 'sizeHeq-s1'])]
    xs, ms = [], []
    for k, vs in eqa:
        if all(v in pv for v in vs):
            xs.append(N[k])
            ms.append(np.mean([pv[v]['best']['primary'] for v in vs]))
    if xs:
        xs.append(N['full'])
        ms.append(np.mean([pv[v]['best']['primary'] for v in ('A_s0', 'A_s1')]))
        ax.plot(xs, ms, '--s', color=ORANGE, lw=2.5, ms=11, zorder=4)
        ax.text(xs[0] / 1.04, ms[0], '参考(固定の外): ステップ数をそろえた', va='center', ha='right', fontsize=16, color=ORANGE,
                bbox=BOX)
    if not xs and all(v in pv for v in ('sizeHeq-s0', 'sizeH-s0', 'A_s0')):
        # 10/16 の電源断で種 1 だけ削った: 種 0 だけの参考(同じ種 0 の H(30 エポック)・H(59 エポック)・全部)
        y30, y59, yf = (pv[v]['best']['primary'] for v in ('sizeH-s0', 'sizeHeq-s0', 'A_s0'))
        ax.plot([N['H'], N['full']], [y59, yf], ':s', color=ORANGE, lw=2.5, ms=12, zorder=4)
        ax.annotate('', xy=(N['H'], y59), xytext=(N['H'], y30),
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))
        ax.text(N['Q'], yf - 4.6, '■ 参考(固定の外・種 0 だけ):\nステップ数をそろえた H %.1f(30 エポックは %.1f)' % (y59, y30),
                va='center', ha='left', fontsize=16, color=ORANGE, bbox=BOX)
    names = {'Q': '29%', 'H': '52%', 'T': '79%', 'full': '全部'}
    ax.set_xscale('log', base=2)
    ax.set_xticks([N[k] for k, vs in arms if all(v in pv for v in vs)])
    ax.set_xticklabels(['%s\n%d' % (names[k], N[k]) for k, vs in arms if all(v in pv for v in vs)])
    ax.set_xlim(N['Q'] / (2.2 if xs else 1.25), N['full'] * 1.9)
    m = s['by_metric']['primary/best']
    b = s['bootstrap_primary_best']
    txt = ('2 倍にしたときの見込み(主指標・best)\n'
           's2 = %.2f µs / 2 倍(95%% 区間 %.2f〜%.2f)\nG2_pow = %.2f µs(%.2f〜%.2f)\n読み: %s' % (
               m['slopes']['s2'], *b['G2_lin_ci95'], m['slopes']['G2_pow'], *b['G2_pow_ci95'], s['reading_text']))
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=16, va='top', ha='right', color=INK, bbox=dict(BOX, alpha=0.95))
    ax.set_xlabel('学習に使った実機のサンプル数(間引き 3 の後・対数の目盛り)')
    ax.set_ylabel('保留の L1 [µs](best、小さいほど良い)')
    ax.set_title('実機のきれいな走行のデータの量と精度(点 = 種 0・1、線 = 平均)')
    fig.tight_layout()
    return _save(fig, out, 'eval_size_curve.png')


def fig_ceiling(res, out):
    """天井の図: 今の A・今夜の最良・下調べの 10 本の平均・人の走りのばらつきから見た底(22〜44 µs)"""
    plt = _setup()
    proto = json.loads((REPORT / 'protocol.json').read_text(encoding='utf-8'))
    exp = proto['expectations_from_ceiling']
    pv = res['per_variant']
    items = [('A(今の作り方)', np.mean([pv['A_s0']['best']['primary'], pv['A_s1']['best']['primary']]), BLUE),
             ('モデル B(今の走行)', pv['B']['best']['primary'], GRAY)]
    for lab, vs, col in (('ENS(新しい種)', ['P2', 'P3'], ORANGE), ('ROT', ['rot-s0', 'rot-s1'], PURPLE),
                         ('GAIN 組 (ii)(新しい種)', ['GAIN(v0-s2)', 'GAIN(v0-s3)'], GREEN),
                         ('ENS + GAIN(参考)', ['GAIN(P2)', 'GAIN(P3)'], GREEN)):
        if all(v in pv for v in vs):
            items.append((lab, float(np.mean([pv[v]['best']['primary'] for v in vs])), col))
    items.append(('下調べ: 10 本の平均(見たあと)', exp['achieved_upper_bound_us'], GRAY))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    lo, hi = exp['floor_us']['range']
    ax.axvspan(lo, hi, color=GREEN, alpha=0.12, lw=0)
    ax.axvline(exp['floor_us']['central_normal'], color=GREEN, lw=2, ls='--')
    ax.text(exp['floor_us']['central_normal'], len(items) - 0.35, '人の走りのばらつきから見た底\n%d〜%d µs(中心 約 %.0f µs)' % (
        lo, hi, exp['floor_us']['central_normal']), ha='center', va='bottom', fontsize=16, color=GREEN, bbox=BOX)
    a = items[0][1]
    ax.axvline(a - 10, color=RED, lw=1.5, ls=':')
    ax.text(a - 10, -0.9, '実用上の差 10 µs', ha='center', fontsize=16, color=RED, bbox=BOX)
    for k, (lab, v, col) in enumerate(items):
        y = len(items) - 1 - k
        ax.barh(y, v, color=col, alpha=0.85, height=0.6)
        ax.text(v + 0.4, y, '%.1f µs' % v, va='center', fontsize=16, color=INK)
    ax.set_yticks([len(items) - 1 - k for k in range(len(items))])
    ax.set_yticklabels([x[0] for x in items])
    ax.set_xlim(15, max(v for _, v, _ in items) + 7)
    ax.set_ylim(-1.3, len(items) + 0.9)
    ax.set_xlabel('保留の L1 [µs](best、2 つの種の平均。小さいほど良い)')
    ax.set_title('天井との関係: 人の走りのばらつきから見た底と、今の誤差')
    fig.tight_layout()
    return _save(fig, out, 'eval_ceiling.png')


def fig_ens_scenes(res, out):
    """ENS のシーン別: A(2 本の平均)と ENS(P2・P3 の平均)の L1、シーンごとの差"""
    ps = res['reference'].get('per_scene_best') or {}
    if not all(k in ps for k in ('A_s0', 'A_s1', 'P2', 'P3')):
        return None
    plt = _setup()
    n = res['reference']['per_scene_frames']
    scenes = list(n)
    a = np.array([(ps['A_s0'][s] + ps['A_s1'][s]) / 2 for s in scenes])
    e = np.array([(ps['P2'][s] + ps['P3'][s]) / 2 for s in scenes])
    same = None
    if 'SAME_A' in ps and 'SAME_new' in ps:
        same = np.array([(ps['SAME_A'][s] + ps['SAME_new'][s]) / 2 for s in scenes])
    order = np.argsort(a)[::-1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    y = np.arange(len(scenes))[::-1]
    for k, i in enumerate(order):
        yy = y[k]
        ax.plot([e[i], a[i]], [yy, yy], color=GRID, lw=4, zorder=1)
        ax.scatter([a[i]], [yy], s=150, color=BLUE, zorder=3)
        ax.scatter([e[i]], [yy], s=150, color=ORANGE, zorder=3)
        if same is not None:
            ax.scatter([same[i]], [yy], s=90, color=GREEN, marker='D', zorder=3)
        xr = max(a[i], e[i], same[i] if same is not None else a[i])
        ax.text(xr + 1.5, yy, '%+.1f' % (e[i] - a[i]), va='center', fontsize=16, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(['%s(%d)%s' % (scenes[i][-11:], n[scenes[i]], ' 電池' if scenes[i] == '20261010-193638-0003' else '')
                        for i in order])
    ax.scatter([], [], s=150, color=BLUE, label='A(種 0・1 の平均)')
    ax.scatter([], [], s=150, color=ORANGE, label='ENS(P2・P3 の平均)')
    if same is not None:
        ax.scatter([], [], s=90, color=GREEN, marker='D', label='SAME(同じ形の平均)')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlabel('シーン別の L1 [µs](best、括弧はフレーム数、右の数字 = ENS − A)')
    ax.set_title('ENS のシーン別: どのシーンで差が出たか')
    fig.tight_layout()
    return _save(fig, out, 'eval_ens_scenes.png')


def make_all(res, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    ps = []
    for fn in (fig_primary, fig_size, fig_ceiling, fig_ens_scenes):
        p = fn(res, out)
        if p:
            ps.append(p)
    return ps


if __name__ == '__main__':
    res = json.loads((REPORT / 'evaluation.json').read_text(encoding='utf-8'))
    for p in make_all(res, REPORT / 'fig'):
        print(p)
