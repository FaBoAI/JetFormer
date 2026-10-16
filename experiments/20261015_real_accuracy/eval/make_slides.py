#!/usr/bin/env python3
"""約 1 分の報告動画の図(video/fig/v1〜v5)を evaluation.json から作る(CPU・matplotlib)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python experiments/20261015_real_accuracy/eval/make_slides.py

日本語・文字は大きめ(動画の中で読めるように 22 pt 以上)・2160×1350。数字はすべて evaluation.json と、
これまでの報告(policy_zoo_20261012・sim_cotrain_20261013・sim_diverse_train_20261015)の値。
"""
import json
from pathlib import Path

import numpy as np

import realacc_figures as F

REPORT = Path('/home/jetson/jetformer_reports/real_accuracy_20261015')
OUT = REPORT / 'video' / 'fig'
CLS = {'better': '良い', 'equivalent': '同等', 'worse': '悪い', None: '—'}
# これまでの比較(保留の主指標・best・2 つの種の平均。各報告の evaluation.json の値)
PAST = [('今の作り方(A)', 49.30), ('多段の融合', 49.30), ('TCP型', 49.73), ('ACT型', 50.38), ('DINOv2型', 50.40),
        ('オフライン強化学習', None), ('シミュ混合(C)', 49.73), ('C+微調整', 49.71), ('いろいろな走り', 50.04)]


def setup():
    plt = F._setup()
    plt.rcParams.update({'font.size': 24, 'axes.titlesize': 30, 'axes.labelsize': 24, 'xtick.labelsize': 22,
                         'ytick.labelsize': 22})
    return plt


def v1_problem(res):
    plt = setup()
    rows = [(n, v) for n, v in PAST if v is not None]
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    for k, (n, v) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.scatter([v], [y], s=320, color=F.BLUE if k == 0 else F.GRAY, zorder=3)
        ax.text(v + 0.12, y, '%.1f' % v, va='center', fontsize=22, color=F.INK)
    ax.axvspan(49.30 - 0.626, 49.30 + 0.626, color=F.BLUE, alpha=0.1, lw=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows][::-1])
    ax.set_xlim(48.0, 51.5)
    ax.set_xlabel('実機の保留データの誤差 [µs](小さいほど良い)')
    ax.set_title('これまで: 形を変えても、混ぜても、誤差は約 49 µs のまま')
    fig.tight_layout()
    return F._save(fig, OUT, 'v1_problem.png')


def v2_plan(res):
    plt = setup()
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    ax.axis('off')
    items = [('平均(ENS)', '形の違う 2 本の予測を平均する(新しい種で確かめる)', F.ORANGE),
             ('電池の補正(GAIN)', '電池が減ってスロットルの効きが落ちた分を、直前 5 秒から見積もって補う', F.GREEN),
             ('時間の平均(TE)', '1・2 フレーム前の予測も混ぜてならす', F.GRAY),
             ('位相を回す(ROT)', '間引きで捨てていた 2/3 のフレームも、エポックごとに使う', F.PURPLE),
             ('データの量(SIZE)', '学習データを 29%・52% に減らして、増やしたときの伸びを見積もる', F.BLUE)]
    for k, (t, d, c) in enumerate(items):
        y = 0.86 - k * 0.18
        ax.add_patch(plt.Rectangle((0.02, y - 0.06), 0.012, 0.12, color=c, transform=ax.transAxes))
        ax.text(0.05, y + 0.02, t, fontsize=30, color=F.INK, transform=ax.transAxes, va='center', weight='bold')
        ax.text(0.05, y - 0.045, d, fontsize=22, color=F.INK2, transform=ax.transAxes, va='center')
    ax.set_title('今回試した 5 つ(判定の規則は学習の前に固定)')
    return F._save(fig, OUT, 'v2_plan.png')


def v3_result(res):
    plt = setup()
    pv, Q = res['per_variant'], res['questions']
    rows = [('今の作り方(A)', ['A_s0', 'A_s1'], None, F.BLUE),
            ('平均(ENS)', ['P2', 'P3'], 'ENS_vs_A', F.ORANGE),
            ('同じ形の平均', ['SAME_A', 'SAME_new'], 'SAME_vs_A', F.GREEN),
            ('電池の補正(新しい種)', ['GAIN(v0-s2)', 'GAIN(v0-s3)'], 'GAIN_ii', F.GREEN),
            ('時間の平均', ['TE(A_s0)', 'TE(A_s1)'], 'TE_i', F.GRAY),
            ('位相を回す', ['rot-s0', 'rot-s1'], 'ROT_vs_A', F.PURPLE)]
    rows = [r for r in rows if all(v in pv for v in r[1])]
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    a = np.mean([pv['A_s0']['best']['primary'], pv['A_s1']['best']['primary']])
    ax.axvspan(a - 0.626, a + 0.626, color=F.BLUE, alpha=0.1, lw=0)
    ax.axvline(a, color=F.BLUE, ls='--', lw=2)
    xs = []
    for k, (n, vs, qid, c) in enumerate(rows):
        y = len(rows) - 1 - k
        v = [pv[x]['best']['primary'] for x in vs]
        xs += v
        ax.scatter(v, [y] * len(v), s=320, color=c, zorder=3)
        if qid and not Q[qid]['pending']:
            q = Q[qid]
            ref = q['primary']['best']['ref_mean']
            ax.text(max(v) + 0.3, y, '%s %+.1f µs' % (CLS[q['final_class']], np.mean(v) - ref), va='center', fontsize=24,
                    color=F.INK, bbox=F.BOX)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(min(xs) - 1, max(xs) + 4.5)
    ax.set_xlabel('実機の保留データの誤差 [µs](点 = 2 つの種)')
    ax.set_title('結果(帯の外で、2 つの種がそろえば「良い」)')
    fig.tight_layout()
    return F._save(fig, OUT, 'v3_result.png')


def v4_size(res):
    s = res.get('size')
    if not s:
        return None
    plt = setup()
    pv = res['per_variant']
    N = s['N']
    pts = [(k, vs) for k, vs in (('Q', ['sizeQ-s0', 'sizeQ-s1']), ('H', ['sizeH-s0', 'sizeH-s1']),
                                 ('T', ['sizeT-s0', 'sizeT-s1']), ('full', ['A_s0', 'A_s1'])) if all(v in pv for v in vs)]
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    x = [N[k] for k, _ in pts]
    y = [np.mean([pv[v]['best']['primary'] for v in vs]) for _, vs in pts]
    ax.plot(x, y, '-o', color=F.BLUE, lw=4, ms=18)
    lab = {'Q': '29%', 'H': '52%', 'T': '79%', 'full': '全部'}
    for (k, _), xx, yy in zip(pts, x, y):
        ax.text(xx, yy + 0.25, '%s\n%.1f' % (lab[k], yy), ha='center', va='bottom', fontsize=22)
    ax.set_xscale('log', base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(['%d' % v for v in x])
    ax.set_ylim(min(y) - 1.5, max(y) + 2.5)
    m = s['by_metric']['primary/best']['slopes']
    ax.set_title('データを 2 倍にすると: 約 %.1f µs 良くなる見込み' % max(m['G2_pow'], 0))
    ax.set_xlabel('学習に使った実機のサンプル数')
    ax.set_ylabel('誤差 [µs]')
    fig.tight_layout()
    return F._save(fig, OUT, 'v4_size.png')


def v5_ceiling(res):
    plt = setup()
    proto = json.loads((REPORT / 'protocol.json').read_text(encoding='utf-8'))
    exp = proto['expectations_from_ceiling']
    pv = res['per_variant']
    items = [('今の作り方', np.mean([pv['A_s0']['best']['primary'], pv['A_s1']['best']['primary']]), F.BLUE)]
    for n, vs, c in (('平均(ENS)', ['P2', 'P3'], F.ORANGE), ('平均 + 電池の補正', ['GAIN(P2)', 'GAIN(P3)'], F.GREEN)):
        if all(v in pv for v in vs):
            items.append((n, float(np.mean([pv[v]['best']['primary'] for v in vs])), c))
    fig, ax = plt.subplots(figsize=F.FIGSIZE)
    lo, hi = exp['floor_us']['range']
    ax.axvspan(lo, hi, color=F.GREEN, alpha=0.12, lw=0)
    ax.text((lo + hi) / 2, len(items) - 0.3, '人の走りのばらつき\nから見た底 %d〜%d' % (lo, hi), ha='center', fontsize=24,
            color=F.GREEN, bbox=F.BOX)
    for k, (n, v, c) in enumerate(items):
        y = len(items) - 1 - k
        ax.scatter([v], [y], s=420, color=c, zorder=3)
        ax.text(v + 0.8, y, '%.1f' % v, va='center', fontsize=24)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([x[0] for x in items][::-1])
    ax.set_xlim(15, 56)
    ax.set_ylim(-0.8, len(items) + 0.6)
    ax.set_xlabel('実機の保留データの誤差 [µs]')
    ax.set_title('底(人のばらつき)まで、まだ距離がある')
    fig.tight_layout()
    return F._save(fig, OUT, 'v5_ceiling.png')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((REPORT / 'evaluation.json').read_text(encoding='utf-8'))
    for fn in (v1_problem, v2_plan, v3_result, v4_size, v5_ceiling):
        p = fn(res)
        if p:
            print(p)
