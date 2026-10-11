"""図: 4本の方策の学習データの内訳と、場所ごとの「目標の偏り」(target_shift.json)。CPU だけ。
出力: ../../fig/policy_data_mix.png, ../../fig/policy_target_shift.png
"""
import json
import os
import sys

import numpy as np

J = '/home/jetson/jetformer'
sys.path.insert(0, J + '/tools')
import figures_offline_rl as fo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', '..', 'fig')
sets = json.load(open(J + '/data/rl/ai_mix_20261011/policy/policy_sets.json', encoding='utf-8'))
shift = json.load(open(os.path.join(HERE, '..', 'target_shift.json'), encoding='utf-8'))
plt = fo._setup()

# 1) 学習サンプルの内訳
c = sets['counts']
hum = c['c']['train']['human']
ai_clean = c['c']['train']['ai']
added = c['b_added_vs_c']['train']['samples_stride']
labels = ['(c) filtered BC', '(b) AWR', '(a) 印つき', '(n) 陰性対照']
fig, ax = plt.subplots(figsize=(14, 6.2))
y = np.arange(4)[::-1]
for i, v in enumerate(['c', 'b', 'a', 'n']):
    yy = y[i]
    parts = [(hum, fo.BLUE, '人(きれいな走行)'), (ai_clean, fo.AQUA, 'AI(失敗の前後以外)')]
    if v != 'c':
        parts.append((added, fo.RED, 'AI の失敗の前後(衝突の1秒前〜2秒後・交代の2秒前)'))
    x0 = 0
    for n, col, lab in parts:
        ax.barh(yy, n, left=x0, color=col, height=0.62, label=lab if i == 1 else None)
        x0 += n
    txt = '%d' % x0
    if v in ('a', 'n'):
        g = c[v]['train']['good_samples']
        txt += '(うち「〜・良」で学習 %d = %.0f%%)' % (g, 100 * g / x0)
    if v == 'b':
        txt += '(重み w=clip(exp(A/β),0,20)、β=%.2f)' % sets['awr']['beta']
    ax.text(x0 + 120, yy, txt, va='center', fontsize=14, color=fo.INK)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(0, 19000)
ax.set_xlabel('学習サンプル数(間引き 1/3 の後。検証は (c) %d、ほか %d)' % (c['c']['val']['samples_stride'], c['b']['val']['samples_stride']))
ax.set_title('4本の方策の学習データ(初期値はどれもモデル B。違うのはデータの使い方だけ)')
ax.legend(frameon=False, loc='upper right', fontsize=13)
ax.grid(axis='y', visible=False)
fo._save(fig, os.path.join(FIG, 'policy_data_mix.png'))

# 2) 目標の偏り(学習データの未来10フレームの平均。(c) との差)
locs = [('all/all', '全体'), ('R1/all', 'R1'), ('DE/all', 'DE'), ('GM/all', 'GM'), ('HS/all', 'HS'), ('LH/all', 'LH')]
series = [('b_minus_c', '(b) AWR(重み付き)', fo.ORANGE), ('a_good_minus_c', '(a) 「〜・良」の目標', fo.BLUE),
          ('n_good_minus_c', '(n) 陰性対照の「〜・良」', fo.RED)]
fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
xx = np.arange(len(locs))
wdt = 0.26
for j, (key, lab, col) in enumerate(series):
    thr = [shift['train'][k][key][1] for k, _ in locs]
    st = [shift['train'][k][key][0] for k, _ in locs]
    axes[0].bar(xx + (j - 1) * wdt, thr, wdt, color=col, label=lab)
    axes[1].bar(xx + (j - 1) * wdt, st, wdt, color=col, label=lab)
for ax in axes:
    ax.axhline(0, color=fo.INK2, lw=1)
    ax.set_xticks(xx)
    ax.set_xticklabels([n for _, n in locs])
    ax.grid(axis='x', visible=False)
axes[0].set_title('スロットルの目標の (c) との差 [µs]\n+ = 中立寄り(遅い)、− = 前進寄り(速い)')
axes[1].set_title('ステアの目標の (c) との差 [µs]\n+ = 左、− = 右')
axes[0].legend(frameon=False, fontsize=13)
fig.text(0.5, 0.012, '学習データ(train)の未来10フレームの操作の平均。方策の出力ではなく「何をまねるように仕向けたか」。'
         '(b) の差には、足した失敗の前後のフレーム自体の違いも入る', ha='center', fontsize=13, color=fo.INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(FIG, 'policy_target_shift.png'), dpi=110)
print('図:', os.path.join(FIG, 'policy_target_shift.png'))
