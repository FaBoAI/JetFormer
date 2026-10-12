"""(動画 2 枚目)結果の一覧: 主指標(± δ の帯の中に全部が入る)・推論の速さ・ばらつきとの比べ。
数字は eval/zoo_eval.json から読む(手で写さない)。横軸は v0 ± δ を全部見せる(差を大きく見せない)。"""
import numpy as np

from zoolib import *  # noqa: F401,F403
from zoolib import Canvas, load, PCOL, PNAME, POLICIES, INK, INK2, MUTED, GRID, GREEN, RED, tint

D = load()
P = D['policy']
ev = D['ev']
delta = D['delta']
v0m = P['jetformer_v0']['L_mean']

c = Canvas()
c.title('結果: 新しい 4 つは v0 を超えなかった',
        '保留した人の走行(11 シーン・4,670 フレーム)で、人の操作との差(主指標 L1)を比べた。各方策を種 0・種 1 の 2 本ずつ学習')

rows = ['B'] + list(POLICIES)
labels = {'B': 'モデル B(今の走行)', 'jetformer_v0': 'TransFuser軽量型(基準)', 'act': 'ACT型', 'tcp': 'TCP型',
          'msfusion': 'TransFuser多段型', 'dinov2': 'DINOv2型'}
ys = {k: len(rows) - 1 - i for i, k in enumerate(rows)}

# ---- 左: 主指標
ax = c.fig.add_axes([0.205, 0.335, 0.43, 0.545])
ax.axvspan(v0m - delta, v0m + delta, color=tint(PCOL['jetformer_v0'], 0.10), zorder=0)
ax.plot([v0m, v0m], [-0.7, len(rows) - 0.85], color=PCOL['jetformer_v0'], lw=1.6, zorder=1)
for k in rows:
    y = ys[k]
    col = PCOL[k]
    if k == 'B':
        vals = [D['B']['L']]
        mean = vals[0]
    else:
        vals = P[k]['L_seeds']
        mean = P[k]['L_mean']
    ax.scatter(vals, [y] * len(vals), s=140, color=col, edgecolor='white', linewidth=1.5, zorder=4)
    if k != 'B':
        ax.plot([mean, mean], [y - 0.28, y + 0.28], color=INK, lw=2.6, zorder=5)
    if k in ('B', 'jetformer_v0'):
        lab = '%.1f µs' % mean
    else:
        lab = '%.1f µs(%+.1f)' % (mean, P[k]['diff'])
    ax.text(max(vals) + 0.9, y, lab, va='center', ha='left', fontsize=17, color=INK, zorder=6)
ax.set_yticks([ys[k] for k in rows])
ax.set_yticklabels([labels[k] for k in rows], fontsize=17)
for t, k in zip(ax.get_yticklabels(), rows):
    t.set_color(INK)
ax.set_ylim(-0.7, len(rows) - 0.4)
ax.set_xlim(v0m - delta - 1.5, v0m + delta + 1.5)
ax.set_xticks(np.arange(40, 61, 5))
ax.tick_params(axis='x', labelsize=16)
ax.set_xlabel('主指標 L1 [µs](小さいほど人の運転に近い)', fontsize=17, color=INK2)
ax.grid(axis='x', color=GRID, lw=1.0)
ax.set_axisbelow(True)
ax.text(v0m - delta + 0.4, len(rows) - 0.62, '帯 = v0 ± 10 µs(これより小さい差は区別しない)', fontsize=16, color=INK2,
        ha='left', va='center', zorder=7, bbox=dict(fc=tint(PCOL['jetformer_v0'], 0.10), ec='none', pad=1.5))
c.txt(1.2, 19.6, '● = 学習した 1 本(種 0・種 1)、縦の棒 = 2 本の平均。モデル B = 旧コードで学習した今の走行のモデル(1 本)',
      size=16, color=INK2, ha='left')

# ---- 右: 推論の速さ
ax2 = c.fig.add_axes([0.715, 0.335, 0.255, 0.545])
for k in rows:
    y = ys[k]
    ms = D['B']['ms'] if k == 'B' else P[k]['ms']
    ax2.barh(y, ms, height=0.5, color=tint(PCOL[k], 0.75), edgecolor=PCOL[k], lw=1.5, zorder=3)
    ax2.text(ms + 0.15, y, '%.1f' % ms, va='center', ha='left', fontsize=17, color=INK, zorder=6)
ax2.axvline(6.0, color=RED, lw=1.6, ls=(0, (5, 3)), zorder=2)
ax2.text(5.9, len(rows) - 0.62, 'ゲート 6 ms', fontsize=16, color=RED, ha='right', va='center')
ax2.set_yticks([ys[k] for k in rows])
ax2.set_yticklabels([''] * len(rows))
ax2.tick_params(axis='y', length=0)
ax2.set_ylim(-0.7, len(rows) - 0.4)
ax2.set_xlim(0, 7.0)
ax2.set_xticks([0, 2, 4, 6])
ax2.tick_params(axis='x', labelsize=16)
ax2.set_xlabel('推論 1 回 [ms](中央値)', fontsize=17, color=INK2)
ax2.grid(axis='x', color=GRID, lw=1.0)
ax2.set_axisbelow(True)

# ---- 下: 4 つの数字
br = [abs(v) for v in ev['primary']['best_minus_last_us'].values()]
sr = [ev['primary']['seed_range_us'][p]['best'] for p in POLICIES]
diffs = [P[p]['diff'] for p in POLICIES if p != 'jetformer_v0']
r1 = ev['r1']['per_policy']
de = ev['de']['per_policy']
n_good = sum(1 for p in r1 if r1[p]['good'] or de[p]['good'])
items = [
    ('%.1f〜%.1f µs' % (min(diffs), max(diffs)), '方策と v0 の差(主指標)', '区別の幅 δ は %.0f µs' % delta, INK),
    ('%.2f〜%.2f µs' % (min(sr), max(sr)), '同じ方策で種だけ変えた差', 'best と last の差も最大 %.2f' % max(br), INK),
    ('%d / 4' % n_good, 'R1・DE で「良い」', 'v0 の種どうしでも ±8 µs 動く', INK),
    ('0 / 4', '直線で前進が弱まる「崩れ」', 'どの方策にも無い', INK),
]
TW, TH = 32.4, 12.0
for i, (big, l1, l2, col) in enumerate(items):
    c.tile(1.0 + i * (TW + 1.4), 2.6, TW, TH, big, l1, l2, color=col)
c.save('zoo_results_overview.png')
