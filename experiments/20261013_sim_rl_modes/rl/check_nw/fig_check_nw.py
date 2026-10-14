#!/usr/bin/env python3
# no_wall の確認 r1 の図(2026-10-14)。左: 組ごとの衝突の割合(シールドあり・なし)。右: 組ごとの回の最小すき間(衝突は 0 に × で描く)
import json, sys, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

R = sys.argv[1]
OUT = sys.argv[2]
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
S1, S2 = '#2a78d6', '#eb6834'   # 参考の配色の 1 番(青)・2 番(橙)


def rows(name):
    return [json.loads(l) for l in open(f'{R}/rows_{name}.jsonl') if l.strip()]


rep = rows('REPLAY')
fn = [r for r in rows('FNOM_FAST_LOWMU') if r['set'] == 'FNOM']
fdc = rows('FDR_CORNER')
fdr = [r for r in fdc if r['set'] == 'FDR']
cor = [r for r in fdc if r['set'] == 'CORNER']
fdroff = rows('FDROFF')
coroff = rows('CORNEROFF')
probe = json.load(open(f'{R}/probe_g16.json'))
judged = [r for r in rep if r['test'] in ('NW-N60', 'NW-STD', 'NW-G', 'NW-R200', 'NW-FC10')]
off260 = [r for r in rep if r['test'] == 'NW-OFF']

groups = [
    ('試験の組の作り直し(判定 340)', judged, off260),
    ('新しい開始・標準の物理(150)', fn, None),
    ('新しい開始・DR-T 一様(400)', fdr, fdroff),
] + [(f'DR-T の隅 {k}(115)', [r for r in cor if r['corner'] == k], [r for r in coroff if r['corner'] == k]) for k in ['C1', 'C2', 'C3', 'C4']]
pr = [p for p in probe if p['g'] == 16 and p['sr'] <= .85]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURF, gridspec_kw={'width_ratios': [1.15, 1]})
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)

labels = [g[0] for g in groups] + ['ゲート 16・舵 0.8〜0.85・遅延 1(15)']
on = [(sum(1 for r in g[1] if r['collisions']), len(g[1])) for g in groups] + [(sum(1 for p in pr if p['col']), len(pr))]
offv = [(sum(1 for r in g[2] if r['collisions']), len(g[2])) if g[2] else None for g in groups] + [None]
y = list(range(len(labels)))[::-1]
h = .36
for yi, (c, n), o in zip(y, on, offv):
    ax1.barh(yi + h / 2, 100 * c / n, height=h - .04, color=S1)
    ax1.text(100 * c / n + 1, yi + h / 2, f'{c}/{n}', va='center', fontsize=8.5, color=INK)
    if o:
        ax1.barh(yi - h / 2, 100 * o[0] / o[1], height=h - .04, color=S2)
        ax1.text(100 * o[0] / o[1] + 1, yi - h / 2, f'{o[0]}/{o[1]}', va='center', fontsize=8.5, color=INK2)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=9, color=INK)
ax1.set_xlim(0, 100)
ax1.set_xlabel('衝突した回の割合 [%]', color=INK2, fontsize=9)
ax1.xaxis.grid(True, color=GRID, lw=.8)
ax1.set_axisbelow(True)
ax1.set_title('衝突(シールドあり = 青、なし = 橙)', fontsize=11, color=INK, loc='left')
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=S1, label='シールドあり'), Patch(color=S2, label='シールドなし')], loc='upper right', frameon=False, fontsize=9)

# 右: 最小すき間
sets2 = [('試験 N60+STD', [r for r in judged if r['test'] in ('NW-N60', 'NW-STD')]),
         ('試験 G+R200+FC', [r for r in judged if r['test'] in ('NW-G', 'NW-R200', 'NW-FC10')]),
         ('新・標準 150', fn), ('新・DR-T 400', fdr), ('隅 C1〜C4 460', cor)]
rnd = random.Random(7)
for i, (nm, rs) in enumerate(sets2):
    xs = [i + rnd.uniform(-.28, .28) for _ in rs]
    ys = [max(0, r['min_clearance']) * 100 for r in rs]
    ok = [(x, v) for x, v, r in zip(xs, ys, rs) if not r['collisions']]
    bad = [(x, v) for x, v, r in zip(xs, ys, rs) if r['collisions']]
    ax2.scatter([a for a, _ in ok], [b for _, b in ok], s=9, color=S1, alpha=.55, linewidths=0)
    if bad:
        ax2.scatter([a for a, _ in bad], [b for _, b in bad], s=22, marker='x', color=S2, linewidths=1.2)
    ax2.text(i, 16.6, f'最小 {min(ys):.1f}', ha='center', fontsize=8.5, color=INK2)
for yv, t in ((8, '8 cm(N2)'), (5, '5 cm(N3)')):
    ax2.axhline(yv, color=INK2, lw=1, ls='--')
    ax2.text(-.45, yv + .25, t, fontsize=8.5, color=INK2, ha='left')
ax2.set_xticks(range(len(sets2)))
ax2.set_xticklabels([s[0] for s in sets2], fontsize=8.5, color=INK, rotation=12)
ax2.set_ylim(-.6, 17.5)
ax2.set_ylabel('回の最小すき間 [cm](× = 衝突)', color=INK2, fontsize=9)
ax2.yaxis.grid(True, color=GRID, lw=.8)
ax2.set_axisbelow(True)
ax2.set_title('最小すき間(シールドあり)', fontsize=11, color=INK, loc='left')
fig.suptitle('no_wall r1 の確認: 決めた試験の組は再現(N3 4.69 cm で不合格)、範囲の中の悪い組み合わせでは衝突する', fontsize=11.5, color=INK, x=.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(OUT, dpi=130, facecolor=SURF)
print(OUT)
