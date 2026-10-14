#!/usr/bin/env python3
# no_wall 確認 r4 の図: 組ごとの「回の最小すき間」(cm)。左 = 標準の物理(判定の線 8 cm)、右 = 物理のずれ(判定の線 5 cm)。
# 青の点 = 1 回、橙の × = 抜け出しで後ろへ動いた回(衝突・止まりがあれば赤の印と数を書く)。使い方: python3 fig_ck4.py <png> <runs のフォルダー>...
import json, glob, sys, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ['Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'TakaoPGothic', 'VL PGothic']:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
out, dirs = sys.argv[1], sys.argv[2:]
rows = []
for d in dirs:
    for fn in glob.glob(f'{d}/rows_*.jsonl'):
        rows += [json.loads(l) for l in open(fn) if l.strip()]
rows = [r for r in rows if not r.get('error') and not r.get('invalid_spawn')]
isdr = lambda r: bool(r.get('phys')) or r.get('fc') or r.get('delay')
STD = ['G17F', 'G2023', 'CORN', 'LOWC', 'LS', 'U', 'LONG', 'APP', 'UOFF', 'FAST']
DR = ['HCL', 'VXALL', 'HC', 'DIN', 'DRL', 'VXC', 'VXS', 'DRU', 'VVN', 'DLS', 'LONG', 'DELAY2', 'LOWMU']
NOTE = {'APP': 'APP(分布の外・報告)', 'UOFF': 'UOFF(シールドなし・報告)', 'FAST': 'FAST(最速・報告)', 'DELAY2': 'DELAY2(範囲の外)', 'LOWMU': 'LOWMU(範囲の外)'}
SURF, INK, MUTED, BLUE, ORANGE, RED = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#c62828'
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor=SURF)
rnd = random.Random(1)
for ax, names, dr, thr, title in [(axes[0], STD, False, 8, '標準の物理(記録の物理)'), (axes[1], DR, True, 5, '物理のずれ(DR-T の範囲の中)')]:
    ax.set_facecolor(SURF)
    names = [n for n in names if any(r['set'] == n and bool(isdr(r)) == dr for r in rows)]
    for i, n in enumerate(names):
        L = [r for r in rows if r['set'] == n and bool(isdr(r)) == dr]
        xs = [max(-1, min(20, 100 * r['min_clearance'])) for r in L]
        ys = [i + rnd.uniform(-.3, .3) for _ in L]
        ax.scatter(xs, ys, s=8, color=BLUE, alpha=.35, linewidths=0)
        rv = [(x, y) for x, y, r in zip(xs, ys, L) if r.get('reverse_frames', 0) > 0]
        if rv:
            ax.scatter([a for a, _ in rv], [b for _, b in rv], s=40, marker='x', color=ORANGE, linewidths=1.5)
        nc = sum(1 for r in L if r['collisions'] > 0)
        ns = sum(1 for r in L if r.get('stall'))
        lab = f"{len(L)} 回・最小 {min(100 * r['min_clearance'] for r in L):.2f} cm"
        if nc: lab += f"・衝突 {nc}"
        if ns: lab += f"・止まり {ns}"
        ax.text(20.6, i, lab, va='center', fontsize=8, color=RED if (nc or ns) and n not in NOTE else MUTED)
    ax.axvline(thr, color=MUTED, lw=1, ls='--')
    ax.text(thr, len(names) - .4, f' 判定の線 {thr} cm', color=MUTED, fontsize=8, va='bottom')
    if not dr:
        ax.axvline(8.1, color=MUTED, lw=.6, ls=':')
    ax.set_yticks(range(len(names)), [NOTE.get(n, n) for n in names], fontsize=9, color=INK)
    ax.set_xlim(-1, 31); ax.set_xticks([0, 5, 10, 15, 20]); ax.set_ylim(-.7, len(names) - .3)
    ax.set_xlabel('回の最小すき間 [cm](自前の幾何。0 未満 = 衝突、20 以上は 20 に寄せた)', fontsize=9, color=INK)
    ax.set_title(title, fontsize=11, color=INK, loc='left')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='x', color='#e6e5e0', lw=.6)
fig.suptitle('no_wall r4 の確認(2026-10-15): 組ごとの回の最小すき間。橙の × = 抜け出しで後ろへ動いた回', fontsize=12, color=INK, x=.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(out, dpi=130, facecolor=SURF)
print('saved', out)
