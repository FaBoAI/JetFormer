#!/usr/bin/env python3
# no_wall の確認 r2 の図: 組ごとの衝突した回の割合(シールドあり / なし)と、回の最小すき間の分布(5 cm・8 cm の線)
#   python3 fig_check_nw_r2.py <runs> <out.png>
import json, sys, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

RUNS, OUT = sys.argv[1], sys.argv[2]
for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf']:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
        break
by = defaultdict(list)
for p in glob.glob(os.path.join(RUNS, 'rows_*.jsonl')):
    if any(p.endswith(f'_{k}.jsonl') for k in range(8)):
        continue
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if not r.get('error') and not r.get('invalid_spawn'):
                key = r['set']
                if key == 'REPLAY2':
                    key = 'REPLAY2-OFF' if r['test'] == 'NW-OFF' else 'REPLAY2'
                if key == 'LONG':
                    key = 'LONG-N' if r['id'].startswith('LN') else 'LONG-DR'
                if key == 'ABX':
                    if not r['id'].startswith('K0_'):
                        continue
                    key = 'K0'
                if key == 'ABXOFF':
                    key = 'K0OFF'
                by[key].append(r)
groups = [('REPLAY2', 'REPLAY2-OFF', '試験 r2\n(判定 340)'), ('FN3', 'FN3OFF', '標準の物理\n新しい開始 600'), ('NG', None, '標準の物理\n開始の格子'),
          ('FD3', 'FD3OFF', 'DR-T 一様\n新しい開始 600'), ('VX', 'VXOFF', 'DR-T の頂点\n480'), ('LONG-N', None, '120 秒\n標準'), ('LONG-DR', None, '120 秒\nDR-T'),
          ('K0', 'K0OFF', '範囲の中の隅 K0\n33 ゲート'), ('D2', None, '範囲の外\n遅延 2')]
groups = [g for g in groups if by.get(g[0])]
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C_ON, C_OFF = '#2a78d6', '#eb6834'
fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), facecolor=SURF)
ax = axes[0]
ax.set_facecolor(SURF)
x = range(len(groups))
w = .36
for i, (on, off, lab) in enumerate(groups):
    rs = by[on]
    k = sum(1 for r in rs if r['collisions'])
    ax.bar(i - w / 2 - .01, 100 * k / len(rs), w, color=C_ON, label='シールドあり' if i == 0 else None)
    ax.text(i - w / 2 - .01, 100 * k / len(rs) + .3, f'{k}/{len(rs)}', ha='center', va='bottom', fontsize=8.5, color=INK)
    if off and by.get(off):
        ro = by[off]
        ko = sum(1 for r in ro if r['collisions'])
        ax.bar(i + w / 2 + .01, 100 * ko / len(ro), w, color=C_OFF, label='シールドなし(方策だけ)' if i == 0 else None)
        ax.text(i + w / 2 + .01, 100 * ko / len(ro) + .3, f'{ko}/{len(ro)}', ha='center', va='bottom', fontsize=8.5, color=INK2)
ax.set_xticks(list(x))
ax.set_xticklabels([g[2] for g in groups], fontsize=9, color=INK)
ax.set_ylabel('衝突した回の割合 [%]', color=INK2)
ax.set_title('no_wall r2(方策 153c709c…・シールド v3)— 衝突した回の割合', fontsize=11, color=INK, loc='left')
ax.grid(axis='y', color=GRID, lw=.8)
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, frameon=False, fontsize=9, loc='upper left')
ax = axes[1]
ax.set_facecolor(SURF)
data = [[100 * max(0, r['min_clearance']) for r in by[g[0]]] for g in groups]
bp = ax.boxplot(data, positions=list(x), widths=.45, whis=(0, 100), patch_artist=True, showfliers=False)
for b in bp['boxes']:
    b.set(facecolor='#d6e6f8', edgecolor=C_ON, lw=1.2)
for k in ['whiskers', 'caps', 'medians']:
    for e in bp[k]:
        e.set(color=C_ON, lw=1.2)
for i, d in enumerate(data):
    m = min(d)
    k = sum(1 for v in d if v <= 0)
    ax.text(i + .27, m + .2, f'最小 {m:.2f}' + (f'\n(衝突 {k})' if k else ''), ha='left', va='bottom', fontsize=8, color=INK)
ax.axhline(8, color=INK2, lw=1, ls='--')
ax.axhline(5, color=INK2, lw=1, ls=':')
ax.text(len(groups) - .5, 8.15, 'N2 8 cm', fontsize=8, color=INK2, ha='right', va='bottom')
ax.text(len(groups) - .5, 5.15, 'N3 5 cm', fontsize=8, color=INK2, ha='right', va='bottom')
ax.set_xticks(list(x))
ax.set_xticklabels([g[2] for g in groups], fontsize=9, color=INK)
ax.set_ylabel('回の最小すき間 [cm](シールドあり)', color=INK2)
ax.set_title('回の最小すき間の分布(箱 = 四分位、ひげ = 最小〜最大。自前の形で測った値。衝突した回は 0 cm として描いた)', fontsize=11, color=INK, loc='left')
ax.grid(axis='y', color=GRID, lw=.8)
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=110, facecolor=SURF)
print(OUT)
