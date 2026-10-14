#!/usr/bin/env python3
# 確認 r3 の図: 組ごとの「回の最小すき間」の分布(点)と判定のしきい値(8 cm・5 cm)。衝突・止まりの回は × と文字で示す(色だけに頼らない)。
# 使い方: python3 fig_ck3.py <runs のフォルダー>... <出力 png>
import json, glob, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf']:
    try:
        font_manager.fontManager.addfont(f); plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name(); break
    except Exception:
        pass
dirs, out = sys.argv[1:-1], sys.argv[-1]
rows = [json.loads(l) for d in dirs for f in glob.glob(f'{d}/rows_*.jsonl') for l in open(f) if l.strip()]
rows = [r for r in rows if 'error' not in r and r.get('mode') == 'no_wall']
ORDER = [('FS', '標準・新しい開始 1000'), ('LOWC', '標準・開始のすき間 12〜15 cm'), ('NBR', '標準・5.6 cm の開始のまわり'), ('EDGE2', '標準・分布の角の格子'),
         ('FSOFF', '標準・シールドなし(報告)'), ('DRU', 'DR-T 一様'), ('VXALL', 'DR-T の頂点 512 通り'), ('HCALL', '止まりにくい隅 64 通り'),
         ('HCIN', '隅の近くの内側'), ('VXOFF', '頂点・シールドなし(報告)'), ('HCOFF', '隅・シールドなし(報告)'), ('LONG', '120 秒の連続周回')]
SURF, INK, MUTED, GRID, S1 = '#fcfcfb', '#0b0b0b', '#52514e', '#dcdbd6', '#2a78d6'
fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURF)
ax.set_facecolor(SURF)
labels = []
for i, (s, name) in enumerate(ORDER):
    g = [r for r in rows if r['set'] == s]
    if not g: continue
    y = len(labels)
    ok = [100 * r['min_clearance'] for r in g if r['collisions'] == 0 and not r['stall']]
    jit = [(hash(r['id']) % 1000 / 1000 - .5) * .5 for r in g if r['collisions'] == 0 and not r['stall']]
    ax.scatter([min(v, 20) for v in ok], [y + j for j in jit], s=9, color=S1, alpha=.45, linewidths=0)
    bad = [r for r in g if r['collisions'] > 0 or r['stall']]
    for r in bad:
        x = 0 if r['collisions'] > 0 else 100 * r['min_clearance']
        ax.scatter([x], [y], s=60, marker='x', color=INK, linewidths=1.6)
    nc, ns = sum(r['collisions'] > 0 for r in g), sum(r['stall'] for r in g)
    note = f"n={len(g)}  衝突 {nc}・止まり {ns}"
    ax.text(20.4, y, note, va='center', fontsize=8.5, color=INK if (nc or ns) else MUTED)
    labels.append(name)
for x, t, ha in [(8, '8 cm(N2)', 'left'), (5, '5 cm(N3)', 'right')]:
    ax.axvline(x, color=MUTED, lw=1, ls='--'); ax.text(x + (.15 if ha == 'left' else -.15), -.75, t, fontsize=8, color=MUTED, ha=ha)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9, color=INK)
ax.invert_yaxis(); ax.set_xlim(-1, 20); ax.set_xlabel('回の最小すき間 [cm](20 cm 以上は 20 に寄せた。× = 衝突(0 cm に置く)・止まり)', color=MUTED, fontsize=9)
ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(colors=MUTED)
ax.set_title('no_wall r3 候補(方策 3b099f9b・シールド v4.1)— 確認の段の新しい組', color=INK, fontsize=11, loc='left')
plt.subplots_adjust(left=.27, right=.80, top=.92, bottom=.1)
plt.savefig(out, dpi=130, facecolor=SURF)
print(out)
