#!/usr/bin/env python3
"""図: GM の出口(DE に入る 1.5 秒前)から DE の中までのステア。人(10/10 の人の記録、ぶつからない通過)・AI(10/23 公式)の
ぶつからない通過・ぶつかった通過。de_line_wide.json から(作り方は de_line.py の wide 部分)
    python3 fig_de_line.py → de_line.png"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE = os.path.dirname(os.path.abspath(__file__))
for f in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'TakaoGothic'):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
d = json.load(open(os.path.join(HERE, 'de_line_wide.json')))
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
for k, lab, col in (('human', '人(10/10、ぶつからない通過)', '#2a7a3b'), ('ai_clean', 'AI ぶつからない(10/23 公式)', '#2b6cd4'),
                    ('ai_hit', 'AI ぶつかった(10/23 公式)', '#c0392b')):
    g = d[k]
    ax.plot(g['t'], g['med'], color=col, lw=2.4, label='%s n=%d' % (lab, g['n']))
    ax.fill_between(g['t'], g['q25'], g['q75'], color=col, alpha=0.10)
ax.axhline(0, color='#888', lw=1)
ax.axvline(0, color='#555', lw=1, ls='--')
ax.axvspan(0.8, 1.0, color='#c0392b', alpha=0.08)
ax.set_ylim(-250, 520)
ax.text(0.02, 480, 'DE に入る', fontsize=10, color='#555')
ax.text(0.9, 480, '衝突の多い時刻', ha='center', color='#c0392b', fontsize=10)
ax.text(-0.75, 480, 'GM(左へ大きく切る)', ha='center', fontsize=10, color='#555')
ax.set_xlabel('DE に入ってからの秒(負 = GM の出口の手前)')
ax.set_ylabel('ステア(中央からの µs。+ = 左・− = 右)')
ax.set_title('GM の出口で、人はまっすぐに戻す。AI は右へ行き過ぎ、DE で左へ切り返す')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'de_line.png'))
print('saved de_line.png')
