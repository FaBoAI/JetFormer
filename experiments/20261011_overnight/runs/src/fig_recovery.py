"""fig/recovery_v2_stop_cases.png: the 12 stops that ended in a human takeover, with the recovery-v2 detector timeline
(from recov.json = exact replay of jetformer.assist)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': '#b5b4ad', 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'font.size': 9,
                     'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE = '#2a78d6', '#eb6834'
PLACE = {('ai_run_009', 15.69): 'LH 黒カーテン', ('ai_run_009', 19.62): 'LH 黒カーテン', ('ai_run_009', 24.42): 'GM 入口',
         ('ai_run_009', 46.81): 'HS 駐車枠', ('ai_run_009', 66.36): 'HS 駐車枠', ('ai_run_010', 17.69): 'GM 出口',
         ('ai_run_101', 12.76): 'LH 黒カーテン', ('ai_run_101', 42.7): 'LH 黒カーテン',
         ('ai_run_009', 50.84): 'LH 黒カーテン', ('ai_run_009', 54.64): 'LH 黒カーテン', ('ai_run_010', 12.19): 'LH 黒カーテン',
         ('ai_run_101', 53.13): 'GM 入口(白いシート)'}
rows = json.load(open('recov.json'))
sel = [r for r in rows if (r.get('impact_t') is not None and r['erpm_at_takeover'] > -1000) or
       (r.get('impact_t') is None and r['erpm_at_takeover'] > -300)]
sel.sort(key=lambda r: (r.get('impact_t') is None, -(r['takeover_t'] - (r['impact_t'] if r.get('impact_t') else r['stuck_from_t']))))
fig, ax = plt.subplots(figsize=(10.5, 6.3))
labels = []
for k, r in enumerate(sel):
    y = len(sel) - 1 - k
    ref = r['impact_t'] if r.get('impact_t') is not None else r['stuck_from_t']
    to = r['takeover_t'] - ref
    st = (r['stuck_from_t'] - ref) if r['stuck_from_t'] is not None else to
    if st > 0:
        ax.barh(y, st, left=0, height=0.55, color=BLUE, alpha=0.45, edgecolor=SURF, linewidth=2)
    if to - st > 0:
        ax.barh(y, to - st, left=st, height=0.55, color=ORANGE, edgecolor=SURF, linewidth=2)
    ax.plot([to, to], [y - 0.38, y + 0.38], color=INK, lw=2.2)
    if r['would_fire_t'] is not None:
        fx = r['would_fire_t'] - ref
        ax.plot(fx, y, 'D', ms=7, mfc=SURF, mec=INK, mew=1.4)
        ax.text(max(fx, to) + 0.04, y, '交代 %.2f s / 検出まであと %.2f s(必要 %.1f s)' % (to, r['margin_s'], r['need_s']),
                va='center', fontsize=8, color=INK2)
    else:
        ax.annotate('', xy=(to + r['need_s'] + 0.12, y), xytext=(to + 0.03, y),
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
        ax.text(to + r['need_s'] + 0.16, y, '交代 %.2f s / 跳ね返り中(eRPM %+d)で未カウント' % (to, r['erpm_at_takeover']),
                va='center', fontsize=8, color=INK2)
    kind = '%.1fG' % r['impact_g'] if r.get('impact_t') is not None else '衝撃なし'
    labels.append('%s %.1fs %s %s' % (r['name'], ref, PLACE.get((r['name'], r['takeover_t']), '?'), kind))
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(labels[::-1], fontsize=8.5)
ax.axvline(0.88, color='#9a9993', ls='--', lw=1)
ax.text(0.89, len(sel) - 0.45, '以前のAI走行(007ほか)で人が交代した中央値 0.88 s(止まってから)', fontsize=8, color=INK2)
ax.set_xlim(0, 2.6)
ax.set_xlabel('衝突(衝撃なしの停止は止まり始め)からの秒')
ax.set_title('リカバリーv2が一度も作動しなかった理由: 止まった12回すべてで、検出より先に人が交代した', loc='left')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hs = [Patch(facecolor=BLUE, alpha=0.45, label='まだ動いている・跳ね返って後ろへ転がっている(検出は数えない)'),
      Patch(facecolor=ORANGE, label='止まっている(検出が数えている)'),
      Line2D([], [], color=INK, lw=2.2, label='人が交代'),
      Line2D([], [], marker='D', ls='', mfc=SURF, mec=INK, label='止まったままなら検出された時刻')]
ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.4, -0.11), ncol=2, fontsize=8)
ax.grid(axis='x', color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig('../fig/recovery_v2_stop_cases.png', dpi=110)
print('ok', len(sel))
