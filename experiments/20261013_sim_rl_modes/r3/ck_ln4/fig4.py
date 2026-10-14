# 確認 line r4: 図(組ごとの回の最小すき間。シールドありと、同じ開始のシールドなし)
import json, glob, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
O = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4'
FIG = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/fig'
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'


def load(p):
    out = {}
    for f in sorted(glob.glob(f'{O}/runs/{p}')):
        for line in open(f):
            if line.strip():
                try:
                    r = json.loads(line)
                    out[r['id']] = r
                except json.JSONDecodeError:
                    pass
    return out


T, F, N = load('test_*.jsonl'), load('fresh_*.jsonl'), load('nosh_*.jsonl')
A, AN = load('adv_*.jsonl'), load('advnosh_*.jsonl')
order = ['T-LN-P60-R4', 'T-LN-HEAD60-R4', 'T-LN-EXT60-R4', 'T-LN-CORNER-R4', 'T-LN-BOX120-R4', 'T-LN-G30R120-R4B', 'T-LN-FAM6',
         'U300', 'H200', 'HX120', 'CB', 'VX', 'B1617', 'B1617X', 'G30R', 'G30RX', 'G30W', 'MIR', 'NEAR', 'RND', 'RNDC',
         'ADV-G30', 'ADV-G29G31', 'ADV-G32MM', 'ADV-G31G0MM', 'ADV-G32MMF', 'ADV-G23', 'ADV-G21', 'ADV-G17', 'ADV-G7']
label = {'T-LN-P60-R4': '試験 P60', 'T-LN-HEAD60-R4': '試験 HEAD60', 'T-LN-EXT60-R4': '試験 EXT60', 'T-LN-CORNER-R4': '試験 CORNER116', 'T-LN-BOX120-R4': '試験 BOX120',
         'T-LN-G30R120-R4B': '試験 G30R120', 'T-LN-FAM6': '試験 FAM6', 'U300': '新 一様 300', 'H200': '新 横向き 200', 'HX120': '新 横向き×端の形 120',
         'CB': '新 角の箱 232', 'VX': '新 ちょうどの頂点 115', 'B1617': '新 ゲート16/17 箱 120', 'B1617X': '新 16/17 箱×端の形 40', 'G30R': '新 ゲート28〜32 右寄り 150',
         'G30RX': '新 29〜32 c0≤−0.85 80', 'G30W': '新 27〜32 広め 60', 'MIR': '新 鏡の組 80', 'NEAR': '新 開始のすき間 12〜14 cm 100', 'RND': '新 random 60', 'RNDC': '新 random×角 60',
         'ADV-G30': '攻め G30 (+,+)×c0≤−0.85 60', 'ADV-G29G31': '攻め G29/31 同じ 40', 'ADV-G32MM': '攻め G32 (−,−)×c0≥0.85 50', 'ADV-G31G0MM': '攻め G31/0 (−,−) 30',
         'ADV-G32MMF': '攻め G32 (−,−)×端の形 20', 'ADV-G23': '攻め G23 開始の端 50', 'ADV-G21': '攻め G21 (+,+) 40', 'ADV-G17': '攻め G17 (−,+) 40', 'ADV-G7': '攻め G7 頂点 30'}
groups = collections.defaultdict(list)
for r in list(T.values()) + list(F.values()) + list(A.values()):
    groups[r['set']].append(r)
nosh = collections.defaultdict(list)
for r in N.values():
    nosh[r['set'][5:]].append(r)
for r in AN.values():
    nosh[r['set']].append(r)
rows = [s for s in order if groups.get(s)]
fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 1.6))
rng = np.random.default_rng(7)
for i, s in enumerate(rows):
    y = len(rows) - 1 - i
    a = np.array([min(r['min_c'], .2) for r in groups[s]]) * 100
    col = np.array([r['collisions'] > 0 for r in groups[s]])
    ax.scatter(a[~col], y + .12 + rng.uniform(-.08, .08, (~col).sum()), s=9, color=BLUE, alpha=.55, linewidths=0, label='シールドあり(候補そのもの)' if i == 0 else None)
    if col.any():
        ax.scatter(a[col], np.full(col.sum(), y + .12), s=40, marker='x', color=INK, linewidths=1.5)
    if nosh.get(s):
        b = np.array([min(r['min_c'], .2) for r in nosh[s]]) * 100
        cb = np.array([r['collisions'] > 0 for r in nosh[s]])
        ax.scatter(b[~cb], y - .18 + rng.uniform(-.08, .08, (~cb).sum()), s=9, color=ORANGE, alpha=.55, linewidths=0, label='同じ開始のシールドなし(参考)' if 'lab_n' not in globals() else None)
        globals()['lab_n'] = 1
        if cb.any():
            ax.scatter(np.zeros(cb.sum()), np.full(cb.sum(), y - .18), s=40, marker='x', color=ORANGE, linewidths=1.5)
    g = groups[s]
    txt = f"{sum(r['valid_r3'] for r in g)}/{len(g)} 有効・衝突 {sum(r['collisions'] > 0 for r in g)}・止まり {sum(r['stall'] for r in g)}・最小 {min(r['min_c'] for r in g) * 100:.2f} cm"
    ax.text(20.6, y + .12, txt, fontsize=7.5, color=MUTED, va='center')
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([label.get(s, s) for s in reversed(rows)], fontsize=8.5, color=INK)
for xv, t, ha, dx in [(3, 'L2 3 cm', 'right', -.15), (4, 'シールド 4 cm', 'left', .15)]:
    ax.axvline(xv, color=MUTED, lw=1, ls='--')
    ax.text(xv + dx, -.75, t, fontsize=8, color=MUTED, ha=ha)
ax.set_xlim(0, 20.5)
ax.set_ylim(-1.0, len(rows) - .4)
ax.set_xlabel('回ごとの最小すき間 [cm](20 cm 以上は 20 に寄せた。× は衝突)', color=INK)
ax.grid(axis='x', color=GRID, lw=.8)
ax.set_axisbelow(True)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
fig.legend(loc='upper left', fontsize=8, frameon=False, bbox_to_anchor=(.2, .965), ncol=2)
ax.set_title('確認 line r4: 組ごとの回の最小すき間(元の sim-core.js・自前の幾何・候補 line_r4 = 方策 5c197a2b + シールド v5b)', fontsize=10, loc='left', color=INK, pad=4)
fig.text(.2, .995, '', fontsize=1)
plt.subplots_adjust(left=.2, right=.68, top=1 - 1.1 / (0.42 * len(rows) + 1.6), bottom=.6 / (0.42 * len(rows) + 1.6))
plt.savefig(f'{FIG}/check_line_r4_sets.png', dpi=130)
print(f'{FIG}/check_line_r4_sets.png')
