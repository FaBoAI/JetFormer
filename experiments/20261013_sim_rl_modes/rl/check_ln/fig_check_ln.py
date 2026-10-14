#!/usr/bin/env python3
# 経路いろいろ(line)の確認 r1 の図(2026-10-14、確認の段)。runs/*.jsonl(自前の走らせ方の結果)だけを読む。
#   1. check_line_min_clearance.png: 回ごとの最小すき間とそれが起きた時刻(試験の組 + 新しい開始)。3 cm の線
#   2. check_line_tracking_by_s.png: 新しい開始の有効な回の追従の誤差の 95% 点を R 上の s(0.25 m のマス)ごとに。12 cm の線、S 字と尖りの区間
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013'
RUNS = f'{REP}/check_line_r1/runs'
OUT = f'{REP}/fig'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
L_LOOP = 26.119648724030924
for name in ['Noto Sans CJK JP', 'Noto Sans JP', 'Noto Serif CJK JP']:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams['font.family'] = name
        break


def load(n):
    return [json.loads(l) for l in open(f'{RUNS}/{n}.jsonl')]


def style(ax, title, xl, yl):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel(xl, color=INK2, fontsize=9)
    ax.set_ylabel(yl, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


rep = [r for r in load('REPLAY') if r['test'] in ('LN-P60', 'LN-FAM6')]
fr = load('FRESH')
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
fig.patch.set_facecolor(SURF)
for rows, col, lab, mk in [(fr, S1, '新しい開始 240 回', 'o'), (rep, S2, '試験の組 66 回', 's')]:
    ok = [r for r in rows if r['collisions'] == 0]
    bad = [r for r in rows if r['collisions'] > 0]
    ax.scatter([r['min_t'] for r in ok], [100 * max(r['min_clearance'], 0) for r in ok], s=16, color=col, marker=mk, label=lab, edgecolors=SURF, linewidths=.6, zorder=3)
    ax.scatter([r['min_t'] for r in bad], [100 * max(r['min_clearance'], 0) for r in bad], s=40, color=col, marker='x', linewidths=1.4, zorder=4,
               label=f'{lab}のうち衝突')
ax.axhline(3, color=INK2, linewidth=1, linestyle='--')
ax.text(19.8, 3.3, 'L2 のしきい値 3 cm', color=INK2, fontsize=8, ha='right', va='bottom')
ax.set_xlim(-0.3, 20)
ax.set_ylim(-0.5, 16)
style(ax, '回ごとの最小すき間と、それが起きた時刻(3 cm 未満は全部が開始から 1.2 秒以内)', '最小すき間の時刻 [秒](開始から)', '最小すき間 [cm]')
ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc='upper right', ncol=2)
fig.tight_layout()
fig.savefig(f'{OUT}/check_line_min_clearance.png', facecolor=SURF)

# 2
bins = [[] for _ in range(int(math.ceil(L_LOOP / .25)))]
for r in fr:
    if not r['valid_episode']:
        continue
    for f in r['fr']:
        if f[1] is None or f[1] < 3 or f[4] is None:
            continue
        bins[min(len(bins) - 1, int(f[2] / .25))].append(abs(f[3] - f[4]))


def q(a, p):
    a = sorted(a)
    if not a:
        return float('nan')
    i = p * (len(a) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo] + (a[hi] - a[lo]) * (i - lo)


xs = [(i + .5) * .25 for i in range(len(bins))]
p95 = [100 * q(b, .95) for b in bins]
med = [100 * q(b, .5) for b in bins]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=140)
fig.patch.set_facecolor(SURF)
for z in [(2.0, 4.5), (8.0, 12.5)]:
    ax.axvspan(*z, color='#f1efe8', zorder=0)
ax.axvspan(10.34, 11.29, color='#f6d9cc', zorder=0)
ax.plot(xs, p95, color=S1, linewidth=2, label='95% 点', zorder=3)
ax.plot(xs, med, color=S3, linewidth=2, label='中央値', zorder=3)
ax.axhline(12, color=INK2, linewidth=1, linestyle='--')
ax.text(25.9, 12.6, 'L3 の 95% 点のしきい値 12 cm', color=INK2, fontsize=8, ha='right')
ax.text(3.25, 46, 'S 字', color=INK2, fontsize=8, ha='center')
ax.text(9.0, 46, 'S 字', color=INK2, fontsize=8, ha='center')
ax.annotate('赤い帯: e*·κ_R ≥ 1(左寄り c0 0.9・イン a 0.9 で\n目標の点が R の曲率の中心より先になる所)', xy=(11.2, 30), xytext=(14.2, 30), color=INK2, fontsize=7.5,
            arrowprops=dict(arrowstyle='-', color=INK2, linewidth=.8))
ax.set_xlim(0, L_LOOP)
ax.set_ylim(0, 50)
style(ax, '追従の誤差 |e_y − e*(s)| を R 上の位置ごとに(新しい開始 232 回、0.25 m のマス)', 'R 上の位置 s [m]', '誤差 [cm]')
ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc='upper right')
fig.tight_layout()
fig.savefig(f'{OUT}/check_line_tracking_by_s.png', facecolor=SURF)
print('ok')
