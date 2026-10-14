#!/usr/bin/env python3
# 経路いろいろ(line)の確認 r2 の図(2026-10-14、確認の段)
#   check_line_r2_p45.png: P45 を上から(柵・柱の足・R・目標の線 v2(つなぎあり / なし)・車の軌跡・一番近づいたときの車体)
#   check_line_r2_clearance_vs_heading.png: 回ごとの最小すき間と、開始の車の向きの R に対するずれ |Δψ|(試験の組・新しい開始・大きく向いた開始・範囲の端)
import json, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

for f in ['Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'TakaoGothic', 'VL Gothic']:
    try:
        from matplotlib import font_manager
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = f
            break
    except Exception:
        pass
OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013'
RUNS = f'{OUT}/check_line_r2/runs'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
W = json.load(open(f'{OUT}/check_line_r2/world_fig.json'))


def load(n):
    p = f'{RUNS}/{n}.jsonl'
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---- 図 1: P45 ----
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=140)
path = W['p45']['path']
xs = [p[0] for p in path]; zs = [p[1] for p in path]
xmin, xmax = min(xs[:80]) - .6, max(xs[:80]) + .6
zmin, zmax = min(zs[:80]) - .6, max(zs[:80]) + .6
for s in W['segments']:
    ax.plot([s['a'][0], s['b'][0]], [s['a'][1], s['b'][1]], color=INK, lw=2.2, solid_capstyle='round', zorder=3)
for d in W['disks']:
    if d['kind'] == 'post-foot':
        ax.add_patch(Rectangle((d['x'] - .08, d['z'] - .075), .16, .15, fc='#8a8986', ec=INK, lw=.8, zorder=3))
    else:
        ax.add_patch(Circle((d['x'], d['z']), d['r'] or .02, fc=INK, zorder=3))
ax.plot([p[0] for p in W['R']], [p[1] for p in W['R']], color=MUTED, lw=1, ls=(0, (4, 3)), label='参照ループ R', zorder=2)
ax.plot([p[0] for p in W['target_noblend']], [p[1] for p in W['target_noblend']], color=S4, lw=1.2, ls=(0, (1, 2)), label='目標の線 v2(つなぎなし)', zorder=2)
ax.plot([p[0] for p in W['target_blend']], [p[1] for p in W['target_blend']], color=S3, lw=2, label='目標の線 v2(開始のつなぎあり)', zorder=4)
n = 90
ax.plot(xs[:n], zs[:n], color=S1, lw=2, label='車の中心(0〜3 秒)', zorder=5)
mf = W['p45']['min_frame']
for k, col in [(0, MUTED), (mf + 1, S2)]:
    x, z, y = path[k]
    fx, fz, lx, lz = math.cos(y), -math.sin(y), math.sin(y), math.cos(y)
    c = [(x + .21 * fx + .095 * lx, z + .21 * fz + .095 * lz), (x + .21 * fx - .095 * lx, z + .21 * fz - .095 * lz),
         (x - .21 * fx - .095 * lx, z - .21 * fz - .095 * lz), (x - .21 * fx + .095 * lx, z - .21 * fz + .095 * lz)]
    ax.add_patch(Polygon(c, closed=True, fc='none', ec=col, lw=1.8, zorder=6))
ax.annotate(f"一番近づいた所 {W['p45']['min_clearance'] * 100:.2f} cm\n(開始から {mf / 30:.2f}〜{(mf + 1) / 30:.2f} 秒)", xy=(path[mf + 1][0] - .2, path[mf + 1][1] - .08), xytext=(-190, -110),
            textcoords='offset points', color=INK, fontsize=9, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
ax.annotate('開始(R に対して 72° 横向き・止まった状態)', xy=(path[0][0], path[0][1]), xytext=(-230, -40), textcoords='offset points', color=INK, fontsize=9,
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
ax.set_xlim(xmin, xmax); ax.set_ylim(zmax, zmin)
ax.set_aspect('equal')
ax.set_xlabel('x [m]', color=MUTED); ax.set_ylabel('z [m](下が手前)', color=MUTED)
ax.tick_params(colors=MUTED, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(color=GRID, lw=.5)
ax.legend(loc='lower right', fontsize=8, frameon=True, facecolor='white', edgecolor=GRID)
ax.set_title('P45(試験の組 LN-P60-R2、ゲート 16・左寄り c0 0.76): 開始の回り込みで柱の足へ 1.33 cm', fontsize=10, color=INK, loc='left')
fig.tight_layout()
fig.savefig(f'{OUT}/fig/check_line_r2_p45.png')
plt.close(fig)

# ---- 図 2: 最小すき間と開始の向きのずれ ----
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
sets = [('LN-P60-R2(試験)', [r for r in load('REPLAY') if r['test'] == 'LN-P60'], S1, 'o'),
        ('新しい開始 240 回', load('FRESH'), S2, 's'),
        ('大きく向いた開始 60 回', load('HEAD'), S3, '^')]
ext = load('EXT')
if ext:
    sets.append(('範囲の端 60 回', ext, S4, 'D'))
for nm, rows, col, mk in sets:
    ok = [r for r in rows if not r.get('invalid_spawn') and not r['collisions']]
    xv = [abs(r['start_dpsi_vs_R']) * 180 / math.pi for r in ok]
    yv = [max(r['min_clearance'], 0) * 100 for r in ok]
    ax.scatter(xv, yv, s=22, c=col, marker=mk, edgecolors='white', linewidths=.6, label=nm, zorder=3, alpha=.9)
coll = [r for _, rows, _, _ in sets for r in rows if not r.get('invalid_spawn') and r['collisions']]
ax.scatter([abs(r['start_dpsi_vs_R']) * 180 / math.pi for r in coll], [0] * len(coll), s=40, c=INK, marker='x', linewidths=1.4, label=f'衝突した回({len(coll)} 回、すき間 0)', zorder=4)
ax.axhline(3, color=INK, lw=1, ls=(0, (4, 3)))
ax.text(1, 3.4, 'L2 のしきい値 3 cm', fontsize=8, color=INK)
ax.axvline(57.3, color=MUTED, lw=.8, ls=(0, (1, 2)))
ax.text(58.5, 22, '|Δψ| = 1 rad', fontsize=8, color=MUTED)
ax.set_xlabel('開始の車の向きと R の向きのずれ |Δψ| [°]', color=MUTED)
ax.set_ylabel('回の最小すき間 [cm]', color=MUTED)
ax.set_ylim(-1, 25)
ax.tick_params(colors=MUTED, labelsize=8)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(color=GRID, lw=.5)
ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=2)
ax.set_title('line r2(1ae6be07…)の回ごとの最小すき間(元の sim-core.js・自前の多角形の距離)', fontsize=10, color=INK, loc='left')
fig.tight_layout()
fig.savefig(f'{OUT}/fig/check_line_r2_clearance_vs_heading.png')
plt.close(fig)
print('ok')
