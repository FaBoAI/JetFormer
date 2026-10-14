#!/usr/bin/env python3
# 確認 line r3 の図(2026-10-14、確認の段)。check_line_r3/runs から
#   fig/check_line_r3_clearance_vs_heading.png: 回ごとの最小すき間と、開始の向きの R に対するずれ(組ごと。シールドが働いた回はシールドなしの値も)
#   fig/check_line_r3_worst.png: 一番近づいた回たち(シールドあり)の上から見た軌跡(柵・柱の足・外枠)
import json, glob, math, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3'
FIG = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/fig'

def load(v):
    rows = []
    for f in sorted(glob.glob(f'{OUT}/runs/{v}_*.jsonl')):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    return rows
cand = load('cand'); nosh = load('nosh')
cm = {(r['set'], r['id']): r for r in cand}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
col = {'REPLAY': '#1f77b4', 'U300': '#7f7f7f', 'HEAD120': '#d62728', 'MID120': '#ff7f0e', 'NEAR60': '#9467bd', 'CORNER': '#2ca02c', 'DENSE': '#8c564b', 'EXTHEAD': '#e377c2', 'FAM6V': '#17becf'}
name = {'REPLAY': '決めた試験の組(186)', 'U300': '新しい一様 300', 'HEAD120': '横向き |Δψ|≥1 120', 'MID120': '0.6≤|Δψ|<1 120', 'NEAR60': '開始のすき間 12〜15 cm 60',
        'CORNER': '分布の角・辺', 'DENSE': '危ない 6 ゲートの格子 150', 'EXTHEAD': '横向き × 端の形 60', 'FAM6V': 'FAM6 のずらし 42'}
for s in col:
    rr = [r for r in cand if r['set'] == s]
    if not rr: continue
    ax[0].scatter([abs(r['start_info']['dpsi']) for r in rr], [r['min_clearance'] * 100 for r in rr], s=9, c=col[s], label=f'{name[s]}', alpha=.75)
ax[0].axhline(3, color='k', lw=1, ls='--'); ax[0].axhline(4, color='gray', lw=.8, ls=':')
ax[0].set_xlabel('開始の向きの R に対するずれ |Δψ| [rad]'); ax[0].set_ylabel('回の最小すき間 [cm]')
ax[0].set_title(f'候補 9f10c7b7(シールドあり)の {len(cand)} 回'); ax[0].legend(fontsize=7, loc='upper right'); ax[0].set_ylim(-1, 25)
acted = [r for r in nosh if r['set'] != 'REPLAY' or cm[(r['set'], r['id'])]['shield_frames'] > 0]
for r in acted:
    c = cm[(r['set'], r['id'])]; x = abs(c['start_info']['dpsi'])
    ax[1].plot([x, x], [r['min_clearance'] * 100, c['min_clearance'] * 100], color='#bbbbbb', lw=.8)
    ax[1].scatter([x], [c['min_clearance'] * 100], s=14, c='#1f77b4')
    ax[1].scatter([x], [r['min_clearance'] * 100], s=14, c='#d62728' if r['collisions'] else '#ff7f0e', marker='x')
ax[1].axhline(3, color='k', lw=1, ls='--')
ax[1].set_xlabel('|Δψ| [rad]'); ax[1].set_ylabel('最小すき間 [cm](衝突 = 0)')
ax[1].set_title(f'シールドが働いた {len(acted)} 回: 青 = シールドあり、× = なし(赤 = 衝突)')
fig.tight_layout(); fig.savefig(f'{FIG}/check_line_r3_clearance_vs_heading.png', dpi=110)

# 一番近づいた 6 回(シールドあり)
sys.path.insert(0, '')
world = json.load(open(f'{OUT}/world_fig.json')) if glob.glob(f'{OUT}/world_fig.json') else None
# 崩れた回(repro.json、フレームは 3 つおき + シールドのフレーム)と試験の P37
rep = json.load(open(f'{OUT}/repro.json'))
worst = []
for x in rep:
    o = x['own']
    worst.append({'set': 'REPRO', 'id': x['id'], 'fr': x['frames'], 'start': x['start'], 'line': x['params']['line'], 'min_clearance': max(o['min'], 0), 'min_t': o['min_t'],
                  'shield_frames': o['shield_frames'], 'collision': o['collision'], 'stall': o['stall']})
p37 = [r for r in cand if r['set'] == 'REPLAY' and r['id'] == 'P37'][0]
worst.append({**p37, 'collision': None})
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for a, r in zip(axs.flat, worst):
    fr = r['fr']
    mf = min(range(len(fr)), key=lambda k: abs(fr[k][0] / 30 - r['min_t']))
    lo, hi = 0, min(len(fr), mf + 15)
    xs = [q[1] for q in fr[lo:hi]]; zs = [q[2] for q in fr[lo:hi]]
    if world:
        for sgm in world['segments']:
            a.plot([sgm['a'][0], sgm['b'][0]], [sgm['a'][1], sgm['b'][1]], color='k', lw=2)
        for d in world['feet']:
            a.add_patch(plt.Rectangle((d[0] - .08, d[1] - .075), .16, .15, color='#555555'))
        a.plot(world['loop_x'], world['loop_z'], color='#bbbbbb', lw=1, ls='--')
    a.plot(xs, zs, color='#1f77b4', lw=1.5)
    q = fr[mf]; x, z, y = q[1], q[2], q[3]
    c, s = math.cos(y), math.sin(y)
    pts = [(x + lx * c + lz * s, z - lx * s + lz * c) for lx, lz in [(.21, .095), (.21, -.095), (-.21, -.095), (-.21, .095), (.21, .095)]]
    a.plot([p[0] for p in pts], [p[1] for p in pts], color='#d62728', lw=1.5)
    for k in range(lo, hi):
        if fr[k][11] == 1: a.scatter([fr[k][1]], [fr[k][2]], s=18, c='#ff7f0e', zorder=5)
    a.set_xlim(min(xs) - .4, max(xs) + .4); a.set_ylim(max(zs) + .4, min(zs) - .4); a.set_aspect('equal')
    tag = '衝突' if r.get('collision') else ('止まり' if r.get('stall') else f"最小 {r['min_clearance']*100:.2f} cm")
    a.set_title(f"{r['id']} ゲート {r['start']['gate']}・左右 {r['start']['left_offset_m']:+.3f} m・向き {r['start']['left_yaw_deg']:+.1f}°\n{r['line']['family']} {tag} @ {r['min_t']:.2f} 秒(シールド {r['shield_frames']} フレーム)", fontsize=9)
fig.suptitle('崩れた回の最初の数秒(青 = 軌跡、橙 = シールドが替えたフレーム、赤 = 一番近づいた時の車体)', fontsize=11)
fig.tight_layout(); fig.savefig(f'{FIG}/check_line_r3_worst.png', dpi=100)
print('ok')
