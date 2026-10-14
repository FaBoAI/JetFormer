# 確認 line r4: 図 2(攻めの組で方策だけでは崩れる 3 回を、シールドあり・なしで上から見た図)
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Circle

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
O = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4'
FIG = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/fig'
BLUE, ORANGE, INK, MUTED, OBST = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#9a9893'
D = json.load(open(f'{O}/traj4.json'))
cases = [('ADV-G32MM-37', 'ゲート 32 (−,−) × c0 +0.90(鏡の組)'), ('ADV-G30-1', 'ゲート 30 (+,+) × c0 −0.9 付近'), ('ADV-G21-36', 'ゲート 21 (+,+)')]


def car(x, z, yaw, L=.42, W=.19):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + u * c + w * s, z - u * s + w * c) for u, w in [(L / 2, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2), (L / 2, -W / 2)]]


fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9))
for ax, (cid, title) in zip(axs, cases):
    on, off = D['runs'][f'{cid}:on'], D['runs'][f'{cid}:off']
    fon, foff = np.array([r[:4] for r in on['fr']], float), np.array([r[:4] for r in off['fr']], float)
    nwin = max(on['min_f'], off['min_f']) + 25
    allx = np.concatenate([fon[:nwin, 1], foff[:nwin, 1]]); allz = np.concatenate([fon[:nwin, 2], foff[:nwin, 2]])
    cx, cz = (allx.min() + allx.max()) / 2, (allz.min() + allz.max()) / 2
    half = max(allx.max() - allx.min(), allz.max() - allz.min()) / 2 + .35
    for o in D['obstacles']:
        if o['k'] == 'cap':
            ax.plot([o['a'][0], o['b'][0]], [o['a'][1], o['b'][1]], color=OBST, lw=3, solid_capstyle='round')
        elif o['k'] == 'box':
            ax.add_patch(Polygon(o['V'], closed=True, color=OBST))
        else:
            ax.add_patch(Circle((o['x'], o['z']), o['r'], color=OBST))
    for f, col, lab, run in [(foff, ORANGE, 'シールドなし', off), (fon, BLUE, 'シールドあり', on)]:
        n = nwin if run is on else min(nwin, len(f))
        ax.plot(f[:n, 1], f[:n, 2], color=col, lw=2, label=f"{lab}: 最小 {max(run['min_c'], 0) * 100:.2f} cm" + ('(衝突)' if run['collision'] else ''))
        k = min(run['min_f'], len(f) - 1)
        ax.add_patch(Polygon(car(f[k, 1], f[k, 2], f[k, 3]), closed=True, fill=False, ec=col, lw=1.5))
    shf = [i for i in on['sh_list'] if i < nwin]
    if shf:
        ax.scatter(fon[shf, 1], fon[shf, 2], s=22, color=BLUE, edgecolor='white', linewidths=.8, zorder=5, label=f'シールドが替えたフレーム({len(on["sh_list"])})')
    ax.add_patch(Polygon(car(fon[0, 1], fon[0, 2], fon[0, 3]), closed=True, fill=False, ec=MUTED, lw=1, ls='--'))
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cz + half, cz - half)
    ax.set_aspect('equal')
    L = on['line']
    ax.set_title(f"{title}\n{cid}: 左右 {on['off']:+.3f} m・向き {on['yaw']:+.1f}°・{L['family']} {L.get('c0', L.get('a'))}・c_v {on['c_v']}", fontsize=8.5, color=INK, loc='left')
    ax.legend(fontsize=7.5, loc='lower left', frameon=True, framealpha=.85)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_xlabel('x [m]', fontsize=8, color=MUTED); ax.set_ylabel('z [m]', fontsize=8, color=MUTED)
fig.suptitle('確認 line r4: 方策だけでは崩れる開始(攻めの組)を、シールド v5b が 4 cm で止めている(点線の枠 = 開始、実線の枠 = 一番近づいた所)', fontsize=10, x=.01, ha='left', color=INK)
plt.tight_layout(rect=(0, 0, 1, .95))
plt.savefig(f'{FIG}/check_line_r4_hotspots.png', dpi=130)
print(f'{FIG}/check_line_r4_hotspots.png')
