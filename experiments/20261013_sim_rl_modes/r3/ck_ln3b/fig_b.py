# 確認 line r3(再実行): 図。(左) 回ごとの最小すき間と |Δψ|、(右) ゲート 16・17 の角の箱の開始と結果
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln3b')
from analyze_b import load

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3'
rs = load('cand_*.jsonl')


def outcome(r):
    if r.get('collisions', 0) > 0:
        return 'collision'
    if r.get('stall'):
        return 'stall'
    if r.get('valid_r3') and r['min_c'] < .03:
        return 'lt3'
    if not r.get('valid_r3'):
        return 'invalid'
    return 'ok'


ST = {'ok': dict(c='#9aa0a6', m='o', s=10, label='有効(3 cm 以上)'), 'invalid': dict(c='#1a73e8', m='s', s=22, label='無効(黒シートの横断なし)'),
      'stall': dict(c='#f29900', m='D', s=30, label='止まり'), 'lt3': dict(c='#a142f4', m='v', s=40, label='有効だが 3 cm 未満'),
      'collision': dict(c='#d93025', m='X', s=70, label='衝突')}
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
a = ax[0]
for k, st in ST.items():
    xs = [abs(r['start']['dpsi']) for r in rs if outcome(r) == k]
    ys = [max(r['min_c'], 0) * 100 for r in rs if outcome(r) == k]
    if xs:
        a.scatter(xs, ys, c=st['c'], marker=st['m'], s=st['s'], label=f"{st['label']}({len(xs)})", alpha=.8 if k == 'ok' else 1, linewidths=0)
a.axhline(3, color='#d93025', lw=1, ls='--')
a.axhline(4, color='#5f6368', lw=.8, ls=':')
a.axvline(1, color='#5f6368', lw=.8, ls=':')
a.set_xlabel('開始の向きのずれ |Δψ|(R に対して、rad)')
a.set_ylabel('回の最小すき間 [cm]')
a.set_title(f'新しい組 {len(rs)} 回(候補 + シールド 4 cm)')
a.set_ylim(-0.5, 20)
a.legend(fontsize=8, loc='upper right')
b = ax[1]
for gate, col in ((16, '#1a73e8'), (17, '#188038')):
    sub = [r for r in rs if r['gate'] == gate and r['off'] <= -.15 and r['yaw'] >= 10]
    for k, st in ST.items():
        xs = [r['off'] for r in sub if outcome(r) == k]
        ys = [r['yaw'] for r in sub if outcome(r) == k]
        if xs:
            b.scatter(xs, ys, c=st['c'], marker=st['m'], s=st['s'] * 1.3, edgecolors=col if k == 'ok' else 'none', linewidths=.6 if k == 'ok' else 0)
    n = len(sub)
    nf = sum(outcome(r) != 'ok' for r in sub)
    b.text(.02, .96 - (gate - 16) * .06, f'ゲート {gate}: {n} 回のうち崩れ {nf}(枠の色 = ゲート)', transform=b.transAxes, color=col, fontsize=9, va='top')
b.set_xlabel('開始の左右のずれ [m](負 = 右)')
b.set_ylabel('開始の向き [度](正 = 左)')
b.set_title('ゲート 16・17 の角の箱(左右 −0.20〜−0.16 m・向き +12〜+15°)')
b.set_xlim(-.205, -.155)
b.set_ylim(11.8, 15.2)
c = ax[2]
sub = [r for r in rs if r['gate'] in (30, 31) and r['off'] >= .09 and r['yaw'] >= 7]
for k, st in ST.items():
    pts = [r for r in sub if outcome(r) == k]
    if pts:
        c.scatter([r['off'] for r in pts], [r['yaw'] for r in pts], c=st['c'], marker=st['m'], s=st['s'] * 1.3,
                  edgecolors=['#1a73e8' if (r['params']['line'] or {}).get('family') == 'const' and r['params']['line'].get('c0', 0) <= -.6 else 'none' for r in pts] if k == 'ok' else 'none', linewidths=.8)
n = len(sub); nf = sum(outcome(r) != 'ok' for r in sub)
c.text(.02, .97, f'ゲート 30・31: {n} 回のうち崩れ {nf}(青い枠 = 右寄りの形 c0 ≤ −0.6)', transform=c.transAxes, fontsize=9, va='top')
c.set_xlabel('開始の左右のずれ [m](正 = 左)')
c.set_ylabel('開始の向き [度](正 = 左)')
c.set_title('ゲート 30・31 の角(左右 +、向き +)。新しく見つけた崩れ')
plt.tight_layout()
plt.savefig(f'{OUT}/fig/check_line_r3b_corner.png', dpi=110)
print('ok')

# 2 枚目: 崩れた回を上から(コースの障害物・R・軌跡。× = 最小すき間の所、● = 開始)
W = json.load(open(f'{OUT}/check_line_r3b/world_b.json'))
FF = json.load(open(f'{OUT}/check_line_r3b/fail_frames.json'))
fig, ax = plt.subplots(1, 1, figsize=(13, 8.5))
for o in W['ob']:
    if o['type'] == 'cap':
        ax.plot([o['a'][0], o['b'][0]], [o['a'][1], o['b'][1]], color='#3c4043', lw=1.6)
    elif o['type'] == 'box':
        P = o['P'] + [o['P'][0]]
        ax.plot([p[0] for p in P], [p[1] for p in P], color='#3c4043', lw=.8)
    else:
        ax.add_patch(plt.Circle((o['x'], o['z']), o['r'], color='#3c4043'))
ax.plot([p[0] for p in W['R']], [p[1] for p in W['R']], color='#bdc1c6', lw=1, ls='--', label='R(参照ループ)')
cols = {'CB264-213': '#a142f4', 'G30R-2': '#d93025', 'BOX16-0': '#f29900', 'BOX17-57': '#188038'}
for x in FF:
    if x['id'] not in cols:
        continue
    fr = x['fr']
    n = min(len(fr), 180)
    lab = f"{x['id']} {'シールドあり' if x['shield'] else 'シールドなし'}: {x['reason'] if x['reason'] != 'collection_duration_complete' else '20 秒'}" \
          f"{'・止まり' if x['stall_first'] else ''}・最小 {max(x['min_c'], 0) * 100:.2f} cm"
    ax.plot([r[1] for r in fr[:n]], [r[2] for r in fr[:n]], color=cols[x['id']], lw=1.4 if x['shield'] else .9, ls='-' if x['shield'] else ':', label=lab)
    ax.plot(fr[0][1], fr[0][2], 'o', color=cols[x['id']], ms=5)
    if x['shield']:
        m = min(range(len(fr)), key=lambda i: fr[i][10])
        ax.plot(fr[m][1], fr[m][2], 'x', color=cols[x['id']], ms=11, mew=2)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('z [m]')
ax.invert_yaxis()
ax.set_title('崩れた回(最初の 6 秒)。実線 = シールドあり、点線 = シールドなし。× = 最小すき間の所')
ax.legend(fontsize=8, loc='center')
plt.tight_layout()
plt.savefig(f'{OUT}/fig/check_line_r3b_worst.png', dpi=110)
print('ok2')
