import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False; plt.rcParams['axes.spines.right'] = False
from rruns3 import get
from eff import NEUT, CENTER
OUT = '/home/jetson/jetformer/data/reports/ai_runs_20261010/'
C1, C2, C3, C4, GR, INK = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#8a8984', '#52514e'
rows_all = json.load(open('rruns3.json')); rows = [r for r in rows_all if r['cat'] != 'after_R_reverse']; E = json.load(open('episodes.json')); CLS = json.load(open('ep_class.json'))
KK = {r['ep']: r for r in json.load(open('fwdkick.json'))}
SIM = json.load(open('simdet.json'))
# ---------------- Fig 1: ESC rule
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, prev, ttl in [(axes[0], 'F', '直前の非中立指令 = 前進 (F)'), (axes[1], 'R', '直前の非中立指令 = 後退側 (R: ブレーキ/後退)')]:
    sub = [r for r in rows if (r['prev'] in ('F', 'start')) == (prev == 'F') and r['prev'] != 'start' or (prev == 'F' and r['prev'] == 'start')]
    for eng, col, lab, mk in [(False, C2, '後退せず (ブレーキ扱い)', 'x'), (True, C1, '後退が始まった', 'o')]:
        s = [r for r in sub if r['engaged'] == eng]
        x = [max(r['gap'], 0.012) for r in s]; y = [r['e0'] for r in s]
        ax.scatter(x, y, s=34 if mk == 'o' else 40, marker=mk, c=col, label='%s (n=%d)' % (lab, len(s)), linewidths=1.5, alpha=0.9,
                   edgecolors='white' if mk == 'o' else None)
    ax.axhline(-150, color=GR, lw=1, ls='--'); ax.axhline(0, color=GR, lw=0.6)
    ax.set_xscale('log'); ax.set_xlim(0.01, 2.0)
    ax.set_yscale('symlog', linthresh=300); ax.set_yticks([-5000, -1000, -300, 0, 300, 1000])
    ax.set_yticklabels(['-5000', '-1000', '-300', '0', '300', '1000'])
    ax.set_xlabel('R指令の直前の中立時間 [s] (log)')
    ax.set_title(ttl, fontsize=10, loc='left')
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False, loc='lower right')
axes[0].set_ylabel('R指令開始時のeRPM (負=前進中)')
axes[1].text(0.012, -200, '← この線より下 = まだ前に転がっている', fontsize=7.5, color=INK, va='top')
fig.suptitle('ESCの後退条件: 「R(ブレーキ) → 中立 → R」かつ モーター停止(eRPM ≥ -150) の時だけ後退する   (R指令 %d 回、後退継続中のパルス %d 回は除外)' % (len(rows), len(rows_all) - len(rows)), fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT + 'fig_recovery_esc_rule.png', dpi=110); plt.close(fig)

# ---------------- Fig 2: example timelines
def timeline(ax1, ax2, k, pre=1.0, post=None, notes=()):
    x = E[k - 1]; d = get(x['log']); t0 = x['t_stop']
    tend = t0 + (x['handback_after_stop'] + 0.8 if x['handback_after_stop'] and x['handback_after_stop'] < 8 else (x['free_after_stop'] or x['fwd_after_stop'] + 1) + 0.6)
    m = (d.vt >= t0 - pre) & (d.vt <= tend); tt = d.vt[m] - t0
    for ax in (ax1, ax2):
        sel = d.vsel[m]
        if sel.any(): ax.fill_between(tt, 0, 1, where=sel, transform=ax.get_xaxis_transform(), color=GR, alpha=0.13, lw=0)
        ax.grid(alpha=0.25, lw=0.5); ax.axvline(0, color=GR, lw=0.8, ls='--')
    ax1.axhspan(NEUT - 20, NEUT + 40, color=GR, alpha=0.2, lw=0)
    ax1.plot(tt, d.thr[m], color=C1, lw=1.6, label='スロットル指令 (実効)')
    ax1.plot(tt, d.st[m], color=C2, lw=1.6, label='ステアリング指令 (実効)')
    ax1.set_ylim(950, 2050); ax1.set_ylabel('µs')
    ax2.plot(tt, d.erpm[m], color=C3, lw=1.6); ax2.axhline(0, color=GR, lw=0.6); ax2.set_ylabel('eRPM (+ = 後退)')
    ax2.set_xlabel('停止からの時間 [s]')
    for tx, txt, y in notes:
        ax1.annotate(txt, (tx, y), fontsize=7.5, color=INK, ha='left', va='bottom')
    return ax1, ax2
fig = plt.figure(figsize=(12, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.08, wspace=0.15)
a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1)
timeline(a1, a2, 9, notes=[(-0.9, '衝突 7.0g\n(右フルロック)', 1960), (0.45, '前進を\n試す', 1080), (1.0, 'ブレーキ\n1775µs', 1830), (1.42, 'R 1813µs', 1960), (2.0, '前進+右切り', 1080)])
a1.set_title('#9 人手のみ (152707, t=77.3s)', fontsize=10, loc='left'); a1.legend(fontsize=8, frameon=False, loc='lower left', ncol=2)
plt.setp(a1.get_xticklabels(), visible=False)
b1 = fig.add_subplot(gs[0, 1], sharey=a1); b2 = fig.add_subplot(gs[1, 1], sharex=b1, sharey=a2)
timeline(b1, b2, 19, notes=[(0.3, 'AIがR 2000µsを1.3s\n→ ブレーキのまま動かない', 1170), (1.36, '人が交代\n中立0.58s', 1590), (1.95, 'R 1750µs', 1960), (2.55, '前進キック', 1080), (3.6, 'AIへ返す', 1960)])
b1.set_title('#19 AI走行中の交代 (173445, t=35.2s)   灰色 = AI側', fontsize=10, loc='left')
plt.setp(b1.get_xticklabels(), visible=False)
fig.savefig(OUT + 'fig_recovery_examples.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---------------- Fig 3: parameter distributions (A episodes)
A = [(k + 1, x) for k, x in enumerate(E) if CLS[str(k + 1)] == 'A']
params = [('停止→R指令(ブレーキ) [s]', lambda x, k: x['first_R_after_stop']), ('停止→後退開始 [s]', lambda x, k: x['engage_after_stop']),
          ('効いた中立時間 [s]', lambda x, k: x['gaps'][0]), ('後退指令 中央値 [µs]', lambda x, k: x['rev_thr_med']),
          ('後退指令の長さ [s]', lambda x, k: x['rev_cmd_s']), ('後退の移動時間 [s]', lambda x, k: x['rev_motion_s']),
          ('後退距離 [hallカウント]', lambda x, k: x['rev_counts']), ('後退ピークeRPM', lambda x, k: x['rev_peak_erpm']),
          ('後退中ステア (中央比) [µs]', lambda x, k: x['st_rev']), ('前進キック ステア [µs]', lambda x, k: KK[k].get('kick_st_med')),
          ('前進キック 長さ [s]', lambda x, k: KK[k].get('kick_dur')), ('停止→通常走行 [s]', lambda x, k: x['free_after_stop'])]
fig, axes = plt.subplots(3, 4, figsize=(13, 7.2))
rng = np.random.default_rng(0)
for ax, (name, f) in zip(axes.flat, params):
    for j, (src, col, lab) in enumerate([('human-only', C1, '人手のみ (n=9)'), ('AI-run takeover', C2, 'AI走行中の交代 (n=10)')]):
        v = np.array([f(x, k) for k, x in A if (x['source'] == 'human-only') == (src == 'human-only') and f(x, k) is not None], float)
        y = j + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(v, y, s=26, c=col, alpha=0.85, edgecolors='white', linewidths=0.6, label=lab)
        ax.plot([np.median(v)] * 2, [j - 0.3, j + 0.3], color=INK, lw=2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['人手', 'AI交代'], fontsize=8); ax.set_ylim(-0.6, 1.6)
    ax.set_title(name, fontsize=9, loc='left'); ax.grid(axis='x', alpha=0.25, lw=0.5); ax.tick_params(axis='x', labelsize=8)
axes[0, 0].legend(fontsize=7, frameon=False, loc='upper right')
fig.suptitle('衝突・スタック回復(クラスA, 19件)のパラメータ分布   黒線 = 中央値', fontsize=11)
fig.tight_layout(); fig.savefig(OUT + 'fig_recovery_params.png', dpi=110); plt.close(fig)

# ---------------- Fig 4: detection vs human (AI-run episodes)
fig, ax = plt.subplots(figsize=(9, 4.4))
ai_eps = [(k, x) for k, x in A if x['source'] != 'human-only']
for j, (k, x) in enumerate(ai_eps):
    trig = [s for s in SIM if s['ep'] == k]
    tdet = min(s['lat_from_stop'] for s in trig) if trig else None
    to = x['takeover_after_stop']; eng = x['engage_after_stop']; hb = x['handback_after_stop']
    ax.plot([0, hb], [j, j], color=GR, lw=1)
    ax.scatter([to], [j], marker='s', s=40, c=C2, label='人の交代 (CH3)' if j == 0 else None, zorder=3)
    ax.scatter([eng], [j], marker='o', s=40, c=C1, label='後退が始まった' if j == 0 else None, zorder=3)
    ax.scatter([hb], [j], marker='D', s=36, c=C4, label='AIへ返却' if j == 0 else None, zorder=3)
    if tdet is not None: ax.scatter([tdet], [j], marker='|', s=260, c=INK, linewidths=2.2, label='提案検知(D1/D2)' if j == 0 else None, zorder=4)
ax.set_yticks(range(len(ai_eps))); ax.set_yticklabels(['#%d t=%.0fs' % (k, x['t_stop']) for k, x in ai_eps], fontsize=8)
ax.axvline(0, color=GR, ls='--', lw=0.8); ax.set_xlabel('停止(eRPM > -300)からの時間 [s]'); ax.invert_yaxis()
ax.grid(axis='x', alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False, loc='lower right')
ax.set_title('AI走行中の10件: 提案検知(停止+0.5s) と 人の対応の比較', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(OUT + 'fig_recovery_detection.png', dpi=110); plt.close(fig)
print('ok')
