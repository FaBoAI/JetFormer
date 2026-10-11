import json, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ld import *
from match import REF_A, REF_B, match
from psi import course_psi
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
REP = '/home/jetson/jetformer/data/reports/ai_runs_20261010'
SURF, TXT, TXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COL = {'AI 006': '#2a78d6', 'AI 007': '#eb6834', '人(学習データ)': '#1baf7a'}
def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b2')
    ax.tick_params(colors=TXT2, labelsize=9); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
C = json.load(open('classified.json'))
# ---------- reference lap path ----------
d = load('20261010-172910'); d.heading()
v = np.interp(d.it, d.vt, -d.erpm) / 1000.0; dt = np.diff(d.it, prepend=d.it[0])
m = (d.it >= REF_A - 1.6) & (d.it < REF_B + 0.3)
i0 = np.nonzero(d.it >= REF_A)[0][0]
h = np.deg2rad(d.hd - d.hd[i0] + 90)
x = np.cumsum(v * np.cos(h) * dt); y = np.cumsum(v * np.sin(h) * dt)
x -= x[i0]; y -= y[i0]
T, X, Y = d.it[m], x[m], y[m]
from destat import de_table
ref_tg = [r['tg'] for r in de_table('20261010-172910', False) if abs(r['a'] - REF_A) < 0.2][0]
print('ref_tg', ref_tg)
def pos(tau, off=0.0):
    tau = REF_A + ((tau - REF_A) % (REF_B - REF_A))
    k = np.searchsorted(T, tau); k = np.clip(k, 1, len(T) - 2)
    dx, dy = X[k + 1] - X[k - 1], Y[k + 1] - Y[k - 1]; n = np.hypot(dx, dy) + 1e-9
    return X[k] - dy / n * off, Y[k] + dx / n * off
def tau_of(e):
    s = e['sec'].rstrip('?')
    if s in ('R1', 'GM_in') and e['a'] is not None and e['rel'] is not None and e['rel'] < 3.5: return REF_A + e['rel']
    if s in ('GM_out', 'DE') and e['tg'] is not None and e['relg'] is not None and e['relg'] < 2.2: return ref_tg + e['relg']
    return None
groups = {'AI 006': ('20261010-172910',), 'AI 007': ('20261010-173445',), '人(学習データ)': ('20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707')}
REC = {('20261010-173445', 100.17), ('20261010-173445', 101.00), ('20261010-173445', 106.20)}
fig, ax = plt.subplots(figsize=(8.0, 9.6), facecolor=SURF)
style(ax)
mm = (T >= REF_A) & (T < REF_B)
ax.plot(X[mm], Y[mm], color='#8a8984', lw=6, alpha=0.25, solid_capstyle='round', zorder=1)
ax.plot(X[mm], Y[mm], color='#8a8984', lw=1.2, zorder=2)
# direction arrows
for tt in np.arange(REF_A + 0.5, REF_B, 1.0):
    k = np.searchsorted(T, tt)
    ax.annotate('', xy=(X[k + 3], Y[k + 3]), xytext=(X[k], Y[k]), arrowprops=dict(arrowstyle='->', color='#8a8984', lw=1.2))
labels = [(REF_A + 0.25, 'R1 右ヘアピン\n(白い仕切り板に正面衝突)', (1.6, 2.2)), (REF_A + 1.4, 'GM 緑マット左ターン\n(内側の板端 / 外側の板)', (0.0, -1.7)),
          (ref_tg + 1.15, 'DE バックストレート終端\n右外壁の角\n(緑矢印看板の下)', (-4.2, 0.3)), (REF_A + 5.2, 'SW 奥の左スイーパー\n(仕切り板レーン)', (0.3, -2.0)),
          (REF_A + 7.6, 'HS ホームストレート', (2.6, 1.6)), (REF_A + 9.1, 'LH 黒カーテン前の\n左ヘアピン', (2.6, -0.9))]
for tau, txt, (ox, oy) in labels:
    px, py = pos(tau)
    ax.annotate(txt, xy=(px, py), xytext=(px + ox, py + oy), fontsize=9, color=TXT, ha='center', bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec='none', alpha=0.85),
                arrowprops=dict(arrowstyle='-', color='#b9b8b2', lw=0.8))
offs = {'AI 006': 0.35, 'AI 007': -0.35, '人(学習データ)': 0.0}
mk = {'AI 006': 'o', 'AI 007': '^', '人(学習データ)': 'X'}
counts = {}
rng = np.random.default_rng(0)
for gname, logs in groups.items():
    pts = []
    for log in logs:
        for e in C.get(log, []):
            if 'sel' in e and log.startswith('20261010-17') and not e['sel']: continue
            if (log, round(e['t'], 2)) in REC: continue
            tau = tau_of(e)
            if tau is None:
                dd = load(log) if log.startswith('20261010-17') else D(log)
                continue_match = True
                # fall back to matched phase stored in sec (SW/HS/LH): place at section centre +- jitter
                cen = {'SW': REF_A + 5.2, 'HS': REF_A + 7.6, 'LH': REF_A + 9.1, 'R1': REF_A + 0.3, 'GM_in': REF_A + 1.4, 'DE': ref_tg + 1.15, 'GM_out': ref_tg}[e['sec'].rstrip('?')]
                tau = cen + rng.uniform(-0.3, 0.3)
            px, py = pos(tau, offs[gname] + rng.uniform(-0.12, 0.12))
            pts.append((px, py, e['g']))
    pts = np.array(pts); counts[gname] = len(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=12 + 9 * pts[:, 2], marker=mk[gname], facecolor=COL[gname], edgecolor=SURF, linewidth=1.0, alpha=0.85, zorder=3,
               label='%s 衝突 %d件' % (gname, len(pts)))
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.12)
for s in ('left', 'bottom'): ax.spines[s].set_visible(False)
ax.set_title('衝突マップ(ジャイロ+eRPMの推定コース、縮尺は任意)', fontsize=12, color=TXT, loc='left')
leg = ax.legend(loc='upper right', fontsize=9, frameon=False, title='マーカーの大きさ = 衝撃 |a|(g)', title_fontsize=8)
for t_ in leg.get_texts(): t_.set_color(TXT)
fig.tight_layout(); fig.savefig(REP + '/cmap_fig_course_map.png', dpi=110, facecolor=SURF)
print(counts)
# ---------- rates ----------
R1P = pickle.load(open('r1pass.pkl', 'rb')); DEP = pickle.load(open('depass.pkl', 'rb'))
def r1rate(sel):
    L = [x for x in R1P['AI' if sel != 'HU' else 'HU'] if (sel == 'HU' or x[0] == sel)]
    return sum(1 for x in L if x[3] >= 4), len(L)
def derate(sel):
    L = [x for x in DEP['AI' if sel != 'HU' else 'HU'] if (sel == 'HU' or x[0] == sel)]
    return sum(1 for x in L if x[5] >= 4), len(L)
def gmrate(sel):
    logs = groups['人(学習データ)'] if sel == 'HU' else (sel,)
    n_ev = sum(1 for log in logs for e in C.get(log, []) if e['sec'].rstrip('?') in ('GM_in', 'GM_out') and (not log.startswith('20261010-17') or e['sel']) and (log, round(e['t'], 2)) not in REC)
    k, n = r1rate(sel)
    return n_ev, n - k
rows = {'AI 006': '20261010-172910', 'AI 007': '20261010-173445', '人(学習データ)': 'HU'}
spots = [('R1 右ヘアピン\n(強い衝突≥4g / 通過)', r1rate), ('GM 緑マット\n(接触≥2.5g / 通過)', gmrate), ('DE 右外壁の角\n(強い衝突≥4g / 通過)', derate)]
fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURF); style(ax)
w = 0.26
res = {}
for j, (gname, sel) in enumerate(rows.items()):
    for i, (lab, fn) in enumerate(spots):
        k, n = fn(sel); r = k / n if n else 0; res[(gname, lab.split()[0])] = (k, n)
        b = ax.bar(i + (j - 1) * (w + 0.02), r * 100, width=w, color=COL[gname], edgecolor=SURF, linewidth=2, label=gname if i == 0 else None)
        ax.text(i + (j - 1) * (w + 0.02), r * 100 + 1.5, '%d/%d' % (k, n), ha='center', va='bottom', fontsize=8.5, color=TXT2)
ax.set_xticks(range(3)); ax.set_xticklabels([s[0] for s in spots], fontsize=9.5, color=TXT)
ax.set_ylabel('通過あたりの割合 (%)', color=TXT2, fontsize=9.5); ax.set_ylim(0, 85)
ax.set_title('場所ごとの衝突率(AI側の周回のみ、人は学習用5ログ合計)', fontsize=11.5, color=TXT, loc='left')
leg = ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper left')
fig.tight_layout(); fig.savefig(REP + '/cmap_fig_spot_rates.png', dpi=110, facecolor=SURF)
print(res)
# ---------- R1 entry speed ----------
fig, ax = plt.subplots(figsize=(8.6, 3.6), facecolor=SURF); style(ax)
ax.axvspan(5500, 6200, color='#eda100', alpha=0.10, lw=0)
for j, (gname, sel) in enumerate(rows.items()):
    L = [x for x in R1P['AI' if sel != 'HU' else 'HU'] if (sel == 'HU' or x[0] == sel)]
    vv = np.array([x[2] for x in L]); gg = np.array([x[3] for x in L])
    yj = 2 - j + rng.uniform(-0.12, 0.12, len(vv))
    hit = gg >= 4
    ax.scatter(vv[~hit], yj[~hit], s=46, facecolor='none', edgecolor=COL[gname], linewidth=1.6, zorder=3)
    ax.scatter(vv[hit], yj[hit], s=58, facecolor=COL[gname], edgecolor=SURF, linewidth=1.0, zorder=4)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(list(rows.keys()), fontsize=9.5, color=TXT)
ax.set_xlabel('R1 進入速度 = 右ヘアピン開始 −0.4〜+0.15 s の最大 eRPM(前進)', fontsize=9.5, color=TXT2)
ax.set_title('R1 右ヘアピン: 塗り = 強い衝突(≥4 g)、白抜き = 通過(軽い接触含む)', fontsize=11, color=TXT, loc='left')
ax.text(5850, -0.42, '5500〜6200: 境目', ha='center', fontsize=8.5, color=TXT2)
ax.set_ylim(-0.5, 2.6)
fig.tight_layout(); fig.savefig(REP + '/cmap_fig_r1_entry_speed.png', dpi=110, facecolor=SURF)
# ---------- DE heading vs speed ----------
fig, ax = plt.subplots(figsize=(7.6, 5.4), facecolor=SURF); style(ax)
for gname, sel in rows.items():
    L = [x for x in DEP['AI' if sel != 'HU' else 'HU'] if (sel == 'HU' or x[0] == sel)]
    vv = np.array([x[2] for x in L]); pp = np.array([x[4] for x in L]); gg = np.array([x[5] for x in L]); hit = gg >= 4
    ax.scatter(pp[~hit], vv[~hit], s=46, facecolor='none', edgecolor=COL[gname], linewidth=1.6, marker=mk[gname] if gname != '人(学習データ)' else 's', label='%s 通過' % gname)
    ax.scatter(pp[hit], vv[hit], s=58, facecolor=COL[gname], edgecolor=SURF, linewidth=1.0, marker=mk[gname] if gname != '人(学習データ)' else 's', label='%s 衝突≥4g' % gname)
ax.axvline(0, color='#b9b8b2', lw=1)
ax.set_xlabel('緑マット出口+0.6 s の向き(度、R1開始時の向き基準、+ = 左 / − = 右の外壁側)', fontsize=9, color=TXT2)
ax.set_ylabel('速度 eRPM(出口+0.4〜0.9 s)', fontsize=9, color=TXT2)
ax.set_title('DE: AIは人より右(外壁側)を向いて進入する', fontsize=11.5, color=TXT, loc='left')
leg = ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16))
fig.tight_layout(); fig.savefig(REP + '/cmap_fig_de_heading.png', dpi=110, facecolor=SURF)
print('figs ok')
