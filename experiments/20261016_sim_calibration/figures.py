#!/usr/bin/env python3
"""シミュレーター較正(2026-10-16)S1 の図: 同定の結果(ident/*.json・profiles)と当てはめの窓から描く

    jf-heavy --mem 1500M --lock none -- nice -n 10 python -I figures.py

出力: /home/jetson/jetformer_reports/sim_calibration_20261016/fig/*.png。読み取りだけ(CPU)。
色: 実際 = 墨、シミュレーターの今の物理 = 橙、較正 = 青、AI = 水色(人と AI を並べるとき、人 = 青)。
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib_model as cm  # noqa: E402
import ident  # noqa: E402

REPORT = ident.REPORT
FIG = REPORT / 'fig'
OUTD = ident.OUTD
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8f8e88', '#e4e3de', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

for f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({
    'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.titlesize': 11, 'axes.titlecolor': INK,
    'legend.frameon': False, 'lines.linewidth': 2.0, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.unicode_minus': False,
})


def j(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', FIG / name)


# ------------------------------------------------------------------ 舵
def fig_steering(st, d4, P_cal):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    nn = np.linspace(-1, 1, 401)
    ax.plot(nn, np.tan(nn * 0.52) / ident.L_WB, color=ORANGE, label='シミュレーターの今(直線、±0.52 rad)')
    for op, col, mk in (('human', BLUE, 'o'), ('ai', AQUA, 's')):
        d = st[op]
        tab = np.array(d['bins_low_speed'])
        own = tab
        if op == 'ai' and d['bins_filled_from_human']:
            # 自分の帯だけ点で描く(埋めた帯は描かない)
            filled = [0.5 * (a + b) for a, b in d['bins_filled_from_human']]
            keep = [not any(abs(x - f) < 0.03 for f in filled) for x in tab[:, 0]]
            own = tab[np.array(keep)]
        ax.scatter(own[:, 0], np.tan(own[:, 2]) / ident.L_WB, s=np.clip(own[:, 1] / 8, 12, 60), color=col, marker=mk, alpha=0.75,
                   edgecolor=SURF, linewidth=0.8, zorder=3)
        if op == P_cal['steering']['map_from']:
            stp = P_cal['steering']
            xs_ = np.concatenate([-np.array(stp['knots_n'])[::-1], np.array(stp['knots_n'])[1:]])
            ys_ = np.concatenate([-np.array(stp['delta_R_rad'])[::-1], np.array(stp['delta_L_rad'])[1:]])
            ys_[len(stp['knots_n']) - 1] = stp['delta_L_rad'][0]
            ax.plot(xs_, np.tan(ys_) / ident.L_WB, color=col, zorder=4,
                    label='較正の値・%s の対応(中心 = %s)' % ('人' if op == 'human' else 'AI', '0' if stp.get('center_choice') == 'zero_center' else 'I1'))
            ax.plot(ident.STEER_FIT_KNOTS, np.tan(np.array(d['delta_rad'])) / ident.L_WB, color=col, lw=1.2, ls=':', zorder=4,
                    label='I1 の当てはめのまま(中心 +%.2f /m)' % (math.tan(d['delta_rad'][5]) / ident.L_WB))
        else:
            ax.plot(ident.STEER_FIT_KNOTS, np.tan(np.array(d['delta_rad'])) / ident.L_WB, color=col,
                    label='実車の当てはめ・%s(参考)' % ('人' if op == 'human' else 'AI'), zorder=4)
    ax.axvline(ident.HUMAN_LEFT_MAX, color=MUTED, lw=1, ls=':')
    ax.text(ident.HUMAN_LEFT_MAX - 0.02, -1.9, '人の左の上限 0.81', ha='right', color=INK2, fontsize=8)
    ax.set_xlabel('舵の指令(正規化 n = u_s / 0.52、+ = 左)')
    ax.set_ylabel('低速の曲率 κ0 [1/m]')
    ax.set_title('(a) 舵の PWM → 曲率(0.8〜2.2 m/s、舵が 0.3 秒一定、GM を除く)')
    ax.legend(loc='upper left', fontsize=8)
    ax.text(0.98, 0.04, '点の大きさ = 帯のサンプル数', transform=ax.transAxes, ha='right', color=INK2, fontsize=8)
    # (b) アンダーステア: κ/κ0 対 速さ
    ax = axs[1]
    S, W, logs = ident.load_cache()
    vv = np.linspace(0, 4.2, 100)
    for op, col, mk in ((0, BLUE, 'o'), (1, AQUA, 's')):
        f = st[ident.OPS[op]]
        m = ident.steer_samples(S, op, 0.8, 4.0)
        k0 = ident.kappa_of(S['n_s'][m], np.array(f['delta_rad']))
        v = S['v5'][m]; kap = S['r'][m] / v
        ok = (np.abs(k0) > 0.4) & (np.abs(k0) * v * v <= ident.A_LIN_MAX)
        ratio = kap[ok] / k0[ok]
        edges = np.arange(0.8, 4.01, 0.4)
        xs, ys, ns = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            k = (v[ok] >= a) & (v[ok] < b)
            if k.sum() >= 20:
                xs.append(np.median(v[ok][k])); ys.append(np.median(ratio[k])); ns.append(k.sum())
        ax.scatter(xs, ys, color=col, marker=mk, s=40, zorder=3, edgecolor=SURF, linewidth=0.8)
        K = f['K_us_I2']
        ax.plot(vv, 1 / (1 + K * vv * vv), color=col, label='%s: K = %.3f%s' % ('人' if op == 0 else 'AI', K,
                                                                            '(使う)' if abs(K - P_cal['steering']['K_us']) < 1e-12 and P_cal['steering']['map_from'] == ident.OPS[op] else ''))
    ax.axhline(1.0, color=ORANGE, lw=2, label='シミュレーターの今(K = 0)')
    ax.set_ylim(0.4, 1.3)
    ax.set_xlabel('速さ [m/s]')
    ax.set_ylabel('曲率 / 低速の曲率 κ/κ0')
    ax.set_title('(b) アンダーステア(|κ0| > 0.4、要求の横加速度 ≤ 5 m/s²、帯の中央値)')
    ax.legend(loc='lower left', fontsize=8)
    save(fig, 'steer_map_understeer.png')


def fig_latency(d4, i8):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    rows = i8['by_log']
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.plot([r['p50'], r['p99']], [i, i], color=MUTED, lw=2)
        ax.scatter([r['p50']], [i], color=INK, s=30, zorder=3)
        ax.scatter([r['p99']], [i], color=INK2, s=30, marker='|', zorder=3)
    ax.axvline(1000 / 30, color=ORANGE, lw=2)
    ax.text(1000 / 30 + 2, len(rows) - 1.5, 'アプリの遅れ\n1 フレーム 33 ms', color=INK2, fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([r['log'][9:] for r in rows], fontsize=8)
    ax.set_xlabel('撮像 → PWM [ms](点 = p50、縦線 = p99)')
    ax.set_title('(a) AI の撮像 → PWM の遅れ(当てはめのシーン)')
    seq = ['#c9dcf3', '#a6c6ee', '#7eabe5', '#5690db', '#2a78d6', '#1f5fae', '#164784', '#0e2f59']
    for k, op in enumerate(('ai', 'human')):
        ax = axs[k + 1]
        tab = np.array(d4[op]['yaw_rmse_table'])
        for jj, rs in enumerate(d4['r_s_grid']):
            ax.plot(np.array(d4['d_s_grid']) * 1000, tab[:, jj], color=seq[jj], lw=1.6,
                    label='%s rad/s' % ('∞' if rs is None else ('%g' % rs)))
        ax.axhline(d4[op]['yaw_rmse_sim_default_lat_only'], color=ORANGE, lw=2, ls='--')
        ax.text(2, d4[op]['yaw_rmse_sim_default_lat_only'] + 0.005, 'シミュレーターの今(遅れ 0・1.2 rad/s・直線の対応)', color=INK2, fontsize=8)
        ax.scatter([d4[op]['d_s'] * 1000], [d4[op]['yaw_rmse_best']], color=INK, s=60, zorder=5, marker='*')
        ax.annotate('最良 %.0f ms・%s rad/s' % (d4[op]['d_s'] * 1000, d4[op]['r_s']), (d4[op]['d_s'] * 1000, d4[op]['yaw_rmse_best']),
                    xytext=(10, -18), textcoords='offset points', fontsize=8, color=INK)
        ax.set_xlabel('舵の遅れ d_s [ms](PWM の時刻から)')
        ax.set_ylabel('ヨーレートの RMSE [rad/s]')
        ax.set_title('(%s) 舵の遅れと速さ・%s の窓 %d(横だけの再生)' % ('b' if k == 0 else 'c', 'AI' if op == 'ai' else '人', d4[op]['n_windows']))
        if k == 0:
            ax.legend(title='舵の速さ r_s', fontsize=7, title_fontsize=8, ncol=2, loc='upper right')
    save(fig, 'latency_steer_dynamics.png')


def fig_lateral(gr, P_cal):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    secs = ['R1', 'GM', 'DE', 'LH', 'HS', 'SW']
    surf = {'R1': 'shortcut', 'GM': 'turf', 'DE': 'carpet', 'LH': 'carpet', 'HS': 'carpet', 'SW': 'carpet'}
    y = np.arange(len(secs) + 1)[::-1]
    S, W, logs = ident.load_cache()
    secn = np.array([ident.cd.SEC_NAME[ident.cd.LOC[c]] for c in S['loc']])
    base = S['clean'] & np.isfinite(S['a_lat']) & (S['v5'] >= 1.0)
    p95 = {sn: float(np.percentile(np.abs(S['a_lat'][base & (secn == sn)]), 95)) for sn in secs if (base & (secn == sn)).sum() > 200}
    for i, sname in enumerate(secs):
        b = gr['by_section'][sname]
        yy = y[i]
        if b['median_alat'] is not None:
            ax.plot([b['p5'], b['p95']], [yy, yy], color=INK2, lw=2)
            ax.scatter([b['median_alat']], [yy], color=INK if b['enough'] else SURF, edgecolor=INK, s=50, zorder=3)
            ax.text(0.2, yy + 0.22, 'n = %d%s' % (b['n'], '' if b['enough'] else '(< 200)'), va='center', fontsize=8, color=INK2)
        if sname in p95:
            ax.scatter([p95[sname]], [yy], color=MUTED, marker='^', s=40, zorder=3)
        ax.scatter([ident.SIM_MU[surf[sname]] * cm.G], [yy], color=ORANGE, marker='D', s=40, zorder=4)
        ax.scatter([P_cal['surfaces_mu'][surf[sname]] * cm.G], [yy], color=BLUE, marker='|', s=300, lw=2.5, zorder=4)
    c = gr['surfaces']['carpet']
    yy = y[-1]
    ax.plot([c['mu_p5'] * cm.G, c['mu_p95'] * cm.G], [yy, yy], color=INK2, lw=2)
    ax.scatter([c['mu'] * cm.G], [yy], color=INK, s=50, zorder=3)
    ax.scatter([0.8 * cm.G], [yy], color=ORANGE, marker='D', s=40, zorder=4)
    ax.scatter([P_cal['surfaces_mu']['carpet'] * cm.G], [yy], color=BLUE, marker='|', s=300, lw=2.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(['%s(%s)' % (s, surf[s]) for s in secs] + ['カーペットにまとめた\n(DE・LH・HS・SW)'])
    ax.set_xlabel('横加速度の上限 μ g [m/s²]')
    ax.set_title('路面ごとの横の限界: 滑りの時刻の |a_lat| の中央値(線 = 90% 区間)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, marker='o', ls='', label='実車の滑りの時刻の中央値(白抜き = 200 未満)'),
         Line2D([], [], color=BLUE, marker='|', ls='', markersize=14, mew=2.5, label='較正の値(路面、I3 × 倍率 %.1f)' % P_cal.get('mu_scale', 1.0)),
         Line2D([], [], color=ORANGE, marker='D', ls='', label='シミュレーターの今の μ g'),
         Line2D([], [], color=MUTED, marker='^', ls='', label='参考: 走っている時刻の |a_lat| の p95(v ≥ 1 m/s)')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    ax.set_xlim(0, 11)
    save(fig, 'lateral_limits.png')


def fig_speed(P_cal, extra):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    uu = np.linspace(0, 7.24, 300)
    ax.plot(uu, uu, color=ORANGE, label='シミュレーターの今(T(u) = u)')
    lon = P_cal['longitudinal']
    ax.plot(uu, cm.speed_target(uu, {'u_db': lon['u_db'], 'T': lon['T_values']}), color=BLUE, label='較正の値(人の窓 + 定常の表を弱い事前に)')
    b_ = ident.i6_boot_summary(0, 'human')
    if b_ is not None:
        lo = cm.speed_target(uu, {'u_db': lon['u_db'], 'T': b_['T_p5']})
        hi = cm.speed_target(uu, {'u_db': lon['u_db'], 'T': b_['T_p95']})
        ax.fill_between(uu, lo, hi, color=BLUE, alpha=0.15, lw=0)
    fh = ident.jload(OUTD / 'i6' / 'human_final_full.json')
    ax.plot(uu, cm.speed_target(uu, fh['prm']), color=BLUE, ls=':', lw=1.6, label='人の窓だけの当てはめ(事前なし、参考)')
    pa = OUTD / 'i6' / 'ai_input_map.json'
    if pa.exists():
        fa = ident.jload(pa)
        ax.plot(uu, cm.speed_target(uu, fa['prm']), color=AQUA, ls='--', lw=1.6, label='AI の窓の当てはめ(人の T(a u + b)、参考)')
    # 人の一定のスロットルの表(evidence_simulator.md 6.3)
    tab = extra['steady_check']['table_fit_scenes_human']
    ax.scatter([r_['u'] for r_ in tab], [r_['v_median'] for r_ in tab], color=INK, s=36, zorder=4,
               label='人の一定のスロットルの速さ(当てはめのシーン、直線)')
    ax.set_xlabel('速さの指令 u_v [m/s](スロットル PWM = 1498 − 68.8 u_v)')
    ax.set_ylabel('目標の速さ T(u_v) [m/s]')
    ax.set_title('(a) スロットル → 速さの対応(帯 = 90% 区間、ブートストラップ)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0, 7.3); ax.set_ylim(0, 7.3)
    # (b) 例の窓
    ax = axs[1]
    S, W, logs = ident.load_cache()
    op = 0 if P_cal['longitudinal']['map_from'] == 'human' else 1   # 'ai_scaled' → AI の窓
    Wx = ident.sub_windows(W, W['op'] == op)
    tru = cm.truth(Wx)
    gr = ident.jload(OUTD / 'i3_grip.json')
    mu = ident.mu_windows(Wx, {k: v['mu'] for k, v in gr['surfaces'].items()})
    mu0 = np.array([ident.SIM_MU[s] for s in Wx['surface']])
    rd = cm.replay(Wx, 'sim', dict(ident.SIM_DEFAULT_CFG), mode='lon', mu=mu0, record=('v',))
    rc = cm.replay(Wx, 'cal', ident.cal_prm_from_profile(P_cal), mode='lon', mu=mu, k_batt=Wx['k_batt'], record=('v',))
    # 速さが大きく変わる窓を 1 つ選ぶ(加速 → 戻し)
    dv = tru['v'][:, -1] - tru['v'][:, 0]
    tv = np.abs(np.diff(Wx['u_v'][:, 70:171], axis=1)).sum(axis=1)
    cand = np.nonzero(np.abs(dv) > 0.8)[0]
    i = int(cand[np.argmin(tv[cand])]) if len(cand) else int(np.argmax(np.abs(dv)))
    t = np.arange(101) * 0.01
    ax.plot(t, tru['v'][i], color=INK, label='実車(eRPM)')
    ax.plot(t, rd['v'][i], color=ORANGE, label='シミュレーターの今')
    ax.plot(t, rc['v'][i], color=BLUE, label='較正(P_cal)')
    ax.plot(t, Wx['u_v'][i, 70:171], color=MUTED, lw=1.2, ls='--', label='速さの指令 u_v')
    ax.set_xlabel('窓の始まりからの時間 [s]')
    ax.set_ylabel('速さ [m/s]')
    ax.set_title('(b) 例: %s の窓(%s・%s、速さだけの再生)' % ('人' if op == 0 else 'AI', Wx['scene_name'][i], Wx['sec'][i]))
    ax.legend(fontsize=8, loc='best')
    save(fig, 'speed_map_fit.png')


def fig_speed_examples(P_cal):
    S, W, logs = ident.load_cache()
    tru = cm.truth(W)
    gr = ident.jload(OUTD / 'i3_grip.json')
    mu = ident.mu_windows(W, {k: v['mu'] for k, v in gr['surfaces'].items()})
    mu0 = np.array([ident.SIM_MU[s] for s in W['surface']])
    rd = cm.replay(W, 'sim', dict(ident.SIM_DEFAULT_CFG), mode='coupled', mu=mu0)
    rc = cm.replay(W, 'cal', ident.cal_prm_from_profile(P_cal), mode='coupled', mu=mu, k_batt=W['k_batt'])
    rng = np.random.default_rng(7)
    picks = []
    for op, sec in ((1, 'R1'), (1, 'GM'), (0, 'R1'), (0, 'GM'), (0, 'DE'), (1, 'LH')):
        idx = np.nonzero((W['op'] == op) & (W['sec'] == sec) & (np.abs(tru['psi'][:, -1]) > 0.8))[0]
        if len(idx) == 0:
            idx = np.nonzero((W['op'] == op) & (W['sec'] == sec))[0]
        if len(idx):
            picks.append(int(rng.choice(idx)))
    fig, axs = plt.subplots(2, len(picks), figsize=(3.0 * len(picks), 5.6), sharex=True)
    t = np.arange(101) * 0.01
    for c, i in enumerate(picks):
        ax = axs[0, c]
        r_sim_d = np.concatenate([[np.nan], np.diff(rd['psi'][i]) / 0.01])
        r_sim_c = np.concatenate([[np.nan], np.diff(rc['psi'][i]) / 0.01])
        ax.plot(t, tru['r'][i], color=INK, lw=1.8, label='実車(ジャイロ)')
        ax.plot(t, r_sim_d, color=ORANGE, lw=1.8, label='シミュレーターの今')
        ax.plot(t, r_sim_c, color=BLUE, lw=1.8, label='較正(P_cal)')
        ax.set_title('%s・%s\n%s' % ('AI' if W['op'][i] else '人', W['sec'][i], W['scene_name'][i][9:]), fontsize=9)
        if c == 0:
            ax.set_ylabel('ヨーレート [rad/s]')
        ax = axs[1, c]
        ax.plot(t, tru['v'][i], color=INK, lw=1.8)
        ax.plot(t, rd['v'][i], color=ORANGE, lw=1.8)
        ax.plot(t, rc['v'][i], color=BLUE, lw=1.8)
        ax.set_xlabel('時間 [s]')
        if c == 0:
            ax.set_ylabel('速さ [m/s]')
    axs[0, 0].legend(fontsize=7, loc='best')
    fig.suptitle('当てはめの窓の例: 実車の指令を 1 秒流した結果(結合した再生、同じ始めの速さ)', color=INK, fontsize=11)
    save(fig, 'replay_examples_fit_windows.png')


def fig_coast_brake(c5, b7):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axs[0]
    S, W, logs = ident.load_cache()
    m = ident.coast_mask(S)
    al = np.abs(S['v5'][m] * S['r'][m]); dc = -S['acc'][m]
    ax.scatter(al, dc, s=5, color=MUTED, alpha=0.35, lw=0, label='惰性の時刻(100 Hz)')
    cc = b7['coast_curve_used_d5']
    ax.plot(cc['alat'], cc['decel'], color=INK, marker='o', ms=6, lw=1.5, label='帯の中央値(実測の惰性の曲線)')
    xx = np.linspace(0, 11, 50)
    ax.plot(xx, c5['c0'] + c5['k_s'] * xx, color=BLUE, label='較正: c0 + k_s |a_lat| = %.2f + %.2f |a_lat|' % (c5['c0'], c5['k_s']))
    ax.axhline(1.2, color=ORANGE, label='シミュレーターの今(一律 1.2)')
    ax.set_ylim(-1, 8); ax.set_xlim(0, 11.5)
    ax.set_xlabel('横加速度 |a_lat| = |v r| [m/s²]')
    ax.set_ylabel('減速 [m/s²]')
    ax.set_title('(a) 惰性(中立 ±12 µs を 0.3 秒以上)の減速とこすれ')
    ax.legend(fontsize=8, loc='upper left')
    ax = axs[1]
    ev = b7['events']
    for e in ev:
        mk = 'o' if e['op'] == 'human' else 's'
        ax.scatter([e['off_us']], [e['a_b_d5']], color=INK if e['op'] == 'human' else INK2, marker=mk, s=40, zorder=3,
                   facecolor=SURF if e['in_heldout_scene'] else (INK if e['op'] == 'human' else INK2))
    oo = np.linspace(0, 400, 200)
    for key, col, ls, lab in (('nominal', BLUE, '-', '名目'), ('low', BLUE, ':', '弱め'), ('high', BLUE, '--', '強め')):
        sc = b7['scenarios'][key]
        ax.plot(oo, sc['s_b'] * np.maximum(0, oo - sc['db_b']), color=col, ls=ls, lw=2 if key == 'nominal' else 1.5,
                label='%s: +200 µs で %.2f m/s²' % (lab, sc['a_b_at_200us']))
    lit = b7['protocol_literal']
    ax.plot(oo, lit['s_b'] * np.maximum(0, oo - lit['db_b']), color=MUTED, lw=1.5, label='字義どおりの I7(参考)')
    ax.plot(oo, np.where(oo > 0, 1.2 - c5['c0'], 0), color=ORANGE, label='シミュレーターの今(一律 1.2 − 惰性)')
    ax.axvline(200, color=GRID, lw=1)
    ax.text(203, -1.3, '上限の層のブレーキ +200 µs', fontsize=8, color=INK2)
    ax.set_xlabel('スロットルの中立からのずれ off [µs](+ = ブレーキ側)')
    ax.set_ylabel('惰性より多い減速 a_b [m/s²]')
    ax.set_title('(b) ESC のブレーキ(きれいな 16 回。○ 人・□ AI、白抜き = 保留のシーン)')
    ax.legend(fontsize=7, loc='upper left')
    save(fig, 'coast_brake.png')


def main():
    st = j(OUTD / 'i1_i2_steering.json'); d4 = j(OUTD / 'i4_steer_dynamics.json'); gr = j(OUTD / 'i3_grip.json')
    c5 = j(OUTD / 'i5_coast.json'); b7 = j(OUTD / 'i7_brake.json'); P_cal = j(REPORT / 'profiles' / 'P_cal.json')
    extra = j(OUTD / 'final_decisions.json')
    fig_steering(st, d4, P_cal)
    fig_latency(d4, extra['i8'])
    fig_lateral(gr, P_cal)
    fig_speed(P_cal, extra)
    fig_coast_brake(c5, b7)
    fig_speed_examples(P_cal)


if __name__ == '__main__':
    main()
