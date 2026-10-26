"""原因 C の診断(5): レポートの数のまとめと図。2026-10-26

入力: out/launches.json・out/dyn_fit.json・out/sim.json・out/check_dyn.json
出力: out/summary.json と /home/jetson/jetformer_reports/fix_20261026/fig_launch/*.png

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I report_figs.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim as S  # noqa: E402

OUT = S.OUT
FIG = Path('/home/jetson/jetformer_reports/fix_20261026/fig_launch')
FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FP)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FP).get_name()
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 15, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.spines.top': False,
                     'axes.spines.right': False})


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(st.median(xs)), 3) if xs else None


def rng(xs):
    xs = [x for x in xs if x is not None]
    return [round(float(min(xs)), 3), round(float(max(xs)), 3)] if xs else None


def group_stats(rows):
    lp = [r['loss_parts'] for r in rows]
    tg = [r['tgrid'] for r in rows]

    def t_reach(g, key, thr):
        t, y = np.array(g['t']), np.array(g[key], float)
        k = np.nonzero(y >= thr)[0]
        return float(t[k[0]]) if len(k) else None

    def peak(g, key, w=1.0):
        t, y = np.array(g['t']), np.array(g[key], float)
        return float(np.nanmax(y[t < w]))
    return {
        'n': len(rows), 'logs': [(r['log'], r['no']) for r in rows],
        'to_r1_s': [med([r['to_r1_s'] for r in rows]), rng([r['to_r1_s'] for r in rows])],
        'dist_m': [med([r['dist_m'] for r in rows]), rng([r['dist_m'] for r in rows])],
        'loss_s': [med([r['loss_s'] for r in rows]), rng([r['loss_s'] for r in rows])],
        'dead_s': [med([x['dead_s'] for x in lp]), rng([x['dead_s'] for x in lp])],
        'straight_s': [med([x['straight_s'] for x in lp]), rng([x['straight_s'] for x in lp])],
        'lh_to_r1_s': [med([x['lh_to_r1_s'] for x in lp]), rng([x['lh_to_r1_s'] for x in lp])],
        'x_lh_m': [med([r['x_lh_m'] for r in rows]), rng([r['x_lh_m'] for r in rows])],
        't_launch_to_lh': [med([r['t_launch_to_lh'] for r in rows]), rng([r['t_launch_to_lh'] for r in rows])],
        't_fly_to_lh': [med([r['t_fly_to_lh'] for r in rows]), rng([r['t_fly_to_lh'] for r in rows])],
        'v_launch_at_lh': [med([r['v_launch_at_lh'] for r in rows]), rng([r['v_launch_at_lh'] for r in rows])],
        'v_fly_at_lh': [med([r['v_fly_at_lh'] for r in rows]), rng([r['v_fly_at_lh'] for r in rows])],
        'straight_f_out_launch': med([r.get('straight_f_out_launch') for r in rows]),
        'straight_f_out_fly': med([r.get('straight_f_out_fly') for r in rows]),
        'straight_v_launch': med([r.get('straight_v_launch') for r in rows]),
        'straight_v_fly': med([r.get('straight_v_fly') for r in rows]),
        'lh_v_launch': med([r.get('lh_v_launch') for r in rows]), 'lh_v_fly': med([r.get('lh_v_fly') for r in rows]),
        'lh_f_out_launch': med([r.get('lh_f_out_launch') for r in rows]),
        'lh_f_out_fly': med([r.get('lh_f_out_fly') for r in rows]),
        'kick_frames': [med([r['kick_frames'] for r in rows]), rng([r['kick_frames'] for r in rows])],
        'n_kick': sum(r['kick'] for r in rows),
        'n_model_brake_start': sum(min(r['f_model_first5'][:3]) <= -40 for r in rows),
        'n_model_below40_first3': sum(max(r['f_model_first5'][:3]) < 40 for r in rows),
        'move_0p2_s': [med([r['move_0p2_s'] for r in rows]), rng([r['move_0p2_s'] for r in rows])],
        'model80_s': [med([r['model80_s'] for r in rows]), rng([r['model80_s'] for r in rows])],
        't_f150_s': [med([t_reach(g, 'f_model', 150) for g in tg]), rng([t_reach(g, 'f_model', 150) for g in tg])],
        't_v2_s': [med([t_reach(g, 'v', 2.0) for g in tg]), rng([t_reach(g, 'v', 2.0) for g in tg])],
        'f_model_mean_0p5': med([r['f_model_mean_0.5s'] for r in rows]),
        'f_model_mean_1p0': med([r['f_model_mean_1.0s'] for r in rows]),
        'peak_a_imu_1s': [med([peak(g, 'a_imu') for g in tg]), rng([peak(g, 'a_imu') for g in tg])],
        'slip_ratio': [med([(r.get('slip') or {}).get('ratio') for r in rows]),
                       rng([(r.get('slip') or {}).get('ratio') for r in rows])],
        's_abs_0p3': med([float(np.nanmean(np.abs(np.array(g['s_out'], float)[np.array(g['t']) < 0.3]))) for g in tg]),
        'dpsi_0p5_abs': [med([abs(float(np.trapz(np.array(g['gz'])[np.array(g['t']) < 0.5],
                                                  np.array(g['t'])[np.array(g['t']) < 0.5]))) for g in tg]),
                         rng([abs(float(np.trapz(np.array(g['gz'])[np.array(g['t']) < 0.5],
                                                  np.array(g['t'])[np.array(g['t']) < 0.5]))) for g in tg])],
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    L = json.loads((OUT / 'launches.json').read_text())
    sim = json.loads((OUT / 'sim.json').read_text())
    fit = json.loads((OUT / 'dyn_fit.json').read_text())
    R = L['launches']
    b24 = {tuple(x) for x in sim['summary']['B_1024_logs']}
    alls = {tuple(x) for x in sim['summary']['all_logs']}
    rows_b = [r for r in R if (r['log'], r['no']) in b24 and r.get('grid')]
    rows_a = [r for r in R if (r['log'], r['no']) in alls and r.get('grid')]
    inv = {'logs_with_ai': len({r['log'] for r in R}), 'launches': len(R), 'kick': sum(r['kick'] for r in R),
           'no_kick': sum(not r['kick'] for r in R), 'reached_r1': sum(r.get('to_r1_s') is not None for r in R),
           'with_fly_ref': sum(r.get('loss_s') is not None for r in R),
           'no_r1_short_seg': sum(r.get('why_no_r1') == 'AI の区間が短い' for r in R),
           'no_r1_15s': sum(r.get('why_no_r1') == '15 秒以内に起点なし' for r in R),
           'start_kick_param_none': sum(r['start_kick_param'] is None for r in R)}
    # 補助あり・なし(動き出し)
    kick_rows = [r for r in R if r['kick'] and r['move_0p2_s'] is not None]
    nok_rows = [r for r in R if not r['kick'] and r['move_0p2_s'] is not None and r['start_kick_param'] == 1]
    old_rows = [r for r in R if r['start_kick_param'] is None]
    kick_cmp = {'kick': {'n': len(kick_rows), 'move_0p2_s': [med([r['move_0p2_s'] for r in kick_rows]), rng([r['move_0p2_s'] for r in kick_rows])],
                         'kick_s': [med([r['kick_s'] for r in kick_rows]), rng([r['kick_s'] for r in kick_rows])],
                         'kick_end_s': [med([r['kick_end_s'] for r in kick_rows]), rng([r['kick_end_s'] for r in kick_rows])]},
                'no_kick_param1': {'n': len(nok_rows), 'move_0p2_s': [med([r['move_0p2_s'] for r in nok_rows]), rng([r['move_0p2_s'] for r in nok_rows])]},
                'before_kick_1023': {'n': len(old_rows), 'never_moved': sum(r['move_0p2_s'] is None for r in old_rows),
                                     'move_0p2_s': [med([r['move_0p2_s'] for r in old_rows]), rng([r['move_0p2_s'] for r in old_rows])]}}
    summ = {'inventory': inv, 'kick_cmp': kick_cmp, 'B_1024': group_stats(rows_b), 'all_0p2_1p2': group_stats(rows_a),
            'car_model': sim['car_model'], 'validation_summary': sim['validation_summary'],
            'fit_dyn_best': fit['best'], 'a_imu_p90_low_speed_f250': fit['a_imu_p90_low_speed_f250'],
            'a_imu_p99_low_speed': fit['a_imu_p99_low_speed']}
    (OUT / 'summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))

    # ---- 図 1: 速さ ↔ 発進からの距離(10/24 の B 9 回)。記録の発進・周の同じ場所・案 L250・3.0 の再生
    cm = sim['car_model']
    p = [cm['alpha_mps2_per_us'], cm['f0_us'], cm['beta_per_s'], cm['gamma_mps2'], cm['delay_s']]
    X = np.arange(0, 4.7, 0.05)
    VL, VF, VB, TL, TB, FL, FB = [], [], [], [], [], [], []
    T = np.arange(0, 1.6, 1 / 30)
    for r in rows_b:
        gr = r['grid']
        x = np.array(gr['x'])
        VL.append(np.interp(X, x, gr['v_launch'], right=np.nan))
        VF.append(np.interp(X, x, gr['v_fly'], right=np.nan))
        prof = {'x': x, 'f_model': np.nan_to_num(np.array(gr['f_model_launch'], float)),
                's': np.nan_to_num(np.array(gr['s_out_launch'], float)), 'v_fly': np.array(gr['v_fly'], float)}
        car = S.Car(p, S.c_ratio(fit, r['session']), cm['a_cap_mps2'])
        tt, vl, hist = S.sim_policy(car, {'kind': 'launch', 'U': 250.0, 'vt': 3.0}, prof, r['tgrid']['v'][0], float(x[-1]))
        h = np.array(hist)
        vb = np.interp(X, h[:, 2], h[:, 1], right=np.nan)
        vb[X > r['x_lh_m']] = np.nan            # 車のモデルはまっすぐだけ: LH の入口から先は描かない
        VB.append(vb)
        k_lh = np.nonzero(h[:, 2] >= r['x_lh_m'])[0]
        t_lh = float(h[k_lh[0], 0]) if len(k_lh) else np.inf     # 案の再生で LH の入口に着いた時刻
        tb = np.interp(T, h[:, 0], h[:, 1])
        tb[T > t_lh] = np.nan                   # LH の入口から先は描かない
        TB.append(tb)
        # 指令は段のまま(補間しない)。記録の線と同じく「出した時刻」(車に効くのは δ あと)にそろえる
        kz = np.clip(np.searchsorted(h[:, 0], T + cm['delay_s'] + 0.01, side='right') - 1, 0, len(h) - 1)
        fb = h[kz, 3].astype(float)
        fb[T > t_lh] = np.nan
        FB.append(fb)
        g = r['tgrid']
        TL.append(np.interp(T, g['t'], g['v']))
        FL.append(np.interp(T, g['t'], np.nan_to_num(np.array(g['f_out'], float))))
    nvalid = np.sum(np.isfinite(np.stack(VB)), axis=0)
    VL, VF, VB = (np.nanmedian(np.stack(a), axis=0) for a in (VL, VF, VB))
    VB[nvalid < 5] = np.nan                     # 半分より少ない発進しか残らないところは描かない
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
    ax.plot(X, VF, color=C2, lw=2, label='走りながら(周の同じ場所)')
    ax.plot(X, VL, color=C1, lw=2, label='止まってから(記録)')
    ax.plot(X, VB, color=C3, lw=2, ls='--', label='案 L250・3.0 m/s(再生。LH の入口まで)')
    xl = med([r['x_lh_m'] for r in rows_b])
    ax.axvline(xl, color=INK2, lw=1, ls=':')
    ax.text(xl + 0.05, 4.35, 'LH の入口(中央 %.1f m)' % xl, color=INK2, fontsize=13)
    ax.set_xlabel('発進からの車輪の距離 [m]')
    ax.set_ylabel('速さ [m/s]')
    ax.set_ylim(0, 4.6)
    ax.set_xlim(0, 4.7)
    ax.set_title('発進(10/24 の B・9 回の中央値)と走りながらの速さ', loc='left', color=INK, fontsize=16)
    ax.legend(loc='lower right', frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / 'launch_speed_vs_distance.png')
    plt.close(fig)

    # ---- 図 2: 発進からの時間(速さと、車に出した前進の指令)。2 つの図に分ける(軸は 1 つずつ)
    nv_t = np.sum(np.isfinite(np.stack(TB)), axis=0)
    TLm, TBm, FLm, FBm = (np.nanmedian(np.stack(a), axis=0) for a in (TL, TB, FL, FB))
    TBm[nv_t < 5] = np.nan
    FBm[nv_t < 5] = np.nan
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), dpi=150, sharex=True)
    axs[0].plot(T, TLm, color=C1, lw=2, label='記録(今の補助)')
    axs[0].plot(T, TBm, color=C3, lw=2, ls='--', label='案 L250・3.0 m/s(再生。LH の入口まで)')
    axs[0].set_ylabel('速さ [m/s]')
    axs[0].legend(loc='lower right', frameon=False, fontsize=13)
    axs[0].set_title('発進からの時間(10/24 の B・9 回の中央値)', loc='left', color=INK, fontsize=16)
    axs[1].step(T, FLm, where='post', color=C1, lw=2, label='記録: 車に出した前進')
    axs[1].step(T, FBm, where='post', color=C3, lw=2, ls='--', label='案: 車に出す前進')
    axs[1].axhline(80, color=INK2, lw=1, ls=':')
    axs[1].text(0.62, 60, '今の補助 80 µs', color=INK2, fontsize=12)
    axs[1].set_ylim(0, 270)
    axs[1].set_ylabel('前進の指令 [µs](中立から)')
    axs[1].set_xlabel('AI 側にしてからの時間 [秒]')
    axs[1].legend(loc='lower right', frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / 'launch_time_series.png')
    plt.close(fig)

    # ---- 図 3: 案ごとの縮む時間(10/24 の B・9 回の中央値。ひげ = 加速の上限 5〜9 m/s²)
    names = [('KICK150', '補助を 150 µs に(0.4 m/s まで)'), ('KICK250', '補助を 250 µs に(0.4 m/s まで)'),
             ('L80_3.0', '補助 80 µs のまま長く(3.0 m/s まで)'), ('L150_3.0', '前進 150 µs を 3.0 m/s まで'),
             ('L200_2.5', '前進 200 µs を 2.5 m/s まで'), ('L250_2.5', '前進 250 µs を 2.5 m/s まで'),
             ('L250_3.0', '前進 250 µs を 3.0 m/s まで'), ('L300_3.0', '前進 300 µs を 3.0 m/s まで'),
             ('IDEAL', '下限の目安(上限の加速で周の速さへ)')]
    sb = sim['summary']['B_1024']
    vals = [sb['a7'][k]['save_median'] for k, _ in names]
    lo = [sb['a5'][k]['save_median'] for k, _ in names]
    hi = [sb['a9'][k]['save_median'] for k, _ in names]
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=150)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=C1, height=0.6)
    ax.errorbar(vals, y, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt='none',
                ecolor=INK2, elinewidth=1.2, capsize=4)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi + 0.18, '%.2f' % v, color=INK, fontsize=12, va='bottom')
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in names], fontsize=13, color=INK)
    ax.set_xlabel('縮む時間 [秒](発進 → LH の入口。加速の上限 7.1 m/s²。ひげ = 5〜9 m/s²)')
    ax.set_title('案ごとの見積もり(10/24 の B・9 回の中央値。今の失う時間は 0.63 秒)', loc='left', color=INK, fontsize=15)
    ax.set_xlim(0, 0.5)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(FIG / 'launch_savings.png')
    plt.close(fig)
    print('figs ->', FIG)


if __name__ == '__main__':
    main()
