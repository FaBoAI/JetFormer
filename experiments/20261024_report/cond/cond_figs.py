#!/usr/bin/env python3
"""解析 B の図と数(2026-10-25)。cond_metrics.py・cond_extra.py・cond_replay.py〜cond_replay5.py の出力を読んで、
図 4 枚(2160×1350)と cond_summary.json を作る。読むだけ。

    jf-heavy --lock none --mem 1500M -- python3 -I cond_figs.py

追加で計算するもの:
  動きの式の電池の効き: 直線(DE・SW・HS)で d eRPM/dt = α·cmd(0.1 秒前)− β·eRPM + γ を当てはめ(β は 20 本で共通の
  中央値)、指令 200 µs で行きつく eRPM = (200α + γ)/β。90% 区間 = 5 秒ずつのかたまりの復元抽出(400 回)
  左右の旋回の効き: 速さ 1.5〜2.5 m/s・|舵(直前 0.2 秒の平均)| ≥ 150 µs の κ ÷ 舵 の中央値(×1e4)を左右で
"""
import json
from pathlib import Path

import numpy as np

D = Path('/home/jetson/jetformer_reports/bench_20261024_report/cond_data')
FIG = Path('/home/jetson/jetformer_reports/bench_20261024_report/fig')
COL = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}
GRAY, INK, INK2, SURF, RED = '#8a8984', '#0b0b0b', '#52514e', '#fcfcfb', '#e34948'
SESS = {1: '1 回目(13:46〜)', 2: '2 回目(16:01〜、満充電)', 3: '3 回目(18:31〜、後輪の修理のあと)'}
FOCUS = (9, 11, 12, 18)
OFFS = {9: (0, 13, 'center'), 11: (8, -22, 'left'), 12: (8, 12, 'left'), 18: (0, 13, 'center')}
PRE_B, POST_B = (1, 4, 9, 10), (11, 12, 14, 17, 18, 20)
LAG = 3


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 13, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
                         'axes.titlelocation': 'left', 'axes.titlepad': 10, 'legend.frameon': False})
    return plt


def frame(plt, title, sub):
    fig, ax = plt.subplots(2, 2, figsize=(14.4, 9.0), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.855, bottom=0.115, hspace=0.45, wspace=0.2)
    fig.text(0.07, 0.955, title, ha='left', fontsize=16, color=INK, fontweight='bold')
    fig.text(0.07, 0.915, sub, ha='left', fontsize=11.5, color=INK2)
    return fig, ax


def session_legend(fig, extra=()):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker='o', ls='', color=COL[s], ms=9, label=SESS[s]) for s in (1, 2, 3)]
    hs += [Line2D([], [], color=RED, lw=1.2, ls=(0, (4, 3)), label='後輪の修理(No.10 と No.11 の間)')]
    hs += list(extra)
    fig.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.52, 0.005), ncol=len(hs), fontsize=11.5)


def dyn_fit(z, idx=None, beta=None):
    t, e, u, drv, loc = z['t'], z['erpm'], z['cmd'], z['drv'], z['loc']
    n = len(t)
    es = np.convolve(e, np.ones(5) / 5, 'same')
    us = np.convolve(np.nan_to_num(u), np.ones(3) / 3, 'full')[:n]
    i = np.arange(LAG + 3, n - 3)
    dt = t[i + 1] - t[i - 1]
    ok = (drv[i] & drv[i - LAG - 2] & drv[i + 1] & np.isin(loc[i], (4, 5, 6)) & (dt > 0.05) & (dt < 0.08)
          & (es[i] > 1500) & (us[i - LAG] > 40))
    i, dt = i[ok], dt[ok]
    if idx is not None:
        i, dt = idx(i, dt)
    y = (es[i + 1] - es[i - 1]) / dt
    U, E = us[i - LAG], es[i]
    if beta is None:
        a, b, c = np.linalg.lstsq(np.c_[U, -E, np.ones_like(U)], y, rcond=None)[0]
        return a, b, c, len(y), i
    a, c = np.linalg.lstsq(np.c_[U, np.ones_like(U)], y + beta * E, rcond=None)[0]
    return a, beta, c, len(y), i


def boot_ess(z, beta, B=400, block_s=5.0, seed=0):
    a, b, c, n, i_all = dyn_fit(z, beta=beta)
    est = (200 * a + c) / b
    blk = np.floor(z['t'][i_all] / block_s).astype(int)
    ub = np.unique(blk)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.choice(ub, len(ub), replace=True)
        sel = np.concatenate([np.flatnonzero(blk == k) for k in pick])
        a2, b2, c2, _, _ = dyn_fit(z, idx=lambda i, dt, sel=sel: (i[sel], dt[sel]), beta=beta)
        vals.append((200 * a2 + c2) / b2)
    return est, n, (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))


def trailing_mean(x, w):
    c = np.cumsum(np.r_[0.0, x])
    n = len(x)
    i0 = np.maximum(np.arange(n) - w + 1, 0)
    return (c[np.arange(n) + 1] - c[i0]) / (np.arange(n) + 1 - i0)


def main():
    plt = setup()
    from matplotlib.lines import Line2D
    M = json.loads((D / 'cond_metrics.json').read_text(encoding='utf-8'))
    X = json.loads((D / 'cond_extra.json').read_text(encoding='utf-8'))
    RP = json.loads((D / 'cond_replay.json').read_text(encoding='utf-8'))
    RP2 = json.loads((D / 'cond_replay2.json').read_text(encoding='utf-8'))
    RP3 = json.loads((D / 'cond_replay3.json').read_text(encoding='utf-8'))
    RP4 = json.loads((D / 'cond_replay4.json').read_text(encoding='utf-8'))
    RP5 = json.loads((D / 'cond_replay5.json').read_text(encoding='utf-8'))
    runs = {r['no']: r for r in M['runs']}
    xrun = {r['log']: r for r in X['runs']}
    nos = sorted(runs)
    S = {}

    def sess_of(no):
        return runs[no]['session']

    # ---------- 動きの式の電池の効き
    Z = {}
    betas = []
    for no in nos:
        r = runs[no]
        if not r['ai_s'] or r['ai_s'] < 15:
            continue
        Z[no] = np.load(D / ('frames_%s.npz' % r['log']))
        if r['ai_s'] > 25:
            betas.append(dyn_fit(Z[no])[1])
    beta = float(np.median(betas))
    dyn = {}
    for no, z in Z.items():
        est, n, ci = boot_ess(z, beta, seed=no)
        dyn[no] = {'ess200': est, 'ci': ci, 'n': n}
    S['dyn_beta'] = round(beta, 3)
    S['dyn_ess200'] = {no: {'ess200': round(v['ess200']), 'ci90': [round(v['ci'][0]), round(v['ci'][1])], 'n': v['n']}
                       for no, v in dyn.items()}

    def per_run(a, key_fn, ylabel, title, ci_fn=None, fmt='%.0f', ref=None, pad=0.18):
        vals = []
        for no in nos:
            v = key_fn(no)
            if v is None:
                continue
            vals.append(v)
            s = sess_of(no)
            big = no in FOCUS
            if ci_fn is not None and ci_fn(no) is not None:
                lo, hi = ci_fn(no)
                a.plot([no, no], [lo, hi], color=COL[s], lw=2, alpha=0.5, solid_capstyle='round')
                vals += [lo, hi]
            a.plot([no], [v], 'o', ms=12 if big else 7, color=COL[s], mec=SURF, mew=2, zorder=3)
            if big:
                dx, dy, ha = OFFS[no]
                a.annotate(('No.%d  ' + fmt) % (no, v), (no, v), xytext=(dx, dy), textcoords='offset points', ha=ha,
                           fontsize=10.5, color=INK, zorder=4)
        a.axvline(4.5, color=GRAY, lw=0.8, ls=(0, (4, 3)))
        a.axvline(10.5, color=RED, lw=1.2, ls=(0, (4, 3)))
        a.set_xticks(nos)
        a.set_xlim(0.3, 20.7)
        a.set_xlabel('AI 走行の番号(No.)')
        a.set_ylabel(ylabel)
        a.set_title(title)
        if ref is not None:
            a.axhline(ref, color=GRAY, lw=0.8)
            vals.append(ref)
        if vals:
            lo, hi = min(vals), max(vals)
            a.set_ylim(lo - 0.08 * (hi - lo), hi + pad * (hi - lo))

    # ---------- 図 1: 電池
    fig, ax = frame(plt, '電池の印: 同じ指令で回る速さは 3 回目も No.9 とほぼ同じ(−8〜+2%)。3 回目の直線が遅いのは、指令そのものが小さいから',
                    '10/24 の AI 走行 No.1〜20。大きな点 = B・速さ 0.90・ステア 1.0(No.9・11・12・18)。電圧の記録が無いので、回転数(eRPM)と前進の指令(µs)から作った代わりの印')
    per_run(ax[0, 0], lambda no: runs[no]['eff_raw'], 'eRPM ÷ µs', '(a) 電池の効き = 直線の eRPM ÷ 前進の指令', fmt='%.1f')
    per_run(ax[0, 1], lambda no: dyn[no]['ess200'] if no in dyn and runs[no]['ai_s'] > 25 else None, 'eRPM',
            '(a\') 指令 200 µs で行きつく回転数(加速の式・90% 区間)', ci_fn=lambda no: dyn[no]['ci'] if no in dyn else None)
    per_run(ax[1, 0], lambda no: runs[no]['erpm_p90_ai'] if runs[no]['ai_s'] and runs[no]['ai_s'] > 15 else None,
            'eRPM(× 0.0005 = m/s)', '(b) 直線の最高速 = AI 中の eRPM の 90% 点')
    a = ax[1, 1]
    for no in FOCUS:
        b = runs[no]['bins']
        xs = [x['t_ai_s'] for x in b if x['erpm_p90'] is not None]
        ys = [x['erpm_p90'] for x in b if x['erpm_p90'] is not None]
        a.plot(xs, ys, '-', color=COL[sess_of(no)], lw=2.2 if no in (9, 11) else 1.6, marker='o', ms=8, mec=SURF, mew=1.5,
               alpha=1.0 if no in (9, 11) else 0.7)
        a.annotate('No.%d' % no, (xs[-1], ys[-1]), xytext=(7, 0), textcoords='offset points', va='center', fontsize=11,
                   color=INK)
    a.set_xlabel('AI で走った時間 [秒](20 秒ごと)')
    a.set_ylabel('eRPM の 90% 点')
    a.set_title('(c) 1 本の中の変化(B・速さ 0.90)')
    a.set_xlim(0, 128)
    session_legend(fig)
    fig.savefig(FIG / 'cond_battery.png', dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---------- 図 2: 明るさ
    tr = X['train']
    tb = [t['bright_mean'] for t in tr]
    ts = [t['bright_std'] for t in tr]
    fig, ax = frame(plt, '明るさの印: 画像の明るさ・コントラストは 20 本でほぼ同じ。机上で画像を ±10% 変えても、スロットルの変化は平均 1 µs 以内',
                    '輝度 Y(0〜255)。灰色の帯 = B の学習データ(10/10 の昼 13:54〜15:27・夜 19:36〜20:26 の人の走行 9 本)の最小〜最大。AI のフレーム 2 枚に 1 枚')
    per_run(ax[0, 0], lambda no: xrun[runs[no]['log']]['bright_mean'], '輝度 Y の平均', '(a) 画像の平均の明るさ(走行ごと)', fmt='%.1f')
    ax[0, 0].axhspan(min(tb), max(tb), color=GRAY, alpha=0.15, lw=0)
    ax[0, 0].set_ylim(98, 112)
    per_run(ax[0, 1], lambda no: xrun[runs[no]['log']]['bright_std'], '輝度 Y の標準偏差', '(b) コントラスト(走行ごと)', fmt='%.1f')
    ax[0, 1].axhspan(min(ts), max(ts), color=GRAY, alpha=0.15, lw=0)
    ax[0, 1].set_ylim(28, 42)
    a = ax[1, 0]
    secs = ['R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
    names = {'R1': 'R1', 'GM_in': 'GM 入口', 'GM_out': 'GM 出口', 'DE': 'DE', 'SW': 'SW', 'HS': 'HS', 'LH': 'LH'}
    for j, s in enumerate(secs):
        v = [t['sections'].get(s, {}).get('bright_mean') for t in tr]
        v = [x for x in v if x is not None]
        a.fill_between([j - 0.38, j + 0.38], min(v), max(v), color=GRAY, alpha=0.15, lw=0)
    for k, no in enumerate(FOCUS):
        xr = xrun[runs[no]['log']]
        ys = [xr['sections'].get(s, {}).get('bright_mean') for s in secs]
        xs = np.arange(len(secs)) + (k - 1.5) * 0.16
        a.plot(xs, ys, 'o', ms=10 if no == 9 else 7, color=COL[sess_of(no)], mec=SURF, mew=1.5,
               label='No.%d' % no, alpha=1.0 if no in (9, 11) else 0.75)
    a.set_xticks(range(len(secs)))
    a.set_xticklabels([names[s] for s in secs])
    a.set_ylabel('輝度 Y の平均')
    a.set_ylim(72, 128)
    a.set_title('(c) 区間ごとの明るさ(GM 入口・LH の差は車の向きで写る物が変わるため)')
    a.legend(ncol=4, fontsize=10.5, loc='lower left')
    a = ax[1, 1]
    labs = ['画像 ×1.10', '画像 ×0.90', 'eRPM ×1.15', 'eRPM ×0.87']
    keys = ['d_thr_img_up', 'd_thr_img_dn', 'd_thr_erpm_up', 'd_thr_erpm_dn']
    for k, no in enumerate(FOCUS):
        v = [RP['logs'][str(no)]['all'][kk] for kk in keys]
        xs = np.arange(len(keys)) + (k - 1.5) * 0.19
        a.bar(xs, v, width=0.17, color=COL[sess_of(no)], alpha=1.0 if no in (9, 11) else 0.7, label='No.%d' % no)
    hs_gap = [RP['logs']['9']['sections']['HS']['thr_orig_p50'] - RP['logs'][str(no)]['sections']['HS']['thr_orig_p50']
              for no in (11, 12, 18)]
    S['hs_raw_thr_gap_us'] = [round(x, 1) for x in hs_gap]
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(range(len(keys)))
    a.set_xticklabels(labs)
    a.set_ylabel('モデルのスロットルの変化 [µs]')
    a.set_ylim(-3, 3)
    a.set_title('(d) 机上の試し: 入力を 1 つ変えたときのモデルのスロットル')
    a.text(0.02, 0.05, '参考: HS での No.9 と 3 回目のモデルのスロットルの差は %d〜%d µs' % (min(hs_gap), max(hs_gap)),
           transform=a.transAxes, fontsize=11, color=INK)
    a.legend(ncol=4, fontsize=10.5, loc='upper right')
    session_legend(fig, [Line2D([], [], color=GRAY, lw=8, alpha=0.3, label='学習データの幅')])
    fig.savefig(FIG / 'cond_brightness.png', dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---------- 図 3: 修理
    lr = {}
    for no, z in Z.items():
        sa = trailing_mean(z['steer'], 6)
        v = z['speed']
        kap = np.radians(z['gz']) / np.maximum(v, 0.1)
        m = z['drv'] & (v >= 1.5) & (v < 2.5) & (np.abs(sa) >= 150)
        L, R = m & (sa > 0), m & (sa < 0)
        if L.sum() > 20 and R.sum() > 20:
            gl, gr = np.median(kap[L] / sa[L]) * 1e4, np.median(kap[R] / sa[R]) * 1e4
            lr[no] = {'left': round(float(gl), 1), 'right': round(float(gr), 1), 'ratio': round(float(gl / gr), 3),
                      'n_left': int(L.sum()), 'n_right': int(R.sum())}
    S['turn_gain_1p5_2p5'] = lr
    pre = [lr[n]['ratio'] for n in lr if n <= 10]
    post = [lr[n]['ratio'] for n in lr if n >= 11]
    S['turn_ratio_pre_range'] = [min(pre), max(pre), len(pre)]
    S['turn_ratio_post_range'] = [min(post), max(post), len(post)]
    fig, ax = frame(plt, '修理の前後: 3 回目は「左に曲がりにくく・右に曲がりやすい」車になり、左の U ターン(GM)で舵が増え、DE で当たりやすくなった',
                    '赤い点線 = 後輪の修理(No.10 で壊れた → No.11 の前に直した)。曲がり具合 = ヨー ÷ 速さ。舵 = 車に出した指令 − 中央 1441 µs(+ = 左)')
    per_run(ax[0, 0], lambda no: lr[no]['ratio'] if no in lr else None, '左 ÷ 右',
            '(a) 左右の旋回の効きの比(曲がり具合 ÷ 舵、1.5〜2.5 m/s)', fmt='%.2f', ref=1.0)
    ax[0, 0].set_ylim(0.6, 1.05)
    ax[0, 0].text(0.02, 0.05, '修理の前 %d 本 %.2f〜%.2f  /  後 %d 本 %.2f〜%.2f(重ならない)' % (
        len(pre), min(pre), max(pre), len(post), min(post), max(post)), transform=ax[0, 0].transAxes, fontsize=11, color=INK)
    per_run(ax[0, 1], lambda no: runs[no]['sections'].get('GM_in', {}).get('steer_p50') if runs[no]['ai_s'] and runs[no]['ai_s'] > 15 else None,
            '舵 [µs](+ = 左)', '(b) GM 入口(左の U ターン)の舵の中央値', fmt='%.0f')
    a = ax[1, 0]
    for no in nos:
        pl = runs[no]['de_pass_list']
        if not pl or runs[no]['model'] != 'B':
            continue
        s = sess_of(no)
        for p in pl:
            if p['late_left_us'] is None:
                continue
            jx = no + (np.random.default_rng(int(p['t_s'] * 10)).random() - 0.5) * 0.45
            if p['hit']:
                a.plot([jx], [p['late_left_us']], 'x', ms=10, color=COL[s], mew=2.4)
            else:
                a.plot([jx], [p['late_left_us']], 'o', ms=7, color=COL[s], mec=SURF, mew=1.2, alpha=0.85)
    a.axvline(10.5, color=RED, lw=1.2, ls=(0, (4, 3)))
    a.set_xticks([n_ for n_ in nos if runs[n_]['model'] == 'B'])
    a.set_xlim(0.3, 20.7)
    a.set_ylim(-10, 470)
    a.set_xlabel('AI 走行の番号(No.、モデル B だけ)')
    a.set_ylabel('切り返し [µs]')
    a.set_title('(c) DE の舵の切り返し(0.6〜1.0 秒の左への最大。1 通過 = 1 点、× = 当たった)')
    pre_p = [p for no in PRE_B for p in runs[no]['de_pass_list']]
    post_p = [p for no in POST_B for p in runs[no]['de_pass_list']]
    S['de_hits_pre'] = [sum(p['hit'] for p in pre_p), len(pre_p)]
    S['de_hits_post'] = [sum(p['hit'] for p in post_p), len(post_p)]
    a.text(0.02, 0.92, '当たった通過(ステア 1.0): 前 No.1・4・9・10 = %d/%d  →  後 No.11・12・14・17・18・20 = %d/%d' % (
        S['de_hits_pre'][0], S['de_hits_pre'][1], S['de_hits_post'][0], S['de_hits_post'][1]), transform=a.transAxes,
        fontsize=11, color=INK)
    per_run(ax[1, 1], lambda no: runs[no]['yaw_bias_dps'] if runs[no]['yaw_bias_n'] > 50 else None, 'deg/s(+ = 左)',
            '(d) 直線のヨーの偏り(舵が中央 ±25 µs のときの gz の平均)', fmt='%+.1f', ref=0)
    ax[1, 1].set_ylim(-42, 30)
    ax[1, 1].text(0.02, 0.05, 'No.9 だけが +。修理の前(No.1・2・4・7・8)も − なので、修理だけでは説明できない',
                  transform=ax[1, 1].transAxes, fontsize=11, color=INK)
    session_legend(fig)
    fig.savefig(FIG / 'cond_repair.png', dpi=150, facecolor=SURF)
    plt.close(fig)

    # ---------- 図 4: 入れ替えの試し
    pairs = ['11<-9', '12<-9', '18<-9']

    def closed(J, key, sec='all'):
        v = []
        for p in pairs:
            row = J['pairs'][p]['all'] if (sec == 'all' and 'all' in J['pairs'][p]) else J['pairs'][p]['sections'].get(sec, {})
            if row.get(key) is not None:
                v.append(row[key])
        return 100 * float(np.mean(v)) if v else np.nan

    fig, ax = plt.subplots(1, 2, figsize=(14.4, 9.0), dpi=150, gridspec_kw={'width_ratios': [1.0, 1.15]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.13, wspace=0.42)
    fig.text(0.07, 0.955, '入れ替えの試し: 3 回目のスロットルの小ささは、IMU(前後の加速度)と画像で半分ずつ。明るさ・回転数・ToF はほぼ 0',
             ha='left', fontsize=15, color=INK, fontweight='bold')
    fig.text(0.07, 0.912, '3 回目(No.11・12・18)のフレームの入力を 1 つだけ No.9 の「同じ区間・同じ進み具合」のフレームのものに替えて、モデル B に入れた(机上・開ループ。'
             '記録との差 0.13 µs で再現)。', ha='left', fontsize=11.5, color=INK2)
    fig.text(0.07, 0.88, '0% = 変わらない、100% = No.9 のスロットルと同じになった。3 本の平均。', ha='left', fontsize=11.5, color=INK2)
    a = ax[0]
    secs2 = ['GM_in', 'LH', 'SW', 'HS']
    sw = ['img', 'imu', 'tof', 'erpm']
    swn = {'img': 'カメラ画像', 'imu': 'IMU', 'tof': 'ToF', 'erpm': '回転数'}
    swc = {'img': '#2a78d6', 'imu': '#eb6834', 'tof': '#1baf7a', 'erpm': '#eda100'}
    width = 0.2
    for k, s in enumerate(sw):
        vals = [closed(RP2, 'closed_' + s, sec) for sec in secs2]
        xs = np.arange(len(secs2)) + (k - 1.5) * width
        a.bar(xs, vals, width=width * 0.88, color=swc[s], label=swn[s])
        for x_, v_ in zip(xs, vals):
            if np.isfinite(v_) and abs(v_) >= 5:
                a.annotate('%d' % round(v_), (x_, v_), xytext=(0, 3 if v_ >= 0 else -14), textcoords='offset points',
                           ha='center', fontsize=10, color=INK2)
    a.axhline(0, color=INK2, lw=0.8)
    a.axhline(100, color=GRAY, lw=0.8, ls=(0, (4, 3)))
    a.set_xticks(range(len(secs2)))
    a.set_xticklabels([names[s] for s in secs2])
    a.set_ylim(-15, 128)
    a.set_ylabel('埋まった差の割合 [%]')
    a.set_title('(a) 区間ごと: 左の曲がりは IMU、SW・HS は画像')
    a.legend(ncol=2, fontsize=11, loc='upper right')
    a = ax[1]
    rows = [('画像 ぜんぶ', closed(RP2, 'closed_img'), '#2a78d6'),
            ('画像の下 56 画素だけ(ボンネット)', closed(RP4, 'closed_hood'), '#2a78d6'),
            ('画像の上 168 画素だけ', closed(RP4, 'closed_scene'), '#2a78d6'),
            ('IMU ぜんぶ', closed(RP2, 'closed_imu'), '#eb6834'),
            ('IMU 前後の加速度だけ', closed(RP3, 'closed_acc_lon'), '#eb6834'),
            ('IMU 横の加速度だけ', closed(RP3, 'closed_acc_lat'), '#eb6834'),
            ('IMU ヨーだけ', closed(RP3, 'closed_gyro_z'), '#eb6834'),
            ('IMU 上下・ロール・ピッチ', closed(RP3, 'closed_acc_vert') + closed(RP3, 'closed_gyro_xy'), '#eb6834'),
            ('ToF', closed(RP2, 'closed_tof'), '#1baf7a'),
            ('回転数', closed(RP2, 'closed_erpm'), '#eda100')]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, c) in zip(ys, rows):
        a.barh([y], [v], height=0.62, color=c, alpha=1.0 if 'ぜんぶ' in lab or lab in ('ToF', '回転数') else 0.6)
        a.annotate('%d%%' % round(v), (max(v, 0), y), xytext=(5, 0), textcoords='offset points', va='center', fontsize=11,
                   color=INK)
    a.set_yticks(ys)
    a.set_yticklabels([r[0] for r in rows])
    a.axvline(0, color=INK2, lw=0.8)
    a.set_xlim(-15, 80)
    a.set_xlabel('埋まった差の割合 [%](全区間)')
    a.set_title('(b) 何を替えると差が埋まるか(全区間)')
    # 速さをそろえた相手
    sp = []
    for p in pairs:
        al = RP5['pairs'][p]['sections']['all']
        al2 = RP2['pairs'][p]['all']
        sp.append({'pair': p, 'gap_any': round(al2['thr_donor'] - al2['thr_self'], 1),
                   'gap_same_speed': round(al['thr_donor'] - al['thr_none'], 1),
                   'img_same_speed': round(al['thr_img'] - al['thr_none'], 1),
                   'imu_same_speed': round(al['thr_imu'] - al['thr_none'], 1),
                   'img_any': round(al2['thr_img'] - al2['thr_self'], 1), 'imu_any': round(al2['thr_imu'] - al2['thr_self'], 1)})
    S['same_speed_swap'] = sp
    txt = ('速さ(eRPM)もそろえた No.9 のフレームを相手にすると: 入力すべてで +%.0f〜%.0f µs(そろえないと +%.0f〜%.0f)、'
           '画像だけ +%.0f〜%.0f(そろえないと +%.0f〜%.0f)、IMU だけ +%.0f〜%.0f(+%.0f〜%.0f)') % (
        min(x['gap_same_speed'] for x in sp), max(x['gap_same_speed'] for x in sp), min(x['gap_any'] for x in sp),
        max(x['gap_any'] for x in sp), min(x['img_same_speed'] for x in sp), max(x['img_same_speed'] for x in sp),
        min(x['img_any'] for x in sp), max(x['img_any'] for x in sp), min(x['imu_same_speed'] for x in sp),
        max(x['imu_same_speed'] for x in sp), min(x['imu_any'] for x in sp), max(x['imu_any'] for x in sp))
    fig.text(0.07, 0.035, txt, ha='left', fontsize=11.5, color=INK)
    fig.text(0.07, 0.008, '→ 画像の効きの大半は「もう速いかどうか」の違い(速さをそろえると小さくなる)。IMU の効きは同じ速さでも残る(左の曲がりで減速しているか)',
             ha='left', fontsize=11.5, color=INK)
    fig.savefig(FIG / 'cond_swap.png', dpi=150, facecolor=SURF)
    plt.close(fig)
    (D / 'cond_summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in S.items() if k not in ('dyn_ess200', 'turn_gain_1p5_2p5')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
