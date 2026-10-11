"""Figures for analysis.md"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

from load import L, seg_mask, KV, NEUTRAL  # noqa: E402
from takeovers import LOC, board_resets  # noqa: E402

SRC = '/home/jetson/jetformer_reports/overnight_20261011/runs/src/'
FIG = '/home/jetson/jetformer_reports/overnight_20261011/runs/fig/'
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(fp):
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                                                           '#008300', '#4a3aa7', '#e34948')
WARN, SERIOUS, CRIT = '#fab219', '#ec835a', '#d03b3b'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlecolor': INK, 'axes.titlesize': 11, 'font.size': 9, 'legend.frameon': False})
LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
LOCJ = {'LH': 'LH 黒カーテン', 'GM_in': 'GM 入口', 'GM_out': 'GM 出口', 'DE': 'DE', 'DE2': 'DE後半〜SW',
        'HS_P': 'HS 駐車枠', 'R1': 'R1'}


def timeline(log):
    d = L(log)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8.0), sharex=True, gridspec_kw={'height_ratios': [1.3, 1, 1]})
    segs = seg_mask(d.sel)
    for a in ax:
        for i0, i1 in segs:
            a.axvspan(d.at[i0], d.at[i1], color=BLUE, alpha=0.08, lw=0)
    v = -d.erpm * KV
    ax[0].plot(d.vt, v, color=INK2, lw=0.8, label='速度 m/s(eRPM×0.5e-3、+ = 前進)')
    ai_v = np.where(np.interp(d.vt, d.at, d.sel.astype(float)) > 0.5, v, np.nan)
    ax[0].plot(d.vt, ai_v, color=BLUE, lw=1.4, label='AI側')
    SHORT = {'LH': 'LH', 'GM_in': 'GMi', 'GM_out': 'GMo', 'DE': 'DE', 'DE2': 'DE2', 'HS_P': 'HSP', 'R1': 'R1'}
    kk = 0
    for e in d.events(2.5):
        if not e['ai']:
            continue
        lab = LOC.get((log, round(e['t'], 2)), '')
        ax[0].plot(e['t'], -e['erpm_before'] * KV, 'v', color=CRIT, ms=8, mec=SURF, mew=1)
        ax[0].annotate('%s %.0fG' % (SHORT.get(lab, lab), e['g']), (e['t'], -e['erpm_before'] * KV),
                       xytext=(0, 8 + 13 * (kk % 2)), textcoords='offset points', ha='center', fontsize=7.5, color=INK2)
        kk += 1
    for i0, i1 in segs:
        if i1 + 1 < len(d.at):
            ax[0].axvline(d.at[i1], color=ORANGE, lw=1.0, ls='--')
    for r in board_resets(d):
        ax[0].axvline(r, color=VIOLET, lw=1.6, ls=':')
        ax[0].text(r, 5.3, '基板再起動', color=VIOLET, fontsize=7.5, ha='center')
    ax[0].set_ylim(-2.5, 5.8)
    ax[0].set_ylabel('速度 m/s')
    ax[0].plot([], [], 'v', color=CRIT, label='衝撃 ≥2.5G(AI側、場所)')
    ax[0].plot([], [], ls='--', color=ORANGE, label='人が交代')
    ax[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8)
    a = d.sel & ~d.hold
    st = np.where(a, d.out[:, 0] - 1443, np.nan)
    ax[1].plot(d.at, st, color=BLUE, lw=1.0, label='AIのステア出力(中心から µs、+ = 左)')
    ax[1].set_ylabel('ステア µs')
    la = d.la.astype(float)
    prev = None
    for i0, i1 in segs:
        if la[i0] != prev:
            ax[1].text(d.at[i0], 520, '先読み %d' % la[i0], fontsize=7.5, color=INK2)
            prev = la[i0]
    ax[1].set_ylim(-520, 600)
    ax[1].legend(loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=8)
    ttc = np.clip(d.ttc, 0, 1.5)
    ax[2].plot(d.at, np.where(a, ttc, np.nan), color=INK2, lw=0.9, label='TTC(AI側、1.5秒で頭打ち)')
    g1 = a & (d.guard == 1)
    g2 = a & (d.guard == 2)
    ax[2].fill_between(d.at, 0, 1, where=g1, transform=ax[2].get_xaxis_transform(), color=WARN, alpha=0.45, lw=0,
                       label='ガード: 減速(記録だけ)')
    ax[2].fill_between(d.at, 0, 1, where=g2, transform=ax[2].get_xaxis_transform(), color=CRIT, alpha=0.35, lw=0,
                       label='ガード: ブレーキ(記録だけ)')
    ax[2].axhline(0.5, color=MUTED, lw=0.8, ls='--')
    ax[2].axhline(0.3, color=MUTED, lw=0.8, ls=':')
    for e in d.events(2.5):
        if e['ai']:
            ax[2].axvline(e['t'], color=CRIT, lw=1.0)
    ax[2].set_ylim(0, 1.55)
    ax[2].set_ylabel('TTC 秒')
    ax[2].set_xlabel('ログ開始からの秒(青い帯 = AI側)')
    ax[2].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    p = d.params
    fig.suptitle('%s (%s)  開始時: ステア %.2f / 速度 %.2f / 先読み %d、リカバリー ON、ToFガード=記録だけ\n'
                 '衝撃ラベル: DE=バックストレート右の仕切り / DE2=その先〜SW / LH=黒カーテン / GMi・GMo=緑マット入口・出口 / HSP=ホームストレートの駐車枠' % (
        d.name, log, p.get('steer_gain'), p.get('throttle_gain'), p.get('lookahead')), x=0.01, ha='left', fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG + 'timeline_%s.png' % d.name, dpi=90)
    plt.close(fig)


def takeover_fig():
    """impact/stop -> takeover vs recovery detector requirement"""
    r = json.load(open(SRC + 'takeovers.json'))
    rows = []
    for log, v in r.items():
        name = v['name']
        for s in v['segments']:
            if s['end'] != 'takeover':
                continue
            cnt = s['detector']['slow_since_to_end_s'] or 0.0
            if s['impacts_in_seg'] and s['last_impact_to_end_s'] is not None and s['last_impact_to_end_s'] < 1.2:
                imp = next(x for x in v['impacts'] if abs((s['t1'] - x['t']) - s['last_impact_to_end_s']) < 0.02)
                if imp['glance']:
                    continue
                rows.append(('%s %.1fs %s' % (name, imp['t'], LOCJ.get(imp['loc'], imp['loc'])), 'impact',
                             s['last_impact_to_end_s'], 0.5, cnt))
            elif s['stopped_run_at_end_s'] >= 0.25:
                rows.append(('%s %.1fs 停止(衝撃なし)' % (name, s['t1'] - s['stopped_run_at_end_s']), 'plain',
                             s['stopped_run_at_end_s'], 0.8, cnt))
    rows.sort(key=lambda x: (x[1], x[2]))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(rows))
    for k, (lab, kind, dt, need, cnt) in enumerate(rows):
        pre = max(dt - cnt, 0.0)
        ax.barh(k, pre, color=ORANGE, height=0.55, alpha=0.35)
        ax.barh(k, cnt, left=pre, color=ORANGE if kind == 'impact' else YELLOW, height=0.55)
        fire = pre + need
        ax.plot([fire], [k], marker='|', ms=16, mew=2.5, color=INK)
        ax.text(max(dt, fire) + 0.04, k, '交代 %.2f s(止まって数えた %.2f / 必要 %.1f)' % (dt, cnt, need), va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([x[0] for x in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.88, color=MUTED, lw=1, ls='--')
    ax.text(0.89, -0.75, 'ai_run_007 で人が交代した中央値 0.88 s(停止から)', fontsize=7.5, color=INK2)
    ax.barh([0], [0], color=ORANGE, alpha=0.35, label='衝撃のあと、まだ動いている(跳ね返り・後ろへ転がる)= 検出は数えない')
    ax.barh([0], [0], color=ORANGE, label='止まっている時間(衝撃のあと、検出が数える)')
    ax.barh([0], [0], color=YELLOW, label='止まっている時間(衝撃なしの停止)')
    ax.plot([], [], marker='|', ms=12, mew=2.5, color=INK, ls='none',
            label='止まったままなら検出された時刻(衝撃後 0.5 s / 衝撃なし 0.8 s)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.12), ncol=2, fontsize=8)
    ax.set_xlim(0, 2.0)
    ax.set_xlabel('衝撃(または停止)からの秒 → 棒の右端 = 人が交代した時刻', labelpad=2)
    ax.set_title('リカバリー v2 が一度も動かなかった理由: 12 件すべてで、人の交代が検出より早かった', loc='left')
    fig.tight_layout()
    fig.savefig(FIG + 'recovery_takeover_timing.png', dpi=90)
    plt.close(fig)


def tof_fig():
    sw = json.load(open(SRC + 'tof_sweep.json'))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={'width_ratios': [1.3, 1, 1]})
    # (a) lead time per collision
    names = {'20261010-203924': 'ai_run_009', '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}
    cols = {'ai_run_009': BLUE, 'ai_run_010': ORANGE, 'ai_run_101': AQUA}
    from tof import evaluate, collisions
    k = 0
    for log in LOGS:
        d = L(log)
        ev = evaluate(d, d.guard >= 1)
        for rr in ev['collisions']:
            lead = rr['lead_s'] if rr['lead_s'] is not None else 0
            ax[0].bar(k, min(lead, 1.2), color=cols[names[log]], width=0.7)
            if lead > 1.2:
                ax[0].text(k, 1.21, '%.1f' % lead, ha='center', fontsize=7, color=INK2)
            if rr['lead_s'] is None:
                ax[0].text(k, 0.03, '×', ha='center', fontsize=10, color=CRIT)
            k += 1
    ax[0].axhline(0.3, color=INK, lw=1, ls='--')
    ax[0].text(-0.6, 0.31, '0.3 s', ha='left', va='bottom', fontsize=8, color=INK)
    for n, c in cols.items():
        ax[0].bar([0], [0], color=c, label=n)
    ax[0].legend(fontsize=8, loc='upper left')
    ax[0].set_ylim(0, 1.3)
    ax[0].set_ylabel('警告の先行時間 s(TTC<0.5 の立ち上がり → 衝撃)')
    ax[0].set_xlabel('AI側の衝突 21件(時刻順)')
    ax[0].set_title('衝突の前に警告できたか: 20/21 件が 0.3 s 以上前', loc='left')
    # (b) false alarms / min per run: now vs recalibrated floor
    runs = list(sw['per_run'].keys())
    x = np.arange(len(runs))
    fa_now = [sw['per_run'][r]['logged: TTC<0.5 (now)']['fa'] / sw['per_run'][r]['logged: TTC<0.5 (now)']['minutes'] for r in runs]
    fa_rec = [sw['per_run'][r]['recal floor TTC<0.5']['fa'] / sw['per_run'][r]['recal floor TTC<0.5']['minutes'] for r in runs]
    ax[1].bar(x - 0.18, fa_now, 0.34, color=BLUE, label='今の床の基準')
    ax[1].bar(x + 0.18, fa_rec, 0.34, color=AQUA, label='床の基準を作り直した場合(推定)')
    for i in range(len(runs)):
        ax[1].text(x[i] - 0.18, fa_now[i] + 0.4, '%.0f' % fa_now[i], ha='center', fontsize=8, color=INK2)
        ax[1].text(x[i] + 0.18, fa_rec[i] + 0.4, '%.0f' % fa_rec[i], ha='center', fontsize=8, color=INK2)
    ax[1].axhline(13.2, color=MUTED, ls='--', lw=1)
    ax[1].text(0.5, 13.5, '事前の見込み 13/分', fontsize=8, color=INK2, ha='center')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([names[r] for r in runs])
    ax[1].set_ylabel('誤警報 / 分(AI側の前進中)')
    ax[1].set_title('誤警報の回数', loc='left')
    ax[1].legend(fontsize=8, loc='upper left')
    ax[1].set_ylim(0, 30)
    # (c) warning-time ratio
    w_now = [100 * sw['per_run'][r]['logged: TTC<0.5 (now)']['warn_ratio'] for r in runs]
    w_rec = [100 * sw['per_run'][r]['recal floor TTC<0.5']['warn_ratio'] for r in runs]
    ax[2].bar(x - 0.18, w_now, 0.34, color=BLUE, label='今の床の基準')
    ax[2].bar(x + 0.18, w_rec, 0.34, color=AQUA, label='作り直した場合(推定)')
    for i in range(len(runs)):
        ax[2].text(x[i] - 0.18, w_now[i] + 0.8, '%.0f%%' % w_now[i], ha='center', fontsize=8, color=INK2)
        ax[2].text(x[i] + 0.18, w_rec[i] + 0.8, '%.0f%%' % w_rec[i], ha='center', fontsize=8, color=INK2)
    ax[2].axhline(11, color=MUTED, ls='--', lw=1)
    ax[2].text(1.5, 11.8, '見込み 11%', fontsize=8, color=INK2, ha='center')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([names[r] for r in runs])
    ax[2].set_ylabel('警告が出ている時間の割合 %(衝突のない前進中)')
    ax[2].set_title('警告時間率(減速の実コストに近い)', loc='left')
    ax[2].set_ylim(0, 48)
    fig.tight_layout()
    fig.savefig(FIG + 'tof_guard_eval.png', dpi=90)
    plt.close(fig)


def floor_fig():
    import datetime
    ref = np.load('/home/jetson/jetformer/data/tof_floor_ref.npy').astype(float)
    RAW = '/home/jetson/jetformer/data/raw'
    pts = []
    for log in ['20261010-172910', '20261010-173445', '20261010-181504', '20261010-192141', '20261010-193638',
                '20261010-200916', '20261010-201626', '20261010-202633'] + LOGS:
        root = os.path.join(RAW, log)
        for s in sorted(x for x in os.listdir(root) if x.startswith(log + '-')):
            try:
                T = np.load(os.path.join(root, s, 'TOF_FRONT.npz')); V = np.load(os.path.join(root, s, 'VEHICLE.npz'))
                I = np.load(os.path.join(root, s, 'IMU.npz'))
            except Exception:
                continue
            if len(V['timestamps_us']) == 0 or len(T['timestamps_us']) == 0:
                continue
            er = np.interp(T['timestamps_us'], V['timestamps_us'], V['erpm'])
            gz = np.interp(T['timestamps_us'], I['timestamps_us'], I['gyro'][:, 2])
            m = (np.abs(er) > 1500) & (np.abs(gz) < 60)
            if m.sum() < 20:
                continue
            raw = T['depth'][m]; dd = (raw & 0x1FFF).astype(float); conf = raw >> 13
            with np.errstate(all='ignore'):
                fl = np.nanpercentile(np.where(conf == 4, dd, np.nan), 80, axis=0)
                off9 = 100 * np.nanmedian(((fl - ref) / ref)[9])
                off18 = 100 * np.nanmedian(((fl - ref) / ref)[18])
            tt = datetime.datetime.fromtimestamp(T['timestamps_us'][0] / 1e6)
            pts.append((tt, off9, off18, log in LOGS))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    t = [p[0] for p in pts]
    ax.plot(t, [p[1] for p in pts], 'o', color=BLUE, ms=5, label='遠い床(行9、約2.0 m)')
    ax.plot(t, [p[2] for p in pts], 's', color=ORANGE, ms=5, label='中くらい(行18、約0.85 m)')
    ai = [p for p in pts if p[3]]
    ax.axvspan(ai[0][0], ai[-1][0] + datetime.timedelta(seconds=20), color=MUTED, alpha=0.12, lw=0)
    ax.text(ai[0][0], 4, ' 今回のAI走行3本', fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axhline(-20, color=CRIT, lw=1, ls='--')
    ax.text(t[0], -19, '床を「物」と見なす境目(床より20%近い)', fontsize=8, color=CRIT, va='bottom')
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_ylabel('床までの距離 − 基準 [%]')
    ax.set_xlabel('2026-10-10 の時刻(シーンごと)')
    ax.set_ylim(-32, 10)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('ToFの床の見え方が 20:11 頃から基準より近くなっていた(ToFガードの床の基準は 18:13 に作成)', loc='left')
    fig.tight_layout()
    fig.savefig(FIG + 'tof_floor_drift.png', dpi=90)
    plt.close(fig)


def hotspot_fig():
    r = json.load(open(SRC + 'takeovers.json'))
    order = ['DE', 'DE2', 'LH', 'GM_in', 'GM_out', 'HS_P', 'R1']
    names = [v['name'] for v in r.values()]
    cnt = {n: {o: 0 for o in order} for n in names}
    stop = {o: 0 for o in order}
    for v in r.values():
        for x in v['impacts']:
            cnt[v['name']][x['loc']] += 1
            if not x['glance']:
                stop[x['loc']] += 1
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(order))
    cols = [BLUE, ORANGE, AQUA]
    bottom = np.zeros(len(order))
    for n, c in zip(names, cols):
        vals = np.array([cnt[n][o] for o in order], float)
        ax.bar(x, vals, 0.6, bottom=bottom, color=c, label=n, edgecolor=SURF, linewidth=2)
        bottom += vals
    for i, o in enumerate(order):
        ax.text(i, bottom[i] + 0.15, '%d(止まった %d)' % (bottom[i], stop[o]), ha='center', fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([LOCJ[o] for o in order])
    ax.set_ylabel('AI側の衝突(水平 ≥2.5G)')
    ax.set_ylim(0, 8)
    ax.legend(fontsize=8)
    ax.set_title('衝突の場所(21件、カメラ画像で目視分類)。DE(バックストレート右の仕切り)が 10/21', loc='left')
    fig.tight_layout()
    fig.savefig(FIG + 'hotspots.png', dpi=90)
    plt.close(fig)


def params_fig():
    pf = json.load(open(SRC + 'params_fx.json'))
    sel = [p for p in pf if p['log'] in LOGS and p['ai_s'] >= 3]
    labels = ['%s\nst%.2f th%.2f 先読み%d\n(AI %.0f s)' % (p['name'], p['steer_gain'], p['throttle_gain'], p['lookahead'], p['ai_s'])
              for p in sel]
    mets = [('steer_flips_per_s', 'ステアの左右反転 / 秒'), ('steer_limit', 'ステアが限界(1000/2000)にいる割合'),
            ('pred_jitter_us', '使った予測のフレーム間の揺れ µs'), ('impacts_per_100m', '衝突 / 100 m')]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    for a, (k, t) in zip(ax, mets):
        vals = [p[k] if p[k] is not None else np.nan for p in sel]
        c = [ORANGE if p['lookahead'] == 10 else BLUE for p in sel]
        a.barh(np.arange(len(sel)), vals, color=c, height=0.6)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                a.text(v, i, ' %.2f' % v if v < 10 else ' %.0f' % v, va='center', fontsize=8, color=INK2)
        a.set_yticks(np.arange(len(sel)))
        a.set_yticklabels(labels if a is ax[0] else [''] * len(sel), fontsize=7.5)
        a.invert_yaxis()
        a.set_title(t, loc='left', fontsize=10)
    fig.suptitle('設定ごとの比較(オレンジ = 先読み10)。設定ごとのAI時間は 3〜30 秒しかなく、先読み10以外の差は誤差の範囲', x=0.01, ha='left',
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG + 'params.png', dpi=90)
    plt.close(fig)


if __name__ == '__main__':
    for log in LOGS:
        timeline(log)
    takeover_fig()
    tof_fig()
    floor_fig()
    hotspot_fig()
    params_fig()
    print('ok')
