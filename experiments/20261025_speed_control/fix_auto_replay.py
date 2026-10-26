#!/usr/bin/env python3
"""fix.md(2026-10-25): 自動合わせの物差しを「周の速さ p90」から「周の時間」に替えたあとの再生。読むだけ。

    cd ~/jetformer/experiments/20261025_speed_control/replay
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I rr_cache.py   # 初回だけ
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I ../fix_auto_replay.py

反証 V-replay の仮想の閉ループ(replay/rr_ceiling.virtual。むだ時間 0.05 秒・モデルの反応なし・天井は実装の ceil_step)を
そのまま使う。自動合わせは実装の純粋な関数(jetformer.speed_auto の lap_stats・fb_lap・end_summary)。電圧の記録は無いので m_ff = 1。
  A. 周の水準の閉ループ(天井 ON): 周 k は m_k = m_fb(k) をゲイン倍 gm にして記録の周を仮想の車で走らせ、仮想の周の時間・
     天井の割合で次の m_fb を決める。物差し 周の時間(既定・fix)と p90(仕様の文のまま)を並べる
  B. ゲイン倍の掃き出し(No.9): gm ごとの仮想の周の時間・天井の割合・p90(目標 8.22 秒はどこか)
  C. 目安で次の走行の throttle_gain を決める(人が 1 走行 = 1 設定で直す): 天井 ON の影(gm 1)の目安 → その gm で同じ記録を
     走らせ直す → もう一度目安、を 3 回。止まってからの 3 周 = 続いた 3 周の最速 + 0.63 秒(analysis_tt 5 章)
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'replay'))
import rr_cache  # noqa: E402
import rr_ceiling as RC  # noqa: E402
import rr_common as C  # noqa: E402
from jetformer import speed_auto as sa  # noqa: E402

AP_LAP = sa.AutoParams()
AP_P90 = sa.auto_params(metric=sa.METRIC_P90)
RUNS = {'10/24 No.9': '20261024-162033', '10/24 No.10': '20261024-162305', '10/24 No.7': '20261024-160654',
        '10/24 No.8': '20261024-160847', '10/24 No.11': '20261024-183123', '10/24 No.12': '20261024-183437',
        '10/24 No.17': '20261024-184549', '10/24 No.14': '20261024-183826', '10/24 No.2': '20261024-134907',
        '10/23 No.17': '20261023-211828'}
OUT = os.path.join(HERE, 'fix_auto_replay.json')


def holds_of(log, d):
    """AI 側のまま中立で止めたフレーム(短ければ周を使う。fix.md)。キャッシュには無いので、ラベルから作り直す"""
    info, fr = C.le.label_log(log)
    hold = fr['ai'].astype(bool) & ~fr['dry_run'].astype(bool) & fr['hold'].astype(bool)
    assert len(hold) == len(d['t']) and np.allclose(fr['t_s'].astype(float), d['t'])
    return hold


def laps_sorted(d):
    return sorted([dict(lp, t_end=round(lp['t_start'] + lp['lap_s'], 3)) for lp in d['laps']], key=lambda x: x['t_start'])


def virtual_lap(d, p, b, lp, gm):
    """記録の周 lp を、ゲイン倍 gm・天井 ON の仮想の車で → (LapFrame の列, 仮想の周の時間, 仮想の p90, 天井の割合)"""
    ij = RC.idx(d, lp['t_start'], lp['t_end'])
    if ij is None:
        return None
    o = RC.virtual(d, p, ij[0], ij[1], b, dead=0.05, gm=gm, ceil=True)
    n = ij[1] - ij[0]
    g = d['tg'] * gm
    hold = d.get('hold_fix')
    fr = [sa.LapFrame(float(d['t'][ij[0] + k]), float(o['vm'][k]) if d['ai'][ij[0] + k] else float('nan'),
                      bool(d['ai'][ij[0] + k]), int(d['assist'][ij[0] + k]),
                      bool(d['guard'][ij[0] + k]), float(d['hg'][ij[0] + k]), float(d['a3'][ij[0] + k]), bool(o['changed'][k]),
                      bool(hold[ij[0] + k]) if hold is not None else False,
                      g if d['ai'][ij[0] + k] else float('nan')) for k in range(n)]
    lap_v = lp['lap_s'] + float(o['loss'].sum())
    return fr, lap_v, float(np.percentile(o['vm'], 90)), float(o['changed'].mean())


def closed_loop(d, p, b, ap, gm0=1.0):
    """A. 周の水準の閉ループ(1 走行、m_fb は 1.00 から。gm = gm0 × m_fb)"""
    st = sa.FbState()
    rows = []
    for lp in laps_sorted(d):
        m = st.m_fb
        r = virtual_lap(d, p, b, lp, gm0 * m)
        if r is None:
            continue
        fr, lap_v, v90, frac = r
        ls = sa.lap_stats(fr, lp['t_start'], lp['t_end'], ap)
        ls = ls._replace(dur=lap_v)          # 周の時間は仮想の車の時間(ほかの「使える周」の条件は記録のまま)
        st, ev, dl = sa.fb_lap(st, ls, 1.0, ap)
        rows.append({'t': round(lp['t_start'], 2), 'm_fb': round(m, 4), 'lap_v': round(lap_v, 3), 'v90': round(v90, 3),
                     'ceil_frac': round(frac, 3), 'ok': ls.ok, 'reason': ls.reason, 'event': sa.EVENT_NAMES[ev],
                     'delta': round(dl, 4)})
    return rows, sa.end_summary(st, True, d['tg'] * gm0, ap), st


def run_gm(d, p, b, gm):
    """ゲイン倍 gm(一定)・天井 ON で全部の周 → 周の時間の列(記録の順)と続いた 3 周の最速"""
    laps = []
    for lp in laps_sorted(d):
        r = virtual_lap(d, p, b, lp, gm)
        if r is not None:
            laps.append((lp, r))
    rec = [lp for lp, _ in laps]
    times = [r[1] for _, r in laps]
    cont = [k for k in range(len(rec) - 2)
            if abs(rec[k]['t_end'] - rec[k + 1]['t_start']) < 0.02 and abs(rec[k + 1]['t_end'] - rec[k + 2]['t_start']) < 0.02]
    w3 = [sum(times[k:k + 3]) for k in cont]
    return laps, times, (min(w3) if w3 else None)


def suggest_loop(d, p, b, iters=3):
    """C. 目安 → throttle_gain を直す → 同じ記録を走らせ直す(自動合わせ OFF の影 = 掛けたゲインは throttle_gain)"""
    out = []
    gm = 1.0
    for it in range(iters + 1):
        laps, times, w3 = run_gm(d, p, b, gm)
        st = sa.FbState()
        for lp, (fr, lap_v, v90, frac) in laps:
            ls = sa.lap_stats(fr, lp['t_start'], lp['t_end'], AP_LAP)._replace(dur=lap_v)
            st, ev, dl = sa.fb_lap(st, ls, 1.0, AP_LAP)
        end = sa.end_summary(st, False, d['tg'] * gm, AP_LAP)
        fr_all = [r[3] for _, r in laps]
        out.append({'iter': it, 'gain': round(d['tg'] * gm, 3), 'gm': round(gm, 4), 'laps': len(times),
                    'lap_mean': round(float(np.mean(times)), 3) if times else None,
                    'lap_med': round(float(np.median(times)), 3) if times else None,
                    'w3_best': None if w3 is None else round(w3, 3), 'stand3_best': None if w3 is None else round(w3 + 0.63, 3),
                    'ceil_frac_mean': round(float(np.mean(fr_all)), 3) if fr_all else None,
                    'laps_used': end['laps_used'], 'suggest_fixed': end['suggest_throttle_gain_fixed']})
        if end['suggest_throttle_gain_fixed'] is None:
            break
        gm = end['suggest_throttle_gain_fixed'] / d['tg']
    return out


def main():
    t0 = time.time()
    D = rr_cache.load()['runs']
    p = C.P('impl')
    res = {'note': __doc__, 'target_lap_s': AP_LAP.target_lap_s(), 'A': [], 'B': [], 'C': []}
    for name, log in RUNS.items():
        if log not in D or not D[log]['laps']:
            print('skip', name, log)
            continue
        d = dict(D[log])
        d['hold_fix'] = holds_of(log, d)
        b = RC.run_b(d)
        for metric, ap in (('lap_s', AP_LAP), ('p90', AP_P90)):
            rows, end, st = closed_loop(d, p, b, ap)
            used = [r for r in rows if r['ok']]
            res['A'].append({'run': name, 'log': log, 'tg': d['tg'], 'metric': metric, 'laps': rows,
                             'm_fb_end': round(st.m_fb, 4), 'laps_used': st.laps_used, 'warn_seen': list(st.warn_seen),
                             'events': {k: sum(1 for r in rows if r['event'] == k) for k in set(r['event'] for r in rows)},
                             'lap_v_used_first': used[0]['lap_v'] if used else None,
                             'lap_v_used_last': used[-1]['lap_v'] if used else None,
                             'suggest_auto': end['suggest_throttle_gain_auto'], 'suggest_fixed': end['suggest_throttle_gain_fixed']})
            a = res['A'][-1]
            print('A %-12s %-5s tg %.2f laps %2d used %2d m_fb_end %.3f  lap first %s last %s  events %s  suggest %s / %s' % (
                name, metric, d['tg'], len(rows), a['laps_used'], a['m_fb_end'], a['lap_v_used_first'], a['lap_v_used_last'],
                a['events'], a['suggest_auto'], a['suggest_fixed']))
        res['C'].append({'run': name, 'log': log, 'tg': d['tg'], 'iters': suggest_loop(d, p, b)})
        for it in res['C'][-1]['iters']:
            print('C %-12s iter %d gain %.3f lap mean %s med %s stand3 %s ceil %.3f used %d -> suggest %s' % (
                name, it['iter'], it['gain'], it['lap_mean'], it['lap_med'], it['stand3_best'], it['ceil_frac_mean'] or 0,
                it['laps_used'], it['suggest_fixed']))
    d = D[RUNS['10/24 No.9']]
    b = RC.run_b(d)
    for gm in (1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15):
        laps, times, w3 = run_gm(d, p, b, gm)
        fr = [r[3] for _, r in laps]
        v90 = [r[2] for _, r in laps]
        res['B'].append({'gm': gm, 'gain': round(d['tg'] * gm, 3), 'lap_mean': round(float(np.mean(times)), 3),
                         'v90_mean': round(float(np.mean(v90)), 3), 'ceil_frac': round(float(np.mean(fr)), 3),
                         'stand3_best': None if w3 is None else round(w3 + 0.63, 3)})
        print('B No.9 gm %.2f gain %.3f lap %.3f v90 %.3f ceil %.3f stand3 %s' % (
            gm, d['tg'] * gm, res['B'][-1]['lap_mean'], res['B'][-1]['v90_mean'], res['B'][-1]['ceil_frac'],
            res['B'][-1]['stand3_best']))
    C.jdump(res, OUT)
    print('done %.1f s -> %s' % (time.time() - t0, OUT))


if __name__ == '__main__':
    main()
