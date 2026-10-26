#!/usr/bin/env python3
"""反証 V-replay: 速さのゲインの自動合わせが、各走行でどのゲインを選ぶか(2026-10-25)。読むだけ。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I rr_auto.py

実装の純粋な関数(jetformer.speed_auto の lap_stats・fb_lap・end_summary)をそのまま使う。周は R1 の起点 → 次の起点
(tools の段の偽の起点を外したあと。section_scorecard.score_log)。電圧の記録は無いので m_ff = 1(源 0)。
  A. 開ループの影: 記録のまま(OFF の日の記録と同じ。天井の印は開ループの「変えたはず」)
  B. 周の水準の閉ループ: 周ごとに、その時の m(= m_fb)を記録のモデルの指令に掛けた仮想の車(rr_ceiling.virtual。
     むだ時間 0.05 秒・モデルの反応なし)で周を走らせ、その周の p90・天井の割合で次の m_fb を決める。天井 ON / OFF
  C. 記録のゲインの差から、ゲインの弾力 ε(ゲイン 1% で周の p90 が何 % 変わるか)
  D. 電圧の前送りの計算(α = 1・V_ref 8.35 V)
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_cache  # noqa: E402
import rr_ceiling as RC  # noqa: E402
import rr_common as C  # noqa: E402
from rr_common import sc  # noqa: E402
from jetformer import speed_auto as sa  # noqa: E402

AP = sa.AutoParams()


def frames_of(d, ceil_lim):
    return [sa.LapFrame(float(d['t'][i]), float(d['v'][i]), bool(d['ai'][i]), int(d['assist'][i]), bool(d['guard'][i]),
                        float(d['hg'][i]), float(d['a3'][i]), bool(ceil_lim[i])) for i in range(len(d['t']))]


def laps_sorted(d):
    return sorted([dict(lp, t_end=round(lp['t_start'] + lp['lap_s'], 3)) for lp in d['laps']], key=lambda x: x['t_start'])


def shadow(d, ol):
    """A. 開ループの影(記録の周のまま)"""
    fr = frames_of(d, ol['mode'] >= sc.CEIL_LIMIT)
    st = sa.FbState()
    rows = []
    for lp in laps_sorted(d):
        ls = sa.lap_stats(fr, lp['t_start'], lp['t_end'])
        st, ev, dl = sa.fb_lap(st, ls, 1.0)
        rows.append({'t': lp['t_start'], 'lap_s': round(ls.dur, 2), 'v90': ls.v90, 'ok': ls.ok, 'reason': ls.reason,
                     'event': sa.EVENT_NAMES[ev], 'ceil_frac': ls.ceil_frac, 'm_fb': st.m_fb})
    return rows, sa.end_summary(st, False, d['tg'])


def closed(d, B, ceil=True, start_m=1.0):
    """B. 周の水準の閉ループ。周 k は m_k = m_fb(k) でその周の記録を仮想の車で走らせる(各周は記録の速さから始める)"""
    p = C.P('impl')
    st = sa.FbState(m_fb=start_m)
    rows = []
    for lp in laps_sorted(d):
        ij = RC.idx(d, lp['t_start'], lp['t_end'])
        if ij is None:
            continue
        m = st.m_fb
        o = RC.virtual(d, p, ij[0], ij[1], B, dead=0.05, gm=m, ceil=ceil)
        n = ij[1] - ij[0]
        # 周の数は、仮想の速さ・天井の印で作る(使える周の条件は記録のまま)
        fr = [sa.LapFrame(float(d['t'][ij[0] + k]), float(o['vm'][k]), bool(d['ai'][ij[0] + k]), int(d['assist'][ij[0] + k]),
                          bool(d['guard'][ij[0] + k]), float(d['hg'][ij[0] + k]), float(d['a3'][ij[0] + k]),
                          bool(o['changed'][k] and ceil)) for k in range(n)]
        ls = sa.lap_stats(fr, lp['t_start'], lp['t_end'])
        lap_v = lp['lap_s'] + float(o['loss'].sum())
        st, ev, dl = sa.fb_lap(st, ls, 1.0)
        rows.append({'t': lp['t_start'], 'm': m, 'lap_rec': lp['lap_s'], 'lap_v': lap_v, 'v90_v': ls.v90, 'ok': ls.ok,
                     'reason': ls.reason, 'event': sa.EVENT_NAMES[ev], 'ceil_frac': ls.ceil_frac, 'm_fb_next': st.m_fb,
                     'warn': st.warn})
    return rows, sa.end_summary(st, True, d['tg'])


def elasticity(D):
    """C. 同じ回・同じモデル・同じステアの走行どうしで、ゲインと周の p90 の中央値(記録の周、全部の AI の周)"""
    groups = {}
    for log, d in D.items():
        if not d['laps']:
            continue
        model = 'B' if ('clean_driving' in d['desc'] or 'cand=B' in d['desc'] or 'model=' not in d['desc']) else 'other'
        tt = C.TT_RUNS.get(log, {}).get('setting', '')
        if tt:
            model = tt.split()[0]
        steer = C.desc_tok(d['desc'], 'steer') or '1.0'
        if tt and 'ステア1.10' in tt:
            steer = '1.1'
        key = (d['session'].split()[0] if d['session'].startswith('10/23') else d['session'], model, steer)
        v90 = []
        for lp in laps_sorted(d):
            ij = RC.idx(d, lp['t_start'], lp['t_end'])
            if ij:
                v90.append(float(np.percentile(d['v'][ij[0]:ij[1]], 90)))
        if v90:
            groups.setdefault(key, []).append({'log': log, 'no': d['no'], 'tg': round(d['tg'], 3), 'v90_med': float(np.median(v90)),
                                               'laps': len(v90), 'lap_med': float(np.median([lp['lap_s'] for lp in d['laps']]))})
    out = []
    for key, rows in groups.items():
        gs = sorted(set(r['tg'] for r in rows))
        if len(gs) < 2:
            continue
        x = np.log([r['tg'] for r in rows])
        y = np.log([r['v90_med'] for r in rows])
        w = np.array([r['laps'] for r in rows], float)
        A = np.vstack([x, np.ones_like(x)]).T
        coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        out.append({'group': ' '.join(key), 'runs': rows, 'eps': float(coef[0])})
    return out


def main():
    t0 = time.time()
    D = rr_cache.load()['runs']
    B = {log: RC.run_b(d) for log, d in D.items()}
    res = {'note': __doc__, 'params': AP._asdict(), 'shadow': [], 'closed': []}
    for log, d in D.items():
        if not d['laps']:
            continue
        ol = C.open_loop(d, C.P('impl'))
        rows, end = shadow(d, ol)
        res['shadow'].append({'log': log, 'no': d['no'], 'session': d['session'], 'tg': d['tg'], 'laps': rows, 'end': end})
    print('shadow %.1f s' % (time.time() - t0))
    for log in ['20261024-162033', '20261024-162305', '20261024-160654', '20261024-160847', '20261024-183123',
                '20261024-183437', '20261024-184549', '20261024-183826', '20261024-134607', '20261024-134907',
                '20261023-211828', '20261023-211209']:
        d = D[log]
        for ceil in (True, False):
            rows, end = closed(d, B[log], ceil=ceil)
            res['closed'].append({'log': log, 'no': d['no'], 'session': d['session'], 'tg': d['tg'], 'ceil': ceil,
                                  'laps': rows, 'end': end})
    print('closed %.1f s' % (time.time() - t0))
    res['elasticity'] = elasticity(D)
    res['ff'] = {str(v): sa.ff_of_voltage(v) for v in (7.4, 7.6, 7.7, 7.8, 8.0, 8.2, 8.35, 8.4, 8.49)}
    # 周の時間の式(14.52 − 1.50 × p90)が、天井の仮想の周で成り立つか: No.9 の周(記録・天井あり ×1.00〜1.10)
    C.jdump(res, os.path.join(C.HERE, 'rr_auto.json'))
    print('done %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
