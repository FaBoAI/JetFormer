#!/usr/bin/env python3
"""fix9 tonight s24-analysis (2026-10-29): 速さのゲイン × 速さの天井で、周と止まってからの 3 周はどこまで縮むか(机の上の見積もり)。読むだけ。

    cd /home/jetson/jetformer/experiments/20261026_fix/fix9/tonight/s24-analysis
    PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1200M --need-mb 2000 -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I s24_gain_sweep.py

- 仮想の閉ループは 10/25 の反証 V-replay のもの(experiments/20261025_speed_control/replay/rr_ceiling.virtual。むだ時間 0.05 秒。
  前後だけ・道すじと舵は記録のまま・横の滑りと新しい衝突は入れない)をそのまま使う。天井は実装の純粋な関数(safety_caps.ceil_step)。
- 10/25 の rr_gainup.py との違い: 天井を今の既定の「全体 4.5・左 4.1」(ユーザーが決めた値)にした / 4.2・上積み U1(4.8/4.2)・天井なし
  も並べた / ゲインは絶対の値(0.90〜1.10)/ 止まってからの 3 周を「今の発進(+0.63 秒)」と「発進の前進 ON(+0.35 秒)」の 2 つで出す /
  区間(R1+GM・DE+SW・HS+LH)ごとの仮想の時間 / 3 周のうちに出っ張りに 1 回でも当たる見込み。
- 記録は C.load_run でその場で読む(キャッシュ cache_runs.pkl は作らない・書かない)。data/raw・data/labels は変えない。
- モデルの反応 kv: 0(モデルは記録のまま)と 50 µs/(m/s)(モデル B が「自分が速いと踏む」形。解析 B 6 章の粗い見積もり)。本物は間とみる。
出力: s24_gain_sweep.json(このフォルダ)
"""
import json
import os
import sys
import time

import numpy as np

REPLAY = '/home/jetson/jetformer/experiments/20261025_speed_control/replay'
sys.path.insert(0, REPLAY)
import rr_common as C  # noqa: E402
import rr_ceiling as RC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {'No.9(2 回目・満充電・修理の前)': '20261024-162033',
        'No.12(3 回目・修理のあと)': '20261024-183437',
        'No.17(3 回目・修理のあと)': '20261024-184549'}
GAINS = (0.90, 0.95, 1.00, 1.05, 1.10)
CEILS = {'天井なし': None, '天井 4.5/左 4.1(今の既定)': (4.5, 4.1), '天井 4.5/左 4.2': (4.5, 4.2), '天井 4.8/左 4.2(U1)': (4.8, 4.2)}
KVS = (0.0, 50.0)
LOSS = {'now': 0.63, 'boost': 0.35}
GROUP_OF = {'R1': 'R1+GM', 'GM_in': 'R1+GM', 'GM_out': 'R1+GM', 'DE': 'DE+SW', 'SW': 'DE+SW', 'HS': 'HS+LH', 'LH': 'HS+LH'}


def one(d, b, gain, ceil, kv):
    gm = gain / d['tg']
    p = C.P('impl', cap_all=ceil[0], cap_left=ceil[1]) if ceil else C.P('impl')
    laps = []
    for ch in RC.lap_chains(d):
        ij = RC.idx(d, ch[0]['t_start'], ch[-1]['t_end'])
        if ij is None:
            continue
        o = RC.virtual(d, p, ij[0], ij[1], b, dead=0.05, gm=gm, ceil=bool(ceil), kv=kv)
        tt = d['t'][ij[0]:ij[1]]
        locd = d['locd'][ij[0]:ij[1]]
        for lp in ch:
            m = (tt >= lp['t_start']) & (tt < lp['t_end'])
            if not m.any():
                continue
            sw = m & (locd == C.LOC_DETAIL.index('SW'))
            r1 = np.nonzero(m & (locd == C.LOC_DETAIL.index('R1')))[0]
            hs = m & (locd == C.LOC_DETAIL.index('HS'))
            grp = {'R1+GM': 0.0, 'DE+SW': 0.0, 'HS+LH': 0.0}
            for j, name in enumerate(C.LOC_DETAIL):
                if name in GROUP_OF:
                    grp[GROUP_OF[name]] += float(o['loss'][m & (locd == j)].sum())
            laps.append({'t_start': round(lp['t_start'], 3), 'rec_lap_s': lp['lap_s'], 'clean': bool(lp.get('clean', True)),
                         'lap_s': lp['lap_s'] + float(o['loss'][m].sum()), 'loss_by_group': grp,
                         'v90': float(np.percentile(o['vm'][m], 90)),
                         'sw_vmax': float(o['vm'][sw].max()) if sw.any() else None,
                         'hs_vmax': float(o['vm'][hs].max()) if hs.any() else None,
                         'r1_entry': float(o['vm'][max(r1[0] - 12, 0):r1[0] + 1].max()) if len(r1) else None,
                         'changed': float((o['changed'] & m).sum() / m.sum())})
    rec = sorted(d['laps'], key=lambda x: x['t_start'])
    cont = [k for k in range(len(rec) - 2) if abs(rec[k]['t_start'] + rec[k]['lap_s'] - rec[k + 1]['t_start']) < 0.02
            and abs(rec[k + 1]['t_start'] + rec[k + 1]['lap_s'] - rec[k + 2]['t_start']) < 0.02]
    lt = {x['t_start']: x for x in laps}
    w3 = []
    for k in cont:
        keys = [round(rec[k + j]['t_start'], 3) for j in range(3)]
        if all(kk in lt for kk in keys):
            w3.append(sum(lt[kk]['lap_s'] for kk in keys))
    sw = [x['sw_vmax'] for x in laps if x['sw_vmax'] is not None]
    rates = [RC.sw_rate(v) for v in sw]
    p_lap = float(np.mean(rates)) if rates else None
    return {
        'gain': gain, 'gm': round(gm, 4), 'kv': kv, 'laps': len(laps),
        'lap_mean': float(np.mean([x['lap_s'] for x in laps])), 'lap_median': float(np.median([x['lap_s'] for x in laps])),
        'lap_best': float(np.min([x['lap_s'] for x in laps])),
        'w3_n': len(w3), 'w3_best': min(w3) if w3 else None, 'w3_median': float(np.median(w3)) if w3 else None,
        'stand_best_now': min(w3) + LOSS['now'] if w3 else None, 'stand_best_boost': min(w3) + LOSS['boost'] if w3 else None,
        'stand_med_now': float(np.median(w3)) + LOSS['now'] if w3 else None,
        'stand_med_boost': float(np.median(w3)) + LOSS['boost'] if w3 else None,
        'loss_by_group_mean': {g: float(np.mean([x['loss_by_group'][g] for x in laps])) for g in ('R1+GM', 'DE+SW', 'HS+LH')},
        'v90_mean': float(np.mean([x['v90'] for x in laps])),
        'sw_vmax_med': float(np.median(sw)) if sw else None, 'sw_vmax_max': float(np.max(sw)) if sw else None,
        'hs_vmax_med': float(np.median([x['hs_vmax'] for x in laps if x['hs_vmax'] is not None])),
        'bump_hit_per_lap': p_lap, 'bump_hit_in_3laps': (1 - (1 - p_lap) ** 3) if p_lap is not None else None,
        'r1_entry_med': float(np.median([x['r1_entry'] for x in laps if x['r1_entry'] is not None])),
        'ceil_changed': float(np.mean([x['changed'] for x in laps])),
    }


def main():
    t0 = time.time()
    out = {'created_by': 'experiments/20261026_fix/fix9/tonight/s24-analysis/s24_gain_sweep.py', 'loss': LOSS, 'runs': {}}
    for name, log in RUNS.items():
        t1 = time.time()
        d = C.load_run(log)
        b = RC.run_b(d)
        rec_laps = sorted(d['laps'], key=lambda x: x['t_start'])
        res = {'log': log, 'tg': d['tg'], 'b': b, 'rec_laps': [round(x['lap_s'], 3) for x in rec_laps], 'rows': []}
        for cname, ceil in CEILS.items():
            for gain in GAINS:
                for kv in KVS:
                    if ceil is None and kv == 50.0 and gain > 1.0:
                        pass
                    r = one(d, b, gain, ceil, kv)
                    r['ceil'] = cname
                    res['rows'].append(r)
                    print('%s %-24s g %.2f kv %2.0f | lap mean %.3f best %.3f | w3 best %s med %s | stand(now) %s (boost) %s | '
                          'v90 %.2f sw %.2f hs %.2f hit3 %s r1 %.2f chg %.2f' % (
                              name[:5], cname, gain, kv, r['lap_mean'], r['lap_best'],
                              None if r['w3_best'] is None else round(r['w3_best'], 2),
                              None if r['w3_median'] is None else round(r['w3_median'], 2),
                              None if r['stand_best_now'] is None else round(r['stand_best_now'], 2),
                              None if r['stand_best_boost'] is None else round(r['stand_best_boost'], 2),
                              r['v90_mean'], r['sw_vmax_med'] or float('nan'), r['hs_vmax_med'],
                              None if r['bump_hit_in_3laps'] is None else round(r['bump_hit_in_3laps'], 3),
                              r['r1_entry_med'], r['ceil_changed']), flush=True)
        out['runs'][name] = res
        print('  (%s: %.1f 秒)' % (name, time.time() - t1), flush=True)
        del d
    out['elapsed_s'] = time.time() - t0
    C.jdump(out, os.path.join(HERE, 's24_gain_sweep.json'))
    print('elapsed %.1f s' % out['elapsed_s'])


if __name__ == '__main__':
    main()
