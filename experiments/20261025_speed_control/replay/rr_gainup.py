#!/usr/bin/env python3
"""反証 V-replay: 自動合わせがゲインを上げたら(No.9 の記録 × gm)、天井の下で周と 3 周の窓はどうなるか(2026-10-25)。読むだけ。
仮想の閉ループ rr_ceiling.virtual(むだ時間 0.05 秒)。モデルの反応 kv = 0 / 50 µs/(m/s)。止まってからの 3 周 = 走りながらの 3 周 + 0.63 秒
(analysis_tt 5 章)。当たる率は SW の中の最高の速さの帯(analysis_collisions 4 章)。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_cache  # noqa: E402
import rr_ceiling as RC  # noqa: E402
import rr_common as C  # noqa: E402

LOGS = {'No.9': '20261024-162033', 'No.11': '20261024-183123', 'No.17': '20261024-184549', 'No.10': '20261024-162305'}


def main():
    D = rr_cache.load()['runs']
    out = []
    p = C.P('impl')
    for name, log in LOGS.items():
        d = D[log]
        b = RC.run_b(d)
        for gm in (1.0, 1.03, 1.07, 1.10, 1.15):
            for ceil in (False, True):
                for kv in (0.0, 50.0):
                    laps = []
                    for ch in RC.lap_chains(d):
                        ij = RC.idx(d, ch[0]['t_start'], ch[-1]['t_end'])
                        if ij is None:
                            continue
                        o = RC.virtual(d, p, ij[0], ij[1], b, dead=0.05, gm=gm, ceil=ceil, kv=kv)
                        tt = d['t'][ij[0]:ij[1]]
                        locd = d['locd'][ij[0]:ij[1]]
                        for lp in ch:
                            m = (tt >= lp['t_start']) & (tt < lp['t_end'])
                            sw = m & (locd == C.LOC_DETAIL.index('SW'))
                            r1 = np.nonzero(m & (locd == C.LOC_DETAIL.index('R1')))[0]
                            laps.append({'t_start': lp['t_start'], 'lap_s': lp['lap_s'] + float(o['loss'][m].sum()),
                                         'v90': float(np.percentile(o['vm'][m], 90)),
                                         'sw_vmax': float(o['vm'][sw].max()) if sw.any() else None,
                                         'r1_entry': float(o['vm'][max(r1[0] - 12, 0):r1[0] + 1].max()) if len(r1) else None,
                                         'changed': float((o['changed'] & m).sum() / m.sum())})
                    w3 = [sum(x['lap_s'] for x in laps[k:k + 3]) for k in range(len(laps) - 2)
                          if all(abs(laps[k + j]['t_start'] + (d['laps'][0]['lap_s'] * 0) - laps[k + j]['t_start']) < 1 for j in range(3))]
                    # 続いた 3 周だけ(記録の周の時刻で)
                    rec = sorted(d['laps'], key=lambda x: x['t_start'])
                    cont = [k for k in range(len(rec) - 2) if abs(rec[k]['t_start'] + rec[k]['lap_s'] - rec[k + 1]['t_start']) < 0.02
                            and abs(rec[k + 1]['t_start'] + rec[k + 1]['lap_s'] - rec[k + 2]['t_start']) < 0.02]
                    lt = {round(x['t_start'], 3): x for x in laps}
                    w3 = [sum(lt[round(rec[k + j]['t_start'], 3)]['lap_s'] for j in range(3)) for k in cont
                          if all(round(rec[k + j]['t_start'], 3) in lt for j in range(3))]
                    sw = [x['sw_vmax'] for x in laps if x['sw_vmax'] is not None]
                    out.append({'run': name, 'log': log, 'tg': d['tg'], 'gm': gm, 'gain': round(d['tg'] * gm, 3), 'ceil': ceil, 'kv': kv,
                                'laps': len(laps), 'lap_mean': float(np.mean([x['lap_s'] for x in laps])),
                                'lap_best': float(np.min([x['lap_s'] for x in laps])),
                                'v90_mean': float(np.mean([x['v90'] for x in laps])),
                                'w3_best': min(w3) if w3 else None, 'w3_med': float(np.median(w3)) if w3 else None,
                                'stand3_best': min(w3) + 0.63 if w3 else None,
                                'sw_vmax_med': float(np.median(sw)) if sw else None,
                                'sw_hits_expected_per_lap': float(np.mean([RC.sw_rate(v) for v in sw])) if sw else None,
                                'r1_entry_med': float(np.median([x['r1_entry'] for x in laps if x['r1_entry'] is not None])),
                                'changed': float(np.mean([x['changed'] for x in laps]))})
                    r = out[-1]
                    print('%s gm %.2f ceil %d kv %2.0f: lap %.3f best %.2f v90 %.2f w3 %s stand %s sw %.2f hit/lap %.3f r1 %.2f chg %.2f' % (
                        name, gm, ceil, kv, r['lap_mean'], r['lap_best'], r['v90_mean'], None if r['w3_best'] is None else round(r['w3_best'], 2),
                        None if r['stand3_best'] is None else round(r['stand3_best'], 2), r['sw_vmax_med'], r['sw_hits_expected_per_lap'],
                        r['r1_entry_med'], r['changed']))
    C.jdump(out, os.path.join(C.HERE, 'rr_gainup.json'))


if __name__ == '__main__':
    main()
