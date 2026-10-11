#!/usr/bin/env python3
"""区間を大きくまとめて(ヘアピン群 R1+GM、後半 DE+SW+HS、LH)、人と AI の周の時間・距離・平均速度を比べる。
ψ のしきい値で切った細かい区間(DE/SW/HS)は、ψ の揺れで境界が前後するので、まとめた値を主に使う。"""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '../sections.json'))
L = [l for x in d['logs'] for l in x['laps'] if all(l['sec'][s] for s in ['R1', 'GM', 'DE', 'SW', 'HS', 'LH'])]
def parts(l):
    s = l['sec']
    t = lambda ks: sum(s[k]['t_s'] for k in ks)
    dist = lambda ks: sum(s[k]['t_s'] * s[k]['v_mean'] for k in ks)
    return {'hp_t': t(['R1', 'GM']), 'back_t': t(['DE', 'SW', 'HS']), 'lh_t': t(['LH']),
            'hp_d': dist(['R1', 'GM']), 'back_d': dist(['DE', 'SW', 'HS']), 'lh_d': dist(['LH']),
            'back_vmax': max(s[k]['v_max'] for k in ['DE', 'SW', 'HS']),
            'back_alat90': float(np.median([s[k]['alat_kin_p90'] for k in ['SW']]))}
G = {
 'human_fast_clean(<=9.2s)': lambda l: l['driver'] == 'human' and l['n_imp'] == 0 and l['lap_s'] <= 9.2,
 'human_clean': lambda l: l['driver'] == 'human' and l['n_imp'] == 0,
 'human_all': lambda l: l['driver'] == 'human',
 'ai_clean': lambda l: l['driver'] == 'ai' and l['n_imp'] == 0,
 'ai_all': lambda l: l['driver'] == 'ai',
 'ai_006': lambda l: l['who'] == 'ai_run_006',
 'ai_007': lambda l: l['who'] == 'ai_run_007',
}
out = {}
for name, f in G.items():
    g = [l for l in L if f(l)]
    P = [parts(l) for l in g]
    med = {k: round(float(np.median([p[k] for p in P])), 2) for k in P[0]}
    med['lap'] = round(float(np.median([l['lap_s'] for l in g])), 2)
    med['n'] = len(g)
    med['back_v'] = round(med['back_d'] / med['back_t'], 2)
    med['hp_v'] = round(med['hp_d'] / med['hp_t'], 2)
    med['lh_v'] = round(med['lh_d'] / med['lh_t'], 2)
    out[name] = med
    print('%-26s' % name, med)
json.dump(out, open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(sys.argv[1] if len(sys.argv) > 1 else '../sections.json')), 'sections_halves.json'), 'w'), ensure_ascii=False, indent=1)
