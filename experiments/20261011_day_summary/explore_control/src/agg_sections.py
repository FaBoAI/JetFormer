#!/usr/bin/env python3
"""sections.json を人・AI のグループで集計して表示する(区間タイムの中央値、速度、R1 進入速度、区間ごとの衝突)"""
import json, sys
from collections import Counter
import numpy as np
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '../sections.json'))
L = [l for x in d['logs'] for l in x['laps'] if all(l['sec'][s] for s in ['R1', 'GM', 'DE', 'SW', 'HS', 'LH'])]
SECS = ['R1', 'GM', 'DE', 'SW', 'HS', 'LH']
out = {}
def grp(name, f):
    g = [l for l in L if f(l)]
    if not g:
        print(name, 'none'); return None
    lt = np.array([l['lap_s'] for l in g])
    row = {s: float(np.median([l['sec'][s]['t_s'] for l in g])) for s in SECS}
    vm = {s: float(np.median([l['sec'][s]['v_mean'] for l in g])) for s in SECS}
    vx = {s: float(np.median([l['sec'][s]['v_max'] for l in g])) for s in SECS}
    e = [l['r1_entry_erpm'] for l in g if l['r1_entry_erpm']]
    print(f"{name:26s} n={len(g):2d} lap med {np.median(lt):.2f} min {lt.min():.2f} sumsec {sum(row.values()):.2f} | " + ' '.join(f"{s}:{row[s]:.2f}" for s in SECS))
    print(' ' * 30 + 'v_mean ' + ' '.join(f"{s}:{vm[s]:.2f}" for s in SECS))
    print(' ' * 30 + 'v_max  ' + ' '.join(f"{s}:{vx[s]:.2f}" for s in SECS))
    print(' ' * 30 + 'R1 entry erpm med %.0f p90 %.0f; impacts/lap %.2f; vmax lap med %.2f' % (np.median(e), np.percentile(e, 90), np.mean([l['n_imp'] for l in g]), np.median([l['v_max_lap'] for l in g])))
    out[name] = {'n': len(g), 'lap_med': float(np.median(lt)), 'lap_min': float(lt.min()), 'sec_t_med': row, 'sec_v_mean': vm, 'sec_v_max': vx,
                 'r1_entry_med': float(np.median(e)), 'impacts_per_lap': float(np.mean([l['n_imp'] for l in g]))}
    return row
G = {
 'human_fast_clean(<=9.2s)': lambda l: l['driver'] == 'human' and l['n_imp'] == 0 and l['lap_s'] <= 9.2,
 'human_clean': lambda l: l['driver'] == 'human' and l['n_imp'] == 0,
 'human_all': lambda l: l['driver'] == 'human',
 'human_afternoon': lambda l: l['driver'] == 'human' and l['log'] < '20261010-19',
 'human_night': lambda l: l['driver'] == 'human' and l['log'] > '20261010-19',
 'ai_clean': lambda l: l['driver'] == 'ai' and l['n_imp'] == 0,
 'ai_all': lambda l: l['driver'] == 'ai',
 'ai_006': lambda l: l['who'] == 'ai_run_006',
 'ai_007': lambda l: l['who'] == 'ai_run_007',
}
rows = {k: grp(k, f) for k, f in G.items()}
for nm in ('human', 'ai'):
    c = Counter(s for l in L if l['driver'] == nm for s in l['impact_secs'])
    n = sum(1 for l in L if l['driver'] == nm)
    print(nm, 'laps', n, 'impacts by sec', dict(c))
    out['impacts_by_sec_' + nm] = {'laps': n, **dict(c)}
def delta(a, b):
    r = {s: round(rows[a][s] - rows[b][s], 2) for s in SECS}
    r['sum'] = round(sum(rows[a].values()) - sum(rows[b].values()), 2)
    print('delta', a, '-', b, r)
    out['delta|%s-%s' % (a, b)] = r
delta('ai_clean', 'human_fast_clean(<=9.2s)')
delta('ai_all', 'human_clean')
delta('ai_clean', 'human_clean')
json.dump(out, open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(sys.argv[1] if len(sys.argv) > 1 else '../sections.json')), 'sections_agg.json'), 'w'), ensure_ascii=False, indent=1)
