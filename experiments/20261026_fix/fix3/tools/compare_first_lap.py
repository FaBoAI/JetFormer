#!/usr/bin/env python3
"""係 tools(fix3、tools:F9): 本物の記録で、collect.py --day の「最初の周」の終わり(first_lap_end_s)と
standing_3lap の 1 周目の線(line_ends[0])を発進ごとに比べる。読むだけ。
    python3 compare_first_lap.py launches_<日>.json standing_<日>.json"""
import json
import sys

L = json.load(open(sys.argv[1], encoding='utf-8'))['launches']
S = json.load(open(sys.argv[2], encoding='utf-8'))['results']
st = {}
for r in S:
    for s in r.get('standing') or []:
        st[(r['log'], round(s['t_ai'], 2))] = s
n = same = 0
worst = []
for r in L:
    if r.get('first_lap_end') is None:
        continue
    s = st.get((r['log'], round(r['t_ai'], 2)))
    le = (s or {}).get('line_ends') or []
    line = 'line' if r['first_lap_end'] == 'line' else r['first_lap_end']
    if not le:
        print('%s %.2f %-6s collect %.3f  standing: 線なし(%s)  A2 %s  d=%s' % (r['log'], r['t_ai'], line, r['first_lap_end_s'],
              (s or {}).get('why'), r.get('first_lap_a2_s'), r.get('dist_m')))
        continue
    n += 1
    dd = r['first_lap_end_s'] - le[0]
    same += abs(dd) <= 0.02
    worst.append((abs(dd), r['log'], r['t_ai'], r['first_lap_end_s'], le[0]))
    print('%s %.2f %-6s collect %.3f standing %.3f 差 %+.3f  A2 %s  first_lap_hits %d' % (r['log'], r['t_ai'], line, r['first_lap_end_s'],
          le[0], dd, r.get('first_lap_a2_s'), len(r.get('collisions_first_lap') or [])))
print('比べた %d・±0.02 秒で同じ %d・最大の差 %s' % (n, same, max(worst)[0] if worst else None))
