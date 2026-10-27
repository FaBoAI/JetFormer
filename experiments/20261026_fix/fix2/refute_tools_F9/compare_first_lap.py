"""tools:F9 の確かめ(読むだけ): collect.py の collisions_first_lap(発進 → A2)と standing_3lap の collisions_by_lap[0]
(AI 側 → 1 周目の線 = A2 の d_pre 手前)を同じ発進で並べ、A2 の d_pre 手前 → A2 の当たり(2 周目の LH など)を数える。"""
import json, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent / 'out'
tot = {'pairs': 0, 'diff': 0, 'extra_hits': 0}
for day in sys.argv[1:] or ['20261024', '20261023']:
    L = json.loads((OUT / ('launches_%s.json' % day)).read_text())['launches']
    S = json.loads((OUT / ('s3_%s.json' % day)).read_text())['results']
    sidx = {}
    for r in S:
        for s in r.get('standing') or []:
            sidx[(r['log'], round(s['t_ai'], 2))] = s
    for l in L:
        if l.get('to_r1_s') is None or 'collisions_first_lap' not in l:
            continue
        s = sidx.get((l['log'], round(l['t_ai'], 2)))
        if not s or not s.get('line_ends'):
            continue
        A = s.get('anchors') or []
        e1 = s['line_ends'][0]
        cfl = l['collisions_first_lap']
        in_line = [h for h in cfl if h[0] < e1]
        extra = [h for h in cfl if h[0] >= e1]
        by = s.get('collisions_by_lap')
        tot['pairs'] += 1
        if extra:
            tot['diff'] += 1
            tot['extra_hits'] += len(extra)
        print('%s t_ai=%7.2f A1=%s A2=%s line1=%7.2f d_pre=%s first_lap_end=%s | collect:%d (線まで %d・線→A2 %d %s) | s3 by_lap=%s' % (
            l['log'], l['t_ai'], A[0] if A else None, A[1] if len(A) > 1 else None, e1, s.get('d_pre_m'),
            l.get('first_lap_end_s'), len(cfl), len(in_line), len(extra), [[round(h[0], 2), round(h[1], 2), h[2]] for h in extra], by))
print(tot)
