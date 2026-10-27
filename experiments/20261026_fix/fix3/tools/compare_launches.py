#!/usr/bin/env python3
"""係 tools(fix3): collect.py --day の出力(launches_<日>.json)を直す前と後で比べる。読むだけ。
    python3 compare_launches.py BEFORE.json AFTER.json
欄ごとに: 同じ / 違う(値)/ 足した。発進の行は (log, t_ai) で合わせる。summary は丸ごと比べる"""
import json
import sys

a, b = (json.load(open(p, encoding='utf-8')) for p in sys.argv[1:3])
print('top keys same' if list(a) == list(b) else 'top keys: %s → %s' % (list(a), list(b)))
for k in a:
    if k in ('launches',):
        continue
    print('%-20s %s' % (k, 'SAME' if a[k] == b.get(k) else 'DIFF'))
A = {(r['log'], r['t_ai']): r for r in a['launches']}
B = {(r['log'], r['t_ai']): r for r in b['launches']}
print('launches', len(A), len(B), 'same keys' if set(A) == set(B) else 'DIFF rows')
added, changed = {}, {}
for key, ra in A.items():
    rb = B[key]
    for f in rb:
        if f not in ra:
            added[f] = added.get(f, 0) + 1
        elif ra[f] != rb[f]:
            changed.setdefault(f, []).append((key[0], ra[f], rb[f]))
    for f in ra:
        if f not in rb:
            print('removed', f, key)
print('added fields:', added)
for f, xs in changed.items():
    print('changed %s: %d rows' % (f, len(xs)))
    for x in xs[:40]:
        print('   ', x)
