"""exp_grid の出力で、keep 1 の名前ごとに「値の止まりなし(ref)と同じ場面の keep 0 のどちらよりも +0.05 m を超えて下がった」
場面(worse)と、escape が前の壁を押した場面を数える。
    python -I cmp.py out/x.json name1,name2 k0name [only_window]
"""
import json
import sys

d = json.load(open(sys.argv[1]))
names = sys.argv[2].split(',')
k0 = sys.argv[3]
win = len(sys.argv) > 4 and sys.argv[4] == 'win'
for nm in names:
    ref = d['ref'][nm.partition('/')[0]]['back_m']
    worse, push, mx = [], [], (None, -9.0)
    for k, row in d['grid'].items():
        d0 = float(k.split()[0][3:])
        L = float(k.split()[1][2:])
        if win and not (-0.06 - 1e-9 <= d0 <= 1e-9 and 0.45 - 1e-9 <= L <= 0.6 + 1e-9):
            continue
        m, m0 = row[nm], row[k0]
        lim = max(ref, m0['back_m'])
        if m['back_m'] > lim + 0.05:
            worse.append((k, m['back_m'], m0['back_m']))
        if m['esc_push_s'] > 0:
            push.append((k, m['esc_push_s'], m['back_m']))
        if m['back_m'] - lim > mx[1]:
            mx = (k, round(m['back_m'] - lim, 3))
    print('== %s ref %.3f vs %s: worse %d push %d max(+) %s' % (nm, ref, k0, len(worse), len(push), mx))
    for w in worse:
        print('   worse', w)
    for w in push:
        print('   push', w)
