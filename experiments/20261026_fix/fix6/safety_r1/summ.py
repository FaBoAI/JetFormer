"""exp_grid の出力を表にする: 値の止まりなし(ref)より +0.05 m を超えて下がった・escape が前の壁を押した・中止 の数と一覧
    python -I summ.py out/grid_rev_0.4_300.json [names]
"""
import json
import sys

d = json.load(open(sys.argv[1]))
names = sys.argv[2].split(',') if len(sys.argv) > 2 else [k for k in d['ref'] if not k.endswith('_keep0')]
print('kind', d['kind'], 'rev_s', d['rev_s'], 'rev_us', d['rev_us'], 't_on', d.get('t_on'))
for nm in names:
    ref = d['ref'][nm]['back_m']
    over, push, gu, under, mx = [], [], [], [], (None, -9)
    for k, row in d['grid'].items():
        m = row.get(nm)
        if m is None:
            continue
        diff = m['back_m'] - ref
        g = [x[0] for x in m['guards']]
        if 'hold_long' in g:
            gu.append(k)
        if diff > 0.05 + 1e-9:
            over.append((k, round(diff, 3)))
        if m['esc_push_s'] > 0:
            push.append((k, m['esc_push_s'], m['back_m']))
        if diff < -0.4 and 'hold_long' not in g:
            under.append((k, m['back_m']))
        if diff > mx[1]:
            mx = (k, round(diff, 3))
    print('== %s ref %.3f: over+0.05 %d, esc_push %d, hold_long %d, under-0.4 %d, max diff %s' % (
        nm, ref, len(over), len(push), len(gu), len(under), mx))
    print('  over', over)
    print('  push', push)
    print('  under', under[:40])
