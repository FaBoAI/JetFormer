"""exp_grid の出力(複数)を名前ごとにまとめる(読むだけ)。
  window = 指摘の窓(d0 −0.06〜0 × L 0.45〜0.6)
  worse  = 値の止まりなし(ref)と同じ場面の keep 0 のどちらよりも +0.05 m を超えて下がった場面
  push   = escape が前の壁を押した場面 / front = 最初のリカバリーの間(escape の 0.6 秒あとまで)に前の壁に当たった場面
    python -I analyze.py out/grid_*_v5.json
"""
import json
import sys


def win(k):
    d0 = float(k.split()[0][3:])
    L = float(k.split()[1][2:])
    return -0.06 - 1e-9 <= d0 <= 1e-9 and 0.45 - 1e-9 <= L <= 0.6 + 1e-9


rows = []
for f in sys.argv[1:]:
    d = json.load(open(f))
    names = list(next(iter(d['grid'].values())).keys())
    ref = next(d['ref'][k]['back_m'] for k in ('now', 'final', 'fix6_before') if k in d['ref'])
    k0 = 'now/k0'
    for nm in names:
        r = {'file': f.split('/')[-1], 'name': nm, 'ref': ref, 'win_max': -9.0, 'win_push': 0, 'win_front': 0,
             'worse': 0, 'push': 0, 'front': 0, 'front_v': 0.0, 'min_back': 9.0}
        for k, row in d['grid'].items():
            m = row[nm]
            g = [x[0] for x in m['guards']]
            fh = [h[2] for h in m.get('front_hits', [])]
            if win(k):
                r['win_max'] = max(r['win_max'], m['back_m'] - ref)
                r['win_push'] += m['esc_push_s'] > 0
                r['win_front'] += bool(fh)
            if m['back_m'] > max(ref, row[k0]['back_m']) + 0.05:
                r['worse'] += 1
            r['push'] += m['esc_push_s'] > 0
            r['front'] += bool(fh)
            r['front_v'] = max([r['front_v']] + fh)
            if 'hold_long' not in g:
                r['min_back'] = min(r['min_back'], m['back_m'])
        rows.append(r)
cur = None
for r in rows:
    if r['file'] != cur:
        cur = r['file']
        print('## %s  ref %.3f' % (cur, r['ref']))
        print('   %-24s %8s %8s %9s %6s %5s %6s %7s %8s' % ('name', 'win_max', 'win_push', 'win_front', 'worse', 'push',
                                                         'front', 'front_v', 'min_back'))
    print('   %-24s %+8.3f %8d %9d %6d %5d %6d %7.2f %8.3f' % (r['name'], r['win_max'], r['win_push'], r['win_front'],
                                                           r['worse'], r['push'], r['front'], r['front_v'], r['min_back']))
