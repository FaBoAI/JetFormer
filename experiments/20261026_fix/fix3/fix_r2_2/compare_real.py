#!/usr/bin/env python3
"""直す 2 回目・2 組目: run_real.sh の出力(直す前 out_old と後 out_new)を比べる。読むだけ。
    python3 -I compare_real.py <run_real.sh の作業場所>"""
import glob
import json
import os
import sys

W = sys.argv[1]
O, N = os.path.join(W, 'out_old'), os.path.join(W, 'out_new')


def rd(p):
    return open(p, encoding='utf-8').read()


def rows_diff(a, b):
    """発進の行の違い: {欄: 本数}"""
    out = {}
    for x, y in zip(a, b):
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k):
                out[k] = out.get(k, 0) + 1
    return out


print('# 本物の記録 10/23・10/24: 直す前と後(直す 2 回目・2 組目)')
for f in ('launches.json', 'launches.stdout'):
    print('%-44s %s' % ('collect 引数なし ' + f, 'ビット単位で同じ' if rd(os.path.join(O, f)) == rd(os.path.join(N, f)) else '違う'))
for d in ('20261023', '20261024'):
    for n in ('plain', 'first-only', 'ignore-gain'):
        ja, jb = json.loads(rd(os.path.join(O, 'launches_%s_%s.json' % (d, n)))), json.loads(rd(os.path.join(N, 'launches_%s_%s.json' % (d, n))))
        assert len(ja['launches']) == len(jb['launches'])
        rd_ = rows_diff(ja['launches'], jb['launches'])
        ai_end = [(r['log'], r['t_ai']) for r in jb['launches'] if r.get('first_lap_end') == 'ai_end']
        moved = []
        for x, y in zip(ja['launches'], jb['launches']):
            if x.get('collisions_first_lap') != y.get('collisions_first_lap'):
                moved.append('%s@%.2f %s %s→%s' % (y['log'], y['t_ai'], y.get('first_lap_end'), len(x.get('collisions_first_lap') or []),
                                                   len(y.get('collisions_first_lap') or [])))
        sa, sb = ja['summary'], jb['summary']
        sdiff = []
        for ga, gb in zip(sa, sb):
            for k in sorted(set(ga) | set(gb)):
                if ga.get(k) != gb.get(k):
                    sdiff.append('%s %s.%s' % (ga['session'], ga['gain'], k))
        oa = [x for x in rd(os.path.join(O, 'launches_%s_%s.stdout' % (d, n))).splitlines() if not x.startswith('出力 ')]
        ob = [x for x in rd(os.path.join(N, 'launches_%s_%s.stdout' % (d, n))).splitlines() if not x.startswith('出力 ')]
        extra_keys = sorted(set(jb) - set(ja))
        print('collect --day %s %-11s 行 %d: 違う欄 %s。first_lap_end=ai_end %d 本。最初の周の衝突が変わった %s。まとめの違い %s(組 %d→%d)。'
              '標準出力の違う行 %d。JSON の新しい欄 %s'
              % (d, n, len(jb['launches']), rd_, len(ai_end), moved or 'なし', sdiff or 'なし', len(sa), len(sb),
                 sum(1 for x, y in zip(oa, ob) if x != y) + abs(len(oa) - len(ob)), extra_keys or 'なし'))
        if n == 'plain':
            for x, y in zip(oa, ob):
                if x != y:
                    print('    前: %s\n    今: %s' % (x[:220], y[:220]))
# car_check
diffs, n = [], 0
for f in sorted(glob.glob(os.path.join(O, 'car_check', '*.json'))):
    a, b = json.loads(rd(f)), json.loads(rd(f.replace(O, N)))
    for x in (a, b):
        x.pop('created_at', None)
        x.pop('seconds', None)
    n += 1
    if a != b:
        ks = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        ta = [x for x in rd(f.replace('.json', '.txt')).splitlines() if not x.startswith('  (')]
        tb = [x for x in rd(f.replace(O, N).replace('.json', '.txt')).splitlines() if not x.startswith('  (')]
        dl = [(x, y) for x, y in zip(ta, tb) if x != y]
        diffs.append((os.path.basename(f)[:-5], ks, a['judge'] == b['judge'], (b.get('collisions') or {}).get('after_ai'), dl))
print('car_check %d 本: 違う本 %d' % (n, len(diffs)))
for lg, ks, jsame, aa, dl in diffs:
    print('  %s 違う欄 %s・判定 %s・人側にした直後 %s' % (lg, ks, '同じ' if jsame else '違う', aa))
    for x, y in dl:
        print('    前: %s\n    今: %s' % (x[:240], y[:240]))
rc_a, rc_b = rd(os.path.join(O, 'car_check', 'rc.txt')), rd(os.path.join(N, 'car_check', 'rc.txt'))
print('car_check の終わりの番号(rc)', '全部同じ' if rc_a == rc_b else '違う')
