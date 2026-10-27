#!/usr/bin/env python3
"""直す 1 回目・2 組目: run_real.sh の出力(直す前 out_old と後 out_new)を比べる。読むだけ。
    python3 -I compare_real.py <run_real.sh の作業場所>"""
import difflib
import glob
import json
import os
import sys

W = sys.argv[1]
O, N = os.path.join(W, 'out_old'), os.path.join(W, 'out_new')
NEWG = ('standing_clean_le25', 'standing_clean_best_s', 'standing_best_coll')


def rd(p):
    return open(p, encoding='utf-8').read()


def strip_goal(md):
    out, ch = [], None
    for x in md.splitlines():
        if x.startswith('## '):
            ch = x[:5]
        if ch == '## 2.' and x.startswith('|'):
            c = x.split(' | ')
            x = ' | '.join(c[:-2] + c[-1:]) if not x.startswith('|---') else x[:-4]
        if x.startswith('- ぶつからずに 25 秒以下 = '):
            if out and out[-1] == '':
                out.pop()
            continue
        out.append(x)
    return '\n'.join(out) + '\n'


print('# 本物の記录 10/23・10/24: 直す前と後'.replace('录', '録'))
for d in ('20261023', '20261024'):
    for f in ('standing_%s.json' % d, 'standing_%s.md' % d, 'standing_%s.stdout' % d):
        print('%-34s %s' % (f, 'ビット単位で同じ' if rd(os.path.join(O, f)) == rd(os.path.join(N, f)) else '違う'))
for f in ('launches.json', 'launches.stdout'):
    print('%-34s %s' % ('collect 引数なし ' + f, 'ビット単位で同じ' if rd(os.path.join(O, f)) == rd(os.path.join(N, f)) else '違う'))
for d in ('20261023', '20261024'):
    a, b = rd(os.path.join(O, 'qp_%s/by_params.md' % d)), rd(os.path.join(N, 'qp_%s/by_params.md' % d))
    print('%-34s %s' % ('by_params.md %s' % d, '新しい列と説明の 1 行のほかは同じ' if strip_goal(b) == a else '違う(列のほかも)'))
    ja, jb = json.loads(rd(os.path.join(O, 'qp_%s/by_params.json' % d))), json.loads(rd(os.path.join(N, 'qp_%s/by_params.json' % d)))
    gb = [{k: v for k, v in g.items() if k not in NEWG} for g in jb['groups']]
    print('%-34s %s' % ('by_params.json %s' % d, '組の新しい 3 欄のほかは同じ' if (gb == ja['groups'] and ja['runs'] == jb['runs']
                                                          and ja['results'] == jb['results']) else '違う'))
    ta, tb = rd(os.path.join(O, 'runs_%s.stdout' % d)).strip().splitlines(), rd(os.path.join(N, 'runs_%s.stdout' % d)).strip().splitlines()
    ra, rb = json.loads(ta[-1]), json.loads(tb[-1])
    same = ta[:-1] == tb[:-1] and [{k: v for k, v in r.items() if k != 'options'} for r in rb] == ra
    print('%-34s %s' % ('runs_1103 %s' % d, '表は同じ・JSON は options の欄だけ足した' if same else '違う'))
diffs = []
n = 0
for f in sorted(glob.glob(os.path.join(O, 'car_check', '*.json'))):
    a, b = json.loads(rd(f)), json.loads(rd(f.replace(O, N)))
    for x in (a, b):
        x.pop('created_at', None)
        x.pop('seconds', None)
    n += 1
    if a != b:
        diffs.append((os.path.basename(f)[:-5], a['judge']['code'], b['judge']['code'],
                      [i['text'] for i in b['judge']['items'] if i['item'] == 'r1_bump']))
print('car_check %d 本: 違うのは %d 本' % (n, len(diffs)))
for x in diffs:
    print('   %s 判定の番号 %s → %s: %s' % x)
for d in ('20261023', '20261024', '20261024_first'):
    a, b = json.loads(rd(os.path.join(O, 'launches_%s.json' % d))), json.loads(rd(os.path.join(N, 'launches_%s.json' % d)))
    A = {(r['log'], r['t_ai']): r for r in a['launches']}
    B = {(r['log'], r['t_ai']): r for r in b['launches']}
    added, changed = {}, {}
    for k, ra in A.items():
        rb = B[k]
        for f in rb:
            if f not in ra:
                added[f] = added.get(f, 0) + 1
            elif ra[f] != rb[f]:
                changed[f] = changed.get(f, 0) + 1
        for f in ra:
            if f not in rb:
                changed['(消えた) ' + f] = changed.get('(消えた) ' + f, 0) + 1
    print('collect --day %s: 発進 %d → %d・足した欄 %s・値の変わった欄 %s' % (d, len(A), len(B), added, changed))
    sa = rd(os.path.join(O, 'launches_%s.stdout' % d)).split('発進の前進の比べ', 1)[1].splitlines()
    sb = rd(os.path.join(N, 'launches_%s.stdout' % d)).split('発進の前進の比べ', 1)[1].splitlines()
    for x in difflib.unified_diff([y for y in sa if not y.startswith('出力')], [y for y in sb if not y.startswith('出力')],
                                  lineterm='', n=0):
        if not x.startswith(('---', '+++', '@@')):
            print('   ' + x[:260])
