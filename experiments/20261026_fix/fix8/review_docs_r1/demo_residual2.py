"""反証 8-1 docs: approvals 14 の 2 行目・fix7/README 3 章の 2 行目(s4-c の残り「やり直して最大 0.2 秒の後退ぶん多く下がる」
「起きるのは、下がっている途中で後ろに当たる・引っかかるときなど」)を、自分で流した出力と照らす(読むだけ)。

元の出力(この場所の rerun/):
- rerun/exp10.py dip_skip on → rerun/out/exp10_dip_skip.json(閉じたループ。本物の drive._loop・偽物の ESC と車。今の ON = 両方 1。
  fix8/proposal/exp10.py の写しで一時の場所だけ変えた)
- rerun/exp10.py dip_norm on,off → rerun/out/exp10_dip_norm.json(ふつうの後退の途中の途切れ。ON・OFF)
- rerun/unit_real10_b2.py b2 → rerun/out/unit_real10_b2.json(本物の後退の eRPM の並び 132 を 'gap' を飛ばした後退に置く。今の assist.py)
"""
import collections
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REP = Path('/home/jetson/jetformer_reports/fix_20261026')


def snip(text, pat, before=0, after=200):
    i = text.find(pat)
    return text[max(0, i - before):i + len(pat) + after] if i >= 0 else '(無い)'


ap = (REP / 'fix3/approvals.md').read_text(encoding='utf-8')
f7 = (REP / 'fix7/README.md').read_text(encoding='utf-8')
print('== 文書')
print('approvals 14 の 2 行目:', snip(ap, '本当に後ろへ動き始めたのに', 0, 330))
print('fix7/README 3 章の 2 行目:', snip(f7, 's4-c の残り', 0, 200))

print()
print('== 閉じたループ dip_skip(今の ON。途切れ = 本当に下がっている途中の 1 サンプル(w=1)か 0.04 秒(w=2)の 0)')
d = json.loads((HERE / 'rerun/out/exp10_dip_skip.json').read_text(encoding='utf-8'))
cells = d['cells']
cl = {k[:-len(' clean')]: r for k, r in cells.items() if k.endswith('clean')}
chg, same_long = [], []
by_rd = collections.defaultdict(list)
for k, r in cells.items():
    if k.endswith('clean'):
        continue
    pre = next(x for x in cl if k.startswith(x + ' '))
    c, b = cl[pre]['on'], r['on']
    db = (b['back'] or 0) - (c['back'] or 0)
    if b['n_retry'] != c['n_retry']:
        chg.append(db)
        by_rd[pre.split()[0]].append((b['back'], b['esc_push'] or 0, len(b['rear_hits'] or []) > len(c['rear_hits'] or [])))
    elif db > 0.05:
        same_long.append(db)
print('組 %d(clean 8)。やり直し・early_none の数が clean と変わった組 %d: 下がる距離の差(今 − clean)%+.3f〜%+.3f m(中央 %+.3f)'
      '・clean より長く下がった組 %d・後ろの当たりが増えた組 %d' % (
          len(cells) - len(cl), len(chg), min(chg), max(chg), float(np.median(chg)), sum(x > 0.05 for x in chg),
          sum(x[2] for v in by_rd.values() for x in v)))
for rd, v in sorted(by_rd.items()):
    print('  %s(ESC が後退側から下がり始めるまで): 変わった組 %d・下がった距離 %.3f〜%.3f m・escape が前の壁を 0.05 秒より押した組 %d' % (
        rd, len(v), min(x[0] for x in v), max(x[0] for x in v), sum(1 for x in v if x[1] > 0.05)))
for pre, r in sorted(cl.items()):
    if pre.endswith('rear=None'):
        print('  clean %s: 下がった距離 %.3f m・前の壁を押した秒 %s' % (pre, r['on']['back'], r['on']['esc_push']))
print('決めは変わらず clean より長く下がった組 %d(+%.3f〜+%.3f m)= 途切れそのもので下がる時間がのびた形' % (
    len(same_long), min(same_long), max(same_long)))

print()
print('== 閉じたループ dip_norm(ふつうの後退の途中の途切れ。ON・OFF)')
d2 = json.loads((HERE / 'rerun/out/exp10_dip_norm.json').read_text(encoding='utf-8'))
cells2 = d2['cells']
cl2 = {k[:-len(' clean')]: r for k, r in cells2.items() if k.endswith('clean')}
for v in d2['variants']:
    longer = []
    for k, r in cells2.items():
        if k.endswith('clean'):
            continue
        pre = next(x for x in cl2 if k.startswith(x + ' '))
        c, b = cl2[pre][v], r[v]
        if b['n_retry'] == c['n_retry'] and (b['back'] or 0) - (c['back'] or 0) > 0.05:
            longer.append((b['back'] or 0) - (c['back'] or 0))
    print('  %s: 決めは変わらず clean より長く下がった組 %d / %d(+%.3f〜+%.3f m)' % (
        v, len(longer), len(cells2) - len(cl2), min(longer), max(longer)))

print()
print('== 本物の後退の eRPM の並びを、今の assist.py の \'gap\' を飛ばした後退に置く(unit_real10 の (b2))')
u = json.loads((HERE / 'rerun/out/unit_real10_b2.json').read_text(encoding='utf-8'))
b2 = u['b2']['base_on']
print('  並び %d・置き方 %d: 下がり始めたあとの early_none %d(並び %d)・ほか %s' % (
    u['n_backs'], sum(b2.values()), b2.get('early_none_after_onset', 0), u['b2_lost_files']['base_on'],
    {k: v for k, v in b2.items() if k != 'early_none_after_onset'}))
names = sorted(set(x[0] for x in u['b2_lost_examples']['base_on']))
print('  例:', '・'.join(names))
