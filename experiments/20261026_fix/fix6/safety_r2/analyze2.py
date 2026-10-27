"""grid2.py の出力(out/grid2_<grid>[_<part>].json)をまとめる(読むだけ)。
各候補について、最初のリカバリーで
  fail   = 下がった距離 < 0.1 m で escape が前の壁を押した(指摘の「0 m・壁押し」)
  push   = escape が前の壁を押した場面・front = 最初のリカバリーの間に前の壁に当たった場面
  more   = 下がった距離が「値の止まりなし・同じ場面の直す前の写し(fix5 のあと)・keep 0」の多い方より +0.05 m を超えた場面(後ろの当たりの見込み)
  over   = 下がった距離が「値の止まりなし・同じ場面の keep 0」の多い方より +0.05 m を超えた場面(fix6 r1 の決まり。overmax = その最大)
  short  = 下がった距離が 0.5 m より短い場面
  less   = 直す前の写しより 0.05 m を超えて短い場面・lessmax = その最大の差
  dead   = 後退側を書いているのに止まっていた秒の最大
    python -I analyze2.py [grid ...]
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(prefix, grid):
    fs = sorted(glob.glob(str(HERE / 'out' / ('%s_%s.json' % (prefix, grid))))) or \
        sorted(glob.glob(str(HERE / 'out' / ('%s_%s_[0-9].json' % (prefix, grid)))))
    if not fs:
        return None
    out = None
    for f in fs:
        d = json.loads(Path(f).read_text())
        if out is None:
            out = d
        else:
            out['cells'].update(d['cells'])
    return out


def load(grid):
    """1 回目の組(grid2_*)に、2 回目の組(grid2b_*・grid2c_*。候補だけ)の候補を場面の鍵で足す"""
    out = _load('grid2', grid)
    if out is None:
        return None
    for pre in ('grid2b', 'grid2c'):
        d = _load(pre, grid)
        if d is None:
            continue
        for v in d['variants']:
            if v not in out['variants']:
                out['variants'].append(v)
        for k, row in d['cells'].items():
            out['cells'].setdefault(k, {}).update(row)
    out['cells'] = {k: r for k, r in out['cells'].items() if all(v in r for v in out['variants'])}
    return out


def summ(d):
    ng = d['ng']['back']
    res = {}
    for v in d['variants']:
        c = {'n': 0, 'fail': 0, 'push': 0, 'front': 0, 'more': 0, 'moremax': 0.0, 'over': 0, 'overmax': 0.0, 'less': 0,
             'lessmax': 0.0, 'same_before': 0, 'short': 0, 'dead': 0.0, 'none': 0}
        for key, row in d['cells'].items():
            m, b = row[v], row['before']
            if m is None:
                c['none'] += 1
                continue
            c['n'] += 1
            c['fail'] += m['back'] < 0.1 and m['push'] > 0
            c['push'] += m['push'] > 0
            c['front'] += bool(m['front'])
            refs = [ng] + [x['back'] for x in (b, row.get('k0')) if x is not None]
            over = m['back'] - max(refs)
            if over > 0.05:
                c['more'] += 1
                c['moremax'] = max(c['moremax'], round(over, 3))
            refs2 = [ng] + [x['back'] for x in (row.get('k0'),) if x is not None]
            over2 = m['back'] - max(refs2)
            if over2 > 0.05:
                c['over'] += 1
                c['overmax'] = max(c['overmax'], round(over2, 3))
            c['short'] += m['back'] < 0.5
            if b is not None:
                diff = b['back'] - m['back']
                if diff > 0.05:
                    c['less'] += 1
                    c['lessmax'] = max(c['lessmax'], round(diff, 3))
                c['same_before'] += abs(diff) <= 0.005
            c['dead'] = max(c['dead'], m['dead'])
        res[v] = c
    return res


def main(grids):
    if not grids:
        grids = sorted({Path(f).stem[6:].rsplit('_', 1)[0] if Path(f).stem[-2] == '_' and Path(f).stem[-1].isdigit()
                        else Path(f).stem[6:] for f in glob.glob(str(HERE / 'out' / 'grid2_*.json'))})
    allres = {}
    for g in grids:
        d = load(g)
        if d is None:
            continue
        r = summ(d)
        allres[g] = {'ng': d['ng']['back'], 'cells': len(d['cells']), 'res': r}
        print('==', g, 'cells', len(d['cells']), 'ng back', d['ng']['back'])
        for v, c in r.items():
            print('  %-24s' % v, ' '.join('%s=%s' % kv for kv in c.items()))
    (HERE / 'out' / 'analyze2.json').write_text(json.dumps(allres, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
