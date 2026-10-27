"""exp8_hold・exp8_align の出力をまとめる(読むだけ)。fail = 最初のリカバリーで 0.1 m 未満しか下がらず escape が前の壁を
0.05 秒より押した。better/worse = on と off の下がった距離・fail を同じ場面で比べる。
    python -I an8.py out/exp8_*.json"""
import json
import sys
from collections import Counter


def fail(m):
    return (m['back'] or 0.0) < 0.1 and (m['esc_push'] or 0) > 0.05


def main(paths):
    for pth in paths:
        d = json.load(open(pth))
        c = Counter()
        ex = []
        for key, row in d['cells'].items():
            c['cells'] += 1
            vs = [v for v in ('on', 'off', 'f7b_on', 'on_nohold', 'on1') if v in row]
            for v in vs:
                if fail(row[v]):
                    c[v + '_fail'] += 1
                if row[v]['rear']:
                    c[v + '_rear'] += 1
            if fail(row['on']) and not fail(row['off']):
                c['on_fail_off_ok'] += 1
                if len(ex) < 8:
                    ex.append((key, row.get('glitch'), {v: (row[v]['back'], row[v]['esc_push'], row[v]['rev_dead'],
                                                           row[v]['all_guards'][:4]) for v in vs}))
            if fail(row['on']) and 'on_nohold' in row and not fail(row['on_nohold']):
                c['on_fail_nohold_ok'] += 1
            if fail(row['off']) and not fail(row['on']):
                c['off_fail_on_ok'] += 1
        print('==', pth, dict(sorted(c.items())))
        for x in ex:
            print('   ', x)


if __name__ == '__main__':
    main(sys.argv[1:])
