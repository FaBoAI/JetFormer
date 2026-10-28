"""票 A: sweep_A.json の、ESC の模型ごとの「距離 d ごとの ON 速い / OFF 速い / ON だけ / OFF だけ」(16 場面)と、
ON − OFF の速さの差の中央値。→ 標準出力・out/ana_A2.json"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TH = 0.1


def main():
    J = json.loads((HERE / 'out' / 'sweep_A.json').read_text())
    ds = J['ds']
    out = {}
    for esc in ('fit', 'fit_coast1.5', 'fit_lag0.15', 'fit_const', 'cl', 'human'):
        rows = [r for r in J['rows'] if r['esc'] == esc]
        tab = {}
        for d in ds:
            c = collections.Counter()
            dif = []
            for r in rows:
                a, b = r['on']['hit'][str(d)], r['off']['hit'][str(d)]
                if a is None and b is None:
                    c['-'] += 1
                elif b is None:
                    c['ON only'] += 1
                elif a is None:
                    c['OFF only'] += 1
                else:
                    dif.append(a - b)
                    c['ON+' if a > b + TH else ('OFF+' if b > a + TH else '=')] += 1
            dif.sort()
            tab[str(d)] = {'c': dict(c), 'med_on_minus_off': round(dif[len(dif) // 2], 3) if dif else None}
        out[esc] = tab
        print('==', esc, '(16 場面)')
        line = []
        for d in ds:
            if d > 1.6:
                break
            t = tab[str(d)]
            line.append('%.2f %s med%s' % (d, ','.join('%s%d' % (k, v) for k, v in sorted(t['c'].items())), t['med_on_minus_off']))
        for k in range(0, len(line), 4):
            print('   ' + ' | '.join(line[k:k + 4]))
    (HERE / 'out' / 'ana_A2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
