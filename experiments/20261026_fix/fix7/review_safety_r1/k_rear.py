"""fuzz_onoff で ON が OFF・ON の止まりなしより速く後ろに当たった場面を作り直し、当たりが「ふつうの下がる量の中」か
(その手順の 'brake' の始めの位置から後ろの物まで ≤ 1.50 m = 止まりなしの位相の最大 1.452 m + 0.05)、「ふつうより多く下がった」か
(> 1.50 m)に分ける。ふつうより多く下がった場面は、同じ場面の fix5 のあと(f6b)・fix6 r1 のあと(f6r2b)の ON とも比べる。読むだけ。
  python -I k_rear.py out/fuzz_onoff_7101.json out/fuzz_onoff_7102.json   → out/k_rear.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
import fuzz_onoff as FZ  # noqa: E402
from replay_fz import scene_of  # noqa: E402

LIM = 1.50


def classify(res):
    sim, rec, pca, ph = res
    starts = [t for t, p0, p1, *_ in ph if p0 is None and p1 in ('brake', 'giveup')]
    out = []
    for h in sim.car.hits:
        if h[1] != 'rear':
            continue
        st = [t for t in starts if t <= h[0]]
        if not st:
            out.append({'hit': h, 'start': None})
            continue
        t0 = st[-1]
        x0 = next(x for t, x, *_ in sim.log if t >= t0 - 1e-9)
        dist = x0 - sim.s['xr']
        kept = [(t, p0, p1, hr) for t, p0, p1, hs, hr, es, bs, e in ph if hs and t0 <= t <= h[0] and p0 is not None]
        out.append({'hit': h, 'start': t0, 'x0': round(x0, 3), 'dist_to_rear': round(dist, 3), 'over': dist > LIM,
                    'kept_holds': kept[:6]})
    return out


def main(paths):
    res = {'cases': []}
    try:
        for pth in paths:
            d = json.load(open(pth))
            for r in d['runs']:
                on, off, onc = r['on'], r['off'], r['onc']
                if not (on['rear_max_v'] > off['rear_max_v'] + 0.05 and on['rear_max_v'] > onc['rear_max_v'] + 0.05):
                    continue
                sc, p = scene_of(d['seed'], r['i'])
                c = {'tag': '%d#%d' % (d['seed'], r['i']), 'on': classify(rv7.run('now', rv7.ON(p), sc))}
                if any(x.get('over') for x in c['on']):
                    for v in ('f6b', 'f6r2b'):
                        rr = rv7.run(v, rv7.ON(p), sc)
                        c['on_' + v] = [h for h in rr[0].car.hits if h[1] == 'rear']
                res['cases'].append(c)
                print(c['tag'], [(x['hit'], x.get('dist_to_rear'), x.get('over'), x.get('kept_holds')) for x in c['on']],
                      {k: v for k, v in c.items() if k.startswith('on_')}, flush=True)
    finally:
        rv7.cleanup()
    (HERE / 'out' / 'k_rear.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1:])
