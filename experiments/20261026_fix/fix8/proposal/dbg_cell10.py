"""fix8 係 safety-proposal: exp10 の 1 組をたどる(段の移り・書いた値・車の位置)。python -I dbg_cell10.py <kind> <key の一部> [形,...]"""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV9_TAG', 'dbg')
sys.path.insert(0, str(HERE))
import h10  # noqa: E402
import exp10  # noqa: E402

kind, pat = sys.argv[1], sys.argv[2]
vs = (sys.argv[3] if len(sys.argv) > 3 else 'on,pA,pB,pC').split(',')
try:
    for key, sc, kw, tag in exp10.cells(kind):
        if pat not in key:
            continue
        print('==', key)
        p = h10.base_params(**kw)
        for v in vs:
            name, f, force, patch = exp10.VARIANTS[v]
            res = h10.run(name, f(p), sc, force=force, patch=patch)
            m = exp10.summ(res)
            sim, rec, pca, ph = res
            tb = m['t_brake']
            tr = [(e[0], e[1], e[2], e[6], e[7]) for e in ph if tb - 0.05 <= e[0] <= tb + 3.0 and e[1] != e[2]]
            print(' ', v, {k: m[k] for k in ('back', 'esc_push', 'rear_hits', 'max_back_v', 't_back_end', 't_to_escape',
                                             'rear_push', 'n_retry', 'all_guards', 'rev_on_t')})
            print('    段:', ' '.join('%.3f %s→%s(b%.2f e%d)' % x for x in tr))
            xs = [(t, x, v_, th, ro) for t, x, v_, th, con, ro, sl in sim.log if tb <= t <= tb + 2.6]
            print('    位置:', ' '.join('%.2f:%.2f/%.2f/%d%s' % (t, x, v_, th - h10.N, '*' if ro else '') for t, x, v_, th, ro in xs[::5]))
        break
finally:
    h10.cleanup()
