from destat import *
import pickle
P = {'AI': [], 'HU': []}
for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM]:
    if hum:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
    else:
        d, rows, _, tk, segs = analyze(log)
    for a, dep, h in true_r1(d, r1_anchors(d), rows):
        if not hum:
            m = (d.at >= a - 2.5) & (d.at < a + 0.5)
            if not d.sel[m].all(): continue
        vin = float(np.max(-d.erpm[(d.vt >= a - 0.4) & (d.vt < a + 0.15)]))
        g = max([x['g'] for x in h]) if h else 0.0
        P['HU' if hum else 'AI'].append((log, a, vin, g))
pickle.dump(P, open('r1pass.pkl', 'wb'))
for k, L in P.items():
    v = np.array([x[2] for x in L]); g = np.array([x[3] for x in L])
    for lo, hi in [(0, 5500), (5500, 6200), (6200, 99999)]:
        m = (v >= lo) & (v < hi)
        print('%s v_in %5d-%5d: n=%2d  hit>=4g %2d (%.0f%%)  any>=2.5g %2d' % (k, lo, hi, m.sum(), (g[m] >= 4).sum(), 100 * np.mean(g[m] >= 4) if m.any() else 0, (g[m] >= 2.5).sum()))
