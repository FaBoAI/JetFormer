from destat import *
from psi import course_psi
import pickle
allR = pickle.load(open('de.pkl', 'rb'))
P = {'AI': [], 'HU': []}
for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM]:
    if hum:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
    else:
        d, rows, _, tk, segs = analyze(log)
    an = [a for a, dep, hit in true_r1(d, r1_anchors(d), rows)]
    psi, _ = course_psi(d, an)
    gz = d.gyr[:, 2] - d.bias
    for r in allR[log]:
        tg = r['tg']
        v = float(np.mean(-d.erpm[(d.vt >= tg + 0.4) & (d.vt < tg + 0.9)]))
        y = float(np.mean(gz[(d.it >= tg + 0.2) & (d.it < tg + 0.8)]))
        g = max([h[1] for h in r['hit']]) if r['hit'] else 0.0
        P['HU' if hum else 'AI'].append((log, r['a'], v, y, wrap(np.interp(tg + 0.6, d.it, psi)), g))
pickle.dump(P, open('depass.pkl', 'wb'))
for k, L in P.items():
    v = np.array([x[2] for x in L]); y = np.array([x[3] for x in L]); g = np.array([x[5] for x in L])
    for lo, hi in [(0, 5000), (5000, 6000), (6000, 99999)]:
        m = (v >= lo) & (v < hi)
        print('%s v %5d-%5d: n=%2d hit>=4g %2d (%.0f%%)' % (k, lo, hi, m.sum(), (g[m] >= 4).sum(), 100 * np.mean(g[m] >= 4) if m.any() else 0))
    for lo, hi in [(-999, -30), (-30, 0), (0, 999)]:
        m = (y >= lo) & (y < hi)
        print('%s yaw %4d..%4d: n=%2d hit>=4g %2d (%.0f%%)' % (k, lo, hi, m.sum(), (g[m] >= 4).sum(), 100 * np.mean(g[m] >= 4) if m.any() else 0))
    print(k, 'median v %.0f yaw %+.0f' % (np.median(v), np.median(y)))
