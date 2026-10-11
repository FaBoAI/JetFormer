from destat import *
from psi import course_psi
import pickle
allR = pickle.load(open('de.pkl', 'rb'))
for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM]:
    if hum:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
    else:
        d, rows, _, tk, segs = analyze(log)
    an = [a for a, dep, hit in true_r1(d, r1_anchors(d), rows)]
    psi, _ = course_psi(d, an)
    gz = d.gyr[:, 2] - d.bias
    H, C = [], []
    for r in allR[log]:
        tg = r['tg']; strong = any(h[1] >= 4 for h in r['hit'])
        p6 = wrap(np.interp(tg + 0.6, d.it, psi)); y = float(np.mean(gz[(d.it >= tg + 0.2) & (d.it < tg + 0.8)]))
        if not hum:
            st = float(np.mean(d.out[(d.at >= tg + 0.2) & (d.at < tg + 0.8), 0] - 1443)); stmin = float(np.min(d.out[(d.at >= tg - 0.6) & (d.at < tg + 0.6), 0] - 1443))
        else:
            st = float(np.mean(d.pwm[(d.vt >= tg + 0.2) & (d.vt < tg + 0.8), 0] - 1443)); stmin = float(np.min(d.pwm[(d.vt >= tg - 0.6) & (d.vt < tg + 0.6), 0] - 1443))
        v = float(np.mean(-d.erpm[(d.vt >= tg + 0.4) & (d.vt < tg + 0.9)]))
        (H if strong else C).append((p6, y, st, stmin, v))
    def s(L, i): return '%+.0f' % np.median([x[i] for x in L]) if L else '-'
    print('%s DE strong-hit n=%d psi(+0.6) %s yaw %s steer %s minSteer(R2) %s v %s || clean n=%d psi %s yaw %s steer %s minSteer %s v %s' % (log[-6:], len(H), s(H,0), s(H,1), s(H,2), s(H,3), s(H,4), len(C), s(C,0), s(C,1), s(C,2), s(C,3), s(C,4)))
