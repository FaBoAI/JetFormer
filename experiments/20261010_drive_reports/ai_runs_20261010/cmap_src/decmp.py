from destat import *
import pickle, sys
allR = pickle.load(open('de.pkl', 'rb'))
def rowsheet(log, picks, out, offs=(0.4, 0.7, 0.9, 1.1)):
    d = load(log) if log.startswith('20261010-17') else D(log)
    paths, labs = [], []
    for r in allR[log]:
        if not any(abs(r['a'] - p) < 0.3 for p in picks): continue
        hit = [h for h in r['hit'] if h[1] >= 4]
        for o in offs:
            t = r['tg'] + o; i = min(np.searchsorted(d.ct, t), len(d.ct) - 1)
            paths.append(d.cf[i]); labs.append('%s %.0f %s\nGMexit%+.1fs e%.0f' % (log[-6:], r['a'], 'HIT@%.1f' % hit[0][0] if hit else 'clean', o, -np.interp(t, d.vt, d.erpm)))
    sheet(paths, labs, len(offs), out, 0.8)
log = sys.argv[1]; picks = [float(x) for x in sys.argv[2].split(',')]; rowsheet(log, picks, sys.argv[3])
