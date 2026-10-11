from ld import *
import sys
log = sys.argv[1]; anchors = [float(x) for x in sys.argv[2].split(',')]; out = sys.argv[3]
d = load(log); d.heading()
offs = np.arange(-0.6, 0.81, 0.2)
paths, labs = [], []
for a in anchors:
    for o in offs:
        t = a + o; i = min(np.searchsorted(d.ct, t), len(d.ct)-1)
        paths.append(d.cf[i])
        st = d.at_t(d.at, d.out, t)
        labs.append('%.1f %+.1f\ns%+.0f t%+.0f e%.0f' % (a, o, st[0]-1443, st[1]-1498, -np.interp(t, d.vt, d.erpm)))
sheet(paths, labs, len(offs), out, 0.75)
