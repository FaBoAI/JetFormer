from ld import *
import sys
log, t0, a, b, step, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
d = load(log); d.heading()
paths, labs = [], []
for o in np.arange(a, b + 1e-6, step):
    t = t0 + o; i = min(np.searchsorted(d.ct, t), len(d.ct) - 1)
    w = (d.it >= t - 0.017) & (d.it < t + 0.017)
    ax_ = d.acc[w, 0].mean() / G if w.any() else 0; ay_ = d.acc[w, 1].mean() / G if w.any() else 0
    st = d.at_t(d.at, d.out, t)
    paths.append(d.cf[i]); labs.append('%+.2fs s%+.0f e%.0f\nlat%+.1f lon%+.1f yaw%+.0f' % (o, st[0] - 1443, -np.interp(t, d.vt, d.erpm), ax_, ay_, np.interp(t, d.it, d.gyr[:, 2])))
sheet(paths, labs, 6, out)
