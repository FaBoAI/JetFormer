from ev import *
def wrap(a): return (a + 180) % 360 - 180
def course_psi(d, anchors):
    hk = np.array([np.interp(a, d.it, d.hd) for a in anchors])
    n = np.round((hk - hk[0]) / 360); r = hk - hk[0] - 360 * n
    ok = np.ones(len(r), bool)
    for _ in range(2):
        med = np.array([np.median(r[ok][max(0, i-3):i+4]) if ok.sum() else 0 for i in range(len(r))])
        # neighbor median computed over ok anchors near index i (approx)
        ok = np.abs(r - med) < 25
    ta = np.array(anchors)[ok]; ra = r[ok]
    drift = np.interp(d.it, ta, ra)
    psi = d.hd - hk[0] - drift
    return psi, (np.array(anchors), r, ok)
if __name__ == '__main__':
    for log in ['20261010-172910','20261010-173445']:
        d, rows, anchors, tk, segs = analyze(log)
        psi, (ta, r, ok) = course_psi(d, anchors)
        print(log, 'anchor residuals', np.round(r).astype(int), ok.astype(int))
        for rr in rows:
            p = np.interp(rr['t'], d.it, psi); pb = np.interp(rr['t']-1.0, d.it, psi)
            print('  %7.2f psi=%+5.0f  psi(-1s)=%+5.0f' % (rr['t'], wrap(p), wrap(pb)))
