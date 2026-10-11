from regime import *
def de_table(log, hum):
    if hum:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
    else:
        d, rows, _, tk, segs = analyze(log)
    an = [a for a, dep, hit in true_r1(d, r1_anchors(d), rows)]
    R = []
    for k, a in enumerate(an):
        b = an[k+1] if k+1 < len(an) else a + 10
        m = (d.it >= a) & (d.it < a + 3.5)
        h = d.hd[m] - np.interp(a, d.it, d.hd); tt = d.it[m]
        imin = np.argmin(h)
        if h[imin] > -90: continue
        j = np.nonzero(h[imin:] >= -25)[0]
        if not len(j): continue
        tg = tt[imin + j[0]]
        if not hum:
            mm = (d.at >= a - 0.5) & (d.at < tg + 1.6)
            if not d.sel[mm].all(): continue
            st = d.out[(d.at >= tg) & (d.at < tg + 0.8), 0] - 1443
            th = d.out[(d.at >= tg) & (d.at < tg + 0.8), 1] - 1498
        else:
            st = d.pwm[(d.vt >= tg) & (d.vt < tg + 0.8), 0] - 1443
            th = d.pwm[(d.vt >= tg) & (d.vt < tg + 0.8), 1] - 1498
        v = -d.erpm[(d.vt >= tg) & (d.vt < tg + 0.8)]
        hits = [r for r in rows if tg - 0.3 <= r['t'] < tg + 2.0]
        gm_hits = [r for r in rows if a + 0.9 <= r['t'] < tg - 0.3]
        yaw = (d.gyr[:,2]-d.bias)[(d.it >= tg) & (d.it < tg + 0.8)]
        R.append(dict(a=a, tg=tg, dt_gm=tg - a, v=float(np.mean(v)), vmax=float(np.max(v)), st=float(np.mean(st)), th=float(np.mean(th)),
                      yaw=float(np.mean(yaw)), r1min=float(h[imin]), hit=[(round(r['t'] - tg, 2), r['g']) for r in hits],
                      gmhit=[(round(r['t'] - a, 2), r['g']) for r in gm_hits]))
    return R
if __name__ == '__main__':
    import pickle
    allR = {}
    for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM]:
        R = de_table(log, hum); allR[log] = R
        print(log, len(R))
        for r in R:
            print('  %6.1f gmx+%.2f R1min %+4.0f v %5.0f vmax %5.0f st %+4.0f th %+4.0f yaw %+4.0f DE %s GM %s' % (r['a'], r['dt_gm'], r['r1min'], r['v'], r['vmax'], r['st'], r['th'], r['yaw'], r['hit'], r['gmhit']))
        for f in (True, False):
            s = [r for r in R if bool([h for h in r['hit'] if h[1] >= 4]) == f]
            if s: print('   strongDEhit=%s n=%d v med %.0f vmax med %.0f st med %+.0f yaw med %+.0f R1min med %+.0f dt_gm %.2f' % (f, len(s), np.median([r['v'] for r in s]), np.median([r['vmax'] for r in s]), np.median([r['st'] for r in s]), np.median([r['yaw'] for r in s]), np.median([r['r1min'] for r in s]), np.median([r['dt_gm'] for r in s])))
    pickle.dump(allR, open('de.pkl', 'wb'))
