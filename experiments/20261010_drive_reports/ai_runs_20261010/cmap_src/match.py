from hum2 import *
REF_A, REF_B = 63.62, 73.40     # reference lap (006): R1 start .. next R1 start
SECT = [(0.00, 'R1'), (0.11, 'GM'), (0.24, 'DE'), (0.41, 'SW'), (0.70, 'HS'), (0.85, 'LH')]
def section(ph):
    s = SECT[0][1]
    for a, n in SECT:
        if ph >= a: s = n
    return s
_ref = {}
def ref():
    if not _ref:
        d, rows, anchors, tk, segs = analyze('20261010-172910')
        psi, _ = course_psi(d, anchors)
        _ref.update(d=d, psi=psi)
    return _ref['d'], _ref['psi']
def match(d, psi, t, win=(1.6, 0.1), step=0.05, tsa=None):
    dr, pr = ref()
    ks = np.arange(win[0], win[1]-1e-9, -step)
    pe = np.interp(t - ks, d.it, psi)
    taus = np.arange(REF_A, REF_B, 0.02)
    if tsa is not None and tsa < 3.0:
        taus = np.arange(REF_A, REF_A + 0.36 * (REF_B - REF_A), 0.02)
    best = (1e9, None, None)
    for tau in taus:
        pf = np.interp(tau - ks, dr.it, pr)
        diff = wrap(pe - pf); c = np.clip(np.median(diff), -30, 30)
        cost = np.mean(np.abs(wrap(diff - c))) + 0.3 * abs(c)
        if cost < best[0]: best = (cost, tau, c)
    cost, tau, c = best
    ph = (tau - REF_A) / (REF_B - REF_A)
    # predicted position ahead: events occur at t, history ends t-0.1 -> add 0.1 s to tau
    ph = ((tau + win[1] - REF_A) / (REF_B - REF_A)) % 1.0
    return ph, cost, c
if __name__ == '__main__':
    import json
    res = {}
    for log in ['20261010-172910','20261010-173445']:
        d, rows, anchors, tk, segs = analyze(log); psi, _ = course_psi(d, anchors)
        for r in rows:
            ph, cost, c = match(d, psi, r['t'], tsa=r['t_since_R1']); r.update(phase=round(float(ph),3), cost=round(float(cost),1), off=round(float(c)), sect=section(ph))
            print(log[-6:], '%7.2f ph %.2f %s cost %4.1f off %+3.0f g %.1f' % (r['t'], ph, r['sect'], cost, c, r['g']))
        res[log] = rows
    for log in HUM:
        d, rows, anchors, segs = analyze_h(log)
        if len(anchors) < 2:
            print(log, 'too few anchors'); continue
        psi, _ = course_psi(d, anchors)
        for r in rows:
            ph, cost, c = match(d, psi, r['t'], tsa=r['t_since_R1']); r.update(phase=round(float(ph),3), cost=round(float(cost),1), off=round(float(c)), sect=section(ph))
            print(log[-6:], '%7.2f ph %.2f %s cost %4.1f off %+3.0f g %.1f' % (r['t'], ph, r['sect'], cost, c, r['g']))
        res[log] = rows
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        raise TypeError(type(o))
    json.dump(res, open('events_all.json', 'w'), indent=1, default=conv)
