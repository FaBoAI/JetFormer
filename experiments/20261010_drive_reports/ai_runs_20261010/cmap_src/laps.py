from match import *
def r1_anchors(d):
    segs, gs = rsegs(d)
    out = []
    for (a, b, dh, mn) in segs:
        if mn > -120 or dh > -40: continue
        # preceded by a left turn > 90 deg within the prior 2.0 s
        h_prev = np.interp(a, d.it, d.hd) - np.interp(a - 2.0, d.it, d.hd)
        m = (d.it >= a - 3.0) & (d.it < a - 0.6)
        hw = d.hd[m]; drop = np.min(hw - np.maximum.accumulate(hw)) if m.any() else 0
        if drop < -60:
            continue
        if h_prev > 90 or dh < -90:
            if out and a - out[-1] < 5.0: continue
            out.append(a)
    return out
def lap_table(log, human=False):
    if human:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
    else:
        d, rows, _, tk, segs = analyze(log)
    an = r1_anchors(d)
    psi, _ = course_psi(d, an)
    gz = np.convolve(d.gyr[:,2]-d.bias, np.ones(11)/11, 'same')
    sel_t = d.at if not human else None
    laps = []
    for k in range(len(an) - 1):
        a, b = an[k], an[k+1]
        L = dict(t0=a, dur=b-a)
        if not human:
            m = (d.at >= a) & (d.at < b); L['ai'] = bool(d.sel[m].all())
        else:
            L['ai'] = False
        def er(x, y): 
            m = (d.vt >= a + x) & (d.vt < a + y); return float(np.mean(-d.erpm[m])) if m.any() else np.nan
        def erx(x, y):
            m = (d.vt >= a + x) & (d.vt < a + y); return float(np.max(-d.erpm[m])) if m.any() else np.nan
        L['v_hs'] = er(-2.5, -1.3)       # home straight end
        L['v_lh'] = er(-1.0, 0.0)        # left hairpin
        L['v_r1'] = er(0.0, 0.6)
        L['v_de'] = erx(2.4, 4.2)
        m = (d.it >= a) & (d.it < a + 1.5)
        L['r1_depth'] = float(np.min(psi[m] - np.interp(a, d.it, psi)))
        L['yaw_r1_min'] = float(gz[m].min())
        st = (lambda arr, x, y: arr[(d.at >= a + x) & (d.at < a + y)]) if not human else None
        if not human:
            o = d.out[(d.at >= a) & (d.at < a + 0.8), 0] - 1443
            L['r1_sat'] = float(np.mean(np.abs(o) >= 440)) if len(o) else np.nan
            o = d.out[(d.at >= a - 1.0) & (d.at < a), 0] - 1443
            L['lh_steer'] = float(np.mean(o)) if len(o) else np.nan
            o = d.out[(d.at >= a + 2.4) & (d.at < a + 3.6), 0] - 1443
            L['de_steer'] = float(np.mean(o)) if len(o) else np.nan
        else:
            o = d.pwm[(d.vt >= a) & (d.vt < a + 0.8), 0] - 1443
            L['r1_sat'] = float(np.mean(np.abs(o) >= 440)); 
            L['lh_steer'] = float(np.mean(d.pwm[(d.vt >= a - 1.0) & (d.vt < a), 0] - 1443))
            L['de_steer'] = float(np.mean(d.pwm[(d.vt >= a + 2.4) & (d.vt < a + 3.6), 0] - 1443))
        # events in this lap
        evs = [r for r in rows if a - 0.05 <= r['t'] < b - 0.05]
        L['ev'] = [(round(r['t'] - a, 2), r['g']) for r in evs]
        L['hit_r1'] = any(r['t'] - a < 1.0 for r in evs)
        L['hit_gm'] = any(1.0 <= r['t'] - a < 2.6 for r in evs)
        L['hit_de'] = any(2.6 <= r['t'] - a < 4.8 for r in evs)
        L['hit_other'] = any(r['t'] - a >= 4.8 for r in evs)
        laps.append(L)
    return d, an, laps
if __name__ == '__main__':
    import pickle
    allres = {}
    for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM[3:]]:
        d, an, laps = lap_table(log, hum)
        allres[log] = laps
        print(log, 'anchors', len(an))
        for L in laps:
            print('  %6.1f dur %5.2f %s vHS %5.0f vLH %5.0f vR1 %5.0f vDE %5.0f depth %+5.0f yawmin %+5.0f sat %.2f lhSt %+4.0f deSt %+4.0f %s' % (
                L['t0'], L['dur'], 'AI' if L['ai'] else '--', L['v_hs'], L['v_lh'], L['v_r1'], L['v_de'], L['r1_depth'], L['yaw_r1_min'], L['r1_sat'], L['lh_steer'], L['de_steer'], L['ev']))
    pickle.dump(allres, open('laps.pkl', 'wb'))
