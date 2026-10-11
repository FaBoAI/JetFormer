from laps import *
def true_r1(d, an, rows):
    out = []
    gz = np.convolve(d.gyr[:,2]-d.bias, np.ones(11)/11, 'same')
    for a in an:
        m = (d.it >= a) & (d.it < a + 1.5)
        depth = float(np.min(d.hd[m] - np.interp(a, d.it, d.hd))); ymin = float(gz[m].min())
        hit = [r for r in rows if a - 0.05 <= r['t'] < a + 1.0]
        if ymin <= -170 and (depth <= -90 or hit):
            out.append((a, depth, hit))
    return out
if __name__ == '__main__':  # _MAINGUARD_
    res = {}
    for log, hum in [('20261010-172910', False), ('20261010-173445', False), ('20261010-152405', True), ('20261010-152707', True)]:
        if hum:
            d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
        else:
            d, rows, _, tk, segs = analyze(log)
        an = r1_anchors(d)
        R = []
        for a, depth, hit in true_r1(d, an, rows):
            if not hum:
                m = (d.at >= a - 2.5) & (d.at < a + 0.5)
                if not d.sel[m].all(): continue
                thr = d.out[(d.at >= a - 0.5) & (d.at < a), 1] - 1498
                thr_raw = d.raw[(d.at >= a - 0.5) & (d.at < a), 1] - 1498
            else:
                thr = d.pwm[(d.vt >= a - 0.5) & (d.vt < a), 1] - 1498; thr_raw = thr
            mv = (d.vt >= a - 0.4) & (d.vt < a + 0.15)
            v_in = float(np.max(-d.erpm[mv]))
            mv2 = (d.vt >= a - 1.5) & (d.vt < a - 0.5)
            v_lh = float(np.mean(-d.erpm[mv2]))
            R.append(dict(a=a, hit=bool(hit), g=max([h['g'] for h in hit]) if hit else 0, v_in=v_in, v_lh=v_lh, thr=float(np.mean(thr)), thr_min=float(np.min(thr)), thr_raw=float(np.mean(thr_raw))))
        res[log] = R
        print(log)
        for r in R:
            print('  %6.1f %s g%4.1f v_in %5.0f v_lh %5.0f thr %+5.0f min %+5.0f raw %+5.0f' % (r['a'], 'HIT' if r['hit'] else '   ', r['g'], r['v_in'], r['v_lh'], r['thr'], r['thr_min'], r['thr_raw']))
        for flag in (True, False):
            s = [r for r in R if r['hit'] == flag]
            if s: print('   hit=%s n=%d v_in med %.0f thr med %.0f v_lh med %.0f' % (flag, len(s), np.median([r['v_in'] for r in s]), np.median([r['thr'] for r in s]), np.median([r['v_lh'] for r in s])))
    import pickle; pickle.dump(res, open('r1.pkl', 'wb'))
