from psi import *
def si(x):
    return int(x) if np.isfinite(x) else -99999
HUM = ['20261010-135449','20261010-144028','20261010-150424','20261010-152405','20261010-152707']
def analyze_h(log):
    d = D(log); d.heading(bias=-0.0915)
    segs, gs = rsegs(d)
    anchors = [s[0] for s in segs if s[2] < -90]
    rows = []
    for e in [e for e in d.events(2.5) if not e['suspect']]:
        t = e['t']
        w = np.nonzero(np.abs(d.it - t) <= 0.04)[0]
        hz = np.linalg.norm(d.acc[w,:2], axis=1); j = w[np.argmax(hz)]
        er_b = -mean_in(d.vt, d.erpm[:,None], t-0.5, t-0.1)[0]; er_a = -mean_in(d.vt, d.erpm[:,None], t+0.3, t+0.8)[0]
        p_b = mean_in(d.vt, d.pwm, t-0.5, t-0.05)
        gz_b = mean_in(d.it, (d.gyr[:,2]-d.bias)[:,None], t-0.4, t-0.05)[0]
        prev = [a for a in anchors if a <= t + 0.05]
        rows.append(dict(t=round(t,2), g=round(e['g'],1), ax=round(d.acc[j,0]/G,1), ay=round(d.acc[j,1]/G,1), erpm_b=si(er_b), erpm_a=si(er_a),
                         gz_b=si(gz_b), steer_b=si(p_b[0]-1443), thr_b=si(p_b[1]-1498), sel=False, takeover=None,
                         t_since_R1=round(t-prev[-1],2) if prev else None,
                         h_since_R1=int(np.interp(t,d.it,d.hd)-np.interp(prev[-1],d.it,d.hd)) if prev else None))
    return d, rows, anchors, segs
if __name__ == '__main__':
    for log in HUM:
        d, rows, anchors, segs = analyze_h(log)
        print(log, 'erpm nonzero frac %.2f' % (np.abs(d.erpm)>0).mean(), 'gyro z p99 %.0f' % np.percentile(np.abs(d.gyr[:,2]),99))
        for r in rows:
            print('  %7.2f g%4.1f ax%+5.1f ay%+5.1f er %5d->%5d yaw %+4d st %+4d th %+4d tR1 %s hR1 %s' % (r['t'], r['g'], r['ax'], r['ay'], r['erpm_b'], r['erpm_a'], r['gz_b'], r['steer_b'], r['thr_b'], r['t_since_R1'], r['h_since_R1']))
