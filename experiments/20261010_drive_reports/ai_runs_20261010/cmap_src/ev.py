from ld import *
import json
def rsegs(d, thr=-30):
    gz = d.gyr[:,2]-d.bias; gs = np.convolve(gz, np.ones(11)/11, 'same')
    neg = gs < thr; out=[]; i=0; n=len(neg)
    while i<n:
        if neg[i]:
            j=i
            while j<n and neg[j]: j+=1
            out.append((d.it[i], d.it[j-1], d.hd[j-1]-d.hd[i], gs[i:j].min())); i=j
        else: i+=1
    return out, gs
def mean_in(ts, arr, a, b):
    m = (ts>=a)&(ts<b)
    return arr[m].mean(axis=0) if m.any() else np.nan*np.ones(arr.shape[1:])
def analyze(log):
    d = load(log); d.heading()
    segs, gs = rsegs(d)
    anchors = [s[0] for s in segs if s[2] < -90]
    # takeovers
    s = d.sel.astype(int); ch = np.nonzero(np.diff(s))[0]
    tk = [d.at[i+1] for i in ch if s[i+1]==0]
    ai_on = [d.at[i+1] for i in ch if s[i+1]==1]
    evs = [e for e in d.events(2.5) if not e['suspect']]
    rows = []
    for e in evs:
        t = e['t']; i = e['i']
        # peak accel vector (in g) over +-30ms by max horizontal
        w = np.nonzero(np.abs(d.it - t) <= 0.04)[0]
        hz = np.linalg.norm(d.acc[w,:2], axis=1); j = w[np.argmax(hz)]
        ax_, ay_ = d.acc[j,0]/G, d.acc[j,1]/G
        er_b = -mean_in(d.vt, d.erpm[:,None], t-0.5, t-0.1)[0]
        er_a = -mean_in(d.vt, d.erpm[:,None], t+0.3, t+0.8)[0]
        gz_b = mean_in(d.it, (d.gyr[:,2]-d.bias)[:,None], t-0.4, t-0.05)[0]
        o_b = mean_in(d.at, d.out, t-0.5, t-0.05)
        r_b = mean_in(d.at, d.raw, t-0.5, t-0.05)
        sel = bool(d.at_t(d.at, d.sel, t))
        nxt_tk = [x for x in tk if t - 0.3 <= x <= t + 2.5]
        prev_a = [a for a in anchors if a <= t + 0.05]
        tsa = t - prev_a[-1] if prev_a else np.nan
        hsa = np.interp(t, d.it, d.hd) - np.interp(prev_a[-1], d.it, d.hd) if prev_a else np.nan
        # inside a right-turn segment?
        inr = [sg for sg in segs if sg[0]-0.05 <= t <= sg[1]+0.15 and sg[3] < -100]
        # stuck: |erpm| < 300 for > 0.5 s within 2 s after on AI side
        m = (d.vt > t) & (d.vt < t + 2.0)
        stuck_frac = (np.abs(d.erpm[m]) < 300).mean()
        rows.append(dict(t=round(t,2), g=round(e['g'],1), ax=round(ax_,1), ay=round(ay_,1), erpm_b=int(er_b), erpm_a=int(er_a),
            gz_b=int(gz_b), steer_b=int(o_b[0]-1443), thr_b=int(o_b[1]-1498), steer_raw_b=int(r_b[0]-1443), sel=sel,
            takeover=round(nxt_tk[0]-t,2) if nxt_tk else None, t_since_R1=round(tsa,2), h_since_R1=int(hsa) if np.isfinite(hsa) else None,
            in_right=bool(inr), stuck_frac=round(stuck_frac,2)))
    return d, rows, anchors, tk, segs
if __name__ == '__main__':
    out = {}
    for log in ['20261010-172910','20261010-173445']:
        d, rows, anchors, tk, segs = analyze(log)
        print(log, 'anchors', np.round(anchors,1)); print(' takeovers', np.round(tk,2))
        print('   t     g    ax   ay  erB  erA  gzB stB thB stRaw sel  tk   tR1  hR1 inR stuck')
        for r in rows:
            print('%7.2f %4.1f %5.1f %5.1f %5d %5d %4d %4d %4d %4d %s %5s %5.2f %5s %s %.2f' % (r['t'], r['g'], r['ax'], r['ay'], r['erpm_b'], r['erpm_a'], r['gz_b'], r['steer_b'], r['thr_b'], r['steer_raw_b'], 'AI' if r['sel'] else 'HU', r['takeover'], r['t_since_R1'], r['h_since_R1'], 'R' if r['in_right'] else '-', r['stuck_frac']))
        out[log] = rows
    json.dump(out, open('events.json','w'), indent=1)
