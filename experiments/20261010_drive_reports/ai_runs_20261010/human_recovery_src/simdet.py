import numpy as np, json
from rruns3 import get, LOGS
from eff import NEUT, runs
E = json.load(open('episodes.json')); CLS = json.load(open('ep_class.json'))
def det(d, T2=0.6, confirm=0.5, gthr=2.5, still_erpm=150, startup=1.0, stop_win=0.6, fwd_counts=20):
    t = d.vt; e = d.erpm; h = d.hall; n = len(t)
    dh = np.abs(h - np.r_[np.full(10, h[0]), h[:-10]])
    still = (np.abs(e) < still_erpm) & (dh <= 2)
    beyond = (d.thr < NEUT - 20) | (d.thr > NEUT + 40)
    # startup mask: within `startup` s after a select rising edge or log start
    sel = d.vsel.astype(int)
    rise = np.r_[0, np.nonzero(np.diff(sel) == 1)[0] + 1]
    since_start = np.full(n, 99.0)
    for r in rise:
        k = np.arange(r, n); since_start[k] = np.minimum(since_start[k], t[k] - t[r])
    trig = []
    # D2
    cnt = 0.0
    for k in range(n):
        if still[k] and beyond[k] and since_start[k] > startup:
            cnt += 0.01
            if abs(cnt - T2) < 0.005: trig.append(('D2', t[k]))
        else:
            cnt = 0.0
    # D1
    gi = d.g; it = d.it
    i = 0; last = -9
    while i < len(gi):
        if gi[i] > gthr and it[i] - last > 0.6:
            near = (np.abs(it - it[i]) <= 0.05) & (gi > 1.0)
            last = it[i]
            if near.sum() > 1:
                k0 = np.searchsorted(t, it[i])
                if np.median(e[max(0, k0 - 20):k0]) < -500:     # was driving forward
                    ks = [k for k in range(k0, min(n, k0 + int(stop_win * 100))) if e[k] > -300]
                    if ks:
                        k1 = ks[0]; k2 = min(n, k1 + int(confirm * 100))
                        if k2 < n and -(h[k2] - h[k1]) < fwd_counts:
                            trig.append(('D1', t[k2]))
        i += 1
    return sorted(trig, key=lambda x: x[1])

def evaluate(**kw):
    rows = []
    for l in LOGS:
        d = get(l)
        tr = det(d, **kw)
        # merge triggers within 1.5 s
        m = []
        for kind, tt in tr:
            if m and tt - m[-1][1] < 1.5: continue
            m.append((kind, tt))
        for kind, tt in m:
            ep = [k + 1 for k, x in enumerate(E) if x['log'] == l and x['t_stop'] - 0.3 <= tt <= x['t_stop'] + max(x['engage_after_stop'], 0.5) + 0.3]
            k = np.searchsorted(d.vt, tt)
            rows.append(dict(log=l, kind=kind, t=round(float(tt), 2), ai=bool(d.vsel[min(k, len(d.vt) - 1)]), ep=ep[0] if ep else None,
                             cls=CLS[str(ep[0])] if ep else None,
                             lat_from_stop=round(float(tt - E[ep[0] - 1]['t_stop']), 2) if ep else None))
    return rows
if __name__ == '__main__':
    import sys
    for T2 in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        for conf in [0.3, 0.5]:
            rows = evaluate(T2=T2, confirm=conf)
            eps_hit = sorted(set(r['ep'] for r in rows if r['ep']))
            A_eps = [int(k) for k, v in CLS.items() if v == 'A']
            hitA = [e for e in eps_hit if e in A_eps]
            fp = [r for r in rows if r['ep'] is None]
            lat = [min(r['lat_from_stop'] for r in rows if r['ep'] == e) for e in hitA]
            print('T2=%.1f confirm=%.1f  A-episodes detected %d/%d  (AI-side A %d/%d)  other-trigger %d  latency med %.2f max %.2f' % (
                T2, conf, len(hitA), len(A_eps), len([e for e in hitA if e >= 19]), len([e for e in A_eps if e >= 19]), len(fp), np.median(lat), np.max(lat)))
    rows = evaluate(T2=0.6, confirm=0.5)
    json.dump(rows, open('simdet.json', 'w'), indent=1)
    for r in rows: print(r)
