from ev import *
d, rows, anchors, tk, segs = analyze('20261010-173445')
for t in [35.12, 58.19, 92.99, 143.17, 157.78, 210.43, 244.30, 271.11]:
    m = (d.at > t) & (d.at < t + 2.5) & d.sel
    th = d.out[m, 1]; st = d.out[m, 0]; tt = d.at[m]
    raw = d.raw[m, 1]
    neut = np.abs(th - 1498) < 40
    # longest neutral run
    best = 0; cur = 0
    for x in neut:
        cur = cur + 1 if x else 0; best = max(best, cur)
    mv = (d.vt > t + 0.2) & (d.vt < t + 2.5) & d.vsel
    tko = [x for x in tk if t < x < t + 5]
    print('%6.2f AI-side %.1fs after: thr out p10/p50/p90 %4.0f/%4.0f/%4.0f (raw p90 %4.0f) rev(>1600) %.0f%% longest neutral %dms | |eRPM|<300 %.0f%% | steer p50 %+.0f | takeover +%s' % (
        t, m.sum()/30, *np.percentile(th, [10, 50, 90]), np.percentile(raw, 90), 100*np.mean(th > 1600), best * 33, 100*np.mean(np.abs(d.erpm[mv]) < 300) if mv.any() else -1, np.median(st) - 1443, '%.1f' % (tko[0]-t) if tko else '-'))
