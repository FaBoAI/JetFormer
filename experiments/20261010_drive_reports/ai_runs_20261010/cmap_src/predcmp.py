from destat import *
import pickle
allR = pickle.load(open('de.pkl', 'rb'))
out = {}
for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM]:
    if hum:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
    else:
        d, rows, _, tk, segs = analyze(log)
    key = 'HU' if hum else log[-6:]
    # DE
    for r in allR[log]:
        tg = r['tg']; hit = any(h[1] >= 4 for h in r['hit'])
        if hum:
            s0 = np.interp(tg + 0.6, d.vt, d.pwm[:, 0]) - 1443; s9 = np.interp(tg + 0.93, d.vt, d.pwm[:, 0]) - 1443
            s_late = np.interp(tg + 1.1, d.vt, d.pwm[:, 0]) - 1443
        else:
            j = np.searchsorted(d.at, tg + 0.6); s0 = d.pred[j, 0, 0] - 1443; s9 = d.pred[j, 9, 0] - 1443
            s_late = d.at_t(d.at, d.raw[:, 0], tg + 1.1) - 1443
        out.setdefault((key, 'DE', hit), []).append((s0, s9, s_late))
    # R1
    for a, dep, h in true_r1(d, r1_anchors(d), rows):
        g = max([x['g'] for x in h]) if h else 0; hit = g >= 4
        if hum:
            th = [np.interp(a - 0.6 + k / 30, d.vt, d.pwm[:, 1]) - 1498 for k in (0, 9)]
            st = [np.interp(a - 0.6 + k / 30, d.vt, d.pwm[:, 0]) - 1443 for k in (0, 9)]
        else:
            m = (d.at >= a - 2.5) & (d.at < a + 0.5)
            if not d.sel[m].all(): continue
            j = np.searchsorted(d.at, a - 0.6); th = [d.pred[j, k, 1] - 1498 for k in (0, 9)]; st = [d.pred[j, k, 0] - 1443 for k in (0, 9)]
        out.setdefault((key, 'R1', hit), []).append((th[0], th[1], st[0], st[1]))
for k in sorted(out, key=str):
    a = np.array(out[k])
    if k[1] == 'DE':
        print('%-6s DE hit=%-5s n=%2d steer(GMexit+0.6) %+4.0f -> +0.33s %+4.0f  ; at +1.1s %+4.0f   (raw/pre-gain for AI)' % (k[0], k[2], len(a), *np.median(a, axis=0)))
    else:
        print('%-6s R1 hit=%-5s n=%2d thr(R1-0.6s) %+4.0f -> +0.3s %+4.0f ; steer %+4.0f -> %+4.0f (raw/pre-gain for AI)' % (k[0], k[2], len(a), *np.median(a, axis=0)))
