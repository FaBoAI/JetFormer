from regime import *
from destat import de_table
S = {}
for log, hum in [('20261010-172910', False), ('20261010-173445', False)] + [(h, True) for h in HUM]:
    if hum:
        d = D(log); d.heading(bias=-0.0915); rows = analyze_h(log)[1]
    else:
        d, rows, _, tk, segs = analyze(log)
    R1 = []
    for a, dep, hit in true_r1(d, r1_anchors(d), rows):
        if not hum:
            m = (d.at >= a - 2.5) & (d.at < a + 0.5)
            if not d.sel[m].all(): continue
        g = max([h['g'] for h in hit]) if hit else 0
        mv = (d.vt >= a - 0.4) & (d.vt < a + 0.15)
        stuck = False
        if hit:
            th = hit[0]['t']; mm = (d.vt > th + 0.2) & (d.vt < th + 1.5); stuck = bool(np.mean(np.abs(d.erpm[mm]) < 300) > 0.5)
        R1.append((a, g, float(np.max(-d.erpm[mv])), stuck))
    DE = de_table(log, hum)
    n1 = len(R1); h1 = sum(1 for r in R1 if r[1] >= 2.5); s1 = sum(1 for r in R1 if r[1] >= 4); st1 = sum(1 for r in R1 if r[3])
    nd = len(DE); hd_ = sum(1 for r in DE if r['hit']); sd = sum(1 for r in DE if any(h[1] >= 4 for h in r['hit']))
    gm = sum(1 for r in DE if r['gmhit'])
    vin_hit = [r[2] for r in R1 if r[1] >= 2.5]; vin_ok = [r[2] for r in R1 if r[1] < 2.5]
    S[log] = dict(n1=n1, h1=h1, s1=s1, st1=st1, nd=nd, hd=hd_, sd=sd, gm=gm, vin_hit=vin_hit, vin_ok=vin_ok)
    print('%s R1 passes %2d hit %2d strong %2d stuck %d | v_in hit med %s ok med %s | DE passes %2d hit %2d strong %2d | GM hits %d' % (log, n1, h1, s1, st1,
        '%.0f' % np.median(vin_hit) if vin_hit else '-', '%.0f' % np.median(vin_ok) if vin_ok else '-', nd, hd_, sd, gm))
import pickle; pickle.dump(S, open('rates.pkl', 'wb'))
