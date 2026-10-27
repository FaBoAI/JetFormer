"""反証 safety1-2(読むだけ): 本物の StartKick の区間で、補助の始まりから 0.4 m/s(eRPM -800)に届くまでに進んだ距離"""
import json, os, sys, glob
import numpy as np
root = '/home/jetson/jetformer/data/raw'
rows = json.load(open(sys.argv[1]))['rows']
out = []
for r in rows:
    vt, ve = [], []
    for p in sorted(glob.glob(os.path.join(root, r['log'], '*', 'VEHICLE.npz'))):
        if os.path.getsize(p) == 0:
            continue
        try:
            z = np.load(p); vt.append(z['timestamps_us'] / 1e6); ve.append(z['erpm'].astype(float))
        except Exception:
            pass
    vt, ve = np.concatenate(vt), np.concatenate(ve); o = np.argsort(vt); vt, ve = vt[o], ve[o]
    t0 = r['t_ai']
    w = (vt >= t0) & (vt <= t0 + 2.0)
    tt, vv = vt[w], np.clip(-ve[w] * 0.5e-3, 0, None)
    k = np.where(-ve[w] >= 800)[0]
    if not len(k):
        out.append({'log': r['log'], 'row': r['sec_row'], 'd_to_0p4_m': None}); continue
    k = int(k[0])
    d = float(np.trapz(vv[:k + 1], tt[:k + 1]))
    out.append({'log': r['log'], 'row': r['sec_row'], 'boost_would': r['boost_would'] if abs(r['erpm_at_ai']) < 400 else 'moving',
                't_to_0p4_s': round(float(tt[k] - t0), 2), 'd_to_0p4_m': round(d, 3)})
ds = [x['d_to_0p4_m'] for x in out if x['d_to_0p4_m'] is not None]
s = {'n': len(out), 'reached_0p4': len(ds), 'd_max_m': max(ds), 'd_median_m': float(np.median(ds))}
json.dump({'summary': s, 'rows': out}, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print(s)
print([x for x in out if x.get('boost_would') in ('near', 'ttc_n/tof')])
