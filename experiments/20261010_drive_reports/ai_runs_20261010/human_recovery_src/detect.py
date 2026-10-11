import numpy as np, json
from rruns3 import get, LOGS
from eff import NEUT, runs
from stall import still_mask
E = json.load(open('episodes.json')); IMP = json.load(open('impacts.json'))
CLS = json.load(open('ep_class.json'))
# --- D1: impact + still
print('=== D1 impact -> still')
d1 = []
for x in IMP:
    d = get(x['log']); t = d.vt; st = still_mask(d)
    i = np.searchsorted(t, x['t'])
    # first time within 0.8 s after impact where car is 'still' for 0.2 s continuously, or erpm > -300 (stop/rebound)
    stop_i = None
    for k in range(i, min(len(t) - 20, i + 80)):
        if d.erpm[k] > -300: stop_i = k; break
    still_i = None
    if stop_i is not None:
        for k in range(stop_i, min(len(t) - 20, stop_i + 150)):
            if st[k:k + 20].all(): still_i = k + 20; break
    ep = [k + 1 for k, e in enumerate(E) if e['log'] == x['log'] and abs(e['t_stop'] - x['t']) <= 1.0]
    lab = CLS.get(str(ep[0]), '?') if ep else ('glancing' if stop_i is None else x['next'][:40])
    rec = dict(log=x['log'], t=x['t'], g=x['g'], side=x['side'], stop_after=None if stop_i is None else round(float(t[stop_i] - x['t']), 2),
               still_trigger_after=None if still_i is None else round(float(t[still_i] - x['t']), 2), ep=ep[0] if ep else None, label=lab)
    d1.append(rec)
    if stop_i is not None:
        print(rec)
json.dump(d1, open('d1.json', 'w'), indent=1)
# --- start latency: still >= 0.5 s then command beyond deadband -> time to |erpm|>300
print('=== start-from-rest latency')
lat = []
for l in LOGS:
    d = get(l); t = d.vt; st = still_mask(d)
    F = d.thr < NEUT - 20; R = d.thr > NEUT + 40
    for k in range(50, len(t) - 200):
        if (F[k] or R[k]) and not (F[k - 1] or R[k - 1]) and st[k - 50:k].all():
            seg = np.abs(d.erpm[k:k + 300]) > 300
            j = np.argmax(seg) if seg.any() else None
            # did the command stay beyond deadband?
            lat.append(dict(log=l, t=round(float(t[k]), 2), dir='F' if F[k] else 'R', ai=bool(d.vsel[k]), cmd=round(float(np.median(d.thr[k:k + 20]))),
                            latency=None if j is None else round(float(t[k + j] - t[k]), 2), moved_dir=None if j is None else ('fwd' if d.erpm[k + j] < 0 else 'rev')))
for x in lat: print(x)
json.dump(lat, open('startlat.json', 'w'), indent=1)
