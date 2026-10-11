import numpy as np, json
from rruns3 import get
from eff import NEUT, CENTER, runs
E = json.load(open('episodes.json')); CLS = json.load(open('ep_class.json'))
res = []
for k, x in enumerate(E):
    if CLS[str(k + 1)] != 'A': continue
    d = get(x['log']); t = d.vt
    i0 = x['i_rev_last'] - 30   # a bit before reverse end
    s = np.zeros(len(t), int); s[d.thr < NEUT - 20] = -1; s[d.thr > NEUT + 40] = 1
    rr = [(a, b, v) for a, b, v in runs(s[i0:i0 + 400]) if v == -1 and b - a >= 5]
    if not rr: res.append(dict(ep=k + 1)); continue
    a, b, _ = rr[0]; a += i0; b += i0
    # include continuation: forward command runs separated by <0.15 s neutral
    rec = dict(ep=k + 1, log=x['log'][-6:], src=x['source'], kick_start_after_rev_end=round(float(t[a] - t[x['i_rev_last']]), 2), kick_dur=round(float(t[b - 1] - t[a] + 0.01), 2),
               kick_thr_med=round(float(np.median(d.thr[a:b]))), kick_thr_min=round(float(d.thr[a:b].min())), kick_st_med=round(float(np.median(d.st[a:b]) - CENTER)),
               kick_counts=round(float(-(d.hall[min(b + 50, len(t) - 1)] - d.hall[a]))), erpm_end_kick=round(float(d.erpm[b - 1])), contact_ay=x['impact']['ay'], st_stop=x['st_stop'], st_rev=x['st_rev'])
    res.append(rec)
for r in res: print(r)
json.dump(res, open('fwdkick.json', 'w'), indent=1)
def q(v):
    v = np.array(v, float); return 'median %.2f IQR %.2f-%.2f range %.2f-%.2f n=%d' % (np.median(v), *np.percentile(v, [25, 75]), v.min(), v.max(), len(v))
for key in ['kick_start_after_rev_end', 'kick_dur', 'kick_thr_med', 'kick_thr_min', 'kick_st_med', 'kick_counts', 'erpm_end_kick']:
    for src in ['human-only', 'AI-run takeover']:
        v = [r[key] for r in res if key in r and r['src'] == src]
        print(key, src, q(v))
