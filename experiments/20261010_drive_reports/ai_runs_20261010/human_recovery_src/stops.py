import numpy as np, json
from rruns3 import get, LOGS
from eff import NEUT, CENTER, runs
E = json.load(open('episodes.json'))
def impacts(d, gthr=2.5):
    ev = []
    for i in np.nonzero(d.g > gthr)[0]:
        if ev and d.it[i] - ev[-1]['t'] <= 0.6:
            if d.g[i] > ev[-1]['g']: ev[-1].update(g=float(d.g[i]), ay=float(d.acc[i, 1] / 9.81), ax=float(d.acc[i, 0] / 9.81))
            continue
        near = (np.abs(d.it - d.it[i]) <= 0.05) & (d.g > 1.0)
        ev.append({'t': float(d.it[i]), 'g': float(d.g[i]), 'ay': float(d.acc[i, 1] / 9.81), 'ax': float(d.acc[i, 0] / 9.81), 'suspect': bool(near.sum() <= 1)})
    return ev
out = []
for l in LOGS:
    d = get(l)
    e = d.erpm; t = d.vt
    for ev in impacts(d):
        if ev['suspect']: continue
        i = np.searchsorted(t, ev['t'])
        pre = float(np.median(e[max(0, i - 30):i]))
        post = e[i:i + 50]
        # min |erpm| within 0.5 s after
        stopped = bool((np.abs(post) < 300).any() or (post > 0).any())
        kept = float(np.median(e[i + 30:i + 60])) if i + 60 < len(e) else np.nan
        side = 'AI' if d.vsel[max(0, i - 5)] else 'H'
        # what happened next (within 4 s)
        ep = [x for x in E if x['log'] == l and -1.0 <= x['t_stop'] - ev['t'] <= 1.0]
        nxt = 'reverse-episode #%d' % (E.index(ep[0]) + 1) if ep else ''
        if not ep and stopped:
            # forward-only? find time until erpm < -1500 sustained 0.3 s
            j = i
            seg = e[i:i + 600]
            fw = np.nonzero(np.convolve((seg < -1500).astype(int), np.ones(30, int), 'valid') >= 30)[0]
            tf = float(t[i + fw[0]] - t[i]) if len(fw) else None
            # human reverse within 4 s?
            rv = (seg[:400] > 800).sum() * 0.01
            sel_after = d.vsel[i:i + 400]
            to = bool(sel_after[:1].all() and (~sel_after).any())
            nxt = 'stopped -> forward again after %s s (rev motion %.2fs)%s' % (None if tf is None else round(tf, 2), rv, ' takeover' if to else '')
        if not stopped: nxt = 'kept moving (glancing)'
        out.append(dict(log=l, t=round(ev['t'], 2), g=round(ev['g'], 1), ay=round(ev['ay'], 1), ax=round(ev['ax'], 1), side=side, erpm_pre=round(pre), erpm_post=None if np.isnan(kept) else round(kept), stopped=stopped, next=nxt))
json.dump(out, open('impacts.json', 'w'), indent=1)
from collections import Counter
for o in out: print(o['log'][-6:], o['t'], o['side'], 'g', o['g'], 'ay', o['ay'], 'pre', o['erpm_pre'], 'post', o['erpm_post'], o['next'])
