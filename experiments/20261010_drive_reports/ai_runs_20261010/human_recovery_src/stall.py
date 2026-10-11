import numpy as np, json
from rruns3 import get, LOGS
from eff import NEUT, runs
E = json.load(open('episodes.json'))
IMP = json.load(open('impacts.json'))
def still_mask(d, erpm_thr=100, hall_win=10, hall_tol=1):
    h = d.hall
    dh = np.abs(h - np.r_[np.full(hall_win, h[0]), h[:-hall_win]])
    return (np.abs(d.erpm) < erpm_thr) & (dh <= hall_tol)
res = []
for l in LOGS:
    d = get(l); t = d.vt; e = d.erpm
    F = d.thr < NEUT - 20; R = d.thr > NEUT + 40
    st = still_mask(d)
    for name, cmd in [('F', F), ('R', R), ('any', F | R)]:
        m = cmd & st
        for a, b, v in runs(m.astype(int)):
            if v != 1: continue
            dur = (b - a) * 0.01
            if dur < 0.05: continue
            ai = bool(d.vsel[a:b].mean() > 0.5)
            # outcome at end of run
            nxt = e[b:b + 30]
            if b >= len(e) - 2: out = 'log end'
            elif ai and not d.vsel[min(b + 1, len(e) - 1)]: out = 'takeover'
            elif ai and (~d.vsel[a:b]).any(): out = 'takeover'
            elif (np.abs(nxt) > 300).any(): out = 'moved ' + ('fwd' if nxt[np.argmax(np.abs(nxt) > 300)] < 0 else 'rev')
            else: out = 'cmd ended'
            # prior context: impact in previous 1.5 s?
            imp = [x for x in IMP if x['log'] == l and 0 <= t[a] - x['t'] <= 1.5]
            # stuck-episode context: inside [t_stop, first engagement] of an episode
            ep = [k + 1 for k, x in enumerate(E) if x['log'] == l and x['t_stop'] - 0.2 <= t[a] <= x['t_first_eng']]
            # time since car was last moving (|erpm| > 1000)
            mv = np.nonzero(np.abs(e[:a]) > 1000)[0]
            since_move = float(t[a] - t[mv[-1]]) if len(mv) else None
            res.append(dict(log=l, kind=name, t=round(float(t[a]), 2), dur=round(dur, 2), ai=ai, out=out, impact=bool(imp), ep=ep[0] if ep else None,
                            cmd=round(float(np.median(d.thr[a:b]))), since_move=None if since_move is None else round(since_move, 2),
                            at_start=bool(t[a] < 3.0)))
json.dump(res, open('stalls.json', 'w'), indent=1)
for k in ['F', 'R', 'any']:
    print('=====', k)
    for r in res:
        if r['kind'] == k and r['dur'] >= 0.15:
            print('%s %7.2f %s dur %.2f cmd %d out=%-10s impact=%d ep=%s since_move=%s start=%d' % (r['log'][-6:], r['t'], 'AI' if r['ai'] else 'H ', r['dur'], r['cmd'], r['out'], r['impact'], r['ep'], r['since_move'], r['at_start']))
