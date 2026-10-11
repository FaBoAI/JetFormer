"""ToF guard (log-only) evaluation on ai_run_009/010/101.

Definitions follow data/reports/ai_runs_20261010/tof_predictor.md section 4:
  warning episode = consecutive frames with the rule true (v > 0.5 m/s), gaps < 0.3 s merged
  caught         = a warning is on within the 0.3 s before the impact; lead = impact - episode onset
  false alarm    = episode onset with no impact (>= 2.0 g horizontal, any) within 1.5 s after the onset
  per minute     = AI-side forward driving (v > 0.5 m/s) minus 1 s after each impact
Collisions = AI-side impacts with horizontal |a| >= 2.5 g and forward speed > 0.5 m/s before.
"""
import json

import numpy as np

from load import L, seg_mask, KV, NEUTRAL

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
A_COAST, A_BRAKE = 1.0, 2.8      # m/s^2 (tof_predictor.md section 8)


def frame_speed(d):
    return np.maximum(-d.erpm_at(d.at) * KV, 0.0)


def episodes(mask, t, gap=0.3):
    eps = []
    for a, b in seg_mask(mask):
        if eps and t[a] - t[eps[-1][1]] < gap:
            eps[-1] = (eps[-1][0], b)
        else:
            eps.append((a, b))
    return eps


def collisions(d, gmin=2.5):
    return [e for e in d.events(2.5) if e['ai'] and e['g'] >= gmin and -e['erpm_before'] * KV > 0.5]


def evaluate(d, rule, ai_only=True, coll=None, all_ev=None):
    """rule: boolean per AI frame. Returns dict of metrics + per-collision rows + episodes."""
    v = frame_speed(d)
    base = (d.sel if ai_only else ~d.sel) & ~d.hold
    on = rule & base & (v > 0.5)
    eps = episodes(on, d.at)
    coll = coll if coll is not None else collisions(d)
    all_ev = all_ev if all_ev is not None else [e for e in d.events(2.0)]
    rows = []
    for e in coll:
        t = e['t']
        lead = None
        for a, b in eps:
            if d.at[a] <= t and d.at[b] >= t - 0.3:
                lead = t - d.at[a]
        m = (d.at >= t - 1.0) & (d.at <= t)
        rows.append({'t': round(t, 2), 'g': round(e['g'], 1), 'lead_s': None if lead is None else round(lead, 2),
                     'caught_0.3': bool(lead is not None and lead >= 0.3),
                     'caught_0.15': bool(lead is not None and lead >= 0.15),
                     'ttc_min_1s': round(float(np.nanmin(d.ttc[m])), 2) if m.any() else None})
    fa = []
    for a, b in eps:
        t0 = d.at[a]
        hit = any(t0 - 0.05 <= e['t'] <= t0 + 1.5 for e in all_ev)
        if not hit:
            fa.append((a, b))
    # exposure: forward AI frames, minus 1 s after each impact
    expo = base & (v > 0.5)
    for e in all_ev:
        expo &= ~((d.at > e['t']) & (d.at < e['t'] + 1.0))
    dt = np.diff(d.at, append=d.at[-1] + 1 / 30)
    dt = np.clip(dt, 0, 0.2)
    minutes = float(np.sum(dt[expo]) / 60.0)
    # warning time ratio over forward frames with no impact within 1.5 s
    safe = expo.copy()
    for e in all_ev:
        safe &= ~((d.at > e['t'] - 1.5) & (d.at <= e['t']))
    wtr = float(np.sum((on & safe) * dt) / max(np.sum(safe * dt), 1e-9))
    return {'episodes': len(eps), 'false_alarms': len(fa), 'minutes': minutes, 'fa_per_min': len(fa) / max(minutes, 1e-9),
            'warn_time_ratio': wtr, 'collisions': rows, 'eps': eps, 'fa': fa}


def lap_cost(d, g_code, a_r=2.0, delay=0.15):
    """open-loop time-loss estimate of applying the guard (slow: forward blended to neutral by u, brake: -2.8 m/s^2)
    g_code: 0/1/2 per frame. Returns total time loss [s] on AI forward frames."""
    v = frame_speed(d)
    t = d.at
    # actual accel from eRPM (smoothed, at frame times)
    vs = -d.erpm * KV
    k = 15
    vsm = np.convolve(vs, np.ones(k) / k, 'same')
    acc = np.gradient(vsm, d.vt)
    a_act = np.interp(t + delay, d.vt, acc)
    u = np.clip((0.5 - d.ttc) / 0.2, 0, 1)
    dt = np.clip(np.diff(t, append=t[-1] + 1 / 30), 0, 0.1)
    dv, D, loss = 0.0, 0.0, 0.0
    base = d.sel & ~d.hold
    for i in range(len(t)):
        if not base[i]:
            dv, D = 0.0, 0.0
            continue
        if g_code[i] == 2:
            a_cf = -A_BRAKE
        elif g_code[i] == 1:
            a_cf = (1 - u[i]) * a_act[i] + u[i] * (-A_COAST)
        else:
            a_cf = None
        if a_cf is not None:
            dv += max(a_act[i] - a_cf, 0.0) * dt[i]
            dv = min(dv, v[i])
        else:
            dv = max(dv - a_r * dt[i], 0.0)
        if v[i] > 0.3:
            loss += dv * dt[i] / v[i]
    return loss


def guard_code_from(d, t_warn, t_brake, nconsec=1, vmin=0.5):
    v = frame_speed(d)
    w = (d.ttc < t_warn) & (v >= vmin)
    if nconsec > 1:
        c = np.zeros(len(w), int)
        run = 0
        for i, x in enumerate(w):
            run = run + 1 if x else 0
            c[i] = run
        w = c >= nconsec
    g = np.where(w, 1, 0)
    g[w & (d.ttc < t_brake)] = 2
    return g


if __name__ == '__main__':
    res = {}
    for log in LOGS:
        d = L(log)
        v = frame_speed(d)
        # sanity: logged guard vs ttc rule
        g_re = guard_code_from(d, 0.5, 0.3)
        ok = np.mean((g_re == d.guard)[~d.hold])
        ev = evaluate(d, d.guard >= 1)
        evb = evaluate(d, d.guard >= 2)
        evh = evaluate(d, d.guard >= 1, ai_only=False)
        cost = lap_cost(d, np.where(d.sel, d.guard, 0))
        print('=====', d.name, 'guard rule reproduces log: %.4f' % ok)
        print(' AI: episodes %d FA %d  %.2f min  FA/min %.1f  warn time %.1f%%' % (
            ev['episodes'], ev['false_alarms'], ev['minutes'], ev['fa_per_min'], 100 * ev['warn_time_ratio']))
        print(' AI brake(<0.3): episodes %d FA %d FA/min %.1f warn %.1f%%' % (
            evb['episodes'], evb['false_alarms'], evb['fa_per_min'], 100 * evb['warn_time_ratio']))
        print(' human side: episodes %d FA %d %.2f min FA/min %.1f warn %.1f%%' % (
            evh['episodes'], evh['false_alarms'], evh['minutes'], evh['fa_per_min'], 100 * evh['warn_time_ratio']))
        for r, rb in zip(ev['collisions'], evb['collisions']):
            print('   coll', r, 'brake lead', rb['lead_s'])
        print(' open-loop time loss on AI side: %.2f s' % cost)
