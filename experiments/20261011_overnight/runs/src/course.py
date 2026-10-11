"""Course position from gyro: R1 anchors (the one sharp right hairpin per lap), heading psi relative to R1 start,
and a location label for any time t (same rules as collision_map.md section 1.3)."""
import numpy as np

from load import L, KV


def heading(d):
    gz = d.gyr[:, 2].astype(float)
    er = np.interp(d.it, d.vt, d.erpm)
    m = (np.abs(er) < 1) & (np.abs(gz) < 5)
    d.bias = float(np.median(gz[m])) if m.sum() > 50 else -0.09
    dt = np.diff(d.it, prepend=d.it[0])
    dt[dt > 0.2] = 0
    d.hd = np.cumsum((gz - d.bias) * dt)
    d.gzs = np.convolve(gz - d.bias, np.ones(11) / 11, 'same')
    return d.hd


def right_turns(d, thr=-170.0):
    """segments where smoothed yaw < thr: (t_start, t_end, dheading, min_rate)"""
    m = d.gzs < thr
    out = []
    i = 0
    n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and (m[j + 1] or (j + 8 < n and m[j + 1:j + 8].any())):
                j += 1
            # extend start back to where yaw crosses -60
            a = i
            while a > 0 and d.gzs[a - 1] < -60:
                a -= 1
            b = j
            while b + 1 < n and d.gzs[b + 1] < -60:
                b += 1
            out.append((float(d.it[a]), float(d.it[b]), float(d.hd[b] - d.hd[a]), float(d.gzs[i:j + 1].min())))
            i = b + 1
        else:
            i += 1
    return out


def r1_anchors(d):
    """R1 start times: right turn deeper than -90 deg (or ending in impact) preceded by a big left turn (LH)"""
    if not hasattr(d, 'hd'):
        heading(d)
    an = []
    for a, b, dh, mn in right_turns(d):
        h_prev = np.interp(a, d.it, d.hd) - np.interp(a - 2.0, d.it, d.hd)
        moving = np.median(np.abs(d.erpm[(d.vt > a - 0.5) & (d.vt < a)])) if ((d.vt > a - 0.5) & (d.vt < a)).any() else 0
        if moving < 1500:
            continue
        if dh < -90 or (dh < -40 and h_prev > 90):
            if h_prev > 60:
                if an and a - an[-1] < 5.0:
                    continue
                an.append(a)
    return an


def psi_course(d, an):
    """heading relative to the most recent R1 start (0 at R1 start, + = left)"""
    psi = np.full(len(d.it), np.nan)
    for k, a in enumerate(an):
        b = an[k + 1] if k + 1 < len(an) else d.it[-1] + 1
        m = (d.it >= a) & (d.it < b)
        psi[m] = d.hd[m] - np.interp(a, d.it, d.hd)
    return psi


def locate(d, an, t):
    """label for time t (collision_map rules, simplified): R1 / GM_in / GM_out / DE / SW / HS / LH / ?"""
    prev = [a for a in an if a <= t + 0.05]
    if not prev:
        return '?', np.nan
    a = prev[-1]
    dt = t - a
    nxt = [x for x in an if x > t + 0.05]
    if nxt and nxt[0] - t < 1.2:
        return 'LH', dt        # just before the next R1 = left hairpin (black curtain)
    if dt < 1.0:
        return 'R1', dt
    m = d.it >= a
    psi = d.hd[m] - np.interp(a, d.it, d.hd)
    tt = d.it[m]
    # GM exit: first time after R1 where psi comes back up to -25 deg
    after = (tt > a + 0.3)
    k = np.nonzero(after & (psi > -25))[0]
    t_gm = tt[k[0]] if len(k) else np.inf
    p = np.interp(t, d.it, d.hd) - np.interp(a, d.it, d.hd)
    if t < t_gm - 0.3:
        return 'GM_in', dt
    if abs(t - t_gm) <= 0.3:
        return 'GM_out', dt
    if t - t_gm <= 2.0:
        return 'DE', dt
    if p < 150:
        return 'SW', dt
    if p < 215:
        return 'HS', dt
    return 'LH', dt
