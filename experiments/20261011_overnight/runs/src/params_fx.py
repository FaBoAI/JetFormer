"""Per-configuration stats (gains / lookahead as logged per frame in AI_CMD)."""
import json

import numpy as np

from load import L, KV, NEUTRAL
from course import heading

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605', '20261010-172910', '20261010-173445']


def stats(d, m):
    """m: AI-frame mask"""
    a = m & d.sel & ~d.hold
    t = d.at
    dt = np.clip(np.diff(t, append=t[-1] + 1 / 30), 0, 0.1)
    T = float(np.sum(dt[a]))
    if T < 1:
        return None
    # vehicle samples inside frames of the mask
    i = np.clip(np.searchsorted(d.at, d.vt, side='right') - 1, 0, len(d.at) - 1)
    mv = a[i] & (d.vt <= d.at[-1])
    dv = np.clip(np.diff(d.vt, append=d.vt[-1]), 0, 0.2)
    dist = float(np.sum((-d.erpm * KV * dv)[mv & (d.erpm < 0)]))
    j = np.clip(np.searchsorted(d.at, d.it, side='right') - 1, 0, len(d.at) - 1)
    mi = a[j]
    di = np.clip(np.diff(d.it, append=d.it[-1]), 0, 0.2)
    hl = float(np.sum(((d.gyr[:, 2] - d.bias) * di)[mi]) / 360.0)
    ev = [e for e in d.events(2.5) if e['ai'] and a[min(np.searchsorted(d.at, e['t']), len(a) - 1)]]
    so = d.out[a, 0]; dev = so - 1443
    flips = np.sum((np.sign(dev[:-1]) * np.sign(dev[1:]) < 0) & (np.diff(np.nonzero(a)[0]) == 1))
    pr = d.pred[a]                     # (n,10,2)
    # jitter of the used prediction between consecutive frames
    k = np.clip(d.la[a] - 1, 0, 9)
    used = pr[np.arange(len(k)), k, 0]
    jit = float(np.median(np.abs(np.diff(used)))) if len(used) > 2 else np.nan
    er = -d.erpm[mv]
    er = er[er > 1000]
    return {'ai_s': round(T, 1), 'dist_m': round(dist, 1), 'heading_laps': round(hl, 2), 'impacts': len(ev),
            'impacts_per_lap': round(len(ev) / hl, 2) if hl > 0.3 else None,
            'impacts_per_100m': round(100 * len(ev) / dist, 2) if dist > 5 else None,
            'steer_flips_per_s': round(float(flips) / T, 2), 'steer_limit': round(float(np.mean((so <= 1001) | (so >= 1999))), 3),
            'steer_absdev_med': round(float(np.median(np.abs(dev))), 1), 'pred_jitter_us': round(jit, 1),
            'thr_out_med': round(float(np.median(d.out[a, 1] - NEUTRAL))), 'thr_out_p10': round(float(np.percentile(d.out[a, 1] - NEUTRAL, 10))),
            'raw_thr_med': round(float(np.median(d.raw[a, 1] - NEUTRAL))),
            'erpm_p50': int(np.median(er)) if len(er) else None, 'erpm_p90': int(np.percentile(er, 90)) if len(er) else None}


if __name__ == '__main__':
    out = []
    for log in LOGS:
        d = L(log)
        heading(d)
        keys = sorted(set((round(float(g[0]), 2), round(float(g[1]), 2), int(la)) for g, la, s in zip(d.gains, d.la, d.sel) if s))
        for key in keys:
            m = (np.round(d.gains[:, 0], 2) == key[0]) & (np.round(d.gains[:, 1], 2) == key[1]) & (d.la == key[2])
            s = stats(d, m)
            if s:
                s.update({'log': log, 'name': d.name, 'steer_gain': key[0], 'throttle_gain': key[1], 'lookahead': key[2]})
                out.append(s)
                print(json.dumps(s))
    json.dump(out, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/params_fx.json', 'w'), indent=1)
