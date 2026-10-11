import numpy as np, warnings
from jetformer import tof_format
def obstacle_dist(d, rows=(0, 21), cols=(6, 48), ratio=0.7, minpix=25):
    """Per-frame distance [m] to the nearest object that is clearly closer than the floor/background baseline
    seen during free forward driving. inf = nothing clearly closer than baseline."""
    D = tof_format.distance_mm(d.depth[:, rows[0]:rows[1], cols[0]:cols[1]].astype(np.uint16)).astype(float)
    D[D <= 0] = np.nan
    # free driving frames: |erpm| large forward
    e_at = np.interp(d.tt, d.vt, d.erpm)
    free = e_at < -3000
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base = np.nanmedian(D[free], axis=0) if free.sum() > 50 else np.nanmedian(D, axis=0)
        obs = D < ratio * base[None]
        out = np.full(len(D), np.inf)
        for i in range(len(D)):
            v = D[i][obs[i]]
            if v.size >= minpix:
                out[i] = np.percentile(v, 20) / 1000.0
    return out, base
