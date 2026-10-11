"""Recompute the guard TTC offline (jetformer.tof_guard.ToFGuard, numpy only) with
 (a) the floor reference used online (validation: must reproduce the logged ttc),
 (b) a floor reference rebuilt from the same run's straight driving (what a re-calibration would give).
Saves per-run arrays to src/ttc_<log>.npz"""
import sys

import numpy as np

from load import L
from jetformer.tof_guard import ToFGuard

REF = '/home/jetson/jetformer/data/tof_floor_ref.npy'


def build_ref(d):
    raw = d.depth
    dd = (raw & 0x1FFF).astype(float)
    conf = raw >> 13
    er = d.erpm_at(d.tt)
    gz = np.interp(d.tt, d.it, d.gyr[:, 2])
    m = (np.abs(er) > 1500) & (np.abs(gz) < 60)
    x = np.where(conf == 4, dd, np.nan)[m]
    fl = np.nanpercentile(x, 80, axis=0)
    ref0 = np.load(REF).astype(float)
    fl = np.where(np.isfinite(fl), fl, ref0)
    return fl, int(m.sum())


def recompute(d, ref):
    g = ToFGuard(ref, center=1443.0, neutral=1498.0)
    n = len(d.at)
    ttc = np.full(n, np.nan); xf = np.full(n, np.nan); nobs = np.zeros(n, int)
    ti = np.clip(np.searchsorted(d.tt, d.at, side='right') - 1, 0, len(d.tt) - 1)
    ii = np.clip(np.searchsorted(d.it, d.at, side='right') - 1, 0, len(d.it) - 1)
    for k in range(n):
        if d.hold[k]:
            continue
        er = float(d.erpm_at(d.at[k]))
        f = g.features(d.depth[ti[k]], er, float(d.gyr[ii[k], 2]), float(d.mout[k, 0]))
        ttc[k], xf[k], nobs[k] = f['ttc'], f['x_free'], f['n_obs']
    return ttc, xf, nobs


if __name__ == '__main__':
    for log in sys.argv[1:]:
        d = L(log, tof=True)
        ref0 = np.load(REF)
        t0, x0, n0 = recompute(d, ref0)
        ok = np.isfinite(d.ttc) & np.isfinite(t0)
        agree = np.mean(np.abs(np.minimum(t0[ok], 3) - np.minimum(d.ttc[ok], 3)) < 0.02)
        agree_w = np.mean((t0[ok] < 0.5) == (d.ttc[ok] < 0.5))
        ref1, nref = build_ref(d)
        t1, x1, n1 = recompute(d, ref1)
        print(d.name, 'reproduce logged ttc (|d|<0.02 s): %.3f  warn agreement %.3f  ref frames %d' % (agree, agree_w, nref))
        np.savez('/home/jetson/jetformer_reports/overnight_20261011/runs/src/ttc_%s.npz' % log, ttc0=t0, x0=x0, n0=n0,
                 ttc1=t1, x1=x1, n1=n1, ref1=ref1)
