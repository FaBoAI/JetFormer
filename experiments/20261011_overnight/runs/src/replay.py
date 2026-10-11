"""Offline replay of jetformer.assist.Assist (recovery v2 + ESC neutralising) on the logged frames.

Inputs per AI_CMD frame exactly as drive.py builds them:
  model_us = AI_CMD.model_out (ToF guard is log-only, so base_us == model_us)
  erpm     = latest VEHICLE sample at the camera time
  select   = AI_CMD.select (hold frames skip the helper and reset it)
  impact_g = max horizontal |a| over the IMU samples of the last 150 ms (last 10 samples)
  now      = camera time + latency (monotonic offset is irrelevant)
Also a counterfactual run with select forced True after each takeover for up to `extend` s is NOT physically
meaningful (the car was driven by the human), so instead we report the detector state at the takeover.
"""
import numpy as np

from load import L  # noqa (adds the repo to sys.path)
from jetformer import assist as A


def impact_series(d):
    """impact_g at each AI frame (max over IMU samples in (t-0.15, t])"""
    out = np.zeros(len(d.at))
    j0 = 0
    for k, t in enumerate(d.at):
        a = np.searchsorted(d.it, t - 0.15, side='right')
        b = np.searchsorted(d.it, t, side='right')
        a = max(a, b - 10)
        out[k] = d.hg[a:b].max() if b > a else 0.0
    return out


def replay(d, rev_us=300.0, rev_s=0.4, trace=False):
    ref = (1443.0, 1498.0)
    h = A.Assist(ref)
    ig = impact_series(d)
    kinds = np.zeros(len(d.at), int)
    outs = np.full((len(d.at), 2), np.nan)
    det = []                     # per frame: (armed, slow_since, impact_t, impact_moving, phase)
    for k, t in enumerate(d.at):
        er = float(d.erpm_at(t))
        if d.hold[k]:
            h.reset()
            det.append((False, None, None, False, None))
            continue
        now = t + (d.lat[k] / 1000.0 if np.isfinite(d.lat[k]) else 0.03)
        o, kind = h.step((float(d.mout[k, 0]), float(d.mout[k, 1])), er, bool(d.sel[k]), now,
                         esc_on=True, rec_on=True, impact_g=float(ig[k]), rev_us=rev_us, rev_s=rev_s)
        kinds[k] = kind
        outs[k] = o
        r = h.rec
        det.append((r.armed, r.slow_since, r.impact_t, r.impact_moving, r.phase))
    return kinds, outs, det, ig


if __name__ == '__main__':
    for log in ['20261010-203924', '20261010-204208', '20261010-204605']:
        d = L(log)
        kinds, outs, det, ig = replay(d)
        same = np.mean(kinds == d.assist)
        mism = np.nonzero(kinds != d.assist)[0]
        print(d.name, 'replay==log %.4f' % same, 'replay counts', np.bincount(kinds, minlength=4),
              'log counts', np.bincount(d.assist, minlength=4), 'mismatch t', [round(d.at[i], 2) for i in mism[:10]])
        dout = np.nanmax(np.abs(outs - d.out)[d.sel & ~d.hold])
        print('   max |replay out - logged out| on AI frames: %.2f us' % dout)
