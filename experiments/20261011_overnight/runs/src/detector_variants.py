"""Would recovery have started before the human took over? Detector variants replayed on the logged AI frames.

Only the detection part is replayed (the car's later motion was the human's, so nothing after the takeover is
simulated). For each AI segment we report the first detection time (if any) and compare it with the takeover.
False-trigger check: detections where the car moved forward again by itself (eRPM < -1500) within 1.0 s after
the detection time while still on the AI side (the AI would have recovered alone).
"""
import json

import numpy as np

from load import L, seg_mask
from replay import impact_series
from jetformer import assist as A

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605', '20261010-172910', '20261010-173445']


class Det(A.Recovery):
    def __init__(self, ref, after_imp_s=0.5, plain_s=0.8, bounce_counts=False):
        super().__init__(ref)
        self.STUCK_AFTER_IMPACT_S = after_imp_s
        self.STUCK_S = plain_s
        self.bounce = bounce_counts

    def _detect(self, model_us, erpm, now, impact_g):
        if not self.bounce:
            return super()._detect(model_us, erpm, now, impact_g)
        # variant: after a moving impact, backward roll (bounce) also counts as "not progressing"
        self.history.append((now, model_us[0]))
        if impact_g >= self.IMPACT_G:
            self.impact_t, self.impact_moving = now, erpm < -500 or self.impact_moving
        if self.impact_t is not None and now - self.impact_t > 1.2 + self.STUCK_AFTER_IMPACT_S:
            self.impact_t, self.impact_moving = None, False
        if abs(erpm) >= self.ARM_ERPM and erpm < 0:
            self.armed = True
        if not self.armed or now < self.cooldown_until:
            self.slow_since = None
            return False
        after_imp = self.impact_t is not None and self.impact_moving
        slow = (erpm > -self.STOP_ERPM) if after_imp else (-self.STOP_ERPM < erpm < A.BACKING_ERPM)
        if slow:
            if self.slow_since is None:
                self.slow_since = now
            need = self.STUCK_AFTER_IMPACT_S if after_imp else self.STUCK_S
            return now - self.slow_since >= need
        self.slow_since = None
        return False


VARIANTS = {
    'v2 now (impact 0.5 s / plain 0.8 s)': dict(after_imp_s=0.5, plain_s=0.8, bounce_counts=False),
    'bounce counts (0.5 / 0.8)': dict(after_imp_s=0.5, plain_s=0.8, bounce_counts=True),
    'bounce counts (0.4 / 0.6)': dict(after_imp_s=0.4, plain_s=0.6, bounce_counts=True),
    'bounce counts (0.3 / 0.5)': dict(after_imp_s=0.3, plain_s=0.5, bounce_counts=True),
    'no bounce (0.3 / 0.5)': dict(after_imp_s=0.3, plain_s=0.5, bounce_counts=False),
}


def detections(d, ig, **kw):
    det = Det((1443.0, 1498.0), **kw)
    mo = d.mout if d.mout.ndim == 2 else d.out          # 006/007 have no model_out
    out = []
    for i0, i1 in seg_mask(d.sel & ~d.hold):
        det.reset()
        for k in range(i0, i1 + 1):
            t = d.at[k]
            if det._detect((float(mo[k, 0]), float(mo[k, 1])), float(d.erpm_at(t)), t, float(ig[k])):
                out.append((float(t), float(d.at[i1]), i1 + 1 < len(d.at)))
                det.slow_since = None
                det.cooldown_until = t + 3.0          # one detection per stop
                det.impact_t, det.impact_moving = None, False
    return out


if __name__ == '__main__':
    res = {}
    for log in LOGS:
        d = L(log)
        ig = impact_series(d)
        for name, kw in VARIANTS.items():
            ds = detections(d, ig, **kw)
            rows = []
            for t, seg_end, takeover in ds:
                # did the AI move forward by itself within 1.0 s (only while still on the AI side)?
                m = (d.vt > t) & (d.vt < min(t + 1.0, seg_end))
                self_rec = bool(np.any(d.erpm[m] < -1500))
                rows.append({'t': round(t, 2), 'seg_end': round(seg_end, 2), 'before_takeover_s': round(seg_end - t, 2),
                             'ai_moved_forward_within_1s': self_rec})
            res.setdefault(log, {})[name] = rows
            print(d.name, '%-36s detections %2d  self-moved-after %d  times %s' % (
                name, len(rows), sum(r['ai_moved_forward_within_1s'] for r in rows), [r['t'] for r in rows]))
    json.dump(res, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/detector_variants.json', 'w'), indent=1)
