"""Review check: AUROC of 'collision within 1 s' from course position alone (privileged input of the value head),
leave-one-log-out per-bin rates, on the 5 AI logs that have a course-map fit. Same frame filter as risk_baselines.py."""
import json, sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV  # noqa
from risk_baselines import auroc  # noqa  (module re-runs; guarded below)

LOGS = ['20261010-172910', '20261010-173445', '20261010-203924', '20261010-204208', '20261010-204605']
TR = '/home/jetson/jetformer_reports/overnight_20261011/course/tracks_%s.npz'
LAP = 25.48
BIN = 0.5
data = {}
for log in LOGS:
    d = L(log)
    tr = np.load(TR % log)
    ev = [e for e in d.events() if e['ai'] and not e['suspect']]
    et = np.array([e['t'] for e in ev])
    t = d.at; n = len(t)
    v = -d.erpm_at(t) * KV
    assist = d.assist if len(d.assist) == n else np.zeros(n, int)
    hold = d.hold if len(d.hold) == n else np.zeros(n, bool)
    keep = d.sel & (assist == 0) & ~hold & (v > 0.5)
    dt_next = np.array([np.min(et[et > x] - x, initial=np.inf) for x in t]) if len(et) else np.full(n, np.inf)
    dt_prev = np.array([np.min(x - et[et <= x], initial=np.inf) for x in t]) if len(et) else np.full(n, np.inf)
    keep &= dt_prev > 2.0
    y = dt_next <= 1.0
    # map to track samples by absolute time
    tus = d.t0 + (t * 1e6)
    j = np.clip(np.searchsorted(tr['t_us'], tus), 0, len(tr['t_us']) - 1)
    close = np.abs(tr['t_us'][j] - tus) < 50_000
    ok = keep & close & tr['valid'][j]
    pos = np.mod(tr['progress_cum_m'][j], LAP)
    b = np.floor(pos / BIN).astype(int)
    data[log] = {'b': b[ok], 'y': y[ok], 'v': v[ok], 'sec': tr['section'][j][ok]}
    del d
nb = int(np.ceil(LAP / BIN)) + 1
scores, ys, sp, secs = [], [], [], []
for log in LOGS:
    others = [data[o] for o in LOGS if o != log]
    cnt = np.zeros(nb); pos = np.zeros(nb)
    for o in others:
        np.add.at(cnt, o['b'], 1); np.add.at(pos, o['b'], o['y'])
    rate = (pos + 0.144 * 5) / (cnt + 5)            # shrink to the pooled rate
    scores += list(rate[data[log]['b']]); ys += list(data[log]['y']); sp += list(data[log]['v'])
y = np.array(ys, bool); s = np.array(scores); v = np.array(sp)
# position x speed: rank-normalised product
rankdata = lambda a: np.argsort(np.argsort(a, kind='mergesort'), kind='mergesort') + 1.0
comb = rankdata(s) / len(s) + rankdata(v) / len(v)
res = {'frames': int(len(y)), 'pos': int(y.sum()), 'auroc_position_LOLO': round(auroc(s, y), 3),
       'auroc_speed_same_frames': round(auroc(v, y), 3), 'auroc_position_plus_speed_rank': round(auroc(comb, y), 3),
       'bin_m': BIN}
json.dump(res, open('/home/jetson/jetformer_reports/overnight_20261011/research/review_src/position_baseline.json', 'w'), indent=1)
print(res)
