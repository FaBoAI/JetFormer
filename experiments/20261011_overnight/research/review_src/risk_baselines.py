"""Review check (read-only): how easy is the gate 'AUROC >= 0.8 for collision within 1 s'?
Baselines that need no learned features: forward speed only, ToF TTC (logged AI_CMD.ttc), and a crude
'speed + gyro' score. Also per-lap overdispersion of AI impacts (for the power table).
AI frames: select=True, assist==0, not hold, forward speed > 0.5 m/s, not within 2 s after an impact.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV  # noqa

LOGS = ['20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
        '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
        '20261010-204605']


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, bool)
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def main():
    rows = {'speed': [], 'ttc': [], 'speed_gyro': [], 'y': [], 'log': []}
    disp = {}
    for log in LOGS:
        try:
            d = L(log)
        except Exception as e:  # noqa
            print('skip', log, e); continue
        ev = [e for e in d.events() if e['ai'] and not e['suspect']]
        et = np.array([e['t'] for e in ev])
        t = d.at
        v = -d.erpm_at(t) * KV
        gz = np.interp(t, d.it, np.abs(d.gyr[:, 2]))
        n = len(t)
        assist = d.assist if len(d.assist) == n else np.zeros(n, int)
        hold = d.hold if len(d.hold) == n else np.zeros(n, bool)
        keep = d.sel & (assist == 0) & ~hold & (v > 0.5)
        if len(et):
            dt_next = np.array([np.min(et[et > x] - x, initial=np.inf) for x in t])
            dt_prev = np.array([np.min(x - et[et <= x], initial=np.inf) for x in t])
        else:
            dt_next = np.full(len(t), np.inf); dt_prev = np.full(len(t), np.inf)
        keep &= dt_prev > 2.0
        y = dt_next <= 1.0
        ttc = d.ttc.copy() if len(d.ttc) == n else np.full(n, np.nan)
        has_ttc = np.isfinite(ttc).any()
        ttc[~np.isfinite(ttc)] = 99.0
        if not has_ttc: ttc[:] = np.nan
        rows['speed'] += list(v[keep]); rows['ttc'] += list(-np.minimum(ttc[keep], 99.0))
        rows['speed_gyro'] += list((v * (1 + gz / 200.0))[keep])
        rows['y'] += list(y[keep]); rows['log'] += [log] * int(keep.sum())
        # overdispersion: AI heading-lap windows (gyro z integrated on AI side)
        ai_i = d.sel_at(d.it)
        dth = np.zeros(len(d.it)); dth[1:] = np.diff(d.it) * d.gyr[1:, 2] * ai_i[1:]
        lap_idx = np.floor(np.abs(np.cumsum(dth)) / 360.0).astype(int)
        ev_lap = np.array([lap_idx[min(e['i'], len(lap_idx) - 1)] for e in ev], int)
        nl = int(lap_idx.max()) if len(lap_idx) else 0
        if nl >= 3:
            cnt = np.bincount(ev_lap[ev_lap < nl], minlength=nl)[:nl]
            disp[d.name] = {'full_laps': nl, 'mean': round(float(cnt.mean()), 2), 'var': round(float(cnt.var(ddof=1)), 2),
                            'var_over_mean': round(float(cnt.var(ddof=1) / max(cnt.mean(), 1e-9)), 2),
                            'counts': cnt.tolist()}
        del d

    y = np.array(rows['y'])
    res = {'frames': int(len(y)), 'positives': int(y.sum()), 'pos_rate': round(float(y.mean()), 3),
           'logs': sorted(set(rows['log'])),
           'auroc_speed': round(auroc(rows['speed'], y), 3),
           'auroc_ttc': round(auroc(rows['ttc'], y), 3),
           'auroc_speed_gyro': round(auroc(rows['speed_gyro'], y), 3),
           'per_log_auroc_speed': {}, 'overdispersion': disp}
    lg = np.array(rows['log'])
    for log in sorted(set(rows['log'])):
        m = lg == log
        res['per_log_auroc_speed'][log] = {'n': int(m.sum()), 'pos': int(y[m].sum()),
                                           'auroc_speed': round(auroc(np.array(rows['speed'])[m], y[m]), 3),
                                           'auroc_ttc': round(auroc(np.array(rows['ttc'])[m], y[m]), 3)}
    json.dump(res, open('/home/jetson/jetformer_reports/overnight_20261011/research/review_src/risk_baselines.json', 'w'),
              indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ('per_log_auroc_speed',)}, ensure_ascii=False)[:3000])
    print(json.dumps(res['per_log_auroc_speed'])[:2000])


if __name__ == '__main__':
    main()
