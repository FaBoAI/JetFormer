"""Geometric variants of the ToF guard (offline, numpy only) on ai_run_009/010/101 — aimed at the false alarms
from side walls / parking-bay boards entering the predicted path band.

TTC is recomputed per AI frame with a copy of ToFGuard.features() that exposes:
  hw_margin : half width of the path band (now 0.10 + 0.05 = 0.15 m)
  arcs      : 'both' (gyro arc OR steer arc, now) / 'gyro' / 'steer'
  vert_min  : rows of consistent distance needed for a 'standing object' (now 4)
  floor     : 'now' (data/tof_floor_ref.npy) or 'recal' (rebuilt from this run, ttc_<log>.npz:ref1)
Evaluation = tof.evaluate (same definitions as analysis.md section 6). Output tof_variants.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV  # noqa: E402
from tof import evaluate, collisions  # noqa: E402
from jetformer.tof_guard import ToFGuard, K_LEFT, K_RIGHT, K_MAX_L, K_MAX_R  # noqa: E402

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
S = '/home/jetson/jetformer_reports/overnight_20261011/runs/src/'
REF = '/home/jetson/jetformer/data/tof_floor_ref.npy'
LAPS = {'20261010-203924': 1.26, '20261010-204208': 3.88, '20261010-204605': 2.19}


def x_free(g, raw, erpm, gz_dps, steer_us, hw, arcs):
    raw = np.asarray(raw)[g.ROWS]
    d = (raw & 0x1FFF).astype(np.float32)
    d[d == 0x1FFF] = 0
    d *= 0.001
    conf = raw >> 13
    m = g.mask(d, conf)
    v = max(-float(erpm) * KV, 0.0)
    x = (d * g.fwd)[m]
    y = (d * g.left)[m] + g.Y_OFF
    k_gyro = np.clip(np.deg2rad(gz_dps) / max(v, 0.5), -2.0, 2.0)
    s = steer_us - g.center
    k_cmd = float(np.clip(s * (K_LEFT if s > 0 else K_RIGHT), -K_MAX_R, K_MAX_L))
    a = np.abs(y - 0.5 * k_gyro * x * x) < hw
    b = np.abs(y - 0.5 * k_cmd * x * x) < hw
    inpath = a | b if arcs == 'both' else (a if arcs == 'gyro' else b)
    xs = x[inpath]
    return float(np.partition(xs, g.K - 1)[g.K - 1]) if xs.size >= g.K else 9.0


def run(d, ref, hw=0.15, arcs='both', vert_min=4):
    g = ToFGuard(ref, center=1443.0, neutral=1498.0)
    g.VERT_MIN = vert_min
    n = len(d.at)
    xf = np.full(n, np.nan)
    ti = np.clip(np.searchsorted(d.tt, d.at, side='right') - 1, 0, len(d.tt) - 1)
    ii = np.clip(np.searchsorted(d.it, d.at, side='right') - 1, 0, len(d.it) - 1)
    er = d.erpm_at(d.at)
    for k in range(n):
        if d.hold[k] or not d.sel[k]:
            continue
        xf[k] = x_free(g, d.depth[ti[k]], er[k], float(d.gyr[ii[k], 2]), float(d.mout[k, 0]), hw, arcs)
    v = np.maximum(-er * KV, 0.0)
    return xf / np.maximum(v, 0.3)


VARIANTS = [
    ('now', dict()),
    ('band 0.12 m', dict(hw=0.12)),
    ('band 0.10 m', dict(hw=0.10)),
    ('gyro arc only', dict(arcs='gyro')),
    ('vert_min 6', dict(vert_min=6)),
    ('band 0.12 + gyro only', dict(hw=0.12, arcs='gyro')),
]


def main():
    agg = {}
    for log in LOGS:
        d = L(log, tof=True)
        tt = dict(np.load(S + 'ttc_%s.npz' % log))
        refs = {'now': np.load(REF), 'recal': tt['ref1']}
        coll = collisions(d)
        all_ev = d.events(2.0)
        for fl, ref in refs.items():
            for name, kw in VARIANTS:
                ttc = run(d, ref, **kw)
                if fl == 'now' and name == 'now':
                    ok = np.isfinite(ttc) & np.isfinite(d.ttc)
                    print(log, 'check vs logged ttc: warn agreement %.4f' % np.mean((ttc[ok] < 0.5) == (d.ttc[ok] < 0.5)))
                for tw in (0.5, 0.4):
                    ev = evaluate(d, np.nan_to_num(ttc, nan=9.0) < tw, coll=coll, all_ev=all_ev)
                    key = '%s | floor %s | TTC<%.1f' % (name, fl, tw)
                    a = agg.setdefault(key, {'n_coll': 0, 'caught03': 0, 'caught015': 0, 'fa': 0, 'minutes': 0.0,
                                             'warn_min': 0.0, 'leads': []})
                    a['n_coll'] += len(ev['collisions'])
                    a['caught03'] += sum(r['caught_0.3'] for r in ev['collisions'])
                    a['caught015'] += sum(r['caught_0.15'] for r in ev['collisions'])
                    a['fa'] += ev['false_alarms']
                    a['minutes'] += ev['minutes']
                    a['warn_min'] += ev['warn_time_ratio'] * ev['minutes']
                    a['leads'] += [r['lead_s'] for r in ev['collisions'] if r['lead_s'] is not None]
        print('done', log)
    rows = []
    for k, a in agg.items():
        r = {'variant': k, 'caught_0.3': '%d/%d' % (a['caught03'], a['n_coll']),
             'caught_0.15': '%d/%d' % (a['caught015'], a['n_coll']),
             'lead_median_s': round(float(np.median(a['leads'])), 2) if a['leads'] else None,
             'fa': a['fa'], 'fa_per_min': round(a['fa'] / a['minutes'], 1),
             'warn_time_frac': round(a['warn_min'] / a['minutes'], 3)}
        rows.append(r)
        print(r)
    json.dump(rows, open(S + 'tof_variants.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
