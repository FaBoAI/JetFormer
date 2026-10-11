"""Threshold sweep for the ToF guard + lap-time cost estimates (open-loop and self-limiting)."""
import json

import numpy as np

from load import L, KV
from tof import evaluate, collisions, frame_speed, A_BRAKE, A_COAST
from basic import run_numbers

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
OUT = '/home/jetson/jetformer_reports/overnight_20261011/runs/src/'


def accel_at_frames(d, delay=0.15):
    vs = -d.erpm * KV
    vsm = np.convolve(vs, np.ones(15) / 15, 'same')
    acc = np.gradient(vsm, d.vt)
    return np.interp(d.at + delay, d.vt, acc)


def cost_selflimit(d, x_free, t_warn=0.5, t_brake=0.3, a_r=2.0, nconsec=1, vmin=0.5):
    """counterfactual speed v_cf with the guard acting on x_free/v_cf (x_free from the log, indexed by time)."""
    v = frame_speed(d)
    a_act = accel_at_frames(d)
    dt = np.clip(np.diff(d.at, append=d.at[-1] + 1 / 30), 0, 0.1)
    base = d.sel & ~d.hold
    vcf = None
    loss = 0.0
    run = 0
    for i in range(len(d.at)):
        if not base[i] or not np.isfinite(x_free[i]):
            vcf = None
            run = 0
            continue
        if vcf is None:
            vcf = v[i]
        ttc = x_free[i] / max(vcf, 0.3)
        w = ttc < t_warn and vcf >= vmin
        run = run + 1 if w else 0
        act = run >= nconsec
        if act and ttc < t_brake:
            vcf = max(vcf - A_BRAKE * dt[i], 0.0)
        elif act:
            u = min(1.0, (t_warn - ttc) / (t_warn - t_brake))
            a_cf = (1 - u) * a_act[i] + u * (-A_COAST)
            vcf = max(vcf + min(a_cf, a_act[i]) * dt[i], 0.0)
        else:
            # follow the actual speed profile, catching up at most a_r faster than the actual accel
            if vcf < v[i]:
                dv_act = max(v[i] - v[i - 1], 0.0) if i > 0 else 0.0
                vcf = min(v[i], vcf + dv_act + a_r * dt[i])
            else:
                vcf = v[i]
        vcf = min(vcf, v[i])
        if v[i] > 0.3:
            loss += (v[i] - vcf) * dt[i] / v[i]
    return loss


def rules(d, tt):
    v = frame_speed(d)
    t0, t1 = tt['ttc0'], tt['ttc1']

    def consec(w, n):
        c = np.zeros(len(w), int)
        r = 0
        for i, x in enumerate(w):
            r = r + 1 if x else 0
            c[i] = r
        return c >= n
    return {
        'logged: TTC<0.5 (now)': (t0 < 0.5, tt['x0'], 0.5, 1),
        'TTC<0.4': (t0 < 0.4, tt['x0'], 0.4, 1),
        'TTC<0.3': (t0 < 0.3, tt['x0'], 0.3, 1),
        'TTC<0.5 x2 frames': (consec(t0 < 0.5, 2), tt['x0'], 0.5, 2),
        'TTC<0.4 x2 frames': (consec(t0 < 0.4, 2), tt['x0'], 0.4, 2),
        'recal floor TTC<0.5': (t1 < 0.5, tt['x1'], 0.5, 1),
        'recal floor TTC<0.4': (t1 < 0.4, tt['x1'], 0.4, 1),
        'recal floor TTC<0.4 x2': (consec(t1 < 0.4, 2), tt['x1'], 0.4, 2),
        'recal floor TTC<0.35 x2': (consec(t1 < 0.35, 2), tt['x1'], 0.35, 2),
    }


if __name__ == '__main__':
    agg = {}
    per = {}
    laps = {}
    for log in LOGS:
        d = L(log)
        _, bn = run_numbers(log)
        laps[log] = bn['ai_heading_laps']
        tt = dict(np.load(OUT + 'ttc_%s.npz' % log))
        coll = collisions(d)
        all_ev = d.events(2.0)
        for name, (w, xf, tw, nc) in rules(d, tt).items():
            ev = evaluate(d, w, coll=coll, all_ev=all_ev)
            tb = min(0.3, tw - 0.05)
            c_sl = cost_selflimit(d, xf, t_warn=tw, t_brake=tb, nconsec=nc)
            leads = [r['lead_s'] for r in ev['collisions'] if r['lead_s'] is not None]
            row = {'n_coll': len(coll), 'caught03': sum(r['caught_0.3'] for r in ev['collisions']),
                   'caught015': sum(r['caught_0.15'] for r in ev['collisions']), 'leads': leads,
                   'fa': ev['false_alarms'], 'episodes': ev['episodes'], 'minutes': ev['minutes'],
                   'warn_s': ev['warn_time_ratio'] * ev['minutes'] * 60, 'safe_min': ev['minutes'],
                   'warn_ratio': ev['warn_time_ratio'], 'cost_sl_s': c_sl}
            per.setdefault(log, {})[name] = row
            a = agg.setdefault(name, {'n_coll': 0, 'caught03': 0, 'caught015': 0, 'leads': [], 'fa': 0, 'minutes': 0.0,
                                      'cost_sl_s': 0.0, 'warn_num': 0.0, 'laps': 0.0})
            for k in ('n_coll', 'caught03', 'caught015', 'fa', 'minutes', 'cost_sl_s'):
                a[k] += row[k]
            a['leads'] += leads
            a['warn_num'] += ev['warn_time_ratio'] * ev['minutes']
            a['laps'] += laps[log]
    print('pooled (AI side, 3 runs):')
    table = []
    for name, a in agg.items():
        r = {'rule': name, 'caught_0.3': '%d/%d' % (a['caught03'], a['n_coll']),
             'caught_0.15': '%d/%d' % (a['caught015'], a['n_coll']),
             'lead_median_s': round(float(np.median(a['leads'])), 2) if a['leads'] else None,
             'fa_per_min': round(a['fa'] / a['minutes'], 1), 'fa': a['fa'], 'minutes': round(a['minutes'], 2),
             'warn_ratio_approx': round(a['warn_num'] / a['minutes'], 3),
             'cost_selflimit_s_per_lap': round(a['cost_sl_s'] / a['laps'], 2)}
        table.append(r)
        print(r)
    for log in LOGS:
        print(log)
        for name, r in per[log].items():
            print('   %-26s caught %d/%d FA %d (%.1f/min) warn %.1f%% cost_sl %.2f s' % (
                name, r['caught03'], r['n_coll'], r['fa'], r['fa'] / max(r['minutes'], 1e-9), 100 * r['warn_ratio'],
                r['cost_sl_s']))
    json.dump({'pooled': table, 'per_run': {k: {n: {kk: (vv if kk != 'leads' else [round(x, 2) for x in vv])
                                                     for kk, vv in r.items()} for n, r in v.items()}
                                            for k, v in per.items()}, 'ai_heading_laps': laps},
              open(OUT + 'tof_sweep.json', 'w'), indent=1, ensure_ascii=False, default=float)


def split_cost(d, w, x_free, t_warn, t_brake, all_ev, coll):
    """cost of the guard restricted to false-alarm episodes (true warnings excluded) — self-limiting model"""
    ev = evaluate(d, w, coll=coll, all_ev=all_ev)
    keep = np.zeros(len(d.at), bool)
    for a, b in ev['fa']:
        keep[a:b + 1] = True
    xf = np.where(keep, x_free, 99.0)       # outside FA episodes: no obstacle
    return cost_selflimit(d, xf, t_warn=t_warn, t_brake=t_brake), ev


def vcf_trace(d, x_free, t_warn=0.5, t_brake=0.3, a_r=2.0):
    """same model as cost_selflimit, returns the counterfactual speed per frame"""
    v = frame_speed(d)
    a_act = accel_at_frames(d)
    dt = np.clip(np.diff(d.at, append=d.at[-1] + 1 / 30), 0, 0.1)
    base = d.sel & ~d.hold
    out = np.full(len(d.at), np.nan)
    vcf = None
    for i in range(len(d.at)):
        if not base[i] or not np.isfinite(x_free[i]):
            vcf = None
            continue
        if vcf is None:
            vcf = v[i]
        ttc = x_free[i] / max(vcf, 0.3)
        if ttc < t_warn and vcf >= 0.5 and ttc < t_brake:
            vcf = max(vcf - A_BRAKE * dt[i], 0.0)
        elif ttc < t_warn and vcf >= 0.5:
            u = min(1.0, (t_warn - ttc) / (t_warn - t_brake))
            vcf = max(vcf + min((1 - u) * a_act[i] + u * (-A_COAST), a_act[i]) * dt[i], 0.0)
        elif vcf < v[i]:
            dv_act = max(v[i] - v[i - 1], 0.0) if i > 0 else 0.0
            vcf = min(v[i], vcf + dv_act + a_r * dt[i])
        else:
            vcf = v[i]
        vcf = min(vcf, v[i])
        out[i] = vcf
    return out
