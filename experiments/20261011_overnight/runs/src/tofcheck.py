"""Independent re-check of the logged ToF guard (mode 1 = log only) on ai_run_009/010/101, using only the logged
AI_CMD fields (ttc, guard). Also a floor-drift check (raw depth vs data/tof_floor_ref.npy).

warning episode: AI side, not hold, v > 0.5 m/s, guard >= 1 (or >= 2); gaps <= 0.3 s merged
collision: AI-side impact, horizontal >= 2.5 g, v_before > 0.5 m/s
lead: impact time - onset of the episode that is on within the 0.3 s before the impact
false alarm: episode onset with no >= 2.0 g event (any side) within [onset-0.05, onset+1.5 s]
exposure: AI forward (v > 0.5) frames, minus 1 s after each >= 2.0 g event
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, seg_mask, KV  # noqa: E402

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
REF = '/home/jetson/jetformer/data/tof_floor_ref.npy'


def eps(mask, t, gap=0.3):
    out = []
    for a, b in seg_mask(mask):
        if out and t[a] - t[out[-1][1]] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def check(d):
    v = np.maximum(-d.erpm_at(d.at) * KV, 0)
    base = d.sel & ~d.hold & (v > 0.5)
    dt = np.clip(np.diff(d.at, append=d.at[-1] + 1 / 30), 0, 0.1)
    ev2 = d.events(2.0)
    coll = [e for e in d.events(2.5) if e['ai'] and not e['suspect'] and -e['erpm_before'] * KV > 0.5]
    expo = base.copy()
    for e in ev2:
        expo &= ~((d.at > e['t']) & (d.at < e['t'] + 1.0))
    minutes = float(np.sum(dt[expo]) / 60)
    res = {'exposure_min': round(minutes, 3)}
    for lvl, name in ((1, 'warn'), (2, 'brake')):
        E = eps((d.guard >= lvl) & base, d.at)
        rows = []
        for e in coll:
            t = e['t']
            lead = None
            for a, b in E:
                if d.at[a] <= t and d.at[b] >= t - 0.3:
                    lead = t - d.at[a]
            m = (d.at >= t - 1.0) & (d.at <= t)
            rows.append({'t': round(t, 2), 'g': round(e['g'], 1), 'v': round(-e['erpm_before'] * KV, 2),
                         'lead_s': None if lead is None else round(lead, 2),
                         'ttc_min_1s': round(float(np.nanmin(d.ttc[m])), 2)})
        fa = [(a, b) for a, b in E if not any(d.at[a] - 0.05 <= x['t'] <= d.at[a] + 1.5 for x in ev2)]
        warn_s = float(np.sum(dt[(d.guard >= lvl) & expo]))
        res[name] = {'episodes': len(E), 'episodes_per_min': round(len(E) / minutes, 1),
                     'false_alarms': len(fa), 'fa_per_min': round(len(fa) / minutes, 1),
                     'fa_list': [{'t': round(float(d.at[a]), 2), 'dur_s': round(float(d.at[b] - d.at[a] + 1 / 30), 2),
                                  'ttc_min': round(float(np.nanmin(d.ttc[a:b + 1])), 2),
                                  'v': round(float(v[a]), 2)} for a, b in fa],
                     'warn_time_frac': round(warn_s / 60 / minutes, 3),
                     'caught_0.3': sum(1 for r in rows if r['lead_s'] is not None and r['lead_s'] >= 0.3),
                     'caught_0.15': sum(1 for r in rows if r['lead_s'] is not None and r['lead_s'] >= 0.15),
                     'n_coll': len(rows), 'collisions': rows}
    return res


def floor_drift(d):
    """median over straight fast frames of depth / ref for rows 9 and 18, centre columns 20-34"""
    ref = np.load(REF).astype(float)
    raw = d.depth
    dd = (raw & 0x1FFF).astype(float)
    conf = raw >> 13
    er = d.erpm_at(d.tt)
    gz = np.interp(d.tt, d.it, d.gyr[:, 2])
    m = (er < -1500) & (np.abs(gz) < 40)
    out = {}
    for r in (9, 18):
        x = np.where(conf[m, r, 20:35] == 4, dd[m, r, 20:35], np.nan)
        ratio = np.nanmedian(x, axis=1) / np.median(ref[r, 20:35])
        out['row%d_pct' % r] = round(float((np.nanmedian(ratio) - 1) * 100), 1)
    out['frames'] = int(m.sum())
    return out


if __name__ == '__main__':
    allr = {}
    for log in sys.argv[1:] or LOGS:
        d = L(log, tof=True)
        r = check(d) if len(d.guard) == len(d.at) else {}
        r['floor'] = floor_drift(d)
        allr[log] = r
        print('=====', log, d.name, 'exposure %.3f min' % r.get('exposure_min', 0), 'floor', r['floor'])
        for k in ('warn', 'brake'):
            if k in r:
                x = r[k]
                print('  %s: episodes %d (%.1f/min) FA %d (%.1f/min) warn-time %.1f%% caught0.3 %d/%d caught0.15 %d/%d' % (
                    k, x['episodes'], x['episodes_per_min'], x['false_alarms'], x['fa_per_min'],
                    100 * x['warn_time_frac'], x['caught_0.3'], x['n_coll'], x['caught_0.15'], x['n_coll']))
                for c in x['collisions']:
                    print('     ', c)
                print('     FA', x['fa_list'])
    if not sys.argv[1:]:
        json.dump(allr, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/tofcheck.json', 'w'), indent=1)
