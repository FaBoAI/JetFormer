"""Independent re-check (2026-10-11 06:xx, after the Jetson reboot) of the main numbers for ai_run_009/010/101.

- AI segments / takeovers (select from AI_CMD; a select drop that coincides with a board reboot is not a takeover)
- LapCounter (jetformer.laps) fed with the IMU samples exactly as drive.py does (gz deg/s, erpm of the latest
  VEHICLE sample, ai = latest AI_CMD.select, impact_g = hypot(ax, ay)/g); recovery noted on assist==2 frames
- impacts (horizontal >= 2.5 g, merged 0.6 s), stuck events, assist==1 episodes, guard onsets, holds, gaps
Output: verify.json next to this file.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, seg_mask, KV, NEUTRAL, G  # noqa: E402
from jetformer.laps import LapCounter  # noqa: E402

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
OLD = ['20261010-173445', '20261010-181504']
OUT = '/home/jetson/jetformer_reports/overnight_20261011/runs/src/verify.json'


def board_resets(d):
    k = np.nonzero(np.diff(d.board) < 0)[0]
    return [float(d.vt[i + 1]) for i in k]


def laps(d):
    lc = LapCounter()
    vi = np.clip(np.searchsorted(d.vt, d.it, side='right') - 1, 0, len(d.vt) - 1)
    ai_i = np.searchsorted(d.at, d.it, side='right') - 1
    rec_t = d.at[d.assist == 2] if len(d.assist) == len(d.at) else np.zeros(0)
    rj = 0
    out = []
    for k in range(len(d.it)):
        t = float(d.it[k])
        if ai_i[k] < 0 or t > d.at[-1] + 0.1:
            continue            # drive loop not running
        while rj < len(rec_t) and rec_t[rj] <= t:
            lc.note('recovery'); rj += 1
        lap = lc.add(t, float(d.gyr[k, 2]), float(d.erpm[vi[k]]), bool(d.sel[ai_i[k]]), float(d.hg[k]))
        if lap:
            lap = dict(lap)
            lap['t_start'] = round(lap['t_end'] - lap['time_s'], 2)
            out.append(lap)
    s = lc.summary()
    return out, s


def stuck(d, segs):
    """AI side, after |eRPM|>=1000 in this segment, -300<eRPM<300 for >=0.25 s"""
    res = []
    for a, b in segs:
        t0, t1 = d.at[a], d.at[b]
        m = (d.vt >= t0) & (d.vt <= t1 + 0.034)
        idx = np.nonzero(m)[0]
        armed = False
        k = 0
        while k < len(idx):
            i = idx[k]
            if abs(d.erpm[i]) >= 1000:
                armed = True
            if armed and -300 < d.erpm[i] < 300:
                j = k
                while j + 1 < len(idx) and -300 < d.erpm[idx[j + 1]] < 300:
                    j += 1
                dur = d.vt[idx[j]] - d.vt[i]
                if dur >= 0.25:
                    res.append({'t': round(float(d.vt[i]), 2), 'dur_s': round(float(dur), 2),
                                'end': 'takeover' if j == len(idx) - 1 else 'ai_moved_on'})
                k = j + 1
                continue
            k += 1
    return res


def episodes(mask, t, gap=0.0):
    eps = []
    for a, b in seg_mask(mask):
        if eps and t[a] - t[eps[-1][1]] <= gap:
            eps[-1] = (eps[-1][0], b)
        else:
            eps.append((a, b))
    return eps


def main():
    allr = {}
    for log in LOGS + OLD:
        d = L(log)
        r = {'name': d.name, 'duration_s': round(float(d.at[-1]), 2), 'frames': int(len(d.at))}
        resets = board_resets(d)
        r['board_resets_t'] = [round(x, 2) for x in resets]
        segs = seg_mask(d.sel)
        dt = np.clip(np.diff(d.at, append=d.at[-1] + 1 / 30), 0, 0.1)
        r['ai_time_s'] = round(float(np.sum(dt[d.sel])), 1)
        r['ai_time_nohold_s'] = round(float(np.sum(dt[d.sel & ~d.hold])), 1)
        seginfo = []
        for a, b in segs:
            t0, t1 = float(d.at[a]), float(d.at[b])
            if b + 1 >= len(d.at):
                end = 'end_of_log'
            elif any(t1 - 0.3 <= x <= t1 + 0.3 for x in resets):
                end = 'board_reset'
            else:
                end = 'takeover'
            g = d.gains[a:b + 1]
            seginfo.append({'t0': round(t0, 2), 't1': round(t1, 2), 'dur_s': round(t1 - t0 + 1 / 30, 2), 'end': end,
                            'steer_gain': sorted(set(np.round(g[:, 0], 2).tolist())),
                            'throttle_gain': sorted(set(np.round(g[:, 1], 2).tolist())),
                            'lookahead': sorted(set(d.la[a:b + 1].tolist())), 'hold_frames': int(d.hold[a:b + 1].sum())})
        r['segments'] = seginfo
        r['takeovers'] = sum(s['end'] == 'takeover' for s in seginfo)
        # distance on AI side
        vi = np.clip(np.searchsorted(d.at, d.vt, side='right') - 1, 0, len(d.at) - 1)
        av = d.sel[vi] & (d.vt >= d.at[0])
        dv = np.clip(np.diff(d.vt, append=d.vt[-1]), 0, 0.2)
        r['ai_dist_m'] = round(float(np.sum(np.maximum(-d.erpm, 0)[av] * KV * dv[av])), 1)
        # impacts
        ev = d.events(2.5)
        evai = [e for e in ev if e['ai'] and not e['suspect']]
        r['impacts_ai'] = len(evai)
        r['impacts_ai_suspect'] = sum(1 for e in ev if e['ai'] and e['suspect'])
        r['impacts_all'] = len([e for e in ev if not e['suspect']])
        r['impacts_ai_list'] = [{'t': round(e['t'], 2), 'g': round(e['g'], 1), 'v': round(-e['erpm_before'] * KV, 2)}
                                for e in evai]
        # laps
        lp, s = laps(d)
        r['laps'] = lp
        r['lap_summary'] = {k: s[k] for k in ('laps', 'best_ai_s', 'median_ai_s', 'clean_laps', 'ai_laps',
                                               'median_ai_all_s', 'direction')}
        r['stuck'] = stuck(d, segs)
        # assist episodes
        ae = []
        if len(d.assist) != len(d.at):
            d.assist = np.zeros(len(d.at), int)
        for a, b in seg_mask(d.assist == 1):
            t = float(d.at[a])
            prev = [e for e in ev if t - 1.5 <= e['t'] <= t]
            ae.append({'t0': round(t, 2), 't1': round(float(d.at[b]), 2), 'frames': int(b - a + 1),
                       'model_thr_us_max': round(float(np.max(d.mout[a:b + 1, 1]) - NEUTRAL)),
                       'out_thr_us': round(float(np.median(d.out[a:b + 1, 1]) - NEUTRAL)),
                       'erpm': int(d.erpm_at(t)), 'impact_before_s': round(t - prev[-1]['t'], 2) if prev else None,
                       'ai_seg_end_after_s': round(next(float(d.at[y]) for x, y in segs if x <= a <= y) - t, 2)})
        r['esc_assist'] = ae
        r['assist_counts'] = {int(k): int(v) for k, v in zip(*np.unique(d.assist, return_counts=True))}
        # guard onsets (logged) on AI side
        v = np.maximum(-d.erpm_at(d.at) * KV, 0)
        base = d.sel & ~d.hold
        fwd_min = float(np.sum(dt[base & (v > 0.5)]) / 60)
        if len(d.guard) != len(d.at):
            d.guard = np.zeros(len(d.at), int); d.gapplied = np.zeros(len(d.at), bool)
        if len(d.assist) != len(d.at):
            d.assist = np.zeros(len(d.at), int)
        g1 = episodes((d.guard >= 1) & base, d.at)
        g2 = episodes((d.guard >= 2) & base, d.at)
        r['guard'] = {'ai_forward_min': round(fwd_min, 3), 'warn_onsets': len(g1), 'brake_onsets': len(g2),
                      'warn_per_min': round(len(g1) / fwd_min, 1) if fwd_min else None,
                      'brake_per_min': round(len(g2) / fwd_min, 1) if fwd_min else None,
                      'warn_time_frac_fwd': round(float(np.sum(dt[(d.guard >= 1) & base & (v > 0.5)]) / 60 / fwd_min), 3),
                      'applied_frames': int(d.gapplied.sum()),
                      'human_side_warn_onsets': len(episodes((d.guard >= 1) & ~d.sel & ~d.hold, d.at))}
        # holds and gaps
        he = []
        for a, b in seg_mask(d.hold):
            he.append({'t0': round(float(d.at[a]), 2), 't1': round(float(d.at[b]), 2), 'frames': int(b - a + 1),
                       'select': bool(d.sel[a])})
        r['hold_episodes'] = he
        cg = np.diff(d.ct)
        r['cam_gaps'] = [{'t': round(float(d.ct[i]), 2), 'gap_s': round(float(cg[i]), 3),
                          'ai': bool(d.sel_at(d.ct[i]))} for i in np.nonzero(cg > 0.1)[0]]
        ag = np.diff(d.at)
        r['aicmd_gaps'] = [{'t': round(float(d.at[i]), 2), 'gap_s': round(float(ag[i]), 3), 'ai': bool(d.sel[i])}
                           for i in np.nonzero(ag > 0.1)[0]]
        vg = np.diff(d.vt)
        r['vehicle_gaps'] = [{'t': round(float(d.vt[i]), 2), 'gap_s': round(float(vg[i]), 3)}
                             for i in np.nonzero(vg > 0.1)[0]]
        r['lat_ms_p50_p99'] = [round(float(np.nanmedian(d.lat[base])), 1), round(float(np.nanpercentile(d.lat[base], 99)), 1)]
        allr[log] = r
        print('=====', log, d.name, json.dumps({k: r[k] for k in ('duration_s', 'ai_time_s', 'ai_time_nohold_s',
                                                                    'takeovers', 'ai_dist_m', 'impacts_ai',
                                                                    'impacts_ai_suspect', 'board_resets_t',
                                                                    'lap_summary', 'assist_counts', 'guard',
                                                                    'lat_ms_p50_p99')}, ensure_ascii=False))
        for s_ in seginfo:
            print('   seg', s_)
        for x in lp:
            print('   lap', x)
        print('   stuck', r['stuck'])
        print('   esc', r['esc_assist'])
        print('   holds', he[:10], '... n', len(he))
        print('   camgaps', r['cam_gaps'][:20])
        print('   aicmd gaps', r['aicmd_gaps'][:20])
        print('   veh gaps', r['vehicle_gaps'][:20])
    json.dump(allr, open(OUT, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
