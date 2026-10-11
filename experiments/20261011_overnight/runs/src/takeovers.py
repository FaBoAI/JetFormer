"""Per-takeover / per-impact analysis: why recovery v2 never fired."""
import json

import numpy as np

from load import L, seg_mask, KV, NEUTRAL
from replay import replay

# visual location labels (checked on fig/impacts_*.jpg; see analysis.md)
LOC = {
    ('20261010-203924', 15.16): 'LH', ('20261010-203924', 19.23): 'LH', ('20261010-203924', 23.98): 'GM_in',
    ('20261010-203924', 45.80): 'HS_P', ('20261010-203924', 61.56): 'DE', ('20261010-203924', 65.72): 'HS_P',
    ('20261010-204208', 16.89): 'GM_out', ('20261010-204208', 28.24): 'DE', ('20261010-204208', 29.98): 'DE2',
    ('20261010-204208', 36.42): 'GM_in', ('20261010-204208', 38.49): 'DE', ('20261010-204208', 40.59): 'DE2',
    ('20261010-204208', 65.85): 'DE', ('20261010-204208', 75.67): 'DE',
    ('20261010-204605', 12.51): 'LH', ('20261010-204605', 25.81): 'DE', ('20261010-204605', 27.58): 'DE2',
    ('20261010-204605', 32.08): 'R1', ('20261010-204605', 34.96): 'DE', ('20261010-204605', 37.07): 'DE2',
    ('20261010-204605', 42.48): 'LH',
}


def loc(log, t):
    for (l, tt), v in LOC.items():
        if l == log and abs(tt - t) < 0.05:
            return v
    return '?'


def board_resets(d):
    k = np.nonzero(np.diff(d.board) < 0)[0]
    return [float(d.vt[i + 1]) for i in k]


def analyze(log):
    d = L(log)
    kinds, outs, det, ig = replay(d)
    ev = [e for e in d.events(2.5)]
    resets = board_resets(d)
    segs = seg_mask(d.sel)
    res = {'log': log, 'name': d.name, 'segments': [], 'impacts': []}
    for e in ev:
        if not e['ai']:
            continue
        t = e['t']
        after = (d.vt > t) & (d.vt < t + 1.5)
        # first time eRPM > -300 (stopped/bounced) after the impact
        k = np.nonzero(after & (d.erpm > -300))[0]
        t_stop = float(d.vt[k[0]]) if len(k) else None
        e05 = float(np.median(d.erpm[(d.vt > t + 0.4) & (d.vt < t + 0.6)])) if ((d.vt > t + 0.4) & (d.vt < t + 0.6)).any() else np.nan
        # end of this AI segment
        seg_end = next((float(d.at[i1]) for i0, i1 in segs if d.at[i0] <= t <= d.at[i1] + 0.05), None)
        # model output after the impact (on AI frames)
        m = (d.at > t) & (d.at < min(t + 1.0, seg_end or t + 1.0)) & d.sel
        thr = d.mout[m, 1] if m.any() else np.array([])
        res['impacts'].append({
            't': round(t, 2), 'g': round(e['g'], 1), 'ax_g': round(e['ax'], 1), 'ay_g': round(e['ay'], 1),
            'loc': loc(log, t), 'v_before_ms': round(-e['erpm_before'] * KV, 2), 'erpm_before': int(e['erpm_before']),
            't_stop_after_s': None if t_stop is None else round(t_stop - t, 2),
            'erpm_at_+0.5s': None if not np.isfinite(e05) else int(e05),
            'glance': bool(np.isfinite(e05) and e05 < -1500),
            'ai_seg_end_after_s': None if seg_end is None else round(seg_end - t, 2),
            'model_thr_after_us_median': None if not len(thr) else round(float(np.median(thr - NEUTRAL))),
            'model_reverse_frames': int(np.sum(thr > NEUTRAL + 40)),
            'ttc_min_before': None,
        })
    for i0, i1 in segs:
        t0, t1 = float(d.at[i0]), float(d.at[i1])
        nxt = d.at[i1 + 1] if i1 + 1 < len(d.at) else None
        end_kind = 'end_of_log'
        if nxt is not None:
            end_kind = 'board_reset' if any(t1 - 0.2 <= r <= t1 + 0.3 for r in resets) else 'takeover'
        # detector state at the last AI frame
        armed, slow_since, impact_t, impact_moving, phase = det[i1]
        need = 0.5 if (impact_t is not None and impact_moving) else 0.8
        imp = [x for x in res['impacts'] if t0 <= x['t'] <= t1 + 0.05]
        last_imp = imp[-1]['t'] if imp else None
        # stopped time before takeover
        m = (d.vt >= t0) & (d.vt <= t1)
        stopped = (np.abs(d.erpm[m]) < 300)
        # length of the final stopped run
        run = 0.0
        if stopped.any() and stopped[-1]:
            j = len(stopped) - 1
            while j > 0 and stopped[j - 1]:
                j -= 1
            run = float(d.vt[m][-1] - d.vt[m][j])
        # what the human did after the takeover: reversed? time until AI again
        nseg = next(((float(d.at[a0])) for a0, a1 in segs if d.at[a0] > t1), None)
        hm = (d.vt > t1) & (d.vt < (nseg if nseg else t1 + 30))
        h_rev = bool(np.any(d.erpm[hm] > 300))
        h_rev_t = float(d.vt[hm][np.argmax(d.erpm[hm] > 300)] - t1) if h_rev else None
        res['segments'].append({
            't0': round(t0, 2), 't1': round(t1, 2), 'dur_s': round(t1 - t0 + 1 / 30, 2), 'end': end_kind,
            'gains': [round(float(x), 2) for x in d.gains[i1]], 'lookahead': int(d.la[i1]),
            'impacts_in_seg': len(imp), 'last_impact_to_end_s': None if last_imp is None else round(t1 - last_imp, 2),
            'erpm_at_end': int(d.erpm_at(t1)), 'stopped_run_at_end_s': round(run, 2),
            'detector': {'armed': bool(armed), 'slow_since_to_end_s': None if slow_since is None else round(
                t1 + 0.03 - slow_since, 2), 'needed_s': need, 'impact_pending': impact_t is not None},
            'human_reversed': h_rev, 'human_reverse_after_s': None if h_rev_t is None else round(h_rev_t, 2),
            'back_to_ai_after_s': None if nseg is None else round(nseg - t1, 2),
        })
    return d, res


if __name__ == '__main__':
    allr = {}
    for log in ['20261010-203924', '20261010-204208', '20261010-204605']:
        d, r = analyze(log)
        allr[log] = r
        print('=====', d.name)
        for s in r['segments']:
            print('  seg', s)
        for x in r['impacts']:
            print('  imp', x)
    json.dump(allr, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/takeovers.json', 'w'), indent=1)
