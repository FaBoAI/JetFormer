"""Basic per-run numbers: AI time, segments, takeovers, distance/laps on the AI side, collisions per lap,
stuck events, holds, sensor gaps, speed. Also for the earlier runs 006/007/008 with the same definitions."""
import json

import numpy as np

from load import L, seg_mask, KV, NEUTRAL, G
from course import heading
from takeovers import board_resets

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
OLD = ['20261010-172910', '20261010-173445', '20261010-181504']


def ai_mask_v(d):
    """AI side at the vehicle sample times (from AI_CMD.select, sample-and-hold)"""
    i = np.clip(np.searchsorted(d.at, d.vt, side='right') - 1, 0, len(d.at) - 1)
    return d.sel[i] & (d.vt >= d.at[0]) & (d.vt <= d.at[-1] + 0.05)


def stuck_events(d, min_s=0.3):
    """AI side, after the car has moved (|eRPM|>=1000 since the AI got control), -300 < eRPM < 300 for >= min_s"""
    av = ai_mask_v(d)
    out = []
    for i0, i1 in seg_mask(av):
        armed = False
        k = i0
        while k <= i1:
            if abs(d.erpm[k]) >= 1000:
                armed = True
            if armed and -300 < d.erpm[k] < 300:
                j = k
                while j + 1 <= i1 and -300 < d.erpm[j + 1] < 300:
                    j += 1
                dur = d.vt[j] - d.vt[k]
                if dur >= min_s:
                    end = 'takeover' if j == i1 else 'ai_moved'
                    out.append({'t': round(float(d.vt[k]), 2), 'dur_s': round(float(dur), 2), 'end': end})
                k = j + 1
                continue
            k += 1
    return out


def run_numbers(log, old=False):
    d = L(log)
    heading(d)
    r = {'log': log, 'name': d.name, 'duration_s': round(float(d.at[-1]), 1)}
    r['ai_s'] = round(float(d.sel.sum() / 30.0), 1)   # frames are 30 fps nominal
    # more exact AI time from timestamps
    segs = seg_mask(d.sel)
    r['ai_s_ts'] = round(float(sum(d.at[b] - d.at[a] + 1 / 30 for a, b in segs)), 1)
    r['ai_segments'] = len(segs)
    resets = board_resets(d)
    ends = []
    for a, b in segs:
        if b + 1 >= len(d.at):
            ends.append('end')
        elif any(d.at[b] - 0.2 <= x <= d.at[b] + 0.3 for x in resets):
            ends.append('board_reset')
        else:
            ends.append('takeover')
    r['takeovers'] = ends.count('takeover')
    r['segment_ends'] = ends
    r['segments_s'] = [round(float(d.at[b] - d.at[a] + 1 / 30), 2) for a, b in segs]
    av = ai_mask_v(d)
    dt = np.diff(d.vt, append=d.vt[-1])
    dt[dt > 0.2] = 0
    fwd = av & (d.erpm < 0)
    r['ai_dist_m'] = round(float(np.sum(-d.erpm[fwd] * KV * dt[fwd])), 1)
    # AI-side heading turned (CCW +) via gyro
    ai_i = np.clip(np.searchsorted(d.at, d.it, side='right') - 1, 0, len(d.at) - 1)
    ai_imu = d.sel[ai_i]
    dti = np.diff(d.it, append=d.it[-1]); dti[dti > 0.2] = 0
    r['ai_heading_laps'] = round(float(np.sum((d.gyr[:, 2] - d.bias)[ai_imu] * dti[ai_imu]) / 360.0), 2)
    ev = d.events(2.5)
    evh = [e for e in ev if e['ai']]
    r['impacts_ai_h25'] = len(evh)
    ev3 = [e for e in d.events(2.5, horiz=False) if e['ai'] and not e['suspect']]
    r['impacts_ai_g3_25'] = len(ev3)
    r['impacts_ai_h4'] = sum(1 for e in evh if e['g'] >= 4)
    r['erpm_ai_p50'] = int(np.median(-d.erpm[av & (d.erpm < -1000)])) if (av & (d.erpm < -1000)).any() else None
    r['erpm_ai_p90'] = int(np.percentile(-d.erpm[av & (d.erpm < -1000)], 90)) if (av & (d.erpm < -1000)).any() else None
    r['stuck'] = stuck_events(d)
    r['board_resets'] = [round(x, 2) for x in resets]
    a = d.sel & ~d.hold
    so = d.out[a, 0]
    r['steer_limit_ratio'] = round(float(np.mean((so <= 1001) | (so >= 1999))), 3)
    r['steer_limit_right'] = round(float(np.mean(so <= 1001)), 3)
    r['steer_limit_left'] = round(float(np.mean(so >= 1999)), 3)
    dev = so - 1443
    r['steer_absdev_med'] = round(float(np.median(np.abs(dev))), 1)
    r['steer_flips_per_s'] = round(float(np.sum(np.sign(dev[:-1]) * np.sign(dev[1:]) < 0) / max(a.sum() / 30, 1e-6)), 2)
    r['throttle_out_p10_us'] = round(float(np.percentile(d.out[a, 1] - NEUTRAL, 10)))
    r['throttle_raw_med_us'] = round(float(np.median(d.raw[a, 1] - NEUTRAL)))
    r['latency_p50'] = round(float(np.nanmedian(d.lat[a])), 1)
    r['latency_p99'] = round(float(np.nanpercentile(d.lat[a], 99)), 1)
    return d, r


if __name__ == '__main__':
    out = {}
    for log in LOGS + OLD:
        try:
            d, r = run_numbers(log)
        except Exception as e:  # noqa
            print('fail', log, e)
            continue
        out[log] = r
        print(json.dumps(r, ensure_ascii=False))
    json.dump(out, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/basic.json', 'w'), indent=1,
              ensure_ascii=False)
