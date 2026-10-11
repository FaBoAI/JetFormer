"""Recovery v2 detector timeline for every AI segment end (takeover) of ai_run_009/010/101, from the exact
offline replay of jetformer.assist (replay.py reproduces the logged assist codes 100 %).

Per segment end: last impact (>= 2.5 g, AI side) before the end, rebound (max eRPM within 0.6 s after the impact;
> +300 = 'backing' for the detector, which resets the stuck timer), the frame where the detector armed the
'impact while moving' path, the start of the final stuck run (slow_since), what it needed (0.5 / 0.8 s),
the earliest time it would have fired had the car stayed still, and the human takeover time.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, seg_mask, KV  # noqa: E402
from replay import replay  # noqa: E402

LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']


def main():
    rows = []
    for log in LOGS:
        d = L(log)
        kinds, outs, det, ig = replay(d)
        ev = [e for e in d.events(2.5) if e['ai'] and not e['suspect']]
        segs = seg_mask(d.sel)
        resets = [float(d.vt[i + 1]) for i in np.nonzero(np.diff(d.board) < 0)[0]]
        for a, b in segs:
            t0, t1 = float(d.at[a]), float(d.at[b])
            if b + 1 >= len(d.at) or any(t1 - 0.3 <= x <= t1 + 0.3 for x in resets):
                continue
            imp = [e for e in ev if t0 <= e['t'] <= t1 + 0.05]
            last = imp[-1] if imp and t1 - imp[-1]['t'] < 2.0 else None
            r = {'log': log, 'name': d.name, 'seg_t0': round(t0, 2), 'takeover_t': round(t1 + 1 / 30, 2),
                 'seg_s': round(t1 - t0 + 1 / 30, 2)}
            armed, slow_since, impact_t, impact_moving, phase = det[b]
            r['armed'] = bool(armed)
            r['impact_path'] = bool(impact_t is not None and impact_moving)
            need = 0.5 if r['impact_path'] else 0.8
            r['need_s'] = need
            r['erpm_at_takeover'] = int(d.erpm_at(t1))
            if last is not None:
                r['impact_t'] = round(last['t'], 2)
                r['impact_g'] = round(last['g'], 1)
                r['v_before'] = round(-last['erpm_before'] * KV, 2)
                m = (d.vt > last['t']) & (d.vt < last['t'] + 0.6)
                r['rebound_erpm_max'] = int(np.max(d.erpm[m])) if m.any() else None
                k = np.nonzero(m & (d.erpm > -300))[0]
                r['stop_after_s'] = round(float(d.vt[k[0]] - last['t']), 2) if len(k) else None
                r['takeover_after_impact_s'] = round(t1 + 1 / 30 - last['t'], 2)
                # did the detector see the impact as 'moving'? first frame after the impact with ig >= 2.5
                fk = np.nonzero((d.at >= last['t']) & (ig >= 2.5))[0]
                r['impact_seen_erpm'] = int(d.erpm_at(d.at[fk[0]])) if len(fk) else None
            else:
                r['impact_t'] = None
            if slow_since is not None:
                # slow_since is in replay 'now' units (camera t + latency); convert back approx with the frame latency
                lat = d.lat[b] / 1000.0 if np.isfinite(d.lat[b]) else 0.03
                ss = slow_since - lat
                r['stuck_from_t'] = round(ss, 2)
                r['stuck_counted_s'] = round(t1 + 1 / 30 - ss, 2)
                r['would_fire_t'] = round(ss + need, 2)
                r['margin_s'] = round(ss + need - (t1 + 1 / 30), 2)
            else:
                r['stuck_from_t'] = None
                r['stuck_counted_s'] = 0.0
                r['would_fire_t'] = None
                r['margin_s'] = None
            # model throttle in the last 0.5 s (positive = reverse/brake request), assist==1 frames
            m = (d.at > t1 - 0.5) & (d.at <= t1) & d.sel
            r['model_rev_req_frames_last0.5s'] = int(np.sum(d.mout[m, 1] > 1498 + 40))
            r['esc_neutralised_frames_seg'] = int(np.sum(d.assist[a:b + 1] == 1))
            # human after takeover: first reverse motion (> +300 eRPM) and time back to AI
            nxt = [float(d.at[x]) for x, y in segs if d.at[x] > t1]
            hm = (d.vt > t1) & (d.vt < (nxt[0] if nxt else t1 + 20))
            kk = np.nonzero(hm & (d.erpm > 300))[0]
            r['human_reverse_after_s'] = round(float(d.vt[kk[0]] - t1), 2) if len(kk) else None
            r['back_to_ai_after_s'] = round(nxt[0] - t1, 2) if nxt else None
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False))
    json.dump(rows, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/recov.json', 'w'), indent=1,
              ensure_ascii=False)


if __name__ == '__main__':
    main()
