"""Build runs/summary.json (v2, 2026-10-11 after the Jetson reboot) from the verified intermediate files.
The previous version (02:54, written before the reboot, no analysis.md) is kept as src/summary_prev_0254.json."""
import json

import numpy as np

S = '/home/jetson/jetformer_reports/overnight_20261011/runs/src/'
J = lambda n: json.load(open(S + n))  # noqa: E731
prev = J('summary_prev_0254.json')
ver = J('verify.json')
rec = J('recov.json')
tof = J('tofcheck.json')
sw = J('tof_sweep.json')
tv = J('tof_variants.json')
dv = J('detector_variants.json')
geo = J('geom_timeline.json')
pf = J('params_fx.json')
mo = J('moments2.json')
fa = J('false_alarms.json')
LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']
HLAPS = {'20261010-203924': 1.26, '20261010-204208': 3.88, '20261010-204605': 2.19}
AI_LAP_WINDOWS = {
    '20261010-203924': [],
    '20261010-204208': [{'lap': 1, 't': [30.01, 40.63], 'time_s': 10.62, 'ai_frac': 1.0, 'impacts': [[36.42, 3.0], [38.49, 5.7], [40.59, 5.7]],
                         'note': 'LapCounter flags it first/mixed because its first-lap stats accumulate from the start of timing'},
                        {'lap': 3, 't': [67.9, 78.05], 'time_s': 10.15, 'ai_frac': 1.0, 'impacts': [[75.67, 8.1]],
                         'note': 'LapCounter ai_only=True; includes a 0.2 s neutral hold (board reboot 72.64) and a 0.13 s camera gap'}],
    '20261010-204605': [{'lap': 1, 't': [27.61, 36.95], 'time_s': 9.34, 'ai_frac': 1.0, 'impacts': [[32.08, 4.0], [34.96, 6.3]],
                         'note': 'LapCounter flags it first/mixed (accumulated stats); DE2 27.58 is 0.03 s before the lap start'}],
}
TAKEOVER_CLASS = prev['runs']

out = {
    'generated': '2026-10-11 (re-verified after the Jetson reboot; previous draft src/summary_prev_0254.json)',
    'model': 'first_trial_20261010-s3-20261010-153629 (trained on logs 135449-152707), genre 5 (low speed) in all runs',
    'definitions': dict(prev['definitions'], **{
        'ai_time_s': 'sum of AI_CMD frame intervals (clipped 0.1 s) with select=True',
        'takeover': 'end of an AI segment (select True->False) that is not the end of the log and not a board reboot',
        'stuck_event': 'AI side, after |eRPM|>=1000 in the segment, -300<eRPM<300 for >= 0.25 s',
        'laps': 'jetformer.laps.LapCounter fed with every IMU sample (gz, eRPM, AI_CMD.select, hypot(ax,ay)/g); recovery noted on assist==2 frames (none)',
        'guard_rates': 'onsets of guard>=1 / >=2 on AI forward frames (v>0.5 m/s), gaps <= 0.3 s merged, per minute of exposure (AI forward minus 1 s after each >=2 g event)',
        'recovery_detector': 'exact offline replay of jetformer.assist.Assist (src/replay.py reproduces the logged assist codes and outputs 100 %)'}),
    'runs': {},
}
for log in LOGS:
    v, p = ver[log], prev['runs'][log]
    t = tof[log]
    ai_min = v['ai_time_s'] / 60
    out['runs'][log] = {
        'name': v['name'], 'duration_s': v['duration_s'], 'ai_time_s': v['ai_time_s'], 'ai_distance_m': v['ai_dist_m'],
        'ai_heading_laps': HLAPS[log], 'ai_segments': v['segments'], 'takeovers': v['takeovers'],
        'takeover_reasons': p['takeover_reasons'],
        'impacts_ai': v['impacts_ai'], 'impacts_ai_ge4G': p['impacts_ai_ge4G'],
        'impacts_per_heading_lap': round(v['impacts_ai'] / HLAPS[log], 2),
        'impacts_per_100m': round(100 * v['impacts_ai'] / v['ai_dist_m'], 1),
        'impacts_per_ai_min': round(v['impacts_ai'] / ai_min, 1),
        'impacts': p['impacts'],
        'laps_lapcounter': v['laps'], 'lap_summary': v['lap_summary'], 'ai_driven_lap_windows': AI_LAP_WINDOWS[log],
        'clean_laps': 0,
        'stuck_events_ai': v['stuck'],
        'recovery_v2': {'episodes': 0, 'giveups': 0, 'assist_counts': v['assist_counts'],
                        'segment_ends_detector': [r for r in rec if r['log'] == log]},
        'esc_assist_episodes': v['esc_assist'],
        'tof_guard_logged': {'exposure_min': t['exposure_min'], 'warn': {k: t['warn'][k] for k in t['warn'] if k != 'fa_list'},
                             'brake': {k: t['brake'][k] for k in t['brake'] if k != 'fa_list'},
                             'false_alarms_warn': t['warn']['fa_list'], 'raw_onsets_per_ai_forward_min': v['guard'],
                             'floor_vs_ref_pct': t['floor'],
                             'lap_cost_selflimit_s': round(sw['per_run'][log]['logged: TTC<0.5 (now)']['cost_sl_s'], 2)},
        'board_resets_t': v['board_resets_t'], 'hold_episodes': v['hold_episodes'],
        'camera_gaps': v['cam_gaps'], 'vehicle_gaps': v['vehicle_gaps'], 'holds_note': p['holds_note'],
        'latency_ms_p50_p99': v['lat_ms_p50_p99'],
        'erpm_ai_p50_p90': p['erpm_ai_p50_p90'],
    }
R = out['runs']
tot = lambda k: sum(R[l][k] for l in LOGS)  # noqa: E731
ai_s, dist, imp = tot('ai_time_s'), tot('ai_distance_m'), tot('impacts_ai')
hl = sum(HLAPS.values())
stop_rows = [r for r in rec if r.get('impact_t') is not None and r['erpm_at_takeover'] > -1000] + \
            [r for r in rec if r.get('impact_t') is None and r['erpm_at_takeover'] > -300]
imp_rows = [r for r in stop_rows if r.get('impact_t') is not None]
expo = sum(tof[l]['exposure_min'] for l in LOGS)
out['pooled_new_runs'] = {
    'ai_time_s': round(ai_s, 1), 'ai_distance_m': round(dist, 1), 'ai_heading_laps': round(hl, 2),
    'impacts_ai': imp, 'impacts_per_heading_lap': round(imp / hl, 2),
    'impacts_per_heading_lap_95ci_poisson': prev['pooled_new_runs']['impacts_per_lap_95ci_poisson'],
    'impacts_per_100m': round(100 * imp / dist, 1), 'impacts_per_ai_min': round(imp / (ai_s / 60), 1),
    'hotspots': prev['pooled_new_runs']['hotspots'], 'glancing_impacts_kept_driving': 13, 'stopping_impacts': 8,
    'takeovers': tot('takeovers'), 'takeovers_per_ai_min': round(tot('takeovers') / (ai_s / 60), 1),
    'takeover_kinds': {'after_stopping_impact': 8, 'plain_stop_no_impact': 4, 'while_moving_after_glance': 1, 'pre_emptive_while_moving': 2},
    'lapcounter_laps': 5, 'lapcounter_ai_only_laps': 1, 'ai_driven_lap_windows': 3, 'clean_laps': 0,
    'ai_lap_times_s': [10.62, 10.15, 9.34], 'stuck_events_ai': sum(len(R[l]['stuck_events_ai']) for l in LOGS),
    'stuck_events_ended_by_takeover': sum(len(R[l]['stuck_events_ai']) for l in LOGS),
    'recovery_v2_episodes': 0, 'recovery_giveups': 0,
    'recovery_stop_cases': len(stop_rows),
    'recovery_takeover_after_impact_s': sorted(r['takeover_after_impact_s'] for r in imp_rows),
    'recovery_takeover_after_impact_median_s': float(np.median([r['takeover_after_impact_s'] for r in imp_rows])),
    'recovery_stuck_counted_at_takeover_s': [r['stuck_counted_s'] for r in stop_rows],
    'recovery_rebounding_at_takeover': sum(1 for r in imp_rows if r['erpm_at_takeover'] > 300),
    'esc_assist_episodes': sum(len(R[l]['esc_assist_episodes']) for l in LOGS),
    'esc_assist_frames': sum(R[l]['recovery_v2']['assist_counts'].get('1', 0) for l in LOGS),
    'tof_guard': {
        'exposure_min': round(expo, 3),
        'warn_episodes_per_min': round(sum(tof[l]['warn']['episodes'] for l in LOGS) / expo, 1),
        'brake_episodes_per_min': round(sum(tof[l]['brake']['episodes'] for l in LOGS) / expo, 1),
        'false_alarms_warn': sum(tof[l]['warn']['false_alarms'] for l in LOGS),
        'false_alarms_warn_per_min': round(sum(tof[l]['warn']['false_alarms'] for l in LOGS) / expo, 1),
        'false_alarms_brake_per_min': round(sum(tof[l]['brake']['false_alarms'] for l in LOGS) / expo, 1),
        'false_alarms_near_miss_ttc_lt_0.25': sum(1 for l in LOGS for x in tof[l]['warn']['fa_list'] if x['ttc_min'] < 0.25),
        'warn_time_frac_incl_pre_impact': round(sum(tof[l]['warn']['warn_time_frac'] * tof[l]['exposure_min'] for l in LOGS) / expo, 3),
        'warn_time_frac_no_impact_within_1.5s': 0.183,
        'caught_lead_ge_0.3s': '%d/%d' % (sum(tof[l]['warn']['caught_0.3'] for l in LOGS), sum(tof[l]['warn']['n_coll'] for l in LOGS)),
        'caught_brake_level_lead_ge_0.3s': '%d/%d' % (sum(tof[l]['brake']['caught_0.3'] for l in LOGS), sum(tof[l]['brake']['n_coll'] for l in LOGS)),
        'lead_median_s': 0.46,
        'ttc_min_before_collisions_range_s': [0.07, 0.24],
        'missed': '20261010-204605 32.08 R1 4.0 G: only single-frame TTC 0.32/0.24 s at 31.55-31.61 (0.5 s before), then the car steered full right and the obstacle left the forward ToF view; side contact (lateral 3.9 g)',
        'lap_cost_s_per_lap': prev['pooled_new_runs']['tof_lap_cost_s_per_lap'],
        'impact_speed_if_applied': prev['pooled_new_runs']['tof_impact_speed_if_applied'],
        'threshold_sweep': prev['pooled_new_runs']['tof_guard_rules'],
        'geometry_variants': tv,
        'suggested': {'first': 'fix the ToF/camera mount and rebuild data/tof_floor_ref.npy',
                      'then_mode2': 'T_WARN 0.4 / T_BRAKE 0.3 with a rebuilt floor: caught 17/21 (>=0.3 s), 20/21 (>=0.15 s), FA 12.7/min, warn time 7.8 %, est. cost 0.72 s/lap',
                      'not_helpful': 'narrower path band, gyro-arc only, VERT_MIN 6 (detection drops more than false alarms)'},
    },
    'board_resets': 2, 'hold_frames_ai_side': 6,
}
out['earlier_runs'] = prev['earlier_runs']
out['earlier_runs']['20261010-173445']['lapcounter'] = ver['20261010-173445']['lap_summary']
out['earlier_runs']['20261010-173445']['stuck_events_ai'] = ver['20261010-173445']['stuck']
out['earlier_runs']['20261010-181504']['lapcounter'] = ver['20261010-181504']['lap_summary']
out['recovery_detector_variants'] = dv
out['params_effects'] = pf
out['sensor_geometry'] = {'method': 'camera: rows of the own blue bumper at the image bottom (median of 150 frames); ToF: straight fast driving, '
                                    'rows 9 / 18, centre columns 20-34, median depth vs data/tof_floor_ref.npy',
                          'per_log': geo}
out['video_moments'] = mo
out['false_alarm_list_prev'] = fa
json.dump(out, open('/home/jetson/jetformer_reports/overnight_20261011/runs/summary.json', 'w'), indent=1, ensure_ascii=False,
          default=float)
print(json.dumps(out['pooled_new_runs'], ensure_ascii=False, default=float)[:3000])
