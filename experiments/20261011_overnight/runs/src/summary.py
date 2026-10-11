"""Collect the main numbers into runs/summary.json"""
import json

import numpy as np

S = '/home/jetson/jetformer_reports/overnight_20261011/runs/src/'
basic = json.load(open(S + 'basic.json'))
tk = json.load(open(S + 'takeovers.json'))
sw = json.load(open(S + 'tof_sweep.json'))
dv = json.load(open(S + 'detector_variants.json'))
pf = json.load(open(S + 'params_fx.json'))
fa = json.load(open(S + 'false_alarms.json'))
cf = json.load(open(S + 'impact_speed_cf.json'))
mo = json.load(open(S + 'moments.json'))
LOGS = ['20261010-203924', '20261010-204208', '20261010-204605']

TAKEOVER_CLASS = {
    '20261010-203924': ['pre-emptive (heading into HS parking-bay boards, no impact)', 'after stopping impact LH 15.16',
                        'after stopping impact LH 19.23', 'after stopping impact GM_in 23.98',
                        'after stopping impact HS_P 45.80', 'plain stop (LH, curtain, no impact >=2.5G)',
                        'plain stop (LH, curtain, no impact >=2.5G)', 'after stopping impact HS_P 65.72'],
    '20261010-204208': ['plain stop (LH, soft curtain contact)', 'after stopping impact GM_out 16.89',
                        'while moving after DE2 glance 40.59', 'pre-emptive before DE (3.1G under human at 88.58)'],
    '20261010-204605': ['after stopping impact LH 12.51', 'after stopping impact LH 42.48', 'plain stop (GM entry, white PTFE sheet)'],
}
LAPS = {
    '20261010-203924': {'lapcounter_laps': 1, 'ai_only_laps': 0, 'clean_laps': 0, 'lap_times_s': [37.53], 'note': 'only the first (mixed) lap'},
    '20261010-204208': {'lapcounter_laps': 3, 'ai_only_laps': 1, 'clean_laps': 0, 'lap_times_s': [10.62, 27.27, 10.15],
                        'ai_only_lap': {'t': [67.9, 78.05], 'time_s': 10.15, 'impacts': 1, 'impact': 'DE 75.67 8.1G glance'}},
    '20261010-204605': {'lapcounter_laps': 1, 'ai_only_laps': 0, 'clean_laps': 0, 'lap_times_s': [9.34], 'note': 'first (mixed) lap'},
}
out = {'generated': '2026-10-11', 'model': 'first_trial_20261010-s3-20261010-153629 (all three runs)',
       'definitions': {
           't': 'seconds from the first AI_CMD frame of the log (same as tools/drive_report.py)',
           'impact': 'horizontal |a| = hypot(ax, ay) >= 2.5 G, merged within 0.6 s (same quantity as drive.py / LapCounter)',
           'ai_heading_laps': 'gyro-z heading integrated on the AI side / 360 deg (lap-equivalents driven by the AI)',
           'clean_lap': 'LapCounter: AI only, not first, no recovery, no impact',
           'tof_false_alarm': 'warning onset (TTC<0.5, v>0.5 m/s, gaps<0.3 s merged) with no >=2.0G event within 1.5 s',
           'tof_caught': 'warning on within 0.3 s before the impact and started >= 0.3 s before it'},
       'runs': {}, 'pooled_new_runs': {}, 'earlier_runs': {}}
for log in LOGS:
    b = basic[log]
    t = tk[log]
    p = sw['per_run'][log]
    out['runs'][log] = {
        'name': b['name'], 'duration_s': b['duration_s'], 'ai_time_s': b['ai_s_ts'], 'ai_segments': b['ai_segments'],
        'ai_segments_s': b['segments_s'], 'segment_ends': b['segment_ends'], 'human_takeovers': b['takeovers'],
        'takeover_reasons': TAKEOVER_CLASS[log],
        'ai_distance_m_from_erpm': b['ai_dist_m'], 'ai_heading_laps': b['ai_heading_laps'],
        'laps': LAPS[log],
        'impacts_ai': b['impacts_ai_h25'], 'impacts_ai_ge4G': b['impacts_ai_h4'],
        'impacts_per_ai_lap': round(b['impacts_ai_h25'] / b['ai_heading_laps'], 2),
        'impacts_per_ai_min': round(b['impacts_ai_h25'] / (b['ai_s_ts'] / 60), 1),
        'impacts': t['impacts'],
        'stuck_events_ai': b['stuck'],
        'recovery_v2': {'episodes': 0, 'giveups': 0, 'reason': 'human took over before the detector fired in all 12 stop cases'},
        'esc_assist_frames': {'20261010-203924': 26, '20261010-204208': 0, '20261010-204605': 13}[log],
        'esc_assist_episodes': {'20261010-203924': 5, '20261010-204208': 0, '20261010-204605': 1}[log],
        'tof_guard_logged': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p['logged: TTC<0.5 (now)'].items()},
        'board_resets_t': b['board_resets'],
        'holds_recorded_frames': {'20261010-203924': 0, '20261010-204208': 6, '20261010-204605': 781}[log],
        'holds_note': {'20261010-203924': 'none',
                       '20261010-204208': '72.81-72.98 (6 frames) = 2nd board reboot (vehicle data gap 0.36 s); 1st reboot at 64.68 dropped CH3 select for 0.1 s; camera gaps on AI side 16.29 (0.20 s), 36.04 (0.20 s), 73.64 (0.13 s) -> unrecorded neutral holds',
                       '20261010-204605': '74.52-100.55 on the human side after the transmitter signal was invalid (71.99) and vehicle data stopped (74.36) = end of session; camera gap on AI side 33.04 (0.13 s)'}[log],
        'erpm_ai_p50_p90': [b['erpm_ai_p50'], b['erpm_ai_p90']],
        'steer_limit_ratio': b['steer_limit_ratio'], 'steer_flips_per_s': b['steer_flips_per_s'],
        'latency_ms_p50_p99': [b['latency_p50'], b['latency_p99']],
    }
tot_imp = sum(basic[l]['impacts_ai_h25'] for l in LOGS)
tot_laps = sum(basic[l]['ai_heading_laps'] for l in LOGS)
tot_ai = sum(basic[l]['ai_s_ts'] for l in LOGS)
out['pooled_new_runs'] = {
    'ai_time_s': round(tot_ai, 1), 'ai_heading_laps': round(tot_laps, 2), 'impacts_ai': tot_imp,
    'impacts_per_lap': round(tot_imp / tot_laps, 2), 'impacts_per_lap_95ci_poisson': [round(13.0 / tot_laps, 2), round(32.1 / tot_laps, 2)],
    'impacts_per_ai_min': round(tot_imp / (tot_ai / 60), 1),
    'clean_laps': 0, 'ai_only_laps': 1,
    'takeovers': sum(basic[l]['takeovers'] for l in LOGS), 'board_resets': 2,
    'hotspots': {'DE (back straight right partition)': 7, 'DE2 (after DE to SW)': 4, 'LH (black curtain)': 4, 'GM_in': 2,
                 'GM_out': 1, 'HS_P (parking-bay boards, new)': 2, 'R1': 1},
    'stopping_impacts': 8, 'plain_stops': 4,
    'recovery_v2_episodes': 0,
    'impact_to_takeover_s_median_range': [0.455, 0.19, 0.98],
    'tof_guard_rules': sw['pooled'],
    'tof_impact_speed_if_applied': {'v_actual_median_ms': 3.07, 'v_with_guard_median_ms': 1.96,
                                    'energy_reduction_median': 0.65, 'below_1ms': '3/21', 'model': 'self-limiting, 0.15 s delay, coast 1.0 / brake 2.8 m/s2'},
    'tof_lap_cost_s_per_lap': {'false_alarms_only_now': 0.26, 'false_alarms_only_recal_floor': 0.12,
                                'all_warnings_now': 1.02, 'all_warnings_recal_floor': 0.84},
    'tof_floor_offset_row9_pct': {'until_20:10': '0 +-3', '201626-203924': '-8 to -12', '204208': '-13 to -18', '204605': '-14 to -29'},
    'detector_variants': {k: {n: len(v) for n, v in dv[k].items()} for k in dv},
}
out['earlier_runs'] = {
    log: {'name': basic[log]['name'], 'ai_time_s': basic[log]['ai_s_ts'], 'ai_heading_laps': basic[log]['ai_heading_laps'],
          'impacts_ai': basic[log]['impacts_ai_h25'],
          'impacts_per_lap': round(basic[log]['impacts_ai_h25'] / basic[log]['ai_heading_laps'], 2),
          'erpm_ai_p50_p90': [basic[log]['erpm_ai_p50'], basic[log]['erpm_ai_p90']],
          'takeovers': basic[log]['takeovers'], 'board_resets': len(basic[log]['board_resets'])}
    for log in ['20261010-172910', '20261010-173445', '20261010-181504']}
out['earlier_runs']['20261010-181504']['recovery_v1'] = {
    'episodes': 4, 'reversed': 0, 'giveups': 1,
    'detail': '26.31 and 29.81: +180us reverse, eRPM stayed 0, give-up 33.27; 65.88 and 81.17 ended by takeover within 0.1-0.5 s'}
out['earlier_runs']['20261010-173445']['note'] = 'no recovery; AI held reverse/brake without neutral; v2 detector replay fires on 5 of 6 stuck events before the takeover'
out['params'] = pf
out['video_moments'] = mo
out['false_alarms'] = fa
json.dump(out, open('/home/jetson/jetformer_reports/overnight_20261011/runs/summary.json', 'w'), indent=1, ensure_ascii=False,
          default=float)
print('ok', tot_imp, tot_laps, tot_imp / tot_laps)
