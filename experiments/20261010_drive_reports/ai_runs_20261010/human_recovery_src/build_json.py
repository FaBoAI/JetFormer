import json, numpy as np
from simdet import evaluate
S = json.load(open('summary.json')); E = json.load(open('episodes.json')); CLS = json.load(open('ep_class.json'))
KK = {r['ep']: r for r in json.load(open('fwdkick.json'))}
LAT = json.load(open('startlat.json')); IMP = json.load(open('impacts.json'))
sweep = []
A_eps = [int(k) for k, v in CLS.items() if v == 'A']
for T2 in [0.3, 0.6, 1.0]:
    for conf in [0.3, 0.5]:
        rows = evaluate(T2=T2, confirm=conf)
        hit = sorted(set(r['ep'] for r in rows if r['ep'] in A_eps))
        sweep.append(dict(T2_s=T2, confirm_s=conf, A_detected=len(hit), A_total=len(A_eps), AI_run_A_detected=len([e for e in hit if e >= 19]), AI_run_A_total=10,
                          triggers_outside_A=len([r for r in rows if r['ep'] is None or r['ep'] not in A_eps]),
                          triggers_outside_A_on_AI_side=len([r for r in rows if (r['ep'] is None or r['ep'] not in A_eps) and r['ai']]),
                          latency_from_stop_median=float(np.median([min(r['lat_from_stop'] for r in rows if r['ep'] == e) for e in hit]))))
fl = [x['latency'] for x in LAT if x['dir'] == 'F' and x['latency'] is not None]
eps = []
for k, x in enumerate(E):
    kk = KK.get(k + 1, {})
    eps.append(dict(id=k + 1, cls=CLS[str(k + 1)], log=x['log'], t_stop_s=x['t_stop'], source=x['source'], impact_g=x['impact']['g'], impact_ay_g=x['impact']['ay'],
                    speed_pre_erpm=x['speed_pre'], steer_stop_us=x['st_stop'], takeover_after_stop_s=x['takeover_after_stop'], first_R_after_stop_s=x['first_R_after_stop'],
                    reverse_engaged_after_stop_s=x['engage_after_stop'], neutral_gaps_s=x['gaps'], brake_pulses=x['brake_pulses'], onset_after_R_s=[None if v is None else round(v, 2) for v in x['t_on']],
                    failed_R_attempts=x['failed'], fail_reasons=x['fail_reasons'], n_reverse_engagements=x['n_engagements'],
                    reverse_thr_median_us=x['rev_thr_med'], reverse_thr_max_us=x['rev_thr_max'], reverse_cmd_s=x['rev_cmd_s'], reverse_motion_s=x['rev_motion_s'],
                    reverse_hall_counts=x['rev_counts'], reverse_peak_erpm=x['rev_peak_erpm'], steer_reverse_us=x['st_rev'], steer_reverse_vs_prestop=x['st_rev_rel'],
                    forward_after_reverse_s=x['fwd_after_rev'], steer_forward_us=x['st_fwd'], steer_forward_vs_prestop=x['st_fwd_rel'],
                    kick_thr_us=kk.get('kick_thr_med'), kick_s=kk.get('kick_dur'), kick_steer_us=kk.get('kick_st_med'),
                    forward_after_stop_s=x['fwd_after_stop'], normal_driving_after_stop_s=x['free_after_stop'], handback_after_stop_s=x['handback_after_stop'],
                    tof_obstacle_at_stop_m=x['tof_obs_stop'], tof_obstacle_at_reverse_end_m=x['tof_obs_rev_end']))
J = {
 'title': 'Human collision/stuck recovery -> automatic recovery manoeuvre parameters (JetFormer, 2026-10-10)',
 'data': dict(human_only_logs=['20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707'],
              ai_run_logs_human_segments=['20261010-163247', '20261010-172910', '20261010-173445'],
              skipped_corrupted_scenes=['20261010-163247-0021', '20261010-163247-0022'],
              driving_time_s=dict(ai_side=424, human_side=606),
              conventions=dict(steer_center_us=1443, throttle_neutral_us=1498, forward='below neutral', reverse_brake='above neutral', erpm='negative = forward', hall_counts='erpm/10 per second; 6 per electrical rev; increases when reversing'),
              command_bands=dict(F='thr < 1478', N='1478..1538', R='thr > 1538'),
              note='effective command = AI out while CH3 selects AI, transmitter pwm otherwise'),
 'esc_reverse_rule': dict(
     threshold_sensitivity='R band threshold 20-60 us above neutral: identical conclusions (human after-F 0/29-0/31, rolling-forward 0/9-0/11, after-brake 33/37); at 80 us one human after-F case flips because a weak brake (<1578 us) is then counted as neutral',
     statement='Reverse drive engages only if (1) the last non-neutral command was R (a brake) of >= ~1560 us for >= ~0.1 s, (2) then neutral, (3) the motor is not turning forward (eRPM >= -150) when R is applied again. The first R after any forward command (even a 30 ms forward blip) only brakes, indefinitely. Reverse->forward has no lockout.',
     counts=S['esc_rule_table'], success_stats=S['esc_success'], failures_after_brake=S['esc_failures_after_brake'], rolling_forward=S['rolling_fwd_attempts'],
     ai_long_R_after_F_stationary=S['ai_R_after_F_long'],
     ai_natural_experiment='173445 t=99.0-100.2: AI output R1665 (brake) -> N 0.20 s -> R1701 and the car reversed 112 counts (then hit something behind at t=100.17, 2.6 g)'),
 'episodes_classification': dict(A='collision/stuck against wall/obstacle (n=19: 9 human-only, 10 AI-run takeovers)', B='braked to stop near obstacle, no impact (n=4)', C='manoeuvring/repositioning without obstacle (n=5, excluded from parameters)'),
 'episode_stats': dict(A_all=S['episodes_A_all'], A_human_only=S['episodes_A_human_only'], A_ai_takeover=S['episodes_A_ai_takeover'], B=S['episodes_B']),
 'forward_kick': {k: v for k, v in [('human_only', dict(thr_us_median=1266, dur_s_median=0.80, steer_us_median=-168, counts_median=443)),
                                     ('ai_takeover', dict(thr_us_median=1369, dur_s_median=0.60, steer_us_median=-428, counts_median=178)),
                                     ('starts_before_reverse_motion_ends_s', 0.15)]},
 'reverse_counts_during_command': dict(median=38, iqr=[28, 62], fraction_of_total_median=0.38, erpm_at_release_median=2211),
 'handback': dict(per_episode=S['handback'], summary=S['handback_summary']),
 'repeat_stuck_after_handback_173445': S['repeat_stuck_173445'],
 'detection': dict(
     forward_start_latency_from_rest_s=dict(n=len(fl), median=float(np.median(fl)), max=float(np.max(fl))),
     forward_stall_observed='forward command with |eRPM|<100 and no hall motion >=0.15 s: only 2 runs (0.18 s each, both normal starts) in 1030 s; the AI stuck signature is R commanded while stationary (ESC brake lockout) for 1.08-2.06 s until takeover',
     impacts_ge_2p5g=dict(ai_side=sum(1 for x in IMP if x['side'] == 'AI'), ai_side_glancing=sum(1 for x in IMP if x['side'] == 'AI' and 'glancing' in x['next']),
                          human_side=sum(1 for x in IMP if x['side'] == 'H'), human_side_glancing=sum(1 for x in IMP if x['side'] == 'H' and 'glancing' in x['next'])),
     D1='impact |a|-1g >= 2.5 g (>=2 samples >1 g within +-50 ms) while moving forward (median eRPM over previous 0.2 s < -500); eRPM > -300 within 0.6 s; forward progress < 20 hall counts during the next 0.5 s -> trigger',
     D2='still (|eRPM|<150 and |dHall|<=2 per 0.1 s) while command outside the neutral band (thr<1478 or thr>1538) continuously for >= 0.6 s; masked for 1.0 s after AI start / hand-back',
     sweep=sweep),
 'recommended_state_machine': {
     'units': 'absolute PWM us, bypass apply_gains (throttle_gain must not scale these)',
     'DETECT': 'D1 or D2 (see detection); latency = stop + 0.5 s (D1) / stop + >=0.6 s (D2)',
     'BRAKE': dict(thr_us=1700, steer_us=1443, duration_s=0.25, why='first R after forward is always a brake; also kills forward roll'),
     'NEUTRAL': dict(thr_us=1498, steer_us=1443, min_s=0.15, until='eRPM >= -150 (not rolling forward)', timeout_s=1.0),
     'REVERSE': dict(thr_us=1720, allowed_range_us=[1680, 1800], steer='centre 1443 (or <=150 us opposite to the pre-stop steering); never same side', engage_check='hall +5 within 0.30 s else NEUTRAL 0.2 s and retry (max 3 cycles)',
                     release_when='hall >= +40 counts since engagement or eRPM >= 2200 or 0.5 s of R', expected_total='~100-120 counts incl. coast (~0.6 s of motion)', abort='impact >= 2.5 g while reversing -> NEUTRAL, go to ESCAPE'),
     'COAST': dict(thr_us=1498, duration_s=0.3, or_until='eRPM < 1000'),
     'ESCAPE_FORWARD': dict(thr_us=1360, duration_s=0.6, or_until='forward hall >= 150 counts', steer_us='full lock away from the contact side: impact a_y < -2 g (left contact) -> ~1000 (right); a_y > +2 g -> ~1890 (left); otherwise the pre-stop steering direction'),
     'HANDBACK': dict(condition='eRPM < -1500 (moving forward) after ESCAPE, sticks neutral up to 0.4 s', expected_after_stop_s=2.3, human_reference_after_stop_s=3.39, human_after_reverse_end_s=1.08, human_erpm_at_handback=-1778),
     'COOLDOWN': 'D2 masked 1.0 s after hand-back; D1 always armed (needs a new impact)',
     'ESCALATE': 'if re-stuck within 10 s after hand-back: reverse target x1.5 (~170 counts) and ESCAPE 0.8 s at full lock',
     'GIVE_UP': '(a) no reverse engagement after 3 NEUTRAL->REVERSE cycles (~1.5 s); (b) 3 recoveries within 20 s; (c) re-stuck within 5 s after hand-back twice in a row -> hold neutral, request human (CH3)'},
 'episodes': eps,
}
json.dump(J, open('/home/jetson/jetformer/data/reports/ai_runs_20261010/human_recovery.json', 'w'), indent=1, ensure_ascii=False)
for s in sweep: print(s)
print('start latency', J['detection']['forward_start_latency_from_rest_s'], J['detection']['impacts_ge_2p5g'])
