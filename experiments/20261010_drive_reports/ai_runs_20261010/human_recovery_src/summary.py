import numpy as np, json
from rruns3 import get, LOGS, rrun_table
from eff import NEUT, CENTER
E = json.load(open('episodes.json')); CLS = json.load(open('ep_class.json'))
rows = json.load(open('rruns3.json'))
def q(v):
    v = np.array([x for x in v if x is not None], float)
    if not len(v): return None
    return dict(n=int(len(v)), median=round(float(np.median(v)), 3), p25=round(float(np.percentile(v, 25)), 3), p75=round(float(np.percentile(v, 75)), 3), min=round(float(v.min()), 3), max=round(float(v.max()), 3))
out = {}
# ---- ESC rule
tab = {}
for r in rows:
    k = r['cat'] + ('/human' if r['human'] else '/AI')
    tab.setdefault(k, [0, 0]); tab[k][0] += int(r['engaged']); tab[k][1] += 1
out['esc_rule_table'] = {k: dict(engaged=v[0], n=v[1]) for k, v in sorted(tab.items())}
succ = [r for r in rows if r['human'] and r['cat'] == 'after_R_brake' and r['engaged']]
fail = [r for r in rows if r['human'] and r['cat'] == 'after_R_brake' and not r['engaged']]
out['esc_success'] = dict(neutral_gap_s=q([r['gap'] for r in succ]), onset_after_R_s=q([r['t_on'] for r in succ]),
                          brake_pulse_us=q([r['prev_thr'] for r in succ]), brake_pulse_s=q([r['prev_dur'] for r in succ]),
                          reverse_cmd_us=q([r['thr'] for r in succ]), erpm_at_R=q([r['e0'] for r in succ]))
out['esc_failures_after_brake'] = [dict(log=r['log'], t=r['t'], thr=r['thr'], dur=r['dur'], gap=r['gap'], prev_thr=r['prev_thr'], prev_dur=r['prev_dur'], dh=r['dh'], peak=r['peak']) for r in fail]
rf = [r for r in rows if r['cat'] == 'after_R_rolling_fwd']
out['rolling_fwd_attempts'] = dict(n=len(rf), engaged=sum(r['engaged'] for r in rf), erpm_at_R=q([r['e0'] for r in rf]), gap=q([r['gap'] for r in rf]))
# AI long R after F while still: displacement
d7 = get('20261010-173445')
ai_long = [r for r in rows if not r['human'] and r['cat'] == 'after_F' and r['dur'] >= 0.5]
lst = []
for r in ai_long:
    d = get(r['log']); i, j = r['i'], r['j']
    a = i + 50
    lst.append(dict(log=r['log'], t=r['t'], dur=r['dur'], thr=r['thr'], disp_after_0p5s=float(d.hall[min(j, len(d.hall)-1)] - d.hall[min(a, len(d.hall)-1)]) if j > a else None))
out['ai_R_after_F_long'] = lst
# ---- episodes
A = [(k + 1, x) for k, x in enumerate(E) if CLS[str(k + 1)] == 'A']
B = [(k + 1, x) for k, x in enumerate(E) if CLS[str(k + 1)] == 'B']
def side(v, thr=100):
    return 'C' if abs(v) < thr else ('L' if v > 0 else 'R')
def epstats(L):
    s = {}
    s['n'] = len(L)
    s['impact_g'] = q([x['impact']['g'] for _, x in L])
    s['speed_pre_erpm'] = q([x['speed_pre'] for _, x in L])
    s['first_R_after_stop_s'] = q([x['first_R_after_stop'] for _, x in L])
    s['engage_after_stop_s'] = q([x['engage_after_stop'] for _, x in L])
    s['neutral_gap_s'] = q([x['gaps'][0] for _, x in L])
    s['n_engagements'] = q([x['n_engagements'] for _, x in L])
    s['failed_R_attempts'] = q([x['failed'] for _, x in L])
    s['rev_thr_med_us'] = q([x['rev_thr_med'] for _, x in L])
    s['rev_thr_max_us'] = q([x['rev_thr_max'] for _, x in L])
    s['rev_cmd_s'] = q([x['rev_cmd_s'] for _, x in L])
    s['rev_motion_s'] = q([x['rev_motion_s'] for _, x in L])
    s['rev_counts'] = q([x['rev_counts'] for _, x in L])
    s['rev_peak_erpm'] = q([x['rev_peak_erpm'] for _, x in L])
    s['fwd_after_rev_s'] = q([x['fwd_after_rev'] for _, x in L])
    s['fwd_after_stop_s'] = q([x['fwd_after_stop'] for _, x in L])
    s['free_after_stop_s'] = q([x['free_after_stop'] for _, x in L])
    s['st_stop_us'] = q([x['st_stop'] for _, x in L])
    s['st_rev_us'] = q([x['st_rev'] for _, x in L])
    s['st_fwd_us'] = q([x['st_fwd'] for _, x in L])
    from collections import Counter
    s['st_rev_rel'] = dict(Counter(x['st_rev_rel'] for _, x in L))
    s['st_fwd_rel'] = dict(Counter(x['st_fwd_rel'] for _, x in L))
    s['st_fwd_vs_rev'] = dict(Counter(x['st_fwd_vs_rev'] for _, x in L))
    # fwd steer vs contact side (ay sign, |ay|>=2)
    c = Counter()
    for _, x in L:
        ay = x['impact']['ay']
        if abs(ay) < 2 or x['st_fwd'] is None: c['contact unclear'] += 1; continue
        contact = 'L' if ay < 0 else 'R'
        fs = side(x['st_fwd'])
        c['away' if (fs != 'C' and fs != contact) else ('toward' if fs == contact else 'centred')] += 1
    s['st_fwd_vs_contact'] = dict(c)
    s['stop_steer_side'] = dict(Counter(side(x['st_stop']) for _, x in L))
    s['contact_side'] = dict(Counter(('L' if x['impact']['ay'] <= -2 else ('R' if x['impact']['ay'] >= 2 else '?')) for _, x in L))
    s['tof_obs_stop_m'] = q([x['tof_obs_stop'] for _, x in L])
    s['tof_obs_rev_end_m'] = q([x['tof_obs_rev_end'] for _, x in L])
    return s
out['episodes_A_all'] = epstats(A)
out['episodes_A_human_only'] = epstats([p for p in A if p[1]['source'] == 'human-only'])
out['episodes_A_ai_takeover'] = epstats([p for p in A if p[1]['source'] != 'human-only'])
out['episodes_B'] = epstats(B)
# ---- takeover / hand-back (AI-run takeovers, class A)
hb = []
for k, x in A:
    if x['source'] == 'human-only' or x['handback_after_stop'] is None: continue
    d = get(x['log']); t = d.vt
    ts = x['t_stop']; thb = ts + x['handback_after_stop']
    tf = ts + x['fwd_after_stop'] if x['fwd_after_stop'] is not None else None
    ih = np.searchsorted(t, thb)
    rec = dict(ep=k, t_stop=ts, takeover_after_stop=x['takeover_after_stop'], handback_after_stop=x['handback_after_stop'],
               handback_after_rev_end=round(x['handback_after_stop'] - x['rev_end_after_stop'], 2),
               fwd_phase_s=None if tf is None else round(thb - tf, 2))
    if tf is not None and thb > tf:
        m = (t >= tf) & (t < thb)
        rec.update(fwd_thr_med=round(float(np.median(d.thr[m]))), fwd_st_med=round(float(np.median(d.st[m]) - CENTER)),
                   fwd_counts=round(float(-(d.hall[np.searchsorted(t, thb)] - d.hall[np.searchsorted(t, tf)]))),
                   erpm_at_handback=round(float(np.median(d.erpm[ih - 5:ih + 1]))), st_at_handback=round(float(np.median(d.st_h[ih - 5:ih]) - CENTER)))
    hb.append(rec)
out['handback'] = hb
out['handback_summary'] = dict(takeover_after_stop=q([h['takeover_after_stop'] for h in hb]), handback_after_stop=q([h['handback_after_stop'] for h in hb]),
                               handback_after_rev_end=q([h['handback_after_rev_end'] for h in hb]), fwd_phase_s=q([h.get('fwd_phase_s') for h in hb]),
                               fwd_counts=q([h.get('fwd_counts') for h in hb]), fwd_thr_med=q([h.get('fwd_thr_med') for h in hb]), fwd_st_med=q([h.get('fwd_st_med') for h in hb]),
                               erpm_at_handback=q([h.get('erpm_at_handback') for h in hb]))
# ---- repeat collisions in 173445 (time from hand-back to next stuck)
eps7 = [(k, x) for k, x in A if x['log'] == '20261010-173445']
rep = []
for (k1, a), (k2, b) in zip(eps7[:-1], eps7[1:]):
    if a['handback_after_stop'] is None: continue
    rep.append(dict(ep=k1, next_ep=k2, gap_s=round(b['t_stop'] - (a['t_stop'] + a['handback_after_stop']), 2)))
out['repeat_stuck_173445'] = rep
json.dump(out, open('summary.json', 'w'), indent=1, ensure_ascii=False)
for k, v in out.items():
    print('##', k); print(json.dumps(v, ensure_ascii=False)[:2500])
