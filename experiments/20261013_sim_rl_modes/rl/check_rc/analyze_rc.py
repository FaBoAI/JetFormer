"""離脱復活(recover)の確認 r1: 走らせた回(rows_*.jsonl)から判定 C1〜C6・ラベル・ずる・ばらつきを自前で計算する(2026-10-14)。

学習・評価のコード(eval_modes.mjs・variation_metrics.mjs・build_nominal_line.mjs)を使わず、protocol.json definitions の文から書いた。
  python -I rl/check_rc/analyze_rc.py <runs_dir> <eval_test.json> <out.json>
"""
from __future__ import annotations

import glob
import json
import math
import sys

import numpy as np

RUNS, EVAL_TEST, OUT = sys.argv[1:4]
REF = json.load(open('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl/reference_loop.json'))
L = REF['length_m']
NB = int(math.ceil(L / 0.05))
MODES = json.load(open('/home/jetson/jetformer_reports/sim_rl_modes_20261013/modes.json'))
REC = MODES['common']['perturbation_recording']
ZONES = MODES['common']['zones']

rows = []
for p in sorted(glob.glob(RUNS + '/rows_*.jsonl')):
    for line in open(p):
        rows.append(json.loads(line))
by = {}
for r in rows:
    by.setdefault(r['set'], []).append(r)
for k in by:
    by[k].sort(key=lambda r: (len(r['id']), r['id']))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def q(a, p):
    a = sorted(a)
    if not a:
        return None
    # 線形補間の分位点(numpy の既定と同じ)
    return float(np.quantile(np.asarray(a, float), p))


# ---- 乱しなしの線(definitions.nominal_line): 8 回、開始から 3 m 以降の最初の通過の e_y・ψ を 5 cm のマスで平均 ----
def nominal_line(runs):
    ey = [[] for _ in range(NB)]
    ps = [[] for _ in range(NB)]
    for r in runs:
        s, e, h = r['pf']['s'], r['pf']['ey'], r['pf']['epsi']
        s0 = s[0]
        seen = set()
        for i in range(len(s)):
            if s[i] - s0 < 3:
                continue
            b = int((s[i] % L) / 0.05) % NB
            if b in seen:
                continue
            seen.add(b)
            ey[b].append(e[i])
            ps[b].append(h[i])
    eyM = np.full(NB, np.nan)
    psM = np.full(NB, np.nan)
    for b in range(NB):
        if ey[b]:
            eyM[b] = np.mean(ey[b])
            psM[b] = math.atan2(np.mean(np.sin(ps[b])), np.mean(np.cos(ps[b])))
    # 抜けたマスは近いマスで埋める(周をまたいで)
    have = np.nonzero(~np.isnan(eyM))[0]
    filled = 0
    for b in range(NB):
        if np.isnan(eyM[b]):
            d = np.minimum(np.abs(have - b), NB - np.abs(have - b))
            j = have[np.argmin(d)]
            eyM[b], psM[b] = eyM[j], psM[j]
            filled += 1
    return {'ey': eyM, 'psi': psM, 'filled': filled, 'runs': len(runs),
            'valid': sum(1 for r in runs if r['valid_episode'])}


NL = {'recover': nominal_line(by['NOM-recover']), 'fastest': nominal_line(by['NOM-fastest'])}


def stall_wrong(r):
    v, s = r['pf']['v'], r['pf']['s']
    T = len(v)
    stall = [False] * T
    wrong = [False] * T
    run = 0
    smax = -1e9
    for f in range(T):
        run = run + 1 if (f / 30 >= 1 and v[f] < 0.2) else 0
        stall[f] = run >= 60
        smax = max(smax, s[f])
        wrong[f] = s[f] < smax - 1
    return stall, wrong


def recoveries(r, nl):
    """出来事ごとの立て直し(definitions.recovered・recovery_success)。3 つの読み方:
    A = 15 フレーム続いた区間の最初のフレーム(評価のコードと同じ読み)、B = 15 フレーム目(続いたと分かったフレーム)、
    C = A で、次の出来事の始まりまでに 15 フレームが終わること"""
    pf = r['pf']
    T = len(pf['s'])
    stall, wrong = stall_wrong(r)
    evs = r['events']
    collF = T if r['collisions'] else math.inf
    out = []
    for k, ev in enumerate(evs):
        pe = ev['pe'] if ev['pe'] is not None else T
        nxt = evs[k + 1]['ps'] if k + 1 < len(evs) else T
        until = min(nxt, T)
        runlen = 0
        recA = recB = None
        for f in range(pe, T):
            b = int((pf['s'][f] % L) / 0.05) % NB
            ok = abs(pf['ey'][f] - nl['ey'][b]) <= 0.10 and abs(wrap(pf['epsi'][f] - nl['psi'][b])) <= 0.15
            runlen = runlen + 1 if ok else 0
            if runlen >= 15:
                recA, recB = f - 14, f
                break
        bad = collF <= until
        why = 'collision' if bad else None
        if not bad:
            for f in range(pe, until):
                if stall[f]:
                    bad, why = True, 'stall'
                    break
                if wrong[f]:
                    bad, why = True, 'wrong_way'
                    break
        tA = None if recA is None else (recA - pe) / 30
        tB = None if recB is None else (recB - pe) / 30
        out.append({'n': ev['n'], 'type': ev['type'], 'cls': ev['cls'], 'ps': ev['ps'], 'pe': ev['pe'], 's': ev['s'],
                    'tA': tA, 'tB': tB, 'bad': bad, 'why': why,
                    'okA': tA is not None and tA <= 3.0 and not bad,
                    'okB': tB is not None and tB <= 3.0 and not bad,
                    'okC': tA is not None and tA <= 3.0 and not bad and recB < until,
                    'v_ps': pf['v'][ev['ps']] if ev['ps'] < T else None,
                    'smod': (pf['s'][ev['ps']] % L) if ev['ps'] < T else None})
    return out


def mask_frac(r):
    l = r['pf']['l']
    n = len(l)
    m = np.zeros(n, bool)
    for j in range(n):
        if l[j] == 0:
            m[max(0, j - 10):j] = True
    return float(m.sum()) / 600, m


def p_summary(rs, nl):
    ev = []
    epm, mfs = [], []
    for r in rs:
        rec = recoveries(r, nl)
        r['_rec'] = rec
        ev += rec
        epm.append(len(r['events']))
        mfs.append(mask_frac(r)[0])
    valid = [r for r in rs if r['valid_episode']]
    tA = [e['tA'] for e in ev if e['okA']]
    tB = [e['tB'] for e in ev if e['okB']]
    out = {
        'episodes': len(rs), 'valid': len(valid),
        'events': len(ev),
        'C2_A': sum(e['okA'] for e in ev) / max(1, len(ev)), 'C2_B': sum(e['okB'] for e in ev) / max(1, len(ev)),
        'C2_C': sum(e['okC'] for e in ev) / max(1, len(ev)),
        'C3_A': {'median': q(tA, .5), 'p90': q(tA, .9)}, 'C3_B': {'median': q(tB, .5), 'p90': q(tB, .9)},
        'C4_events_per_episode': float(np.mean(epm)), 'C4_masked_fraction': float(np.mean(mfs)),
        'valid_only_events_per_episode': float(np.mean([len(r['events']) for r in valid])) if valid else None,
        'valid_only_masked_fraction': float(np.mean([mask_frac(r)[0] for r in valid])) if valid else None,
        'fail_why': {}, 'by_type': {}, 'by_class': {},
        'end_reasons': {},
        'min_clearance_valid': min([r['min_clearance'] for r in valid], default=None),
    }
    for e in ev:
        if not e['okA']:
            w = e['why'] or ('not_recovered' if e['tA'] is None else 'late')
            out['fail_why'][w] = out['fail_why'].get(w, 0) + 1
        for key, val in (('by_type', e['type']), ('by_class', e['cls'])):
            d = out[key].setdefault(val, {'n': 0, 'ok': 0})
            d['n'] += 1
            d['ok'] += int(e['okA'])
    for r in rs:
        out['end_reasons'][r['reason']] = out['end_reasons'].get(r['reason'], 0) + 1
    return out


res = {'nominal_line': {k: {'runs': v['runs'], 'valid': v['valid'], 'filled_bins': v['filled']} for k, v in NL.items()}}

# ---- 1) 試験の組の作り直し ----
P = by['REPLAY-P100']
res['REPLAY_P100'] = p_summary(P, NL['recover'])
K = by['REPLAY-KICK']
for kind in ('yaw', 'lateral'):
    j = [r for r in K if r['spec']['kind'] == kind and r['kick']['allowed']]
    res.setdefault('REPLAY_KICK', {})[kind] = {'judged': len(j), 'survived': sum(r['valid_episode'] for r in j),
                                               'all': len([r for r in K if r['spec']['kind'] == kind]),
                                               'all_survived': sum(r['valid_episode'] for r in K if r['spec']['kind'] == kind)}
std = by['REPLAY-STD'][0]
rec60 = by['REPLAY-REC60']
res['REPLAY_C6'] = {'lap_from_stop': std['lap_from_stop'], 'std_reason': std['reason'], 'std_min_clearance': std['min_clearance'],
                    'rec60_valid': sum(r['valid_episode'] for r in rec60), 'rec60_n': len(rec60),
                    'rec60_min_clearance': min(r['min_clearance'] for r in rec60),
                    'rec60_below_3cm': sorted([(r['id'], round(r['min_clearance'], 5), r['min_kind'], r['min_frame']) for r in rec60 if r['min_clearance'] < 0.03]),
                    'rec60_below_4cm': len([r for r in rec60 if r['min_clearance'] < 0.04])}
res['FAST_STD_lap_from_stop'] = by['FAST-STD'][0]['lap_from_stop']

# 学習の段の試験(eval_test.json)と 1 回ずつ比べる
E = json.load(open(EVAL_TEST))['tests']
cmp = {'p100_valid_mismatch': 0, 'p100_frames_mismatch': 0, 'p100_events_mismatch': 0, 'p100_success_mismatch': 0, 'p100_success_mismatch_ids': [],
       'p100_clear_maxdiff_valid': 0.0, 'kick_survive_mismatch': 0, 'kick_allowed_mismatch': 0, 'rec60_clear_maxdiff': 0.0, 'rec60_valid_mismatch': 0,
       'p100_rec_time_maxdiff': 0.0}
eP = {r['id']: r for r in E['RC-P100']}
for r in P:
    e = eP[r['id']]
    cmp['p100_valid_mismatch'] += int(e['valid_episode'] != r['valid_episode'])
    cmp['p100_frames_mismatch'] += int(e['frames'] != r['frames'])
    evs_e = [(x['type'], x['cls'], x['ps'], x['pe']) for x in e['events']]
    evs_r = [(x['type'], x['cls'], x['ps'], x['pe']) for x in r['events']]
    cmp['p100_events_mismatch'] += int(evs_e != evs_r)
    for a, b in zip(e['recovery'], r['_rec']):
        if a['success'] != b['okA']:
            cmp['p100_success_mismatch'] += 1
            cmp['p100_success_mismatch_ids'].append((r['id'], a['n'], a['recovered_s'], b['tA'], a['bad_after'], b['bad'], b['why']))
        if a['recovered_s'] is not None and b['tA'] is not None:
            cmp['p100_rec_time_maxdiff'] = max(cmp['p100_rec_time_maxdiff'], abs(a['recovered_s'] - b['tA']))
    if r['valid_episode'] and e.get('min_clearance') is not None:
        cmp['p100_clear_maxdiff_valid'] = max(cmp['p100_clear_maxdiff_valid'], abs(e['min_clearance'] - r['min_clearance']))
eK = {r['id']: r for r in E['RC-KICK']}
for r in K:
    e = eK[r['id']]
    cmp['kick_survive_mismatch'] += int(e['survived'] != r['valid_episode'])
    cmp['kick_allowed_mismatch'] += int(e['kick_allowed'] != r['kick']['allowed'])
eR = {r['id']: r for r in E['RC-REC60']}
for r in rec60:
    e = eR[r['id']]
    cmp['rec60_clear_maxdiff'] = max(cmp['rec60_clear_maxdiff'], abs(e['min_clearance'] - r['min_clearance']))
    cmp['rec60_valid_mismatch'] += int(e['valid_episode'] != r['valid_episode'])
cmp['std_lap_diff'] = abs(E['RC-STD'][0]['lap_from_stop'] - std['lap_from_stop']) if std['lap_from_stop'] else None
res['compare_with_training_eval'] = cmp

# ---- 2) 新しい開始・新しい種 ----
res['FRESH_P'] = p_summary(by['FRESH-P'], NL['recover'])
res['FRESH_P_FAST'] = p_summary(by['FRESH-P-FAST'], NL['fastest'])
res['FRESH_WM'] = p_summary(by['FRESH-WM'], NL['recover'])
for name in ('FRESH-KICK', 'FRESH-KICK-FAST'):
    d = {}
    for kind in ('yaw', 'lateral'):
        pre = [r for r in by[name] if r['spec']['kind'] == kind and not r['kick']]
        j = [r for r in by[name] if r['spec']['kind'] == kind and r['kick'] and r['kick']['allowed']]
        a = [r for r in by[name] if r['spec']['kind'] == kind and r['kick']]
        d[kind] = {'judged': len(j), 'survived': sum(r['valid_episode'] for r in j), 'all': len(a), 'all_survived': sum(r['valid_episode'] for r in a),
                   'ended_before_kick': len(pre), 'ended_before_kick_reasons': [(r['id'], r['reason'], r['frames'], r['spec']['kick']['frame']) for r in pre]}
    res[name.replace('-', '_')] = d

# ---- 3) ラベル・乱しの決まりの確かめ(乱しのある組の全部) ----
lab = {'episodes': 0, 'frames': 0, 'changed_but_l1': 0, 'l0_not_changed': 0, 'l0_outside_event_window': 0, 'event_l0_count_mismatch': 0,
       'info_bad_keys': 0, 'p0_missing': 0, 'p0_policy_prefix_bad': 0, 'spec_violations': [], 'zone_violations': 0, 'first_event_early': 0,
       'gap_short': 0, 'late_start': 0, 'steer_value_mismatch': 0, 'events': 0, 'types': {}, 'aborted_excursions': 0,
       'start_smod_hist': {}, 'mask_k_rule_mismatch': 0}
CL = REC['classes']
for name in ('REPLAY-P100', 'FRESH-P', 'FRESH-WM'):
    for r in by[name]:
        pf = r['pf']
        T = len(pf['l'])
        lab['episodes'] += 1
        lab['frames'] += T
        lab['changed_but_l1'] += r['labels']['changed_but_l1']
        lab['info_bad_keys'] += r['labels']['info_bad_keys']
        if not r['p0']:
            lab['p0_missing'] += 1
        elif r['p0'].get('policy') != '7720f2754972':
            lab['p0_policy_prefix_bad'] += 1
        inwin = np.zeros(T, bool)
        prev_end = None
        for k, ev in enumerate(r['events']):
            lab['events'] += 1
            lab['types'][ev['type']] = lab['types'].get(ev['type'], 0) + 1
            ps, pe = ev['ps'], ev['pe'] if ev['pe'] is not None else T
            inwin[ps:pe] = True
            n0 = sum(1 for f in range(ps, min(pe, T)) if pf['l'][f] == 0)
            if n0 != min(pe, T) - ps:
                lab['event_l0_count_mismatch'] += 1
            # 予定の決まり
            if k == 0 and ps < 60:
                lab['first_event_early'] += 1
            if prev_end is not None and ps - prev_end < 60:
                lab['gap_short'] += 1
            if ps > 555:
                lab['late_start'] += 1
            prev_end = ev['pe']
            sm = pf['s'][ps] % L
            bad_zone = False
            for d in np.arange(0, 2.0 + 1e-9, 0.05):
                qq = (sm + d) % L
                for z in (ZONES['black_sheet_zone_s'], ZONES['back_straight_zone_s']):
                    if z[0] <= qq <= z[1]:
                        bad_zone = True
            lab['zone_violations'] += int(bad_zone)
            hb = round(sm * 4) / 4
            lab['start_smod_hist'][hb] = lab['start_smod_hist'].get(hb, 0) + 1
            c = CL[ev['cls']]
            if ev['type'] == 'steer_pulse':
                fr, kk = ev['frames'], ev['k']
                if not (c['steer_pulse']['frames'][0] <= fr <= c['steer_pulse']['frames'][1]) or not (c['steer_pulse']['k'][0] - 1e-12 <= kk <= c['steer_pulse']['k'][1] + 1e-12):
                    lab['spec_violations'].append((r['set'], r['id'], ev['n'], 'steer', fr, kk))
                for f in range(ps, min(pe, T)):
                    if abs(pf['st'][f] - ev['sign'] * 0.52 * kk) > 2e-4:
                        lab['steer_value_mismatch'] += 1
            elif ev['type'] == 'excursion':
                if not (c['excursion']['d_m'][0] - 1e-12 <= ev['d'] <= c['excursion']['d_m'][1] + 1e-12) or not (c['excursion']['T_s'][0] - 1e-12 <= ev['T'] <= c['excursion']['T_s'][1] + 1e-12):
                    lab['spec_violations'].append((r['set'], r['id'], ev['n'], 'excursion', ev['d'], ev['T']))
                if ev['aborted']:
                    lab['aborted_excursions'] += 1
            elif ev['type'] == 'speed_pulse':
                rng = c['speed_pulse']['brake_frames' if ev['kind'] == 'brake' else 'burst_frames']
                if not (rng[0] <= ev['frames'] <= rng[1]):
                    lab['spec_violations'].append((r['set'], r['id'], ev['n'], 'speed', ev['kind'], ev['frames']))
            else:
                lab['spec_violations'].append((r['set'], r['id'], ev['n'], 'type', ev['type']))
        for f in range(T):
            if pf['l'][f] == 0 and not inwin[f]:
                lab['l0_outside_event_window'] += 1
            if pf['l'][f] == 0 and pf['ch'][f] == 0:
                lab['l0_not_changed'] += 1
        # 外す観測の決まり: k ∈ [ps − 10, pe − 2](記録の npz の正解 = k+1〜k+10 の操作)
        _, m = mask_frac(r)
        m2 = np.zeros(T, bool)
        for ev in r['events']:
            pe = ev['pe'] if ev['pe'] is not None else T
            m2[max(0, ev['ps'] - 10):max(0, pe - 1)] = True
        lab['mask_k_rule_mismatch'] += int((m != m2).sum())
lab['start_smod_hist'] = dict(sorted(lab['start_smod_hist'].items()))
lab['spec_violations'] = lab['spec_violations'][:20]
res['labels'] = lab

# ---- 4) ずる: すり抜け・ゲート・黒シート ----
gate_s = sorted(REF['gate_s'])
max_gap = max((gate_s[(i + 1) % 33] - gate_s[i]) % L for i in range(33))
# R が z = 0(x > 0)を横切る s(黒シート)
xs, zs, ss = REF['x'], REF['z'], REF['s']
bs_s = None
for i in range(len(xs)):
    j = (i + 1) % len(xs)
    if zs[i] > 0 >= zs[j] and xs[i] > 0:
        bs_s = ss[i]
ex = {'max_gate_gap_m': max_gap, 'black_sheet_s': bs_s, 'overlap_without_collision': 0, 'audit_pos_err_max': 0.0, 'gate_stuck_episodes': [],
      'virtual_miss_episodes': 0, 'valid_with_zero_laps': 0, 'black_sheet_pass_without_inside_crossing': [], 'proj_vs_runtime_max': [0, 0], 'valid_eps': 0}
for name, rs in by.items():
    for r in rs:
        if r['collisions'] == 0 and r['overlap_frames']:
            ex['overlap_without_collision'] += 1
        if r['reason'] in ('collection_duration_complete', 'route_complete') and r['audit_pos_err'] is not None:
            ex['audit_pos_err_max'] = max(ex['audit_pos_err_max'], r['audit_pos_err'])
        ex['proj_vs_runtime_max'][0] = max(ex['proj_vs_runtime_max'][0], r['proj_vs_runtime']['ds'])
        ex['proj_vs_runtime_max'][1] = max(ex['proj_vs_runtime_max'][1], r['proj_vs_runtime']['dey'])
        if r['virtual_misses']:
            ex['virtual_miss_episodes'] += 1
        s, g = r['pf']['s'], r['pf']['g']
        last_s = s[0]
        stuck = 0.0
        for f in range(1, len(s)):
            if g[f] != g[f - 1]:
                last_s = s[f]
            stuck = max(stuck, s[f] - last_s)
        if stuck > max_gap + 0.5:
            ex['gate_stuck_episodes'].append((name, r['id'], round(stuck, 2), r['valid_episode']))
        if r['valid_episode']:
            ex['valid_eps'] += 1
            if not r['laps']:
                ex['valid_with_zero_laps'] += 1
            # 黒シートの s を通った回数と、開口の中の横断の数
            passes = 0
            for f in range(1, len(s)):
                if math.floor((s[f - 1] - bs_s) / L) != math.floor((s[f] - bs_s) / L) and s[f] > s[f - 1]:
                    passes += 1
            inside = sum(1 for c in r['own_crossings'] if c['inside'])
            if inside < passes:
                ex['black_sheet_pass_without_inside_crossing'].append((name, r['id'], passes, inside))
ex['gate_stuck_episodes'] = ex['gate_stuck_episodes'][:30]
res['exploits'] = ex


# ---- 5) ばらつき・最速との違い ----
def vm7(rs, use_mask=False, N=9000, seed=12345):
    pts = []
    for r in rs:
        pf = r['pf']
        s0 = pf['s'][0]
        m = mask_frac(r)[1] if use_mask else None
        for i in range(len(pf['s'])):
            if pf['s'][i] - s0 < 3:
                continue
            if use_mask and m[i]:
                continue
            pts.append((pf['s'][i], pf['ey'][i], pf['epsi'][i], pf['v'][i]))
    if len(pts) > N:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pts))[:N]
        pts = [pts[i] for i in idx]
    cells = set()
    for s, e, h, v in pts:
        cells.add((int((s % L) // 0.25), math.floor(e / 0.05), math.floor(h / 0.1), math.floor(v / 0.5)))
    return len(cells), len(pts)


def first_valid(rs, n=15):
    return [r for r in rs if r['valid_episode']][:n]


var = {}
for name in ('FRESH-P', 'FRESH-P-FAST', 'FRESH-U', 'FRESH-U-FAST', 'FRESH-WM', 'REPLAY-REC60'):
    v = first_valid(by[name])
    c, n = vm7(v)
    cm, nm = vm7(v, use_mask=True)
    # 有効な回から 15 シーンを 20 回引いたときの幅
    allv = [r for r in by[name] if r['valid_episode']]
    rng = np.random.default_rng(7)
    draws = []
    if len(allv) >= 15:
        for _ in range(20):
            pick = [allv[i] for i in rng.choice(len(allv), 15, replace=False)]
            draws.append(vm7(pick)[0])
    var[name] = {'valid': len(allv), 'of': len(by[name]), 'vm7_first15': c, 'frames': n, 'vm7_first15_unmasked_only': cm, 'frames_unmasked': nm,
                 'vm7_draws_min_med_max': [min(draws), float(np.median(draws)), max(draws)] if draws else None,
                 'events_per_valid_scene': float(np.mean([len(r['events']) for r in allv])) if allv else None,
                 'masked_fraction_valid': float(np.mean([mask_frac(r)[0] for r in allv])) if allv else None,
                 'min_clearance_valid': min([r['min_clearance'] for r in allv], default=None),
                 'clearance_valid_p5_of_scene_min': q([r['min_clearance'] for r in allv], .05)}
res['variation'] = var


def profile(r, key):
    pf = r['pf']
    s0 = pf['s'][0]
    out = np.full(NB, np.nan)
    for i in range(len(pf['s'])):
        if pf['s'][i] - s0 < 3:
            continue
        b = int((pf['s'][i] % L) / 0.05) % NB
        if np.isnan(out[b]):
            out[b] = pf[key][i]
    return out


# 乱しなし: 離脱復活の方策 vs 最速の方策(同じ開始・同じ種)
U, UF = {r['id']: r for r in by['FRESH-U']}, {r['id']: r for r in by['FRESH-U-FAST']}
d_ey, d_v, lap_r, lap_f, cl_r, cl_f, vr, vf = [], [], [], [], [], [], [], []
for k in U:
    a, b = U[k], UF[k]
    if a['valid_episode'] and b['valid_episode']:
        pa, pb = profile(a, 'ey'), profile(b, 'ey')
        ok = ~np.isnan(pa) & ~np.isnan(pb)
        d_ey.append(float(np.mean(np.abs(pa[ok] - pb[ok]))))
        va, vb = profile(a, 'v'), profile(b, 'v')
        d_v.append(float(np.mean(va[ok] - vb[ok])))
        if a['laps']:
            lap_r.append(a['laps'][0]['elapsed'])
        if b['laps']:
            lap_f.append(b['laps'][0]['elapsed'])
        cl_r.append(a['min_clearance'])
        cl_f.append(b['min_clearance'])
    vr += [x for x in a['pf']['v'] if x > 0.3]
    vf += [x for x in b['pf']['v'] if x > 0.3]
# 方策の中の線のばらつき(開始から 3 m 以降のマスで、全シーンの平均からの |差| の平均)
def spread(rs, key):
    profs = [profile(r, key) for r in rs if r['valid_episode']]
    if not profs:
        return None
    A = np.vstack(profs)
    ok = ~np.isnan(A).any(axis=0)
    A = A[:, ok]
    return float(np.mean(np.abs(A - A.mean(axis=0))))


res['vs_fastest_unperturbed'] = {
    'pairs_both_valid': len(d_ey), 'valid_recover': sum(r['valid_episode'] for r in by['FRESH-U']), 'valid_fastest': sum(r['valid_episode'] for r in by['FRESH-U-FAST']),
    'mean_abs_dey_m': float(np.mean(d_ey)) if d_ey else None, 'p90_abs_dey_m': q(d_ey, .9),
    'mean_dv_mps(recover-fastest)': float(np.mean(d_v)) if d_v else None,
    'mean_running_speed': [float(np.mean(vr)), float(np.mean(vf))],
    'lap1_median_s': [q(lap_r, .5), q(lap_f, .5)],
    'min_clearance_median_cm': [q(cl_r, .5) * 100 if cl_r else None, q(cl_f, .5) * 100 if cl_f else None],
    'min_clearance_min_cm': [min(cl_r) * 100 if cl_r else None, min(cl_f) * 100 if cl_f else None],
    'spread_ey_m_recover_vs_fastest': [spread(by['FRESH-U'], 'ey'), spread(by['FRESH-U-FAST'], 'ey')],
    'spread_ey_m_perturbed_recover_vs_fastestpolicy': [spread(by['FRESH-P'], 'ey'), spread(by['FRESH-P-FAST'], 'ey')],
}
# 乱しあり: 同じ開始・同じ種・同じ予定(方策だけ違う)。種と開始が同じでも走りが変われば乱しの位置は変わる
FP, FPF = {r['id']: r for r in by['FRESH-P']}, {r['id']: r for r in by['FRESH-P-FAST']}
both = sum(1 for k in FP if FP[k]['valid_episode'] and FPF[k]['valid_episode'])
only_r = sum(1 for k in FP if FP[k]['valid_episode'] and not FPF[k]['valid_episode'])
only_f = sum(1 for k in FP if not FP[k]['valid_episode'] and FPF[k]['valid_episode'])
res['vs_fastest_perturbed'] = {'both_valid': both, 'only_recover_valid': only_r, 'only_fastest_valid': only_f,
                               'valid': [sum(r['valid_episode'] for r in by['FRESH-P']), sum(r['valid_episode'] for r in by['FRESH-P-FAST'])]}
# 立て直しの成否と乱しの始まりの速さ
ev = [e for r in by['FRESH-P'] + by['REPLAY-P100'] for e in r['_rec']]
sp = [e for e in ev if e['type'] == 'steer_pulse' and e['v_ps'] is not None]
bins = [(0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 9)]
res['steer_pulse_success_by_speed'] = {f'{a}-{b}': {'n': len([e for e in sp if a <= e['v_ps'] < b]), 'ok': len([e for e in sp if a <= e['v_ps'] < b and e['okA']])} for a, b in bins}
res['success_by_start_zone'] = {}
for name, (a, b) in {'black_sheet_exit_8.00-8.25': (8.0, 8.25), 'back_straight_exit_19.30-19.55': (19.3, 19.55)}.items():
    sel = [e for e in ev if e['smod'] is not None and a <= e['smod'] < b]
    res['success_by_start_zone'][name] = {'n': len(sel), 'ok': sum(e['okA'] for e in sel), 'share': len(sel) / max(1, len(ev))}
rest = [e for e in ev if e['smod'] is not None and not (8.0 <= e['smod'] < 8.25 or 19.3 <= e['smod'] < 19.55)]
res['success_by_start_zone']['other'] = {'n': len(rest), 'ok': sum(e['okA'] for e in rest), 'share': len(rest) / max(1, len(ev))}
# 衝突したときの乱しとの関係(衝突の前の最後の出来事の種類と、その終わりからの時間)
coll = []
for r in by['FRESH-P'] + by['REPLAY-P100']:
    if r['collisions']:
        T = r['frames']
        last = [e for e in r['events'] if e['ps'] <= T]
        if last:
            e = last[-1]
            pe = e['pe'] if e['pe'] is not None else T
            coll.append({'type': e['type'], 'cls': e['cls'], 'during': e['pe'] is None, 'after_s': (T - pe) / 30})
        else:
            coll.append({'type': None, 'cls': None, 'during': False, 'after_s': None})
res['collisions_context'] = {'n': len(coll), 'no_event_before': sum(1 for c in coll if c['type'] is None),
                             'during_event': sum(1 for c in coll if c['during']),
                             'within_3s_after': sum(1 for c in coll if c['after_s'] is not None and not c['during'] and c['after_s'] <= 3),
                             'later_than_3s': sum(1 for c in coll if c['after_s'] is not None and c['after_s'] > 3),
                             'by_type': {t: sum(1 for c in coll if c['type'] == t) for t in ('steer_pulse', 'excursion', 'speed_pulse', None)}}
for r in rows:
    r.pop('_rec', None)
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))[:20000])
