#!/usr/bin/env python3
# 経路いろいろ(line)の確認 r2: 走らせた組(runs/*.jsonl)から判定 L1〜L6 と、ずる・ラベル・最速との違いを自前で計算する(2026-10-14、確認の段)。
# 学習・評価のコード(eval_modes.mjs・variation_metrics.mjs・line_r2/*)を使わず、protocol.json acceptance.line・definitions・variation_metrics と
# protocol_line_r2.json の文言から書いた(r1 の確認の analyze_ln.py を r2 向けに直したもの)。判定の追従の誤差は自前の e*(own_line_v2.mjs)。
#   python3 analyze_ln2.py > summary.json
import json, math, os, statistics
from collections import Counter, defaultdict

REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013'
RUNS = f'{REP}/check_line_r2/runs'
L_LOOP = 26.119648724030924
NB = math.ceil(L_LOOP / 0.05)
HAIRPIN = (9.75, 11.75)   # R の左の急な曲がり(学習の段が 12 cm 越えの 97% と書いた区間)
S_ZONES = [(2.0, 4.5), (8.0, 12.5)]


def load(name):
    p = f'{RUNS}/{name}.jsonl'
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def q(a, p):
    a = sorted(a)
    if not a:
        return None
    i = p * (len(a) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo] + (a[hi] - a[lo]) * (i - lo)


def spread(profs):
    # VM2/VM3 の式: 全員がそろうマスで、各人と群の平均の |差| の平均
    if len(profs) < 2:
        return None, 0
    tot = n = bins = 0
    for b in range(len(profs[0])):
        v = [p[b] for p in profs]
        if any(x is None for x in v):
            continue
        m = sum(v) / len(v)
        tot += sum(abs(x - m) for x in v)
        n += len(v)
        bins += 1
    return (tot / n if n else None), bins


def tracking(rows, which='own'):
    # |e_y − e*(s)|、開始から 3 m 以降。fr = [t, s−s0, s mod L, e_y, e*_rt, e*_own, すき間, v, 舵, 速さ]
    out = []
    for r in rows:
        for f in r.get('fr') or []:
            if f[1] is None or f[1] < 3:
                continue
            es = f[5] if which == 'own' else f[4]
            if es is None:
                continue
            out.append((abs(f[3] - es), f[2], r['id']))
    return out


def judge(p60, fam6):
    ok = [r for r in p60 if r['valid_episode']]
    J = {}
    J['L1_valid'] = {'value': f'{len(ok)}/{len(p60)}', 'pass': len(ok) >= 57 and len(p60) == 60}
    pool = ok + fam6
    mc = min(r['min_clearance'] for r in pool) if pool else None
    srt = sorted(pool, key=lambda r: r['min_clearance'])
    J['L2_clearance'] = {'value': mc, 'pass': mc is not None and mc >= 0.03, 'lowest5': [(r['id'], round(r['min_clearance'], 4)) for r in srt[:5]]}
    te = [x[0] for x in tracking(ok, 'own')]
    te_rt = [x[0] for x in tracking(ok, 'rt')]
    J['L3_tracking'] = {'value': {'median': q(te, .5), 'p95': q(te, .95), 'n': len(te)}, 'runtime_estar': {'median': q(te_rt, .5), 'p95': q(te_rt, .95)},
                        'pass': bool(te) and q(te, .5) <= 0.05 and q(te, .95) <= 0.12}
    F = {r['id']: r for r in fam6}
    if len(F) == 6 and all(r['valid_episode'] for r in F.values()):
        ids = list(F)
        bins = [b for b in range(NB) if all(F[i]['prof']['ey'][b] is not None for i in ids)]
        d = lambda a, b: [abs(F[a]['prof']['ey'][k] - F[b]['prof']['ey'][k]) for k in bins]
        lr = statistics.mean(d('c+0.8', 'c-0.8'))
        io = statistics.mean(d('inside', 'outside'))
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dd = d(ids[i], ids[j])
                # 1 m の移動平均(20 マス)。きびしめに、周をまたいでつながない(r1 の確認と同じ)
                best = max((sum(dd[k:k + 20]) / 20 for k in range(0, len(dd) - 19)), default=0)
                pairs.append((ids[i], ids[j], best))
        mp = min(p[2] for p in pairs)
        J['L4_distinct'] = {'value': {'left_right_mean': lr, 'inside_outside_mean': io, 'min_pair_max_1m_avg': mp, 'min_pair': [p for p in pairs if p[2] == mp][0][:2], 'bins': len(bins)},
                            'pass': lr >= .2 and io >= .1 and mp >= .08}
    else:
        J['L4_distinct'] = {'value': None, 'pass': False, 'note': 'FAM6 が 6 回そろって有効でない'}
    vm3, nb = spread([r['prof']['ey'] for r in ok])
    J['L5_vm3'] = {'value': vm3, 'bins': nb, 'pass': vm3 is not None and vm3 >= .08}
    c0 = F.get('c0')
    lap = c0['laps'][0]['elapsed'] if c0 and c0['laps'] else None
    J['L6_lap'] = {'value': lap, 'pass': bool(c0 and c0['valid_episode'] and lap is not None and lap <= 10.0)}
    J['all_pass'] = all(v['pass'] for v in J.values() if isinstance(v, dict) and 'pass' in v)
    return J


def exploit_checks(rows):
    c = Counter()
    worst = {'audit_pos_err': 0, 'sample_exact_minus_sample': -1, 'es_own_minus_rt': 0, 'es_info_mm_minus_rt': 0, 'proj_dey': 0, 'proj_ds': 0}
    worst_ids = {}
    for r in rows:
        if r.get('invalid_spawn'):
            c['invalid_spawn'] += 1
            continue
        c['runs'] += 1
        if r['overlap_frames'] > 0 and r['collisions'] == 0:
            c['overlap_without_collision'] += 1
        if r['collisions'] > 0:
            c['collided'] += 1
        worst['audit_pos_err'] = max(worst['audit_pos_err'], r['audit_pos_err'])
        sc = r['sample_check']
        if sc['n'] and sc['max_exact_minus_sample'] is not None:
            worst['sample_exact_minus_sample'] = max(worst['sample_exact_minus_sample'], sc['max_exact_minus_sample'])
        if r['es_check']['n']:
            if r['es_check']['max_abs_own_minus_rt'] > worst['es_own_minus_rt']:
                worst_ids['es_own_minus_rt'] = r['id']
            worst['es_own_minus_rt'] = max(worst['es_own_minus_rt'], r['es_check']['max_abs_own_minus_rt'])
            worst['es_info_mm_minus_rt'] = max(worst['es_info_mm_minus_rt'], r['es_check']['max_abs_info_mm_minus_rt'])
        if r['proj_check']['max_dey_vs_runtime'] > worst['proj_dey']:
            worst_ids['proj_dey'] = r['id']
        worst['proj_dey'] = max(worst['proj_dey'], r['proj_check']['max_dey_vs_runtime'])
        worst['proj_ds'] = max(worst['proj_ds'], r['proj_check']['max_ds_vs_runtime'])
        if r['valid_episode']:
            for lap in r['laps']:
                if lap['gates'] != 33 or lap['crossings'] != 1 or lap['collisions'] != 0:
                    c['bad_lap'] += 1
            if r['passed_gates'] < r['gates_by_s'] - 1:
                c['gate_count_short'] += 1
            if r['gate_stuck_max'] > 0:
                c['gate_stuck'] += 1
            nin = sum(1 for x in r['own_crossings'] if x['inside'])
            c['own_crossings'] += len(r['own_crossings'])
            if nin != len(r['own_crossings']):
                c['crossing_outside_opening'] += 1
            if len(r['laps']) < 2:
                c['valid_but_lt2_laps'] += 1
        L = r['labels']
        if L['l0']:
            c['label_l0_runs'] += 1
        if set(L['ph']) - {'drive'}:
            c['label_ph_not_drive'] += 1
        if L['ev_nonzero'] or L['sh_key'] or L['bad_keys']:
            c['label_bad'] += 1
        if r['mode'] == 'line' and L['es_missing']:
            c['es_missing'] += 1
        if L['cmd_over_cap']:
            c['cmd_over_cap'] += 1
        c['bytes_max'] = max(c['bytes_max'], L['bytes_max'])
        p0 = L['p0'] or {}
        if r['mode'] == 'line':
            if p0.get('line') != r['params'].get('line') or p0.get('c_v') != r['params'].get('c_v'):
                c['p0_mismatch'] += 1
            if p0.get('line_profile') != 'v2' or L.get('meta_line_profile') != 'jf-line-profile/v2':
                c['not_v2'] += 1
    return dict(c), worst, worst_ids


def low_runs(rows, thr=.05):
    low = [r for r in rows if not r.get('invalid_spawn') and r['min_clearance'] < thr]
    return [{'id': r['id'], 'family': r['family'], 'line': r['params'].get('line'), 'c_v': r['params'].get('c_v'),
             'start': {k: round(v, 3) if isinstance(v, float) else v for k, v in r['start'].items()},
             'start_dpsi_vs_R_rad': round(r['start_dpsi_vs_R'], 3), 'start_s': round(r['start_s_mod'], 2), 'start_ey': round(r['start_ey'], 3),
             'min_clearance': round(r['min_clearance'], 4), 'min_t': round(r['min_t'], 2), 'kind': r['min_kind'], 'min_s_mod': round(r['min_s_mod'], 2),
             'min_s_from_start': round(r['min_s_from_start'], 2), 'min_v': round(r['min_v'], 2), 'min_ey': round(r['min_ey'], 3),
             'min_es': None if r['min_es'] is None else round(r['min_es'], 3), 'R_static_clr_there': round(r['min_R_static_clearance'], 4),
             'collided': r['collisions'] > 0, 'collision': r.get('collision'), 'start_clearance': round(r['start_clearance'], 3)}
            for r in sorted(low, key=lambda r: r['min_clearance'])]


def start_phase(rows):
    ok = [r for r in rows if not r.get('invalid_spawn')]
    val = [r for r in ok if r['valid_episode']]
    m = lambda k, rs: min((r[k] for r in rs if r[k] is not None), default=None)
    return {'runs': len(ok), 'collided': sum(1 for r in ok if r['collisions']), 'collision_t': sorted(round(r['collision']['t'], 2) for r in ok if r.get('collision')),
            'min_lt3cm_valid': sum(1 for r in val if r['min_clearance'] < .03), 'min_lt5cm_valid': sum(1 for r in val if r['min_clearance'] < .05),
            'clr_first_1p5s_min_valid': m('clr_first_1p5s', val), 'clr_after_1p5s_min_valid': m('clr_after_1p5s', val),
            'clr_after_blend_2m_min_valid': m('clr_after_blend_2m', val), 'clr_after_3m_min_valid': m('clr_after_3m', val)}


def zone_share(rows):
    te = tracking([r for r in rows if r['valid_episode']])
    if not te:
        return None
    big = [x for x in te if x[0] > .12]
    inH = [x for x in big if HAIRPIN[0] <= x[1] <= HAIRPIN[1]]
    outH = [x[0] for x in te if not (HAIRPIN[0] <= x[1] <= HAIRPIN[1])]
    return {'frames': len(te), 'frac_gt12': len(big) / len(te), 'share_of_gt12_in_hairpin': len(inH) / len(big) if big else None,
            'hairpin_frame_share': sum(1 for x in te if HAIRPIN[0] <= x[1] <= HAIRPIN[1]) / len(te), 'p95_outside_hairpin': q(outH, .95)}


def by_family(rows):
    g = defaultdict(list)
    for r in rows:
        g[r['family']].append(r)
    out = {}
    for f, rs in g.items():
        ok = [r for r in rs if r['valid_episode']]
        te = [x[0] for x in tracking(ok)]
        out[f] = {'n': len(rs), 'valid': len(ok), 'trk_median': q(te, .5), 'trk_p95': q(te, .95),
                  'lap1_mean': statistics.mean([r['laps'][0]['elapsed'] for r in ok if r['laps']]) if ok else None,
                  'min_clr': min(r['min_clearance'] for r in rs), 'v_mean': statistics.mean(r['speed']['mean_running'] for r in rs)}
    return out


def vm1(rows):
    ss = [[f[8] for f in r['fr']] for r in rows if r['valid_episode'] and len(r.get('fr') or []) == 600]
    tot = n = 0
    for t in range(600):
        v = [s[t] for s in ss]
        m = sum(v) / len(v)
        tot += sum(abs(x - m) for x in v)
        n += len(v)
    return tot / n if n else None


def compare_trainer(rep_rows):
    ev = json.load(open(f'{REP}/train/line/r2/eval_test_r2.json'))
    T = {r['id']: r for k in ev['tests'] for r in ev['tests'][k]}
    diffs = {'valid_mismatch': [], 'reason_mismatch': [], 'max_abs_min_clearance_diff': 0, 'laps_mismatch': [], 'max_abs_tracking_p95_diff_per_run': 0, 'n': 0}
    for r in rep_rows:
        t = T.get(r['id'])
        if not t:
            continue
        diffs['n'] += 1
        if t['valid_episode'] != r['valid_episode']:
            diffs['valid_mismatch'].append(r['id'])
        if t['reason'] != r['reason']:
            diffs['reason_mismatch'].append(r['id'])
        diffs['max_abs_min_clearance_diff'] = max(diffs['max_abs_min_clearance_diff'], abs(t['min_clearance'] - r['min_clearance']))
        if [round(x, 6) for x in t['laps']] != [round(l['elapsed'], 6) for l in r['laps']]:
            diffs['laps_mismatch'].append(r['id'])
        if t.get('tracking') and r.get('tracking_own'):
            diffs['max_abs_tracking_p95_diff_per_run'] = max(diffs['max_abs_tracking_p95_diff_per_run'], abs(t['tracking']['p95'] - r['tracking_own']['p95']))
    diffs['trainer_judged'] = {k: v.get('value') for k, v in ev['judged'].items()}
    return diffs


def vs_fastest(line_rows, fast_rows):
    F = {r['id']: r for r in fast_rows}
    per = []
    for r in line_rows:
        f = F.get(r['id'])
        if not f or not r['valid_episode'] or not f['valid_episode']:
            continue
        a, b = r['prof']['ey'], f['prof']['ey']
        d = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
        per.append({'id': r['id'], 'family': r['family'], 'mean_abs_dey': statistics.mean(d), 'lap1_line': r['laps'][0]['elapsed'], 'lap1_fast': f['laps'][0]['elapsed'],
                    'v_line': r['speed']['mean_running'], 'v_fast': f['speed']['mean_running'], 'clr_line': r['min_clearance'], 'clr_fast': f['min_clearance']})
    if not per:
        return None
    fam = defaultdict(list)
    for p in per:
        fam[p['family']].append(p['mean_abs_dey'])
    ids = {p['id'] for p in per}
    return {'pairs': len(per), 'mean_abs_dey_mean': statistics.mean(p['mean_abs_dey'] for p in per), 'mean_abs_dey_min': min(p['mean_abs_dey'] for p in per),
            'mean_abs_dey_by_family': {k: statistics.mean(v) for k, v in fam.items()},
            'lap1_line_mean': statistics.mean(p['lap1_line'] for p in per), 'lap1_fast_mean': statistics.mean(p['lap1_fast'] for p in per),
            'v_line_mean': statistics.mean(p['v_line'] for p in per), 'v_fast_mean': statistics.mean(p['v_fast'] for p in per),
            'clr_line_median': q([p['clr_line'] for p in per], .5), 'clr_fast_median': q([p['clr_fast'] for p in per], .5),
            'clr_line_min': min(p['clr_line'] for p in per), 'clr_fast_min': min(p['clr_fast'] for p in per),
            'vm3_fast_same_starts': spread([F[i]['prof']['ey'] for i in ids])[0], 'vm3_line_same_starts': spread([r['prof']['ey'] for r in line_rows if r['id'] in ids])[0],
            'fast_valid': f"{sum(1 for r in fast_rows if r['valid_episode'])}/{len(fast_rows)}"}


def l2_fail_prob(rows, n_test=60):
    # 1 回あたりの「有効で最小すき間 < 3 cm」の割合 p から、60 回の試験で 1 回以上出る確率 1 − (1 − p)^60
    ok = [r for r in rows if not r.get('invalid_spawn')]
    k = sum(1 for r in ok if r['valid_episode'] and r['min_clearance'] < .03)
    p = k / len(ok) if ok else None
    return {'k': k, 'n': len(ok), 'p': p, 'p_test_fail': None if p is None else 1 - (1 - p) ** n_test}


def main():
    out = {}
    rep = load('REPLAY')
    p60 = [r for r in rep if r['test'] == 'LN-P60']
    fam6 = [r for r in rep if r['test'] == 'LN-FAM6']
    if rep:
        out['replay'] = {'judged': judge(p60, fam6), 'exploit': exploit_checks(rep), 'start_phase': start_phase(p60 + fam6), 'low_runs_lt5cm': low_runs(p60 + fam6),
                         'zones': zone_share(p60), 'by_family': by_family(p60), 'vm1_us': vm1(p60), 'vs_trainer': compare_trainer(rep),
                         'invalid': [{'id': r['id'], 'reason': r['reason'], 'collision': r['collision'], 'family': r['family'], 'line': r['params'].get('line'),
                                      'min_s_mod': round(r['min_s_mod'], 2), 'min_v': round(r['min_v'], 2)} for r in p60 if not r['valid_episode']],
                         'start_dpsi_ge_1rad': [{'id': r['id'], 'dpsi': round(r['start_dpsi_vs_R'], 3), 'min_clr': round(r['min_clearance'], 4), 'valid': r['valid_episode']}
                                                for r in p60 if abs(r['start_dpsi_vs_R']) >= 1.0],
                         'vm4_speed_spread': spread([r['prof']['v'] for r in p60 if r['valid_episode']])[0]}
    fr = load('FRESH')
    if fr:
        valid = [r for r in fr if r['valid_episode']]
        blocks = []
        for b in range(len(fr) // 60):
            blk = fr[60 * b:60 * (b + 1)]
            j = judge(blk, [])
            blocks.append({'block': b, 'L1': j['L1_valid']['value'], 'L2': j['L2_clearance']['value'], 'L2_low': j['L2_clearance']['lowest5'][:3],
                           'L3_median': j['L3_tracking']['value']['median'], 'L3_p95': j['L3_tracking']['value']['p95'], 'L5_vm3': j['L5_vm3']['value'],
                           'pass_L1': j['L1_valid']['pass'], 'pass_L2': j['L2_clearance']['pass'], 'pass_L3': j['L3_tracking']['pass'], 'pass_L5': j['L5_vm3']['pass']})
        te = [x[0] for x in tracking(valid)]
        mc = sorted(r['min_clearance'] for r in valid)
        hi = [r for r in fr if abs(r['start_dpsi_vs_R']) >= 1.0]
        out['fresh'] = {'runs': len(fr), 'valid': len(valid), 'reasons': dict(Counter(r['reason'] for r in fr)), 'blocks_of_60': blocks,
                        'pooled_tracking': {'median': q(te, .5), 'p95': q(te, .95), 'frac_gt12': sum(1 for x in te if x > .12) / len(te)},
                        'vm3_all_valid': spread([r['prof']['ey'] for r in valid])[0], 'vm1_us': vm1(valid),
                        'min_clr_valid': {'min': mc[0], 'p05': q(mc, .05), 'median': q(mc, .5), 'lt3cm': sum(1 for x in mc if x < .03), 'lt5cm': sum(1 for x in mc if x < .05)},
                        'start_phase': start_phase(fr), 'low_runs_lt5cm': low_runs(fr), 'zones': zone_share(fr), 'by_family': by_family(fr), 'exploit': exploit_checks(fr),
                        'abs_dpsi_ge_1rad': {'n': len(hi), 'valid': sum(1 for r in hi if r['valid_episode']), 'min_clr': min((r['min_clearance'] for r in hi), default=None)},
                        'l2_fail_prob': l2_fail_prob(fr), 'invalid': [{'id': r['id'], 'reason': r['reason'], 'collision': r['collision'], 'family': r['family'],
                                                                     'dpsi': round(r['start_dpsi_vs_R'], 3), 'gate': r['start']['gate']} for r in fr if not r['valid_episode']]}
        fast = load('FRESH_FAST')
        if fast:
            out['vs_fastest'] = vs_fastest(fr, fast)
            out['fastest_fresh'] = {'runs': len(fast), 'valid': sum(1 for r in fast if r['valid_episode']), 'min_clr': min(r['min_clearance'] for r in fast),
                                    'exploit': exploit_checks(fast)[0]}
    hd = load('HEAD')
    if hd:
        ok = [r for r in hd if r['valid_episode']]
        out['head'] = {'runs': len(hd), 'valid': len(ok), 'collided': sum(1 for r in hd if r['collisions']), 'l2_fail_prob': l2_fail_prob(hd),
                       'min_clr_valid': min((r['min_clearance'] for r in ok), default=None), 'lt3cm_valid': sum(1 for r in ok if r['min_clearance'] < .03),
                       'lt5cm_valid': sum(1 for r in ok if r['min_clearance'] < .05), 'by_gate': {}, 'low_runs_lt5cm': low_runs(hd),
                       'invalid': [{'id': r['id'], 'reason': r['reason'], 'collision': r['collision'], 'family': r['family'], 'gate': r['start']['gate'],
                                    'dpsi': round(r['start_dpsi_vs_R'], 3), 'start': r['start']} for r in hd if not r['valid_episode']],
                       'exploit': exploit_checks(hd)[0], 'start_phase': start_phase(hd)}
        g = defaultdict(list)
        for r in hd:
            g[r['start']['gate']].append(r)
        out['head']['by_gate'] = {k: {'n': len(v), 'valid': sum(1 for r in v if r['valid_episode']), 'lt3cm_valid': sum(1 for r in v if r['valid_episode'] and r['min_clearance'] < .03)}
                                  for k, v in sorted(g.items())}
    ex = load('EXT')
    if ex:
        ok = [r for r in ex if r['valid_episode']]
        te = [x[0] for x in tracking(ok)]
        out['ext'] = {'runs': len(ex), 'valid': len(ok), 'l2_fail_prob': l2_fail_prob(ex), 'min_clr_valid': min((r['min_clearance'] for r in ok), default=None),
                      'tracking': {'median': q(te, .5), 'p95': q(te, .95)}, 'vm3': spread([r['prof']['ey'] for r in ok])[0], 'by_family': by_family(ex),
                      'low_runs_lt5cm': low_runs(ex), 'invalid': [{'id': r['id'], 'reason': r['reason'], 'collision': r['collision'], 'family': r['family'],
                                                                   'dpsi': round(r['start_dpsi_vs_R'], 3), 'gate': r['start']['gate']} for r in ex if not r['valid_episode']],
                      'exploit': exploit_checks(ex)[0], 'start_phase': start_phase(ex)}
    fb = load('FB_MIX')
    if fb:
        ok = [r for r in fb if r['valid_episode']]
        l2 = min(r['min_clearance'] for r in ok) if ok else None
        vm3 = spread([r['prof']['ey'] for r in ok])[0]
        out['fallback_mix'] = {'L1': f'{len(ok)}/{len(fb)}', 'L2': l2, 'L2_argmin': min(ok, key=lambda r: r['min_clearance'])['id'] if ok else None, 'L5_vm3': vm3,
                               'pass': len(ok) >= 57 and l2 is not None and l2 >= .03 and vm3 is not None and vm3 >= .08, 'policies': sorted({r['policy_sha256'][:12] for r in fb})}
    v = load('FAM6V')
    if v:
        res = {}
        allv = [r for r in v if r['test'] == 'FAM6V']
        for k in range(len(allv) // 6):
            blk = allv[6 * k:6 * (k + 1)]
            for r in blk:
                r['id'] = r['family']
            j = judge([], blk)
            res[f'start_{k}'] = {'start': blk[0]['start'], 'L2': j['L2_clearance']['value'], 'L4': j['L4_distinct']['value'], 'L4_pass': j['L4_distinct']['pass'],
                                 'L6': j['L6_lap']['value'], 'valid': sum(1 for r in blk if r['valid_episode'])}
        c9 = [r for r in v if r['test'] == 'FAM6_CV09']
        for r in c9:
            r['id'] = r['family']
        j = judge([], c9)
        res['cv0.9_std'] = {'L2': j['L2_clearance']['value'], 'L4': j['L4_distinct']['value'], 'L4_pass': j['L4_distinct']['pass'], 'L6': j['L6_lap']['value'],
                            'valid': sum(1 for r in c9 if r['valid_episode'])}
        fs = [r for r in v if r['test'] == 'FAST_STD']
        if fs and rep:
            res['fastest_std_continuous'] = {'laps': [l['elapsed'] for l in fs[0]['laps']], 'min_clr': fs[0]['min_clearance']}
            F = {r['id']: r for r in rep if r['test'] == 'LN-FAM6'}
            res['fam6_vs_fastest_mean_abs_dey'] = {k: statistics.mean([abs(a - b) for a, b in zip(F[k]['prof']['ey'], fs[0]['prof']['ey']) if a is not None and b is not None]) for k in F}
        res['L4_pair_min_over_starts'] = min((x['L4']['min_pair_max_1m_avg'] for k, x in res.items() if k.startswith('start_') and x['L4']), default=None)
        res['L4_fail_starts'] = [k for k, x in res.items() if k.startswith('start_') and not x['L4_pass']]
        out['fam6_variants'] = res
    print(json.dumps(out, ensure_ascii=False, default=lambda o: None))


main()
