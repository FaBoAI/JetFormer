#!/usr/bin/env python3
# 経路いろいろ(line)の確認 r1: 走らせた組(runs/*.jsonl)から判定 L1〜L6 と、ずる・ラベル・最速との違いを自前で計算する(2026-10-14、確認の段)。
# 学習・評価のコード(eval_modes.mjs・variation_metrics.mjs)を使わず、protocol.json acceptance.line・definitions・variation_metrics の文言から書いた。
#   python3 analyze_ln.py > summary.json
import json, math, os, statistics, sys
from collections import Counter, defaultdict

REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013'
RUNS = f'{REP}/check_line_r1/runs'
L_LOOP = 26.119648724030924
NB = math.ceil(L_LOOP / 0.05)
S_ZONES = [(2.0, 4.5), (8.0, 12.5)]
CUSP = (10.2, 11.4)   # e*·κ_R が 0.7 を越える左の曲がり(target_audit)


def load(name):
    p = f'{RUNS}/{name}.jsonl'
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


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


def tracking(rows, which='rt'):
    # |e_y − e*(s)|、開始から 3 m 以降のフレーム。fr = [t, s−s0, s mod L, e_y, e*_rt, e*_own, すき間, v, 舵, 速さ]
    out = []
    for r in rows:
        for f in r['fr']:
            if f[1] is None or f[1] < 3:
                continue
            es = f[4] if which == 'rt' else (f[5] if f[5] is not None else f[4])
            if es is None:
                continue
            out.append((abs(f[3] - es), f[2], r['id']))
    return out


def in_zone(s, z):
    return z[0] <= s <= z[1]


def judge(p60, fam6):
    ok = [r for r in p60 if r['valid_episode']]
    J = {}
    J['L1_valid'] = {'value': f'{len(ok)}/{len(p60)}', 'pass': len(ok) >= 57 and len(p60) == 60}
    pool = ok + fam6
    mc = min(r['min_clearance'] for r in pool) if pool else None
    J['L2_clearance'] = {'value': mc, 'pass': mc is not None and mc >= 0.03, 'argmin': min(pool, key=lambda r: r['min_clearance'])['id'] if pool else None}
    te = [x[0] for x in tracking(ok)]
    te_own = [x[0] for x in tracking(ok, 'own')]
    te_half = []
    for r in ok:   # 学習の段と同じ間引き(2 フレームに 1 つ)でも
        a = [abs(f[3] - f[4]) for f in r['fr'] if f[1] is not None and f[1] >= 3 and f[4] is not None]
        te_half += a[::2]
    J['L3_tracking'] = {'value': {'median': q(te, .5), 'p95': q(te, .95), 'n': len(te)},
                        'own_estar': {'median': q(te_own, .5), 'p95': q(te_own, .95)},
                        'every_2nd_frame': {'median': q(te_half, .5), 'p95': q(te_half, .95)},
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
    worst = {'audit_pos_err': 0, 'sample_exact_minus_sample': -1, 'es_own_minus_rt': 0, 'proj_dey': 0}
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
            worst['es_own_minus_rt'] = max(worst['es_own_minus_rt'], r['es_check']['max_abs_own_minus_rt'])
        worst['proj_dey'] = max(worst['proj_dey'], r['proj_check']['max_dey_vs_runtime'])
        if r['valid_episode']:
            for lap in r['laps']:
                if lap['gates'] != 33 or lap['crossings'] != 1 or lap['collisions'] != 0:
                    c['bad_lap'] += 1
            if r['passed_gates'] < r['gates_by_s'] - 1:
                c['gate_count_short'] += 1
            if r['gate_stuck_max'] > 0:
                c['gate_stuck'] += 1
            nin = sum(1 for x in r['own_crossings'] if x['inside'])
            if nin != len(r['own_crossings']):
                c['crossing_outside_opening'] += 1
            if len(r['laps']) < 2:
                c['valid_but_lt2_laps'] += 1
            if not (24.5 <= min(l['distance'] for l in r['laps']) and max(l['distance'] for l in r['laps']) <= 27.5) if r['laps'] else False:
                c['lap_distance_odd'] += 1
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
        p0 = L['p0'] or {}
        if r['mode'] == 'line' and (p0.get('line') != r['params'].get('line') or p0.get('c_v') != r['params'].get('c_v')):
            c['p0_mismatch'] += 1
    return dict(c), worst


def start_phase(rows):
    # すき間の低い所が開始の直後か
    low = [r for r in rows if not r.get('invalid_spawn') and r['min_clearance'] < .03]
    return {'runs_min_lt3cm': len(low), 'of': len([r for r in rows if not r.get('invalid_spawn')]),
            'min_t_s': sorted(round(r['min_t'], 2) for r in low),
            'within_1_5s': sum(1 for r in low if r['min_t'] <= 1.5),
            'clr_after_1_5s_min_valid': min((r['clr_after_1_5s'] for r in rows if r.get('valid_episode')), default=None),
            'clr_after_3m_min_valid': min((r['clr_after_3m'] for r in rows if r.get('valid_episode') and r['clr_after_3m'] is not None), default=None),
            'low_runs': [{'id': r['id'], 'family': r['family'], 'line': r['params'].get('line'), 'start': {k: round(v, 3) if isinstance(v, float) else v for k, v in r['start'].items()},
                          'min_clearance': round(r['min_clearance'], 4), 'min_t': round(r['min_t'], 2), 'kind': r['min_kind'], 'collided': r['collisions'] > 0,
                          'start_clearance': round(r['start_clearance'], 3)} for r in sorted(low, key=lambda r: r['min_clearance'])]}


def zone_share(rows):
    te = tracking([r for r in rows if r['valid_episode']])
    big = [x for x in te if x[0] > .12]
    inS = [x for x in big if any(in_zone(x[1], z) for z in S_ZONES)]
    inC = [x for x in big if in_zone(x[1], CUSP)]
    allS = [x for x in te if any(in_zone(x[1], z) for z in S_ZONES)]
    out_s = [x[0] for x in te if not any(in_zone(x[1], z) for z in S_ZONES)]
    return {'frames': len(te), 'frac_gt12': len(big) / len(te) if te else None, 'share_of_gt12_in_S_zones': len(inS) / len(big) if big else None,
            'share_of_gt12_in_cusp_10_2_11_4': len(inC) / len(big) if big else None, 'S_zone_frame_share': len(allS) / len(te) if te else None,
            'p95_outside_S_zones': q(out_s, .95), 'median_outside_S_zones': q(out_s, .5)}


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
    # 時刻でそろえた舵 PWM の平均からの |差| の平均(600 フレームそろう回だけ)
    ss = [[f[8] for f in r['fr']] for r in rows if r['valid_episode'] and len(r['fr']) == 600]
    tot = n = 0
    for t in range(600):
        v = [s[t] for s in ss]
        m = sum(v) / len(v)
        tot += sum(abs(x - m) for x in v)
        n += len(v)
    return tot / n if n else None


def compare_trainer(rep_rows):
    ev = json.load(open(f'{REP}/train/line/eval_test_retry.json'))
    T = {r['id']: r for k in ev['tests'] for r in ev['tests'][k]}
    diffs = {'valid_mismatch': [], 'reason_mismatch': [], 'max_abs_min_clearance_diff': 0, 'laps_mismatch': [], 'max_abs_tracking_p95_diff_per_run': 0}
    for r in rep_rows:
        t = T.get(r['id'])
        if not t:
            continue
        if t['valid_episode'] != r['valid_episode']:
            diffs['valid_mismatch'].append(r['id'])
        if t['reason'] != r['reason']:
            diffs['reason_mismatch'].append(r['id'])
        diffs['max_abs_min_clearance_diff'] = max(diffs['max_abs_min_clearance_diff'], abs(t['min_clearance'] - r['min_clearance']))
        if [round(x, 6) for x in t['laps']] != [round(l['elapsed'], 6) for l in r['laps']]:
            diffs['laps_mismatch'].append(r['id'])
        if t.get('tracking') and r.get('tracking_rt'):
            diffs['max_abs_tracking_p95_diff_per_run'] = max(diffs['max_abs_tracking_p95_diff_per_run'], abs(t['tracking']['p95'] - r['tracking_rt']['p95']))
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
        per.append({'id': r['id'], 'family': r['family'], 'mean_abs_dey': statistics.mean(d), 'p90_abs_dey': q(d, .9),
                    'lap1_line': r['laps'][0]['elapsed'], 'lap1_fast': f['laps'][0]['elapsed'],
                    'v_line': r['speed']['mean_running'], 'v_fast': f['speed']['mean_running'], 'clr_line': r['min_clearance'], 'clr_fast': f['min_clearance']})
    if not per:
        return None
    vm3_fast, _ = spread([r['prof']['ey'] for r in fast_rows if r['valid_episode']])
    vm3_line_same, _ = spread([r['prof']['ey'] for r in line_rows if r['valid_episode'] and r['id'] in F])
    fam = defaultdict(list)
    for p in per:
        fam[p['family']].append(p['mean_abs_dey'])
    return {'pairs': len(per), 'mean_abs_dey_mean': statistics.mean(p['mean_abs_dey'] for p in per), 'mean_abs_dey_min': min(p['mean_abs_dey'] for p in per),
            'mean_abs_dey_by_family': {k: statistics.mean(v) for k, v in fam.items()},
            'lap1_line_mean': statistics.mean(p['lap1_line'] for p in per), 'lap1_fast_mean': statistics.mean(p['lap1_fast'] for p in per),
            'v_line_mean': statistics.mean(p['v_line'] for p in per), 'v_fast_mean': statistics.mean(p['v_fast'] for p in per),
            'clr_line_median': q([p['clr_line'] for p in per], .5), 'clr_fast_median': q([p['clr_fast'] for p in per], .5),
            'clr_line_min': min(p['clr_line'] for p in per), 'clr_fast_min': min(p['clr_fast'] for p in per),
            'vm3_fast_group': vm3_fast, 'vm3_line_same_starts': vm3_line_same,
            'fast_valid': f"{sum(1 for r in fast_rows if r['valid_episode'])}/{len(fast_rows)}"}


def main():
    out = {}
    rep = load('REPLAY')
    p60 = [r for r in rep if r['test'] == 'LN-P60']
    fam6 = [r for r in rep if r['test'] == 'LN-FAM6']
    if rep:
        out['replay'] = {'judged': judge(p60, fam6), 'exploit': exploit_checks(rep), 'start_phase': start_phase(p60 + fam6), 'zones': zone_share(p60),
                         'by_family': by_family(p60), 'vm1_us': vm1(p60), 'vs_trainer': compare_trainer(rep)}
    fr = load('FRESH')
    if fr:
        valid = [r for r in fr if r['valid_episode']]
        # 記録の前もって走らせる決まり(600 フレーム・衝突 0 でなければ替える)のあとに残るもの = 有効な回
        mc = sorted(r['min_clearance'] for r in valid)
        # 60 回ずつの塊で L1・L2・L3・L5(新しい開始で同じ大きさの試験を 4 回)
        blocks = []
        for b in range(len(fr) // 60):
            blk = fr[60 * b:60 * (b + 1)]
            j = judge(blk, [])
            blocks.append({'block': b, 'L1': j['L1_valid']['value'], 'L2': j['L2_clearance']['value'], 'L3_median': j['L3_tracking']['value']['median'],
                           'L3_p95': j['L3_tracking']['value']['p95'], 'L5_vm3': j['L5_vm3']['value'],
                           'pass_L1': j['L1_valid']['pass'], 'pass_L2': j['L2_clearance']['pass'], 'pass_L3': j['L3_tracking']['pass'], 'pass_L5': j['L5_vm3']['pass']})
        allj = judge(fr[:60], [])
        te = [x[0] for x in tracking(valid)]
        out['fresh'] = {'runs': len(fr), 'valid': len(valid), 'collided': sum(1 for r in fr if r['collisions']),
                        'reasons': dict(Counter(r['reason'] for r in fr)),
                        'blocks_of_60': blocks, 'pooled_tracking': {'median': q(te, .5), 'p95': q(te, .95), 'frac_gt12': sum(1 for x in te if x > .12) / len(te)},
                        'vm3_all_valid': spread([r['prof']['ey'] for r in valid])[0], 'vm1_us': vm1(valid),
                        'min_clr_valid': {'min': mc[0], 'p05': q(mc, .05), 'median': q(mc, .5), 'lt3cm': sum(1 for x in mc if x < .03), 'lt5cm': sum(1 for x in mc if x < .05)},
                        'start_phase': start_phase(fr), 'zones': zone_share(fr), 'by_family': by_family(fr), 'exploit': exploit_checks(fr),
                        'frames_lt3cm_in_kept_scenes': sum(sum(1 for f in r['fr'] if f[6] is not None and f[6] < .03) for r in valid),
                        'scenes_with_frames_lt3cm_kept': sum(1 for r in valid if any(f[6] is not None and f[6] < .03 for f in r['fr']))}
        fast = load('FRESH_FAST')
        if fast:
            out['vs_fastest'] = vs_fastest(fr, fast)
            out['fastest_fresh'] = {'runs': len(fast), 'valid': sum(1 for r in fast if r['valid_episode']), 'min_clr': min(r['min_clearance'] for r in fast),
                                    'exploit': exploit_checks(fast)}
    v = load('FAM6V')
    if v:
        groups = defaultdict(list)
        for r in v:
            if r['test'] == 'FAM6V':
                groups[r['id'].split('-')[0]].append(r)
        res = {}
        # V0..V35: 6 つずつ同じ開始
        allv = [r for r in v if r['test'] == 'FAM6V']
        for k in range(6):
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
        if fs:
            res['fastest_std_continuous'] = {'laps': [l['elapsed'] for l in fs[0]['laps']], 'min_clr': fs[0]['min_clearance']}
            # FAM6(標準の開始)と最速の線の違い
            F = {r['family']: r for r in rep if r['test'] == 'LN-FAM6'}
            res['fam6_vs_fastest_mean_abs_dey'] = {k: statistics.mean([abs(a - b) for a, b in zip(F[k]['prof']['ey'], fs[0]['prof']['ey']) if a is not None and b is not None]) for k in F}
        out['fam6_variants'] = res
    fb = load('FB_MIX')
    if fb:
        ok = [r for r in fb if r['valid_episode']]
        out['fallback_mix'] = {'L1': f'{len(ok)}/{len(fb)}', 'L2': min(r['min_clearance'] for r in ok) if ok else None, 'L5_vm3': spread([r['prof']['ey'] for r in ok])[0],
                               'pass': len(ok) >= 57 and ok and min(r['min_clearance'] for r in ok) >= .03 and spread([r['prof']['ey'] for r in ok])[0] >= .08}
    print(json.dumps(out, ensure_ascii=False, default=lambda o: None))


main()
