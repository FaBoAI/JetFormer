#!/usr/bin/env python3
# no_wall の確認 r2 の集計(2026-10-14、確認の段 r2)。runs/rows_*.jsonl → summary.json
#   python3 analyze_nw_r2.py <runs> <summary.json>
import json, math, sys, glob, os
from collections import defaultdict, Counter

RUNS, OUT = sys.argv[1], sys.argv[2]
REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013'
rows_all = []
for p in sorted(glob.glob(os.path.join(RUNS, 'rows_*.jsonl'))):
    if any(p.endswith(f'_{k}.jsonl') for k in range(8)):
        continue
    for line in open(p):
        if line.strip():
            rows_all.append(json.loads(line))
by = defaultdict(list)
for r in rows_all:
    by[r['set']].append(r)


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = min(len(xs) - 1, f + 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def ok_lap(r):
    return r.get('valid_episode') and r.get('valid_laps', 0) >= 1


def basic(rows):
    rows = [r for r in rows if not r.get('error') and not r.get('invalid_spawn')]
    errs = [r for r in rows_all if r.get('error')]
    n = len(rows)
    if not n:
        return {'n': 0}
    fr = sum(r['labels']['frames'] for r in rows)
    sh = sum(r['labels']['sh'] for r in rows)
    gov = sum(r['labels'].get('governor', 0) for r in rows)
    fb = Counter()
    for r in rows:
        fb.update(r['labels'].get('fb', {}))
    mins = [r['min_clearance'] for r in rows]
    cols = [r for r in rows if r['collisions']]
    kn = [r['labels'].get('known_frame') for r in rows if r['shield']]
    rho_n = sum(r['rho']['n'] for r in rows)
    out = {
        'n': n, 'collisions': len(cols),
        'collision_runs': [{'id': r['id'], 'gate': r['start'].get('gate'), 'off': round(r['start'].get('left_offset_m', 0), 3), 'yaw': round(r['start'].get('left_yaw_deg', 0), 1),
                            't': (r.get('collision') or {}).get('time'), 'kind': (r.get('collision') or {}).get('kind'), 'sh': r['shield_frames'],
                            'fb': r['labels'].get('fb'), 'dr': r.get('dr'), 'ident': r['labels'].get('ident_last')} for r in cols][:40],
        'min_clearance_cm': round(100 * min(mins), 3), 'median_min_clearance_cm': round(100 * q(mins, .5), 3),
        'below_5cm': sum(1 for m in mins if m < .05), 'below_8cm': sum(1 for m in mins if m < .08),
        'worst': [{'id': r['id'], 'min_cm': round(100 * r['min_clearance'], 3), 'kind': r['min_kind'], 'frame': r['min_frame'], 'gate': r['start'].get('gate'),
                   'off': round(r['start'].get('left_offset_m', 0), 3), 'yaw': round(r['start'].get('left_yaw_deg', 0), 1), 'sh': r['shield_frames'], 'col': r['collisions'],
                   'dr': r.get('dr')} for r in sorted(rows, key=lambda r: r['min_clearance'])[:8]],
        'valid_episode': sum(1 for r in rows if r.get('valid_episode')), 'valid_and_lap': sum(1 for r in rows if ok_lap(r)),
        'not_valid': [{'id': r['id'], 'reason': r['reason'], 'frames': r['frames'], 'col': r['collisions'], 'laps': r.get('valid_laps'), 'sh': r['shield_frames']} for r in rows if not ok_lap(r)][:20],
        'shield_fraction': sh / fr if fr else None, 'governor_fraction': gov / fr if fr else None, 'fallbacks': dict(fb),
        'runs_with_shield': sum(1 for r in rows if r['labels']['sh'] > 0),
        'shield_lt5s_frames': sum(r['labels'].get('sh_lt5s', 0) for r in rows),
        'known_frame': {'null': sum(1 for k in kn if k is None), 'max': max([k for k in kn if k is not None], default=None), 'p99': q([k for k in kn if k is not None], .99)},
        'broken_frames': sum(r['labels'].get('broken_frames', 0) for r in rows),
        'rho_req_gt1': sum(r['rho']['req_gt1'] for r in rows) / rho_n if rho_n else None,
        'rho_comb_gt1': sum(r['rho']['comb_gt1'] for r in rows) / max(1, sum(r['rho']['comb_n'] for r in rows)),
        'rho_cmd_gt1': sum(r['rho'].get('cmd_gt1', 0) for r in rows) / rho_n if rho_n else None,
        'rho_req_max': max(r['rho']['req_max'] for r in rows), 'rho_comb_max': max(r['rho']['comb_max'] for r in rows),
        'max_scene_rho_req_gt1': max((r['rho']['req_gt1'] / r['rho']['n'] if r['rho']['n'] else 0) for r in rows),
        'max_scene_rho_comb_gt1': max((r['rho']['comb_gt1'] / r['rho']['comb_n'] if r['rho']['comb_n'] else 0) for r in rows),
        'audit_pos_err_max': max(r['audit_pos_err'] for r in rows),
        'overlap_frames_without_collision': sum(r['overlap_frames'] for r in rows if not r['collisions']),
        'own_crossings_outside': sum(1 for r in rows for c in r['own_crossings'] if not c['inside']),
        'laps_not_33_or_cross': sum(1 for r in rows for l in r['laps'] if l['collisions'] == 0 and (l['gates'] != 33 or l['crossings'] != 1)),
        'lap_distance_range': [min([l['distance'] for r in rows for l in r['laps']], default=None), max([l['distance'] for r in rows for l in r['laps']], default=None)],
        'labels_l0': sum(r['labels']['l0'] for r in rows), 'bad_keys': sum(r['labels']['bad_keys'] for r in rows),
        'mean_running_speed': sum(r['mean_running_speed'] for r in rows) / n,
        'lap1_median_s': q([r['laps'][0]['elapsed'] for r in rows if r['laps']], .5),
        'errors': len(errs),
    }
    return out


S = {'sets': {}}
for name in ['FN3', 'FN3OFF', 'FD3', 'FD3OFF', 'VX', 'VXOFF', 'NG', 'LONG', 'D2', 'FAST3']:
    if by.get(name):
        S['sets'][name] = basic(by[name])
# LONG を標準 / DR に分ける
if by.get('LONG'):
    S['sets']['LONG_nominal'] = basic([r for r in by['LONG'] if r['id'].startswith('LN')])
    S['sets']['LONG_dr'] = basic([r for r in by['LONG'] if r['id'].startswith('LD')])
# DR の中の分け(FD3・VX): 遅延・摩擦円・舵 < 0.9
for name in ['FD3', 'VX']:
    rs = by.get(name, [])
    if rs:
        S['sets'][name + '_split'] = {
            'delay1': basic([r for r in rs if r['dr']['delay'] == 1]), 'fc': basic([r for r in rs if r['dr'].get('frictionCircle')]),
            'steer_lt09': basic([r for r in rs if r['dr']['physics']['steerRate'] < .9]),
            'steer_lt09_delay1': basic([r for r in rs if r['dr']['physics']['steerRate'] < .9 and r['dr']['delay'] == 1]),
        }
# シールドあり / なしの対 = 救った・起こした
for a, b in [('FN3', 'FN3OFF'), ('FD3', 'FD3OFF'), ('VX', 'VXOFF')]:
    A = {r['id']: r for r in by.get(a, [])}
    B = {r['id']: r for r in by.get(b, [])}
    if A and B:
        k = set(A) & set(B)
        S['sets'][f'{a}_vs_{b}'] = {'pairs': len(k), 'saved': sum(1 for i in k if B[i]['collisions'] and not A[i]['collisions']),
                                     'caused': sum(1 for i in k if A[i]['collisions'] and not B[i]['collisions']),
                                     'both': sum(1 for i in k if A[i]['collisions'] and B[i]['collisions']),
                                     'shield_zero_frames_collision': sum(1 for i in k if A[i]['collisions'] and A[i]['shield_frames'] == 0),
                                     'off_below_5cm': sum(1 for i in k if B[i]['min_clearance'] < .05), 'off_min_cm': round(100 * min(B[i]['min_clearance'] for i in k), 3),
                                     'identical_paths_hint_same_min': sum(1 for i in k if abs(A[i]['min_clearance'] - B[i]['min_clearance']) < 1e-12)}

# ---- 試験の組 r2 の作り直し(REPLAY2)と判定 ----
rp = by.get('REPLAY2', [])
if rp:
    T = json.load(open(f'{REP}/eval/no_wall/no_wall_test_r2.json'))['tests']
    theirs = {(t, r['id']): r for t, rs in T.items() for r in rs}
    t = defaultdict(list)
    for r in rp:
        t[r['test']].append(r)
    judged = t['NW-N60'] + t['NW-STD'] + t['NW-G'] + t['NW-R200'] + t['NW-FC10']
    n60 = t['NW-N60']
    N = {}
    N['N1'] = {'runs': len(judged), 'collisions': sum(1 for r in judged if r['collisions']), 'pass': len(judged) == 340 and all(r['collisions'] == 0 for r in judged)}
    nom = n60 + t['NW-STD']
    N['N2'] = {'min_cm': round(100 * min(r['min_clearance'] for r in nom), 4), 'pass': min(r['min_clearance'] for r in nom) >= .08}
    rnd = t['NW-G'] + t['NW-R200'] + t['NW-FC10']
    w3 = min(rnd, key=lambda r: r['min_clearance'])
    N['N3'] = {'min_cm': round(100 * w3['min_clearance'], 4), 'where': w3['id'], 'pass': w3['min_clearance'] >= .05,
               'below_8cm': sorted([(r['id'], round(100 * r['min_clearance'], 2)) for r in rnd if r['min_clearance'] < .08], key=lambda x: x[1])}
    std1 = [r for r in t['NW-STD'] if r['episode_mode'] == 'single'][0]
    N['N4'] = {'lap_from_stop_s': std1['lap_from_stop'], 'pass': std1['lap_from_stop'] is not None and std1['lap_from_stop'] <= 12.0}
    stdc = [r for r in t['NW-STD'] if r['episode_mode'] == 'continuous'][0]
    r200 = t['NW-R200']
    N['N5'] = {'n60': f"{sum(1 for r in n60 if ok_lap(r))}/{len(n60)}", 'std': bool(ok_lap(stdc) and std1['lap_from_stop'] is not None),
               'r200': f"{sum(1 for r in r200 if ok_lap(r))}/{len(r200)}",
               'pass': all(ok_lap(r) for r in n60) and ok_lap(stdc) and std1['lap_from_stop'] is not None and sum(1 for r in r200 if ok_lap(r)) >= 190}
    rn = sum(r['rho']['n'] for r in n60)
    N['N6'] = {'rho_req_gt1': sum(r['rho']['req_gt1'] for r in n60) / rn, 'rho_comb_gt1': sum(r['rho']['comb_gt1'] for r in n60) / sum(r['rho']['comb_n'] for r in n60),
               'rho_cmd_gt1_ref': sum(r['rho'].get('cmd_gt1', 0) for r in n60) / rn}
    N['N6']['pass'] = N['N6']['rho_req_gt1'] <= .02 and N['N6']['rho_comb_gt1'] <= .02
    N['N7'] = {'fraction': sum(r['labels']['sh'] for r in n60) / sum(r['labels']['frames'] for r in n60)}
    N['N7']['pass'] = N['N7']['fraction'] <= .05
    N['all_pass'] = all(v['pass'] for v in N.values() if isinstance(v, dict) and 'pass' in v)
    off = t['NW-OFF']
    N['NW_OFF'] = {'runs': len(off), 'collisions': sum(1 for r in off if r['collisions']), 'collision_ids': [r['id'] for r in off if r['collisions']],
                   'below_5cm': sum(1 for r in off if r['min_clearance'] < .05), 'below_8cm': sum(1 for r in off if r['min_clearance'] < .08)}
    N['R200_shield_fraction'] = sum(r['labels']['sh'] for r in r200) / sum(r['labels']['frames'] for r in r200)
    # 1 回ずつの比べ(試験の JSON)
    dmax, mism = 0, []
    for r in rp:
        o = theirs.get((r['test'], r['id']))
        if o is None:
            mism.append((r['test'], r['id'], 'missing'))
            continue
        for k in ['collisions', 'reason', 'frames']:
            if o.get(k) != r.get(k):
                mism.append((r['test'], r['id'], k, o.get(k), r.get(k)))
        if o.get('shield_frames') is not None and o.get('shield_frames') != r['shield_frames']:
            mism.append((r['test'], r['id'], 'shield_frames', o.get('shield_frames'), r['shield_frames']))
        if not r['collisions'] and o.get('min_clearance') is not None:
            dmax = max(dmax, abs(o['min_clearance'] - r['min_clearance']))
        if r['valid_laps'] != o.get('valid_laps') and r['test'] != 'NW-STD':
            mism.append((r['test'], r['id'], 'valid_laps', o.get('valid_laps'), r['valid_laps']))
    N['per_run_compare'] = {'runs': len(rp), 'mismatches': mism[:30], 'n_mismatch': len(mism), 'max_clearance_diff_m_no_collision': dmax}
    N['own_checks'] = {'audit_pos_err_max': max(r['audit_pos_err'] for r in rp), 'overlap_without_collision': sum(r['overlap_frames'] for r in rp if not r['collisions']),
                       'own_crossings_outside': sum(1 for r in rp for c in r['own_crossings'] if not c['inside']),
                       'laps_not_33_or_cross': sum(1 for r in rp for l in r['laps'] if l['collisions'] == 0 and (l['gates'] != 33 or l['crossings'] != 1)),
                       'labels_l0': sum(r['labels']['l0'] for r in rp), 'bad_keys': sum(r['labels']['bad_keys'] for r in rp),
                       'n60_fallbacks': dict(sum((Counter(r['labels'].get('fb', {})) for r in n60), Counter())),
                       'n60_known_frame_max': max((r['labels'].get('known_frame') or 0) for r in n60)}
    S['REPLAY2'] = N

# ---- 最速との比べ(FN3 の最初の 150 と FAST3 は同じ開始)----
fn = {r['id'][2:]: r for r in by.get('FN3', []) if 'path' in r}
ff = {r['id'][2:]: r for r in by.get('FAST3', [])}
if fn and ff:
    L = 26.12

    def binned(rows, key):
        B = defaultdict(list)
        for r in rows:
            seen, dist, prev = set(), 0, None
            for p in r['path']:
                x, z, v, s, ey = p[:5]
                if prev is not None:
                    dist += math.hypot(x - prev[0], z - prev[1])
                prev = (x, z)
                if dist < 3:
                    continue
                b = int((s % L) / .05)
                if b in seen:
                    continue
                seen.add(b)
                B[b].append(ey if key == 'ey' else v)
        return B

    def spread(B, nmin):
        out = []
        for b, xs in B.items():
            if len(xs) >= nmin:
                m = sum(xs) / len(xs)
                out.append(sum(abs(x - m) for x in xs) / len(xs))
        return sum(out) / max(1, len(out)), len(out)

    common = sorted(set(fn) & set(ff), key=int)
    rn = [fn[k] for k in common if fn[k]['collisions'] == 0 and fn[k]['valid_episode']]
    rf = [ff[k] for k in common if ff[k]['collisions'] == 0 and ff[k]['valid_episode']]
    Bn, Bf = binned(rn, 'ey'), binned(rf, 'ey')
    Vn, Vf = binned(rn, 'v'), binned(rf, 'v')
    nmin_n, nmin_f = int(.8 * len(rn)), int(.8 * len(rf))
    diffs, vd = [], []
    for b in Bn:
        if b in Bf and len(Bn[b]) >= nmin_n and len(Bf[b]) >= nmin_f:
            diffs.append(abs(sum(Bn[b]) / len(Bn[b]) - sum(Bf[b]) / len(Bf[b])))
            vd.append(sum(Vn[b]) / len(Vn[b]) - sum(Vf[b]) / len(Vf[b]))

    def pts_after(r, d0=3):
        out, dist, prev = [], 0, None
        for p in r['path']:
            if prev is not None:
                dist += math.hypot(p[0] - prev[0], p[1] - prev[1])
            prev = p
            if dist >= d0:
                out.append((p[0], p[1]))
        return out

    def ptseg(px, pz, a, b):
        dx, dz = b[0] - a[0], b[1] - a[1]
        l2 = dx * dx + dz * dz
        t_ = 0 if l2 == 0 else max(0, min(1, ((px - a[0]) * dx + (pz - a[1]) * dz) / l2))
        return math.hypot(px - a[0] - t_ * dx, pz - a[1] - t_ * dz)

    def dist_to_path(P, Q):
        return [min(ptseg(x, z, Q[i], Q[i + 1]) for i in range(len(Q) - 1)) for (x, z) in P[::3]]

    geo = []
    for k in common[:40]:
        a, b = fn[k], ff[k]
        if a['collisions'] or b['collisions']:
            continue
        P, Q = pts_after(a), pts_after(b)
        if len(P) > 10 and len(Q) > 10:
            geo.extend(dist_to_path(P, Q))

    def within(rows):
        ref = pts_after(rows[0], 6)
        ds = []
        for r in rows[1:30]:
            ds.extend(dist_to_path(pts_after(r, 6), ref))
        return ds
    wn, wf = within(rn), within(rf)
    sn, nb = spread(Bn, nmin_n)
    sf, nbf = spread(Bf, nmin_f)
    S['COMPARE'] = {
        'starts': len(common), 'no_wall_valid': len(rn), 'fastest_valid': len(rf),
        'fastest_collisions': sum(1 for k in common if ff[k]['collisions']),
        'fastest_min_clearance_median_cm': 100 * q([ff[k]['min_clearance'] for k in common if not ff[k]['collisions']], .5),
        'no_wall_min_clearance_median_cm': 100 * q([fn[k]['min_clearance'] for k in common], .5),
        'fastest_below_8cm': sum(1 for k in common if ff[k]['min_clearance'] < .08),
        'no_wall_below_8cm': sum(1 for k in common if fn[k]['min_clearance'] < .08),
        'line_diff_mean_cm': 100 * sum(diffs) / max(1, len(diffs)), 'line_diff_p90_cm': 100 * q(diffs, .9), 'line_diff_max_cm': 100 * max(diffs), 'bins': len(diffs),
        'speed_diff_mean_mps': sum(vd) / max(1, len(vd)), 'speed_diff_p10_mps': q(vd, .1), 'speed_diff_p90_mps': q(vd, .9),
        'geo_dist_to_fastest_cm': {'mean': 100 * sum(geo) / max(1, len(geo)), 'median': 100 * q(geo, .5), 'p90': 100 * q(geo, .9), 'max': 100 * max(geo)},
        'within_mode_spread_ey_cm': {'no_wall': 100 * sn, 'fastest': 100 * sf, 'bins': [nb, nbf]},
        'within_mode_geo_cm': {'no_wall_mean': 100 * sum(wn) / max(1, len(wn)), 'no_wall_p90': 100 * q(wn, .9),
                               'fastest_mean': 100 * sum(wf) / max(1, len(wf)), 'fastest_p90': 100 * q(wf, .9)},
        'mean_running_speed': {'no_wall': sum(r['mean_running_speed'] for r in rn) / len(rn), 'fastest': sum(r['mean_running_speed'] for r in rf) / len(rf)},
        'lap1_median_s': {'no_wall': q([r['laps'][0]['elapsed'] for r in rn if r['laps']], .5), 'fastest': q([r['laps'][0]['elapsed'] for r in rf if r['laps']], .5)},
        'rho': {'no_wall': [sum(r['rho']['req_gt1'] for r in rn) / sum(r['rho']['n'] for r in rn), sum(r['rho']['comb_gt1'] for r in rn) / sum(r['rho']['comb_n'] for r in rn)],
                'fastest': [sum(r['rho']['req_gt1'] for r in rf) / sum(r['rho']['n'] for r in rf), sum(r['rho']['comb_gt1'] for r in rf) / sum(r['rho']['comb_n'] for r in rf)]},
    }

# ABX: 速い舵・弱いブレーキ・低い μ・摩擦円の隅の原因分け(隅ごと)
if by.get('ABX'):
    G = defaultdict(list)
    for r in by['ABX']:
        G[r['id'].rsplit('-g', 1)[0]].append(r)
    S['ABX'] = {k: {x: v for x, v in basic(rs).items() if x in ('n', 'collisions', 'min_clearance_cm', 'below_5cm', 'below_8cm', 'valid_and_lap', 'shield_fraction', 'governor_fraction')}
                | {'collision_gates': [r['start']['gate'] for r in rs if r['collisions']], 'collision_t': [round((r.get('collision') or {}).get('time', 0), 2) for r in rs if r['collisions']],
                   'zero_shield_collisions': sum(1 for r in rs if r['collisions'] and r['shield_frames'] == 0)} for k, rs in G.items()}
    if by.get('ABXOFF'):
        A = {r['id']: r for r in by['ABX']}
        B = {r['id']: r for r in by['ABXOFF']}
        k = set(A) & set(B)
        S['ABX']['K0_vs_off'] = {'pairs': len(k), 'off_collisions': sum(1 for i in k if B[i]['collisions']), 'saved': sum(1 for i in k if B[i]['collisions'] and not A[i]['collisions']),
                                 'caused': sum(1 for i in k if A[i]['collisions'] and not B[i]['collisions'])}

json.dump(S, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1)[:30000])
