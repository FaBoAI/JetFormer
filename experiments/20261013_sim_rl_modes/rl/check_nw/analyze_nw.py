#!/usr/bin/env python3
# no_wall の確認 r1 の集計(2026-10-14、確認の段)。verify_nw.mjs の JSONL を読み、判定 N1〜N7 の作り直しと、新しい組・隅の組・最速との比べを数える。
#   python3 analyze_nw.py <runs のフォルダー> <出力 summary.json>
import json, math, sys, glob, os
from collections import defaultdict

RUNS = sys.argv[1]
OUT = sys.argv[2]
TEST = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/eval/no_wall/no_wall_test_r1.json'
WORLD = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/sim/world-v3.json'
G = 9.80665


def load(name):
    p = os.path.join(RUNS, f'rows_{name}.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    i = int(math.floor(k))
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (k - i)


def ok_lap(r):
    return r.get('valid_episode') and r.get('valid_laps', 0) >= 1


def basic(rows):
    rows = [r for r in rows if not r.get('error')]
    mc = [r['min_clearance'] for r in rows]
    return {
        'runs': len(rows),
        'errors': 0,
        'collisions_runs': sum(1 for r in rows if r['collisions'] > 0),
        'collided_ids': [r['id'] for r in rows if r['collisions'] > 0][:40],
        'min_clearance': min(mc) if mc else None,
        'p5_clearance': q(mc, .05),
        'median_clearance': q(mc, .5),
        'n_below_5cm': sum(1 for c in mc if c < .05),
        'n_below_8cm': sum(1 for c in mc if c < .08),
        'below_5cm': sorted([(r['id'], round(r['min_clearance'] * 100, 2)) for r in rows if r['min_clearance'] < .05], key=lambda x: x[1])[:40],
        'valid_with_lap': sum(1 for r in rows if ok_lap(r)),
        'shield_frames': sum(r['shield_frames'] for r in rows),
        'frames': sum(r['frames'] for r in rows),
        'shield_fraction': sum(r['shield_frames'] for r in rows) / max(1, sum(r['frames'] for r in rows)),
        'runs_with_shield': sum(1 for r in rows if r['shield_frames'] > 0),
        'max_audit_pos_err': max((r['audit_pos_err'] for r in rows), default=None),
        'overlap_without_collision': sum(1 for r in rows if r['overlap_frames'] > 0 and r['collisions'] == 0),
        'labels_l0': sum(r['labels']['l0'] for r in rows),
        'labels_bad_keys': sum(r['labels']['bad_keys'] for r in rows),
        'mean_running_speed': sum(r['mean_running_speed'] for r in rows) / max(1, len(rows)),
        'lap1': [r['laps'][0]['elapsed'] for r in rows if r['laps']],
        'lap_distance_range': [min((l['distance'] for r in rows for l in r['laps']), default=None), max((l['distance'] for r in rows for l in r['laps']), default=None)],
        'lap_crossing_mismatch': sum(1 for r in rows for l in r['laps'] if l['crossings'] != 1),
        'own_crossing_outside': sum(1 for r in rows for c in r.get('own_crossings', []) if not c['inside']),
        'reasons': dict(sorted(defaultdict(int, {k: sum(1 for r in rows if r['reason'] == k) for k in set(r['reason'] for r in rows)}).items())),
    }


def rho(rows):
    n = sum(r['rho']['n'] for r in rows)
    cn = sum(r['rho']['comb_n'] for r in rows)
    return {'rho_req_gt1': sum(r['rho']['req_gt1'] for r in rows) / max(1, n), 'rho_comb_gt1': sum(r['rho']['comb_gt1'] for r in rows) / max(1, cn),
            'req_max': max(r['rho']['req_max'] for r in rows), 'comb_max': max(r['rho']['comb_max'] for r in rows), 'frames': n}


world = json.load(open(WORLD))
surfaces = world['surfaces']


def mu_at(x, z):
    mu = .8
    for s in surfaces:
        if abs(x - s['x']) <= s['w'] / 2 and abs(z - s['z']) <= s['d'] / 2:
            mu = s['mu']
    return mu


def rho_cmd(rows):
    # 参考: 指令の舵(要求)で測った ρ = v²·|tan δ_cmd|/(0.26·μ·g)(v = そのフレームの初めの速さ、v > 0.3)
    n = k = 0
    for r in rows:
        for p in r.get('path', []):
            x, z, v, s, ey, st, sp, sh = p
            if v > .3:
                n += 1
                if v * v * abs(math.tan(st)) / (.26 * mu_at(x, z) * G) > 1:
                    k += 1
    return {'frames': n, 'gt1': k / max(1, n)}


S = {'created': 'check_no_wall_r1'}

# ---- REPLAY(試験の組の作り直し)----
rep = load('REPLAY')
if rep:
    T = defaultdict(list)
    for r in rep:
        T[r['test']].append(r)
    judged = T['NW-N60'] + T['NW-STD'] + T['NW-G'] + T['NW-R200'] + T['NW-FC10']
    nom = T['NW-N60'] + T['NW-STD']
    rnd = T['NW-G'] + T['NW-R200'] + T['NW-FC10']
    s1 = [r for r in T['NW-STD'] if r['episode_mode'] == 'single'][0]
    sc = [r for r in T['NW-STD'] if r['episode_mode'] == 'continuous'][0]
    n60 = T['NW-N60']
    J = {
        'N1': {'runs': len(judged), 'collided': sum(1 for r in judged if r['collisions'] > 0), 'pass': len(judged) == 340 and all(r['collisions'] == 0 for r in judged)},
        'N2': {'value': min(r['min_clearance'] for r in nom), 'pass': min(r['min_clearance'] for r in nom) >= .08},
        'N3': {'value': min(r['min_clearance'] for r in rnd), 'pass': min(r['min_clearance'] for r in rnd) >= .05,
               'below_5cm': sorted([(r['test'], r['id'], round(r['min_clearance'] * 100, 3)) for r in rnd if r['min_clearance'] < .05], key=lambda x: x[2]),
               'below_8cm': sorted([(r['test'], r['id'], round(r['min_clearance'] * 100, 3)) for r in rnd if r['min_clearance'] < .08], key=lambda x: x[2])},
        'N4': {'value': s1['lap_from_stop'], 'pass': s1['lap_from_stop'] is not None and s1['lap_from_stop'] <= 12.0},
        'N5': {'n60': sum(1 for r in n60 if ok_lap(r)), 'std': bool(ok_lap(sc)), 'r200': sum(1 for r in T['NW-R200'] if ok_lap(r)),
               'pass': all(ok_lap(r) for r in n60) and bool(ok_lap(sc)) and sum(1 for r in T['NW-R200'] if ok_lap(r)) >= 190},
        'N6': {**rho(n60)},
        'N7': {'value': sum(r['shield_frames'] for r in n60) / sum(r['frames'] for r in n60)},
    }
    J['N6']['pass'] = J['N6']['rho_req_gt1'] <= .02 and J['N6']['rho_comb_gt1'] <= .02
    J['N7']['pass'] = J['N7']['value'] <= .05
    S['REPLAY_judged'] = J
    S['REPLAY_sets'] = {k: basic(v) for k, v in T.items()}
    for k in S['REPLAY_sets']:
        S['REPLAY_sets'][k].pop('lap1', None)
    # 実装の段の数字と 1 回ずつ比べる
    theirs = json.load(open(TEST))['tests']
    diffs = []
    for k, v in T.items():
        their = {r['id']: r for r in theirs.get(k, [])}
        for r in v:
            t = their.get(r['id'])
            if not t:
                diffs.append({'test': k, 'id': r['id'], 'missing': True})
                continue
            d = {'test': k, 'id': r['id'], 'dmin': r['min_clearance'] - (t.get('min_clearance') or 0), 'col': r['collisions'] != t['collisions'],
                 'reason': r['reason'] != t['reason'], 'shield': r['shield_frames'] != t.get('shield_frames'), 'frames': r['frames'] != t['frames']}
            diffs.append(d)
    S['REPLAY_vs_reported'] = {
        'compared': len(diffs), 'missing': sum(1 for d in diffs if d.get('missing')),
        'max_abs_dmin_m': max(abs(d['dmin']) for d in diffs if 'dmin' in d),
        'own_lower_than_theirs_max_m': -min(d['dmin'] for d in diffs if 'dmin' in d),
        'collision_mismatch': sum(1 for d in diffs if d.get('col')), 'reason_mismatch': sum(1 for d in diffs if d.get('reason')),
        'shield_frames_mismatch': sum(1 for d in diffs if d.get('shield')), 'frames_mismatch': sum(1 for d in diffs if d.get('frames')),
    }

# ---- 新しい組 ----
for name in ['FNOM', 'FDR', 'FDROFF', 'LOWMU', 'FAST']:
    rows = load(name) if name not in ('FNOM', 'FAST', 'LOWMU') else [r for r in load('FNOM_FAST_LOWMU') if r['set'] == name]
    if name in ('FDR', 'FDROFF') and not rows:
        rows = [r for r in load('FDR_CORNER') if r['set'] == name]
    if not rows:
        continue
    b = basic(rows)
    b['rho_nominal_def'] = rho([r for r in rows if not r.get('error')])
    if name in ('FNOM', 'FAST'):
        b['rho_cmd_ref'] = rho_cmd(rows)
    lap1 = b.pop('lap1')
    b['lap1_s'] = {'n': len(lap1), 'min': min(lap1) if lap1 else None, 'median': q(lap1, .5), 'max': max(lap1) if lap1 else None}
    if name in ('FDR', 'FDROFF'):
        fc = [r for r in rows if r['dr'] and r['dr'].get('frictionCircle')]
        d1 = [r for r in rows if r['dr'] and r['dr'].get('delay') == 1]
        b['by'] = {'friction_circle': {'runs': len(fc), 'collisions': sum(1 for r in fc if r['collisions']), 'min': min((r['min_clearance'] for r in fc), default=None)},
                   'delay1': {'runs': len(d1), 'collisions': sum(1 for r in d1 if r['collisions']), 'min': min((r['min_clearance'] for r in d1), default=None)},
                   'steerRate_lt_0.9': {'runs': sum(1 for r in rows if r['dr']['physics']['steerRate'] < .9),
                                        'collisions': sum(1 for r in rows if r['dr']['physics']['steerRate'] < .9 and r['collisions']),
                                        'below5': sum(1 for r in rows if r['dr']['physics']['steerRate'] < .9 and r['min_clearance'] < .05)}}
        b['worst'] = [{'id': r['id'], 'min_cm': round(r['min_clearance'] * 100, 2), 'col': r['collisions'], 'kind': r['min_kind'], 'frame': r['min_frame'],
                       'start': r['start'], 'dr': r['dr'], 'shield_frames': r['shield_frames'], 'reason': r['reason']}
                      for r in sorted(rows, key=lambda r: r['min_clearance'])[:8]]
    S[name] = b

cor = [r for r in load('FDR_CORNER') if r['set'] == 'CORNER']
if cor:
    by = defaultdict(list)
    for r in cor:
        by[r['corner']].append(r)
    S['CORNER'] = {k: basic(v) for k, v in sorted(by.items())}
    for k in S['CORNER']:
        S['CORNER'][k].pop('lap1', None)
    S['CORNER_all'] = basic(cor)
    S['CORNER_all'].pop('lap1', None)
    S['CORNER_worst'] = [{'id': r['id'], 'min_cm': round(r['min_clearance'] * 100, 2), 'col': r['collisions'], 'kind': r['min_kind'], 'frame': r['min_frame'],
                          'shield_frames': r['shield_frames'], 'reason': r['reason'], 'collision': r.get('collision')} for r in sorted(cor, key=lambda r: r['min_clearance'])[:15]]

# ---- 最速との比べ(FNOM と FAST は同じ開始)----
fn = {r['id'][2:]: r for r in load('FNOM_FAST_LOWMU') if r['set'] == 'FNOM'}
ff = {r['id'][2:]: r for r in load('FNOM_FAST_LOWMU') if r['set'] == 'FAST'}
if fn and ff:
    L = 26.12  # 参照ループの長さ(LOOP_LENGTH の近似。マスの数だけに使う)

    def binned(rows, key):
        # 開始から 3 m 以降の最初の通過(s は runtime の追跡の s、5 cm のマス)
        B = defaultdict(list)
        for r in rows:
            seen = set()
            dist = 0
            prev = None
            for p in r['path']:
                x, z, v, s, ey, st, sp, sh = p
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
    diffs = []
    vd = []
    for b in Bn:
        if b in Bf and len(Bn[b]) >= nmin_n and len(Bf[b]) >= nmin_f:
            diffs.append(abs(sum(Bn[b]) / len(Bn[b]) - sum(Bf[b]) / len(Bf[b])))
            vd.append(sum(Vn[b]) / len(Vn[b]) - sum(Vf[b]) / len(Vf[b]))

    # 幾何の比べ(基準の線を使わない): 同じ開始の no_wall の点から最速の軌跡までの距離(開始から 3 m 以降)
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
        t = 0 if l2 == 0 else max(0, min(1, ((px - a[0]) * dx + (pz - a[1]) * dz) / l2))
        return math.hypot(px - a[0] - t * dx, pz - a[1] - t * dz)

    def dist_to_path(P, Q):
        ds = []
        for (x, z) in P[::3]:
            m = min(ptseg(x, z, Q[i], Q[i + 1]) for i in range(len(Q) - 1))
            ds.append(m)
        return ds

    geo = []
    for k in common[:40]:
        a, b = fn[k], ff[k]
        if a['collisions'] or b['collisions']:
            continue
        P, Q = pts_after(a), pts_after(b)
        if len(P) > 10 and len(Q) > 10:
            geo.extend(dist_to_path(P, Q))
    # 同じモードの中: 1 本目の no_wall の軌跡までの距離(開始から 6 m 以降)
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
        'fastest_min_clearance_median_cm': q([ff[k]['min_clearance'] for k in common if not ff[k]['collisions']], .5) * 100,
        'no_wall_min_clearance_median_cm': q([fn[k]['min_clearance'] for k in common], .5) * 100,
        'fastest_below_8cm': sum(1 for k in common if ff[k]['min_clearance'] < .08),
        'line_diff_mean_cm': 100 * sum(diffs) / max(1, len(diffs)), 'line_diff_p90_cm': 100 * q(diffs, .9), 'line_diff_max_cm': 100 * max(diffs), 'bins': len(diffs),
        'speed_diff_mean_mps': sum(vd) / max(1, len(vd)), 'speed_diff_p10_mps': q(vd, .1), 'speed_diff_p90_mps': q(vd, .9),
        'geo_dist_to_fastest_cm': {'mean': 100 * sum(geo) / max(1, len(geo)), 'median': 100 * q(geo, .5), 'p90': 100 * q(geo, .9), 'max': 100 * max(geo)},
        'within_mode_spread_ey_cm': {'no_wall': 100 * sn, 'fastest': 100 * sf, 'bins': [nb, nbf]},
        'within_mode_geo_cm': {'no_wall_mean': 100 * sum(wn) / max(1, len(wn)), 'no_wall_p90': 100 * q(wn, .9),
                               'fastest_mean': 100 * sum(wf) / max(1, len(wf)), 'fastest_p90': 100 * q(wf, .9)},
        'mean_running_speed': {'no_wall': sum(r['mean_running_speed'] for r in rn) / len(rn), 'fastest': sum(r['mean_running_speed'] for r in rf) / len(rf)},
        'lap1_median_s': {'no_wall': q([r['laps'][0]['elapsed'] for r in rn if r['laps']], .5), 'fastest': q([r['laps'][0]['elapsed'] for r in rf if r['laps']], .5)},
    }

json.dump(S, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1)[:20000])
