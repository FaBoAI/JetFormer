#!/usr/bin/env python3
# no_wall 確認 r3 の集計(2026-10-14、確認の段)。run_ck3.mjs の行(rows_*.jsonl)から、組ごとの数字・追補の決まりの形での判定(新しい組で)・
# シールドあり / なしの対・止まり・最速との線の違いを出す。使い方: python3 analyze_ck3.py <runs のフォルダー> <summary.json> [<jobs.meta.json>] [<route.json>]
import json, glob, math, sys, statistics as stt
from collections import Counter, defaultdict

runs, out = sys.argv[1], sys.argv[2]
meta = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
route = json.load(open(sys.argv[4]))['route'] if len(sys.argv) > 4 else None
rows = {}
for f in sorted(glob.glob(f'{runs}/rows_*.jsonl')):
    for l in open(f):
        if l.strip():
            r = json.loads(l); rows[r['id']] = r
R = list(rows.values())
errs = [r for r in R if 'error' in r]
R = [r for r in R if 'error' not in r and not r.get('invalid_spawn')]
by = defaultdict(list)
for r in R: by[r['set']].append(r)

def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))] if xs else None

def setstat(rs):
    if not rs: return None
    cl = [r['min_clearance'] for r in rs if r['collisions'] == 0]
    fr = sum(r['frames'] for r in rs)
    d = {'n': len(rs), 'collisions': sum(1 for r in rs if r['collisions'] > 0),
         'min_clearance_cm': round(100 * min(cl), 3) if cl else None,
         'min_at': min(((r['min_clearance'], r['id']) for r in rs if r['collisions'] == 0), default=(None, None))[1],
         'lt8': sum(1 for r in rs if r['collisions'] == 0 and r['min_clearance'] < .08), 'lt5': sum(1 for r in rs if r['collisions'] == 0 and r['min_clearance'] < .05),
         'lt9': sum(1 for r in rs if r['collisions'] == 0 and r['min_clearance'] < .09),
         'stalls': sum(1 for r in rs if r['stall']), 'valid_r3': sum(1 for r in rs if r['valid_episode_r3']), 'valid_ep': sum(1 for r in rs if r['valid_episode']),
         'zero_laps': sum(1 for r in rs if r['valid_laps'] == 0),
         'shield_fraction': round(sum(r['sh'] for r in rs) / fr, 5) if fr else None, 'shield_run_max': round(max(r['shield_fraction'] for r in rs), 4),
         'probe_frames_per_run': round(sum(r['probe'] for r in rs) / len(rs), 3), 'governor_fraction': round(sum(r['gov'] for r in rs) / fr, 5) if fr else None,
         'first_lap_median_s': round(stt.median([r['first_lap'] for r in rs if r['first_lap']]), 3) if any(r['first_lap'] for r in rs) else None,
         'first_lap_max_s': round(max([r['first_lap'] for r in rs if r['first_lap']]), 3) if any(r['first_lap'] for r in rs) else None,
         'mean_speed': round(stt.mean(r['mean_speed'] for r in rs), 3),
         'rho_req_gt1': round(sum(r['rho']['req1'] for r in rs) / max(1, sum(r['rho']['n'] for r in rs)), 5),
         'rho_comb_gt1': round(sum(r['rho']['comb1'] for r in rs) / max(1, sum(r['rho']['n'] for r in rs)), 5),
         'pos_err_max': max(r['pos_err'] for r in rs), 'overlap_frames': sum(r['overlap_frames'] for r in rs),
         'l0_frames': sum(r['l0'] for r in rs), 'bad_keys': sum(r['bad_keys'] for r in rs),
         'reasons': dict(Counter(r['reason'] for r in rs)),
         'fb_total': dict(sum((Counter(r['fb']) for r in rs), Counter())),
         'runs_with_non_probe_shield': sum(1 for r in rs if r['sh'] - r['probe'] > 0),
         'ms_per_frame_median': round(stt.median(r['ms_per_frame'] for r in rs), 2), 'ms_per_frame_max_run': round(max(r['ms_per_frame'] for r in rs), 2)}
    return d

S = {'rows': len(R), 'errors': len(errs), 'sets': {k: setstat(v) for k, v in sorted(by.items())}}
S['error_examples'] = [e['error'][:400] for e in errs[:3]]

# ---- 衝突・止まり・8 cm 未満の一覧 ----
def brief(r):
    return {'id': r['id'], 'start': r['start'], 'phys': r['phys'], 'delay': r['delay'], 'sg': r['sg'], 'vg': r['vg'], 'fc': r['fc'], 'shield': r['shield'],
            'reason': r['reason'], 'collision': r['collision'], 'min_cm': round(100 * r['min_clearance'], 3), 'min_frame': r['min_frame'], 'min_kind': r['min_kind'],
            'stall_max': r['stall_max'], 'valid_laps': r['valid_laps'], 'sh': r['sh'], 'fb': r['fb'], 'start_clearance_cm': round(100 * r['start_clearance'], 2)}
S['collisions_shield_on'] = [brief(r) for r in R if r['shield'] and r['mode'] == 'no_wall' and r['collisions'] > 0 and not r['set'].startswith('OUT')]
S['stalls'] = [brief(r) for r in R if r['stall'] and r['mode'] == 'no_wall' and r['shield']]
S['lt8_std'] = [brief(r) for r in R if r['shield'] and r['mode'] == 'no_wall' and r['phys'] is None and not r['fc'] and r['delay'] == 0 and r['collisions'] == 0 and r['min_clearance'] < .08]
S['zero_lap_not_stall'] = [brief(r) for r in R if r['shield'] and r['mode'] == 'no_wall' and r['valid_episode'] and not r['stall'] and r['valid_laps'] == 0 and r['T'] == 20]

# ---- シールドあり / なしの対 ----
def pairs(on_set, off_set):
    on = {r['id'].split(':', 1)[1]: r for r in by.get(on_set, [])}
    off = {r['id'].split(':', 1)[1]: r for r in by.get(off_set, [])}
    k = [x for x in on if x in off]
    return {'pairs': len(k), 'off_collisions': sum(1 for x in k if off[x]['collisions'] > 0), 'on_collisions': sum(1 for x in k if on[x]['collisions'] > 0),
            'saved': sum(1 for x in k if off[x]['collisions'] > 0 and on[x]['collisions'] == 0), 'caused': sum(1 for x in k if off[x]['collisions'] == 0 and on[x]['collisions'] > 0),
            'off_min_cm': round(100 * min((off[x]['min_clearance'] for x in k if off[x]['collisions'] == 0), default=float('nan')), 3),
            'off_stalls': sum(1 for x in k if off[x]['stall']), 'on_stalls': sum(1 for x in k if on[x]['stall']),
            'off_collision_list': [brief(off[x]) for x in k if off[x]['collisions'] > 0][:20]}
S['pairs'] = {'FS/FSOFF': pairs('FS', 'FSOFF'), 'VXALL/VXOFF': pairs('VXALL', 'VXOFF'), 'HCALL/HCOFF': pairs('HCALL', 'HCOFF')}

# ---- 隅の組ごと(HCALL の 64 通り)と頂点の要因 ----
hc = defaultdict(list)
for r in by.get('HCALL', []): hc[r['id'].split(':')[1].split('-')[0]].append(r)
S['hcall_by_combo_worst'] = sorted([{'combo': k, 'phys': v[0]['phys'], 'delay': v[0]['delay'], 'sg': v[0]['sg'], 'vg': v[0]['vg'],
    'min_cm': round(100 * min(r['min_clearance'] for r in v), 3), 'coll': sum(r['collisions'] > 0 for r in v), 'stalls': sum(r['stall'] for r in v),
    'first_lap_max': max((r['first_lap'] or 99) for r in v), 'shield_fraction': round(stt.mean(r['shield_fraction'] for r in v), 3)} for k, v in hc.items()], key=lambda d: d['min_cm'])[:8]
vx = by.get('VXALL', [])
if vx:
    fac = {}
    for key in ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier']:
        for val in sorted({r['phys'][key] for r in vx}):
            g = [r for r in vx if r['phys'][key] == val]
            fac[f'{key}={val}'] = {'n': len(g), 'min_cm': round(100 * min(r['min_clearance'] for r in g), 2), 'coll': sum(r['collisions'] > 0 for r in g), 'stalls': sum(r['stall'] for r in g),
                                   'shield_fraction': round(stt.mean(r['shield_fraction'] for r in g), 3), 'first_lap_med': round(stt.median([r['first_lap'] for r in g if r['first_lap']] or [0]), 2)}
    for key in ['delay', 'sg', 'vg', 'fc']:
        for val in sorted({r[key] for r in vx}):
            g = [r for r in vx if r[key] == val]
            fac[f'{key}={val}'] = {'n': len(g), 'min_cm': round(100 * min(r['min_clearance'] for r in g), 2), 'coll': sum(r['collisions'] > 0 for r in g), 'stalls': sum(r['stall'] for r in g),
                                   'shield_fraction': round(stt.mean(r['shield_fraction'] for r in g), 3), 'first_lap_med': round(stt.median([r['first_lap'] for r in g if r['first_lap']] or [0]), 2)}
    S['vxall_factors'] = fac
    r2c = [r for r in vx if r['phys']['steerRate'] == 1.8 and r['phys']['brakeDeceleration'] == .7 and r['phys']['frictionMultiplier'] == .85 and r['fc']]
    S['vxall_r2_corner'] = {'n': len(r2c), 'coll': sum(r['collisions'] > 0 for r in r2c), 'min_cm': round(100 * min(r['min_clearance'] for r in r2c), 2) if r2c else None,
                            'vg104': len([r for r in r2c if r['vg'] == 1.04])}

# ---- 追補の決まりの形の判定を、この段の新しい組で(参考: 組の作り方は試験と違う) ----
std_sets = ['FS', 'LOWC', 'STD']
std = [r for r in R if r['set'] in std_sets and r['shield']] + [r for r in by.get('LONG', []) if r['id'].startswith('LONG:std')]
drs = [r for r in R if r['set'] in ('DRU', 'VXALL', 'HCALL', 'HCIN')] + [r for r in by.get('LONG', []) if not r['id'].startswith('LONG:std')]
fs_ = by.get('FS', [])
J = {}
J['N1'] = {'runs': len(std) + len(drs), 'collisions': sum(r['collisions'] > 0 for r in std + drs)}
J['N2'] = {'runs': len(std), 'min_cm': round(100 * min(r['min_clearance'] for r in std if r['collisions'] == 0), 3) if std else None,
           'FS_only_min_cm': round(100 * min(r['min_clearance'] for r in fs_), 3) if fs_ else None}
J['N3'] = {'runs': len(drs), 'min_cm': round(100 * min(r['min_clearance'] for r in drs if r['collisions'] == 0), 3) if drs else None}
stdc = [r for r in std if not (r['set'] == 'STD' and r['episode'] == 'single')]
J['N5_std'] = {'valid': sum(r['valid_episode_r3'] for r in stdc), 'n': len(stdc), 'not_valid': [r['id'] for r in stdc if not r['valid_episode_r3']]}
for s in ['DRU', 'VXALL', 'HCALL', 'HCIN']:
    g = by.get(s, []); J[f'N5_{s}'] = {'valid': sum(r['valid_episode_r3'] for r in g), 'n': len(g), 'pct': round(100 * sum(r['valid_episode_r3'] for r in g) / max(1, len(g)), 2)}
if fs_:
    n = sum(r['rho']['n'] for r in fs_)
    J['N6_FS'] = {'rho_req_gt1': round(sum(r['rho']['req1'] for r in fs_) / n, 5), 'rho_comb_gt1': round(sum(r['rho']['comb1'] for r in fs_) / n, 5)}
    J['N7_FS'] = {'fraction': round(sum(r['sh'] for r in fs_) / sum(r['frames'] for r in fs_), 5), 'run_max': round(max(r['shield_fraction'] for r in fs_), 4)}
st = [r for r in by.get('STD', []) if r['episode'] == 'single']
J['N4'] = {'lap_from_stop': st[0]['lap_from_stop'] if st else None}
S['judge_on_fresh_sets'] = J
# 記録の分布での止まり・8 cm 未満の割合(FS は分布そのもの、LOWC は 12〜15 cm の層 × 重み)
w = meta.get('lowc_stratum', {}).get('weight')
lo = by.get('LOWC', [])
if fs_ and lo and w:
    fs_low = [r for r in fs_ if r['start_clearance'] <= .15]; fs_hi = [r for r in fs_ if r['start_clearance'] > .15]
    p_lo = sum(r['stall'] for r in lo) / len(lo); p_hi = sum(r['stall'] for r in fs_hi) / max(1, len(fs_hi))
    S['stall_rate_estimate'] = {'FS_raw': f"{sum(r['stall'] for r in fs_)}/{len(fs_)}", 'LOWC': f"{sum(r['stall'] for r in lo)}/{len(lo)}", 'stratum_weight': w,
        'FS_in_stratum': len(fs_low), 'weighted_rate': round(w * p_lo + (1 - w) * p_hi, 5),
        'lt8_LOWC': sum(1 for r in lo if r['collisions'] == 0 and r['min_clearance'] < .08), 'LOWC_min_cm': round(100 * min(r['min_clearance'] for r in lo), 3)}

# ---- 最速との違い(同じ 150 の開始・標準の物理)。経路の輪への自前の射影 ----
if route and by.get('FAST'):
    P = route + [route[0]]
    segs = []; acc = 0.0
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]; L = math.hypot(b[0] - a[0], b[1] - a[1]); segs.append((a, b, L, acc)); acc += L
    TOT = acc
    def proj(x, z):
        best = None
        for a, b, L, s0 in segs:
            dx, dz = b[0] - a[0], b[1] - a[1]
            t = max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[1]) * dz) / (L * L)))
            px, pz = a[0] + t * dx, a[1] + t * dz; d = math.hypot(x - px, z - pz)
            if best is None or d < best[0]:
                # 左 = 進む向き (dx, dz) を左へ 90°(yaw の約束: 左回り = −z 向き。left = (dz, -dx)/L で符号をそろえる)
                e = ((x - px) * (dz / L) + (z - pz) * (-dx / L))
                best = (d, s0 + t * L, e)
        return best[1], best[2]
    def profile(r):
        bins = {}; dist = 0.0; prev = None
        for x, z, v in r['path']:
            if prev: dist += math.hypot(x - prev[0], z - prev[1])
            prev = (x, z)
            if dist < 3.0 or v < .3: continue
            s, e = proj(x, z); k = int(s / .05)
            if k not in bins: bins[k] = (e, v)
        return bins
    nw = {r['id'].split(':')[1]: r for r in by.get('FS', []) if r.get('path')}
    fa = {r['id'].split(':')[1]: r for r in by['FAST']}
    keys = [k for k in nw if k in fa]
    diffs, dv = [], []
    allnw, allfa = defaultdict(list), defaultdict(list)
    for k in keys:
        a, b = profile(nw[k]), profile(fa[k])
        for kk in a:
            allnw[kk].append(a[kk][0])
            if kk in b: diffs.append(abs(a[kk][0] - b[kk][0])); dv.append(a[kk][1] - b[kk][1])
        for kk in b: allfa[kk].append(b[kk][0])
    def spread(dd):
        v = [stt.pstdev(x) for x in dd.values() if len(x) >= 10]
        return round(100 * stt.mean(v), 3) if v else None
    fr = [fa[k] for k in keys]; nr = [nw[k] for k in keys]
    S['vs_fastest'] = {'pairs': len(keys), 'line_diff_mean_cm': round(100 * stt.mean(diffs), 2), 'line_diff_p90_cm': round(100 * q(diffs, .9), 2),
        'speed_diff_mean': round(stt.mean(dv), 3), 'speed_diff_p10': round(q(dv, .1), 3), 'speed_diff_p90': round(q(dv, .9), 3),
        'within_mode_spread_cm_no_wall': spread(allnw), 'within_mode_spread_cm_fastest': spread(allfa),
        'no_wall_mean_speed': round(stt.mean(r['mean_speed'] for r in nr), 3), 'fastest_mean_speed': round(stt.mean(r['mean_speed'] for r in fr), 3),
        'no_wall_first_lap_med': round(stt.median([r['first_lap'] for r in nr if r['first_lap']]), 3), 'fastest_first_lap_med': round(stt.median([r['first_lap'] for r in fr if r['first_lap']]), 3),
        'no_wall_minc_median_cm': round(100 * stt.median(r['min_clearance'] for r in nr), 2), 'fastest_minc_median_cm': round(100 * stt.median(r['min_clearance'] for r in fr if r['collisions'] == 0), 2),
        'fastest_collisions': sum(r['collisions'] > 0 for r in fr), 'fastest_lt8': sum(1 for r in fr if r['collisions'] > 0 or r['min_clearance'] < .08),
        'no_wall_lt8': sum(1 for r in nr if r['min_clearance'] < .08),
        'fastest_rho_req_gt1': round(sum(r['rho']['req1'] for r in fr) / sum(r['rho']['n'] for r in fr), 4), 'no_wall_rho_req_gt1': round(sum(r['rho']['req1'] for r in nr) / sum(r['rho']['n'] for r in nr), 4),
        'route_length_m': round(TOT, 3)}

json.dump(S, open(out, 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: S[k] for k in ['rows', 'errors']}, ensure_ascii=False))
for k, v in S['sets'].items():
    print(k, json.dumps({x: v[x] for x in ['n', 'collisions', 'min_clearance_cm', 'min_at', 'lt8', 'lt9', 'stalls', 'valid_r3', 'zero_laps', 'shield_fraction', 'shield_run_max', 'probe_frames_per_run', 'first_lap_median_s', 'first_lap_max_s', 'mean_speed', 'pos_err_max', 'overlap_frames', 'l0_frames', 'bad_keys', 'ms_per_frame_median']}, ensure_ascii=False))
for k in ['judge_on_fresh_sets', 'pairs', 'stall_rate_estimate', 'vxall_r2_corner', 'vs_fastest', 'hcall_by_combo_worst']:
    if k in S: print(k, json.dumps(S[k], ensure_ascii=False)[:3000])
print('collisions_shield_on', json.dumps(S['collisions_shield_on'], ensure_ascii=False)[:3000])
print('stalls', json.dumps(S['stalls'], ensure_ascii=False)[:3000])
print('lt8_std', json.dumps(S['lt8_std'], ensure_ascii=False)[:2000])
print('zero_lap_not_stall', json.dumps(S['zero_lap_not_stall'], ensure_ascii=False)[:1500])
