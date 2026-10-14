#!/usr/bin/env python3
# no_wall 確認 r4 のまとめ(2026-10-15、確認の段)。rows_*.jsonl(run_ck4.mjs)を数え直し、決まり(protocol_no_wall_r4.json acceptance)の形で判定する。
# 使い方: python3 analyze_ck4.py <runs のフォルダー> <summary.json>
import json, glob, sys, math, statistics, collections

runs_dir, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f'{runs_dir}/rows_*.jsonl')):
    for line in open(f):
        if line.strip():
            rows.append(json.loads(line))
by = collections.defaultdict(list)
for r in rows:
    by[r['set']].append(r)

STD_SETS = ['G17F', 'G2023', 'CORN', 'LOWC', 'LS', 'U', 'STD', 'LONG']
DR_SETS = ['HCL', 'VXALL', 'HC', 'DIN', 'DRL', 'VXC', 'VXS', 'DLS', 'DRU', 'LONG']   # VVN(VV9 のまわりを狙った組)は別に報告


def is_dr(r):
    return bool(r.get('phys')) or r.get('fc') or r.get('delay', 0) or r.get('sg', 1) != 1 or r.get('vg', 1) != 1


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def setsum(L):
    ok = [r for r in L if not r.get('error') and not r.get('invalid_spawn')]
    mins = [r['min_clearance'] for r in ok]
    imin = min(range(len(ok)), key=lambda i: ok[i]['min_clearance']) if ok else None
    cont = [r for r in ok if r.get('episode') == 'continuous']
    s = {
        'n': len(L), 'errors': sum(1 for r in L if r.get('error')), 'invalid_spawn': sum(1 for r in L if r.get('invalid_spawn')),
        'collisions': sum(1 for r in ok if r['collisions'] > 0),
        'min_clearance_cm': round(100 * min(mins), 3) if mins else None,
        'min_at': ({'id': ok[imin]['id'], 'start': ok[imin]['start'], 'frame': ok[imin]['min_frame'], 'kind': ok[imin]['min_kind']} if ok else None),
        'lt8': sum(1 for x in mins if x < .08), 'lt5': sum(1 for x in mins if x < .05), 'in8_9': sum(1 for x in mins if .08 <= x < .09),
        'stalls': sum(1 for r in ok if r.get('stall')),
        'valid_r3': sum(1 for r in cont if r.get('valid_episode_r3')), 'continuous': len(cont),
        'reverse_runs': sum(1 for r in ok if r.get('reverse_frames', 0) > 0), 'reverse_frames': sum(r.get('reverse_frames', 0) for r in ok),
        'reverse_cmd_runs': sum(1 for r in ok if r.get('reverse_cmd_frames', 0) > 0),
        'escape_runs': sum(1 for r in ok if r.get('escapes', 0) > 0), 'escapes': sum(r.get('escapes', 0) for r in ok),
        'shield_fraction_mean': round(statistics.mean(r['shield_fraction'] for r in ok), 5) if ok else None,
        'shield_fraction_max': round(max(r['shield_fraction'] for r in ok), 4) if ok else None,
        'lock_mismatch_runs': sum(1 for r in ok if r.get('lock_mismatch', 0) > 0),
        'overlap_no_collision': sum(1 for r in ok if r.get('overlap_frames', 0) > 0 and r['collisions'] == 0),
        'label_mismatch_sh0_frames': sum(r.get('label_mismatch_sh0', 0) for r in ok), 'label_sh1_unchanged_frames': sum(r.get('label_sh1_unchanged', 0) for r in ok),
        'l0_frames': sum(r.get('l0', 0) for r in ok), 'bad_keys': sum(r.get('bad_keys', 0) for r in ok),
        'ms_p50_median': round(statistics.median(r['ms_p50'] for r in ok), 2) if ok else None,
        'ms_p99_max': round(max(r['ms_p99'] for r in ok), 2) if ok else None,
        'ms_max': round(max(r['ms_max'] for r in ok), 1) if ok else None,
        'frames_over33': sum(r.get('ms_over33', 0) for r in ok), 'frames': sum(r['frames'] for r in ok),
        'first_lap_median': round(statistics.median([r['first_lap'] for r in ok if r.get('first_lap')]), 3) if any(r.get('first_lap') for r in ok) else None,
        'first_lap_max': round(max([r['first_lap'] for r in ok if r.get('first_lap')]), 3) if any(r.get('first_lap') for r in ok) else None,
    }
    return s


summary = {'rows': len(rows), 'sets': {k: setsum(v) for k, v in sorted(by.items())}}

# ---- 決まりの形の判定(この段の組で。組の作り方は試験と違うので「判定の形」)----
std = [r for k in STD_SETS for r in by.get(k, []) if not r.get('error') and not r.get('invalid_spawn') and not is_dr(r) and r.get('shield', True)]
dr = [r for k in DR_SETS for r in by.get(k, []) if not r.get('error') and not r.get('invalid_spawn') and is_dr(r) and r.get('shield', True)]
std_cont = [r for r in std if r.get('episode') == 'continuous']
J = {}
J['N1_collisions'] = {'std': sum(1 for r in std if r['collisions'] > 0), 'dr': sum(1 for r in dr if r['collisions'] > 0), 'n': len(std) + len(dr)}
J['N2_std_min_cm'] = round(100 * min(r['min_clearance'] for r in std), 3) if std else None
J['N2_std_lt8'] = sum(1 for r in std if r['min_clearance'] < .08)
J['N2_n'] = len(std)
J['N3_dr_min_cm'] = round(100 * min(r['min_clearance'] for r in dr), 3) if dr else None
J['N3_dr_lt5'] = sum(1 for r in dr if r['min_clearance'] < .05)
J['N3_n'] = len(dr)
stdS = [r for r in by.get('STD', []) if r.get('episode') == 'single']
J['N4_lap_from_stop_s'] = stdS[0].get('lap_from_stop') if stdS else None
J['N5_std_valid'] = f"{sum(1 for r in std_cont if r.get('valid_episode_r3'))}/{len(std_cont)}"
J['N5_std_fail'] = [{'id': r['id'], 'start': r['start'], 'stall': r.get('stall'), 'valid_laps': r.get('valid_laps'), 'reason': r.get('reason'), 'coll': r['collisions'],
                     'min_cm': round(100 * r['min_clearance'], 2)} for r in std_cont if not r.get('valid_episode_r3')]
for k in ['HCL', 'VXALL', 'HC', 'DIN', 'DRL', 'VXC', 'VXS', 'DLS', 'DRU', 'VVN']:
    L = [r for r in by.get(k, []) if not r.get('error') and not r.get('invalid_spawn')]
    if L:
        J[f'N5_{k}_valid_pct'] = round(100 * sum(1 for r in L if r.get('valid_episode_r3')) / len(L), 2)
J['N5_dr_fail'] = [{'id': r['id'], 'start': r['start'], 'phys': r.get('phys'), 'sg': r.get('sg'), 'vg': r.get('vg'), 'delay': r.get('delay'), 'fc': r.get('fc'),
                    'stall': r.get('stall'), 'valid_laps': r.get('valid_laps'), 'coll': r['collisions'], 'min_cm': round(100 * r['min_clearance'], 2)}
                   for r in dr if r.get('episode') == 'continuous' and not r.get('valid_episode_r3')]
U = [r for r in by.get('U', []) if not r.get('error') and not r.get('invalid_spawn')]
if U:
    n = sum(r['rho']['n'] for r in U)
    J['N6_rho_req_gt1_pct'] = round(100 * sum(r['rho']['req1'] for r in U) / n, 3)
    J['N6_rho_comb_gt1_pct'] = round(100 * sum(r['rho']['comb1'] for r in U) / n, 3)
    J['N7_shield_fraction_pct'] = round(100 * sum(r['sh'] for r in U) / sum(r['frames'] for r in U), 3)
    J['N7_run_max_pct'] = round(100 * max(r['shield_fraction'] for r in U), 2)
    # 一様の組の 8〜9 cm と分布
    J['U_min_cm'] = round(100 * min(r['min_clearance'] for r in U), 3)
    J['U_in8_9'] = sum(1 for r in U if r['min_clearance'] < .09)
    J['U_min_quantiles_cm'] = {p: round(100 * q([r['min_clearance'] for r in U], p), 2) for p in [0.001, 0.01, 0.05, 0.5]}
summary['judged_shape'] = J

# ---- シールドあり / なしの対(U の最初の 300 と UOFF)----
off = {json.dumps(r['start'], sort_keys=True): r for r in by.get('UOFF', []) if not r.get('error')}
pairs = {'n': 0, 'off_coll': 0, 'on_coll': 0, 'saved': 0, 'caused': 0, 'off_min_cm': None, 'off_lt8': 0, 'on_lt8': 0}
offmins = []
for r in U:
    k = json.dumps(r['start'], sort_keys=True)
    if k in off:
        o = off[k]
        pairs['n'] += 1
        a, b = o['collisions'] > 0, r['collisions'] > 0
        pairs['off_coll'] += a; pairs['on_coll'] += b; pairs['saved'] += (a and not b); pairs['caused'] += (b and not a)
        offmins.append(o['min_clearance']); pairs['off_lt8'] += o['min_clearance'] < .08; pairs['on_lt8'] += r['min_clearance'] < .08
if offmins:
    pairs['off_min_cm'] = round(100 * min(offmins), 3)
summary['pairs_U_UOFF'] = pairs

# ---- 止まりにくい隅・頂点の要因ごと ----
def factor_table(L, keyf):
    t = collections.defaultdict(lambda: {'n': 0, 'coll': 0, 'stall': 0, 'min': 9, 'valid': 0})
    for r in L:
        if r.get('error') or r.get('invalid_spawn'):
            continue
        k = keyf(r)
        t[k]['n'] += 1; t[k]['coll'] += r['collisions'] > 0; t[k]['stall'] += bool(r.get('stall')); t[k]['valid'] += bool(r.get('valid_episode_r3'))
        t[k]['min'] = min(t[k]['min'], r['min_clearance'])
    return {k: {**v, 'min': round(100 * v['min'], 2)} for k, v in sorted(t.items())}

hc = by.get('HC', []) + by.get('HCL', [])
summary['hard_corner_by'] = {
    'steerRate': factor_table(hc, lambda r: r['phys']['steerRate']),
    'vg': factor_table(hc, lambda r: r['vg']),
    'delay': factor_table(hc, lambda r: r['delay']),
    'r2_corner(1.8,vg1.04)': factor_table([r for r in hc if r['phys']['steerRate'] == 1.8 and r['vg'] == 1.04], lambda r: f"d{r['delay']}"),
}
vx = by.get('VXALL', []) + by.get('VXC', [])
summary['vertex_by'] = {k: factor_table(vx, (lambda kk: (lambda r: (r['phys'] or {}).get(kk, r.get(kk))))(k)) for k in
                        ['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier', 'sg', 'vg', 'delay', 'fc']}

# ---- 最速との比べ(U の最初の 150 と FAST、同じ開始)----
ROUTE = [[-.85, 2.1], [.5, 2.1], [1.8, 2.1], [3.35, 2.1], [3.85, 1.93], [4.04, 1.4], [3.85, .87], [3.35, .7], [2.51, .7], [2.16, .6062], [1.9038, .35], [1.81, 0],
         [1.9038, -.35], [2.16, -.6062], [2.51, -.7], [3.5, -.7], [3.85, -.95], [3.98, -1.45], [3.83, -1.93], [3.35, -2.25], [2.3, -2.4], [.8, -2.4], [-.5, -2.42],
         [-1.6, -2.42], [-2.65, -2.4], [-3.6, -2.3], [-4.2, -1.9], [-4.45, -1.3], [-4.45, 0], [-4.4, 1.2], [-4.15, 1.8], [-3.5, 2.1], [-2.2, 2.1]]
segs = []
acc = 0.0
for i in range(len(ROUTE)):
    a, b = ROUTE[i], ROUTE[(i + 1) % len(ROUTE)]
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    segs.append((a, b, L, acc)); acc += L
LOOP = acc


def project(x, z, hint=None):
    best = None
    idxs = range(len(segs)) if hint is None else [(hint + d) % len(segs) for d in range(-2, 4)]
    for i in idxs:
        a, b, L, s0 = segs[i]
        tx, tz = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        u = max(0, min(L, (x - a[0]) * tx + (z - a[1]) * tz))
        px, pz = a[0] + u * tx, a[1] + u * tz
        d = math.hypot(x - px, z - pz)
        if best is None or d < best[0]:
            ey = -(x - a[0]) * tz + (z - a[1]) * tx   # 進む向きの左を正(x 右・z 下の向きの約束に合わせない符号。差だけを使う)
            best = (d, s0 + u, ey, i)
    return best


def binned(path):
    out = {}
    hint = None
    s_start = None
    travelled = 0
    prev = None
    for x, z, v, *_ in path:
        if hint is None:
            d, s, ey, hint = project(x, z)
        else:
            d, s, ey, hint = project(x, z, hint)
        if prev is not None:
            travelled += math.hypot(x - prev[0], z - prev[1])
        prev = (x, z)
        if travelled < 3.0:
            continue
        k = int((s % LOOP) / .05)
        if k not in out:
            out[k] = (ey, v)
    return out


nw = {json.dumps(r['start'], sort_keys=True): r for r in U if r.get('path')}
fa = {json.dumps(r['start'], sort_keys=True): r for r in by.get('FAST', []) if r.get('path') and not r.get('error')}
diffs, dv, nwb_all, fab_all = [], [], collections.defaultdict(list), collections.defaultdict(list)
comp = {'n': 0}
lapn, lapf, minn, minf, lt8n, lt8f, colln, collf = [], [], [], [], 0, 0, 0, 0
for k, r in nw.items():
    if k not in fa:
        continue
    f = fa[k]
    comp['n'] += 1
    bn, bf = binned(r['path']), binned(f['path'])
    for kk, (ey, v) in bn.items():
        nwb_all[kk].append(ey)
        if kk in bf:
            diffs.append(abs(ey - bf[kk][0])); dv.append(v - bf[kk][1])
    for kk, (ey, v) in bf.items():
        fab_all[kk].append(ey)
    if r.get('first_lap'): lapn.append(r['first_lap'])
    if f.get('first_lap'): lapf.append(f['first_lap'])
    minn.append(r['min_clearance']); minf.append(f['min_clearance'])
    lt8n += r['min_clearance'] < .08; lt8f += f['min_clearance'] < .08; colln += r['collisions'] > 0; collf += f['collisions'] > 0


def spread(d):
    v = [statistics.pstdev(x) for x in d.values() if len(x) >= 10]
    return round(100 * statistics.mean(v), 2) if v else None


if comp['n']:
    comp.update({'line_diff_mean_cm': round(100 * statistics.mean(diffs), 2), 'line_diff_p90_cm': round(100 * q(diffs, .9), 2),
                 'speed_diff_mean_mps': round(statistics.mean(dv), 3), 'first_lap_median_nw_fast': [round(statistics.median(lapn), 3), round(statistics.median(lapf), 3)],
                 'min_clear_median_cm_nw_fast': [round(100 * statistics.median(minn), 2), round(100 * statistics.median(minf), 2)],
                 'lt8_nw_fast': [lt8n, lt8f], 'coll_nw_fast': [colln, collf],
                 'within_mode_spread_cm_nw_fast': [spread(nwb_all), spread(fab_all)],
                 'mean_speed_nw_fast': [round(statistics.mean(r['mean_speed'] for r in nw.values()), 3), round(statistics.mean(f['mean_speed'] for f in fa.values()), 3)]})
summary['fastest_comparison'] = comp

# ---- 1 フレームの時間(全部の回)----
allok = [r for r in rows if not r.get('error') and not r.get('invalid_spawn') and r.get('ms_p50') is not None]
summary['frame_time'] = {'frames': sum(r['frames'] for r in allok), 'over33': sum(r.get('ms_over33', 0) for r in allok),
                         'max_ms': round(max(r['ms_max'] for r in allok), 1) if allok else None,
                         'p99_of_runs_median': round(statistics.median(r['ms_p99'] for r in allok), 2) if allok else None,
                         'escape_runs_ms_max': round(max([r['ms_max'] for r in allok if r.get('escapes', 0) > 0] or [0]), 1)}
# 見つかった崩れ(衝突・8 cm 未満・止まり・有効でない・後ろへ動いた)の一覧
bad = []
for r in rows:
    if r.get('error'):
        bad.append({'id': r['id'], 'error': r['error'][:200]}); continue
    if r.get('invalid_spawn'):
        continue
    if r['set'] in ('UOFF', 'FAST', 'LOWMU', 'DELAY2'):
        continue
    flags = []
    if r['collisions'] > 0: flags.append('collision')
    if r['min_clearance'] < (.08 if not is_dr(r) else .05): flags.append('below_threshold')
    elif not is_dr(r) and r['min_clearance'] < .0805: flags.append('at_threshold')
    if r.get('stall'): flags.append('stall')
    if r.get('episode') == 'continuous' and not r.get('valid_episode_r3'): flags.append('not_valid_r3')
    if r.get('reverse_frames', 0) > 0: flags.append('reverse')
    if r.get('lock_mismatch', 0) > 0: flags.append('lock_mismatch')
    if r.get('label_mismatch_sh0', 0) > 0: flags.append('label_mismatch')
    if flags and flags != ['at_threshold']:
        bad.append({'id': r['id'], 'set': r['set'], 'flags': flags, 'start': r['start'], 'phys': r.get('phys'), 'sg': r.get('sg'), 'vg': r.get('vg'), 'delay': r.get('delay'),
                    'fc': r.get('fc'), 'coll': r.get('collision'), 'min_cm': round(100 * r['min_clearance'], 3), 'stall_max': r.get('stall_max'),
                    'valid_laps': r.get('valid_laps'), 'reverse_frames': r.get('reverse_frames'), 'escapes': r.get('escapes'), 'fb': r.get('fb')})
summary['findings'] = bad
json.dump(summary, open(out, 'w'), ensure_ascii=False, indent=1)
print(json.dumps({'rows': len(rows), 'judged_shape': J, 'pairs': pairs, 'fastest': comp, 'n_findings': len(bad)}, ensure_ascii=False, indent=1))
for b in bad[:40]:
    print(json.dumps(b, ensure_ascii=False))
