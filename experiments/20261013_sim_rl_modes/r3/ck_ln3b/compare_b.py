# 確認 line r3(再実行): (1) 学習の段の試験の組 186 回を自前の走らせ方で走らせた結果と、eval_test_r3.json の行を照らす
# (2) 判定 L1〜L8 を自前で計算する (3) 最速との線の違い (4) シールドなしとのくらべ
import json, glob, math, collections, statistics, sys
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/ck_ln3b')
from analyze_b import load, q, vm3, summ, fails

OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b'
res = {}

# (1)(2) 試験の組
T = load('test_*.jsonl')
if T:
    ev = json.load(open('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3/eval_test_r3.json'))
    ref = {f"T-{r['id']}": r for k, rs in ev['tests'].items() for r in rs}
    diffs = {'n': 0, 'reason': 0, 'valid': 0, 'laps_max_abs': 0.0, 'min_c_max_abs': 0.0, 'shield_frames': 0, 'stall': 0}
    for r in T:
        e = ref[r['id']]
        diffs['n'] += 1
        diffs['reason'] += r['reason'] != e['reason']
        diffs['valid'] += r['valid_r3'] != e['valid_r3']
        la, lb = [l['t'] for l in r['laps']], e['laps']
        if len(la) != len(lb):
            diffs['laps_max_abs'] = float('inf')
        else:
            for x, y in zip(la, lb):
                diffs['laps_max_abs'] = max(diffs['laps_max_abs'], abs(x - y))
        if r['collisions'] == 0:
            diffs['min_c_max_abs'] = max(diffs['min_c_max_abs'], abs(r['min_c'] - e['min_clearance']))
        diffs['shield_frames'] += r['sh_frames'] != e['shield_frames']
        diffs['stall'] += r['stall'] != e['stall']
    res['test_vs_training'] = diffs
    by = collections.defaultdict(list)
    for r in T:
        by[r['set']].append(r)
    P, H, E, F = by['T-LN-P60-R3'], by['T-LN-HEAD60-R3'], by['T-LN-EXT60-R3'], by['T-LN-FAM6']
    vP = [r for r in P if r['valid_r3']]
    trk = [x / 1e4 for r in vP for x in r['trk_all']]
    # L4: LN-FAM6 の 6 つ(c+0.8・c-0.8・inside・outside など)の e_y のマス
    fam = {r['id'].replace('T-', ''): r for r in F}
    nb = len(F[0]['bins_ey'])
    common = [j for j in range(nb) if all(r['bins_ey'][j] is not None for r in F)]
    def mean_abs(a, b):
        return statistics.mean(abs(fam[a]['bins_ey'][j] - fam[b]['bins_ey'][j]) for j in common)
    keys = sorted(fam)
    def max_1m(a, b):
        d = [abs(fam[a]['bins_ey'][j] - fam[b]['bins_ey'][j]) for j in range(nb)]
        best = 0
        for j in range(nb):
            w = [d[(j + k) % nb] for k in range(20)]
            if any(math.isnan(x) for x in w) or any(fam[a]['bins_ey'][(j + k) % nb] is None or fam[b]['bins_ey'][(j + k) % nb] is None for k in range(20)):
                continue
            best = max(best, sum(w) / 20)
        return best
    lr = mean_abs('c+0.8', 'c-0.8') if 'c+0.8' in fam else None
    io = mean_abs('inside', 'outside') if 'inside' in fam else None
    pair = min(max_1m(a, b) for i, a in enumerate(keys) for b in keys[i + 1:])
    c0 = [r for r in F if (r['params']['line'] or {}).get('family') == 'const' and (r['params']['line'] or {}).get('c0') == 0]
    res['judged_own'] = {
        'L1': {k: f"{sum(r['valid_r3'] for r in v)}/{len(v)}" for k, v in (('P', P), ('HEAD', H), ('EXT', E))},
        'L2_min_clearance': min(r['min_c'] for r in [x for x in P + H + E if x['valid_r3']] + F),
        'L3': {'median': q(trk, .5), 'p95': q(trk, .95), 'frames': len(trk)},
        'L4': {'left_right': lr, 'inside_outside': io, 'min_pair_max_1m': pair, 'fam_keys': keys, 'bins': len(common)},
        'L5_vm3': vm3(vP)[0],
        'L6_lap1_c0': c0[0]['lap1'] if c0 else None,
        'L7': {'P_fraction': sum(r['sh_frames'] for r in P) / sum(r['frames'] for r in P), 'max_per_run_P_FAM6': max(r['sh_frac'] for r in P + F)},
        'L8_collisions': sum(r['collisions'] > 0 for r in T),
        'stalls': sum(r['stall'] for r in T),
        'large_heading_min': min(r['min_c'] for r in T if abs(r['start']['dpsi']) >= 1),
    }
    res['test_fails'] = fails(T)

# (3) 最速とのくらべ(同じ開始)
FS = load('fast_*.jsonl')
if FS:
    cand = {r['id']: r for r in load('cand_*.jsonl') + T}
    rows = []
    for f in FS:
        cid = f['id'][2:]
        c = cand.get(cid)
        if not c or not c['valid_r3'] or not f['valid']:
            continue
        d = [abs(a - b) for a, b in zip(c['bins_ey'], f['bins_ey']) if a is not None and b is not None]
        if d:
            rows.append({'id': cid, 'mean_dey': statistics.mean(d), 'lap1_line': c['lap1'], 'lap1_fast': f['lap1'], 'v_line': c['v_run'], 'v_fast': f['v_run'],
                         'minc_line': c['min_c'], 'minc_fast': f['min_c']})
    fv = [f for f in FS if f['valid']]
    res['vs_fastest'] = {
        'pairs': len(rows), 'mean_dey_mean': statistics.mean(r['mean_dey'] for r in rows), 'mean_dey_min': min(r['mean_dey'] for r in rows),
        'lap1_line_median': q([r['lap1_line'] for r in rows if r['lap1_line']], .5), 'lap1_fast_median': q([r['lap1_fast'] for r in rows if r['lap1_fast']], .5),
        'v_line_mean': statistics.mean(r['v_line'] for r in rows), 'v_fast_mean': statistics.mean(r['v_fast'] for r in rows),
        'minc_line_median': q([r['minc_line'] for r in rows], .5), 'minc_fast_median': q([r['minc_fast'] for r in rows], .5),
        'fast_collisions': sum(f['collisions'] > 0 for f in FS), 'fast_n': len(FS),
        'vm3_fast_U60': vm3([f for f in fv if f['set'] == 'FAST-U360'])[0], 'vm3_fast_P60': vm3([f for f in fv if f['set'] == 'FAST-T-LN-P60-R3'])[0],
        'vm3_line_U60': vm3([cand[f['id'][2:]] for f in FS if f['set'] == 'FAST-U360' and cand[f['id'][2:]]['valid_r3']])[0],
        'vm3_line_P60': vm3([cand[f['id'][2:]] for f in FS if f['set'] == 'FAST-T-LN-P60-R3' and cand[f['id'][2:]]['valid_r3']])[0],
    }

# (4) シールドなし
NS = load('nosh_*.jsonl')
if NS:
    cand = {r['id']: r for r in load('cand_*.jsonl') + T}
    rows = []
    for r in NS:
        c = cand[r['id']]
        rows.append({'id': r['id'], 'set': r['set'], 'gate': r['gate'], 'off': round(r['off'], 3), 'yaw': round(r['yaw'], 2), 'dpsi': r['start']['dpsi'],
                     'with_shield': {'reason': c['reason'], 'min_c': c['min_c'], 'stall': c['stall'], 'sh': c['sh_frames'], 'valid_r3': c['valid_r3']},
                     'no_shield': {'reason': r['reason'], 'min_c': r['min_c'], 'stall': r['stall'], 'collision': r['collision'], 'valid_r3': r['valid_r3']}})
    res['no_shield'] = {'n': len(rows), 'collisions': sum(x['no_shield']['reason'] == 'collision' for x in rows),
                        'below_3cm_valid': sum(1 for x in rows if x['no_shield']['valid_r3'] and x['no_shield']['min_c'] < .03),
                        'by_set': {k: {'n': len(v), 'collisions': sum(x['no_shield']['reason'] == 'collision' for x in v),
                                       'below_3cm_valid': sum(1 for x in v if x['no_shield']['valid_r3'] and x['no_shield']['min_c'] < .03)}
                                   for k, v in collections.Counter(x['set'] for x in rows).items() for v in [[x for x in rows if x['set'] == k]]},
                        'rows': rows}

json.dump(res, open(f'{OUT}/summary_compare.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('no_shield',)}, ensure_ascii=False, indent=1)[:6000])
if 'no_shield' in res:
    print(json.dumps({k: v for k, v in res['no_shield'].items() if k != 'rows'}, ensure_ascii=False))
