# 確認 line r4: 比べる値(判定のあと) — 最速との線の違い、シールドなし(方策そのもの)の弱さ
import json, glob, collections
import numpy as np
from ana4 import load, ok3, vm3, summary

O = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4'
T, F = load('test_*.jsonl'), load('fresh_*.jsonl')
FA, N = load('fast_*.jsonl'), load('nosh_*.jsonl')
out = {}
if FA:
    base = {**T, **F}
    pairs = []
    for r in FA.values():
        l = base.get(r['id'][5:])
        if l is None:
            continue
        a = np.array([np.nan if x is None else x for x in l['bins_ey']], float)
        b = np.array([np.nan if x is None else x for x in r['bins_ey']], float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        pairs.append({'id': l['id'], 'line_ok': ok3(l), 'fast_ok': ok3(r), 'dey': float(np.mean(np.abs(a[ok] - b[ok]))) if ok.sum() else None,
                      'lap1_line': l.get('lap1'), 'lap1_fast': r.get('lap1'), 'v_line': l.get('v_run'), 'v_fast': r.get('v_run'),
                      'min_line': l['min_c'], 'min_fast': r['min_c'], 'fast_collision': r['collisions'] > 0, 'fast_stall': r['stall'], 'set': r['set']})
    both = [p for p in pairs if p['line_ok'] and p['fast_ok'] and p['dey'] is not None]
    d = np.array([p['dey'] for p in both])
    for grp in ['FAST-T-LN-P60-R4', 'FAST-U300']:
        fr = [r for r in FA.values() if r['set'] == grp and ok3(r)]
        lr = [base[r['id'][5:]] for r in FA.values() if r['set'] == grp and ok3(base[r['id'][5:]])]
        out[f'vm3_{grp}'] = {'fastest': round(vm3(fr)[0], 4) if fr else None, 'line': round(vm3(lr)[0], 4) if lr else None, 'n_fast': len(fr), 'n_line': len(lr)}
    out['fastest'] = {'pairs': len(pairs), 'both_valid': len(both), 'dey_mean': round(float(d.mean()), 4), 'dey_min': round(float(d.min()), 4), 'dey_p10': round(float(np.percentile(d, 10)), 4),
                      'lap1_median_line': round(float(np.median([p['lap1_line'] for p in both])), 3), 'lap1_median_fast': round(float(np.median([p['lap1_fast'] for p in both])), 3),
                      'v_run_line': round(float(np.mean([p['v_line'] for p in both])), 3), 'v_run_fast': round(float(np.mean([p['v_fast'] for p in both])), 3),
                      'min_c_median_line': round(float(np.median([p['min_line'] for p in both])), 4), 'min_c_median_fast': round(float(np.median([p['min_fast'] for p in both])), 4),
                      'fast_collisions': sum(p['fast_collision'] for p in pairs), 'fast_stalls': sum(p['fast_stall'] for p in pairs), 'fast_not_valid': sum(not p['fast_ok'] for p in pairs)}
if N:
    sets = collections.defaultdict(list)
    for r in N.values():
        sets[r['set'][5:]].append(r)
    out['noshield_by_set'] = {k: {x: v[x] for x in ['n', 'valid_r3', 'collisions', 'stalls', 'below3_valid', 'below4_all', 'min_valid', 'min_all', 'not_valid_reasons', 'infl3_hits', 'own_below3']}
                              for k, v in ((k, summary(rs)) for k, rs in sorted(sets.items()))}
    out['noshield_all'] = {x: summary(N.values())[x] for x in ['n', 'valid_r3', 'collisions', 'stalls', 'below3_valid', 'below4_all', 'min_valid', 'min_all', 'infl3_hits', 'own_below3']}
    bad = []
    for r in N.values():
        if r['collisions'] > 0 or r['stall'] or r['min_c'] < .03:
            bad.append({'id': r['id'], 'gate': r['gate'], 'off': round(r['off'], 4), 'yaw': round(r['yaw'], 3), 'line': r['line'], 'c_v': r['c_v'], 'dpsi': r['start']['dpsi'],
                        'min_c': r['min_c'], 'min_id': r.get('min_id'), 'min_f': r.get('min_f'), 'collision': r.get('collision'), 'stall': r['stall'],
                        'with_shield_min': F[r['id'][5:]]['min_c'], 'with_shield_sh': F[r['id'][5:]]['sh_frames'], 'with_shield_valid_r3': F[r['id'][5:]]['valid_r3']})
    out['noshield_flagged'] = sorted(bad, key=lambda b: b['min_c'])
json.dump(out, open(f'{O}/compare4.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != 'noshield_flagged'}, ensure_ascii=False)[:5000])
print('noshield_flagged', len(out.get('noshield_flagged', [])), json.dumps(out.get('noshield_flagged', [])[:12], ensure_ascii=False)[:4000])
