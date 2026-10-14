# 確認 line r4: 集計と、自前の判定(protocol.json acceptance.line・variation_metrics と protocol_line_r3/r4/r4b の文言から書いた。
# 学習・評価の判定のコード(eval_line_r4b.mjs judge・variation_metrics.mjs)も、前の確認の analyze_b.py も使わない)
import json, glob, math, sys, collections
import numpy as np

O = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4'


def load(pattern):
    out = {}
    for f in sorted(glob.glob(f'{O}/runs/{pattern}')):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r['id']] = r
    return out


def fam(r):
    L = r.get('line') or {}
    if L.get('family') == 'const':
        c = L['c0']
        return 'const_left' if c > .5 else 'const_right' if c < -.5 else 'const_center'
    return L.get('family', '?')


def ok3(r):
    return bool(r.get('valid_r3'))


def esc(r):
    return (r.get('esc_frames') or 0) > 0


def summary(runs):
    rs = list(runs)
    if not rs:
        return None
    v = [r for r in rs if ok3(r)]
    mins_v = [r['min_c'] for r in v]
    mins = [r['min_c'] for r in rs if not r.get('invalid_spawn')]
    trk = np.concatenate([np.array(r['trk_mm10'], float) / 1e4 for r in v if r.get('trk_mm10')]) if v else np.array([])
    reasons = collections.Counter(r.get('reason') for r in rs if not ok3(r))
    infl3 = [r for r in rs if (r.get('infl') or {}).get('0.03')]
    own3 = [r for r in rs if r.get('min_c_own', 1) < .03]
    shf = sum(r.get('sh_frames', 0) for r in rs) / max(1, sum(r.get('frames', 0) for r in rs))
    return {
        'n': len(rs), 'valid': sum(1 for r in rs if r.get('valid')), 'valid_r3': len(v),
        'collisions': sum(1 for r in rs if r.get('collisions', 0) > 0), 'stalls': sum(1 for r in rs if r.get('stall')),
        'not_valid_reasons': dict(reasons),
        'below3_valid': sum(1 for x in mins_v if x < .03), 'below4_all': sum(1 for x in mins if x < .04),
        'min_valid': round(min(mins_v), 5) if mins_v else None, 'min_all': round(min(mins), 5) if mins else None,
        'p5_min_valid': round(float(np.percentile(mins_v, 5)), 5) if mins_v else None,
        'shield_runs': sum(1 for r in rs if r.get('sh_frames', 0) > 0), 'shield_fraction': round(shf, 5),
        'shield_gt10': sum(1 for r in rs if r.get('sh_frac', 0) > .10), 'shield_gt5': sum(1 for r in rs if r.get('sh_frac', 0) > .05),
        'escape_runs': sum(1 for r in rs if esc(r)), 'brake_first_frames': sum(r.get('brake_first_frames', 0) for r in rs),
        'brake_first_runs': sum(1 for r in rs if r.get('brake_first_frames', 0) > 0),
        'reverse_runs': sum(1 for r in rs if r.get('reverse_frames', 0) > 0),
        'fallbacks': dict(sum((collections.Counter(r.get('fb') or {}) for r in rs), collections.Counter())),
        'infl3_hits': len(infl3), 'own_below3': len(own3),
        'infl3_vs_own_disagree': sorted({r['id'] for r in infl3} ^ {r['id'] for r in own3}),
        'trk_median': round(float(np.median(trk)), 5) if trk.size else None, 'trk_p95': round(float(np.percentile(trk, 95)), 5) if trk.size else None,
        'large_heading': sum(1 for r in rs if abs(r['start']['dpsi']) >= 1), 'large_heading_min': (lambda a: round(min(a), 5) if a else None)([r['min_c'] for r in rs if abs(r['start']['dpsi']) >= 1 and not r.get('invalid_spawn')]),
        'integ_err_max': max((r.get('integ_err', 0) for r in rs), default=0), 'ey_vs_runtime_max': max((r.get('ey_vs_runtime_max') or 0 for r in rs), default=0),
        'v_run_mean': round(float(np.mean([r['v_run'] for r in v])), 4) if v else None,
        'lap1_median': round(float(np.median([r['lap1'] for r in v if r.get('lap1')])), 3) if v else None,
    }


def labels(runs):
    keys = ['l_not1', 'ev_not0', 'sh_missing', 'sh_ph_mismatch', 'es_missing', 'cl_missing', 'p0_bad', 'p_later', 'unknown_key', 'bad_text', 'over_cap', 'steer_out', 'nonfinite', 'neg_cmd']
    out = {k: sum(r['labels'][k] for r in runs if 'labels' in r) for k in keys}
    out['frames'] = sum(r['labels']['n'] for r in runs if 'labels' in r)
    out['sh1_frames'] = sum(r['labels']['sh1'] for r in runs if 'labels' in r)
    out['max_bytes'] = max((r['labels']['max_bytes'] for r in runs if 'labels' in r), default=0)
    out['meta_shield_present'] = sum(1 for r in runs if r.get('meta_shield'))
    out['meta_shield_v5b'] = sum(1 for r in runs if (r.get('meta_shield') or {}).get('select') == 'v5b')
    out['runs'] = len(runs)
    return out


def binsmat(rs):
    return np.array([[np.nan if x is None else x for x in r['bins_ey']] for r in rs], float)


def vm3(rs):
    M = binsmat(rs)
    ok = ~np.isnan(M).any(axis=0)
    if ok.sum() == 0:
        return None, 0
    X = M[:, ok]
    return float(np.mean(np.mean(np.abs(X - X.mean(axis=0)), axis=0))), int(ok.sum())


def l4(fam6):
    by = {}
    for r in fam6:
        L = r['line']
        k = f"c{L['c0']:+.1f}" if L['family'] == 'const' else L['family']
        by[k] = np.array([np.nan if x is None else x for x in r['bins_ey']], float)
    keys = list(by)
    M = np.array([by[k] for k in keys])
    ok = ~np.isnan(M).any(axis=0)
    lr = float(np.mean(np.abs(by['c+0.8'][ok] - by['c-0.8'][ok])))
    io = float(np.mean(np.abs(by['inside'][ok] - by['outside'][ok])))
    nb = M.shape[1]
    pair = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = np.abs(by[keys[i]] - by[keys[j]])
            best = 0.0
            for s in range(nb):
                idx = [(s + k) % nb for k in range(20)]
                if not ok[idx].all():
                    continue
                best = max(best, float(np.mean(d[idx])))
            pair[f'{keys[i]}|{keys[j]}'] = round(best, 5)
    return {'left_right_mean': round(lr, 5), 'inside_outside_mean': round(io, 5), 'min_pair_max_1m_avg': min(pair.values()), 'bins': int(ok.sum()), 'pairs': pair}


def judge_test(T):
    sets = collections.defaultdict(list)
    for r in T.values():
        sets[r['set'][2:]].append(r)
    need = {'LN-P60-R4': 57, 'LN-HEAD60-R4': 57, 'LN-EXT60-R4': 57, 'LN-CORNER-R4': 111, 'LN-BOX120-R4': 114, 'LN-G30R120-R4B': 114}
    J = {}
    J['L1'] = {k: f"{sum(ok3(r) for r in sets[k])}/{len(sets[k])}" for k in need}
    J['L1_pass'] = all(sum(ok3(r) for r in sets[k]) >= need[k] for k in need)
    l2 = [r['min_c'] for k, rs in sets.items() for r in rs if ok3(r) or k == 'LN-FAM6']
    J['L2'] = round(min(l2), 5); J['L2_pass'] = min(l2) >= .03
    P = [r for r in sets['LN-P60-R4'] if ok3(r)]
    trk = np.concatenate([np.array(r['trk_mm10'], float) / 1e4 for r in P])
    J['L3'] = {'median': round(float(np.median(trk)), 5), 'p95': round(float(np.percentile(trk, 95)), 5), 'frames': int(trk.size)}
    J['L3_pass'] = J['L3']['median'] <= .05 and J['L3']['p95'] <= .12
    J['L4'] = l4(sets['LN-FAM6'])
    J['L4_pass'] = len(sets['LN-FAM6']) == 6 and all(ok3(r) for r in sets['LN-FAM6']) and J['L4']['left_right_mean'] >= .2 and J['L4']['inside_outside_mean'] >= .1 and J['L4']['min_pair_max_1m_avg'] >= .08
    v, nb = vm3(P)
    J['L5'] = {'vm3': round(v, 5), 'bins': nb}; J['L5_pass'] = v >= .08
    c0 = [r for r in sets['LN-FAM6'] if r['line']['family'] == 'const' and r['line']['c0'] == 0][0]
    J['L6'] = round(c0['lap1'], 4); J['L6_pass'] = c0['lap1'] <= 10.0
    pf = sum(r['sh_frames'] for r in sets['LN-P60-R4']) / sum(r['frames'] for r in sets['LN-P60-R4'])
    mx = max(r['sh_frac'] for r in sets['LN-P60-R4'] + sets['LN-FAM6'])
    J['L7'] = {'P_fraction': round(pf, 5), 'max_run_P_FAM6': round(mx, 5)}; J['L7_pass'] = pf <= .02 and mx <= .10
    allr = list(T.values())
    J['L8'] = sum(1 for r in allr if r['collisions'] > 0); J['L8_pass'] = J['L8'] == 0
    J['L9'] = sum(1 for r in allr if r['stall']); J['L9_pass'] = J['L9'] == 0
    J['L10'] = sum(r['reverse_frames'] for r in sets['LN-P60-R4'] + sets['LN-FAM6']); J['L10_pass'] = J['L10'] == 0
    J['L11'] = sum(1 for r in allr if esc(r)); J['L11_pass'] = J['L11'] <= math.floor(.01 * len(allr))
    J['all_pass'] = all(v for k, v in J.items() if k.endswith('_pass'))
    return J, sets


def compare_ref(T):
    d = {'n': 0, 'valid_r3_diff': [], 'stall_diff': [], 'sh_diff': [], 'min_absdiff_max': 0.0, 'lap_absdiff_max': 0.0, 'reason_diff': []}
    jobs = {j['id']: j for j in json.load(open(f'{O}/testjobs4.json'))['jobs']}
    for i, r in T.items():
        ref = jobs[i]['ref']
        d['n'] += 1
        if bool(ref['valid_r3']) != bool(r['valid_r3']):
            d['valid_r3_diff'].append(i)
        if bool(ref['stall']) != bool(r['stall']):
            d['stall_diff'].append(i)
        if ref['sh'] != r['sh_frames']:
            d['sh_diff'].append(i)
        if ref['reason'] != r['reason']:
            d['reason_diff'].append(i)
        if r['collisions'] == 0:
            d['min_absdiff_max'] = max(d['min_absdiff_max'], abs(ref['min'] - r['min_c']))
        for a, b in zip(ref['laps'], [l['t'] for l in r['laps']]):
            d['lap_absdiff_max'] = max(d['lap_absdiff_max'], abs(a - b))
    return d


if __name__ == '__main__':
    T = load('test_*.jsonl')
    F = load('fresh_*.jsonl')
    out = {}
    if T:
        out['test_n'] = len(T)
        if len(T) == 542:
            J, sets = judge_test(T)
            out['test_judged'] = J
            out['test_by_set'] = {k: summary(v) for k, v in sets.items()}
        out['test_vs_training_stage'] = compare_ref(T)
        out['test_labels'] = labels(list(T.values()))
    if F:
        sets = collections.defaultdict(list)
        for r in F.values():
            sets[r['set']].append(r)
        out['fresh_n'] = len(F)
        out['fresh_by_set'] = {k: summary(v) for k, v in sorted(sets.items())}
        out['fresh_all'] = summary(F.values())
        out['fresh_labels'] = labels(list(F.values()))
        # 一様な組を 60 回ずつ 5 つ(L1・L2・L3・L5 の水準)
        U = sorted(sets.get('U300', []), key=lambda r: int(r['id'].split('-')[1]))
        blocks = []
        for b in range(0, len(U), 60):
            B = U[b:b + 60]
            if len(B) < 60:
                break
            V = [r for r in B if ok3(r)]
            trk = np.concatenate([np.array(r['trk_mm10'], float) / 1e4 for r in V])
            vv, nb = vm3(V)
            blocks.append({'L1': f'{len(V)}/60', 'L2': round(min(r['min_c'] for r in V), 5), 'L3': [round(float(np.median(trk)), 5), round(float(np.percentile(trk, 95)), 5)],
                           'L5': round(vv, 5), 'sh_frames': sum(r['sh_frames'] for r in B), 'collisions': sum(r['collisions'] > 0 for r in B), 'stalls': sum(r['stall'] for r in B)})
        out['U300_blocks'] = blocks
        # 形ごと(U300・H200・HX120・NEAR・RND の有効な回)
        byf = collections.defaultdict(list)
        for k in ['U300', 'H200', 'NEAR', 'RND']:
            for r in sets.get(k, []):
                if ok3(r):
                    byf[fam(r)].append(r)
        out['by_family'] = {}
        for k, rs in sorted(byf.items()):
            trk = np.concatenate([np.array(r['trk_mm10'], float) / 1e4 for r in rs])
            out['by_family'][k] = {'n': len(rs), 'trk_med': round(float(np.median(trk)), 5), 'trk_p95': round(float(np.percentile(trk, 95)), 5),
                                   'lap1_med': round(float(np.median([r['lap1'] for r in rs])), 3), 'min': round(min(r['min_c'] for r in rs), 5)}
        # 崩れた回の一覧
        bad = []
        for r in F.values():
            why = []
            if r.get('collisions', 0) > 0: why.append('collision')
            if r.get('stall'): why.append('stall')
            if ok3(r) and r['min_c'] < .03: why.append('below3_valid')
            if not ok3(r) and r.get('min_c', 1) < .03 and not r.get('collisions'): why.append('below3_invalid')
            if esc(r): why.append('escape')
            if r.get('brake_first_frames', 0) > 0: why.append('brake_first')
            if r.get('sh_frac', 0) > .10: why.append('shield_gt10')
            if r.get('reverse_frames', 0) > 0: why.append('reverse')
            if not ok3(r) and not why: why.append('invalid:' + str(r.get('reason')) + (':no_valid_lap' if r.get('valid') else ''))
            if why:
                bad.append({'id': r['id'], 'why': why, 'gate': r['gate'], 'off': round(r['off'], 4), 'yaw': round(r['yaw'], 3), 'line': r['line'], 'c_v': r['c_v'],
                            'dpsi': r['start']['dpsi'], 'min_c': r['min_c'], 'min_id': r.get('min_id'), 'min_f': r.get('min_f'), 'sh': r.get('sh_frames'), 'fb': r.get('fb'),
                            'collision': r.get('collision'), 'laps': [l['t'] for l in r.get('laps', [])], 'stall_first': r.get('stall_first')})
        out['fresh_flagged'] = sorted(bad, key=lambda b: b['id'])
    A, AN = load('adv_*.jsonl'), load('advnosh_*.jsonl')
    if A:
        sa = collections.defaultdict(list)
        for r in A.values():
            sa[r['set']].append(r)
        out['adv_n'] = len(A)
        out['adv_by_set'] = {k: summary(v) for k, v in sorted(sa.items())}
        out['adv_all'] = summary(A.values())
        out['adv_labels'] = labels(list(A.values()))
        out['adv_flagged'] = [{'id': r['id'], 'reason': r['reason'], 'min_c': r['min_c'], 'stall': r['stall'], 'esc': r.get('esc_frames'), 'bf': r.get('brake_first_frames'), 'sh': r['sh_frames'], 'fb': r['fb']}
                              for r in A.values() if not ok3(r) or r['min_c'] < .03 or esc(r) or r.get('brake_first_frames', 0) > 0 or r['sh_frac'] > .1 or r['reverse_frames'] > 0]
    if AN:
        sn = collections.defaultdict(list)
        for r in AN.values():
            sn[r['set']].append(r)
        out['advnosh_by_set'] = {k: {x: v[x] for x in ['n', 'valid_r3', 'collisions', 'stalls', 'below3_valid', 'below4_all', 'min_valid', 'min_all']} for k, v in ((k, summary(rs)) for k, rs in sorted(sn.items()))}
        out['advnosh_all'] = {x: summary(AN.values())[x] for x in ['n', 'valid_r3', 'collisions', 'stalls', 'below3_valid', 'below4_all', 'min_valid', 'min_all']}
    json.dump(out, open(f'{O}/summary4.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in out if k in ('test_n', 'fresh_n', 'test_judged', 'test_vs_training_stage', 'U300_blocks')}, ensure_ascii=False)[:4000])
    print('flagged', json.dumps(out.get('fresh_flagged', []), ensure_ascii=False)[:3000])
