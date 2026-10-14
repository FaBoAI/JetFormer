#!/usr/bin/env python3
# 確認 line r3 の集計と判定(2026-10-14、確認の段)。学習・評価のコード(eval_line_r3.mjs judge・variation_metrics.mjs・summ_ln.py)を使わず、
# protocol.json acceptance.line・variation_metrics と protocol_line_r3.json acceptance の文言から書いた。
#   入力: check_line_r3/runs/{cand,nosh,web,fast}_*.jsonl、sets.json。比べる相手: 学習の段の eval_test_r3.json・compare/*.json
#   出力: check_line_r3/summary.json
import json, glob, math, statistics, collections, sys
OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3'
TR = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3'
L_LOOP = None

def load(variant):
    rows = []
    for f in sorted(glob.glob(f'{OUT}/runs/{variant}_*.jsonl')):
        for l in open(f):
            if l.strip():
                rows.append(json.loads(l))
    return rows

def quant(a, p):
    if not a: return None
    b = sorted(a); i = p * (len(b) - 1); lo = math.floor(i); hi = math.ceil(i)
    return b[lo] + (b[hi] - b[lo]) * (i - lo)

def vm3(runs, key='ey'):
    # VM2 と同じ式: 全部の回がそろうマスで、平均からの |差| の平均(回について)→ マスについて平均
    if not runs: return None, 0
    arrs = [r['bins'][key] for r in runs]
    nb = len(arrs[0]); vals = []
    for j in range(nb):
        col = [a[j] for a in arrs]
        if any(c is None for c in col): continue
        m = sum(col) / len(col)
        vals.append(sum(abs(c - m) for c in col) / len(col))
    return (sum(vals) / len(vals) if vals else None), len(vals)

def l4(fam):
    names = {}
    for r in fam:
        ln = r['line']
        if ln['family'] == 'const':
            nm = 'c0' if abs(ln['c0']) < 1e-9 else ('c+0.8' if ln['c0'] > 0 else 'c-0.8')
        else:
            nm = ln['family']
        names[nm] = r['bins']['ey']
    need = ['c-0.8', 'c0', 'c+0.8', 'inside', 'outside', 'late_apex']
    if any(n not in names for n in need): return None
    nb = len(names['c0'])
    ok = [all(names[n][j] is not None for n in need) for j in range(nb)]
    idx = [j for j in range(nb) if ok[j]]
    lr = statistics.mean(abs(names['c+0.8'][j] - names['c-0.8'][j]) for j in idx)
    io = statistics.mean(abs(names['inside'][j] - names['outside'][j]) for j in idx)
    pairs = []
    for a in range(6):
        for b in range(a + 1, 6):
            A, B = names[need[a]], names[need[b]]
            best = 0
            # 1 m(20 マス)の移動平均。続いてそろっているマスだけ(周の継ぎ目はまたがない)
            for j0 in range(nb - 19):
                if all(ok[j0 + k] for k in range(20)):
                    best = max(best, statistics.mean(abs(A[j0 + k] - B[j0 + k]) for k in range(20)))
            pairs.append({'a': need[a], 'b': need[b], 'max_1m_avg': best})
    return {'left_right_mean': lr, 'inside_outside_mean': io, 'min_pair_max_1m_avg': min(p['max_1m_avg'] for p in pairs),
            'closest_pair': min(pairs, key=lambda p: p['max_1m_avg']), 'bins': len(idx), 'pairs': pairs}

def setstats(runs, label=None):
    v = [r for r in runs if r.get('valid_r3')]
    trk = [x / 1e4 for r in v for x in r.get('trk_e4', [])]
    col = [r for r in runs if r['collisions'] > 0]
    st = [r for r in runs if r.get('stall')]
    nolap = [r for r in runs if r.get('valid_episode') and r['valid_laps'] < 1]
    vmin = min((r['min_clearance'] for r in v), default=None)
    sh = sum(r['shield_frames'] for r in runs); fr = sum(r['frames'] for r in runs)
    vm, nb = vm3(v)
    def brief(r):
        return {'id': r['id'], 'gate': r['start']['gate'], 'off': round(r['start']['left_offset_m'], 3), 'yaw': round(r['start']['left_yaw_deg'], 2),
                'dpsi': round(r['start_info']['dpsi'], 3), 'c_start': round(r['start_info']['clearance'], 4), 'family': r.get('family'), 'line': r['line'], 'c_v': r['c_v'],
                'reason': r['reason'], 'collision': r.get('collision'), 'min_clearance': round(r['min_clearance'], 5), 'min_t': round(r['min_t'], 2),
                'min_what': r.get('min_what'), 'min_v': r.get('min_v'), 'shield_frames': r['shield_frames'], 'shield_fb': r.get('shield_fb'),
                'stall': r.get('stall'), 'stall_first': r.get('stall_first'), 'stall_longest': r.get('stall_longest'), 'valid_laps': r['valid_laps']}
    return {'n': len(runs), 'valid_r3': len(v), 'valid_episode': sum(1 for r in runs if r.get('valid_episode')), 'collisions': len(col), 'stalls': len(st),
            'valid_but_no_lap': len(nolap),
            'min_clearance_valid': vmin, 'min_clearance_all': min((r['min_clearance'] for r in runs), default=None),
            'below_3cm': [brief(r) for r in runs if r['min_clearance'] < .03], 'below_5cm': [brief(r) for r in runs if r['min_clearance'] < .05],
            'collided': [brief(r) for r in col], 'stalled': [brief(r) for r in st], 'invalid': [brief(r) for r in runs if not r.get('valid_r3')],
            'tracking_median': quant(trk, .5), 'tracking_p95': quant(trk, .95), 'tracking_frames': len(trk),
            'over12_frac': (sum(1 for x in trk if x > .12) / len(trk)) if trk else None,
            'vm3': vm, 'vm3_bins': nb, 'shield_frames': sh, 'shield_fraction': sh / fr if fr else None,
            'shield_runs': sum(1 for r in runs if r['shield_frames'] > 0), 'shield_max_run': max((r['shield_fraction'] for r in runs), default=0),
            'shield_gt5pct': [brief(r) for r in runs if r['shield_fraction'] > .05], 'shield_gt10pct': [brief(r) for r in runs if r['shield_fraction'] > .10],
            'shield_fb': dict(collections.Counter(k for r in runs for k, c in (r.get('shield_fb') or {}).items() for _ in range(c))),
            'clr_median': quant([r['min_clearance'] for r in runs], .5), 'lap1_median': quant([r['lap1'] for r in runs if r.get('lap1')], .5),
            'speed_running_mean': statistics.mean(r['speed']['mean_running'] for r in runs) if runs else None,
            'large_heading': {'n': sum(1 for r in runs if abs(r['start_info']['dpsi']) >= 1),
                              'min_clearance': min((r['min_clearance'] for r in runs if abs(r['start_info']['dpsi']) >= 1), default=None),
                              'min_first_3s': min((r['clr_first_3s'] for r in runs if abs(r['start_info']['dpsi']) >= 1 and r.get('clr_first_3s') is not None), default=None),
                              'invalid': sum(1 for r in runs if abs(r['start_info']['dpsi']) >= 1 and not r.get('valid_r3'))}}

def judge(rep):
    by = collections.defaultdict(list)
    for r in rep: by[r['test']].append(r)
    P, H, E, F = by['LN-P60-R3'], by['LN-HEAD60-R3'], by['LN-EXT60-R3'], by['LN-FAM6']
    j = {}
    j['L1_r3'] = {'P': sum(r['valid_r3'] for r in P), 'HEAD': sum(r['valid_r3'] for r in H), 'EXT': sum(r['valid_r3'] for r in E)}
    j['L1_r3']['pass'] = all(j['L1_r3'][k] >= 57 for k in ('P', 'HEAD', 'EXT')) and len(P) == len(H) == len(E) == 60
    pool = [r for r in P + H + E if r['valid_r3']] + F
    m = min(pool, key=lambda r: r['min_clearance'])
    j['L2_r3'] = {'value': m['min_clearance'], 'where': m['id'], 'test': m['test'], 'pass': m['min_clearance'] >= .03}
    trk = [x / 1e4 for r in P if r['valid_r3'] for x in r['trk_e4']]
    j['L3'] = {'median': quant(trk, .5), 'p95': quant(trk, .95), 'frames': len(trk)}
    j['L3']['pass'] = j['L3']['median'] <= .05 and j['L3']['p95'] <= .12
    l4v = l4(F)
    j['L4'] = {k: v for k, v in l4v.items() if k != 'pairs'}
    j['L4']['fam6_all_valid_r3'] = all(r['valid_r3'] for r in F)
    j['L4']['pass'] = j['L4']['fam6_all_valid_r3'] and l4v['left_right_mean'] >= .2 and l4v['inside_outside_mean'] >= .1 and l4v['min_pair_max_1m_avg'] >= .08
    vm, nb = vm3([r for r in P if r['valid_r3']])
    j['L5'] = {'vm3': vm, 'bins': nb, 'pass': vm is not None and vm >= .08}
    c0 = [r for r in F if r['line']['family'] == 'const' and abs(r['line']['c0']) < 1e-9][0]
    j['L6'] = {'lap1': c0['lap1'], 'pass': c0['lap1'] is not None and c0['lap1'] <= 10.0 and c0['valid_r3']}
    shP = sum(r['shield_frames'] for r in P) / sum(r['frames'] for r in P)
    j['L7'] = {'P_fraction': shP, 'max_per_run_P_FAM6': max(r['shield_fraction'] for r in P + F)}
    j['L7']['pass'] = shP <= .02 and j['L7']['max_per_run_P_FAM6'] <= .10
    j['L8'] = {'collisions': sum(r['collisions'] > 0 for r in P + H + E + F), 'runs': len(P + H + E + F)}
    j['L8']['pass'] = j['L8']['collisions'] == 0
    j['all_pass'] = all(v['pass'] for v in j.values())
    return j, by

def main():
    cand = load('cand'); nosh = load('nosh'); web = load('web'); fast = load('fast')
    S = {'n_runs': {'cand': len(cand), 'nosh': len(nosh), 'web': len(web), 'fast': len(fast)}}
    rep = [r for r in cand if r['set'] == 'REPLAY']
    j, by = judge(rep)
    S['judged_replay'] = j
    S['replay_sets'] = {k: setstats(v) for k, v in by.items()}
    # 学習の段の行と照らす
    ev = json.load(open(f'{TR}/eval_test_r3.json'))
    trow = {r['id']: r for k, v in ev['tests'].items() for r in v}
    cmp = {'n': 0, 'reason_diff': [], 'valid_diff': [], 'lap_maxdiff': 0, 'minclr_maxdiff': 0, 'shield_frames_diff': []}
    for r in rep:
        t = trow.get(r['id'])
        if not t: continue
        cmp['n'] += 1
        if t['reason'] != r['reason']: cmp['reason_diff'].append(r['id'])
        if t.get('valid_r3') != r['valid_r3']: cmp['valid_diff'].append(r['id'])
        la = [l['elapsed'] for l in r['laps']]; lb = t.get('laps') or []
        if len(la) != len(lb): cmp['reason_diff'].append(r['id'] + ':laps')
        else:
            for a, b in zip(la, lb): cmp['lap_maxdiff'] = max(cmp['lap_maxdiff'], abs(a - b))
        if r['collisions'] == 0: cmp['minclr_maxdiff'] = max(cmp['minclr_maxdiff'], abs(r['min_clearance'] - t['min_clearance']))
        if r['shield_frames'] != t.get('shield_frames'): cmp['shield_frames_diff'].append(r['id'])
    S['replay_vs_training'] = cmp
    # 新しい組
    fresh = collections.defaultdict(list)
    for r in cand:
        if r['set'] != 'REPLAY': fresh[r['set']].append(r)
    S['fresh'] = {k: setstats(v) for k, v in fresh.items()}
    # 60 回ずつ(U300)
    u = sorted(fresh['U300'], key=lambda r: int(r['id'].split('-')[1]))
    S['U300_groups'] = []
    for g in range(5):
        st = setstats(u[60 * g:60 * g + 60])
        S['U300_groups'].append({k: st[k] for k in ('n', 'valid_r3', 'collisions', 'stalls', 'min_clearance_valid', 'tracking_median', 'tracking_p95', 'vm3', 'shield_fraction', 'shield_max_run')})
    # 全部の新しい組を合わせて
    allf = [r for v in fresh.values() for r in v]
    S['fresh_all'] = {k: v for k, v in setstats(allf).items() if k in ('n', 'valid_r3', 'collisions', 'stalls', 'min_clearance_valid', 'min_clearance_all', 'shield_frames',
                                                                         'shield_fraction', 'shield_runs', 'shield_max_run', 'valid_but_no_lap')}
    # 向きのずれの帯ごと(新しい組と REPLAY を合わせる)
    allc = cand
    bands = [(0, .3), (.3, .6), (.6, .8), (.8, 1.0), (1.0, 1.3), (1.3, 9)]
    S['by_dpsi_band'] = []
    for lo, hi in bands:
        rr = [r for r in allc if lo <= abs(r['start_info']['dpsi']) < hi]
        S['by_dpsi_band'].append({'band': [lo, hi], 'n': len(rr), 'invalid': sum(not r['valid_r3'] for r in rr), 'collisions': sum(r['collisions'] > 0 for r in rr),
                                   'min_clearance': min((r['min_clearance'] for r in rr), default=None), 'p05_clearance': quant([r['min_clearance'] for r in rr], .05),
                                   'shield_runs': sum(r['shield_frames'] > 0 for r in rr)})
    # 開始のすき間の帯ごと
    S['by_start_clearance'] = []
    for lo, hi in [(.12, .14), (.14, .16), (.16, .2), (.2, .3), (.3, 9)]:
        rr = [r for r in allc if lo <= r['start_info']['clearance'] < hi]
        S['by_start_clearance'].append({'band': [lo, hi], 'n': len(rr), 'collisions': sum(r['collisions'] > 0 for r in rr), 'min_clearance': min((r['min_clearance'] for r in rr), default=None),
                                        'shield_runs': sum(r['shield_frames'] > 0 for r in rr)})
    # 形ごと(新しい U300・HEAD・MID・NEAR の valid、追従)
    fam = collections.defaultdict(list)
    for k in ('U300', 'HEAD120', 'MID120', 'NEAR60'):
        for r in fresh.get(k, []):
            fam[r['family']].append(r)
    S['by_family_fresh'] = {}
    for k, v in fam.items():
        vv = [r for r in v if r['valid_r3']]; trk = [x / 1e4 for r in vv for x in r['trk_e4']]
        S['by_family_fresh'][k] = {'n': len(v), 'valid_r3': len(vv), 'tracking_p95': quant(trk, .95), 'tracking_median': quant(trk, .5),
                                   'lap1_median': quant([r['lap1'] for r in vv if r['lap1']], .5), 'min_clearance': min((r['min_clearance'] for r in v), default=None)}
    # 12 cm 越えの場所
    sm = [x for r in cand if r['set'] in ('U300',) and r['valid_r3'] for x in r.get('over12_s_mod', [])]
    S['over12_U300'] = {'frames': len(sm), 'share_s_9_12': (sum(1 for x in sm if 9 <= x <= 12) / len(sm)) if sm else None,
                        'share_s_9p75_11p75': (sum(1 for x in sm if 9.75 <= x <= 11.75) / len(sm)) if sm else None}
    # シールドなし(シールドが働いた回だけ。働かなかった回は同じ走り)
    cmap = {(r['set'], r['id']): r for r in cand}
    acted = [r for r in nosh if r['set'] != 'REPLAY' or cmap[(r['set'], r['id'])]['shield_frames'] > 0]
    S['noshield_where_shield_acted'] = {'n': len(acted), 'collisions': sum(r['collisions'] > 0 for r in acted), 'below_3cm': sum(r['min_clearance'] < .03 for r in acted),
                                        'stalls': sum(bool(r['stall']) for r in acted),
                                        'runs': [{'set': r['set'], 'id': r['id'], 'gate': r['start']['gate'], 'dpsi': round(r['start_info']['dpsi'], 3), 'c_start': round(r['start_info']['clearance'], 4),
                                                  'line': r['line'], 'c_v': r['c_v'], 'noshield_reason': r['reason'], 'noshield_collision': r['collision'],
                                                  'noshield_min': round(r['min_clearance'], 4), 'shield_min': round(cmap[(r['set'], r['id'])]['min_clearance'], 4),
                                                  'shield_frames': cmap[(r['set'], r['id'])]['shield_frames'], 'shield_at': cmap[(r['set'], r['id'])]['shield_at'][:10]} for r in acted]}
    # シールドなしの REPLAY(学習の段の compare と照らす)と、共有の web/rl(シールドなし)が同じか
    nr = [r for r in nosh if r['set'] == 'REPLAY']
    if len(nr) == 186:
        jn, _ = judge(nr)
        S['judged_replay_noshield'] = jn
    if web:
        nm = {r['id']: r for r in nr}
        S['web_vs_noshield_replay'] = {'n': len(web), 'mismatch': [r['id'] for r in web if r['id'] in nm and (r['reason'] != nm[r['id']]['reason'] or abs(r['min_clearance'] - nm[r['id']]['min_clearance']) > 1e-12
                                                                                         or [l['elapsed'] for l in r['laps']] != [l['elapsed'] for l in nm[r['id']]['laps']])]}
    # 最速との違い(同じ開始)
    if fast:
        fm = {(r['set'] if r['set'] != 'REPLAY' else 'REPLAY', r['id']): r for r in fast}
        pairs = []
        for r in cand:
            f = fm.get((r['set'], r['id']))
            if not f or r['set'] not in ('U300', 'REPLAY'): continue
            a, b = r['bins']['ey'], f['bins']['ey']
            d = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
            if d: pairs.append({'id': r['id'], 'set': r['set'], 'family': r['family'], 'mean_abs_dey': statistics.mean(d), 'lap1_line': r['lap1'], 'lap1_fast': f['lap1'],
                                'v_line': r['speed']['mean_running'], 'v_fast': f['speed']['mean_running'], 'clr_line': r['min_clearance'], 'clr_fast': f['min_clearance'],
                                'fast_valid': f['valid_episode'], 'line_valid': r['valid_r3']})
        fv = [f for f in fast if f['set'] != 'FAST_STD']
        fs_u = [f for f in fast if f['set'] == 'U300']; ls_u = [cmap[('U300', f['id'])] for f in fs_u]
        fs_p = [f for f in fast if f['set'] == 'REPLAY']; ls_p = [cmap[('REPLAY', f['id'])] for f in fs_p]
        bf = collections.defaultdict(list)
        for p in pairs: bf[p['family']].append(p['mean_abs_dey'])
        S['vs_fastest'] = {'pairs': len(pairs), 'mean_abs_dey_mean': statistics.mean(p['mean_abs_dey'] for p in pairs), 'mean_abs_dey_min': min(p['mean_abs_dey'] for p in pairs),
                           'by_family': {k: statistics.mean(v) for k, v in bf.items()},
                           'lap1_line_median': quant([p['lap1_line'] for p in pairs if p['lap1_line']], .5), 'lap1_fast_median': quant([p['lap1_fast'] for p in pairs if p['lap1_fast']], .5),
                           'v_line_mean': statistics.mean(p['v_line'] for p in pairs), 'v_fast_mean': statistics.mean(p['v_fast'] for p in pairs),
                           'clr_line_median': quant([p['clr_line'] for p in pairs], .5), 'clr_fast_median': quant([p['clr_fast'] for p in pairs], .5),
                           'fast_valid': sum(p['fast_valid'] for p in pairs), 'line_valid': sum(p['line_valid'] for p in pairs),
                           'vm3_U60': {'line': vm3([r for r in ls_u if r['valid_r3']])[0], 'fastest': vm3([f for f in fs_u if f['valid_episode']])[0]},
                           'vm3_P60': {'line': vm3([r for r in ls_p if r['valid_r3']])[0], 'fastest': vm3([f for f in fs_p if f['valid_episode']])[0]},
                           'fast_collided': [{'set': f['set'], 'id': f['id'], 'collision': f['collision']} for f in fv if f['collisions'] > 0]}
    # ラベル・ずる
    lab = collections.Counter(); ph = collections.Counter(); extra = set(); bytes_max = 0; clv = 0; prof = collections.Counter(); meta_sh = collections.Counter()
    integ = 0; samp = -1; samp_n = 0; cross_bad = 0; cross_n = 0; lapds = []
    for r in cand:
        L = r['labels']
        for k in ('l_not1', 'ev_nonzero', 'sh_missing', 'sh_ph_mismatch', 'es_missing', 'cl_missing', 'p_after0', 'bad_keys', 'over_cap', 'steer_out', 'cmd_nonfinite', 'frames', 'sh1'):
            lab[k] += L[k]
        lab['p0_bad'] += 0 if L['p0_ok'] else 1
        for k, c in L['ph'].items(): ph[k] += c
        extra |= set(L['extra_keys']); bytes_max = max(bytes_max, L['bytes_max']); clv = max(clv, L['cl_vs_own_max'])
        prof[L['meta_line_profile']] += 1; meta_sh[L['meta_has_shield']] += 1
        if r['reason'] == 'collection_duration_complete': integ = max(integ, r['integ_err'])
        if r['sample_check']['n']: samp_n += r['sample_check']['n']; samp = max(samp, r['sample_check']['max_exact_minus_sample'])
        for c in r['crossings']:
            cross_n += 1; cross_bad += 0 if c['inside'] else 1
        for l in r['laps']:
            if l['gates'] == 33: lapds.append(l['own_ds'])
    S['labels'] = {**lab, 'ph': dict(ph), 'extra_keys': sorted(extra), 'bytes_max': bytes_max, 'runtime_dmin_vs_own_max': clv, 'meta_line_profile': dict(prof),
                   'meta_has_shield': {str(k): v for k, v in meta_sh.items()}}
    S['integrity'] = {'own_integrator_vs_sim_max': integ, 'sample_check_points': samp_n, 'max_exact_minus_sampled': samp, 'crossings': cross_n, 'crossings_outside_opening': cross_bad,
                      'lap_own_ds_min': min(lapds) if lapds else None, 'lap_own_ds_max': max(lapds) if lapds else None, 'laps': len(lapds)}
    json.dump(S, open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1)
    # 画面へ短く
    print(json.dumps({'n': S['n_runs'], 'judged': {k: (v if k == 'all_pass' else {kk: vv for kk, vv in v.items() if kk != 'closest_pair'}) for k, v in j.items()},
                      'replay_vs_training': cmp}, ensure_ascii=False, default=str)[:3000])
    for k, v in S['fresh'].items():
        print(k, {kk: v[kk] for kk in ('n', 'valid_r3', 'collisions', 'stalls', 'min_clearance_valid', 'min_clearance_all', 'tracking_median', 'tracking_p95', 'vm3', 'shield_frames', 'shield_runs', 'shield_max_run')})
        for kk in ('collided', 'stalled', 'below_3cm'):
            if v[kk]: print('  ', kk, json.dumps(v[kk], ensure_ascii=False)[:1500])
    print('U300 groups', json.dumps(S['U300_groups']))
    print('dpsi bands', json.dumps(S['by_dpsi_band']))
    print('start clr bands', json.dumps(S['by_start_clearance']))
    print('noshield', json.dumps({k: v for k, v in S['noshield_where_shield_acted'].items() if k != 'runs'}))
    print('labels', json.dumps(S['labels'], ensure_ascii=False))
    print('integrity', json.dumps(S['integrity']))
    if 'vs_fastest' in S: print('vs_fastest', json.dumps({k: v for k, v in S['vs_fastest'].items()}, ensure_ascii=False)[:2000])
    if 'judged_replay_noshield' in S: print('noshield replay', json.dumps(S['judged_replay_noshield'], default=str)[:1500])
    if 'web_vs_noshield_replay' in S: print('web vs nosh', S['web_vs_noshield_replay'])

if __name__ == '__main__':
    main()
