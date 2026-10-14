# 速さいろいろ(speed)の確認 r1 の集計(2026-10-14、確認の段)。run_sp.mjs の JSONL だけを読み、判定 S1〜S5 を protocol の文言から自前で数える。
# 学習の段の試験の結果(eval/speed/test_cond_step_001015808.json)とは、回ごとに比べるためだけに読む。
import json, sys, math, statistics as stt
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
RUNS = REP / 'check_speed_r1/runs'
LEVELS = [1.0, .875, .75, .625, .5]


def load(name):
    p = RUNS / f'{name}.jsonl'
    return [json.loads(l) for l in open(p)] if p.exists() else []


def q(a, p):
    b = sorted(a)
    if not b:
        return None
    i = p * (len(b) - 1); lo = math.floor(i); hi = math.ceil(i)
    return b[lo] + (b[hi] - b[lo]) * (i - lo)


def spread(profiles):
    """全員がそろうマスで、群の平均からの |差| の平均(VM2〜VM4 の式)"""
    if len(profiles) < 2:
        return None, 0
    nb = len(profiles[0]); s = 0.0; n = 0; bins = 0
    for b in range(nb):
        vals = [p[b] for p in profiles]
        if any(v is None for v in vals):
            continue
        m = sum(vals) / len(vals)
        s += sum(abs(v - m) for v in vals); n += len(vals); bins += 1
    return (s / n if n else None), bins


def vm1(rows):
    s = 0.0; n = 0
    for t in range(600):
        v = [r['st_time'][t] for r in rows if len(r['st_time']) > t]
        if len(v) != len(rows):
            continue
        m = sum(v) / len(v); s += sum(abs(x - m) for x in v); n += len(v)
    return s / n if n else None


def judge(rows, std, label):
    p60 = [r for r in rows if r['id'].startswith('P') or r['id'].startswith('F')]
    ok = [r for r in p60 if r['valid_episode']]
    out = {'label': label, 'runs': len(p60), 'valid': len(ok)}
    out['S1'] = {'value': f'{len(ok)}/{len(p60)}', 'pass': len(ok) >= math.ceil(57 / 60 * len(p60))}
    mc = min((r['min_clearance'] for r in ok), default=None)
    out['S2'] = {'value': mc, 'pass': mc is not None and mc >= .03,
                 'below_3cm': sorted([{'id': r['id'], 'c_v': r['c_v'], 'gate': r['start']['gate'], 'off': round(r['start']['left_offset_m'], 3),
                                       'yaw': round(r['start']['left_yaw_deg'], 1), 'start_clear': round(r['start_clearance'], 4),
                                       'min': round(r['min_clearance'], 4), 'kind': r['min_kind'], 'frame': r['min_frame'],
                                       's_from_start': round(r['min_at']['s_from_start'], 2), 's_mod': round(r['min_at']['s_mod'], 2),
                                       'v_at_min': round(r['min_at']['v'], 2), 'after_3m': round(r['clear_after_3m'], 4)}
                                      for r in ok if r['min_clearance'] < .03], key=lambda x: x['min'])}
    if std:
        t = {r['c_v']: r for r in std}
        def lt(cv):
            r = t.get(cv)
            return r['lap_from_stop'] if r else None
        ks = sorted(t.keys(), reverse=True)
        t1, t75, t5 = lt(ks[0]), lt(ks[1]), lt(ks[2])
        allc = all(r['lap_from_stop'] is not None for r in std)
        out['S3'] = {'value': {'t1': t1, 't75': t75, 't5': t5, 'diff': (t5 - t1) if allc else None}, 'pass': allc and t1 < t75 < t5 and t5 - t1 >= 3.0,
                     'crossings': [r['single_crossings'] for r in std], 'gates': [r['passed_gates'] for r in std]}
    cvs = sorted(set(r['c_v'] for r in p60))
    per = []
    for cv in cvs:
        rs = [r for r in ok if r['c_v'] == cv]
        per.append({'c_v': cv, 'n': len(rs), 'mean_speed': sum(r['speed']['mean_running'] for r in rs) / len(rs) if rs else None,
                    'p95_max': max((r['speed']['p95'] for r in rs), default=None), 'v_max': max((r['speed']['max'] for r in rs), default=None),
                    'cap': 4.5 * cv})
    mono = all(per[i]['mean_speed'] - per[i - 1]['mean_speed'] >= .05 for i in range(1, len(per)))
    capok = all(x['n'] > 0 and x['p95_max'] <= 4.5 * x['c_v'] + .05 for x in per)
    out['S4'] = {'value': per, 'diffs': [per[i]['mean_speed'] - per[i - 1]['mean_speed'] for i in range(1, len(per))], 'pass': mono and capok}
    v4, bins = spread([r['prof']['v'] for r in ok])
    out['S5'] = {'value': v4, 'bins': bins, 'pass': v4 is not None and v4 >= .4}
    out['VM'] = {'VM1_us': vm1(ok), 'VM2_us': spread([r['prof']['st'] for r in ok])[0], 'VM3_m': spread([r['prof']['ey'] for r in ok])[0]}
    lap1 = [r['laps'][0]['elapsed'] for r in ok if r['laps']]
    out['lap1'] = {'n': len(lap1), 'min': min(lap1, default=None), 'max': max(lap1, default=None), 'range': (max(lap1) - min(lap1)) if lap1 else None}
    out['per_cv'] = []
    for cv in cvs:
        rs = [r for r in p60 if r['c_v'] == cv]; vk = [r for r in rs if r['valid_episode']]
        l1 = [r['laps'][0]['elapsed'] for r in vk if r['laps']]
        out['per_cv'].append({'c_v': cv, 'n': len(rs), 'valid': len(vk), 'collisions': sum(1 for r in rs if r['collisions']),
                              'lap1_mean': sum(l1) / len(l1) if l1 else None, 'lap1_min': min(l1, default=None), 'lap1_max': max(l1, default=None),
                              'min_clear': min((r['min_clearance'] for r in vk), default=None), 'median_clear': q([r['min_clearance'] for r in vk], .5),
                              'lt3': sum(1 for r in vk if r['min_clearance'] < .03), 'lt5': sum(1 for r in vk if r['min_clearance'] < .05),
                              'track_R_median': q([r['tracking_R']['median'] for r in vk if r['tracking_R']['median'] is not None], .5),
                              'track_R_p95_med': q([r['tracking_R']['p95'] for r in vk if r['tracking_R']['p95'] is not None], .5),
                              'rho_req_gt1': sum(r['rho_req_gt1'] for r in vk) / len(vk) if vk else None})
    out['all_pass'] = all(out[k]['pass'] for k in ['S1', 'S2', 'S3', 'S4', 'S5'] if k in out)
    return out


def integrity(rows):
    allr = [r for r in rows if not r.get('invalid_spawn')]
    sc = [r['sample_check'] for r in allr if r['sample_check']['n']]
    laps = [l for r in allr for l in r['laps']]
    cross = [c for r in allr for c in r['own_crossings']]
    lab = [r['labels'] for r in allr]
    return {'runs': len(allr), 'audit_pos_err_max': max(r['audit_pos_err'] for r in allr),
            'overlap_frames_no_collision': sum(r['overlap_frames'] for r in allr if r['collisions'] == 0),
            'collision_runs': sum(1 for r in allr if r['collisions']),
            'sample_checked_poses': sum(s['n'] for s in sc),
            'exact_minus_sample_max': max((s['max_exact_minus_sample'] for s in sc), default=None),
            'sample_minus_exact_max': max((s['max_sample_minus_exact'] for s in sc), default=None),
            'laps': len(laps), 'laps_not_33_gates': sum(1 for l in laps if l['gates'] != 33), 'laps_crossings_ne1': sum(1 for l in laps if l['crossings'] != 1),
            'lap_distance_min': min((l['distance'] for l in laps), default=None), 'lap_distance_max': max((l['distance'] for l in laps), default=None),
            'own_crossings': len(cross), 'own_crossings_outside': sum(1 for c in cross if not c['inside']),
            'label_l0': sum(x['l0'] for x in lab), 'label_frames': sum(x['frames'] for x in lab), 'ph': {k: sum(x['ph'].get(k, 0) for x in lab) for k in set(k for x in lab for k in x['ph'])},
            'sh_key': sum(x['sh_key'] for x in lab), 'bad_keys': sum(x['bad_keys'] for x in lab), 'cmd_over_cap': sum(x['cmd_over_cap'] for x in lab),
            'p0_cv_mismatch': sum(1 for r in allr if r['policy'] == 'speed' and (r['labels']['p0'] or {}).get('c_v') != r['params'].get('c_v')),
            'p0_line_not_c0': sum(1 for r in allr if r['policy'] == 'speed' and (r['labels']['p0'] or {}).get('line') != {'family': 'const', 'c0': 0}),
            'es_values': sorted(set(v for x in lab for v in x['es_vals'])),
            'proj_ds_max': max(r['proj_check']['max_ds_vs_runtime'] for r in allr), 'proj_dey_max': max(r['proj_check']['max_dey_vs_runtime'] for r in allr),
            'stops_after_1s_runs': sum(1 for r in allr if r['stops_after_1s'] > 0 and r['collisions'] == 0)}


def compare_training(rows):
    d = json.load(open(REP / 'eval/speed/test_cond_step_001015808.json'))
    tr = {r['id']: r for r in d['tests']['SP-P60']}
    std = {r['id']: r for r in d['tests']['SP-STD3']}
    diffs = []; mism = []
    for r in rows:
        t = tr.get(r['id']) or std.get(r['id'])
        if not t:
            continue
        if (t['valid_episode'] if r['id'].startswith('P') else t['lap_complete']) != (r['valid_episode'] if r['id'].startswith('P') else r['lap_from_stop'] is not None):
            mism.append(r['id'])
        if t.get('collisions') != r['collisions']:
            mism.append(r['id'] + ':col')
        if t.get('min_clearance') is not None:
            diffs.append(abs(t['min_clearance'] - r['min_clearance']))
        tl = t.get('laps') or []
        rl = [l['elapsed'] for l in r['laps']]
        if r['id'].startswith('P') and (len(tl) != len(rl) or any(abs(a - b) > 1e-9 for a, b in zip(tl, rl))):
            mism.append(r['id'] + ':laps')
    return {'compared': len(diffs), 'max_clearance_diff_m': max(diffs, default=None), 'mismatches': mism, 'training_judged': d['judged'],
            'training_vm': d['reported']['variation']}


def pair(a_rows, b_rows):
    """同じ開始の 2 つの組を比べる(線・速さ・すき間)"""
    B = {r['id']: r for r in b_rows}
    out = []
    for r in a_rows:
        o = B.get(r['id'])
        if not o or not r['valid_episode'] or not o['valid_episode']:
            continue
        de = [abs(x - y) for x, y in zip(r['prof']['ey'], o['prof']['ey']) if x is not None and y is not None]
        dv = [x - y for x, y in zip(r['prof']['v'], o['prof']['v']) if x is not None and y is not None]
        out.append({'id': r['id'], 'c_v': r['c_v'], 'line_diff_mean': sum(de) / len(de) if de else None, 'line_diff_p90': q(de, .9),
                    'speed_diff_mean': sum(dv) / len(dv) if dv else None, 'clear_a': r['min_clearance'], 'clear_b': o['min_clearance'],
                    'lap1_a': r['laps'][0]['elapsed'] if r['laps'] else None, 'lap1_b': o['laps'][0]['elapsed'] if o['laps'] else None})
    return out


def summarize_pairs(P):
    res = {}
    for cv in sorted(set(x['c_v'] for x in P)):
        xs = [x for x in P if x['c_v'] == cv]
        res[str(cv)] = {'n': len(xs), 'line_diff_mean_cm': 100 * stt.mean(x['line_diff_mean'] for x in xs), 'line_diff_p90_cm_median': 100 * q([x['line_diff_p90'] for x in xs], .5),
                        'speed_diff_mean': stt.mean(x['speed_diff_mean'] for x in xs),
                        'clear_a_median_cm': 100 * q([x['clear_a'] for x in xs], .5), 'clear_b_median_cm': 100 * q([x['clear_b'] for x in xs], .5),
                        'lap1_a_mean': stt.mean(x['lap1_a'] for x in xs if x['lap1_a']), 'lap1_b_mean': stt.mean(x['lap1_b'] for x in xs if x['lap1_b'])}
    return res


def within_line_spread(rows, cv):
    ok = [r for r in rows if r['valid_episode'] and abs(r['c_v'] - cv) < 1e-9]
    return spread([r['prof']['ey'] for r in ok])[0], spread([r['prof']['v'] for r in ok])[0], len(ok)


def fresh_summary(rows, label):
    allr = [r for r in rows if not r.get('invalid_spawn')]
    ok = [r for r in allr if r['valid_episode']]
    o = {'label': label, 'runs': len(allr), 'valid': len(ok), 'collisions': sum(1 for r in allr if r['collisions']),
         'not_600': sum(1 for r in allr if r['reason'] != 'collection_duration_complete'),
         'min_clear': min((r['min_clearance'] for r in ok), default=None), 'lt3': sum(1 for r in ok if r['min_clearance'] < .03),
         'lt3_start_lt3': sum(1 for r in ok if r['min_clearance'] < .03 and r['start_clearance'] < .03),
         'lt3_within_3m': sum(1 for r in ok if r['min_clearance'] < .03 and r['min_at']['s_from_start'] < 3),
         'lt3_after_3m': sum(1 for r in ok if r['clear_after_3m'] < .03), 'min_after_3m': min((r['clear_after_3m'] for r in ok), default=None),
         'lt5': sum(1 for r in ok if r['min_clearance'] < .05), 'start_lt3': sum(1 for r in allr if r['start_clearance'] < .03),
         'start_lt8': sum(1 for r in allr if r['start_clearance'] < .08),
         'collided': [{'id': r['id'], 'c_v': r['c_v'], 'gate': r['start']['gate'], 'off': round(r['start']['left_offset_m'], 3), 'yaw': round(r['start']['left_yaw_deg'], 1),
                       'start_clear': round(r['start_clearance'], 4), 'reason': r['reason'], 'frame': r['frames'], 'collision': r['collision']} for r in allr if r['collisions'] or r['reason'] != 'collection_duration_complete'],
         'lt3_list': sorted([{'id': r['id'], 'c_v': r['c_v'], 'gate': r['start']['gate'], 'off': round(r['start']['left_offset_m'], 3), 'yaw': round(r['start']['left_yaw_deg'], 1),
                              'start_clear': round(r['start_clearance'], 4), 'min': round(r['min_clearance'], 4), 'kind': r['min_kind'], 's_from_start': round(r['min_at']['s_from_start'], 2),
                              's_mod': round(r['min_at']['s_mod'], 2), 'v': round(r['min_at']['v'], 2)} for r in ok if r['min_clearance'] < .03], key=lambda x: x['min'])}
    # 8 cm 以上で始まる回だけ(第 2 回の提案「開始のすき間 ≥ 8 cm」の確かめ)
    ok8 = [r for r in ok if r['start_clearance'] >= .08]
    all8 = [r for r in allr if r['start_clearance'] >= .08]
    o['start_ge8'] = {'runs': len(all8), 'valid': len(ok8), 'lt3': sum(1 for r in ok8 if r['min_clearance'] < .03), 'min': min((r['min_clearance'] for r in ok8), default=None)}
    # 黒シートの入口(s ≈ 6.95)の 2 m 手前から出ない + 開始 ≥ 8 cm
    def near_entry(r):
        s0 = (r['min_at']['s_mod'] - r['min_at']['s_from_start']) % 26.1196
        return 4.95 <= s0 <= 6.95
    okF = [r for r in ok8 if not near_entry(r)]
    o['start_ge8_not_entry'] = {'valid': len(okF), 'lt3': sum(1 for r in okF if r['min_clearance'] < .03), 'min': min((r['min_clearance'] for r in okF), default=None)}
    return o


def main():
    res = {}
    rep = load('REPLAY')
    if rep:
        p = [r for r in rep if r['id'].startswith('P')]; s = [r for r in rep if r['id'].startswith('cv')]
        res['REPLAY'] = judge(p, s, 'cond 101 万 試験の組(作り直し)')
        res['REPLAY_vs_training'] = compare_training(rep)
        res['REPLAY_integrity'] = integrity(rep)
        res['REPLAY_start_clear'] = sorted([(r['id'], round(r['start_clearance'], 4)) for r in p], key=lambda x: x[1])[:6]
    fb = load('REPLAY_FB')
    if fb:
        res['REPLAY_FB'] = judge([r for r in fb if r['id'].startswith('P')], [r for r in fb if r['id'].startswith('cv')], '代わりの方法: 最速 + 上限')
        res['REPLAY_FB_integrity'] = integrity(fb)
    fr = load('FRESH')
    if fr:
        res['FRESH_judge'] = judge(fr, None, 'cond 新しい開始 300')
        res['FRESH'] = fresh_summary(fr, 'cond 新しい開始')
        res['FRESH_integrity'] = integrity(fr)
    fc = load('FRESH_CAP')
    if fc:
        res['FRESH_CAP_judge'] = judge(fc, None, '最速 + 同じ上限 新しい開始 300')
        res['FRESH_CAP'] = fresh_summary(fc, '最速 + 同じ上限')
        if fr:
            res['pair_cond_vs_cap'] = summarize_pairs(pair(fr, fc))
            res['within'] = {str(cv): {'cond': within_line_spread(fr, cv), 'cap': within_line_spread(fc, cv)} for cv in LEVELS}
    ff = load('FRESH_FAST')
    if ff:
        res['FRESH_FAST'] = fresh_summary(ff, '最速(上限なし)')
        if fr:
            res['pair_cond1_vs_fastest'] = summarize_pairs(pair([r for r in fr if r['c_v'] == 1.0], ff))
            # c_v の段ごとの cond と最速(上限なし)の線の違い(同じ開始の組が無いので、マスの平均の線どうし)
            def mean_prof(rows, key):
                ok = [r for r in rows if r['valid_episode']]
                nb = len(ok[0]['prof'][key]); out = []
                for b in range(nb):
                    v = [r['prof'][key][b] for r in ok if r['prof'][key][b] is not None]
                    out.append(sum(v) / len(v) if v else None)
                return out
            fp = mean_prof(ff, 'ey'); fv = mean_prof(ff, 'v')
            res['mean_line_vs_fastest'] = {}
            for cv in LEVELS:
                cp = mean_prof([r for r in fr if r['c_v'] == cv], 'ey'); cvv = mean_prof([r for r in fr if r['c_v'] == cv], 'v')
                d = [abs(a - b) for a, b in zip(cp, fp) if a is not None and b is not None]
                dv = [a - b for a, b in zip(cvv, fv) if a is not None and b is not None]
                res['mean_line_vs_fastest'][str(cv)] = {'line_mean_cm': 100 * sum(d) / len(d), 'line_p90_cm': 100 * q(d, .9), 'line_max_cm': 100 * max(d),
                                                         'speed_diff_mean': sum(dv) / len(dv)}
            if fc:
                for cv in LEVELS:
                    cp = mean_prof([r for r in fc if r['c_v'] == cv], 'ey')
                    d = [abs(a - b) for a, b in zip(cp, fp) if a is not None and b is not None]
                    res['mean_line_vs_fastest'][str(cv)]['cap_line_mean_cm'] = 100 * sum(d) / len(d)
    if fr and fc:
        # c_v の段で線を変えているか: c_v 0.5 と 1.0 の平均の線の違い(cond と、最速 + 上限)
        def mprof(rows, key):
            ok = [r for r in rows if r['valid_episode']]
            if not ok:
                return [None] * 523
            nb = len(ok[0]['prof'][key]); out = []
            for b in range(nb):
                v = [r['prof'][key][b] for r in ok if r['prof'][key][b] is not None]
                out.append(sum(v) / len(v) if v else None)
            return out
        def dline(a, b):
            d = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
            if not d:
                return None
            return {'mean_cm': 100 * sum(d) / len(d), 'p90_cm': 100 * q(d, .9), 'max_cm': 100 * max(d)}
        res['line_by_cv'] = {}
        for nm, rows in (('cond', fr), ('cap', fc)):
            P = {cv: mprof([r for r in rows if r['c_v'] == cv], 'ey') for cv in LEVELS}
            res['line_by_cv'][nm] = {f'{a}-{b}': dline(P[a], P[b]) for a, b in ((1.0, .5), (1.0, .75), (.75, .5))}
    sv = load('STDV')
    if sv:
        starts = {}
        for r in sv:
            k = (round(r['start']['left_offset_m'], 3), round(r['start']['left_yaw_deg'], 1))
            starts.setdefault(k, {})[r['c_v']] = r['lap_from_stop']
        rows = []
        for k, t in sorted(starts.items()):
            ts = [t.get(cv) for cv in LEVELS]
            ok3 = None not in (t.get(1.0), t.get(.75), t.get(.5)) and t[1.0] < t[.75] < t[.5] and t[.5] - t[1.0] >= 3
            mono5 = None not in ts and all(ts[i] < ts[i + 1] for i in range(4))
            rows.append({'start': k, 'laps': ts, 'S3_like': ok3, 'mono5': mono5})
        res['STDV'] = rows
        res['STDV_integrity'] = integrity(sv)
    dr = load('DR')
    if dr:
        res['DR'] = fresh_summary(dr, 'DR-T(参考)')
        res['DR']['per_delay'] = {str(dl): {'n': sum(1 for r in dr if r['dr']['delay'] == dl), 'collisions': sum(1 for r in dr if r['dr']['delay'] == dl and r['collisions'])} for dl in (0, 1)}
    sc = load('SEEDCHK')
    if sc:
        res['SEEDCHK'] = {'identical': len(set(json.dumps([r['laps'], r['min_clearance'], r['distance']]) for r in sc)) == 1, 'n': len(sc)}
    out = REP / 'check_speed_r1/summary.json'
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=lambda o: None)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: None)[:20000])


if __name__ == '__main__':
    main()
