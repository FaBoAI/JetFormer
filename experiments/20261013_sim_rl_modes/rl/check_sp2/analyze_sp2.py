# 速さいろいろ(speed)の確認 r2 の集計(2026-10-14、確認の段)。run_sp2.mjs の JSONL だけを読み、判定 S1〜S5 を protocol.json と protocol_speed_r2.json の文言から自前で数える。
# 学習の段の試験の結果(eval/speed/r2/test_r2_step_002015232.json)とは、回ごとに比べるためだけに読む。
import json, math, statistics as stt
from collections import Counter, defaultdict
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
RUNS = REP / 'check_speed_r2/runs'
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


def a3(r):
    """開始から 3 m 以降の最小すき間(3 m 進まなかった回 = 止まったままの回は None)"""
    v = r.get('clear_after_3m')
    return v if v is not None and math.isfinite(v) else None


def stalled(r):
    """止まったままの回: 有効な回(600 フレーム・衝突 0)なのに、有効な周が 0 で、進んだ距離が 3 m 未満"""
    return bool(r.get('valid_episode')) and r.get('valid_laps', 0) == 0 and (r.get('distance') or 0) < 3


def clr(r):
    """第 2 回の S2 の数え方: 衝突・開始できない・測れない回は 0"""
    if r.get('invalid_spawn') or r.get('collisions'):
        return 0.0
    c = r.get('min_clearance')
    return c if c is not None and math.isfinite(c) else 0.0


def brief(r):
    return {'id': r['id'], 'c_v': r['c_v'], 'gate': r['start']['gate'], 'off': round(r['start']['left_offset_m'], 3), 'yaw': round(r['start']['left_yaw_deg'], 1),
            'start_clear': round(r.get('start_clearance') or 0, 4), 'min': round(clr(r), 4), 'kind': r.get('min_kind'), 'reason': r.get('reason'),
            's_from_start': round(r['min_at']['s_from_start'], 2) if r.get('min_at') else None, 's_mod': round(r['min_at']['s_mod'], 2) if r.get('min_at') else None,
            'v_at_min': round(r['min_at']['v'], 2) if r.get('min_at') else None, 'after_3m': round(a3(r), 4) if a3(r) is not None else None, 'stalled': stalled(r), 'distance': round(r.get('distance') or 0, 2)}


def judge(rows, std, label):
    p60 = [r for r in rows if r['id'][0] in 'PFGHE']
    ok = [r for r in p60 if r['valid_episode']]
    out = {'label': label, 'runs': len(p60), 'valid': len(ok)}
    out['S1'] = {'value': f'{len(ok)}/{len(p60)}', 'pass': len(ok) >= math.ceil(57 / 60 * len(p60))}
    mc = min((clr(r) for r in p60), default=None)
    mcv = min((r['min_clearance'] for r in ok), default=None)
    out['S2'] = {'value': mc, 'pass': mc is not None and mc >= .03, 'value_valid_only_r1_rule': mcv,
                 'below_3cm': sorted([brief(r) for r in p60 if clr(r) < .03], key=lambda x: x['min']),
                 'below_5cm_n': sum(1 for r in p60 if clr(r) < .05), 'collided': [r['id'] for r in p60 if r.get('collisions')],
                 'lowest5': sorted([brief(r) for r in p60], key=lambda x: x['min'])[:5]}
    if std:
        t = {r['c_v']: r for r in std}
        ks = sorted(t.keys(), reverse=True)
        t1, t75, t5 = (t[k]['lap_from_stop'] for k in ks[:3])
        allc = all(r['lap_from_stop'] is not None for r in std)
        out['S3'] = {'value': {'t1': t1, 't75': t75, 't5': t5, 'diff': (t5 - t1) if allc else None}, 'pass': allc and t1 < t75 < t5 and t5 - t1 >= 3.0,
                     'crossings': [r['single_crossings'] for r in std], 'gates': [r['passed_gates'] for r in std], 'std_min_clear': min(r['min_clearance'] for r in std)}
    cvs = sorted(set(r['c_v'] for r in p60))
    per = []
    for cv in cvs:
        rs = [r for r in ok if r['c_v'] == cv]
        per.append({'c_v': cv, 'n': len(rs), 'mean_speed': sum(r['speed']['mean_running'] for r in rs) / len(rs) if rs else None,
                    'p95_max': max((r['speed']['p95'] for r in rs), default=None), 'v_max': max((r['speed']['max'] for r in rs), default=None), 'cap': 4.5 * cv})
    mono = all(per[i]['mean_speed'] is not None and per[i - 1]['mean_speed'] is not None and per[i]['mean_speed'] - per[i - 1]['mean_speed'] >= .05 for i in range(1, len(per)))
    capok = all(x['n'] > 0 and x['p95_max'] <= 4.5 * x['c_v'] + .05 for x in per)
    out['S4'] = {'value': per, 'diffs': [per[i]['mean_speed'] - per[i - 1]['mean_speed'] for i in range(1, len(per)) if per[i]['mean_speed'] and per[i - 1]['mean_speed']], 'pass': mono and capok}
    v4, bins = spread([r['prof']['v'] for r in ok])
    out['S5'] = {'value': v4, 'bins': bins, 'pass': v4 is not None and v4 >= .4}
    out['VM'] = {'VM1_us': vm1(ok), 'VM2_us': spread([r['prof']['st'] for r in ok])[0], 'VM3_m': spread([r['prof']['ey'] for r in ok])[0]}
    lap1 = [r['laps'][0]['elapsed'] for r in ok if r['laps']]
    out['lap1'] = {'n': len(lap1), 'min': min(lap1, default=None), 'max': max(lap1, default=None), 'range': (max(lap1) - min(lap1)) if lap1 else None}
    out['per_cv'] = []
    for cv in cvs:
        rs = [r for r in p60 if r['c_v'] == cv]; vk = [r for r in rs if r['valid_episode']]
        l1 = [r['laps'][0]['elapsed'] for r in vk if r['laps']]
        out['per_cv'].append({'c_v': cv, 'n': len(rs), 'valid': len(vk), 'collisions': sum(1 for r in rs if r.get('collisions')),
                              'not600': sum(1 for r in rs if r.get('reason') != 'collection_duration_complete'),
                              'lap1_mean': sum(l1) / len(l1) if l1 else None, 'lap1_min': min(l1, default=None), 'lap1_max': max(l1, default=None),
                              'min_clear': min((clr(r) for r in rs), default=None), 'median_clear': q([clr(r) for r in rs], .5),
                              'lt3': sum(1 for r in rs if clr(r) < .03), 'lt5': sum(1 for r in rs if clr(r) < .05),
                              'after3m_min': min((a3(r) for r in vk if a3(r) is not None), default=None),
                              'post_zone_min': min((r['post_zone_min'] for r in vk if r.get('post_zone_min') is not None), default=None),
                              'post_zone_median': q([r['post_zone_min'] for r in vk if r.get('post_zone_min') is not None], .5),
                              'sat_frac': (sum(r['labels']['sat'] for r in vk) / max(1, sum(r['labels']['run'] for r in vk))) if vk else None,
                              'track_R_median': q([r['tracking_R']['median'] for r in vk if r['tracking_R']['median'] is not None], .5),
                              'rho_req_gt1': sum(r['rho_req_gt1'] for r in vk) / len(vk) if vk else None})
    out['all_pass'] = all(out[k]['pass'] for k in ['S1', 'S2', 'S3', 'S4', 'S5'] if k in out)
    return out


def integrity(rows):
    allr = [r for r in rows if not r.get('invalid_spawn')]
    sc = [r['sample_check'] for r in allr if r['sample_check']['n']]
    laps = [l for r in allr for l in r['laps']]
    cross = [c for r in allr for c in r['own_crossings']]
    lab = [r['labels'] for r in allr]
    spd = [r for r in allr if r['policy'].startswith('speed')]
    return {'runs': len(allr), 'invalid_spawn': sum(1 for r in rows if r.get('invalid_spawn')), 'audit_pos_err_max': max(r['audit_pos_err'] for r in allr),
            'overlap_frames_no_collision': sum(r['overlap_frames'] for r in allr if r['collisions'] == 0),
            'collision_runs': sum(1 for r in allr if r['collisions']), 'early_end_runs': sum(1 for r in allr if r['episode_mode'] == 'continuous' and r['reason'] != 'collection_duration_complete'),
            'reasons': dict(Counter(r['reason'] for r in allr)),
            'sample_checked_poses': sum(s['n'] for s in sc),
            'exact_minus_sample_max': max((s['max_exact_minus_sample'] for s in sc), default=None),
            'sample_minus_exact_max': max((s['max_sample_minus_exact'] for s in sc), default=None),
            'laps': len(laps), 'laps_not_33_gates': sum(1 for l in laps if l['gates'] != 33), 'laps_crossings_ne1': sum(1 for l in laps if l['crossings'] != 1),
            'lap_distance_min': min((l['distance'] for l in laps), default=None), 'lap_distance_max': max((l['distance'] for l in laps), default=None),
            'own_crossings': len(cross), 'own_crossings_outside': sum(1 for c in cross if not c['inside']),
            'label_l0': sum(x['l0'] for x in lab), 'label_frames': sum(x['frames'] for x in lab), 'ph': {k: sum(x['ph'].get(k, 0) for x in lab) for k in set(k for x in lab for k in x['ph'])},
            'sh_key': sum(x['sh_key'] for x in lab), 'bad_keys': sum(x['bad_keys'] for x in lab), 'cmd_over_cap': sum(x['cmd_over_cap'] for x in lab),
            'neg_speed_cmd': sum(x.get('neg_speed_cmd', 0) for x in lab),
            'p0_cv_mismatch': sum(1 for r in spd if (r['labels']['p0'] or {}).get('c_v') != r['params'].get('c_v')),
            'p0_line_not_c0': sum(1 for r in spd if (r['labels']['p0'] or {}).get('line') != {'family': 'const', 'c0': 0}),
            'p0_policy': sorted(set(str((r['labels']['p0'] or {}).get('policy')) for r in allr)),
            'proj_ds_max': max(r['proj_check']['max_ds_vs_runtime'] for r in allr), 'proj_dey_max': max(r['proj_check']['max_dey_vs_runtime'] for r in allr),
            'runtime_dmin_vs_own_max': max(r['proj_check']['max_dmin_vs_own'] for r in allr),
            'stops_after_1s_runs': sum(1 for r in allr if r['stops_after_1s'] > 0 and r['collisions'] == 0), 'stalled_runs': sum(1 for r in allr if stalled(r)),
            'runtimes': sorted(set(r['runtime'] for r in allr))}


def compare_training(rows):
    d = json.load(open(REP / 'eval/speed/r2/test_r2_step_002015232.json'))
    tr = {r['id']: r for r in d['tests']['SP-P60']}
    std = {r['id']: r for r in d['tests']['SP-STD3']}
    diffs = []; mism = []; start_mism = []
    for r in rows:
        t = tr.get(r['id']) or std.get(r['id'])
        if not t:
            mism.append(r['id'] + ':missing')
            continue
        if r['id'].startswith('P'):
            ts, rs = t['start'], r['start']
            if ts.get('halton_index') != rs.get('halton_index') or abs(ts['left_offset_m'] - rs['left_offset_m']) > 1e-12 or abs(ts['left_yaw_deg'] - rs['left_yaw_deg']) > 1e-9 \
                    or ts['gate'] != rs['gate'] or ts['seed'] != rs['seed'] or abs((t.get('c_v') or 0) - r['c_v']) > 1e-12:
                start_mism.append(r['id'])
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
        if r['id'].startswith('cv') and t.get('lap_from_stop') is not None and abs(t['lap_from_stop'] - r['lap_from_stop']) > 1e-9:
            mism.append(r['id'] + ':lfs')
    return {'compared': len(diffs), 'max_clearance_diff_m': max(diffs, default=None), 'mismatches': mism, 'start_mismatches': start_mism,
            'training_judged': d['judged'], 'training_vm': d['reported']['variation'], 'training_ids': len(tr) + len(std)}


def compare_same(a_rows, b_rows):
    """同じ組を 2 つの走行中の仕組みで: 回ごとに舵の列・周・すき間が同じか"""
    B = {r['id']: r for r in b_rows}
    n = 0; diff = []; st_max = 0.0
    for r in a_rows:
        o = B.get(r['id'])
        if not o:
            diff.append(r['id'] + ':missing'); continue
        n += 1
        if r['reason'] != o['reason'] or r['frames'] != o['frames'] or [l['elapsed'] for l in r['laps']] != [l['elapsed'] for l in o['laps']] or abs(r['min_clearance'] - o['min_clearance']) > 0:
            diff.append(r['id'])
        if r.get('st_time') and o.get('st_time'):
            st_max = max(st_max, max(abs(x - y) for x, y in zip(r['st_time'], o['st_time'])))
    return {'compared': n, 'different': diff, 'steer_us_max_diff': st_max}


def pair(a_rows, b_rows):
    B = {r['id']: r for r in b_rows}
    out = []
    for r in a_rows:
        o = B.get(r['id'])
        if not o or not r['valid_episode'] or not o['valid_episode']:
            continue
        de = [abs(x - y) for x, y in zip(r['prof']['ey'], o['prof']['ey']) if x is not None and y is not None]
        dv = [x - y for x, y in zip(r['prof']['v'], o['prof']['v']) if x is not None and y is not None]
        dva = [abs(x) for x in dv]
        if not de or not dv:   # 止まったままの回(マスが空)は比べない
            continue
        out.append({'id': r['id'], 'c_v': r['c_v'], 'line_diff_mean': sum(de) / len(de) if de else None, 'line_diff_p90': q(de, .9),
                    'speed_diff_mean': sum(dv) / len(dv) if dv else None, 'speed_absdiff_mean': sum(dva) / len(dva) if dva else None,
                    'clear_a': clr(r), 'clear_b': clr(o), 'lap1_a': r['laps'][0]['elapsed'] if r['laps'] else None, 'lap1_b': o['laps'][0]['elapsed'] if o['laps'] else None,
                    'sat_a': r['labels']['sat'] / max(1, r['labels']['run']), 'sat_b': o['labels']['sat'] / max(1, o['labels']['run'])})
    return out


def summarize_pairs(P):
    res = {}
    for cv in sorted(set(x['c_v'] for x in P)):
        xs = [x for x in P if x['c_v'] == cv]
        res[str(cv)] = {'n': len(xs), 'line_diff_mean_cm': 100 * stt.mean(x['line_diff_mean'] for x in xs), 'line_diff_p90_cm_median': 100 * q([x['line_diff_p90'] for x in xs], .5),
                        'speed_diff_mean': stt.mean(x['speed_diff_mean'] for x in xs), 'speed_absdiff_mean': stt.mean(x['speed_absdiff_mean'] for x in xs),
                        'clear_a_median_cm': 100 * q([x['clear_a'] for x in xs], .5), 'clear_b_median_cm': 100 * q([x['clear_b'] for x in xs], .5),
                        'clear_a_min_cm': 100 * min(x['clear_a'] for x in xs), 'clear_b_min_cm': 100 * min(x['clear_b'] for x in xs),
                        'lap1_a_mean': stt.mean(x['lap1_a'] for x in xs if x['lap1_a']), 'lap1_b_mean': stt.mean(x['lap1_b'] for x in xs if x['lap1_b']),
                        'sat_a': stt.mean(x['sat_a'] for x in xs), 'sat_b': stt.mean(x['sat_b'] for x in xs)}
    return res


def mprof(rows, key):
    ok = [r for r in rows if r['valid_episode']]
    if not ok:
        return None
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


def within_line_spread(rows, cv):
    ok = [r for r in rows if r['valid_episode'] and abs(r['c_v'] - cv) < 1e-9]
    return {'VM3_cm': 100 * (spread([r['prof']['ey'] for r in ok])[0] or 0), 'VM4': spread([r['prof']['v'] for r in ok])[0], 'n': len(ok)}


def p_pass60(rate):
    return (1 - rate) ** 60


def upper95(k, n):
    """二項の 95% 上側(Clopper-Pearson の片側)。k = 0 なら 1 − 0.05^(1/n)"""
    if k == 0:
        return 1 - .05 ** (1 / n)
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        # P(X ≤ k | p = mid)
        c = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        if c > .05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fresh_summary(rows, label):
    allr = list(rows)
    ok = [r for r in allr if r['valid_episode']]
    n = len(allr); k3 = sum(1 for r in allr if clr(r) < .03); k5 = sum(1 for r in allr if clr(r) < .05)
    o = {'label': label, 'runs': n, 'valid': len(ok), 'collisions': sum(1 for r in allr if r.get('collisions')),
         'not_600': sum(1 for r in allr if r.get('reason') != 'collection_duration_complete'),
         'min_clear_all_r2rule': min((clr(r) for r in allr), default=None), 'min_clear_valid': min((r['min_clearance'] for r in ok), default=None),
         'lt3': k3, 'lt5': k5, 'lt3_rate_upper95': upper95(k3, n) if n else None,
         'p_pass60_point': p_pass60(k3 / n) if n else None, 'p_pass60_at_upper95': p_pass60(upper95(k3, n)) if n else None,
         'lt3_after_3m': sum(1 for r in ok if a3(r) is not None and a3(r) < .03), 'min_after_3m': min((a3(r) for r in ok if a3(r) is not None), default=None),
         'stalled': sum(1 for r in allr if stalled(r)), 'stalled_list': [brief(r) for r in allr if stalled(r)],
         'not600_list': [brief(r) for r in allr if r.get('reason') != 'collection_duration_complete' or r.get('collisions')],
         'lowest10': sorted([brief(r) for r in allr], key=lambda x: x['min'])[:10],
         'per_cv': {}}
    for cv in sorted(set(r['c_v'] for r in allr)):
        rs = [r for r in allr if r['c_v'] == cv]; vk = [r for r in rs if r['valid_episode']]
        l1 = [r['laps'][0]['elapsed'] for r in vk if r['laps']]
        o['per_cv'][str(cv)] = {'n': len(rs), 'valid': len(vk), 'lt3': sum(1 for r in rs if clr(r) < .03), 'lt5': sum(1 for r in rs if clr(r) < .05),
                                'min': min(clr(r) for r in rs), 'median': q([clr(r) for r in rs], .5), 'after3m_min': min((a3(r) for r in vk if a3(r) is not None), default=None),
                                'post_zone_min': min((r['post_zone_min'] for r in vk if r.get('post_zone_min') is not None), default=None),
                                'post_zone_median': q([r['post_zone_min'] for r in vk if r.get('post_zone_min') is not None], .5),
                                'lap1_mean': sum(l1) / len(l1) if l1 else None, 'mean_speed': stt.mean(r['speed']['mean_running'] for r in vk) if vk else None}
    g = defaultdict(list)
    for r in allr:
        g[r['start']['gate']].append(clr(r))
    o['by_gate_min'] = {str(k): round(min(v), 4) for k, v in sorted(g.items())}
    o['by_gate_lt5'] = {str(k): sum(1 for x in v if x < .05) for k, v in sorted(g.items()) if any(x < .05 for x in v)}
    return o


def main():
    res = {}
    rep = load('REPLAY')
    if rep:
        p = [r for r in rep if r['id'].startswith('P')]; s = [r for r in rep if r['id'].startswith('cv')]
        res['REPLAY'] = judge(p, s, 'speed r2 試験の組 SP-P60-R2(作り直し、rt_current)')
        res['REPLAY_vs_training'] = compare_training(rep)
        res['REPLAY_integrity'] = integrity(rep)
        res['REPLAY_start_clear_lowest'] = sorted([(r['id'], round(r['start_clearance'], 4)) for r in p], key=lambda x: x[1])[:5]
        res['REPLAY_halton_range'] = [p[0]['start']['halton_index'], p[-1]['start']['halton_index']]
    web = load('REPLAY_WEB')
    if web and rep:
        res['REPLAY_WEB_vs_current'] = compare_same(rep, web)
    r1 = load('REPLAY_R1')
    if r1:
        res['REPLAY_R1'] = judge([r for r in r1 if r['id'].startswith('P')], [r for r in r1 if r['id'].startswith('cv')], 'speed r1 を同じ試験の組で(参考)')
    fb = load('REPLAY_FB')
    if fb:
        res['REPLAY_FB'] = judge([r for r in fb if r['id'].startswith('P')], [r for r in fb if r['id'].startswith('cv')], '代わりの方法: 最速 + 上限(参考)')
    fr = load('FRESH')
    if fr:
        res['FRESH_judge'] = judge(fr, None, 'speed r2 新しい開始')
        res['FRESH'] = fresh_summary(fr, 'speed r2 新しい開始')
        res['FRESH_integrity'] = integrity(fr)
        # 60 回ずつに分けて S1〜S5 の形で(どれだけの 60 回の組が通るか)
        chunks = []
        for i in range(0, len(fr) - 59, 60):
            j = judge(fr[i:i + 60], None, f'F{i}〜')
            chunks.append({'from': i, 'S1': j['S1']['pass'], 'S2': j['S2']['value'], 'S2_pass': j['S2']['pass'], 'S4': j['S4']['pass'], 'S5': j['S5']['value']})
        res['FRESH_chunks60'] = chunks
    fr2 = load('FRESH2')
    if fr2:
        res['FRESH2'] = fresh_summary(fr2, 'speed r2 2 つ目の新しい開始')
        res['FRESH2_integrity'] = integrity(fr2)
        both = fr + fr2
        res['FRESH_ALL'] = fresh_summary(both, 'speed r2 新しい開始(FRESH + FRESH2)')
        res['FRESH_ALL']['bad_any'] = sum(1 for r in both if r.get('collisions') or clr(r) < .03 or stalled(r))
        res['FRESH_ALL']['bad_any_upper95'] = upper95(res['FRESH_ALL']['bad_any'], len(both))
        res['FRESH_ALL']['p_pass60_at_upper95_bad_any'] = p_pass60(res['FRESH_ALL']['bad_any_upper95'])
        chunks = []
        for i in range(0, len(fr2) - 59, 60):
            j = judge(fr2[i:i + 60], None, f'E{i}〜')
            chunks.append({'from': i, 'all': j['S1']['pass'] and j['S2']['pass'] and j['S4']['pass'] and j['S5']['pass'], 'S2': j['S2']['value'], 'S5': j['S5']['value']})
        res['FRESH2_chunks60'] = chunks
    gr = load('GRID')
    if gr:
        res['GRID'] = fresh_summary(gr, '開始の範囲の端と真ん中(33 ゲート × 9)× c_v 3 段')
        res['GRID_integrity'] = integrity(gr)
    ho = []
    for nm, lab in (('HOT_G45', 'ゲート 4・5 のまわり(左右 0.05 m・向き 2.5° おき)× c_v 5 段'), ('HOT_G17', 'ゲート 16〜18 のまわり × c_v 1.0'),
                    ('HOT_G78', 'ゲート 7・8 のまわり × c_v 0.5・0.625'), ('HOT_G45_R1', '参考: speed r1 をゲート 4・5 のまわり × c_v 1.0 で')):
        h = load(nm)
        if h:
            res[nm] = fresh_summary(h, lab)
            res[nm + '_integrity'] = integrity(h)
            if not nm.endswith('_R1'):
                ho += h
    fc = load('FRESH_CAP')
    if fc:
        res['FRESH_CAP'] = fresh_summary(fc, '最速 + 同じ上限 4.5·c_v')
        res['FRESH_CAP_judge'] = judge(fc, None, '最速 + 同じ上限')
        if fr:
            res['pair_r2_vs_cap'] = summarize_pairs(pair(fr, fc))
            res['within'] = {str(cv): {'r2': within_line_spread(fr[:300], cv), 'cap': within_line_spread(fc, cv)} for cv in LEVELS}
            res['line_by_cv'] = {}
            for nm, rows in (('r2', fr[:300]), ('cap', fc)):
                P = {cv: mprof([r for r in rows if r['c_v'] == cv], 'ey') for cv in LEVELS}
                res['line_by_cv'][nm] = {f'{a}-{b}': dline(P[a], P[b]) for a, b in ((1.0, .5), (1.0, .75), (.75, .5))}
    fr1 = load('FRESH_R1')
    if fr1:
        res['FRESH_R1'] = fresh_summary(fr1, 'speed r1 新しい開始(参考)')
        if fr:
            res['pair_r2_vs_r1'] = summarize_pairs(pair(fr, fr1))
    ff = load('FRESH_FAST')
    if ff:
        res['FRESH_FAST'] = fresh_summary(ff, '最速(上限なし)')
        if fr:
            res['pair_r2cv1_vs_fastest'] = summarize_pairs(pair([r for r in fr if r['c_v'] == 1.0], ff))
            fp = mprof(ff, 'ey'); fv = mprof(ff, 'v')
            res['mean_line_vs_fastest'] = {}
            for cv in LEVELS:
                cp = mprof([r for r in fr[:300] if r['c_v'] == cv], 'ey'); cvv = mprof([r for r in fr[:300] if r['c_v'] == cv], 'v')
                dv = [a - b for a, b in zip(cvv, fv) if a is not None and b is not None]
                res['mean_line_vs_fastest'][str(cv)] = {**(dline(cp, fp) or {}), 'speed_diff_mean': sum(dv) / len(dv)}
                if fc:
                    res['mean_line_vs_fastest'][str(cv)]['cap_line_mean_cm'] = (dline(mprof([r for r in fc if r['c_v'] == cv], 'ey'), fp) or {}).get('mean_cm')
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
            rows.append({'start': k, 'laps': ts, 'S3_like': ok3, 'mono5': mono5, 'diff': (t[.5] - t[1.0]) if None not in (t.get(.5), t.get(1.0)) else None})
        res['STDV'] = rows
        res['STDV_min_clear'] = min(r['min_clearance'] for r in sv)
        res['STDV_integrity'] = integrity(sv)
    v67 = load('VAL67')
    if v67:
        res['VAL67'] = [{**brief(r), 'laps': [(round(l['elapsed'], 3), l['gates'], l['crossings']) for l in r['laps']], 'frames': r['frames'], 'passed_gates': r['passed_gates'],
                         'required_crossings': r['required_crossings'], 'own_crossings': r['own_crossings'], 'valid': r['valid_episode']} for r in v67]
    dr = load('DR')
    if dr:
        res['DR'] = fresh_summary(dr, 'DR-T(参考)')
        res['DR']['per_delay'] = {str(dl): {'n': sum(1 for r in dr if r['dr']['delay'] == dl), 'collisions': sum(1 for r in dr if r['dr']['delay'] == dl and r['collisions'])} for dl in (0, 1)}
    sc = load('SEEDCHK')
    if sc:
        res['SEEDCHK'] = {'identical': len(set(json.dumps([[l['elapsed'] for l in r['laps']], r['min_clearance'], r['distance']]) for r in sc)) == 1, 'n': len(sc)}
    allrows = rep + web + r1 + fb + fr + (fr2 or []) + gr + ho + fc + fr1 + ff + sv + v67 + sc
    if allrows:
        res['ALL_integrity_nominal'] = integrity(allrows)
    out = REP / 'check_speed_r2/summary.json'
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=lambda o: None)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: None)[:6000])


if __name__ == '__main__':
    main()
