#!/usr/bin/env python3
"""lap_quality.json → explore_data.md の表の数値(読み取りのみ、CPU)

    python summarize.py lap_quality.json > summary.json
"""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'lap_quality.json'))
L = r['logs']
AFT = [k for k, v in L.items() if v['kind'] == 'human' and v['group'] == 'afternoon']
EVE = [k for k, v in L.items() if v['kind'] == 'human' and v['group'] == 'evening']
HUM = AFT + EVE
AI67 = ['20261010-172910', '20261010-173445']
AIALL = [k for k, v in L.items() if v['kind'] == 'ai']
out = {}


def q(x, p):
    return round(float(np.percentile(x, p)), 2) if len(x) else None


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {'n': 0}
    return {'n': len(xs), 'median': q(xs, 50), 'p25': q(xs, 25), 'p75': q(xs, 75), 'min': round(min(xs), 2), 'max': round(max(xs), 2)}


def laps(logs, ai=False, valid=True):
    res = []
    for lg in logs:
        for x in L[lg]['laps']:
            if valid and not x['valid']:
                continue
            if ai and not x.get('ai_only'):
                continue
            y = dict(x)
            y['log'] = lg
            y['genre'] = L[lg]['genre']
            res.append(y)
    return res


# 1. 周
groups = {'human_afternoon': laps(AFT), 'human_evening': laps(EVE), 'human_all': laps(HUM),
          'ai_006_007': laps(AI67, ai=True), 'ai_all': laps(AIALL, ai=True)}
out['laps'] = {}
for g, ls in groups.items():
    clean = [x for x in ls if x['impacts'] == 0]
    d = {'valid_laps': len(ls), 'impacts': sum(x['impacts'] for x in ls),
         'impacts_per_lap': round(sum(x['impacts'] for x in ls) / len(ls), 2) if ls else None,
         'clean_laps': len(clean), 'time_all': stats([x['time_s'] for x in ls]), 'time_clean': stats([x['time_s'] for x in clean])}
    for nm in ('R1GM', 'DESW', 'HSLH'):
        d['sector_' + nm + '_clean'] = stats([x['sector_s'][nm] for x in clean if x.get('sector_s')])
        d['erpm_' + nm + '_clean'] = stats([x['sector_erpm'][nm] for x in clean if x.get('sector_erpm')])
    d['erpm_p90_clean'] = stats([x.get('erpm_p90') for x in clean])
    d['dist_m_clean'] = stats([x.get('dist_m') for x in clean])
    d['v_in_clean'] = stats([x['v_in'] for x in clean])
    out['laps'][g] = d
# 速い人の周(衝突なし、上位)
hc = sorted([x for x in groups['human_all'] if x['impacts'] == 0], key=lambda x: x['time_s'])
out['human_clean_laps_sorted'] = [{'log': x['log'][-6:], 'time_s': x['time_s'], 'v_in': round(x['v_in']) if x['v_in'] else None,
                                   'de_psi': round(x['de_psi']) if x['de_psi'] is not None else None, 'sector_s': x.get('sector_s'),
                                   'sector_erpm': x.get('sector_erpm'), 'erpm_p90': round(x['erpm_p90']) if x.get('erpm_p90') else None,
                                   'genre': x['genre'], 'split': x['split']} for x in hc]
fast = [x for x in hc if x['time_s'] < 9.3]
out['human_fast_clean'] = {'threshold_s': 9.3, 'n': len(fast), 'time': stats([x['time_s'] for x in fast])}
for nm in ('R1GM', 'DESW', 'HSLH'):
    out['human_fast_clean']['sector_' + nm] = stats([x['sector_s'][nm] for x in fast if x.get('sector_s')])
    out['human_fast_clean']['erpm_' + nm] = stats([x['sector_erpm'][nm] for x in fast if x.get('sector_erpm')])

# 2. ラップと衝突の関係(人の有効な周)
hl = groups['human_all']
bins = [(0, 9.0), (9.0, 10.0), (10.0, 99)]
out['human_time_vs_impacts'] = []
for a, b in bins:
    s = [x for x in hl if a <= x['time_s'] < b]
    out['human_time_vs_impacts'].append({'range_s': [a, b], 'laps': len(s), 'impacts': sum(x['impacts'] for x in s),
                                         'clean': sum(1 for x in s if x['impacts'] == 0)})
def _rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind='mergesort')
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    for v in np.unique(a):                 # 同順位は平均順位
        m = a == v
        r[m] = r[m].mean()
    return r


def spearmanr(x, y, n_perm=20000, seed=0):
    """Spearman の順位相関と、並べ替え検定の両側 p 値(scipy が無いので自前)"""
    rx, ry = _rank(x), _rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1)


if len(hl) > 3:
    rho, p = spearmanr([x['time_s'] for x in hl], [x['impacts'] for x in hl])
    out['human_time_vs_impacts_spearman'] = {'rho': round(float(rho), 2), 'p': round(float(p), 3), 'n': len(hl)}
    s = [x for x in hl if x['impacts'] == 0 and x['v_in'] is not None]
    rho, p = spearmanr([x['v_in'] for x in s], [x['time_s'] for x in s])
    out['human_clean_vin_vs_time_spearman'] = {'rho': round(float(rho), 2), 'p': round(float(p), 3), 'n': len(s)}
    s2 = [x for x in s if x.get('sector_s')]
    rho, p = spearmanr([x['v_in'] for x in s2], [x['sector_s']['R1GM'] for x in s2])
    out['human_clean_vin_vs_R1GM_spearman'] = {'rho': round(float(rho), 2), 'p': round(float(p), 3), 'n': len(s2)}
    s3 = [x for x in hl if x.get('sector_s')]
    for nm in ('R1GM', 'DESW', 'HSLH'):
        rho, p = spearmanr([x['sector_s'][nm] for x in s3], [x['time_s'] for x in s3])
        out['human_sector_vs_time_spearman_' + nm] = {'rho': round(float(rho), 2), 'p': round(float(p), 3), 'n': len(s3)}
    a3 = [x for x in groups['ai_006_007'] if x.get('sector_s')]
    for nm in ('R1GM', 'DESW', 'HSLH'):
        rho, p = spearmanr([x['sector_s'][nm] for x in a3], [x['time_s'] for x in a3])
        out['ai_sector_vs_time_spearman_' + nm] = {'rho': round(float(rho), 2), 'p': round(float(p), 3), 'n': len(a3)}


# 3. R1 の進入速度と衝突(通過ごと。周が有効でなくてもよい)
def passes(logs, ai_side=None, key='ai'):
    res = []
    for lg in logs:
        for P in L[lg]['passes']:
            if P['v_in'] is None:
                continue
            if ai_side is True and not P.get(key):
                continue
            y = dict(P)
            y['log'] = lg
            y['genre'] = L[lg]['genre']
            res.append(y)
    return res


def r1_table(ps):
    rows = []
    for a, b in [(0, 5500), (5500, 6200), (6200, 99999)]:
        s = [p for p in ps if a <= p['v_in'] < b]
        rows.append({'v_in': [a, b], 'n': len(s), 'hit': sum(1 for p in s if p['r1_hits']),
                     'strong': sum(1 for p in s if any(g >= 4 for g in p['r1_hits']))})
    return rows


out['r1'] = {'human_afternoon': r1_table(passes(AFT)), 'human_evening': r1_table(passes(EVE)),
             'human_all': r1_table(passes(HUM)), 'ai_006_007': r1_table(passes(AI67, True)), 'ai_all': r1_table(passes(AIALL, True))}
for g, ps in {'human_afternoon': passes(AFT), 'human_evening': passes(EVE), 'ai_006_007': passes(AI67, True)}.items():
    v = [p['v_in'] for p in ps]
    out['r1'][g + '_vin'] = {'n': len(v), 'median': q(v, 50), 'p90': q(v, 90), 'frac_ge_6200': round(np.mean(np.array(v) >= 6200), 2),
                             'frac_ge_5500': round(np.mean(np.array(v) >= 5500), 2)}
# clean_driving に入っている通過だけ(ジャンル別)
cd = []
for lg in HUM:
    sc = {x['a']: x for x in L[lg]['laps']}
    for P in L[lg]['passes']:
        if P['v_in'] is None:
            continue
        cd.append((lg, P))
out['r1']['by_log'] = {}
for lg in HUM:
    v = [P['v_in'] for P in L[lg]['passes'] if P['v_in'] is not None]
    h = [P for P in L[lg]['passes'] if P['v_in'] is not None and P['r1_hits']]
    out['r1']['by_log'][lg[-6:]] = {'genre': L[lg]['genre'], 'n': len(v), 'median': q(v, 50), 'ge6200': int(np.sum(np.array(v) >= 6200)),
                                    'ge5500': int(np.sum(np.array(v) >= 5500)), 'hits': len(h)}


# 4. DE の向きと衝突
def de_table(ps):
    rows = []
    for a, b in [(-999, -10), (-10, 0), (0, 15), (15, 999)]:
        s = [p for p in ps if p.get('de_psi') is not None and a <= p['de_psi'] < b]
        rows.append({'psi': [a, b], 'n': len(s), 'hit': sum(1 for p in s if p.get('de_hits')),
                     'strong': sum(1 for p in s if any(g >= 4 for g in p.get('de_hits') or []))})
    return rows


out['de'] = {'human_afternoon': de_table(passes(AFT)), 'human_evening': de_table(passes(EVE)), 'human_all': de_table(passes(HUM)),
             'ai_006_007': de_table(passes(AI67, True, 'ai_de')), 'ai_all': de_table(passes(AIALL, True, 'ai_de'))}
DEG = {'human_afternoon': passes(AFT), 'human_evening': passes(EVE), 'ai_006_007': passes(AI67, True, 'ai_de')}
for g, ps in DEG.items():
    ps = [p for p in ps if p.get('de_psi') is not None]
    v = [p['de_psi'] for p in ps]
    out['de'][g + '_psi'] = {'n': len(v), 'median': q(v, 50), 'p25': q(v, 25), 'p75': q(v, 75), 'frac_ge0': round(np.mean(np.array(v) >= 0), 2),
                             'hit': sum(1 for p in ps if p.get('de_hits')), 'strong': sum(1 for p in ps if any(g_ >= 4 for g_ in p.get('de_hits') or []))}
    for k in ('de_psi12', 'de_yaw', 'de_psi_min', 'de_v', 'de_vmax'):
        hit = [p[k] for p in ps if p.get(k) is not None and p.get('de_hits')]
        ok = [p[k] for p in ps if p.get(k) is not None and not p.get('de_hits')]
        out['de'][g + '_' + k] = {'all': stats([p[k] for p in ps if p.get(k) is not None]), 'hit': stats(hit), 'no_hit': stats(ok)}
# DE の衝突率を、tg+1.2秒の向き(壁へ向かう量)で分ける
def de_table12(ps):
    rows = []
    for a, b in [(-999, -20), (-20, -5), (-5, 10), (10, 999)]:
        s_ = [p for p in ps if p.get('de_psi12') is not None and a <= p['de_psi12'] < b]
        rows.append({'psi12': [a, b], 'n': len(s_), 'hit': sum(1 for p in s_ if p.get('de_hits'))})
    return rows
out['de']['by_psi12'] = {g: de_table12(ps) for g, ps in DEG.items()}
def de_table_v(ps):
    rows = []
    for a, b in [(0, 5000), (5000, 6000), (6000, 99999)]:
        s_ = [p for p in ps if p.get('de_vmax') is not None and a <= p['de_vmax'] < b]
        rows.append({'de_vmax': [a, b], 'n': len(s_), 'hit': sum(1 for p in s_ if p.get('de_hits'))})
    return rows
out['de']['by_vmax'] = {g: de_table_v(ps) for g, ps in DEG.items()}

# 5. 合格基準(roadmap 3.3)
crit = []
for x in groups['human_all']:
    thr = 6200 if x['genre'] == 4 else 5500
    ok_imp = x['impacts'] == 0
    ok_r1 = x['v_in'] is not None and x['v_in'] < thr
    ok_de = x['de_psi'] is not None and x['de_psi'] >= 0
    crit.append((x, ok_imp, ok_r1, ok_de))
out['quality_gate'] = {'valid_laps': len(crit), 'no_impact': sum(c[1] for c in crit), 'r1_ok': sum(c[2] for c in crit),
                       'de_ok': sum(c[3] for c in crit), 'all_three': sum(c[1] and c[2] and c[3] for c in crit),
                       'all_three_times': sorted([c[0]['time_s'] for c in crit if c[1] and c[2] and c[3]]),
                       'no_impact_and_r1': sum(c[1] and c[2] for c in crit)}

# 6. フレームの内訳(動いているカメラフレーム)
fb = {'R1': 0, 'GM': 0, 'DE': 0, 'other': 0}
fb_cd = {'R1': 0, 'GM': 0, 'DE': 0, 'other': 0}
for lg in HUM:
    for z in fb:
        fb[z] += L[lg]['frames'][z]
        fb_cd[z] += L[lg]['frames']['cd'][z]
tot = sum(fb.values())
totcd = sum(fb_cd.values())
out['frames_human'] = {z: {'n': fb[z], 'share': round(fb[z] / tot, 3)} for z in fb}
out['frames_clean_driving'] = {z: {'n': fb_cd[z], 'share': round(fb_cd[z] / totcd, 3)} for z in fb_cd}
out['passes_count'] = {'human_all': len(passes(HUM)), 'human_clean_driving_scenes': None}

json.dump(out, sys.stdout, ensure_ascii=False, indent=1)

# 7. R1 の進入速度と直線の速さの関係(人の衝突なしの周)、合成ラップ
s = [x for x in groups['human_all'] if x['impacts'] == 0 and x.get('sector_erpm') and x['v_in'] is not None]
if len(s) > 3:
    for nm in ('DESW', 'HSLH'):
        rho, p = spearmanr([x['v_in'] for x in s], [x['sector_erpm'][nm] for x in s])
        out['human_clean_vin_vs_erpm_' + nm] = {'rho': round(rho, 2), 'p': round(p, 3), 'n': len(s)}
slow_in = [x for x in s if x['v_in'] < 5500]
fast_in = [x for x in s if x['v_in'] >= 6200]
out['human_clean_slow_vs_fast_entry'] = {
    'slow_in(<5500)': {'n': len(slow_in), 'time': stats([x['time_s'] for x in slow_in]),
                       **{nm: stats([x['sector_s'][nm] for x in slow_in]) for nm in ('R1GM', 'DESW', 'HSLH')}},
    'fast_in(>=6200)': {'n': len(fast_in), 'time': stats([x['time_s'] for x in fast_in]),
                        **{nm: stats([x['sector_s'][nm] for x in fast_in]) for nm in ('R1GM', 'DESW', 'HSLH')}}}
# 合成: R1+GM は R1 を遅く入った衝突なしの周の中央値、DE+SW と HS+LH は衝突なしの周の速いほう 1/4(25パーセンタイル)
comp = {'R1GM': q([x['sector_s']['R1GM'] for x in slow_in], 50),
        'DESW': q([x['sector_s']['DESW'] for x in s], 25), 'HSLH': q([x['sector_s']['HSLH'] for x in s], 25)}
comp['total'] = round(sum(comp.values()), 2)
out['composite_lap_s'] = comp
ai = [x for x in groups['ai_006_007'] if x['impacts'] == 0 and x.get('sector_s')]
out['ai_clean_sectors'] = {nm: q([x['sector_s'][nm] for x in ai], 50) for nm in ('R1GM', 'DESW', 'HSLH')}
out['ai_clean_sectors']['n'] = len(ai)
json.dump({k: out[k] for k in ('human_clean_vin_vs_erpm_DESW', 'human_clean_vin_vs_erpm_HSLH', 'human_clean_slow_vs_fast_entry',
                               'composite_lap_s', 'ai_clean_sectors')}, open('summary_extra.json', 'w'), ensure_ascii=False, indent=1)
