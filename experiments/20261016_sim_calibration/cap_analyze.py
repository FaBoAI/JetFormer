"""上限の試験(S4)の集計と判定(2026-10-16)。protocol.json cap_test.metrics / decision_rules / sim_real_consistency のとおり。

  python cap_analyze.py   → cap_test/cap_test.json(全部の数字)・fig/cap_*.png

cap_test.mjs が書いた cap_test/runs/<条件>.json を読むだけ。判定の決まりは protocol のまま(結果を見て変えない)。
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

REPORT = '/home/jetson/jetformer_reports/sim_calibration_20261016'
RUNS = os.path.join(REPORT, 'cap_test', 'runs')
OUT = os.path.join(REPORT, 'cap_test', 'cap_test.json')
FIG = os.path.join(REPORT, 'fig')
SEED = 20261016
NBOOT = 2000
Z90 = 1.6448536269514722
DRIVERS = ['PP-1.0', 'PP-1.2', 'PP-1.4', 'RL-fastest', 'RL-speed-0.75']
SECTIONS = ['HS', 'LH', 'R1', 'GM', 'DE', 'SW']
LOOP_L = 26.119648724030924


def wilson(k, n, z=Z90):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_runs():
    out = {}
    for f in sorted(os.listdir(RUNS)):
        if f.endswith('.json'):
            d = json.load(open(os.path.join(RUNS, f)))
            out[d['id']] = d
    return out


def cid(driver, gov, physics='P_cal', brake='nominal', world='W0', kb=1):
    parts = [driver, gov, physics, None if brake == 'nominal' else 'brake-' + brake, None if world == 'W0' else world,
             None if kb == 1 else 'kb' + (str(kb).rstrip('0').rstrip('.') if isinstance(kb, float) else str(kb))]
    return '__'.join(p for p in parts if p)


def per_start(d):
    """開始ごとの数(ブートストラップの単位)"""
    rows = []
    for r in d['runs']:
        p = r['passes']
        sec = Counter()
        if r.get('collision'):
            sec[r['collision']['section']] += 1
        rows.append({
            'start': r['start'], 'laps': r['laps_driven'], 'col': r['collisions'] > 0, 'complete': r['complete'],
            'r1_n': len(p['R1']), 'r1_c': sum(q['collision'] for q in p['R1']), 'r1_fast': sum(q['entry_v'] > 2.5 for q in p['R1']),
            'gm_n': len(p['GM']), 'gm_c': sum(q['collision'] for q in p['GM']),
            'de_n': len(p['DE']), 'de_c': sum(q['collision'] for q in p['DE']),
            'sec': sec,
        })
    return rows


def summarize(d):
    runs = d['runs']
    rows = per_start(d)
    r1 = [q for r in runs for q in r['passes']['R1']]
    gm = [q for r in runs for q in r['passes']['GM']]
    de = [q for r in runs for q in r['passes']['DE']]
    laps_driven = sum(r['laps_driven'] for r in runs)
    ncol = sum(r['collisions'] > 0 for r in runs)
    sec = Counter(r['collision']['section'] for r in runs if r.get('collision'))
    objs = Counter((r['collision']['section'], r['collision']['collider']) for r in runs if r.get('collision'))
    laps_ok = [l['elapsed'] for r in runs for l in r['lap_events'] if l['collisions'] == 0]
    ev = np.array([q['entry_v'] for q in r1]) if r1 else np.array([])
    s = {
        'id': d['id'], 'condition': d['condition'], 'starts': len(runs),
        'complete': sum(r['complete'] for r in runs), 'collisions': ncol, 'laps_driven': round(laps_driven, 3),
        'collisions_per_lap': round(ncol / laps_driven, 4) if laps_driven else None,
        'reasons': dict(Counter(r['reason'] for r in runs)),
        'collisions_by_section': {k: sec.get(k, 0) for k in SECTIONS},
        'collisions_by_section_per_lap': {k: round(sec.get(k, 0) / laps_driven, 4) for k in SECTIONS} if laps_driven else None,
        'collision_objects': [{'section': k[0], 'collider': k[1], 'n': n} for k, n in objs.most_common()],
        'collision_places_xz': [[round(r['collision']['x'], 3), round(r['collision']['z'], 3), r['collision']['section'], round(r['collision']['v'], 3)]
                                for r in runs if r.get('collision')],
        'R1': {'passes': len(r1), 'collisions': sum(q['collision'] for q in r1),
               'rate': round(sum(q['collision'] for q in r1) / len(r1), 4) if r1 else None,
               'rate_wilson90': wilson(sum(q['collision'] for q in r1), len(r1)),
               'entry_v_median': round(float(np.median(ev)), 3) if len(ev) else None,
               'entry_v_p90': round(float(np.percentile(ev, 90)), 3) if len(ev) else None,
               'entry_v_max': round(float(ev.max()), 3) if len(ev) else None,
               'entry_gt_2_5': int((ev > 2.5).sum()) if len(ev) else 0,
               'entry_gt_2_5_frac': round(float((ev > 2.5).mean()), 4) if len(ev) else None,
               'entry_gt_2_75_frac': round(float((ev > 2.75).mean()), 4) if len(ev) else None},
        'GM': {'passes': len(gm), 'collisions': sum(q['collision'] for q in gm),
               'rate': round(sum(q['collision'] for q in gm) / len(gm), 4) if gm else None,
               'rate_wilson90': wilson(sum(q['collision'] for q in gm), len(gm)),
               'start_v_median': round(float(np.median([q['start_v'] for q in gm if q['start_v'] is not None])), 3) if any(q['start_v'] is not None for q in gm) else None},
        'DE': {'passes': len(de), 'collisions': sum(q['collision'] for q in de),
               'rate': round(sum(q['collision'] for q in de) / len(de), 4) if de else None,
               'rate_wilson90': wilson(sum(q['collision'] for q in de), len(de)),
               'max_v_median': round(float(np.median([q['max_v'] for q in de if q['max_v'] is not None])), 3) if any(q['max_v'] is not None for q in de) else None},
        'lap_time': {'n': len(laps_ok), 'median': round(float(np.median(laps_ok)), 3) if laps_ok else None,
                     'mean': round(float(np.mean(laps_ok)), 3) if laps_ok else None},
        'tracker': {'jumps_total': sum(r['tracker']['jumps'] for r in runs), 'max_dist_m': max(r['tracker']['max_dist_m'] for r in runs)},
    }
    if d['condition']['gov'] != 'OFF':
        zones = defaultdict(lambda: Counter())
        inv = Counter()
        for r in runs:
            for z, v in r['governor']['zones'].items():
                zones[z].update({k: v[k] for k in ('frames', 'active', 'cap', 'neutral', 'brake')})
                zones[z]['cut_mps_s'] += v['cut_mps_s']
            inv.update(r['invariants'])
            inv['reverse_frames'] += r['reverse_frames']
        s['governor'] = {z: {'frames': int(v['frames']), 'active_frac': round(v['active'] / v['frames'], 4) if v['frames'] else None,
                             'cap_frames': int(v['cap']), 'neutral_frames': int(v['neutral']), 'brake_frames': int(v['brake']),
                             'cut_mps_s': round(v['cut_mps_s'], 3)} for z, v in sorted(zones.items())}
        s['governor_brake_starts'] = sum(r['governor']['brake_starts'] for r in runs)
        s['invariants'] = dict(inv)
        s['invariants_ok'] = all(inv[k] == 0 for k in ('brake_in_R1turn_GM_DE', 'brake_start_below_1mps', 'brake_neutral_roundtrip_lt_0_2s',
                                                        'u_increased_frames', 'reverse_frames'))
    return s


def boot_paired(on, off, rng, fn):
    """開始(24)単位のブートストラップ。on・off は開始の並びが同じ行のリスト。fn(rows) → 値"""
    n = len(on)
    assert [r['start'] for r in on] == [r['start'] for r in off]
    vals = []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        a, b = fn([on[i] for i in idx]), fn([off[i] for i in idx])
        if a is None or b is None:
            continue
        vals.append(a - b)
    if not vals:
        return None
    return [round(float(np.percentile(vals, 5)), 4), round(float(np.percentile(vals, 95)), 4)]


def ratio(key_c, key_n):
    def f(rows):
        n = sum(r[key_n] for r in rows)
        return None if n == 0 else sum(r[key_c] for r in rows) / n
    return f


def col_per_lap(rows):
    L = sum(r['laps'] for r in rows)
    return None if L == 0 else sum(r['col'] for r in rows) / L


def sec_per_lap(sec):
    def f(rows):
        L = sum(r['laps'] for r in rows)
        return None if L == 0 else sum(r['sec'].get(sec, 0) for r in rows) / L
    return f


def lap_pairs(on, off):
    """同じ開始・同じ周の番号の、衝突の無い周の時間の差(ON − OFF)"""
    lo = {r['start']: {l['lap']: l['elapsed'] for l in r['lap_events'] if l['collisions'] == 0} for r in off['runs']}
    diffs, flying = [], []
    for r in on['runs']:
        for l in r['lap_events']:
            if l['collisions'] == 0 and l['lap'] in lo.get(r['start'], {}):
                d = l['elapsed'] - lo[r['start']][l['lap']]
                diffs.append(d)
                if l['lap'] >= 2:
                    flying.append(d)
    return diffs, flying


def compare(on, off, rng):
    """ON と OFF の比べ(同じ運転手・物理・世界・電池)"""
    A, B = per_start(on), per_start(off)
    so, sf = summarize(on), summarize(off)
    r1_on, r1_off = so['R1']['rate'], sf['R1']['rate']
    rel = None if (r1_off in (None, 0) or r1_on is None) else round(1 - r1_on / r1_off, 4)
    r1_ci = boot_paired(A, B, rng, ratio('r1_c', 'r1_n'))
    cpl_ci = boot_paired(A, B, rng, col_per_lap)
    sec_diff = {}
    for sname in SECTIONS:
        a, b = sec_per_lap(sname)(A), sec_per_lap(sname)(B)
        sec_diff[sname] = None if a is None or b is None else round(a - b, 4)
    diffs, flying = lap_pairs(on, off)
    informative = sf['R1']['passes'] >= 30 and (sf['R1']['entry_gt_2_5_frac'] or 0) >= 0.2
    r1_eff = bool(rel is not None and rel >= 0.5 and r1_ci is not None and r1_ci[1] < 0)
    no_harm = cpl_ci is not None and cpl_ci[1] <= 0.05
    flags = [k for k, v in sec_diff.items() if v is not None and v >= 0.05]
    out = {
        'on': on['id'], 'off': off['id'],
        'informative': informative,
        'informative_detail': {'off_R1_passes': sf['R1']['passes'], 'off_R1_entry_gt_2_5_frac': sf['R1']['entry_gt_2_5_frac']},
        'R1_rate_off': r1_off, 'R1_rate_on': r1_on, 'R1_rel_reduction': rel, 'R1_rate_diff_ci90': r1_ci,
        'R1_effective': r1_eff,
        'GM_rate_off': sf['GM']['rate'], 'GM_rate_on': so['GM']['rate'], 'GM_rate_diff_ci90': boot_paired(A, B, rng, ratio('gm_c', 'gm_n')),
        'DE_rate_off': sf['DE']['rate'], 'DE_rate_on': so['DE']['rate'],
        'col_per_lap_off': sf['collisions_per_lap'], 'col_per_lap_on': so['collisions_per_lap'],
        'col_per_lap_diff_ci90': cpl_ci, 'no_harm': no_harm,
        'section_col_per_lap_diff': sec_diff, 'section_flags_ge_0_05': flags,
        'complete_off': sf['complete'], 'complete_on': so['complete'],
        'lap_diff_n': len(diffs), 'lap_diff_median_s': round(float(np.median(diffs)), 4) if diffs else None,
        'lap_diff_flying_median_s': round(float(np.median(flying)), 4) if flying else None,
        'lap_cost_ok': (float(np.median(diffs)) <= 0.3) if diffs else None,
        'entry_v_median_off': sf['R1']['entry_v_median'], 'entry_v_median_on': so['R1']['entry_v_median'],
        'entry_v_max_on': so['R1']['entry_v_max'],
        'invariants_ok': so.get('invariants_ok'),
    }
    return out


def logistic_v20(passes):
    """R1 の衝突の確率を進入の速さのロジスティック回帰で(Newton 法)。確率 20% の速さ v20"""
    x = np.array([p[0] for p in passes], float)
    y = np.array([p[1] for p in passes], float)
    res = {'n': int(len(x)), 'collisions': int(y.sum()), 'v_range': [round(float(x.min()), 3), round(float(x.max()), 3)] if len(x) else None}
    if len(x) < 5 or y.sum() == 0 or y.sum() == len(y):
        res['v20'] = None
        res['note'] = '衝突が 0(または全部)で当てはまらない'
        return res
    # 完全な分離の確かめ
    if x[y == 1].min() > x[y == 0].max() or x[y == 1].max() < x[y == 0].min():
        res['separated'] = True
    X = np.c_[np.ones_like(x), x]
    b = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    res['b0'], res['b1'] = round(float(b[0]), 4), round(float(b[1]), 4)
    res['v20'] = round(float((math.log(0.2 / 0.8) - b[0]) / b[1]), 3) if abs(b[1]) > 1e-9 else None
    # 速さの帯ごとの実数
    bands = [(0, 2.25), (2.25, 2.5), (2.5, 2.75), (2.75, 3.1), (3.1, 9)]
    res['bands'] = [{'band': f'{a}-{c}', 'n': int(((x >= a) & (x < c)).sum()), 'col': int(y[(x >= a) & (x < c)].sum())} for a, c in bands]
    return res


def main():
    runs = load_runs()
    rng = np.random.default_rng(SEED)
    summaries = {k: summarize(v) for k, v in runs.items()}
    comps = {}

    def cmp(on_id, off_id):
        if on_id in runs and off_id in runs:
            comps[on_id] = compare(runs[on_id], runs[off_id], rng)

    for phys in ('P_cal', 'P_default'):
        for d in DRIVERS:
            for g in ('ON-O25', 'ON-O275', 'ON-G25'):
                cmp(cid(d, g, phys), cid(d, 'OFF', phys))
    for b in ('low', 'high'):
        for d in DRIVERS:
            cmp(cid(d, 'ON-O25', brake=b), cid(d, 'OFF'))
    for d in DRIVERS:
        cmp(cid(d, 'ON-O25', world='W+'), cid(d, 'OFF', world='W+'))
        for kb in (0.8, 1.12):
            cmp(cid(d, 'ON-O25', kb=kb), cid(d, 'OFF', kb=kb))

    # ---- 判定(protocol decision_rules)。主 = P_cal・ON-O25・名目のブレーキ・W0
    def verdict(phys):
        prim = {d: comps.get(cid(d, 'ON-O25', phys)) for d in DRIVERS}
        prim = {d: c for d, c in prim.items() if c}
        inf = [d for d, c in prim.items() if c['informative']]
        eff = {d: prim[d]['R1_effective'] for d in inf}
        no_harm_all = all(c['no_harm'] for c in prim.values())
        lap_ok = {d: prim[d]['lap_cost_ok'] for d in ('PP-1.0', 'PP-1.2') if d in prim}
        inv_ok = all(c['invariants_ok'] for c in prim.values())
        pp12, pp14 = prim.get('PP-1.2'), prim.get('PP-1.4')
        pp_inf = [d for d in ('PP-1.2', 'PP-1.4') if d in inf]
        pp_eff = [d for d in pp_inf if eff[d]]
        brake_low_breaks = []
        for d in ('PP-1.2', 'PP-1.4', 'RL-fastest'):
            cl = comps.get(cid(d, 'ON-O25', brake='low'))
            if cl and d in inf and eff.get(d) and not cl['R1_effective']:
                brake_low_breaks.append(d)
        if not no_harm_all or not inv_ok:
            v = '直す'
        elif not inf:
            v = '効く場面なし'
        elif len(pp_inf) == 2 and len(pp_eff) == 2 and all(lap_ok.values()) and not brake_low_breaks:
            v = 'シミュレーターで支持'
        elif len(pp_eff) == 1 or brake_low_breaks:
            v = '支持(条件つき)'
        else:
            v = 'どの区分にも当たらない(支持の条件を満たさない)'
        return {'physics': phys, 'verdict': v, 'informative_drivers': inf, 'R1_effective': eff, 'PP_informative': pp_inf, 'PP_effective': pp_eff,
                'no_harm_all': no_harm_all, 'no_harm': {d: c['no_harm'] for d, c in prim.items()},
                'section_flags': {d: c['section_flags_ge_0_05'] for d, c in prim.items() if c['section_flags_ge_0_05']},
                'lap_cost': {d: {'median_s': prim[d]['lap_diff_median_s'], 'ok': prim[d]['lap_cost_ok']} for d in prim},
                'lap_cost_ok_PP10_PP12': all(lap_ok.values()) if lap_ok else None,
                'invariants_ok': inv_ok, 'brake_low_breaks': brake_low_breaks}

    verdicts = {p: verdict(p) for p in ('P_cal', 'P_default')}
    # R1 2.75 と 2.5
    r275 = {}
    for phys in ('P_cal', 'P_default'):
        for d in DRIVERS:
            a, b = comps.get(cid(d, 'ON-O25', phys)), comps.get(cid(d, 'ON-O275', phys))
            if a and b:
                red25 = None if a['R1_rate_off'] in (None, 0) else a['R1_rate_off'] - a['R1_rate_on']
                red275 = None if b['R1_rate_off'] in (None, 0) else b['R1_rate_off'] - b['R1_rate_on']
                r275[f'{phys}/{d}'] = {'R1_reduction_2_5': red25, 'R1_reduction_2_75': red275,
                                      'share_of_2_5': None if not red25 else round(red275 / red25, 3),
                                      'lap_diff_median_2_5': a['lap_diff_median_s'], 'lap_diff_median_2_75': b['lap_diff_median_s'],
                                      'col_per_lap_on_2_5': a['col_per_lap_on'], 'col_per_lap_on_2_75': b['col_per_lap_on']}
    # シミュレーターと実車の突き合わせ(OFF の走行、PP と RL を合わせる)
    consistency = {}
    for phys in ('P_cal', 'P_default'):
        passes, objs = [], Counter()
        for d in DRIVERS:
            r = runs.get(cid(d, 'OFF', phys))
            if not r:
                continue
            for run in r['runs']:
                for q in run['passes']['R1']:
                    passes.append((q['entry_v'], int(q['collision'])))
                c = run.get('collision')
                if c and c['section'] == 'R1':
                    objs[c['collider']] += 1
        res = logistic_v20(passes)
        res['R1_collision_objects'] = dict(objs)
        v20 = res.get('v20')
        res['rule'] = 'v20 が 2.4〜3.3 m/s なら「R1 の物理は実車と矛盾しない」'
        res['consistent'] = None if v20 is None else (2.4 <= v20 <= 3.3)
        consistency[phys] = res
    out = {'schema': 'jf-cap-test-summary/v1', 'protocol': 'protocol.json cap_test(sha256 は protocol.sha256)', 'bootstrap': {'n': NBOOT, 'seed': SEED, 'unit': 'start(24)'},
           'verdicts': verdicts, 'comparisons': comps, 'r1_275_vs_25': r275, 'sim_real_consistency': consistency,
           'real_reference': {'AI': {'lt_2_75': '1/21', '2_75_to_3_1': '2/7', 'ge_3_1': '6/8'}, 'human': {'lt_2_75': '1/53', 'ge_3_1': '5/12'},
                              'DE_human': {'lt_3_0': '1/23', 'ge_3_0': '10/37'}},
           'summaries': summaries}
    with open(OUT + '.tmp', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(OUT + '.tmp', OUT)
    print(json.dumps(verdicts, ensure_ascii=False, indent=1))
    if '--fig' in sys.argv:
        figures(runs, summaries, comps)


def figures(runs, summaries, comps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    os.makedirs(FIG, exist_ok=True)
    # 図 1: R1 の進入の速さ(上限なし・あり)と R1 の衝突。物理 2 つ × 運転手 5
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True)
    bins = np.arange(1.5, 3.31, 0.05)
    for i, phys in enumerate(('P_cal', 'P_default')):
        for j, d in enumerate(DRIVERS):
            ax = axes[i, j]
            for g, col in (('OFF', '#c0392b'), ('ON-O25', '#2471a3')):
                r = runs.get(cid(d, g, phys))
                if not r:
                    continue
                ev = [q['entry_v'] for run in r['runs'] for q in run['passes']['R1']]
                evc = [q['entry_v'] for run in r['runs'] for q in run['passes']['R1'] if q['collision']]
                lab = ('上限なし' if g == 'OFF' else '上限あり') + f' n={len(ev)}・衝突 {len(evc)}'
                ax.hist(ev, bins=bins, alpha=.55, color=col, label=lab)
                if evc:
                    ax.hist(evc, bins=bins, histtype='step', color='k', lw=1.4, hatch='//')
            ax.axvline(2.5, color='k', ls='--', lw=.8)
            ax.set_title(f'{d}({"較正後" if phys == "P_cal" else "較正前"})', fontsize=10)
            ax.legend(fontsize=7)
            if i == 1:
                ax.set_xlabel('R1 の進入の速さ [m/s]')
    fig.suptitle('R1 の進入の速さ(R1 の始まりの手前 0.4 秒の最大)。黒の斜線 = R1 で衝突した通過。点線 = 上限 2.5 m/s', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'cap_r1_entry.png'), dpi=110)
    plt.close(fig)
    # 図 2: 衝突/周(区間ごと)と完走(主の条件)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for k, phys in enumerate(('P_cal', 'P_default')):
        ax = axes[k]
        x = np.arange(len(DRIVERS))
        bottom_off = np.zeros(len(DRIVERS))
        bottom_on = np.zeros(len(DRIVERS))
        colors = {'HS': '#7f8c8d', 'LH': '#8e44ad', 'R1': '#c0392b', 'GM': '#27ae60', 'DE': '#e67e22', 'SW': '#2c3e50'}
        for sname in SECTIONS:
            off = [summaries.get(cid(d, 'OFF', phys), {}).get('collisions_by_section_per_lap', {}) or {} for d in DRIVERS]
            on = [summaries.get(cid(d, 'ON-O25', phys), {}).get('collisions_by_section_per_lap', {}) or {} for d in DRIVERS]
            vo = np.array([o.get(sname, 0) for o in off])
            vn = np.array([o.get(sname, 0) for o in on])
            ax.bar(x - .2, vo, .38, bottom=bottom_off, color=colors[sname], label=sname)
            ax.bar(x + .2, vn, .38, bottom=bottom_on, color=colors[sname], hatch='..', edgecolor='white')
            bottom_off += vo
            bottom_on += vn
        for i, d in enumerate(DRIVERS):
            so, sn = summaries.get(cid(d, 'OFF', phys)), summaries.get(cid(d, 'ON-O25', phys))
            if so and sn:
                ax.text(i - .2, bottom_off[i] + .005, f"完走\n{so['complete']}/24", ha='center', fontsize=7)
                ax.text(i + .2, bottom_on[i] + .005, f"完走\n{sn['complete']}/24", ha='center', fontsize=7)
        ax.set_ylim(0, max(bottom_off.max(), bottom_on.max(), .05) * 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels([d + '\n左: なし 右: あり' for d in DRIVERS], fontsize=8)
        ax.set_ylabel('衝突 / 走った周')
        ax.set_title(f'{"較正後 P_cal" if phys == "P_cal" else "較正前 P_default"}: 区間ごとの衝突/周(左 = 上限なし、右 = 上限あり)', fontsize=10)
        ax.legend(fontsize=8, ncol=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'cap_collisions_sections.png'), dpi=110)
    plt.close(fig)
    # 図 3: コースの上の衝突の場所(較正後、上限なし・あり)と、RL-speed-0.75 の開始 0 の軌跡(色 = 速さ)
    world = json.load(open('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/sim/world-v3.json'))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    for k, g in enumerate(('OFF', 'ON-O25')):
        ax = axes[k]
        for sg in world['segments']:
            ax.plot([sg['a'][0], sg['b'][0]], [sg['a'][1], sg['b'][1]], color='#555', lw=1.2)
        for dk in world['disks']:
            if dk['kind'] == 'post':
                ax.plot(dk['x'], dk['z'], 'o', color='#555', ms=2)
        for sf in world['surfaces']:
            if sf['id'] != 'carpet':
                ax.add_patch(plt.Rectangle((sf['x'] - sf['w'] / 2, sf['z'] - sf['d'] / 2), sf['w'], sf['d'], color={'turf': '#9fd39f', 'ptfe': '#e8e8ff', 'shortcut': '#f5d7a3', 'blue-mat': '#a9c9ef'}.get(sf['id'], '#eee'), alpha=.6, zorder=0))
        r = runs.get(cid('RL-speed-0.75', g))
        if r:
            tr = next((x['trace'] for x in r['runs'] if x['start'] == 0 and 'trace' in x), None)
            if tr:
                sc = ax.scatter(tr['x'], tr['z'], c=tr['v'][:len(tr['x'])], cmap='viridis', vmin=0, vmax=4, s=3)
        marks = {'PP-1.0': 'o', 'PP-1.2': 's', 'PP-1.4': '^', 'RL-fastest': 'X', 'RL-speed-0.75': 'D'}
        for d in DRIVERS:
            sm = summaries.get(cid(d, g))
            if sm and sm['collision_places_xz']:
                xs = [c[0] for c in sm['collision_places_xz']]
                zs = [c[1] for c in sm['collision_places_xz']]
                ax.scatter(xs, zs, marker=marks[d], facecolor='none', edgecolor='#c0392b', s=60, lw=1.3, label=f'{d} の衝突 {len(xs)}')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'較正後 P_cal・{"上限なし" if g == "OFF" else "上限あり"}: 衝突の場所(赤)と\nRL-speed-0.75 の開始 0 の軌跡(色 = 速さ)', fontsize=10)
        ax.legend(fontsize=7, loc='lower left')
    fig.colorbar(sc, ax=axes, shrink=.7, label='速さ [m/s]')
    fig.savefig(os.path.join(FIG, 'cap_map_collisions.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
