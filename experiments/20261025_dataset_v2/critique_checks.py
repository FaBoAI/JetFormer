"""設計 A・B の批評のための数え直し(2026-10-25)。読むだけ。

numpy も torch も使わない(json だけ)。data/ は読まない。読むのは
~/jetformer_reports/dataset_v2_plan_20261025/ の inventory.json・design_numbers.json・
train_seed_spread.json・train_budget.json だけ。出力は同じ場所の critique_checks.json。

動かし方:
  /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I \
    experiments/20261025_dataset_v2/critique_checks.py
"""
import json
import math
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025')


def load(name):
    with open(REP / name, encoding='utf-8') as f:
        return json.load(f)


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)), m


def three_lap_windows(inv):
    """AI の続きの 3 周(途切れなし・7 秒未満の疑いの周なし)の和"""
    out = []
    for r in inv['records']:
        ll = [x for x in (r.get('lap_list') or []) if x['who'] == 'ai']
        for i in range(len(ll) - 2):
            a, b, c = ll[i:i + 3]
            if abs(a['t_start'] + a['lap_s'] - b['t_start']) > 0.3:
                continue
            if abs(b['t_start'] + b['lap_s'] - c['t_start']) > 0.3:
                continue
            if any(x['suspect'] for x in (a, b, c)):
                continue
            out.append({'sum_s': round(a['lap_s'] + b['lap_s'] + c['lap_s'], 2), 'log': r['log'],
                        'session': r.get('session'), 'model': a['model'], 'throttle_gain': a['throttle_gain'],
                        'clean_laps': sum(bool(x['clean']) for x in (a, b, c)),
                        'laps': [x['lap_s'] for x in (a, b, c)]})
    out.sort(key=lambda w: w['sum_s'])
    return out


def main():
    inv = load('inventory.json')
    dn = load('design_numbers.json')
    ss = load('train_seed_spread.json')
    tb = load('train_budget.json')
    res = {'schema': 'jf-dataset-v2-critique-checks/v1', 'tool': 'experiments/20261025_dataset_v2/critique_checks.py'}

    # 1. 3 周の窓
    w = three_lap_windows(inv)
    clean3 = [x for x in w if x['clean_laps'] == 3]
    res['three_lap'] = {
        'best': w[:8],
        'best_all_clean': clean3[:3],
        'n_le_25p6': sum(1 for x in w if x['sum_s'] <= 25.6),
        'logs_le_26': sorted({x['log'] for x in w if x['sum_s'] <= 26.0}),
    }

    # 2. 種のばらつきと閾値の強さ
    fam = ['A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B']
    pr = ss['per_run']
    st = {}
    for which in ('best', 'last'):
        xs = [pr[k][which]['l1'] for k in fam]
        s, m = sd(xs)
        se_2v5 = s * math.sqrt(1 / 2 + 1 / 5)
        rng = max(xs) - min(xs)
        st[which] = {'mean': round(m, 3), 'sd': round(s, 3), 'range': round(rng, 3), 'thr_2x_range': round(2 * rng, 3),
                     'se_diff_2_vs_5': round(se_2v5, 3), 'thr_in_se': round(2 * rng / se_2v5, 2),
                     'min': round(min(xs), 3), 'max': round(max(xs), 3)}
    rec = dn['recipes']['recipes']
    v2m = rec['主: 人 + AI の B <9.0(重み 1)+ 9.0〜9.5(重み 0.5)']['l1_if_ai_counts_full']['range']
    v2h = rec['人だけ(封をしたうえで)']['l1_if_ai_counts_0']['range']
    st['expected_gain_vs_family_mean'] = {
        'v2-H': [round(st['best']['mean'] - v2h[1], 2), round(st['best']['mean'] - v2h[0], 2)],
        'v2-M(AI を人と同じに数える)': [round(st['best']['mean'] - v2m[1], 2), round(st['best']['mean'] - v2m[0], 2)],
    }
    # 良い・悪いに要る平均
    st['good_needs_mean_below'] = round(st['best']['mean'] - st['best']['thr_2x_range'], 2)
    st['bad_needs_mean_above'] = round(st['best']['mean'] + st['best']['thr_2x_range'], 2)
    st['steer_bias_veto_halfwidth'] = ss['families']['v0(B と同じ作り方)']['best']['steer_bias']['thr_2x_range']
    st['r1_app_seed_range'] = ss['families']['v0(B と同じ作り方)']['best']['r1_app']['range']
    res['seed_power'] = st

    # 3. 人の走行の速さ(直線)。10/23〜24 の人は置き直しが多い
    eff = dn['effectiveness']['straight_by_session_who_model_gain']
    res['human_straight'] = {k: {'n': v['n'], 'u_p50': v['u_p50'], 'erpm_p50': v['erpm_p50'], 'erpm_p90': v['erpm_p90']}
                             for k, v in eff.items() if '|human|' in k}

    # 4. ジャンル 9 の正解のスロットル(B の直線、車に出た値)と行き着く速さ
    keep = ['10/23 1 回目(20:37〜)|ai|B|1.0', '10/24 1 回目(13:46〜)|ai|B|1.0', '10/24 2 回目(16:01〜)|ai|B|0.9',
            '10/24 3 回目(18:31〜)|ai|B|0.9']
    dyn = dn['effectiveness']['dyn_ai_by_session_model_gain']
    g9 = {}
    for k in keep:
        v = eff.get(k)
        dk = k.replace('|ai|', '|')
        d = dyn.get(dk, {})
        g9[k] = {'u_p50': v and v['u_p50'], 'erpm_p90': v and v['erpm_p90'], 'e_ss_u250': d.get('e_ss_u250')}
    res['genre9_label_throttle'] = g9

    # 5. 2 本の平均の実車の記録
    ens = {'records': [], 'ai_s': 0.0, 'collisions_ai': 0, 'laps_ai': 0, 'clean_ai': 0}
    for r in inv['records']:
        if r.get('models') and any('2本の平均' in str(m) for m in (r['models'] if isinstance(r['models'], list) else [r['models']])):
            ca = r['collisions']['ai'] if isinstance(r.get('collisions'), dict) else None
            la = r['laps']['ai'] if isinstance(r.get('laps'), dict) else None
            ens['records'].append({'log': r['log'], 'ai_no': r.get('ai_no'), 'ai_s': r.get('ai_s'), 'collisions': r.get('collisions'),
                                   'laps': r.get('laps'), 'straight_erpm_p90': r.get('straight_erpm_p90')})
            ens['ai_s'] += float(r.get('ai_s') or 0)
            if isinstance(ca, (int, float)):
                ens['collisions_ai'] += ca
    res['ens_on_car'] = ens

    # 6. 走行中の推論の倍率
    dt = tb['drive_timing']
    res['drive_timing'] = {'infer_p50': dt['infer_ms']['p50'], 'infer_p99': dt['infer_ms']['p99'], 'engine_B_ms': 2.34,
                           'ratio_p50': round(dt['infer_ms']['p50'] / 2.34, 2), 'ratio_p99': round(dt['infer_ms']['p99'] / 2.34, 2),
                           'latency_p99': dt['latency_ms']['p99']}

    # 7. v2-M のサンプル数の数え直し(新しい人は (a) で落とす → 1.046 倍しない)
    hp = dn['recipes']['human_new_pool_total']
    res['v2m_count'] = {'design': 8879, 'if_drop_mask_keeps_only_a': 5912 + hp + 1195,
                        'ai_share_design': round(1195 / 8879, 4), 'ai_share_recount': round(1195 / (5912 + hp + 1195), 4)}

    out = REP / 'critique_checks.json'
    tmp = str(out) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    Path(tmp).replace(out)
    print(json.dumps({k: res[k] for k in ('three_lap', 'seed_power', 'v2m_count', 'drive_timing')}, ensure_ascii=False)[:4000])
    print(json.dumps(res['human_straight'], ensure_ascii=False)[:2500])
    print(json.dumps(res['genre9_label_throttle'], ensure_ascii=False))
    print(json.dumps(res['ens_on_car'], ensure_ascii=False)[:1500])


if __name__ == '__main__':
    main()
