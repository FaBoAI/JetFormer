#!/usr/bin/env python3
"""tools/car_check.py を 10/23・10/24 の AI の記録に当てて、10/24 の台本の数(cond_metrics.json・steer_offset.json・
r1_bump.json)と比べる(段 tools、spec 6.3 T-4)。修理の前後の差(左右の比・s0)が出るかも見る。読むだけ。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I car_check_1024.py
出力: /home/jetson/jetformer_reports/speed_control_20261025/car_check_1024.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import car_check as cc  # noqa: E402
import section_scorecard as ss  # noqa: E402

REP = Path('/home/jetson/jetformer_reports/bench_20261024_report')
COND = REP / 'cond_data' / 'cond_metrics.json'
S0 = Path('/home/jetson/jetformer/experiments/20261024_report/col/steer_offset.json')
BUMP = REP / 'fix_data' / 'r1_bump.json'
OUT = Path('/home/jetson/jetformer_reports/speed_control_20261025/car_check_1024.json')
LAB = Path('/home/jetson/jetformer/data/labels')


def main():
    cond = {r['log']: r for r in json.loads(COND.read_text(encoding='utf-8'))['runs']}
    s0 = {r['log']: r for r in json.loads(S0.read_text(encoding='utf-8'))}
    bump = json.loads(BUMP.read_text(encoding='utf-8'))
    bmax = {}
    for p in bump['passes']:
        bmax[p['log']] = max(bmax.get(p['log'], 0.0), p['a3_max'])
    brun = {r['log']: r for r in bump['runs']}
    logs = sorted(p.stem for p in LAB.glob('2026102[34]-*.json') if json.loads(p.read_text(encoding='utf-8')).get('ai_log'))
    rows = []
    for lg in logs:
        r = cc.check_log(lg)
        if r.get('error'):
            rows.append({'log': lg, 'error': r['error']})
            continue
        c, s, b = cond.get(lg), s0.get(lg), brun.get(lg)
        ref_ratio = None
        if c and c.get('turn_gain_left') and c.get('turn_gain_right'):
            ref_ratio = c['turn_gain_left'] / c['turn_gain_right']
        row = {'log': lg, 'no': r.get('no'), 'session': (c or {}).get('session'), 'seconds': r['seconds'],
               'ratio': r['turn']['ratio'], 'left': r['turn']['left'], 'right': r['turn']['right'],
               'n_left': r['turn']['n_left'], 'n_right': r['turn']['n_right'],
               'ref_ratio': ss._r(ref_ratio, 3), 'ref_left': (c or {}).get('turn_gain_left'), 'ref_right': (c or {}).get('turn_gain_right'),
               's0': r['s0']['s0_us'], 's0_n': r['s0']['n'], 'ref_s0': (s or {}).get('s0_us'), 'ref_s0_n': (s or {}).get('frames'),
               'mps_p90': r['top_speed']['mps_p90'], 'ref_erpm_p90': (c or {}).get('erpm_p90_ai'), 'erpm_p90': r['top_speed']['erpm_p90'],
               'erpm_p90_HS': r['top_speed']['erpm_p90_HS'], 'ref_erpm_p90_HS': (c or {}).get('erpm_p90_HS'),
               'a3_median': r['r1_bump']['a3_median'], 'a3_max': r['r1_bump']['a3_max'], 'r1_passes': r['r1_bump']['passes'],
               'ref_a3_median': (b or {}).get('a3_max_median'), 'ref_a3_max': bmax.get(lg), 'ref_r1_passes': (b or {}).get('passes'),
               'laps': r['laps']['n'], 'lap_p90_median': r['laps']['p90_median_mps'], 'sw_max_median': r['laps']['sw_max_median_mps'],
               'code': r['judge']['code'], 'label': r['judge']['label'],
               'items': {x['item']: x['status'] for x in r['judge']['items']}, 'speed': r['judge']['speed_advice'].get('text')}
        d = lambda a, b_: None if a is None or b_ is None else round(a - b_, 4)      # noqa: E731
        row['d_ratio'], row['d_s0'] = d(row['ratio'], row['ref_ratio']), d(row['s0'], row['ref_s0'])
        row['d_a3_median'], row['d_a3_max'] = d(row['a3_median'], row['ref_a3_median']), d(row['a3_max'], row['ref_a3_max'])
        rows.append(row)
        print('%s No.%-3s 比 %s(台本 %s)s0 %s(台本 %s)板 %s/%s(台本 %s/%s)判定 %s %s 秒' % (
            lg, row['no'], row['ratio'], row['ref_ratio'], row['s0'], row['ref_s0'], row['a3_median'], row['a3_max'],
            row['ref_a3_median'], row['ref_a3_max'], row['label'], row['seconds']), flush=True)

    def rng(xs):
        xs = [x for x in xs if x is not None]
        return [min(xs), max(xs)] if xs else None
    d24 = [r for r in rows if r['log'].startswith('20261024') and 'error' not in r]
    before = [r for r in d24 if r.get('session') in (1, 2) and r['n_left'] >= 30 and r['n_right'] >= 30]
    after = [r for r in d24 if r.get('session') == 3 and r['n_left'] >= 30 and r['n_right'] >= 30]
    s0b = [r['s0'] for r in d24 if r.get('session') in (1, 2) and r['s0'] is not None]
    s0a = [r['s0'] for r in d24 if r.get('session') == 3 and r['s0'] is not None]
    summ = {'ratio_before_repair': rng([r['ratio'] for r in before]), 'ratio_after_repair': rng([r['ratio'] for r in after]),
            'ratio_before_nos': [r['no'] for r in before], 'ratio_after_nos': [r['no'] for r in after],
            's0_before_median': float(np.median(s0b)) if s0b else None, 's0_after_median': float(np.median(s0a)) if s0a else None,
            's0_before_range': rng(s0b), 's0_after_range': rng(s0a),
            'max_abs_d_ratio': max(abs(r['d_ratio']) for r in rows if r.get('d_ratio') is not None),
            'max_abs_d_s0': max(abs(r['d_s0']) for r in rows if r.get('d_s0') is not None),
            'max_abs_d_a3_median': max(abs(r['d_a3_median']) for r in rows if r.get('d_a3_median') is not None),
            'max_abs_d_a3_max': max(abs(r['d_a3_max']) for r in rows if r.get('d_a3_max') is not None),
            'max_seconds': max(r['seconds'] for r in rows if 'seconds' in r)}
    for k in ('s0_before_median', 's0_after_median'):
        summ[k] = None if summ[k] is None else round(summ[k], 1)
    summ['s0_shift_us'] = None if summ['s0_after_median'] is None else round(summ['s0_after_median'] - summ['s0_before_median'], 1)
    OUT.write_text(json.dumps({'created_by': 'experiments/20261025_speed_control/car_check_1024.py', 'summary': summ, 'rows': rows},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False))
    print('->', OUT)


if __name__ == '__main__':
    main()
