#!/usr/bin/env python3
"""fix9 tonight s24-analysis (2026-10-29): 止まってから 3 周で 24 秒を切るための数。読むだけ。

    PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1200M --need-mb 2000 -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I s24_numbers.py

入力(読むだけ):
  - /home/jetson/jetformer_reports/bench_20261024_report/tt_data.json(解析 A の周・窓・発進。10/23 公式 + 10/24)
  - 同じフォルダの standing_20261023.json・standing_20261024.json(tools/standing_3lap.py を 10/29 に流し直した出力)
出力: s24_numbers.json(このフォルダ)と標準出力の表
"""
import json
import os
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TT = json.load(open('/home/jetson/jetformer_reports/bench_20261024_report/tt_data.json'))
GROUPS = ['R1+GM', 'DE+SW', 'HS+LH']
TARGET = 24.0
LOSS_NOW = 0.63          # 止まってからの損(B・9 回の中央値。analysis_tt 5.1)
LOSS_BOOST = 0.63 - 0.28  # 発進の前進の見積もり(build_launch.md 5 章: 0.28 秒縮む)


def med(x):
    return float(st.median(x)) if x else None


def laps_of(pred):
    return [lp for lp in TT['laps'] if not lp['false_end'] and pred(lp)]


def group_table(laps):
    out = {}
    for g in GROUPS + ['lap']:
        vals = [(lp['groups_s'][g] if g != 'lap' else lp['lap_s']) for lp in laps]
        if not vals:
            continue
        i = int(np.argmin(vals))
        out[g] = {'n': len(vals), 'best': round(min(vals), 3), 'p25': round(float(np.percentile(vals, 25)), 3),
                  'median': round(med(vals), 3), 'mean': round(float(np.mean(vals)), 3),
                  'best_from': '%s No.%s %.1f 秒' % (laps[i]['session'], laps[i]['no'], laps[i]['t_start'])}
    out['sum_best'] = round(sum(out[g]['best'] for g in GROUPS), 3)
    out['sum_p25'] = round(sum(out[g]['p25'] for g in GROUPS), 3)
    out['sum_median'] = round(sum(out[g]['median'] for g in GROUPS), 3)
    return out


def main():
    res = {}
    # ---- 1. 周と区間 ----
    sets = {
        'No.9(10/24 2 回目 B 0.90)全部の周': laps_of(lambda lp: lp['session'] == '10/24 2回目' and lp['no'] == '9'),
        'No.9 きれいな周': laps_of(lambda lp: lp['session'] == '10/24 2回目' and lp['no'] == '9' and lp['clean']),
        'B のきれいな周(全部の回)': laps_of(lambda lp: lp['model'] == 'B' and lp['clean']),
        '全部のモデルの周(偽の周を除く)': laps_of(lambda lp: True),
        '3 回目 B 0.90 ステア 1.00(No.11・12・18)': laps_of(lambda lp: lp['session'] == '10/24 3回目' and lp['model'] == 'B'
                                                       and abs(lp['tg'] - 0.90) < 1e-6 and abs(lp['sg'] - 1.0) < 1e-6),
    }
    res['groups'] = {k: group_table(v) for k, v in sets.items()}
    # No.9 の周ごと
    no9 = sorted(sets['No.9(10/24 2 回目 B 0.90)全部の周'], key=lambda lp: lp['t_start'])
    res['no9_laps'] = [{'t_start': lp['t_start'], 'lap_s': lp['lap_s'], 'kind': lp['kind'],
                        **{g: lp['groups_s'][g] for g in GROUPS}, 'p90': lp['speed_mps']['p90'],
                        'v': {g: lp['speed_mps'][g] for g in GROUPS}} for lp in no9]
    # ---- 2. 周の時間と速さ(B の周。解析 A の式の確かめ) ----
    bl = laps_of(lambda lp: lp['model'] == 'B')
    x = np.array([lp['speed_mps']['p90'] for lp in bl])
    y = np.array([lp['lap_s'] for lp in bl])
    a, b = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    res['lap_vs_p90'] = {'n': len(bl), 'slope': round(float(a), 3), 'intercept': round(float(b), 3), 'r': round(r, 3),
                         'p90_range': [round(float(x.min()), 2), round(float(x.max()), 2)]}
    # 区間ごと: 区間の時間 ~ 区間の速さ(中央値)
    res['group_vs_speed'] = {}
    for g in GROUPS:
        xs = np.array([lp['speed_mps'][g] for lp in bl])
        ys = np.array([lp['groups_s'][g] for lp in bl])
        # 距離 = 時間 × 速さ(区間の中央値の速さで近似)。速さを k 倍 → 時間 1/k 倍とみたときの「区間の長さ」
        dist = ys * xs
        res['group_vs_speed'][g] = {'r': round(float(np.corrcoef(xs, ys)[0, 1]), 3),
                                    'v_med': round(float(np.median(xs)), 2), 'v_max': round(float(xs.max()), 2),
                                    't_med': round(float(np.median(ys)), 3),
                                    'dist_med_m': round(float(np.median(dist)), 2)}
    # ---- 3. 24 秒の予算 ----
    bud = {}
    for name, loss in (('今の発進(損 0.63 秒)', LOSS_NOW), ('発進の前進 ON(損 約 0.35 秒)', LOSS_BOOST)):
        fly3 = TARGET - loss
        lap = fly3 / 3
        p90 = (b - lap) / (-a) if a < 0 else None
        bud[name] = {'flying_3lap_s': round(fly3, 3), 'per_lap_s': round(lap, 3), 'p90_needed_by_fit': round(p90, 2)}
    # 23.9 秒(少しの余裕)も
    for name, loss in (('発進の前進 ON・23.9 秒(0.1 秒の余裕)', LOSS_BOOST),):
        fly3 = 23.9 - loss
        bud[name] = {'flying_3lap_s': round(fly3, 3), 'per_lap_s': round(fly3 / 3, 3),
                     'p90_needed_by_fit': round((b - fly3 / 3) / (-a), 2)}
    res['budget'] = bud
    # No.9 の水準からの差
    g9 = res['groups']['No.9 きれいな周']
    need = bud['発進の前進 ON(損 約 0.35 秒)']['per_lap_s']
    res['gap_from_no9'] = {
        'no9_clean_median_lap': g9['lap']['median'], 'no9_clean_mean_lap': g9['lap']['mean'],
        'need_per_lap': need, 'cut_per_lap_vs_median': round(g9['lap']['median'] - need, 3),
        'cut_pct_vs_median': round(100 * (g9['lap']['median'] - need) / g9['lap']['median'], 1),
        'sum_best_no9_clean': g9['sum_best'], 'sum_median_no9_clean': g9['sum_median'],
    }
    # 区間ごとに要る時間: No.9 のきれいな周の中央値から、「縮めやすさ」= 中央値 − いちばん速い値 の比で配る
    room = {g: g9[g]['median'] - g9[g]['best'] for g in GROUPS}
    cut = g9['sum_median'] - need
    tot = sum(room.values())
    res['section_plan'] = {g: {'no9_median': g9[g]['median'], 'no9_best': g9[g]['best'],
                               'target': round(g9[g]['median'] - cut * room[g] / tot, 3),
                               'cut': round(cut * room[g] / tot, 3)} for g in GROUPS}
    res['section_plan']['_note'] = 'No.9 のきれいな周の中央値の和 %.3f 秒 → 要る %.3f 秒。縮める量を(中央値 − 最速)の比で配った' % (
        g9['sum_median'], need)
    # ---- 4. 止まってから 3 周(standing_3lap の出力) ----
    rows = []
    for day in ('20261023', '20261024'):
        js = json.load(open(os.path.join(HERE, 'standing_%s.json' % day)))
        for rr in js['results']:
            for s in rr.get('standing', []):
                if s.get('from_move_s') is None:
                    continue
                coll = s.get('collisions') or {}
                rows.append({'log': rr['log'], 'no': rr.get('ai_no'), 'setting': rr.get('settings_text'),
                             'model': (rr.get('settings') or {}).get('model'),
                             'gain': (rr.get('settings') or {}).get('throttle_gain'),
                             'from_move_s': s['from_move_s'], 'from_ai_s': s.get('from_ai_s'),
                             'valid': s.get('valid'), 'counted': s.get('counted'), 'why': s.get('why'),
                             'method': s.get('method'), 'line_laps': s.get('line_laps_s') or s.get('laps_line_s'),
                             'coll_n': s.get('coll_n', coll.get('n')), 'coll_strong': s.get('coll_strong', coll.get('strong')),
                             'dist3_from_ai_s': s.get('dist3_from_ai_s'), 'move_after_s': s.get('move_after_s'),
                             'keys': sorted(s.keys())})
    res['standing_rows'] = rows
    counted = [r for r in rows if r['counted']]
    res['standing_summary'] = {
        'counted_n': len(counted),
        'best': min(counted, key=lambda r: r['from_move_s']) if counted else None,
        'median_all_counted': med([r['from_move_s'] for r in counted]),
        'median_B_counted': med([r['from_move_s'] for r in counted if r['model'] == 'B']),
    }
    json.dump(res, open(os.path.join(HERE, 's24_numbers.json'), 'w'), ensure_ascii=False, indent=1)
    # ---- 表示 ----
    print('== 周の時間 ~ p90:', res['lap_vs_p90'])
    print('== 区間ごとの速さ:', json.dumps(res['group_vs_speed'], ensure_ascii=False))
    for k, v in res['groups'].items():
        print('==', k)
        for g in GROUPS + ['lap']:
            if g in v:
                print('   %-6s n=%d best %.3f p25 %.3f med %.3f mean %.3f (%s)' % (
                    g, v[g]['n'], v[g]['best'], v[g]['p25'], v[g]['median'], v[g]['mean'], v[g]['best_from']))
        print('   sum_best %.3f sum_p25 %.3f sum_median %.3f' % (v['sum_best'], v['sum_p25'], v['sum_median']))
    print('== 予算:', json.dumps(res['budget'], ensure_ascii=False))
    print('== No.9 からの差:', json.dumps(res['gap_from_no9'], ensure_ascii=False))
    print('== 区間の目標:', json.dumps(res['section_plan'], ensure_ascii=False))
    print('== No.9 の周:')
    for lp in res['no9_laps']:
        print('   %.1f %.3f %s R1+GM %.3f DE+SW %.3f HS+LH %.3f p90 %.2f' % (
            lp['t_start'], lp['lap_s'], lp['kind'], lp['R1+GM'], lp['DE+SW'], lp['HS+LH'], lp['p90']))
    print('== 止まってから 3 周(standing_3lap):', json.dumps(res['standing_summary'], ensure_ascii=False)[:600])
    print('   keys of a standing row:', rows[0]['keys'] if rows else None)


if __name__ == '__main__':
    main()
