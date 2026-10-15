#!/usr/bin/env python3
"""確認(sim_diverse_train_20261015 の Q1、2026-10-15): eval_sim.json の完走の数・判定を、走行の生の記録(ブラウザのアプリが保存した JSON)から
数え直す。10/14 の review/recount.py の写しを広げたもの。

    python3 -I recount_q1.py

方針: 評価の段の道具(eval/*.py・*.mjs・runs.csv・eval_sim.json)は使わない。生の JSON の outcome・metrics・trace・world・context だけを読む。
  - 完走 = outcome.reason == route_complete かつ 衝突 0・33 ゲート・黒シートの横断 1 回(protocol eval_Q1.run_success の記録の側)
  - 最初の直線を抜けた fs = ゲート 0 の開始で、衝突の前に trace の x > 2.6 m
  - 点 4 のゲート g4 = ゲート 0 の開始で、sim が数えたゲート ≥ 4(protocol の定義)。参考に、点 3 → 点 4 の線を越えた横の位置も
  - 進んだ道のり(別の作り方): world.route の 33 点を結んだ閉じた折れ線に、trace の位置を射影して(前の位置の近くの区間だけを探す)
    折れ線に沿った長さを続けてたどる。評価の道具(最速の先生の参照の周 R)とは違う線なので、値は少し違う。周の割合 × 26.1 m に直して
    d の判定(1.5 × d(A1) など)が同じになるかを見る
  - 判定は protocol.json verdicts_Q1 の文をそのままコードにした(ゲート 0 の開始は挙げてある 9 本)
出力: /home/jetson/jetformer_reports/sim_diverse_train_20261015/review/recount_q1.json
"""
import glob
import json
import math
import os
import re
import statistics

REP = '/home/jetson/jetformer_reports/sim_diverse_train_20261015'
PREV = '/home/jetson/jetformer_reports/sim_dart_20261014'
BEST = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013'
OUT = os.path.join(REP, 'review', 'recount_q1.json')
L_REF = 26.1   # protocol baselines_already_measured.Q1.lap_length_m


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def start_name(r, plan_starts):
    sv = r['context']['start_variation']
    return plan_starts.get((sv['gate'], round(sv['left_offset_m'], 4), round(sv['left_yaw_deg'], 4)))


def gate4_crossing(r):
    route = r['world']['route']
    p3, p4, p5 = route[3], route[4], route[5]
    dx, dz = p4[0] - p3[0], p4[1] - p3[1]
    n = math.hypot(dx, dz)
    ux, uz = dx / n, dz / n
    lx, lz = -uz, ux
    if (p5[0] - p4[0]) * lx + (p5[1] - p4[1]) * lz > 0:
        lx, lz = -lx, -lz
    prev = None
    for t in r['trace']:
        a = (t['x'] - p4[0]) * ux + (t['z'] - p4[1]) * uz
        if prev is not None and prev[0] < 0 <= a:
            w = -prev[0] / (a - prev[0])
            x = prev[1]['x'] + w * (t['x'] - prev[1]['x'])
            z = prev[1]['z'] + w * (t['z'] - prev[1]['z'])
            return {'t': round(prev[1]['time'] + w * (t['time'] - prev[1]['time']), 3), 'lateral_m': round((x - p4[0]) * lx + (z - p4[1]) * lz, 3),
                    'inside_gate': abs((x - p4[0]) * lx + (z - p4[1]) * lz) <= 0.65}
        prev = (a, t)
    return None


def polyline_progress(r):
    """world.route の閉じた折れ線に沿って進んだ長さ(前の区間の近く ±2 区間だけを探して、続けてたどる)"""
    route = r['world']['route']
    n = len(route)
    seg_len = [math.dist(route[i], route[(i + 1) % n]) for i in range(n)]
    cum = [0.0]
    for v in seg_len:
        cum.append(cum[-1] + v)
    total = cum[-1]

    def proj(x, z, i):
        ax, az = route[i]
        bx, bz = route[(i + 1) % n]
        dx, dz = bx - ax, bz - az
        L2 = dx * dx + dz * dz
        u = max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / L2)) if L2 > 0 else 0.0
        return math.hypot(x - ax - u * dx, z - az - u * dz), cum[i] + u * seg_len[i]

    tr = r['trace']
    # 最初の位置は全部の区間から探す
    d0, s0, i0 = min((proj(tr[0]['x'], tr[0]['z'], i) + (i,) for i in range(n)), key=lambda v: v[0])
    seg, s_prev, unwrapped = i0, s0, 0.0
    for t in tr[1:]:
        best = min((proj(t['x'], t['z'], (seg + k) % n) + ((seg + k) % n,) for k in (-2, -1, 0, 1, 2)), key=lambda v: v[0])
        _, s, seg = best
        ds = s - s_prev
        if ds < -total / 2:
            ds += total
        elif ds > total / 2:
            ds -= total
        unwrapped += ds
        s_prev = s
    return {'progress_polyline_m': round(unwrapped, 3), 'polyline_length_m': round(total, 3), 'lap_fraction': round(unwrapped / total, 4)}


def one(p, plan_starts, gate0):
    r = load(p)
    m, o = r['metrics'], r['outcome']
    ce = (m.get('collision_events') or [None])[0]
    cross = len(m.get('required_crossing_events') or [])
    completed = o['reason'] == 'route_complete' and m['collisions'] == 0 and m['passed_gates'] == 33 and cross == 1
    st = start_name(r, plan_starts)
    tr = r['trace']
    fs = None
    if st in gate0:
        cf = ce['frame'] if ce else 10 ** 9
        fs = any(t['x'] > 2.6 and t['frame'] <= cf for t in tr)
    pp = polyline_progress(r)
    return {
        'name': r.get('name') or os.path.basename(p)[:-5], 'file': p, 'start': st, 'reason': o['reason'], 'completed': completed,
        'batch_status': r.get('batch_status'), 'elapsed_s': m['elapsed'], 'passed_gates': m['passed_gates'], 'collisions': m['collisions'],
        'crossings': cross, 'collider_kind': ce['kind'] if ce else None, 'collision_xz': [round(ce['x'], 2), round(ce['z'], 2)] if ce else None,
        'frames': len(r['actions']), 'model_sha256': r['context']['policy_info'].get('model_sha256'),
        'control_delay_ticks': r['context']['execution']['control_delay']['ticks'],
        'lookahead': r['context']['policy_info'].get('lookahead'), 'steer_gain': r['context']['policy_info'].get('steer_gain'),
        'throttle_gain': r['context']['policy_info'].get('throttle_gain'), 'mode_id': r['context']['policy_info'].get('mode_id'),
        'episode_mode': r['config'].get('episodeMode', 'single'), 'time_limit': r['config']['timeLimit'],
        'first_straight_passed': fs, 'g4': (m['passed_gates'] >= 4) if st in gate0 else None, 'gate4_line': gate4_crossing(r) if st in gate0 else None,
        # d の 1 本ぶん(別の作り方): 完走は 1 周、ほかは折れ線に沿った周の割合 × 26.1 m
        'd_value_polyline_scaled_m': L_REF if completed else round(pp['lap_fraction'] * L_REF, 3), **pp,
    }


def main():
    prot = load(os.path.join(REP, 'protocol.json'))
    V = prot['verdicts_Q1']
    starts = prot['eval_Q1']['starts']
    gate0 = starts['gate0_starts']
    plan_starts = {(v[0], round(v[1], 4), round(v[2], 4)): k for k, v in starts.items() if re.fullmatch(r'E\d+', k)}
    T = f'{REP}/eval/closed_loop'
    groups = {
        'A0': sorted(glob.glob(f'{BEST}/eval/closed_loop/SR-s0/SR-s0-E?-single.json')) + sorted(glob.glob(f'{PREV}/eval/closed_loop/A0-s0/*-single.json')),
        'A1': sorted(glob.glob(f'{PREV}/eval/closed_loop/A1-s0/A1-s0-E*-single.json')),
        'A2': sorted(glob.glob(f'{PREV}/eval/closed_loop/A2-s0/A2-s0-E*-single.json')),
        'A2@1015': sorted(glob.glob(f'{T}/A2-s0/A2-s0-E*-single.json')),
        'S-DIV': sorted(glob.glob(f'{T}/SDIV-s0/SDIV-s0-E*-single*.json')),
        'S-MOD': sorted(glob.glob(f'{T}/SMOD-s0/SMOD-s0-E*-single.json')),
        'A4': sorted(glob.glob(f'{T}/A4-s0/A4-s0-E*-single.json')),
        'EXT_today': sorted(glob.glob(f'{T}/EXT/EXT-E0-single.json')),
        'A1_today': sorted(glob.glob(f'{T}/A1-s0/A1-s0-E0-single.json')),
        'flying_today': sorted(glob.glob(f'{T}/*/*-flying.json')),
    }
    out = {'note': '生の JSON から数え直した(評価の段の道具は使っていない)', 'arms': {}}
    for g, files in groups.items():
        if not files:
            continue
        rows = [one(p, plan_starts, gate0) for p in files]
        pr = [x for x in rows if not x['name'].endswith('rep2')]
        fs_rows = [x for x in pr if x['start'] in gate0]
        out['arms'][g] = {
            'n': len(pr), 'starts': [x['start'] for x in pr],
            'c': sum(x['completed'] for x in pr), 'completed_starts': [x['start'] for x in pr if x['completed']],
            'laps_s': {x['start']: round(x['elapsed_s'], 4) for x in pr if x['completed']},
            'fs': sum(1 for x in fs_rows if x['first_straight_passed']), 'gate0_n': len(fs_rows),
            'g4': sum(1 for x in fs_rows if x['g4']),
            'd_polyline_scaled_median_m': round(statistics.median([x['d_value_polyline_scaled_m'] for x in pr]), 3) if pr else None,
            'all_batch_completed': all(x['batch_status'] == 'completed' for x in rows),
            'settings_set': sorted({(x['control_delay_ticks'], x['lookahead'], x['steer_gain'], x['mode_id'], x['time_limit']) for x in rows}, key=str),
            'model_sha256_set': sorted({x['model_sha256'] for x in rows}),
            'rows': rows,
        }
    a = out['arms']
    v = {}
    if 'S-DIV' in a:
        refs = [x for x in ('A0', 'A1', 'A2', 'A2@1015') if x in a]
        c_ref = max(a[x]['c'] for x in refs)
        cs, fs, g4 = a['S-DIV']['c'], a['S-DIV']['fs'], a['S-DIV']['g4']
        if cs >= 6 and cs - c_ref >= 4:
            vv = '効いた'
        elif cs - c_ref >= 2 or g4 >= 5:
            vv = '一部効いた'
        elif cs <= c_ref and fs <= 4:
            vv = '悪くなった'
        else:
            vv = '差なし'
        v['Q1_S-DIV_vs_yesterday'] = {'verdict': vv, 'c': cs, 'c_ref': c_ref, 'fs': fs, 'g4': g4, 'rule': V['Q1_S-DIV_vs_yesterday']}
    if 'S-MOD' in a:
        sm, a1 = a['S-MOD'], a['A1']
        dA1 = a1['d_polyline_scaled_median_m']
        if sm['c'] - a1['c'] >= 2 or (sm['c'] == a1['c'] == 0 and sm['g4'] - a1['g4'] >= 3) or \
                (sm['c'] == a1['c'] == 0 and sm['g4'] == a1['g4'] == 0 and sm['d_polyline_scaled_median_m'] >= 1.5 * dA1):
            vv = '量よりばらつき'
        else:
            vv = 'ばらつきの効果は見えない'
        v['Q1_variety_vs_volume'] = {'verdict': vv, 'c_S-MOD': sm['c'], 'c_A1': a1['c'], 'g4_S-MOD': sm['g4'], 'g4_A1': a1['g4'],
                                     'd_S-MOD_polyline_scaled_m': sm['d_polyline_scaled_median_m'], 'd_A1_polyline_scaled_m': dA1}
        if 'S-DIV' in a:
            sd = a['S-DIV']
            dc, dg, df = sd['c'] - sm['c'], sd['g4'] - sm['g4'], sd['fs'] - sm['fs']
            both0 = sd['c'] == 0 and sm['c'] == 0
            vv = '上乗せあり' if (dc >= 2 or (both0 and (dg >= 3 or df >= 3))) else ('悪くなった' if (dc <= -2 or (both0 and (dg <= -3 or df <= -3))) else '差なし')
            v['Q1_DART_margin_addon'] = {'verdict': vv, 'c_diff': dc, 'g4_diff': dg, 'fs_diff': df}
    if 'A4' in a and 'S-DIV' in a:
        b4, sd = a['A4'], a['S-DIV']
        dc, dg, df = b4['c'] - sd['c'], b4['g4'] - sd['g4'], b4['fs'] - sd['fs']
        both0 = b4['c'] == 0 and sd['c'] == 0
        vv = '上乗せあり' if (dc >= 2 or (both0 and dg >= 3)) else ('悪くなった' if (dc <= -2 or (both0 and (dg <= -3 or df <= -3))) else '差なし')
        v['Q1b_DAgger'] = {'verdict': vv, 'c_diff': dc, 'g4_diff': dg, 'fs_diff': df, 'base': 'S-DIV'}
    out['verdicts'] = v
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for g, x in a.items():
        print(f"{g:12s} n={x['n']:2d} 完走={x['c']} {x['laps_s']} 直線={x['fs']}/{x['gate0_n']} 点4={x['g4']} d(折れ線)={x['d_polyline_scaled_median_m']} "
              f"設定={x['settings_set']}")
    print('判定', json.dumps(v, ensure_ascii=False))


if __name__ == '__main__':
    main()
