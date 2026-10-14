#!/usr/bin/env python3
"""確認の段(review、2026-10-14): evaluation.md の数を、走行の生の記録(ブラウザのアプリが保存した JSON)から数え直す

    python3 -I recount.py

方針: 評価の段の道具(eval/*.py・*.mjs・runs.csv・evaluation.json)は使わない。生の JSON の outcome・metrics・trace・world だけを読む。
  - 完走 = outcome.reason == route_complete かつ 衝突 0・33 ゲート・黒シートの横断 1 回(protocol eval.run_success の記録の側)
  - 最初の直線を抜けた = ゲート 0 の開始で、衝突の前に trace の x > 2.6 m(protocol eval.per_run_metrics)
  - 点 4 のゲート = 元の経路の点 4(ゲート 0 の開始では world.route[4])。点 3 → 点 4 の向きの線を前へ越えた最初の時刻と、そのときの
    ゲートの中心からの横の距離(線形補間)。横の向きは「次の点(点 5)のある側 = 曲がりの内側」を負とする
  - 判定は protocol.json verdicts の文をそのままコードにした(ゲート 0 の開始は挙げてある 9 本)
出力: /home/jetson/jetformer_reports/sim_dart_20261014/review/recount.json
"""
import glob
import json
import math
import os
import re
import statistics

REP = '/home/jetson/jetformer_reports/sim_dart_20261014'
BEST = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013'
OUT = os.path.join(REP, 'review', 'recount.json')
GATE0 = ['E0', 'E1', 'E2', 'E3', 'E4', 'E6', 'E7', 'E8', 'E9']


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def start_name(r, plan_starts):
    sv = r['context']['start_variation']
    key = (sv['gate'], round(sv['left_offset_m'], 4), round(sv['left_yaw_deg'], 4))
    return plan_starts.get(key)


def gate4_crossing(r):
    """ゲート 0 の開始だけ。点 3 → 点 4 の向きで点 4 の線を前へ越えた最初の時刻と横の位置"""
    route = r['world']['route']
    p3, p4, p5 = route[3], route[4], route[5]
    dx, dz = p4[0] - p3[0], p4[1] - p3[1]
    n = math.hypot(dx, dz)
    ux, uz = dx / n, dz / n
    # 横の単位ベクトル。点 5 の側(曲がりの内側)を負にする
    lx, lz = -uz, ux
    if (p5[0] - p4[0]) * lx + (p5[1] - p4[1]) * lz > 0:
        lx, lz = -lx, -lz
    tr = r['trace']
    prev = None
    for t in tr:
        a = (t['x'] - p4[0]) * ux + (t['z'] - p4[1]) * uz
        if prev is not None and prev[0] < 0 <= a:
            w = -prev[0] / (a - prev[0])
            x = prev[1]['x'] + w * (t['x'] - prev[1]['x'])
            z = prev[1]['z'] + w * (t['z'] - prev[1]['z'])
            lat = (x - p4[0]) * lx + (z - p4[1]) * lz
            tm = prev[1]['time'] + w * (t['time'] - prev[1]['time'])
            return {'t': round(tm, 3), 'lateral_m': round(lat, 3), 'inside_gate': abs(lat) <= 0.65}
        prev = (a, t)
    return None


def one(p, plan_starts):
    r = load(p)
    m, o = r['metrics'], r['outcome']
    ce = (m.get('collision_events') or [None])[0]
    cross = len(m.get('required_crossing_events') or [])
    completed = o['reason'] == 'route_complete' and m['collisions'] == 0 and m['passed_gates'] == 33 and cross == 1
    st = start_name(r, plan_starts)
    tr = r['trace']
    fs = None
    if st in GATE0:
        cf = ce['frame'] if ce else 10 ** 9
        fs = any(t['x'] > 2.6 and t['frame'] <= cf for t in tr)
        if ce is None and not fs:
            fs = False
    g4 = gate4_crossing(r) if st in GATE0 else None
    return {
        'name': r.get('name') or os.path.basename(p)[:-5], 'file': p, 'start': st, 'reason': o['reason'], 'completed': completed,
        'elapsed_s': m['elapsed'], 'passed_gates': m['passed_gates'], 'collisions': m['collisions'], 'crossings': cross,
        'collider_kind': ce['kind'] if ce else None, 'collision_xz': [round(ce['x'], 2), round(ce['z'], 2)] if ce else None,
        'frames': len(r['actions']), 'model_sha256': r['context']['policy_info'].get('model_sha256'),
        'control_delay_ticks': r['context']['execution']['control_delay']['ticks'],
        'lookahead': r['context']['policy_info'].get('lookahead'), 'steer_gain': r['context']['policy_info'].get('steer_gain'),
        'throttle_gain': r['context']['policy_info'].get('throttle_gain'), 'mode_id': r['context']['policy_info'].get('mode_id'),
        'episode_mode': r['config'].get('episodeMode', 'single'), 'time_limit': r['config']['timeLimit'],
        'first_straight_passed': fs, 'gate4': g4,
    }


def main():
    prot = load(os.path.join(REP, 'protocol.json'))
    plan_starts = {(g, round(o, 4), round(y, 4)): k for k, v in prot['eval']['starts'].items() if re.fullmatch(r'E\d+', k) for (g, o, y) in [v]}
    groups = {
        'A0-s0': sorted(glob.glob(f'{BEST}/eval/closed_loop/SR-s0/SR-s0-E?-single.json')) + sorted(glob.glob(f'{REP}/eval/closed_loop/A0-s0/*-single.json')),
        'A1-s0': sorted(glob.glob(f'{REP}/eval/closed_loop/A1-s0/A1-s0-E*-single.json')),
        'A2-s0': sorted(glob.glob(f'{REP}/eval/closed_loop/A2-s0/A2-s0-E*-single.json')),
        'A0-s1': sorted(glob.glob(f'{BEST}/eval/closed_loop/SR-s1/SR-s1-E?-single.json')),
        'ST-s0': sorted(glob.glob(f'{BEST}/eval/closed_loop/ST-s0/ST-s0-E?-single.json')),
        'ST-s1': sorted(glob.glob(f'{BEST}/eval/closed_loop/ST-s1/ST-s1-E?-single.json')),
        'EXT': sorted(glob.glob(f'{BEST}/eval/closed_loop/EXT/EXT-E?-single.json')),
        'EXT_today': sorted(glob.glob(f'{REP}/eval/closed_loop/EXT/EXT-E0-single.json')),
        'flying_today': sorted(glob.glob(f'{REP}/eval/closed_loop/A?-s0/*-flying.json')),
    }
    out = {'note': '生の JSON から数え直した(評価の段の道具は使っていない)', 'arms': {}}
    for g, files in groups.items():
        rows = [one(p, plan_starts) for p in files]
        protocol_rows = [x for x in rows if not x['name'].endswith('rep2')]
        stops = [x['elapsed_s'] for x in protocol_rows if not x['completed']]
        fs_rows = [x for x in protocol_rows if x['start'] in GATE0]
        out['arms'][g] = {
            'n': len(protocol_rows), 'starts': [x['start'] for x in protocol_rows],
            'completed': sum(x['completed'] for x in protocol_rows), 'completed_starts': [x['start'] for x in protocol_rows if x['completed']],
            'laps_s': {x['start']: round(x['elapsed_s'], 3) for x in protocol_rows if x['completed']},
            'stop_time_median_s': round(statistics.median([x['elapsed_s'] for x in protocol_rows]), 4) if protocol_rows else None,
            'stop_time_median_failed_s': round(statistics.median(stops), 4) if stops else None,
            'first_straight_passed': sum(1 for x in fs_rows if x['first_straight_passed']), 'first_straight_starts': len(fs_rows),
            'gate4_missed_starts': [x['start'] for x in fs_rows if x['gate4'] and not x['gate4']['inside_gate']],
            'gate4_counted': sum(1 for x in protocol_rows if x['passed_gates'] >= 4 and x['start'] in GATE0),
            'rows': rows,
        }
    a = out['arms']

    def verdict(arm):
        c, c0 = a[arm]['completed'], a['A0-s0']['completed']
        fsn = a[arm]['first_straight_passed']
        if c >= 6 and c - c0 >= 4:
            return '効いた'
        if c - c0 >= 2 or fsn >= 8:
            return '一部効いた'
        return '効かなかった'
    out['verdicts'] = {'Q1_DART': verdict('A1-s0'), 'Q2_MARGIN': verdict('A2-s0')}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for g, v in a.items():
        print(f"{g:13s} n={v['n']:2d} 完走={v['completed']} {v['laps_s']} 止まる中央値={v['stop_time_median_s']} "
              f"直線を抜けた={v['first_straight_passed']}/{v['first_straight_starts']} 点4外し={v['gate4_missed_starts']}")
    print('判定', out['verdicts'])


if __name__ == '__main__':
    main()
