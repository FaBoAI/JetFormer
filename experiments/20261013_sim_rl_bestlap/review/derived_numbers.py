"""確認の段(review、2026-10-14): README 5.3 節・6 章の「失敗の形」の数字を、保存した走行から計算し直して JSON に残す

    python3 -I derived_numbers.py <出力 JSON>

これまで JSON に無かった数字(図を見て書いた数字)を、ここで計算できる形にする:
  - RL の線・教師の線の、S-R がぶつかったレール(segment-11)までの最小のすき間(最初の直線、x 0〜2.6 m)
  - S-R(E0〜E4 の 10 本)の、ぶつかった時点と 0.5 秒の時点での RL の線からのずれ(同じ開始の RL の走り・E0 の RL の走りの 2 通り)
  - ゲート(経路の点 4)を通った横の位置: 教師・S-T 種 0 の E1・作者の ONNX の E0
  - 付け直しの舵(同じ状態で RL の方策が出す舵)と S-R 種 0 の E0 が使った舵の差(フレーム 15〜26)
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import independent_lap_check as ilc  # noqa: E402

REP = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013'


def load(p):
    return json.load(open(p, encoding='utf-8'))


def seg_clear(rep, seg_id, xlo=-9, xhi=9):
    """車体の長方形と、ある線分(半径つき)の最小のすき間 [m] と、そのフレーム"""
    w = ilc.World(rep['world'])
    s = next(x for x in w.segments if x['id'] == seg_id)
    hx, hz = rep['config']['vehicleLength'] / 2, rep['config']['vehicleWidth'] / 2
    best = (float('inf'), None)
    for t in rep['trace']:
        if not xlo <= t['x'] <= xhi:
            continue
        d = ilc.seg_box_dist(ilc.to_local(s['a'][0], s['a'][1], t['x'], t['z'], t['yaw']),
                             ilc.to_local(s['b'][0], s['b'][1], t['x'], t['z'], t['yaw']), hx, hz) - s['r']
        if d < best[0]:
            best = (d, {'frame': t['frame'], 'x': round(t['x'], 3), 'z': round(t['z'], 3)})
    return {'segment': seg_id, 'kind': s['kind'], 'a': s['a'], 'b': s['b'], 'radius': s['r'], 'min_clearance_m': best[0], 'at': best[1]}


def nearest_obstacles_first_straight(rep, x_end=2.6, top=4):
    """最初の直線(開始から、車の x が初めて x_end を超えるまで)で、障害物ごとの最小のすき間(近い順 top 個)"""
    w = ilc.World(rep['world'])
    hx, hz = rep['config']['vehicleLength'] / 2, rep['config']['vehicleWidth'] / 2
    best = {}
    for t in rep['trace']:
        if t['x'] > x_end:
            break
        car = ilc.corners(t['x'], t['z'], t['yaw'], hx, hz)
        for s in w.segments:
            d = ilc.seg_box_dist(ilc.to_local(s['a'][0], s['a'][1], t['x'], t['z'], t['yaw']),
                                 ilc.to_local(s['b'][0], s['b'][1], t['x'], t['z'], t['yaw']), hx, hz) - s['r']
            if d < best.get(s['id'], {'d': 9})['d']:
                best[s['id']] = {'d': d, 'kind': s['kind'], 'frame': t['frame'], 'x': round(t['x'], 3), 'z': round(t['z'], 3),
                                 'obstacle': [s['a'], s['b']]}
        for e in w.disks:
            if 'box' in e:
                bw, bd = e['box'][0] / 2, e['box'][1] / 2
                box = [(e['x'] - bw, e['z'] - bd), (e['x'] + bw, e['z'] - bd), (e['x'] + bw, e['z'] + bd), (e['x'] - bw, e['z'] + bd)]
                d = 0.0 if ilc.poly_overlap(car, box) else min(
                    min(ilc.pt_seg_dist(p, box[i], box[(i + 1) % 4]) for p in car for i in range(4)),
                    min(ilc.pt_seg_dist(q, car[i], car[(i + 1) % 4]) for q in box for i in range(4)))
            else:
                u, ww = ilc.to_local(e['x'], e['z'], t['x'], t['z'], t['yaw'])
                d = ilc.pt_rect_dist(u, ww, hx, hz) - e['r']
            if d < best.get(e['id'], {'d': 9})['d']:
                best[e['id']] = {'d': d, 'kind': e['kind'], 'frame': t['frame'], 'x': round(t['x'], 3), 'z': round(t['z'], 3),
                                 'obstacle': [e['x'], e['z']]}
    rows = sorted(({'id': k, **v} for k, v in best.items()), key=lambda r: r['d'])[:top]
    return rows


def dist_to_line(px, pz, tr):
    best = float('inf')
    for a, b in zip(tr[:-1], tr[1:]):
        best = min(best, ilc.pt_seg_dist((px, pz), (a['x'], a['z']), (b['x'], b['z'])))
    return best


def gate_lateral(rep, route_point):
    """元の経路の点 route_point のゲートを越えた点の横のずれ(最初の 1 回)。自前の積分の副ステップ(約 1 ms)で求める"""
    r = ilc.check_report(rep)
    g = next((x for x in r['gate_laterals'] if x['route_point'] == route_point), None)
    if g is None:
        return None
    return {'route_point': route_point, 'lateral_m': g['lateral_m'], 'margin_to_edge_m': rep['config']['gateHalfWidth'] - g['lateral_m'],
            'passed': g['passed'], 'method': 'substep'}


def gate_lateral_frames(rep, route_point):
    """(参考)フレームの終わりの姿勢(30 Hz)を直線でつないだときの値。副ステップの値と数 mm 違う"""
    w = ilc.World(rep['world'])
    g0 = (rep.get('context') or {}).get('start_variation', {}).get('gate', 0) or 0
    tr = [{'x': w.spawn['x'], 'z': w.spawn['z']}] + rep['trace']
    n = len(w.targets)
    i = 0
    for a, b in zip(tr[:-1], tr[1:]):
        if i >= n:
            break
        tgt = w.targets[i]
        prev = w.targets[i - 1] if i else w.route[0]
        ln = math.hypot(tgt[0] - prev[0], tgt[1] - prev[1])
        nx, nz = (tgt[0] - prev[0]) / ln, (tgt[1] - prev[1]) / ln
        sa, sb = (a['x'] - tgt[0]) * nx + (a['z'] - tgt[1]) * nz, (b['x'] - tgt[0]) * nx + (b['z'] - tgt[1]) * nz
        if sa < 0 <= sb:
            f = -sa / (sb - sa)
            gx, gz = a['x'] + (b['x'] - a['x']) * f, a['z'] + (b['z'] - a['z']) * f
            lat = abs(-(gx - tgt[0]) * nz + (gz - tgt[1]) * nx)
            orig = (g0 + i + 1) % n
            if orig == route_point:
                return {'route_point': orig, 'lateral_m': lat, 'margin_to_edge_m': rep['config']['gateHalfWidth'] - lat, 'frame': b.get('frame')}
            if lat <= rep['config']['gateHalfWidth']:
                i += 1
    return None


def main():
    out = sys.argv[1]
    rl = {e: load(f'{REP}/rl/final/records/rl_main_{e}_single.json') for e in ('E0', 'E1', 'E2', 'E3', 'E4')}
    teacher = load(f'{REP}/rl/final/records/teacher_E0_single.json')
    res = {'note': '確認の段で計算(review/derived_numbers.py)。trace はフレームの終わりの姿勢(30 Hz)'}
    # 1) segment-11(S-R がぶつかったレール)までのすき間: 最初の直線(x 0〜2.6 m)
    res['rail_segment11_clearance_first_straight'] = {
        'RL_E0': seg_clear(rl['E0'], 'segment-11', 0, 2.6), 'teacher_E0': seg_clear(teacher, 'segment-11', 0, 2.6),
        'RL_E0_whole_lap_min_clearance_any_m': ilc.check_report(rl['E0'])['min_clearance_own_m'],
        'teacher_E0_whole_lap_min_clearance_any_m': ilc.check_report(teacher)['min_clearance_own_m']}
    res['first_straight_nearest_obstacles'] = {'definition': '開始から車の x が初めて 2.6 m を超えるまで。車体の長方形と障害物の最小距離',
        'RL_E0': nearest_obstacles_first_straight(rl['E0']), 'teacher_E0': nearest_obstacles_first_straight(teacher)}
    # 2) S-R の RL の線からのずれ
    rows = []
    for lab in ('SR-s0', 'SR-s1'):
        for e in ('E0', 'E1', 'E2', 'E3', 'E4'):
            r = load(f'{REP}/eval/closed_loop/{lab}/{lab}-{e}-single.json')
            tr = r['trace']
            last = tr[-1]
            ce = r['metrics']['collision_events'][0]
            f15 = tr[min(14, len(tr) - 1)]          # trace[14] = フレーム 15 の終わり = 0.5 秒
            rows.append({'run': f'{lab}-{e}', 'collider': ce['collider'], 'kind': ce['kind'], 'crash_time_s': ce['time'],
                         'dev_at_crash_same_start_m': dist_to_line(last['x'], last['z'], rl[e]['trace']),
                         'dev_at_crash_vs_E0_m': dist_to_line(last['x'], last['z'], rl['E0']['trace']),
                         'dev_at_0p5s_same_start_m': dist_to_line(f15['x'], f15['z'], rl[e]['trace']),
                         'time_at_0p5s': f15['time']})
    res['SR_deviation_from_RL_line'] = {'rows': rows,
        'at_crash_same_start_range_m': [min(r['dev_at_crash_same_start_m'] for r in rows), max(r['dev_at_crash_same_start_m'] for r in rows)],
        'at_crash_vs_E0_range_m': [min(r['dev_at_crash_vs_E0_m'] for r in rows), max(r['dev_at_crash_vs_E0_m'] for r in rows)],
        'at_0p5s_same_start_range_m': [min(r['dev_at_0p5s_same_start_m'] for r in rows), max(r['dev_at_0p5s_same_start_m'] for r in rows)]}
    # 3) 経路の点 4 のゲート
    res['gate_point4'] = {
        'teacher_E0': gate_lateral(teacher, 4),
        'ST-s0-E1': gate_lateral(load(f'{REP}/eval/closed_loop/ST-s0/ST-s0-E1-single.json'), 4),
        'ST-s0-E3': gate_lateral(load(f'{REP}/eval/closed_loop/ST-s0/ST-s0-E3-single.json'), 4),
        'EXT-E0': gate_lateral(load(f'{REP}/eval/closed_loop/EXT/EXT-E0-single.json'), 4),
        'RL_E0': gate_lateral(rl['E0'], 4)}
    # 4) 付け直しの舵(takeover.json の per_frame: [フレーム, 使った舵, RL の舵, 使った速さ, RL の速さ])
    tk = load(f'{REP}/eval/takeover.json')
    row = next(r for r in tk['rows'] if r['name'] == 'SR-s0-E0-single')
    pf = [q for q in row['relabel']['per_frame'] if 15 <= q[0] <= 26]
    res['SR_s0_E0_relabel_steer_frames_15_26'] = {'used_rad': [q[1] for q in pf], 'rl_rad': [q[2] for q in pf],
                                                  'rl_minus_used_range_rad': [min(q[2] - q[1] for q in pf), max(q[2] - q[1] for q in pf)]}
    # 5) S-R の衝突の種類(12 本)
    kinds = {}
    for f in sorted(glob.glob(f'{REP}/eval/closed_loop/SR-s*/SR-s?-E?-single.json')):
        r = load(f)
        ce = r['metrics']['collision_events'][0]
        kinds[os.path.basename(f)[:-5]] = {'kind': ce['kind'], 'collider': ce['collider'], 'time_s': ce['time'], 'passed_gates': r['metrics']['passed_gates']}
    res['SR_collisions_12'] = kinds
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('SR_collisions_12',)}, ensure_ascii=False, indent=1)[:6000])


if __name__ == '__main__':
    main()
