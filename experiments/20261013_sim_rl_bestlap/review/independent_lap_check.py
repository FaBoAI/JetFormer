"""確認の段(review、2026-10-14): 保存したレポート(sim.report() + world)の周が本物かを、Python の自前の積分と判定で数え直す

    python3 -I independent_lap_check.py <出力 JSON> <レポート.json> [...]

JS(sim-core.js)のコードは使わない。レポートの world・config・actions だけを読み、
  1. 物理を自前で積分する(planar の自転車モデル。前後は real-drive-response-estimate-v1、横は μg/|v| でヨーレートを頭打ち、
     副ステップ min(substep, 0.012 / 最大の動きの速さ, 残り))。衝突は自前の幾何(車体の長方形と、半径つき線分・円・脚の箱・外枠)
  2. ゲート: 目標 i = 回した経路の i+1 番目の点。前の目標から目標への向きの線を、符号つき距離が負 → 0 以上に越えた点の横のずれ ≤ 半幅(0.65 m)
  3. 黒シート: 必須の横断の線(z = g.z、向き g.direction)を越えた点で、車体の幅(向きで変わる)がすべて開口 [minX, maxX] の中、
     かつその点が黒シートの面の上。周の中で数え、「ちょうど 1 回」を求める(sim は 2 回目を数えないので、ここでは数える)
  4. 周: 単周は 33 ゲート・距離 ≥ 半周・黒シート ≥ 1 回で完走。時間は完走した副ステップの終わり。連続周回は周ごとに
     ゲートの線を越えた瞬間で補間した時間(sim の lap_events と同じ定義)
  5. sim のレポートの軌跡(trace)と自前の積分を比べる(差の最大)。周の数・時間・衝突も比べる
ベストラップの定義(protocol best_lap_definition): 33/33 ゲート・黒シートの有効な横断ちょうど 1 回・衝突 0・距離の条件。
"""
import json
import math
import sys

G = 9.80665
NEAR = 0.3          # これより遠い障害物は厳密な距離を計算しない(すき間は NEAR 以上とだけ数える)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def approach(v, target, dmax):
    return v + clamp(target - v, -dmax, dmax)


# ---------------------------------------------------------------- 幾何(自前)
def corners(x, z, yaw, hx, hz):
    """車体の 4 隅(世界座標)。前 = (cos yaw, −sin yaw)、左 = (−sin yaw, −cos yaw)"""
    fx, fz = math.cos(yaw), -math.sin(yaw)
    lx, lz = -math.sin(yaw), -math.cos(yaw)
    return [(x + sx * hx * fx + sz * hz * lx, z + sx * hx * fz + sz * hz * lz) for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def to_local(px, pz, x, z, yaw):
    """世界の点 → 車の座標(前 u、左 w)"""
    dx, dz = px - x, pz - z
    return dx * math.cos(yaw) - dz * math.sin(yaw), -dx * math.sin(yaw) - dz * math.cos(yaw)


def pt_rect_dist(u, w, hx, hz):
    return math.hypot(max(0.0, abs(u) - hx), max(0.0, abs(w) - hz))


def pt_seg_dist(p, a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dz * dz
    t = clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / L, 0.0, 1.0) if L > 0 else 0.0
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dz)


def seg_hits_box(a, b, hx, hz):
    """車の座標の線分 a-b が箱 [−hx, hx] × [−hz, hz] に入るか(パラメーターの区間を狭める方法)"""
    t0, t1 = 0.0, 1.0
    for o, d, h in ((a[0], b[0] - a[0], hx), (a[1], b[1] - a[1], hz)):
        if abs(d) < 1e-15:
            if abs(o) > h:
                return False
            continue
        lo, hi = sorted(((-h - o) / d, (h - o) / d))
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return False
    return True


def seg_box_dist(a, b, hx, hz):
    if seg_hits_box(a, b, hx, hz):
        return 0.0
    d = min(pt_rect_dist(a[0], a[1], hx, hz), pt_rect_dist(b[0], b[1], hx, hz))
    for c in ((hx, hz), (hx, -hz), (-hx, hz), (-hx, -hz)):
        d = min(d, pt_seg_dist(c, a, b))
    return d


def poly_overlap(P, Q):
    """凸多角形 P・Q が重なるか(分離軸。境界で触れるのも重なりとする)"""
    for poly in (P, Q):
        n = len(poly)
        for i in range(n):
            ex, ez = poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1]
            ax, az = -ez, ex
            pa = [p[0] * ax + p[1] * az for p in P]
            qa = [q[0] * ax + q[1] * az for q in Q]
            if max(pa) < min(qa) - 1e-12 or max(qa) < min(pa) - 1e-12:
                return False
    return True


class World(object):
    """レポートの world を sim と同じ約束で整える(線分の半径の既定 0.01・高さ 0 は除く・柱の脚は 0.16 × 0.15 m の箱)"""

    def __init__(self, w):
        self.segments = []
        for i, s in enumerate(w.get('segments') or []):
            if s.get('height', 0.1) <= 0:
                continue
            self.segments.append({'id': 'segment-%d' % i, 'a': tuple(s['a'][:2]), 'b': tuple(s['b'][:2]), 'r': s.get('radius', 0.01),
                                  'kind': s.get('kind', 'rail')})
        self.disks = []
        for i, d in enumerate(w.get('disks') or []):
            if d.get('height', 0.1) <= 0:
                continue
            e = {'id': 'disk-%d' % i, 'x': d['x'], 'z': d['z'], 'r': d.get('radius', 0.02), 'kind': d.get('kind', 'post')}
            if e['kind'] == 'post-foot':
                e['box'] = (0.16, 0.15)
            self.disks.append(e)
        self.surfaces = [s for s in (w.get('surfaces') or [])]
        route = [tuple(p[:2]) for p in w['route']]
        if len(route) > 1 and math.hypot(route[0][0] - route[-1][0], route[0][1] - route[-1][1]) < 1e-8:
            route.pop()
        self.route = route
        self.targets = route[1:] + [route[0]]
        self.route_length = sum(math.hypot(self.targets[i][0] - (self.targets[i - 1] if i else route[0])[0],
                                           self.targets[i][1] - (self.targets[i - 1] if i else route[0])[1]) for i in range(len(self.targets)))
        self.bounds = w['bounds']
        self.crossings = w.get('requiredCrossings') or []
        self.spawn = w['spawn']
        # 遠い障害物を飛ばすための外接円(中心・半径)。厳密な判定は近いものだけ(NEAR より遠いものはすき間 ≥ NEAR)
        for s in self.segments:
            s['c'] = ((s['a'][0] + s['b'][0]) / 2, (s['a'][1] + s['b'][1]) / 2)
            s['ext'] = math.hypot(s['b'][0] - s['a'][0], s['b'][1] - s['a'][1]) / 2 + s['r']
        for e in self.disks:
            e['c'] = (e['x'], e['z'])
            e['ext'] = math.hypot(e['box'][0] / 2, e['box'][1] / 2) if 'box' in e else e['r']

    def mu(self, x, z, fm):
        mu = 0.8
        for s in self.surfaces:
            if abs(x - s['x']) <= s['w'] / 2 and abs(z - s['z']) <= s['d'] / 2:
                mu = s.get('mu', 0.8)
        return mu * fm

    def surface_id(self, x, z):
        sid = 'default'
        for s in self.surfaces:
            if abs(x - s['x']) <= s['w'] / 2 and abs(z - s['z']) <= s['d'] / 2:
                sid = s.get('id')
        return sid

    def hit(self, x, z, yaw, hx, hz):
        """衝突した障害物(無ければ None)と、そのときの最小のすき間 [m]"""
        best = (None, NEAR)
        car_r = math.hypot(hx, hz)
        for s in self.segments:
            if math.hypot(x - s['c'][0], z - s['c'][1]) - s['ext'] - car_r > NEAR:
                continue
            d = seg_box_dist(to_local(s['a'][0], s['a'][1], x, z, yaw), to_local(s['b'][0], s['b'][1], x, z, yaw), hx, hz) - s['r']
            if d <= 1e-9 and best[0] is None:
                best = (s, d)
            if best[0] is None and d < best[1]:
                best = (None, d)
        car = corners(x, z, yaw, hx, hz)
        for e in self.disks:
            if math.hypot(x - e['c'][0], z - e['c'][1]) - e['ext'] - car_r > NEAR:
                continue
            if 'box' in e:
                bw, bd = e['box'][0] / 2, e['box'][1] / 2
                box = [(e['x'] - bw, e['z'] - bd), (e['x'] + bw, e['z'] - bd), (e['x'] + bw, e['z'] + bd), (e['x'] - bw, e['z'] + bd)]
                if poly_overlap(car, box):
                    if best[0] is None:
                        best = (e, 0.0)
                    continue
                # 箱どうしの距離(角と辺の距離の最小)
                d = min(min(pt_seg_dist(p, box[i], box[(i + 1) % 4]) for p in car for i in range(4)),
                        min(pt_seg_dist(q, car[i], car[(i + 1) % 4]) for q in box for i in range(4)))
            else:
                u, w = to_local(e['x'], e['z'], x, z, yaw)
                d = pt_rect_dist(u, w, hx, hz) - e['r']
                if d <= 1e-9 and best[0] is None:
                    best = (e, d)
            if best[0] is None and d < best[1]:
                best = (None, d)
        b = self.bounds
        for cx, cz in car:
            if cx < b['minX'] or cx > b['maxX'] or cz < b['minZ'] or cz > b['maxZ']:
                return ({'id': 'bounds', 'kind': 'bounds'}, 0.0)
            if best[0] is None:
                best = (None, min(best[1], cx - b['minX'], b['maxX'] - cx, cz - b['minZ'], b['maxZ'] - cz))
        return best


# ---------------------------------------------------------------- 積分と判定(自前)
def check_report(rep):
    cfg, world = rep['config'], World(rep['world'])
    continuous = cfg.get('episodeMode') == 'continuous-laps-v1'
    L, Wd, wb = cfg['vehicleLength'], cfg['vehicleWidth'], cfg['wheelbase']
    hx, hz = L / 2, Wd / 2
    vmax, smax, dt, sub, tlim = cfg['maxSpeed'], cfg['maxSteer'], cfg['dt'], cfg['substep'], cfg['timeLimit']
    fm = cfg.get('frictionMultiplier', 1.0)
    rc = cfg.get('longitudinalModel') == 'real-drive-response-estimate-v1'
    radius = math.hypot(L, Wd) / 2
    sp = world.spawn
    x, z, yaw, v, st, t = sp['x'], sp['z'], wrap(sp['yaw']), 0.0, 0.0, 0.0
    route_i, dist_total = 0, 0.0
    lap_start_t, lap_start_d = 0.0, 0.0
    lap_gates, lap_cross, laps = 0, [], []
    all_cross, gate_lat = [], []
    terminal, reason, collisions, min_clear = False, None, [], float('inf')
    my_trace = []
    n_t = len(world.targets)
    start_gate = ((rep.get('context') or {}).get('start_variation') or {}).get('gate', 0) or 0
    for k, act in enumerate(rep['actions']):
        if terminal:
            break
        steer_cmd = clamp(act['steering'], -smax, smax)
        speed_cmd = clamp(act['speed'], -vmax, vmax)
        remaining = min(dt, tlim - t)
        el = 0.0
        while el < remaining - 1e-12 and not terminal:
            rate_max = vmax + radius * vmax / wb * math.tan(smax)
            h = min(sub, 0.012 / rate_max, remaining - el)
            mu = world.mu(x, z, fm)
            if rc:
                if v >= 0 and speed_cmd > v:
                    r = min(cfg['driveAccelerationCap'], mu * G, (speed_cmd - v) / cfg['driveTau'])
                else:
                    r = cfg['brakeDeceleration']
                nv = approach(v, speed_cmd, r * h)
            else:
                nv = approach(v, speed_cmd, cfg['acceleration'] * h)
            ns = approach(st, steer_cmd, cfg['steerRate'] * h)
            raw = nv / wb * math.tan(ns)
            lim = mu * G / max(abs(nv), 0.01)
            om = clamp(raw, -lim, lim)
            ym = yaw + om * h / 2
            tx, tz, tyaw = x + nv * math.cos(ym) * h, z - nv * math.sin(ym) * h, wrap(yaw + om * h)
            obj, clear = world.hit(tx, tz, tyaw, hx, hz)
            if obj is not None:
                collisions.append({'time': t + el + h, 'frame': k, 'collider': obj['id'], 'kind': obj.get('kind'), 'x': tx, 'z': tz})
                v, st = 0.0, ns
                if cfg.get('stopOnCollision', True):
                    terminal, reason = True, 'collision'
            else:
                min_clear = min(min_clear, clear)
                px, pz, pyaw = x, z, yaw
                x, z, yaw, v, st = tx, tz, tyaw, nv, ns
                seg = math.hypot(x - px, z - pz)
                dist_total += seg
                tnow = t + el + h
                # 黒シート(数えるのは自前の約束: 周の中の有効な横断をすべて)
                for g in world.crossings:
                    a, b = (pz - g['z']) * g['direction'], (z - g['z']) * g['direction']
                    if not (a < 0 <= b):
                        continue
                    f = -a / (b - a)
                    cx = px + (x - px) * f
                    cyaw = pyaw + wrap(yaw - pyaw) * f
                    half = hx * abs(math.cos(cyaw)) + hz * abs(math.sin(cyaw))
                    ok_open = cx - half >= g['minX'] - 1e-12 and cx + half <= g['maxX'] + 1e-12
                    ok_surf = (not g.get('surface_id')) or any(
                        s.get('id') == g['surface_id'] and abs(cx - s['x']) <= s['w'] / 2 and abs(g['z'] - s['z']) <= s['d'] / 2
                        for s in world.surfaces)
                    ev = {'id': g['id'], 'time': tnow - (1 - f) * h, 'x': cx, 'valid': bool(ok_open and ok_surf),
                          'margin_m': min(cx - half - g['minX'], g['maxX'] - cx - half)}
                    all_cross.append(ev)
                    if ev['valid']:
                        lap_cross.append(ev)
                # ゲート
                if route_i < n_t:
                    tgt = world.targets[route_i]
                    prev = world.targets[route_i - 1] if route_i else world.route[0]
                    ln = math.hypot(tgt[0] - prev[0], tgt[1] - prev[1])
                    nx, nz = (tgt[0] - prev[0]) / ln, (tgt[1] - prev[1]) / ln
                    sa = (px - tgt[0]) * nx + (pz - tgt[1]) * nz
                    sb = (x - tgt[0]) * nx + (z - tgt[1]) * nz
                    if sa < 0 <= sb:
                        f = -sa / (sb - sa)
                        gx, gz = px + (x - px) * f, pz + (z - pz) * f
                        lat = abs(-(gx - tgt[0]) * nz + (gz - tgt[1]) * nx)
                        gate_lat.append({'lap': len(laps) + 1, 'target': route_i, 'lateral_m': lat, 'passed': lat <= cfg['gateHalfWidth']})
                        if lat <= cfg['gateHalfWidth']:
                            route_i += 1
                            lap_gates += 1
                            if continuous and lap_gates == n_t:
                                end_t = tnow - (1 - f) * h
                                end_d = dist_total - (1 - f) * seg
                                ids = {g['id'] for g in world.crossings}
                                crossed = ids <= {e['id'] for e in lap_cross}
                                enough = end_d - lap_start_d >= max(L, world.route_length * 0.5)
                                n_valid = len([e for e in lap_cross])
                                lap_col = len([c for c in collisions if c['time'] > lap_start_t])
                                laps.append({'lap': len(laps) + 1, 'time_s': end_t - lap_start_t, 'gates': lap_gates, 'valid_crossings': n_valid,
                                             'collisions': lap_col, 'distance_ok': enough,
                                             'best_lap_valid': bool(crossed and enough and n_valid == 1 and lap_col == 0)})
                                if not crossed or not enough:
                                    terminal, reason, v = True, ('lap_distance_insufficient' if crossed else 'required_crossing_missing'), 0.0
                                lap_start_t, lap_start_d = end_t, end_d
                                route_i, lap_gates, lap_cross = 0, 0, []
                            elif not continuous and route_i == n_t and dist_total >= max(L, world.route_length * 0.5):
                                ids = {g['id'] for g in world.crossings}
                                crossed = ids <= {e['id'] for e in lap_cross}
                                terminal, reason, v = True, ('route_complete' if crossed else 'required_crossing_missing'), 0.0
                                laps.append({'lap': 1, 'time_s': tnow, 'gates': lap_gates, 'valid_crossings': len(lap_cross),
                                             'collisions': len(collisions), 'distance_ok': True,
                                             'best_lap_valid': bool(crossed and len(lap_cross) == 1 and not collisions)})
            el += h
        t = min(tlim, t + el)
        if not terminal and t >= tlim - 1e-10:
            terminal = True
            reason = 'collection_duration_complete' if continuous else 'time_limit'
            if not continuous:
                v = 0.0
        my_trace.append((x, z, yaw, v, st))
    # sim のレポートと比べる
    tr = rep['trace']
    n = min(len(tr), len(my_trace))
    dmax = 0.0
    for i in range(n):
        a, b = my_trace[i], tr[i]
        dmax = max(dmax, abs(a[0] - b['x']), abs(a[1] - b['z']), abs(wrap(a[2] - b['yaw'])), abs(a[3] - b['speed']), abs(a[4] - b['steering']))
    m = rep['metrics']
    sim_laps = [l['elapsed'] for l in (m.get('lap_events') or [])] if continuous else (
        [m['elapsed']] if rep['outcome']['reason'] == 'route_complete' else [])
    my_laps = [l['time_s'] for l in laps]
    agree = (len(tr) == len(my_trace) and reason == rep['outcome']['reason'] and len(collisions) == m['collisions']
             and len(sim_laps) == len(my_laps) and all(abs(a - b) < 1e-6 for a, b in zip(sim_laps, my_laps)) and dmax < 1e-6)
    return {'frames': len(rep['actions']), 'reason_own': reason, 'reason_sim': rep['outcome']['reason'],
            'collisions_own': collisions, 'collisions_sim': m['collisions'], 'laps_own': laps, 'laps_sim_s': sim_laps,
            'max_trace_diff': dmax, 'agree_with_sim': bool(agree), 'min_clearance_own_m': min_clear,
            'gate_lateral_max_m': max((g['lateral_m'] for g in gate_lat if g['passed']), default=None),
            'gate_misses': [g for g in gate_lat if not g['passed']], 'crossing_events': all_cross,
            'gate_laterals': [{'lap': g['lap'], 'target': g['target'], 'route_point': (start_gate + g['target'] + 1) % n_t,
                               'lateral_m': g['lateral_m'], 'passed': g['passed']} for g in gate_lat],
            'start': rep.get('context', {}).get('start_variation')}


def main():
    out, files = sys.argv[1], sys.argv[2:]
    res = {}
    for f in files:
        rep = json.load(open(f, encoding='utf-8'))
        if not isinstance(rep, dict) or not all(k in rep for k in ('config', 'world', 'actions', 'trace')):
            print('%-34s 走行のレポートではないので飛ばします' % f.rsplit('/', 1)[-1][:34])
            continue
        r = check_report(rep)
        r['file'] = f
        res[f.rsplit('/', 1)[-1].rsplit('.json', 1)[0]] = r
        print('%-34s own=%-28s sim=%-28s laps_own=%s valid=%s trace_diff=%.2e agree=%s clear=%.3f' % (
            f.rsplit('/', 1)[-1][:34], r['reason_own'], r['reason_sim'], [round(l['time_s'], 6) for l in r['laps_own']],
            [l['best_lap_valid'] for l in r['laps_own']], r['max_trace_diff'], r['agree_with_sim'], r['min_clearance_own_m']))
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
