#!/usr/bin/env python3
"""シミュレーターのコースと実車のコース図が同じものかを確かめ、区間(HS・LH・R1・GM・DE・SW)をシミュレーターの座標に写す(2026-10-16)

    python -I section_map.py            # → /home/jetson/jetformer_reports/sim_calibration_20261016/section_map.json

読み取りだけ(CPU、数秒)。ユーザーのシミュレーターのファイルは読むだけで変えない。
座標: シミュレーター (x, z) ↔ コース図 (X, Z) は X = x + 5.15、Z = 2.8 − z(course_map.json の meta.frame)。
"""
import hashlib
import json
from pathlib import Path

import numpy as np

COURSE_MAP = Path('/home/jetson/jetformer_reports/overnight_20261011/course/course_map.json')
WORLD = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/sim/world-v3.json')
REF_LOOP = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/rl/reference_loop.json')
OUT = Path('/home/jetson/jetformer_reports/sim_calibration_20261016/section_map.json')
DX, DZ = 5.15, 2.8


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def to_sim(X, Z):
    return np.asarray(X, float) - DX, DZ - np.asarray(Z, float)


def rect_of_poly(poly):
    p = np.asarray(poly, float)
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())


def main():
    cm = json.loads(COURSE_MAP.read_text(encoding='utf-8'))
    w = json.loads(WORLD.read_text(encoding='utf-8'))
    ref = json.loads(REF_LOOP.read_text(encoding='utf-8'))
    out = {'schema': 'jf-sim-calib-section-map/v1', 'frame': 'X = x + 5.15, Z = 2.8 - z(sim → コース図)',
           'sources': {str(COURSE_MAP): sha(COURSE_MAP), str(WORLD): sha(WORLD), str(REF_LOOP): sha(REF_LOOP)}}

    # ---- 1. 路面: シミュレーターの surfaces(中心 x,z・幅 w・奥行き d)をコース図の座標の矩形にして、コース図の areas と比べる
    surf = {}
    for s in w['surfaces']:
        X0, X1 = s['x'] - s['w'] / 2 + DX, s['x'] + s['w'] / 2 + DX
        Z0, Z1 = DZ - (s['z'] + s['d'] / 2), DZ - (s['z'] - s['d'] / 2)
        surf[s['id']] = {'mu_sim': s['mu'], 'course_rect_XZ': [round(X0, 4), round(X1, 4), round(Z0, 4), round(Z1, 4)]}
    areas = cm['areas']
    pairs = {'turf': rect_of_poly(areas['turf_low_mu']['polygon']), 'ptfe': rect_of_poly(areas['ptfe_white_sheet']['polygon']),
             'blue-mat': rect_of_poly(areas['blue_start_mat']['polygon']), 'shortcut': tuple(np.asarray(areas['shortcut']['tape_outline'], float)[[0, 1, 0, 2], [0, 0, 1, 1]].tolist())}
    for k, r in pairs.items():
        a = np.asarray(surf[k]['course_rect_XZ'])
        surf[k]['course_map_rect_XZ'] = [round(x, 4) for x in r]
        surf[k]['max_edge_diff_m'] = round(float(np.max(np.abs(a - np.asarray(r)))), 4)
    out['surfaces'] = surf

    # ---- 2. レール(コース図の rails)とシミュレーターの線分の一致(端点どうし 3 cm 以内)
    segs = [(np.asarray(s['a'], float), np.asarray(s['b'], float), s.get('kind', 'rail')) for s in w['segments']]
    matched, unmatched = 0, []
    for r in cm['rails']:
        a = np.array(to_sim(*r['a'])); b = np.array(to_sim(*r['b']))
        best = min(min(np.linalg.norm(a - s[0]) + np.linalg.norm(b - s[1]), np.linalg.norm(a - s[1]) + np.linalg.norm(b - s[0]))
                   for s in segs)
        if best <= 0.06:
            matched += 1
        else:
            unmatched.append({'rail_XZ': [r['a'], r['b']], 'best_endpoint_sum_m': round(float(best), 3)})
    out['rails'] = {'course_map_rails': len(cm['rails']), 'matched_within_3cm_each_end': matched, 'unmatched': unmatched[:20],
                    'sim_segments': len(segs)}
    # 柱(posts)
    disks = np.array([[d['x'], d['z']] for d in w['disks']], float)
    pm = 0
    for p in cm.get('posts', []):
        xy = p.get('xz') if isinstance(p, dict) else p
        if xy is None:
            continue
        q = np.array(to_sim(*xy))
        if np.min(np.linalg.norm(disks - q, axis=1)) <= 0.03:
            pm += 1
    out['posts'] = {'course_map_posts': len(cm.get('posts', [])), 'matched_within_3cm': pm, 'sim_disks': len(disks)}

    # ---- 3. 区間: コース図の基準線(左回り、s=0 は HS の始まり)をシミュレーターの座標に写し、RL の参照ループ R の各点を最も近い基準線の点の区間にする
    cl = cm['centerline']
    cx, cz = to_sim(np.asarray(cl['xz'])[:, 0], np.asarray(cl['xz'])[:, 1])
    csec = np.asarray(cl['section'])
    cs = np.asarray(cl['s'], float)
    rx, rz, rs = np.asarray(ref['x'], float), np.asarray(ref['z'], float), np.asarray(ref['s'], float)
    d2 = (rx[:, None] - cx[None, :]) ** 2 + (rz[:, None] - cz[None, :]) ** 2
    near = np.argmin(d2, axis=1)
    rsec = csec[near]
    dist = np.sqrt(d2[np.arange(len(rx)), near])
    runs = []
    i0 = 0
    for i in range(1, len(rsec) + 1):
        if i == len(rsec) or rsec[i] != rsec[i0]:
            runs.append({'section': str(rsec[i0]), 'ref_s_start_m': round(float(rs[i0]), 3), 'ref_s_end_m': round(float(rs[i - 1]), 3),
                         'course_s_start_m': round(float(cs[near[i0]]), 3), 'course_s_end_m': round(float(cs[near[i - 1]]), 3),
                         'sim_xz_start': [round(float(rx[i0]), 3), round(float(rz[i0]), 3)],
                         'sim_xz_end': [round(float(rx[i - 1]), 3), round(float(rz[i - 1]), 3)], 'n_points': i - i0})
            i0 = i
    out['reference_loop'] = {'length_m': ref['length_m'], 'n': len(rx), 'max_dist_to_course_centerline_m': round(float(dist.max()), 3),
                             'median_dist_m': round(float(np.median(dist)), 3), 'section_runs': runs}
    # ゲート(33)ごとの区間
    gates = []
    for gi, (g, gs) in enumerate(zip(ref['route'], ref['gate_s'])):
        k = int(np.argmin((cx - g[0]) ** 2 + (cz - g[1]) ** 2))
        gates.append({'gate': gi, 'sim_xz': g, 'ref_s_m': round(float(gs), 3), 'section': str(csec[k]), 'course_s_m': round(float(cs[k]), 3)})
    out['gates'] = gates
    # 区間ごとの路面(R の点の上の路面。最後に一致した矩形 = sim-core.js の surfaceAt と同じ決め方)
    def surface_at(x, z):
        sid = 'default'
        for s in w['surfaces']:
            if abs(x - s['x']) <= s['w'] / 2 and abs(z - s['z']) <= s['d'] / 2:
                sid = s['id']
        return sid
    sec_surf = {}
    for x, z, sec in zip(rx, rz, rsec):
        sec_surf.setdefault(str(sec), {}).setdefault(surface_at(x, z), 0)
        sec_surf[str(sec)][surface_at(x, z)] += 1
    out['section_surfaces_on_reference_loop_points'] = sec_surf
    # 難所(hotspots)をシミュレーターの座標へ
    out['hotspots_sim'] = {h['id']: {'center_sim': [round(float(v), 3) for v in to_sim(*h['center'])],
                                     'contact_sim': [round(float(v), 3) for v in to_sim(*h['contact']['xz'])],
                                     'radius_m': h['radius_m']} for h in cm['hotspots']}
    # 寸法
    out['dimensions'] = {'course_map_W_D_m': [cm['course']['width'], cm['course']['depth']],
                         'sim_bounds': w['bounds'],
                         'sim_bounds_as_course_XZ': [w['bounds']['minX'] + DX, w['bounds']['maxX'] + DX, DZ - w['bounds']['maxZ'], DZ - w['bounds']['minZ']],
                         'sim_route_gates': len(w['route']), 'required_crossing': w.get('requiredCrossings')}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('surfaces', 'rails', 'posts', 'dimensions')}, ensure_ascii=False, indent=1)[:4000])
    print(json.dumps(out['reference_loop'], ensure_ascii=False, indent=1)[:4000])
    print(json.dumps(sec_surf, ensure_ascii=False))
    print([(g['gate'], g['section']) for g in gates])


if __name__ == '__main__':
    main()
