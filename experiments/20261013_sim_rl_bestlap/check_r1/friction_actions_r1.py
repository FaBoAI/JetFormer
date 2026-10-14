# 摩擦円の超え方と操作の荒さを、保存した周の記録(フレームごとの軌跡と操作)から数える(2026-10-13、確認の段 r1)
#   python -I check_r1/friction_actions_r1.py
# sim-core は縦(加速 6 m/s²・ブレーキ 1.2 m/s²)と横(μg)を別々に頭打ちにする(摩擦円ではない)。
# 縦と横の合成 √(a_long² + a_lat²) が μg を超える時間を、RL・教師・学習済み ONNX で比べる。
import json, math, os, sys
REC = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/final/records'
OUT = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl_check_r1/friction_actions.json'
G = 9.80665
def wrap(a): return math.atan2(math.sin(a), math.cos(a))
def pct(a, p):
    b = sorted(a); return b[min(len(b) - 1, int(p * (len(b) - 1) + .5))] if b else None
res = {}
for name in ['rl_main_STD_single', 'rl_main_STD_continuous', 'teacher_STD_single', 'teacher_STD_continuous', 'learned_onnx_E0_single']:
    r = json.load(open(f'{REC}/{name}.json'))
    mu = {s['id']: s['mu'] for s in r['world']['surfaces']}
    tr, act, dt = r['trace'], r['actions'], r['config']['dt']
    prev = {'x': r['world']['spawn']['x'], 'z': r['world']['spawn']['z'], 'yaw': r['world']['spawn']['yaw'], 'speed': 0.0}
    over0 = over10 = over25 = moving = 0; ratios = []; lat_ratio = []; long_pos_at_lat_limit = 0
    for k, p in enumerate(tr):
        v0, v1 = prev['speed'], p['speed']
        if p is tr[-1] and r['outcome']['reason'] == 'route_complete':
            break   # 完走のフレームは sim が速さを 0 にするので除く
        a_long = (v1 - v0) / dt
        w = wrap(p['yaw'] - prev['yaw']) / dt
        a_lat = (v0 + v1) / 2 * w
        m = mu.get(p.get('surface'), .8) * G
        if (v0 + v1) / 2 > .3:
            moving += 1
            q = math.hypot(a_long, a_lat) / m
            ratios.append(q); lat_ratio.append(abs(a_lat) / m)
            over0 += q > 1.0 + 1e-6; over10 += q > 1.10; over25 += q > 1.25
            if abs(a_lat) / m > .97 and a_long > .5: long_pos_at_lat_limit += 1
        prev = p
    # 操作
    st = [a['steering'] for a in act]; sp = [a['speed'] for a in act]
    rate = [abs(st[i] - st[i - 1]) / dt for i in range(1, len(st))]
    flips = sum(1 for i in range(1, len(st)) if abs(st[i]) >= .494 and abs(st[i - 1]) >= .494 and st[i] * st[i - 1] < 0)
    # 舵の指令が端から反対の端へ 5 フレーム(0.17 秒)以内に移る回数
    edge_idx = [(i, 1 if s > 0 else -1) for i, s in enumerate(st) if abs(s) >= .494]
    fast_switch = sum(1 for (i, a), (j, b) in zip(edge_idx, edge_idx[1:]) if a != b and j - i <= 5)
    sat_hi = sum(1 for v in sp if v >= 4.45) / len(sp); sat_lo = sum(1 for v in sp if v <= .05) / len(sp)
    trans = sum(1 for i in range(1, len(sp)) if (sp[i] >= 4.45) != (sp[i - 1] >= 4.45) or (sp[i] <= .05) != (sp[i - 1] <= .05))
    res[name] = {'moving_frames': moving,
                 'friction_circle_ratio_p95': pct(ratios, .95), 'friction_circle_ratio_max': max(ratios),
                 'frac_over_mu_g': over0 / moving, 'frac_over_1.10_mu_g': over10 / moving, 'frac_over_1.25_mu_g': over25 / moving,
                 'frac_lat_limit_and_accelerating': long_pos_at_lat_limit / moving,
                 'lateral_ratio_p95': pct(lat_ratio, .95),
                 'steer_cmd_rate_p95': pct(rate, .95), 'steer_cmd_rate_max': max(rate), 'steer_edge_frac': sum(1 for s in st if abs(s) >= .494) / len(st),
                 'steer_edge_to_opposite_edge_within_5_frames': fast_switch, 'steer_edge_sign_flip_in_1_frame': flips,
                 'speed_cmd_at_cap_frac': sat_hi, 'speed_cmd_zero_frac': sat_lo, 'speed_cmd_saturation_transitions_per_s': trans / (len(sp) * dt),
                 'speed_cmd_min': min(sp), 'speed_cmd_max': max(sp), 'steer_cmd_abs_max': max(abs(s) for s in st)}
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}, ensure_ascii=False))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
