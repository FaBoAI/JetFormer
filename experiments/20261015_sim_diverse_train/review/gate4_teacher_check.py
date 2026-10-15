#!/usr/bin/env python3
"""確認の段(review、2026-10-14): 先生の線が点 4 のゲートをどこで通るかを、記録したデータ(frames.jsonl の pose_pre_action)から
線形補間で測る(evaluation.md 5 章は「sim が数えた最初のフレームの位置」で測ったおよその値)。読むだけ。

    python3 -I gate4_teacher_check.py
出力: /home/jetson/jetformer_reports/sim_dart_20261014/review/gate4_teacher_check.json
"""
import glob, json, math, statistics
ROUTE = json.load(open('/home/jetson/jetformer_reports/sim_dart_20261014/eval/closed_loop/A1-s0/A1-s0-E0-single.json'))['world']['route']
SETS = {
    'T_margin(D2)': '/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-margin-line-10min_20261014-070934_48d04e6b',
    'T_fast(A0 のデータ)': '/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0',
    'T_fast + DART(D1)': '/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-dart-fast-10min_20261014-062216_4446c73d',
}
p3, p4, p5 = ROUTE[3], ROUTE[4], ROUTE[5]
dx, dz = p4[0] - p3[0], p4[1] - p3[1]; n = math.hypot(dx, dz); ux, uz = dx / n, dz / n
lx, lz = -uz, ux
if (p5[0] - p4[0]) * lx + (p5[1] - p4[1]) * lz > 0:
    lx, lz = -lx, -lz            # 曲がりの内側(点 5 の側)を負

def first_cross(poses):
    prev = None
    for q in poses:
        a = (q[0] - p4[0]) * ux + (q[1] - p4[1]) * uz
        if prev is not None and prev[0] < 0 <= a and abs(a - prev[0]) < 0.5:
            w = -prev[0] / (a - prev[0])
            x = prev[1][0] + w * (q[0] - prev[1][0]); z = prev[1][1] + w * (q[1] - prev[1][1])
            return (x - p4[0]) * lx + (z - p4[1]) * lz
        prev = (a, q)
    return None

out = {}
for name, ds in SETS.items():
    lats = []
    for p in sorted(glob.glob(ds + '/simulation/scenes/*/frames.jsonl')):
        poses = []
        for line in open(p, encoding='utf-8'):
            f = json.loads(line); q = f['pose_pre_action']; poses.append((q['x'], q['z']))
        lat = first_cross(poses)
        if lat is not None:
            lats.append(lat)
    m = [0.65 - abs(v) for v in lats]
    out[name] = {'scenes': len(lats), 'lateral_m_min_max': [round(min(lats), 3), round(max(lats), 3)] if lats else None,
                 'margin_cm_min_median_max': [round(min(m) * 100, 1), round(statistics.median(m) * 100, 1), round(max(m) * 100, 1)] if m else None,
                 'missed': sum(1 for v in lats if abs(v) > 0.65)}
    print(name, out[name])
json.dump(out, open('/home/jetson/jetformer_reports/sim_dart_20261014/review/gate4_teacher_check.json', 'w'), ensure_ascii=False, indent=1)
