#!/usr/bin/env python3
"""第 3〜4 回の審査(2026-10-15): 昇格の段のばらつき(r3/variation.json、rl/variation_metrics.mjs)を、そのコードを使わずに別のやり方で数え直す。
review/indep_variation.py(2026-10-14)の写しに、経路いろいろ・r4 の見本と「線のばらつき」の別の数え方を足した。

- VM1 と同じ式(時刻でそろえた舵 PWM の |平均からの差| の平均 µs)を自前で
- 訪れた状態のマス(参照ループへの射影を使わない): 開始から 3 m 以降のフレームを (x 10 cm × z 10 cm × 向き 0.2 rad × 速さ 0.5 m/s) に入れた数
  (1 群 15 シーン × 540 フレームに間引き。5 モード合わせは 1 シーン 108 フレーム)
- 線のばらつき(RL の参照ループを使わない): 前の教師の 1 シーン目の 1 周の軌跡を 5 cm おきの折れ線にし、各フレームをいちばん近い点に
  載せて符号つきの横ずれを出す。シーンごとに「3 m 以降に最初に通った各点」の横ずれを取り、点ごとにシーンどうしの |平均からの差| の平均 → 全点の平均
    python3 -I indep_variation_r4.py > 結果.json
"""
import glob
import json
import math
import os

import numpy as np

D = '/home/jetson/jetformer/data/datasets'
TEACHER = 'jetformer-simulator-id8-20min_20261012-150956_b55616cf'
GROUPS = {'teacher': TEACHER, 'fastest': 'sim-rl-fastest_20261014-1125', 'no_wall': 'sim-rl-no_wall_20261015-0302',
          'recover': 'sim-rl-recover_20261015-0302', 'speed': 'sim-rl-speed_20261014-1125', 'line': 'sim-rl-line_20261015-0302',
          'no_wall_r2(前の版)': 'sim-rl-no_wall_20261014-1125', 'recover_r1(前の版)': 'sim-rl-recover_20261014-1125'}
FIVE = ('fastest', 'no_wall', 'recover', 'speed', 'line')


def load(ds, n=15):
    scenes = sorted(glob.glob(os.path.join(D, ds, 'simulation', 'scenes', '*', 'frames.jsonl')))[:n]
    return [[json.loads(l) for l in open(p, encoding='utf-8')] for p in scenes]


def vm1(scenes):
    T = min(len(s) for s in scenes)
    tot = 0.0
    for t in range(T):
        v = [s[t]['applied_pwm_us'][0] for s in scenes]
        m = sum(v) / len(v)
        tot += sum(abs(x - m) for x in v) / len(v)
    return tot / T


def after3m_idx(fr):
    d, out = 0.0, []
    for i in range(1, len(fr)):
        a, b = fr[i - 1]['pose_pre_action'], fr[i]['pose_pre_action']
        d += math.hypot(b['x'] - a['x'], b['z'] - a['z'])
        if d >= 3.0:
            out.append(i)
    return out


def cells(scenes, per_scene):
    S = set()
    for fr in scenes:
        f = after3m_idx(fr)
        step = max(1, len(f) / per_scene)
        k = 0.0
        while int(k) < len(f):
            i = f[int(k)]
            p = fr[i]['pose_pre_action']
            sp = fr[i - 1]['state_post_action']['speed'] if i > 0 else 0.0
            yaw = (p['yaw'] + math.pi) % (2 * math.pi)
            S.add((int(math.floor(p['x'] / .1)), int(math.floor(p['z'] / .1)), int(yaw // .2), int(sp // .5)))
            k += step
    return len(S)


def teacher_polyline(tscene):
    """前の教師の 1 シーン目の、開始から 1 周(元の位置に戻るまで)の軌跡を 5 cm おきに"""
    P = np.array([[f['pose_pre_action']['x'], f['pose_pre_action']['z']] for f in tscene])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    # 1 周 = 開始から 10 m 以上走ったあと、開始の 0.3 m 以内に最初に戻るところ
    back = [i for i in range(len(P)) if d[i] > 10 and np.hypot(*(P[i] - P[0])) < 0.3]
    end = back[0] if back else len(P) - 1
    s = np.arange(0, d[end], 0.05)
    x = np.interp(s, d[:end + 1], P[:end + 1, 0])
    z = np.interp(s, d[:end + 1], P[:end + 1, 1])
    Q = np.stack([x, z], 1)
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    return Q, N, float(d[end])


def lateral_profiles(scenes, Q, N):
    """シーンごとに、3 m 以降に最初に通った各点の符号つきの横ずれ(点の数 = len(Q)。通らない点は nan)"""
    out = []
    for fr in scenes:
        idx = after3m_idx(fr)
        P = np.array([[fr[i]['pose_pre_action']['x'], fr[i]['pose_pre_action']['z']] for i in idx])
        prof = np.full(len(Q), np.nan)
        for p in P:
            dd = np.hypot(Q[:, 0] - p[0], Q[:, 1] - p[1])
            j = int(np.argmin(dd))
            if np.isnan(prof[j]) and dd[j] < 0.6:
                prof[j] = float(np.dot(p - Q[j], N[j]))
        out.append(prof)
    return np.array(out)


def spread(profs):
    """点ごとにシーンどうしの |平均からの差| の平均(3 シーン以上が通った点だけ)→ 全点の平均 [m]"""
    vals = []
    for j in range(profs.shape[1]):
        c = profs[:, j]
        c = c[~np.isnan(c)]
        if len(c) >= 3:
            vals.append(float(np.mean(np.abs(c - c.mean()))))
    return float(np.mean(vals)), len(vals)


def main():
    loaded = {g: load(ds) for g, ds in GROUPS.items()}
    Q, N, lap_len = teacher_polyline(loaded['teacher'][0])
    res = {'reference_polyline': {'points': len(Q), 'length_m': round(lap_len, 2), 'from': TEACHER + ' の 1 シーン目の 1 周'}}
    per_scene = 540
    profs = {}
    for g, sc in loaded.items():
        profs[g] = lateral_profiles(sc, Q, N)
        sp, nb = spread(profs[g])
        res[g] = {'dataset': GROUPS[g], 'scenes': len(sc), 'VM1_us_own': round(vm1(sc), 3), 'state_cells_own': cells(sc, per_scene),
                  'lateral_spread_own_cm': round(100 * sp, 3), 'lateral_points_used': nb}
    pooled = [fr for g in FIVE for fr in loaded[g]]
    sp, nb = spread(np.concatenate([profs[g] for g in FIVE]))
    res['pooled_5modes'] = {'scenes': len(pooled), 'state_cells_own': cells(pooled, per_scene / 5), 'lateral_spread_own_cm': round(100 * sp, 3),
                            'lateral_points_used': nb}
    t = res['teacher']
    for g in list(res):
        if isinstance(res[g], dict) and 'state_cells_own' in res[g]:
            res[g]['state_cells_ratio_vs_teacher'] = round(res[g]['state_cells_own'] / t['state_cells_own'], 2)
            res[g]['lateral_spread_ratio_vs_teacher'] = round(res[g]['lateral_spread_own_cm'] / t['lateral_spread_own_cm'], 2)
    print(json.dumps(res, ensure_ascii=False, indent=1))


main()
