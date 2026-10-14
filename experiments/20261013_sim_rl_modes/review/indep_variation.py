#!/usr/bin/env python3
"""審査の段(2026-10-14): ばらつきを、記録の段のコード(rl/variation_metrics.mjs)を使わずに、記録した frames.jsonl から別のやり方で数え直す。
- VM1 と同じ式(時刻でそろえた舵 PWM の |平均からの差| の平均 µs)を自前で。
- 参照ループへの射影を使わない「訪れた状態のマス」: 開始から 3 m 走ったあとのフレームを (x 10 cm × z 10 cm × 向き 0.2 rad × 速さ 0.5 m/s) のマスに入れた数。
  1 群 15 シーン、各シーンの 3 m 以降のフレームを決まった間引き(群の合計がそろうように)で数える。
- 速さ(3 m 以降の走っているフレームの平均)のシーン間の広がり、シーンの開始ゲート・ずれ。
    python3 -I review/indep_variation.py > 結果.json
"""
import glob
import json
import math
import os
import sys

D = '/home/jetson/jetformer/data/datasets'
GROUPS = {'teacher(前の教師・先頭 15)': 'jetformer-simulator-id8-20min_20261012-150956_b55616cf',
          'fastest': 'sim-rl-fastest_20261014-1125', 'no_wall': 'sim-rl-no_wall_20261014-1125',
          'recover': 'sim-rl-recover_20261014-1125', 'speed': 'sim-rl-speed_20261014-1125'}


def load(ds, n=None):
    scenes = sorted(glob.glob(os.path.join(D, ds, 'simulation', 'scenes', '*', 'frames.jsonl')))
    if n:
        scenes = scenes[:n]
    out = []
    for p in scenes:
        fr = [json.loads(l) for l in open(p, encoding='utf-8')]
        out.append(fr)
    return out


def vm1(scenes):
    T = min(len(s) for s in scenes)
    tot = 0.0
    for t in range(T):
        v = [s[t]['applied_pwm_us'][0] for s in scenes]
        m = sum(v) / len(v)
        tot += sum(abs(x - m) for x in v) / len(v)
    return tot / T


def after3m(fr):
    d, out = 0.0, []
    for i in range(1, len(fr)):
        a, b = fr[i - 1]['pose_pre_action'], fr[i]['pose_pre_action']
        d += math.hypot(b['x'] - a['x'], b['z'] - a['z'])
        if d >= 3.0:
            out.append(fr[i])
    return out


def cells(scenes, per_scene):
    S = set()
    for fr in scenes:
        f = after3m(fr)
        step = max(1, len(f) / per_scene)
        k = 0.0
        while int(k) < len(f):
            q = f[int(k)]
            p, st = q['pose_pre_action'], fr[max(0, q['frame'] - 1)]['state_post_action'] if q['frame'] > 0 else {'speed': 0}
            yaw = (p['yaw'] + math.pi) % (2 * math.pi)
            S.add((int(math.floor(p['x'] / .1)), int(math.floor(p['z'] / .1)), int(yaw // .2), int(st['speed'] // .5)))
            k += step
    return len(S)


def main():
    res = {}
    loaded = {}
    for g, ds in GROUPS.items():
        loaded[g] = load(ds, 15)
    per_scene = 540   # 15 シーン × 540 = 8,100 フレーム(どの群も 3 m 以降にこれ以上ある)
    for g, sc in loaded.items():
        sp = []
        for fr in sc:
            f = after3m(fr)
            v = [fr[q['frame'] - 1]['state_post_action']['speed'] for q in f if q['frame'] > 0]
            v = [x for x in v if x > .3]
            sp.append(sum(v) / len(v))
        m = sum(sp) / len(sp)
        st0 = [fr[0]['pose_pre_action'] for fr in sc]
        res[g] = {'scenes': len(sc), 'VM1_us_own': round(vm1(sc), 3), 'state_cells_own': cells(sc, per_scene),
                  'mean_speed_per_scene_mps': {'min': round(min(sp), 3), 'max': round(max(sp), 3), 'mean_abs_dev': round(sum(abs(x - m) for x in sp) / len(sp), 3)},
                  'distinct_start_xz_10cm': len({(round(p['x'], 1), round(p['z'], 1)) for p in st0})}
    pooled = [fr for g in ('fastest', 'no_wall', 'recover', 'speed') for fr in loaded[g]]
    res['pooled_4modes(60 シーン、マスは 1 シーン 135 フレーム)'] = {'scenes': 60, 'state_cells_own': cells(pooled, per_scene / 4)}
    t = res['teacher(前の教師・先頭 15)']['state_cells_own']
    for g in res:
        res[g]['state_cells_ratio_vs_teacher'] = round(res[g]['state_cells_own'] / t, 2)
    # 前の教師の全 60 シーンの VM1(今日の 7.6 µs の数え直し)
    res['teacher_all60_VM1_us_own'] = round(vm1(load(GROUPS['teacher(前の教師・先頭 15)'])), 3)
    print(json.dumps(res, ensure_ascii=False, indent=1))


main()
