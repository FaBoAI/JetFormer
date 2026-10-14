#!/usr/bin/env python3
"""確認の段(review、2026-10-14): 「D1 で乱しを入れたのは 33 区間のうち 18 区間」「点 3 → 4 の区間で始まった乱しは 0 回」を、
eval/dart_coverage.py を使わずに D1 の frames.jsonl から数え直す(読むだけ)

    python3 -I coverage_check.py

区間 i = 元の経路の点 i → 点 i+1。各フレームの位置(pose_pre_action)を、今の区間・次の区間・その次の区間のうち、いちばん近い線分に当てる
(シーンはゲート 0 から始まる。2 周目に入っても続けて数える)。乱しのフレーム = action_info.dart.noise があるフレーム、
乱しの始まり = 乱しの無いフレームの次の乱しのフレーム。
出力: /home/jetson/jetformer_reports/sim_dart_20261014/review/coverage_check.json
"""
import glob
import json
import math

DS = '/home/jetson/jetformer/data/datasets/jetformer-simulator-rl-dart-fast-10min_20261014-062216_4446c73d'
OUT = '/home/jetson/jetformer_reports/sim_dart_20261014/review/coverage_check.json'
ROUTE_FILE = '/home/jetson/jetformer_reports/sim_dart_20261014/eval/closed_loop/A1-s0/A1-s0-E0-single.json'


def seg_d(p, a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dz * dz
    u = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / L))
    return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dz)


def main():
    route = json.load(open(ROUTE_FILE))['world']['route']   # ゲート 0 の開始の走行なので元の経路のまま
    n = len(route)
    noisy = [0] * n
    starts = [0] * n
    frames = [0] * n
    for p in sorted(glob.glob(f'{DS}/simulation/scenes/*/frames.jsonl')):
        cur, prev_noisy = 0, False
        for line in open(p, encoding='utf-8'):
            f = json.loads(line)
            q = f['pose_pre_action']
            pt = (q['x'], q['z'])
            cand = [(seg_d(pt, route[(cur + j) % n], route[(cur + j + 1) % n]), (cur + j) % n) for j in range(3)]
            cur = min(cand)[1]
            dart = (f.get('action_info') or {}).get('dart') or {}
            is_noisy = bool(dart.get('noise'))
            frames[cur] += 1
            noisy[cur] += int(is_noisy)
            starts[cur] += int(is_noisy and not prev_noisy)
            prev_noisy = is_noisy
    res = {'frames_per_segment': frames, 'noisy_frames_per_segment': noisy, 'event_starts_per_segment': starts,
           'segments_with_noise': [i for i in range(n) if noisy[i]], 'n_segments_with_noise': sum(1 for x in noisy if x),
           'unperturbed_segments': [i for i in range(n) if not noisy[i]], 'noisy_total': sum(noisy), 'starts_total': sum(starts),
           'segment_3_to_4': {'noisy_frames': noisy[3], 'event_starts': starts[3]}}
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'frames_per_segment'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
