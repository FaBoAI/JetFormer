#!/usr/bin/env python3
"""DAgger の記録の確かめ(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.go_no_go_gates.G2_browser と procedure 4 の検証)

    python3 -I verify_dagger_scenes.py --scenes <frames.jsonl のあるシーンのフォルダー ...> --out <結果.json>
    python3 -I verify_dagger_scenes.py --dataset <取り込んだデータセットのフォルダー> --out <結果.json>

シーンごとに:
  - G2: フレーム k の位置(frames.jsonl の pose_pre_action)が、状態集めの走行(ブラウザの評価の記録の trace[k−1] = フレーム k の操作の前の位置)と
    ≤ 2 cm。k までのすべてのフレームの位置の差の最大も書く
  - info.dart が全フレームの action_info にあり、info.dart.dagger.replay がフレーム < k で真・≥ k で偽、action_info.jf が無い
  - 再生した操作(command_si)が状態集めの走行の操作と同じ(≤ 1e-12)
  - 600 フレーム・カメラの画像がすべて違う(取り込んだデータセットのとき。samples/ の画像の sha1)
状態集めの走行と k は web/dagger_plan.mjs(prepare_record.py が書いたもの)から読む。データは読むだけ。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXP = Path('/home/jetson/jetformer/experiments/20261015_sim_diverse_train')
CLD = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015/eval/closed_loop_dagger')
TOL_M = 0.02


def load_plan():
    t = (EXP / 'web' / 'dagger_plan.mjs').read_text(encoding='utf-8')
    body = t[t.index('Object.freeze(') + len('Object.freeze('):t.rindex(');')]
    return json.loads(body)


def check_scene(sd, plan, dataset_root=None):
    rows = [json.loads(l) for l in (sd / 'frames.jsonl').read_text().splitlines() if l.strip()]
    out = {'scene': sd.name, 'frames': len(rows)}
    why = []
    p0 = ((rows[0].get('action_info') or {}).get('dart') or {}).get('p') or {}
    key = p0.get('start')
    sc = plan['scenes'].get(key)
    if not sc:
        return out | {'pass': False, 'why': [f'計画にない開始 {key}']}
    k = sc['k']
    out.update({'start': key, 'name': sc['name'], 'k': k, 'source_run': sc['source_run']})
    src_file = None
    for p in CLD.glob(f"*/{sc['source_run']}.json"):
        src_file = p
    if src_file is None:
        return out | {'pass': False, 'why': ['状態集めの走行が見つからない']}
    src = json.loads(src_file.read_text())
    tr, acts = src['trace'], src['actions']
    # info.dart・jf
    miss = [i for i, r in enumerate(rows) if not ((r.get('action_info') or {}).get('dart'))]
    jf = [i for i, r in enumerate(rows) if (r.get('action_info') or {}).get('jf')]
    flags = [bool(((r.get('action_info') or {}).get('dart') or {}).get('dagger', {}).get('replay')) for r in rows]
    bad_flag = [i for i, f in enumerate(flags) if f != (i < k)]
    if miss:
        why.append(f'info.dart の無いフレーム {len(miss)}')
    if jf:
        why.append(f'action_info.jf のあるフレーム {len(jf)}')
    if bad_flag:
        why.append(f'replay の印が k と合わないフレーム {len(bad_flag)}')
    # 再生した操作
    amax = 0.0
    for i in range(min(k, len(rows))):
        c = rows[i]['command_si']
        amax = max(amax, abs(c['steering'] - acts[i]['steering']), abs(c['speed'] - acts[i]['speed']))
    if amax > 1e-12:
        why.append(f'再生した操作が状態集めと違う({amax})')
    # 位置: フレーム i の操作の前の位置 = 状態集めの trace[i − 1](i ≥ 1)
    dmax, dk = 0.0, None
    for i in range(1, min(k, len(rows) - 1) + 1):
        a, b = rows[i]['pose_pre_action'], tr[i - 1]
        d = math.hypot(a['x'] - b['x'], a['z'] - b['z'])
        dmax = max(dmax, d)
        if i == k:
            dk = d
    if k == 0:
        dk = 0.0
    if dk is None or dk > TOL_M:
        why.append(f'フレーム k の位置の差 {dk}(> {TOL_M} m)')
    out.update({'pose_diff_at_k_m': dk, 'pose_diff_max_before_k_m': dmax, 'replay_action_max_diff': amax, 'replay_frames': sum(flags)})
    if len(rows) != 600:
        why.append(f'フレーム {len(rows)}')
    if dataset_root is not None:
        z = np.load(dataset_root / 'aligned' / f'{sd.name}.npz', allow_pickle=False)
        cams = list(z['cam_files'])
        hs = []
        for c in cams:
            p = dataset_root / c if not Path(c).is_absolute() else Path(c)   # cam_files はデータセットのフォルダーからの相対(samples/CAM_FRONT/…)
            hs.append(hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else None)
        out['images'] = len(hs)
        out['images_unique'] = len({h for h in hs if h})
        out['images_missing'] = sum(1 for h in hs if h is None)
        # 同じ画像の組: 車が動いていない(操作の前の位置・向きが同じ)フレームどうしなら正しい(状態集めの走行は遅延 1 なので、フレーム 0 の操作は 0 で、
        # フレーム 0 と 1 は同じ位置。10/14 の DART は先生がフレーム 0 から動くので、すべて違った)
        first = {}
        dups = []
        for i, h in enumerate(hs):
            if h in first:
                j = first[h]
                a_, b_ = rows[j]['pose_pre_action'], rows[i]['pose_pre_action']
                dups.append({'frames': [j, i], 'same_pose': all(abs(a_[k] - b_[k]) < 1e-12 for k in ('x', 'z', 'yaw'))})
            else:
                first[h] = i
        out['image_duplicates'] = dups
        bad = [d for d in dups if not d['same_pose']]
        if bad or out['images_missing']:
            why.append(f"違う位置で同じ画像 {len(bad)} 組(無い {out['images_missing']})")
    out['why'] = why
    out['pass'] = not why
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='*', default=[])
    ap.add_argument('--dataset', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plan = load_plan()
    res = []
    if a.dataset:
        root = Path(a.dataset)
        sp = json.loads((root / 'splits.json').read_text())
        for s in list(sp.get('train', [])) + list(sp.get('val', [])):
            res.append(check_scene(root / 'simulation' / 'scenes' / s, plan, root))
    for s in a.scenes:
        res.append(check_scene(Path(s), plan))
    out = {'schema': 'jf-sim-diverse-dagger-verify/v1', 'teacher': plan['teacher'], 'plan_sha256': plan['plan_sha256'], 'tolerance_m': TOL_M,
           'scenes': res, 'all_pass': bool(res) and all(r['pass'] for r in res)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({'all_pass': out['all_pass'], 'scenes': [(r.get('name'), r.get('k'), r.get('pose_diff_at_k_m'), r['why']) for r in res]},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
