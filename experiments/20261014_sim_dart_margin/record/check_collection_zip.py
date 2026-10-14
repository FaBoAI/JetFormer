#!/usr/bin/env python3
# (sim_dart_20261014 の記録の段の写し。中身は変えていない)
"""アプリが作った集約 ZIP(20 秒シーンを1つにまとめたもの)を、元の20分データ(ID8・取り込み済み)と同じ形式かどうか確かめる
(2026-10-13、強化学習のベストラップ)。データは変更しない。

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/check_collection_zip.py <集約ZIP> [--out 結果.json]

  - ZIP の中身がマニフェスト(sha256・バイト数)と一致するか(ユーザーの qa/verify-id8-collection-zip.py と同じ検査を、シーン数を固定せずに)
  - 各シーンの aligned NPZ のキー・型・形が元の20分データと同じか、ジャンル 8・num_modes 9・未来ラベル(next1..10)・最後の10フレームの除外・合成フラグ
  - PWM: 各フレームの記録(simulation/scenes/<scene>/frames.jsonl の command_si)から、収集器と同じ式(dataset_recorder.action_to_pwm)で
    計算し直した値が vehicle と一致するか(操舵中心 1442 µs・スロットル中立 1498 µs・最大 0.52 rad・7.24 m/s)
  - dataset_card.json の項目が元と同じか、context.teacher(教師の識別)が何か
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ORIG = Path('/home/jetson/jetformer/data/datasets/jetformer-simulator-id8-20min_20261012-150956_b55616cf')


def action_to_pwm(action, c, m):
    """dataset_recorder.action_to_pwm と同じ式(写し)。ユーザーの bridge 実装は import しない"""
    fraction = max(-1., min(1., action['steering'] / c['max_steering_rad'] / c['steering_sign']))
    center, neutral = m['steer_center_us'], m['throttle_neutral_us']
    span = c['pwm_max_us'] - center if fraction >= 0 else center - c['pwm_min_us']
    steer = center + fraction * span
    speed = action['speed']
    if speed >= 0:
        throttle = neutral - min(1., speed / c['max_speed_mps']) * (neutral - c['pwm_min_us'])
    else:
        throttle = neutral + min(1., -speed / c['max_reverse_speed_mps']) * (c['pwm_max_us'] - neutral)
    return steer, throttle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip')
    ap.add_argument('--out')
    a = ap.parse_args()
    res = {'archive': a.zip, 'checks': {}}
    ref_npz = np.load(sorted((ORIG / 'aligned').glob('*.npz'))[0])
    ref_spec = {k: (str(ref_npz[k].dtype), list(ref_npz[k].shape[1:])) for k in ref_npz.files}
    ref_card = json.loads((ORIG / 'dataset_card.json').read_text())
    with zipfile.ZipFile(a.zip) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        cards = [n for n in names if n.endswith('/dataset_card.json')]
        assert len(cards) == 1
        prefix = cards[0][:-len('dataset_card.json')]
        read = lambda rel: z.read(prefix + rel)  # noqa: E731
        card, splits, man = (json.loads(read(x)) for x in ('dataset_card.json', 'splits.json', 'manifest.json'))
        files = {e['path']: e for e in man['files']}
        assert set(files) == {n[len(prefix):] for n in names if n != prefix + 'manifest.json'}, 'マニフェストと中身が違う'
        for p, e in files.items():
            d = read(p)
            assert len(d) == e['bytes'] and hashlib.sha256(d).hexdigest() == e['sha256'], p
        res['checks']['manifest_sha256_all_files'] = len(files)
        res['card'] = {k: card.get(k) for k in ('name', 'scenes', 'frames', 'training_frames', 'num_modes', 'mode_frames', 'controller',
                                                 'training_label_mode', 'synthetic', 'train_scenes', 'val_scenes', 'test_scenes')}
        res['card_keys_same_as_original'] = sorted(card) == sorted(ref_card)
        res['card_keys_diff'] = {'only_new': sorted(set(card) - set(ref_card)), 'only_original': sorted(set(ref_card) - set(card))}
        assert card['num_modes'] == 9 and card['mode_frames']['8'] == card['frames'] and sum(card['mode_frames'].values()) == card['frames']
        assert not set(splits['train']) & set(splits['val'])
        coll = json.loads(read('simulation/collection.json'))
        res['collection_contract_teacher'] = {k: (coll.get('contract') or {}).get('teacher', {}).get(k)
                                              for k in ('id', 'schema', 'controller_kind', 'profile', 'not_learned_policy')}
        scenes, max_pwm_err, frames_total, eligible = [], 0.0, 0, 0
        for split in ('train', 'val', 'test'):
            for scene in splits[split]:
                arr = np.load(io.BytesIO(read(f'aligned/{scene}.npz')), allow_pickle=False)
                n = len(arr['mode'])
                spec = {k: (str(arr[k].dtype), list(arr[k].shape[1:])) for k in arr.files}
                assert spec == ref_spec, f'NPZ の形式が元と違う: {scene} {set(spec.items()) ^ set(ref_spec.items())}'
                assert (arr['mode'] == 8).all()
                assert np.array_equal(arr['future_valid'], np.arange(n) < n - 10)
                assert np.array_equal(arr['future_actions'], arr['vehicle'][np.minimum(np.arange(n)[:, None] + np.arange(1, 11), n - 1), :2])
                assert arr['synthetic'].all() and not arr['human_control'].any() and arr['vehicle_valid'].all()
                assert np.all(np.max(arr['imu_timestamps_us'], axis=1) <= arr['timestamps_us'] + 1)
                sdir = f'simulation/scenes/{scene}/'
                session = json.loads(read(sdir + 'session.json'))
                rows = [json.loads(x) for x in read(sdir + 'frames.jsonl').splitlines() if x]
                c, m = session['calibration'], session['model_config']
                pwm = np.array([action_to_pwm(r['command_si'], c, m) for r in rows], np.float32)
                err = float(np.max(np.abs(pwm - arr['vehicle'][:, :2])))
                max_pwm_err = max(max_pwm_err, err)
                erpm = np.array([r['state_post_action']['speed'] for r in rows])
                v = arr['vehicle']
                scenes.append({'scene': scene, 'split': split, 'frames': n, 'pwm_recompute_max_abs_us': err,
                               'steer_center_us': m['steer_center_us'], 'throttle_neutral_us': m['throttle_neutral_us'],
                               'max_speed_mps': c['max_speed_mps'], 'max_steering_rad': c['max_steering_rad'],
                               'steer_pwm_min_max': [float(v[:, 0].min()), float(v[:, 0].max())],
                               'throttle_pwm_min_max': [float(v[:, 1].min()), float(v[:, 1].max())],
                               'ch3': sorted(set(v[:, 2].tolist())), 'erpm_min': float(v[:, 3].min()),
                               'post_speed_max': float(erpm.max()),
                               'teacher_id': session['context'].get('teacher', {}).get('id'),
                               'start_variation': session['context'].get('start_variation')})
                frames_total += n
                eligible += int(arr['future_valid'].sum())
        res['scenes'] = scenes
        res['frames'], res['training_frames'] = frames_total, eligible
        res['pwm_recompute_max_abs_us'] = max_pwm_err
        assert max_pwm_err < 0.01, max_pwm_err
    res['pass'] = True
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
