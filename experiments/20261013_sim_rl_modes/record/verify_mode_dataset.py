#!/usr/bin/env python3
"""記録の段: 強化学習モードで記録して JetFormer に取り込んだデータセット(data/datasets/sim-rl-<モード>_<日時>)を、
ユーザーの qa/verify-rc-teacher-dataset.py と同じ検査で 1 シーンずつ確かめる(2026-10-14、sim_rl_modes)。データは変えない。

    python -I record/verify_mode_dataset.py <data/datasets/名前> [--out 結果.json]

ユーザーの検査(1 シーンの ZIP 用)からの違い:
  - 取り込んだあとの形(集約: aligned/<シーン>.npz・simulation/scenes/<シーン>/・can_bus/<シーン>_*.json・v1.0-<名前>/ の表)を読む。
    ファイルごとの sha256(manifest)は、取り込みのとき rl_import.audit_scene がシーンの ZIP の manifest と照らしている(ここでは見ない)
  - 教師の印: not_learned_policy(台本の RC 教師の印)の代わりに、context.teacher が jetformer-sim-rl-teacher/v1・id jf-rl-<モード>-v1・
    方策の sha256 が登録簿(data/sim_rl/modes/<モード>/mode.json)と同じで、データセットの全シーンで同じか
  - ラベル: 乱した操作を正解に含むフレームの学習除外(JetFormer の label_mask、k ∈ [j − 10, j])を frames.jsonl の action_info.jf.l から
    数え直し、future_valid = (i < 590) かつ外していない・mode = 8(外したフレームは 0)と全フレーム一致するか。vehicle_valid は全部 true のまま
  - 足したもの: 衝突 0・20 秒の連続周回を走り切った・有効な周(33 ゲート・黒シート 1 回・衝突 0)の数・乱しの出来事・シールドの作動、
    データセットのカード(ジャンル 8・num_modes 9・タグ・rl・label_mask の数)、シーンタブに展開したシーン(data/raw/ds-<名前>/)
  - センサー・時刻・IMU・PWM・未来ラベル・画像の重複の検査は元のまま(同じ式)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

JF = Path('/home/jetson/jetformer')
HORIZON = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, label, tol=1e-9):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label + ' shape/finite')
    error = float(np.max(np.abs(a - b), initial=0))
    require(error <= tol, label + f' error {error}')
    return error


def mask_from_learn(learn):
    learn = np.asarray(learn).astype(bool)
    mask = np.zeros(len(learn), bool)
    for j in np.nonzero(~learn)[0]:
        mask[max(0, j - HORIZON):j + 1] = True
    return mask


def verify_scene(root, scene, tables, key, policy_sha):
    sd_dir = root / 'simulation' / 'scenes' / scene

    def read(p):
        return json.loads((sd_dir / p).read_text())
    s, r = read('session.json'), read('evaluation-report.json')
    rows = [json.loads(l) for l in (sd_dir / 'frames.jsonl').read_text().splitlines() if l.strip()]
    t = s['context']['teacher']
    require(s['controller'] == r['context']['controller'] == 'demo' and s['synthetic'], 'Not synthetic teacher')
    require(t.get('schema') == 'jetformer-sim-rl-teacher/v1' and t.get('id') == f'jf-rl-{key}-v1' and t.get('policy_sha256') == policy_sha
            and t.get('mode') == key, 'Teacher marker (RL mode)')
    a = np.load(root / 'aligned' / f'{scene}.npz', allow_pickle=False)
    n = len(rows)
    require(n == len(r['actions']) == len(r['trace']) == 600, 'Coverage')
    require(all(len(a[k]) == 600 for k in a.files), 'NPZ row coverage')
    # 学習除外(JetFormer の label_mask)を数え直す
    learn = [((row.get('action_info') or {}).get('jf') or {}).get('l', 0) == 1 for row in rows]
    mask = mask_from_learn(learn)
    require(np.array_equal(a['mode'], np.where(mask, 0, 8).astype(a['mode'].dtype)), 'Genre8 / label-mask mode')
    require(a['synthetic'].all() and not a['human_control'].any(), 'Synthetic flags')
    require(a['vehicle_valid'].all() and a['erpm_valid'].all(), 'Labels/eRPM validity')
    require(np.array_equal(a['future_valid'], (np.arange(600) < 590) & ~mask), 'Future validity (last 10 + label mask)')
    require((a['tof_age_ms'] == 0).all() and (a['vehicle_age_ms'] == 0).all(), 'Synchronous ages')
    sd, ep, sam = tables
    rawimu = {v['utime']: v for v in json.loads((root / 'can_bus' / f'{scene}_ms_imu.json').read_text())}
    rawvehicle = json.loads((root / 'can_bus' / f'{scene}_vehicle_monitor.json').read_text())
    epoch = rows[0]['timestamp_us']
    history = [((i - 9) / 100, [0, 0, 9.80665, 0, 0, 0]) for i in range(10)]
    next_t = .01
    image_hashes, tof_hashes, confidence = set(), set(), set()
    imu_error = 0.
    previous = {'x': rows[0]['pose_pre_action']['x'], 'z': rows[0]['pose_pre_action']['z'], 'yaw': rows[0]['pose_pre_action']['yaw'], 'speed': 0, 'sim_time': 0}
    c, m = s['calibration'], s['model_config']
    rgb_mean = []
    for i, row in enumerate(rows):
        require(row['frame'] == i, 'Contiguous frame')
        require(row['timestamp_us'] == int(a['timestamps_us'][i]) == epoch + round(row['sim_time'] * 1e6), 'Camera timestamp')
        near(row['sim_time'], i / 30, '30Hz simtime')
        near([row['pose_pre_action'][k] for k in ['x', 'z', 'yaw']], [previous[k] for k in ['x', 'z', 'yaw']], 'Preaction pose')
        for channel, rel, shape in [('CAM_FRONT', str(a['cam_files'][i]), (224, 224, 3)),
                                    ('TOF_FRONT', str(a['cam_files'][i]).replace('CAM_FRONT', 'TOF_FRONT').replace('.jpg', '.png'), (42, 54))]:
            require(rel in sd and (root / rel).is_file(), 'Covered sensor file')
            im = cv2.imread(str(root / rel), cv2.IMREAD_UNCHANGED)
            require(im is not None and im.shape == shape, 'Sensor shape ' + rel)
            entry = sd[rel]
            pose = ep[entry['ego_pose_token']]
            sample = sam[entry['sample_token']]
            require(entry['timestamp'] == pose['timestamp'] == sample['timestamp'] == row['timestamp_us'], 'Image/time/pose mapping')
            p = row['pose_pre_action']
            near(pose['translation'], [p['x'], -p['z'], 0], 'Ego translation')
            near(pose['rotation'], [math.cos(p['yaw'] / 2), 0, 0, math.sin(p['yaw'] / 2)], 'Ego orientation')
            require(pose['synthetic_ground_truth'], 'Groundtruth pose marker')
            h = hashlib.sha256(im.tobytes()).hexdigest()
            if channel == 'CAM_FRONT':
                image_hashes.add(h)
                rgb_mean.append(float(im.mean()))
            else:
                require(im.dtype == np.uint16 and np.array_equal(im, a['tof'][i]), 'Packed ToF exact PNG/NPZ parity')
                tof_hashes.add(h)
                confidence.update(int(x) for x in np.unique(im >> 13))
        imu_expected = np.asarray([v for t_, v in history], np.float32)
        times_expected = np.asarray([epoch + round(t_ * 1e6) for t_, v in history], np.int64)
        require(np.array_equal(times_expected, a['imu_timestamps_us'][i]), 'Causal IMU timestamp synthesis')
        imu_error = max(imu_error, near(a['imu'][i], imu_expected, 'IMU independently derived from motion', 2e-6))
        require(np.max(times_expected) <= row['timestamp_us'] + 1 and row['timestamp_us'] - max(times_expected) <= 10001, 'Causal fresh IMU')
        require(row['imu_padding_before_episode'] == sum(t_ < 0 for t_, v in history), 'IMU reset-padding marker')
        for t_, v in zip(times_expected, a['imu'][i]):
            raw = rawimu[int(t_)]
            near(v, raw['linear_accel'] + raw['rotation_rate'], 'Raw/aligned IMU', 0)
        require(np.float32(previous['speed'] * c['erpm_per_mps'] * c['erpm_forward_sign']) == a['vehicle'][i, 3], 'Preaction speed/eRPM')
        fraction = np.clip(row['command_si']['steering'] / c['max_steering_rad'] / c['steering_sign'], -1, 1)
        center, neutral, speed = m['steer_center_us'], m['throttle_neutral_us'], row['command_si']['speed']
        expected = [center + fraction * (c['pwm_max_us'] - center if fraction >= 0 else center - c['pwm_min_us']),
                    neutral - min(1, speed / c['max_speed_mps']) * (neutral - c['pwm_min_us']) if speed >= 0
                    else neutral + min(1, -speed / c['max_reverse_speed_mps']) * (c['pwm_max_us'] - neutral)]
        near(row['applied_pwm_us'], expected, 'Actual SI inverse PWM')
        require(np.array_equal(np.asarray(expected, np.float32), a['vehicle'][i, :2]), 'Serialized PWM')
        rv = rawvehicle[i]
        require(rv['utime'] == row['timestamp_us'] and rv['controller'] == 'demo' and rv['synthetic'] and not rv['human_control'], 'Raw vehicle flags/time')
        near([rv[k] for k in ['steering_us', 'throttle_us', 'mode_us', 'erpm']], a['vehicle'][i], 'Raw vehicle', 0)
        require(np.array_equal(a['future_actions'][i], a['vehicle'][np.minimum(np.arange(i + 1, i + 11), 599), :2]), 'Future labels next1..10')
        current = row['state_post_action']
        dt = current['sim_time'] - previous['sim_time']
        dyaw = math.atan2(math.sin(current['yaw'] - previous['yaw']), math.cos(current['yaw'] - previous['yaw']))
        yawrate, acc = dyaw / dt, (current['speed'] - previous['speed']) / dt
        while next_t <= current['sim_time'] + 1e-9:
            history.append((next_t, [-current['speed'] * yawrate, acc, 9.80665, 0, 0, yawrate]))
            next_t += .01
        history = history[-10:]
        previous = current
    require(len(image_hashes) > 590 and len(tof_hashes) > 590, 'Unexpected frozen/repeated sensor frames')
    # 足したもの: 走りの結果
    met = r.get('metrics') or {}
    require(r.get('outcome', {}).get('reason') == 'collection_duration_complete', '20 秒の連続周回を走り切っていません')
    require(met.get('collisions') == 0, '衝突があります')
    valid_laps = [l for l in (met.get('lap_events') or []) if l.get('passed_gates') == 33 and len(l.get('required_crossing_events') or []) == 1
                  and l.get('collisions') == 0]
    jf = [((row.get('action_info') or {}).get('jf') or {}) for row in rows]
    events, prev_on = 0, False
    for q in jf:
        on = q.get('ph') == 'perturb' or q.get('l') == 0
        events += int(on and not prev_on)
        prev_on = on
    return {'scene': scene, 'pass': True, 'frames': n, 'unique_decoded_camera_images': len(image_hashes), 'unique_packed_tof_frames': len(tof_hashes),
            'confidence_codes': sorted(confidence), 'max_imu_float32_error': imu_error, 'rgb_mean': float(np.mean(rgb_mean)),
            'rgb_mean_std_over_frames': float(np.std(rgb_mean)), 'tof_valid_fraction': float(np.mean((a['tof'] & 0x1FFF) != 8191)),
            'collisions': met.get('collisions'), 'completed_laps': met.get('completed_laps'), 'valid_laps': len(valid_laps),
            'lap_times_s': [round(l['elapsed'], 3) for l in valid_laps], 'perturb_events': events, 'learn0_frames': int(sum(1 for x in learn if not x)),
            'masked_frames': int(mask.sum()), 'shield_frames': sum(1 for q in jf if q.get('sh') == 1), 'scenario': r.get('scenario'),
            'seed': s['config'].get('seed'), 'start': s['context'].get('start_variation'), 'teacher': t}


def verify(root):
    root = Path(root).resolve()
    card = json.loads((root / 'dataset_card.json').read_text())
    splits = json.loads((root / 'splits.json').read_text())
    coll = json.loads((root / 'simulation' / 'collection.json').read_text())
    key = card['rl']['mode']
    mode_json = json.loads((JF / 'data/sim_rl/modes' / key / 'mode.json').read_text())
    policy_sha = mode_json['policy_sha256']
    require(card['rl']['policy_sha256'] == policy_sha, 'カードの方策の sha256 が登録簿と違います')
    require(card['num_modes'] == 9 and card['training_label_mode'] == 'synthetic_controller' and card['synthetic'] is True, 'Genre/label contract (card)')
    want_tags = {'simulator', 'sim-rl', 'sim-rl-' + key}
    require(want_tags <= set(card.get('tags') or []), 'タグが足りません')
    inv = root / card['version']
    sd = {x['filename']: x for x in json.loads((inv / 'sample_data.json').read_text())}
    ep = {x['token']: x for x in json.loads((inv / 'ego_pose.json').read_text())}
    sam = {x['token']: x for x in json.loads((inv / 'sample.json').read_text())}
    scenes = [s['scene'] for s in coll['scenes']]
    require(sorted(scenes) == sorted(splits['train'] + splits['val'] + splits['test']), 'splits と collection のシーンが違います')
    require(len(sd) == 1200 * len(scenes) and len(ep) == len(sam) == 600 * len(scenes), 'nuScenes coverage')
    starts = [json.dumps(s['start'], sort_keys=True) for s in coll['scenes']]
    require(len(set(starts)) == len(starts), '同じ開始姿勢が 2 つあります')
    per = []
    teachers = set()
    for sc in scenes:
        try:
            res = verify_scene(root, sc, (sd, ep, sam), key, policy_sha)
        except Exception as e:  # noqa: BLE001
            res = {'scene': sc, 'pass': False, 'error': f'{type(e).__name__}: {e}'}
        res['split'] = 'val' if sc in splits['val'] else 'train'
        per.append(res)
        if res.get('teacher'):
            teachers.add(json.dumps(res.pop('teacher'), sort_keys=True))
        print(sc, 'pass' if res['pass'] else 'FAIL ' + res.get('error', ''), flush=True)
    masked = sum(p.get('masked_frames', 0) for p in per)
    lm = card.get('label_mask') or {}
    card_ok = {'genre': card['mode_frames'].get('8', 0) + card['mode_frames'].get('0', 0) == card['frames'] == 600 * len(scenes),
               'mode0_frames_equal_masked': int(card['mode_frames'].get('0', 0)) == masked,
               'label_mask_masked_frames': lm.get('masked_frames') == masked if lm else masked == 0}
    # training_frames は future_valid の数(数え直し)
    fv = 0
    for sc in scenes:
        with np.load(root / 'aligned' / f'{sc}.npz', allow_pickle=False) as a:
            fv += int(a['future_valid'].sum())
    card_ok['training_frames'] = card['training_frames'] == fv
    # シーンタブに展開したシーン
    imp = json.loads((root / 'import_source.json').read_text())
    created = (imp.get('scenes') or {}).get('created') or []
    raw = JF / 'data' / 'raw' / (imp.get('scenes') or {}).get('log', '')
    expanded = []
    for name in created:
        sj = raw / name / 'scene.json'
        ok = sj.is_file()
        info = {'name': name, 'exists': ok}
        if ok:
            d = json.loads(sj.read_text())
            zero = sum(1 for seg in d.get('mode_segments') or [] if seg.get('mode_id') == 0)
            info.update(tags_ok=want_tags <= set(d.get('tags') or []), mode_id=d.get('mode_id'), counts=d.get('counts'), zero_segments=zero,
                        unset_segments=len(((d.get('label_mask') or {}).get('unset_segments')) or []))
        expanded.append(info)
    out = {'schema': 'jf-sim-rl-mode-dataset-integrity/v1', 'dataset': root.name, 'mode': key, 'policy_sha256': policy_sha,
           'scope': 'ユーザーの qa/verify-rc-teacher-dataset.py の検査(センサー・時刻・姿勢・IMU・PWM・未来ラベル・画像の重複)を取り込んだ形で 1 シーンずつ + 学習除外・教師の印・走りの結果・カード・シーンタブ',
           'scenes': len(scenes), 'train': len(splits['train']), 'val': len(splits['val']), 'per_scene': per,
           'pass_scenes': sum(1 for p in per if p['pass']), 'teacher_same_all_scenes': len(teachers) == 1, 'card': card_ok,
           'tags': card.get('tags'), 'genre_frames': card['mode_frames'], 'masked_frames': masked,
           'expanded_scenes': {'log': (imp.get('scenes') or {}).get('log'), 'created': len(created), 'all_exist': all(e['exists'] for e in expanded),
                               'all_tags_ok': all(e.get('tags_ok') for e in expanded), 'per_scene': expanded}}
    out['pass'] = (out['pass_scenes'] == len(scenes) and out['teacher_same_all_scenes'] and all(v for v in card_ok.values() if v is not None)
                   and len(created) == len(scenes) and out['expanded_scenes']['all_exist'] and out['expanded_scenes']['all_tags_ok'])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--out')
    a = ap.parse_args()
    res = verify(a.dataset)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(json.dumps({k: res[k] for k in ('dataset', 'mode', 'scenes', 'pass_scenes', 'teacher_same_all_scenes', 'card', 'masked_frames', 'pass')}, ensure_ascii=False))
