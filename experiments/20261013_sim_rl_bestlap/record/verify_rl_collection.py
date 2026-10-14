#!/usr/bin/env python3
"""記録の段: ブラウザで記録した RL のベストラップの 60 シーンを、ユーザーの qa/verify-rc-collection.py と同じやり方で確かめる
(2026-10-14、design.md 4.5 節)。データは変えない。

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/record/verify_rl_collection.py <session_ids.json> [--workers 2]

ユーザーの verify-rc-collection.py との違い(RC 教師専用の部分だけを RL 用にした):
  - 教師の出どころ: not_learned_policy・profile speed-rc の代わりに、context.teacher.id = rl-bestlap-ppo-v1・profile speed-rc・
    方策の sha256(install.json)を確かめる
  - 物理の段: RC 教師専用の verify-rc-teacher-recording.mjs の代わりに、汎用の verify_recording.mjs
    (記録した操作を元の sim-core.js で再生して軌跡が一致、周回・ゲート・黒シート・衝突の集計、自前の積分と数え直し)
  - 教師の段: teacher_mismatch_probe.mjs で、方策を計算し直して記録の操作と比べる。合否は「ブラウザが記録した状態」から計算した値
    (≤ 1e-9)。Node で再生した状態(ブラウザとの差 ~1e-14)から計算した値も書く。後者は参照ループの頂点で観測(向きのずれ)が
    跳ぶフレームがあるため、合否には使わない(2026-10-14 の 1 回目の検証で 60 シーン中 2 シーン・各 1 フレームで起きた。record.md)
  - センサーの段: ユーザーの verify-rc-teacher-dataset.py の写し(教師の印の 1 行だけ変えた verify_rc_teacher_dataset_rl.py)
  - 足したもの: 記録前の予想(prediction.json)との照合(周回時間・操作列の sha256)、全フレームの RGB・ToF が互いに違うか、
    RGB の平均・標準偏差、ToF の当たりの割合(教師の 20 分データと比べる)
結果: /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/verify/collection-integrity.json(シーンごとは verify/<id>/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
SIM = REPORT / 'sim'
TEACHER = SIM / 'dist' / 'rl' / 'rl-teacher-bestlap.js'
TEACHER_DATA = Path('/home/jetson/jetformer/data/datasets/jetformer-simulator-id8-20min_20261012-150956_b55616cf')
SOURCES = [SIM / 'dist/sim-core.js', SIM / 'dist/rc-speed-controller.js', SIM / 'dist/world.js', SIM / 'dist/scene.js',
           SIM / 'qa/load-course-world.mjs', TEACHER, SIM / 'dist/rl/features.mjs', SIM / 'dist/rl/policy_forward.mjs',
           SIM / 'dist/rl/policy_weights.mjs', SIM / 'dist/rl/bestlap_manifest.mjs', EXP / 'verify_recording.mjs',
           EXP / 'rl/lap_audit.mjs', HERE / 'verify_rc_teacher_dataset_rl.py', HERE / 'teacher_mismatch_probe.mjs', Path(__file__).resolve()]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False))
    tmp.replace(path)


def resolve_scene(datasets, sid):
    require(re.fullmatch(r'[0-9a-f]{32}', sid), 'セッション ID の形が違います')
    paths = list((datasets / sid).glob('*/simulation/session.json'))
    require(len(paths) == 1, f'{sid}: シーンが 1 つではありません')
    root = paths[0].parent.parent
    require((root / 'manifest.json').is_file() and (root / 'simulation/evaluation-report.json').is_file(), f'{sid}: 記録が終わっていません')
    return root


def actions_sha256(report):
    """predict_rec60.mjs と同じ: 各フレームの (steering, speed) を float64 リトルエンディアンで並べた sha256"""
    h = hashlib.sha256()
    for a in report['actions']:
        h.update(struct.pack('<dd', a['steering'], a['speed']))
    return h.hexdigest()


def scene_metadata(root, expected, policy_sha):
    """ユーザーの verify-rc-collection.py の scene_metadata と同じ検査(教師の出どころだけ RL 用)"""
    s = read(root / 'simulation/session.json'); r = read(root / 'simulation/evaluation-report.json')
    card = read(root / 'dataset_card.json'); splits = read(root / 'splits.json')
    name = s['display_name']; require(name in expected, f'計画に無い名前です: {name}')
    p = expected[name]; v = r['context']['start_variation']; c = r['config']
    require(r['seed'] == c['seed'] == p['seed'] and r['scenario'] == p['scenario'], '種・場面が計画と違います')
    require(v['gate'] == p['start-gate'] == 0 and abs(v['left_offset_m'] - p['spawn-offset']) < 1e-12 and abs(v['left_yaw_deg'] - p['spawn-yaw']) < 1e-12,
            '開始姿勢が計画と違います')
    require(c['noiseStd'] == (.025 if p['scenario'] == 'noise' else 0), '雑音が計画と違います')
    require(c['episodeMode'] == 'continuous-laps-v1' and c['timeLimit'] == 20 and c['stopOnCollision'], '回の設定が違います')
    require(s['controller'] == r['context']['controller'] == 'demo' and s['synthetic'] and s['training_label_mode'] == 'synthetic_controller',
            '合成の教師の記録ではありません')
    t = r['context']['teacher']
    require(t.get('id') == 'rl-bestlap-ppo-v1' and t.get('profile') == 'speed-rc' and t.get('uses_ground_truth') is True
            and t.get('policy_sha256') == policy_sha and t.get('control_delay_frames') == 0, '教師の出どころ(RL の方策)が違います')
    require(card['num_modes'] == 9 and card['mode_frames'] == {'8': 600}, 'ジャンル 8 ではありません')
    scene = card['source_scenes']; require(len(scene) == 1, '1 シーンの記録ではありません')
    require(splits.get(p['dataset-split']) == scene, 'train / val が計画と違います')
    require(all(not value for key, value in splits.items() if key in ('train', 'val', 'test') and key != p['dataset-split']), '2 つの分割に入っています')
    physical = {k: v2 for k, v2 in c.items() if k not in ('seed', 'noiseStd')}
    sensors = r['context']['sensors']; rgb = sensors['rgb']; tof = sensors['tof']
    for field, key in [('height_m', 'camera-height'), ('pitch_down_deg', 'camera-pitch'), ('vertical_fov_deg', 'fov'), ('forward_offset_m', 'camera-forward'),
                       ('left_offset_m', 'camera-left')]:
        require(abs(rgb[field] - p[key]) < 1e-12, 'カメラの設定が違います: ' + field)
    for field, key in [('height_m', 'tof-height'), ('pitch_down_deg', 'tof-pitch'), ('vertical_fov_deg', 'tof-vfov'), ('forward_offset_m', 'tof-forward'),
                       ('left_offset_m', 'tof-left')]:
        require(abs(tof[field] - p[key]) < 1e-12, 'ToF の設定が違います: ' + field)
    require(r['context']['sensor_profile'] == p['sensor-profile'], 'センサーの設定が違います')
    manifest = read(root / 'manifest.json')
    fingerprints = {ch: canonical([x['sha256'] for x in sorted(manifest['files'], key=lambda x: x['path']) if x['path'].startswith(f'samples/{ch}/')])
                    for ch in ('CAM_FRONT', 'TOF_FRONT')}
    return {'name': name, 'index': p['_index'], 'split': p['dataset-split'], 'scenario': p['scenario'], 'seed': p['seed'], 'start_variation': v,
            'physical_contract_sha256': canonical(physical), 'sensor_contract_sha256': canonical({'sensors': sensors, 'ego_sensor_profile': r['context']['ego_sensor_profile']}),
            'teacher_contract_sha256': canonical(t), 'world_hash': r['world_hash'], 'content_fingerprints': fingerprints,
            'dataset': str(root), 'dataset_name': card['name'], 'actions_sha256': actions_sha256(r),
            'camera_sha256_list': [x['sha256'] for x in manifest['files'] if x['path'].startswith('samples/CAM_FRONT/')],
            'tof_sha256_list': [x['sha256'] for x in manifest['files'] if x['path'].startswith('samples/TOF_FRONT/')]}


def image_stats(root):
    """RGB の平均・標準偏差(30 フレームおき)と ToF の当たりの割合(下位 13 ビットが 8191 でない画素)"""
    import cv2
    a = np.load(next((root / 'aligned').glob('*.npz')), allow_pickle=False)
    means, stds = [], []
    for rel in a['cam_files'][::30]:
        im = cv2.imread(str(root / str(rel)), cv2.IMREAD_COLOR)
        means.append(float(im.mean())); stds.append(float(im.std()))
    tof = a['tof']
    hit = float(((tof & 8191) != 8191).mean())
    return {'rgb_mean': float(np.mean(means)), 'rgb_std_mean': float(np.mean(stds)), 'rgb_std_min': float(np.min(stds)),
            'tof_hit_fraction': hit, 'tof_unique_frames': int(len({t.tobytes() for t in tof}))}


def teacher_reference_stats():
    """教師の 20 分データ(取り込み済み)の ToF の当たりの割合と RGB の平均(比べる相手)"""
    import cv2
    hits, means = [], []
    for npz in sorted((TEACHER_DATA / 'aligned').glob('*.npz')):
        a = np.load(npz, allow_pickle=False)
        hits.append(float(((a['tof'] & 8191) != 8191).mean()))
        for rel in a['cam_files'][::60]:
            im = cv2.imread(str(TEACHER_DATA / str(rel)), cv2.IMREAD_COLOR)
            if im is not None:
                means.append(float(im.mean()))
    return {'dataset': TEACHER_DATA.name, 'scenes': len(hits), 'tof_hit_fraction_mean': float(np.mean(hits)),
            'tof_hit_fraction_min': float(np.min(hits)), 'tof_hit_fraction_max': float(np.max(hits)), 'rgb_mean': float(np.mean(means)) if means else None}


def audit_scene(sid, datasets, expected, out, node, policy_sha, prediction):
    row = {'session_id': sid, 'pass': False}; stages = []
    try:
        root = resolve_scene(datasets, sid)
        row.update(scene_metadata(root, expected, policy_sha))
        folder = out / sid; folder.mkdir(parents=True, exist_ok=True)
        for stage, cmd in [
                ('physics', [node, str(EXP / 'verify_recording.mjs'), str(root), '--out', str(folder / 'physics.json')]),
                ('teacher', [node, str(HERE / 'teacher_mismatch_probe.mjs'), str(root), '--out', str(folder / 'teacher.json')]),
                ('sensors', [sys.executable, '-I', str(HERE / 'verify_rc_teacher_dataset_rl.py'), str(root), '--output', str(folder / 'sensors.json')])]:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            (folder / (stage + '.log')).write_text(res.stdout[-20000:] + res.stderr[-20000:])
            require(res.returncode == 0, f'{stage} の検証に失敗: ' + (res.stderr or res.stdout)[-2000:])
            payload = read(folder / (stage + '.json'))
            if stage == 'teacher':
                payload['pass'] = payload['max_err_from_browser_state'] <= 1e-9 and payload['frames'] == 600
            require(payload.get('pass') is True, f'{stage}: 合格になりません')
            stages.append(stage)
            row[stage + '_audit'] = {'path': str(folder / (stage + '.json')), 'sha256': sha(folder / (stage + '.json'))}
            if stage == 'physics':
                rc = payload['independent_recount']
                row.update(frames=payload['frames'], reason=payload['outcome']['reason'], collisions=payload['collisions'],
                           passed_gates=payload['passed_gates'], completed_laps=len(payload['laps']),
                           lap_times=[x['elapsed_s'] for x in payload['laps']], lap_gates=[x['gates'] for x in payload['laps']],
                           lap_crossings=[x['shortcut'] for x in payload['laps']], max_abs_error=payload['max_abs_error'],
                           teacher_recomputed=payload['teacher_recomputed'], recount_every_lap_valid=rc['every_lap_valid'],
                           recount_min_clearance_m=rc['min_clearance_m'], recount_gate_margin_min_m=rc['gate_margin_min_m'],
                           recount_gate_misses=rc['gate_misses'], recount_collisions=rc['collisions'])
            elif stage == 'teacher':
                mf = payload['mismatched_frames']
                row.update(teacher_max_err_from_browser_state=payload['max_err_from_browser_state'],
                           teacher_max_err_from_node_replay_state=payload['max_err_from_node_replay_state'],
                           teacher_node_replay_mismatch_frames=[m['frame'] for m in mf],
                           teacher_node_replay_first_jump=[j['name'] for j in mf[0]['obs_jumps_node_vs_browser_state']] if mf else [],
                           max_state_diff_browser_vs_node_replay=payload['max_state_diff_browser_vs_node_replay'])
            else:
                row.update(manifest_files_checked=payload['manifest_files_checked'], genre_id=payload['genre_id'],
                           future_valid_rows=payload['future_valid_rows'], max_imu_error=payload['max_imu_float32_error'],
                           no_imu_reset=payload['no_imu_reset_after_start'], unique_camera_images=payload['unique_decoded_camera_images'],
                           unique_tof_frames=payload['unique_packed_tof_frames'], confidence_codes=payload['confidence_codes'])
        row['image_stats'] = image_stats(root)
        # 記録前の予想との照合(Node + 元の sim-core.js + 同じ差し込みモジュール)
        pred = prediction.get(row['name'])
        require(pred is not None, '予想にないシーンです')
        row['prediction'] = {'actions_sha256_equal': pred['actions_sha256'] == row['actions_sha256'],
                             'max_lap_abs_diff_s': max(abs(a - b['elapsed_s']) for a, b in zip(row['lap_times'], pred['laps']))
                             if len(row['lap_times']) == len(pred['laps']) else None,
                             'predicted_laps': [x['elapsed_s'] for x in pred['laps']], 'predicted_min_clearance_m': pred['min_clearance_m']}
        require(row['frames'] == 600 and row['genre_id'] == 8 and row['completed_laps'] >= 2 and row['collisions'] == 0
                and row['reason'] == 'collection_duration_complete' and row['recount_every_lap_valid'] and row['no_imu_reset']
                and all(g == 33 for g in row['lap_gates']) and all(c == 1 for c in row['lap_crossings']), '20 秒・2 周以上・衝突 0・全周が有効になりません')
        row['pass'] = True
    except Exception as e:  # noqa: BLE001
        row['error'] = str(e); row['completed_stages'] = stages
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sessions_file')
    ap.add_argument('--datasets-root', default=str(SIM / 'runtime/datasets'))
    ap.add_argument('--plan', default=str(REPORT / 'recording/plan_rlbest60.json'))
    ap.add_argument('--prediction', default=str(REPORT / 'recording/prediction.json'))
    ap.add_argument('--node', default=shutil.which('node') or '/usr/bin/node')
    ap.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    ap.add_argument('--output-dir', default=str(REPORT / 'recording/verify'))
    a = ap.parse_args()
    v = read(a.sessions_file); ids = v.get('session_ids') if isinstance(v, dict) else v
    require(isinstance(ids, list) and len(ids) == 60 and len(set(ids)) == 60, 'セッション ID は重複なしの 60 個が要ります(フォルダーを探しには行きません)')
    plan = read(a.plan); require(len(plan) == 60 and len({p['name'] for p in plan}) == 60, '計画が 60 条件ではありません')
    expected = {p['name']: dict(p, _index=i) for i, p in enumerate(plan)}
    require(sum(p['dataset-split'] == 'train' for p in plan) == 48 and sum(p['dataset-split'] == 'val' for p in plan) == 12, '計画が 48 / 12 ではありません')
    install = read(REPORT / 'recording/install.json'); policy_sha = install['manifest']['policy_json_sha256']
    prediction = {s['name']: s for s in read(a.prediction)['scenes']}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(); before = {str(p): sha(p) for p in SOURCES}; rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(audit_scene, sid, Path(a.datasets_root).resolve(), expected, out, a.node, policy_sha, prediction): sid for sid in ids}
        for fut in as_completed(futures):
            row = fut.result(); rows.append(row)
            save(out / 'progress.json', {'started': started, 'complete': len(rows), 'pass_count': sum(r['pass'] for r in rows)})
            print(f"[{len(rows):02}/60] {'合格' if row['pass'] else '不合格'} {row.get('name', row['session_id'])}" + ((': ' + row['error']) if 'error' in row else ''),
                  flush=True)
    rows.sort(key=lambda x: x.get('index', 999)); good = [r for r in rows if r['pass']]; errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)
    after = {str(p): sha(p) for p in SOURCES}; check(before == after, '検証中に検証のファイルが変わりました')
    check(len(good) == 60, '不合格のシーンがあります'); check({r.get('name') for r in rows} == set(expected), '計画の名前がそろっていません')
    for field in ('physical_contract_sha256', 'sensor_contract_sha256', 'teacher_contract_sha256'):
        check(len({r[field] for r in good}) == 1, f'{field} がシーンで違います')
    check(len({canonical(r['start_variation']) for r in good}) == 60, '開始姿勢が 60 通りではありません')
    check(len({canonical(r['content_fingerprints']) for r in good}) == 60, 'カメラ・ToF の列が 60 通りではありません')
    total_frames = sum(r['frames'] for r in good); check(total_frames == 36000, '36,000 フレームではありません')
    cam_all = [h for r in good for h in r['camera_sha256_list']]; tof_all = [h for r in good for h in r['tof_sha256_list']]
    pred_equal = sum(r['prediction']['actions_sha256_equal'] for r in good)
    lap1 = [r['lap_times'][0] for r in good]; lap2 = [r['lap_times'][1] for r in good if len(r['lap_times']) > 1]
    stats = lambda x: {'n': len(x), 'mean': float(np.mean(x)), 'min': float(np.min(x)), 'max': float(np.max(x))} if x else None  # noqa: E731
    ref = teacher_reference_stats()
    hits = [r['image_stats']['tof_hit_fraction'] for r in good]
    hit_mean = float(np.mean(hits)) if hits else None
    check(hit_mean is not None and abs(hit_mean - ref['tof_hit_fraction_mean']) <= .10 * ref['tof_hit_fraction_mean'], 'ToF の当たりの割合が教師データと ±10% 以上違います')
    check(all(r['image_stats']['rgb_std_min'] > 1 and r['image_stats']['rgb_mean'] > 1 for r in good), 'RGB が真っ黒・一様な画像があります')
    for r in good:
        r.pop('camera_sha256_list', None); r.pop('tof_sha256_list', None)
    result = {'schema': 'rl-bestlap-60-scene-integrity/v1', 'pass': not errors,
              'scope': 'session ID で渡した 60 シーンだけ。記録した操作の再生(元の sim-core.js)・方策の再計算・周回の数え直し・保存したセンサーとラベルの整合。'
                       'WebGL の画素の再描画の一致や実車での正しさは確かめていない。運転したのは正解の状態を見る RL の方策(学習した方策)で、カメラで走る AI ではない。',
              'started': started, 'completed': datetime.now(timezone.utc).isoformat(), 'input_sessions_file': str(Path(a.sessions_file).resolve()),
              'input_sessions_sha256': sha(a.sessions_file), 'plan_sha256': sha(a.plan), 'session_ids': ids, 'source_sha256': before,
              'source_unchanged': before == after, 'errors': errors, 'scenes': len(rows), 'passed_scenes': len(good),
              'train_scenes': sum(r.get('split') == 'train' for r in good), 'val_scenes': sum(r.get('split') == 'val' for r in good),
              'noise_scenes': sum(r.get('scenario') == 'noise' for r in good), 'frames': total_frames, 'seconds': total_frames / 30, 'genre_id': 8,
              'future_valid_rows': sum(r['future_valid_rows'] for r in good), 'completed_laps': sum(r['completed_laps'] for r in good),
              'collisions': sum(r['collisions'] for r in good), 'manifest_files_checked': sum(r['manifest_files_checked'] for r in good),
              'max_abs_error': {k: max((r['max_abs_error'].get(k, 0) for r in good), default=None) for k in ('spawn', 'pre', 'time', 'stored', 'replay', 'post', 'own_physics.x', 'own_physics.yaw', 'own_lap_elapsed')},
              'teacher_max_err_from_browser_state': max((r['teacher_max_err_from_browser_state'] for r in good), default=None),
              'teacher_max_err_from_node_replay_state': max((r['teacher_max_err_from_node_replay_state'] for r in good), default=None),
              'teacher_node_replay_scenes_over_1e-9': [r['name'] for r in good if r['teacher_max_err_from_node_replay_state'] > 1e-9],
              'teacher_node_replay_first_jumps': sorted({n for r in good for n in r['teacher_node_replay_first_jump']}),
              'max_imu_error': max((r['max_imu_error'] for r in good), default=None),
              'unique_camera_images_per_scene_min': min((r['unique_camera_images'] for r in good), default=None),
              'unique_tof_frames_per_scene_min': min((r['unique_tof_frames'] for r in good), default=None),
              'camera_files_total': len(cam_all), 'camera_files_distinct_sha256': len(set(cam_all)),
              'tof_files_total': len(tof_all), 'tof_files_distinct_sha256': len(set(tof_all)),
              'prediction_actions_sha256_equal_scenes': pred_equal,
              'prediction_max_lap_abs_diff_s': max((r['prediction']['max_lap_abs_diff_s'] for r in good if r['prediction']['max_lap_abs_diff_s'] is not None), default=None),
              'lap1': stats(lap1), 'lap2': stats(lap2),
              'min_clearance_m': min((r['recount_min_clearance_m'] for r in good), default=None),
              'gate_margin_min_m': min((r['recount_gate_margin_min_m'] for r in good), default=None),
              'tof_hit_fraction_mean': hit_mean, 'teacher_reference': ref,
              'rgb_mean_mean': float(np.mean([r['image_stats']['rgb_mean'] for r in good])) if good else None,
              'rows': rows}
    save(out / 'collection-integrity.json', result)
    print(json.dumps({k: v2 for k, v2 in result.items() if k not in ('rows', 'session_ids', 'source_sha256')}, ensure_ascii=False, indent=1))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
