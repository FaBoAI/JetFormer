#!/usr/bin/env python3
"""確認(理解の段・UI 側): 教師の action に付けた info が、ユーザーの収集器(dataset_recorder.py)と集約
(qa/combine_scene_datasets.py)を通って、シーンごとの simulation/.../frames.jsonl の action_info に残るか

    /home/jetson/jetformer/data/sim/venv/bin/python -B -I recorder_info_probe.py

- app.js は記録のとき boundedAction = {...action, steering, speed} を /api/datasets/frame に送るので、
  教師(差し込み口)が返す {steering, speed, info: {...}} の info もそのまま届く(app.js:387)。
- ここではブラウザを使わず、収集器の DatasetManager を直接呼ぶ(ブリッジの /api/datasets/* と同じ関数)。
- ユーザーのファイルは読むだけ(-B で __pycache__ も作らない)。書くのは --out の下だけ。
- 集約は 20 GiB の空きを要求するので、同じディスク(/home)の下に作る。
"""
import argparse
import base64
import json
import shutil
import sys
import zipfile
from pathlib import Path

RELEASE = Path('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5')
OUT = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/probe_ui/recorder_info')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--frames', type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    root = out / 'datasets'
    root.mkdir(parents=True)
    sys.path.insert(0, str(RELEASE))
    import cv2
    import numpy as np
    from dataset_recorder import DatasetManager

    m = DatasetManager(root)
    n = a.frames
    png = base64.b64encode(cv2.imencode('.png', np.full((224, 224, 3), 128, np.uint8))[1].tobytes()).decode()
    sensors = {'rgb': {'vertical_fov_deg': 36.4, 'render_width': 320, 'render_height': 180, 'pitch_down_deg': 11.8,
                       'forward_offset_m': -.15, 'left_offset_m': 0, 'height_m': .25, 'ground_offset_m': .018},
               'tof': {'vertical_fov_deg': 36.46, 'horizontal_tangent_step': .0193, 'pitch_down_deg': 16,
                       'forward_offset_m': -.15, 'left_offset_m': 0, 'height_m': .2, 'ground_offset_m': .018}}
    config = {'dt': 1 / 30, 'seed': 7, 'episodeMode': 'continuous-laps-v1', 'timeLimit': n / 30, 'maxSpeed': 7.24,
              'maxSteer': .52, 'acceleration': 1.2, 'steerRate': 1.2, 'vehicleLength': .42, 'vehicleWidth': .19,
              'wheelbase': .26, 'stopOnCollision': True, 'frictionMultiplier': 1, 'noiseStd': 0,
              'longitudinalModel': 'real-drive-response-estimate-v1', 'driveTau': .4, 'driveAccelerationCap': 6,
              'brakeDeceleration': 1.2}
    teacher = {'schema': 'jetformer-sim-rl-teacher/v1', 'id': 'probe-recover-v0', 'mode': 'recover'}
    context = {'world_version': 'probe', 'route': 'probe', 'sensor_profile': 'measured-height-approx', 'sensors': sensors,
               'episode_kind': 'continuous-lap-collection', 'teacher': teacher}
    sids, infos = [], []
    for scene_i, split in enumerate(('train', 'val')):
        s = m.start({'name': 'probe-recover-%s-%02d' % (split, scene_i), 'controller': 'demo', 'mode_id': 8, 'split': split,
                     'include_automated_labels': True, 'context': context, 'config': config})
        sid = s['session_id']
        x, z, yaw = -0.85 + 0.01 * scene_i, 2.1, 0.0
        for f in range(n):
            t = f / 30
            # 「離脱復活」の想定: 8〜11 フレームは乱れ(学習しない)、ほかは学習する
            phase = 'perturb' if 8 <= f < 12 else 'drive'
            info = {'jf': {'mode': 'recover', 'phase': phase, 'learn': phase != 'perturb', 'layout': 'L0'}}
            infos.append(info)
            nx = x + 0.05
            obs = {'sim_time': t, 'frame': f, 'episode_id': 'probe-%d' % scene_i, 'dt': 1 / 30, 'speed': 1.5,
                   'camera': {'width': 224, 'height': 224, 'encoding': 'png-base64', 'data': png},
                   'depth': {'width': 54, 'height': 42, 'packed_data': [1000 + scene_i] * 2268},
                   'imu': {'samples': [[0, 0, 9.80665, 0, 0, 0]] * 10, 'timestamps': [t - .09 + .01 * k for k in range(10)]}}
            m.append({'session_id': sid, 'observation': obs,
                      'action': {'steering': .52 if phase == 'perturb' else 0.0, 'speed': 1.5, 'info': info},
                      'pose': {'x': x, 'z': z, 'yaw': yaw}, 'state': {'x': nx, 'z': z, 'yaw': yaw}})
            x = nx
        report = {'config': config, 'episode_kind': 'continuous-lap-collection',
                  'outcome': {'terminal': True, 'success': False, 'reason': 'collection_duration_complete'},
                  'metrics': {'finished': False, 'elapsed': n / 30, 'collisions': 0}, 'context': context}
        r = m.finish({'session_id': sid, 'report': report})
        sids.append(sid)
    col = m.collection({'name': 'probe-recover', 'session_ids': sids, 'expected_frames': n,
                        'expected_duration_s': n / 30, 'require_collision_free': True})
    zpath = root / col['session_id'] / col['filename']
    found = {}
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        ledgers = [x for x in names if x.endswith('/frames.jsonl')]
        for name in ledgers:
            rows = [json.loads(line) for line in zf.read(name).decode().splitlines() if line]
            found[name.split('/', 1)[1]] = {
                'rows': len(rows),
                'action_info_frame0': rows[0]['action_info'],
                'perturb_frames': [r['frame'] for r in rows if r['action_info'].get('jf', {}).get('learn') is False],
                'applied_pwm_steer_perturb': rows[9]['applied_pwm_us'][0],
            }
        coll = json.loads(zf.read([x for x in names if x.endswith('simulation/collection.json')][0]))
        card = json.loads(zf.read([x for x in names if x.endswith('/dataset_card.json')][0]))
        with np.load(zf.open([x for x in names if x.endswith('.npz')][0])) as z0:
            fv = z0['future_valid']
    result = {'schema': 'sim-rl-modes-probe-recorder-info/v1', 'release': RELEASE.name, 'frames_per_scene': n,
              'collection': {k: col[k] for k in ('filename', 'scenes', 'frames', 'training_frames', 'bytes')},
              'ledgers': found, 'contract_teacher': coll['contract']['teacher'],
              'card_tags': card.get('tags'), 'npz_future_valid_true': int(fv.sum()),
              'conclusion': 'action.info は frames.jsonl の action_info に残り、集約後も simulation/scenes/<シーン>/frames.jsonl に残る。'
                            'future_valid は収集器が最後の10フレームだけ false にする(フレームごとの除外は JetFormer の取り込みで行う必要がある)'}
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    shutil.rmtree(root)                       # 作った ZIP は確認用なので消す(結果の JSON だけ残す)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
