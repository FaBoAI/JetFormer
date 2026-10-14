"""データセットのシーンを、収集したシーンと同じ形(data/raw/<ログ>/<シーン>/)に展開する

取り込んだデータセット(シミュレーターで作ったもの・Hugging Face から取ったものなど)には、収集したときの生のシーンが無い。
そのままではシーンタブで1フレームずつ見たり、ジャンル・タグを直したり、別のシーンと混ぜて新しいデータセットを作ったりできない。
そこで、データセットの aligned/<シーン>.npz と画像から、収集と同じ形のシーンを作る:

  CAM_FRONT.npz   timestamps_us・files(画像はデータセットの画像へのハードリンク。別のディスクならコピー)
  TOF_FRONT.npz   timestamps_us(= カメラ時刻 − tof_age_ms)・depth(生の値)
  IMU.npz         各フレームの直前 10 サンプル(100Hz)から作り直した列。gyro は deg/s に戻す(aligned は rad/s)
  VEHICLE.npz     各フレームの「今の操作値」[ステア, スロットル, CH3]・eRPM(30fps)。select はすべて人の側
  scene.json      ジャンルの区間(aligned の mode から)・タグ(データセットのタグ、シミュレーターなら simulator)・取り込み元
  thumb.jpg       中ほどのフレームのカメラ + ToF

ログの名前は「ds-<データセット名>」、シーンは「ds-<データセット名>-0001」から。元のシーンがすでにこの Jetson にある
(自分のデータセットを Hugging Face から取り直したなど)ものは作らない。
"""

import datetime
import json
import os
import shutil
import uuid
from collections import Counter

import cv2
import numpy as np

from . import config, dataset, modes
from .recorder import depth_colormap

LOG_PREFIX = 'ds-'
IMU_DT_US = 10000                     # 100Hz


def log_name_for(dataset_name):
    return LOG_PREFIX + dataset_name


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _segments(mode_arr, t_rel):
    """フレームごとのジャンル → 区間 [{'start_s', 'end_s', 'mode_id'}]"""
    segs = []
    if not len(mode_arr):
        return segs
    start = 0
    for i in range(1, len(mode_arr) + 1):
        if i == len(mode_arr) or mode_arr[i] != mode_arr[start]:
            segs.append({'start_s': round(float(t_rel[start]), 3), 'end_s': round(float(t_rel[i - 1]), 3),
                         'mode_id': int(mode_arr[start])})
            start = i
    return segs


def _imu_stream(cam_t, imu):
    """各フレームの直前 10 サンプルから 100Hz の列を作り直す(重なったサンプルは 5ms 以内なら1つにまとめる)"""
    n, k, _ = imu.shape
    t = (cam_t[:, None] - (k - 1 - np.arange(k))[None, :] * IMU_DT_US).reshape(-1)
    v = imu.reshape(-1, 6)
    order = np.argsort(t, kind='stable')
    t, v = t[order], v[order]
    keep = np.ones(len(t), bool)
    last = None
    for i, ti in enumerate(t):
        if last is not None and ti - last < IMU_DT_US // 2:
            keep[i] = False
        else:
            last = ti
    t, v = t[keep], v[keep]
    return {'timestamps_us': t.astype(np.int64), 'accel': v[:, :3].astype(np.float32),
            'gyro': np.rad2deg(v[:, 3:]).astype(np.float32), 'mag': np.zeros((len(t), 3), np.float32),
            'temp': np.zeros(len(t), np.float32)}


def _existing_scene_names():
    return {s['name'] for s in dataset.list_scenes()}


def dataset_to_scenes(name, progress=None, p0=0.0, p1=1.0):
    """データセット name のシーンを data/raw/ds-<name>/ に作る。戻り値 {'log', 'created': [...], 'skipped': [...]}"""
    root = config.DATASET_DIR / name
    if '..' in name or '/' in name or not (root / 'dataset_card.json').exists():
        raise KeyError(name)
    card = json.loads((root / 'dataset_card.json').read_text(encoding='utf-8'))
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    scene_names = list(splits.get('train', [])) + list(splits.get('val', []))
    sim = card.get('source') in ('simulator', 'mixed') or 'simulator' in (card.get('tags') or [])
    log = log_name_for(name)
    log_dir = config.RAW_DIR / log
    if log_dir.exists():
        marker = log_dir / 'imported_dataset.json'
        if not marker.exists():
            raise FileExistsError('ログ %s が別の用途で使われています' % log)
        shutil.rmtree(log_dir)                       # 同じデータセットから作り直す
    existing = _existing_scene_names()
    known_modes = {m['id'] for m in modes.load()}
    created, skipped = [], []
    log_dir.mkdir(parents=True)
    with open(log_dir / 'imported_dataset.json', 'w', encoding='utf-8') as f:
        json.dump({'dataset': name, 'created_at': datetime.datetime.now().isoformat(timespec='seconds')}, f, ensure_ascii=False)
    log_token = uuid.uuid4().hex
    log_json = {'token': log_token, 'logfile': log, 'vehicle': 'simulator' if sim else 'dataset:%s' % name,
                'date_captured': datetime.date.today().isoformat(), 'location': '',
                'description': 'データセット %s から展開' % name, 'tags': ['simulator'] if sim else [],
                'scene_seconds': None, 'sensors': {}, 'calibration': getattr(config, 'CALIBRATION', {}), 'drive': None,
                'imported_from': {'dataset': name}}
    with open(log_dir / 'log.json', 'w', encoding='utf-8') as f:
        json.dump(log_json, f, ensure_ascii=False, indent=1)
    for idx, orig in enumerate(scene_names, start=1):
        if orig in existing:                         # 元のシーンがこの Jetson にある
            skipped.append(orig)
            continue
        with np.load(root / 'aligned' / ('%s.npz' % orig), allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        cam_t = a['timestamps_us'].astype(np.int64)
        n = len(cam_t)
        if n == 0:
            skipped.append(orig)
            continue
        scene = '%s-%04d' % (log, idx)
        d = log_dir / scene
        (d / 'CAM_FRONT').mkdir(parents=True)
        files = []
        for i in range(n):
            rel = 'CAM_FRONT/%d.jpg' % cam_t[i]
            src = root / str(a['cam_files'][i])
            if src.exists():
                _link_or_copy(src, d / rel)
            files.append(rel)
        np.savez(d / 'CAM_FRONT.npz', timestamps_us=cam_t, seq=np.arange(n, dtype=np.int64), files=np.array(files))
        tof_t = cam_t - (a['tof_age_ms'].astype(np.float64) * 1000).astype(np.int64) if 'tof_age_ms' in a else cam_t
        u, first = np.unique(tof_t, return_index=True)
        np.savez(d / 'TOF_FRONT.npz', timestamps_us=u, seq=np.arange(len(u), dtype=np.int64), depth=a['tof'][first])
        imu = _imu_stream(cam_t, a['imu'].astype(np.float32))
        np.savez(d / 'IMU.npz', **imu)
        veh = a['vehicle'].astype(np.float64)
        veh_t = cam_t - (a['vehicle_age_ms'].astype(np.float64) * 1000).astype(np.int64) if 'vehicle_age_ms' in a else cam_t
        valid = a['vehicle_valid'].astype(bool)
        erpm = np.round(veh[:, 3]).astype(np.int32)
        np.savez(d / 'VEHICLE.npz', timestamps_us=veh_t.astype(np.int64), pwm=np.round(veh[:, :3]).astype(np.int32),
                 valid=np.repeat(valid[:, None], 3, axis=1), select=np.zeros(n, bool), moving=np.abs(erpm) > 100,
                 erpm=erpm, hall_count=np.zeros(n, np.int64), hall_errors=np.zeros(n, np.int32),
                 hall_state=np.zeros(n, np.uint8), seq=np.arange(n, dtype=np.int64), board_t_us=veh_t.astype(np.int64),
                 quality=np.array('snapshot'))
        mode_arr = a['mode'].astype(int) if 'mode' in a else np.zeros(n, int)
        mode_arr = np.where(np.isin(mode_arr, list(known_modes)), mode_arr, 0)
        t_rel = (cam_t - cam_t[0]) / 1e6
        nz = [m for m in mode_arr.tolist() if m]
        mode_id = Counter(nz).most_common(1)[0][0] if nz else 0
        dur = round(float(t_rel[-1]), 2)
        gaps = np.diff(cam_t) / 1000.0 if n > 1 else np.zeros(1)
        steer, thr = veh[valid, 0], veh[valid, 1]
        st = lambda x: {'min': int(x.min()), 'max': int(x.max()), 'mean': round(float(x.mean()), 1)} if len(x) else None  # noqa: E731
        tags = dataset.normalize_tags(list(card.get('tags') or []) + (['simulator'] if sim else []))
        meta = {
            'token': uuid.uuid4().hex, 'name': scene, 'log_token': log_token, 'log': log,
            'description': '%s から取り込み(%s)' % (name, orig), 'tags': tags,
            'mode_id': int(mode_id), 'mode_segments': _segments(mode_arr, t_rel),
            'start_us': int(cam_t[0]), 'end_us': int(cam_t[-1]), 'duration_s': dur, 'complete': True,
            'counts': {'CAM_FRONT': n, 'TOF_FRONT': int(len(u)), 'IMU': int(len(imu['timestamps_us'])), 'VEHICLE': n},
            'rates': {'CAM_FRONT': round(n / max(dur, 1e-3), 1)},
            'stats': {'steering_us': st(steer), 'throttle_us': st(thr),
                      'erpm': {'min': int(erpm.min()), 'max': int(erpm.max()), 'mean': round(float(erpm.mean()), 1)},
                      'ai_driving_ratio': 0.0, 'rc_lost_ratio': round(float(1 - valid.mean()), 3),
                      'camera_max_gap_ms': round(float(gaps.max()), 1)},
            'vehicle_quality': 'snapshot', 'drive': None,
            'source': 'simulator' if sim else 'dataset',
            'imported_from': {'dataset': name, 'scene': orig},
            'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        lm = ((card.get('label_mask') or {}).get('per_scene') or {}).get(orig)
        if lm is not None:
            # シミュレーターの強化学習モードのシーン: 乱した操作を正解に含むフレームは「未設定」(mode 0)。ジャンルを付け直しても
            # この区間は未設定のまま残す(dataset.update_scene)。消すと乱した操作が正解に戻ってしまう(確認 recover r2)
            meta['label_mask'] = {'schema': card['label_mask'].get('schema'), 'rule': card['label_mask'].get('rule_id'),
                                  'masked_frames': int(lm.get('masked', 0)), 'teacher_mode': lm.get('teacher_mode'),
                                  'unset_segments': [sg for sg in meta['mode_segments'] if sg['mode_id'] == 0]}
        with open(d / 'scene.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        mid = n // 2
        img = cv2.imread(str(d / files[mid]))
        if img is not None:
            thumb = np.hstack([img, depth_colormap(a['tof'][mid], img.shape[1], img.shape[0])])
            cv2.imwrite(str(d / 'thumb.jpg'), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        created.append(scene)
        if progress:
            progress(p0 + (p1 - p0) * idx / max(len(scene_names), 1), 'シーンに展開中 %d / %d' % (idx, len(scene_names)))
    if not created:
        shutil.rmtree(log_dir, ignore_errors=True)
    info = {'log': log if created else None, 'created': created, 'skipped': skipped}
    card_path = root / 'dataset_card.json'
    card = json.loads(card_path.read_text(encoding='utf-8'))
    card['scene_log'] = info['log']
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding='utf-8')
    return info
