"""シーン管理と、nuScenes形式のデータセット生成・エクスポート

データセットのレイアウト(data/datasets/<name>/):
  v1.0-<name>/   nuScenes互換の表(JSON)
      log, scene, sample, sample_data, sensor, calibrated_sensor, ego_pose
      category, attribute, visibility, instance, sample_annotation, map(空。devkit互換用)
  samples/CAM_FRONT/<scene>/<timestamp_us>.jpg       カメラ(30Hz、全フレームがキーフレーム)
  samples/TOF_FRONT/<scene>/<timestamp_us>.png       ToF距離(16bit PNG)
  can_bus/<scene>_vehicle_monitor.json               PWM・モーター(nuScenes CAN busエクスパンション風)
  can_bus/<scene>_ms_imu.json                        IMU
  aligned/<scene>.npz                                学習用: カメラ各フレームに揃えた全センサー
  splits.json / dataset_card.json
"""

import datetime
import hashlib
import json
import os
import shutil
import tarfile

import cv2
import numpy as np

from . import config, modes, tof_format

SENSORS = [
    ('CAM_FRONT', 'camera'),
    ('TOF_FRONT', 'tof'),
    ('IMU', 'imu'),
    ('VEHICLE', 'can_bus'),
]
IMU_HISTORY = 10          # 学習用: 各フレームの直前10サンプル(100ms)のIMU
FUTURE_STEPS = 10         # 学習用: 各フレームから先の操作値(33ms刻み=約0.33秒先まで)
VEHICLE_MAX_AGE_MS = 100  # 学習用: これより古い操作値しか無いフレームは無効にする


def _token(*parts):
    """入力から決まる再現性のあるトークン(同じデータから作り直しても変わらない)"""
    return hashlib.md5('/'.join(str(p) for p in parts).encode()).hexdigest()


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError('JSONに変換できない型: %s' % type(o).__name__)


def _write_json(path, obj, indent=None):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_np_default)
    os.replace(tmp, path)


# ---------------------------------------------------------------- シーン管理
def _with_modes(s):
    """ジャンル情報の無い古いシーンは「未設定」とし、名前はジャンル表から引き直す"""
    if 'mode_id' not in s:
        s['mode_id'] = modes.UNSET
        s['mode_segments'] = [{'start_s': 0.0, 'end_s': s.get('duration_s', 0.0), 'mode_id': modes.UNSET}]
    s['mode_name'] = modes.name_of(s['mode_id'])
    for sg in s.get('mode_segments', []):
        sg['mode_name'] = modes.name_of(sg['mode_id'])
    s['mode_mixed'] = len({sg['mode_id'] for sg in s.get('mode_segments', [])}) > 1
    return s


def list_scenes():
    scenes = []
    if not config.RAW_DIR.exists():
        return scenes
    for log_dir in sorted(config.RAW_DIR.iterdir()):
        if not log_dir.is_dir():
            continue
        for scene_dir in sorted(log_dir.iterdir()):
            meta = scene_dir / 'scene.json'
            if meta.exists():
                try:
                    s = json.loads(meta.read_text(encoding='utf-8'))
                except ValueError:
                    continue
                s['path'] = str(scene_dir)
                scenes.append(_with_modes(s))
    return scenes


def scene_path(name):
    log = name.rsplit('-', 1)[0]
    p = config.RAW_DIR / log / name
    if not (p / 'scene.json').exists() or '..' in name:
        raise KeyError(name)
    return p


def load_scene_meta(name):
    p = scene_path(name)
    s = json.loads((p / 'scene.json').read_text(encoding='utf-8'))
    s['path'] = str(p)
    return _with_modes(s)


def update_scene(name, description=None, tags=None, mode_id=None):
    p = scene_path(name)
    s = json.loads((p / 'scene.json').read_text(encoding='utf-8'))
    if mode_id is not None:
        mode_id = modes.validate(mode_id)
        s['mode_id'] = mode_id
        s['mode_segments'] = [{'start_s': 0.0, 'end_s': s.get('duration_s', 0.0), 'mode_id': mode_id}]
    if description is not None:
        s['description'] = description
    if tags is not None:
        s['tags'] = [t for t in tags if t]
    for sg in s.get('mode_segments', []):
        sg.pop('mode_name', None)
    s.pop('mode_name', None)
    s.pop('mode_mixed', None)
    _write_json(p / 'scene.json', s, indent=1)
    return _with_modes(s)


def delete_scene(name):
    p = scene_path(name)
    shutil.rmtree(p)
    log_dir = p.parent
    if not any(d.is_dir() for d in log_dir.iterdir()):
        shutil.rmtree(log_dir)


def _load(p, sensor):
    f = p / ('%s.npz' % sensor)
    return dict(np.load(f, allow_pickle=False)) if f.exists() else None


def scene_timeseries(name, max_points=600):
    """シーン詳細画面のグラフ用(間引き済み)"""
    p = scene_path(name)
    out = {}
    cam = _load(p, 'CAM_FRONT')
    t0 = int(cam['timestamps_us'][0]) if cam is not None else 0
    out['frames'] = len(cam['timestamps_us']) if cam is not None else 0
    out['frame_t'] = ((cam['timestamps_us'] - t0) / 1e6).round(3).tolist() if cam is not None else []

    def thin(t, arrs):
        idx = np.linspace(0, len(t) - 1, min(len(t), max_points)).astype(int)
        return ((t[idx] - t0) / 1e6).round(3).tolist(), [a[idx] for a in arrs]

    veh = _load(p, 'VEHICLE')
    if veh is not None:
        n = len(veh['timestamps_us'])
        erpm = veh['erpm'] if 'erpm' in veh else np.zeros(n, np.int32)
        select = veh['select'] if 'select' in veh else np.zeros(n, bool)
        t, (pwm, er, sel, val) = thin(veh['timestamps_us'], [veh['pwm'], erpm, select, veh['valid']])
        ok = val[:, 0] & val[:, 1]
        out['vehicle'] = {'t': t, 'steering': np.where(ok, pwm[:, 0], 0).tolist(),
                          'throttle': np.where(ok, pwm[:, 1], 0).tolist(), 'mode': pwm[:, 2].tolist(),
                          'erpm': er.tolist(), 'select': sel.astype(int).tolist()}
    imu = _load(p, 'IMU')
    if imu is not None:
        t, (acc, gyr) = thin(imu['timestamps_us'], [imu['accel'], imu['gyro']])
        out['imu'] = {'t': t, 'accel': acc.round(3).tolist(), 'gyro': gyr.round(2).tolist()}
    tof = _load(p, 'TOF_FRONT')
    if tof is not None:
        d = tof_format.distance_mm(tof['depth'].reshape(len(tof['depth']), -1))
        d[d == 0] = np.nan
        with np.errstate(all='ignore'):
            mn, md = np.nan_to_num(np.nanmin(d, axis=1)), np.nan_to_num(np.nanmedian(d, axis=1))
        t, (mn, md) = thin(tof['timestamps_us'], [mn, md])
        out['tof'] = {'t': t, 'min': mn.tolist(), 'median': md.tolist()}
    audio = _load(p, 'AUDIO')
    if audio is not None and len(audio['timestamps_us']):
        n = len(audio['timestamps_us'])
        x = audio['pcm'].astype(np.float64) / 2.0 ** 31
        edges = np.concatenate([[0], np.cumsum(audio['chunk_samples'])])
        rms = np.array([np.sqrt(np.mean(x[edges[i]:edges[i + 1]] ** 2)) if edges[i + 1] > edges[i] else 0.0 for i in range(n)])
        t, (lv,) = thin(audio['timestamps_us'], [20 * np.log10(np.maximum(rms, 1e-12))])
        out['audio'] = {'t': t, 'rms_dbfs': np.maximum(lv, -120).round(1).tolist()}
    ai = _load(p, 'AI_CMD')
    if ai is not None and len(ai['timestamps_us']):
        t, (raw, outv, hold, sel) = thin(ai['timestamps_us'], [ai['raw'], ai['out'], ai['hold'], ai['select']])
        nan_none = lambda a: [None if not np.isfinite(x) else round(float(x), 1) for x in a]    # JSONはNaN不可
        out['ai'] = {'t': t, 'raw_steer': nan_none(raw[:, 0]), 'raw_throttle': nan_none(raw[:, 1]),
                     'out_steer': nan_none(outv[:, 0]), 'out_throttle': nan_none(outv[:, 1]),
                     'hold': hold.astype(int).tolist(), 'select': sel.astype(int).tolist()}
    return out


def scene_audio_wav(name):
    """シーンのマイク音声を、聞きやすいよう最大値をそろえた16bit WAV(bytes)にする"""
    import io
    import wave
    audio = _load(scene_path(name), 'AUDIO')
    if audio is None or not len(audio['pcm']):
        raise KeyError(name)
    x = audio['pcm'].astype(np.float64)
    peak = np.abs(x).max()
    y = (x / peak * 0.89 * 32767).astype('<i2') if peak > 0 else np.zeros(len(x), '<i2')   # 約-1dBFSにそろえる
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(audio['sample_rate']))
        w.writeframes(y.tobytes())
    return buf.getvalue()


def scene_frame(name, index):
    """(カメラ画像, そのフレーム時刻に最も近いToF距離画像) を返す"""
    p = scene_path(name)
    cam = _load(p, 'CAM_FRONT')
    index = int(np.clip(index, 0, len(cam['files']) - 1))
    image = cv2.imread(str(p / str(cam['files'][index])))
    depth = None
    tof = _load(p, 'TOF_FRONT')
    if tof is not None and len(tof['timestamps_us']):
        j = int(np.argmin(np.abs(tof['timestamps_us'] - cam['timestamps_us'][index])))
        depth = tof['depth'][j]
    return image, depth, int(cam['timestamps_us'][index])


# ---------------------------------------------------------------- データセット生成
def _latest_index(src_t, query_t):
    """各query_tに対し、src_t <= query_t となる最新のインデックス(無ければ-1)"""
    return np.searchsorted(src_t, query_t, side='right') - 1


def _frame_modes(meta, cam_t):
    """シーンのジャンル区間から、各カメラフレームのジャンルIDを求める"""
    rel = (cam_t - cam_t[0]) / 1e6
    out = np.full(len(cam_t), meta.get('mode_id', modes.UNSET), dtype=np.int16)
    for sg in meta.get('mode_segments', []):
        out[(rel >= sg['start_s'] - 1e-3) & (rel <= sg['end_s'] + 1e-3)] = sg['mode_id']
    return out


def _aligned_arrays(cam_t, tof, imu, veh):
    n = len(cam_t)
    out = {'timestamps_us': cam_t}
    if tof is not None:
        j = np.clip(_latest_index(tof['timestamps_us'], cam_t), 0, None)
        out['tof'] = tof['depth'][j]
        out['tof_age_ms'] = ((cam_t - tof['timestamps_us'][j]) / 1000.0).astype(np.float32)
    if imu is not None:
        j = _latest_index(imu['timestamps_us'], cam_t)
        hist = np.zeros((n, IMU_HISTORY, 6), dtype=np.float32)
        feats = np.concatenate([imu['accel'], np.deg2rad(imu['gyro'])], axis=1)   # m/s^2, rad/s
        for k in range(IMU_HISTORY):
            idx = np.clip(j - (IMU_HISTORY - 1 - k), 0, len(feats) - 1)
            hist[:, k] = feats[idx]
        out['imu'] = hist
    if veh is not None:
        # 人が操縦しているサンプルだけを使う: ステア・スロットルとも受信中で、SELECTがAI側でないこと
        # (信号断の0や、AI走行中のプロポの値を学習に混ぜない)
        nv = len(veh['timestamps_us'])
        select = veh['select'] if 'select' in veh else np.zeros(nv, bool)
        erpm = veh['erpm'] if 'erpm' in veh else np.zeros(nv, np.int32)
        ok = veh['valid'][:, 0] & veh['valid'][:, 1] & ~select
        state_all = np.concatenate([veh['pwm'][:, :3].astype(np.float32), erpm[:, None].astype(np.float32)], axis=1)
        vt, state = veh['timestamps_us'][ok], state_all[ok]
        max_age_us = VEHICLE_MAX_AGE_MS * 1000
        if len(vt) == 0:
            out['vehicle'] = np.zeros((n, 4), dtype=np.float32)
            out['erpm_valid'] = np.zeros(n, dtype=bool)
            out['vehicle_valid'] = np.zeros(n, dtype=bool)
            out['future_actions'] = np.zeros((n, FUTURE_STEPS, 2), dtype=np.float32)
            out['future_valid'] = np.zeros(n, dtype=bool)
        else:
            j = _latest_index(vt, cam_t)
            jc = np.clip(j, 0, len(vt) - 1)
            out['vehicle'] = state[jc]                                   # 現在の [steering_us, throttle_us, mode_us, erpm]
            out['vehicle_age_ms'] = ((cam_t - vt[jc]) / 1000.0).astype(np.float32)
            out['vehicle_valid'] = (j >= 0) & (cam_t - vt[jc] <= max_age_us)
            fut = np.zeros((n, FUTURE_STEPS, 2), dtype=np.float32)
            fut_ok = np.ones(n, dtype=bool)
            frame_us = 1e6 / config.CAMERA['fps']
            for k in range(FUTURE_STEPS):
                tk = cam_t + int((k + 1) * frame_us)
                jk = _latest_index(vt, tk)
                jkc = np.clip(jk, 0, len(vt) - 1)
                fut[:, k] = state[jkc, :2]
                fut_ok &= (jk >= 0) & (tk - vt[jkc] <= max_age_us)
            out['future_actions'] = fut                                  # 未来の [steering_us, throttle_us]
            horizon = cam_t + int(FUTURE_STEPS * frame_us)
            out['future_valid'] = fut_ok & (horizon <= veh['timestamps_us'][-1])
            out['erpm_valid'] = np.full(n, 'erpm' in veh)              # 旧ファームには回転数が無い(0で埋めている)
    return out


def build_dataset(name, scene_names, val_ratio=0.2, min_duration=5.0, seed=0, progress=None):
    """選んだシーンからnuScenes形式のデータセットを作る。dataset_card(dict)を返す"""
    if not name or any(c in name for c in '/\\.') or ' ' in name:
        raise ValueError('データセット名に使えない文字が含まれています')
    root = config.DATASET_DIR / name
    if root.exists():
        raise ValueError('同名のデータセットがあります: %s' % name)
    metas, skipped_legacy = [], []
    for sn in scene_names:
        m = load_scene_meta(sn)
        if m.get('vehicle_quality') == 'legacy':      # 旧ファームv1の操作値はチャネルが混ざるので使わない
            skipped_legacy.append(sn)
        elif m['duration_s'] >= min_duration:
            metas.append(m)
    if not metas:
        raise ValueError('条件を満たすシーンがありません(最短 %.1f 秒%s)' % (
            min_duration, '、旧ファームv1のシーン %d 件は使えません' % len(skipped_legacy) if skipped_legacy else ''))

    version = 'v1.0-%s' % name
    tables_dir = root / version
    for d in (tables_dir, root / 'samples' / 'CAM_FRONT', root / 'samples' / 'TOF_FRONT', root / 'can_bus', root / 'aligned'):
        d.mkdir(parents=True, exist_ok=True)

    sensor_tbl = [{'token': _token('sensor', ch), 'channel': ch, 'modality': mod} for ch, mod in SENSORS]
    sensor_tok = {s['channel']: s['token'] for s in sensor_tbl}
    T = {k: [] for k in ('log', 'scene', 'sample', 'sample_data', 'calibrated_sensor', 'ego_pose')}
    logs_done = {}
    calib_tok = {}
    totals = {'frames': 0, 'tof_frames': 0, 'seconds': 0.0}
    mode_frames = {}

    for n_done, meta in enumerate(metas):
        p = scene_path(meta['name'])
        log_json = json.loads((p.parent / 'log.json').read_text(encoding='utf-8'))
        if log_json['token'] not in logs_done:
            T['log'].append({'token': log_json['token'], 'logfile': log_json['logfile'], 'vehicle': log_json['vehicle'],
                             'date_captured': log_json['date_captured'], 'location': log_json.get('location', '')})
            calib = log_json.get('calibration', config.CALIBRATION)
            for ch, _ in SENSORS:
                c = calib.get(ch, {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]})
                tok = _token('calib', log_json['token'], ch)
                T['calibrated_sensor'].append({'token': tok, 'sensor_token': sensor_tok[ch],
                                               'translation': c['translation'], 'rotation': c['rotation'],
                                               'camera_intrinsic': c.get('camera_intrinsic', [])})
                calib_tok[(log_json['token'], ch)] = tok
            logs_done[log_json['token']] = log_json

        cam = _load(p, 'CAM_FRONT')
        tof = _load(p, 'TOF_FRONT')
        imu = _load(p, 'IMU')
        veh = _load(p, 'VEHICLE')
        cam_t = cam['timestamps_us'].astype(np.int64)
        scene_tok = meta['token']
        sname = meta['name']

        # 自車姿勢: 位置は推定できないため原点、向きはジャイロz軸の積分(ヨー角)
        yaw_at = None
        if imu is not None and len(imu['timestamps_us']) > 1:
            it = imu['timestamps_us'].astype(np.int64)
            dt = np.diff(it, prepend=it[0]) / 1e6
            yaw = np.cumsum(np.deg2rad(imu['gyro'][:, 2]) * dt)
            yaw_at = lambda t: float(np.interp(t, it, yaw))

        def ego(t):
            tok = _token('ego', sname, t)
            y = yaw_at(t) if yaw_at else 0.0
            T['ego_pose'].append({'token': tok, 'timestamp': int(t), 'translation': [0.0, 0.0, 0.0],
                                  'rotation': [float(np.cos(y / 2)), 0.0, 0.0, float(np.sin(y / 2))]})
            return tok

        # sample(全カメラフレーム) + CAMのsample_data
        sample_toks = [_token('sample', sname, int(t)) for t in cam_t]
        cam_dir = root / 'samples' / 'CAM_FRONT' / sname
        cam_dir.mkdir(exist_ok=True)
        cam_sd = []
        for i, t in enumerate(cam_t):
            T['sample'].append({'token': sample_toks[i], 'timestamp': int(t), 'scene_token': scene_tok,
                                'prev': sample_toks[i - 1] if i else '',
                                'next': sample_toks[i + 1] if i + 1 < len(cam_t) else ''})
            src = p / str(cam['files'][i])
            rel = 'samples/CAM_FRONT/%s/%d.jpg' % (sname, int(t))
            dst = root / rel
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            cam_sd.append({'token': _token('sd', 'CAM_FRONT', sname, int(t)), 'sample_token': sample_toks[i],
                           'ego_pose_token': ego(t), 'calibrated_sensor_token': calib_tok[(log_json['token'], 'CAM_FRONT')],
                           'timestamp': int(t), 'fileformat': 'jpg', 'is_key_frame': True,
                           'height': config.CAMERA['height'], 'width': config.CAMERA['width'], 'filename': rel})
        # ToFのsample_data(最も近いsampleに紐付け)
        tof_sd = []
        if tof is not None:
            tof_dir = root / 'samples' / 'TOF_FRONT' / sname
            tof_dir.mkdir(exist_ok=True)
            tt = tof['timestamps_us'].astype(np.int64)
            nearest = np.clip(np.searchsorted(cam_t, tt), 0, len(cam_t) - 1)
            for k, t in enumerate(tt):
                i = int(nearest[k])
                if i > 0 and abs(cam_t[i - 1] - t) < abs(cam_t[i] - t):
                    i -= 1
                rel = 'samples/TOF_FRONT/%s/%d.png' % (sname, int(t))
                cv2.imwrite(str(root / rel), tof['depth'][k].astype(np.uint16))
                h, w = tof['depth'][k].shape
                tof_sd.append({'token': _token('sd', 'TOF_FRONT', sname, int(t)), 'sample_token': sample_toks[i],
                               'ego_pose_token': ego(t), 'calibrated_sensor_token': calib_tok[(log_json['token'], 'TOF_FRONT')],
                               'timestamp': int(t), 'fileformat': 'png16', 'is_key_frame': bool(abs(cam_t[i] - t) < 17000),
                               'height': int(h), 'width': int(w), 'filename': rel})
        for sd_list in (cam_sd, tof_sd):
            for k, sd in enumerate(sd_list):
                sd['prev'] = sd_list[k - 1]['token'] if k else ''
                sd['next'] = sd_list[k + 1]['token'] if k + 1 < len(sd_list) else ''
            T['sample_data'].extend(sd_list)

        T['scene'].append({'token': scene_tok, 'log_token': log_json['token'], 'nbr_samples': len(cam_t),
                           'first_sample_token': sample_toks[0], 'last_sample_token': sample_toks[-1],
                           'name': sname, 'description': meta.get('description', ''),
                           'tags': meta.get('tags', []), 'duration_s': meta['duration_s'],
                           'mode_id': meta['mode_id'], 'mode_name': meta['mode_name'],
                           'mode_segments': meta['mode_segments'],
                           'vehicle_quality': meta.get('vehicle_quality')})

        # CAN bus風の時系列
        if veh is not None:
            nv = len(veh['timestamps_us'])
            erpm = veh['erpm'] if 'erpm' in veh else np.zeros(nv, np.int32)
            select = veh['select'] if 'select' in veh else np.zeros(nv, bool)
            _write_json(root / 'can_bus' / ('%s_vehicle_monitor.json' % sname), [
                {'utime': int(t), 'steering_us': int(pw[0]), 'throttle_us': int(pw[1]), 'mode_us': int(pw[2]),
                 'erpm': int(e), 'ai_driving': bool(sel), 'valid': [bool(x) for x in v[:3]]}
                for t, pw, e, sel, v in zip(veh['timestamps_us'], veh['pwm'], erpm, select, veh['valid'])])
        if imu is not None:
            _write_json(root / 'can_bus' / ('%s_ms_imu.json' % sname), [
                {'utime': int(t), 'linear_accel': [round(float(x), 4) for x in a],
                 'rotation_rate': [round(float(x), 5) for x in np.deg2rad(g)], 'temperature': round(float(tp), 2)}
                for t, a, g, tp in zip(imu['timestamps_us'], imu['accel'], imu['gyro'], imu['temp'])])

        # 学習用にカメラフレームへ揃えた配列
        aligned = _aligned_arrays(cam_t, tof, imu, veh)
        aligned['cam_files'] = np.array(['samples/CAM_FRONT/%s/%d.jpg' % (sname, int(t)) for t in cam_t])
        aligned['mode'] = _frame_modes(meta, cam_t)
        for m, c in zip(*np.unique(aligned['mode'], return_counts=True)):
            mode_frames[int(m)] = mode_frames.get(int(m), 0) + int(c)
        np.savez_compressed(root / 'aligned' / ('%s.npz' % sname), **aligned)

        totals['frames'] += len(cam_t)
        totals['tof_frames'] += len(tof_sd)
        totals['seconds'] += meta['duration_s']
        if progress:
            progress((n_done + 1) / len(metas), sname)

    for k, rows in T.items():
        _write_json(tables_dir / ('%s.json' % k), rows)
    _write_json(tables_dir / 'sensor.json', sensor_tbl)
    for k in ('category', 'attribute', 'visibility', 'instance', 'sample_annotation', 'map'):
        _write_json(tables_dir / ('%s.json' % k), [])

    # シーン単位で学習/検証に分割(同じシーンのフレームが両方に入らないように)
    rng = np.random.default_rng(seed)
    names = [m['name'] for m in metas]
    order = rng.permutation(len(names))
    n_val = int(round(len(names) * val_ratio)) if len(names) > 1 else 0
    val = sorted(names[i] for i in order[:n_val])
    train = sorted(names[i] for i in order[n_val:])
    _write_json(root / 'splits.json', {'train': train, 'val': val}, indent=1)

    card = {
        'name': name, 'version': version,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'scenes': len(metas), 'train_scenes': len(train), 'val_scenes': len(val),
        'frames': totals['frames'], 'tof_frames': totals['tof_frames'], 'hours': round(totals['seconds'] / 3600, 3),
        'sensors': {
            'CAM_FRONT': {'width': config.CAMERA['width'], 'height': config.CAMERA['height'], 'fps': config.CAMERA['fps'],
                          'format': 'jpg BGR', 'pipeline': 'V4L2 RAW + 簡易ISP(v4l2_camera.py)'},
            'TOF_FRONT': {'model': 'VL53L9', 'zones': '54x42 (binning 2)',
                          'format': '16bit PNG(生の値: 下位13bit=距離mm、0x1FFF=無効、上位3bit=信頼度0-7)'},
            'IMU': {'model': 'ICM-20948', 'rate_hz': config.IMU['rate_hz'], 'accel': 'm/s^2', 'gyro': 'rad/s (can_bus, aligned)'},
            'VEHICLE': {'board': 'コントローラ基板 XIAO ESP32-S3 (leracer_ctrl v2)',
                        'pwm': 'µs [CH1 steering, CH2 throttle, CH3 mode]', 'erpm': '電気角rpm(符号=回転方向)',
                        'ai_driving': 'SELECT=AI走行側のサンプルは学習の操作値に使わない'},
        },
        'aligned_npz': {
            'timestamps_us': '[N] カメラ時刻', 'cam_files': '[N] 画像の相対パス',
            'tof': '[N,42,54] uint16 直前のToF(生の値。jetformer.tof_format で距離mm・信頼度に分解)', 'tof_age_ms': '[N]',
            'imu': '[N,%d,6] 直前%dサンプルの accel(m/s^2) + gyro(rad/s)' % (IMU_HISTORY, IMU_HISTORY),
            'vehicle': '[N,4] 現在の steering_us, throttle_us, mode_us(CH3), erpm',
            'vehicle_valid': '[N] 100ms以内に有効な操作値(ステア・スロットル)があるか',
            'vehicle_age_ms': '[N] 使った操作値の古さ',
            'future_actions': '[N,%d,2] 1〜%dフレーム先の steering_us, throttle_us' % (FUTURE_STEPS, FUTURE_STEPS),
            'future_valid': '[N] 未来の操作値が全て有効(100ms以内)でシーン内に収まるか',
            'mode': '[N] int16 フレームごとの走行ジャンルID(modes参照、0=未設定)',
            'erpm_valid': '[N] 回転数が計測されているか(旧ファームは無いので偽)',
        },
        'vehicle_quality': sorted({str(m.get('vehicle_quality')) for m in metas}),
        'modes': modes.load(),                    # ジャンル表(IDはモデルの出力ヘッド番号)
        'mode_frames': {str(k): v for k, v in sorted(mode_frames.items())},
        'ego_pose': '位置は未推定(原点)。向きはIMUジャイロのヨー積分',
        'source_scenes': names,
        'skipped_legacy_scenes': skipped_legacy,
    }
    _write_json(root / 'dataset_card.json', card, indent=1)
    return card


def list_datasets():
    out = []
    if config.DATASET_DIR.exists():
        for d in sorted(config.DATASET_DIR.iterdir()):
            card = d / 'dataset_card.json'
            if card.exists():
                c = json.loads(card.read_text(encoding='utf-8'))
                exp = config.EXPORT_DIR / ('%s.tar' % c['name'])
                c['export'] = {'file': exp.name, 'bytes': exp.stat().st_size} if exp.exists() else None
                out.append(c)
    return out


def delete_dataset(name):
    root = config.DATASET_DIR / name
    if '..' in name or not (root / 'dataset_card.json').exists():
        raise KeyError(name)
    shutil.rmtree(root)
    exp = config.EXPORT_DIR / ('%s.tar' % name)
    for f in (exp, exp.with_suffix('.tar.sha256')):
        if f.exists():
            f.unlink()


def export_dataset(name, progress=None):
    """データセットを1つのtarにまとめる(画像は圧縮済みなので無圧縮tar)。パスを返す"""
    root = config.DATASET_DIR / name
    if '..' in name or not (root / 'dataset_card.json').exists():
        raise KeyError(name)
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EXPORT_DIR / ('%s.tar' % name)
    tmp = out.with_suffix('.tar.part')
    files = [f for f in root.rglob('*') if f.is_file()]
    sha = hashlib.sha256()
    with tarfile.open(tmp, 'w') as tar:
        for k, f in enumerate(files):
            tar.add(f, arcname=str(f.relative_to(config.DATASET_DIR)), recursive=False)
            if progress and k % 200 == 0:
                progress(k / max(len(files), 1), f.name)
    with open(tmp, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            sha.update(block)
    os.replace(tmp, out)
    out.with_suffix('.tar.sha256').write_text('%s  %s\n' % (sha.hexdigest(), out.name))
    if progress:
        progress(1.0, out.name)
    return out
