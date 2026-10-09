"""収集: センサーのサンプルを20秒単位のシーンに分けて保存する(nuScenes風)

保存レイアウト:
  data/raw/<log>/log.json
  data/raw/<log>/<scene>/scene.json
                         CAM_FRONT/<timestamp_us>.jpg
                         CAM_FRONT.npz   timestamps_us, seq, files
                         TOF_FRONT.npz   timestamps_us, seq, depth[N,H,W] uint16
                         IMU.npz         timestamps_us, accel[N,3] m/s^2, gyro[N,3] deg/s, mag[N,3] µT, temp[N]
                         VEHICLE.npz     timestamps_us, pwm[N,3] µs, motor_pct[N], motor_us[N], valid[N,4], seq[N], board_t_ms[N]
                         thumb.jpg

サンプルは計測時刻で「どのシーンに属するか」を決めるので、センサー間でシーン境界がずれない。
"""

import datetime
import json
import os
import queue
import shutil
import threading
import time
import uuid

import cv2
import numpy as np

from . import clock, config, modes
from .sensors import SENSOR_CHANNELS


def new_token():
    return uuid.uuid4().hex


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def depth_colormap(depth, width=None, height=None, vmin=None, vmax=None):
    d = depth.astype(np.float32)
    lo = float(np.percentile(d, 2)) if vmin is None else vmin
    hi = float(np.percentile(d, 98)) if vmax is None else vmax
    norm = np.clip((d - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)   # 近い=赤, 遠い=青
    if width and height:
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_NEAREST)
    return color


def mode_segments(events, t_start, t_end):
    """ジャンル切り替え履歴 [(t_ns, mode_id)] から、[t_start, t_end] の区間リストを作る"""
    segs = []
    current = events[0][1]
    for t, m in events:
        if t <= t_start:
            current = m
    cursor = t_start
    for t, m in events:
        if t_start < t < t_end and m != current:
            segs.append({'start_s': cursor, 'end_s': t, 'mode_id': current})
            cursor, current = t, m
    segs.append({'start_s': cursor, 'end_s': t_end, 'mode_id': current})
    for sg in segs:
        sg['mode_name'] = modes.name_of(sg['mode_id'])
    return segs


class _SceneBuffer(object):
    def __init__(self, index, name, path, start_ns, end_ns):
        self.index = index
        self.name = name
        self.path = path
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.token = new_token()
        self.cam = []        # (t_ns, seq, filename)
        self.tof = []        # (t_ns, seq, depth)
        self.imu = []        # (t_ns, accel, gyro, mag, temp)
        self.veh = []        # (t_ns, data)
        self.thumb = None
        self.last_depth = None
        (path / 'CAM_FRONT').mkdir(parents=True, exist_ok=True)


class Recorder(object):
    def __init__(self, hub, on_scene_saved=None):
        self.hub = hub
        self.on_scene_saved = on_scene_saved
        self.lock = threading.Lock()
        self.recording = False
        self.log = None
        self.scenes = {}           # index -> _SceneBuffer
        self.saved = []            # 保存済みシーン名
        self.max_scenes = None
        self._io = queue.Queue(maxsize=600)
        self._io_thread = threading.Thread(target=self._io_worker, name='rec-io', daemon=True)
        self._io_thread.start()
        self._finalizer = None
        self.last_error = ''
        self.dropped_writes = 0

    # ------------------------------------------------------------------ 制御
    def start(self, description='', tags=None, location='', max_scenes=None, mode_id=modes.UNSET):
        mode_id = modes.validate(mode_id)
        if self.recording:
            raise RuntimeError('すでに収集中です')
        self._warm_up()
        with self.lock:
            if self.recording:
                raise RuntimeError('すでに収集中です')
            now = clock.now_ns()
            stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = config.RAW_DIR / stamp
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log = {
                'token': new_token(), 'name': stamp, 'path': log_dir, 'start_ns': now,
                'description': description, 'tags': list(tags or []), 'location': location,
                'mode_events': [(now, mode_id)],     # (時刻, ジャンルID) 収集中の切り替えを記録
            }
            write_json(log_dir / 'log.json', {
                'token': self.log['token'], 'logfile': stamp, 'vehicle': config.VEHICLE_NAME,
                'date_captured': datetime.date.today().isoformat(), 'location': location,
                'description': description, 'tags': list(tags or []),
                'mode_id': mode_id, 'mode_name': modes.name_of(mode_id),
                'scene_seconds': config.SCENE_SECONDS,
                'sensors': {'CAM_FRONT': config.CAMERA, 'TOF_FRONT': config.TOF, 'IMU': config.IMU,
                            'VEHICLE': {k: v for k, v in config.VEHICLE.items()}},
                'calibration': config.CALIBRATION,
                'software': 'jetformer',
            })
            self.scenes = {}
            self.saved = []
            self.max_scenes = max_scenes
            self.recording = True
            for name in SENSOR_CHANNELS:
                self.hub[name].subscribe(self._on_sample)
            self._finalizer = threading.Thread(target=self._finalize_loop, name='rec-final', daemon=True)
            self._finalizer.start()
            return self.log['name']

    def stop(self):
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            for name in SENSOR_CHANNELS:
                self.hub[name].unsubscribe(self._on_sample)
        self._io.join()
        with self.lock:
            pending = sorted(self.scenes)
        for idx in pending:
            self._finalize(idx, partial=True)
        self.log = None

    WARMUP_SAMPLES = {'CAM_FRONT': 45, 'TOF_FRONT': 15, 'IMU': 15, 'VEHICLE': 15}   # カメラは自動露出の収束も待つ(1.5秒)

    def _warm_up(self, timeout=8.0):
        """センサーが止まっていたら起動し、全センサーが安定して値を出すまで待つ。
        立ち上がり直後のデータ(IMUのリセット直後の値、カメラのストリーム開始時の
        間隔の乱れ・自動露出の収束前の画像)がシーンの先頭に混ざらないようにする。"""
        self.hub.start()
        start = {name: self.hub[name].count for name in SENSOR_CHANNELS}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(self.hub[n].count - start[n] >= self.WARMUP_SAMPLES[n] for n in SENSOR_CHANNELS
                   if self.hub[n].connected or n != 'VEHICLE'):
                return
            time.sleep(0.1)
        missing = [n for n in SENSOR_CHANNELS if self.hub[n].count - start[n] < self.WARMUP_SAMPLES[n]]
        self.last_error = 'センサーの立ち上がりを待ちきれませんでした: %s' % ', '.join(missing)

    def set_mode(self, mode_id):
        """収集中にジャンルを切り替える(この瞬間から新しいジャンルとして記録)"""
        mode_id = modes.validate(mode_id)
        with self.lock:
            if not self.recording:
                raise RuntimeError('収集中ではありません')
            if self.log['mode_events'][-1][1] != mode_id:
                self.log['mode_events'].append((clock.now_ns(), mode_id))
        return mode_id

    def status(self):
        with self.lock:
            if not self.recording or self.log is None:
                return {'recording': False, 'saved_scenes': list(self.saved), 'last_error': self.last_error}
            elapsed = (clock.now_ns() - self.log['start_ns']) / 1e9
            cur = int(elapsed // config.SCENE_SECONDS)
            buf = self.scenes.get(cur)
            return {
                'recording': True, 'log': self.log['name'], 'elapsed_s': round(elapsed, 1),
                'scene_index': cur + 1, 'scene_elapsed_s': round(elapsed - cur * config.SCENE_SECONDS, 1),
                'scene_seconds': config.SCENE_SECONDS, 'max_scenes': self.max_scenes,
                'current_counts': {'CAM_FRONT': len(buf.cam), 'TOF_FRONT': len(buf.tof),
                                   'IMU': len(buf.imu), 'VEHICLE': len(buf.veh)} if buf else {},
                'mode_id': self.log['mode_events'][-1][1], 'mode_name': modes.name_of(self.log['mode_events'][-1][1]),
                'saved_scenes': list(self.saved), 'io_queue': self._io.qsize(),
                'dropped_writes': self.dropped_writes, 'last_error': self.last_error,
            }

    # ------------------------------------------------------------------ 受信
    def _scene_for(self, t_ns):
        start = self.log['start_ns']
        if t_ns < start:
            return None
        period = int(config.SCENE_SECONDS * 1e9)
        idx = (t_ns - start) // period
        if self.max_scenes is not None and idx >= self.max_scenes:
            return None
        buf = self.scenes.get(idx)
        if buf is None:
            name = '%s-%04d' % (self.log['name'], idx + 1)
            buf = _SceneBuffer(idx, name, self.log['path'] / name, start + idx * period, start + (idx + 1) * period)
            self.scenes[idx] = buf
        return buf

    def _on_sample(self, channel, sample):
        with self.lock:
            if not self.recording:
                return
            buf = self._scene_for(sample.t_ns)
            if buf is None:
                return
            d = sample.data
            if channel == 'CAM_FRONT':
                ts_us = clock.to_epoch_us(sample.t_ns)
                fname = 'CAM_FRONT/%d.jpg' % ts_us
                buf.cam.append((sample.t_ns, d['seq'], fname))
                if buf.thumb is None and len(buf.cam) >= 30:
                    buf.thumb = d['image']
                try:
                    self._io.put_nowait((buf.path / fname, d['image']))
                except queue.Full:
                    self.dropped_writes += 1
            elif channel == 'TOF_FRONT':
                buf.tof.append((sample.t_ns, d['seq'], d['depth']))
                buf.last_depth = d['depth']
            elif channel == 'IMU':
                buf.imu.append((sample.t_ns, d['accel'], d['gyro'], d['mag'], d['temp']))
            elif channel == 'VEHICLE':
                buf.veh.append((sample.t_ns, d))

    def _io_worker(self):
        params = [cv2.IMWRITE_JPEG_QUALITY, config.CAMERA['jpeg_quality']]
        while True:
            path, image = self._io.get()
            try:
                cv2.imwrite(str(path), image, params)
            except Exception as e:
                self.last_error = 'JPEG保存失敗: %s' % e
            finally:
                self._io.task_done()

    # ------------------------------------------------------------------ 保存
    def _finalize_loop(self):
        # シーン終了から1秒経ったら保存する(遅れて届くToF/車両のサンプルを待つ)
        while True:
            time.sleep(0.5)
            with self.lock:
                if not self.recording:
                    return
                now = clock.now_ns()
                done = [i for i, b in self.scenes.items() if now > b.end_ns + 1_000_000_000]
                reached_max = self.max_scenes is not None and \
                    now > self.log['start_ns'] + int(self.max_scenes * config.SCENE_SECONDS * 1e9) + 1_000_000_000
            for idx in done:
                self._io.join()
                self._finalize(idx, partial=False)
            if reached_max:
                threading.Thread(target=self.stop, daemon=True).start()
                return

    def _finalize(self, idx, partial):
        with self.lock:
            buf = self.scenes.pop(idx, None)
            log = dict(self.log, mode_events=list(self.log['mode_events'])) if self.log else None
        if buf is None or log is None:
            return
        if not buf.cam:
            shutil.rmtree(buf.path, ignore_errors=True)
            return
        try:
            self._save_scene(buf, log, partial)
            self.saved.append(buf.name)
            if self.on_scene_saved:
                self.on_scene_saved(buf.name)
        except Exception as e:
            self.last_error = 'シーン保存失敗 %s: %s' % (buf.name, e)

    def _save_scene(self, buf, log, partial):
        ts = lambda arr: np.array([clock.to_epoch_us(t) for t in arr], dtype=np.int64)
        np.savez(buf.path / 'CAM_FRONT.npz',
                 timestamps_us=ts([c[0] for c in buf.cam]),
                 seq=np.array([c[1] for c in buf.cam], dtype=np.int64),
                 files=np.array([c[2] for c in buf.cam]))
        if buf.tof:
            np.savez_compressed(buf.path / 'TOF_FRONT.npz',
                                timestamps_us=ts([t[0] for t in buf.tof]),
                                seq=np.array([t[1] for t in buf.tof], dtype=np.int64),
                                depth=np.stack([t[2] for t in buf.tof]).astype(np.uint16))
        if buf.imu:
            np.savez(buf.path / 'IMU.npz',
                     timestamps_us=ts([s[0] for s in buf.imu]),
                     accel=np.stack([s[1] for s in buf.imu]).astype(np.float32),
                     gyro=np.stack([s[2] for s in buf.imu]).astype(np.float32),
                     mag=np.stack([s[3] for s in buf.imu]).astype(np.float32),
                     temp=np.array([s[4] for s in buf.imu], dtype=np.float32))
        if buf.veh:
            v = [s[1] for s in buf.veh]
            np.savez(buf.path / 'VEHICLE.npz',
                     timestamps_us=ts([s[0] for s in buf.veh]),
                     pwm=np.array([x['pwm'] for x in v], dtype=np.int32),
                     motor_pct=np.array([x['motor_pct'] for x in v], dtype=np.int32),
                     motor_us=np.array([x['motor_us'] for x in v], dtype=np.int32),
                     valid=np.array([x['valid'] for x in v], dtype=bool),
                     seq=np.array([x['seq'] for x in v], dtype=np.int64),
                     board_t_ms=np.array([x['board_t_ms'] for x in v], dtype=np.int64),
                     quality=np.array(v[0]['quality']))
        if buf.thumb is not None:
            thumb = buf.thumb
            if buf.last_depth is not None:
                thumb = np.hstack([thumb, depth_colormap(buf.last_depth, thumb.shape[1], thumb.shape[0])])
            cv2.imwrite(str(buf.path / 'thumb.jpg'), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        start_us = clock.to_epoch_us(buf.start_ns)
        cam_ts = [c[0] for c in buf.cam]
        duration = (cam_ts[-1] - cam_ts[0]) / 1e9 if len(cam_ts) > 1 else 0.0
        counts = {'CAM_FRONT': len(buf.cam), 'TOF_FRONT': len(buf.tof), 'IMU': len(buf.imu), 'VEHICLE': len(buf.veh)}
        rates = {k: round(n / duration, 1) if duration > 0 else 0.0 for k, n in counts.items()}
        stats = {}
        if buf.veh:
            pwm = np.array([s[1]['pwm'] for s in buf.veh], dtype=np.float32)
            mp = np.array([s[1]['motor_pct'] for s in buf.veh], dtype=np.float32)
            for k, name in enumerate(('steering', 'throttle', 'aux')):
                col = pwm[:, k][pwm[:, k] > 0]
                if col.size:
                    stats[name + '_us'] = {'min': int(col.min()), 'max': int(col.max()), 'mean': round(float(col.mean()), 1)}
            stats['motor_pct'] = {'min': int(mp.min()), 'max': int(mp.max()), 'mean': round(float(mp.mean()), 1)}
        if buf.imu:
            acc = np.stack([s[1] for s in buf.imu])
            stats['accel_peak_ms2'] = round(float(np.abs(np.linalg.norm(acc, axis=1) - 9.80665).max()), 2)
        segments = mode_segments(log['mode_events'], cam_ts[0], cam_ts[-1])
        main_mode = max(segments, key=lambda sg: sg['end_s'] - sg['start_s'])['mode_id']
        for sg in segments:
            sg['start_s'] = round((sg['start_s'] - cam_ts[0]) / 1e9, 3)
            sg['end_s'] = round((sg['end_s'] - cam_ts[0]) / 1e9, 3)
        gaps = np.diff(np.array(cam_ts)) / 1e6 if len(cam_ts) > 1 else np.array([0.0])
        stats['camera_max_gap_ms'] = round(float(gaps.max()), 1)
        write_json(buf.path / 'scene.json', {
            'token': buf.token, 'name': buf.name, 'log_token': log['token'], 'log': log['name'],
            'description': log['description'], 'tags': log['tags'],
            'mode_id': main_mode, 'mode_name': modes.name_of(main_mode), 'mode_segments': segments,
            'start_us': start_us, 'end_us': clock.to_epoch_us(cam_ts[-1]),
            'duration_s': round(duration, 2), 'complete': not partial and duration >= config.SCENE_SECONDS * 0.95,
            'counts': counts, 'rates': rates, 'stats': stats,
            'vehicle_quality': buf.veh[0][1]['quality'] if buf.veh else None,
            'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        })
