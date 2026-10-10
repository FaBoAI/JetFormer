"""収集: センサーのサンプルを20秒単位のシーンに分けて保存する(nuScenes風)

保存レイアウト:
  data/raw/<log>/log.json
  data/raw/<log>/<scene>/scene.json
                         CAM_FRONT/<timestamp_us>.jpg
                         CAM_FRONT.npz   timestamps_us, seq, files
                         TOF_FRONT.npz   timestamps_us, seq, depth[N,H,W] uint16
                         IMU.npz         timestamps_us, accel[N,3] m/s^2, gyro[N,3] deg/s, mag[N,3] µT, temp[N]
                         VEHICLE.npz     timestamps_us, pwm[N,3] µs, valid[N,3], select[N], erpm[N], ...
                         AUDIO.npz       timestamps_us[M](20msごとの受信時刻), pcm[M*320] int32 16kHz モノラル
                         AI_CMD.npz      (AI走行中のみ)timestamps_us(=推論に使ったカメラ時刻), pred[N,10,2] µs,
                                         raw[N,2] / out[N,2] µs(ゲイン適用前/後), gains[N,2], hold[N], ...
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
import traceback
import uuid

import cv2
import numpy as np

from . import clock, config, modes, tof_format
from .sensors import OPTIONAL_CHANNELS


def new_token():
    return uuid.uuid4().hex


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def depth_colormap(raw, width=None, height=None, vmin=None, vmax=None):
    """ToFの生の値 → カラー画像(近い=赤, 遠い=青, 無効=黒)"""
    d = tof_format.distance_mm(raw)
    ok = d > 0
    vals = d[ok] if ok.any() else np.array([0.0, 1.0])
    lo = float(np.percentile(vals, 2)) if vmin is None else vmin
    hi = float(np.percentile(vals, 98)) if vmax is None else vmax
    norm = np.clip((d - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
    color[~ok] = 0
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
        self.ai = []         # (t_ns, data) AI走行の指令
        self.audio = []      # (t_ns, pcm int32[320]) マイク
        self.thumb = None
        self.last_depth = None
        self.run = None          # 走行ごとに区切る収集: 何回目の走行か
        self.planned_s = None
        self.run_complete = False
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
        self._closed = set()       # 保存済みシーン番号(遅れて届いた値で作り直さない)
        self.late_dropped = 0
        self.runs_mode = False     # 走行ごとに区切る収集か
        self.run_count = 0
        self.active_run = None
        self.listeners = []        # 出来事の通知先 cb(event, **info)(音声案内など)

    def _emit(self, event, **info):
        for cb in list(self.listeners):
            try:
                cb(event, **info)
            except Exception:
                traceback.print_exc()

    @property
    def channels(self):
        return getattr(self.hub, 'channels', ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE'))

    # ------------------------------------------------------------------ 制御
    def start(self, description='', tags=None, location='', max_scenes=None, mode_id=modes.UNSET, drive=None,
              runs_mode=False):
        """runs_mode=True: 走行ごとに区切る(begin_run〜end_run の間だけ記録し、1回の走行を1シーンにする)。
        False: 開始から20秒ごとに区切って記録し続ける"""
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
                'drive': drive,                      # AI走行中の収集なら、そのモデル・設定
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
                'drive': drive,
                'software': 'jetformer',
            })
            self.scenes = {}
            self.saved = []
            self._closed = set()
            self.runs_mode = bool(runs_mode)
            self.run_count = 0
            self.active_run = None
            self.max_scenes = None if runs_mode else max_scenes
            self.recording = True
            for name in self.channels:
                self.hub[name].subscribe(self._on_sample)
            self._finalizer = threading.Thread(target=self._finalize_loop, name='rec-final', daemon=True)
            self._finalizer.start()
            name = self.log['name']
        self._emit('record_start', log=name, mode_id=mode_id, mode_name=modes.name_of(mode_id), runs=self.runs_mode)
        return name

    # ------------------------------------------------------------ 走行ごとの区切り
    def begin_run(self, planned_s):
        """走行を始める(この瞬間から記録)。戻り値 (何回目か, シーン名)"""
        with self.lock:
            if not self.recording or not self.runs_mode:
                raise RuntimeError('走行ごとの収集をしていません')
            if self.active_run is not None:
                raise RuntimeError('前の走行が終わっていません')
            idx = self.run_count
            self.run_count += 1
            now = clock.now_ns()
            name = '%s-%04d' % (self.log['name'], idx + 1)
            buf = _SceneBuffer(idx, name, self.log['path'] / name, now, now + int(planned_s * 1e9))
            buf.run, buf.planned_s = idx + 1, float(planned_s)
            self.scenes[idx] = buf
            self.active_run = idx
            return idx + 1, name

    def end_run(self, complete=True):
        """走行を終える(この瞬間までを記録)。complete=False は途中で打ち切った走行。戻り値: 何回目か(走行中でなければ None)"""
        with self.lock:
            idx, self.active_run = self.active_run, None
            buf = self.scenes.get(idx) if idx is not None else None
            if buf is None:
                return None
            buf.end_ns = min(buf.end_ns, clock.now_ns())
            buf.run_complete = complete
            return idx + 1

    def stop(self, only_log=None):
        """収集を止めて残りのシーンを保存する。only_log を渡すと、そのログの収集のときだけ止める。止めたら True"""
        if self.recording and getattr(self, 'active_run', None) is not None and \
                (only_log is None or (self.log and self.log['name'] == only_log)):
            self.end_run(complete=False)             # 走行の途中で止めた
        with self.lock:
            if not self.recording or (only_log is not None and self.log['name'] != only_log):
                return False
            self.recording = False
            for name in self.channels:
                self.hub[name].unsubscribe(self._on_sample)
        self._io.join()
        with self.lock:
            pending = sorted(self.scenes)
        for idx in pending:
            self._finalize(idx, partial=True)
        self.log = None
        self._emit('record_stop', scenes=len(self.saved), runs=self.runs_mode)
        return True

    WARMUP_SAMPLES = {'CAM_FRONT': 45, 'TOF_FRONT': 15, 'IMU': 15, 'VEHICLE': 15, 'AUDIO': 10}   # カメラは自動露出の収束も待つ(1.5秒)

    def _warm_up(self, timeout=8.0):
        """センサーが止まっていたら起動し、全センサーが安定して値を出すまで待つ。
        立ち上がり直後のデータ(IMUのリセット直後の値、カメラのストリーム開始時の
        間隔の乱れ・自動露出の収束前の画像)がシーンの先頭に混ざらないようにする。"""
        self.hub.start()
        start = {name: self.hub[name].count for name in self.channels}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(self.hub[n].count - start[n] >= self.WARMUP_SAMPLES[n] for n in self.channels
                   if self.hub[n].connected or n not in OPTIONAL_CHANNELS):
                return
            time.sleep(0.1)
        missing = [n for n in self.channels if self.hub[n].count - start[n] < self.WARMUP_SAMPLES[n]
                   and (self.hub[n].connected or n not in OPTIONAL_CHANNELS)]
        self.last_error = 'センサーの立ち上がりを待ちきれませんでした: %s' % ', '.join(missing)

    def set_mode(self, mode_id):
        """収集中にジャンルを切り替える(この瞬間から新しいジャンルとして記録)"""
        mode_id = modes.validate(mode_id)
        with self.lock:
            if not self.recording:
                raise RuntimeError('収集中ではありません')
            changed = self.log['mode_events'][-1][1] != mode_id
            if changed:
                self.log['mode_events'].append((clock.now_ns(), mode_id))
        if changed:
            self._emit('mode', mode_id=mode_id, mode_name=modes.name_of(mode_id))
        return mode_id

    def status(self):
        with self.lock:
            if not self.recording or self.log is None:
                return {'recording': False, 'saved_scenes': list(self.saved), 'last_error': self.last_error}
            elapsed = (clock.now_ns() - self.log['start_ns']) / 1e9
            cur = int(elapsed // config.SCENE_SECONDS)
            buf = self.scenes.get(cur)
            if self.runs_mode:
                buf = self.scenes.get(self.active_run) if self.active_run is not None else None
                return {
                    'recording': True, 'runs_mode': True, 'log': self.log['name'], 'elapsed_s': round(elapsed, 1),
                    'run_index': buf.run if buf else None, 'runs_started': self.run_count,
                    'run_elapsed_s': round((clock.now_ns() - buf.start_ns) / 1e9, 1) if buf else None,
                    'run_planned_s': buf.planned_s if buf else None,
                    'current_counts': {'CAM_FRONT': len(buf.cam), 'TOF_FRONT': len(buf.tof), 'IMU': len(buf.imu),
                                       'VEHICLE': len(buf.veh)} if buf else {},
                    'mode_id': self.log['mode_events'][-1][1], 'mode_name': modes.name_of(self.log['mode_events'][-1][1]),
                    'saved_scenes': list(self.saved), 'io_queue': self._io.qsize(),
                    'dropped_writes': self.dropped_writes, 'last_error': self.last_error,
                }
            return {
                'recording': True, 'log': self.log['name'], 'elapsed_s': round(elapsed, 1),
                'scene_index': cur + 1, 'scene_elapsed_s': round(elapsed - cur * config.SCENE_SECONDS, 1),
                'scene_seconds': config.SCENE_SECONDS, 'max_scenes': self.max_scenes,
                'current_counts': {'CAM_FRONT': len(buf.cam), 'TOF_FRONT': len(buf.tof),
                                   'IMU': len(buf.imu), 'VEHICLE': len(buf.veh), 'AI_CMD': len(buf.ai),
                                   'AUDIO': len(buf.audio)} if buf else {},
                'mode_id': self.log['mode_events'][-1][1], 'mode_name': modes.name_of(self.log['mode_events'][-1][1]),
                'saved_scenes': list(self.saved), 'io_queue': self._io.qsize(),
                'dropped_writes': self.dropped_writes, 'last_error': self.last_error,
            }

    # ------------------------------------------------------------------ 受信
    def add_ai(self, t_ns, data):
        """AI走行の指令を記録する(収集中でなければ何もしない)。t_ns は推論に使ったカメラ画像の時刻"""
        with self.lock:
            if not self.recording:
                return
            buf = self._scene_for(t_ns)
            if buf is not None:
                buf.ai.append((t_ns, data))

    def _scene_for(self, t_ns):
        if self.runs_mode:                 # 走行ごと: 走行中(と終わったばかりの走行)の時間内だけ記録する
            for buf in self.scenes.values():
                if buf.start_ns <= t_ns < buf.end_ns:
                    return buf
            return None
        start = self.log['start_ns']
        if t_ns < start:
            return None
        period = int(config.SCENE_SECONDS * 1e9)
        idx = (t_ns - start) // period
        if self.max_scenes is not None and idx >= self.max_scenes:
            return None
        if idx in self._closed:            # 保存済みのシーンに遅れて届いた値は捨てる
            self.late_dropped += 1
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
            elif channel == 'AUDIO':
                buf.audio.append((sample.t_ns, d['pcm']))

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
                done = [i for i, b in self.scenes.items()
                        if now > b.end_ns + 1_000_000_000 and i != self.active_run]   # 走行中のシーンは終わるまで保存しない
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
            self._closed.add(idx)
            log = dict(self.log, mode_events=list(self.log['mode_events'])) if self.log else None
        if buf is None or log is None:
            return
        if not buf.cam:
            if not (buf.path / 'scene.json').exists():     # 保存済みのシーンは消さない
                shutil.rmtree(buf.path, ignore_errors=True)
            return
        try:
            self._save_scene(buf, log, partial)
            self.saved.append(buf.name)
            self._emit('scene_saved', name=buf.name)
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
                     pwm=np.array([x['pwm'] for x in v], dtype=np.int32),          # [steering, throttle, mode] µs
                     valid=np.array([x['valid'] for x in v], dtype=bool),          # CH1-3 受信中
                     select=np.array([x['select'] for x in v], dtype=bool),        # True = AI走行側
                     moving=np.array([x['moving'] for x in v], dtype=bool),
                     erpm=np.array([x['erpm'] for x in v], dtype=np.int32),        # 電気角rpm(符号=向き)
                     hall_count=np.array([x['hall_count'] for x in v], dtype=np.int64),
                     hall_errors=np.array([x['hall_errors'] for x in v], dtype=np.int32),
                     hall_state=np.array([x['hall_state'] for x in v], dtype=np.uint8),
                     seq=np.array([x['seq'] for x in v], dtype=np.int64),
                     board_t_us=np.array([x['board_t_us'] for x in v], dtype=np.int64),
                     quality=np.array(v[0]['quality']))
        if buf.audio:
            np.savez(buf.path / 'AUDIO.npz',
                     timestamps_us=ts([s[0] for s in buf.audio]),             # 各20msチャンクの末尾の受信時刻
                     chunk_samples=np.array([len(s[1]) for s in buf.audio], dtype=np.int32),
                     pcm=np.concatenate([s[1] for s in buf.audio]).astype(np.int32),   # 16kHz モノラル(24bit精度)
                     sample_rate=np.array(config.AUDIO['rate']))
        if buf.ai:
            a = [s[1] for s in buf.ai]
            np.savez(buf.path / 'AI_CMD.npz',
                     timestamps_us=ts([s[0] for s in buf.ai]),
                     pred=np.stack([x['pred'] for x in a]).astype(np.float32),        # [N,10,2] 未来の [ステア, スロットル] µs
                     raw=np.array([x['raw'] for x in a], dtype=np.float32),           # 採用した予測(ゲイン適用前)
                     out=np.array([x['out'] for x in a], dtype=np.float32),           # PCA9685へ出した値
                     gains=np.array([x['gains'] for x in a], dtype=np.float32),       # [ステア, 速度]
                     trim=np.array([x['trim'] for x in a], dtype=np.float32),
                     lookahead=np.array([x['lookahead'] for x in a], dtype=np.int16),
                     mode_id=np.array([x['mode_id'] for x in a], dtype=np.int16),
                     hold=np.array([x['hold'] for x in a], dtype=bool),               # センサー途切れ等で中立を出した
                     select=np.array([x['select'] for x in a], dtype=bool),           # True = 車にAIの出力が届いていた
                     latency_ms=np.array([x['latency_ms'] for x in a], dtype=np.float32),   # 撮像 → 出力
                     infer_ms=np.array([x['infer_ms'] for x in a], dtype=np.float32),
                     dry_run=np.array([x['dry_run'] for x in a], dtype=bool))
        if buf.thumb is not None:
            thumb = buf.thumb
            if buf.last_depth is not None:
                thumb = np.hstack([thumb, depth_colormap(buf.last_depth, thumb.shape[1], thumb.shape[0])])
            cv2.imwrite(str(buf.path / 'thumb.jpg'), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        start_us = clock.to_epoch_us(buf.start_ns)
        cam_ts = [c[0] for c in buf.cam]
        duration = (cam_ts[-1] - cam_ts[0]) / 1e9 if len(cam_ts) > 1 else 0.0
        counts = {'CAM_FRONT': len(buf.cam), 'TOF_FRONT': len(buf.tof), 'IMU': len(buf.imu), 'VEHICLE': len(buf.veh)}
        if buf.ai:
            counts['AI_CMD'] = len(buf.ai)
        if buf.audio:
            counts['AUDIO'] = len(buf.audio)
        rates = {k: round(n / duration, 1) if duration > 0 else 0.0 for k, n in counts.items()}
        stats = {}
        if buf.veh:
            vd = [s[1] for s in buf.veh]
            pwm = np.array([x['pwm'] for x in vd], dtype=np.float32)
            valid = np.array([x['valid'] for x in vd], dtype=bool)
            erpm = np.array([x['erpm'] for x in vd], dtype=np.float32)
            for k, name in enumerate(('steering', 'throttle', 'mode')):
                col = pwm[:, k][valid[:, k] & (pwm[:, k] > 0)]
                if col.size:
                    stats[name + '_us'] = {'min': int(col.min()), 'max': int(col.max()), 'mean': round(float(col.mean()), 1)}
            stats['erpm'] = {'min': int(erpm.min()), 'max': int(erpm.max()), 'mean': round(float(erpm.mean()), 1)}
            stats['ai_driving_ratio'] = round(float(np.mean([x['select'] for x in vd])), 3)
            stats['rc_lost_ratio'] = round(float(1.0 - np.mean(valid[:, 0] & valid[:, 1])), 3)
        if buf.ai:
            a = [s[1] for s in buf.ai]
            lat = np.array([x['latency_ms'] for x in a if not x['hold']] or [0.0])
            inf = np.array([x['infer_ms'] for x in a if not x['hold']] or [0.0])
            stats['ai'] = {'latency_ms_mean': round(float(lat.mean()), 1), 'latency_ms_max': round(float(lat.max()), 1),
                           'infer_ms_mean': round(float(inf.mean()), 2),
                           'hold_ratio': round(float(np.mean([x['hold'] for x in a])), 3),
                           'applied_ratio': round(float(np.mean([x['select'] and not x['dry_run'] for x in a])), 3)}
        if buf.audio:
            pcm = np.concatenate([s[1] for s in buf.audio]).astype(np.float64) / 2.0 ** 31
            stats['audio'] = {'rms_dbfs': round(float(20 * np.log10(max(np.sqrt(np.mean(pcm * pcm)), 1e-12))), 1),
                              'peak_dbfs': round(float(20 * np.log10(max(np.abs(pcm).max(), 1e-12))), 1),
                              'no_signal': bool(not pcm.any())}
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
            'duration_s': round(duration, 2),
            'complete': buf.run_complete if buf.run else (not partial and duration >= config.SCENE_SECONDS * 0.95),
            'run_index': buf.run, 'planned_s': buf.planned_s,
            'counts': counts, 'rates': rates, 'stats': stats,
            'vehicle_quality': buf.veh[0][1]['quality'] if buf.veh else None,
            'drive': log.get('drive'),
            'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        })
