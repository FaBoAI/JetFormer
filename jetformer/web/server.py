"""JetFormer 収集管理Webサーバー

    python -m jetformer.web.server          # http://<jetson>:8080/
"""

import atexit
import collections
import contextlib
import queue
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import actuator, config, dataset, drive, hf_export, model_export, modes, policies, session, sound, tof_format
from ..recorder import Recorder, depth_colormap
from ..sensors import SensorHub
from ..training import TrainingManager
from ..training import start_args as training_start_args

STATIC = config.ROOT / 'jetformer' / 'web' / 'static'

hub = SensorHub()
recorder = Recorder(hub)
trainer = TrainingManager()
driver = drive.Driver(hub, recorder)
SHUTDOWN = threading.Event()        # 終了の合図(映像配信などの終わらない応答を閉じる)
player = sound.SoundPlayer()        # 車のスピーカー(収集・走行・学習の開始と終了などを音声で知らせる)


def _announce(event, **info):
    sound.announce(player, event, **info)


for _src in (recorder, driver, trainer):
    _src.listeners.append(_announce)
collect = session.CollectSession(recorder, player)     # 走行ごとに区切る収集(カウントダウンと終了の合図つき)


def _release_outputs():
    """走行していなければ、PCA9685のステアリング・スロットルをパルスなしにする(前のプロセスが残した出力を止める)"""
    if driver.running:
        return
    try:
        actuator.outputs_off(config.PCA9685)
    except Exception:
        pass


@atexit.register
def _on_exit():
    # 通常の終了処理を通らなかった場合の後始末: 走行を止め、止まりきらなければ出力を切る
    if driver.running:
        driver.shutdown(1.5)
        if driver.running:
            try:
                actuator.outputs_off(config.PCA9685)
            except Exception:
                pass


@contextlib.asynccontextmanager
async def lifespan(_app):
    _release_outputs()
    names = [m['name'] for m in modes.load() if m['id'] != modes.UNSET]
    player.prepare(sound.COMMON_TEXTS + ['データ収集を開始します。%s' % n for n in names] + ['ジャンル、%s' % n for n in names])
    _announce('startup')
    yield
    # 終了時: 走行中なら中立を出して止め、収集中なら最後のシーンを保存してからセンサーを止め、学習中なら停止を指示する
    driver.shutdown()
    if collect.active:
        collect.stop()
        collect.thread.join(10)
    if recorder.recording:
        recorder.stop()
    hub.stop()
    if trainer.running:
        try:
            trainer.stop()
        except RuntimeError:
            pass
    print('JetFormer: 終了処理が完了しました(走行停止・収集の保存・センサー停止)', flush=True)


app = FastAPI(title='JetFormer', lifespan=lifespan)


# ---------------------------------------------------------------- ライブ値の履歴(グラフ用)
class LiveHistory(object):
    def __init__(self, seconds=10.0, rate_hz=25.0):
        self.min_dt = 1.0 / rate_hz
        self.imu = collections.deque(maxlen=int(seconds * rate_hz))
        self.veh = collections.deque(maxlen=int(seconds * rate_hz))
        self._last = {'IMU': 0.0, 'VEHICLE': 0.0}
        hub['IMU'].subscribe(self.on_sample)
        hub['VEHICLE'].subscribe(self.on_sample)

    def on_sample(self, channel, sample):
        now = time.monotonic()
        if now - self._last[channel] < self.min_dt:
            return
        self._last[channel] = now
        if channel == 'IMU':
            a, g = sample.data['accel'], sample.data['gyro']
            self.imu.append((now, float(np.linalg.norm(a)), float(a[0]), float(a[1]), float(g[2])))
        else:
            d = sample.data
            ok = d['valid'][0] and d['valid'][1]
            self.veh.append((now, d['pwm'][0] if ok else 0, d['pwm'][1] if ok else 0, d['pwm'][2], d['erpm']))

    def snapshot(self):
        now = time.monotonic()
        imu, veh = list(self.imu), list(self.veh)
        return {
            'imu': {'t': [round(r[0] - now, 2) for r in imu], 'accel_norm': [round(r[1], 2) for r in imu],
                    'accel_x': [round(r[2], 2) for r in imu], 'accel_y': [round(r[3], 2) for r in imu],
                    'gyro_z': [round(r[4], 1) for r in imu]},
            'vehicle': {'t': [round(r[0] - now, 2) for r in veh], 'steering': [r[1] for r in veh],
                        'throttle': [r[2] for r in veh], 'mode': [r[3] for r in veh], 'erpm': [r[4] for r in veh]},
        }


history = LiveHistory()


# ---------------------------------------------------------------- バックグラウンドジョブ
class Jobs(object):
    def __init__(self):
        self.jobs = collections.OrderedDict()
        self.lock = threading.Lock()

    def submit(self, kind, title, fn):
        job_id = uuid.uuid4().hex[:8]
        job = {'id': job_id, 'kind': kind, 'title': title, 'state': 'running', 'progress': 0.0,
               'message': '', 'result': None, 'error': None, 'started': time.time(), 'finished': None}
        with self.lock:
            self.jobs[job_id] = job
            while len(self.jobs) > 30:
                self.jobs.popitem(last=False)

        def progress(p, msg=''):
            job['progress'] = round(float(p), 3)
            job['message'] = str(msg)

        def run():
            try:
                job['result'] = fn(progress)
                job['state'] = 'done'
                job['progress'] = 1.0
            except Exception as e:
                job['state'] = 'error'
                job['error'] = '%s: %s' % (type(e).__name__, e)
            job['finished'] = time.time()

        threading.Thread(target=run, name='job-' + job_id, daemon=True).start()
        return job

    def list(self):
        with self.lock:
            return list(reversed(self.jobs.values()))


jobs = Jobs()


# ---------------------------------------------------------------- 画像ユーティリティ
def _jpeg(image, quality=80):
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _placeholder(text, w=672, h=336):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return img


def _compose(image, depth, scale=2):
    """カメラ | ToF(距離カラーマップ) を横に並べる"""
    if image is None:
        image = np.zeros((224, 224, 3), np.uint8)
    h, w = image.shape[0] * scale, image.shape[1] * scale
    cam = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    if depth is None:
        tof = np.full((h, w, 3), 30, np.uint8)
        cv2.putText(tof, 'ToF: no data', (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    else:
        tof = depth_colormap(depth, w, h)
        mm = tof_format.distance_mm(depth)
        if (mm > 0).any():
            lo, hi = int(np.percentile(mm[mm > 0], 2)), int(np.percentile(mm[mm > 0], 98))
            cv2.putText(tof, '%d-%d mm' % (lo, hi), (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.hstack([cam, np.full((h, 4, 3), 255, np.uint8), tof])


# ---------------------------------------------------------------- API: 状態・センサー・収集
@app.get('/api/status')
def api_status():
    du = shutil.disk_usage(str(config.DATA_DIR if config.DATA_DIR.exists() else config.ROOT))
    return {
        'time': time.time(),
        'sensors_running': hub.running,
        'sensors': hub.status(),
        'recorder': recorder.status(),
        'session': collect.status(),
        'disk': {'free_gb': round(du.free / 1e9, 1), 'total_gb': round(du.total / 1e9, 1)},
        'training': trainer.running,
        'driving': driver.running,
        'config': {'scene_seconds': config.SCENE_SECONDS, 'vehicle_protocol': config.VEHICLE['protocol'],
                   'tof_unit': config.TOF.get('unit', ''), 'camera': '%dx%d@%d' % (
                       config.CAMERA['width'], config.CAMERA['height'], config.CAMERA['fps'])},
    }


@app.post('/api/sensors/start')
def api_sensors_start():
    hub.start()
    return {'ok': True}


@app.post('/api/sensors/stop')
def api_sensors_stop():
    if recorder.recording or collect.active or driver.running:
        raise HTTPException(409, '収集中・走行中はセンサーを停止できません')
    hub.stop()
    return {'ok': True}


LAST_SENSOR_CHECK = {}


def _record_meta(body):
    """収集に残すメモ: コースの配置・電池(画面で入力)と、15分以内のセンサーの向きの点検"""
    meta = {k: str(body.get(k) or '').strip() for k in ('layout', 'battery') if body.get(k)}
    chk = LAST_SENSOR_CHECK.get('result')
    if chk and time.time() - chk.get('time', 0) < 900:
        meta['sensor_check'] = chk
    return meta


@app.post('/api/sensor_check')
def api_sensor_check():
    """センサーの向きの点検(約1秒。車を止め、前に障害物のない床の上で)"""
    from .. import sensor_check
    hub.start()
    deadline = time.time() + 8.0                 # 起動直後はカメラが映るまで数秒かかる
    while time.time() < deadline and (hub['CAM_FRONT'].latest is None or hub['TOF_FRONT'].latest is None):
        time.sleep(0.1)
    res = sensor_check.run(hub, seconds=1.0)
    LAST_SENSOR_CHECK['result'] = res
    return res


@app.get('/api/sensor_check')
def api_sensor_check_last():
    return LAST_SENSOR_CHECK.get('result') or {}


@app.post('/api/record/start')
def api_record_start(body: dict = Body(default={})):
    if trainer.running:
        raise HTTPException(409, '学習中は収集できません(GPU・CPUを奪い合い、30fpsの収集が乱れるため)')
    if collect.active:
        raise HTTPException(409, 'すでに収集中です')
    common = dict(description=body.get('description', ''), tags=body.get('tags', []), location=body.get('location', ''),
                  mode_id=int(body.get('mode_id', modes.UNSET)), meta=_record_meta(body))
    try:
        modes.validate(common['mode_id'])
        if body.get('method', 'runs') == 'runs':
            # 走行ごと: 「n回目の走行をスタート、3、2、1、ピ」〜「ピ、n回目の走行終了」を回数分(間は pause_s 秒停止)
            collect.start(runs=body.get('runs', 1), run_seconds=body.get('run_seconds', config.SCENE_SECONDS),
                          pause_s=body.get('pause_s', 10), **common)
            return {'ok': True, 'method': 'runs'}
        max_scenes = body.get('max_scenes')
        name = recorder.start(max_scenes=int(max_scenes) if max_scenes else None, **common)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True, 'method': 'continuous', 'log': name}


@app.post('/api/record/next')
def api_record_next():
    """走行ごとの収集で、停止中の待ちを飛ばして次の走行を始める"""
    try:
        collect.next_run()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True}


@app.post('/api/record/end_run')
def api_record_end_run():
    """走行ごとの収集で、今の走行を(時間前でも)終える"""
    try:
        collect.end_run()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True}


@app.post('/api/record/mode')
def api_record_mode(body: dict = Body(...)):
    """収集中に走行ジャンルを切り替える(この瞬間から新しいジャンルとして記録)"""
    try:
        mode_id = collect.set_mode(int(body.get('mode_id'))) if collect.active else recorder.set_mode(int(body.get('mode_id')))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True, 'mode_id': mode_id, 'mode_name': modes.name_of(mode_id)}


@app.post('/api/record/stop')
def api_record_stop():
    if collect.active:
        collect.stop()
    else:
        threading.Thread(target=recorder.stop, daemon=True).start()
    return {'ok': True}


@app.get('/api/live/values')
def api_live_values():
    out = history.snapshot()
    tof = hub['TOF_FRONT'].latest
    if tof is not None:
        mm = tof_format.distance_mm(tof.data['depth'])
        v = mm[mm > 0]
        out['tof'] = {'min': int(v.min()) if v.size else 0, 'center': int(mm[mm.shape[0] // 2, mm.shape[1] // 2]),
                      'median': int(np.median(v)) if v.size else 0, 'invalid': int((mm == 0).sum()),
                      'confidence': round(float(tof_format.confidence(tof.data['depth']).mean()), 2)}
    imu = hub['IMU'].latest
    if imu is not None:
        out['imu_now'] = {'accel': [round(float(x), 2) for x in imu.data['accel']],
                          'gyro': [round(float(x), 1) for x in imu.data['gyro']], 'temp': round(imu.data['temp'], 1)}
    veh = hub['VEHICLE'].latest
    if veh is not None:
        out['vehicle_now'] = {k: veh.data[k] for k in ('pwm', 'erpm', 'valid', 'select', 'moving', 'quality')}
    aud = hub['AUDIO'].latest if 'AUDIO' in hub.sensors else None
    if aud is not None and hub['AUDIO'].connected:
        out['audio_now'] = {k: aud.data[k] for k in ('rms_dbfs', 'peak_dbfs', 'channel', 'zero')}
    return out


@app.get('/api/live/audio.f32')
def api_live_audio():
    """マイクの音を流し続ける(float32 リトルエンディアン、16kHz、モノラル)。管理画面の「聞く」で再生する"""
    if 'AUDIO' not in hub.sensors:
        raise HTTPException(404, 'マイクは無効です(JETFORMER_MIC=1 で有効)')
    q = queue.Queue(maxsize=100)

    def on_sample(_ch, sample):
        try:
            q.put_nowait(sample.data['pcm'])
        except queue.Full:
            pass                       # 受け手が遅いときは捨てる(遅れをためない)
    hub['AUDIO'].subscribe(on_sample)

    def gen():
        try:
            while not SHUTDOWN.is_set():
                try:
                    pcm = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                yield (pcm.astype(np.float32) / 2.0 ** 31).astype('<f4').tobytes()
        finally:
            hub['AUDIO'].unsubscribe(on_sample)

    return StreamingResponse(gen(), media_type='application/octet-stream',
                             headers={'X-Sample-Rate': str(config.AUDIO['rate']), 'Cache-Control': 'no-cache'})


@app.get('/api/live.mjpg')
def api_live_mjpeg(fps: float = 15.0):
    boundary = 'frame'

    def gen():
        period = 1.0 / max(min(fps, 30.0), 1.0)
        while not SHUTDOWN.is_set():
            t0 = time.monotonic()
            cam, tof = hub['CAM_FRONT'].latest, hub['TOF_FRONT'].latest
            if not hub.running or cam is None:
                frame = _placeholder('Sensors stopped' if not hub.running else 'Waiting for camera...')
            else:
                frame = _compose(cam.data['image'], tof.data['depth'] if tof is not None else None)
            data = _jpeg(frame, 75)
            yield (b'--' + boundary.encode() + b'\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                   str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
            time.sleep(max(period - (time.monotonic() - t0), 0.005))

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=%s' % boundary)


# ---------------------------------------------------------------- API: シーン
@app.get('/api/scenes')
def api_scenes():
    out = []
    for s in dataset.list_scenes():
        s.pop('path', None)
        out.append(s)
    return out


@app.get('/api/scenes/{name}')
def api_scene(name: str):
    try:
        meta = dataset.load_scene_meta(name)
        meta.pop('path', None)
        meta['series'] = dataset.scene_timeseries(name)
        return meta
    except KeyError:
        raise HTTPException(404, 'シーンがありません')


@app.get('/api/scenes/{name}/thumb.jpg')
def api_scene_thumb(name: str):
    try:
        p = dataset.scene_path(name) / 'thumb.jpg'
    except KeyError:
        raise HTTPException(404)
    if not p.exists():
        return Response(_jpeg(_placeholder('no thumbnail', 448, 224)), media_type='image/jpeg')
    return FileResponse(p, media_type='image/jpeg', headers={'Cache-Control': 'max-age=3600'})


@app.get('/api/scenes/{name}/audio.wav')
def api_scene_audio(name: str):
    try:
        return Response(dataset.scene_audio_wav(name), media_type='audio/wav')
    except KeyError:
        raise HTTPException(404, 'このシーンには音声がありません')


@app.get('/api/scenes/{name}/frame/{index}.jpg')
def api_scene_frame(name: str, index: int):
    try:
        image, depth, ts = dataset.scene_frame(name, index)
    except KeyError:
        raise HTTPException(404)
    return Response(_jpeg(_compose(image, depth), 85), media_type='image/jpeg',
                    headers={'X-Timestamp-Us': str(ts), 'Cache-Control': 'max-age=3600'})


@app.patch('/api/scenes/{name}')
def api_scene_update(name: str, body: dict = Body(...)):
    try:
        return dataset.update_scene(name, body.get('description'), body.get('tags'), body.get('mode_id'))
    except KeyError:
        raise HTTPException(404)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post('/api/scenes/bulk')
def api_scenes_bulk(body: dict = Body(...)):
    """複数シーンのジャンルをまとめて設定する"""
    names = body.get('names') or []
    mode_id = body.get('mode_id')
    if not names or mode_id is None:
        raise HTTPException(400, 'シーンとジャンルを指定してください')
    try:
        mode_id = modes.validate(mode_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    done = []
    for n in names:
        try:
            dataset.update_scene(n, mode_id=mode_id)
            done.append(n)
        except KeyError:
            pass
    return {'ok': True, 'updated': len(done)}


# ---------------------------------------------------------------- API: 走行ジャンル
@app.get('/api/modes')
def api_modes():
    scenes = dataset.list_scenes()
    out = []
    for m in modes.load():
        sc = [s for s in scenes if s['mode_id'] == m['id']]
        out.append(dict(m, scenes=len(sc), minutes=round(sum(s['duration_s'] for s in sc) / 60, 1)))
    return out


@app.post('/api/modes')
def api_mode_add(body: dict = Body(...)):
    try:
        return modes.add(body.get('name'), body.get('description', ''))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch('/api/modes/{mode_id}')
def api_mode_update(mode_id: int, body: dict = Body(...)):
    try:
        return modes.update(mode_id, body.get('name'), body.get('description'))
    except KeyError:
        raise HTTPException(404)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete('/api/scenes/{name}')
def api_scene_delete(name: str):
    if recorder.recording and recorder.log and name.startswith(recorder.log['name']):
        raise HTTPException(409, '収集中のログのシーンは削除できません')
    try:
        dataset.delete_scene(name)
    except KeyError:
        raise HTTPException(404)
    return {'ok': True}


# ---------------------------------------------------------------- API: データセット
@app.get('/api/datasets')
def api_datasets():
    return dataset.list_datasets()


@app.post('/api/datasets')
def api_dataset_create(body: dict = Body(...)):
    name = (body.get('name') or '').strip()
    scenes = body.get('scenes') or []
    if not name or not scenes:
        raise HTTPException(400, 'データセット名とシーンを指定してください')
    val_ratio = float(body.get('val_ratio', 0.2))
    min_duration = float(body.get('min_duration', 5.0))
    job = jobs.submit('build', 'データセット作成: %s' % name,
                      lambda progress: dataset.build_dataset(name, scenes, val_ratio, min_duration, progress=progress,
                                                             val_scenes=body.get('val_scenes'), tags=body.get('tags')))
    return job


@app.post('/api/datasets/{name}/tags')
def api_dataset_tags(name: str, body: dict = Body(...)):
    """データセットのタグを付け直す(例 {"tags": ["simulator"]})。シミュレーターのシーンが入っていれば simulator は外れない"""
    try:
        return dataset.set_dataset_tags(name, body.get('tags') or [])
    except KeyError:
        raise HTTPException(404, 'データセットがありません')


@app.post('/api/datasets/{name}/export')
def api_dataset_export(name: str):
    job = jobs.submit('export', 'エクスポート: %s' % name,
                      lambda progress: {'file': dataset.export_dataset(name, progress=progress).name})
    return job


@app.get('/api/datasets/{name}/download')
def api_dataset_download(name: str):
    """データセットをそのままブラウザにダウンロードする(その場で tar にしながら送る。Jetson にファイルは作らない)"""
    if '..' in name or '/' in name or not (config.DATASET_DIR / name / 'dataset_card.json').exists():
        raise HTTPException(404, 'データセットがありません')
    return StreamingResponse(dataset.iter_dataset_tar(name), media_type='application/x-tar',
                             headers={'Content-Disposition': 'attachment; filename="%s.tar"' % name,
                                      'X-Dataset-Bytes': str(dataset.dataset_bytes(config.DATASET_DIR / name))})


@app.delete('/api/datasets/{name}')
def api_dataset_delete(name: str):
    try:
        dataset.delete_dataset(name)
    except KeyError:
        raise HTTPException(404)
    return {'ok': True}


_hf_status = {'t': 0.0, 'value': None}
_hf_uploading = set()


@app.get('/api/hf/status')
def api_hf_status(refresh: bool = False):
    """Hugging Face にログインしているか(Jetsonで hf auth login 済みか)"""
    if refresh or _hf_status['value'] is None or time.time() - _hf_status['t'] > 60:
        _hf_status['value'], _hf_status['t'] = hf_export.login_status(), time.time()
    return _hf_status['value']


@app.post('/api/datasets/{name}/hf')
def api_dataset_hf(name: str, body: dict = Body(...)):
    """データセットを Hugging Face Hub に上げる(Dataset Viewer で見られる Parquet + nuScenes の tar + カード)"""
    if not (config.DATASET_DIR / name / 'dataset_card.json').exists() or '/' in name or '..' in name:
        raise HTTPException(404, 'データセットがありません')
    repo_id = str(body.get('repo_id', '')).strip()
    if repo_id.count('/') != 1 or not all(repo_id.split('/')):
        raise HTTPException(400, 'リポジトリ名は「ユーザー名/名前」で指定してください')
    if not api_hf_status(refresh=True).get('logged_in'):
        raise HTTPException(409, 'Hugging Face にログインしていません。Jetsonで hf auth login を実行してください')
    if name in _hf_uploading:
        raise HTTPException(409, 'このデータセットはアップロード中です')
    _hf_uploading.add(name)

    def run(progress):
        try:
            return hf_export.upload(name, repo_id, private=bool(body.get('private', True)),
                                    include_nuscenes=bool(body.get('include_nuscenes', True)),
                                    license_id=str(body.get('license') or 'other'), progress=progress)
        finally:
            _hf_uploading.discard(name)
    return jobs.submit('hf', 'Hugging Face へ: %s → %s' % (name, repo_id), run)


_hf_downloading = set()


def _hub_error(e):
    """Hub の例外を、画面に出せる HTTP エラーにする"""
    if isinstance(e, ValueError):
        return HTTPException(400, str(e))
    name = type(e).__name__
    if name in ('RepositoryNotFoundError', 'GatedRepoError', 'RevisionNotFoundError'):
        return HTTPException(404, 'リポジトリが見つかりません(名前の誤り、または非公開でログインが必要): %s' % name)
    return HTTPException(502, 'Hugging Face に接続できません: %s' % name)


@app.get('/api/hf/hub_datasets')
def api_hf_hub_datasets(author: str = '', search: str = '', all_tags: bool = False, kind: str = 'all'):
    """Hub の JetFormer のデータセットを探す(author = ユーザー名または組織、search = 名前の一部、
    kind = all・real(実機)・simulator(シミュレーター))"""
    if kind not in ('all', 'real', 'simulator'):
        raise HTTPException(400, 'kind は all・real・simulator のどれか')
    try:
        return hf_export.list_hub_datasets(author=author, search=search, only_jetformer=not all_tags, kind=kind)
    except Exception as e:
        raise _hub_error(e)


@app.get('/api/hf/hub_files')
def api_hf_hub_files(repo_id: str, revision: str = ''):
    """リポジトリにある学習用の tar(nuscenes/*.tar)と大きさ。同じ名前のデータセットが既にあるかも返す"""
    try:
        return {'repo_id': repo_id, 'files': hf_export.hub_dataset_files(repo_id, revision or None)}
    except Exception as e:
        raise _hub_error(e)


@app.post('/api/hf/download')
def api_hf_download(body: dict = Body(...)):
    """Hub のデータセットを Jetson の data/datasets/ に取り込む(処理状況に進み具合が出る)"""
    repo_id = str(body.get('repo_id', '')).strip()
    revision = str(body.get('revision') or '').strip() or None
    if repo_id.count('/') != 1 or not all(repo_id.split('/')):
        raise HTTPException(400, 'リポジトリ名は「ユーザー名または組織/名前」で指定してください')
    if repo_id in _hf_downloading:
        raise HTTPException(409, 'このリポジトリはダウンロード中です')
    _hf_downloading.add(repo_id)

    def run(progress):
        try:
            return hf_export.download(repo_id, revision=revision, filename=body.get('file') or None, progress=progress)
        finally:
            _hf_downloading.discard(repo_id)
    return jobs.submit('hf_download', 'Hugging Face から: %s' % repo_id, run)


@app.get('/api/jobs')
def api_jobs():
    return jobs.list()


@app.get('/api/exports/{filename}')
def api_export_download(filename: str):
    p = config.EXPORT_DIR / filename
    if '/' in filename or '..' in filename or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=filename)


# ---------------------------------------------------------------- API: 学習
@app.post('/api/train/start')
def api_train_start(body: dict = Body(...)):
    if recorder.recording or collect.active:
        raise HTTPException(409, '収集中は学習できません(収集を止めてから開始してください)')
    if driver.running:
        raise HTTPException(409, 'AI走行中は学習できません(GPUを奪い合い、30fpsの走行が乱れるため)')
    try:
        # モデルの種類・衝突の加速度・乱数の種も(無ければ jetformer_v0 = 古い画面のキャッシュからでも今までと同じ学習)
        name = trainer.start(body.get('dataset', ''), **training_start_args(body))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True, 'run': name}


@app.get('/api/policies')
def api_policies():
    """学習できるモデルの種類(方策の登録簿。torch を読まない): 名前・表示名・説明・状態・設定・目安"""
    return policies.list_specs()


@app.post('/api/train/stop')
def api_train_stop():
    try:
        trainer.stop()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True}


@app.get('/api/train/status')
def api_train_status():
    return trainer.status()


@app.get('/api/runs')
def api_runs():
    return trainer.list_runs()


@app.delete('/api/runs/{name}')
def api_run_delete(name: str):
    if driver.running and driver.st.get('run') == name:
        raise HTTPException(409, '走行に使っているモデルは削除できません')
    try:
        trainer.delete_run(name)
    except KeyError:
        raise HTTPException(404)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True}


@app.get('/api/runs/{name}/files/{filename}')
def api_run_file(name: str, filename: str):
    try:
        return FileResponse(trainer.artifact_path(name, filename), filename='%s_%s' % (name, filename))
    except KeyError:
        raise HTTPException(404)


# ---------------------------------------------------------------- API: 学習済みモデルの書き出し(エミュレーター用)
@app.get('/api/models')
def api_models():
    """ONNX と model_config.json がある学習済みモデルの一覧(エミュレーターなど Jetson の外で使う用)"""
    return model_export.list_models()


@app.get('/api/models/{name}/io_spec')
def api_model_io_spec(name: str):
    """入力の作り方・出力の読み方"""
    try:
        return model_export.io_spec(name)
    except KeyError:
        raise HTTPException(404, 'モデルがありません')


@app.get('/api/models/{name}/bundle.zip')
def api_model_bundle(name: str, pth: bool = False):
    """エミュレーター用の一式(ONNX・model_config.json・io_spec.json・README・動かす例。pth=true で best.pth も)を zip で"""
    try:
        body = model_export.iter_bundle_zip(name, pth=pth)
    except KeyError:
        raise HTTPException(404, 'モデルがありません')
    return StreamingResponse(body, media_type='application/zip',
                             headers={'Content-Disposition': 'attachment; filename="%s%s.zip"' % (name, '_pth' if pth else '')})


# ---------------------------------------------------------------- API: AI走行
@app.get('/api/drive/models')
def api_drive_models():
    return drive.list_models()


@app.get('/api/drive/status')
def api_drive_status(token: str = Query(default='')):
    """走行画面が0.25秒ごとに呼ぶ。走行を始めたページの token 付きの問い合わせが途絶えると走行を止める
    (画面を閉じた・スマホがスリープした・通信断)。token なしの問い合わせは表示用で、生存確認にならない"""
    alive = driver.heartbeat(token)
    st = driver.status()
    st['controller'] = alive
    return st


DRIVE_CLIENT_VERSION = 6        # 生存確認(token)を送る画面の版。これより古い画面(キャッシュ)からは走行を始めない


@app.post('/api/drive/start')
def api_drive_start(body: dict = Body(...)):
    if int(body.get('client') or 0) < DRIVE_CLIENT_VERSION:
        raise HTTPException(409, '管理画面が古いままです。ページを再読み込み(更新)してから走行を開始してください')
    if trainer.running:
        raise HTTPException(409, '学習中は走行できません(GPUを奪い合い、30fpsを守れないため)')
    try:
        run, token = driver.start(body.get('run', ''), body.get('mode_id', -1), record=bool(body.get('record')),
                                  description=body.get('description', ''), tags=body.get('tags') or [],
                                  dry_run=bool(body.get('dry_run')), params=body.get('params') or {})
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True, 'run': run, 'token': token}


@app.post('/api/drive/stop')
def api_drive_stop(body: dict = Body(default={})):
    try:
        driver.stop(body.get('reason') or '停止ボタン')
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True}


@app.post('/api/drive/takeover_reason')
def api_drive_takeover_reason(body: dict = Body(default={})):
    try:
        return driver.takeover_reason(str(body.get('reason', '')))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post('/api/drive/params')
def api_drive_params(body: dict = Body(...)):
    """ゲイン・トリム・先読み・時計補正・ジャンルを変える(走行中もすぐ反映)"""
    try:
        return driver.set_params(body)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- API: サウンド(車のスピーカー)
@app.get('/api/sound')
def api_sound():
    return player.status()


@app.post('/api/sound')
def api_sound_update(body: dict = Body(...)):
    try:
        return player.update(body)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post('/api/sound/test')
def api_sound_test():
    player.enqueue('スピーカーのテストです。音量 %d' % player.settings['volume'], 'start')
    return {'ok': True}


@app.post('/api/sound/say')
def api_sound_say(body: dict = Body(...)):
    text = str(body.get('text', '')).strip()[:100]
    if not text:
        raise HTTPException(400, '話す内容を入れてください')
    if not player.enqueue(text, 'info'):
        raise HTTPException(409, '再生待ちが多すぎます')
    return {'ok': True}


# ---------------------------------------------------------------- 画面
class _NoCacheStatic(StaticFiles):
    """画面のファイルは毎回更新を確かめさせる(古い app.js のまま動かないように)"""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers['Cache-Control'] = 'no-cache'
        return resp


app.mount('/static', _NoCacheStatic(directory=str(STATIC)), name='static')


@app.get('/')
def index():
    # app.js・style.css に更新時刻を付けて、更新したら必ず読み直させる
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    for name in ('app.js', 'style.css'):
        html = html.replace('/static/%s"' % name, '/static/%s?v=%d"' % (name, int((STATIC / name).stat().st_mtime)))
    return Response(html, media_type='text/html', headers={'Cache-Control': 'no-cache'})


def main():
    import uvicorn

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            # 終了の合図を受けたらすぐ走行を止める(uvicornは開いている接続が閉じるまで終了処理を待つため、
            # lifespan を待っていると走行が続いてしまう)
            SHUTDOWN.set()
            try:
                driver.stop('サーバー終了')
            except RuntimeError:
                pass
            super().handle_exit(sig, frame)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        Server(uvicorn.Config(app, host=config.WEB_HOST, port=config.WEB_PORT, log_level='warning',
                              timeout_graceful_shutdown=2)).run()
    except KeyboardInterrupt:          # uvicornが終了後に受けたSIGINTを投げ直すため(終了処理は済んでいる)
        pass


if __name__ == '__main__':
    main()
