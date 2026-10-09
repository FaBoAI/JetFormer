"""JetFormer 収集管理Webサーバー

    python -m jetformer.web.server          # http://<jetson>:8080/
"""

import collections
import contextlib
import shutil
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import config, dataset, modes
from ..recorder import Recorder, depth_colormap
from ..sensors import SensorHub

STATIC = config.ROOT / 'jetformer' / 'web' / 'static'

hub = SensorHub()
recorder = Recorder(hub)


@contextlib.asynccontextmanager
async def lifespan(_app):
    yield
    # 終了時: 収集中なら最後のシーンを保存してからセンサーを止める
    if recorder.recording:
        recorder.stop()
    hub.stop()


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
            self.veh.append((now, d['pwm'][0], d['pwm'][1], d['pwm'][2], d['motor_pct']))

    def snapshot(self):
        now = time.monotonic()
        imu, veh = list(self.imu), list(self.veh)
        return {
            'imu': {'t': [round(r[0] - now, 2) for r in imu], 'accel_norm': [round(r[1], 2) for r in imu],
                    'accel_x': [round(r[2], 2) for r in imu], 'accel_y': [round(r[3], 2) for r in imu],
                    'gyro_z': [round(r[4], 1) for r in imu]},
            'vehicle': {'t': [round(r[0] - now, 2) for r in veh], 'steering': [r[1] for r in veh],
                        'throttle': [r[2] for r in veh], 'aux': [r[3] for r in veh], 'motor_pct': [r[4] for r in veh]},
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
        lo, hi = int(np.percentile(depth, 2)), int(np.percentile(depth, 98))
        cv2.putText(tof, '%d-%d %s' % (lo, hi, config.TOF.get('unit', '')), (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
        'disk': {'free_gb': round(du.free / 1e9, 1), 'total_gb': round(du.total / 1e9, 1)},
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
    if recorder.recording:
        raise HTTPException(409, '収集中はセンサーを停止できません')
    hub.stop()
    return {'ok': True}


@app.post('/api/record/start')
def api_record_start(body: dict = Body(default={})):
    try:
        max_scenes = body.get('max_scenes')
        name = recorder.start(description=body.get('description', ''), tags=body.get('tags', []),
                              location=body.get('location', ''),
                              max_scenes=int(max_scenes) if max_scenes else None,
                              mode_id=int(body.get('mode_id', modes.UNSET)))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True, 'log': name}


@app.post('/api/record/mode')
def api_record_mode(body: dict = Body(...)):
    """収集中に走行ジャンルを切り替える(この瞬間から新しいジャンルとして記録)"""
    try:
        mode_id = recorder.set_mode(int(body.get('mode_id')))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {'ok': True, 'mode_id': mode_id, 'mode_name': modes.name_of(mode_id)}


@app.post('/api/record/stop')
def api_record_stop():
    threading.Thread(target=recorder.stop, daemon=True).start()
    return {'ok': True}


@app.get('/api/live/values')
def api_live_values():
    out = history.snapshot()
    tof = hub['TOF_FRONT'].latest
    if tof is not None:
        d = tof.data['depth']
        out['tof'] = {'min': int(d.min()), 'center': int(d[d.shape[0] // 2, d.shape[1] // 2]),
                      'median': int(np.median(d)), 'max': int(d.max())}
    imu = hub['IMU'].latest
    if imu is not None:
        out['imu_now'] = {'accel': [round(float(x), 2) for x in imu.data['accel']],
                          'gyro': [round(float(x), 1) for x in imu.data['gyro']], 'temp': round(imu.data['temp'], 1)}
    veh = hub['VEHICLE'].latest
    if veh is not None:
        out['vehicle_now'] = {k: veh.data[k] for k in ('pwm', 'motor_pct', 'valid', 'quality')}
    return out


@app.get('/api/live.mjpg')
def api_live_mjpeg(fps: float = 15.0):
    boundary = 'frame'

    def gen():
        period = 1.0 / max(min(fps, 30.0), 1.0)
        while True:
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
                      lambda progress: dataset.build_dataset(name, scenes, val_ratio, min_duration, progress=progress))
    return job


@app.post('/api/datasets/{name}/export')
def api_dataset_export(name: str):
    job = jobs.submit('export', 'エクスポート: %s' % name,
                      lambda progress: {'file': dataset.export_dataset(name, progress=progress).name})
    return job


@app.delete('/api/datasets/{name}')
def api_dataset_delete(name: str):
    try:
        dataset.delete_dataset(name)
    except KeyError:
        raise HTTPException(404)
    return {'ok': True}


@app.get('/api/jobs')
def api_jobs():
    return jobs.list()


@app.get('/api/exports/{filename}')
def api_export_download(filename: str):
    p = config.EXPORT_DIR / filename
    if '/' in filename or '..' in filename or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=filename)


# ---------------------------------------------------------------- 画面
app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-cache'})


def main():
    import uvicorn
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT, log_level='warning')


if __name__ == '__main__':
    main()
