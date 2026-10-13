"""中継(/sim/bridge/api/*)の遅れを測る: 同じ要求をブリッジへ直接と JetFormer の中継経由で送り比べる

    python -I proxy_latency.py [--base http://127.0.0.1:8080] [--n 40]

ブリッジは JetFormer の API で起動・停止する。収集のセッションは最後に消す。CPU だけ(GPU・実車は使わない)。
"""
import argparse
import base64
import json
import statistics
import sys
import time

import cv2
import httpx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='http://127.0.0.1:8080')
ap.add_argument('--n', type=int, default=40)
ap.add_argument('--out', default='/home/jetson/jetformer_reports/sim_integration_20261013/verify/proxy_latency.json')
a = ap.parse_args()
c = httpx.Client(base_url=a.base, timeout=120, trust_env=False)
rng = np.random.default_rng(0)
img = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
img = cv2.GaussianBlur(img, (5, 5), 0)                 # 実際の画面に近い大きさの PNG(約 100 KB)
png = cv2.imencode('.png', img)[1].tobytes()
packed = np.full((42, 54), (4 << 13) | 900, np.uint16).flatten().tolist()


def obs(i, episode='lat'):
    t = i / 30
    latest = int(np.floor((t + 1e-9) * 100)) / 100
    return {'schema': 'course-sim-observation-v1', 'episode_id': episode, 'frame': i, 'sim_time': t, 'dt': 1 / 30, 'speed': 0, 'steering': 0,
            'ranges': [{'angle': 0, 'distance': 2}],
            'camera': {'width': 224, 'height': 224, 'encoding': 'png-base64', 'data': base64.b64encode(png).decode()},
            'depth': {'width': 54, 'height': 42, 'packed_data': packed},
            'imu': {'units': {'acceleration': 'm/s2', 'gyro': 'rad/s'}, 'samples': [[0, 0, 9.80665, 0, 0, 0]] * 10,
                    'timestamps': [latest - (9 - j) / 100 for j in range(10)]},
            'actuator_limits': {'max_speed_mps': 1.6, 'max_steer_rad': 0.52}}


def stats(xs):
    xs = sorted(xs)
    return {'n': len(xs), 'median_ms': round(statistics.median(xs), 1), 'p90_ms': round(xs[int(0.9 * (len(xs) - 1))], 1),
            'max_ms': round(xs[-1], 1)}


res = {'body_bytes': len(json.dumps(obs(0)))}
started_here = False
sid = None
try:
    st = c.get('/api/sim/status').json()['bridge']
    if st['state'] != 'running':
        c.post('/api/sim/bridge/start')
        started_here = True
        for _ in range(60):
            st = c.get('/api/sim/status').json()['bridge']
            if st['state'] == 'running':
                break
            time.sleep(0.5)
    port = st['port']
    direct = httpx.Client(base_url='http://127.0.0.1:%d' % port, timeout=60, trust_env=False,
                          headers={'Host': '127.0.0.1:%d' % port})
    # 1. GET status
    for name, cl, path in (('status_direct', direct, '/api/status'), ('status_proxy', c, '/sim/bridge/api/status')):
        lat = []
        for _ in range(a.n):
            t0 = time.perf_counter()
            r = cl.get(path)
            lat.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200, r.text
        res[name] = stats(lat)
    # 2. 収集の frame(約 100 KB の JSON)
    r = c.post('/sim/bridge/api/datasets/start', json={'name': 'latency-probe', 'mode_id': 8, 'controller': 'manual', 'split': 'train'})
    sid = r.json()['session_id']
    i = 0
    for name, cl, path in (('frame_direct', direct, '/api/datasets/frame'), ('frame_proxy', c, '/sim/bridge/api/datasets/frame')):
        lat = []
        for _ in range(a.n):
            body = {'session_id': sid, 'observation': obs(i), 'action': {'steering': 0.0, 'speed': 0.1},
                    'pose': {'x': .01 * i, 'z': 0, 'yaw': 0}, 'state': {'x': .01 * (i + 1), 'z': 0, 'yaw': 0}}
            raw = json.dumps(body).encode()
            t0 = time.perf_counter()
            r = cl.post(path, content=raw, headers={'Content-Type': 'application/json'})
            lat.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200, r.text[:300]
            i += 1
        res[name] = stats(lat)
    fin = c.post('/sim/bridge/api/datasets/finish', json={'session_id': sid})
    res['finish_status'] = fin.status_code
finally:
    if sid:
        d = c.delete('/api/sim/datasets/%s' % sid)
        res['cleanup_session'] = d.status_code
    if started_here:
        c.post('/api/sim/bridge/stop', json={'force': True})
        res['bridge_stopped'] = True
    with open(a.out, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
