"""E2E 確認(design.md §8.3 の 1〜9・11)を試験インスタンスに対して行い、結果を JSON で出す

    python e2e_check.py http://127.0.0.1:18080 <結果の JSON>
"""
import base64
import hashlib
import io
import json
import re
import socket
import subprocess
import sys
import time
import zipfile

import cv2
import httpx
import numpy as np

BASE, OUT = sys.argv[1], sys.argv[2]
REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5'
c = httpx.Client(base_url=BASE, timeout=120, trust_env=False)
res = {'steps': [], 'started': time.strftime('%Y-%m-%d %H:%M:%S')}


def step(name, ok, **info):
    res['steps'].append(dict(name=name, ok=bool(ok), **info))
    print(('OK  ' if ok else 'NG  ') + name, json.dumps(info, ensure_ascii=False)[:400], flush=True)


def wait_job(job, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for j in c.get('/api/jobs').json():
            if j['id'] == job['id'] and j['state'] != 'running':
                return j
        time.sleep(0.5)
    raise RuntimeError('job timeout')


def sha(b):
    return hashlib.sha256(b).hexdigest()


# 1. 状態
s = c.get('/api/sim/status').json()
inbox = {z['file']: z for z in s['inbox']}
step('1 status: 導入済み・inbox に本体', s['release']['installed'] and inbox.get('course-simulator-complete.zip', {}).get('kind') == 'simulator',
     release=s['release']['id'], inbox=list(inbox), env_ready=s['env']['ready'], bridge=s['bridge']['state'])

# 2. 導入(もう一度押しても展開し直さない)
before = subprocess.run(['stat', '-c', '%s %Y', '/home/jetson/sim/course-simulator-complete.zip'], capture_output=True, text=True).stdout
j = wait_job(c.post('/api/sim/install', json={'file': 'course-simulator-complete.zip'}).json())
after = subprocess.run(['stat', '-c', '%s %Y', '/home/jetson/sim/course-simulator-complete.zip'], capture_output=True, text=True).stdout
import os
step('2 install: job done・already_installed・dataset-downloads なし・~/sim は変わらない',
     j['state'] == 'done' and j['result']['already_installed'] and not os.path.exists(REL + '/dataset-downloads') and before == after,
     result=j.get('result') and {k: j['result'][k] for k in ('release_id', 'files', 'bytes', 'already_installed')}, error=j.get('error'))

# 3. 準備(済み)
j = wait_job(c.post('/api/sim/setup', json={}).json())
st = c.get('/api/sim/status').json()['env']
step('3 setup: 版が一致・self-test 済み', j['state'] == 'done' and st['ready'] and st['packages'].get('onnxruntime') == '1.31.0',
     packages=st['packages'], python=st['python_version'], job=j['state'], already=(j.get('result') or {}).get('already_ready'))

# 4. ブリッジの起動
r = c.post('/api/sim/bridge/start')
t0 = time.time()
while time.time() - t0 < 30:
    b = c.get('/api/sim/status').json()['bridge']
    if b['state'] in ('running', 'failed'):
        break
    time.sleep(0.3)
port = b.get('port')
ss = subprocess.run(['ss', '-ltnH', 'sport = :%s' % port], capture_output=True, text=True).stdout.strip()
step('4 bridge start: running・127.0.0.1 だけで待ち受け', b['state'] == 'running' and ss and all('127.0.0.1:%s' % port in line for line in ss.splitlines()),
     seconds=round(time.time() - t0, 1), port=port, pid=b.get('pid'), rss_mb=b.get('rss_mb'), ss=ss)

# 5. 中継
r = c.get('/sim/bridge/api/status')
r2 = c.get('/sim/bridge/api/status', headers={'Host': '192.168.5.14:8080', 'Origin': 'http://192.168.5.14:8080'})
lan_ip = subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.split()[0]
try:
    socket.create_connection((lan_ip, port), timeout=2).close()
    lan_refused = False
except OSError:
    lan_refused = True
step('5 proxy: /sim/bridge/api/status 200(LAN の Host でも)・ブリッジのポートに LAN からつながらない',
     r.status_code == 200 and r.json()['policy_loaded'] is False and r2.status_code == 200 and lan_refused,
     status=r.status_code, lan_host_status=r2.status_code, lan_ip=lan_ip, lan_refused=lan_refused)

# 7(先). コースのアップロード(Default の雛形を書き出し → world_version と板 1 枚を変える)
z = zipfile.ZipFile(io.BytesIO(c.get('/api/sim/courses/default/package.zip').content))
course = json.loads(z.read('default/course.json'))
world = json.loads(z.read('default/physics-world.json'))
course.update(world_version='e2e-course-1', id='e2e-course-1', name='E2E 確認コース')
course.pop('world_hash', None)
world['segments'].append({'a': [-1.0, -1.4], 'b': [-0.6, -1.4], 'radius': 0.0095, 'height': 0.119, 'kind': 'rail'})
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w') as zz:
    zz.writestr('e2e/course.json', json.dumps(course, ensure_ascii=False))
    zz.writestr('e2e/physics-world.json', json.dumps(world))
pkg = buf.getvalue()
r = c.put('/api/sim/courses/import?filename=e2e.zip', content=pkg)
up = r.json()
r_again = c.put('/api/sim/courses/import?filename=e2e.zip', content=pkg)
world2 = json.loads(json.dumps(world))
world2['segments'][-1]['b'] = [-0.5, -1.4]
buf2 = io.BytesIO()
with zipfile.ZipFile(buf2, 'w') as zz:
    zz.writestr('course.json', json.dumps(course, ensure_ascii=False))
    zz.writestr('physics-world.json', json.dumps(world2))
r_conflict = c.put('/api/sim/courses/import?filename=e2e2.zip', content=buf2.getvalue())
buf3 = io.BytesIO()
with zipfile.ZipFile(buf3, 'w') as zz:
    zz.writestr('../physics-world.json', json.dumps(world2))
r_attack = c.put('/api/sim/courses/import?filename=atk.zip', content=buf3.getvalue())
step('7 course upload: data で登録・再送は登録済み・形違いは 409・../ は 400',
     r.status_code == 200 and up['course']['kind'] == 'data' and r_again.json().get('already_registered') and r_conflict.status_code == 409
     and r_attack.status_code == 400,
     course={k: up['course'][k] for k in ('id', 'kind', 'world_hash', 'stats')} if r.status_code == 200 else r.text,
     again=r_again.status_code, conflict=r_conflict.status_code, attack=r_attack.status_code, attack_detail=r_attack.json().get('detail'))
cid = up['course']['id'] if r.status_code == 200 else None

# 6. ページ
expect = {'default': (sha(open(REL + '/dist/scene.js', 'rb').read()), sha(open(REL + '/dist/world.js', 'rb').read())),
          'indoor-course-eval-2-shortcut': ('6e4a72655f5a9191516e237e4d0a0beeb7ac1a9ad43bd5e87079c98fb3e5c824',
                                             'ec3a813d7d4e9b3c80bba895c6aa7eb788a14c398d80f03654a8a3ae648265e6')}
static = '/home/jetson/jetformer/jetformer/sim/static/'
if cid:
    expect[cid] = (sha(open(static + 'course-scene.js', 'rb').read()), sha(open(static + 'course-world.js', 'rb').read()))
pages = {}
for k, (s1, s2) in expect.items():
    html = c.get('/sim/c/%s/' % k).text
    sc, wc = c.get('/sim/c/%s/scene.js' % k), c.get('/sim/c/%s/world.js' % k)
    img = c.get('/sim/c/%s/assets/carpet.jpg' % k)
    cfg = c.get('/sim/c/%s/_jf/config.js' % k).text
    pages[k] = {'shim': 'sim-shim.js' in html and './_jf/config.js' in html, 'scene': sha(sc.content) == s1, 'world': sha(wc.content) == s2,
                'carpet': img.status_code == 200 and img.headers['content-type'] == 'image/jpeg', 'config': '"id":"%s"' % k in cfg}
data_mod = c.get('/sim/c/%s/_jf/course-data.js' % cid).text if cid else ''
step('6 pages: shim 入り・scene/world の sha256・carpet.jpg', all(all(v.values()) for v in pages.values()) and data_mod.startswith('export default'),
     pages=pages)
default_hash = c.get('/api/sim/courses/default').json()['world_hash']
step('6b Default は既知の v3(hash 0a464384)', default_hash == '0a464384', world_hash=default_hash)

# 8. モデル(サーバー側で bundle → activate)→ predict
models = [m for m in c.get('/api/sim/models').json() if m['sim_compatible']]
run = next((m['name'] for m in models if 'jetformer_v0' in m['name']), models[0]['name'] if models else None)
t0 = time.time()
r = c.post('/api/sim/bridge/load_model', json={'run': run}, timeout=600)
load = r.json()
st = c.get('/sim/bridge/api/status').json()
png = cv2.imencode('.png', np.full((224, 224, 3), 120, np.uint8))[1].tobytes()
packed = np.full((42, 54), (4 << 13) | 900, np.uint16).flatten().tolist()


def obs(i, episode='e2e'):
    t = i / 30
    latest = int(np.floor((t + 1e-9) * 100)) / 100
    return {'schema': 'course-sim-observation-v1', 'episode_id': episode, 'frame': i, 'sim_time': t, 'dt': 1 / 30, 'speed': 0, 'steering': 0,
            'ranges': [{'angle': 0, 'distance': 2}],
            'camera': {'width': 224, 'height': 224, 'encoding': 'png-base64', 'data': base64.b64encode(png).decode()},
            'depth': {'width': 54, 'height': 42, 'packed_data': packed},
            'imu': {'units': {'acceleration': 'm/s2', 'gyro': 'rad/s'}, 'samples': [[0, 0, 9.80665, 0, 0, 0]] * 10,
                    'timestamps': [latest - (9 - j) / 100 for j in range(10)]},
            'actuator_limits': {'max_speed_mps': 1.6, 'max_steer_rad': 0.52}}


lat = []
pred = None
for i in range(5):
    t1 = time.time()
    p = c.post('/sim/bridge/api/predict', json=obs(i))
    lat.append(round((time.time() - t1) * 1000))
    pred = p
step('8 model: bundle → activate → policy_loaded・predict 200',
     r.status_code == 200 and st.get('policy_loaded') and pred.status_code == 200 and 'steering' in pred.json(),
     run=run, load=load, load_s=round(time.time() - t0, 1), predict_status=pred.status_code,
     predict=pred.json() if pred.status_code == 200 else pred.text[:300], latency_ms=lat)

# 9. データ(中継経由で収集 → 一覧 → 取り込み → 片付け)
name = 'e2e_sim_%s' % time.strftime('%H%M%S')
start = c.post('/sim/bridge/api/datasets/start', json={'name': name, 'mode_id': 5, 'controller': 'manual', 'split': 'train'})
sid = start.json().get('session_id')
frames_ok = True
for i in range(15):
    f = c.post('/sim/bridge/api/datasets/frame', json={'session_id': sid, 'observation': obs(i, 'e2e-rec'),
                                                       'action': {'steering': .001 * i, 'speed': .01 * i},
                                                       'pose': {'x': .01 * i, 'z': 0, 'yaw': 0}, 'state': {'x': .01 * (i + 1), 'z': 0, 'yaw': 0}})
    frames_ok = frames_ok and f.status_code == 200
fin = c.post('/sim/bridge/api/datasets/finish', json={'session_id': sid})
dl = c.get('/sim/bridge' + fin.json().get('download_url', '/x'))
lst = c.get('/api/sim/datasets').json()
item = next((d for d in lst['datasets'] if d['session_id'] == sid), None)
j = wait_job(c.post('/api/sim/datasets/%s/import' % sid).json())
lst2 = c.get('/api/sim/datasets').json()
item2 = next((d for d in lst2['datasets'] if d['session_id'] == sid), None)
ds_dir = os.path.join(os.environ.get('E2E_DATA', ''), 'datasets', (j.get('result') or {}).get('name') or 'none')
step('9 dataset: start → frame×15 → finish → 一覧 → 取り込み',
     start.status_code == 200 and frames_ok and fin.status_code == 200 and dl.status_code == 200 and item and j['state'] == 'done'
     and item2 and item2['imported'],
     session=sid, finish={k: fin.json().get(k) for k in ('filename', 'bytes', 'training_frames')}, download_bytes=len(dl.content),
     listed=item and {k: item[k] for k in ('kind', 'frames', 'dataset_name')}, job=j['state'], job_error=j.get('error'),
     imported_name=(j.get('result') or {}).get('name'), scenes=(j.get('result') or {}).get('scenes'))
# 片付け: ブリッジの作業フォルダー(取り込んだデータセットは試験用の DATA にある)
d = c.delete('/api/sim/datasets/%s' % sid)
step('9b runtime の片付け', d.status_code == 200, status=d.status_code)
if cid:
    c.delete('/api/sim/courses/%s' % cid)

# ログ
logs = c.get('/api/sim/bridge/log?lines=20').json()
res['bridge_log_tail'] = logs

# 11. 停止
b = c.post('/api/sim/bridge/stop', json={}).json()
time.sleep(0.5)
alive = subprocess.run(['pgrep', '-f', 'bridge_runner.py'], capture_output=True, text=True).stdout.strip()
step('11 bridge stop: stopped・子プロセスなし', b['state'] == 'stopped' and not alive, state=b['state'], pgrep=alive)

res['ok'] = all(s['ok'] for s in res['steps'])
res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('ALL OK' if res['ok'] else 'SOME NG')
