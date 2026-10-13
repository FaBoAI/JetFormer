"""review の修正の確認(試験インスタンス 127.0.0.1:18080。jetformer-web は再起動しない)"""
import concurrent.futures as cf
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SCR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
REAL = Path('/home/jetson/jetformer/data')
DATA = SCR / 'data'
PORT = 18080
REL = REAL / 'sim/releases/20261013-144013-6f610dd5'
res = {}


def http(method, path, body=None, headers=None, timeout=30):
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (PORT, path), data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def js(method, path, value=None):
    body = json.dumps(value).encode() if value is not None else None
    st, raw = http(method, path, body, {'Content-Type': 'application/json'} if body else {})
    try:
        return st, json.loads(raw)
    except ValueError:
        return st, raw[:200]


DATA.mkdir(parents=True, exist_ok=True)
for n in ('sim', 'runs'):
    if not (DATA / n).exists():
        (DATA / n).symlink_to(REAL / n)
log = open(SCR / 'e2e_server.log', 'w')
srv = subprocess.Popen([sys.executable, '/home/jetson/jetformer_reports/sim_integration_20261013/e2e/e2e_server.py', str(DATA), str(PORT)],
                       stdout=log, stderr=subprocess.STDOUT, cwd=str(SCR))
try:
    for _ in range(100):
        try:
            if http('GET', '/api/sim/status')[0] == 200:
                break
        except OSError:
            time.sleep(0.2)
    st, s = js('GET', '/api/sim/status')
    res['status'] = (st, s['release']['id'], s['env']['ready'], s['bridge']['state'])

    # 1) ブリッジ: 起動・oom_score_adj・--parent-pid
    js('POST', '/api/sim/bridge/start')
    for _ in range(100):
        st, s = js('GET', '/api/sim/status')
        if s['bridge']['state'] in ('running', 'failed'):
            break
        time.sleep(0.2)
    b = s['bridge']
    pid = b['pid']
    cmd = Path('/proc/%d/cmdline' % pid).read_bytes().split(b'\0')
    res['bridge'] = {'state': b['state'], 'pid': pid, 'port': b['port'],
                     'oom_score_adj': int(Path('/proc/%d/oom_score_adj' % pid).read_text()),
                     'parent_pid_arg': cmd[cmd.index(b'--parent-pid') + 1].decode(), 'server_pid': srv.pid}
    res['proxy_status'] = http('GET', '/sim/bridge/api/status', headers={'Host': '192.168.1.20:8080', 'Origin': 'http://192.168.1.20:8080'})[0]
    res['proxy_newline'] = [http('GET', '/sim/bridge/api/' + p)[0] for p in ('status%0a', 'status%0d%0a', 'status%0aX')]

    # 2) コース: 本物の版の組(v2 のスナップショット)は trusted-js、版のほかの JS(app.js)を scene.js にしたものは data
    snap = REL / 'qa/course-snapshots/indoor-course-eval-2-shortcut'
    man = json.loads((snap / 'manifest.json').read_text())

    def course_zip(scene, world, wv):
        m = dict(man, world_version=wv)
        m['files'] = dict(man['files'], **{'scene.js': hashlib.sha256(scene).hexdigest(), 'world.js': hashlib.sha256(world).hexdigest()})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('c/manifest.json', json.dumps(m))
            z.writestr('c/physics-world.json', (snap / 'physics-world.json').read_bytes())
            z.writestr('c/scene.js', scene)
            z.writestr('c/world.js', world)
        return buf.getvalue()
    good = course_zip((snap / 'scene.js').read_bytes(), (snap / 'world.js').read_bytes(), 'review-v2-pair')
    bad = course_zip((REL / 'dist/app.js').read_bytes(), (snap / 'world.js').read_bytes(), 'review-v2-appjs')
    swapped = course_zip((REL / 'dist/world.js').read_bytes(), (REL / 'dist/scene.js').read_bytes(), 'review-v3-swapped')

    def up(data, name):
        st, raw = http('PUT', '/api/sim/courses/import?filename=%s' % name, data, {'Content-Type': 'application/zip'})
        try:
            v = json.loads(raw)
        except ValueError:
            v = {'raw': raw[:200].decode('utf-8', 'replace')}
        return st, v
    st, v = up(good, 'good.zip')
    res['course_pair'] = (st, v.get('course', {}).get('kind'), v.get('course', {}).get('world_hash'))
    st, v = up(bad, 'bad.zip')
    res['course_appjs'] = (st, v.get('course', {}).get('kind'), [w for w in v.get('course', {}).get('warnings', []) if 'JS' in w][:1])
    st, v = up(swapped, 'swapped.zip')
    res['course_swapped'] = (st, v.get('course', {}).get('kind'))
    # 同じものを 4 つ同時に
    same = course_zip((snap / 'scene.js').read_bytes(), (snap / 'world.js').read_bytes(), 'review-v2-concurrent')
    with cf.ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda i: up(same, 'same%d.zip' % i), range(4)))
    res['concurrent'] = sorted((st, v.get('already_registered')) for st, v in outs)
    st, page = http('GET', '/sim/c/review-v2-pair/scene.js')
    res['pair_scene_served_sha_ok'] = hashlib.sha256(page).hexdigest() == man['files']['scene.js']
    st, page = http('GET', '/sim/c/review-v2-appjs/scene.js')
    res['appjs_course_scene_is_generic'] = b'jetformer-generic-course' in page
    for cid in ('review-v2-pair', 'review-v2-appjs', 'review-v3-swapped', 'review-v2-concurrent'):
        js('DELETE', '/api/sim/courses/%s' % cid)
    res['courses_after_delete'] = [c['id'] for c in js('GET', '/api/sim/courses')[1]['courses']]
    res['course_dir_left'] = sorted(os.listdir(DATA / 'sim_courses'))

    # 3) JetFormer(試験インスタンス)を kill -9 → 孤児のブリッジが自分で止まる
    t0 = time.time()
    os.kill(srv.pid, signal.SIGKILL)
    srv.wait()

    def alive(p):
        try:
            return Path('/proc/%d/stat' % p).read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except OSError:
            return False
    while alive(pid) and time.time() - t0 < 15:
        time.sleep(0.1)
    res['orphan_bridge_exited'] = not alive(pid)
    res['orphan_exit_s'] = round(time.time() - t0, 2)
finally:
    if srv.poll() is None:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
    if 'bridge' in res and Path('/proc/%d' % res['bridge']['pid']).exists():
        try:
            os.kill(res['bridge']['pid'], signal.SIGTERM)
        except OSError:
            pass
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))
