"""シミュレーターの経路(jetformer/sim/web.py)のテスト。FastAPI に Router だけを付けて試す(サーバー全体は立てない)"""

import io
import json
import os
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import _sim_testlib as L
from jetformer.sim import install, paths, web


class FakeJobs(object):
    def __init__(self):
        self.jobs = []

    def submit(self, kind, title, fn):
        job = {'id': 'j%d' % len(self.jobs), 'kind': kind, 'title': title, 'state': 'running'}
        try:
            job['result'] = fn(lambda p, m='': None)
            job['state'] = 'done'
        except Exception as e:
            job['state'], job['error'] = 'error', str(e)
        self.jobs.append(job)
        return job


class FakeBridge(object):
    def __init__(self, port=None, state='stopped'):
        self._port, self.state = port, state
        self.started = 0

    @property
    def port(self):
        return self._port if self.state == 'running' else None

    def running(self):
        return self.state in ('starting', 'running', 'stopping')

    def note_activity(self):
        pass

    def status(self):
        return {'state': self.state, 'pid': None, 'port': self.port, 'started_at': None, 'restarts': 0, 'restarted_at': None,
                'last_exit': None, 'last_error': None, 'rss_mb': None}

    def start(self):
        self.started += 1
        self.state = 'starting'
        return self.status()

    def stop(self, timeout=5.0):
        self.state = 'stopped'
        return self.status()

    def logs(self, n=200):
        return {'lines': ['a', 'b'][-n:] if n else [], 'suppressed': {'predict': 3, 'frame': 4}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    L.install_fake_release(tmp_path, monkeypatch)
    jobs, fb = FakeJobs(), FakeBridge()
    orig = web.bridge
    web.init(jobs=jobs, busy=lambda: {'training': False, 'driving': False, 'recording': False}, bridge=fb)
    app = FastAPI()
    app.include_router(web.router)
    c = TestClient(app)
    c.jobs, c.fb, c.tmp = jobs, fb, tmp_path
    yield c
    web.init(bridge=orig)


def upload_course(c, files, name='c.zip', replace=0):
    return c.put('/api/sim/courses/import?filename=%s&replace=%d' % (name, replace), content=L.course_zip(files),
                 headers={'Content-Type': 'application/zip'})


def data_course(c, name='自宅コース', wv='home-1'):
    r = upload_course(c, {'course.json': {'format': 'jetformer-sim-course/v1', 'name': name, 'world_version': wv},
                          'physics-world.json': L.simple_world(), 'assets/floor.jpg': L.tiny_jpg()})
    assert r.status_code == 200, r.text
    return r.json()['course']


def test_redirects(client):
    r = client.get('/sim', follow_redirects=False)
    assert r.status_code == 307 and r.headers['location'] == '/sim/'
    r = client.get('/sim/?model=run1', follow_redirects=False)
    assert r.status_code == 307 and r.headers['location'] == '/sim/c/default/?model=run1'
    r = client.get('/sim/c/indoor-course-eval-2-shortcut', follow_redirects=False)
    assert r.status_code == 308 and r.headers['location'] == '/sim/c/indoor-course-eval-2-shortcut/'
    cid = data_course(client)['id']
    client.post('/api/sim/courses/%s/default' % cid)
    assert client.get('/sim/', follow_redirects=False).headers['location'] == '/sim/c/%s/' % cid
    assert client.get('/sim/c/BAD', follow_redirects=False).status_code == 404


def test_index_injects_three_lines(client):
    r = client.get('/sim/c/default/')
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/html')
    html = r.text
    head = html.split('<meta charset="utf-8">', 1)[1]
    assert head.lstrip().startswith('<script src="./_jf/config.js"></script>')
    assert '/sim/static/sim-shim.js?v=' in head and '/sim/static/sim-shim.css?v=' in head
    assert head.index('sim-shim.js') < head.index('<script type="importmap">') < head.index('src="./app.js"')
    assert r.headers['cache-control'] == 'no-cache' and r.headers['x-content-type-options'] == 'nosniff'
    assert client.head('/sim/c/default/').status_code == 200
    assert client.get('/sim/c/default/index.html').text == html


def test_index_shape_change_is_500(client):
    p = install.active_release().dist / 'index.html'
    os.chmod(p, 0o644)
    p.write_text(L.INDEX_HTML.replace('<script type="importmap">', '<script>'))
    r = client.get('/sim/c/default/')
    assert r.status_code == 500 and '版が変わった' in r.json()['detail']


def test_config_js_escapes_user_strings(client):
    cid = data_course(client, name='</script><img>&', wv='esc-1')['id']
    r = client.get('/sim/c/%s/_jf/config.js' % cid)
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/javascript')
    body = r.text
    assert '<' not in body and '>' not in body and '&' not in body
    cfg = json.loads(body[len('window.JETFORMER_SIM = Object.freeze('):-len(');\n')])
    assert cfg['course']['name'] == '</script><img>&' and cfg['course']['renderer'] == 'generic'
    assert cfg['course']['features'] == {'teachers': ['slow', 'speed'], 'collection_plan': False}
    assert cfg['bridge'] == '/sim/bridge' and cfg['release'] == install.active_id()
    d = json.loads(client.get('/sim/c/default/_jf/config.js').text.split('freeze(', 1)[1][:-3])
    assert d['course']['features']['collection_plan'] is True


def test_file_resolution_per_course(client):
    rel = install.active_release()
    assert client.get('/sim/c/default/scene.js').content == L.SCENE_JS
    assert client.get('/sim/c/default/world.js').content == L.WORLD_JS
    assert client.get('/sim/c/indoor-course-eval-2-shortcut/scene.js').content == L.V2_SCENE_JS
    assert client.get('/sim/c/indoor-course-eval-2-shortcut/app.js').content == (rel.dist / 'app.js').read_bytes()
    cid = data_course(client)['id']
    r = client.get('/sim/c/%s/scene.js' % cid)
    assert r.content == (paths.STATIC / 'course-scene.js').read_bytes() and r.headers['content-type'] == 'text/javascript; charset=utf-8'
    assert client.get('/sim/c/%s/world.js' % cid).content == (paths.STATIC / 'course-world.js').read_bytes()
    r = client.get('/sim/c/%s/_jf/course-data.js' % cid)
    assert r.status_code == 200 and r.text.startswith('export default {')
    assert client.get('/sim/c/default/_jf/course-data.js').status_code == 404
    assert client.get('/sim/c/%s/assets/floor.jpg' % cid).content[:3] == b'\xff\xd8\xff'
    assert client.get('/sim/c/%s/assets/turf.jpg' % cid).content == (rel.dist / 'assets' / 'turf.jpg').read_bytes()
    assert client.get('/sim/c/default/vendor/three.module.js').status_code == 200
    assert client.get('/sim/c/default/diagnosis/index.html').headers['content-type'].startswith('text/html')
    r = client.get('/sim/c/default/assets/carpet.jpg')
    assert r.headers['content-type'] == 'image/jpeg' and r.headers['x-content-type-options'] == 'nosniff'
    assert client.get('/sim/c/default/styles.css').headers['content-type'].startswith('text/css')


@pytest.mark.parametrize('path', ['%2e%2e/bridge.py', '..%2fbridge.py', 'a%5cb.js', 'x%00.js', '_jf/other.js', 'nothing.js',
                                  'assets/../../bridge.py', 'vendor//three.module.js', 'c:x.js', '%2e', 'assets/'])
def test_bad_paths_404(client, path):
    assert client.get('/sim/c/default/' + path, follow_redirects=False).status_code == 404


def test_unknown_course_and_not_installed(client, monkeypatch):
    assert client.get('/sim/c/nothing/').status_code == 404
    monkeypatch.setattr(install, 'active_release', lambda: None)
    r = client.get('/sim/c/default/')
    assert r.status_code == 503 and '導入' in r.json()['detail']


def test_symlink_in_dist_is_refused(client):
    rel = install.active_release()
    os.chmod(rel.dist, 0o755)
    os.symlink('/etc/hostname', rel.dist / 'leak.js')
    assert client.get('/sim/c/default/leak.js').status_code == 404


def test_etag_304(client):
    r = client.get('/sim/c/default/app.js')
    etag = r.headers['etag']
    r2 = client.get('/sim/c/default/app.js', headers={'If-None-Match': etag})
    assert r2.status_code == 304 and r2.content == b''
    r3 = client.get('/sim/c/default/_jf/config.js')
    assert client.get('/sim/c/default/_jf/config.js', headers={'If-None-Match': r3.headers['etag']}).status_code == 304


def test_static_files(client):
    for n in ('sim-shim.js', 'course-scene.js', 'course-world.js'):
        r = client.get('/sim/static/' + n)
        assert r.status_code == 200 and r.headers['content-type'] == 'text/javascript; charset=utf-8'
    assert client.get('/sim/static/sim-shim.css').headers['content-type'].startswith('text/css')
    assert client.get('/sim/static/../web.py').status_code == 404
    assert client.get('/sim/static/web.py').status_code == 404


def test_status_api(client):
    s = client.get('/api/sim/status').json()
    assert s['release']['installed'] and s['release']['id'] == install.active_id()
    assert s['release']['default_world_version'] == 'indoor-course-eval-3-turf-board-removed'
    assert s['inbox'][0]['installed'] and s['inbox'][0]['kind'] == 'simulator'
    assert s['env']['ready'] is False and s['bridge']['state'] == 'stopped' and s['bridge']['policy_loaded'] is False
    assert s['default_course'] == 'default' and s['busy'] == {'training': False, 'driving': False, 'recording': False}


def test_install_and_setup_api(client, monkeypatch):
    from jetformer.sim import env
    calls = []
    monkeypatch.setattr(env, 'setup', lambda force=False, progress=None: calls.append(force) or {'ready': True})
    files = L.release_files()
    files['README.md'] = b'v2\n'
    L.make_release_zip(client.tmp / 'inbox' / 'new.zip', files=files, manifest=L.manifest_for(files, created_at='2026-12-01T00:00:00+09:00'))
    r = client.post('/api/sim/install', json={'file': 'new.zip'})
    assert r.status_code == 200 and r.json()['kind'] == 'sim_install' and r.json()['state'] == 'done'
    assert r.json()['result']['release_id'].startswith('20261201-000000-')
    assert client.post('/api/sim/install', json={'file': '../new.zip'}).status_code == 400
    assert client.post('/api/sim/install', json={'file': 'none.zip'}).status_code == 404
    rels = client.get('/api/sim/status').json()['releases']
    assert len(rels) == 2
    old = [x['id'] for x in rels if not x['active']][0]
    client.fb.state = 'running'
    assert client.post('/api/sim/releases/%s/activate' % old).status_code == 409
    assert client.post('/api/sim/setup', json={}).status_code == 409
    client.fb.state = 'stopped'
    assert client.post('/api/sim/releases/%s/activate' % old).json()['active'] == old
    r = client.post('/api/sim/setup', json={'force': True})
    assert r.json()['kind'] == 'sim_setup' and r.json()['state'] == 'done' and calls == [True]   # pip は動かさない


def test_bridge_api(client):
    assert client.post('/api/sim/bridge/start').json()['state'] == 'starting'
    assert client.fb.started == 1
    assert client.post('/api/sim/bridge/stop', json={}).json()['state'] == 'stopped'
    r = client.get('/api/sim/bridge/log?lines=1').json()
    assert r == {'lines': ['b'], 'suppressed': {'predict': 3, 'frame': 4}}
    assert client.get('/api/sim/bridge/log?lines=999').status_code == 422


def test_bridge_start_error_is_409(client, monkeypatch):
    from jetformer.sim import bridge_proc

    def boom():
        raise bridge_proc.BridgeError('先に Python 環境を準備してください')
    monkeypatch.setattr(client.fb, 'start', boom)
    r = client.post('/api/sim/bridge/start')
    assert r.status_code == 409 and '準備' in r.json()['detail']


def test_courses_api(client):
    r = client.get('/api/sim/courses').json()
    assert r['default'] == 'default' and [c['id'] for c in r['courses']] == ['default', 'indoor-course-eval-2-shortcut']
    assert r['courses'][0]['open_url'] == '/sim/c/default/'
    c = data_course(client)
    assert client.get('/api/sim/courses/%s' % c['id']).json()['kind'] == 'data'
    bad = upload_course(client, {'course.json': {'format': 'jetformer-sim-course/v1', 'name': 'x', 'world_version': 'bad v'},
                                 'physics-world.json': dict(L.simple_world(), route=[[0, 0]])})
    assert bad.status_code == 400 and len(bad.json()['errors']) >= 2 and bad.json()['detail']
    dup = upload_course(client, {'course.json': {'format': 'jetformer-sim-course/v1', 'name': 'x', 'world_version': 'home-1'},
                                 'physics-world.json': dict(L.simple_world(), spawn={'x': 0, 'z': .75, 'yaw': 0})})
    assert dup.status_code == 409
    big = client.put('/api/sim/courses/import?filename=x.zip', content=b'x' * 10, headers={'Content-Length': str(70 << 20)})
    assert big.status_code == 413
    z = client.get('/api/sim/courses/%s/package.zip' % c['id'])
    assert z.status_code == 200 and z.headers['content-type'] == 'application/zip'
    assert 'home-1/course.json' in zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert client.post('/api/sim/courses/nothing/default').status_code == 404
    assert client.delete('/api/sim/courses/default').status_code == 409
    assert client.delete('/api/sim/courses/%s' % c['id']).json() == {'ok': True}
    assert not list(paths.courses_dir().glob('.upload-*'))
