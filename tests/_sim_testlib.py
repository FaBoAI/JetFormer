"""シミュレーター統合のテスト用の材料(偽のシミュレーター本体の ZIP・偽のブリッジ)。pytest は集めない(test_ で始まらない)"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

FIX = Path(__file__).resolve().parent / 'fixtures' / 'sim'
CREATED_AT = '2026-10-13T14:40:13.846207+09:00'

# 本物の dist/index.html と同じ head の形(meta charset・importmap・./app.js がちょうど 1 つ)
INDEX_HTML = ('<!doctype html>\n<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
              '<title>Course Lab — AI走行評価</title><link rel="stylesheet" href="./styles.css" />\n'
              '<script type="importmap">{"imports":{"three":"./vendor/three.module.js","three/addons/":"./vendor/addons/"}}</script>\n'
              '</head><body>\n<span class="view-label">COURSE / 01</span>\n<script type="module" src="./app.js"></script>\n</body></html>\n')

# 偽のブリッジ(標準ライブラリだけ。本物と同じく 127.0.0.1 に bind し、Host を確かめる)
STUB_BRIDGE = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PolicyRunner:
    def __init__(self):
        self.name = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, value):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        print('[bridge] %s %s -> %d' % (self.command, self.path.split('?')[0], code), file=sys.stderr, flush=True)

    def do_GET(self):
        if (self.headers.get('Host') or '').lower() not in self.server.allowed_hosts:
            return self._send(403, {'error': {'code': 'host_not_allowed', 'message': 'host'}})
        if self.path == '/api/status':
            return self._send(200, {'policy_loaded': False, 'name': None, 'model_upload': {'available': True},
                                    'dataset_collection': {'available': True, 'recording': False},
                                    'runtime_root': str(self.server.runtime_root)})
        return self._send(404, {'error': {'code': 'not_found', 'message': 'x'}})


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, static_root, runner, runtime_root=None):
        self.runtime_root = runtime_root
        self.static_root = static_root
        super().__init__(('127.0.0.1', port), Handler)
        p = self.server_address[1]
        self.allowed_hosts = {'127.0.0.1:%d' % p, 'localhost:%d' % p}


def main(arguments=None):
    if arguments and '--self-test' in arguments:
        print('test_one ... ok', file=sys.stderr)
        print('Ran 13 tests in 0.1s', file=sys.stderr)
        print('OK', file=sys.stderr)
        return 0
    return 2
'''

SCENE_JS = b"// fake scene\nexport function buildCourse(scene) { return {physics: {}}; }\n"
WORLD_JS = b"export const WORLD_VERSION = 'indoor-course-eval-3-turf-board-removed';\nexport function makeWorld(p) { return p; }\n"
V2_SCENE_JS = b"// fake v2 scene\nexport function buildCourse(scene) { return {physics: {}}; }\n"
V2_WORLD_JS = b"export const WORLD_VERSION = 'indoor-course-eval-2-shortcut';\nexport function makeWorld(p) { return p; }\n"


def tiny_jpg(w=8, h=8, color=(40, 120, 200)):
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def tiny_png(w=8, h=8):
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = 200
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def release_files():
    """偽のシミュレーター本体のファイル {相対パス: bytes}(COMPLETE-MANIFEST.json に載せるもの)"""
    v2_world = (FIX / 'v2-physics-world.json').read_bytes()
    v2_manifest = json.loads((FIX / 'v2-manifest.json').read_text(encoding='utf-8'))
    v2_manifest['files'] = {'scene.js': hashlib.sha256(V2_SCENE_JS).hexdigest(), 'world.js': hashlib.sha256(V2_WORLD_JS).hexdigest(),
                            'physics-world.json': hashlib.sha256(v2_world).hexdigest()}
    snap = 'qa/course-snapshots/indoor-course-eval-2-shortcut/'
    return {
        'bridge.py': STUB_BRIDGE.encode(),
        'dataset_recorder.py': b'# stub\n',
        'jetformer_policy.py': b'# stub\n',
        'jetformer_upload.py': b'# stub\n',
        'calibration.example.json': b'{}\n',
        'requirements-model.txt': b'# pinned\nnumpy==2.2.6\nopencv-python-headless==4.13.0.92\nonnxruntime==1.31.0\n',
        'README.md': b'# readme\n',
        'dist/index.html': INDEX_HTML.encode(),
        'dist/app.js': b"import { buildCourse } from './scene.js';\n",
        'dist/scene.js': SCENE_JS,
        'dist/world.js': WORLD_JS,
        'dist/sim-core.js': b'// sim-core\n',
        'dist/styles.css': b'body{margin:0}\n',
        'dist/vendor/three.module.js': b'// three\n',
        'dist/assets/carpet.jpg': tiny_jpg(),
        'dist/assets/turf.jpg': tiny_jpg(color=(30, 160, 60)),
        'dist/diagnosis/index.html': b'<html></html>\n',
        'qa/combine_scene_datasets.py': b'# stub\n',
        'training/speed_physics.py': b'# stub\n',
        snap + 'manifest.json': json.dumps(v2_manifest, indent=2).encode(),
        snap + 'physics-world.json': v2_world,
        snap + 'scene.js': V2_SCENE_JS,
        snap + 'world.js': V2_WORLD_JS,
        'dataset-downloads/big.zip': b'PK-not-really-a-zip' * 10,
        'runtime/datasets/x/evaluation-report.json': b'{}',
    }


def manifest_for(files, created_at=CREATED_AT, schema='course-simulator-complete-v1'):
    return {'schema': schema, 'created_at': created_at,
            'files': [{'path': p, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for p, b in sorted(files.items())]}


def make_release_zip(path, files=None, manifest=None, top='course-simulator', extra=None, drop=(), mutate=None, entries=None):
    """偽のシミュレーター本体の ZIP を作る。extra = manifest に載せないファイル、drop = ZIP から抜くファイル、
    mutate = ZIP に入れる中身だけを変えるファイル(manifest は元のまま)"""
    files = dict(release_files() if files is None else files)
    manifest = manifest or manifest_for(files)
    raw_manifest = json.dumps(manifest, indent=1).encode()
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('%s/' % top, b'')
        z.writestr('%s/COMPLETE-MANIFEST.json' % top, raw_manifest)
        for p, b in sorted(files.items()):
            if p in drop:
                continue
            z.writestr('%s/%s' % (top, p), (mutate or {}).get(p, b))
        for p, b in (extra or {}).items():
            z.writestr('%s/%s' % (top, p), b)
        for info, b in (entries or []):
            z.writestr(info, b)
    return hashlib.sha256(raw_manifest).hexdigest()


def course_zip(files, top=None):
    """コースの ZIP(bytes)"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            if isinstance(data, (dict, list)):
                data = json.dumps(data)
            if isinstance(data, str):
                data = data.encode('utf-8')
            z.writestr(('%s/%s' % (top, name)) if top else name, data)
    return buf.getvalue()


def v3_world():
    return json.loads((FIX / 'v3-physics-world.json').read_text(encoding='utf-8'))


def v2_world():
    return json.loads((FIX / 'v2-physics-world.json').read_text(encoding='utf-8'))


def simple_world():
    """小さな四角いコース(4 m × 3 m)。physics-world.json の形"""
    rails = [((-2, -1.5), (2, -1.5)), ((2, -1.5), (2, 1.5)), ((2, 1.5), (-2, 1.5)), ((-2, 1.5), (-2, -1.5)),
             ((-1, 0), (1, 0))]
    return {'segments': [{'a': list(a), 'b': list(b), 'radius': .0095, 'height': .119, 'kind': 'rail'} for a, b in rails],
            'disks': [{'x': -2, 'z': -1.5, 'radius': .11, 'height': .03, 'kind': 'post-foot'},
                      {'x': -2, 'z': -1.5, 'radius': .018, 'height': .155, 'kind': 'post'}],
            'surfaces': [{'id': 'carpet', 'x': 0, 'z': 0, 'w': 4.5, 'd': 3.5, 'mu': .8},
                         {'id': 'shortcut', 'x': 0, 'z': -.75, 'w': 1.0, 'd': .6, 'mu': .7}],
            'bounds': {'minX': -2.5, 'maxX': 2.5, 'minZ': -2, 'maxZ': 2},
            'route': [[-1.5, .75], [0, .75], [1.5, .75], [1.5, -.75], [0, -.75], [-1.5, -.75]],
            'requiredCrossings': [{'id': 'shortcut', 'surface_id': 'shortcut', 'minX': -.5, 'maxX': .5, 'z': -.75, 'direction': -1}],
            'spawn': {'x': -1.5, 'z': .75, 'yaw': 0}}


def install_fake_release(tmp_path, monkeypatch, name='course-simulator-complete.zip', **kw):
    """config.DATA_DIR と inbox を tmp に向けて、偽の本体を導入する。戻り値 install_release の結果"""
    from jetformer import config
    from jetformer.sim import install
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path / 'data' / 'datasets')
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path / 'data' / 'raw')
    inbox = tmp_path / 'inbox'
    inbox.mkdir(exist_ok=True)
    monkeypatch.setenv('JETFORMER_SIM_INBOX', str(inbox))
    make_release_zip(inbox / name, **kw)
    return install.install_release(name)
