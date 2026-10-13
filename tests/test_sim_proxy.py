"""ブリッジの API の中継(jetformer/sim/proxy.py)のテスト。偽のブリッジ(本物と同じく Host・Origin・Content-Length・chunked を確かめる)を
127.0.0.1 の空きポートで動かし、TestClient から中継する"""

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jetformer.sim import proxy as proxy_mod
from jetformer.sim import web


class FakeBridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, value, extra=None):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Access-Control-Allow-Origin', 'http://evil')
        self.send_header('Connection', 'close')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _check(self):
        srv = self.server
        srv.seen.append({'method': self.command, 'path': self.path, 'headers': dict(self.headers)})
        host = (self.headers.get('Host') or '').lower()
        if host not in srv.allowed_hosts:
            self._send(403, {'error': {'code': 'host_not_allowed', 'message': 'host %s' % host}})
            return False
        origin = self.headers.get('Origin')
        if origin is not None and origin not in srv.allowed_origins:
            self._send(403, {'error': {'code': 'origin_not_allowed', 'message': 'origin'}})
            return False
        return True

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._check():
            return
        if self.path == '/api/status':
            if self.server.delay:
                time.sleep(self.server.delay)
            return self._send(200, {'policy_loaded': False, 'dataset_collection': {'recording': False}})
        if self.path.startswith('/api/datasets/download/'):
            n = self.server.download_bytes
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(n))
            self.send_header('Content-Disposition', 'attachment; filename="scene.zip"')
            self.send_header('Connection', 'close')
            self.end_headers()
            if self.command == 'GET':
                block = b'z' * 65536
                left = n
                while left:
                    k = min(left, len(block))
                    self.wfile.write(block[:k])
                    left -= k
            return
        if self.path.startswith('/api/jetformer/status/'):
            return self._send(200, {'id': self.path.rsplit('/', 1)[1], 'state': 'done'})
        return self._send(404, {'error': {'code': 'not_found', 'message': 'nope'}})

    def do_POST(self):
        if not self._check():
            return
        if 'Transfer-Encoding' in self.headers:
            return self._send(400, {'error': {'code': 'chunked_not_supported', 'message': 'chunked'}})
        if self.headers.get('Content-Length') is None:
            return self._send(411, {'error': {'code': 'length_required', 'message': 'length'}})
        n = int(self.headers['Content-Length'])
        h = hashlib.sha256()
        left = n
        while left:
            chunk = self.rfile.read(min(left, 1 << 20))
            if not chunk:
                break
            h.update(chunk)
            left -= len(chunk)
            if self.path.startswith('/api/jetformer/'):
                self.server.last_body = chunk
        self._send(200, {'path': self.path, 'bytes': n - left, 'sha256': h.hexdigest(),
                         'content_type': self.headers.get('Content-Type')})


@pytest.fixture
def fake_bridge():
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakeBridgeHandler)
    srv.daemon_threads = True
    port = srv.server_address[1]
    srv.allowed_hosts = {'127.0.0.1:%d' % port, 'localhost:%d' % port}
    srv.allowed_origins = {'null', 'http://127.0.0.1:%d' % port}
    srv.seen, srv.delay, srv.download_bytes, srv.last_body = [], 0, 3 << 20, b''
    t = threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


class Port(object):
    def __init__(self, port):
        self._port = port

    @property
    def port(self):
        return self._port

    def note_activity(self):
        self.noted = True

    def running(self):
        return bool(self._port)


@pytest.fixture
def client(fake_bridge):
    orig = web.bridge
    fb = Port(fake_bridge.server_address[1])
    web.init(bridge=fb, web_port=8080)
    app = FastAPI()
    app.include_router(web.router)
    c = TestClient(app)
    c.fb, c.srv = fb, fake_bridge
    yield c
    web.init(bridge=orig)


def test_status_rewrites_host_and_drops_origin(client):
    r = client.get('/sim/bridge/api/status', headers={'Origin': 'http://192.168.5.14:8080', 'Referer': 'http://x/', 'Cookie': 'a=b',
                                                       'Host': '192.168.5.14:8080'})
    assert r.status_code == 200 and r.json()['policy_loaded'] is False
    seen = client.srv.seen[-1]['headers']
    assert seen['Host'] == '127.0.0.1:%d' % client.srv.server_address[1]
    assert 'Origin' not in seen and 'Referer' not in seen and 'Cookie' not in seen
    assert 'access-control-allow-origin' not in r.headers and r.headers['cache-control'] == 'no-store'
    assert client.head('/sim/bridge/api/status').status_code == 200
    assert getattr(client.fb, 'noted', False)


def test_post_streams_with_content_length_not_chunked(client):
    body = b'\x01\x02' * (25 << 20)                       # 50 MB
    r = client.post('/sim/bridge/api/models/bundle?name=run1.zip&x=1', content=body, headers={'Content-Type': 'application/zip'})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j['bytes'] == len(body) and j['sha256'] == hashlib.sha256(body).hexdigest()
    assert j['path'] == '/api/models/bundle?name=run1.zip&x=1' and j['content_type'] == 'application/zip'
    h = client.srv.seen[-1]['headers']
    assert h['Content-Length'] == str(len(body)) and 'Transfer-Encoding' not in h


def test_query_dropped_where_not_needed(client):
    r = client.post('/sim/bridge/api/predict?secret=1', json={'a': 1})
    assert r.json()['path'] == '/api/predict'


def test_limits_and_framing(client):
    r = client.post('/sim/bridge/api/predict', content=b'x' * 10, headers={'Content-Length': str(3 << 20)})
    assert r.status_code == 413 and r.json()['error']['code'] == 'body_too_large'

    def chunks():
        yield b'{"a":'
        yield b'1}'
    r = client.post('/sim/bridge/api/predict', content=chunks())
    assert r.status_code in (400, 411)
    assert r.json()['error']['code'] in ('chunked_not_supported', 'length_required')


def test_not_proxied_and_method(client):
    for m, p in (('GET', 'models/list'), ('POST', 'shutdown'), ('GET', '%2e%2e/%2e%2e/etc/passwd'), ('GET', 'datasets/download/xyz')):
        r = client.request(m, '/sim/bridge/api/' + p)
        assert r.status_code == 404 and r.json()['error']['code'] == 'not_proxied'
    r = client.post('/sim/bridge/api/status', content=b'')
    assert r.status_code == 405


def test_jetformer_url_forced_to_this_jetson(client):
    r = client.post('/sim/bridge/api/jetformer/check', json={'url': 'http://192.168.1.50:9999', 'session_id': 'a' * 32})
    assert r.status_code == 200
    sent = json.loads(client.srv.last_body)
    assert sent == {'url': 'http://127.0.0.1:8080', 'session_id': 'a' * 32}
    assert r.json()['bytes'] == len(client.srv.last_body)
    r = client.post('/sim/bridge/api/jetformer/upload', content=b'[1,2]', headers={'Content-Type': 'application/json'})
    assert r.status_code == 400
    r = client.get('/sim/bridge/api/jetformer/status/abc123')
    assert r.json() == {'id': 'abc123', 'state': 'done'}


def test_download_streams_with_headers(client):
    sid = 'f' * 32
    with client.stream('GET', '/sim/bridge/api/datasets/download/' + sid) as r:
        assert r.status_code == 200
        assert r.headers['content-disposition'] == 'attachment; filename="scene.zip"'
        assert r.headers['content-length'] == str(3 << 20)
        n = sum(len(c) for c in r.iter_bytes())
    assert n == 3 << 20
    assert web.proxy.busy() == 0                    # 終わったら数が戻る


def test_bridge_not_running(client):
    client.fb._port = None
    r = client.get('/sim/bridge/api/status')
    assert r.status_code == 503 and r.json()['error']['code'] == 'bridge_not_running'
    assert '起動' in r.json()['error']['message']


def test_unreachable_and_timeout(client, monkeypatch):
    client.fb._port = 9                               # 何も待ち受けていない
    r = client.get('/sim/bridge/api/status')
    assert r.status_code == 502 and r.json()['error']['code'] == 'bridge_unreachable'
    client.fb._port = client.srv.server_address[1]
    client.srv.delay = 1.5
    patched = [(m, rx, lim, (0.3 if rx.pattern == '^(?:status)$' else t), k, q) for m, rx, lim, t, k, q in proxy_mod._COMPILED]
    monkeypatch.setattr(proxy_mod, '_COMPILED', patched)
    r = client.get('/sim/bridge/api/status')
    assert r.status_code == 504 and r.json()['error']['code'] == 'bridge_timeout'
    assert web.proxy.busy() == 0


def test_route_table_matches_design():
    assert proxy_mod.match('POST', 'models/bundle')['limit'] == 300 << 20
    assert proxy_mod.match('POST', 'datasets/collection')['timeout'] == 900
    assert proxy_mod.match('POST', 'datasets/finish')['limit'] == 16 << 20
    assert proxy_mod.match('GET', 'datasets/download/' + 'a' * 32)['kind'] == 'download'
    assert proxy_mod.match('GET', 'datasets/download/' + 'A' * 32) is None
    assert proxy_mod.match('DELETE', 'status') == 'method'


def test_trailing_newline_is_not_proxied(client):
    """re の $ は末尾の改行の前でも一致するので、match だと 'status\\n' が通り、httpx の InvalidURL(500)になりうる(review で fullmatch に)。
    Starlette の経路は末尾の %0a を落として 'status' として渡すので、今の経路からは届かない(多層防御)。どちらでも 500 にならず、
    ブリッジに改行入りのパスが届かないことを確かめる"""
    for m, p in (('GET', 'status\n'), ('POST', 'predict\n'), ('GET', 'datasets/download/' + 'a' * 32 + '\n'),
                 ('GET', 'jetformer/status/abc\n')):
        assert proxy_mod.match(m, p) is None
    for p in ('status%0a', 'status%0A', 'status%0d%0a', 'datasets/download/' + 'a' * 32 + '%0a', 'status%0aX'):
        r = client.get('/sim/bridge/api/' + p)
        assert r.status_code in (200, 404), (p, r.status_code)
    assert all('\n' not in x['path'] and '%0' not in x['path'].lower() for x in client.srv.seen)
    assert web.proxy.busy() == 0
