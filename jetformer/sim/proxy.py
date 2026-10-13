"""ブリッジの API の中継(/sim/bridge/api/* → http://127.0.0.1:<ブリッジのポート>/api/*)

- 決まったメソッドとパスの組み合わせだけを通す(それ以外は 404 not_proxied)。本文の上限と読み取りの時間切れは組み合わせごと。
- Host を 127.0.0.1:<ポート> にし、渡すヘッダーは Content-Type・Content-Length・Accept だけ(Origin・Referer・Cookie は捨てる。
  ブリッジは LAN の Origin・Host を 403 にするため)。
- POST は Content-Length が必須(なければ 411)、chunked は 400。本文はためずに流し、Content-Length を明示する
  (明示すれば httpx は Transfer-Encoding: chunked を付けない。ブリッジは chunked を拒否する)。
- /api/jetformer/check・upload は本文の url をこの Jetson の JetFormer(http://127.0.0.1:<WEB_PORT>)に置き換える
  (ブリッジに LAN の別のアドレスへ GET・PUT させない)。
- エラーはブリッジと同じ形 {"error": {"code", "message"}}(シミュレーターの画面は error.message を出す)。
- 通信は httpx.AsyncClient だけ(ブリッジの /api/jetformer/* はブリッジから JetFormer を同期的に呼ぶので、
  JetFormer 側で同期の通信をイベントループの中で待つとデッドロックになる)。
"""

import asyncio
import json
import re
import threading

import httpx
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from .. import config

MiB = 1 << 20
# (メソッド, パスの正規表現, 本文の上限, 読み取りの時間切れ[秒], 種類, クエリを渡すか)
ROUTES = [
    (('GET', 'HEAD'), r'status', 0, 10, None, False),
    (('POST',), r'predict', 2 * MiB, 15, None, False),
    (('POST',), r'models/upload', 256 * MiB, 300, None, True),
    (('POST',), r'models/bundle', 300 * MiB, 300, None, True),
    (('POST',), r'models/activate', 2 * MiB, 180, None, False),
    (('POST',), r'datasets/(start|frame)', 2 * MiB, 30, None, False),
    (('POST',), r'datasets/finish', 16 * MiB, 300, 'finish', False),
    (('POST',), r'datasets/collection', 2 * MiB, 900, 'collection', False),
    (('GET', 'HEAD'), r'datasets/download/[0-9a-f]{32}', 0, 60, 'download', False),
    (('POST',), r'jetformer/check', 64 << 10, 120, 'jetformer', False),
    (('POST',), r'jetformer/upload', 64 << 10, 60, 'jetformer', False),
    (('GET',), r'jetformer/status/[A-Za-z0-9_-]{1,80}', 0, 15, None, False),
]
_COMPILED = [(m, re.compile(r'^(?:%s)$' % p), lim, t, kind, q) for m, p, lim, t, kind, q in ROUTES]
PASS_REQUEST = ('content-type', 'accept')
PASS_RESPONSE = ('content-type', 'content-length', 'content-disposition', 'cache-control', 'x-content-type-options')


def error(status, code, message):
    return JSONResponse({'error': {'code': code, 'message': message}}, status_code=status,
                        headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})


def match(method, path):
    for methods, rx, limit, timeout, kind, query in _COMPILED:
        # fullmatch: re の $ は末尾の改行の前でも一致するので、match だと 'status\n'(%0a)が通って httpx の InvalidURL(500)になる
        if rx.fullmatch(path):
            if method in methods:
                return {'limit': limit, 'timeout': timeout, 'kind': kind, 'query': query}
            return 'method'
    return None


class _BodyError(Exception):
    pass


class Proxy(object):
    """ブリッジへの中継。port() は running のときだけポート番号を返す関数"""

    def __init__(self, port, note_activity=None, web_port=None):
        self.port = port
        self.note_activity = note_activity or (lambda: None)
        self.web_port = web_port
        self.in_flight = {'finish': 0, 'collection': 0, 'download': 0}
        self._lock = threading.Lock()
        self._client = None
        self._loop = None

    def busy(self):
        """データの保存・まとめ・ダウンロードが中継中か(作成中の ZIP を消さないため)"""
        with self._lock:
            return sum(self.in_flight.values())

    def client(self):
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or self._client.is_closed:
            self._client = httpx.AsyncClient(limits=httpx.Limits(max_connections=16, max_keepalive_connections=0),
                                             trust_env=False, follow_redirects=False)
            self._loop = loop
        return self._client

    def close(self):
        """終了時。接続は持ち越さない(max_keepalive_connections=0)ので参照を捨てるだけ"""
        self._client = None
        self._loop = None

    def _track(self, kind, delta):
        if kind in self.in_flight:
            with self._lock:
                self.in_flight[kind] += delta

    async def handle(self, request, path):
        route = match(request.method, path)
        if route is None:
            return error(404, 'not_proxied', 'JetFormer はこの API を中継しません: %s %s' % (request.method, path[:80]))
        if route == 'method':
            return error(405, 'method_not_allowed', 'この API には使えないメソッドです: %s' % request.method)
        port = self.port()
        if not port:
            return error(503, 'bridge_not_running',
                         'シミュレーターのブリッジが起動していません。JetFormer の「シミュレーター」タブで起動してください。')
        headers = {'Host': '127.0.0.1:%d' % port}
        for h in PASS_REQUEST:
            if h in request.headers:
                headers[h] = request.headers[h]
        content = None
        if request.method == 'POST':
            if 'transfer-encoding' in request.headers:
                return error(400, 'chunked_not_supported', '本文は Content-Length を付けて送ってください(chunked は使えません)')
            cl = request.headers.get('content-length')
            if cl is None:
                return error(411, 'length_required', 'Content-Length が必要です')
            try:
                length = int(cl)
                if length < 0:
                    raise ValueError
            except ValueError:
                return error(400, 'invalid_length', 'Content-Length が不正です')
            if length > route['limit']:
                return error(413, 'body_too_large', '本文が大きすぎます(%d バイトまで)' % route['limit'])
            if route['kind'] == 'jetformer':
                body = b''
                async for chunk in request.stream():
                    body += chunk
                    if len(body) > route['limit']:
                        return error(413, 'body_too_large', '本文が大きすぎます')
                if len(body) != length:
                    return error(400, 'invalid_length', '本文の長さが Content-Length と違います')
                try:
                    value = json.loads(body.decode('utf-8'))
                except (UnicodeDecodeError, ValueError):
                    return error(400, 'invalid_json', 'JSON を読めません')
                if not isinstance(value, dict):
                    return error(400, 'invalid_json', 'JSON のオブジェクトを送ってください')
                value['url'] = 'http://127.0.0.1:%d' % (self.web_port or config.WEB_PORT)
                content = json.dumps(value).encode('utf-8')
                headers['content-type'] = 'application/json'
                headers['Content-Length'] = str(len(content))
            else:
                headers['Content-Length'] = str(length)
                content = self._stream(request, length)
        url = 'http://127.0.0.1:%d/api/%s' % (port, path)
        if route['query'] and request.url.query:
            url += '?' + request.url.query
        timeout = httpx.Timeout(connect=3.0, read=float(route['timeout']), write=300.0, pool=5.0)
        client = self.client()
        req = client.build_request(request.method, url, headers=headers, content=content, timeout=timeout)
        for h in ('accept-encoding', 'user-agent'):
            req.headers.pop(h, None)
        self.note_activity()
        kind = route['kind']
        self._track(kind, 1)
        try:
            upstream = await client.send(req, stream=True)
        except _BodyError as e:
            self._track(kind, -1)
            return error(400, 'invalid_body', str(e))
        except httpx.TimeoutException:
            self._track(kind, -1)
            return error(504, 'bridge_timeout', 'シミュレーターのブリッジが時間内に応答しませんでした')
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as e:
            self._track(kind, -1)
            if isinstance(e.__cause__, _BodyError) or isinstance(e.__context__, _BodyError):
                return error(400, 'invalid_body', str(e.__cause__ or e.__context__))
            return error(502, 'bridge_unreachable', 'シミュレーターのブリッジにつながりません(%s)' % type(e).__name__)
        except Exception:
            self._track(kind, -1)
            raise
        out_headers = {h: upstream.headers[h] for h in PASS_RESPONSE if h in upstream.headers}
        done = []

        async def close():
            if done:
                return
            done.append(1)
            try:
                await upstream.aclose()
            finally:
                self._track(kind, -1)
                self.note_activity()

        async def body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            except httpx.HTTPError:
                return
            finally:
                await close()             # 途中で切られても数を戻す
        return StreamingResponse(body(), status_code=upstream.status_code, headers=out_headers, background=BackgroundTask(close))

    @staticmethod
    async def _stream(request, length):
        """本文をためずに流す(実際の長さが宣言と違えば打ち切る)"""
        n = 0
        async for chunk in request.stream():
            if not chunk:
                continue
            n += len(chunk)
            if n > length:
                raise _BodyError('本文が Content-Length より長いです')
            yield chunk
        if n != length:
            raise _BodyError('本文が Content-Length より短いです')

    async def bridge_status(self, timeout=1.0):
        """ブリッジの /api/status(dict)。取れなければ None"""
        port = self.port()
        if not port:
            return None
        try:
            r = await self.client().get('http://127.0.0.1:%d/api/status' % port, headers={'Host': '127.0.0.1:%d' % port},
                                        timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except (httpx.HTTPError, ValueError):
            pass
        return None
