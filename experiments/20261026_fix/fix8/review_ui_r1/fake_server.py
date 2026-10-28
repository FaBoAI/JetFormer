"""作り物のサーバー(この係。読むだけ): 本物の index.html・app.js・style.css をそのまま返し(index は本物の server.py と同じに ?v= を付ける)、
走行タブの GET の答えは本物の jetformer-web から GET で取った JSON(status_live.json・models_live.json・bench_live.json。今日を 11/3 に)。
POST は 405(何も変えない)。ポート 18771・127.0.0.1 だけ。本物の基板・jetformer-web には触れない"""
import http.server
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = Path('/home/jetson/jetformer/jetformer/web/static')
BENCH = json.loads((HERE / 'bench_live.json').read_text(encoding='utf-8'))
BENCH['today'] = '2026-11-03'
DOWN = [False]
J = {'/api/drive/status': json.loads((HERE / 'status_live.json').read_text(encoding='utf-8')),
     '/api/drive/models': json.loads((HERE / 'models_live.json').read_text(encoding='utf-8')),
     '/api/bench/candidates': BENCH}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            html = (STATIC / 'index.html').read_text(encoding='utf-8')
            for name in ('app.js', 'style.css'):
                html = html.replace('/static/%s"' % name, '/static/%s?v=%d"' % (name, int((STATIC / name).stat().st_mtime)))
            return self.send(200, html, 'text/html; charset=utf-8')
        if path == '/phone':        # 幅 W の iframe(スマホの画面の幅。media query は iframe の幅で決まる)
            w = int((self.path.split('w=')[1] if 'w=' in self.path else '420').split('&')[0])
            return self.send(200, '<!doctype html><body style="margin:0"><iframe id="f" src="/" style="width:%dpx;height:860px;border:0"></iframe></body>' % w,
                             'text/html; charset=utf-8')
        if path.startswith('/static/'):
            p = STATIC / path[len('/static/'):]
            if p.is_file() and p.parent == STATIC:
                ct = {'.js': 'application/javascript', '.css': 'text/css', '.html': 'text/html'}.get(p.suffix, 'application/octet-stream')
                return self.send(200, p.read_bytes(), ct + '; charset=utf-8')
        if path == '/fake/set':      # この係の作り物だけ: status の params・boot_at・落ちる を変える(?j=JSON)
            import urllib.parse
            q = json.loads(urllib.parse.unquote(self.path.split('j=', 1)[1]))
            st = J['/api/drive/status']
            st['params'].update(q.get('params') or {})
            if 'boot_at' in q:
                st['boot_at'] = q['boot_at']
            if 'down' in q:
                DOWN[0] = bool(q['down'])
            return self.send(200, json.dumps({'ok': True}), 'application/json')
        if DOWN[0] and path.startswith('/api/'):
            self.close_connection = True
            return self.send(503, json.dumps({'detail': 'fake: 落ちている'}), 'application/json')
        if path in J:
            return self.send(200, json.dumps(J[path], ensure_ascii=False), 'application/json')
        return self.send(404, json.dumps({'detail': 'fake: ' + path}), 'application/json')

    def do_POST(self):
        return self.send(405, json.dumps({'detail': 'fake: POST しない'}), 'application/json')


if __name__ == '__main__':
    http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 18771), H).serve_forever()
