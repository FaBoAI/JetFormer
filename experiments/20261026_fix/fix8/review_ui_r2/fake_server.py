"""作り物のサーバー(反証 8-2 回目・係 ui。8-1 の係の fake_server.py の写しに running・iframe の高さ・前の style.css を足した。読むだけ): 本物の index.html・app.js・style.css をそのまま返し(index は本物の server.py と同じに ?v= を付ける)、
走行タブの GET の答えは本物の jetformer-web から GET で取った JSON(status_live.json・models_live.json・bench_live.json。今日を 11/3 に)。
POST は 405(何も変えない)。ポートは引数(既定 18797)・127.0.0.1 だけ。本物の基板・jetformer-web には触れない"""
import http.server
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = Path('/home/jetson/jetformer/jetformer/web/static')
BENCH = json.loads((HERE / 'bench_live.json').read_text(encoding='utf-8'))
BENCH['today'] = '2026-11-03'
DOWN = [False]
RUN = [False]
BEFORE_CSS = Path('/home/jetson/jetformer/tests/fixtures/fix8_ui_r1_before/style.css')
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
            if 'css=before' in self.path:
                html = html.replace('/static/style.css"', '/before/style.css"')
            for name in ('app.js', 'style.css'):
                html = html.replace('/static/%s"' % name, '/static/%s?v=%d"' % (name, int((STATIC / name).stat().st_mtime)))
            return self.send(200, html, 'text/html; charset=utf-8')
        if path == '/phone':        # 幅 W の iframe(スマホの画面の幅。media query は iframe の幅で決まる)
            import urllib.parse
            q = urllib.parse.parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
            w = int(q.get('w', ['420'])[0]); h = int(q.get('h', ['860'])[0]); css = q.get('css', [''])[0]
            return self.send(200, '<!doctype html><body style="margin:0"><iframe id="f" src="/%s" style="width:%dpx;height:%dpx;border:0"></iframe></body>' % ('?css=' + css if css else '', w, h),
                             'text/html; charset=utf-8')
        if path == '/before/style.css':
            return self.send(200, BEFORE_CSS.read_bytes(), 'text/css; charset=utf-8')
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
            if 'running' in q:
                RUN[0] = bool(q['running'])
            return self.send(200, json.dumps({'ok': True}), 'application/json')
        if DOWN[0] and path.startswith('/api/'):
            self.close_connection = True
            return self.send(503, json.dumps({'detail': 'fake: 落ちている'}), 'application/json')
        if path == '/api/drive/status' and RUN[0]:
            st = json.loads(json.dumps(J[path]))
            st.update({'running': True, 'phase': 'driving', 'controller': True, 'run': st.get('last', {}).get('run'), 'mode_id': 5, 'ai_no': 3,
                       'now': {'select': True, 'hold': False, 'out': [1500, 1580], 'raw': [1500, 1500], 'assist': None, 'ttc': None},
                       'stats': {'fps': 30, 'latency_ms': 28, 'latency_ms_max': 40, 'infer_ms': 9, 'frames': 300, 'skipped': 0, 'holds': 0}})
            return self.send(200, json.dumps(st, ensure_ascii=False), 'application/json')
        if path in J:
            return self.send(200, json.dumps(J[path], ensure_ascii=False), 'application/json')
        return self.send(404, json.dumps({'detail': 'fake: ' + path}), 'application/json')

    def do_POST(self):
        return self.send(405, json.dumps({'detail': 'fake: POST しない'}), 'application/json')


if __name__ == '__main__':
    http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 18797), H).serve_forever()
