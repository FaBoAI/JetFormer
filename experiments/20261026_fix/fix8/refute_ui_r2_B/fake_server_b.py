"""打ち消しの票 B(反証 8-2 回目・係 ui の float-stop-half-width-under-toast)の作り物のサーバー。
review_ui_r2/fake_server.py の写しに「POST /api/drive/stop を覚えて 200 を返す」だけを足した(ほかの POST は 405)。
本物の index.html・app.js・style.css をそのまま返す。走行タブの GET の答えは review_ui_r2 の *_live.json(読むだけ)。
127.0.0.1 だけ。本物の jetformer-web・基板には触れない。
"""
import http.server
import json
import sys
import time
import urllib.parse
from pathlib import Path

R2 = Path('/home/jetson/jetformer/experiments/20261026_fix/fix8/review_ui_r2')
STATIC = Path('/home/jetson/jetformer/jetformer/web/static')
BENCH = json.loads((R2 / 'bench_live.json').read_text(encoding='utf-8'))
BENCH['today'] = '2026-11-03'
RUN = [False]
POSTS = []
J = {'/api/drive/status': json.loads((R2 / 'status_live.json').read_text(encoding='utf-8')),
     '/api/drive/models': json.loads((R2 / 'models_live.json').read_text(encoding='utf-8')),
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
        if path == '/phone':        # 幅 W・高さ H の iframe(スマホの画面。media query は iframe の幅で決まる)
            q = urllib.parse.parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
            w = int(q.get('w', ['420'])[0]); h = int(q.get('h', ['860'])[0])
            return self.send(200, '<!doctype html><body style="margin:0"><iframe id="f" src="/" style="width:%dpx;height:%dpx;border:0;display:block"></iframe></body>' % (w, h),
                             'text/html; charset=utf-8')
        if path.startswith('/static/'):
            p = STATIC / path[len('/static/'):]
            if p.is_file() and p.parent == STATIC:
                ct = {'.js': 'application/javascript', '.css': 'text/css', '.html': 'text/html'}.get(p.suffix, 'application/octet-stream')
                return self.send(200, p.read_bytes(), ct + '; charset=utf-8')
        if path == '/fake/set':
            q = json.loads(urllib.parse.unquote(self.path.split('j=', 1)[1]))
            if 'running' in q:
                RUN[0] = bool(q['running'])
            if q.get('clear'):
                POSTS.clear()
            return self.send(200, json.dumps({'ok': True}), 'application/json')
        if path == '/fake/posts':
            return self.send(200, json.dumps(POSTS, ensure_ascii=False), 'application/json')
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
        path = self.path.split('?')[0]
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n).decode('utf-8', 'replace') if n else ''
        if path == '/api/drive/stop':
            POSTS.append({'t': time.time(), 'path': path, 'body': body})
            return self.send(200, json.dumps({'ok': True}), 'application/json')
        return self.send(405, json.dumps({'detail': 'fake: POST しない'}), 'application/json')


if __name__ == '__main__':
    http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 18823), H).serve_forever()
