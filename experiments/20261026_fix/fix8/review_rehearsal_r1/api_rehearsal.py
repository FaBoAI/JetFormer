"""反証 8-1 回目・係 rehearsal: 手順書 3 章・5 章の curl のコマンド(止まっているときの API)を、手順書の文から抜き出してそのまま流す。
相手は作り物のサーバー(別のポート 18080。本物の drive.Driver.set_params を、設定ファイルを作業場所に替えて呼ぶ。FastAPI と同じに
400 は {"detail": 文})。本物の jetformer-web・基板には触れない(本物の server.py は読み込まない)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B api_rehearsal.py
"""
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402

RB = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md')
TMP = JF / 'experiments/20261026_fix/fix8/tmp_rehearsal/api'
TMP.mkdir(parents=True, exist_ok=True)
drive.SETTINGS_PATH = TMP / 'drive_settings.json'          # 本物の data/drive_settings.json は読まない・書かない
PORT = 18080


class Alive(object):
    def __init__(self):
        self.on = False

    def is_alive(self):
        return self.on


DRV = drive.Driver.__new__(drive.Driver)
DRV.lock = threading.Lock()
DRV.params = dict(drive.DEFAULT_PARAMS, steer_gain=1.0, throttle_gain=1.0, lookahead=3, start_kick=1)
DRV.st = {}
DRV.thread = Alive()
DRV.params_changes = []
DRV.recorder = None
DRV.last = {"run": None, "mode_id": None}
DRV.boot_at = 1793116404.711


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith('/api/drive/status'):
            return self._send(200, {'running': DRV.running, 'params': dict(DRV.params), 'boot_at': DRV.boot_at})
        return self._send(404, {'detail': 'Not Found'})

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(n) or b'{}')
        if self.path == '/api/drive/params':
            try:
                return self._send(200, DRV.set_params(body))
            except (ValueError, TypeError) as e:
                return self._send(400, {'detail': str(e)})
        return self._send(404, {'detail': 'Not Found'})


def commands():
    """手順書の ```bash の塊から curl -s -X POST の行(と直前の # の行)を抜き出す"""
    s = RB.read_text(encoding='utf-8')
    out = []
    for blk in re.findall(r'```bash\n(.*?)```', s, re.S):
        note = ''
        for ln in blk.splitlines():
            if ln.startswith('#'):
                note = ln
            elif ln.startswith('curl -s -X POST'):
                out.append((note, ln))
    return out


def sh(cmd):
    r = subprocess.run(['bash', '-c', cmd.replace('127.0.0.1:8080', '127.0.0.1:%d' % PORT)], capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr).strip()


def main():
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cmds = commands()
    lines = ['curl の行 %d 本(手順書から)' % len(cmds)]
    guard1, both, left42, us200 = [c for _, c in cmds]
    guard0 = guard1.replace('{"rec_back_guard": 1}', '{"rec_back_guard": 0}')      # 手順書 3 章「API なら上の curl の 1 の `1` を `0` に」
    left41 = left42.replace('4.2', '4.1')
    plan = [
        ('朝: 残すだけ(見張りが OFF)を送った形 = 手順書の 2 本目の残すだけ', both.replace('"rec_back_guard": 1, ', '')),
        ('朝: 見張り(3 章の 1 本目)', guard1),
        ('朝: 見張りと残す(3 章の 2 本目)', both),
        ('2-5 の前: 左 4.2', left42),
        ('空転 > 1.3: launch_us 200', us200),
        ('最初のリカバリーがおかしい: 見張り OFF(1 → 0)', guard0),
        ('2-6 の前: 左 4.1', left41),
    ]
    for name, c in plan:
        lines.append('## %s\n$ %s\n%s' % (name, c.replace('127.0.0.1:8080', '127.0.0.1:%d' % PORT)[:160] + ' …', sh(c)))
    # 走行中(400 の文)
    DRV.thread.on = True
    for name, c in [('走行中に見張り ON', guard1), ('走行中に左 4.2', left42), ('走行中に launch_us 200', us200), ('走行中に見張り OFF', guard0)]:
        lines.append('## %s\n%s' % (name, sh(c)))
    DRV.thread.on = False
    # 本物と同じ形の GET(作り物のサーバー)。手順書 3.1 の 1 行目の読み方
    lines.append('## GET(作り物)\n' + sh('curl -s http://127.0.0.1:8080/api/drive/status | python3 -c "import json,sys; s=json.load(sys.stdin); '
                                          'p=s[\'params\']; print({k: p.get(k) for k in (\'launch_boost\',\'rec_back_guard\',\'rec_hold_keep\','
                                          '\'launch_no_kick\',\'speed_ceiling\',\'speed_auto\',\'zone_caps\',\'ceil_left_mps\',\'launch_us\')}, '
                                          '\'boot_at\', s.get(\'boot_at\'))"'))
    srv.shutdown()
    txt = '\n'.join(lines) + '\n'
    (HERE / 'out' / 'api_rehearsal.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
