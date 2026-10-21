"""/home/jetson/bin/jf-heavy の「走行中・収集中は待つ」(2026-10-21 段 S4)のテスト。torch なし・GPU なし・
管理画面(:8080)には問い合わせない(作り物の status のサーバーを 127.0.0.1 の空いた番号で立てる)。車は動かさない

- running か recording が true のあいだは始めない。両方 false になったら始める
- 管理画面に届かない・答えが読めない・HTTP のエラーなら、今まで通りすぐ始める
- --no-drive-wait で待たない。--mem・--lock・--need-mb の今の使い方はそのまま動く

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M --no-drive-wait -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q tests/test_policy_ens_jf_heavy.py
"""

import http.server
import json
import os
import shutil
import socket
import subprocess
import threading
import time

import pytest

JF = os.environ.get('JF_HEAVY_BIN', '/home/jetson/bin/jf-heavy')
POLL_S = '0.2'

pytestmark = pytest.mark.skipif(not (os.path.exists(JF) and shutil.which('systemd-run') and shutil.which('curl')),
                                reason='jf-heavy・systemd-run・curl が要る')


class _Status(object):
    def __init__(self):
        self.body = {'running': False, 'recording': False}
        self.code = 200
        self.hits = 0


def _server(status):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            status.hits += 1
            raw = status.body if isinstance(status.body, bytes) else json.dumps(status.body).encode()
            self.send_response(status.code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def status():
    st = _Status()
    srv = _server(st)
    st.url = 'http://127.0.0.1:%d/api/drive/status' % srv.server_address[1]
    yield st
    srv.shutdown()


def _env(url):
    return dict(os.environ, JF_HEAVY_DRIVE_URL=url, JF_HEAVY_DRIVE_POLL_S=POLL_S)


def _start(url, out, *opts):
    cmd = [JF, '--lock', 'none', '--mem', '200M', '--need-mb', '1'] + list(opts) + ['--', 'sh', '-c', 'echo ran > "%s"' % out]
    return subprocess.Popen(cmd, env=_env(url), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _free_port_url():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()                                   # 誰も聞いていない番号 = 管理画面に届かない
    return 'http://127.0.0.1:%d/api/drive/status' % port


def test_runs_immediately_when_idle(status, tmp_path):
    out = tmp_path / 'o'
    p = _start(status.url, out)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 0 and out.read_text().strip() == 'ran', err
    assert status.hits >= 1 and '走行中' not in err


@pytest.mark.parametrize('busy', [{'running': True, 'recording': False}, {'running': False, 'recording': True}])
def test_waits_while_driving_or_recording(status, tmp_path, busy):
    status.body = dict(busy, phase='driving')
    out = tmp_path / 'o'
    p = _start(status.url, out)
    time.sleep(1.5)
    assert p.poll() is None and not out.exists()             # 待っている
    hits = status.hits
    assert hits >= 2                                          # 問い合わせ続けている
    status.body = {'running': False, 'recording': False, 'phase': 'idle'}
    _, err = p.communicate(timeout=30)
    assert p.returncode == 0 and out.read_text().strip() == 'ran'
    assert '走行中・収集中のため待っています' in err


def test_no_drive_wait_skips_the_check(status, tmp_path):
    status.body = {'running': True, 'recording': True}
    out = tmp_path / 'o'
    p = _start(status.url, out, '--no-drive-wait')
    _, err = p.communicate(timeout=30)
    assert p.returncode == 0 and out.read_text().strip() == 'ran', err
    assert status.hits == 0


@pytest.mark.parametrize('body,code', [(b'not json', 200), (b'[1, 2]', 200), (b'{"running": "yes"}', 200),
                                       (b'{"detail": "x"}', 500)])
def test_unreadable_answer_runs_as_before(status, tmp_path, body, code):
    status.body, status.code = body, code
    out = tmp_path / 'o'
    p = _start(status.url, out)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 0 and out.read_text().strip() == 'ran', err


def test_unreachable_server_runs_as_before(tmp_path):
    out = tmp_path / 'o'
    t0 = time.time()
    p = _start(_free_port_url(), out)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 0 and out.read_text().strip() == 'ran', err
    assert time.time() - t0 < 15


def test_existing_options_still_work(status, tmp_path):
    """--mem・--lock(名前つきの順番待ち)・--need-mb・終了コードの受け渡し・-- なしの書き方"""
    lock = 'pytest-s4-%d' % os.getpid()
    p = subprocess.run([JF, '--mem', '300M', '--lock', lock, '--need-mb', '1', '--', 'sh', '-c', 'exit 3'],
                       env=_env(status.url), capture_output=True, text=True, timeout=30)
    assert p.returncode == 3, p.stderr
    assert os.path.exists('/tmp/jf-heavy-%s.lock' % lock)
    os.remove('/tmp/jf-heavy-%s.lock' % lock)
    p = subprocess.run([JF, '--lock', 'none', '--need-mb', '1', 'true'], env=_env(status.url), capture_output=True,
                       text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    p = subprocess.run([JF, '--lock', 'none'], env=_env(status.url), capture_output=True, text=True, timeout=30)
    assert p.returncode == 2 and '--no-drive-wait' in p.stderr       # 使い方
    # メモリの上限は今まで通り(上限を超えたコマンドだけが止まる)
    p = subprocess.run([JF, '--lock', 'none', '--mem', '64M', '--need-mb', '1', '--', '/usr/bin/python3', '-c',
                        'b = bytearray(400 * 1024 * 1024); print(len(b))'], env=_env(status.url), capture_output=True,
                       text=True, timeout=60)
    assert p.returncode != 0
