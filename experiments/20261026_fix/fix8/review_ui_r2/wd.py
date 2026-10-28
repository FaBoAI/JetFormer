"""(反証 8-2 回目・係 ui: 8-1 の係の wd.py の写し。ポートだけ替えた)headless の Firefox(geckodriver の WebDriver。urllib だけ)で、作り物のサーバー(fake_server.py)の画面を開く小さな道具(この係)"""
import base64
import json
import subprocess
import time
import urllib.request

import os
WD = 'http://127.0.0.1:%s' % os.environ.get('JF_WD_PORT', '4497')
SRV_PORT = os.environ.get('JF_FAKE_PORT', '18797')


def req(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(WD + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())['value']


class Browser:
    def __init__(self, width=420, height=900, scale=None):
        self.srv = subprocess.Popen(['python3', '-I', '-B', str(__import__('pathlib').Path(__file__).resolve().parent / 'fake_server.py'), SRV_PORT], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.gd = subprocess.Popen(['/snap/bin/geckodriver', '--port', WD.rsplit(':', 1)[1]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                req('GET', '/status')
                break
            except Exception:
                time.sleep(0.5)
        v = req('POST', '/session', {'capabilities': {'alwaysMatch': {'moz:firefoxOptions': dict({'args': ['-headless']}, **({'prefs': {'layout.css.devPixelsPerPx': str(scale), 'widget.gtk.overlay-scrollbars.enabled': True, 'widget.non-native-theme.scrollbar.style': 1}} if scale else {}))}}}, timeout=120)
        self.sid = v['sessionId']
        req('POST', '/session/%s/window/rect' % self.sid, {'width': width, 'height': height})

    def go(self, url):
        req('POST', '/session/%s/url' % self.sid, {'url': url})

    def js(self, script, *args):
        return req('POST', '/session/%s/execute/sync' % self.sid, {'script': script, 'args': list(args)})

    def shot(self, path, full=False):
        v = req('GET', '/session/%s/%s' % (self.sid, 'moz/screenshot/full' if full else 'screenshot'))
        open(path, 'wb').write(base64.b64decode(v))

    def close(self):
        try:
            req('DELETE', '/session/%s' % self.sid)
        except Exception:
            pass
        for p in (self.gd, self.srv):
            p.terminate()
            try:
                p.wait(10)
            except Exception:
                p.kill()

    def frame(self, i=0):
        req('POST', '/session/%s/frame' % self.sid, {'id': i})

    def top(self):
        req('POST', '/session/%s/frame' % self.sid, {'id': None})
