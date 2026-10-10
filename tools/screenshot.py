"""管理画面のスクリーンショット(ヘッドレスFirefox)。geckodriver --port 4444 を起動してから実行する

    W=1440 H=1000 python tools/screenshot.py "live.png::window.jetformer.showTab('live')" "drive.png::window.jetformer.showTab('drive')"
    各引数は「保存先::実行するJavaScript」。JavaScriptを実行してから WAIT 秒(既定3)待って撮る
"""
import base64
import json
import os
import sys
import time
import urllib.request

WD = os.environ.get('WEBDRIVER', 'http://127.0.0.1:4444')
URL = os.environ.get('JETFORMER_URL', 'http://127.0.0.1:8080/')
WIDTH, HEIGHT = int(os.environ.get('W', '1440')), int(os.environ.get('H', '1000'))


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(WD + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=90).read())['value']


def main(args):
    caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
            'moz:firefoxOptions': {'args': ['-headless', '--width=%d' % WIDTH, '--height=%d' % HEIGHT]}}}}
    sid = req('POST', '/session', caps)['sessionId']
    try:
        req('POST', '/session/%s/url' % sid, {'url': URL})
        req('POST', '/session/%s/window/rect' % sid, {'width': WIDTH, 'height': HEIGHT})
        time.sleep(1.5)
        for arg in args:
            name, _, script = arg.partition('::')
            if script:
                req('POST', '/session/%s/execute/sync' % sid, {'script': script, 'args': []})
            time.sleep(float(os.environ.get('WAIT', '3')))
            with open(name, 'wb') as f:
                f.write(base64.b64decode(req('GET', '/session/%s/screenshot' % sid)))
            print('saved', name)
    finally:
        req('DELETE', '/session/%s' % sid)


if __name__ == '__main__':
    main(sys.argv[1:])
