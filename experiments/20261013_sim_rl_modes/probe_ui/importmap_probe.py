#!/usr/bin/env python3
"""確認(理解の段・UI 側): JetFormer の /sim/c/<コース>/ と同じ階層で、import map の imports + scopes により
  - app.js が読む ./rc-speed-controller.js を ./_jf/rl/teacher.js に差し替え、差し替え側からは元のファイルを読める
  - ./scene.js を ./_jf/scene-overlay.js に差し替え、重ねる側からは元の scene.js を読める(three は 1 つ)
  - 差し替えモジュールのトップレベル await(重みの fetch)で app.js が待つ
  - world.disks が scene の physics.disks と同じ配列(後から障害物を入れ替えるとリセットで使われる)
をヘッドレス Firefox(WebGL 不要)で確かめる。ページは非ループバックのアドレス(http://192.168.55.1:8801)で開く。

    jf-heavy --mem 2500M --lock big -- python3 -I importmap_probe.py

静的なファイルは probe_ui/importmap/www/(ユーザーのファイルは使わない)。終わったら Firefox・geckodriver・サーバーを止める。
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013/probe_ui/importmap')
WWW = OUT / 'www'
PROFILE = OUT / 'ff_profile'
GECKO, GECKO_PORT, HTTP_PORT = '/snap/bin/geckodriver', 4447, 8801
URL = 'http://192.168.55.1:%d/sim/c/default/' % HTTP_PORT


def req(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request('http://127.0.0.1:%d%s' % (GECKO_PORT, path), data=data, method=method,
                               headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(r, timeout=timeout) as f:
        return json.loads(f.read().decode())['value']


def main():
    PROFILE.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, '-I', '-m', 'http.server', str(HTTP_PORT), '--bind', '192.168.55.1',
                            '--directory', str(WWW)], stdout=open(OUT / 'http.log', 'w'), stderr=subprocess.STDOUT)
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=open(OUT / 'geckodriver.log', 'w'), stderr=subprocess.STDOUT)
    sid, result = None, {'url': URL}
    try:
        for _ in range(60):
            try:
                req('GET', '/status', timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox',
                                                 'moz:firefoxOptions': {'args': ['-headless', '-profile', str(PROFILE)]}}}}
        sid = req('POST', '/session', caps, timeout=180)['sessionId']
        req('POST', '/session/%s/url' % sid, {'url': URL})
        for _ in range(40):
            v = req('POST', '/session/%s/execute/sync' % sid, {'script': 'return window.__result || null;', 'args': []})
            if v:
                result['result'] = v
                break
            time.sleep(0.25)
        result['browser'] = req('POST', '/session/%s/execute/sync' % sid, {'script': 'return navigator.userAgent;', 'args': []})
        r = result.get('result') or {}
        result['ok'] = (r.get('teacher', {}).get('who') == 'jf' and r.get('teacher', {}).get('base') == 'orig'
                        and r.get('teacher_rc', {}).get('who') == 'orig' and r.get('plan') == 'orig-plan'
                        and r.get('assert') == 'orig-assert' and r.get('overlay', {}).get('three_same') is True
                        and r.get('live_active_disks') == ['post', 'obstacle'])
    finally:
        if sid:
            try:
                req('DELETE', '/session/%s' % sid, timeout=30)
            except Exception as e:                                    # noqa: BLE001
                result['close_error'] = str(e)
        gd.terminate()
        srv.terminate()
        for p in (gd, srv):
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
