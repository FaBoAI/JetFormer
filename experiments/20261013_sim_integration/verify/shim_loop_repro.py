"""sim-shim.js の #jetformer-open の書き換えが MutationObserver の無限ループになるかを確かめる(修正の前後で実行する)

    jf-heavy --mem 2500M --lock big -- python -I shim_loop_repro.py --tag before|after [--display :0]

ページを 127.0.0.1 で開き、アプリと同じように #jetformer-open の href を変える(収集の終了・「接続・取り込み状況を確認」で起きること)。
その後 10 秒間、ページが応答するか(WebDriver の execute が返るか)と Firefox のメモリを記録する。ブリッジは使わない。
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

GECKO = '/snap/bin/geckodriver'
PORT = 4444
WORK = Path('/home/jetson/jetformer_reports/sim_integration_20261013/verify')
PROFILE = WORK / 'ff_profile_repro'
XAUTH = '/run/user/1000/gdm/Xauthority'
ap = argparse.ArgumentParser()
ap.add_argument('--base', default='http://127.0.0.1:8080')
ap.add_argument('--display', default=':0')
ap.add_argument('--tag', required=True)
a = ap.parse_args()


def wdreq(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request('http://127.0.0.1:%d%s' % (PORT, path), data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read() or b'{}').get('value')


def scope_mb(scope):
    p = Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice') / scope / 'memory.current'
    try:
        return round(int(p.read_text()) / 1e6)
    except OSError:
        return None


res = {'tag': a.tag, 'samples': []}
gd = None
sid = None
try:
    env = dict(os.environ, DISPLAY=a.display, XAUTHORITY=XAUTH)
    gd = subprocess.Popen([GECKO, '--port', str(PORT)], stdout=open(WORK / ('geckodriver_repro_%s.log' % a.tag), 'w'),
                          stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        try:
            wdreq('GET', '/status', timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    shutil.rmtree(PROFILE, ignore_errors=True)
    PROFILE.mkdir(parents=True)
    caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager', 'timeouts': {'script': 5000},
                                             'moz:firefoxOptions': {'args': ['-profile', str(PROFILE)], 'env': {'DISPLAY': a.display, 'XAUTHORITY': XAUTH}}}}}
    sid = wdreq('POST', '/session', caps, timeout=180)['sessionId']
    scope = Path('/proc/%d/cgroup' % gd.pid).read_text().strip().splitlines()[-1].rsplit('/', 1)[-1]
    subprocess.run(['systemctl', '--user', 'set-property', '--runtime', scope, 'MemoryMax=2500M', 'MemorySwapMax=0'], check=False)
    js = lambda s, t=8: wdreq('POST', '/session/%s/execute/sync' % sid, {'script': s, 'args': []}, timeout=t)  # noqa: E731
    wdreq('POST', '/session/%s/url' % sid, {'url': a.base + '/sim/c/default/'})
    for _ in range(60):
        if js("return !!window.courseSimulator && !!document.querySelector('#jf-simbar')"):
            break
        time.sleep(0.5)
    time.sleep(2)
    res['before'] = js("return document.querySelector('#jetformer-open').getAttribute('href')")
    # アプリの refreshLink() と同じ: jetformer-url(shim が http://127.0.0.1:8080 にしたもの)を href に入れる
    try:
        t1 = time.time()
        js("document.querySelector('#jetformer-open').href = document.querySelector('#jetformer-url').value; return 1")
        res['set_ms'] = round((time.time() - t1) * 1000)
    except Exception as e:  # noqa: BLE001   直す前はここで固まった(8 秒で時間切れ)
        res['set_error'] = str(e)[:160]
    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            t1 = time.time()
            v = js("return document.querySelector('#jetformer-open').getAttribute('href')", 8)
            res['samples'].append({'t': round(time.time() - t0, 1), 'responds': True, 'ms': round((time.time() - t1) * 1000), 'href': v,
                                   'scope_mb': scope_mb(scope)})
        except Exception as e:  # noqa: BLE001
            res['samples'].append({'t': round(time.time() - t0, 1), 'responds': False, 'error': str(e)[:160], 'scope_mb': scope_mb(scope)})
        time.sleep(1)
    res['hung'] = 'set_error' in res or any(not s['responds'] for s in res['samples'])
finally:
    if gd:
        gd.kill()                        # ページが固まっていても止める(Firefox も geckodriver のスコープごと止まる)
        subprocess.run(['pkill', '-f', 'ff_profile_repro'], check=False)
    time.sleep(1)
    shutil.rmtree(PROFILE, ignore_errors=True)
    res['leftover'] = subprocess.run(['pgrep', '-af', 'ff_profile_repro|geckodriver'], capture_output=True, text=True).stdout.strip()
    (WORK / ('shim_loop_repro_%s.json' % a.tag)).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
