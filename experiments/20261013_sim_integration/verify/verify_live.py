"""本番の jetformer-web(:8080)でシミュレーター統合を端から端まで確かめる(sim_integration_20261013 / verify)

    jf-heavy --mem 2500M --lock big -- python -I verify_live.py [--display :0] [--base http://127.0.0.1:8080]

- geckodriver(snap)をポート 4444 で起動し、専用のプロファイルで Firefox を開く(--display :0 なら WebGL が使える。
  ヘッドレスの Firefox ではこの Jetson では WebGL 2 が使えないことを確認済み: sim_rl_bestlap_20261013/webgl_probe/default)。
- snap の Firefox は自分の systemd スコープに移って jf-heavy の上限の外に出るので、そのスコープにも MemoryMax を付ける。
- 実車・PCA9685・GPU は使わない(ブリッジは CPU の onnxruntime)。
- 作ったもの(データセット・シーン・ブリッジの作業フォルダー・コース)は最後に消す。Firefox・geckodriver・ブリッジも止める。
"""
import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

GECKO = '/snap/bin/geckodriver'
PORT = 4444
ROOT = Path('/home/jetson/jetformer_reports/sim_integration_20261013')
SHOTS = ROOT / 'screens'
WORK = ROOT / 'verify'
PROFILE = WORK / 'ff_profile'
REL = Path('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5')
SNAP_V2 = REL / 'qa/course-snapshots/indoor-course-eval-2-shortcut'
ELEM = 'element-6066-11e4-a52e-4f735466cecf'
XAUTH = '/run/user/1000/gdm/Xauthority'

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='http://127.0.0.1:8080')
ap.add_argument('--display', default=':0')
ap.add_argument('--run', default='clean_driving_20261011-nocmd-s3-20261011-024626')     # モデル B
ap.add_argument('--firefox-mem', default='2500M')
a = ap.parse_args()
SHOTS.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime('%H%M%S')
res = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'base': a.base, 'display': a.display or 'headless', 'model': a.run, 'checks': []}
created = {'sim_sessions': [], 'datasets': [], 'scenes': [], 'courses': []}


def log(*x):
    print(*x, flush=True)


def check(name, ok, **info):
    res['checks'].append(dict(name=name, ok=bool(ok), **info))
    log(('OK  ' if ok else 'NG  ') + name, json.dumps(info, ensure_ascii=False, default=str)[:700])


def http(method, path, body=None, headers=None, timeout=60, raw=False):
    data = body if isinstance(body, (bytes, bytearray)) else (None if body is None else json.dumps(body).encode())
    h = {'Content-Type': 'application/json'} if (body is not None and not isinstance(body, (bytes, bytearray))) else {}
    h.update(headers or {})
    req = urllib.request.Request(a.base + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = r.read()
            return (r.status, b) if raw else (r.status, json.loads(b or b'null'))
    except urllib.error.HTTPError as e:
        b = e.read()
        if raw:
            return e.code, b
        try:
            return e.code, json.loads(b or b'null')
        except ValueError:
            return e.code, b.decode('utf-8', 'replace')


def api(method, path, body=None, timeout=60):
    st, v = http(method, path, body, timeout=timeout)
    if st >= 400:
        raise RuntimeError('%s %s -> %s %s' % (method, path, st, v))
    return v


def sha(b):
    return hashlib.sha256(b).hexdigest()


def busy_state():
    t = api('GET', '/api/train/status')
    d = api('GET', '/api/drive/status')
    s = api('GET', '/api/status')
    j = api('GET', '/api/jobs')
    return {'training': t.get('running'), 'driving': d.get('running'), 'drive_phase': d.get('phase'),
            'recording': s['recorder']['recording'], 'session': s['session']['active'],
            'running_jobs': [x['title'] for x in j if x['state'] == 'running']}


def predicts():
    return api('GET', '/api/sim/bridge/log?lines=1')['suppressed'].get('predict', 0)


class WD(object):
    def __init__(self):
        self.base, self.sid = 'http://127.0.0.1:%d' % PORT, None

    def req(self, method, path, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read() or b'{}').get('value')
        except urllib.error.HTTPError as e:
            raise RuntimeError('%s %s: %s %r' % (method, path, e.code, e.read()[:400]))

    def s(self, method, path, body=None, timeout=120):
        return self.req(method, '/session/%s%s' % (self.sid, path), body, timeout)

    def js(self, script, *args):
        return self.s('POST', '/execute/sync', {'script': script, 'args': list(args)})

    def get(self, url):
        return self.s('POST', '/url', {'url': url})

    def find(self, css):
        return self.s('POST', '/element', {'using': 'css selector', 'value': css})[ELEM]

    def click(self, css):
        eid = self.find(css)
        self.s('POST', '/execute/sync', {'script': "arguments[0].scrollIntoView({block:'center'})", 'args': [{ELEM: eid}]})
        self.s('POST', '/element/%s/click' % eid, {})

    def keys(self, css, text):
        self.s('POST', '/element/%s/value' % self.find(css), {'text': text})

    def handles(self):
        return self.s('GET', '/window/handles')

    def switch(self, h):
        self.s('POST', '/window', {'handle': h})

    def close(self):
        return self.s('DELETE', '/window')

    def accept_alert(self, timeout=5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                txt = self.s('GET', '/alert/text')
                self.s('POST', '/alert/accept', {})
                return txt
            except RuntimeError:
                time.sleep(0.2)
        return None

    def shot(self, name, full=False):
        b64 = self.s('GET', '/moz/screenshot/full' if full else '/screenshot')
        (SHOTS / name).write_bytes(base64.b64decode(b64))
        return name

    def elem_shot(self, css, name):
        (SHOTS / name).write_bytes(base64.b64decode(self.s('GET', '/element/%s/screenshot' % self.find(css))))
        return name


def wait(fn, timeout, interval=0.5):
    t0 = time.time()
    v = None
    while time.time() - t0 < timeout:
        try:
            v = fn()
        except Exception as e:  # noqa: BLE001
            v = {'error': str(e)}
        if v and not (isinstance(v, dict) and v.get('error')):
            return v, round(time.time() - t0, 1)
        time.sleep(interval)
    return v, None


def cap_snap_scope(pid, mem):
    """snap の Firefox は snap.firefox.geckodriver-*.scope に移り jf-heavy の上限の外に出るので、そのスコープにも上限を付ける"""
    try:
        scope = Path('/proc/%d/cgroup' % pid).read_text().strip().splitlines()[-1].rsplit('/', 1)[-1]
    except OSError as e:
        return {'error': str(e)}
    if not scope.startswith('snap.'):
        return {'scope': scope, 'note': 'snap のスコープではない(jf-heavy の上限の中)'}
    r = subprocess.run(['systemctl', '--user', 'set-property', '--runtime', scope, 'MemoryMax=%s' % mem, 'MemorySwapMax=0'],
                       capture_output=True, text=True)
    shown = subprocess.run(['systemctl', '--user', 'show', scope, '-p', 'MemoryMax', '-p', 'MemoryCurrent'],
                           capture_output=True, text=True).stdout.split()
    return {'scope': scope, 'returncode': r.returncode, 'stderr': r.stderr.strip(), 'show': shown}


PAGE = r"""
const cs = window.courseSimulator, q = (s) => document.querySelector(s), cfg = window.JETFORMER_SIM || {};
let snap = null; try { snap = cs ? cs.snapshot().state : null; } catch (e) {}
let gl2 = false; try { gl2 = !!document.createElement('canvas').getContext('webgl2'); } catch (e) {}
return {app: !!cs, fatal: q('#fatal') ? (q('#fatal').hidden ? '' : q('#fatal').textContent) : null, webgl2: gl2,
  course: cfg.course ? cfg.course.id : null, course_kind_renderer: cfg.course ? cfg.course.renderer : null,
  bar: !!q('#jf-simbar'), options: q('#jf-simbar select') ? [...q('#jf-simbar select').options].map(o => o.value) : [],
  bar_text: q('#jf-simbar') ? q('#jf-simbar').innerText : null,
  label: q('.view-label') && q('.view-label').textContent, size: q('.view-note span') && q('.view-note span').textContent,
  teachers: q('#teacher-profile') ? [...q('#teacher-profile').options].map(o => o.value) : null,
  planHidden: q('#collection-plan') ? q('#collection-plan').hidden : null, endpointReadOnly: q('#endpoint') && q('#endpoint').readOnly,
  conn: q('#connection-status') && q('#connection-status').textContent, upload: q('#upload-status') && q('#upload-status').textContent,
  model: q('.jf-model') && q('.jf-model').textContent, bridge: q('.jf-pill') && q('.jf-pill').textContent,
  ready: cs ? cs.sensorSummary().ready : false, hash: cs ? cs.report().world_hash : null,
  version: cs ? cs.report().context.world_version : null, title: document.title, status: q('#status') && q('#status').textContent,
  run_disabled: q('#run') ? q('#run').disabled : null, run_text: q('#run') ? q('#run').textContent : null,
  record: q('#record-status') && q('#record-status').textContent, download: q('#dataset-download') ? q('#dataset-download').getAttribute('href') : null,
  jf_status: q('#jetformer-status') && q('#jetformer-status').textContent, jf_url: q('#jetformer-url') && q('#jetformer-url').value,
  elapsed: q('#elapsed') && q('#elapsed').textContent, state: snap ? {frame: snap.frame, x: snap.x, z: snap.z, yaw: snap.yaw, speed: snap.speed, terminal: snap.terminal} : null,
  inference: cs ? (cs.report().inference || []).length : null, href: location.pathname + location.search};
"""

RESOURCES = "return performance.getEntriesByType('resource').map(e => [e.name.replace(location.origin, ''), e.responseStatus || 0])"
CAMERA = "const c=document.querySelector('#camera'); return c ? c.toDataURL('image/png') : null"


def save_camera(wd, name):
    """シミュレーターの前方カメラ(#camera)の画像を保存し、画素のばらつき(真っ黒・単色でないこと)を返す"""
    try:
        url = wd.js(CAMERA)
        raw = base64.b64decode(url.split(',', 1)[1])
        (SHOTS / name).write_bytes(raw)
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return {'file': name, 'shape': list(img.shape), 'std': round(float(img.std()), 1), 'mean': round(float(img.mean()), 1)}
    except Exception as e:  # noqa: BLE001
        return {'error': str(e)}


def set_value(wd, css, value, event='change'):
    return wd.js("const e=document.querySelector(arguments[0]); e.value=arguments[1]; e.dispatchEvent(new Event(arguments[2],{bubbles:true})); return e.value",
                 css, value, event)


def open_sim_page(wd, path_or_url, label, want_hash, shot):
    """ページを開いて準備ができるのを待ち、主な状態を確かめて写真を撮る"""
    if path_or_url:
        wd.get(path_or_url if path_or_url.startswith('http') else a.base + path_or_url)
    st, t = wait(lambda: (lambda s: s if (s['app'] and s['ready']) or s['fatal'] else None)(wd.js(PAGE)), 90)
    time.sleep(1.5)                                       # 描画が 2〜3 フレーム進むのを待つ
    st = wd.js(PAGE)
    bad = [r for r in wd.js(RESOURCES) if r[1] >= 400 and r[0] != '/favicon.ico']
    cam = save_camera(wd, shot.replace('.png', '_camera.png'))
    wd.shot(shot)
    return st, t, bad, cam


gd = None
wd = WD()
fails = []
try:
    # ---------------------------------------------------------------- 0. 前提(何も動いていない)
    res['busy_before'] = busy_state()
    check('0 前提: 学習・走行・収集・ジョブが動いていない', not any([res['busy_before']['training'], res['busy_before']['driving'],
                                                      res['busy_before']['recording'], res['busy_before']['session'],
                                                      res['busy_before']['running_jobs']]), **res['busy_before'])
    sim0 = api('GET', '/api/sim/status')
    res['sim_status_before'] = {'release': sim0['release']['id'], 'env_ready': sim0['env']['ready'], 'bridge': sim0['bridge']['state']}

    # ---------------------------------------------------------------- Firefox
    env = dict(os.environ)
    if a.display:
        env.update(DISPLAY=a.display, XAUTHORITY=XAUTH)
    gd = subprocess.Popen([GECKO, '--port', str(PORT)], stdout=open(WORK / 'geckodriver.log', 'w'), stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        try:
            wd.req('GET', '/status', timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    if PROFILE.exists():
        shutil.rmtree(PROFILE)
    PROFILE.mkdir(parents=True)
    args = ['-profile', str(PROFILE)] + ([] if a.display else ['-headless'])
    fenv = {'MOZ_HEADLESS_WIDTH': '1400', 'MOZ_HEADLESS_HEIGHT': '1000'}
    if a.display:
        fenv.update(DISPLAY=a.display, XAUTHORITY=XAUTH)
    caps = {'capabilities': {'alwaysMatch': {
        'browserName': 'firefox', 'pageLoadStrategy': 'eager', 'unhandledPromptBehavior': 'ignore',
        'moz:firefoxOptions': {'args': args, 'env': fenv,
                               'prefs': {'dom.disable_open_during_load': False, 'browser.link.open_newwindow': 3,
                                         'browser.tabs.warnOnClose': False, 'browser.shell.checkDefaultBrowser': False}}}}}
    v = wd.req('POST', '/session', caps, timeout=180)
    wd.sid = v['sessionId']
    res['browser'] = {k: v['capabilities'].get(k) for k in ('browserVersion', 'moz:headless')}
    res['memory_cap'] = cap_snap_scope(gd.pid, a.firefox_mem)
    wd.s('POST', '/window/rect', {'width': 1400, 'height': 1000})
    main = wd.s('GET', '/window')

    # ---------------------------------------------------------------- 1. 管理画面の「シミュレーター」タブ
    wd.get(a.base + '/')
    wait(lambda: wd.js("return !!(window.jetformer && document.querySelector('button.tab[data-tab=sim]'))"), 20)
    wd.click('button.tab[data-tab="sim"]')
    TAB = r"""const q=(s)=>document.querySelector(s), cs=[...document.querySelectorAll('#sim-courses .sim-course')];
      return {active: q('#tab-sim').classList.contains('active'), release: q('#sim-release').innerText, env: q('#sim-env').innerText,
        env_pill: q('#sim-env-pill').innerText, bridge_pill: q('#sim-bridge-pill').innerText, bridge: q('#sim-bridge').innerText,
        start_disabled: q('#btn-sim-start').disabled, stop_disabled: q('#btn-sim-stop').disabled,
        courses: cs.map(c => c.querySelector('b').textContent), course_meta: cs.map(c => c.querySelector('.meta').innerText),
        open_courses: [...q('#sim-open-course').options].map(o => o.value),
        models: [...q('#sim-open-model').options].filter(o => o.value && !o.disabled).map(o => o.value),
        models_disabled: [...q('#sim-open-model').options].filter(o => o.disabled).length,
        datasets: q('#sim-datasets').innerText, jobs: q('#sim-jobs').innerText, upload: q('#sim-course-upload').innerText};"""
    tab, t = wait(lambda: (lambda s: s if s['active'] and len(s['courses']) >= 2 and s['bridge_pill'] else None)(wd.js(TAB)), 30)
    ok = bool(tab) and not tab.get('error') and tab['open_courses'][:2] == ['default', 'indoor-course-eval-2-shortcut'] \
        and '停止中' in tab['bridge_pill'] and a.run in tab['models'] and '1.31.0' in tab['env']
    check('1 シミュレーター タブ: 版・Python 環境・ブリッジの状態・コース Default と v2・モデル B が選べる', ok, seconds=t,
          release=tab and tab.get('release', '')[:120], env=tab and tab.get('env'), bridge_pill=tab and tab.get('bridge_pill'),
          courses=tab and tab.get('courses'), course_meta=tab and tab.get('course_meta'), open_courses=tab and tab.get('open_courses'),
          models=tab and len(tab.get('models', [])), models_disabled=tab and tab.get('models_disabled'))
    wd.shot('01_sim_tab.png', full=True)

    # ---------------------------------------------------------------- 2. ブリッジを画面から起動
    wd.click('#btn-sim-start')
    tab2, t = wait(lambda: (lambda s: s if '動作中' in s['bridge_pill'] else None)(wd.js(TAB)), 40)
    b = api('GET', '/api/sim/status')['bridge']
    port = b.get('port')
    ss = subprocess.run(['ss', '-ltnH', 'sport = :%s' % port], capture_output=True, text=True).stdout.strip() if port else ''
    lan_ip = subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.split()[0]
    try:
        socket.create_connection((lan_ip, port), timeout=2).close()
        lan_refused = False
    except OSError:
        lan_refused = True
    st_p, body_p = http('GET', '/sim/bridge/api/status')
    st_lan, _ = http('GET', '/sim/bridge/api/status', headers={'Host': '%s:8080' % lan_ip, 'Origin': 'http://%s:8080' % lan_ip})
    check('2 画面の「起動」でブリッジが動作中になる(127.0.0.1 だけで待ち受け・中継は LAN の Host でも 200)',
          tab2 and b['state'] == 'running' and ss and all('127.0.0.1:%s' % port in ln for ln in ss.splitlines()) and lan_refused
          and st_p == 200 and st_lan == 200, seconds=t, bridge_text=tab2 and tab2.get('bridge'), port=port, pid=b.get('pid'),
          rss_mb=b.get('rss_mb'), ss=ss, lan_ip=lan_ip, lan_refused=lan_refused, proxy_status=st_p, proxy_lan_host_status=st_lan,
          policy_loaded=isinstance(body_p, dict) and body_p.get('policy_loaded'))
    wd.elem_shot('#tab-sim .sim-grid > div:first-child', '02_bridge_running.png')

    # ---------------------------------------------------------------- 3. /sim/ → Default(WebGL)
    expect = {'default': (sha((REL / 'dist/scene.js').read_bytes()), sha((REL / 'dist/world.js').read_bytes())),
              'indoor-course-eval-2-shortcut': (sha((SNAP_V2 / 'scene.js').read_bytes()), sha((SNAP_V2 / 'world.js').read_bytes()))}
    served = {}
    for cid, (s1, s2) in expect.items():
        _, sc = http('GET', '/sim/c/%s/scene.js' % cid, raw=True)
        _, wc = http('GET', '/sim/c/%s/world.js' % cid, raw=True)
        _, html = http('GET', '/sim/c/%s/' % cid, raw=True)
        served[cid] = {'scene_sha_ok': sha(sc) == s1, 'world_sha_ok': sha(wc) == s2, 'shim': b'sim-shim.js' in html and b'./_jf/config.js' in html}
    check('3a 配信: scene.js・world.js の sha256 がリリース/スナップショットと一致し、shim が入っている', all(all(x.values()) for x in served.values()),
          served=served)
    wd.get(a.base + '/sim/')
    st, t, bad, cam = open_sim_page(wd, None, 'default', '0a464384', '03_default.png')
    check('3b /sim/ → /sim/c/default/: WebGL で描画・world hash 0a464384・教師 5 つ・帯とコース選択', st['href'] == '/sim/c/default/' and st['webgl2']
          and not st['fatal'] and st['ready'] and st['hash'] == '0a464384' and st['version'] == 'indoor-course-eval-3-turf-board-removed'
          and len(st['teachers']) == 5 and st['bar'] and st['options'][:2] == ['default', 'indoor-course-eval-2-shortcut'] and not bad
          and cam.get('std', 0) > 5, seconds=t, href=st['href'], webgl2=st['webgl2'], hash=st['hash'], version=st['version'],
          teachers=st['teachers'], options=st['options'], label=st['label'], conn=st['conn'], bad=bad, camera=cam, bar=st['bar_text'])

    # ---------------------------------------------------------------- 4. v2(WebGL)と帯でのコース切り替え
    st, t, bad, cam = open_sim_page(wd, '/sim/c/indoor-course-eval-2-shortcut/', 'v2', '547d7e80', '04_v2.png')
    check('4a 旧 v2: WebGL で描画・world hash 547d7e80・教師は slow/speed・20分の収集条件を隠す', st['webgl2'] and not st['fatal'] and st['ready']
          and st['hash'] == '547d7e80' and st['version'] == 'indoor-course-eval-2-shortcut' and st['teachers'] == ['slow', 'speed']
          and st['planHidden'] and not bad and cam.get('std', 0) > 5, seconds=t, hash=st['hash'], version=st['version'],
          teachers=st['teachers'], label=st['label'], title=st['title'], bad=bad, camera=cam)
    set_value(wd, '#jf-simbar select', 'default')
    sw, t = wait(lambda: (lambda s: s if s['app'] and s['href'] == '/sim/c/default/' and s['ready'] else None)(wd.js(PAGE)), 60)
    check('4b 帯のコース選択で v2 → Default に切り替わる', sw and sw.get('hash') == '0a464384', seconds=t, href=sw and sw.get('href'),
          hash=sw and sw.get('hash'))

    # ---------------------------------------------------------------- 5. モデル B の受け渡し(タブから「新しいタブで開く」)→ AI 評価
    wd.get(a.base + '/')
    wait(lambda: wd.js("return !!(window.jetformer && document.querySelector('button.tab[data-tab=sim]'))"), 20)
    wd.click('button.tab[data-tab="sim"]')
    wait(lambda: (lambda s: s if s['active'] and len(s['open_courses']) >= 2 and a.run in s['models'] else None)(wd.js(TAB)), 30)
    set_value(wd, '#sim-open-course', 'default')
    set_value(wd, '#sim-open-model', a.run)
    before_handles = set(wd.handles())
    wd.click('#btn-sim-open')
    newh, _ = wait(lambda: list(set(wd.handles()) - before_handles), 15)
    if not newh or isinstance(newh, dict):
        raise RuntimeError('新しいタブが開きません')
    simtab = newh[0]
    wd.switch(simtab)
    st, t_app = wait(lambda: (lambda s: s if (s['app'] and s['ready']) or s['fatal'] else None)(wd.js(PAGE)), 90)
    loaded, t_model = wait(lambda: (lambda b: b if b.get('policy_loaded') else None)(api('GET', '/api/sim/status')['bridge']), 240, 1)
    time.sleep(1.5)
    st = wd.js(PAGE)
    check('5a モデル B を ?model= で受け渡し、ブリッジに読み込まれる(ZIP → bundle → activate)', loaded and isinstance(loaded, dict)
          and a.run in (loaded.get('policy_name') or '') and 'model=' not in st['href'] and '読み込み完了' in (st['upload'] or ''),
          seconds_app=t_app, seconds_model=t_model, policy_name=loaded and loaded.get('policy_name'), upload=st['upload'],
          model_bar=st['model'], conn=st['conn'], href=st['href'])
    wd.js("window.scrollTo(0, 0)")
    wd.shot('05a_model_b_loaded.png')
    p0 = predicts()
    s0 = wd.js(PAGE)['state']
    set_value(wd, '#controller', 'external')
    time.sleep(1)
    wd.click('#run')
    time.sleep(8)
    st = wd.js(PAGE)
    p1 = predicts()
    wd.js("window.scrollTo(0, 0)")
    time.sleep(0.3)
    wd.shot('05b_model_b_ai_running.png')
    cam = save_camera(wd, '05b_model_b_camera.png')
    if '一時停止' in (wd.js(PAGE)['run_text'] or ''):        # 走行中なら一時停止
        wd.click('#run')
    time.sleep(0.5)
    s1 = wd.js(PAGE)['state']
    moved = s0 and s1 and ((s1['x'] - s0['x']) ** 2 + (s1['z'] - s0['z']) ** 2) ** 0.5
    check('5b AI 評価を約 8 秒: 推論が中継経由でブリッジに届き、予測が返って車が動く', p1 - p0 >= 10 and st['inference'] and st['inference'] > 0
          and moved and moved > 0.05, predicts=p1 - p0, inference_log=st['inference'], frames=s1 and s1['frame'],
          moved_m=moved and round(moved, 3), elapsed=st['elapsed'], status=(st['status'] or '')[:160], run_text=st['run_text'], camera=cam)

    # ---------------------------------------------------------------- 6. 収集 A(数秒)→ シミュレーター内の「JetFormerへ送る」
    def record(name, seconds=3.5):
        set_value(wd, '#controller', 'demo')
        time.sleep(0.5)
        wd.click('#reset')
        time.sleep(0.8)
        set_value(wd, '#dataset-name', name, 'input')
        wd.js("const c=document.querySelector('#automated-labels'); if (!c.checked) c.click(); return c.checked")
        wd.click('#record-start')
        r1, _ = wait(lambda: (lambda s: s if '収集待機' in (s['record'] or '') or '開始できません' in (s['record'] or '') else None)(wd.js(PAGE)), 20)
        wd.click('#run')
        time.sleep(seconds)
        # 走行中は毎フレームの送信の間「収集終了」が押せない(app.js: disabled = pending)。利用者と同じく一時停止してから押す
        if '一時停止' in (wd.js(PAGE)['run_text'] or ''):
            wd.click('#run')
        wait(lambda: wd.js("return !document.querySelector('#record-finish').disabled"), 15, 0.2)
        wd.click('#record-finish')
        r2, t = wait(lambda: (lambda s: s if '収集完了' in (s['record'] or '') or '失敗' in (s['record'] or '') else None)(wd.js(PAGE)), 120)
        href = ((r2 or {}).get('download') or '').replace(a.base, '')      # shim は絶対 URL(location.origin + /sim/bridge/...)にする
        sid = href.rstrip('/').rsplit('/', 1)[-1] if '/datasets/download/' in href else None
        if sid:
            created['sim_sessions'].append(sid)
        return r1, r2, sid, t, href

    name_a = 'verify-sim-a-%s' % STAMP
    r1, r2, sid_a, t, href = record(name_a, 5.0)
    dl_st, dl = http('GET', href, raw=True, timeout=60) if href else (0, b'')
    check('6a 収集 A(経路追従デモ・自動走行の教師値を使う・約 5 秒)→ ZIP 作成・保存リンクは /sim/bridge/api/datasets/download/<id>', sid_a and href.startswith('/sim/bridge/api/datasets/download/')
          and dl_st == 200 and dl[:2] == b'PK', record_start=r1 and r1.get('record'), record=r2 and r2.get('record'), seconds=t, href=href,
          download_status=dl_st, download_bytes=len(dl))
    wait(lambda: wd.js("return !document.querySelector('#jetformer-check').disabled"), 15, 0.2)
    wd.click('#jetformer-check')
    BUSY = ('確認中', '送信中', '準備中', '開始します', '検証・取り込み中', '再取得')
    c1, t1 = wait(lambda: (lambda s: s if s['jf_status'] and not any(k in s['jf_status'] for k in BUSY) and any(
        k in s['jf_status'] for k in ('接続OK', 'できません', '取り込み済み ·', '同名', '接続しました')) else None)(wd.js(PAGE)), 90)
    can_upload = not wd.js("return document.querySelector('#jetformer-upload').disabled")
    if can_upload:
        wd.click('#jetformer-upload')
    c2, t2 = wait(lambda: (lambda s: s if s['jf_status'] and not any(k in s['jf_status'] for k in BUSY) and any(
        k in s['jf_status'] for k in ('取り込み完了', '取り込み済みを確認', 'できません', '未確認', '同名')) else None)(wd.js(PAGE)), 300, 1)
    res['jetformer_upload_clicked'] = can_upload
    wd.js("document.querySelector('#jetformer-link').scrollIntoView({block:'center'})")
    time.sleep(0.5)
    wd.shot('06_dataset_a_sent_to_jetformer.png')
    sims = api('GET', '/api/sim/datasets')
    item_a = next((d for d in sims['datasets'] if d['session_id'] == sid_a), None)
    ds_a = item_a and item_a.get('imported') and item_a.get('import_name')
    if ds_a:
        created['datasets'].append(ds_a)
    check('6b シミュレーター内の「JetFormerへ送る」(接続先は shim が http://127.0.0.1:8080 に固定)で JetFormer に取り込まれる',
          c2 and '取り込み完了' in (c2.get('jf_status') or '') and item_a and item_a.get('imported'), jetformer_url=c1 and c1.get('jf_url'),
          check_status=c1 and c1.get('jf_status'), seconds_check=t1, final_status=c2 and c2.get('jf_status'), seconds_upload=t2,
          imported_name=ds_a)

    # ---------------------------------------------------------------- 7. 収集 B → タブの「JetFormer に取り込む」
    name_b = 'verify-sim-b-%s' % STAMP
    r1, r2, sid_b, t, href = record(name_b, 5.0)
    check('7a 収集 B(同じ条件・約 5 秒)→ ZIP 作成', bool(sid_b), record=r2 and r2.get('record'), seconds=t, href=href)
    wd.close()                                             # シミュレーターのタブを閉じる(WebGL を 1 つだけにする)
    wd.switch(main)
    wd.js("window.jetformer.showTab('sim')")
    imp_btn = '[data-sim-import="%s"]' % sid_b
    wait(lambda: wd.js("return !!document.querySelector(arguments[0])", imp_btn), 20)
    wd.click(imp_btn)
    item_b, t = wait(lambda: (lambda d: d if d and d.get('imported') else None)(
        next((d for d in api('GET', '/api/sim/datasets')['datasets'] if d['session_id'] == sid_b), None)), 120, 1)
    item_b = item_b if isinstance(item_b, dict) and item_b.get('session_id') else None
    job = next((j for j in api('GET', '/api/jobs') if j['kind'] == 'import' and item_b and item_b['filename'] in j['title']), None)
    ds_b = item_b and item_b.get('import_name')
    if ds_b:
        created['datasets'].append(ds_b)
    wd.js("window.jetformer.loadSim()")
    tabb, _ = wait(lambda: (lambda s: s if ds_b and ds_b in s['datasets'] and '完了' in s['jobs'] else None)(wd.js(TAB)), 20)
    wd.elem_shot('#tab-sim .sim-grid > div:last-child', '07_sim_tab_datasets_jobs.png')
    check('7b タブの「JetFormer に取り込む」でジョブが完了し、一覧が「取り込み済み」になる', bool(ds_b) and job and job['state'] == 'done',
          seconds=t, imported_name=ds_b, job=job and {k: job[k] for k in ('title', 'state', 'message')}, tab_text=tabb and tabb.get('datasets', '')[:400])

    # ---------------------------------------------------------------- 8. データセット タブとシーン タブ
    wd.click('button.tab[data-tab="datasets"]')
    dtab, t = wait(lambda: wd.js("const t=document.querySelector('#dataset-list').innerText; return [...arguments].every(n => t.includes(n)) ? t.length : 0",
                                 *[n for n in (ds_a, ds_b) if n]), 20)
    wd.js("const h=[...document.querySelectorAll('#dataset-list h3')].find(e => e.textContent.includes(arguments[0])); if (h) h.scrollIntoView({block:'start'})",
          ds_b or ds_a or '')
    time.sleep(0.5)
    wd.shot('08_datasets_tab.png')
    all_ds = {d['name']: d for d in api('GET', '/api/datasets')}
    dinfo = {n: {k: all_ds.get(n, {}).get(k) for k in ('scenes', 'frames', 'source', 'tags')} for n in (ds_a, ds_b) if n}
    check('8a データセット タブに 2 つとも出る(シミュレーターの印つき)', bool(ds_a and ds_b) and dtab and all(n in all_ds for n in (ds_a, ds_b)),
          datasets=dinfo)
    scenes = [s['name'] for s in api('GET', '/api/scenes') if any(s['name'].startswith('ds-%s-' % n) for n in (ds_a, ds_b) if n)]
    created['scenes'] = scenes
    wd.click('button.tab[data-tab="scenes"]')
    time.sleep(1)
    set_value(wd, '#scene-filter', 'verify-sim-', 'input')
    stab, t = wait(lambda: wd.js("const t=document.querySelector('#tab-scenes').innerText; return [...arguments].every(n => t.includes(n)) ? 1 : 0", *scenes)
                   if scenes else 0, 20)
    time.sleep(0.5)
    wd.shot('09_scenes_tab.png')
    check('8b シーン タブに展開されたシーン(ds-<データセット名>-0001)が出る', len(scenes) >= 2 and bool(stab), scenes=scenes)

    # ---------------------------------------------------------------- 9. 確認用のデータを消す
    dele = {}
    for n in created['datasets']:
        dele['dataset ' + n] = http('DELETE', '/api/datasets/%s' % n)[0]
    for s in created['scenes']:
        dele['scene ' + s] = http('DELETE', '/api/scenes/%s' % s)[0]
    for sid in created['sim_sessions']:
        dele['sim ' + sid[:8]] = http('DELETE', '/api/sim/datasets/%s' % sid)[0]
    left_ds = [d['name'] for d in api('GET', '/api/datasets') if d['name'] in created['datasets']]
    left_sc = [s['name'] for s in api('GET', '/api/scenes') if s['name'] in created['scenes']]
    left_sim = [d['session_id'] for d in api('GET', '/api/sim/datasets')['datasets'] if d['session_id'] in created['sim_sessions']]
    raw_dirs = [str(p) for n in created['datasets'] for p in [Path('/home/jetson/jetformer/data/raw/ds-%s' % n)] if p.exists()]
    check('9 確認用のデータセット・シーン・ブリッジの作業フォルダーを消した', all(v == 200 for v in dele.values()) and not left_ds and not left_sc
          and not left_sim and not raw_dirs, deleted=dele, left=[left_ds, left_sc, left_sim, raw_dirs])

    # ---------------------------------------------------------------- 10. コースのアップロード(v2 のスナップショットの写しを新しい名前で)
    new_wv = 'v2-copy-verify-%s' % STAMP
    man = json.loads((SNAP_V2 / 'manifest.json').read_text())
    man['world_version'] = new_wv
    man['reason'] = '確認用: 旧 v2 のスナップショットの写し(world_version だけを変えたもの)'
    zpath = WORK / ('%s.zip' % new_wv)
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('%s/manifest.json' % new_wv, json.dumps(man, ensure_ascii=False, indent=1))
        for f in ('physics-world.json', 'scene.js', 'world.js'):
            z.write(SNAP_V2 / f, '%s/%s' % (new_wv, f))
    wd.click('button.tab[data-tab="sim"]')
    wait(lambda: (lambda s: s if s['active'] and len(s['courses']) >= 2 else None)(wd.js(TAB)), 20)
    wd.keys('#sim-course-file', str(zpath))
    up, t = wait(lambda: (lambda s: s if '登録' in s['upload'] else None)(wd.js(TAB)), 30)
    courses = api('GET', '/api/sim/courses')['courses']
    newc = next((c for c in courses if c['world_version'] == new_wv), None)
    cid = newc and newc['id']
    if cid:
        created['courses'].append(cid)
    tabc, _ = wait(lambda: (lambda s: s if cid and cid in s['open_courses'] else None)(wd.js(TAB)), 20)
    check('10a 画面からコースの ZIP(v2 の写し)をアップロード → 登録・一覧と選択肢に出る', newc and newc['world_hash'] == '547d7e80'
          and tabc and newc['deletable'], seconds=t, upload=up and up.get('upload'), course=newc and {k: newc[k] for k in
          ('id', 'name', 'kind', 'renderer', 'world_hash', 'stats', 'warnings', 'features')}, open_courses=tabc and tabc.get('open_courses'))
    wd.js("const e=[...document.querySelectorAll('#sim-courses .sim-course')].find(c => c.innerText.includes(arguments[0])); if (e) e.scrollIntoView({block:'center'})",
          new_wv)
    time.sleep(0.4)
    wd.shot('10a_course_uploaded.png')
    set_value(wd, '#sim-open-course', cid or '')
    set_value(wd, '#sim-open-model', '')
    note = wd.js("return document.querySelector('#sim-open-note').textContent")
    before_handles = set(wd.handles())
    wd.click('#btn-sim-open')
    newh, _ = wait(lambda: list(set(wd.handles()) - before_handles), 15)
    if not newh or isinstance(newh, dict):
        raise RuntimeError('コースのタブが開きません')
    wd.switch(newh[0])
    st, t, bad, cam = open_sim_page(wd, None, cid, '547d7e80', '10b_uploaded_course_open.png')
    check('10b 選んだコースを開く: WebGL で描画・world hash 547d7e80・帯にコース名', st['course'] == cid and st['webgl2'] and not st['fatal']
          and st['ready'] and st['hash'] == '547d7e80' and st['label'] == (newc or {}).get('name') and cid in st['options'] and not bad,
          seconds=t, href=st['href'], course=st['course'], renderer=st['course_kind_renderer'], hash=st['hash'], version=st['version'],
          label=st['label'], teachers=st['teachers'], options=st['options'], open_note=note, bad=bad, camera=cam)
    wd.close()
    wd.switch(main)
    wd.js("window.jetformer.showTab('sim')")
    delbtn = '[data-sim-delcourse="%s"]' % cid
    wait(lambda: wd.js("return !!document.querySelector(arguments[0])", delbtn), 20)
    wd.js("document.querySelector(arguments[0]).scrollIntoView({block:'center'})", delbtn)
    wd.click(delbtn)
    alert_text = wd.accept_alert()
    gone, t = wait(lambda: (lambda s: s if cid not in s['open_courses'] and not any(new_wv in m for m in s['course_meta']) else None)(wd.js(TAB)), 20)
    st_c, _ = http('GET', '/api/sim/courses/%s' % cid)
    st_p, _ = http('GET', '/sim/c/%s/' % cid, raw=True)
    if gone:
        created['courses'].remove(cid)
    wd.shot('10c_course_deleted.png')
    upload_msg = wd.js("return document.querySelector('#sim-course-upload').textContent")
    check('10c 画面の「削除」(確認ダイアログ)でコースが消える・ページも 404・「登録しました」の表示が消える', bool(gone) and st_c == 404
          and st_p == 404 and alert_text and upload_msg == '', alert=alert_text, seconds=t, api_status=st_c, page_status=st_p,
          upload_msg=upload_msg)

    # ---------------------------------------------------------------- 11. ブリッジを画面から止める
    wd.js("document.querySelector('#btn-sim-stop').scrollIntoView({block:'center'})")
    wd.click('#btn-sim-stop')
    tab3, t = wait(lambda: (lambda s: s if '停止中' in s['bridge_pill'] and '…' not in s['bridge_pill'] else None)(wd.js(TAB)), 30)
    time.sleep(0.5)
    alive = subprocess.run(['pgrep', '-f', 'bridge_runner.py'], capture_output=True, text=True).stdout.strip()
    check('11 画面の「停止」でブリッジが止まり、子プロセスが残らない', tab3 and not alive and api('GET', '/api/sim/status')['bridge']['state'] == 'stopped',
          seconds=t, pgrep=alive)
except Exception as e:  # noqa: BLE001
    import traceback
    res['exception'] = traceback.format_exc()
    log(res['exception'])
    try:
        wd.shot('99_error.png')
    except Exception:  # noqa: BLE001
        pass
finally:
    # 片付け(途中で止まっても残さない)
    for n in list(created['datasets']):
        http('DELETE', '/api/datasets/%s' % n)
    try:
        extra = [x['name'] for x in api('GET', '/api/scenes') if x['name'].startswith('ds-verify-sim-') and STAMP in x['name']]
    except Exception:  # noqa: BLE001
        extra = []
    for s in sorted(set(created['scenes']) | set(extra)):
        http('DELETE', '/api/scenes/%s' % s)
    for sid in created['sim_sessions']:
        http('DELETE', '/api/sim/datasets/%s' % sid)
    for cid in list(created['courses']):
        http('DELETE', '/api/sim/courses/%s' % cid)
    try:
        if api('GET', '/api/sim/status')['bridge']['state'] != 'stopped':
            http('POST', '/api/sim/bridge/stop', {'force': True})
    except Exception as e:  # noqa: BLE001
        log('bridge stop:', e)
    if wd.sid:
        try:
            wd.req('DELETE', '/session/%s' % wd.sid, timeout=60)
        except Exception as e:  # noqa: BLE001
            log('session end:', e)
    if gd:
        gd.terminate()
        try:
            gd.wait(10)
        except subprocess.TimeoutExpired:
            gd.kill()
    time.sleep(1)
    res['leftover_processes'] = subprocess.run(['pgrep', '-af', 'firefox|geckodriver|bridge_runner'], capture_output=True, text=True).stdout.strip()
    try:
        res['busy_after'] = busy_state()
    except Exception as e:  # noqa: BLE001
        res['busy_after'] = str(e)
    shutil.rmtree(PROFILE, ignore_errors=True)
    res['ok'] = bool(res['checks']) and all(c['ok'] for c in res['checks']) and 'exception' not in res
    res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (WORK / 'verify_result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    log('ALL OK' if res['ok'] else 'SOME NG', len(res['checks']))
