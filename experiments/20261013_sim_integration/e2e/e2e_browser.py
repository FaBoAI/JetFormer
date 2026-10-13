"""E2E 確認(design.md §8.3 の 10 を広げたもの): Firefox でシミュレーターのページを開き、shim・コース・モデルの受け渡しを確かめる

    jf-heavy --mem 2500M --lock big -- python -I e2e_browser.py --base http://127.0.0.1:18080 [--display :0]

- geckodriver(snap)をポート 4447 で起動し、専用のプロファイルで Firefox を開く。--display :0 なら WebGL が使える。
- 終わったら Firefox と geckodriver を必ず止める。
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GECKO = '/snap/bin/geckodriver'
PORT = 4447
OUT = Path('/home/jetson/jetformer_reports/sim_integration_20261013/e2e/browser')
PROFILE = Path('/home/jetson/jetformer_reports/sim_integration_20261013/e2e/ff_profile')

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='http://127.0.0.1:18080')
ap.add_argument('--display', default='')
ap.add_argument('--run', default='clean_driving_20261011-jetformer_v0-s1-ft-real-lr1e-4-e12')
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
PROFILE.mkdir(parents=True, exist_ok=True)
res = {'checks': [], 'display': a.display or 'headless'}


def check(name, ok, **info):
    res['checks'].append(dict(name=name, ok=bool(ok), **info))
    print(('OK  ' if ok else 'NG  ') + name, json.dumps(info, ensure_ascii=False)[:500], flush=True)


def api(method, path, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(a.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b'null')


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
            raise RuntimeError('%s %s: %s %r' % (method, path, e.code, e.read()[:500]))

    def s(self, method, path, body=None, timeout=120):
        return self.req(method, '/session/%s%s' % (self.sid, path), body, timeout)

    def js(self, script, *args):
        return self.s('POST', '/execute/sync', {'script': script, 'args': list(args)})

    def get(self, url):
        return self.s('POST', '/url', {'url': url})

    def shot(self, name):
        (OUT / name).write_bytes(base64.b64decode(self.s('GET', '/screenshot')))


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


PAGE_STATE = r"""
const cs = window.courseSimulator;
const q = (s) => document.querySelector(s);
return {app: !!cs, fatal: q('#fatal') ? (q('#fatal').hidden ? '' : q('#fatal').textContent) : null,
  bar: !!q('#jf-simbar'), options: q('#jf-simbar select') ? [...q('#jf-simbar select').options].map(o => o.value) : [],
  label: q('.view-label') && q('.view-label').textContent, size: q('.view-note span') && q('.view-note span').textContent,
  teachers: q('#teacher-profile') ? [...q('#teacher-profile').options].map(o => o.value) : null,
  planHidden: q('#collection-plan') ? q('#collection-plan').hidden : null, endpointReadOnly: q('#endpoint') && q('#endpoint').readOnly,
  endpoint: q('#endpoint') && q('#endpoint').value, conn: q('#connection-status') && q('#connection-status').textContent,
  upload: q('#upload-status') && q('#upload-status').textContent, model: q('.jf-model') && q('.jf-model').textContent,
  note: q('.jf-note') && q('.jf-note').textContent, bridge: q('.jf-pill') && q('.jf-pill').textContent,
  ready: cs ? cs.sensorSummary().ready : false, hash: cs ? cs.report().world_hash : null,
  version: cs ? cs.report().context.world_version : null, title: document.title, status: q('#status') && q('#status').textContent,
  jfurl: localStorage.getItem('course-jetformer-url')};
"""

gd = None
wd = WD()
try:
    env = dict(os.environ)
    if a.display:
        env.update(DISPLAY=a.display, XAUTHORITY='/run/user/1000/gdm/Xauthority')
    gd = subprocess.Popen([GECKO, '--port', str(PORT)], stdout=open(OUT / 'geckodriver.log', 'w'), stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        try:
            wd.req('GET', '/status', timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    args = ['-profile', str(PROFILE)] + ([] if a.display else ['-headless'])
    fenv = {'MOZ_HEADLESS_WIDTH': '1400', 'MOZ_HEADLESS_HEIGHT': '1000'}
    if a.display:
        fenv.update(DISPLAY=a.display, XAUTHORITY='/run/user/1000/gdm/Xauthority')
    caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
                                             'moz:firefoxOptions': {'args': args, 'env': fenv}}}}
    v = wd.req('POST', '/session', caps, timeout=180)
    wd.sid = v['sessionId']
    res['browser'] = {k: v['capabilities'].get(k) for k in ('browserVersion', 'moz:headless')}
    wd.s('POST', '/window/rect', {'width': 1400, 'height': 1000})

    # 0. JetFormer の画面の「シミュレーター」タブ
    wd.get(a.base + '/')
    time.sleep(2)
    wd.js("window.jetformer.showTab('sim')")
    tab, _ = wait(lambda: wd.js("const t=document.querySelector('#sim-release').textContent; const n=document.querySelectorAll('#sim-courses .sim-course').length;"
                                "return t.includes('20261013-144013-6f610dd5') && n >= 2 ? {release: t, courses: n, env: document.querySelector('#sim-env').textContent,"
                                "open: [...document.querySelector('#sim-open-course').options].map(o=>o.value), models: document.querySelector('#sim-open-model').options.length} : null"), 20)
    check('0 管理画面のシミュレーター タブが表示される', tab and tab.get('courses', 0) >= 2 and '1.31.0' in tab.get('env', ''), tab=tab)
    wd.shot('00_jetformer_sim_tab.png')

    # コースを 1 つ登録(Default の雛形から world_version と板 1 枚を変える)
    import io
    import zipfile
    with urllib.request.urlopen(a.base + '/api/sim/courses/default/package.zip') as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    course = json.loads(z.read('default/course.json'))
    world = json.loads(z.read('default/physics-world.json'))
    course.update(world_version='browser-course-1', id='browser-course-1', name='ブラウザ確認コース')
    course.pop('world_hash', None)
    world['segments'].append({'a': [-1.0, -1.4], 'b': [-0.6, -1.4], 'radius': 0.0095, 'height': 0.119, 'kind': 'rail'})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zz:
        zz.writestr('course.json', json.dumps(course, ensure_ascii=False))
        zz.writestr('physics-world.json', json.dumps(world))
    req = urllib.request.Request(a.base + '/api/sim/courses/import?filename=b.zip', data=buf.getvalue(), method='PUT')
    with urllib.request.urlopen(req) as r:
        reg = json.loads(r.read())['course']
    cid, chash = reg['id'], reg['world_hash']
    api('POST', '/api/sim/bridge/start')
    wait(lambda: api('GET', '/api/sim/status')['bridge']['state'] == 'running', 30)

    # 1. アップロードしたコース(簡易描画)+ ?model= で JetFormer のモデルを読み込む
    wd.get('%s/sim/c/%s/?model=%s' % (a.base, cid, a.run))
    st, t_app = wait(lambda: (lambda s: s if s['app'] or s['fatal'] else None)(wd.js(PAGE_STATE)), 60)
    check('1a 簡易描画のコースのページが動く(全モジュールを読み込む)', st and st['app'], seconds=t_app, fatal=st and st.get('fatal'))
    res_list = wd.js("return performance.getEntriesByType('resource').map(e => [e.name.replace(location.origin, ''), e.responseStatus || 0])")
    bad = [r for r in res_list if r[1] and r[1] >= 400 and r[0] != '/favicon.ico']      # favicon はブラウザが勝手に取りに行く
    check('1b 読み込んだファイルがすべて 200(scene.js・world.js・_jf/course-data.js を含む)', not bad and
          any(r[0].endswith('/scene.js') for r in res_list) and any('course-data.js' in r[0] for r in res_list),
          resources=len(res_list), bad=bad)
    ready, t_ready = wait(lambda: wd.js(PAGE_STATE)['ready'], 30)
    st = wd.js(PAGE_STATE)
    check('1c テクスチャを読み終えて「評価を開始」が使える(assetsReady)', ready and not wd.js("return document.querySelector('#run').disabled"),
          seconds=t_ready)
    check('1d world hash が登録時と同じ', st['hash'] == chash and st['version'] == 'browser-course-1', page=st['hash'], registered=chash)
    check('1e 帯・コース名・教師の制限・収集条件を隠す・接続先は読み取り専用', st['bar'] and st['label'] == 'ブラウザ確認コース'
          and st['teachers'] == ['slow', 'speed'] and st['planHidden'] and st['endpointReadOnly'] and len(st['options']) >= 3
          and st['jfurl'] == 'http://127.0.0.1:8080',
          label=st['label'], teachers=st['teachers'], options=st['options'], title=st['title'], size=st['size'])
    loaded, t_model = wait(lambda: api('GET', '/api/sim/status')['bridge'].get('policy_loaded'), 180, 1)
    time.sleep(1)
    st = wd.js(PAGE_STATE)
    check('1f ?model= の学習済みモデルが自動で読み込まれる(ZIP → bundle → activate)', loaded and 'model=' not in wd.js('return location.search'),
          seconds=t_model, upload=st['upload'], model_bar=st['model'], conn=st['conn'])
    wd.shot('01_data_course_model.png')
    rw = wd.js("return [window.JETFORMER_SIM_REWRITE(document.querySelector('#endpoint').value + '/api/status'), window.JETFORMER_SIM_REWRITE('/api/models')]")
    check('1g 接続先の /api/ だけを /sim/bridge/api/ に書き換える', rw[0] == a.base + '/sim/bridge/api/status' and rw[1] is None, rewrite=rw)

    # 2. 学習済み AI で数秒走らせる(推論は中継経由でブリッジ)
    before = api('GET', '/api/sim/bridge/log?lines=1')['suppressed']['predict']
    wd.js("const c=document.querySelector('#controller'); c.value='external'; c.dispatchEvent(new Event('change',{bubbles:true}));")
    time.sleep(1)
    wd.js("document.querySelector('#run').click()")
    time.sleep(8)
    st = wd.js(PAGE_STATE)
    elapsed = wd.js("return document.querySelector('#elapsed').textContent")
    after = api('GET', '/api/sim/bridge/log?lines=1')['suppressed']['predict']
    wd.shot('02_data_course_ai_running.png')
    wd.js("const r=document.querySelector('#run'); if (!r.disabled) r.click();")
    check('2 AI 評価で走る(推論が中継経由でブリッジに届く)', after - before >= 5 and float(elapsed or 0) > 0,
          predicts=after - before, sim_elapsed_s=elapsed, status=st['status'][:120])

    # 3. Default と同梱 v2
    for cid2, want in (('default', '0a464384'), ('indoor-course-eval-2-shortcut', '547d7e80')):
        wd.get('%s/sim/c/%s/' % (a.base, cid2))
        st, t = wait(lambda: (lambda s: s if (s['app'] and s['ready']) or s['fatal'] else None)(wd.js(PAGE_STATE)), 60)
        check('3 %s が動き world hash が %s' % (cid2, want), st and st['hash'] == want and st['ready'],
              seconds=t, hash=st and st['hash'], teachers=st and st['teachers'], label=st and st['label'])
        wd.shot('03_%s.png' % cid2)
finally:
    try:
        api('POST', '/api/sim/bridge/stop', {'force': True})
    except Exception as e:  # noqa: BLE001
        print('bridge stop:', e)
    try:
        api('DELETE', '/api/sim/courses/browser-course-1')
    except Exception:  # noqa: BLE001
        pass
    if wd.sid:
        try:
            wd.req('DELETE', '/session/%s' % wd.sid, timeout=60)
        except Exception as e:  # noqa: BLE001
            print('session end:', e)
    if gd:
        gd.terminate()
        try:
            gd.wait(10)
        except subprocess.TimeoutExpired:
            gd.kill()
res['ok'] = all(c['ok'] for c in res['checks'])
(OUT / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print('ALL OK' if res['ok'] else 'SOME NG', len(res['checks']))
