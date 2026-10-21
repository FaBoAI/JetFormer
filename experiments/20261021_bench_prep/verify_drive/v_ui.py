#!/usr/bin/env python3
"""V-drive: 走行タブの新しいボタンが「保存」「走行開始」をしないかを、壊すつもりで確かめる(ヘッドレス Firefox)。

段 S2 の確認(ui_s2/ui_check.py)とは別に作った。違い:
  - 偽のサーバーは、/api/ の **どの道・どの方法(POST/PUT/PATCH/DELETE)** の要求も全部記録する(S2 の偽物は
    /api/drive/params と /api/drive/start の POST だけを記録していた)。
  - 走行中(running = true)の画面も作る(上限のスイッチ・基準の設定・ブロックが押せないか)。
  - キーボード(Enter・スペース)・連打・再読み込み・古いサーバー(zone_caps を知らない)も試す。
jetformer-web(8080)・装置には触れない。走行開始は偽のサーバーが 409 を返す(走らない)。

    /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python -I v_ui.py --out <置き場>
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
WD_PORT, PORT = 4471, 18093

from fastapi import Body, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from jetformer import drive  # noqa: E402
from jetformer.web import bench  # noqa: E402

spec = importlib.util.spec_from_file_location('s2_fake', str(HERE.parent / 'ui_s2' / 'fake_server.py'))
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)          # make_runs(偽の学習結果と bench_candidates.json)だけ使う
STATIC = ROOT / 'jetformer' / 'web' / 'static'
B, SLOW, FIRST = s2.B, s2.SLOW, s2.FIRST


def build(work):
    s2.make_runs(work)
    drive.RUNS_DIR = work / 'runs'
    drive.RECOMMENDED_PATH = work / 'drive_recommended.json'
    bench.BENCH_PATH = work / 'bench_candidates.json'
    S = {'calls': [], 'running': False, 'old': False,
         'params': dict(drive.DEFAULT_PARAMS, steer_gain=1.5, throttle_gain=1.0, lookahead=3, tof_guard=1)}
    app = FastAPI()
    app.include_router(bench.router)

    @app.middleware('http')
    async def record(request: Request, call_next):
        p = request.url.path
        if request.method != 'GET' and not p.startswith('/api/_'):
            body = await request.body()
            try:
                body = json.loads(body or b'null')
            except ValueError:
                body = body.decode('utf-8', 'replace')
            S['calls'].append({'method': request.method, 'path': p, 'body': body, 't': time.time()})
        return await call_next(request)

    @app.get('/')
    def index():
        return Response((STATIC / 'index.html').read_text(encoding='utf-8'), media_type='text/html',
                        headers={'Cache-Control': 'no-cache'})

    @app.get('/api/status')
    def status():
        return {'time': time.time(), 'sensors_running': False, 'sensors': {}, 'recorder': {'recording': False, 'saved_scenes': []},
                'session': {}, 'disk': {'free_gb': 41.2, 'total_gb': 250.0}, 'training': False, 'driving': S['running'],
                'config': {'scene_seconds': 20, 'vehicle_protocol': '', 'tof_unit': '', 'camera': ''}}

    @app.get('/api/drive/models')
    def models():
        return drive.list_models()

    @app.get('/api/drive/status')
    def drive_status():
        params = dict(S['params'])
        if S['old']:
            params.pop('zone_caps', None)
        out = {'phase': 'driving' if S['running'] else 'idle', 'message': '', 'error': None, 'running': S['running'],
               'params': params, 'last': {'run': FIRST, 'mode_id': 5}, 'recording': False, 'elapsed_s': 0,
               'stats': {'frames': 10, 'zone_applied': 0, 'zone_brakes': 0} if S['running'] else {},
               'now': None, 'history': {'t': []}, 'ref': None, 'run': B, 'mode_id': 5,
               'pca9685': {'address': 64, 'freq_hz': 60, 'steering_ch': 0, 'throttle_ch': 1},
               'limits': {k: list(v) for k, v in drive.PARAM_LIMITS.items()}}
        if not S['old']:
            out['zone_caps_spec'] = drive.ZONE_CAPS_SPEC
        return out

    @app.post('/api/drive/params')
    def drive_params(body: dict = Body(...)):
        for k, v in body.items():
            if k in S['params'] and not (S['old'] and k == 'zone_caps'):
                S['params'][k] = v
        p = dict(S['params'])
        if S['old']:
            p.pop('zone_caps', None)
        return p

    @app.post('/api/drive/start')
    def drive_start(body: dict = Body(...)):
        raise HTTPException(409, '確認用のサーバーなので走りません')

    @app.get('/api/_calls')
    def get_calls():
        return S['calls']

    @app.post('/api/_ctl')
    def ctl(body: dict = Body(...)):
        for k in ('running', 'old'):
            if k in body:
                S[k] = bool(body[k])
        if 'params' in body:
            S['params'].update(body['params'])
        if body.get('clear'):
            S['calls'].clear()
        return {'ok': True}

    @app.api_route('/api/{rest:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
    def anything(rest: str, request: Request):
        if request.method == 'GET' and rest in ('scenes', 'modes', 'jobs', 'datasets', 'runs', 'policies', 'models'):
            return []
        raise HTTPException(404)

    app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')
    return app


def http(method, path, body=None, port=PORT):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request('http://127.0.0.1:%d%s' % (port, path), data=data, method=method,
                               headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='vdui-'))
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(build(work), host='127.0.0.1', port=PORT, log_level='warning'))
    threading.Thread(target=server.run, daemon=True).start()
    gd = shutil.which('geckodriver') or '/snap/bin/geckodriver'
    gp = subprocess.Popen([gd, '--port', str(WD_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []

    def check(name, ok, detail=''):
        results.append({'name': name, 'ok': bool(ok), 'detail': str(detail)[:400]})
        print(('PASS ' if ok else 'FAIL ') + name, '' if ok else detail, flush=True)

    def wd(method, path, body=None):
        return http(method, path, body, WD_PORT)['value']

    try:
        for _ in range(80):
            try:
                http('GET', '/api/status')
                http('GET', '/status', port=WD_PORT)
                break
            except Exception:
                time.sleep(0.5)
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
                'moz:firefoxOptions': {'args': ['-headless', '--width=1400', '--height=1600']}}}}
        sid = wd('POST', '/session', caps)['sessionId']
        js = lambda s: wd('POST', '/session/%s/execute/sync' % sid, {'script': s, 'args': []})   # noqa: E731
        calls = lambda: http('GET', '/api/_calls')                                                # noqa: E731
        ctl = lambda **kw: http('POST', '/api/_ctl', kw)                                          # noqa: E731

        def open_page():
            wd('POST', '/session/%s/url' % sid, {'url': 'http://127.0.0.1:%d/' % PORT})
            time.sleep(2)
            js("window._errs = []; window.addEventListener('error', (e) => window._errs.push(String(e.message)));"
               "window._confirms = []; window._answer = false;"
               "window.confirm = (m) => { window._confirms.push(String(m)); return window._answer; };"
               "window.jetformer.showTab('drive')")
            time.sleep(3)

        def vals():
            return js("const q = (s) => document.querySelector(s); return {model: q('#dr-model').value, mode: q('#dr-mode').value,"
                      " steer: +q('#dr-steer-gain').value, thr: +q('#dr-throttle-gain').value, look: +q('#dr-look').value,"
                      " zone: q('#dr-zone-caps').checked, zoneDis: q('#dr-zone-caps').disabled, baseDis: q('#btn-dr-baseline').disabled,"
                      " rec: q('#dr-record').checked, desc: q('#dr-desc').value,"
                      " blocks: [...document.querySelectorAll('#bench-days button')].map(b => [b.textContent, b.disabled]),"
                      " pending: Object.keys(window.jetformer.state.drive.pending).sort(), errs: window._errs,"
                      " confirms: window._confirms.length}")

        open_page()
        v = vals()
        check('開いた直後: 何も送っていない(GET 以外 0)', calls() == [], calls())
        check('開いた直後: 区間の速度上限 OFF', v['zone'] is False and v['zoneDis'] is False, v)

        # 1) 走らない操作を全部・何度も(押す・キーボード・この日のステア・ブロック・チェック・取り消し)
        js("const q = (s) => document.querySelector(s);"
           "for (let i = 0; i < 3; i++) q('#btn-dr-baseline').click();"
           "const s = q('#dr-base-steer'); s.value = '1.3'; s.dispatchEvent(new Event('change')); q('#btn-dr-baseline').click();"
           "s.value = ''; s.dispatchEvent(new Event('change'));"
           "[...document.querySelectorAll('#bench-days button')].forEach(b => { b.click(); b.click(); });"
           "document.querySelectorAll('.dr-check').forEach(c => c.click()); q('#btn-dr-check-clear').click();")
        # キーボード: 基準の設定のボタンに焦点を当てて Enter・スペース(ボタンの既定の動き = click)
        el = wd('POST', '/session/%s/element' % sid, {'using': 'css selector', 'value': '#btn-dr-baseline'})
        eid = list(el.values())[0]
        wd('POST', '/session/%s/element/%s/value' % (sid, eid), {'text': ''})      # Enter
        wd('POST', '/session/%s/element/%s/value' % (sid, eid), {'text': ' '})
        time.sleep(3.0)                                     # 表示の同期(1 秒ごと)を 3 回
        v = vals()
        c = calls()
        check('基準の設定・ブロック・この日のステア・チェック・キーボード: サーバーへ何も送らない', c == [], c)
        check('同期で戻らない(pending のまま)', v['steer'] == 1.0 and 'steer_gain' in v['pending'], v)
        check('走行開始の確認(confirm)も出ていない', v['confirms'] == 0, v)
        check('JavaScript のエラーなし', v['errs'] == [], v['errs'])
        js("document.querySelector('#btn-dr-base-undo').click()")
        time.sleep(1.5)
        v = vals()
        check('取り消し: 送らない・サーバーの値(ステア 1.5)に戻る', calls() == [] and v['steer'] == 1.5 and v['pending'] == [], v)

        # 2) 区間の速度上限: 確認でやめる → 送らない。強引に change を出しても(確認 false)送らない
        js("window._answer = false; document.querySelector('#dr-zone-caps').click();"
           "const z = document.querySelector('#dr-zone-caps'); z.checked = true; z.dispatchEvent(new Event('change'));")
        time.sleep(1.5)
        v = vals()
        check('上限: 確認でやめたら送らない・OFF のまま', calls() == [] and v['zone'] is False, (calls(), v))

        # 3) 走行中(別の画面から始めた)の画面: 基準の設定・ブロック・上限の ON は押せない・送らない
        js("document.querySelector('#btn-dr-baseline').click()")       # pending を作ってから走り出す
        time.sleep(0.5)
        ctl(running=True)
        time.sleep(2.5)
        v = vals()
        check('走行中: pending は空(サーバーの値が正)', v['pending'] == [], v['pending'])
        check('走行中: 基準の設定は押せない・ブロックも押せない',
              v['baseDis'] is True and all(b[1] for b in v['blocks']), v)
        check('走行中: 上限のスイッチは押せない(OFF のとき)', v['zoneDis'] is True, v)
        js("window._answer = true; const z = document.querySelector('#dr-zone-caps');"
           "z.disabled = false; z.click();"                                   # 押せないはずのものを無理に押す
           "document.querySelector('#btn-dr-baseline').disabled = false; document.querySelector('#btn-dr-baseline').click();"
           "[...document.querySelectorAll('#bench-days button')].forEach(b => { b.disabled = false; b.click(); });")
        time.sleep(1.5)
        c = calls()
        check('走行中: 無理に押しても上限 ON・値は送らない', c == [], c)
        v = vals()
        check('走行中: 上限のスイッチは OFF のまま', v['zone'] is False, v)
        ctl(params={'zone_caps': 1})
        time.sleep(1.5)
        v = vals()
        check('走行中・サーバーが ON: OFF にはできる(押せる)', v['zone'] is True and v['zoneDis'] is False, v)
        js("document.querySelector('#dr-zone-caps').click()")
        time.sleep(1.0)
        c = calls()
        check('走行中: OFF にすると zone_caps 0 だけを送る', c == [{'method': 'POST', 'path': '/api/drive/params',
                                                               'body': {'zone_caps': 0}, 't': c[0]['t']}] if c else False, c)
        ctl(running=False, clear=True)
        time.sleep(1.5)

        # 4) サーバーが ON のまま再読み込み → 画面も ON(サーバーに合わせる)。基準の設定 → OFF(pending)→ 走行開始で 0
        ctl(params={'zone_caps': 1})
        open_page()
        v = vals()
        check('再読み込み: サーバーの ON に合わせる(勝手に OFF/ON にしない)', v['zone'] is True and calls() == [], (v, calls()))
        js("document.querySelector('#btn-dr-baseline').click()")
        time.sleep(2.5)
        v = vals()
        check('基準の設定: 上限を OFF に(pending・同期で戻らない)', v['zone'] is False and 'zone_caps' in v['pending'], v)
        js("window._answer = true; document.querySelector('#btn-dr-start').click()")
        time.sleep(1.5)
        c = calls()
        st = [x for x in c if x['path'] == '/api/drive/start']
        check('走行開始のボタン 1 回 = start 1 回・zone_caps 0・ステア 1.0',
              len(st) == 1 and len(c) == 1 and st[0]['body']['params']['zone_caps'] == 0
              and st[0]['body']['params']['steer_gain'] == 1.0 and st[0]['body']['run'] == B, c)
        ctl(clear=True, params={'zone_caps': 0})

        # 5) サーバーが OFF に戻ったあとで再読み込み → 上限 OFF。確認なしの ON は無い
        open_page()
        v = vals()
        check('再読み込み(サーバー OFF): OFF', v['zone'] is False and calls() == [], v)

        # 6) 古いサーバー(zone_caps を知らない): スイッチは押せない。走行開始の値に zone_caps 0(読み飛ばされる)
        ctl(old=True)
        time.sleep(2.0)
        v = vals()
        check('古いサーバー: 上限のスイッチは押せない・OFF', v['zoneDis'] is True and v['zone'] is False, v)
        js("window._answer = true; const z = document.querySelector('#dr-zone-caps'); z.disabled = false; z.click();")
        time.sleep(1.5)
        c = calls()
        check('古いサーバー: 無理に押すと zone_caps 1 を送るが、サーバーは知らない項目として捨て、画面は OFF に戻す',
              (not c or c[-1]['body'] == {'zone_caps': 1}) and vals()['zone'] is False, (c, vals()))
        ctl(old=False, clear=True)
        v = vals()
        check('全体: JavaScript のエラーなし', v['errs'] == [], v['errs'])
        wd('DELETE', '/session/%s' % sid)
    finally:
        gp.terminate()
        server.should_exit = True
        shutil.rmtree(work, ignore_errors=True)
    with open(os.path.join(a.out, 'v_ui.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    n = sum(r['ok'] for r in results)
    print('%d / %d PASS' % (n, len(results)))


if __name__ == '__main__':
    main()
