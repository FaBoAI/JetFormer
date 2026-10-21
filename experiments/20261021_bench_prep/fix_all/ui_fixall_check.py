"""Fix-all の画面の確認(ヘッドレス Firefox、偽のサーバー)。jetformer-web(8080)・装置には触れない。
  - 走行中の「AI の時間」の行(m:ss・止める目安・5:00 を過ぎた注意・古いサーバー)
  - 「この日のステア」を再読み込みのあとも覚える(その日だけ。日付が違えば基準どおり)。サーバーへは何も送らない

    /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python ui_fixall_check.py --out <置き場> [--live URL]
  --live URL: 本物のサーバー(読むだけ。押すのは「この日のステア」の選択と再読み込みだけ。書き込みの要求を見張る)
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WD_PORT, PORT = 4447, 18083
WATCH = """
window._writes = window._writes || [];
if (!window._fetchWrapped) {
  window._fetchWrapped = true;
  const f0 = window.fetch;
  window.fetch = function(u, o) { const m = ((o && o.method) || 'GET').toUpperCase(); if (m !== 'GET') window._writes.push([m, String(u)]); return f0.apply(window, arguments); };
  const x0 = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function(m, u) { if (String(m).toUpperCase() !== 'GET') window._writes.push([m, String(u)]); return x0.apply(this, arguments); };
}
window.onerror = (m) => { (window._errs = window._errs || []).push(String(m)); };
"""


def wd(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request('http://127.0.0.1:%d%s' % (WD_PORT, path), data=data, method=method,
                               headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())['value']


def post(port, path, body):
    r = urllib.request.Request('http://127.0.0.1:%d%s' % (port, path), data=json.dumps(body).encode(), method='POST',
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=5))


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=5))


def wait_http(url, t=40):
    end = time.time() + t
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--live', default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix='fixall-ui-')
    procs, results = [], []

    def check(name, ok, detail=''):
        results.append((name, bool(ok), str(detail)[:300]))
        print(('PASS ' if ok else 'FAIL ') + name, detail if not ok else '', flush=True)
    base_url = a.live.rstrip('/') + '/' if a.live else 'http://127.0.0.1:%d/' % PORT
    try:
        if not a.live:
            procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, 'fake_server_fixall.py'), '--port', str(PORT),
                                           '--work', os.path.join(work, 'srv')]))
            assert wait_http('http://127.0.0.1:%d/api/status' % PORT)
        gd = shutil.which('geckodriver') or '/snap/bin/geckodriver'
        procs.append(subprocess.Popen([gd, '--port', str(WD_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        assert wait_http('http://127.0.0.1:%d/status' % WD_PORT)
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
                'moz:firefoxOptions': {'args': ['-headless', '--width=1440', '--height=1800']}}}}
        sid = wd('POST', '/session', caps)['sessionId']
        js = lambda s: wd('POST', '/session/%s/execute/sync' % sid, {'script': s, 'args': []})   # noqa: E731

        def shot(name):
            with open(os.path.join(a.out, name), 'wb') as f:
                f.write(base64.b64decode(wd('GET', '/session/%s/screenshot' % sid)))

        def open_drive():
            wd('POST', '/session/%s/url' % sid, {'url': base_url})
            time.sleep(2)
            js(WATCH)
            js("window.jetformer.showTab('drive')")
            time.sleep(2.5)
        try:
            wd('POST', '/session/%s/window/rect' % sid, {'width': 1440, 'height': 1800})
            open_drive()
            sel = js("return document.querySelector('#dr-base-steer').value")
            check('この日のステア: 初めは基準どおり', sel == '', sel)
            # 1.3 を選ぶ → 再読み込み → 1.3 のまま
            js("const s = document.querySelector('#dr-base-steer'); s.value = '1.3'; s.dispatchEvent(new Event('change'))")
            time.sleep(0.5)
            open_drive()
            v = js("return {sel: document.querySelector('#dr-base-steer').value, info: document.querySelector('#dr-base-info').textContent}")
            check('この日のステア: 再読み込みのあとも 1.30', v['sel'] == '1.3' and 'ステア 1.30(この日)' in v['info'], v)
            if not a.live:
                js("document.querySelector('#btn-dr-baseline').click()")
                time.sleep(1.5)
                st = js("return +document.querySelector('#dr-steer-gain').value")
                check('この日のステア: 基準の設定で 1.30 が入る', st == 1.3, st)
            # 前の日の値は使わない
            js("localStorage.setItem('jetformer.bench.daySteer', JSON.stringify({day: '2026-10-20', value: '1.3'}))")
            open_drive()
            sel = js("return document.querySelector('#dr-base-steer').value")
            check('この日のステア: 前の日の値は使わない(基準どおり)', sel == '', sel)
            # 基準どおりに戻す → 覚えた値を消す
            js("const s = document.querySelector('#dr-base-steer'); s.value = '1.3'; s.dispatchEvent(new Event('change'))")
            js("const s = document.querySelector('#dr-base-steer'); s.value = ''; s.dispatchEvent(new Event('change'))")
            ls = js("return localStorage.getItem('jetformer.bench.daySteer')")
            check('この日のステア: 基準どおりに戻すと覚えた値を消す', ls is None, ls)
            open_drive()
            check('この日のステア: 戻したあと再読み込みでも基準どおり', js("return document.querySelector('#dr-base-steer').value") == '')
            writes = js("return window._writes")
            check('この日のステアの選択・再読み込みは何も送らない(書き込みの要求 0)', writes == [], writes)
            if a.live:
                st = get(a.live.rstrip('/') + '/api/drive/status')
                check('本物のサーバー: 走っていない・zone_caps 0', st.get('running') is False and st['params'].get('zone_caps') == 0,
                      {'running': st.get('running'), 'zone_caps': st['params'].get('zone_caps')})
                check('本物のサーバー: stats に ai_s の欄がある(止まっている間は前の走行の値か空)',
                      'ai_s' in (st.get('stats') or {}) or not st.get('stats'), st.get('stats'))
            else:
                # 走行中の AI の時間(偽の状態)
                post(PORT, '/api/_fixall_status', {'running': True, 'phase': 'driving', 'message': '走行中',
                                                   'stats': {'frames': 3000, 'fps': 30.0, 'ai_s': 125.4}, 'elapsed_s': 150})
                time.sleep(1.6)
                v = js("const e = document.querySelector('#dr-ai-time'); return {t: e ? e.textContent : null,"
                       " row: [...document.querySelectorAll('#dr-stats .k')].map(x => x.textContent),"
                       " txt: document.querySelector('#dr-stats').textContent}")
                check('走行中: AI の時間の行が出る(2:05)', v['t'] == '2:05' and 'AI の時間' in v['row'], v['t'])
                check('走行中: 走行時間の次に AI の時間', v['row'].index('AI の時間') == v['row'].index('走行時間') + 1, v['row'])
                check('走行中: 止める目安(5:00)を出す', '5:00' in v['txt'] and '過ぎました' not in v['txt'], v['txt'][-200:])
                shot('fixall_ai_time.png')
                post(PORT, '/api/_fixall_status', {'running': True, 'phase': 'driving', 'message': '走行中',
                                                   'stats': {'frames': 9000, 'fps': 30.0, 'ai_s': 301.0}, 'elapsed_s': 330})
                time.sleep(1.6)
                v = js("return {t: document.querySelector('#dr-ai-time').textContent, txt: document.querySelector('#dr-stats').textContent}")
                check('5:00 を過ぎた: 止める注意を出す', v['t'] == '5:01' and '5:00 を過ぎました' in v['txt'], v)
                shot('fixall_ai_time_over.png')
                post(PORT, '/api/_fixall_status', {'running': True, 'phase': 'driving', 'message': '走行中',
                                                   'stats': {'frames': 300, 'fps': 30.0}, 'elapsed_s': 10})
                time.sleep(1.6)
                v = js("return document.querySelector('#dr-stats').textContent")
                check('古いサーバー(ai_s なし): 「古いサーバー」と出す', '古いサーバー' in v, v[-120:])
                post(PORT, '/api/_fixall_status', {})
                time.sleep(1.6)
                v = js("return document.querySelector('#dr-stats').textContent")
                check('止まっている(フレーム 0): 行を出さない', 'AI の時間' not in v, v[:80])
                calls = get('http://127.0.0.1:%d/api/_calls' % PORT)
                check('偽のサーバーへの書き込み 0', calls == [], calls)
            # スマホの幅
            wd('POST', '/session/%s/window/rect' % sid, {'width': 500, 'height': 1000})
            time.sleep(1)
            sw = js("const c = document.querySelector('#tab-drive') || document.body; return [c.scrollWidth, c.clientWidth]")
            check('狭い幅で走行タブがはみ出さない', sw[0] <= sw[1] + 1, sw)
            errs = js("return window._errs || []")
            check('JavaScript のエラーなし', not errs, errs)
        finally:
            wd('DELETE', '/session/%s' % sid)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except Exception:
                p.kill()
        shutil.rmtree(work, ignore_errors=True)
    n = sum(ok for _, ok, _ in results)
    print('%d/%d passed' % (n, len(results)))
    with open(os.path.join(a.out, 'fixall_ui_check%s.json' % ('_live' if a.live else '')), 'w', encoding='utf-8') as f:
        json.dump({'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'live': a.live, 'passed': n, 'total': len(results),
                   'results': [{'name': a_, 'ok': b_, 'detail': c_} for a_, b_, c_ in results]}, f, ensure_ascii=False, indent=1)
    return 0 if n == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
