"""管理画面(走行タブ)のブラウザ確認。geckodriver --port 4444 を起動してから実行する(確認モードで走行を開始・停止する)

    geckodriver --port 4444 &
    python tools/ui_check.py
"""
import base64
import json
import time
import urllib.request

W = 'http://127.0.0.1:4444'
def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(W + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=90).read())['value']
def api(path):
    return json.load(urllib.request.urlopen('http://127.0.0.1:8080' + path, timeout=5))
caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
        'moz:firefoxOptions': {'args': ['-headless', '--width=1440', '--height=1000']}}}}
sid = req('POST', '/session', caps)['sessionId']
js = lambda s: req('POST', f'/session/{sid}/execute/sync', {'script': s, 'args': []})
def shot(name):
    open(name, 'wb').write(base64.b64decode(req('GET', f'/session/{sid}/screenshot')))
results = []
def check(name, ok, detail=''):
    results.append(ok); print(('PASS ' if ok else 'FAIL ') + name, detail)
try:
    req('POST', f'/session/{sid}/url', {'url': 'http://127.0.0.1:8080/'})
    req('POST', f'/session/{sid}/window/rect', {'width': 1440, 'height': 1000})
    time.sleep(2); js("window.jetformer.showTab('drive')"); time.sleep(2.5)
    v = js("return [document.querySelector('#dr-model').value, document.querySelector('#dr-mode').value, +document.querySelector('#dr-steer-gain').value, +document.querySelector('#dr-throttle-gain').value]")
    s0 = api('/api/drive/status')
    want = [s0['last']['run'], str(s0['last']['mode_id']), s0['params']['steer_gain'], s0['params']['throttle_gain']]
    check('restores last model/genre/gains', v == want, (v, want))
    # 確認モードで開始 → このページが操作している
    js("document.querySelector('#dr-dry').checked = true; document.querySelector('#btn-dr-start').click()")
    time.sleep(7)
    st = api('/api/drive/status')
    check('dry run started from the page', st['running'] and st['phase'] == 'driving', st['message'])
    warn = js("return document.querySelector('#dr-message').textContent")
    check('this page is the controller (no warning)', '別の画面' not in warn, warn[:60])
    shot('ui_drive_running.png')
    # ジャンルを選んだ直後(セレクトにフォーカス)でもスペースで止まる
    js("const s = document.querySelector('#dr-mode'); s.focus(); s.dispatchEvent(new KeyboardEvent('keydown', {key: ' ', code: 'Space', bubbles: true, cancelable: true}))")
    time.sleep(1.5)
    st = api('/api/drive/status')
    check('Space stops while genre select focused', not st['running'], st['message'])
    # もう一度開始して、シーン詳細を開いた状態の Escape は詳細を閉じるだけ
    js("document.querySelector('#btn-dr-start').click()"); time.sleep(7)
    scene = api('/api/scenes')[-1]['name']
    js(f"window.jetformer.openScene('{scene}')"); time.sleep(2)
    js("document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', code: 'Escape', bubbles: true, cancelable: true}))"); time.sleep(1.5)
    st = api('/api/drive/status'); modal = js("return !document.querySelector('#modal').classList.contains('hidden')")
    check('Escape in scene modal closes modal only', st['running'] and not modal, (st['running'], modal))
    # 別のタブでも画面下の停止ボタンが出て、押すと止まる
    js("window.jetformer.showTab('live')"); time.sleep(2)
    vis = js("return !document.querySelector('#btn-float-stop').classList.contains('hidden')")
    check('floating stop visible on live tab', vis)
    shot('ui_live_floatstop.png')
    js("document.querySelector('#btn-float-stop').click()"); time.sleep(1.5)
    check('floating stop stops the drive', not api('/api/drive/status')['running'])
    # トーストが停止ボタンのクリックを邪魔しない
    pe = js("return getComputedStyle(document.querySelector('#toast')).pointerEvents")
    check('toast does not capture clicks', pe == 'none', pe)
finally:
    try:
        if api('/api/drive/status')['running']:
            urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8080/api/drive/stop', data=b'{}', method='POST', headers={'Content-Type': 'application/json'}))
    except Exception: pass
    req('DELETE', f'/session/{sid}')
print('%d/%d passed' % (sum(results), len(results)))
