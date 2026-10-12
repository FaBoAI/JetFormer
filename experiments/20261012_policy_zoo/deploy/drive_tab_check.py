"""走行タブの既定のモデルを、本物の画面(headless Firefox)で確かめる。GET だけ(走行・センサーの開始はしない)"""
import base64, json, sys, time, urllib.request
W = 'http://127.0.0.1:4444'
OUT = sys.argv[1]
def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(W + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())['value']
caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless', '-width', '1280', '-height', '1600']}}}}
sid = req('POST', '/session', caps)['sessionId']
js = lambda s: req('POST', f'/session/{sid}/execute/sync', {'script': s, 'args': []})
try:
    req('POST', f'/session/{sid}/url', {'url': 'http://127.0.0.1:8080/'})
    time.sleep(3)
    js("showTab('drive'); return 1")
    time.sleep(4)
    r = js("""const sel = document.querySelector('#dr-model');
      return {value: sel.value, n: sel.options.length,
              first: [...sel.options].slice(0, 3).map((o) => o.textContent + (o.disabled ? ' [disabled]' : '')),
              disabled: [...sel.options].filter((o) => o.disabled).map((o) => o.textContent),
              info: document.querySelector('#dr-model-info').innerText,
              mode: document.querySelector('#dr-mode').value,
              start_disabled: document.querySelector('#btn-dr-start').disabled,
              running: !!(window.jetformer.state.drive.st && window.jetformer.state.drive.st.running)};""")
    print(json.dumps(r, ensure_ascii=False, indent=1))
    js("document.querySelector('#dr-model').scrollIntoView(); return 1")
    time.sleep(0.5)
    png = req('GET', f'/session/{sid}/screenshot')
    open(OUT, 'wb').write(base64.b64decode(png))
finally:
    req('DELETE', f'/session/{sid}')
