import json, time, urllib.request
W = 'http://127.0.0.1:4444'
def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(W + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())['value']
caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless']}}}}
sid = req('POST', '/session', caps)['sessionId']
js = lambda s: req('POST', f'/session/{sid}/execute/sync', {'script': s, 'args': []})
try:
    req('POST', f'/session/{sid}/url', {'url': 'http://127.0.0.1:8080/'})
    time.sleep(5)
    print(js("const b = document.querySelector('.tab[data-tab=\"train\"]'); if (b) b.click(); return [!!b, window.jetformer.state.tab]"))
    for i in range(10):
        time.sleep(1)
        r = js("const p = document.querySelector('#tr-policy'); return [p.options.length, p.value, document.querySelectorAll('#run-list .ds').length]")
        if r[0]:
            break
    print(i + 1, 's:', r)
finally:
    req('DELETE', f'/session/{sid}')
