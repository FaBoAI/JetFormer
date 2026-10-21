"""Fix-drive の画面の確認(ヘッドレス Firefox)。段 S2 の偽のサーバー(../ui_s2/fake_server.py、変えずに使う)で本物の画面を開き、
4 つめの上限(位相が不確か)の表示・ON の確認の文・走行開始の確認の文・OFF の日の失敗の文・サーバーが断ったときの戻り方を見る。
走行開始は確認でやめる(偽のサーバーの start も 409)。jetformer-web(8080)・装置には触れない

    /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python ui_fix_check.py --out <置き場>
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
FAKE = os.path.join(HERE, '..', 'ui_s2', 'fake_server.py')
WD_PORT, PORT, PORT_OLD = 4447, 18091, 18092


def wd(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request('http://127.0.0.1:%d%s' % (WD_PORT, path), data=data, method=method,
                               headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())['value']


def get(port, path):
    return json.load(urllib.request.urlopen('http://127.0.0.1:%d%s' % (port, path), timeout=5))


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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix='fixdrive-ui-')
    procs, results = [], []

    def check(name, ok, detail=''):
        results.append((name, bool(ok), str(detail)[:400]))
        print(('PASS ' if ok else 'FAIL ') + name, detail if not ok else '', flush=True)

    try:
        for port, extra in ((PORT, []), (PORT_OLD, ['--old'])):
            procs.append(subprocess.Popen([sys.executable, FAKE, '--port', str(port), '--work', os.path.join(work, str(port))]
                                          + extra))
        gd = shutil.which('geckodriver') or '/snap/bin/geckodriver'
        procs.append(subprocess.Popen([gd, '--port', str(WD_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        assert wait_http('http://127.0.0.1:%d/api/status' % PORT) and wait_http('http://127.0.0.1:%d/api/status' % PORT_OLD)
        assert wait_http('http://127.0.0.1:%d/status' % WD_PORT)
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
                'moz:firefoxOptions': {'args': ['-headless', '--width=1440', '--height=1800']}}}}
        sid = wd('POST', '/session', caps)['sessionId']
        js = lambda s: wd('POST', '/session/%s/execute/sync' % sid, {'script': s, 'args': []})   # noqa: E731

        def shot(name):
            with open(os.path.join(a.out, name), 'wb') as f:
                f.write(base64.b64decode(wd('GET', '/session/%s/screenshot' % sid)))

        def open_drive(port):
            wd('POST', '/session/%s/url' % sid, {'url': 'http://127.0.0.1:%d/' % port})
            time.sleep(2)
            js("window.onerror = (m) => { (window._errs = window._errs || []).push(String(m)); };")
            js("window.jetformer.showTab('drive')")
            time.sleep(3)

        try:
            wd('POST', '/session/%s/window/rect' % sid, {'width': 1440, 'height': 1800})
            open_drive(PORT)
            v = js("const u = document.querySelector('#dr-zone-unknown'); return {unk: u.textContent, hidden: u.hidden,"
                   " shown: u.offsetParent !== null, spec: document.querySelector('#dr-zone-spec').textContent,"
                   " zone: document.querySelector('#dr-zone-caps').checked, dis: document.querySelector('#dr-zone-caps').disabled}")
            check('パネルに 4 つめの上限の説明が出る(サーバーの文)',
                  v['shown'] and not v['hidden'] and '4 つめ' in v['unk'] and '1 周目' in v['unk'] and '3 区間' in v['unk']
                  and '16 秒' in v['unk'], v)
            check('スイッチの横の値にも 4 つめ', 'R1 2.5' in v['spec'] and '4 つめ' in v['spec'] and '3.0' in v['spec'], v['spec'])
            check('スイッチは OFF・押せる', v['zone'] is False and v['dis'] is False, v)
            js("document.querySelector('.zone-panel').scrollIntoView({block: 'center'})")
            time.sleep(0.5)
            shot('fix_drive_zone_panel.png')

            # ON の確認の文(やめる → 送らない)
            js("window._msg = null; window.confirm = (m) => { window._msg = m; return false; };"
               " document.querySelector('#dr-zone-caps').click()")
            time.sleep(1)
            msg = js("return window._msg") or ''
            check('ON の確認の文: 4 つめの説明と、了承が 4 つめにも出ているかを聞く',
                  '4 つめの上限' in msg and '1 周目' in msg and '3 区間' in msg and '4 つめ(位相が不確かなときの上限)にも' in msg
                  and 'G1 に合格し、ユーザーが了承したあと' in msg and 'R1 2.5' in msg, msg)
            check('確認でやめると OFF のまま・何も送らない',
                  js("return document.querySelector('#dr-zone-caps').checked") is False and get(PORT, '/api/_calls') == [],
                  get(PORT, '/api/_calls'))

            # ON(OK)→ 走行開始の確認の文にも 4 つめ(確認でやめる → start は送らない)
            js("window.confirm = () => true; document.querySelector('#dr-zone-caps').click()")
            time.sleep(1.5)
            calls = get(PORT, '/api/_calls')
            check('ON(OK)で zone_caps 1 を送る', calls == [{'path': '/api/drive/params', 'body': {'zone_caps': 1}}], calls)
            js("window._msg = null; window.confirm = (m) => { window._msg = m; return false; };"
               " document.querySelector('#dr-dry').checked = false; document.querySelector('#btn-dr-start').click()")
            time.sleep(1)
            msg = js("return window._msg") or ''
            check('走行開始の確認の文: ON と 4 つめ', '区間の速度上限: ON' in msg and '4 つめ' in msg
                  and '位相が不確かなとき(1 周目のほぼ全部)も上限が掛かります' in msg, msg)
            check('走行開始は確認でやめた → start を送らない',
                  [c for c in get(PORT, '/api/_calls') if c['path'] == '/api/drive/start'] == [])

            # サーバーが ON を断った(走行中に別の画面が始めた、など)→ 理由を出し、スイッチを OFF に戻す
            js("window.confirm = () => true; document.querySelector('#dr-zone-caps').click()")      # いったん OFF へ
            time.sleep(1.5)
            js("window._realApi = window.api; window.api = async (m, u, b) => { if (u === '/api/drive/params' && b && b.zone_caps === 1)"
               " throw new Error('走行中は区間の速度上限を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)');"
               " return window._realApi(m, u, b); };")
            js("window.confirm = () => true; document.querySelector('#dr-zone-caps').click()")
            time.sleep(0.8)
            v = js("return [document.querySelector('#dr-zone-caps').checked, document.querySelector('#toast').textContent,"
                   " document.querySelector('#toast').className]")
            check('サーバーが断ったら: スイッチは OFF・理由を出す', v[0] is False and '走行中は区間の速度上限を ON にできません' in v[1]
                  and 'error' in v[2], v)
            js("window.api = window._realApi;")

            # 失敗の文: OFF の日 / ON の日(表示の関数を直接呼ぶ。次の同期の前に読む)
            v = js("const st = JSON.parse(JSON.stringify(window.jetformer.state.drive.st));"
                   " st.zone_caps_error = 'RuntimeError: 試験'; st.params.zone_caps = 0; st.running = false;"
                   " renderZoneSwitch(st); const off = document.querySelector('#dr-zone-note').textContent;"
                   " st.params.zone_caps = 1; document.querySelector('#dr-zone-caps').checked = true; renderZoneSwitch(st);"
                   " const on = document.querySelector('#dr-zone-note').textContent;"
                   " document.querySelector('#dr-zone-caps').checked = false;"
                   " return {off, on, liveOff: zoneLiveText({error: 'E', on: false}, {}), liveOn: zoneLiveText({error: 'E', on: true}, {})}")
            check('失敗の文(OFF の日): 影の判定が止まった・出力は変わらない',
                  '影の判定が計算の失敗で止まりました' in v['off'] and '出力は変わりません' in v['off'] and '外しました' not in v['off'], v)
            check('失敗の文(ON の日): この走行では上限を外しました', 'この走行では上限を外しました' in v['on'], v)
            check('走行中の表の文も OFF / ON で分ける', '影の判定' in v['liveOff'] and '外しました' not in v['liveOff']
                  and '外しました' in v['liveOn'], v)
            time.sleep(1.5)
            # スマホの幅
            wd('POST', '/session/%s/window/rect' % sid, {'width': 390, 'height': 1800})
            time.sleep(1.5)
            sw = js("return [document.documentElement.scrollWidth, window.innerWidth]")
            check('スマホの幅で横にはみ出さない', sw[0] <= sw[1] + 1, sw)
            js("document.querySelector('.zone-panel').scrollIntoView({block: 'center'})")
            time.sleep(0.5)
            shot('fix_drive_zone_phone.png')
            errs = js("return window._errs || []")
            check('JavaScript のエラーなし(今のサーバー)', not errs, errs)

            # 再起動前のサーバー: スイッチは押せない・4 つめの注意は控えの文で出す
            wd('POST', '/session/%s/window/rect' % sid, {'width': 1440, 'height': 1800})
            open_drive(PORT_OLD)
            v = js("const u = document.querySelector('#dr-zone-unknown'); return {unk: u.textContent, hidden: u.hidden,"
                   " dis: document.querySelector('#dr-zone-caps').disabled, note: document.querySelector('#dr-zone-note').textContent}")
            check('古いサーバー: スイッチは押せない・4 つめの注意は出る', v['dis'] is True and '再起動' in v['note']
                  and not v['hidden'] and '4 つめ' in v['unk'], v)
            check('古いサーバー: 何も送らない', get(PORT_OLD, '/api/_calls') == [], get(PORT_OLD, '/api/_calls'))
            errs = js("return window._errs || []")
            check('JavaScript のエラーなし(古いサーバー)', not errs, errs)
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
    with open(os.path.join(a.out, 'fix_drive_ui_check.json'), 'w', encoding='utf-8') as f:
        json.dump({'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'passed': n, 'total': len(results),
                   'results': [{'name': a_, 'ok': b_, 'detail': c_} for a_, b_, c_ in results]}, f, ensure_ascii=False, indent=1)
    return 0 if n == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
