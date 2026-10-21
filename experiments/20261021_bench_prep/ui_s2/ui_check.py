"""段 S2 の画面の確認(ヘッドレス Firefox)。偽のサーバー(fake_server.py)で本物の画面を開き、
「基準の設定」・候補のブロック・走る前のチェック・区間の速度上限のスイッチを押して、欄の値と送った API を確かめる。
走行開始のボタンは押さない(偽のサーバーの start は 409)。jetformer-web(8080)には触れない

    /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python ui_check.py --out <画像の置き場>
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
WD_PORT, PORT, PORT_OLD = 4446, 18081, 18082
B = 'clean_driving_20261011-nocmd-s3-20261011-024626'
SLOW = 'bench_20261021-slow-s0'


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
    work = tempfile.mkdtemp(prefix='s2ui-')
    procs = []
    results = []

    def check(name, ok, detail=''):
        results.append((name, bool(ok), str(detail)[:300]))
        print(('PASS ' if ok else 'FAIL ') + name, detail if not ok else '', flush=True)

    try:
        for port, extra in ((PORT, []), (PORT_OLD, ['--old'])):
            procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, 'fake_server.py'), '--port', str(port),
                                           '--work', os.path.join(work, str(port))] + extra))
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

        try:
            # ---------------- 今のサーバー(bench の API あり・上限の項目あり)
            wd('POST', '/session/%s/url' % sid, {'url': 'http://127.0.0.1:%d/' % PORT})
            wd('POST', '/session/%s/window/rect' % sid, {'width': 1440, 'height': 1800})
            time.sleep(2)
            js("window.onerror = (m) => { (window._errs = window._errs || []).push(String(m)); };")
            js("window.jetformer.showTab('drive')")
            time.sleep(3)
            v = js("return {model: document.querySelector('#dr-model').value, steer: +document.querySelector('#dr-steer-gain').value,"
                   " card: !document.querySelector('#bench-card').classList.contains('hidden'),"
                   " zone: document.querySelector('#dr-zone-caps').checked, zoneDis: document.querySelector('#dr-zone-caps').disabled,"
                   " base: document.querySelector('#dr-check-base').textContent, disk: document.querySelector('#dr-check-disk').textContent,"
                   " count: document.querySelector('#dr-check-count').textContent}")
            check('初めは推奨の B・保存値のステア 1.5', v['model'] == B and v['steer'] == 1.5, v)
            check('候補のパネルが出る', v['card'], v)
            check('区間の速度上限は OFF・押せる', v['zone'] is False and v['zoneDis'] is False, v)
            check('チェック: 基準と違う(ステア 1.5)を出す', '基準と違う' in v['base'] and 'ステア 1.50' in v['base'], v['base'])
            check('チェック: 空き容量を出す', '41.2 GB' in v['disk'], v['disk'])
            check('チェック: 0 / 6', v['count'] == '0 / 6', v['count'])
            shot('S2_drive_before.png')
            blocks = js("return [...document.querySelectorAll('#bench-days button')].map(b => [b.textContent, b.disabled, b.title])")
            check('ブロックが 6 つ並ぶ(10/23)', [b[0] for b in blocks] == ['1. B', '2. SLOW', '3. ENS', '4. B', '5. MSF', '6. B'], blocks)
            check('走れない ENS は押せない・理由', blocks[2][1] is True and 'TensorRT' in blocks[2][2], blocks[2])
            why = js("return document.querySelector('#bench-days').textContent")
            check('走れない理由を画面に出す', 'ブロック 3(ENS): 走れません' in why, why[:200])

            # 基準の設定 → 欄に入る・送らない・同期で戻らない
            js("document.querySelector('#btn-dr-baseline').click()")
            time.sleep(2.5)                                          # 表示の同期(1 秒ごと)を 2 回待つ
            v = js("return {model: document.querySelector('#dr-model').value, mode: document.querySelector('#dr-mode').value,"
                   " steer: +document.querySelector('#dr-steer-gain').value, thr: +document.querySelector('#dr-throttle-gain').value,"
                   " look: +document.querySelector('#dr-look').value, tof: document.querySelector('#dr-tof-guard').value,"
                   " ens: +document.querySelector('#dr-ens-j').value, zone: document.querySelector('#dr-zone-caps').checked,"
                   " note: document.querySelector('#dr-base-note').textContent, noteShown: !document.querySelector('#dr-base-note').classList.contains('hidden'),"
                   " base: document.querySelector('#dr-check-base').textContent, pending: Object.keys(window.jetformer.state.drive.pending)}")
            check('基準の設定: B・ジャンル 5・ステア 1.0・速度 1.0・先読み 3・ToF 1・重ね合わせ 0・上限 OFF',
                  (v['model'], v['mode'], v['steer'], v['thr'], v['look'], v['tof'], v['ens'], v['zone'])
                  == (B, '5', 1.0, 1.0, 3, '1', 0, False), v)
            check('基準の設定: 同期で戻らない(pending)', v['pending'] == ['steer_gain'], v['pending'])
            check('基準の設定: 変えた所を出す', v['noteShown'] and 'ステア 1.50 → 1.00' in v['note'] and 'まだサーバーへは送っていません' in v['note'], v['note'])
            check('チェック: 基準と同じ', '画面の値は基準と同じ' in v['base'], v['base'])
            calls = get(PORT, '/api/_calls')
            check('基準の設定はサーバーへ何も送らない', calls == [], calls)
            shot('S2_drive_baseline.png')

            # ブロック 2(SLOW)→ モデル・説明欄・収集 ON
            js("document.querySelector('#bench-days button[data-bench-i=\"1\"]').click()")
            time.sleep(1.5)
            v = js("return {model: document.querySelector('#dr-model').value, desc: document.querySelector('#dr-desc').value,"
                   " rec: document.querySelector('#dr-record').checked, steer: +document.querySelector('#dr-steer-gain').value,"
                   " active: [...document.querySelectorAll('#bench-days button.active')].map(b => b.textContent),"
                   " note: document.querySelector('#dr-base-note').textContent}")
            want = ('bench=real_bench_20261023 day=2026-10-23 block=2 cand=SLOW model=%s '
                    'battery=#__ V_before=__ V_after=__ check=__ memo=' % SLOW)
            check('ブロック 2: モデル SLOW・説明欄の雛形・収集 ON', v['model'] == SLOW and v['desc'] == want and v['rec'], v)
            check('ブロック 2: 基準の値のまま(ステア 1.0)・押したブロックを示す', v['steer'] == 1.0 and v['active'] == ['2. SLOW'], v)
            check('ブロック 2: 電池を書き足す案内', 'battery' in v['note'], v['note'])
            check('ブロックを押しても走行は始まらない・送らない', get(PORT, '/api/_calls') == [], get(PORT, '/api/_calls'))
            shot('S2_drive_block2.png')

            # この日のステア 1.3 → チェックが違いを出す → 基準の設定で 1.3 が入る
            js("const s = document.querySelector('#dr-base-steer'); s.value = '1.3'; s.dispatchEvent(new Event('change'))")
            time.sleep(1.2)
            base = js("return document.querySelector('#dr-check-base').textContent")
            check('この日のステア 1.3: チェックが違いを出す', 'ステア 1.00(基準 1.30)' in base, base)
            js("document.querySelector('#btn-dr-baseline').click()")
            time.sleep(1.2)
            v = js("return [+document.querySelector('#dr-steer-gain').value, document.querySelector('#dr-model').value,"
                   " document.querySelector('#dr-base-info').textContent]")
            check('この日のステア 1.3 で基準の設定', v[0] == 1.3 and v[1] == B and '1.30(この日)' in v[2], v)

            # 取り消す → サーバーの値に戻る(ステア 1.5・前のモデル)
            js("document.querySelector('#btn-dr-base-undo').click()")
            time.sleep(1.5)
            v = js("return [+document.querySelector('#dr-steer-gain').value, document.querySelector('#dr-model').value,"
                   " Object.keys(window.jetformer.state.drive.pending), document.querySelector('#dr-desc').value]")
            check('取り消す: サーバーの値(ステア 1.5)・前のモデルに戻る', v[0] == 1.5 and v[1] == B and v[2] == [] and v[3] == '', v)

            # 走る前のチェック: 6 つ押すと 6 / 6
            js("document.querySelectorAll('.dr-check').forEach(c => c.click())")
            time.sleep(0.3)
            check('チェック 6 / 6', js("return document.querySelector('#dr-check-count').textContent") == '6 / 6')
            js("document.querySelector('#btn-dr-check-clear').click()")
            check('チェックを全部外す', js("return document.querySelector('#dr-check-count').textContent") == '0 / 6')

            # 区間の速度上限: ON は確認を出す。キャンセルなら送らない、OK なら zone_caps 1 を送る
            js("window.confirm = () => false; document.querySelector('#dr-zone-caps').click()")
            time.sleep(1)
            check('上限 ON: 確認でやめると OFF のまま・送らない',
                  js("return document.querySelector('#dr-zone-caps').checked") is False and get(PORT, '/api/_calls') == [])
            js("window._msg = null; window.confirm = (m) => { window._msg = m; return true; }; document.querySelector('#dr-zone-caps').click()")
            time.sleep(1.5)
            calls = get(PORT, '/api/_calls')
            msg = js("return window._msg") or ''
            check('上限 ON: 確認の文に G1 と了承', 'G1 に合格し、ユーザーが了承したあと' in msg and 'R1 2.5' in msg, msg)
            check('上限 ON: zone_caps 1 を送る', calls and calls[-1] == {'path': '/api/drive/params', 'body': {'zone_caps': 1}}, calls)
            time.sleep(1.2)
            v = js("return [document.querySelector('#dr-zone-caps').checked, document.querySelector('.zone-panel').className,"
                   " document.querySelector('#dr-zone-note').textContent]")
            check('上限 ON の表示', v[0] is True and 'on' in v[1] and 'ON' in v[2], v)
            js("document.querySelector('.zone-panel').scrollIntoView({block: 'center'})")
            time.sleep(0.5)
            shot('S2_drive_zone_on.png')
            js("window.scrollTo(0, 0); document.querySelector('#bench-card details').open = true")
            time.sleep(0.5)
            js("document.querySelector('#bench-card').scrollIntoView({block: 'start'})")
            time.sleep(0.5)
            shot('S2_drive_candidates.png')
            js("window.scrollTo(0, 0)")
            # 基準の設定は上限を OFF にする(送らない・pending)
            js("document.querySelector('#btn-dr-baseline').click()")
            time.sleep(2.5)
            v = js("return [document.querySelector('#dr-zone-caps').checked, Object.keys(window.jetformer.state.drive.pending).sort(),"
                   " document.querySelector('#dr-zone-note').textContent]")
            check('基準の設定: 上限を OFF に(pending・同期で戻らない)', v[0] is False and 'zone_caps' in v[1], v)
            # 走行開始(偽のサーバーは 409)で zone_caps 0 と基準の値が送られる
            js("window.confirm = () => true; document.querySelector('#dr-dry').checked = true; document.querySelector('#btn-dr-start').click()")
            time.sleep(1.5)
            calls = get(PORT, '/api/_calls')
            st = [c for c in calls if c['path'] == '/api/drive/start']
            ok = st and st[-1]['body']['params']['zone_caps'] == 0 and st[-1]['body']['params']['steer_gain'] == 1.3 \
                and st[-1]['body']['client'] == 6 and st[-1]['body']['run'] == B and st[-1]['body']['mode_id'] == 5
            check('走行開始のボタンで(だけ)基準の値が送られる', ok, st[-1] if st else calls)
            js("document.querySelector('#dr-dry').checked = false; window.jetformer.state.drive.pending = {};")

            # スマホの幅
            wd('POST', '/session/%s/window/rect' % sid, {'width': 390, 'height': 1800})
            time.sleep(1.5)
            sw = js("return [document.documentElement.scrollWidth, window.innerWidth]")
            check('スマホの幅で横にはみ出さない', sw[0] <= sw[1] + 1, sw)
            shot('S2_drive_phone.png')
            errs = js("return window._errs || []")
            check('JavaScript のエラーなし(今のサーバー)', not errs, errs)

            # ---------------- 再起動前のサーバー(bench の API なし・上限の項目なし)
            wd('POST', '/session/%s/window/rect' % sid, {'width': 1440, 'height': 1800})
            wd('POST', '/session/%s/url' % sid, {'url': 'http://127.0.0.1:%d/' % PORT_OLD})
            time.sleep(2)
            js("window.onerror = (m) => { (window._errs = window._errs || []).push(String(m)); };")
            js("window.jetformer.showTab('drive')")
            time.sleep(3)
            v = js("return {card: !document.querySelector('#bench-card').classList.contains('hidden'),"
                   " zoneDis: document.querySelector('#dr-zone-caps').disabled, note: document.querySelector('#dr-zone-note').textContent}")
            check('古いサーバー: パネルを出さない', v['card'] is False, v)
            check('古いサーバー: 上限のスイッチは押せない・理由', v['zoneDis'] is True and '再起動' in v['note'], v)
            js("document.querySelector('#btn-dr-baseline').click()")
            time.sleep(2.5)
            v = js("return [document.querySelector('#dr-model').value, document.querySelector('#dr-mode').value,"
                   " +document.querySelector('#dr-steer-gain').value, +document.querySelector('#dr-look').value,"
                   " document.querySelector('#dr-base-info').textContent]")
            check('古いサーバー: 基準の設定は画面の控え(10/18 plan.md 3.4)で入る', v[:4] == [B, '5', 1.0, 3] and '3.4' in v[4], v)
            check('古いサーバー: 何も送らない', get(PORT_OLD, '/api/_calls') == [], get(PORT_OLD, '/api/_calls'))
            errs = js("return window._errs || []")
            check('JavaScript のエラーなし(古いサーバー)', not errs, errs)
            shot('S2_drive_oldserver.png')
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
    with open(os.path.join(a.out, 'S2_ui_check.json'), 'w', encoding='utf-8') as f:
        json.dump({'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'passed': n, 'total': len(results),
                   'results': [{'name': a_, 'ok': b_, 'detail': c_} for a_, b_, c_ in results]}, f, ensure_ascii=False, indent=1)
    return 0 if n == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
