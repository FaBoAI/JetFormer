"""10/24 16:00 の前: 走行タブの「ベンチマーク」で速さの段を選ぶと、モデル・速度ゲイン・ジャンル・発進の補助・説明欄が入るかを、
本物の画面(LAN の住所)でヘッドレス Firefox で確かめる。走行開始・保存は押さない(書き込みの要求は GUARD で止めて記録する)。

    /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python -I ui_tune_check.py
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261021_bench_prep/verify_ui')
import v_ui_live as V  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_20261024_s1/ui_check')
V.OUTDIR = OUT
V.WD_PORT = 4461
PROFILE = OUT / 'ff_profile'
READ = """const q = (s) => document.querySelector(s);
return {model: q('#dr-model').value, mode: q('#dr-mode').value, thr: +q('#dr-throttle-gain').value, steer: +q('#dr-steer-gain').value,
        look: +(q('#dr-lookahead') || {value: NaN}).value, kick: q('#dr-start-kick').checked, zone: q('#dr-zone-caps').checked, dry: q('#dr-dry').checked,
        rec: q('#dr-record').checked, desc: q('#dr-desc').value, pick: q('#dr-bench-pick').value,
        opts: [...q('#dr-bench-pick').options].map((o) => o.textContent), vui: window.__vui || null};"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(Path('/home/jetson/jetformer/data/bench_candidates.json').read_text(encoding='utf-8'))
    day = [x for x in d['days'] if x['date'] == time.strftime('%Y-%m-%d')][0]
    runs = {c['id']: c['run'] for c in d['candidates']}
    st0 = V.get(V.LOCAL, '/api/drive/status')
    sha0 = V.sha(V.SETTINGS)
    gd = subprocess.Popen([V.GECKO, '--port', str(V.WD_PORT)], stdout=open(OUT / 'geckodriver.log', 'w'), stderr=subprocess.STDOUT)
    wd = V.WD()
    ok_all = True
    try:
        for _ in range(60):
            try:
                wd.req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        if PROFILE.exists():
            shutil.rmtree(PROFILE)
        PROFILE.mkdir(parents=True)
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'unhandledPromptBehavior': 'dismiss',
                'moz:firefoxOptions': {'args': ['-headless', '-profile', str(PROFILE)],
                                       'env': {'MOZ_HEADLESS_WIDTH': '1440', 'MOZ_HEADLESS_HEIGHT': '1000'}}}}}
        wd.sid = wd.req('POST', '/session', caps, timeout=180)['sessionId']
        wd.s('POST', '/url', {'url': V.BASE + '/'})
        wd.js(V.GUARD)
        for _ in range(40):
            if wd.js("return !!(window.jetformer && document.querySelector('button.tab[data-tab=drive]'))"):
                break
            time.sleep(0.25)
        wd.click('button.tab[data-tab="drive"]')
        for _ in range(60):
            if wd.js("return document.querySelector('#dr-bench-pick').options.length > 1"):
                break
            time.sleep(0.25)
        time.sleep(2.0)
        s = wd.js(READ)
        print('options:', s['opts'])
        for bi, blk in enumerate(day['blocks']):
            di = d['days'].index(day)
            wd.js("const s = document.querySelector('#dr-bench-pick'); s.value = arguments[0]; s.dispatchEvent(new Event('change'));",
                  '%d:%d' % (di, bi))
            time.sleep(0.6)
            r = wd.js(READ)
            want_thr = (blk.get('params') or {}).get('throttle_gain', d['baseline']['params']['throttle_gain'])
            want_mode = str(blk.get('mode_id', d['baseline']['mode_id']))
            ok = (r['model'] == runs[blk['id']] and abs(r['thr'] - want_thr) < 1e-6 and r['mode'] == want_mode and r['kick']
                  and not r['zone'] and not r['dry'] and r['rec'] and 'official=1' in r['desc']
                  and ('throttle_gain=%s' % want_thr) in r['desc'])
            ok_all &= ok
            print(('OK ' if ok else 'NG '), blk['block'], blk['id'], 'thr', r['thr'], 'mode', r['mode'], 'kick', r['kick'], 'zone', r['zone'],
                  'dry', r['dry'], 'steer', r['steer'], 'look', r['look'], '| desc', r['desc'][-90:])
            if bi in (1, 4, 6):
                wd.shot('ui_tune_block%d.png' % blk['block'])
        v = wd.js('return window.__vui || null')
        print('writes', (v or {}).get('writes'), 'confirms', (v or {}).get('confirms'))
        ok_all &= not (v or {}).get('writes')
    finally:
        try:
            wd.req('DELETE', '/session/%s' % wd.sid)
        except Exception:  # noqa: BLE001
            pass
        gd.terminate()
    st1 = V.get(V.LOCAL, '/api/drive/status')
    same = st1['params'] == st0['params'] and V.sha(V.SETTINGS) == sha0
    print('server params & drive_settings unchanged:', same)
    print('ALL OK' if ok_all and same else 'NOT OK')


if __name__ == '__main__':
    main()
