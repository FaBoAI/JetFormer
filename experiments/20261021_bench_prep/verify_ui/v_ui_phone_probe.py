"""段 V-ui: スマホの幅(390 px)で横にはみ出す要素を探す(読むだけ)。v_ui_live.py の道具を使う

    /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python -I v_ui_phone_probe.py

- 同じ住所(192.168.55.1:8080)の style.css を開き、その中に幅 390 px の iframe で管理画面を開く。
- 走行タブとライブ・収集タブで、文書の幅と、右端が幅を超える要素を集める。GET 以外は送らない(v_ui_live の見張り)。
"""
import json
import shutil
import subprocess
import sys
import time

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261021_bench_prep/verify_ui')
import v_ui_live as V  # noqa: E402

PH = "const w = document.querySelector('#ph').contentWindow; return (new w.Function(arguments[0]))()"
PROBE = r"""
const de = document.documentElement, W = de.clientWidth;
const out = [];
for (const e of document.querySelectorAll('body *')) {
  const r = e.getBoundingClientRect();
  if (r.width > 0 && r.right > W + 0.5) {
    const cs = getComputedStyle(e);
    out.push({ el: e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.className && typeof e.className === 'string' ? '.' + e.className.trim().replace(/\s+/g, '.') : ''),
               right: Math.round(r.right), width: Math.round(r.width), pos: cs.position, ws: cs.whiteSpace,
               parent: e.parentElement ? (e.parentElement.tagName.toLowerCase() + (e.parentElement.id ? '#' + e.parentElement.id : '')) : '',
               text: (e.textContent || '').trim().slice(0, 60) });
  }
}
return { inner: window.innerWidth, client: W, scroll: de.scrollWidth, body: document.body.scrollWidth, n: out.length, items: out.slice(0, 40) };
"""


def main():
    res = {}
    gd = subprocess.Popen([V.GECKO, '--port', str(V.WD_PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wd = V.WD()
    try:
        for _ in range(60):
            try:
                wd.req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        shutil.rmtree(V.PROFILE, ignore_errors=True)
        V.PROFILE.mkdir(parents=True)
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'unhandledPromptBehavior': 'dismiss',
                'moz:firefoxOptions': {'args': ['-headless', '-profile', str(V.PROFILE)],
                                       'env': {'MOZ_HEADLESS_WIDTH': '1440', 'MOZ_HEADLESS_HEIGHT': '1000'}}}}}
        wd.sid = wd.req('POST', '/session', caps, timeout=180)['sessionId']
        V.cap_snap_scope(gd.pid, '2500M')
        wd.s('POST', '/window/rect', {'width': 1440, 'height': 1000})
        wd.s('POST', '/url', {'url': V.BASE + '/static/style.css'})
        wd.js("document.documentElement.innerHTML = '<body style=\"margin:0\"><iframe id=\"ph\" src=\"/\" style=\"width:390px;height:844px;border:0\"></iframe></body>'; return true;")
        for _ in range(60):
            if wd.js("const w = document.querySelector('#ph').contentWindow; return !!(w && w.jetformer && w.document.querySelector('button.tab[data-tab=drive]'))"):
                break
            time.sleep(0.25)
        wd.js(PH, V.GUARD)
        time.sleep(2.0)
        res['live_tab'] = wd.js(PH, PROBE)
        wd.js("document.querySelector('#ph').contentWindow.document.querySelector('button.tab[data-tab=drive]').click()")
        time.sleep(4.0)
        res['drive_tab'] = wd.js(PH, PROBE)
        res['vui'] = wd.js(PH, "return window._vui")
    finally:
        try:
            wd.req('DELETE', '/session/%s' % wd.sid, timeout=60)
        except Exception:  # noqa: BLE001
            pass
        gd.terminate()
        try:
            gd.wait(10)
        except subprocess.TimeoutExpired:
            gd.kill()
        shutil.rmtree(V.PROFILE, ignore_errors=True)
    (V.WORK / 'phone_probe.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])


if __name__ == '__main__':
    main()
