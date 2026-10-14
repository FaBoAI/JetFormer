"""経路いろいろ r4b の候補の F-ID をブラウザ(Firefox)で確かめる(2026-10-14 夜、line r4 の段のやり直し)。

確認 line r3 の再確認の指摘 4「F-ID 2 つが Node とブラウザで同じことを確かめる」のうち、この段でできること:
  - ユーザーの Mac と同じ非ループバックのページ http://192.168.55.1:8080/sim/c/default/ を snap の Firefox(DISPLAY :0、geckodriver 4448)で開き、
  - 候補の走行中の仕組み(candidates/line_r4/runtime/ = r3/line4b/rt4b の写し。共有の web/rl と違うのは shield.mjs・mode_runtime.mjs だけ)を、
    この段で立てる読み出しだけの小さな HTTP の場(192.168.55.1:8457、CORS つき、終わったら止める)から import し、
  - F-ID(Node で作った 600 フレームの snapshot と操作の列)を 1 フレームずつ教師に与えて、統合の段(integrate/browser_check.py)と同じ決まりで比べる:
    操作の |Δ| ≤ 1e-6・info(ラベルとシールドの印)が文字列として同じ・シールドの替え(sh)がどのフレームも同じ・metadata が同じ。
    文字列として全部同じフレームの数も出す(Node と Firefox の Math の最後の桁 1e-16 ほどの違いは、ほかのモードの F-ID と同じく許す)。1 フレームの時間も測る。
共有のファイル(web/rl・teacher.js・bundle)は変えない。取り込んだあとのページの teacher.js を通した確かめは昇格の段。
jf-heavy --mem 2500M --lock big の中で実行すること。

    /home/jetson/bin/jf-heavy --mem 2500M --lock big --need-mb 1500 -- /home/jetson/miniconda/envs/jetracer/bin/python -u r3/line4b/browser_fid_ln4b.py \
        --cand r3/candidates/line_r4 --out ~/jetformer_reports/sim_rl_modes_20261013/r3/line_r4b/browser
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPORTS = Path('/home/jetson/jetformer_reports/sim_rl_modes_20261013')
PROFILE = REPORTS / 'r3/line_r4b/ff_profile'
GECKO, GECKO_PORT = '/snap/bin/geckodriver', 4448
BASE = 'http://192.168.55.1:8080'
SERVE_HOST, SERVE_PORT = '192.168.55.1', 8457
XAUTH = '/run/user/1000/gdm/Xauthority'


class WebDriver:
    def __init__(self, port):
        self.base = f'http://127.0.0.1:{port}'
        self.sid = None

    def _req(self, method, path, body=None, timeout=300):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b'{}').get('value')
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{method} {path}: {e.code} {e.read()[:800]!r}') from None

    def start(self, args, prefs, env):
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {'args': args, 'prefs': prefs, 'env': env}}}}
        v = self._req('POST', '/session', caps, timeout=180)
        self.sid = v['sessionId']
        return v.get('capabilities', {})

    def s(self, method, path, body=None, timeout=300):
        return self._req(method, f'/session/{self.sid}{path}', body, timeout)

    def get(self, url):
        return self.s('POST', '/url', {'url': url})

    def js(self, script, *args, timeout=300):
        return self.s('POST', '/execute/sync', {'script': script, 'args': list(args)}, timeout)

    def js_async(self, script, *args, timeout=900):
        return self.s('POST', '/execute/async', {'script': script, 'args': list(args)}, timeout)

    def quit(self):
        if self.sid:
            try:
                self._req('DELETE', f'/session/{self.sid}', timeout=60)
            except Exception as e:  # noqa: BLE001
                print('[fid] セッション終了の失敗:', e, flush=True)
            self.sid = None


class CorsHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json'}

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):  # 静かに
        pass


FID_JS = r"""
const done = arguments[arguments.length - 1];
const [base, fidName, policyName] = arguments;
(async () => {
  try {
    const RT = await import(base + '/runtime/mode_runtime.mjs');
    const fid = await (await fetch(base + '/' + fidName)).json();
    const policyText = await (await fetch(base + '/' + policyName)).text();
    const weights = JSON.parse(policyText);
    // 非ループバックの http は安全な文脈でないので crypto.subtle が無い(ユーザーの Mac と同じ)。方策の sha256 は配った policy.json を Python で照らす
    const sha = (window.isSecureContext && crypto.subtle) ? [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(policyText)))].map(b => b.toString(16).padStart(2, '0')).join('') : null;
    const t = RT.createModeTeacher(fid.world, fid.frames[0].snapshot.config, fid.variation,
      {mode: 'line', weights, params: fid.params, policySha256: fid.policy_sha256, modeRuntime: fid.mode_runtime});
    // 比べ方(統合の段の browser_check.py と同じ決まり): 操作の |Δ| ≤ 1e-6・info(jf のラベル・シールドの印)が文字列として同じ・metadata が同じ。
    // 文字列として全部同じフレームの数(exact_same)と、シールドの替え(sh)が同じフレームの数も数える(Node と Firefox の Math の最後の桁の違いを分けるため)
    let same = 0, diff = 0, first = null, maxS = 0, maxV = 0, shield = 0, infoMis = 0, firstInfo = null, shSame = 0, ms = [];
    for (const [k, fr] of fid.frames.entries()) {
      const t0 = performance.now();
      const a = t.action(fr.snapshot);
      ms.push(performance.now() - t0);
      if (a.info && a.info.jf && a.info.jf.sh) shield++;
      const b = fr.action;
      maxS = Math.max(maxS, Math.abs(a.steering - b.steering)); maxV = Math.max(maxV, Math.abs(a.speed - b.speed));
      if (JSON.stringify(a) === JSON.stringify(b)) same++; else { diff++; if (first === null) first = k; }
      if (JSON.stringify(a.info) !== JSON.stringify(b.info)) { infoMis++; if (firstInfo === null) firstInfo = {k, a: a.info, b: b.info}; }
      if (((a.info && a.info.jf && a.info.jf.sh) || 0) === ((b.info && b.info.jf && b.info.jf.sh) || 0)) shSame++;
    }
    ms.sort((x, y) => x - y);
    done({file: fidName, frames: fid.frames.length, exact_same: same, exact_diff: diff, first_exact_diff: first, max_abs_steering: maxS, max_abs_speed: maxV, shield_frames: shield,
      info_mismatch: infoMis, first_info_mismatch: firstInfo, sh_same_frames: shSame, shield_frames_node: fid.frames.filter(q => q.action.info && q.action.info.jf && q.action.info.jf.sh).length,
      metadata_same: JSON.stringify(t.metadata) === JSON.stringify(fid.metadata), metadata_shield: t.metadata.shield ? {version: t.metadata.shield.version, select: t.metadata.shield.select, c_nominal: t.metadata.shield.c_nominal} : null,
      policy_sha256_page: sha, policy_bytes_page: policyText.length, policy_sha256_fid: fid.policy_sha256, ms_mean: ms.reduce((x, y) => x + y, 0) / ms.length, ms_p95: ms[Math.floor(0.95 * (ms.length - 1))], ms_max: ms[ms.length - 1],
      pass: maxS <= 1e-6 && maxV <= 1e-6 && infoMis === 0 && shSame === fid.frames.length && JSON.stringify(t.metadata) === JSON.stringify(fid.metadata) && (sha === null || sha === fid.policy_sha256)});
  } catch (e) { done({file: fidName, error: String(e && e.stack || e)}); }
})();"""


def other_firefox():
    out = subprocess.run(['pgrep', '-af', 'firefox'], capture_output=True, text=True).stdout.splitlines()
    return [ln for ln in out if 'pgrep' not in ln and 'browser_fid' not in ln and ('/firefox' in ln.split(' ', 1)[1].split(' ')[0])]


def mem_available_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', type=Path, required=True, help='候補のフォルダー(runtime/・policy.json・fid_*.json)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    cand = a.cand.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    fids = sorted(p.name for p in cand.glob('fid_*.json'))
    if not fids or not (cand / 'runtime/mode_runtime.mjs').exists():
        sys.exit('F-ID か runtime/ がありません')
    res = {'schema': 'jf-sim-rl-line-r4b-browser-fid/v1', 'page': f'{BASE}/sim/c/default/', 'served_from': f'http://{SERVE_HOST}:{SERVE_PORT}/', 'cand': str(cand),
           'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'fids': {}}
    t0 = time.time()
    while other_firefox() or mem_available_mb() < 1500:
        if time.time() - t0 > 3600:
            sys.exit('待ち時間切れ(ほかの Firefox かメモリ)')
        print(f'[fid] 待機: ほかの firefox {len(other_firefox())} / 空き {mem_available_mb()} MB', flush=True)
        time.sleep(30)
    srv = http.server.ThreadingHTTPServer((SERVE_HOST, SERVE_PORT), functools.partial(CorsHandler, directory=str(cand)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    log = open(a.out / 'geckodriver.log', 'w')
    env = {'DISPLAY': ':0', 'XAUTHORITY': XAUTH}
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    d = WebDriver(GECKO_PORT)
    try:
        for _ in range(60):
            try:
                d._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        caps = d.start(['-profile', str(PROFILE)], {}, env)
        res['browser'] = {k: caps.get(k) for k in ('browserName', 'browserVersion', 'platformName')}
        d.get(f'{BASE}/sim/c/default/')
        for _ in range(120):
            st = d.js("return {ready: document.readyState, sim: !!window.courseSimulator, origin: location.origin, secure: window.isSecureContext, uuid: typeof crypto.randomUUID};")
            if st and st.get('ready') == 'complete' and st.get('sim'):
                break
            time.sleep(1)
        res['page_env'] = st
        for f in fids:
            r = d.js_async(FID_JS, f'http://{SERVE_HOST}:{SERVE_PORT}', f, 'policy.json', timeout=900)
            res['fids'][f] = r
            print('[fid]', f, json.dumps({k: r.get(k) for k in ('frames', 'exact_same', 'max_abs_steering', 'max_abs_speed', 'info_mismatch', 'sh_same_frames', 'metadata_same', 'shield_frames', 'shield_frames_node', 'ms_mean', 'ms_p95', 'pass', 'error')}),
                  flush=True)
        res['ua'] = d.js('return navigator.userAgent;')
        import hashlib
        served = hashlib.sha256((cand / 'policy.json').read_bytes()).hexdigest()
        res['policy_sha256_served'] = served
        for r in res['fids'].values():
            if r.get('policy_sha256_fid') != served:
                r['pass'] = False
                r['policy_mismatch'] = True
    finally:
        res['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        res['pass'] = bool(res['fids']) and all(v.get('pass') for v in res['fids'].values())
        (a.out / 'browser_fid.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
        d.quit()
        gd.terminate()
        try:
            gd.wait(10)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
        srv.shutdown()
        srv.server_close()
    print('[fid] 終わり pass =', res['pass'], flush=True)


if __name__ == '__main__':
    main()
