#!/usr/bin/env python3
"""(sim_diverse_train_20261015 の写し: geckodriver 4450・ブリッジ 8821・プロファイル sim_diverse_train_20261015/ff_profile に変えた。元は sim_dart_20261014 の写し)ヘッドレス Firefox で、コピーしたシミュレーター(ブリッジ 127.0.0.1:8791)の WebGL が動くかを確かめる

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/webgl_probe.py [--page rl.html] [--record-scene]
           [--prefs '{"webgl.force-enabled": true}'] [--tag default]

  1. 他の Firefox が動いていないこと・空きメモリが 2 GiB 以上あることを確かめる(無ければ 30 秒ごとに待つ)
  2. geckodriver をポート 4445 で起動し、専用のプロファイル(sim/ff_profile_rl)で firefox -headless を開く
  3. ページを開いてテクスチャの読み込みを待ち、WebGL のレンダラー名・全体のスクリーンショット・車載カメラ(224×224)・
     ToF の画像を保存し、requestAnimationFrame の頻度を測る
  4. --record-scene のときは「20分・60シーンの収集条件を用意」の1本目(rc20-mode8-train-00)だけを「連続試験を開始」で
     走らせ、アプリ自身の収集(画像・ToF・IMU・PWM)で 600 フレームの ZIP を作る。1 フレームあたりの時間を測る。
  結果は /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/webgl_probe/<tag>/ に書く。
  WebDriver は selenium を使わず、HTTP(JSON)を直接送る。実車・I2C には触れない。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GECKO = '/snap/bin/geckodriver'
GECKO_PORT = 4450
BRIDGE = 'http://127.0.0.1:8821'
REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
PROFILE = REPORT / 'ff_profile'      # snap の Firefox が読めるよう、ホームの隠しでないフォルダー
MIN_AVAILABLE_MB = 2048


def mem_available_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) // 1024
    return 0


def other_firefox():
    out = subprocess.run(['pgrep', '-af', 'firefox'], capture_output=True, text=True).stdout.splitlines()
    me = str(os.getpid())
    return [l for l in out if not l.startswith(me + ' ') and 'pgrep' not in l and 'webgl_probe.py' not in l
            and ('/firefox' in l.split(' ', 1)[1].split(' ')[0] or l.split(' ', 1)[1].startswith('firefox'))]


def wait_preflight(max_wait_s=3600):
    t0 = time.time()
    while True:
        others, avail = other_firefox(), mem_available_mb()
        if not others and avail >= MIN_AVAILABLE_MB:
            return {'mem_available_mb': avail, 'waited_s': round(time.time() - t0, 1)}
        if time.time() - t0 > max_wait_s:
            sys.exit(f'待ち時間切れ: 他の firefox {len(others)} 個 / 空きメモリ {avail} MB')
        print(f'[probe] 待機: 他の firefox {len(others)} 個 / 空きメモリ {avail} MB', flush=True)
        time.sleep(30)


class WebDriver:
    def __init__(self, port):
        self.base = f'http://127.0.0.1:{port}'
        self.sid = None

    def _req(self, method, path, body=None, timeout=300):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b'{}').get('value')
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'{method} {path}: {e.code} {e.read()[:800]!r}') from None

    def start(self, args, prefs, env):
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'moz:firefoxOptions': {
            'args': args, 'prefs': prefs, 'env': env, 'log': {'level': 'info'}}}}}
        v = self._req('POST', '/session', caps, timeout=180)
        self.sid = v['sessionId']
        return v.get('capabilities', {})

    def s(self, method, path, body=None, timeout=300):
        return self._req(method, f'/session/{self.sid}{path}', body, timeout)

    def get(self, url):
        return self.s('POST', '/url', {'url': url})

    def js(self, script, *args, timeout=300):
        return self.s('POST', '/execute/sync', {'script': script, 'args': list(args)}, timeout)

    def js_async(self, script, *args, timeout=300):
        return self.s('POST', '/execute/async', {'script': script, 'args': list(args)}, timeout)

    def screenshot(self, path):
        Path(path).write_bytes(base64.b64decode(self.s('GET', '/screenshot')))

    def quit(self):
        if self.sid:
            try:
                self._req('DELETE', f'/session/{self.sid}', timeout=60)
            except Exception as e:  # noqa: BLE001  終了時の失敗は記録だけ
                print('[probe] セッション終了の失敗:', e)
            self.sid = None


WEBGL_INFO = r"""
const out={};
for(const kind of ['webgl2','webgl']){
  try{const c=document.createElement('canvas');const gl=c.getContext(kind);
    if(!gl){out[kind]=null;continue;}
    const ext=gl.getExtension('WEBGL_debug_renderer_info');
    out[kind]={vendor:gl.getParameter(gl.VENDOR),renderer:gl.getParameter(gl.RENDERER),
      unmasked_vendor:ext?gl.getParameter(ext.UNMASKED_VENDOR_WEBGL):null,
      unmasked_renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):null,
      version:gl.getParameter(gl.VERSION),max_texture:gl.getParameter(gl.MAX_TEXTURE_SIZE)};
  }catch(e){out[kind]='error: '+e.message;}
}
out.fatal_hidden=document.getElementById('fatal')?.hidden;
out.fatal_text=document.getElementById('fatal')?.textContent||'';
out.user_agent=navigator.userAgent;
out.ready=!!(window.courseSimulator&&window.courseSimulator.sensorSummary().ready);
out.probe=globalThis.__rlTeacherProbe?JSON.parse(JSON.stringify(globalThis.__rlTeacherProbe)):null;
out.status=document.getElementById('status')?.textContent;
out.sensor=window.courseSimulator?window.courseSimulator.sensorSummary():null;
out.inner=[innerWidth,innerHeight,devicePixelRatio];
return out;"""

RAF_FPS = r"""
const done=arguments[arguments.length-1];let n=0;const t0=performance.now();
function f(t){n++;if(t-t0<3000)requestAnimationFrame(f);else done({frames:n,seconds:(t-t0)/1000,fps:n/((t-t0)/1000)});}
requestAnimationFrame(f);"""

# 1 回の「描画 + 読み出し」の時間: 手動の1ステップ(停止中)ではなく、センサーの描画だけを測るために
# 車載カメラ canvas の toDataURL(PNG 化)と WebGL の描画を、評価を始めずに直接は呼べないので、
# 連続試験の時間(フレーム数 / 秒)で測る。

CANVAS_PNG = "return document.getElementById(arguments[0]).toDataURL('image/png');"


def save_canvas(d, cid, path):
    url = d.js(CANVAS_PNG, cid)
    Path(path).write_bytes(base64.b64decode(url.split(',', 1)[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default='rl.html')
    ap.add_argument('--tag', default='default')
    ap.add_argument('--prefs', default='{}')
    ap.add_argument('--env', default='{}', help='Firefox に渡す環境変数(JSON)')
    ap.add_argument('--display', default='', help='ヘッドレスでなく、この X ディスプレイ(例 :0)に窓を開く')
    ap.add_argument('--record-scene', action='store_true')
    ap.add_argument('--connect', action='store_true', help='「接続を確認」を押す(ブリッジの PWM 換算をアプリに読ませる)')
    ap.add_argument('--no-record', action='store_true', help='連続試験を収集なしで1本だけ走らせる(速度だけ測る)')
    ap.add_argument('--scene-timeout', type=float, default=1800)
    a = ap.parse_args()
    out = REPORT / 'webgl_probe' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    PROFILE.mkdir(parents=True, exist_ok=True)
    result = {'tag': a.tag, 'page': a.page, 'prefs': json.loads(a.prefs), 'preflight': wait_preflight()}
    log = open(out / 'geckodriver.log', 'w')
    gd_env = dict(os.environ)
    if a.display:
        gd_env.update(DISPLAY=a.display, XAUTHORITY='/run/user/1000/gdm/Xauthority')
    gd = subprocess.Popen([GECKO, '--port', str(GECKO_PORT)], stdout=log, stderr=subprocess.STDOUT, env=gd_env)
    d = WebDriver(GECKO_PORT)
    try:
        for _ in range(60):
            try:
                d._req('GET', '/status', timeout=5)
                break
            except Exception:  # noqa: BLE001  起動待ち
                time.sleep(0.5)
        t0 = time.time()
        env = {'MOZ_HEADLESS_WIDTH': '1400', 'MOZ_HEADLESS_HEIGHT': '900', **json.loads(a.env)}
        args = ['-profile', str(PROFILE)]
        if a.display:
            env.update(DISPLAY=a.display, XAUTHORITY='/run/user/1000/gdm/Xauthority')
        else:
            args.insert(0, '-headless')
        result['firefox_args'], result['firefox_env'] = args, env
        caps = d.start(args, json.loads(a.prefs), env)
        result['browser'] = {k: caps.get(k) for k in ('browserName', 'browserVersion', 'platformName', 'moz:headless')}
        result['session_start_s'] = round(time.time() - t0, 1)
        d.get(f'{BRIDGE}/{a.page}')
        t1 = time.time()
        info = None
        while time.time() - t1 < 120:
            info = d.js(WEBGL_INFO)
            if info.get('ready') or not info.get('fatal_hidden', True):
                break
            time.sleep(1)
        result['load_s'] = round(time.time() - t1, 1)
        result['webgl'] = info
        result['mem_available_mb_after_load'] = mem_available_mb()
        d.screenshot(out / 'overview.png')
        save_canvas(d, 'camera', out / 'camera_rgb_224.png')
        save_canvas(d, 'ranges', out / 'tof_view_224.png')
        result['raf'] = d.js_async(RAF_FPS, timeout=30)
        if a.connect:
            d.js("document.getElementById('connect').click();")
            for _ in range(30):
                time.sleep(1)
                cs = d.js("return document.getElementById('connection-status').textContent;")
                if '接続済み' in cs:
                    break
            result['connection_status'] = cs
        print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)
        if a.record_scene or a.no_record:
            # 収集条件(60本)を用意 → 1本目だけを残して連続試験を開始
            d.js("document.getElementById('collection-plan').click();")
            plan = json.loads(d.js("return document.getElementById('batch-plan').value;"))
            first = dict(plan[0])
            first['name'] = 'probe-' + first['name']
            d.js("""const p=document.getElementById('batch-plan');p.value=arguments[0];
                    document.getElementById('batch-record').checked=arguments[1];""",
                 json.dumps([first], ensure_ascii=False), bool(a.record_scene))
            result['scene_plan'] = first
            t2 = time.time()
            d.js("document.getElementById('batch-run').click();")
            samples = []
            while time.time() - t2 < a.scene_timeout:
                time.sleep(5)
                st = d.js("""const s=window.courseSimulator.snapshot();return {frame:s.state.frame,t:s.state.sim_time,
                    laps:s.metrics.completed_laps,coll:s.metrics.collisions,
                    batch:document.getElementById('batch-status').textContent,
                    record:document.getElementById('record-status').textContent};""")
                st['wall_s'] = round(time.time() - t2, 1)
                st['mem_available_mb'] = mem_available_mb()
                samples.append(st)
                print('[probe]', json.dumps(st, ensure_ascii=False), flush=True)
                if '完了' in st['batch'] or '停止' in st['batch'] or '終了' in st['batch']:
                    break
            result['scene_wall_s'] = round(time.time() - t2, 1)
            result['scene_samples'] = samples
            last = samples[-1] if samples else {}
            result['scene_fps'] = round(last.get('frame', 0) / max(result['scene_wall_s'], 1e-6), 2)
            summary = d.js("return document.getElementById('batch-results').value;")
            result['batch_results'] = json.loads(summary or '[]')
            result['probe_after'] = d.js("return globalThis.__rlTeacherProbe?JSON.parse(JSON.stringify(globalThis.__rlTeacherProbe)):null;")
            d.screenshot(out / 'after_scene.png')
            save_canvas(d, 'camera', out / 'camera_rgb_after.png')
    except Exception as e:  # noqa: BLE001  失敗も結果に残す
        result['error'] = f'{type(e).__name__}: {e}'
    finally:
        d.quit()
        gd.terminate()
        try:
            gd.wait(20)
        except subprocess.TimeoutExpired:
            gd.kill()
        log.close()
        time.sleep(2)
        result['firefox_left'] = other_firefox()
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps({k: result.get(k) for k in ('error', 'webgl', 'raf', 'scene_fps', 'scene_wall_s', 'firefox_left')},
                         ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
