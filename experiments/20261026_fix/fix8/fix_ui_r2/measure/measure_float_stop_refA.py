# -*- coding: utf-8 -*-
"""打ち消しの票 A(反証 8-2 回目・係 ui の float-stop-half-width-under-toast)。直さない・本物の jetformer-web に触れない。

作り物のサーバー(この台本の中の FastAPI。127.0.0.1 の空いたポート。本物の index.html・app.js・style.css を読むだけ)と
headless の Firefox で、本物の ▶ の押し方(#btn-dr-start の click → 本物の app.js の確認の文(window.confirm を true に)→
POST /api/drive/start(作り物は受けて running にする)→ 本物の toast() と pollDrive())で走行を始め、
押してから 7.5 秒、50 ms ごとに ページの中で
  - #toast の見え方・箱・文・class、#btn-float-stop の見え方・箱、body.driving
  - ■ の文の 1 字ずつの箱(行に分ける)と、トーストの箱に入る字
  - トーストの上の 1 点(■ の 1 行目の真ん中)の elementFromPoint(pointer-events: none なら ■ が返る)
を取る。指摘の台本(review_ui_r2/)は使わない(toast() を直に呼ばない)。

  jf-heavy --lock big --mem 3000M -- python -I -B measure_float_stop.py
"""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = '/home/jetson/jetformer'
sys.path.insert(0, ROOT)
from jetformer import drive          # noqa: E402
from jetformer.web import bench      # noqa: E402

STATIC = os.path.join(ROOT, 'jetformer', 'web', 'static')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
GECKO = shutil.which('geckodriver') or '/snap/bin/geckodriver'
B = bench.B_RUN
BEFORE_CSS = os.path.join(ROOT, 'experiments', '20261026_fix', 'fix8', 'fix_ui_r1', 'before', 'style.css')   # 8-1 の前(5842642f)


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_app(work):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import Response
    from fastapi.staticfiles import StaticFiles
    runs = work / 'runs'
    d = runs / B
    d.mkdir(parents=True)
    real = drive.RUNS_DIR / B                         # 本物の B の小さい json だけ写す(エンジンは偽)
    for n in ('model_config.json', 'run.json', 'result.json', 'engine_check.json'):
        shutil.copy(real / n, d / n)
    (d / drive.ENGINE_FILE).write_bytes(b'fake')
    drive.RUNS_DIR = runs
    drive.RECOMMENDED_PATH = work / 'rec.json'
    real_bench = bench.BENCH_PATH                    # data/bench_candidates.json(読むだけ)
    bench.BENCH_PATH = work / 'none.json'
    st = {'params': dict(drive.DEFAULT_PARAMS), 'running': False, 'dry': False, 'calls': [], 'plan_diffs': [], 'ai_no': 11,
          'boot_at': 1793140000.0, 'bench': False, 'real_bench': real_bench}
    vbat = {'supported': True, 'v': 8.31, 'age_s': 0.01, 'level': 'ok', 'warn_v': 7.9, 'bad_v': 7.7, 'scale': 1.0, 'run': None}
    app = FastAPI()

    @app.get('/')
    def index():
        return Response(open(os.path.join(STATIC, 'index.html'), encoding='utf-8').read(), media_type='text/html')

    @app.get('/frame')
    def frame(w: int, h: int, c: str = ''):
        # スマホの幅: Firefox の窓は幅 500 より狭くならないので、iframe の大きさ = 画面(左上に置く・枠なし)
        return Response('<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;background:#777">'
                        '<iframe id="f" src="/?c=%s" style="border:0;display:block;width:%dpx;height:%dpx"></iframe></body></html>'
                        % (c, w, h), media_type='text/html')

    @app.get('/before/style.css')
    def before_css():
        return Response(open(BEFORE_CSS, encoding='utf-8').read(), media_type='text/css')

    @app.get('/api/status')
    def status():
        return {'time': time.time(), 'sensors_running': True, 'sensors': {}, 'recorder': {'recording': False, 'saved_scenes': []},
                'session': {}, 'disk': {'free_gb': 40.0, 'total_gb': 250.0}, 'training': False, 'driving': st['running'],
                'config': {'scene_seconds': 20, 'vehicle_protocol': 'snapshot', 'tof_unit': '', 'camera': ''}}

    @app.get('/api/drive/models')
    def models():
        return drive.list_models()

    @app.get('/api/bench/candidates')
    def cands():
        if not st['bench']:
            return bench.load_candidates()               # 無い = パネルを出さない(ふつうの走行)
        d = bench.load_candidates(st['real_bench'])     # 本物の 11/3 の計画。サーバーの日付だけ 11/3 にする
        d['today'] = '2026-11-03'
        return d

    @app.get('/api/drive/status')
    def drive_status():
        run = st['running']
        out = {'phase': 'driving' if run else 'idle', 'message': '', 'error': None, 'running': run, 'dry_run': st['dry'] if run else False,
               'params': dict(st['params']), 'last': {'run': B, 'mode_id': 5}, 'recording': False, 'elapsed_s': 1.0 if run else 0,
               'stats': {}, 'now': None, 'auto_end': None, 'history': {'t': []}, 'ref': None, 'vbat': vbat,
               'boot_at': st['boot_at'], 'ai_no': (st['ai_no'] if run and not st['dry'] else None),
               'pca9685': {'address': 64, 'freq_hz': 60, 'steering_ch': 0, 'throttle_ch': 1},
               'limits': {k: list(v) for k, v in drive.PARAM_LIMITS.items()}, 'zone_caps_spec': drive.ZONE_CAPS_SPEC,
               'speed_ctrl_spec': drive.speed_ctrl_spec(st['params'])}
        return out

    @app.post('/api/drive/params')
    def params(body: dict = Body(...)):
        st['calls'].append({'params': body})
        p = dict(st['params'])
        for k, v in body.items():
            if k in p:
                p[k] = v
        st['params'] = p
        return p

    @app.post('/api/drive/start')
    def start(body: dict = Body(...)):
        st['calls'].append({'start': body})
        if st['running']:
            raise HTTPException(409, '走行中です')
        st['running'], st['dry'] = True, bool(body.get('dry_run'))
        st['ai_no'] += 1
        diffs = list(st['plan_diffs'])
        if not st['dry'] and 'bench=' in (body.get('description') or ''):    # 本物の server.py と同じ照らし合わせ(本物の計画で)
            desc, diffs = bench.plan_check(body.get('description', ''), body.get('run', ''), body.get('mode_id'), body.get('params') or {},
                                           bench.load_candidates(st['real_bench']))
            st['start_desc'] = desc
        st['start_diffs'] = diffs
        return {'ok': True, 'run': body.get('run'), 'token': 'tok%d' % st['ai_no'], 'ai_no': None if st['dry'] else st['ai_no'],
                'plan_diffs': [] if st['dry'] else diffs, 'v_before': 8.31}

    @app.post('/api/drive/stop')
    def stop(body: dict = Body(...)):
        st['calls'].append({'stop': body})
        st['running'] = False
        return {'ok': True}

    @app.post('/api/drive/heartbeat')
    def hb(body: dict = Body(None)):
        return {'ok': True}

    @app.post('/api/bench/after')
    def after(body: dict = Body(None)):
        st['calls'].append({'after': body})
        return {'ok': True}

    @app.get('/api/live/values')
    def live():
        return {'imu': {'t': []}, 'vehicle': {'t': []}, 'vbat': vbat,
                'vehicle_now': {'pwm': [1441, 1496, 1980], 'erpm': 0, 'valid': [True] * 3, 'select': False, 'moving': False,
                                'quality': 'snapshot'}}

    @app.get('/api/{rest:path}')
    def anything(rest: str):
        if rest in ('scenes', 'modes', 'jobs', 'datasets', 'runs', 'policies', 'models', 'sim/courses'):
            return []
        raise HTTPException(404)

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app, st


class WD(object):
    def __init__(self, port):
        self.base, self.sid = 'http://127.0.0.1:%d' % port, None

    def req(self, method, path, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read() or b'{}').get('value')

    def js(self, script):
        return self.req('POST', '/session/%s/execute/sync' % self.sid, {'script': script, 'args': []})

    def to_frame(self):
        self.req('POST', '/session/%s/frame' % self.sid, {'id': None})
        self.req('POST', '/session/%s/frame' % self.sid, {'id': 0})

    def to_top(self):
        self.req('POST', '/session/%s/frame' % self.sid, {'id': None})

    def shot(self, path):
        """iframe(= 画面)の写真。外に出て撮り、iframe に戻る"""
        self.to_top()
        eid = list(self.req('POST', '/session/%s/element' % self.sid, {'using': 'css selector', 'value': '#f'}).values())[0]
        with open(path, 'wb') as f:
            f.write(base64.b64decode(self.req('GET', '/session/%s/element/%s/screenshot' % (self.sid, eid))))
        self.req('POST', '/session/%s/frame' % self.sid, {'id': 0})

    def open_frame(self, port, W, H, tag):
        self.req('POST', '/session/%s/url' % self.sid, {'url': 'http://127.0.0.1:%d/frame?w=%d&h=%d&c=%s' % (port, W, H, tag)})

        def ready():
            self.to_frame()
            return self.js('return !!(window.jetformer && window.jetformer.showTab) && innerWidth === %d && innerHeight === %d' % (W, H))
        assert wait(ready, 40), 'frame'


def wait(fn, t=30.0):
    end = time.time() + t
    while time.time() < end:
        try:
            v = fn()
            if v:
                return v
        except Exception:      # noqa: BLE001
            pass
        time.sleep(0.2)
    return None


SAMPLER = r"""
const f = document.querySelector('#btn-float-stop'), t = document.querySelector('#toast');
window._samp = []; window._t0 = performance.now();
const vis = (e) => !e.classList.contains('hidden') && getComputedStyle(e).display !== 'none';
const R = (r) => [Math.round(r.left * 10) / 10, Math.round(r.top * 10) / 10, Math.round(r.width * 10) / 10, Math.round(r.height * 10) / 10];
function chars() {                 // ■ の文の 1 字ずつの箱
  const tn = f.firstChild, out = [];
  if (!tn || tn.nodeType !== 3) return out;
  const s = tn.textContent;
  for (let i = 0; i < s.length; i++) {
    const rg = document.createRange(); rg.setStart(tn, i); rg.setEnd(tn, i + 1);
    const rs = rg.getClientRects(); if (!rs.length) continue;
    out.push([s[i], rs[0]]);
  }
  return out;
}
function sample() {
  const fv = vis(f), tv = vis(t), fr = f.getBoundingClientRect(), tr = t.getBoundingClientRect();
  const s = { t: Math.round(performance.now() - window._t0), driving: document.body.classList.contains('driving'),
              fv, tv, fr: R(fr), tr: R(tr), tcls: t.className, ttext: t.textContent.slice(0, 60), tlen: t.textContent.length,
              tbg: getComputedStyle(t).backgroundColor, tz: getComputedStyle(t).zIndex, fz: getComputedStyle(f).zIndex,
              tpe: getComputedStyle(t).pointerEvents, iw: innerWidth, ih: innerHeight, cw: document.documentElement.clientWidth };
  if (fv) {
    const cs = chars(), lines = [];
    for (const [c, r] of cs) {
      const top = Math.round(r.top);
      let L = lines.find((l) => Math.abs(l.top - top) < 4);
      if (!L) { L = { top, bottom: Math.round(r.bottom), text: '' }; lines.push(L); }
      L.text += c;
    }
    s.lines = lines.map((l) => l.text); s.line_tops = lines.map((l) => [l.top, l.bottom]);
    if (tv) {
      const ix = Math.max(0, Math.min(fr.right, tr.right) - Math.max(fr.left, tr.left));
      const iy = Math.max(0, Math.min(fr.bottom, tr.bottom) - Math.max(fr.top, tr.top));
      s.ov = [Math.round(ix * 10) / 10, Math.round(iy * 10) / 10, Math.round(ix * iy)];
      s.hidden_chars = cs.filter(([c, r]) => r.right > tr.left && r.left < tr.right && r.bottom > tr.top && r.top < tr.bottom).map(([c]) => c).join('');
      // 字の半分より多くがトーストの下 = 読めない
      s.half_hidden = cs.filter(([c, r]) => {
        const a = Math.max(0, Math.min(r.right, tr.right) - Math.max(r.left, tr.left)) * Math.max(0, Math.min(r.bottom, tr.bottom) - Math.max(r.top, tr.top));
        return a > 0.5 * r.width * r.height; }).map(([c]) => c).join('');
      if (lines.length) {
        const y = (lines[0].top + lines[0].bottom) / 2, x = fr.left + fr.width / 2;
        const e = document.elementFromPoint(x, y);
        s.hit_line1 = e ? (e.id || e.tagName) : null;
        s.line1_under_toast = y > tr.top && y < tr.bottom && x > tr.left && x < tr.right;
      }
    }
  }
  window._samp.push(s);
}
sample();
window._sampTimer = setInterval(sample, 50);
return true;
"""


def run_case(wd, srv, port, W, H, case, shots):
    """case: 'normal'(ベンチマークでない本番の ▶)・'b1103'(11/3 の本物の計画のブロック 1 を選ぶ欄で選んで ▶)・'dry'(推論の確認)"""
    srv['running'], srv['dry'], srv['calls'] = False, False, []
    srv['plan_diffs'], srv['bench'], srv['start_diffs'], srv['start_desc'] = [], case.startswith('b1103'), None, None
    wd.open_frame(port, W, H, '%s_%dx%d' % (case, W, H))
    if case.endswith('_before'):        # 8-1 の前の style.css に替える(index.html・app.js は今のまま)
        wd.js("document.querySelector('link[href*=\"style.css\"]').href = '/before/style.css';")
        assert wait(lambda: wd.js("return getComputedStyle(document.querySelector('#toast')).width !== 'max-content' && "
                                  "[...document.styleSheets].some((x) => (x.href || '').includes('/before/style.css'))"), 10), 'before css'
        time.sleep(0.5)
    wd.js("try { sessionStorage.removeItem('jf-drive-token'); } catch (e) {}"
          "window._errs = []; window.addEventListener('error', (e) => window._errs.push(String(e.message)));"
          "window.addEventListener('unhandledrejection', (e) => window._errs.push('rej ' + String(e.reason)));"
          "window._confirms = []; window.confirm = (m) => { window._confirms.push(m); return true; };")
    wd.js("window.jetformer.showTab('drive')")
    ok = wait(lambda: wd.js("const s = window.jetformer.state.drive; const b = document.querySelector('#btn-dr-start');"
                            "return s.paramsLoaded && !b.disabled && !!document.querySelector('#dr-model').value;"), 30)
    assert ok, 'not synced'
    time.sleep(1.5)
    picked = None
    if case.startswith('b1103'):
        val = wait(lambda: wd.js("const o = [...document.querySelectorAll('#dr-bench-pick option')].find((x) => /:0$/.test(x.value) && !x.disabled);"
                                 "return o ? o.value : null;"), 20)
        assert val, 'no 11/3 block'
        wd.js("const s = document.querySelector('#dr-bench-pick'); s.value = '%s'; s.dispatchEvent(new Event('change', {bubbles: true}));" % val)
        assert wait(lambda: 'block=1' in (wd.js("return document.querySelector('#dr-desc').value") or ''), 10)
        time.sleep(1.5)
        picked = wd.js("return [document.querySelector('#dr-desc').value, document.querySelector('#dr-throttle-gain').value, document.querySelector('#dr-model').value]")
    wd.js("const t = document.querySelector('#toast'); t.classList.add('hidden');")   # 開いたときのトーストを消してから測る
    dry_now = wd.js("return document.querySelector('#dr-dry').checked")
    if (case == 'dry') != bool(dry_now):
        wd.js("document.querySelector('#dr-dry').click()")
        time.sleep(0.5)
    before = wd.js("const f = document.querySelector('#btn-float-stop'); return [f.className, document.body.className]")
    wd.js(SAMPLER)
    wd.js("document.querySelector('#btn-dr-start').click()")
    shot_paths = []
    if shots:
        time.sleep(1.0)
        p = os.path.join(OUT, 'shot_%s_%dx%d_t1s.png' % (case, W, H))
        wd.shot(p)
        shot_paths.append(p)
        time.sleep(6.6 if case.startswith('b1103') else 3.2)
        p = os.path.join(OUT, 'shot_%s_%dx%d_after.png' % (case, W, H))
        wd.shot(p)
        shot_paths.append(p)
        time.sleep(0.2)
    else:
        time.sleep(7.5)
    wd.js('clearInterval(window._sampTimer)')
    samp = wd.js('return window._samp')
    errs = wd.js('return window._errs')
    confirms = wd.js('return window._confirms.length')
    starts = [c for c in srv['calls'] if 'start' in c]
    return {'W': W, 'H': H, 'case': case, 'before_float_class': before[0], 'before_body_class': before[1], 'samples': samp,
            'errs': errs, 'confirms': confirms, 'n_start_posts': len(starts), 'shots': shot_paths, 'picked': picked,
            'server_plan_diffs': srv.get('start_diffs'), 'server_desc': srv.get('start_desc')}


def summarize(r):
    S = r['samples']
    first_drive = next((s['t'] for s in S if s['driving'] and s['fv']), None)
    ov = [s for s in S if s.get('ov') and s['ov'][2] > 0 and s['fv'] and s['tv']]
    toasts = []
    for s in S:
        if s['tv']:
            key = (s['ttext'], s['tcls'])
            if not toasts or toasts[-1]['key'] != key:
                toasts.append({'key': key, 'from': s['t'], 'to': s['t']})
            toasts[-1]['to'] = s['t']
    f_on = [s for s in S if s['fv']]
    lines = f_on[-1]['lines'] if f_on else None
    out = {'W': r['W'], 'H': r['H'], 'case': r['case'], 'inner': [S[0]['iw'], S[0]['ih'], S[0]['cw']],
           'idle_float_hidden': 'hidden' in r['before_float_class'], 'start_posts': r['n_start_posts'], 'confirms': r['confirms'],
           'errs': r['errs'], 'float_shown_at_ms': first_drive,
           'float_rect': f_on[-1]['fr'] if f_on else None, 'float_lines': lines,
           'toasts': [{'text': k['key'][0], 'cls': k['key'][1], 'seen_ms': [k['from'], k['to']]} for k in toasts],
           'overlap_ms': [ov[0]['t'], ov[-1]['t']] if ov else None, 'overlap_samples': len(ov),
           'overlap_max': max((s['ov'] for s in ov), key=lambda v: v[2]) if ov else None,
           'toast_rect_at_overlap': ov[len(ov) // 2]['tr'] if ov else None,
           'hidden_chars': sorted({s['hidden_chars'] for s in ov}) if ov else [],
           'half_hidden_chars': sorted({s['half_hidden'] for s in ov}) if ov else [],
           'line1_center_under_toast': sorted({s.get('line1_under_toast') for s in ov}) if ov else [],
           'hit_line1': sorted({str(s.get('hit_line1')) for s in ov}) if ov else [],
           'toast_bg': ov[0]['tbg'] if ov else None, 'z': [S[-1]['tz'], S[-1]['fz']], 'toast_pe': S[-1]['tpe'],
           'shots': r['shots'], 'picked': r['picked'], 'server_plan_diffs': r['server_plan_diffs']}
    return out


def tap_through(wd, srv, port, W, H):
    """覆われた所(■ の 1 行目の真ん中)を WebDriver の指(pointer)で押すと、本物の stopDrive で止まるか"""
    srv['running'], srv['dry'], srv['calls'], srv['plan_diffs'], srv['bench'] = False, False, [], [], False
    wd.open_frame(port, W, H, 'tap_%dx%d' % (W, H))
    wd.js("try { sessionStorage.removeItem('jf-drive-token'); } catch (e) {} window.confirm = () => true;")
    wd.js("window.jetformer.showTab('drive')")
    assert wait(lambda: wd.js("const s = window.jetformer.state.drive; return s.paramsLoaded && !document.querySelector('#btn-dr-start').disabled"))
    time.sleep(1.5)
    if wd.js("return document.querySelector('#dr-dry').checked"):
        wd.js("document.querySelector('#dr-dry').click()")
    wd.js("document.querySelector('#btn-dr-start').click()")
    assert wait(lambda: wd.js("return document.body.classList.contains('driving') && !document.querySelector('#toast').classList.contains('hidden')"), 5)
    time.sleep(0.3)
    g = wd.js("const f = document.querySelector('#btn-float-stop').getBoundingClientRect(), t = document.querySelector('#toast').getBoundingClientRect();"
              "return {f: [f.left, f.top, f.width, f.height], t: [t.left, t.top, t.width, t.height], text: document.querySelector('#toast').textContent};")
    x = int(g['f'][0] + g['f'][2] / 2)
    y = int(max(g['f'][1], g['t'][1]) + 8)        # トーストの箱の中で ■ の箱の中の点
    assert g['t'][1] < y < g['t'][1] + g['t'][3] and g['f'][1] < y < g['f'][1] + g['f'][3]
    n0 = len([c for c in srv['calls'] if 'stop' in c])
    wd.req('POST', '/session/%s/actions' % wd.sid, {'actions': [{'type': 'pointer', 'id': 'finger', 'parameters': {'pointerType': 'touch'},
                                                                  'actions': [{'type': 'pointerMove', 'x': x, 'y': y, 'origin': 'viewport'},
                                                                              {'type': 'pointerDown', 'button': 0},
                                                                              {'type': 'pointerUp', 'button': 0}]}]})
    stopped = wait(lambda: len([c for c in srv['calls'] if 'stop' in c]) > n0, 5)
    wd.req('DELETE', '/session/%s/actions' % wd.sid)
    how = 'touch' if stopped else None
    if not stopped and srv['running']:             # 指で click にならない作りなら、マウスで同じ点
        wd.req('POST', '/session/%s/actions' % wd.sid, {'actions': [{'type': 'pointer', 'id': 'mouse', 'parameters': {'pointerType': 'mouse'},
                                                                      'actions': [{'type': 'pointerMove', 'x': x, 'y': y, 'origin': 'viewport'},
                                                                                  {'type': 'pointerDown', 'button': 0},
                                                                                  {'type': 'pointerUp', 'button': 0}]}]})
        stopped = wait(lambda: len([c for c in srv['calls'] if 'stop' in c]) > n0, 5)
        wd.req('DELETE', '/session/%s/actions' % wd.sid)
        how = 'mouse' if stopped else None
    return {'W': W, 'H': H, 'how': how, 'tap_xy': [x, y], 'float': g['f'], 'toast': g['t'], 'toast_text': g['text'],
            'stop_posted': bool(stopped), 'stop_body': [c for c in srv['calls'] if 'stop' in c][-1:] if stopped else None,
            'running_after': srv['running']}


def main():
    import uvicorn
    work = Path(tempfile.mkdtemp(prefix='refA-ui-r2-', dir='/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad'))
    app, srv = make_app(work)
    port, wd_port = free_port(), free_port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=server.run, daemon=True).start()
    gd_log = open(os.path.join(OUT, 'geckodriver.log'), 'w')
    gd = subprocess.Popen([GECKO, '--port', str(wd_port)], stdout=gd_log, stderr=subprocess.STDOUT)
    profile = tempfile.mkdtemp(prefix='refA-ui-r2-profile-', dir=os.path.expanduser('~/snap/firefox/common'))
    wd = WD(wd_port)
    res = {'files': {}, 'cases': [], 'tap': []}
    import hashlib
    for n in ('index.html', 'app.js', 'style.css'):
        res['files'][n] = hashlib.sha256(open(os.path.join(STATIC, n), 'rb').read()).hexdigest()[:12]
    try:
        assert wait(lambda: urllib.request.urlopen('http://127.0.0.1:%d/api/status' % port, timeout=2))
        assert wait(lambda: wd.req('GET', '/status', timeout=3))
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
                'moz:firefoxOptions': {'args': ['-headless', '-profile', profile],
                                       'prefs': {'ui.useOverlayScrollbars': 1, 'widget.gtk.overlay-scrollbars.enabled': True},
                                       'env': {'MOZ_HEADLESS_WIDTH': '390', 'MOZ_HEADLESS_HEIGHT': '700'}}}}}
        wd.sid = wd.req('POST', '/session', caps, timeout=180)['sessionId']
        wd.req('POST', '/session/%s/window/rect' % wd.sid, {'width': 1400, 'height': 1150})
        wd.req('POST', '/session/%s/url' % wd.sid, {'url': 'http://127.0.0.1:%d/' % port})
        res['ua'] = wd.js('return navigator.userAgent')
        res['top_inner'] = wd.js('return [innerWidth, innerHeight]')
        sizes = [(360, 700), (375, 667), (390, 660), (390, 700), (390, 844), (393, 760), (412, 800), (430, 900),
                 (600, 800), (768, 900), (844, 390), (1280, 800)]
        shots = {(390, 700), (430, 900)}
        more = ((360, 700), (390, 700), (412, 800), (430, 900), (844, 390))
        for (W, H) in sizes:
            for case in ('normal', 'b1103', 'dry', 'b1103_before'):
                if case != 'normal' and (W, H) not in more:
                    continue
                if case == 'b1103_before' and (W, H) not in ((390, 700), (430, 900)):
                    continue
                r = run_case(wd, srv, port, W, H, case, (W, H) in shots and case != 'dry')
                sm = summarize(r)
                res['cases'].append(sm)
                with open(os.path.join(OUT, 'raw_%s_%dx%d.json' % (case, W, H)), 'w') as f:
                    json.dump(r, f, ensure_ascii=False)
                print(json.dumps({k: sm[k] for k in ('W', 'H', 'case', 'inner', 'float_rect', 'float_lines', 'overlap_ms', 'overlap_max',
                                                     'half_hidden_chars', 'toasts', 'errs', 'server_plan_diffs')}, ensure_ascii=False), flush=True)
        for (W, H) in ((390, 700), (430, 900)):
            t = tap_through(wd, srv, port, W, H)
            res['tap'].append(t)
            print(json.dumps(t, ensure_ascii=False), flush=True)
    finally:
        with open(os.path.join(OUT, 'measure_float_stop.json'), 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        try:
            if wd.sid:
                wd.req('DELETE', '/session/%s' % wd.sid, timeout=30)
        except Exception:      # noqa: BLE001
            pass
        gd.terminate()
        try:
            gd.wait(10)
        except Exception:      # noqa: BLE001
            gd.kill()
        server.should_exit = True
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
