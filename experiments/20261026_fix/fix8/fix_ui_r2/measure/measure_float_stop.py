# -*- coding: utf-8 -*-
"""直す 8-2 回目・係 ui(float-stop-half-width-under-toast)の重なりを、直す前と後で測る。本物の jetformer-web に触れない。

打ち消しの票 A の台本(refute_ui_r2_A/measure_float_stop.py。写し measure_float_stop_refA.py)の写しに、次だけ足した:
  - style.css を選べる(作り物のサーバーが index.html の <link> を /css/<tag>.css に替えて返す。index.html・app.js は本物の今のまま):
      before   = 直す前(tests/fixtures/fix8_ui_r2_before/style.css。dbc8b79a… = 8-1 のあとの版)
      now      = 今の jetformer/web/static/style.css(直したあと)
  - 画面の大きさ: 幅 360・375・390・412・430 × 高さ 660・700・760・800・844・900(ふつうの ▶)、
    11/3 のブロックの ▶(6 秒の赤いトースト)と推論の確認は高さ 700・900、ほかに 600×800・768×900・844×390・1280×800。
  - 走っている間に下までスクロールしたとき、走行タブの最後のカードの下と ■ の上(■ が中身を隠すか)。
  - 止まっているときのトースト(8-1 の直し。本物の toast() に ▶ の断りの出し直し(本物の server._boot_refusal の文)・
    「停止しました」・3 つの走行開始の文を渡す)の箱が、直す前と後で同じか。
本物の ▶ の押し方(#btn-dr-start の click → 確認の文に true → POST /api/drive/start(作り物)→ 本物の toast() と pollDrive())は
票 A と同じ。Firefox の窓は幅 500 より狭くならないので、画面 = 左上に置いた枠なしの iframe。

  cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big --mem 3000M -- /home/jetson/miniconda/envs/jetracer/bin/python -I -B \\
      experiments/20261026_fix/fix8/fix_ui_r2/measure/measure_float_stop.py [tag ...]
出力 out/measure_<tags>.json(summary.py が out/summary.txt にする)。
"""
import base64
import hashlib
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
CSS = {'before': os.path.join(ROOT, 'tests', 'fixtures', 'fix8_ui_r2_before', 'style.css'),
       'now': os.path.join(STATIC, 'style.css')}
TRIAL3 = 'ブロックの計画と違うので「試し」(cand=EXP)として記録します: throttle_gain 0.9(基準 1.0)、speed_ceiling 1(基準 0)、ceil_left_mps 4.2(基準 4.1)'
SCRATCH = '/tmp/claude-1000/-home-jetson-jetracer/f44d13ba-776f-4620-a46a-2574918483a9/scratchpad'


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
    def index(css: str = 'now'):
        html = open(os.path.join(STATIC, 'index.html'), encoding='utf-8').read()
        assert html.count('href="/static/style.css"') == 1
        return Response(html.replace('href="/static/style.css"', 'href="/css/%s.css"' % css), media_type='text/html')

    @app.get('/css/{tag}.css')
    def css_file(tag: str):
        return Response(open(CSS[tag], encoding='utf-8').read(), media_type='text/css', headers={'Cache-Control': 'no-store'})

    @app.get('/frame')
    def frame(w: int, h: int, c: str = '', css: str = 'now'):
        # スマホの幅: Firefox の窓は幅 500 より狭くならないので、iframe の大きさ = 画面(左上に置く・枠なし)
        return Response('<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0;background:#777">'
                        '<iframe id="f" src="/?c=%s&css=%s" style="border:0;display:block;width:%dpx;height:%dpx"></iframe></body></html>'
                        % (c, css, w, h), media_type='text/html')

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

    def js(self, script, *args):
        return self.req('POST', '/session/%s/execute/sync' % self.sid, {'script': script, 'args': list(args)})

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

    def open_frame(self, port, W, H, tag, css):
        self.req('POST', '/session/%s/url' % self.sid, {'url': 'http://127.0.0.1:%d/frame?w=%d&h=%d&c=%s&css=%s' % (port, W, H, tag, css)})

        def ready():
            self.to_frame()
            return self.js('return !!(window.jetformer && window.jetformer.showTab) && innerWidth === %d && innerHeight === %d'
                           ' && [...document.styleSheets].some((x) => (x.href || "").includes("/css/%s.css") && x.cssRules.length > 50)'
                           % (W, H, css))
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
    s.chars_out = cs.filter(([c, r]) => r.left < 0 || r.right > document.documentElement.clientWidth).map(([c]) => c).join('');
    if (tv) {
      const ix = Math.max(0, Math.min(fr.right, tr.right) - Math.max(fr.left, tr.left));
      const iy = Math.max(0, Math.min(fr.bottom, tr.bottom) - Math.max(fr.top, tr.top));
      s.ov = [Math.round(ix * 10) / 10, Math.round(iy * 10) / 10, Math.round(ix * iy)];
      s.gap = Math.round((fr.top - tr.bottom) * 10) / 10;       // トーストの下と ■ の上の間(負 = 重なる)
      s.hidden_chars = cs.filter(([c, r]) => r.right > tr.left && r.left < tr.right && r.bottom > tr.top && r.top < tr.bottom).map(([c]) => c).join('');
      // 字の半分より多くがトーストの下 = 読めない
      s.half_hidden = cs.filter(([c, r]) => {
        const a = Math.max(0, Math.min(r.right, tr.right) - Math.max(r.left, tr.left)) * Math.max(0, Math.min(r.bottom, tr.bottom) - Math.max(r.top, tr.top));
        return a > 0.5 * r.width * r.height; }).map(([c]) => c).join('');
      s.toast_out_top = tr.top < 0;
    }
  }
  window._samp.push(s);
}
sample();
window._sampTimer = setInterval(sample, 50);
return true;
"""

SCROLL_END = r"""
// 走っている間に下までスクロール: 走行タブの最後のカードの下と ■ の上
window.scrollTo(0, document.documentElement.scrollHeight);
const f = document.querySelector('#btn-float-stop').getBoundingClientRect();
const cards = [...document.querySelectorAll('#tab-drive .card')].filter((e) => e.offsetParent !== null);
const last = cards.length ? cards[cards.length - 1].getBoundingClientRect() : null;
return {scrollY: Math.round(scrollY), maxY: document.documentElement.scrollHeight - innerHeight, f_top: Math.round(f.top * 10) / 10,
        last_bottom: last ? Math.round(last.bottom * 10) / 10 : null, hidden_px: last ? Math.max(0, Math.round((last.bottom - f.top) * 10) / 10) : null,
        driving: document.body.classList.contains('driving'), float_visible: !document.querySelector('#btn-float-stop').classList.contains('hidden')};
"""


def run_case(wd, srv, port, W, H, case, css, shots):
    """case: 'normal'(ベンチマークでない本番の ▶)・'b1103'(11/3 の本物の計画のブロック 1 を選ぶ欄で選んで ▶)・'dry'(推論の確認)"""
    srv['running'], srv['dry'], srv['calls'] = False, False, []
    srv['plan_diffs'], srv['bench'], srv['start_diffs'], srv['start_desc'] = [], case.startswith('b1103'), None, None
    wd.open_frame(port, W, H, '%s_%dx%d_%s' % (case, W, H, css), css)
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
        p = os.path.join(OUT, 'shot_%s_%dx%d_%s_t1s.png' % (case, W, H, css))
        wd.shot(p)
        shot_paths.append(p)
        time.sleep(6.5 if case.startswith('b1103') else 3.0)
    else:
        time.sleep(7.5 if case.startswith('b1103') else 4.0)
    wd.js('clearInterval(window._sampTimer)')
    samp = wd.js('return window._samp')
    errs = wd.js('return window._errs')
    confirms = wd.js('return window._confirms.length')
    scroll_end = wd.js(SCROLL_END)
    starts = [c for c in srv['calls'] if 'start' in c]
    return {'W': W, 'H': H, 'case': case, 'css': css, 'before_float_class': before[0], 'before_body_class': before[1], 'samples': samp,
            'errs': errs, 'confirms': confirms, 'n_start_posts': len(starts), 'shots': shot_paths, 'picked': picked,
            'server_plan_diffs': srv.get('start_diffs'), 'server_desc': srv.get('start_desc'), 'scroll_end': scroll_end}


def summarize(r):
    S = r['samples']
    first_drive = next((s['t'] for s in S if s['driving'] and s['fv']), None)
    both = [s for s in S if s['fv'] and s['tv']]
    ov = [s for s in both if s.get('ov') and s['ov'][2] > 0]
    toasts = []
    for s in S:
        if s['tv']:
            key = (s['ttext'], s['tcls'])
            if not toasts or toasts[-1]['key'] != key:
                toasts.append({'key': key, 'from': s['t'], 'to': s['t']})
            toasts[-1]['to'] = s['t']
    f_on = [s for s in S if s['fv']]
    lines = f_on[-1]['lines'] if f_on else None
    out = {'W': r['W'], 'H': r['H'], 'case': r['case'], 'css': r['css'], 'inner': [S[0]['iw'], S[0]['ih'], S[0]['cw']],
           'idle_float_hidden': 'hidden' in r['before_float_class'], 'start_posts': r['n_start_posts'], 'confirms': r['confirms'],
           'errs': r['errs'], 'float_shown_at_ms': first_drive,
           'float_rect': f_on[-1]['fr'] if f_on else None, 'float_lines': lines,
           'float_rects_all': sorted({tuple(s['fr']) for s in f_on}),
           'chars_out': sorted({s.get('chars_out', '') for s in f_on}),
           'toasts': [{'text': k['key'][0], 'cls': k['key'][1], 'seen_ms': [k['from'], k['to']]} for k in toasts],
           'both_samples': len(both),
           'toast_rects_with_float': sorted({tuple(s['tr']) for s in both}),
           'min_gap': min((s['gap'] for s in both), default=None),
           'overlap_ms': [ov[0]['t'], ov[-1]['t']] if ov else None, 'overlap_samples': len(ov),
           'overlap_max': max((s['ov'] for s in ov), key=lambda v: v[2]) if ov else None,
           'toast_rect_at_overlap': ov[len(ov) // 2]['tr'] if ov else None,
           'hidden_chars': sorted({s['hidden_chars'] for s in ov}) if ov else [],
           'half_hidden_chars': sorted({s['half_hidden'] for s in ov}) if ov else [],
           'toast_out_top': any(s.get('toast_out_top') for s in both),
           'z': [S[-1]['tz'], S[-1]['fz']], 'toast_pe': S[-1]['tpe'],
           'shots': r['shots'], 'picked': r['picked'], 'server_plan_diffs': r['server_plan_diffs'], 'scroll_end': r['scroll_end']}
    return out


IDLE = r"""
const [txt, err] = arguments;
document.querySelector('#btn-dr-start').scrollIntoView({block: 'center'});
toast(txt, err);
const t = document.querySelector('#toast'), r = t.getBoundingClientRect();
const ph = document.querySelector('#dr-phase').getBoundingClientRect();
return {toast: [Math.round(r.left * 10) / 10, Math.round(r.top * 10) / 10, Math.round(r.width * 10) / 10, Math.round(r.height * 10) / 10],
        cls: t.className, driving: document.body.classList.contains('driving'), phase_top: Math.round(ph.top * 10) / 10,
        float_hidden: document.querySelector('#btn-float-stop').classList.contains('hidden')};
"""


def idle_toasts(wd, srv, port, W, H, css, texts):
    """止まっているときのトースト(8-1 の直し)の箱。本物の toast() に渡す"""
    srv['running'], srv['dry'], srv['calls'], srv['bench'] = False, False, [], False
    wd.open_frame(port, W, H, 'idle_%dx%d_%s' % (W, H, css), css)
    wd.js("window.jetformer.showTab('drive')")
    assert wait(lambda: wd.js("return window.jetformer.state.drive.paramsLoaded"), 30)
    time.sleep(1.0)
    out = {}
    for name, (txt, err) in texts.items():
        out[name] = wd.js(IDLE, txt, err)
    return {'W': W, 'H': H, 'css': css, 'toasts': out}


def main():
    import uvicorn
    from jetformer.web import server as real_srv      # 断りの文だけ(import だけ。起動しない)
    os.environ['TZ'] = 'Asia/Tokyo'
    time.tzset()
    refusal = '走行は始まっていません(停止中)。' + real_srv._boot_refusal({'boot_at': 1793116404.711}, 1793116404.711 + 3600)
    tags = sys.argv[1:] or ['before', 'now']
    work = Path(tempfile.mkdtemp(prefix='fix-ui-r2-', dir=SCRATCH))
    app, srv = make_app(work)
    port, wd_port = free_port(), free_port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=server.run, daemon=True).start()
    gd_log = open(os.path.join(OUT, 'geckodriver.log'), 'w')
    gd = subprocess.Popen([GECKO, '--port', str(wd_port)], stdout=gd_log, stderr=subprocess.STDOUT)
    profile = tempfile.mkdtemp(prefix='fix-ui-r2-profile-', dir=os.path.expanduser('~/snap/firefox/common'))
    wd = WD(wd_port)
    res = {'files': {}, 'css': {}, 'cases': [], 'idle': []}
    for n in ('index.html', 'app.js', 'style.css'):
        res['files'][n] = hashlib.sha256(open(os.path.join(STATIC, n), 'rb').read()).hexdigest()[:12]
    for t in tags:
        res['css'][t] = hashlib.sha256(open(CSS[t], 'rb').read()).hexdigest()[:12]
    texts = {'refusal': (refusal, True), 'stopped': ('停止しました', False), 'start': ('AI走行 No.12 を開始しました', False),
             'dry': ('推論の確認を開始しました(出力は中立)', False), 'trial3': (TRIAL3, True)}
    t0 = time.time()
    try:
        assert wait(lambda: urllib.request.urlopen('http://127.0.0.1:%d/api/status' % port, timeout=2))
        assert wait(lambda: wd.req('GET', '/status', timeout=3))
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
                'moz:firefoxOptions': {'args': ['-headless', '-profile', profile],
                                       'prefs': {'ui.useOverlayScrollbars': 1, 'widget.gtk.overlay-scrollbars.enabled': True},
                                       'env': {'MOZ_HEADLESS_WIDTH': '1400', 'MOZ_HEADLESS_HEIGHT': '1150'}}}}}
        wd.sid = wd.req('POST', '/session', caps, timeout=180)['sessionId']
        wd.req('POST', '/session/%s/window/rect' % wd.sid, {'width': 1400, 'height': 1150})
        wd.req('POST', '/session/%s/url' % wd.sid, {'url': 'http://127.0.0.1:%d/' % port})
        res['ua'] = wd.js('return navigator.userAgent')
        phone = [(w, h) for w in (360, 375, 390, 412, 430) for h in (660, 700, 760, 800, 844, 900)]
        other = [(600, 800), (768, 900), (844, 390), (1280, 800)]
        more = {(w, h) for w in (360, 375, 390, 412, 430) for h in (700, 900)} | set(other)
        shots = {(360, 700), (390, 660), (430, 900), (768, 900)}
        if os.environ.get('JF_MEASURE_QUICK') == '1':              # 台本の確かめだけ(数は使わない)
            phone, other, more, shots = [(390, 660)], [(1280, 800)], {(390, 660)}, set()
        only = [x.split(':') for x in os.environ.get('JF_MEASURE_ONLY', '').split()]   # 流し直し(例 normal:360x660:before)
        for (W, H) in phone + other:
            for case in ('normal', 'b1103', 'dry'):
                if case != 'normal' and (W, H) not in more:
                    continue
                if case == 'dry' and (W, H) in other and (W, H) != (844, 390):
                    continue
                for css in tags:
                    if only and [case, '%dx%d' % (W, H), css] not in only:
                        continue
                    r = run_case(wd, srv, port, W, H, case, css, (W, H) in shots and case != 'dry')
                    sm = summarize(r)
                    res['cases'].append(sm)
                    with open(os.path.join(OUT, 'raw_%s_%dx%d_%s%s.json' % (case, W, H, css, '_rerun' if only else '')), 'w') as f:
                        json.dump(r, f, ensure_ascii=False)
                    print('%5.0fs' % (time.time() - t0), json.dumps({k: sm[k] for k in ('W', 'H', 'case', 'css', 'float_rect', 'float_lines',
                                                     'min_gap', 'overlap_max', 'half_hidden_chars', 'errs')}, ensure_ascii=False), flush=True)
        idle_sizes = ((360, 660), (360, 760), (390, 700), (390, 844), (412, 800), (430, 900), (600, 800), (768, 900), (1280, 800))
        if os.environ.get('JF_MEASURE_QUICK') == '1':
            idle_sizes = ((390, 700),)
        if os.environ.get('JF_MEASURE_ONLY'):
            idle_sizes = ()
        for (W, H) in idle_sizes:
            for css in tags:
                r = idle_toasts(wd, srv, port, W, H, css, texts)
                res['idle'].append(r)
                print('%5.0fs' % (time.time() - t0), json.dumps(r, ensure_ascii=False)[:300], flush=True)
    finally:
        suffix = '_quick' if os.environ.get('JF_MEASURE_QUICK') == '1' else ('_rerun' if os.environ.get('JF_MEASURE_ONLY') else '')
        with open(os.path.join(OUT, 'measure_%s%s.json' % ('_'.join(tags), suffix)), 'w') as f:
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
