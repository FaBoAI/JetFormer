"""画面の電圧・速さの天井・自動合わせのスイッチ(2026-10-25、speed_control_20261025/spec.md 2.7・3.8・4.2 節、6.2 節 V-5・V-6)。

- いつも: 静的なファイル(index.html・app.js)・app.js の計算(node)・web/bench.py の基準とブロック・計画の照らし合わせ
- JF_UI_FIREFOX=1 のときだけ: 偽のサーバー(本物の画面 + 偽の /api/*。センサー・PCA9685 に触れない)をヘッドレス Firefox で開く。
  Firefox は重いので jf-heavy --lock big で動かす:

    cd ~/jetformer && JF_UI_FIREFOX=1 /home/jetson/bin/jf-heavy --lock big --mem 2500M -- python -m pytest -q tests/test_vbat_ui.py
"""

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request

import pytest

from jetformer import drive
from jetformer.web import bench

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
STATIC = os.path.join(ROOT, 'jetformer', 'web', 'static')
APP_JS = os.path.join(STATIC, 'app.js')
NODE = shutil.which('node')
KEEP = ('steer_trim_us', 'rec_back_guard')     # 基準の設定・ブロックで画面に入れない項目(app.js の benchFieldValues)
B = bench.B_RUN


def _src(name):
    return open(os.path.join(STATIC, name), encoding='utf-8').read()


# ---------------------------------------------------------------- 静的なファイル

def test_static_has_vbat_and_speed_controls():
    html = _src('index.html')
    for el in ('dr-vbat', 'dr-speed-panel', 'dr-speed-ceiling', 'dr-speed-auto', 'dr-ceil-spec', 'dr-auto-spec', 'dr-ceil-note',
               'dr-auto-note', 'dr-after-vnote', 'vbat-calib', 'vbat-meter', 'btn-vbat-calib', 'btn-vbat-calib-reset',
               'vbat-calib-note'):
        assert 'id="%s"' % el in html, el
    for sw in ('dr-speed-ceiling', 'dr-speed-auto'):
        tag = re.search(r'<input type="checkbox" id="%s"[^>]*>' % sw, html).group(0)
        assert 'checked' not in tag and 'autocomplete="off"' in tag             # 既定 OFF・再読み込みで前の ON を戻さない
    for words in ('前進を弱めるだけ', 'ブレーキは出さない', '舵は変えない', '走行中は ON にできません', '天井が ON のときだけ',
                  '0.85〜1.20', '1.10 まで', 'ceil_*・sa_*'):
        assert words in html, words
    assert html.index('id="dr-vbat"') < html.index('class="ds-grid"', html.index('id="tab-drive"'))   # 走行タブの上
    js = _src('app.js')
    assert re.search(r'client: 8\b', js) and js.count("'/api/drive/start'") == 1     # 画面の版 8(2026-10-27 fix3 r2-r2-state-3。前は 7)
    assert "p.speed_ceiling = $('#dr-speed-ceiling').checked ? 1 : 0" in js
    assert "'/api/vbat/calib'" in js
    fn = re.search(r'^function benchApply\(.*?^}\n', js, re.S | re.M).group(0)
    assert '/api/drive/' not in fn and 'api(' not in fn                          # 欄に入れるだけ(送らない)
    if NODE:
        r = subprocess.run([NODE, '--check', APP_JS], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- app.js の計算(node)

STUBS = """
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const __el = {};
const $ = (sel) => (__el[sel] = __el[sel] || { value: '', checked: false, textContent: '', innerHTML: '', className: '' });
"""


def _extract(js, name):
    """app.js の一番外側の定義(function NAME( か const NAME =)を、次の一番外側の文の前まで切り出す"""
    lines = js.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('function %s(' % name) or line.startswith('const %s = ' % name):
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0] in ' \t}])'):
                j += 1
            return '\n'.join(lines[i:j]) + '\n'
    raise AssertionError('app.js に %s がありません' % name)


def _node(names, code):
    js = _src('app.js')
    parts = [STUBS] + [_extract(js, n) for n in names]
    r = subprocess.run([NODE, '-e', '\n'.join(parts) + '\n' + code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


SPEC_FNS = ('SPEED_FALLBACK', 'speedSpec', 'ceilSpecText', 'autoLapText', 'vbatSourceText', 'ceilConfirmText',
            'autoConfirmText', 'speedSupported', 'autoSupported')


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_bench_values_never_turn_speed_switches_on():
    base = bench.fallback_baseline()
    on = dict(base, params=dict(base['params'], speed_ceiling=1, speed_auto=1, zone_caps=1))
    got = _node(['BENCH_FALLBACK', 'benchFieldValues'],
                'process.stdout.write(JSON.stringify([benchFieldValues(%s, ""), benchFieldValues(%s, ""), '
                'benchFieldValues({params: {}}, null), BENCH_FALLBACK]));' % (json.dumps(on), json.dumps(base)))
    a, b, c, fb = got
    assert a['params']['speed_ceiling'] == 0 and a['params']['speed_auto'] == 0 and a['params']['zone_caps'] == 0
    assert any('速さの天井' in n for n in a['notes']) and any('自動合わせ' in n for n in a['notes'])
    assert b['params'] == {k: v for k, v in base['params'].items() if k not in KEEP} and b['notes'] == []   # トリム・後ろの見張りは入れない(2026-10-26 fix2)
    assert c['params'] == {'zone_caps': 0}                                         # 無い項目は足さない(古いサーバー)
    assert fb['params'] == base['params']                                          # 控えの値 = サーバーの基準(6 つの新しい項目も)
    for k in ('speed_ceiling', 'ceil_all_mps', 'ceil_left_mps', 'speed_auto', 'auto_target_mps', 'auto_vref_v'):
        assert fb['params'][k] == drive.DEFAULT_PARAMS[k], k


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_confirm_texts_take_values_from_speed_ctrl_spec():
    params = dict(drive.DEFAULT_PARAMS, ceil_all_mps=4.6, ceil_left_mps=3.9, auto_target_mps=4.15, speed_ceiling=0, speed_auto=0)
    spec = drive.speed_ctrl_spec(params)
    st = {'speed_ctrl_spec': spec, 'params': params, 'vbat': {'supported': True}}
    got = _node(SPEC_FNS, 'const st = %s; const sp = speedSpec(st);\n'
                'process.stdout.write(JSON.stringify({sp, c: ceilConfirmText(sp), a: autoConfirmText(sp, st), '
                'ok: speedSupported(st), aok: autoSupported(st), old: speedSupported({params: {}}), '
                'fb: speedSpec({params: {}})}));' % json.dumps(st))
    sp, c, a = got['sp'], got['c'], got['a']
    cs, au = spec['ceiling'], spec['auto']
    assert (sp['all'], sp['left'], sp['target']) == (cs['all_mps'], cs['left_mps'], au['target_mps']) == (4.6, 3.9, 4.15)
    assert (sp['steer'], sp['gz'], sp['hold']) == (cs['left_rule']['steer_us'], cs['left_rule']['gz_dps'], cs['left_rule']['hold_s'])
    assert [sp['mlo'], sp['mhi']] == au['m_range'] and sp['raise'] == au['raise_max_gain']
    assert '全体 4.6・左旋回中 3.9 m/s' in c and cs['rule_text'] in c and cs['note'] in c    # 文はサーバーの値から
    assert '+%.0f µs' % cs['left_rule']['steer_us'] in c and '+%.0f deg/s' % cs['left_rule']['gz_dps'] in c
    assert 'ブレーキなし' in c and '舵は変えない' in c and '走行中は ON にできません' in c
    # ON は次の走行にも残る(区間の速度上限と同じ)。「この走行のあいだだけ」は動きと合わなかった(fix.md)
    assert 'この走行のあいだだけ' not in c + a and 'OFF にするまで' in c and '次の走行にも残ります' in c + a
    assert '当たりを 0 にはしない' in c                                                 # 「ぶつからない」とは言わない
    # 自動合わせの物差しは周の時間(fix.md): 目標 1 周 = 14.52 − 1.50 × 4.15 = 8.295 秒(表示は 8.29)
    assert '目標 1 周 8.29 秒(周の速さ p90 4.15 m/s 相当)' in a and au['note'] in a and '基板の電圧' in a
    assert sp['lapTarget'] == au['target_lap_s'] == pytest.approx(8.295)
    assert '%.2f〜%.2f 倍' % tuple(au['m_range']) in a and '上げるのは %.2f まで' % au['raise_max_gain'] in a
    assert got['ok'] and got['aok'] and not got['old']
    assert got['fb']['all'] == 4.5 and got['fb']['left'] == 4.0 and got['fb']['target'] == 4.2      # 古いサーバーの控え


@pytest.mark.skipif(NODE is None, reason='node がありません')
def test_js_vbat_text_levels():
    cases = [None, {'supported': False}, {'supported': True, 'v': None, 'age_s': 3.2},
             {'supported': True, 'v': 8.31, 'level': 'ok', 'warn_v': 7.9, 'bad_v': 7.7},
             {'supported': True, 'v': 7.80, 'level': 'warn', 'warn_v': 7.9, 'bad_v': 7.7},
             {'supported': True, 'v': 7.62, 'level': 'bad', 'warn_v': 7.9, 'bad_v': 7.7},
             {'supported': True, 'v': None, 'broken_v': 0.5, 'range_v': [6.0, 9.0]},
             {'supported': True, 'v': 8.31, 'level': 'ok', 'broken_v': 12.75, 'range_v': [6.0, 9.0]}]
    got = _node(['vbatBrokenText', 'vbatText', 'vbatRunText'],
                'process.stdout.write(JSON.stringify([...%s.map(vbatText), vbatRunText({v_before: 8.31, state: "done", '
                'v_after: 8.05, v_min: 7.92, scenes_written: 2}), vbatRunText({v_before: null, human_before: "8.2", '
                'state: "waiting"})]));' % json.dumps(cases))
    old, unsup, stale, ok, warn, bad, broken, broken2, run1, run2 = got
    # 6.0〜9.0 V の外の値(使っていない)は赤で「電圧の値がおかしい」(fix.md)
    assert '電圧の値がおかしい' in broken and '0.50 V' in broken and '6.0〜9.0 V の外' in broken and '来ていません' not in broken
    assert '8.31 V' in broken2 and '電圧の値がおかしい' in broken2 and '12.75 V' in broken2
    assert '電圧の値がおかしい' not in ok
    assert '古いサーバー' in old and '基板が対応していません' in unsup and '来ていません' in stale
    assert 'badge ok' in ok and '8.31 V' in ok and 'パックを替える' not in ok
    assert 'badge warn' in warn and '7.80 V' in warn and '低め' in warn
    assert 'badge bad' in bad and 'パックを替える' in bad and '7.70 V 未満' in bad
    assert '8.31 V(基板)' in run1 and '8.05 V' in run1 and '7.92 V' in run1 and 'シーン 2 個' in run1
    assert '8.2 V(説明欄)' in run2 and '待っています' in run2


# ---------------------------------------------------------------- V-6 ベンチの基準・ブロック・計画の照らし合わせ

def test_bench_never_applies_speed_switches_from_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'BENCH_PATH', tmp_path / 'bench_candidates.json')
    monkeypatch.setattr(drive, 'RUNS_DIR', tmp_path / 'runs')
    doc = {'version': 1, 'bench': 'x', 'baseline': {'mode_id': 5, 'params': {'speed_ceiling': 1, 'speed_auto': 1}},
           'candidates': [{'id': 'B', 'run': B}],
           'days': [{'date': '2026-11-03', 'blocks': [{'block': 1, 'id': 'B',
                                                       'params': {'speed_ceiling': 1, 'speed_auto': 1, 'throttle_gain': 0.9}}]}]}
    (tmp_path / 'bench_candidates.json').write_text(json.dumps(doc), encoding='utf-8')
    d = bench.load_candidates()
    w = ' '.join(d['warnings'])
    assert d['baseline']['params']['speed_ceiling'] == 1                          # 計画の値は残す(画面の benchFieldValues が 0 にする)
    assert '速さの天井を ON にしません' in w and '速さのゲインの自動合わせを ON にしません' in w
    blk = d['days'][0]['blocks'][0]
    assert blk['params'] == {'throttle_gain': 0.9}                                 # ブロックの上書きではスイッチを捨てる
    assert 'speed_ceiling' in w and 'スイッチで入れてください' in w


def test_plan_keys_mark_speed_switches_as_exp():
    assert {'speed_ceiling', 'speed_auto'} <= set(bench.PLAN_KEYS)
    base = bench.fallback_baseline()
    data = {'ok': True, 'baseline': base, 'candidates': [{'id': 'B', 'run': B}],
            'days': [{'date': '2026-11-03', 'blocks': [{'block': 1, 'id': 'B', 'run': B}]}]}
    desc = 'bench=x day=2026-11-03 block=1 cand=B model=%s official=1 memo=' % B
    p = dict(base['params'])
    assert bench.plan_check(desc, B, 5, p, data) == (desc, [])
    out, diffs = bench.plan_check(desc, B, 5, dict(p, speed_ceiling=1), data)
    assert 'cand=EXP' in out and 'official=0' in out and any('speed_ceiling' in x for x in diffs)
    out, diffs = bench.plan_check(desc, B, 5, dict(p, speed_ceiling=1, speed_auto=1), data)
    assert len([x for x in diffs if 'speed_' in x]) == 2
    planned = dict(data, baseline=dict(base, params=dict(base['params'], speed_ceiling=1)))     # 基準に 1 と書いた日
    assert bench.plan_check(desc, B, 5, dict(p, speed_ceiling=1), planned) == (desc, [])


# ---------------------------------------------------------------- V-5 ヘッドレス Firefox(JF_UI_FIREFOX=1・--lock big)

FIREFOX = os.environ.get('JF_UI_FIREFOX') == '1'
GECKO = shutil.which('geckodriver') or ('/snap/bin/geckodriver' if os.path.exists('/snap/bin/geckodriver') else None)


def _free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _fake_app(work):
    """本物の画面と、偽の /api/*(状態は STATE を書き換えて変える)。走行は始めない(start は 409)"""
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import Response
    from fastapi.staticfiles import StaticFiles
    runs = work / 'runs'
    d = runs / B
    d.mkdir(parents=True)
    (d / drive.ENGINE_FILE).write_bytes(b'fake')
    (d / 'model_config.json').write_text(json.dumps({'num_modes': 6, 'trained_modes': [5], 'steer_center_us': 1441.0,
                                                     'throttle_neutral_us': 1496.0}))
    (d / 'run.json').write_text('{}')
    drive.RUNS_DIR = runs
    drive.RECOMMENDED_PATH = work / 'rec.json'
    bench.BENCH_PATH = work / 'none.json'
    state = {'params': dict(drive.DEFAULT_PARAMS), 'running': False, 'vbat': {'supported': True, 'v': 8.31, 'age_s': 0.01,
             'level': 'ok', 'warn_v': 7.9, 'bad_v': 7.7, 'scale': 1.0, 'run': None}, 'calls': [], 'spec': True}
    app = FastAPI()
    app.include_router(bench.router)

    @app.get('/')
    def index():
        return Response(open(os.path.join(STATIC, 'index.html'), encoding='utf-8').read(), media_type='text/html')

    @app.get('/api/status')
    def status():
        return {'time': time.time(), 'sensors_running': True, 'sensors': {}, 'recorder': {'recording': False, 'saved_scenes': []},
                'session': {}, 'disk': {'free_gb': 40.0, 'total_gb': 250.0}, 'training': False, 'driving': state['running'],
                'config': {'scene_seconds': 20, 'vehicle_protocol': 'snapshot', 'tof_unit': '', 'camera': ''}}

    @app.get('/api/drive/models')
    def models():
        return drive.list_models()

    @app.get('/api/drive/status')
    def drive_status():
        out = {'phase': 'driving' if state['running'] else 'idle', 'message': '', 'error': None, 'running': state['running'],
               'params': dict(state['params']), 'last': {'run': B, 'mode_id': 5}, 'recording': False, 'elapsed_s': 0,
               'stats': state.get('stats', {}), 'now': state.get('now'), 'auto_end': state.get('auto_end'),
               'history': {'t': []}, 'ref': None, 'vbat': state['vbat'],
               'pca9685': {'address': 64, 'freq_hz': 60, 'steering_ch': 0, 'throttle_ch': 1},
               'limits': {k: list(v) for k, v in drive.PARAM_LIMITS.items()}, 'zone_caps_spec': drive.ZONE_CAPS_SPEC}
        if state['spec']:
            out['speed_ctrl_spec'] = drive.speed_ctrl_spec(state['params'])
        return out

    @app.post('/api/drive/params')
    def params(body: dict = Body(...)):
        state['calls'].append(body)
        p = dict(state['params'])
        for k, v in body.items():
            if k in p:
                p[k] = v
        if state['running'] and any(int(p[k]) == 1 and int(state['params'][k]) == 0 for k in ('speed_ceiling', 'speed_auto')):
            raise HTTPException(400, '走行中は ON にできません')
        if int(p.get('speed_ceiling', 0)) == 0:
            p['speed_auto'] = 0
        state['params'] = p
        return p

    @app.post('/api/drive/start')
    def start(body: dict = Body(...)):
        state['calls'].append({'start': body})
        raise HTTPException(409, '確認用のサーバーなので走りません')

    @app.get('/api/live/values')
    def live():
        return {'imu': {'t': []}, 'vehicle': {'t': []}, 'vbat': state['vbat'],
                'vehicle_now': {'pwm': [1441, 1496, 1980], 'erpm': 0, 'valid': [True] * 3, 'select': False, 'moving': False,
                                'quality': 'snapshot'}}

    @app.get('/api/{rest:path}')
    def anything(rest: str):
        if rest in ('scenes', 'modes', 'jobs', 'datasets', 'runs', 'policies', 'models', 'sim/courses'):
            return []
        raise HTTPException(404)

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app, state


class _WD(object):
    def __init__(self, port):
        self.base, self.sid = 'http://127.0.0.1:%d' % port, None

    def req(self, method, path, body=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read() or b'{}').get('value')

    def js(self, script):
        return self.req('POST', '/session/%s/execute/sync' % self.sid, {'script': script, 'args': []})

    def elem_shot(self, css, path):
        """要素の写真(WebDriver の要素のスクリーンショット)"""
        import base64
        eid = self.req('POST', '/session/%s/element' % self.sid, {'using': 'css selector', 'value': css})
        eid = list(eid.values())[0]
        self.req('POST', '/session/%s/execute/sync' % self.sid,
                 {'script': "arguments[0].scrollIntoView({block: 'center'})", 'args': [{'element-6066-11e4-a52e-4f735466cecf': eid}]})
        time.sleep(0.3)
        with open(path, 'wb') as f:
            f.write(base64.b64decode(self.req('GET', '/session/%s/element/%s/screenshot' % (self.sid, eid))))


def _wait(fn, t=30.0):
    end = time.time() + t
    while time.time() < end:
        try:
            v = fn()
            if v:
                return v
        except Exception:      # noqa: BLE001
            pass
        time.sleep(0.25)
    return None


@pytest.mark.skipif(not (FIREFOX and GECKO), reason='JF_UI_FIREFOX=1 と geckodriver のときだけ(--lock big)')
def test_firefox_drive_tab_vbat_and_switches():
    import uvicorn
    saved = (drive.RUNS_DIR, drive.RECOMMENDED_PATH, bench.BENCH_PATH)
    work = tempfile.mkdtemp(prefix='vbat-ui-')
    from pathlib import Path
    app, state = _fake_app(Path(work))
    port, wd_port = _free_port(), _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    gd_log = open(os.path.join(work, 'geckodriver.log'), 'w')
    gd = subprocess.Popen([GECKO, '--port', str(wd_port)], stdout=gd_log, stderr=subprocess.STDOUT)
    wd = _WD(wd_port)
    # snap の Firefox は /tmp・隠しフォルダを読めないので、プロファイルは snap の書ける場所に作る
    snap_common = os.path.expanduser('~/snap/firefox/common')
    profile = tempfile.mkdtemp(prefix='vbat-ui-profile-', dir=snap_common if os.path.isdir(snap_common) else work)
    try:
        assert _wait(lambda: urllib.request.urlopen('http://127.0.0.1:%d/api/status' % port, timeout=2))
        assert _wait(lambda: wd.req('GET', '/status', timeout=3))
        caps = {'capabilities': {'alwaysMatch': {'browserName': 'firefox', 'pageLoadStrategy': 'eager',
                'moz:firefoxOptions': {'args': ['-headless', '-profile', profile],
                                       'env': {'MOZ_HEADLESS_WIDTH': '1440', 'MOZ_HEADLESS_HEIGHT': '1800'}}}}}
        wd.sid = wd.req('POST', '/session', caps, timeout=180)['sessionId']
        wd.req('POST', '/session/%s/url' % wd.sid, {'url': 'http://127.0.0.1:%d/' % port})
        assert _wait(lambda: wd.js('return !!(window.jetformer && window.jetformer.showTab)'))
        wd.js("window._errs = []; window.addEventListener('error', (e) => window._errs.push(String(e.message)));"
              "window._confirms = []; window._answer = false; window.confirm = (m) => { window._confirms.push(m); return window._answer; };")
        wd.js("window.jetformer.showTab('drive')")
        q = lambda sel, prop: wd.js("const e = document.querySelector('%s'); return e ? e.%s : null;" % (sel, prop))  # noqa: E731
        # 電圧の帯(走行タブの上)
        assert _wait(lambda: '8.31 V' in (q('#dr-vbat', 'textContent') or ''))
        assert 'badge ok' in q('#dr-vbat', 'innerHTML')
        for v, lv, words in ((7.8, 'warn', '低め'), (7.6, 'bad', 'パックを替える')):
            state['vbat'] = dict(state['vbat'], v=v, level=lv)
            assert _wait(lambda: words in (q('#dr-vbat', 'textContent') or '')), lv
            assert lv in q('#dr-vbat', 'className')
        state['vbat'] = {'supported': False, 'v': None, 'age_s': None, 'level': None, 'warn_v': 7.9, 'bad_v': 7.7, 'run': None}
        assert _wait(lambda: '基板が対応していません' in (q('#dr-vbat', 'textContent') or ''))
        # 読み込みで OFF。天井 OFF のあいだ自動合わせは押せない
        assert q('#dr-speed-ceiling', 'checked') is False and q('#dr-speed-auto', 'checked') is False
        assert _wait(lambda: q('#dr-speed-ceiling', 'disabled') is False)
        assert q('#dr-speed-auto', 'disabled') is True
        # ON の確認の文(キャンセル → 送らない・OFF のまま)。値は speed_ctrl_spec から
        spec = drive.speed_ctrl_spec(state['params'])
        wd.js("document.querySelector('#dr-speed-ceiling').click()")
        assert _wait(lambda: wd.js('return window._confirms.length') == 1)
        msg = wd.js('return window._confirms[0]')
        assert spec['ceiling']['rule_text'] in msg and spec['ceiling']['note'] in msg and '全体 4.5・左旋回中 4.0 m/s' in msg
        assert q('#dr-speed-ceiling', 'checked') is False and not state['calls']
        # 了承 → 送る → 自動合わせが押せる → 自動合わせの確認の文
        wd.js('window._answer = true;')
        wd.js("document.querySelector('#dr-speed-ceiling').click()")
        assert _wait(lambda: state['params']['speed_ceiling'] == 1)
        assert _wait(lambda: q('#dr-speed-auto', 'disabled') is False)
        wd.js("document.querySelector('#dr-speed-auto').click()")
        assert _wait(lambda: state['params']['speed_auto'] == 1)
        amsg = wd.js('return window._confirms[2]')
        assert spec['auto']['note'] in amsg and '目標 1 周 8.12 秒' in amsg
        # 天井を OFF → 自動合わせも OFF(送る)
        wd.js("document.querySelector('#dr-speed-ceiling').click()")
        assert _wait(lambda: state['params']['speed_ceiling'] == 0 and state['params']['speed_auto'] == 0)
        assert {'speed_ceiling': 0, 'speed_auto': 0} in state['calls']
        assert _wait(lambda: q('#dr-speed-auto', 'disabled') is True)
        # 走行中(サーバーは OFF)→ ON にできない
        state['running'] = True
        assert _wait(lambda: q('#dr-speed-ceiling', 'disabled') is True)
        state['running'] = False
        # 古いサーバー(speed_ctrl_spec なし)→ 押せない・知らせ
        state['spec'] = False
        assert _wait(lambda: q('#dr-speed-ceiling', 'disabled') is True and '対応していません' in q('#dr-ceil-note', 'textContent'))
        # 基準の設定で天井を ON にしない(サーバーが ON でも、画面は OFF・送るのは走行開始で)
        state['spec'] = True
        state['params'] = dict(state['params'], speed_ceiling=1)
        assert _wait(lambda: q('#dr-speed-ceiling', 'checked') is True)
        wd.js("document.querySelector('#btn-dr-baseline').click()")
        assert _wait(lambda: q('#dr-speed-ceiling', 'checked') is False)
        assert _wait(lambda: '基準の設定で OFF' in q('#dr-ceil-note', 'textContent'), 5)
        time.sleep(1.5)                                                     # 次の問い合わせ(サーバーは ON)でも戻さない
        assert q('#dr-speed-ceiling', 'checked') is False and '基準の設定で OFF' in q('#dr-ceil-note', 'textContent')
        assert not any('start' in c for c in state['calls'])                # 基準の設定は走行を始めない
        assert wd.js('return window._errs') == []
        shots = os.environ.get('JF_UI_SHOTS')
        if shots:                          # 報告用の写真(走行中の表示は drive.py の now.ceil・now.auto の形の偽の値)
            os.makedirs(shots, exist_ok=True)
            wd.js("document.querySelector('#btn-dr-base-undo').click()")
            state['vbat'] = {'supported': True, 'v': 8.27, 'age_s': 0.01, 'level': 'ok', 'warn_v': 7.9, 'bad_v': 7.7, 'scale': 1.0,
                             'run': {'v_before': 8.31, 'human_before': None, 'state': 'running', 'v_after': None, 'v_min': None}}
            state['params'] = dict(state['params'], speed_ceiling=1, speed_auto=1)
            state['running'] = True
            state['stats'] = {'frames': 1820, 'fps': 30.0, 'ceil_applied': 84, 'ceil_neutral': 3, 'ceil_ms': 0.041, 'ai_s': 60.5}
            state['now'] = {'raw': [1500, 1180], 'out': [1500, 1206], 'select': True, 'hold': False, 'erpm': -8100,
                            'ceil': {'on': True, 'error': None, 'active': True, 'left': True, 'cap': 4.0, 'v': 4.05,
                                     'mode': '弱めた', 'applied': True, 'thr_us': 1206.0},
                            'auto': {'on': True, 'error': None, 'm': 1.042, 'm_ff': 1.010, 'm_fb': 1.032, 'gain': 0.938,
                                     'v': 8.27, 'src': 1, 'lap_p90': 4.12, 'lap_s': 8.31, 'laps_used': 3, 'warn': '',
                                     'warn_text': ''}}
            assert _wait(lambda: '弱めた' in (q('#dr-stats', 'textContent') or ''))
            wd.elem_shot('#dr-vbat', os.path.join(shots, 'drive_vbat_running.png'))
            wd.elem_shot('#dr-speed-panel', os.path.join(shots, 'drive_speed_panel_on.png'))
            wd.elem_shot('#dr-stats', os.path.join(shots, 'drive_stats_running.png'))
            state['running'] = False
            state['vbat'] = dict(state['vbat'], v=8.05, run={'v_before': 8.31, 'human_before': None, 'state': 'done',
                                                            'v_after': 8.05, 'v_min': 7.92, 'scenes_written': 1})
            state['auto_end'] = {'on': True, 'm_fb_end': 1.032, 'laps_used': 6, 'laps_skipped': {}, 'warn': '', 'warn_text': '',
                                 'suggest_throttle_gain': 0.929, 'error': None}
            assert _wait(lambda: '8.05 V' in (q('#dr-vbat', 'textContent') or ''))
            wd.elem_shot('#dr-vbat', os.path.join(shots, 'drive_vbat_after.png'))
            wd.elem_shot('#dr-stats', os.path.join(shots, 'drive_stats_after.png'))
            state['vbat'] = dict(state['vbat'], v=7.62, level='bad')
            wd.js("window.jetformer.showTab('live')")
            assert _wait(lambda: 'パックを替える' in (q('#values', 'textContent') or ''))
            wd.elem_shot('#values', os.path.join(shots, 'live_values_bad.png'))
            assert wd.js('return window._errs') == []
    finally:
        try:
            if wd.sid:
                wd.req('DELETE', '/session/%s' % wd.sid, timeout=30)
        except Exception:      # noqa: BLE001
            pass
        gd.terminate()
        gd_log.close()
        server.should_exit = True
        th.join(10)
        drive.RUNS_DIR, drive.RECOMMENDED_PATH, bench.BENCH_PATH = saved
        if os.environ.get('JF_UI_KEEP') != '1':
            shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)
