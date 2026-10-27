"""反証 1 回目・state(2026-10-26 15:xx)の確かめ。本物の app.js の関数・本物の drive.Driver の決まりで流す。読むだけ。

A. ブロックを押す → 天井のスイッチだけ ON → 表示の同期 → 走行開始、で送る speed_auto(node + drive.Driver の決まり)
B. 速度ゲインのスライダー(index.html の本物の <input type=range step=0.05>)に 0.92 などを入れる・API の 0.92 を表示の同期で
   受けたあと、走行開始で送る値(headless Firefox。画面の写真に結果を書く)

サーバー・車・装置は使わない。drive_settings.json は書かない(Driver の set_params・start は呼ばず、決まりの関数だけ呼ぶ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix2/probe_state_r1.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402

APP_JS = JF / 'jetformer' / 'web' / 'static' / 'app.js'
INDEX = JF / 'jetformer' / 'web' / 'static' / 'index.html'
TMP = JF / 'experiments' / '20261026_fix' / 'fix2' / 'tmp_state'
NODE = shutil.which('node')


def _extract(js, name):
    lines = js.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('function %s(' % name) or line.startswith('const %s = ' % name):
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0] in ' \t}])'):
                j += 1
            return '\n'.join(lines[i:j]) + '\n'
    raise AssertionError('app.js に %s がありません' % name)


def _handler(js, sel):
    m = re.search(r"^\$\('%s'\)\.addEventListener\('change', async \(\) => \{\n.*?^\}\);\n" % re.escape(sel), js, re.S | re.M)
    assert m, sel
    return m.group(0)


# ---------------------------------------------------------------- A
NAMES_A = ('DR_PARAMS', 'DR_FLAGS', 'drBusy', 'readDriveParams', 'setDriveParams', 'BENCH_FALLBACK', 'benchFieldValues',
           'benchSetFields', 'benchSent', 'speedSupported', 'autoSupported', 'SPEED_FALLBACK', 'speedSpec', 'ceilSpecText',
           'autoLapText', 'vbatSourceText', 'ceilConfirmText', 'autoConfirmText', 'speedStartLines', 'renderSpeedSwitches',
           'launchSupported', 'LAUNCH_FALLBACK', 'launchSpec', 'launchSpecText', 'launchStartLine', 'renderLaunchSwitch')
STUBS_A = r"""
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const __el = {};
const mk = () => ({ value: '', checked: false, disabled: false, textContent: '', innerHTML: '', className: '', __ls: {},
  classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
  addEventListener(ev, fn) { this.__ls[ev] = fn; } });
const $ = (sel) => (__el[sel] = __el[sel] || mk());
const document = { activeElement: null };
const state = { drive: { pending: {}, st: null, models: [] }, bench: {} };
const toasts = [], confirms = [], posts = [];
const toast = (m) => { toasts.push(String(m)); };
const confirm = (t) => { confirms.push(String(t).split('\n')[0]); return true; };
let server = null;
function rules(prev, p) {            // drive.Driver._speed_ctrl_rules(止まっているとき)と同じ。B で Python と照らす
  if (+prev.speed_ceiling === 1 && +p.speed_ceiling === 0) p.speed_auto = 0;
  if (+p.speed_auto === 1 && +p.speed_ceiling === 0) throw new Error('auto without ceiling');
  return p;
}
async function api(method, path, body) {
  posts.push([method, path, body]);
  if (method === 'POST' && path === '/api/drive/params') {
    server = rules(server, Object.assign(JSON.parse(JSON.stringify(server)), body));
    return JSON.parse(JSON.stringify(server));
  }
  throw new Error('この確かめでは呼ばない: ' + path);
}
"""
HARNESS_A = r"""
function poll() {
  const p = JSON.parse(JSON.stringify(server));
  state.drive.st = { running: false, params: p, stats: {}, launch: {}, speed_ctrl_spec: SPEC,
                     launch_spec: { us: p.launch_us, mps: p.launch_mps, max_s: p.launch_max_s, note: '' } };
  setDriveParams(p); renderSpeedSwitches(state.drive.st); renderLaunchSwitch(state.drive.st);
}
function press() {
  const f = benchFieldValues(BENCH_FALLBACK, '');
  f.params = Object.assign({}, f.params, { throttle_gain: 0.9 });
  benchSetFields(f.params);
}
async function sw(sel, on) { const el = $(sel); el.checked = on; await el.__ls.change(); }
function snap(step) {
  return [step, $('#dr-speed-ceiling').checked, $('#dr-speed-auto').checked, !!state.drive.pending.speed_ceiling,
          !!state.drive.pending.speed_auto, $('#dr-auto-note').innerHTML.replace(/<[^>]+>/g, '')];
}
async function run(prev, steps) {
  server = JSON.parse(JSON.stringify(DEFAULTS)); Object.assign(server, prev);
  state.drive.pending = {};
  const c0 = confirms.length;
  poll();
  const log = [snap('start_poll')];
  for (const s of steps) {
    if (s === 'press') press();
    else if (s === 'ceil_on') await sw('#dr-speed-ceiling', true);
    else if (s === 'auto_off') await sw('#dr-speed-auto', false);
    else if (s === 'poll') poll();
    log.push(snap(s));
  }
  const line = speedStartLines(state.drive.st), params = readDriveParams();
  const sent = { speed_ceiling: params.speed_ceiling, speed_auto: params.speed_auto, launch_boost: params.launch_boost };
  benchSent();
  return { log, line, sent, confirms: confirms.slice(c0) };
}
(async () => {
  const out = {};
  for (const [name, prev, steps] of CASES) out[name] = await run(prev, steps);
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""
CASES_A = [
    # 名前, 前の本の後にサーバーに残る値, 手順(手順書: 押す → 天井のスイッチ ON。自動合わせには触れない)
    ('ceil_only_after_auto_run', {'speed_ceiling': 1, 'speed_auto': 1}, ['press', 'poll', 'ceil_on', 'poll']),
    ('ceil_only_after_ceil_run', {'speed_ceiling': 1, 'speed_auto': 0}, ['press', 'poll', 'ceil_on', 'poll']),
    ('ceil_only_after_off_run', {'speed_ceiling': 0, 'speed_auto': 0}, ['press', 'poll', 'ceil_on', 'poll']),
    ('off_after_auto_run', {'speed_ceiling': 1, 'speed_auto': 1}, ['press', 'poll']),
]


def part_a():
    js = APP_JS.read_text(encoding='utf-8')
    spec = drive.speed_ctrl_spec(drive.DEFAULT_PARAMS)
    code = '\n'.join([STUBS_A] + [_extract(js, n) for n in NAMES_A]
                     + [_handler(js, '#dr-speed-ceiling'), _handler(js, '#dr-speed-auto'),
                        'const DEFAULTS = %s;' % json.dumps(drive.DEFAULT_PARAMS), 'const SPEC = %s;' % json.dumps(spec),
                        'const CASES = %s;' % json.dumps(CASES_A), HARNESS_A])
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    # サーバー側: 本物の Driver の決まり(set_params・start と同じ順。保存はしない)
    d = object.__new__(drive.Driver)
    d.params = dict(drive.DEFAULT_PARAMS, speed_ceiling=1, speed_auto=1)            # 2-4(天井 + 自動合わせ)のあと
    p = d._launch_rules(d._speed_ctrl_rules(d._clean_params({'speed_ceiling': 1}), False), False)   # 天井のスイッチ ON の POST
    srv_after_switch = {k: p[k] for k in ('speed_ceiling', 'speed_auto')}
    d.params = p
    sent = out['ceil_only_after_auto_run']['sent']
    p2 = d._launch_rules(d._speed_ctrl_rules(d._clean_params(dict(sent)), False), False)            # 走行開始(start と同じ)
    out['python'] = {'server_after_ceil_switch': srv_after_switch,
                     'server_at_start': {k: p2[k] for k in ('speed_ceiling', 'speed_auto')}}
    return out


# ---------------------------------------------------------------- B
NAMES_B = ('DR_PARAMS', 'DR_FLAGS', 'drBusy', 'readDriveParams', 'setDriveParams')


def part_b():
    js = APP_JS.read_text(encoding='utf-8')
    html = INDEX.read_text(encoding='utf-8')
    ranges = re.findall(r'<input type="range" id="dr-[^"]+"[^>]*>', html)
    ids = [re.search(r'id="([^"]+)"', x).group(1) for x in ranges]
    extra = ''.join('<span id="%s-v"></span>' % i for i in ids)
    checks = ''.join('<input type="checkbox" id="%s">' % i for i in
                     ('dr-esc-reverse', 'dr-recovery', 'dr-start-kick', 'dr-zone-caps', 'dr-speed-ceiling', 'dr-speed-auto',
                      'dr-launch-boost'))
    sels = '<select id="dr-tof-guard"><option value="1">1</option></select><select id="dr-tcp-mode"><option value="0">0</option></select>'
    defaults = dict(drive.DEFAULT_PARAMS, throttle_gain=1.0, lookahead=3, start_kick=1)
    script = '\n'.join(["'use strict';", 'const $ = (s) => document.querySelector(s);',
                        'const state = { drive: { pending: {}, st: null, models: [] }, bench: {} };',
                        'async function api() { throw new Error("no"); }', 'function toast() {}']
                       + [_extract(js, n) for n in NAMES_B] + [r"""
const out = [];
const DEF = %s;
// 1. API で throttle_gain 0.92 → 表示の同期(setDriveParams)→ 走行開始で送る値(readDriveParams)
setDriveParams(Object.assign({}, DEF, { throttle_gain: 0.92, steer_trim_us: 17 }));
let rp = readDriveParams();
out.push('API 0.92 -> label ' + $('#dr-throttle-gain-v').textContent + ' / slider ' + $('#dr-throttle-gain').value + ' / sent ' + rp.throttle_gain);
out.push('API trim 17 -> label ' + $('#dr-trim-v').textContent + ' / sent ' + rp.steer_trim_us);
// 2. 手順書の電圧の表・直しの値をスライダーに入れる(人の手でも同じ刻み 0.05 にしかならない)
for (const v of [0.92, 0.97, 1.02, 0.93, 0.87, 1.037]) {
  $('#dr-throttle-gain').value = v;
  out.push('slider ' + v + ' -> sent ' + readDriveParams().throttle_gain);
}
document.getElementById('out').innerHTML = out.join('<br>');
""" % json.dumps(defaults)])
    page = ('<!doctype html><html><head><meta charset="utf-8"></head><body style="font:20px monospace">%s%s%s%s'
            '<div id="out">not run</div><script>%s</script></body></html>' % ('\n'.join(ranges), extra, checks, sels, script))
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / 'slider.html').write_text(page, encoding='utf-8')
    (TMP / 'prof2').mkdir(exist_ok=True)
    png = TMP / 'slider.png'
    subprocess.run(['firefox', '--headless', '--no-remote', '--profile', str(TMP / 'prof2'), '--window-size', '1000,700',
                    '--screenshot', str(png), 'file://%s' % (TMP / 'slider.html')], capture_output=True, timeout=100)
    return {'screenshot': str(png), 'exists': png.exists(), 'ranges': ranges}


if __name__ == '__main__':
    res = {'A': part_a() if NODE else 'node なし'}
    if '--no-firefox' not in sys.argv:
        res['B'] = part_b()
    print(json.dumps(res, ensure_ascii=False, indent=1))
