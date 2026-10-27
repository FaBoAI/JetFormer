"""指摘 state-4 の確かめ(手順書 5 章の curl の確かめの文)。車・サーバー・装置は使わない。
- 本物の drive.Driver と、server.py の /api/drive/params の処理(本文をそのまま切り出す)を FastAPI TestClient で動かし、
  返った本文を手順書の一行の python(手順書から切り出す)にそのまま流す。
- 本物の app.js の関数(node)で、ブロックを押す → curl → 走行開始で送る speed_ceiling と確認の文の行を見る。
drive_settings.json は書かない(SETTINGS_PATH を tmp に差し替える)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix2/refute_state4/check_state4.py
"""
import json, re, shutil, subprocess, sys, tempfile, threading
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402

TMP = Path(tempfile.mkdtemp(dir=str(Path(__file__).parent)))
drive.SETTINGS_PATH = TMP / 'drive_settings.json'
RB = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md').read_text(encoding='utf-8')
ONELINERS = re.findall(r"""\| python3 -c "(.*?)"\n""", RB)
assert len(ONELINERS) == 2, ONELINERS


def runbook_print(body_text, i):
    r = subprocess.run(['python3', '-c', ONELINERS[i]], input=body_text, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


class FakeRecorder:
    recording = False
    def add_event(self, *a, **k): return False


def make_app(driver):
    from fastapi import FastAPI, Body, HTTPException
    src = (JF / 'jetformer/web/server.py').read_text(encoding='utf-8')
    m = re.search(r"@app\.post\('/api/drive/params'\)\ndef api_drive_params.*?\n\n\n", src, re.S)
    app = FastAPI()
    exec(m.group(0), {'app': app, 'Body': Body, 'HTTPException': HTTPException, 'driver': driver})
    return app


def main():
    from fastapi.testclient import TestClient
    out = {}
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    c = TestClient(make_app(d))
    # 2-4 のあと(天井 ON・自動合わせ ON で走った。止まっている)。サーバーに天井 1 が残る
    d.set_params({'speed_ceiling': 1}); d.set_params({'speed_auto': 1})
    out['server_after_2_4'] = {k: d.params[k] for k in ('speed_ceiling', 'speed_auto', 'ceil_left_mps')}
    # (a) 止まっているとき、ベンチマーク 10 を押した(画面だけ。サーバーへは送らない)→ 手順書の curl
    r = c.post('/api/drive/params', json={'ceil_left_mps': 4.2})
    out['a_status'] = r.status_code
    out['a_print'] = runbook_print(r.text, 0)
    # (b) 走行中に打った(走行のスレッドが生きている = running)
    d.set_params({'ceil_left_mps': 4.1})
    ev = threading.Event(); d.thread = threading.Thread(target=ev.wait, daemon=True); d.thread.start()
    assert d.running
    r = c.post('/api/drive/params', json={'ceil_left_mps': 4.2})
    out['b_status'], out['b_body'] = r.status_code, r.json()
    out['b_print'] = runbook_print(r.text, 0)
    r = c.post('/api/drive/params', json={'launch_us': 200})
    out['b2_status'], out['b2_body'] = r.status_code, r.json()
    out['b2_print'] = runbook_print(r.text, 1)
    ev.set(); d.thread.join()
    # 参考: 止まっているときの launch_us 200
    r = c.post('/api/drive/params', json={'launch_us': 200})
    out['stopped_launch_print'] = runbook_print(r.text, 1)
    # 画面(本物の app.js): 2-4 のあとのサーバーの値で、押す → curl → (スイッチ有り/無し)→ 走行開始
    srv = dict(d.params); srv.update({'speed_ceiling': 1, 'speed_auto': 1, 'ceil_left_mps': 4.1, 'launch_us': 250.0})
    out['ui'] = run_node(srv)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    shutil.rmtree(TMP)


NAMES = ('DR_PARAMS', 'DR_FLAGS', 'drBusy', 'readDriveParams', 'setDriveParams', 'BENCH_FALLBACK', 'benchFieldValues',
         'benchSetFields', 'benchSent', 'speedSupported', 'autoSupported', 'SPEED_FALLBACK', 'speedSpec', 'ceilSpecText',
         'autoLapText', 'vbatSourceText', 'ceilConfirmText', 'autoConfirmText', 'speedStartLines', 'renderSpeedSwitches')
STUBS = r"""
const esc = (s) => String(s);
const __el = {};
const mk = () => ({ value: '', checked: false, disabled: false, textContent: '', innerHTML: '', className: '', __ls: {},
  classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
  addEventListener(ev, fn) { this.__ls[ev] = fn; } });
const $ = (sel) => (__el[sel] = __el[sel] || mk());
const document = { activeElement: null };
const state = { drive: { pending: {}, st: null, models: [] }, bench: {} };
const toasts = [], confirms = [];
const toast = (m) => { toasts.push(String(m)); };
const confirm = (t) => { confirms.push(String(t)); return true; };
let server = null;
async function api(method, path, body) {
  if (method === 'POST' && path === '/api/drive/params') { Object.assign(server, body); return JSON.parse(JSON.stringify(server)); }
  throw new Error('no ' + path);
}
function renderLaunchSwitch() {}
"""
HARNESS = r"""
function poll() {
  const p = JSON.parse(JSON.stringify(server));
  state.drive.st = { running: false, params: p, stats: {}, speed_ctrl_spec: SPEC };
  setDriveParams(p); renderSpeedSwitches(state.drive.st);
}
function press() { const f = benchFieldValues(BENCH_FALLBACK, ''); f.params = Object.assign({}, f.params, { throttle_gain: 0.9 }); benchSetFields(f.params); }
async function sw(on) { const el = $('#dr-speed-ceiling'); el.checked = on; await el.__ls.change(); }
async function run(steps) {
  server = JSON.parse(JSON.stringify(SRV)); state.drive.pending = {};
  $('#dr-speed-ceiling').checked = false; $('#dr-speed-auto').checked = false;
  poll();
  const log = [];
  for (const s of steps) {
    if (s === 'press') press();
    else if (s === 'curl42') server.ceil_left_mps = 4.2;       // 手順書の curl(画面の外)
    else if (s === 'sw_on') await sw(true);
    else if (s === 'poll') poll();
    log.push([s, $('#dr-speed-ceiling').checked, !!state.drive.pending.speed_ceiling, $('#dr-ceil-note').innerHTML]);
  }
  const line = speedStartLines(state.drive.st), params = readDriveParams();
  Object.assign(server, params); benchSent();
  return { log, line, sent_speed_ceiling: params.speed_ceiling, server_ceil_left: server.ceil_left_mps };
}
(async () => {
  const out = {};
  out.skip_switch = await run(['poll', 'press', 'poll', 'curl42', 'poll']);
  out.with_switch = await run(['poll', 'press', 'poll', 'curl42', 'poll', 'sw_on', 'poll']);
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _extract(js, name):
    lines = js.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('function %s(' % name) or line.startswith('const %s = ' % name):
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0] in ' \t}])'):
                j += 1
            return '\n'.join(lines[i:j]) + '\n'
    raise AssertionError(name)


def run_node(srv):
    js = (JF / 'jetformer/web/static/app.js').read_text(encoding='utf-8')
    h = re.search(r"^\$\('#dr-speed-ceiling'\)\.addEventListener\('change', async \(\) => \{\n.*?^\}\);\n", js, re.S | re.M).group(0)
    code = '\n'.join([STUBS] + [_extract(js, n) for n in NAMES] + [h, 'const SRV = %s;' % json.dumps(srv),
                     'const SPEC = %s;' % json.dumps(drive.speed_ctrl_spec(srv)), HARNESS])
    r = subprocess.run([shutil.which('node'), '-e', code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


if __name__ == '__main__':
    main()
