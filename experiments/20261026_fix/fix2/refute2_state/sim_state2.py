"""反証 2 回目・state の確かめ(読むだけ。サーバー・車・ブラウザは使わない)。

本物の app.js の関数(ブロックを押す benchApply・表示の同期 setDriveParams・ベンチマークの選択 renderBenchPick・
走行開始のボタンの処理)を node で動かし、偽のサーバー(drive.DEFAULT_PARAMS + data/drive_settings.json の形)で流す。
bench_candidates.json の 11/3 は bench.load_candidates()(読むだけ)から作る。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261026_fix/fix2/refute2_state/sim_state2.py
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
from jetformer.web import bench  # noqa: E402

APP_JS = JF / 'jetformer' / 'web' / 'static' / 'app.js'
NAMES = ('parseTags', 'DR_PARAMS', 'drModel', 'DR_FLAGS', 'readDriveParams', 'drBusy', 'setDriveParams', 'BENCH_FALLBACK',
         'BENCH_LABEL', 'TOF_GUARD_LABEL', 'benchFieldValues', 'benchDiff', 'benchFmt', 'benchDiffText', 'benchBaseline',
         'renderBenchPick', 'benchSetFields', 'benchApply', 'benchSent', 'TCP_MODES')

STUBS = r"""
const esc = (s) => String(s);
const __el = {};
const mk = (sel) => ({ sel, value: '', checked: false, disabled: false, textContent: '', innerHTML: '', className: '', __ls: {},
  options: [], dataset: {},
  classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
  addEventListener(ev, fn) { (this.__ls[ev] = this.__ls[ev] || []).push(fn); } });
const $ = (sel) => (__el[sel] = __el[sel] || mk(sel));
const document = { activeElement: null };
const state = { drive: { pending: {}, st: null, models: [] }, bench: { data: null, applied: null, prev: null }, modes: [] };
const toasts = [], confirms = [], posts = [];
const toast = (m) => { toasts.push(String(m)); };
const confirm = (t) => { confirms.push(String(t)); return true; };
const modeName = (id) => 'ID ' + id;
let server = null;
async function api(method, path, body) {
  posts.push([method, path, JSON.parse(JSON.stringify(body || {}))]);
  if (method === 'POST' && path === '/api/drive/params') { Object.assign(server, body); return JSON.parse(JSON.stringify(server)); }
  if (method === 'POST' && path === '/api/drive/start') { Object.assign(server, body.params); return { token: 't', ai_no: 1 }; }
  throw new Error('この確かめでは呼ばない: ' + path);
}
function renderDriveModel() { $('#dr-mode').options = [{ value: '5', disabled: false }]; if (!$('#dr-mode').value) $('#dr-mode').value = '5'; }
function updateTcpWarn() {} function updateNoPostNote() {} function renderPrepHints() {}
function renderSpeedSwitches() {} function renderLaunchSwitch() {} function labelDriveModels() {}
function renderBench() { renderBenchPick(state.bench.data); }
function zoneSpecText() { return ''; } function zoneUnknownCap() { return null; }
function prepStartLines() { return ''; } function speedStartLines() { return ''; } function launchStartLine() { return ''; }
function setDriveToken() {} function pollDrive() {}
"""

HARNESS = r"""
function poll() {                                  // pollDrive の表示の同期(止まっているとき)
  const p = JSON.parse(JSON.stringify(server));
  state.drive.st = { running: false, params: p, stats: {} };
  setDriveParams(p);
}
async function startClick() {                      // 本物の ▶ 走行開始の処理
  const c0 = confirms.length, p0 = posts.length;
  for (const fn of $('#btn-dr-start').__ls.click) await fn();
  const sent = posts.slice(p0).find((x) => x[1] === '/api/drive/start');
  return { confirm: confirms.slice(c0).join('\n---\n'), sent: sent ? sent[2].params : null };
}
async function pick(value) {                       // ベンチマークの選択(select の change。選び直しは change が来ない)
  const sel = $('#dr-bench-pick');
  const fired = sel.value !== value;               // ブラウザ: 選ばれている option をもう一度選んでも change は来ない
  sel.value = value;
  if (fired) for (const fn of sel.__ls.change) await fn({ target: sel });
  return fired;
}
(async () => {
  const out = {};
  const B = BENCH.baseline.run;
  state.drive.models = [{ name: B, drivable: true }];
  state.bench.data = BENCH;
  $('#dr-model').value = B; $('#dr-mode').value = '5'; $('#dr-throttle-gain').value = '0.9';
  const day = BENCH.days.findIndex((d) => d.date === '2026-11-03');

  // ---- A. car_check の「直す」: 1-1 のあと steer_trim_us +15 をスライダーで入れて(すぐ送る)、次の 1-2 を選ぶ
  server = JSON.parse(JSON.stringify(SERVER0));
  poll();
  await pick(day + ':0');                          // 1-1
  await startClick(); poll();
  $('#dr-trim').value = '15';                      // スライダー(input → 120 ms 後に送る。ここでは送った後の形)
  server.steer_trim_us = 15; poll();
  const before = { slider: +$('#dr-trim').value, server: server.steer_trim_us };
  await pick(day + ':1');                          // 1-2(止まって 3 周)
  const note = $('#dr-base-note-text').innerHTML;
  poll();
  const afterPress = { slider: +$('#dr-trim').value, pending: !!state.drive.pending.steer_trim_us };
  const r = await startClick();
  out.A = { before, afterPress, note, confirm_has_trim: /トリム|trim/.test(r.confirm), sent_trim: r.sent.steer_trim_us,
            server_after: server.steer_trim_us, confirm: r.confirm };

  // ---- B. 同じベンチマークを選び直す(「押し直すと全部 OFF・0.90 に戻る」): 1-4 の天井 ON のあと、1-4 をやり直す
  server = JSON.parse(JSON.stringify(SERVER0));
  state.drive.pending = {}; state.bench.applied = null; poll();
  await pick(day + ':3');                          // 1-4
  server.speed_ceiling = 1; $('#dr-speed-ceiling').checked = true; poll();   // 押したあとにスイッチで天井 ON
  await startClick(); poll();                      // 走った。サーバーは天井 ON・ゲイン 0.90
  $('#dr-throttle-gain').value = '1.1'; server.throttle_gain = 1.1; poll();   // まちがえて速度を動かした(すぐ送る)
  const selAfter = $('#dr-bench-pick').value;
  const fired = await pick(day + ':3');            // 押し直したつもり
  poll();
  const ceilBefore = $('#dr-speed-ceiling').checked;
  // 手順書 2 章「押したあとに入れるもの: 天井 ON」: もう ON のスイッチを押す(クリックで checked が反転する)
  const ce = $('#dr-speed-ceiling'); ce.checked = !ce.checked;
  for (const fn of ce.__ls.change) await fn();
  poll();
  const r2 = await startClick();
  out.B = { select_value_after_run: selAfter, change_fired: fired, ceiling_checked_before_click: ceilBefore,
            sent_gain: r2.sent.throttle_gain, sent_ceiling: r2.sent.speed_ceiling,
            block_gain: BENCH.days[day].blocks[3].params.throttle_gain,
            posts_after_click: posts.filter((x) => x[1] === '/api/drive/params').slice(-1) };
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
    raise SystemExit('app.js に %s がありません' % name)


def _handler(js, sel, ev):
    m = re.search(r"^\$\('%s'\)\.addEventListener\('%s', (async )?\((e)?\) => \{\n.*?^\}\);\n" % (re.escape(sel), ev), js,
                  re.S | re.M)
    if not m:
        raise SystemExit('%s の %s の処理が無い' % (sel, ev))
    return m.group(0)


def main():
    js = APP_JS.read_text(encoding='utf-8')
    data = bench.load_candidates()                  # 読むだけ(data/bench_candidates.json)
    data['today'] = '2026-11-03'                    # 当日の形(サーバーの time.strftime)
    saved = json.loads((JF / 'data' / 'drive_settings.json').read_text(encoding='utf-8'))
    server0 = dict(drive.DEFAULT_PARAMS)
    server0.update({k: v for k, v in saved['params'].items() if k in drive.DEFAULT_PARAMS})
    for k in drive.RESET_ON_BOOT:                   # 起動したあとの形
        server0[k] = drive.DEFAULT_PARAMS[k]
    code = '\n'.join([STUBS] + [_extract(js, n) for n in NAMES]
                     + [_handler(js, '#dr-bench-pick', 'change'), _handler(js, '#btn-dr-start', 'click'),
                        _handler(js, '#dr-speed-ceiling', 'change'),
                        'const BENCH = %s;' % json.dumps(data, ensure_ascii=False),
                        'const SERVER0 = %s;' % json.dumps(server0), HARNESS])
    r = subprocess.run([shutil.which('node'), '-e', code], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise SystemExit(r.stderr)
    out = json.loads(r.stdout)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
