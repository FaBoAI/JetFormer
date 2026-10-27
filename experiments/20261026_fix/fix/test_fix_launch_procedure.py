"""直し(2026-10-26、反証の major 1)の確かめ: 手順書の「発進の前進の入れ方」を、本物の app.js の関数で流す(node)。

ブロックを押す(benchSetFields)→ 表示の同期(setDriveParams・renderLaunchSwitch)→ スイッチ(本物の change の処理)か API
→ 走行開始(launchStartLine・readDriveParams・benchSent)を、偽のサーバー(drive.DEFAULT_PARAMS)で 1 本ずつ回す。
- 新しい手順(押したあとにスイッチ)は、前の本が ON でも OFF でも、走行開始で launch_boost 1 を送り、確認の文に行が出る。
- 古い手順(API)は、前の本が ON だと 0 を送り、確認の文に行が出ない(反証の見つけたこと。手順書から外した理由)。
- OFF の本は、前の本が ON でも 0 を送る。押す前にスイッチを入れると OFF に戻る(手順書 4 章の順番)。
サーバー・車・ブラウザは使わない。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        -m pytest -q -p no:cacheprovider experiments/20261026_fix/fix/test_fix_launch_procedure.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402

APP_JS = JF / 'jetformer' / 'web' / 'static' / 'app.js'
NODE = shutil.which('node')
# drAckAt・drAcked・postDriveParams は 2026-10-26 fix3 直す 1 回目(r1-state3-1・3-2)で app.js に足した、送信の返事の順番と
# 送れなかったときの pending の戻し(発進の前進のスイッチの change の処理が使う)。足しただけで、この試験の流れは同じ
NAMES = ('DR_PARAMS', 'DR_FLAGS', 'drAckAt', 'drAcked', 'postDriveParams', 'drBusy', 'readDriveParams', 'setDriveParams',
         'BENCH_FALLBACK', 'benchFieldValues',
         'benchSetFields', 'benchSent', 'launchSupported', 'LAUNCH_FALLBACK', 'launchSpec', 'launchSpecText',
         'launchConfirmText', 'launchStartLine', 'renderLaunchSwitch')

STUBS = r"""
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
const confirm = (t) => { confirms.push(String(t)); return true; };
let server = null;
async function api(method, path, body) {
  posts.push([method, path, body]);
  if (method === 'POST' && path === '/api/drive/params') { Object.assign(server, body); return JSON.parse(JSON.stringify(server)); }
  throw new Error('この試験では呼ばない: ' + path);
}
function renderSpeedSwitches() {}
"""

HARNESS = r"""
function poll() {                                  // pollDrive の表示の同期(止まっているとき)
  const p = JSON.parse(JSON.stringify(server));
  state.drive.st = { running: false, params: p, stats: {}, launch: {},
                     launch_spec: { us: p.launch_us, mps: p.launch_mps, max_s: p.launch_max_s, note: '' } };
  setDriveParams(p); renderLaunchSwitch(state.drive.st);
}
function press(block) {                            // benchApply の欄に入れるところ(基準 + ブロックの上書き)
  const f = benchFieldValues(BENCH_FALLBACK, '');
  f.params = Object.assign({}, f.params, block.params || {});
  benchSetFields(f.params);
}
async function sw(on) { const el = $('#dr-launch-boost'); el.checked = on; await el.__ls.change(); }
function apiSet(v) { server.launch_boost = v; }    // 手順書の古い curl(画面の外から POST /api/drive/params)
function start() {                                 // 走行開始のボタン(確認の文の行と、送る値)
  const line = launchStartLine(state.drive.st), params = readDriveParams();
  Object.assign(server, params); benchSent();
  return { line, sent: params.launch_boost };
}
async function run(prev, steps) {
  server = JSON.parse(JSON.stringify(DEFAULTS)); server.launch_boost = prev;   // 前の本の値がサーバーに残っている
  const c0 = confirms.length;
  poll();
  const log = [];
  for (const s of steps) {
    if (s === 'press') press({ params: { throttle_gain: 0.9 } });
    else if (s === 'sw_on') await sw(true);
    else if (s === 'api_on') apiSet(1);
    else if (s === 'poll') poll();
    log.push([s, $('#dr-launch-boost').checked, !!state.drive.pending.launch_boost]);
  }
  const r = start();
  return Object.assign(r, { log, server_after: server.launch_boost, confirms: confirms.length - c0 });
}
(async () => {
  const out = {};
  for (const [name, prev, steps] of CASES) out[name] = await run(prev, steps);
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""

CASES = [
    # 名前, 前の本の launch_boost(サーバーに残る値), 手順
    ('new_on_after_on', 1, ['press', 'poll', 'sw_on', 'poll']),       # 3-8(ON)→ 回 4 で ON・ON の本のやり直し
    ('new_on_after_off', 0, ['press', 'poll', 'sw_on', 'poll']),      # 3-1(OFF)→ 3-2(ON)
    ('old_api_after_on', 1, ['press', 'poll', 'api_on', 'poll']),     # 古い手順(反証の見つけた形)
    ('old_api_after_off', 0, ['press', 'poll', 'api_on', 'poll']),    # 古い手順でも前が OFF なら入る(たまたま)
    ('off_after_on', 1, ['press', 'poll']),                           # 3-2(ON)→ 3-3(OFF)
    ('off_after_off', 0, ['press', 'poll']),
    ('sw_before_press', 0, ['sw_on', 'poll', 'press', 'poll']),       # 押す前に入れた(手順書 4 章で「OFF に戻る」)
    ('retry_without_press', 1, ['poll']),                             # 押し直さずに同じ設定でもう 1 本
]


def _extract(js, name):
    """app.js の一番外側の定義(function NAME( か const NAME =)を、次の一番外側の文の前まで切り出す(tests/test_vbat_ui と同じ)"""
    lines = js.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('function %s(' % name) or line.startswith('const %s = ' % name):
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0] in ' \t}])'):
                j += 1
            return '\n'.join(lines[i:j]) + '\n'
    raise AssertionError('app.js に %s がありません' % name)


def _handler(js):
    m = re.search(r"^\$\('#dr-launch-boost'\)\.addEventListener\('change', async \(\) => \{\n.*?^\}\);\n", js, re.S | re.M)
    assert m, '発進の前進のスイッチの change の処理が見つからない'
    return m.group(0)


@pytest.fixture(scope='module')
def sim():
    if NODE is None:
        pytest.skip('node がありません')
    js = APP_JS.read_text(encoding='utf-8')
    code = '\n'.join([STUBS] + [_extract(js, n) for n in NAMES] + [_handler(js),
                     'const DEFAULTS = %s;' % json.dumps(drive.DEFAULT_PARAMS),
                     'const CASES = %s;' % json.dumps(CASES), HARNESS])
    r = subprocess.run([NODE, '-e', code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_new_procedure_switch_after_press_sends_on(sim):
    for name in ('new_on_after_on', 'new_on_after_off'):
        r = sim[name]
        assert r['sent'] == 1 and r['server_after'] == 1, (name, r)
        assert r['line'].startswith('\n発進の前進: ON('), (name, r['line'])          # 確認の文に行が出る
        assert r['confirms'] == 1, name                                                 # スイッチの確認の文に答えた
    # 前が ON だと、押した直後は OFF・pending(「基準の設定で OFF にしました」)。スイッチで pending が消える
    log = sim['new_on_after_on']['log']
    assert log[0] == ['press', False, True] and log[2] == ['sw_on', True, False] and log[3] == ['poll', True, False]


def test_old_api_procedure_silently_off_after_on_run(sim):
    r = sim['old_api_after_on']
    assert r['sent'] == 0 and r['server_after'] == 0 and r['line'] == ''              # 黙って OFF(反証の major 1)
    assert r['log'][-1] == ['poll', False, True]                                        # API で 1 にしても画面は OFF のまま
    r2 = sim['old_api_after_off']
    assert r2['sent'] == 1 and r2['line'].startswith('\n発進の前進: ON(')               # 前が OFF のときだけ入る(当てにしない)


def test_off_blocks_send_off_even_after_on_run(sim):
    for name in ('off_after_on', 'off_after_off'):
        r = sim[name]
        assert r['sent'] == 0 and r['server_after'] == 0 and r['line'] == '', (name, r)


def test_switch_before_press_is_reset_and_retry_keeps_on(sim):
    r = sim['sw_before_press']
    assert r['sent'] == 0 and r['line'] == ''                                           # 押す前に入れると OFF に戻る
    r = sim['retry_without_press']
    assert r['sent'] == 1 and r['line'].startswith('\n発進の前進: ON(')                  # 押し直さなければ ON のまま


def test_bench_day_1103_procedure_text():
    """bench_candidates.json の 11/3: API で発進の前進を入れない・ON の本はスイッチ・回 3/4 は zone_caps 0"""
    d = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
    day = next(x for x in d['days'] if x['date'] == '2026-11-03')
    assert len(day['blocks']) == 23 and '発進の前進は押したあとに画面のスイッチ' in day['note']
    for b in day['blocks']:
        ap, ex = b.get('after_press', ''), b.get('expect', {})
        assert 'launch_boost' not in ap and 'API で launch' not in ap, b['block']
        assert b['params'] == {'throttle_gain': 0.9}, b['block']                       # 上書きは変えていない
        if ex.get('launch_boost') == 1:
            assert 'スイッチ ON' in ap and '1.5 m' in ap and '発進の前進: ON' in ap, b['block']
        if b['block'] >= 12:
            assert ex.get('zone_caps') == 0, b['block']
    assert d['changes'][-1]['at'].startswith('2026-10-26 13:')


def test_runbook_has_no_api_launch_and_has_checks():
    rb = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md').read_text(encoding='utf-8')
    assert '{"launch_boost": 1}' not in rb and "`launch_boost` 1" not in rb             # 古い入れ方が残っていない
    for words in ('スイッチ', '発進の前進: ON', '1.5 m', '区間の速度上限', '車輪を浮かせ', 'dry run では発進の前進は出ない',
                  'CH3 は人側', 'events.jsonl', '出した'):
        assert words in rb, words


def _runs_1103():
    import importlib.util
    spec = importlib.util.spec_from_file_location('runs_1103', JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runs_1103_reads_launch_events(tmp_path):
    """夜の 2 の表: events.jsonl の launch_boost(drive.py の launch_note が書く形)を数える。無い・壊れた行は飛ばす"""
    m = _runs_1103()
    assert m.launch_events(tmp_path) == {'start': 0, 'skip': [], 'end': []} and m.launch_text(m.launch_events(tmp_path)) == '—'
    rows = [{'kind': 'launch_boost', 'event': 'skip', 'why': 'near', 'reason': '前が近い(TTC の中央値 < 3.3 秒)'},
            {'kind': 'recovery', 'event': 'start'},
            {'kind': 'launch_boost', 'event': 'start', 'us': 250.0, 'mps': 2.5, 'max_s': 1.2},
            {'kind': 'launch_boost', 'event': 'end', 'code': 1, 'reason': '速さに届いた', 'frames': 12}]
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n{壊れた行\n',
                                           encoding='utf-8')
    ev = m.launch_events(tmp_path)
    assert ev == {'start': 1, 'skip': ['前が近い(TTC の中央値 < 3.3 秒)'], 'end': ['速さに届いた']}
    assert m.launch_text(ev) == '始め 1・終わり 速さに届いた・始めない 前が近い(TTC の中央値 < 3.3 秒)'
    assert 'zone_caps' in m.KEYS
    # 区間の速度上限: 計画 0 で実際 1 なら印
    assert m.diffs({'zone_caps': 0, 'launch_boost': None}, {'zone_caps': 1}) == ['zone_caps 1(計画 0)']
