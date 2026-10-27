"""反証 state-3 の確かめ(2026-10-26): 手順書どおり回 2 → 回 3 → 回 4 を流したとき、左の天井(ceil_left_mps)が
どう動くか。本物の drive.Driver の param の処理(_clean_params・_speed_ctrl_rules・_launch_rules)・本物の app.js
(benchFieldValues・benchSetFields・readDriveParams・speedStartLines・speedSpec)・本物の runs_1103.diffs と
bench_candidates.json の 11/3 のブロックを使う。サーバー・車・装置は使わない。読むだけ。
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402

bench = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
day = next(d for d in bench['days'] if d['date'] == '2026-11-03')
B = {b['block']: b for b in day['blocks']}


class FakeDriver(object):
    """drive.Driver の param の処理だけ(装置を開かない)"""
    _clean_params = drive.Driver._clean_params
    _speed_ctrl_rules = drive.Driver._speed_ctrl_rules
    _launch_rules = drive.Driver._launch_rules

    def __init__(self):
        self.params = dict(drive.DEFAULT_PARAMS)        # 起動 = 既定(RESET_ON_BOOT で ceil_left_mps 4.1)
        self.lock = threading.Lock()

    def api_params(self, values):                       # POST /api/drive/params(止まっているとき)
        p = self._clean_params(values)
        p = self._speed_ctrl_rules(p, False)
        p = self._launch_rules(p, False)
        self.params = p

    def start(self, params):                            # Driver.start の params の部分
        self.params = self._launch_rules(self._speed_ctrl_rules(self._clean_params(params or {}), False), False)
        return dict(self.params)


# ---- 画面(本物の app.js の関数を node で)
APP_JS = (JF / 'jetformer' / 'web' / 'static' / 'app.js').read_text(encoding='utf-8')
NAMES = ('DR_PARAMS', 'DR_FLAGS', 'readDriveParams', 'BENCH_FALLBACK', 'benchFieldValues', 'benchSetFields',
         'SPEED_FALLBACK', 'speedSpec', 'ceilSpecText', 'vbatSourceText', 'speedStartLines')


def _extract(js, name):
    lines = js.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('function %s(' % name) or line.startswith('const %s = ' % name):
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0] in ' \t}])'):
                j += 1
            return '\n'.join(lines[i:j]) + '\n'
    raise AssertionError(name)


STUBS = r"""
const __el = {};
const mk = () => ({ value: '', checked: false, disabled: false, textContent: '', innerHTML: '',
  classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } }, addEventListener() {} });
const $ = (sel) => (__el[sel] = __el[sel] || mk());
const state = { drive: { pending: {}, st: null }, bench: {} };
function renderSpeedSwitches() {} function renderLaunchSwitch() {}
"""
HARNESS = r"""
const out = [];
for (const c of CASES) {
  // c.server = 押す前のサーバーの params。c.block_params = ブロックの params。c.ceil_on = 押したあとに天井のスイッチ ON
  state.drive.st = { running: false, params: c.server, speed_ctrl_spec: c.spec };
  for (const [sel, key] of DR_PARAMS) $(sel).value = c.server[key];
  const f = benchFieldValues(BENCH_FALLBACK, '');
  f.params = Object.assign({}, f.params, c.block_params);
  benchSetFields(f.params);
  if (c.ceil_on) { $('#dr-speed-ceiling').checked = true; }      // スイッチ(サーバーへは speed_ceiling: 1 だけ送る)
  const sent = readDriveParams();
  out.push({ name: c.name, base_has_left: 'ceil_left_mps' in f.params, base_left: f.params.ceil_left_mps,
             sent_has_left: 'ceil_left_mps' in sent, line: speedStartLines(state.drive.st) });
}
process.stdout.write(JSON.stringify(out));
"""


def screen(cases):
    node = shutil.which('node')
    code = '\n'.join([STUBS] + [_extract(APP_JS, n) for n in NAMES] + ['const CASES = %s;' % json.dumps(cases), HARNESS])
    r = subprocess.run([node, '-e', code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def runs_diffs():
    spec = importlib.util.spec_from_file_location('runs_1103', JF / 'experiments' / '20261026_fix' / 'plan' / 'runs_1103.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.diffs


def one_run(drv, block_no, ceil_on, label, log):
    """1 本: ブロックを押す(画面)→ 天井のスイッチ(API speed_ceiling 1)→ 走行開始(readDriveParams を送る)"""
    srv = dict(drv.params)
    r = screen([{'name': label, 'server': srv, 'spec': drive.speed_ctrl_spec(srv),
                 'block_params': B[block_no]['params'], 'ceil_on': ceil_on}])[0]
    if ceil_on:
        drv.api_params({'speed_ceiling': 1})
    sent = {k: v for k, v in (r and {}).items()}
    # 送る値は readDriveParams(ceil_left_mps を含まない) → サーバーの値が残る
    body = dict(drv.params)
    body.pop('ceil_left_mps')                         # readDriveParams に無い(r['sent_has_left'] で確かめた)
    used = drv.start({k: v for k, v in body.items() if k in drive.PARAM_LIMITS})
    log.append({'run': label, 'block': block_no, 'sent_has_left': r['sent_has_left'], 'base_has_left': r['base_has_left'],
                'confirm_line': r['line'].strip(), 'left_used': used['ceil_left_mps'], 'ceiling': used['speed_ceiling'],
                'after_press': B[block_no]['after_press'][:40], 'expect_left': B[block_no]['expect'].get('ceil_left_mps')})
    return used


def scenario(name, steps):
    drv, log = FakeDriver(), []
    for s in steps:
        if s[0] == 'api':
            drv.api_params(s[1])
            log.append({'run': 'API %s' % json.dumps(s[1]), 'left_now': drv.params['ceil_left_mps']})
        else:
            _, bno, ceil_on, label = s
            one_run(drv, bno, ceil_on, label, log)
    return name, log


def main():
    diffs = runs_diffs()
    res = {}
    # (a) 2-5(4.2)→ 2-6(4.1 に戻す)→ 6 章で「左 4.2」に決まる → 回 3・回 4 は「回 2 のよい形のスイッチ」= 天井 ON だけ
    res['a'] = scenario('a', [('api', {'ceil_left_mps': 4.2}), ('run', 10, True, '2-5'),
                              ('api', {'ceil_left_mps': 4.1}), ('run', 11, True, '2-6'),
                              ('run', 12, True, '3-1'), ('run', 14, True, '3-3'), ('run', 20, True, '4-1'), ('run', 21, True, '4-2')])
    # (b) 2-5(4.2)のあと時間切れで 2-6 を飛ばす → 6 章のよい形は 4.1(天井 ON) → 回 3・回 4
    res['b'] = scenario('b', [('api', {'ceil_left_mps': 4.2}), ('run', 10, True, '2-5'),
                              ('run', 12, True, '3-1'), ('run', 20, True, '4-1'), ('run', 21, True, '4-2')])
    # 夜の runs_1103: 12〜23 の expect は ceil_left_mps null → 4.1 でも 4.2 でも印なし
    exp12, exp20 = B[12]['expect'], B[20]['expect']
    res['runs_1103'] = {
        'expect12': exp12, 'expect20': exp20,
        'diffs_12_left41': diffs(exp12, dict(drive.DEFAULT_PARAMS, speed_ceiling=1, ceil_left_mps=4.1, throttle_gain=0.95)),
        'diffs_12_left42': diffs(exp12, dict(drive.DEFAULT_PARAMS, speed_ceiling=1, ceil_left_mps=4.2, throttle_gain=0.95)),
        'diffs_20_left42': diffs(exp20, dict(drive.DEFAULT_PARAMS, speed_ceiling=1, ceil_left_mps=4.2)),
        'diffs_10_left41': diffs(B[10]['expect'], dict(drive.DEFAULT_PARAMS, speed_ceiling=1, ceil_left_mps=4.1)),
    }
    res['after_press_mentions_api_or_left'] = {b: ('API' in B[b]['after_press'] or 'ceil_left' in B[b]['after_press']
                                                   or '4.2' in B[b]['after_press']) for b in range(12, 24)}
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
