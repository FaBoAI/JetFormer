"""反証 state-4 の確かめ(2026-10-26): 手順書 5 章の curl の一行 python が、(a) 2-4(天井+自動合わせ ON)のあと
ベンチマーク 10 を押してから左 4.2 を入れたとき、(b) 走行中に打ったとき、何を出すか。本物の drive.Driver の決まり・
本物の app.js(benchSetFields・setDriveParams・readDriveParams・speedStartLines)・手順書の一行 python そのものを使う。
サーバー・車・装置は使わない。読むだけ。"""
import json, re, shutil, subprocess, sys, threading
from pathlib import Path
JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import drive  # noqa: E402

RB = Path('/home/jetson/jetformer_reports/fix_20261026/runbook_20261103.md').read_text(encoding='utf-8')
m = re.search(r'''"ceil_left_mps": 4\.2\}' \| python3 -c "(.*?)"\n''', RB)
ONE_LINER = m.group(1)

class FakeDriver(object):
    _clean_params = drive.Driver._clean_params
    _speed_ctrl_rules = drive.Driver._speed_ctrl_rules
    _launch_rules = drive.Driver._launch_rules
    def __init__(self):
        self.params = dict(drive.DEFAULT_PARAMS); self.lock = threading.Lock()
    def post(self, values, running=False):      # server.api_drive_params と同じ: ValueError → 400 {"detail": ...}
        try:
            p = self._launch_rules(self._speed_ctrl_rules(self._clean_params(values), running), running)
        except (ValueError, TypeError) as e:
            return 400, {'detail': str(e)}
        self.params = p
        return 200, dict(p)

def print_like_runbook(body):
    r = subprocess.run([sys.executable, '-c', ONE_LINER], input=json.dumps(body, ensure_ascii=False),
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip()

APP_JS = (JF / 'jetformer' / 'web' / 'static' / 'app.js').read_text(encoding='utf-8')
def _extract(js, name):
    lines = js.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('function %s(' % name) or line.startswith('const %s = ' % name):
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0] in ' \t}])'):
                j += 1
            return '\n'.join(lines[i:j]) + '\n'
    raise AssertionError(name)
NAMES = ('DR_PARAMS', 'DR_FLAGS', 'readDriveParams', 'setDriveParams', 'BENCH_FALLBACK', 'benchFieldValues', 'benchSetFields',
         'SPEED_FALLBACK', 'speedSpec', 'ceilSpecText', 'vbatSourceText', 'speedStartLines')
STUBS = r"""
const __el = {};
const mk = () => ({ value: '', checked: false, disabled: false, textContent: '', innerHTML: '',
  classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } }, addEventListener() {} });
const $ = (sel) => (__el[sel] = __el[sel] || mk());
const document = { activeElement: null };
const drBusy = {};
const state = { drive: { pending: {}, st: null }, bench: {} };
function renderSpeedSwitches() {} function renderLaunchSwitch() {}
"""
HARNESS = r"""
// 2-4 のあと: サーバーと画面は天井 ON・自動合わせ ON
state.drive.st = { running: false, params: C.before, speed_ctrl_spec: C.spec_before };
setDriveParams(C.before);
const sw_before = $('#dr-speed-ceiling').checked;
const f = benchFieldValues(BENCH_FALLBACK, ''); f.params = Object.assign({}, f.params, C.block);
benchSetFields(f.params);                                  // ベンチマーク 10 を押す
const pend = Object.assign({}, state.drive.pending);
// curl のあと、画面の定期の同期でサーバーの params(天井 1・左 4.2)を受ける
state.drive.st = { running: false, params: C.after, speed_ctrl_spec: C.spec_after };
setDriveParams(C.after);
const sent = readDriveParams();
process.stdout.write(JSON.stringify({ sw_before, pending_after_press: pend, switch_after_sync: $('#dr-speed-ceiling').checked,
  sent_speed_ceiling: sent.speed_ceiling, sent_speed_auto: sent.speed_auto, confirm_lines: speedStartLines(state.drive.st) }));
"""

def main():
    res = {'one_liner': ONE_LINER}
    drv = FakeDriver()
    drv.post({'speed_ceiling': 1}); drv.post({'speed_auto': 1})          # 2-4 の形(止まっているとき)
    before = dict(drv.params)
    code, body = drv.post({'ceil_left_mps': 4.2})                          # 押したあとの curl(止まっているとき)
    res['a_http'] = code
    res['a_print'] = print_like_runbook(body)
    bench = json.loads((JF / 'data' / 'bench_candidates.json').read_text(encoding='utf-8'))
    day = next(d for d in bench['days'] if d['date'] == '2026-11-03')
    blk = next(b for b in day['blocks'] if b['block'] == 10)
    C = {'before': before, 'after': body, 'block': blk['params'],
         'spec_before': drive.speed_ctrl_spec(before), 'spec_after': drive.speed_ctrl_spec(body)}
    js = '\n'.join([STUBS] + [_extract(APP_JS, n) for n in NAMES] + ['const C = %s;' % json.dumps(C), HARNESS])
    r = subprocess.run([shutil.which('node'), '-e', js], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    res['a_screen'] = json.loads(r.stdout)
    # (b) 走行中に打つ(左 4.1 → 4.2)・launch_us 200
    drv2 = FakeDriver()
    code_b, body_b = drv2.post({'ceil_left_mps': 4.2}, running=True)
    res['b_http'] = code_b; res['b_body'] = body_b; res['b_print'] = print_like_runbook(body_b)
    code_c, body_c = drv2.post({'launch_us': 200}, running=True)
    res['c_http'] = code_c; res['c_body'] = body_c
    # (d) 止まっているとき・同じ値(4.1)を送る → 200
    code_d, body_d = FakeDriver().post({'ceil_left_mps': 4.1}, running=True)
    res['d_same_value_running_http'] = code_d
    print(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
