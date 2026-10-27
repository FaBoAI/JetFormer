"""state-2 の確かめ: 本物の index.html の入力欄と app.js の関数で、速度ゲイン・トリムが丸められるかを見るページを作る。"""
import re, sys, json, pathlib
root = pathlib.Path('/home/jetson/jetformer/jetformer/web/static')
html = (root / 'index.html').read_text()
js = (root / 'app.js').read_text()
out = pathlib.Path(sys.argv[1])

def elem(id_):
    m = re.search(r'<input[^>]*id="%s"[^>]*>' % re.escape(id_), html)
    if m: return m.group(0)
    m = re.search(r'<select id="%s".*?</select>' % re.escape(id_), html, re.S)
    if m: return m.group(0)
    raise SystemExit('no ' + id_)

def vspan(id_):
    m = re.search(r'<b id="%s-v">[^<]*</b>' % re.escape(id_), html)
    return m.group(0) if m else '<b id="%s-v"></b>' % id_

def block(start):
    i = js.index(start)
    j = js.index('{', i)
    if start.startswith('const '):
        j = js.index('[', i)
        depth, k = 0, j
        while True:
            c = js[k]
            if c == '[': depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0: return js[i:js.index(';', k) + 1]
            k += 1
    depth, k = 0, j
    while True:
        c = js[k]
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return js[i:k + 1]
        k += 1

dr_params = block('const DR_PARAMS = [')
ids = re.findall(r"\['#(dr-[a-z-]+)'", dr_params)
flags = block('const DR_FLAGS = [')
fids = re.findall(r"\['#(dr-[a-z-]+)'", flags)
parts = []
for i in ids: parts.append(vspan(i) + elem(i))
for i in fids: parts.append('<input type="checkbox" id="%s">' % i)
for i in ['dr-tof-guard', 'dr-tcp-mode', 'dr-zone-caps', 'dr-speed-ceiling', 'dr-speed-auto', 'dr-launch-boost']:
    parts.append(elem(i))
funcs = '\n'.join([dr_params, flags, block('function readDriveParams()'), block('function setDriveParams(p)'),
                   block('function benchSetFields(p)')])
test = r'''
const R = {};
const tg = $('#dr-throttle-gain'), tr = $('#dr-trim');
R.attrs = {gain: [tg.min, tg.max, tg.step], trim: [tr.min, tr.max, tr.step]};
// 1) 欄に直接入れる(スライダーに入れる値)
R.direct = {};
for (const v of [0.90, 0.92, 0.95, 0.97, 1.00, 1.02, 0.93, 0.87, 1.037, 1.10]) { tg.value = String(v); R.direct[v] = tg.value; }
R.trim_direct = {};
for (const v of [15, 17, 18, 20, -17]) { tr.value = String(v); R.trim_direct[v] = tr.value; }
// 2) サーバーの値(API で入れた 0.92・17)を表示の同期で受ける → 走行開始で送る値
tg.value = '1'; tr.value = '0';
setDriveParams({throttle_gain: 0.92, steer_trim_us: 17});
R.sync = {gain_slider: tg.value, gain_text: $('#dr-throttle-gain-v').textContent, trim_slider: tr.value, trim_text: $('#dr-trim-v').textContent,
          start_sends: {throttle_gain: readDriveParams().throttle_gain, steer_trim_us: readDriveParams().steer_trim_us}};
// 3) ブロックの params(benchSetFields)に 0.92 があるとき
state.drive.st = {params: {throttle_gain: 1.0, steer_trim_us: 0}};
benchSetFields({throttle_gain: 0.97, steer_trim_us: 17});
R.bench = {gain_slider: tg.value, gain_text: $('#dr-throttle-gain-v').textContent, pending: Object.assign({}, state.drive.pending),
           start_sends: {throttle_gain: readDriveParams().throttle_gain, steer_trim_us: readDriveParams().steer_trim_us}};
R.ua = navigator.userAgent;
const s = JSON.stringify(R);
console.log('STATE2RESULT ' + s);
document.getElementById('out').textContent = JSON.stringify(R, null, 1);
'''
page = '''<!doctype html><html><head><meta charset="utf-8"><title>state2</title></head><body>
<div style="display:none">%s</div><pre id="out" style="font-size:14px"></pre>
<script>
const $ = (s) => document.querySelector(s);
const state = {drive: {pending: {}, st: null}};
const drBusy = {};
function renderSpeedSwitches() {} function renderLaunchSwitch() {}
%s
try { %s } catch (e) { console.log('STATE2ERR ' + e + ' ' + e.stack); document.getElementById('out').textContent = 'ERR ' + e; }
</script></body></html>''' % ('\n'.join(parts), funcs, test)
out.write_text(page)
print('wrote', out, 'ids', ids)
