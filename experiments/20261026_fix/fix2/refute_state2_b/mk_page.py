"""state-2 の反証(別の係): 本物の index.html の <input type=range> と app.js の関数(DR_PARAMS・DR_FLAGS・readDriveParams・setDriveParams)で、
速度ゲイン 0.92 などが丸められるかを headless Firefox で見る。読むだけ。"""
import re, sys, pathlib
root = pathlib.Path('/home/jetson/jetformer/jetformer/web/static')
html = (root / 'index.html').read_text(); js = (root / 'app.js').read_text()
def elem(i):
    m = re.search(r'<input[^>]*id="%s"[^>]*>' % re.escape(i), html) or re.search(r'<select id="%s".*?</select>' % re.escape(i), html, re.S)
    return m.group(0)
def blk(start, open_='{', close='}'):
    i = js.index(start); k = js.index(open_, i); d = 0
    while True:
        c = js[k]
        if c == open_: d += 1
        elif c == close:
            d -= 1
            if d == 0: return js[i:k + 1] + (';' if open_ == '[' else '')
        k += 1
DRP = blk('const DR_PARAMS = [', '[', ']'); DRF = blk('const DR_FLAGS = [', '[', ']')
ids = re.findall(r"\['#(dr-[a-z-]+)'", DRP); fids = re.findall(r"\['#(dr-[a-z-]+)'", DRF)
parts = ['<b id="%s-v"></b>%s' % (i, elem(i)) for i in ids] + ['<input type="checkbox" id="%s">' % i for i in fids]
parts += [elem(i) for i in ['dr-tof-guard', 'dr-tcp-mode', 'dr-zone-caps', 'dr-speed-ceiling', 'dr-speed-auto', 'dr-launch-boost']]
funcs = '\n'.join([DRP, DRF, blk('function readDriveParams()'), blk('function setDriveParams(p)')])
test = r'''
const R = {}; const tg = $('#dr-throttle-gain'), tr = $('#dr-trim');
R.attr = [tg.getAttribute('step'), tr.getAttribute('step')];
R.direct = {}; for (const v of ['0.92','0.97','1.02','0.93','0.87','1.037','1.10','0.85']) { tg.value = v; R.direct[v] = tg.value; }
// キーボードの矢印 1 回 = stepUp
tg.value = '0.90'; tg.stepUp(); R.stepUp_from_090 = tg.value;
R.trim = {}; for (const v of ['15','17','20']) { tr.value = v; R.trim[v] = tr.value; }
tg.value = '1'; tr.value = '0';
setDriveParams({throttle_gain: 0.92, steer_trim_us: 17});
const rp = readDriveParams();
R.sync = {slider: tg.value, text: $('#dr-throttle-gain-v').textContent, start_gain: rp.throttle_gain, start_trim: rp.steer_trim_us,
          confirm_gain: (+tg.value).toFixed(2), trim_text: $('#dr-trim-v').textContent};
dump('REFUTE2B ' + JSON.stringify(R) + '\n'); document.title = 'done';
'''
page = '''<!doctype html><html><head><meta charset="utf-8"><title>x</title></head><body><div>%s</div><script>
const $ = (s) => document.querySelector(s); const state = {drive: {pending: {}, st: null}}; const drBusy = {};
%s
try { %s } catch (e) { dump('REFUTE2B_ERR ' + e + '\\n'); }
</script></body></html>''' % ('\n'.join(parts), funcs, test)
pathlib.Path(sys.argv[1]).write_text(page); print('ids', ids)
