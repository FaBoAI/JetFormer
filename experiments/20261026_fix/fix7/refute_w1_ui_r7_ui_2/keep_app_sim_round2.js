// 本物の app.js をまるごと node の vm で読み込み(DOM・サーバーは偽物)、画面の操作の列を流す(2026-10-26 fix2 直す 2 回目)。
// tests/test_bench_ui_round2.py が使う。もとは experiments/20261026_fix/fix2/refute_state2_1/sim.js(反証の台本)。
//   node app_sim_round2.js APP_JS INPUT_JSON   INPUT = {bench, server_params, steps: [[操作, ...], ...]}
// 操作: ['slide', '#dr-trim', 15]・['pick', 日の番号, ブロックの番号]・['pick_value', '値']・['baseline']・['start']・['poll']
//       ['read', 名前]('trim'・'gain'・'pick'・'pick_first'・'pending'・'note'・'ceil'・'check_base')
// 出力: 操作ごとの結果の列(JSON)。start は {sent: 送った params, confirm: 確認の文}。本物のサーバー・車には触れない
'use strict';
const fs = require('fs'), vm = require('vm');
const [,, APP, INPUT] = process.argv;
const inp = JSON.parse(fs.readFileSync(INPUT, 'utf8'));
const BRUN = inp.bench.baseline.run;

const noop = new Proxy(function () {}, { get: (t, p) => (p === Symbol.iterator ? undefined : (p === Symbol.toPrimitive ? () => '' : noop)), apply: () => noop });
class El {
  constructor(sel) { this.sel = sel; this.value = ''; this.checked = false; this.disabled = false; this.textContent = ''; this._html = '';
    this.className = ''; this.dataset = {}; this.style = {}; this._ls = {}; this.tagName = 'DIV'; this.files = []; this.src = ''; this.href = ''; this.title = ''; this.id = '';
    const cls = new Set(); this.classList = { toggle: (c, on) => { (on === undefined ? !cls.has(c) : on) ? cls.add(c) : cls.delete(c); }, add: (c) => cls.add(c), remove: (c) => cls.delete(c), contains: (c) => cls.has(c) }; }
  get innerHTML() { return this._html; }
  set innerHTML(h) { this._html = String(h);
    this._opts = [...this._html.matchAll(/<option value="([^"]*)"([^>]*)>([^<]*)/g)].map((m) => ({ value: m[1], disabled: /\bdisabled\b/.test(m[2]), dataset: {}, textContent: m[3] }));
    const o = this.options; if (o.length && !o.some((x) => x.value === this.value)) { const f = o.find((x) => !x.disabled); this.value = f ? f.value : ''; } }
  get options() { return this._opts || []; }
  insertBefore() {} get firstChild() { return null; }
  addEventListener(ev, fn) { (this._ls[ev] = this._ls[ev] || []).push(fn); }
  async fire(ev, extra) { const e = Object.assign({ target: this, preventDefault() {}, stopPropagation() {} }, extra || {}); for (const fn of this._ls[ev] || []) await fn(e); }
  querySelector() { return new El('?'); } querySelectorAll() { return []; } closest(s) { return this._closest === undefined ? new El(s) : this._closest; } matches() { return false; }
  insertAdjacentHTML() {} appendChild(x) { return x; } removeChild() {} remove() {} focus() {} blur() {} setAttribute() {} getAttribute() { return null; }
  removeAttribute() {} scrollIntoView() {} getContext() { return noop; } getBoundingClientRect() { return { width: 100, height: 100, left: 0, top: 0 }; }
  play() { return Promise.resolve(); } pause() {} load() {} click() { return this.fire('click'); } contains() { return false; }
}
const els = {};
const document = { querySelector: (s) => (els[s] = els[s] || new El(s)), querySelectorAll: () => [], getElementById: (id) => document.querySelector('#' + id),
  createElement: (t) => new El(t), addEventListener() {}, removeEventListener() {}, body: new El('body'), documentElement: new El('html'), activeElement: null,
  hidden: false, visibilityState: 'visible' };
const store = () => { const m = {}; return { getItem: (k) => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = String(v); }, removeItem: (k) => { delete m[k]; } }; };
const timers = [], confirms = [], toasts = [], calls = [];
const server = JSON.parse(JSON.stringify(inp.server_params));
const MODEL = { name: BRUN, drivable: true, recommended: true, policy_title: 'JetFormer v0', outputs: ['actions'], postprocess: null,
  modes: [{ id: 0, name: '未分類' }, { id: 5, name: 'clean' }], trained_modes: [5], dataset: 'x', stride: 1, best_val_per_mode_us: {} };
function status() { return { running: false, phase: 'idle', params: JSON.parse(JSON.stringify(server)), last: { run: BRUN, mode_id: 5 } }; }
async function fetch(url, opt) {
  const method = (opt && opt.method) || 'GET', body = opt && opt.body ? JSON.parse(opt.body) : undefined;
  calls.push([method, url, body]);
  const ok = (d) => ({ ok: true, status: 200, statusText: 'OK', json: async () => d });
  const path = url.split('?')[0];
  if (method === 'GET' && path === '/api/drive/status') return ok(status());
  if (method === 'GET' && path === '/api/drive/models') return ok([MODEL]);
  if (method === 'GET' && path === '/api/bench/candidates') return ok(inp.bench);
  if (method === 'POST' && path === '/api/drive/params') { Object.assign(server, body); return ok(JSON.parse(JSON.stringify(server))); }
  if (method === 'POST' && path === '/api/drive/start') { Object.assign(server, body.params); return ok({ token: 't', ai_no: 1, plan_diffs: [] }); }
  return { ok: false, status: 404, statusText: 'Not Found', json: async () => ({ detail: 'sim: ' + path }) };
}
const ctx = { document, window: { addEventListener() {}, open() {}, location: { href: '', host: 'x', protocol: 'http:' }, matchMedia: () => ({ matches: false, addEventListener() {} }) },
  localStorage: store(), sessionStorage: store(), fetch, console, navigator: { userAgent: 'sim' }, location: { href: '', host: 'x', protocol: 'http:' },
  setTimeout: (fn, ms) => { timers.push([fn, ms]); return timers.length; }, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
  requestAnimationFrame: () => 0, confirm: (t) => { confirms.push(String(t)); return true; }, alert() {}, performance: { now: () => Date.now() },
  EventSource: class { constructor() {} close() {} addEventListener() {} }, WebSocket: class { constructor() {} close() {} send() {} },
  Audio: class { play() { return Promise.resolve(); } pause() {} }, Image: class {}, URL, URLSearchParams, Blob: class {}, FormData: class { append() {} },
  AbortController, TextDecoder, TextEncoder, Promise, Date, Math, JSON, Object, Array, Number, String, Set, Map, RegExp, Error, Symbol, Intl, isFinite, parseFloat, parseInt };
ctx.window.document = document; ctx.globalThis = ctx;
ctx.toast = undefined;
vm.createContext(ctx);
const src = fs.readFileSync(APP, 'utf8')
  + '\n;globalThis.__T = { state, renderDrive, setDriveParams, loadDrive, readDriveParams, renderPrepHints, benchApply };';
vm.runInContext(src, ctx, { filename: 'app.js' });
const T = ctx.__T, $ = (s) => document.querySelector(s);
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); };
const runShortTimers = async () => { const t = timers.splice(0); const rest = []; for (const x of t) { if (x[1] <= 200) await x[0](); else rest.push(x); } timers.push(...rest); await flush(); };     // 反証用: 長いタイマーを捨てない
const poll = async () => { T.renderDrive(status()); await flush(); };

async function slide(sel, v) {
  const el = $(sel);
  await el.fire('pointerdown'); el.value = String(v); await el.fire('input'); await el.fire('pointerup'); await el.fire('change');
  await runShortTimers();
}
async function pickValue(v) {         // ブラウザと同じ: 選ばれている値と同じなら change を出さない
  const s = $('#dr-bench-pick');
  if (s.value === v) return { fired: false };
  s.value = v; await s.fire('change'); await flush();
  return { fired: true };
}
async function start() {
  const n0 = calls.length; confirms.length = 0;
  await $('#btn-dr-start').fire('click'); await flush();
  const st = calls.slice(n0).find((c) => c[1] === '/api/drive/start');
  return { sent: st ? st[2].params : null, confirm: confirms[0] || '' };
}
function read(what) {
  if (what === 'trim') return +$('#dr-trim').value;
  if (what === 'gain') return +$('#dr-throttle-gain').value;
  if (what === 'pick') return $('#dr-bench-pick').value;
  if (what === 'pick_first') return ($('#dr-bench-pick').options[0] || {}).textContent || '';
  if (what === 'pending') return Object.keys(T.state.drive.pending).sort();
  if (what === 'note') return $('#dr-base-note-text').innerHTML;
  if (what === 'ceil') return !!$('#dr-speed-ceiling').checked;
  if (what === 'check_base') { T.renderPrepHints(); return $('#dr-check-base').innerHTML; }
  if (what === 'server') return JSON.parse(JSON.stringify(server));
  return null;
}

(async () => {
  await flush();
  await T.loadDrive(); await flush(); await runShortTimers();
  await poll();
  const out = [];
  for (const s of inp.steps) {
    const [op, a, b] = s;
    if (op === 'slide') { await slide(a, b); out.push({ op, server: server[{ '#dr-trim': 'steer_trim_us', '#dr-throttle-gain': 'throttle_gain' }[a]] }); }
    else if (op === 'pick') out.push(Object.assign({ op }, await pickValue(`${a}:${b}`)));
    else if (op === 'pick_value') out.push(Object.assign({ op }, await pickValue(a)));
    else if (op === 'baseline') { await $('#btn-dr-baseline').fire('click'); await flush(); out.push({ op }); }
    else if (op === 'ceil_on') { const el = $('#dr-speed-ceiling'); el.checked = true; await el.fire('change'); await flush(); out.push({ op, server: server.speed_ceiling }); }
    else if (op === 'start') out.push(Object.assign({ op }, await start()));
    else if (op === 'poll') { await poll(); out.push({ op }); }
    else if (op === 'read') out.push({ op, what: a, value: read(a) });
    else throw new Error('知らない操作: ' + op);
  }
  console.log(JSON.stringify({ steps: out, toasts }));
})().catch((e) => { console.error('ERR', e && e.stack || e); process.exit(1); });
