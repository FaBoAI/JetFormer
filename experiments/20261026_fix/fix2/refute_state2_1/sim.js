// 指摘 state2-1 の確かめ: 本物の app.js をまるごと vm で読み込み(DOM は偽物)、画面の操作(イベント)を流す。
// 偽のサーバーは prep.py の値(drive の既定 + drive_settings.json、bench.load_candidates())。本物のサーバーには触れない。
'use strict';
const fs = require('fs'), vm = require('vm');
const [,, APP, INPUT, SCEN] = process.argv;
const inp = JSON.parse(fs.readFileSync(INPUT, 'utf8'));
const BRUN = inp.bench.baseline.run;

// ---- 偽の DOM
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
const timers = [];
const confirms = [], toasts = [], calls = [];

// ---- 偽のサーバー(drive.py の params の受け方: 送られた項目だけ上書き)
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
vm.createContext(ctx);
const src = fs.readFileSync(APP, 'utf8')
  + '\n;globalThis.__T = { state, renderDrive, setDriveParams, loadDrive, readDriveParams, renderPrepHints, benchApply };';
vm.runInContext(src, ctx, { filename: 'app.js' });
const T = ctx.__T, $ = (s) => document.querySelector(s);
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); };
const runShortTimers = async () => { const t = timers.splice(0); for (const [fn, ms] of t) if (ms <= 200) await fn(); await flush(); };
const poll = async () => { T.renderDrive(status()); await flush(); };       // pollDrive の表示の同期(止まっているとき)

async function slideTrim(v) {                     // 人がトリムのスライダーを動かす(input → 120 ms 後に POST)
  const el = $('#dr-trim');
  await el.fire('pointerdown'); el.value = String(v); await el.fire('input'); await el.fire('pointerup'); await el.fire('change');
  await runShortTimers();
}
async function pickBlock(di, bi) { const s = $('#dr-bench-pick'); s.value = `${di}:${bi}`; await s.fire('change'); await flush(); }
async function baselineBtn() { await $('#btn-dr-baseline').fire('click'); await flush(); }
async function start() {
  const n0 = calls.length; confirms.length = 0;
  await $('#btn-dr-start').fire('click'); await flush();
  const st = calls.slice(n0).find((c) => c[1] === '/api/drive/start');
  return { sent_trim: st ? st[2].params.steer_trim_us : null, sent_body_params: st ? st[2].params : null, confirm: confirms[0] || '' };
}

(async () => {
  await flush();
  await T.loadDrive(); await flush(); await runShortTimers();
  await poll();
  const day = inp.bench.days.findIndex((d) => d.date === '2026-11-03');
  const res = { scenario: SCEN, day_index: day, block_1_2: inp.bench.days[day].blocks[1] && { block: inp.bench.days[day].blocks[1].block, id: inp.bench.days[day].blocks[1].id, params: inp.bench.days[day].blocks[1].params, drivable: inp.bench.days[day].blocks[1].drivable } };
  if (SCEN === 'A') {           // 1-1 のあと: トリム +15 → ベンチマーク 2 を選ぶ → 走行開始
    await slideTrim(15); res.server_after_slide = server.steer_trim_us; await poll();
    res.field_after_poll = +$('#dr-trim').value;
    await pickBlock(day, 1);
    res.field_after_block = +$('#dr-trim').value; res.pending_after_block = Object.keys(T.state.drive.pending);
    res.note_after_block = $('#dr-base-note-text').innerHTML;
    T.renderPrepHints(); res.prep_check = $('#dr-check-base').innerHTML;
    await poll(); res.field_after_block_poll = +$('#dr-trim').value;      // 表示の同期で戻るか
    const s = await start();
    res.sent_trim = s.sent_trim; res.server_after_start = server.steer_trim_us;
    res.confirm_has_trim = /トリム|trim/.test(s.confirm); res.confirm = s.confirm; res.sent_params = s.sent_body_params;
    // 次の本(ベンチマーク 3)も
    await poll(); await pickBlock(day, 2); const s2 = await start(); res.next_block_sent_trim = s2.sent_trim;
  } else if (SCEN === 'B') {    // 対照: ブロックを選んだあとにトリム +15 → 走行開始
    await pickBlock(day, 1); await slideTrim(15); await poll();
    const s = await start(); res.sent_trim = s.sent_trim; res.server_after_start = server.steer_trim_us; res.confirm_has_trim = /トリム|trim/.test(s.confirm);
  } else if (SCEN === 'C') {    // 手順書 4 章 1 の「基準の設定」→ 2 のベンチマーク(トリム +15 のあと)
    await slideTrim(15); await poll(); await baselineBtn(); res.field_after_baseline = +$('#dr-trim').value;
    await pickBlock(day, 1); const s = await start(); res.sent_trim = s.sent_trim; res.server_after_start = server.steer_trim_us;
  }
  res.toasts_err = toasts;
  console.log(JSON.stringify(res, null, 1));
})().catch((e) => { console.error('ERR', e && e.stack || e); process.exit(1); });
