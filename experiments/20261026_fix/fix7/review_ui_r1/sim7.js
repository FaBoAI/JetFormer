// 反証 7-1 回目 ui の写し(もと tests/fixtures/app_sim_fix6_r2.js。変えたのは 2 行: 本物の 409 の文を (画面の boot_at, 今の boot_at) の組で引く・slide/click の短いタイマーを鳴らすときに長いタイマーを捨てない)
// 2026-10-27 fix6 画面の係 直す 6-2 回目(反証 w2-ui-w1-ui-r62-1。tests/test_ui_fix6_r2.py が使う)。tests/fixtures/app_sim_fix6.js の写し(もとは
// 変えない)に足したもの(操作で選ぶ。選ばなければ app_sim_fix6.js と同じ。ただし下の 2 つの直しは前から入る):
//  - ['models_fail', true/false]: GET /api/drive/models だけ通信の途切れ(TypeError)
//  - ['down_all', true/false]: サーバーが落ちている(GET も POST も全部 TypeError)
//  - ['hold', 道, true/false]: その道の GET を返さない(止まった GET)。返事は ['release', 道] のときのサーバーの値。fetch の signal が
//    打ち切られたら AbortError で断る(ブラウザの fetch と同じ)。['hold_body', 道, true/false]: 頭(200)は返し、本文(json)を返さない
//    (['release', 道] で返す。打ち切りで AbortError)
//  - ['tab', 名前]: 本物の showTab を呼んで待つ。['tick']: 今あるタイマーを長さにかかわらず全部鳴らす(時間が進んだ)。['ntimers']: 残っている
//    タイマーの長さ(ms)の列。['calls']: 前の ['calls'] からの通信の全部([method, url, 返事の番号])
//  - no_init: true なら初めの loadDrive・表示の同期をしない(走行タブをまだ一度も開いていない画面)
//  - 直し 1: 通信の記録の返事の番号を、その通信の行に付ける(前は返事が遅れると最後の行に付いた)
//  - 直し 2: ['timers'] が長いタイマーを戻すときに番号も戻す(前は戻したタイマーを clearTimeout で消せなかった)
// 2026-10-27 fix6 画面の係(反証 6 回目 w1-ui-ui61-2。tests/test_ui_fix6.py が使う)。tests/fixtures/app_sim_fix5.js の写し(もとは変えない)に
// 足したもの(入力・操作で選ぶ。選ばなければ app_sim_fix5.js と同じ):
//  - boot_refusal_texts: {画面の boot_at の文字(null は "null"): 文}。server_boot_check で断るときの 409 の文を、試験が本物の
//    server._boot_refusal で作った文にする(無い鍵は前と同じ「管理画面のサーバーが起動し直しました(sim5: …)」)
//  - ['start_fail']: 次の POST /api/drive/start だけ通信の途切れ(TypeError。GET は通る)
// 2026-10-27 fix5 画面の係(反証 4 回目 t1・t2・t3。tests/test_ui_fix5.py が使う)。もとは experiments/20261026_fix/fix4/review_state/sim4.js
// (反証 4 回目の台本の写し。変えていない)。足したもの(入力で選ぶ。選ばなければ sim4.js と同じ):
//  - server_boot_check: true なら、偽のサーバーの POST /api/drive/start が server.py の _boot_refusal と同じに断る(409): body の
//    boot_at が無い・今の boot_at と違う(起き直した)。boot_at の無いサーバー(入力の boot_at が null)は見ない。既定 false(sim4.js と同じ)
//  - start の結果に boot_at(送った body の boot_at。送らない画面は undefined → null)・status(返事の番号)
// fix4 反証 4 回目(見方: 状態と設定の受け渡し)の写し。もとは tests/fixtures/app_sim_fix3_r3_1.js(変えていない)。足したもの:
//  - ['down', true/false]: サーバーが落ちている(GET /api/drive/status・POST が通信の途切れ)。false で戻る
//  - ['api', {項目: 値}]: 端末の curl(POST /api/drive/params を画面を通さずに。偽のサーバーの決まりで断る・受ける。posts に残る)
//  - ['reboot_on_confirm', 新しい boot_at, {項目: 値}]: 次の confirm() の間にサーバーが起き上がる(down を外し、起こし直しと同じに値を戻す)。
//    ▶ を押したときは落ちていて、確認の文を読んでいる間に systemd が起こし直した、を真似る
// tests/fixtures/app_sim_fix3_r2_1.js の写し(反証 3 回目の experiments/20261026_fix/fix3/refute3_state/sim3.js と同じ足し: models・
// lastStart・checkn・defer・realpoll・eval)に足したもの(2026-10-27 fix3 直す 3 回目・1 組目。tests/test_ui_fix3_r3_1.py が使う。
// もとの写しは変えない):
//  - ['tapday', 日, ブロック]: カードのブロックのボタン(#bench-days の data-bench-day)を押した。結果に confirm(出た確認の文)
//  - read に dry(「モーターに出力しない」)・bench_error(カードの注意)・applied(state.bench.applied)。start の結果に confirms(全部)
// tests/fixtures/app_sim_fix3_r1_2.js の写しに足したもの(2026-10-27 fix3 直す 2 回目・1 組目。tests/test_ui_fix3_r2_1.py が使う。もとの写しは
// 変えない。反証 2 回目の experiments/20261026_fix/fix3/refute2_state/sim2.js と同じ足し + build・script_src):
//  - start の返りに desc(description)・record・dry・build。read に desc・record・st_params・client_ver(▶ の横の版の文)
//  - script_src: document.currentScript.src(本物のブラウザで index が付ける /static/app.js?v=秒。無ければ currentScript は無い)
//  - ['set', sel, 値]: 欄に手で書いた(説明欄など)
// tests/fixtures/app_sim_fix3_r1.js の写しに、サーバーの起動し直し(status の boot_at)を足したもの(2026-10-26 fix3 直す 1 回目・2 組目
// r1-state3-4。tests/test_ui_fix3_r1_2.py が使う。もとの写しは変えない)。足したもの(入力で選ぶ。選ばなければ前と同じ):
//  - boot_at: status の boot_at(drive.Driver の起動の時刻)。無ければ status に入れない(再起動の前の古いサーバー)
//  - ['reboot', 新しい boot_at, {項目: 値}]: サーバーが起動し直した(boot_at を変え、RESET_ON_BOOT の項目を既定に戻す)
// tests/fixtures/app_sim_fix3.js の写しに、送信・問い合わせの順番の入れ替わりと通信の失敗を足したもの(2026-10-26 fix3 直す 1 回目
// r1-state3-1・r1-state3-2。tests/test_ui_fix3_r1.py が使う。もとの写しは変えない)。足したもの(どれも入力で選ぶ。選ばなければ前と同じ):
//  - defer_posts: POST /api/drive/params の返事を ['resolve'](全部)・['resolve1'](いちばん古い 1 本)まで返さない。
//    apply_on_resolve: true なら、サーバーが値を変えるのも返事のとき(問い合わせのほうが先にサーバーに届いた場合)
//  - ['fail_next', N]: 次の N 本の POST /api/drive/params を通信の途切れ(fetch の TypeError)にする
//  - ['gstart']: 本物の pollDrive を呼び、その GET の返事(その時のサーバーの値)を ['gend'](いちばん古い 1 本)まで返さない
//  - ['tap', sel]・['timers'](120 ms のタイマーを待たずに鳴らす)・['poll'] は表示のゲインも返す
//  - 偽のサーバーは走行中の start_kick の 0 → 1 を断る(drive.set_params と同じ)。status に zone_caps_spec(区間の上限のスイッチ)
//  - clearTimeout はブラウザと同じに、まだ鳴っていないタイマーを消す(もとの写しは何もしない)
// 本物の app.js をまるごと node の vm で読み込み(DOM・サーバーは偽物)、画面の操作の列を流す(2026-10-26 fix3 係 ui)。
// tests/test_ui_fix3.py が使う。もとは tests/fixtures/app_sim_round2.js(fix2)。足したもの:
//  - <input type="range"> は本物の index.html の min・max・step で値を丸める(HTML の決まり: 近い段へ、同じ近さなら大きい方)。
//    0.05 刻みのスライダーに 0.92 を入れると 0.9 になる、を node で再現する(ブラウザそのものではない)
//  - status に speed_ctrl_spec・launch_spec(天井・自動合わせ・発進の前進のスイッチが押せる新しいサーバー)・走行中の印
//  - 偽のサーバーは server_params にある項目だけ受け付ける(drive._clean_params と同じ: 知らない項目は読み飛ばす = 古いサーバー)。
//    速さの天井・自動合わせ・区間の上限・発進の前進の決まり(drive._speed_ctrl_rules・_launch_rules)も同じに断る(400)
//  - Event・dispatchEvent(±0.01 のボタンが input を出す)
//   node app_sim_fix3.js APP_JS INDEX_HTML INPUT_JSON   INPUT = {bench, server_params, steps: [[操作, ...], ...], confirm: true}
// 操作: ['slide', sel, 値]・['click', sel]・['check', sel, true/false]・['pick', 日, ブロック]・['baseline']・['undo']・['start']・
//       ['poll']・['running', true/false]・['server_set', {項目: 値}](API = curl で変えた)・['confirm', true/false]・
//       ['read', 名前]・['el', sel, 'value'|'checked'|'disabled'|'text'|'html'|'hidden']・['posts'](前の ['posts'] からの POST)
// status_fail_first: N = 初めの N 回の状態の問い合わせを失敗にし、最初の表示の同期をしない(画面を開いた直後)
// missing_ids: [id, ...] = その id は無い(querySelector が null)。新しい app.js と古い index.html(キャッシュ)の組み合わせ
// 出力: 操作ごとの結果の列(JSON)。start は {sent, client, confirm, refused}。本物のサーバー・車には触れない
'use strict';
const fs = require('fs'), vm = require('vm');
const [,, APP, HTML, INPUT] = process.argv;
const inp = JSON.parse(fs.readFileSync(INPUT, 'utf8'));
const BRUN = inp.bench.baseline.run;
const html = fs.readFileSync(HTML, 'utf8');
const RANGES = {};
for (const m of html.matchAll(/<input type="range" id="([^"]+)"([^>]*)>/g)) {
  const a = (k) => { const x = m[2].match(new RegExp(`\\b${k}="([^"]*)"`)); return x ? x[1] : null; };
  RANGES['#' + m[1]] = { min: +(a('min') ?? 0), max: +(a('max') ?? 100), step: a('step') === 'any' ? null : +(a('step') ?? 1), value: a('value') };
}
const decimals = (x) => { const s = String(x); const i = s.indexOf('.'); return i < 0 ? 0 : s.length - i - 1; };
function snap(r, v) {                  // HTML の range の値の丸め(min・max の中・step の段。同じ近さなら大きい方)
  let x = parseFloat(String(v));
  if (!Number.isFinite(x)) x = r.max < r.min ? r.min : r.min + (r.max - r.min) / 2;
  x = Math.min(r.max, Math.max(r.min, x));
  if (r.step) {
    const n = Math.floor((x - r.min) / r.step + 0.5 + 1e-9);
    x = +(r.min + n * r.step).toFixed(Math.max(decimals(r.step), decimals(r.min)));
    if (x > r.max) x = +(x - r.step).toFixed(decimals(r.step));
  }
  return String(x);
}

const noop = new Proxy(function () {}, { get: (t, p) => (p === Symbol.iterator ? undefined : (p === Symbol.toPrimitive ? () => '' : noop)), apply: () => noop });
class El {
  constructor(sel) { this.sel = sel; this._value = ''; this.checked = false; this.disabled = false; this.textContent = ''; this._html = '';
    this.className = ''; this.dataset = {}; this.style = {}; this._ls = {}; this.tagName = 'DIV'; this.files = []; this.src = ''; this.href = ''; this.title = ''; this.id = '';
    this._range = RANGES[sel] || null; this.min = this._range ? String(this._range.min) : ''; this.max = this._range ? String(this._range.max) : '';
    if (this._range && this._range.value != null) this._value = snap(this._range, this._range.value);
    const cls = new Set(); this._cls = cls;
    this.classList = { toggle: (c, on) => { (on === undefined ? !cls.has(c) : on) ? cls.add(c) : cls.delete(c); }, add: (c) => cls.add(c), remove: (c) => cls.delete(c), contains: (c) => cls.has(c) }; }
  get value() { return this._value; }
  set value(v) { this._value = this._range ? snap(this._range, v) : String(v); }
  get innerHTML() { return this._html; }
  set innerHTML(h) { this._html = String(h);
    this._opts = [...this._html.matchAll(/<option value="([^"]*)"([^>]*)>([^<]*)/g)].map((m) => ({ value: m[1], disabled: /\bdisabled\b/.test(m[2]), dataset: {}, textContent: m[3] }));
    const o = this.options; if (o.length && !o.some((x) => x.value === this.value)) { const f = o.find((x) => !x.disabled); this.value = f ? f.value : ''; } }
  get options() { return this._opts || []; }
  insertBefore() {} get firstChild() { return null; }
  addEventListener(ev, fn) { (this._ls[ev] = this._ls[ev] || []).push(fn); }
  async fire(ev, extra) { const e = Object.assign({ target: this, preventDefault() {}, stopPropagation() {} }, extra || {}); for (const fn of this._ls[ev] || []) await fn(e); }
  dispatchEvent(e) { this.fire(e.type); return true; }
  querySelector() { return new El('?'); } querySelectorAll() { return []; } closest(s) { return this._closest === undefined ? new El(s) : this._closest; } matches() { return false; }
  insertAdjacentHTML() {} appendChild(x) { return x; } removeChild() {} remove() {} focus() {} blur() {} setAttribute() {} getAttribute() { return null; }
  removeAttribute() {} scrollIntoView() {} getContext() { return noop; } getBoundingClientRect() { return { width: 100, height: 100, left: 0, top: 0 }; }
  play() { return Promise.resolve(); } pause() {} load() {} click() { return this.fire('click'); } contains() { return false; }
}
const els = {};
const MISSING = new Set((inp.missing_ids || []).map((i) => '#' + i));   // 古い index.html に無い id(null を返す)
const document = { querySelector: (s) => (MISSING.has(s) ? null : (els[s] = els[s] || new El(s))), querySelectorAll: () => [], getElementById: (id) => document.querySelector('#' + id),
  createElement: (t) => new El(t), addEventListener() {}, removeEventListener() {}, body: new El('body'), documentElement: new El('html'), activeElement: null,
  hidden: false, visibilityState: 'visible' };
const store = () => { const m = {}; return { getItem: (k) => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = String(v); }, removeItem: (k) => { delete m[k]; } }; };
const timers = [], confirms = [], toasts = [], calls = [];
let timerSeq = 0;
let confirmAnswer = inp.confirm === undefined ? true : !!inp.confirm;
const toastEl = document.querySelector('#toast');           // app.js の toast() は #toast の文字を変える。出た文を全部残す
Object.defineProperty(toastEl, 'textContent', { get() { return this._tc || ''; }, set(v) { this._tc = String(v); if (v) toasts.push(String(v)); }, configurable: true });
const server = JSON.parse(JSON.stringify(inp.server_params));
const KNOWN = new Set(Object.keys(server));
let running = false; let lastStart = null;
let deferPosts = !!inp.defer_posts; const deferred = []; let failNext = 0; const applyOnResolve = !!inp.apply_on_resolve;
const deferredGets = []; let deferGet = 0;
let down = false; let rebootOnConfirm = null; let startFail = 0;
let downAll = false; let modelsFail = false; const holdOn = {}, holdBodyOn = {}; const held = [];      // fix6 r2
let statusFail = +(inp.status_fail_first || 0);   // 初めの N 回の GET /api/drive/status を失敗にする(画面を開いた直後)
let bootAt = inp.boot_at === undefined ? null : inp.boot_at;     // r1-state3-4(null = boot_at の無い古いサーバー)
const MODEL = { name: BRUN, drivable: true, recommended: true, policy_title: 'JetFormer v0', outputs: ['actions'], postprocess: null,
  modes: [{ id: 0, name: '未分類' }, { id: 5, name: 'clean' }], trained_modes: [5], dataset: 'x', stride: 1, best_val_per_mode_us: {} };
function status() {
  const st = { running, phase: running ? 'driving' : 'idle', params: JSON.parse(JSON.stringify(server)), last: { run: BRUN, mode_id: 5 } };
  if (KNOWN.has('speed_ceiling')) st.speed_ctrl_spec = { ceiling: { all_mps: server.ceil_all_mps, left_mps: server.ceil_left_mps }, auto: { target_mps: server.auto_target_mps } };
  if (KNOWN.has('launch_boost')) st.launch_spec = { us: server.launch_us, mps: server.launch_mps, max_s: server.launch_max_s };
  // 区間の速度上限のスイッチが押せるサーバー(drive.ZONE_CAPS_SPEC の形。この写しで足した)
  if (KNOWN.has('zone_caps')) st.zone_caps_spec = { version: 'v1.1', caps_mps: { R1: 2.5, GM: 2.2, DE: 3.0, unknown: 3.0 }, unknown_note: '' };
  if (running) Object.assign(st, { run: BRUN, mode_id: 5, now: null, stats: {}, launch: {} });
  if (bootAt !== null) st.boot_at = bootAt;
  return st;
}
// drive.Driver.set_params の決まりのうち画面に関わるもの(_clean_params・_speed_ctrl_rules・_launch_rules・区間の上限)。断るときは文字
function applyParams(body, isRunning) {
  const p = Object.assign({}, server);
  for (const [k, v] of Object.entries(body || {})) if (KNOWN.has(k)) p[k] = v;
  const prev = server, on = (x) => +x === 1;
  if (on(prev.speed_ceiling) && !on(p.speed_ceiling)) p.speed_auto = 0;
  if (isRunning) {
    for (const k of ['zone_caps', 'speed_ceiling', 'speed_auto', 'launch_boost']) if (KNOWN.has(k) && on(p[k]) && !on(prev[k])) return `走行中は ${k} を ON にできません`;
  }
  if (KNOWN.has('speed_auto') && on(p.speed_auto) && !on(p.speed_ceiling)) return '自動合わせは天井が ON のときだけ';
  // drive._opt_rules(2026-10-26 fix3): 見張りを OFF → 中立止めのあとも残す も OFF。走行中は 0 → 1 にできない。残すは見張りが ON のときだけ
  if (KNOWN.has('rec_hold_keep') && on(prev.rec_back_guard) && !on(p.rec_back_guard)) p.rec_hold_keep = 0;
  // drive.set_params(2026-10-26 fix3 r1-rs1-2): 走行中は発進の補助を 0 → 1 にできない
  if (isRunning && KNOWN.has('start_kick') && on(p.start_kick) && !on(prev.start_kick)) return '走行中は発進の補助(start_kick)を ON にできません';
  if (isRunning) for (const k of ['rec_back_guard', 'rec_hold_keep', 'launch_no_kick']) if (KNOWN.has(k) && on(p[k]) && !on(prev[k])) return `走行中は「${k}」を ON にできません`;
  if (KNOWN.has('rec_hold_keep') && on(p.rec_hold_keep) && !on(p.rec_back_guard)) return '中立止めのあとも残すは、後ろの見張りが ON のときだけ';
  Object.assign(server, p);
  return null;
}
async function fetch(url, opt) {
  const i = calls.length;
  const res = await fetch0(url, opt);
  calls[i].push(res.status);                          // [method, url, body, status](fix6 r2: その通信の行に付ける)
  return res;
}
const abortErr = () => new DOMException('The operation was aborted.', 'AbortError');
function holdUntilRelease(path, opt, make) {          // fix6 r2: 返事を ['release', 道] まで返さない(signal の打ち切りで AbortError)
  return new Promise((resolve, reject) => {
    const h = { path, go: () => resolve(make()) };
    held.push(h);
    const sig = opt && opt.signal;
    if (sig) sig.addEventListener('abort', () => { const k = held.indexOf(h); if (k >= 0) held.splice(k, 1); reject(abortErr()); });
  });
}
const path0 = (u) => u.split('?')[0];
async function fetch0(url, opt) {
  const method = (opt && opt.method) || 'GET', body = opt && opt.body ? JSON.parse(opt.body) : undefined;
  calls.push([method, url, body]);
  if (downAll) throw new TypeError('Failed to fetch (sim6r2: サーバーが落ちている。GET も全部)');
  if (modelsFail && path0(url) === '/api/drive/models') throw new TypeError('Failed to fetch (sim6r2: モデルの一覧だけ通らない)');
  if (method === 'GET' && holdOn[path0(url)]) {
    const u = url, o = Object.assign({}, opt || {}); delete o.signal;
    return holdUntilRelease(path0(url), opt, () => fetch0b(u, o));
  }
  if (method === 'GET' && holdBodyOn[path0(url)]) {
    const u = url, o = Object.assign({}, opt || {}), sig = opt && opt.signal; delete o.signal;
    return { ok: true, status: 200, statusText: 'OK', json: () => holdUntilRelease(path0(url), { signal: sig }, () => fetch0b(u, o).then((r) => r.json())) };
  }
  return fetch0b(url, opt);
}
async function fetch0b(url, opt) {
  const method = (opt && opt.method) || 'GET', body = opt && opt.body ? JSON.parse(opt.body) : undefined;
  if (down && (path0(url) === '/api/drive/status' || method === 'POST')) throw new TypeError('Failed to fetch (sim4: サーバーが落ちている)');
  const ok = (d) => ({ ok: true, status: 200, statusText: 'OK', json: async () => d });
  const bad = (code, msg) => ({ ok: false, status: code, statusText: 'ERR', json: async () => ({ detail: msg }) });
  const path = url.split('?')[0];
  if (method === 'GET' && path === '/api/drive/status') { if (statusFail > 0) { statusFail--; return bad(503, 'sim: まだ読めない'); } if (deferGet > 0) { deferGet--; const snap = ok(status()); return new Promise((r) => deferredGets.push(() => r(snap))); } return ok(status()); }
  if (method === 'GET' && path === '/api/drive/models') return ok(inp.models ? inp.models.map((n) => Object.assign({}, MODEL, { name: n, recommended: n === BRUN })) : [MODEL]);
  if (method === 'GET' && path === '/api/bench/candidates') return ok(inp.bench);
  if (method === 'POST' && path === '/api/drive/params' && failNext > 0) { failNext--; throw new TypeError('Failed to fetch (sim: 通信の途切れ)'); }
  if (method === 'POST' && path === '/api/drive/params') {
    const go = () => { const e = applyParams(body, running); return e ? bad(400, e) : ok(JSON.parse(JSON.stringify(server))); };
    if (!deferPosts) return go();
    if (applyOnResolve) return new Promise((r) => deferred.push(() => r(go())));
    const res = go();
    return new Promise((r) => deferred.push(() => r(res)));
  }
  if (method === 'POST' && path === '/api/drive/start' && startFail > 0) { startFail--; throw new TypeError('Failed to fetch (sim6: 走行開始が届かない)'); }
  if (method === 'POST' && path === '/api/drive/start') {
    if ((+body.client || 0) < (inp.server_client_version || 7)) return bad(409, '管理画面が古いままです');
    if (inp.server_boot_check && bootAt !== null) {          // server.py の _boot_refusal と同じ(fix5 t1)
      const seen = body.boot_at === undefined || body.boot_at === null ? null : +body.boot_at;
      const real = inp.boot_refusal_texts && (inp.boot_refusal_texts[String(seen) + '->' + String(bootAt)] || inp.boot_refusal_texts[String(seen)]);     // r7: 画面の boot_at と今の boot_at の組で引く     // 本物の server._boot_refusal の文(fix6)
      if (seen === null || !Number.isFinite(seen) || Math.abs(seen - bootAt) > 5e-4) return bad(409, real || `管理画面のサーバーが起動し直しました(sim5: 画面 ${seen}・今 ${bootAt})`);
    }
    const e = applyParams(body.params, false); if (e) return bad(400, e);
    running = true; lastStart = body; return ok({ token: 't', ai_no: 1, plan_diffs: [] });
  }
  return bad(404, 'sim: ' + path);
}
class Event { constructor(type) { this.type = type; } }
const ctx = { document, window: { addEventListener() {}, open() {}, location: { href: '', host: 'x', protocol: 'http:' }, matchMedia: () => ({ matches: false, addEventListener() {} }) },
  localStorage: store(), sessionStorage: store(), fetch, console, navigator: { userAgent: 'sim' }, location: { href: '', host: 'x', protocol: 'http:' },
  setTimeout: (fn, ms) => { const id = ++timerSeq; timers.push([fn, ms, id]); return id; },
  clearTimeout: (id) => { const i = timers.findIndex((x) => x[2] === id); if (i >= 0) timers.splice(i, 1); },   // ブラウザと同じに消す setInterval: () => 0, clearInterval() {},
  requestAnimationFrame: () => 0, confirm: (t) => { confirms.push(String(t)); if (rebootOnConfirm) { const [ba, rp] = rebootOnConfirm; rebootOnConfirm = null; down = false; bootAt = ba; running = false; Object.assign(server, rp || {}); } return confirmAnswer; }, alert() {}, performance: { now: () => Date.now() },
  EventSource: class { constructor() {} close() {} addEventListener() {} }, WebSocket: class { constructor() {} close() {} send() {} },
  Audio: class { play() { return Promise.resolve(); } pause() {} }, Image: class {}, URL, URLSearchParams, Blob: class {}, FormData: class { append() {} },
  Event, AbortController, TextDecoder, TextEncoder, Promise, Date, Math, JSON, Object, Array, Number, String, Set, Map, RegExp, Error, Symbol, Intl, isFinite, parseFloat, parseInt };
ctx.window.document = document; ctx.globalThis = ctx;
if (inp.script_src) document.currentScript = { src: inp.script_src };     // 本物のブラウザの読み込み中の script
vm.createContext(ctx);
const src = fs.readFileSync(APP, 'utf8')
  + '\n;globalThis.__T = { state, renderDrive, setDriveParams, loadDrive, readDriveParams, renderPrepHints, benchApply, pollDrive };';
vm.runInContext(src, ctx, { filename: 'app.js' });
const T = ctx.__T, $ = (s) => document.querySelector(s);
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); };
const runShortTimers = async () => { const t = timers.splice(0); const rest = []; for (const x of t) { if (x[1] <= 200) await x[0](); else rest.push(x); } timers.push(...rest); await flush(); };     // r7: 長いタイマー(表示の問い合わせの輪)を捨てない(もとは捨てていた)
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
  return { sent: st ? st[2].params : null, client: st ? st[2].client : null, confirm: confirms[0] || '', confirms: confirms.slice(), refused: st ? !running : null,
           desc: st ? st[2].description : null, record: st ? st[2].record : null, dry: st ? st[2].dry_run : null,
           build: st ? st[2].build : null, run: st ? st[2].run : null,
           boot_at: st ? (st[2].boot_at === undefined ? null : st[2].boot_at) : null, status: st ? st[3] : null };
}
function read(what) {
  if (what === 'trim') return +$('#dr-trim').value;
  if (what === 'gain') return +$('#dr-throttle-gain').value;
  if (what === 'gain_text') return $('#dr-throttle-gain-v').textContent;
  if (what === 'pick') return $('#dr-bench-pick').value;
  if (what === 'pending') return Object.keys(T.state.drive.pending).sort();
  if (what === 'note') return $('#dr-base-note-text').innerHTML;
  if (what === 'ceil') return !!$('#dr-speed-ceiling').checked;
  if (what === 'auto') return !!$('#dr-speed-auto').checked;
  if (what === 'launch') return !!$('#dr-launch-boost').checked;
  if (what === 'check_base') { T.renderPrepHints(); return $('#dr-check-base').innerHTML; }
  if (what === 'server') return JSON.parse(JSON.stringify(server));
  if (what === 'toasts') return toasts.slice();
  if (what === 'confirms') return confirms.slice();
  if (what === 'params') return T.readDriveParams();
  if (what === 'desc') return $('#dr-desc').value;
  if (what === 'record') return !!$('#dr-record').checked;
  if (what === 'st_params') return JSON.parse(JSON.stringify((T.state.drive.st || {}).params || {}));
  if (what === 'client_ver') return $('#dr-client-ver').textContent;
  if (what === 'dry') return !!$('#dr-dry').checked;
  if (what === 'bench_error') return $('#bench-error').innerHTML;
  if (what === 'applied') return JSON.parse(JSON.stringify(T.state.bench.applied || null));
  return null;
}
function readEl(sel, prop) {
  const el = $(sel);
  if (prop === 'hidden') return el.classList.contains('hidden');
  if (prop === 'text') return el.textContent;
  if (prop === 'html') return el.innerHTML;
  return el[prop];
}

(async () => {
  await flush();
  if (!inp.no_init) {                // no_init: 走行タブをまだ一度も開いていない画面(fix6 r2。前と同じ初めの読み込みをしない)
    await T.loadDrive(); await flush(); await runShortTimers();
    if (!inp.status_fail_first) await poll();
  }
  const out = [];
  let mark = calls.length;
  let cmark = calls.length;
    for (const s of inp.steps) {
    const [op, a, b] = s;
    if (op === 'slide') { await slide(a, b); out.push({ op, value: $(a).value }); }
    else if (op === 'click') { confirms.length = 0; await $(a).fire('click'); await flush(); await runShortTimers(); out.push({ op, value: $('#dr-throttle-gain').value }); }
    else if (op === 'check') { confirms.length = 0; const el = $(a); el.checked = !!b; await el.fire('change'); await flush(); out.push({ op, checked: el.checked, confirm: confirms[0] || '' }); }
    else if (op === 'pick') out.push(Object.assign({ op }, await pickValue(`${a}:${b}`)));
    else if (op === 'baseline') { await $('#btn-dr-baseline').fire('click'); await flush(); out.push({ op }); }
    else if (op === 'undo') { await $('#btn-dr-base-undo').fire('click'); await flush(); out.push({ op }); }
    else if (op === 'start') out.push(Object.assign({ op }, await start()));
    else if (op === 'poll') { await poll(); out.push({ op, gain: $('#dr-throttle-gain').value, text: $('#dr-throttle-gain-v').textContent }); }
    else if (op === 'fail_next') { failNext = +a || 1; out.push({ op }); }
    else if (op === 'tap') { await $(a).fire('click'); await flush(); out.push({ op, value: $('#dr-throttle-gain').value }); }
    else if (op === 'timers') { const t = timers.splice(0); const rest = []; for (const x of t) { if (x[1] <= 200) x[0](); else rest.push(x); } timers.push(...rest); await flush(); out.push({ op, value: $('#dr-throttle-gain').value }); }
    else if (op === 'resolve') { const d = deferred.splice(0); for (const f of d) f(); await flush(); out.push({ op, n: d.length, value: $('#dr-throttle-gain').value }); }
    else if (op === 'resolve1') { const f = deferred.shift(); if (f) f(); await flush(); out.push({ op, n: f ? 1 : 0, value: $('#dr-throttle-gain').value }); }
    else if (op === 'gstart') { deferGet++; T.pollDrive(); await flush(); out.push({ op }); }
    else if (op === 'gend') { const f = deferredGets.shift(); if (f) f(); await flush(); out.push({ op, n: f ? 1 : 0, value: $('#dr-throttle-gain').value }); }
    else if (op === 'pending_n') out.push({ op, value: { posts: deferred.length, gets: deferredGets.length } });
    else if (op === 'running') { running = !!a; await poll(); out.push({ op, running }); }
    else if (op === 'checkn') { confirms.length = 0; const el = $(a); el.checked = !!b; el.fire('change'); await flush(); out.push({ op, checked: el.checked }); }
    else if (op === 'defer') { deferPosts = !!a; out.push({ op }); }
    else if (op === 'realpoll') { await T.pollDrive(); await flush(); out.push({ op }); }
    else if (op === 'eval') { out.push({ op, value: vm.runInContext(a, ctx) }); }
    else if (op === 'tapday') { confirms.length = 0; const tgt = { closest: () => ({ dataset: { benchDay: String(a), benchI: String(b) }, disabled: false }) };
      await $('#bench-days').fire('click', { target: tgt }); await flush(); out.push({ op, confirm: confirms.slice() }); }
    else if (op === 'server_set') { Object.assign(server, a); out.push({ op }); }
    else if (op === 'api') { const r = await fetch('/api/drive/params', { method: 'POST', body: JSON.stringify(a) }); let j = null; try { j = await r.json(); } catch (e) { /* なし */ } out.push({ op, status: r.status, body: j }); }
    else if (op === 'reboot_on_confirm') { rebootOnConfirm = [a, b]; out.push({ op }); }
    else if (op === 'down') { down = !!a; out.push({ op, down }); }
    else if (op === 'models_fail') { modelsFail = !!a; out.push({ op, modelsFail }); }
    else if (op === 'down_all') { downAll = !!a; out.push({ op, downAll }); }
    else if (op === 'hold') { holdOn[a] = !!b; out.push({ op, path: a, on: !!b }); }
    else if (op === 'hold_body') { holdBodyOn[a] = !!b; out.push({ op, path: a, on: !!b }); }
    else if (op === 'release') { const hs = held.filter((h) => h.path === a); for (const h of hs) held.splice(held.indexOf(h), 1); for (const h of hs) h.go(); await flush(); out.push({ op, path: a, n: hs.length }); }
    else if (op === 'tab') { vm.runInContext(`showTab(${JSON.stringify(a)})`, ctx); await flush(); out.push({ op, tab: a }); }
    else if (op === 'tick') { const t = timers.splice(0); for (const x of t) x[0](); await flush(); out.push({ op, fired: t.map((x) => x[1]) }); }   // 鳴らすだけ(返事を待たない = ブラウザと同じ)
    else if (op === 'ntimers') { out.push({ op, value: timers.map((x) => x[1]) }); }
    else if (op === 'calls') { out.push({ op, value: calls.slice(cmark).map((c) => [c[0], c[1], c[3] === undefined ? null : c[3]]) }); cmark = calls.length; }
    else if (op === 'start_fail') { startFail = +a || 1; out.push({ op }); }
    else if (op === 'reboot') { bootAt = a; running = false; Object.assign(server, b || {}); out.push({ op }); }
    else if (op === 'confirm') { confirmAnswer = !!a; out.push({ op }); }
    else if (op === 'set') { $(a).value = String(b); out.push({ op }); }          // 欄に手で書いた(説明欄など。change は出さない)
    else if (op === 'read') out.push({ op, what: a, value: read(a) });
    else if (op === 'el') out.push({ op, sel: a, prop: b, value: readEl(a, b) });
    else if (op === 'posts') { out.push({ op, value: calls.slice(mark).filter((c) => c[0] === 'POST').map((c) => [c[1], c[2], c[3]]) }); mark = calls.length; }
    else throw new Error('知らない操作: ' + op);
  }
  console.log(JSON.stringify({ steps: out, toasts }));
})().catch((e) => { console.error('ERR', e && e.stack || e); process.exit(1); });
