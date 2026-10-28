// 2026-10-28 fix8 画面の係 直す 8-1 回目(反証 stale-status-after-drive-tab-return。tests/test_ui_fix8_r1.py が使う)。反証 8-1 回目の台本
// experiments/20261026_fix/fix8/review_ui_r1/sim2.js(sha256 6c0f8d2792ae…)をそのまま写した(もとは変えない。この 2 行だけ足した。2 つの画面・タブの行き来)。
// 2026-10-27 fix7 画面の係 直す 7-2 回目(反証 w2-ui-w2-ui-r72-ui-1・-2。tests/test_ui_fix7_r2.py が使う)。反証 7-2 回目の台本
// experiments/20261026_fix/fix7/review_ui_r2/sim_r2.js(sha256 1feef56edf33…)をそのまま写した(もとは変えない。この 2 行だけ足した)。
// 反証 7-2 回目・ui の台本(tests/fixtures/app_sim_fix7.js の写し。もとは変えない)。足したもの:
//  - ['post_lost', N]: 次の N 本の POST /api/drive/params は、サーバーが受けて値を変えたあと、返事が届かない(TypeError)
//  - ['start_lost']: 次の POST /api/drive/start は、サーバーが受けて走行を始めたあと、返事が届かない(TypeError)
//  - ['status_fail_next', N]: 次の N 本の GET /api/drive/status を通信の途切れ(TypeError)にする
//  - ['press_if_off', sel]: 人がスイッチを見て、OFF で押せるときだけ押す(ON・押せないときは何もしない)
//  - ['down_on_confirm']: 次の confirm() の間にサーバーが落ちる(down_all = true。起き上がるのは ['reboot'] と ['down_all', false])
// 2026-10-27 fix7 画面の係 直す 7-1 回目(反証 w1-ui-r7-ui-1・w1-ui-r7-ui-2。tests/test_ui_fix7.py が使う)。tests/fixtures/app_sim_fix6_r2.js の
// 写し(もとは変えない)。変えたもの:
//  - 直し 3(w1-ui-r7-ui-2): タイマーを鳴らす 3 つ('click'・'slide' と初めの読み込みのあとの短いタイマー、['timers']、['tick'])を、
//    本物のブラウザに近づけた: (a) 鳴らさない長いタイマー(表示の問い合わせの輪 pollDrive 1000/250 ms・打ち切り 5000 ms・トースト・上の
//    状態の輪 pollStatus 1000 ms・ライブの値の輪 pollValues 300 ms)は一覧に残したまま(捨てない・並びと番号も変えない)。前の
//    runShortTimers は全部を取り出して 200 ms 以下だけ鳴らし、残りを捨てたので、'click'・'slide' のあとは輪が回らなかった。
//    (b) 鳴らすタイマーは鳴らす直前に一覧から外し、それまでに clearTimeout で消されたものは鳴らさない(前は先に全部取り出したので、
//    先に鳴ったタイマーが消したタイマーも鳴り、消した長いタイマーが一覧に戻ることがあった)。['tick'] の fired は実際に鳴らしたものだけ。
//    初めの読み込みのあとも pollStatus・pollValues が残る(偽のサーバーは /api/status・/api/live/values に 404 を返し、app.js は黙って
//    次を掛ける)ので、['ntimers'] にはその 1000・300 も出る。['dtimers'] は pollStatus・pollValues を除いた長さ(走行タブの輪・打ち切り・
//    トースト・送信)。['timer_names'] は [関数の名前, 長さ](名前の無い関数は '')
//  - boot_refusal_texts の鍵に「画面の boot_at->今の boot_at」(例 "2000->3000")も使える(2 回起き直す場面。無ければ前と同じ鍵)
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

// ===================== ここから下は 反証 8-1 回目・係 ui が足した「2 つの画面」(上はもとの写しのまま) =====================
// 1 つの偽のサーバー(値・起動の時刻・落ちる)を、別々の画面(A・B: 別の document・タイマー・トースト・sessionStorage)が共有する。
// 操作: ['@', 画面, 操作](画面の名前の操作。無ければ A)・['open', 画面](新しい画面を開く = 読み直しは ['reload', 画面])・
//       ['tickall'](全部の画面のタイマーを鳴らす = 時間が進んだ)。ほかの操作はもとの写しと同じ意味(その画面で)
const server = JSON.parse(JSON.stringify(inp.server_params));
const KNOWN = new Set(Object.keys(server));
let running = false; let lastStart = null;
let down = false; let rebootOnConfirm = null; let startFail = 0;
let postLost = 0, startLost = 0, downOnConfirm = false, statusFailNext = 0;
let downAll = false; let modelsFail = false; const holdOn = {}, holdBodyOn = {}; const held = [];
let bootAt = inp.boot_at === undefined ? null : inp.boot_at;
let failNext = 0;
const posted = [];                                  // 届いた POST の全部 [画面, 道, body, status]
const MODEL = { name: BRUN, drivable: true, recommended: true, policy_title: 'JetFormer v0', outputs: ['actions'], postprocess: null,
  modes: [{ id: 0, name: '未分類' }, { id: 5, name: 'clean' }], trained_modes: [5], dataset: 'x', stride: 1, best_val_per_mode_us: {} };
function status() {
  const st = { running, phase: running ? 'driving' : 'idle', params: JSON.parse(JSON.stringify(server)), last: { run: BRUN, mode_id: 5 } };
  if (KNOWN.has('speed_ceiling')) st.speed_ctrl_spec = { ceiling: { all_mps: server.ceil_all_mps, left_mps: server.ceil_left_mps }, auto: { target_mps: server.auto_target_mps } };
  if (KNOWN.has('launch_boost')) st.launch_spec = { us: server.launch_us, mps: server.launch_mps, max_s: server.launch_max_s };
  if (KNOWN.has('zone_caps')) st.zone_caps_spec = { version: 'v1.1', caps_mps: { R1: 2.5, GM: 2.2, DE: 3.0, unknown: 3.0 }, unknown_note: '' };
  if (running) Object.assign(st, { run: BRUN, mode_id: 5, now: null, stats: {}, launch: {} });
  if (bootAt !== null) st.boot_at = bootAt;
  return st;
}
function applyParams(body, isRunning) {             // もとの写しと同じ決まり
  const p = Object.assign({}, server);
  for (const [k, v] of Object.entries(body || {})) if (KNOWN.has(k)) p[k] = v;
  const prev = server, on = (x) => +x === 1;
  if (on(prev.speed_ceiling) && !on(p.speed_ceiling)) p.speed_auto = 0;
  if (isRunning) {
    for (const k of ['zone_caps', 'speed_ceiling', 'speed_auto', 'launch_boost']) if (KNOWN.has(k) && on(p[k]) && !on(prev[k])) return `走行中は ${k} を ON にできません`;
  }
  if (KNOWN.has('speed_auto') && on(p.speed_auto) && !on(p.speed_ceiling)) return '自動合わせは天井が ON のときだけ';
  if (KNOWN.has('rec_hold_keep') && on(prev.rec_back_guard) && !on(p.rec_back_guard)) p.rec_hold_keep = 0;
  if (isRunning && KNOWN.has('start_kick') && on(p.start_kick) && !on(prev.start_kick)) return '走行中は発進の補助(start_kick)を ON にできません';
  if (isRunning) for (const k of ['rec_back_guard', 'rec_hold_keep', 'launch_no_kick']) if (KNOWN.has(k) && on(p[k]) && !on(prev[k])) return `走行中は「${k}」を ON にできません`;
  if (KNOWN.has('rec_hold_keep') && on(p.rec_hold_keep) && !on(p.rec_back_guard)) return '中立止めのあとも残すは、後ろの見張りが ON のときだけ';
  Object.assign(server, p);
  return null;
}
const abortErr = () => new DOMException('The operation was aborted.', 'AbortError');
const path0 = (u) => u.split('?')[0];
function holdUntilRelease(path, opt, make) {
  return new Promise((resolve, reject) => {
    const h = { path, go: () => resolve(make()) };
    held.push(h);
    const sig = opt && opt.signal;
    if (sig) sig.addEventListener('abort', () => { const k = held.indexOf(h); if (k >= 0) held.splice(k, 1); reject(abortErr()); });
  });
}
async function serve(url, opt, pg) {
  const method = (opt && opt.method) || 'GET';
  if (downAll) throw new TypeError('Failed to fetch (sim8: サーバーが落ちている。GET も全部)');
  if (modelsFail && path0(url) === '/api/drive/models') throw new TypeError('Failed to fetch (sim8: モデルの一覧だけ通らない)');
  if (method === 'GET' && holdOn[path0(url)]) { const o = Object.assign({}, opt || {}); delete o.signal; return holdUntilRelease(path0(url), opt, () => serve2(url, o, pg)); }
  return serve2(url, opt, pg);
}
async function serve2(url, opt, pg) {
  const method = (opt && opt.method) || 'GET', body = opt && opt.body ? JSON.parse(opt.body) : undefined;
  if (down && (path0(url) === '/api/drive/status' || method === 'POST')) throw new TypeError('Failed to fetch (sim8: サーバーが落ちている)');
  const ok = (d) => ({ ok: true, status: 200, statusText: 'OK', json: async () => d });
  const bad = (code, msg) => ({ ok: false, status: code, statusText: 'ERR', json: async () => ({ detail: msg }) });
  const path = path0(url);
  const log = (r) => { if (method === 'POST') posted.push([pg ? pg.name : 'curl', path, body, r.status]); return r; };
  if (method === 'GET' && path === '/api/drive/status' && statusFailNext > 0) { statusFailNext--; throw new TypeError('Failed to fetch (sim8: 表示の問い合わせだけ途切れ)'); }
  if (method === 'GET' && path === '/api/drive/status') return ok(status());
  if (method === 'GET' && path === '/api/drive/models') return ok(inp.models ? inp.models.map((n) => Object.assign({}, MODEL, { name: n, recommended: n === BRUN })) : [MODEL]);
  if (method === 'GET' && path === '/api/bench/candidates') return ok(inp.bench);
  if (method === 'POST' && path === '/api/drive/params' && failNext > 0) { failNext--; throw new TypeError('Failed to fetch (sim8: 通信の途切れ)'); }
  if (method === 'POST' && path === '/api/drive/params') { const e = applyParams(body, running); return log(e ? bad(400, e) : ok(JSON.parse(JSON.stringify(server)))); }
  if (method === 'POST' && path === '/api/drive/stop') { running = false; return log(ok({ ok: true })); }
  if (method === 'POST' && path === '/api/drive/start' && startFail > 0) { startFail--; throw new TypeError('Failed to fetch (sim8: 走行開始が届かない)'); }
  if (method === 'POST' && path === '/api/drive/start') {
    if ((+body.client || 0) < (inp.server_client_version || 9)) return log(bad(409, '管理画面が古いままです'));
    if (inp.server_build && body.build && +body.build < +inp.server_build) return log(bad(409, `管理画面が古いままです(この画面の app.js は ${body.build}、今のファイルは ${inp.server_build})`));
    if (inp.server_boot_check && bootAt !== null) {
      const seen = body.boot_at === undefined || body.boot_at === null ? null : +body.boot_at;
      const real = inp.boot_refusal_texts && (inp.boot_refusal_texts[String(seen) + '->' + String(bootAt)] || inp.boot_refusal_texts[String(seen)]);
      if (seen === null || !Number.isFinite(seen) || Math.abs(seen - bootAt) > 5e-4) return log(bad(409, real || `管理画面のサーバーが起動し直しました(sim8: 画面 ${seen}・今 ${bootAt})`));
    }
    if (running) return log(bad(409, 'すでに走行中です'));
    const e = applyParams(body.params, false); if (e) return log(bad(400, e));
    running = true; lastStart = Object.assign({ page: pg ? pg.name : '?' }, body, { server_at_start: JSON.parse(JSON.stringify(server)) });
    if (startLost > 0) { startLost--; throw new TypeError('Failed to fetch (sim8: 走行開始の返事が届かない。サーバーは始めた)'); }
    return log(ok({ token: 't-' + (pg ? pg.name : '?'), ai_no: 1, plan_diffs: [] }));
  }
  return bad(404, 'sim: ' + path);
}
class Event { constructor(type) { this.type = type; } }
const pages = {};
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); };
const store = () => { const m = {}; return { getItem: (k) => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = String(v); }, removeItem: (k) => { delete m[k]; } }; };
function makePage(name, keep) {
  const els = {};
  const document = { querySelector: (s) => (els[s] = els[s] || new El(s)), querySelectorAll: () => [], getElementById: (id) => document.querySelector('#' + id),
    createElement: (t) => new El(t), addEventListener() {}, removeEventListener() {}, body: new El('body'), documentElement: new El('html'), activeElement: null,
    hidden: false, visibilityState: 'visible' };
  const pg = { name, els, document, timers: [], confirms: [], toasts: [], calls: [], timerSeq: 0, confirmAnswer: true, cmark: 0, mark: 0,
               local: keep ? keep.local : store(), session: keep ? keep.session : store() };
  const toastEl = document.querySelector('#toast');
  Object.defineProperty(toastEl, 'textContent', { get() { return this._tc || ''; }, set(v) { this._tc = String(v); if (v) pg.toasts.push(String(v)); }, configurable: true });
  async function fetch(url, opt) {
    const method = (opt && opt.method) || 'GET', body = opt && opt.body ? JSON.parse(opt.body) : undefined;
    const i = pg.calls.length; pg.calls.push([method, url, body]);
    const res = await serve(url, opt, pg);
    pg.calls[i].push(res.status);
    return res;
  }
  const ctx = { document, window: { addEventListener() {}, open() {}, location: { href: '', host: 'x', protocol: 'http:' }, matchMedia: () => ({ matches: false, addEventListener() {} }) },
    localStorage: pg.local, sessionStorage: pg.session, fetch, console, navigator: { userAgent: 'sim' }, location: { href: '', host: 'x', protocol: 'http:' },
    setTimeout: (fn, ms) => { const id = ++pg.timerSeq; pg.timers.push([fn, ms, id]); return id; },
    clearTimeout: (id) => { const i = pg.timers.findIndex((x) => x[2] === id); if (i >= 0) pg.timers.splice(i, 1); }, setInterval: () => 0, clearInterval() {},
    requestAnimationFrame: () => 0,
    confirm: (t) => { pg.confirms.push(String(t)); if (downOnConfirm) { downOnConfirm = false; downAll = true; }
      if (rebootOnConfirm) { const [ba, rp] = rebootOnConfirm; rebootOnConfirm = null; down = false; bootAt = ba; running = false; Object.assign(server, rp || {}); }
      return pg.confirmAnswer; }, alert() {}, performance: { now: () => Date.now() },
    EventSource: class { constructor() {} close() {} addEventListener() {} }, WebSocket: class { constructor() {} close() {} send() {} },
    Audio: class { play() { return Promise.resolve(); } pause() {} }, Image: class {}, URL, URLSearchParams, Blob: class {}, FormData: class { append() {} },
    Event, AbortController, TextDecoder, TextEncoder, Promise, Date, Math, JSON, Object, Array, Number, String, Set, Map, RegExp, Error, Symbol, Intl, isFinite, parseFloat, parseInt };
  ctx.window.document = document; ctx.globalThis = ctx;
  if (inp.script_src) document.currentScript = { src: (inp.page_src && inp.page_src[name]) || inp.script_src };
  vm.createContext(ctx);
  const src = fs.readFileSync((inp.page_app && inp.page_app[name]) || APP, 'utf8')
    + '\n;globalThis.__T = { state, renderDrive, setDriveParams, loadDrive, readDriveParams, renderPrepHints, benchApply, pollDrive };';
  vm.runInContext(src, ctx, { filename: 'app.js' });
  pg.ctx = ctx; pg.T = ctx.__T; pg.$ = (s) => document.querySelector(s);
  return pg;
}
const fireTimers = async (pg, pick, wait) => { const fired = []; for (const x of pg.timers.filter(pick)) { const i = pg.timers.indexOf(x); if (i < 0) continue; pg.timers.splice(i, 1); fired.push(x); const r = x[0](); if (wait) await r; } return fired; };
const runShortTimers = async (pg) => { await fireTimers(pg, (x) => x[1] <= 200, true); await flush(); };

async function startOn(pg) {
  const n0 = pg.calls.length; pg.confirms.length = 0;
  await pg.$('#btn-dr-start').fire('click'); await flush();
  const st = pg.calls.slice(n0).find((c) => c[1] === '/api/drive/start');
  return { sent: st ? st[2].params : null, confirm: pg.confirms[0] || '', confirms: pg.confirms.slice(), desc: st ? st[2].description : null,
           boot_at: st ? (st[2].boot_at === undefined ? null : st[2].boot_at) : null, status: st ? st[3] : null,
           server_at_start: lastStart ? lastStart.server_at_start : null };
}
function readEl(pg, sel, prop) {
  const el = pg.$(sel);
  if (prop === 'hidden') return el.classList.contains('hidden');
  if (prop === 'text') return el.textContent;
  if (prop === 'html') return el.innerHTML;
  return el[prop];
}
async function doStep(pg, s, out) {
  const [op, a, b] = s;
  const $ = pg.$, T = pg.T;
  if (op === 'slide') { const el = $(a); await el.fire('pointerdown'); el.value = String(b); await el.fire('input'); await el.fire('pointerup'); await el.fire('change'); await runShortTimers(pg); out.push({ page: pg.name, op, value: el.value }); }
  else if (op === 'click') { pg.confirms.length = 0; await $(a).fire('click'); await flush(); await runShortTimers(pg); out.push({ page: pg.name, op, value: $('#dr-throttle-gain').value }); }
  else if (op === 'check') { pg.confirms.length = 0; const el = $(a); el.checked = !!b; await el.fire('change'); await flush(); out.push({ page: pg.name, op, sel: a, checked: el.checked, confirm: pg.confirms[0] || '' }); }
  else if (op === 'pick') { const sel = $('#dr-bench-pick'); const v = `${a}:${b}`; if (sel.value === v) out.push({ page: pg.name, op, fired: false }); else { sel.value = v; await sel.fire('change'); await flush(); out.push({ page: pg.name, op, fired: true }); } }
  else if (op === 'baseline') { await $('#btn-dr-baseline').fire('click'); await flush(); out.push({ page: pg.name, op }); }
  else if (op === 'start') out.push(Object.assign({ page: pg.name, op }, await startOn(pg)));
  else if (op === 'stop') { await $('#btn-dr-stop').fire('click'); await flush(); out.push({ page: pg.name, op }); }
  else if (op === 'tab') { vm.runInContext(`showTab(${JSON.stringify(a)})`, pg.ctx); await flush(); out.push({ page: pg.name, op, tab: a }); }
  else if (op === 'tick') { const t = await fireTimers(pg, () => true, false); await flush(); out.push({ page: pg.name, op, fired: t.map((x) => x[1]) }); }
  else if (op === 'eval') out.push({ page: pg.name, op, value: vm.runInContext(a, pg.ctx) });
  else if (op === 'el') out.push({ page: pg.name, op, sel: a, prop: b, value: readEl(pg, a, b) });
  else if (op === 'toasts') out.push({ page: pg.name, op, value: pg.toasts.slice() });
  else if (op === 'calls') { out.push({ page: pg.name, op, value: pg.calls.slice(pg.cmark).map((c) => [c[0], c[1], c[3] === undefined ? null : c[3]]) }); pg.cmark = pg.calls.length; }
  else if (op === 'confirm') { pg.confirmAnswer = !!a; out.push({ page: pg.name, op }); }
  else if (op === 'set') { $(a).value = String(b); out.push({ page: pg.name, op }); }
  else if (op === 'params') out.push({ page: pg.name, op, value: T.readDriveParams() });
  else if (op === 'st_params') out.push({ page: pg.name, op, value: JSON.parse(JSON.stringify((T.state.drive.st || {}).params || {})) });
  else if (op === 'pending') out.push({ page: pg.name, op, value: Object.keys(T.state.drive.pending).sort() });
  else throw new Error('知らない操作: ' + op);
}
(async () => {
  for (const n of inp.pages || ['A']) pages[n] = makePage(n);
  await flush();
  const out = [];
  for (const s0 of inp.steps) {
    let pg = pages.A, s = s0;
    if (s0[0] === '@') { pg = pages[s0[1]]; s = s0[2]; }
    const op = s[0];
    if (op === 'open') { pages[s[1]] = makePage(s[1]); await flush(); out.push({ op, page: s[1] }); continue; }
    if (op === 'reload') { const old = pages[s[1]]; old.timers.length = 0; pages[s[1]] = makePage(s[1], old); await flush(); out.push({ op, page: s[1] }); continue; }
    if (op === 'tickall') { const fired = {}; for (const n of Object.keys(pages)) fired[n] = (await fireTimers(pages[n], () => true, false)).map((x) => x[1]); await flush(); out.push({ op, fired }); continue; }
    if (op === 'server') { out.push({ op, value: JSON.parse(JSON.stringify(server)), running, bootAt }); continue; }
    if (op === 'api') { const e = applyParams(s[1], running); posted.push(['curl', '/api/drive/params', s[1], e ? 400 : 200]); out.push({ op, err: e, value: JSON.parse(JSON.stringify(server)) }); continue; }
    if (op === 'reboot') { bootAt = s[1]; running = false; Object.assign(server, s[2] || {}); out.push({ op }); continue; }
    if (op === 'reboot_on_confirm') { rebootOnConfirm = [s[1], s[2]]; out.push({ op }); continue; }
    if (op === 'down') { down = !!s[1]; out.push({ op, down }); continue; }
    if (op === 'down_all') { downAll = !!s[1]; out.push({ op, downAll }); continue; }
    if (op === 'down_on_confirm') { downOnConfirm = true; out.push({ op }); continue; }
    if (op === 'models_fail') { modelsFail = !!s[1]; out.push({ op }); continue; }
    if (op === 'status_fail_next') { statusFailNext = +s[1] || 1; out.push({ op }); continue; }
    if (op === 'start_fail') { startFail = +s[1] || 1; out.push({ op }); continue; }
    if (op === 'start_lost') { startLost = 1; out.push({ op }); continue; }
    if (op === 'fail_next') { failNext = +s[1] || 1; out.push({ op }); continue; }
    if (op === 'hold') { holdOn[s[1]] = !!s[2]; out.push({ op, path: s[1], on: !!s[2] }); continue; }
    if (op === 'release') { const hs = held.filter((h) => h.path === s[1]); for (const h of hs) held.splice(held.indexOf(h), 1); for (const h of hs) h.go(); await flush(); out.push({ op, path: s[1], n: hs.length }); continue; }
    if (op === 'posted') { out.push({ op, value: posted.splice(0) }); continue; }
    if (op === 'set_running') { running = !!s[1]; out.push({ op, running }); continue; }
    await doStep(pg, s, out);
  }
  const toasts = {}; for (const n of Object.keys(pages)) toasts[n] = pages[n].toasts.slice();
  console.log(JSON.stringify({ steps: out, toasts }));
})().catch((e) => { console.error('ERR', e && e.stack || e); process.exit(1); });
