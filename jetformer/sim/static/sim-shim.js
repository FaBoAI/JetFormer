// JetFormer のシミュレーター用の差し込みスクリプト(classic script。app.js などのモジュールより先に動く)。
// ユーザーのシミュレーターのファイルは変えずに、JetFormer の中で動かすための調整だけを行う。
//  1. ブリッジへの通信の書き換え: 「今の #endpoint(既定 http://127.0.0.1:8766)の origin」宛ての /api/... を、
//     同じオリジンの /sim/bridge/api/... にする(JetFormer がブリッジに中継する)。相対 URL は書き換えない。
//  2. ダウンロードなどのリンクの書き換え  3. 接続先の入力欄を読み取り専用に
//  4. localStorage: JetFormer への送り先を 127.0.0.1:<JetFormer のポート> に、20 秒シーンの記録をコースごとに分ける
//  5. 上の帯(JetFormer に戻る・コースの選択・ブリッジの状態)  6. コースに合わせた表示の調整
//  7. 自動接続  8. ?model=<run> で JetFormer の学習済みモデルを読み込む
// 利用者の文字列(コース名など)は textContent だけで書く。要素が見つからないときはその機能だけを飛ばす。
(() => {
  'use strict';
  const cfg = window.JETFORMER_SIM || {};
  const course = cfg.course || {};
  const BRIDGE = cfg.bridge || '/sim/bridge';
  const LOOP = new Set(['127.0.0.1', 'localhost', '[::1]']);
  const notes = [];
  const warn = (msg) => { notes.push(msg); try { console.warn('[JetFormer]', msg); } catch (e) { /* なし */ } };

  // ---------------------------------------------------------------- 0. 安全でない接続でも動くように
  // crypto.randomUUID は https か localhost(127.0.0.1)のページでしか使えない。Mac などから http://<Jetson の IP>:8080 で
  // 開くと、app.js が読み込みの最初(const pageSession=crypto.randomUUID())で止まり、画面が動かない。
  // どのページでも使える getRandomValues で、同じ形の UUID(v4)を作って補う。
  try {
    const c = window.crypto;
    if (c && typeof c.randomUUID !== 'function' && typeof c.getRandomValues === 'function') {
      Object.defineProperty(c, 'randomUUID', { configurable: true, writable: true, value: function randomUUID() {
        const b = c.getRandomValues(new Uint8Array(16));
        b[6] = (b[6] & 0x0f) | 0x40;                                         // 版 4
        b[8] = (b[8] & 0x3f) | 0x80;                                         // 変種 10xx
        const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
        return h.slice(0, 8) + '-' + h.slice(8, 12) + '-' + h.slice(12, 16) + '-' + h.slice(16, 20) + '-' + h.slice(20);
      } });
      try { console.info('[JetFormer] 安全でない接続(http)で開いたため、crypto.randomUUID を補いました'); } catch (e) { /* なし */ }
    }
  } catch (e) { /* 補えなければそのまま */ }

  // ---------------------------------------------------------------- 1. 通信の書き換え
  function endpointOrigin() {
    try { const el = document.getElementById('endpoint'); if (el && el.value) return new URL(el.value).origin; } catch (e) { /* 既定 */ }
    return 'http://127.0.0.1:8766';
  }
  function rewrite(raw) {
    if (typeof raw !== 'string' || !/^http:\/\//i.test(raw)) return null;       // 相対 URL は触らない
    let u; try { u = new URL(raw); } catch (e) { return null; }
    if (!LOOP.has(u.hostname) || u.origin !== endpointOrigin() || !u.pathname.startsWith('/api/')) return null;
    return location.origin + BRIDGE + u.pathname + u.search;
  }
  window.JETFORMER_SIM_REWRITE = rewrite;            // 確認用
  const origFetch = window.fetch ? window.fetch.bind(window) : null;
  if (origFetch) {
    window.fetch = function (input, init) {
      let raw = null;
      if (typeof input === 'string') raw = input;
      else if (input instanceof URL) raw = input.href;
      else if (typeof Request !== 'undefined' && input instanceof Request) raw = input.url;
      const to = rewrite(raw);
      if (to && typeof Request !== 'undefined' && input instanceof Request) return origFetch(new Request(to, input), init);
      return origFetch(to || input, init);
    };
  }
  const xhrOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    return xhrOpen.call(this, method, rewrite(String(url)) || url, ...rest);
  };

  // ---------------------------------------------------------------- 4. localStorage
  const SCOPED = new Set(['course-20s-scene-ids', 'course-20s-collection']);
  const suffix = ':' + (course.world_version || course.id || 'default');
  try {
    const P = Storage.prototype, get = P.getItem, set = P.setItem, del = P.removeItem;
    P.getItem = function (k) { return get.call(this, SCOPED.has(k) ? k + suffix : k); };
    P.setItem = function (k, v) { return set.call(this, SCOPED.has(k) ? k + suffix : k, v); };
    P.removeItem = function (k) { return del.call(this, SCOPED.has(k) ? k + suffix : k); };
    localStorage.setItem('course-jetformer-url', 'http://127.0.0.1:' + (cfg.jetformer_port || 8080));
  } catch (e) { /* 使えなくても続ける */ }

  // ---------------------------------------------------------------- 上の帯
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const bar = {};
  function buildBar() {
    const root = el('div', 'jf-simbar'); root.id = 'jf-simbar';
    const home = el('a', 'jf-home', '← JetFormer に戻る'); home.href = cfg.home || '/';
    root.appendChild(home);
    const lab = el('label', 'jf-course'); lab.appendChild(el('span', null, 'コース'));
    bar.select = el('select'); bar.select.setAttribute('aria-label', 'コースを選ぶ');
    const cur = el('option', null, course.name || course.id || ''); cur.value = course.id || ''; bar.select.appendChild(cur);
    lab.appendChild(bar.select); root.appendChild(lab);
    bar.bridge = el('span', 'jf-pill', 'ブリッジ: 確認中'); root.appendChild(bar.bridge);
    bar.start = el('button', 'jf-btn', 'ブリッジを起動'); bar.start.type = 'button'; bar.start.hidden = true; root.appendChild(bar.start);
    bar.model = el('span', 'jf-model', ''); root.appendChild(bar.model);
    bar.note = el('span', 'jf-note', ''); root.appendChild(bar.note);
    if (course.renderer === 'generic') {
      root.appendChild(el('span', 'jf-warn', '簡易描画のコース(カメラ画像は Default と同じにはなりません)'));
    }
    document.body.insertBefore(root, document.body.firstChild);
    bar.select.addEventListener('change', () => {
      const id = bar.select.value; if (!id || id === course.id) return;
      const q = new URLSearchParams(location.search); const keep = q.get('model');
      location.href = '/sim/c/' + encodeURIComponent(id) + '/' + (keep ? '?model=' + encodeURIComponent(keep) : '');
    });
    bar.start.addEventListener('click', async () => {
      bar.start.disabled = true;
      try {
        const r = await origFetch('/api/sim/bridge/start', { method: 'POST' });
        if (!r.ok) { let m = r.status + ''; try { m = (await r.json()).detail || m; } catch (e) { /* なし */ } setNote('起動できません: ' + m, true); }
      } catch (e) { setNote('起動できません: ' + e.message, true); }
      bar.start.disabled = false; pollStatus();
    });
  }
  function setNote(text, isError) { if (!bar.note) return; bar.note.textContent = text || ''; bar.note.classList.toggle('jf-error', !!isError); }
  function setModel(text, isError) { if (!bar.model) return; bar.model.textContent = text || ''; bar.model.classList.toggle('jf-error', !!isError); }

  async function loadCourses() {
    try {
      const r = await origFetch(cfg.courses_api || '/api/sim/courses'); if (!r.ok) return;
      const data = await r.json();
      bar.select.textContent = '';
      for (const c of data.courses || []) {
        const o = el('option', null, (c.name || c.id) + (c.renderer === 'generic' ? '(簡易描画)' : '') + (c.is_default ? ' ★' : ''));
        o.value = c.id; if (c.id === course.id) o.selected = true; bar.select.appendChild(o);
      }
    } catch (e) { /* 一覧が取れなくても今のコースのまま */ }
  }

  let bridgeState = null, restartedAt = null;
  const listeners = [];
  async function pollStatus() {
    try {
      const r = await origFetch(cfg.status_api || '/api/sim/status'); if (!r.ok) throw new Error(r.status);
      const s = await r.json(); const b = s.bridge || {};
      bridgeState = b.state;
      const label = { stopped: '停止中', starting: '起動中…', running: '動作中', stopping: '停止中…', failed: '失敗' }[b.state] || b.state;
      bar.bridge.textContent = 'ブリッジ: ' + label + (b.state === 'running' && b.policy_loaded ? '(モデル読み込み済み)' : '');
      bar.bridge.className = 'jf-pill jf-' + (b.state || 'unknown');
      bar.start.hidden = !(b.state === 'stopped' || b.state === 'failed');
      if (b.state === 'failed' && b.last_error) setNote(b.last_error, true);
      if (restartedAt && b.restarted_at && b.restarted_at !== restartedAt) setNote('ブリッジが再起動しました。モデルを読み込み直してください', true);
      restartedAt = b.restarted_at || restartedAt;
      for (const f of listeners.splice(0)) f(b);
    } catch (e) { if (bar.bridge) { bar.bridge.textContent = 'ブリッジ: 状態を取得できません'; bar.bridge.className = 'jf-pill jf-unknown'; } }
  }
  const waitBridge = (timeoutMs) => new Promise((resolve) => {
    const t0 = Date.now();
    const check = (b) => {
      if (b && b.state === 'running') return resolve(true);
      if (Date.now() - t0 > timeoutMs || (b && b.state === 'failed')) return resolve(false);
      setTimeout(() => { listeners.push(check); pollStatus(); }, 1000);
    };
    listeners.push(check); pollStatus();
  });
  const waitApp = (timeoutMs) => new Promise((resolve) => {
    const t0 = Date.now();
    (function loop() { if (window.courseSimulator) return resolve(true); if (Date.now() - t0 > timeoutMs) return resolve(false); setTimeout(loop, 200); })();
  });

  // ---------------------------------------------------------------- 2・3・6. ページの調整
  function fixLinks() {
    const fix = (a) => {
      const cur = a.getAttribute('href') || '';
      let to = rewrite(cur);
      if (!to && a.id === 'jetformer-open') {
        try { if (LOOP.has(new URL(a.href).hostname)) to = location.origin + '/'; } catch (e) { /* そのまま */ }
      }
      // 値が変わるときだけ書く。同じ値でも setAttribute すると MutationObserver がまた呼ばれ、127.0.0.1 で開いたページでは
      // #jetformer-open の書き換えが無限ループになってページが固まり、メモリが増え続けた(2026-10-13 の確認で発見)
      if (to && to !== cur) a.setAttribute('href', to);
    };
    for (const id of ['dataset-download', 'collection-download', 'jetformer-open']) {
      const a = $(id); if (!a) { warn('#' + id + ' が見つかりません(シミュレーターの版が変わった可能性)'); continue; }
      fix(a); new MutationObserver(() => fix(a)).observe(a, { attributes: true, attributeFilter: ['href'] });
    }
  }
  function adjustPage() {
    const ep = $('endpoint');
    if (ep) {
      ep.readOnly = true; ep.title = 'JetFormer 経由で Jetson のブリッジに接続します';
      const hint = el('small', 'jf-hint', 'JetFormer 経由で接続します(この値は変えません)');
      if (ep.parentNode) ep.parentNode.appendChild(hint);
    } else warn('#endpoint が見つかりません');
    const label = document.querySelector('.view-label'); if (label && course.name) label.textContent = course.name;
    const size = document.querySelector('.view-note span');
    if (size && Array.isArray(course.size_m) && course.size_m.length === 2) size.textContent = Number(course.size_m[0]).toFixed(2) + ' × ' + Number(course.size_m[1]).toFixed(2) + ' m';
    if (course.name) document.title = document.title + ' — ' + course.name;
    const feats = course.features || {};
    const teachers = new Set(feats.teachers || []);
    const sel = $('teacher-profile');
    if (sel && teachers.size) {
      const before = sel.value;
      for (const o of [...sel.options]) if (!teachers.has(o.value)) o.remove();
      if (!teachers.has(before)) sel.value = 'slow';
    } else if (!sel) warn('#teacher-profile が見つかりません');
    if (feats.collection_plan === false) {
      const b = $('collection-plan');
      if (b) { b.hidden = true; b.insertAdjacentElement('afterend', el('p', 'jf-hint', '「20分・60シーンの収集条件」は Default コースだけで使えます')); }
    }
    if (course.has_shortcut === false) { const s = $('shortcut-status'); if (s) s.hidden = true; }
    fixLinks();
    if (notes.length) setNote(notes[0], true);
  }

  // ---------------------------------------------------------------- 7. 自動接続  8. モデルの読み込み
  async function autoConnect() {
    const ready = await waitApp(10000);
    if (!ready) { setNote('シミュレーターを起動できませんでした(WebGL が使えるブラウザで開いてください)', true); return false; }
    const running = await waitBridge(30000);
    if (!running) { setNote('ブリッジが動いていません。JetFormer の「シミュレーター」タブで起動してください', true); return false; }
    const c = $('connect'); if (c && !c.disabled) c.click();
    return true;
  }
  async function loadModel(run) {
    setModel('モデル ' + run + ': 準備を待っています');
    const ok = await autoConnect();
    if (!ok) { setModel('モデル ' + run + ': ブリッジにつながらないので読み込めません', true); return; }
    const input = $('model-bundle-file'), button = $('load-bundle');
    if (!input || !button) { setModel('モデルの読み込み欄が見つかりません(シミュレーターの版が変わった可能性)', true); return; }
    let blob;
    try {
      setModel('モデル ' + run + ': JetFormer から取得中');
      const r = await origFetch('/api/models/' + encodeURIComponent(run) + '/bundle.zip');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const total = Number(r.headers.get('content-length')) || 0;
      if (r.body && r.body.getReader) {
        const reader = r.body.getReader(), parts = []; let got = 0;
        for (;;) {
          const { done, value } = await reader.read(); if (done) break;
          parts.push(value); got += value.length;
          setModel('モデル ' + run + ': 取得中 ' + (got / 1e6).toFixed(0) + (total ? ' / ' + (total / 1e6).toFixed(0) : '') + ' MB');
        }
        blob = new Blob(parts, { type: 'application/zip' });
      } else blob = await r.blob();
    } catch (e) { setModel('モデル ' + run + ' を取得できません: ' + e.message, true); return; }
    let file;
    try {
      file = new File([blob], run + '.zip', { type: 'application/zip' });
      const dt = new DataTransfer(); dt.items.add(file); input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    } catch (e) {
      const a = el('a', 'jf-link', 'ZIP をダウンロードして「モデルZIP」で選んでください');
      a.href = '/api/models/' + encodeURIComponent(run) + '/bundle.zip'; a.download = run + '.zip';
      setModel('このブラウザでは自動で読み込めません。'); bar.model.appendChild(a); return;
    }
    const status = $('upload-status');
    if (status) new MutationObserver(() => setModel('モデル ' + run + ': ' + status.textContent, /失敗|エラー/.test(status.textContent))).observe(status, { childList: true, characterData: true, subtree: true });
    const details = $('model-upload'); if (details) details.open = true;
    button.click();
    try { const q = new URLSearchParams(location.search); q.delete('model'); history.replaceState(null, '', location.pathname + (q.toString() ? '?' + q : '') + location.hash); } catch (e) { /* そのまま */ }
  }

  function start() {
    try { buildBar(); } catch (e) { warn('帯を作れません: ' + e.message); }
    try { adjustPage(); } catch (e) { warn('表示の調整に失敗しました: ' + e.message); }
    loadCourses(); pollStatus(); setInterval(pollStatus, 5000);
    const run = new URLSearchParams(location.search).get('model');
    if (run && /^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$/.test(run)) loadModel(run);
    else autoConnect();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
