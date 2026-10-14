// JetFormer: シミュレーターのページに「強化学習モード」の選択欄と、収集プランの実行係を足す(classic script。2026-10-14、sim_rl_modes の統合)。
//
//  - 「反復試験・教師データ収集」の「教師の走行」の下に「強化学習モード」の選択欄(登録簿 /api/sim/rl/modes のモード)。
//    選ぶと方策を JetFormer から読み(sha256 はサーバーが確かめ、X-Policy-SHA256 で受け取る)、コントローラー = 経路追従デモ・
//    教師 = RC加速教師(差し込み口)・物理とセンサーを記録の条件(RC_COLLECTION_PLAN[0])にそろえてリセットする。
//    走るのは _jf/rl/teacher.js(import map で差し替えた教師)。モードを選ばなければ今までどおりの RC加速教師。
//  - モードごとの小さな設定: 速さいろいろ = 速さの倍率、離脱復活 = 乱しの強さ。走行中の様子(すき間・シールド・乱し)も出す。
//  - ?rl=<key> で開くとそのモードを選んだ状態に。?rlplan=<id> で開くと「JetFormer 収集プラン」の枠が出て、シーンを 1 つずつ
//    アプリ自身の「連続試験」(1 条件)で記録し、進み具合を JetFormer に送り、モードが終わるたびにそのモードのシーンを 1 つのデータセットに取り込ませる。
// 利用者の文字列は textContent だけで書く。要素が見つからないときはその機能だけを飛ばす。
(() => {
  'use strict';
  const cfg = window.JETFORMER_SIM || {};
  const rl = cfg.rl || {};
  const RL = window.JETFORMER_RL = window.JETFORMER_RL || {mode: 'rc', weights: {}, modes: {}, sceneParams: null};
  RL.settings = RL.settings || {};
  RL.registry = RL.registry || null;
  if (!rl.enabled) return;
  const API = rl.modes_api || '/api/sim/rl/modes', PLANS = rl.plans_api || '/api/sim/rl/plans';
  const q = new URLSearchParams(location.search);
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const rawFetch = (...a) => window.fetch(...a);
  const ui = {};
  const PH = {drive: '走行', perturb: '乱し中', shield: 'シールド作動'};
  // 経路いろいろの線の形の呼び名(2026-10-15 昇格の段。c0 > 0 が左寄り)
  const LINE_NAME = {inside: 'イン', outside: 'アウト', late_apex: 'レイトエイペックス', early_apex: 'アーリーエイペックス', random: 'ランダム'};
  function lineName(l) {
    if (!l) return '';
    if (l.family === 'const') return (l.c0 > .5 ? '左寄り' : l.c0 < -.5 ? '右寄り' : '中央') + '(c0 ' + (Math.round(l.c0 * 100) / 100) + ')';
    return LINE_NAME[l.family] || l.family;
  }
  // シールドがあるモード(mode.json の runtime.shield。登録簿が無いときは no_wall だけ)
  function hasShield(key) { const m = RL.modes?.[key]; return m?.runtime ? !!m.runtime.shield : key === 'no_wall'; }
  const KEEP = new Set(['name', 'dataset-split', 'seed', 'start-gate', 'spawn-offset', 'spawn-yaw', 'scenario', 'execution-speed', 'dataset-mode']);

  async function getJSON(url, opt) {
    const r = await rawFetch(url, opt);
    let body = null; try { body = await r.json(); } catch (e) { /* なし */ }
    if (!r.ok) throw new Error((body && (body.detail || body.message)) || ('HTTP ' + r.status));
    return body;
  }
  const postJSON = (url, body) => getJSON(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});

  function status(text, isError) { if (!ui.status) return; ui.status.textContent = text || ''; ui.status.classList.toggle('jf-rl-error', !!isError); }
  function busy() {
    const stop = $('batch-stop'), fin = $('record-finish');
    return !!((stop && !stop.disabled) || (fin && !fin.disabled) || RL.planRunning);
  }

  // ---------------------------------------------------------------- 登録簿と方策
  async function loadRegistry() {
    const r = await getJSON(API);
    RL.registry = r;
    return r;
  }
  function modeInfo(key) { return (RL.registry?.modes || []).find((m) => m.key === key) || null; }

  async function loadWeights(key) {
    const m = modeInfo(key);
    if (!m) throw new Error('登録簿にないモードです: ' + key);
    if (!m.selectable) throw new Error(m.name + ' は選べません(' + (m.problems?.join('・') || m.label) + ')');
    if (!RL.weights[key]) {
      status(m.name + ' の方策を読み込んでいます…');
      const r = await rawFetch(API + '/' + encodeURIComponent(key) + '/policy.json', {cache: 'no-cache'});
      if (!r.ok) { let d = 'HTTP ' + r.status; try { d = (await r.json()).detail || d; } catch (e) { /* なし */ } throw new Error(d); }
      const sha = r.headers.get('X-Policy-SHA256');
      if (sha !== m.policy_sha256) throw new Error('方策の sha256 が登録簿と違います(読み込みの途中で差し替わった可能性)');
      const w = await r.json();
      if (!w || !Array.isArray(w.layers) || !Number.isInteger(w.obs_dim)) throw new Error('方策のファイルの形が違います');
      RL.weights[key] = w;
    }
    RL.modes[key] = {name: m.name, policy_sha256: m.policy_sha256, planned_time_s: m.planned_time_s ?? null, state: m.state, runtime: m.runtime ?? null};   // line r4: mode.json の runtime(シールドの切り替え)
    return m;
  }

  // ---------------------------------------------------------------- 選択欄
  function setField(id, v) {
    const e = $(id); if (!e) return false;
    if (e.type === 'checkbox') e.checked = !!v; else e.value = String(v);
    return true;
  }
  function applyBase() {
    const b = RL.basePlan;
    if (!b) throw new Error('記録の条件(RC_COLLECTION_PLAN)を読めません(シミュレーターの版が変わった可能性)');
    for (const [k, v] of Object.entries(b)) if (!KEEP.has(k)) setField(k, v);
    setField('controller', 'demo'); setField('teacher-profile', 'speed-rc'); setField('episode-mode', 'continuous-laps-v1'); setField('limit', 20);
    const sc = $('scenario');
    if (sc && (rl.blocked_scenarios || ['low-friction']).includes(sc.value)) {
      sc.value = 'baseline';
      status('「低摩擦」の場面は強化学習モードでは使えないので baseline にしました(学習・試験した物理の範囲の外)');
    }
  }
  function triggerReset() {
    const tp = $('teacher-profile');
    if (tp) tp.dispatchEvent(new Event('change'));
  }
  function relabelTeacher() {
    const tp = $('teacher-profile'); if (!tp) return;
    const o = [...tp.options].find((x) => x.value === 'speed-rc'); if (!o) return;
    if (!o.dataset.jfOrig) o.dataset.jfOrig = o.textContent;
    o.textContent = RL.mode && RL.mode !== 'rc' ? '強化学習: ' + (RL.modes[RL.mode]?.name || RL.mode) : o.dataset.jfOrig;
  }
  function updateBar() {
    if (!ui.bar) return;
    const m = RL.mode && RL.mode !== 'rc' ? RL.modes[RL.mode] : null;
    ui.bar.textContent = m ? 'RL: ' + m.name : '';
    ui.bar.hidden = !m;
  }

  async function select(key, {reset = true} = {}) {
    if (key !== (RL.mode || 'rc') && busy() && reset) {
      status('連続試験・収集の間はモードを変えられません', true);
      if (ui.sel) ui.sel.value = RL.mode || 'rc';
      return false;
    }
    try {
      if (!key || key === 'rc') {
        RL.mode = 'rc'; RL.sceneParams = null;
        status('強化学習モードを使いません(RC加速教師のまま)');
      } else {
        const m = await loadWeights(key);
        RL.mode = key;
        if (reset) applyBase();
        status(m.name + ' を読み込みました' + (m.state === 'accepted' ? '' : '(' + m.label + ')') + '。「▶ 評価を開始」で走ります');
      }
      if (ui.sel) ui.sel.value = RL.mode;
      relabelTeacher(); updateBar(); renderInfo(); renderOptions();
      if (reset) triggerReset();
      return true;
    } catch (e) {
      status('モードを選べません: ' + e.message, true);
      RL.mode = 'rc';
      if (ui.sel) ui.sel.value = 'rc';
      relabelTeacher(); updateBar(); renderInfo(); renderOptions();
      return false;
    }
  }

  function renderInfo() {
    if (!ui.info) return;
    ui.info.textContent = '';
    const m = RL.mode && RL.mode !== 'rc' ? modeInfo(RL.mode) : null;
    if (!m) { ui.info.appendChild(el('span', 'jf-rl-muted', '教師(RC加速教師)の走りのまま。モードを選ぶと、強化学習の方策が走ります(Default コースだけ)')); return; }
    const head = el('div', null);
    head.appendChild(el('span', 'jf-rl-badge jf-rl-' + m.state, m.label));
    head.appendChild(document.createTextNode(' ' + (m.short || '')));
    ui.info.appendChild(head);
    const c = m.card || {};
    for (const line of [c.headline_ja, ...(c.notes_ja || []).slice(0, 2)]) if (line) ui.info.appendChild(el('div', 'jf-rl-muted', line));
    // 2026-10-15 昇格の段: 手で走らせるときの注意(mode.json の ui_note_ja。止まったまま動かない開始など)
    if (m.ui_note) ui.info.appendChild(el('div', 'jf-rl-muted jf-rl-note', '注意: ' + m.ui_note));
  }

  function renderOptions() {
    if (!ui.opts) return;
    ui.opts.textContent = '';
    const key = RL.mode;
    if (key === 'speed') {
      const lab = el('label', null, '速さの倍率(最速の何 %)');
      const s = el('select'); s.id = 'jf-rl-cv';
      for (const [v, t] of [['', '自動(シーンの種から 5 段)'], ['1', '100%'], ['0.875', '87.5%'], ['0.75', '75%'], ['0.625', '62.5%'], ['0.5', '50%']]) {
        const o = el('option', null, t); o.value = v; s.appendChild(o);
      }
      s.value = RL.settings.speed?.c_v != null ? String(RL.settings.speed.c_v) : '';
      s.addEventListener('change', () => { RL.settings.speed = s.value ? {c_v: Number(s.value)} : {}; if (!busy()) triggerReset(); });
      lab.appendChild(s); ui.opts.appendChild(lab);
    } else if (key === 'recover') {
      const lab = el('label', null, '乱しの強さ');
      const s = el('select'); s.id = 'jf-rl-perturb';
      for (const [v, t] of [['', '自動(出来事ごとに 弱・中・強)'], ['弱', '弱'], ['中', '中'], ['強', '強']]) { const o = el('option', null, t); o.value = v; s.appendChild(o); }
      s.value = RL.settings.recover?.perturb_class || '';
      s.addEventListener('change', () => { RL.settings.recover = s.value ? {perturb_class: s.value} : {}; if (!busy()) triggerReset(); });
      lab.appendChild(s); ui.opts.appendChild(lab);
      ui.opts.appendChild(el('div', 'jf-rl-muted', '乱した操作は記録に印(info.jf.l = 0)が付き、JetFormer に取り込むときに学習から外します'));
    } else if (key === 'no_wall') {
      ui.opts.appendChild(el('div', 'jf-rl-muted', '走行中は安全シールドが方策の指令を見張ります(作動したフレームの数を下に出します)。止められません'));
    }
  }

  function renderLive() {
    if (!ui.live) return;
    const v = RL.live;
    if (!v || !RL.mode || RL.mode === 'rc' || v.mode !== RL.mode) { ui.live.textContent = ''; return; }
    const parts = ['フレーム ' + v.frame, PH[v.ph] || v.ph];
    if (v.cl_mm != null) parts.push('すき間 ' + (v.cl_mm / 10).toFixed(1) + ' cm(最小 ' + (v.min_cl_mm / 10).toFixed(1) + ' cm)');
    if (hasShield(RL.mode)) parts.push('シールド ' + v.shield_frames + ' フレーム');
    if (RL.mode === 'line' && v.params?.line) parts.push('線 ' + lineName(v.params.line));
    if (RL.mode === 'recover') parts.push('乱し ' + v.events + ' 回・学習から外す操作 ' + v.perturb_frames + ' フレーム');
    if (v.params?.c_v != null) parts.push('速さの倍率 ' + Math.round(v.params.c_v * 1000) / 10 + '%');
    if (v.params?.perturb_class) parts.push('強さ ' + v.params.perturb_class);
    if (v.seed != null) parts.push('種 ' + v.seed);
    ui.live.textContent = parts.join(' ・ ') + (v.error ? ' ・ エラー: ' + v.error : '');
    ui.live.classList.toggle('jf-rl-error', !!v.error);
  }

  function buildUI() {
    const tp = $('teacher-profile');
    const host = tp ? tp.closest('label') : null;
    if (!host) { console.warn('[JetFormer] #teacher-profile が見つからないので強化学習モードの欄を出しません'); return false; }
    const box = el('div', 'jf-rl'); box.id = 'jf-rl';
    const lab = el('label', null, '強化学習モード');
    ui.sel = el('select'); ui.sel.id = 'jf-rl-mode';
    const o = el('option', null, '使わない(RC加速教師)'); o.value = 'rc'; ui.sel.appendChild(o);
    lab.appendChild(ui.sel); box.appendChild(lab);
    ui.info = el('div', 'jf-rl-info'); box.appendChild(ui.info);
    ui.opts = el('div', 'jf-rl-opts'); box.appendChild(ui.opts);
    ui.live = el('div', 'jf-rl-live'); box.appendChild(ui.live);
    ui.status = el('div', 'jf-rl-status'); box.appendChild(ui.status);
    host.insertAdjacentElement('afterend', box);
    const bar = $('jf-simbar');
    if (bar) { ui.bar = el('span', 'jf-pill jf-rl-pill', ''); ui.bar.hidden = true; bar.appendChild(ui.bar); }
    if (!rl.course_ok) {
      ui.sel.disabled = true;
      status(rl.reason || '強化学習モードはこのコースでは使えません');
      return false;
    }
    ui.sel.addEventListener('change', () => select(ui.sel.value));
    return true;
  }

  function fillSelect() {
    const reg = RL.registry;
    for (const m of reg?.modes || []) {
      const o = el('option', null, m.name + (m.state === 'accepted' ? '' : '(' + m.label + ')'));
      o.value = m.key; o.disabled = !m.selectable;
      if (!m.selectable && m.problems?.length && m.state !== 'not_published') o.title = m.problems.join(' / ');
      ui.sel.appendChild(o);
    }
  }

  const waitApp = (ms) => new Promise((resolve) => {
    const t0 = Date.now();
    (function loop() { if (window.courseSimulator && RL.basePlan) return resolve(true); if (Date.now() - t0 > ms) return resolve(false); setTimeout(loop, 200); })();
  });

  // ---------------------------------------------------------------- 収集プランの実行係
  const plan = {id: null, data: null, stop: false, log: []};
  const sceneById = (p, id) => p.scenes.find((s) => s.id === id);
  const FINAL = new Set(['done', 'failed', 'skipped']);

  async function calibration() {
    try {
      const b = await getJSON('/sim/bridge/api/status');
      if (!b.policy_loaded) return {ok: true, policy_loaded: false, policy_name: null, steer_center_us: 1441, throttle_neutral_us: 1496, source: 'recorder-default'};
      const mc = b.policy_info?.model_config || {};
      const s = mc.steer_center_us, t = mc.throttle_neutral_us;
      const ok = Number.isFinite(s) && Number.isFinite(t) && Math.abs(s - 1442) <= 5 && Math.abs(t - 1498) <= 5;
      return {ok, policy_loaded: true, policy_name: b.name || b.policy_name || null, steer_center_us: s ?? null, throttle_neutral_us: t ?? null, source: 'bridge-model'};
    } catch (e) {
      return {ok: false, error: 'ブリッジの状態を読めません: ' + e.message};
    }
  }
  const calKey = (c) => JSON.stringify([c.policy_loaded, c.policy_name, c.steer_center_us, c.throttle_neutral_us]);

  function planLog(text, isError) {
    plan.log.unshift((new Date()).toLocaleTimeString() + ' ' + text);
    plan.log = plan.log.slice(0, 40);
    if (ui.planLog) ui.planLog.textContent = plan.log.join('\n');
    if (ui.planStatus) { ui.planStatus.textContent = text; ui.planStatus.classList.toggle('jf-rl-error', !!isError); }
  }

  function renderPlan() {
    const p = plan.data; if (!p || !ui.planTable) return;
    ui.planTable.textContent = '';
    const head = el('tr'); for (const h of ['モード', '予定', '記録', '外れ', '替え', 'データセット']) head.appendChild(el('th', null, h));
    ui.planTable.appendChild(head);
    for (const [key, md] of Object.entries(p.modes)) {
      const ids = p.order.filter((id) => sceneById(p, id)?.mode === key);
      const st = ids.map((id) => p.progress?.[id]?.status);
      const imp = p.imports?.[key];
      const tr = el('tr');
      tr.appendChild(el('td', null, md.name.replace('強化学習モード', '') + (md.state === 'accepted' ? '' : ' ' + md.label)));
      tr.appendChild(el('td', 'jf-rl-nw', md.minutes + ' 分・' + md.scenes));
      tr.appendChild(el('td', null, String(st.filter((x) => x === 'done').length)));
      tr.appendChild(el('td', null, String(st.filter((x) => x === 'failed').length)));
      tr.appendChild(el('td', null, md.spares_used + ' / ' + md.spares.length));
      const td = el('td', null, imp ? ({running: '取り込み中', done: '済み: ' + (imp.dataset || ''), error: 'エラー: ' + (imp.error || '')}[imp.state] || imp.state) : '—');
      if ((!imp || imp.state === 'error') && !RL.planRunning && st.includes('done')) {
        td.textContent = '';
        const b = el('button', 'jf-rl-small', imp ? 'もう一度取り込む' : '記録できた分を取り込む'); b.type = 'button';
        b.addEventListener('click', () => importMode(key));
        td.appendChild(b);
      }
      tr.appendChild(td);
      ui.planTable.appendChild(tr);
    }
    const left = p.order.filter((id) => !FINAL.has(p.progress?.[id]?.status)).length;
    ui.planSummary.textContent = 'プラン ' + p.id + ' ・ ' + p.order.length + ' シーン(のこり ' + left + ')・ 状態 ' + (p.state || '') + (p.message ? ' ・ ' + p.message : '');
    ui.planStart.disabled = RL.planRunning || left === 0;
    ui.planStop.disabled = !RL.planRunning;
  }

  async function refreshPlan() {
    plan.data = await getJSON(PLANS + '/' + encodeURIComponent(plan.id));
    renderPlan();
    return plan.data;
  }
  const progress = async (body) => {
    const r = await postJSON(PLANS + '/' + encodeURIComponent(plan.id) + '/progress', body);
    plan.data = r.plan; renderPlan();
    return r;
  };

  async function importMode(key) {
    try {
      await postJSON('/api/sim/rl/import', {plan: plan.id, mode: key});
      planLog((plan.data.modes[key]?.name || key) + ' のシーンの取り込みを始めました(JetFormer の処理状況に出ます)');
    } catch (e) { planLog('取り込みを始められません: ' + e.message, true); }
    await refreshPlan().catch(() => {});
  }

  function modeComplete(p, key) {
    const ids = p.order.filter((id) => sceneById(p, id)?.mode === key);
    return ids.length && ids.every((id) => FINAL.has(p.progress?.[id]?.status));
  }

  async function runBatchOnce(timeoutMs) {
    const run = $('batch-run'), stopB = $('batch-stop'), st = $('batch-status'), res = $('batch-results');
    for (let i = 0; i < 600 && run.disabled; i++) await sleep(100);                  // テクスチャの読み込み・前の処理を待つ
    if (run.disabled) return {error: '連続試験を始められません(' + st.textContent + ')'};
    const before = st.textContent;
    res.value = '';
    run.click();
    const t0 = Date.now();
    while (stopB.disabled && st.textContent === before && Date.now() - t0 < 5000) await sleep(50);
    if (stopB.disabled) {
      let arr = []; try { arr = JSON.parse(res.value || '[]'); } catch (e) { /* なし */ }
      return arr.length ? arr[arr.length - 1] : {error: st.textContent};
    }
    while (!stopB.disabled) {
      await sleep(400);
      if (Date.now() - t0 > timeoutMs) return {error: '時間切れ(' + Math.round(timeoutMs / 60000) + ' 分)'};
    }
    let arr = []; try { arr = JSON.parse(res.value || '[]'); } catch (e) { /* なし */ }
    const r = arr.length ? arr[arr.length - 1] : {error: st.textContent};
    if (!r.error && /試験停止/.test(st.textContent)) r.error = st.textContent;
    return r;
  }

  async function runPlan() {
    if (RL.planRunning) return;
    if (document.hidden) planLog('このタブを表に出したままにしてください(裏に回ると止まります)', true);
    RL.planRunning = true; plan.stop = false; renderPlan();
    let cal0 = null, errors = 0;
    try {
      const cal = await calibration();
      if (!cal.ok) throw new Error(cal.error || ('ブリッジに読み込んだモデルの換算(操舵の中心 ' + cal.steer_center_us + '・スロットルの中立 ' + cal.throttle_neutral_us
        + ' µs)が実車の値(1442 / 1498 ±5 µs)と違います。そのまま記録すると PWM のラベルがずれるので止めました。モデルを外すか、実車の換算のモデルにしてください'));
      cal0 = cal;
      await progress({plan_state: 'running', calibration: cal, message: '記録中'});
      setField('batch-record', true); setField('automated-labels', true); setField('dataset-mode', 8);
      const exp = $('experiments'); if (exp) exp.open = true;
      for (;;) {
        const p = await refreshPlan();
        for (const key of Object.keys(p.modes)) if (modeComplete(p, key) && !p.imports?.[key]) await importMode(key);
        if (plan.stop) { planLog('止めました'); break; }
        const nextId = p.order.find((id) => !FINAL.has(p.progress?.[id]?.status));
        if (!nextId) { planLog('全部のシーンを記録しました'); break; }
        const sc = sceneById(p, nextId);
        const cal = await calibration();
        if (!cal.ok || calKey(cal) !== calKey(cal0)) throw new Error('記録の途中でブリッジのモデル(換算)が変わりました。同じデータセットの中は同じ換算でなければならないので止めました');
        if (!await select(sc.mode, {reset: false})) throw new Error((p.modes[sc.mode]?.name || sc.mode) + ' を読み込めません');
        RL.sceneParams = {mode: sc.mode, params: sc.params || {}};
        $('batch-plan').value = JSON.stringify([{...RL.basePlan, ...sc.fields}], null, 1);
        setField('batch-record', true); setField('automated-labels', true);
        await progress({scene: sc.id, status: 'recording'});
        planLog(sc.name + '(' + (p.modes[sc.mode]?.name || sc.mode) + ')を記録しています');
        const r = await runBatchOnce(15 * 60 * 1000);
        const m = r.metrics || {};
        const ok = !r.error && r.batch_status === 'completed' && r.frames === 600 && m.collisions === 0 && r.dataset?.session_id;
        const laps = Array.isArray(m.lap_events) ? m.lap_events.map((e) => e.elapsed).filter((x) => typeof x === 'number') : [];
        const body = {scene: sc.id, status: ok ? 'done' : 'failed', frames: r.frames ?? null, collisions: m.collisions ?? null,
          reason: r.outcome?.reason || null, laps, session_id: r.dataset?.session_id || undefined, error: ok ? undefined : (r.error || '20 秒・衝突 0 になりませんでした'),
          replace: !ok && !r.error};
        if (body.session_id && !/^[0-9a-f]{32}$/.test(body.session_id)) delete body.session_id;
        const out = await progress(body);
        if (ok) planLog(sc.name + ': 記録しました(' + laps.length + ' 周)');
        else planLog(sc.name + ': 使えません(' + (body.error || '') + ')' + (out.replacement ? '。替えのシーン ' + out.replacement.name + ' を足しました' : ''), true);
        errors = r.error ? errors + 1 : 0;
        if (r.error && /中断|停止|pause/.test(r.error)) { plan.stop = true; }
        if (errors >= 2) throw new Error('2 シーン続けて記録できませんでした(' + r.error + ')。ブリッジ・ページの状態を確かめてから、もう一度「開始」を押してください');
      }
      const p = await refreshPlan();
      for (const key of Object.keys(p.modes)) if (modeComplete(p, key) && !p.imports?.[key]) await importMode(key);
      await progress({plan_state: plan.stop ? 'stopped' : 'finished', message: plan.stop ? '止めました' : '記録が終わりました'});
    } catch (e) {
      planLog('止まりました: ' + e.message, true);
      try { await progress({plan_state: 'error', message: e.message}); } catch (e2) { /* なし */ }
    } finally {
      RL.planRunning = false; RL.sceneParams = null;
      renderPlan();
    }
  }

  function buildPlanUI(id) {
    plan.id = id;
    const box = el('div', 'jf-rl-plan'); box.id = 'jf-rl-plan';
    box.appendChild(el('div', 'jf-rl-plan-title', 'JetFormer 収集プラン(強化学習モード)'));
    ui.planSummary = el('div', 'jf-rl-muted', '読み込み中…'); box.appendChild(ui.planSummary);
    ui.planTable = el('table', 'jf-rl-table'); box.appendChild(ui.planTable);
    const row = el('div', 'jf-rl-row');
    ui.planStart = el('button', 'jf-rl-btn primary', '開始'); ui.planStart.type = 'button';
    ui.planStop = el('button', 'jf-rl-btn', '今のシーンの後に止める'); ui.planStop.type = 'button'; ui.planStop.disabled = true;
    row.appendChild(ui.planStart); row.appendChild(ui.planStop); box.appendChild(row);
    ui.planStatus = el('div', 'jf-rl-status', ''); box.appendChild(ui.planStatus);
    box.appendChild(el('div', 'jf-rl-muted', '1 シーン(20 秒)ずつアプリの「連続試験」で記録します。このタブは表に出したままにしてください(裏に回ると止まります)。'
      + 'Jetson の Firefox では 1 シーン約 70 秒、Mac の GPU のブラウザなら数十秒です'));
    const det = el('details', 'jf-rl-log'); det.appendChild(el('summary', null, '記録'));
    ui.planLog = el('pre', null, ''); det.appendChild(ui.planLog); box.appendChild(det);
    document.body.appendChild(box);
    ui.planStart.addEventListener('click', () => runPlan());
    ui.planStop.addEventListener('click', () => {
      plan.stop = true; const s = $('batch-stop'); if (s && !s.disabled) s.click();
      planLog('今のシーンが終わったら止めます');
    });
    refreshPlan().then((p) => planLog(p.order.some((id) => !FINAL.has(p.progress?.[id]?.status)) ? 'プランを読み込みました。「開始」で記録します'
      : 'このプランは記録が終わっています(取り込みの様子は表のとおり)')).catch((e) => planLog('プランを読めません: ' + e.message, true));
    setInterval(() => { if (!RL.planRunning && plan.data) refreshPlan().catch(() => {}); }, 5000);
  }

  // ---------------------------------------------------------------- 起動
  async function start() {
    const ok = buildUI();
    if (!ok) return;
    try { await loadRegistry(); fillSelect(); }
    catch (e) { status('強化学習モードの一覧を読めません: ' + e.message, true); return; }
    renderInfo();
    setInterval(() => { if (ui.sel) ui.sel.disabled = busy(); renderLive(); }, 500);
    const ready = await waitApp(20000);
    if (!ready) { status('シミュレーターの準備ができませんでした', true); return; }
    const want = q.get('rl');
    if (want && /^[a-z][a-z0-9_]{0,31}$/.test(want)) await select(want);
    const pid = q.get('rlplan');
    if (pid && /^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$/.test(pid)) buildPlanUI(pid);
  }
  RL.ui = {select, loadWeights, loadRegistry, calibration, runPlan, renderLive};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
