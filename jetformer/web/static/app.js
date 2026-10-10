'use strict';

const $ = (s) => document.querySelector(s);
const state = { tab: 'live', scenes: [], selected: new Set(), status: null, modal: null, playing: false, modes: [] };
const COLORS = { steering: '#2563eb', throttle: '#dc2626', aux: '#9ca3af', motor: '#16a34a',
                 accel: '#7c3aed', ax: '#0891b2', ay: '#f59e0b', gyro: '#db2777', min: '#dc2626', median: '#2563eb' };

// ------------------------------------------------------------ 共通
async function api(method, url, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  const res = await fetch(url, opt);
  let data = null;
  try { data = await res.json(); } catch (e) { /* 本文なし */ }
  if (!res.ok) {
    const msg = (data && data.detail) || `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data;
}
let toastTimer = null;
function toast(msg, error = false) {
  const t = $('#toast');
  t.textContent = msg; t.className = 'toast' + (error ? ' error' : '');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add('hidden'), error ? 6000 : 3000);
}
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtTime = (us) => new Date(us / 1000).toLocaleString('ja-JP', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
const parseTags = (s) => s.split(',').map((x) => x.trim()).filter(Boolean);

// ------------------------------------------------------------ 走行ジャンル
const MODE_COLORS = ['#9ca3af', '#2563eb', '#7c3aed', '#db2777', '#dc2626', '#16a34a', '#d97706', '#0891b2', '#4f46e5', '#65a30d'];
const modeColor = (id) => MODE_COLORS[id % MODE_COLORS.length];
const modeName = (id) => (state.modes.find((m) => m.id === id) || { name: `ID ${id}` }).name;
const modeBadge = (id, mixed) =>
  `<span class="badge mode" style="background:${modeColor(id)}">${id}: ${esc(modeName(id))}${mixed ? ' ほか' : ''}</span>`;

function fillModeSelect(sel, firstOption) {
  const cur = sel.value;
  sel.innerHTML = (firstOption ? `<option value="">${firstOption}</option>` : '') +
    state.modes.map((m) => `<option value="${m.id}">${m.id}: ${esc(m.name)}</option>`).join('');
  if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
}
async function loadModes() {
  try { state.modes = await api('GET', '/api/modes'); } catch (e) { toast(e.message, true); return; }
  const rec = $('#rec-mode');
  const recDefault = !rec.value;
  fillModeSelect(rec);
  if (recDefault && state.modes.length > 1) rec.value = String(state.modes[1].id);
  fillModeSelect($('#scene-mode-filter'), 'すべてのジャンル');
  fillModeSelect($('#bulk-mode'));
  fillModeSelect($('#m-mode'));
  if (state.tab === 'modes') renderModes();
  if (state.tab === 'scenes') renderScenes();
}
function renderModes() {
  $('#mode-table').innerHTML = '<tr><th>ID</th><th>名前</th><th>説明</th><th>シーン</th><th>時間</th><th></th></tr>' +
    state.modes.map((m) => `
      <tr data-id="${m.id}">
        <td><span class="mode-id" style="background:${modeColor(m.id)}">${m.id}</span></td>
        <td>${m.id === 0 ? esc(m.name) : `<input class="mode-name" value="${esc(m.name)}">`}</td>
        <td>${m.id === 0 ? `<span class="muted">${esc(m.description)}</span>` : `<input class="mode-desc" value="${esc(m.description)}">`}</td>
        <td>${m.scenes}</td><td>${m.minutes} 分</td>
        <td>${m.id === 0 ? '' : '<button class="small" data-save-mode="1">保存</button>'}</td>
      </tr>`).join('');
}
$('#mode-table').addEventListener('click', async (e) => {
  const btn = e.target.closest('[data-save-mode]');
  if (!btn) return;
  const tr = btn.closest('tr');
  try {
    await api('PATCH', `/api/modes/${tr.dataset.id}`, { name: tr.querySelector('.mode-name').value, description: tr.querySelector('.mode-desc').value });
    toast('保存しました'); loadModes();
  } catch (err) { toast(err.message, true); }
});
$('#btn-mode-add').addEventListener('click', async () => {
  try {
    const m = await api('POST', '/api/modes', { name: $('#mode-new-name').value, description: $('#mode-new-desc').value });
    toast(`追加しました(ID ${m.id}: ${m.name})`); $('#mode-new-name').value = ''; $('#mode-new-desc').value = ''; loadModes();
  } catch (err) { toast(err.message, true); }
});
$('#rec-mode').addEventListener('change', async () => {
  const st = state.status;
  if (!(st && (st.recorder.recording || (st.session && st.session.active)))) return;
  try { const r = await api('POST', '/api/record/mode', { mode_id: +$('#rec-mode').value }); toast(`ジャンルを「${r.mode_name}」に切り替えました`); }
  catch (err) { toast(err.message, true); }
});

// ------------------------------------------------------------ グラフ(自前の軽量描画)
function drawChart(canvas, series, opt = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.height / (canvas._dpr || 1);
  if (canvas._dpr !== dpr || canvas.width !== Math.round(w * dpr)) {
    canvas._dpr = dpr; canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 44, r: 8, t: 16, b: 18 };
  let xs = [], ys = [];
  for (const s of series) { xs = xs.concat(s.x); ys = ys.concat(s.y.filter((v) => v !== null && isFinite(v))); }
  if (!xs.length || !ys.length) {
    ctx.fillStyle = '#9ca3af'; ctx.font = '12px sans-serif'; ctx.fillText('データなし', pad.l + 4, h / 2); return;
  }
  let x0 = opt.xMin ?? Math.min(...xs), x1 = opt.xMax ?? Math.max(...xs);
  let y0 = opt.yMin ?? Math.min(...ys), y1 = opt.yMax ?? Math.max(...ys);
  if (y1 - y0 < (opt.minSpan || 1e-6)) { const c = (y0 + y1) / 2, s = (opt.minSpan || 1) / 2; y0 = c - s; y1 = c + s; }
  if (x1 <= x0) x1 = x0 + 1;
  const X = (v) => pad.l + (v - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - y0) / (y1 - y0)) * (h - pad.t - pad.b);
  ctx.strokeStyle = '#e5e7eb'; ctx.fillStyle = '#6b7280'; ctx.font = '10px sans-serif'; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const v = y0 + (y1 - y0) * i / 3, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1), 2, y + 3);
  }
  const xLabel = opt.xLabel || ((v) => `${v.toFixed(1)}s`);
  ctx.fillText(xLabel(x0), pad.l, h - 4);
  const right = xLabel(x1);
  ctx.fillText(right, w - pad.r - ctx.measureText(right).width, h - 4);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.x.length; i++) {
      const v = s.y[i];
      if (v === null || !isFinite(v)) { started = false; continue; }
      const px = X(s.x[i]), py = Y(v);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  let lx = pad.l + 6;
  for (const s of series) {
    if (!s.label) continue;
    ctx.fillStyle = s.color; ctx.fillRect(lx, 4, 10, 3);
    ctx.fillStyle = '#374151'; ctx.fillText(s.label, lx + 13, 9); lx += ctx.measureText(s.label).width + 26;
  }
  if (opt.cursor !== undefined && opt.cursor !== null) {
    ctx.strokeStyle = '#111827'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(opt.cursor), pad.t); ctx.lineTo(X(opt.cursor), h - pad.b); ctx.stroke(); ctx.setLineDash([]);
  }
}

// ------------------------------------------------------------ タブ
function showTab(name) {
  state.tab = name;
  document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-page').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + name));
  for (const [tab, id] of [['live', '#live-video'], ['drive', '#drive-video']]) {
    const video = $(id);
    if (name === tab) { if (!video.src.includes('live.mjpg')) video.src = '/api/live.mjpg?fps=15&_=' + Date.now(); }
    else video.removeAttribute('src');                    // 見ていない時は映像を止めて帯域を節約
  }
  if (name === 'scenes') loadScenes();
  if (name === 'modes') loadModes();
  if (name === 'train') loadTrain();
  if (name === 'drive') loadDrive();
  if (name === 'datasets') { loadDatasets(); loadJobs(); updateDsSelected(); }
}
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

// ------------------------------------------------------------ 状態(1秒ごと)
const SENSOR_LABEL = { CAM_FRONT: 'カメラ', TOF_FRONT: 'ToF', IMU: 'IMU', VEHICLE: '車両(PWM)', AUDIO: 'マイク', AI_CMD: 'AI指令' };
const EXPECTED_HZ = { CAM_FRONT: 30, TOF_FRONT: 30, IMU: 100, VEHICLE: 50, AUDIO: 50 };

function sensorExtra(name, s) {
  const parts = [];
  if (name === 'CAM_FRONT' && s.dropped !== undefined) parts.push(`破棄 ${s.dropped}`, s.exposure_us ? `露出 ${(s.exposure_us / 1000).toFixed(1)}ms` : '');
  if (name === 'TOF_FRONT') { parts.push(`破損 ${s.bad_frames}`, `欠落 ${s.skipped}`); }
  if (name === 'IMU') parts.push(s.mag ? '地磁気あり' : '地磁気なし');
  if (name === 'AUDIO' && s.connected) {
    parts.push(s.no_signal ? '<span class="warn-text">信号なし(40ピンのI2Sが無効?)</span>' : `${s.level_dbfs ?? '-'} dBFS`);
    if (s.channel && !s.no_signal) parts.push(`ch ${s.channel}`);
  }
  if (name === 'VEHICLE') {
    if (s.protocol === 'legacy') parts.push('<span class="warn-text">v1ファーム(値の取り違えあり)</span>');
    else parts.push(`v2 / 読み直し ${(s.read_errors || 0) + (s.bus_errors || 0)}${s.skipped_polls ? ` / 飛ばし ${s.skipped_polls}` : ''}`);
    if (s.select !== undefined) parts.push(s.select ? '<span class="warn-text">AI走行中</span>' : '人が操縦中');
  }
  return parts.filter(Boolean).join(' / ');
}

function renderStatus(st) {
  state.status = st;
  const rec = st.recorder;
  const ind = $('#rec-indicator');
  ind.textContent = (st.driving ? 'AI走行中 ' : '') + (rec.recording ? `● 収集中 ${rec.log} ・ ${rec.mode_name}` : st.training ? '学習中' : st.driving ? '' : '待機中');
  ind.className = st.driving ? 'rec-drive' : rec.recording ? 'rec-on' : st.training ? 'rec-train' : 'rec-off';
  if (st.training && !trainTimer) pollTrain();
  $('#mic-card').classList.toggle('hidden', !st.sensors.AUDIO);
  if (st.driving && !driveTimer) pollDrive();
  $('#disk').textContent = `空き ${st.disk.free_gb} GB`;

  const rows = Object.entries(st.sensors).map(([name, s]) => {
    let cls = 'off', text = '停止';
    if (s.running && s.connected) {
      const ok = s.rate_hz >= EXPECTED_HZ[name] * 0.85;
      cls = ok ? 'ok' : 'warn'; text = `${s.rate_hz} Hz`;
    } else if (s.running) { cls = 'ng'; text = '未接続'; }
    const err = s.last_error ? `<div class="small warn-text">${esc(s.last_error)}</div>` : '';
    return `<tr><td><span class="dot ${cls}"></span>${SENSOR_LABEL[name] || name}</td><td>${text}</td>
            <td class="small muted">${sensorExtra(name, s)}${err}</td></tr>`;
  });
  $('#sensor-table').innerHTML = rows.join('');

  const recState = $('#rec-state');
  const ses = st.session || {};
  $('#run-panel').classList.toggle('hidden', !ses.active);
  if (ses.active) renderRunSession(ses, rec);
  else if (rec.recording) {
    recState.innerHTML = `収集中: シーン ${rec.scene_index}${rec.max_scenes ? ' / ' + rec.max_scenes : ''} ${modeBadge(rec.mode_id)}`;
    recState.className = 'rec-state on';
    $('#scene-progress').style.width = `${Math.min(100, rec.scene_elapsed_s / rec.scene_seconds * 100)}%`;
    const c = rec.current_counts || {};
    $('#rec-detail').textContent = `${rec.scene_elapsed_s.toFixed(1)} / ${rec.scene_seconds} 秒  ` +
      `(カメラ ${c.CAM_FRONT || 0} / ToF ${c.TOF_FRONT || 0} / IMU ${c.IMU || 0} / 車両 ${c.VEHICLE || 0})` +
      (rec.dropped_writes ? `  保存遅れ ${rec.dropped_writes}` : '');
  } else {
    const last = ses.phase === 'idle' && ses.log ? ses : {};      // 直前の「走行ごとの収集」の結果
    recState.textContent = last.error ? '止まりました(エラー)' : '待機中';
    recState.className = 'rec-state';
    $('#scene-progress').style.width = '0';
    $('#rec-detail').textContent = last.error ? '⚠ ' + last.error : rec.last_error ? '⚠ ' + rec.last_error
      : last.message ? `前回: ${last.message}` : `1シーン = ${st.config.scene_seconds} 秒`;
  }
  $('#btn-rec-start').disabled = rec.recording || ses.active || st.training;
  $('#btn-rec-stop').disabled = !(rec.recording || ses.active);
  $('#btn-sensors-stop').disabled = rec.recording || ses.active || st.driving || !st.sensors_running;
  $('#btn-sensors-start').disabled = st.sensors_running;
  const saved = rec.saved_scenes || [];
  $('#session-scenes').innerHTML = saved.length
    ? `<div class="muted">保存したシーン (${saved.length})</div>` + saved.slice().reverse().map((n) => `<div>✓ ${esc(n)}</div>`).join('')
    : '';
  if (saved.length !== state._savedCount) { state._savedCount = saved.length; if (state.tab === 'scenes') loadScenes(); }
}

// 走行ごとの収集: カウントダウン(3・2・1)/走行中の経過/次の走行までの停止 を大きく出す
function renderRunSession(ses, rec) {
  const recState = $('#rec-state');
  const n = ses.run || 1, of = ses.runs > 1 ? ` / ${ses.runs}` : '';
  let big = '', detail = '', width = 0;
  if (ses.phase === 'starting') { big = '準備中…'; detail = 'センサーの立ち上がりを待っています'; }
  else if (ses.phase === 'countdown') {
    big = ses.countdown_s === null ? `${n}回目の走行をスタート` : ses.countdown_s > 0 ? String(ses.countdown_s) : 'ピ';
    detail = `${n}回目${of} の走行の準備`;
  } else if (ses.phase === 'running') {
    big = `${n}回目${of} 走行中`;
    detail = `${(ses.run_elapsed_s ?? 0).toFixed(1)} / ${ses.run_seconds} 秒`;
    width = Math.min(100, (ses.run_elapsed_s ?? 0) / ses.run_seconds * 100);
  } else if (ses.phase === 'pause') {
    big = `次の走行まで ${Math.ceil(ses.pause_left_s ?? 0)} 秒`;
    detail = `${ses.done}回の走行を記録済み${of ? '(全' + ses.runs + '回)' : ''}`;
    width = Math.min(100, (1 - (ses.pause_left_s ?? 0) / Math.max(ses.pause_s, 0.1)) * 100);
  } else { big = '保存中…'; }
  $('#run-big').textContent = big;
  $('#run-big').className = 'run-big ' + ses.phase;
  recState.innerHTML = `収集中(走行ごと) ${rec.mode_id !== undefined ? modeBadge(rec.mode_id) : ''}`;
  recState.className = 'rec-state on';
  $('#scene-progress').style.width = `${width}%`;
  $('#rec-detail').textContent = detail + (ses.error ? ` ⚠ ${ses.error}` : '') + (rec.last_error ? ` ⚠ ${rec.last_error}` : '')
    + (ses.sound_error ? '(音声を作れないため音なしで数えています)' : '');
  $('#btn-run-end').disabled = ses.phase !== 'running';
  $('#btn-run-next').disabled = ses.phase !== 'pause';
}

async function pollStatus() {
  let fast = false;
  try { const st = await api('GET', '/api/status'); renderStatus(st); fast = st.session && st.session.active; } catch (e) { /* 再接続待ち */ }
  clearTimeout(state._statusTimer);
  state._statusTimer = setTimeout(pollStatus, fast ? 250 : 1000);
}

// ------------------------------------------------------------ ライブ値(0.3秒ごと)
function renderValues(v) {
  const veh = v.vehicle || {}, imu = v.imu || {};
  drawChart($('#chart-pwm'), [
    { x: veh.t || [], y: (veh.steering || []).map((x) => x || null), color: COLORS.steering, label: 'ステア' },
    { x: veh.t || [], y: (veh.throttle || []).map((x) => x || null), color: COLORS.throttle, label: 'スロットル' },
  ], { xMin: -10, xMax: 0, minSpan: 100 });
  drawChart($('#chart-motor'), [{ x: veh.t || [], y: veh.erpm || [], color: COLORS.motor, label: 'eRPM' }],
    { xMin: -10, xMax: 0, minSpan: 1000 });
  drawChart($('#chart-accel'), [
    { x: imu.t || [], y: imu.accel_norm || [], color: COLORS.accel, label: '|a|' },
    { x: imu.t || [], y: imu.accel_x || [], color: COLORS.ax, label: 'x(前後)' },
    { x: imu.t || [], y: imu.accel_y || [], color: COLORS.ay, label: 'y(左右)' },
  ], { xMin: -10, xMax: 0, minSpan: 4 });
  drawChart($('#chart-gyro'), [{ x: imu.t || [], y: imu.gyro_z || [], color: COLORS.gyro, label: 'z' }],
    { xMin: -10, xMax: 0, minSpan: 20 });
  const rows = [];
  if (v.vehicle_now) {
    const n = v.vehicle_now, p = n.pwm, ok = n.valid || [];
    const ch = (i, label) => [label, ok[i] === false ? '<span class="warn-text">受信なし</span>' : `${p[i]} µs`];
    rows.push(ch(0, 'ステアリング(CH1)'), ch(1, 'スロットル(CH2)'), ch(2, 'モード(CH3)'),
              ['モーター', `${n.erpm} eRPM${n.moving ? '(回転中)' : ''}`],
              ['操縦', n.select ? '<span class="warn-text">AI(学習には使わない)</span>' : '人(プロポ)']);
  }
  if (v.tof) rows.push(['ToF 最小/中央', `${v.tof.min} / ${v.tof.center} mm`], ['ToF 無効ゾーン', `${v.tof.invalid} / 2268`]);
  if (v.imu_now) {
    rows.push(['加速度', v.imu_now.accel.map((x) => x.toFixed(2)).join(', ') + ' m/s²'],
              ['角速度', v.imu_now.gyro.map((x) => x.toFixed(1)).join(', ') + ' °/s'], ['IMU温度', `${v.imu_now.temp} ℃`]);
  }
  $('#values').innerHTML = rows.length ? rows.map(([k, val]) => `<div class="k">${k}</div><div class="v">${val}</div>`).join('')
    : '<div class="muted">センサー停止中</div>';
  const a = v.audio_now;
  $('#mic-bar').style.width = a && !a.zero ? `${Math.max(0, Math.min(100, (a.rms_dbfs + 90) / 90 * 100))}%` : '0';
  $('#mic-level').innerHTML = !a ? 'センサー停止中(またはマイクを開けません)'
    : a.zero ? '<span class="warn-text">信号なし: 40ピンヘッダのI2S(pin12/35/38)が無効です。scripts/i2s_mic/enable.sh で有効にしてください</span>'
    : `${a.rms_dbfs} dBFS(ピーク ${a.peak_dbfs} dBFS)・${a.channel === 1 ? 'R' : 'L'}チャネル`;
}

// ------------------------------------------------------------ サウンド(車のスピーカーで収集・走行の開始/終了などを知らせる)
async function loadSound() {
  let s;
  try { s = await api('GET', '/api/sound'); } catch (e) { return; }
  $('#snd-enabled').checked = s.enabled;
  $('#snd-volume').value = s.volume;
  $('#snd-volume-v').textContent = s.volume;
  $('#snd-events').innerHTML = Object.entries(s.labels).map(([k, label]) =>
    `<label class="inline small"><input type="checkbox" data-snd-event="${k}" ${s.events[k] ? 'checked' : ''}> ${esc(label)}</label>`).join('');
  $('#snd-status').innerHTML = (!s.tts ? '<div class="warn-text">音声合成(Open JTalk)が使えません。チャイムだけ鳴らします</div>' : '')
    + (s.last_error ? `<div class="warn-text">鳴らせませんでした: ${esc(s.last_error)}</div>`
       : s.last_text ? `<div>最後に鳴らした音: ${esc(s.last_text)}</div>` : '');
}
async function saveSound(values) {
  try { await api('POST', '/api/sound', values); } catch (e) { toast(e.message, true); }
}
$('#snd-enabled').addEventListener('change', () => saveSound({ enabled: $('#snd-enabled').checked }));
let sndTimer = null;
$('#snd-volume').addEventListener('input', () => {
  $('#snd-volume-v').textContent = $('#snd-volume').value;
  clearTimeout(sndTimer);
  sndTimer = setTimeout(() => saveSound({ volume: +$('#snd-volume').value }), 200);
});
$('#snd-events').addEventListener('change', (e) => {
  const k = e.target.dataset.sndEvent;
  if (k) saveSound({ events: { [k]: e.target.checked } });
});
$('#btn-snd-test').addEventListener('click', async () => {
  try { await api('POST', '/api/sound/test'); toast('テストの音を鳴らします'); setTimeout(loadSound, 4000); }
  catch (e) { toast(e.message, true); }
});
async function saySound() {
  const text = $('#snd-text').value.trim();
  if (!text) return;
  try { await api('POST', '/api/sound/say', { text }); $('#snd-text').value = ''; setTimeout(loadSound, 4000); }
  catch (e) { toast(e.message, true); }
}
$('#btn-snd-say').addEventListener('click', saySound);
$('#snd-text').addEventListener('keydown', (e) => { if (e.key === 'Enter') saySound(); });

// ------------------------------------------------------------ マイクを聞く(16kHz float32 を受け取り Web Audio で鳴らす)
const micGain = () => Math.pow(10, +$('#mic-gain').value);
let mic = null;
async function startMic() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx(), gain = ctx.createGain(), ctrl = new AbortController();
  gain.gain.value = micGain(); gain.connect(ctx.destination);
  mic = { ctx, gain, ctrl };
  $('#btn-mic-listen').textContent = '■ 止める';
  try {
    const res = await fetch('/api/live/audio.f32', { signal: ctrl.signal });
    const rate = +(res.headers.get('X-Sample-Rate') || 16000);
    const reader = res.body.getReader();
    let rest = new Uint8Array(0), t = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const buf = new Uint8Array(rest.length + value.length);
      buf.set(rest); buf.set(value, rest.length);
      const n = buf.length - (buf.length % 4);
      rest = buf.slice(n);
      if (!n) continue;
      const f = new Float32Array(buf.buffer.slice(0, n));
      const ab = ctx.createBuffer(1, f.length, rate);
      ab.getChannelData(0).set(f);
      const src = ctx.createBufferSource();
      src.buffer = ab; src.connect(gain);
      const now = ctx.currentTime;
      if (t < now + 0.03 || t > now + 0.6) t = now + 0.12;     // 途切れた・遅れがたまったら仕切り直す
      src.start(t); t += ab.duration;
    }
  } catch (e) { if (e.name !== 'AbortError') toast('マイクの音を受け取れません: ' + e.message, true); }
  stopMic();
}
function stopMic() {
  if (!mic) return;
  mic.ctrl.abort(); mic.ctx.close(); mic = null;
  $('#btn-mic-listen').textContent = '🔊 聞く';
}
$('#btn-mic-listen').addEventListener('click', () => (mic ? stopMic() : startMic()));
$('#mic-gain').addEventListener('input', () => {
  $('#mic-gain-v').textContent = `×${Math.round(micGain())}`;
  if (mic) mic.gain.gain.value = micGain();
});
async function pollValues() {
  if (state.tab === 'live' && document.visibilityState === 'visible') {
    try { renderValues(await api('GET', '/api/live/values')); } catch (e) { /* 無視 */ }
  }
  setTimeout(pollValues, 300);
}

// ------------------------------------------------------------ 収集の操作
const recMethod = () => document.querySelector('input[name=rec-method]:checked').value;
function syncRecMethod() {        // ブラウザが前回の選択を復元しても、表示と合うように
  $('#rec-runs-opts').classList.toggle('hidden', recMethod() !== 'runs');
  $('#rec-cont-opts').classList.toggle('hidden', recMethod() === 'runs');
}
document.querySelectorAll('input[name=rec-method]').forEach((r) => r.addEventListener('change', syncRecMethod));
window.addEventListener('pageshow', syncRecMethod);
syncRecMethod();
$('#btn-rec-start').addEventListener('click', async () => {
  const mode = document.querySelector('input[name=rec-mode]:checked').value;
  const body = { method: recMethod(), mode_id: +$('#rec-mode').value, description: $('#rec-desc').value,
                 tags: parseTags($('#rec-tags').value), location: $('#rec-location').value,
                 max_scenes: mode === 'count' ? parseInt($('#rec-count').value, 10) || 1 : null,
                 runs: $('#rec-runs').value, run_seconds: $('#rec-run-seconds').value, pause_s: $('#rec-pause').value };
  try {
    const r = await api('POST', '/api/record/start', body);
    toast(r.method === 'runs' ? `走行ごとの収集を始めます(${body.runs}回)` : `収集を開始しました (${r.log})`);
    pollStatus();
  } catch (e) { toast('開始できません: ' + e.message, true); }
});
$('#btn-rec-stop').addEventListener('click', async () => {
  try { await api('POST', '/api/record/stop'); toast('停止しました。最後のシーンを保存しています…'); }
  catch (e) { toast(e.message, true); }
});
$('#btn-run-end').addEventListener('click', () => api('POST', '/api/record/end_run').catch((e) => toast(e.message, true)));
$('#btn-run-next').addEventListener('click', () => api('POST', '/api/record/next').catch((e) => toast(e.message, true)));
$('#btn-sensors-start').addEventListener('click', () => api('POST', '/api/sensors/start').catch((e) => toast(e.message, true)));
$('#btn-sensors-stop').addEventListener('click', () => api('POST', '/api/sensors/stop').catch((e) => toast(e.message, true)));

// ------------------------------------------------------------ シーン一覧
async function loadScenes() {
  try { state.scenes = await api('GET', '/api/scenes'); } catch (e) { toast(e.message, true); return; }
  const names = new Set(state.scenes.map((s) => s.name));
  for (const n of [...state.selected]) if (!names.has(n)) state.selected.delete(n);
  $('#scene-count').textContent = state.scenes.length;
  renderScenes();
}
function filteredScenes() {
  const q = $('#scene-filter').value.trim().toLowerCase();
  const onlyComplete = $('#only-complete').checked;
  return state.scenes.filter((s) => {
    if (onlyComplete && !s.complete) return false;
    const mf = $('#scene-mode-filter').value;
    if (mf !== '' && s.mode_id !== +mf && !(s.mode_segments || []).some((sg) => sg.mode_id === +mf)) return false;
    if (!q) return true;
    return [s.name, s.description, (s.tags || []).join(' ')].join(' ').toLowerCase().includes(q);
  }).slice().reverse();
}
function sceneBadges(s) {
  const b = [modeBadge(s.mode_id, s.mode_mixed)];
  if (s.run_index) b.push(`<span class="badge ${s.complete ? 'ok' : 'warn'}">${s.run_index}回目 ${Math.round(s.duration_s)}秒${s.complete ? '' : '(途中で停止)'}</span>`);
  else b.push(s.complete ? '<span class="badge ok">20秒</span>' : `<span class="badge warn">${s.duration_s}秒</span>`);
  if (s.vehicle_quality === 'legacy') b.push('<span class="badge warn">PWM:v1(学習不可)</span>');
  else if (s.vehicle_quality) b.push('<span class="badge ok">PWM:v2</span>');
  if (s.drive) b.push(`<span class="badge ai">AIモデル${s.drive.dry_run ? '(出力なし)' : ''}</span>`);
  if (s.stats && s.stats.ai_driving_ratio > 0) b.push(`<span class="badge warn">AI走行 ${Math.round(s.stats.ai_driving_ratio * 100)}%</span>`);
  if (s.stats && s.stats.camera_max_gap_ms > 70) b.push(`<span class="badge warn">映像途切れ ${s.stats.camera_max_gap_ms}ms</span>`);
  for (const t of s.tags || []) b.push(`<span class="badge tag">${esc(t)}</span>`);
  return b.join('');
}
function renderScenes() {
  const list = filteredScenes();
  const sel = state.selected;
  $('#scene-grid').innerHTML = list.map((s) => `
    <div class="scene ${sel.has(s.name) ? 'selected' : ''}" data-name="${esc(s.name)}">
      <div class="check" data-check="1">${sel.has(s.name) ? '✓' : ''}</div>
      <img loading="lazy" src="/api/scenes/${encodeURIComponent(s.name)}/thumb.jpg" alt="">
      <div class="body">
        <div class="name">${esc(s.name)}</div>
        <div class="muted">${fmtTime(s.start_us)} ・ カメラ ${s.counts.CAM_FRONT} / ToF ${s.counts.TOF_FRONT} / IMU ${s.counts.IMU} / 車両 ${s.counts.VEHICLE}${s.counts.AUDIO ? ' / 音声あり' : ''}</div>
        ${s.description ? `<div>${esc(s.description)}</div>` : ''}
        <div>${sceneBadges(s)}</div>
      </div>
    </div>`).join('') || '<div class="muted">シーンがありません。ライブ・収集タブで収集してください。</div>';
  const secs = state.scenes.filter((s) => sel.has(s.name)).reduce((a, s) => a + s.duration_s, 0);
  $('#selection-info').textContent = sel.size ? `${sel.size} シーン選択 (${(secs / 60).toFixed(1)} 分)` : '';
  const total = state.scenes.reduce((a, s) => a + s.duration_s, 0);
  $('#scene-summary').textContent = `全 ${state.scenes.length} シーン / ${(total / 60).toFixed(1)} 分 (表示 ${list.length})`;
}
$('#scene-grid').addEventListener('click', (ev) => {
  const card = ev.target.closest('.scene');
  if (!card) return;
  const name = card.dataset.name;
  if (ev.target.closest('[data-check]') || ev.shiftKey || ev.metaKey || ev.ctrlKey) {
    state.selected.has(name) ? state.selected.delete(name) : state.selected.add(name);
    renderScenes();
  } else openScene(name);
});
$('#scene-filter').addEventListener('input', renderScenes);
$('#only-complete').addEventListener('change', renderScenes);
$('#scene-mode-filter').addEventListener('change', renderScenes);
$('#btn-bulk-mode').addEventListener('click', async () => {
  const names = [...state.selected];
  if (!names.length) { toast('シーンを選択してください(カード左上のチェック)', true); return; }
  const id = +$('#bulk-mode').value;
  if (!confirm(`${names.length} シーンのジャンルを「${modeName(id)}」にします(途中の切り替えは上書きされます)。`)) return;
  try { const r = await api('POST', '/api/scenes/bulk', { names, mode_id: id }); toast(`${r.updated} シーンを更新しました`); loadScenes(); }
  catch (e) { toast(e.message, true); }
});
$('#btn-select-all').addEventListener('click', () => { filteredScenes().forEach((s) => state.selected.add(s.name)); renderScenes(); });
$('#btn-select-none').addEventListener('click', () => { state.selected.clear(); renderScenes(); });
$('#btn-to-dataset').addEventListener('click', () => {
  if (!state.selected.size) { toast('シーンを選択してください(カード左上のチェック)', true); return; }
  showTab('datasets');
});
$('#btn-delete-selected').addEventListener('click', async () => {
  const names = [...state.selected];
  if (!names.length) return;
  if (!confirm(`${names.length} シーンを削除します。元に戻せません。よろしいですか?`)) return;
  for (const n of names) {
    try { await api('DELETE', `/api/scenes/${encodeURIComponent(n)}`); state.selected.delete(n); }
    catch (e) { toast(`${n}: ${e.message}`, true); }
  }
  loadScenes();
});

// ------------------------------------------------------------ シーン詳細
async function openScene(name) {
  let s;
  try { s = await api('GET', `/api/scenes/${encodeURIComponent(name)}`); } catch (e) { toast(e.message, true); return; }
  state.modal = s; state.playing = false;
  $('#m-name').textContent = s.name;
  $('#m-badges').innerHTML = sceneBadges(s);
  const st = s.stats || {};
  const fmt = (o) => (o ? `${o.min} 〜 ${o.max} (平均 ${o.mean})` : '-');
  $('#m-info').innerHTML = [
    ['開始', fmtTime(s.start_us)], ['長さ', `${s.duration_s} 秒`],
    ['件数', `カメラ ${s.counts.CAM_FRONT} / ToF ${s.counts.TOF_FRONT} / IMU ${s.counts.IMU} / 車両 ${s.counts.VEHICLE}`],
    ['レート', Object.entries(s.rates).map(([k, v]) => `${SENSOR_LABEL[k] || k} ${v}Hz`).join(', ')],
    ['ステアリング', fmt(st.steering_us)], ['スロットル', fmt(st.throttle_us)], ['モード(CH3)', fmt(st.mode_us)],
    ['モーター eRPM', fmt(st.erpm)],
    ['AI走行の割合', st.ai_driving_ratio !== undefined ? `${Math.round(st.ai_driving_ratio * 100)} %(学習には使わない)` : '-'],
    ['受信断の割合', st.rc_lost_ratio !== undefined ? `${Math.round(st.rc_lost_ratio * 100)} %` : '-'],
    ['加速度ピーク', st.accel_peak_ms2 !== undefined ? `${st.accel_peak_ms2} m/s²` : '-'],
    ['映像の最大間隔', `${st.camera_max_gap_ms} ms`], ['車両データ', s.vehicle_quality || '-'],
    ['マイク', st.audio ? `平均 ${st.audio.rms_dbfs} dBFS / ピーク ${st.audio.peak_dbfs} dBFS${st.audio.no_signal ? '(信号なし)' : ''}` : '-'],
    ...(s.drive ? [['AIモデル', `${s.drive.run}${s.drive.dry_run ? '(出力なし)' : ''}`],
                   ['走行設定', `ステアリング×${(s.drive.params || {}).steer_gain} / 速度×${(s.drive.params || {}).throttle_gain}(開始時)`]] : []),
    ...(st.ai ? [['AIの遅延', `平均 ${st.ai.latency_ms_mean} ms / 最大 ${st.ai.latency_ms_max} ms(推論 ${st.ai.infer_ms_mean} ms)`],
                 ['AIが車を操作', `${Math.round(st.ai.applied_ratio * 100)} % / 中立で停止 ${Math.round(st.ai.hold_ratio * 100)} %`]] : []),
  ].map(([k, v]) => `<div><span class="muted">${k}:</span> ${esc(v)}</div>`).join('');
  $('#m-mode').value = String(s.mode_id);
  $('#m-segments').innerHTML = s.mode_mixed
    ? '<div class="muted">途中でジャンルを切り替えています(ここで変更するとシーン全体が1つのジャンルになります)</div><div class="segments">' +
      s.mode_segments.map((sg) => `<div>${sg.start_s.toFixed(1)}〜${sg.end_s.toFixed(1)} 秒: ${modeBadge(sg.mode_id)}</div>`).join('') + '</div>'
    : '';
  const au = $('#m-audio');
  au.pause();
  if (s.counts.AUDIO) { au.src = `/api/scenes/${encodeURIComponent(s.name)}/audio.wav`; au.classList.remove('hidden'); }
  else { au.removeAttribute('src'); au.classList.add('hidden'); }
  $('#m-desc').value = s.description || '';
  $('#m-tags').value = (s.tags || []).join(', ');
  const slider = $('#m-slider');
  slider.max = Math.max(0, s.series.frames - 1); slider.value = 0;
  $('#modal').classList.remove('hidden');
  showFrame(0);
}
function frameTime(i) { const t = state.modal.series.frame_t; return t.length ? t[Math.min(i, t.length - 1)] : 0; }
function drawSceneCharts(cursor) {
  const ser = state.modal.series;
  const v = ser.vehicle, imu = ser.imu, tof = ser.tof;
  const xMax = ser.frame_t.length ? ser.frame_t[ser.frame_t.length - 1] : undefined;
  drawChart($('#m-chart-veh'), v ? [
    { x: v.t, y: v.steering.map((x) => x || null), color: COLORS.steering, label: 'ステア[µs]' },
    { x: v.t, y: v.throttle.map((x) => x || null), color: COLORS.throttle, label: 'スロットル[µs]' },
  ] : [], { cursor, xMin: 0, xMax, minSpan: 100 });
  drawChart($('#m-chart-imu'), imu ? [
    { x: imu.t, y: imu.accel.map((a) => Math.hypot(...a)), color: COLORS.accel, label: '|a|' },
    { x: imu.t, y: imu.gyro.map((g) => g[2] / 10), color: COLORS.gyro, label: 'ヨー/10' },
  ] : [], { cursor, xMin: 0, xMax, minSpan: 4 });
  drawChart($('#m-chart-tof'), tof ? [
    { x: tof.t, y: tof.min, color: COLORS.min, label: '最小' },
    { x: tof.t, y: tof.median, color: COLORS.median, label: '中央値' },
  ] : [], { cursor, xMin: 0, xMax });
  const ai = ser.ai;
  $('#m-ai-card').classList.toggle('hidden', !ai);
  if (ai) drawChart($('#m-chart-ai'), [
    { x: ai.t, y: ai.raw_steer, color: '#93c5fd', label: 'ステア予測' },
    { x: ai.t, y: ai.out_steer, color: COLORS.steering, label: 'ステア出力' },
    { x: ai.t, y: ai.raw_throttle, color: '#fca5a5', label: 'スロットル予測' },
    { x: ai.t, y: ai.out_throttle, color: COLORS.throttle, label: 'スロットル出力' },
  ], { cursor, xMin: 0, xMax, minSpan: 100 });
}
function showFrame(i) {
  const s = state.modal;
  if (!s) return;
  i = Math.max(0, Math.min(i, s.series.frames - 1));
  $('#m-slider').value = i;
  $('#m-pos').textContent = `${i + 1} / ${s.series.frames} (${frameTime(i).toFixed(2)} 秒)`;
  $('#m-frame').src = `/api/scenes/${encodeURIComponent(s.name)}/frame/${i}.jpg`;
  drawSceneCharts(frameTime(i));
}
$('#m-slider').addEventListener('input', (e) => { state.playing = false; $('#m-play').textContent = '▶ 再生'; showFrame(+e.target.value); });
$('#m-play').addEventListener('click', () => {
  state.playing = !state.playing;
  $('#m-play').textContent = state.playing ? '❚❚ 停止' : '▶ 再生';
  if (state.playing) playStep();
});
function playStep() {
  if (!state.playing || !state.modal) return;
  const next = (+$('#m-slider').value + 3) % state.modal.series.frames;   // 3フレーム飛ばし(約10枚/秒)
  const img = $('#m-frame');
  img.onload = () => { img.onload = null; setTimeout(playStep, 60); };
  showFrame(next);
}
$('#m-close').addEventListener('click', closeModal);
$('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
document.addEventListener('keydown', (e) => {
  if (!state.modal) return;
  if (e.key === 'Escape') closeModal();
  if (e.key === 'ArrowRight') showFrame(+$('#m-slider').value + 1);
  if (e.key === 'ArrowLeft') showFrame(+$('#m-slider').value - 1);
});
function closeModal() { state.playing = false; state.modal = null; $('#m-audio').pause(); $('#modal').classList.add('hidden'); }
$('#m-save').addEventListener('click', async () => {
  try {
    const body = { description: $('#m-desc').value, tags: parseTags($('#m-tags').value) };
    if (+$('#m-mode').value !== state.modal.mode_id || state.modal.mode_mixed) {
      if (+$('#m-mode').value !== state.modal.mode_id || confirm('シーン全体を1つのジャンルにまとめますか?')) body.mode_id = +$('#m-mode').value;
    }
    const updated = await api('PATCH', `/api/scenes/${encodeURIComponent(state.modal.name)}`, body);
    Object.assign(state.modal, { mode_id: updated.mode_id, mode_mixed: updated.mode_mixed, mode_segments: updated.mode_segments });
    $('#m-badges').innerHTML = sceneBadges(Object.assign({}, state.modal, updated));
    toast('保存しました'); loadScenes();
  } catch (e) { toast(e.message, true); }
});
$('#m-delete').addEventListener('click', async () => {
  const name = state.modal.name;
  if (!confirm(`${name} を削除します。元に戻せません。よろしいですか?`)) return;
  try { await api('DELETE', `/api/scenes/${encodeURIComponent(name)}`); closeModal(); toast('削除しました'); loadScenes(); }
  catch (e) { toast(e.message, true); }
});

// ------------------------------------------------------------ データセット
function updateDsSelected() {
  const sel = state.scenes.filter((s) => state.selected.has(s.name));
  const secs = sel.reduce((a, s) => a + s.duration_s, 0);
  const byMode = {};
  for (const s of sel) { const k = s.mode_id; byMode[k] = byMode[k] || { n: 0, secs: 0 }; byMode[k].n++; byMode[k].secs += s.duration_s; }
  const rows = Object.entries(byMode).sort((a, b) => a[0] - b[0])
    .map(([id, v]) => `<div>${modeBadge(+id)} ${v.n} シーン / ${(v.secs / 60).toFixed(1)} 分</div>`).join('');
  const note = sel.some((s) => s.mode_mixed) ? '<div class="muted small">途中で切り替えたシーンは主なジャンルで集計しています(学習はフレームごとのジャンルで行います)</div>' : '';
  const unset = byMode[0] ? '<div class="warn-text small">「未設定」のシーンは学習に使われません</div>' : '';
  $('#ds-selected').innerHTML = sel.length
    ? `選択中: <b>${sel.length}</b> シーン / ${(secs / 60).toFixed(1)} 分 / 約 ${sel.reduce((a, s) => a + s.counts.CAM_FRONT, 0)} フレーム${rows}${note}${unset}`
    : '<span class="warn-text">シーンタブでシーンを選択してください</span>';
}
$('#btn-ds-create').addEventListener('click', async () => {
  const name = $('#ds-name').value.trim();
  if (!/^[A-Za-z0-9_-]+$/.test(name)) { toast('データセット名は英数字・_・- で指定してください', true); return; }
  if (!state.selected.size) { toast('シーンを選択してください', true); return; }
  try {
    await api('POST', '/api/datasets', { name, scenes: [...state.selected], val_ratio: parseFloat($('#ds-val').value),
                                         min_duration: parseFloat($('#ds-min').value) });
    toast('作成を開始しました'); loadJobs();
  } catch (e) { toast(e.message, true); }
});
let jobTimer = null;
async function loadJobs() {
  let list = [];
  try { list = await api('GET', '/api/jobs'); } catch (e) { return; }
  $('#jobs').innerHTML = list.slice(0, 8).map((j) => `
    <div class="job"><b>${esc(j.title)}</b> <span class="muted">${j.state === 'running' ? '実行中' : j.state === 'done' ? '完了' : 'エラー'}</span>
      <div class="progress blue"><div class="bar" style="width:${Math.round(j.progress * 100)}%"></div></div>
      <div class="muted">${esc(j.message)}</div>${j.error ? `<div class="err">${esc(j.error)}</div>` : ''}
      ${j.result && j.result.url ? `<div><a href="${esc(j.result.url)}" target="_blank" rel="noopener">${esc(j.result.url)}</a></div>` : ''}</div>`).join('')
    || '<div class="muted small">ありません</div>';
  const running = list.some((j) => j.state === 'running');
  if (running) { clearTimeout(jobTimer); jobTimer = setTimeout(loadJobs, 800); }
  if (state._running && !running) loadDatasets();
  state._running = running;
}
// ------------------------------------------------------------ Hugging Face
state.hf = null;
async function loadHf(refresh) {
  try { state.hf = await api('GET', '/api/hf/status' + (refresh ? '?refresh=true' : '')); } catch (e) { state.hf = null; }
  const h = state.hf;
  $('#hf-status').innerHTML = !h ? '' : h.logged_in
    ? `🤗 Hugging Face: <b>${esc(h.user)}</b> でログイン中${h.orgs && h.orgs.length ? `(組織: ${h.orgs.map(esc).join(', ')})` : ''}`
    : '🤗 Hugging Face: <span class="warn-text">未ログイン</span> — Jetsonのターミナルで <code>/home/jetson/miniconda/envs/jetracer/bin/hf auth login</code> を実行し、トークン(Write権限)を入れてください <button class="small" id="btn-hf-refresh">確認</button>';
}
document.addEventListener('click', (e) => { if (e.target.id === 'btn-hf-refresh') loadHf(true).then(loadDatasets); });
function hfRow(d) {
  const h = state.hf || {};
  const owner = h.user || 'ユーザー名';
  const done = d.hf ? `<a href="${esc(d.hf.url)}" target="_blank" rel="noopener">${esc(d.hf.repo_id)}</a>
      <span class="badge ${d.hf.private ? 'tag' : 'ok'}">${d.hf.private ? '非公開' : '公開'}</span>
      <span class="muted small">${esc(d.hf.uploaded_at)}</span>` : '<span class="muted small">まだ上げていません</span>';
  return `<div class="hf-row"><span>🤗</span> ${done} <span class="spacer"></span>
      <button class="small" data-hf-open="${esc(d.name)}" ${h.logged_in ? '' : 'disabled title="Hugging Face にログインしてください"'}>${d.hf ? 'もう一度上げる' : 'Hugging Face へ'}</button></div>
    <div class="hf-form hidden" data-hf-form="${esc(d.name)}">
      <label>リポジトリ(ユーザー名または組織/名前)<input data-hf-repo value="${esc(d.hf ? d.hf.repo_id : owner + '/jetformer-' + d.name)}"></label>
      <div class="row tight">
        <label class="inline"><input type="checkbox" data-hf-public ${d.hf && !d.hf.private ? 'checked' : ''}> 公開する(誰でも Dataset Viewer で見られる)</label>
        <label class="inline"><input type="checkbox" data-hf-nus checked> 学習用のnuScenes形式(tar)も上げる</label>
      </div>
      <label>ライセンス<select data-hf-license>
        ${['other', 'cc-by-4.0', 'cc-by-sa-4.0', 'cc-by-nc-4.0', 'cc-by-nc-sa-4.0', 'mit', 'apache-2.0'].map((l) => `<option ${d.hf && d.hf.license === l ? 'selected' : ''}>${l}</option>`).join('')}
      </select></label>
      <div class="muted small">非公開のデータセットは、Hugging Face の PRO / Enterprise でないと Dataset Viewer で表示されません。
        カメラ画像が含まれるので、公開する前に写っているものを確認してください。</div>
      <div class="row buttons"><button class="primary small" data-hf-upload="${esc(d.name)}">アップロード</button></div>
    </div>`;
}
async function uploadHf(name, form) {
  const repo = form.querySelector('[data-hf-repo]').value.trim();
  const isPublic = form.querySelector('[data-hf-public]').checked;
  const body = { repo_id: repo, private: !isPublic, include_nuscenes: form.querySelector('[data-hf-nus]').checked,
                 license: form.querySelector('[data-hf-license]').value };
  if (!confirm(`データセット ${name} を Hugging Face の ${repo} に${isPublic ? '【公開】' : '非公開'}でアップロードします。`
               + (isPublic ? '\n公開すると誰でも見られます(カメラ画像を含みます)。' : '') + '\nよろしいですか?')) return;
  try { await api('POST', `/api/datasets/${encodeURIComponent(name)}/hf`, body); toast('アップロードを始めました(処理状況に進み具合が出ます)'); loadJobs(); }
  catch (e) { toast(e.message, true); }
}

async function loadDatasets() {
  let list = [];
  try { list = await api('GET', '/api/datasets'); } catch (e) { toast(e.message, true); return; }
  if (state.hf === null) await loadHf(false);
  $('#dataset-list').innerHTML = list.slice().reverse().map((d) => `
    <div class="ds">
      <h3>${esc(d.name)} <span class="muted small">${esc(d.version)} ・ ${esc(d.created_at)}</span></h3>
      <div class="stats">
        <span>シーン <b>${d.scenes}</b> (学習 ${d.train_scenes} / 検証 ${d.val_scenes})</span>
        <span>フレーム <b>${d.frames}</b></span><span>ToF <b>${d.tof_frames}</b></span>
        <span>時間 <b>${(d.hours * 60).toFixed(1)}</b> 分</span>
        <span>車両データ <b>${esc(d.vehicle_quality.join(', '))}</b></span>
      </div>
      <div class="stats">${Object.entries(d.mode_frames || {}).map(([id, n]) => {
        const m = (d.modes || []).find((x) => x.id === +id);
        return `<span><span class="badge mode" style="background:${modeColor(+id)}">${id}: ${esc(m ? m.name : 'ID ' + id)}</span> ${n} フレーム</span>`;
      }).join('')}</div>
      <div class="row">
        ${d.export ? `<a href="/api/exports/${encodeURIComponent(d.export.file)}" download><button class="primary small">⬇ ダウンロード (${(d.export.bytes / 1e6).toFixed(1)} MB)</button></a>
                      <span class="muted small">data/exports/${esc(d.export.file)}(.sha256付き)</span>`
                   : `<button class="small" data-export="${esc(d.name)}">エクスポート(tar作成)</button>`}
        <span class="spacer"></span>
        <button class="small danger" data-del="${esc(d.name)}">削除</button>
      </div>
      ${hfRow(d)}
    </div>`).join('') || '<div class="muted">まだありません</div>';
}
$('#dataset-list').addEventListener('click', async (e) => {
  const open = e.target.closest('[data-hf-open]'), up = e.target.closest('[data-hf-upload]');
  if (open) { document.querySelector(`[data-hf-form="${CSS.escape(open.dataset.hfOpen)}"]`).classList.toggle('hidden'); return; }
  if (up) { uploadHf(up.dataset.hfUpload, up.closest('.hf-form')); return; }
  const exp = e.target.closest('[data-export]'), del = e.target.closest('[data-del]');
  if (exp) {
    try { await api('POST', `/api/datasets/${encodeURIComponent(exp.dataset.export)}/export`); toast('エクスポートを開始しました'); loadJobs(); }
    catch (err) { toast(err.message, true); }
  }
  if (del) {
    if (!confirm(`データセット ${del.dataset.del} を削除します(収集した元シーンは残ります)。`)) return;
    try { await api('DELETE', `/api/datasets/${encodeURIComponent(del.dataset.del)}`); loadDatasets(); }
    catch (err) { toast(err.message, true); }
  }
});


// ------------------------------------------------------------ 学習
const TRAIN_RATE = 200, VAL_RATE = 650;     // Orin Nano実測: 学習 約200枚/秒、検証 約650枚/秒
const fmtDur = (s) => (s === null || s === undefined) ? '-' : s >= 3600 ? `${Math.floor(s / 3600)}時間${Math.round(s % 3600 / 60)}分`
  : s >= 60 ? `${Math.floor(s / 60)}分${Math.round(s % 60)}秒` : `${Math.round(s)}秒`;
state.datasets = [];
async function loadTrain() {
  try { state.datasets = await api('GET', '/api/datasets'); } catch (e) { return; }
  const sel = $('#tr-dataset'), cur = sel.value;
  sel.innerHTML = state.datasets.slice().reverse().map((d) => `<option value="${esc(d.name)}">${esc(d.name)}(${d.scenes}シーン / ${d.frames}フレーム)</option>`).join('')
    || '<option value="">データセットがありません</option>';
  if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  updateEstimate(); loadRuns(); pollTrain();
}
function updateEstimate() {
  const d = state.datasets.find((x) => x.name === $('#tr-dataset').value);
  if (!d) { $('#tr-estimate').textContent = 'データセットを作成してください(データセットタブ)'; $('#tr-ds-info').innerHTML = ''; return; }
  const mf = d.mode_frames || {};
  $('#tr-ds-info').innerHTML = Object.entries(mf).map(([id, n]) =>
    `<span class="badge mode" style="background:${modeColor(+id)}">${id}: ${esc(((d.modes || []).find((m) => m.id === +id) || {}).name || id)}</span> ${n}`).join(' ')
    + (mf['0'] ? '<div class="warn-text">「未設定」のフレームは学習に使いません</div>' : '');
  const stride = +$('#tr-stride').value, epochs = +$('#tr-epochs').value || 1;
  const usable = d.frames - (mf['0'] || 0);
  const trFrac = d.scenes ? d.train_scenes / d.scenes : 1;
  const nTrain = Math.round(usable * trFrac / stride), nVal = Math.round(usable * (1 - trFrac) / stride);
  const perEpoch = nTrain / TRAIN_RATE + nVal / VAL_RATE + 3;
  const total = perEpoch * epochs + 30 + ($('#tr-engine').checked ? 70 : 0);
  $('#tr-estimate').innerHTML = `学習 最大 <b>${nTrain}</b> 枚 / 検証 最大 <b>${nVal}</b> 枚<br>1エポック 約 <b>${fmtDur(perEpoch)}</b> → 合計 約 <b>${fmtDur(total)}</b>`
    + '<div class="muted small">操作値が無効・AI走行中のフレームは除外されるため、実際はこれより少し短くなります</div>';
}
['#tr-dataset', '#tr-stride', '#tr-epochs', '#tr-engine'].forEach((s) => $(s).addEventListener('change', updateEstimate));
$('#tr-epochs').addEventListener('input', updateEstimate);
$('#btn-tr-start').addEventListener('click', async () => {
  const body = { dataset: $('#tr-dataset').value, stride: +$('#tr-stride').value, epochs: +$('#tr-epochs').value,
                 batch: +$('#tr-batch').value, lr: +$('#tr-lr').value, pretrained: $('#tr-pretrained').checked,
                 build_engine: $('#tr-engine').checked, command_input: !$('#tr-nocmd').checked };
  if (!body.dataset) { toast('データセットを選んでください', true); return; }
  try { const r = await api('POST', '/api/train/start', body); toast(`学習を開始しました (${r.run})`); pollTrain(); }
  catch (e) { toast(e.message, true); }
});
$('#btn-tr-stop').addEventListener('click', async () => {
  if (!confirm('学習を停止します。ここまでで最良のモデルを保存して終了します。')) return;
  try { await api('POST', '/api/train/stop'); toast('停止を指示しました'); } catch (e) { toast(e.message, true); }
});
let trainTimer = null, lastRunState = null;
async function pollTrain() {
  clearTimeout(trainTimer); trainTimer = null;
  let st;
  try { st = await api('GET', '/api/train/status'); }
  catch (e) { if (state.tab === 'train' || lastRunState) trainTimer = setTimeout(pollTrain, 2000); return; }
  renderTrain(st);
  if (st.running || state.tab === 'train') trainTimer = setTimeout(pollTrain, st.running ? 1000 : 4000);
  const key = st.name ? `${st.name}:${st.phase}` : null;
  if (key !== lastRunState) { lastRunState = key; loadRuns(); }
}
function renderTrain(st) {
  $('#btn-tr-stop').disabled = !(st.running && st.phase === 'training');
  $('#btn-tr-start').disabled = !!st.running;
  if (!st.name) { $('#tr-current').textContent = '実行中の学習はありません'; return; }
  const info = st.info || {};
  const names = Object.fromEntries((info.modes || []).map((m) => [m.id, m.name]));
  $('#tr-current').innerHTML = `<b>${esc(st.name)}</b> ・ ${esc(st.message || '')}`
    + (info.train ? `<div class="muted">学習 ${info.train} 枚 / 検証 ${info.val} 枚 / 間引き 1/${info.stride}</div>` : '')
    + (st.error ? `<div class="warn-text">${esc(st.error)}</div>` : '');
  const ep = st.epochs || 1, stepsEp = (info.steps_train || 0) + (info.steps_val || 0);
  const inEpoch = stepsEp ? ((st.step_phase === 'val' ? info.steps_train : 0) + st.step) / stepsEp : 0;
  const frac = st.phase === 'done' ? 1 : Math.min(1, ((Math.max(st.epoch, 1) - 1) + inEpoch) / ep);
  $('#tr-progress').style.width = `${Math.round(frac * 100)}%`;
  $('#tr-progress-text').textContent = st.phase === 'training'
    ? `エポック ${st.epoch} / ${ep}(${st.step_phase === 'val' ? '検証' : '学習'} ${st.step}/${st.steps})・経過 ${fmtDur(st.elapsed_s)}・残り 約 ${fmtDur(st.eta_s)}`
    : `経過 ${fmtDur(st.elapsed_s)}`;
  const h = st.history || [];
  drawChart($('#tr-chart'), [
    { x: h.map((r) => r.epoch), y: h.map((r) => r.train_us), color: COLORS.steering, label: '学習誤差 [µs]' },
    { x: h.map((r) => r.epoch), y: h.map((r) => r.val_us), color: COLORS.throttle, label: '検証誤差 [µs]' },
  ], { minSpan: 10, xLabel: (v) => `エポック ${Math.round(v)}` });
  const last = h[h.length - 1];
  $('#tr-modes').innerHTML = last ? '<div class="muted">ジャンル別の検証誤差(最新エポック)</div><div class="mode-loss">' +
    Object.entries(last.val_per_mode_us || {}).map(([id, v]) =>
      `<span><span class="badge mode" style="background:${modeColor(+id)}">${id}: ${esc(names[id] || modeName(+id))}</span> ${v} µs</span>`).join('') + '</div>' : '';
  $('#tr-log').textContent = (st.log || []).join('\n');
}
async function loadRuns() {
  let runs = [];
  try { runs = await api('GET', '/api/runs'); } catch (e) { return; }
  const label = { done: ['ok', '完了'], stopped: ['warn', '途中停止'], stopped_no_model: ['warn', 'モデルなし'], running: ['', '実行中'], error: ['warn', 'エラー'] };
  $('#run-list').innerHTML = runs.map((r) => {
    const [cls, txt] = label[r.state] || ['warn', r.state];
    const names = Object.fromEntries((r.modes || []).map((m) => [m.id, m.name]));
    const files = (r.files || []).filter((f) => f !== 'run.json').map((f) =>
      `<a href="/api/runs/${encodeURIComponent(r.name)}/files/${f}" download><button class="small">⬇ ${f}</button></a>`).join(' ');
    return `<div class="ds">
      <h3>${esc(r.name)} <span class="badge ${cls}">${txt}</span></h3>
      <div class="stats">
        <span>データセット <b>${esc(r.params.dataset)}</b></span><span>間引き <b>1/${r.params.stride}</b></span>
        <span>エポック <b>${r.epochs_done ?? '-'}</b> / ${r.params.epochs}</span>
        <span>最良の検証誤差 <b>${r.best_val_us ?? '-'}</b> µs${r.best_epoch ? `(${r.best_epoch}エポック目)` : ''}</span>
        <span>時間 <b>${fmtDur(r.duration_s)}</b></span><span>学習 ${r.train_samples ?? '-'} 枚</span>
        ${r.params.command_input === false ? '<span class="badge tag">今の操作値なし</span>' : ''}
      </div>
      <div class="mode-loss">${Object.entries(r.best_val_per_mode_us || {}).map(([id, v]) =>
        `<span><span class="badge mode" style="background:${modeColor(+id)}">${id}: ${esc(names[id] || modeName(+id))}</span> ${v} µs</span>`).join('')}</div>
      <div class="row">${files}<span class="spacer"></span>
        ${r.state === 'running' ? '' : `<button class="small danger" data-del-run="${esc(r.name)}">削除</button>`}</div>
    </div>`;
  }).join('') || '<div class="muted">まだありません</div>';
}
$('#run-list').addEventListener('click', async (e) => {
  const b = e.target.closest('[data-del-run]');
  if (!b || !confirm(`学習結果 ${b.dataset.delRun} を削除します。`)) return;
  try { await api('DELETE', `/api/runs/${encodeURIComponent(b.dataset.delRun)}`); loadRuns(); } catch (err) { toast(err.message, true); }
});


// ------------------------------------------------------------ AI走行
state.drive = { models: [], st: null, paramsLoaded: false };
const DR_PARAMS = [
  ['#dr-steer-gain', 'steer_gain', (v) => (+v).toFixed(2)],
  ['#dr-throttle-gain', 'throttle_gain', (v) => (+v).toFixed(2)],
  ['#dr-trim', 'steer_trim_us', (v) => `${v > 0 ? '+' : ''}${(+v).toFixed(0)} µs`],
  ['#dr-look', 'lookahead', (v) => `${v}(約${Math.round(v * 33)}ms先)`],
  ['#dr-osc', 'osc_percent', (v) => `${v > 0 ? '+' : ''}${(+v).toFixed(1)} %`],
];
const drModel = () => state.drive.models.find((m) => m.name === $('#dr-model').value);
async function loadDrive() {
  try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) { toast(e.message, true); return; }
  if (!state.drive.st) { try { state.drive.st = await api('GET', '/api/drive/status'); } catch (e) { /* 後で */ } }
  const sel = $('#dr-model'), cur = sel.value;
  sel.innerHTML = state.drive.models.map((m) =>
    `<option value="${esc(m.name)}">${esc(m.name)}${m.best_val_us != null ? `(検証誤差 ${m.best_val_us}µs)` : ''}</option>`).join('')
    || '<option value="">走行できるモデルがありません(学習タブで「TensorRTエンジンを作る」)</option>';
  const st = state.drive.st;
  const last = !state.drive.restored && st && st.last ? st.last.run : null;     // 初回は前回使ったモデル
  for (const want of [st && st.running && st.run, last, cur]) {
    if (want && [...sel.options].some((o) => o.value === want)) { sel.value = want; break; }
  }
  renderDriveModel();
  state.drive.restored = true;
  pollDrive();
}
function renderDriveModel() {
  const m = drModel();
  const modeSel = $('#dr-mode'), cur = modeSel.value;
  if (!m) { $('#dr-model-info').textContent = ''; modeSel.innerHTML = ''; return; }
  const names = Object.fromEntries((m.modes || []).map((x) => [x.id, x.name]));
  $('#dr-model-info').innerHTML = `データセット ${esc(m.dataset)} ・ 間引き 1/${m.stride ?? '-'} ・ ${esc(m.created_at || '')}`
    + `<div class="mode-loss">${Object.entries(m.best_val_per_mode_us || {}).map(([id, v]) =>
      `<span><span class="badge mode" style="background:${modeColor(+id)}">${id}: ${esc(names[id] || modeName(+id))}</span> ${v} µs</span>`).join('')}</div>`;
  modeSel.innerHTML = (m.modes || []).filter((x) => x.id !== 0).map((x) => {
    const ok = m.trained_modes.includes(x.id);
    return `<option value="${x.id}" ${ok ? '' : 'disabled'}>${x.id}: ${esc(x.name)}${ok ? '' : '(学習データなし)'}</option>`;
  }).join('');
  const st = state.drive.st;
  const running = st && st.running ? String(st.mode_id) : '';
  const last = !state.drive.restored && st && st.last && st.last.mode_id != null ? String(st.last.mode_id) : '';
  for (const v of [running, last, cur]) {
    if (v && [...modeSel.options].some((o) => o.value === v && !o.disabled)) { modeSel.value = v; return; }
  }
  const first = [...modeSel.options].find((o) => !o.disabled);
  if (first) modeSel.value = first.value;
}
$('#dr-model').addEventListener('change', renderDriveModel);
$('#dr-record').addEventListener('change', () => $('#dr-record-opts').classList.toggle('hidden', !$('#dr-record').checked));
// 補助のスイッチ(チェックボックス)。変えたらすぐ送る(走行中も反映)
const DR_FLAGS = [['#dr-esc-reverse', 'esc_reverse'], ['#dr-recovery', 'recovery']];
for (const [sel, key] of DR_FLAGS) {
  $(sel).addEventListener('change', async () => {
    try { await api('POST', '/api/drive/params', { [key]: $(sel).checked ? 1 : 0 }); } catch (e) { toast(e.message, true); }
  });
}
function readDriveParams() {
  const p = {};
  for (const [sel, key] of DR_PARAMS) p[key] = +$(sel).value;
  for (const [sel, key] of DR_FLAGS) p[key] = $(sel).checked ? 1 : 0;
  return p;
}
// スライダーは動かした項目だけを送る(別の画面の値を巻き戻さない)。操作中・送信中でなければサーバーの値に合わせる
const drBusy = {}, drTimers = {};
for (const [sel, key, fmt] of DR_PARAMS) {
  const el = $(sel);
  el.addEventListener('pointerdown', () => { drBusy[key] = (drBusy[key] || 0) | 1; });
  for (const ev of ['pointerup', 'pointercancel', 'change']) el.addEventListener(ev, () => { drBusy[key] = (drBusy[key] || 0) & ~1; });
  el.addEventListener('input', () => {
    $(sel + '-v').textContent = fmt(el.value);
    drBusy[key] = (drBusy[key] || 0) | 2;
    clearTimeout(drTimers[key]);
    drTimers[key] = setTimeout(async () => {
      try { await api('POST', '/api/drive/params', { [key]: +el.value }); } catch (e) { toast(e.message, true); }
      drBusy[key] &= ~2;
    }, 120);
  });
}
function setDriveParams(p) {
  for (const [sel, key] of DR_FLAGS) if (p[key] !== undefined) $(sel).checked = !!+p[key];
  for (const [sel, key, fmt] of DR_PARAMS) {
    if (p[key] === undefined || drBusy[key]) continue;
    if (+$(sel).value !== +p[key]) $(sel).value = p[key];
    $(sel + '-v').textContent = fmt(p[key]);
  }
}
$('#dr-mode').addEventListener('change', async () => {
  const st = state.drive.st;
  if (!(st && st.running)) return;
  try { await api('POST', '/api/drive/params', { mode_id: +$('#dr-mode').value }); toast(`ジャンルを「${modeName(+$('#dr-mode').value)}」に切り替えました`); }
  catch (e) { toast(e.message, true); }
});
$('#btn-dr-start').addEventListener('click', async () => {
  const m = drModel();
  if (!m) { toast('モデルを選んでください', true); return; }
  const dry = $('#dr-dry').checked;
  if (!dry && !confirm(`AI走行を開始します。\nモデル: ${m.name}\nジャンル: ${modeName(+$('#dr-mode').value)} / 速度ゲイン ${(+$('#dr-throttle-gain').value).toFixed(2)}\n\nプロポのCH3をAI側(LED紫)にすると車が動きます。周りに人や物がないことを確認してください。\nこの画面は開いたままにしてください(閉じる・スリープ・通信断が5秒続くと止まります)。`)) return;
  const body = { client: 4, run: m.name, mode_id: +$('#dr-mode').value, dry_run: dry, record: $('#dr-record').checked,
                 description: $('#dr-desc').value || `AI走行 ${m.name}`, tags: parseTags($('#dr-tags').value), params: readDriveParams() };
  try {
    const r = await api('POST', '/api/drive/start', body);
    setDriveToken(r.token);
    toast(dry ? '推論の確認を開始しました(出力は中立)' : '走行を開始しました'); pollDrive();
  } catch (e) { toast(e.message, true); }
});
async function stopDrive(reason) {
  try { await api('POST', '/api/drive/stop', { reason }); toast('停止しました'); } catch (e) { toast(e.message, true); }
  pollDrive();
}
$('#btn-dr-stop').addEventListener('click', () => stopDrive('停止ボタン'));
$('#btn-float-stop').addEventListener('click', () => stopDrive('停止ボタン'));
// 停止キー: Escape はいつでも、スペースは文字入力中以外ならどこでも(ジャンルの選択直後でも)止める。
// シーン詳細を開いているときの Escape は詳細を閉じるだけ(捕捉フェーズで先に判定する)
document.addEventListener('keydown', (e) => {
  const st = state.drive.st;
  if (!(st && st.running) || e.repeat) return;
  if (e.key === 'Escape' && state.modal) return;
  if (e.key !== 'Escape' && e.target.matches('textarea, input:not([type=range]):not([type=checkbox]):not([type=radio]):not([type=button])')) return;
  if (e.code === 'Space' || e.key === 'Escape') { e.preventDefault(); stopDrive('キー操作'); }
}, true);
// 生存確認: 走行を始めたページだけが token を持ち、画面が表示されている間だけ送る
// (閉じた・スリープした・通信が切れて5秒たつと止まる。ほかのページは表示だけ)
function setDriveToken(t) {
  state.drive.token = t || null;
  try { t ? sessionStorage.setItem('jf-drive-token', t) : sessionStorage.removeItem('jf-drive-token'); } catch (e) { /* 無視 */ }
}
try { state.drive.token = sessionStorage.getItem('jf-drive-token'); } catch (e) { state.drive.token = null; }
let driveTimer = null;
async function pollDrive() {
  clearTimeout(driveTimer); driveTimer = null;
  let st;
  // 生存確認: 走行を始めたこの画面から token を送り続ける(タブが裏に回っても、ブラウザが動かしている間は送る)
  const tok = state.drive.token ? `?token=${encodeURIComponent(state.drive.token)}` : '';
  try { st = await api('GET', '/api/drive/status' + tok); } catch (e) { st = null; }
  if (st && !st.running && state.drive.token) setDriveToken(null);
  if (st) renderDrive(st);
  const running = st ? st.running : (state.drive.st && state.drive.st.running);
  if (running || state.tab === 'drive') driveTimer = setTimeout(pollDrive, running ? 250 : 1000);
}
const PHASE = { idle: ['', '停止中'], starting: ['warn', '準備中'], driving: ['ok', '走行中'], error: ['warn', 'エラー'] };
function gaugePos(us) { return `${Math.max(0, Math.min(100, (us - 1000) / 10))}%`; }
function renderDrive(st) {
  const prev = state.drive.st;
  state.drive.st = st;
  if (st.params) setDriveParams(st.params);
  if (prev && prev.running && !st.running && state.tab === 'scenes') loadScenes();
  const [cls, txt] = PHASE[st.phase] || ['', st.phase];
  const ph = $('#dr-phase');
  ph.className = 'badge ' + cls;
  ph.textContent = st.phase === 'driving' && st.dry_run ? '推論の確認中' : txt;
  $('#dr-message').innerHTML = esc(st.message || '') + (st.error ? `<div class="warn-text">${esc(st.error)}</div>` : '')
    + (st.recording_note ? `<div class="muted">${esc(st.recording_note)}</div>` : '')
    + (st.running && st.recording ? '<div><span class="badge warn">● 収集中</span></div>' : '')
    + (st.running && !st.controller ? '<div class="warn-text">別の画面(またはこの画面の以前の表示)が走行を操作しています。この画面を閉じても走行は止まりません。止めるときは停止ボタンを押してください。</div>' : '');
  const now = st.running ? st.now : null;
  const who = $('#dr-who');
  if (now) {
    if (now.hold) who.textContent = now.select ? 'CH3: AI側 ・ 中立で停止中' : 'CH3: 人側 ・ 中立で待機中';
    else if (now.assist === 'リカバリー中止') who.textContent = 'CH3: AI側 ・ リカバリーをやめて中立(CH3を人側へ戻して助けてください)';
    else if (now.assist === 'リカバリー') who.textContent = 'CH3: AI側 ・ ぶつかったので自動で下がっています';
    else who.textContent = now.select ? (st.dry_run ? 'CH3: AI側(出力は中立)' : 'CH3: AI側 ・ 車はAIが操作中') : 'CH3: 人側 ・ プロポで操作中';
    who.className = 'who ' + (now.hold ? 'hold' : now.select && !st.dry_run ? 'ai' : 'human');
  } else { who.textContent = ''; who.className = 'who'; }
  const ref = st.ref || {};
  const setG = (id, us) => { const el = $(id); if (us === null || us === undefined) el.style.display = 'none'; else { el.style.display = ''; el.style.left = gaugePos(us); } };
  setG('#dr-g-steer-ref', ref.steer_center_us); setG('#dr-g-thr-ref', ref.throttle_neutral_us);
  setG('#dr-g-steer-raw', now && now.raw ? now.raw[0] : null); setG('#dr-g-thr-raw', now && now.raw ? now.raw[1] : null);
  setG('#dr-g-steer-out', now ? now.out[0] : null); setG('#dr-g-thr-out', now ? now.out[1] : null);
  $('#dr-out-key').textContent = st.dry_run && st.running ? '出力(ゲイン後の予定値。確認中は車へは中立)' : '出力(ゲイン後)';
  $('#dr-v-steer').textContent = now ? `${now.raw ? Math.round(now.raw[0]) : '-'} → ${Math.round(now.out[0])}` : '';
  $('#dr-v-thr').textContent = now ? `${now.raw ? Math.round(now.raw[1]) : '-'} → ${Math.round(now.out[1])}` : '';
  $('#dr-ref').textContent = st.ref ? `ゲインの中心: ステアリング ${st.ref.steer_center_us} µs / スロットル中立 ${st.ref.throttle_neutral_us} µs(学習データで停止中の操作値)。`
    + `出力: PCA9685 0x${st.pca9685.address.toString(16)} CH${st.pca9685.steering_ch}=ステアリング CH${st.pca9685.throttle_ch}=スロットル ${st.pca9685.freq_hz}Hz` : '';
  const s = st.stats || {};
  const rows = [];
  if (st.running || s.frames) {
    rows.push(['処理速度', `${s.fps ?? '-'} fps(カメラ30fpsに対して)`],
              ['遅延(撮像→出力)', `平均 ${s.latency_ms ?? '-'} ms / 最大 ${s.latency_ms_max ?? '-'} ms`],
              ['推論(TensorRT)', `${s.infer_ms ?? '-'} ms`],
              ['処理したフレーム', `${s.frames ?? 0}(取りこぼし ${s.skipped ?? 0})`],
              ['中立で止めた回数', `${s.holds ?? 0}${st.hold_reason ? ` ・ <span class="warn-text">${esc(st.hold_reason)}</span>` : ''}`],
              ['補助', `リカバリー ${s.recoveries ?? 0}回 ・ 後退の受け付け ${s.esc_assists ?? 0}回`
                + (s.giveups ? ` ・ <span class="warn-text">中止 ${s.giveups}回</span>` : '')
                + (now && now.assist ? ` ・ <b class="warn-text">${esc(now.assist)}中</b>` : '')],
              ['走行時間', fmtDur(st.elapsed_s)]);
    const L = st.laps || {};
    if (L.laps !== undefined) {
      rows.push(['周回(ジャイロ)', `${L.laps}周${L.direction ? `(${L.direction})` : ''} ・ 今の周 ${Math.min(359, L.progress_deg || 0)}°`],
                ['ベスト(AIだけの周)', L.best_ai_s != null ? `<b>${L.best_ai_s.toFixed(2)} 秒</b> ・ 中央値 ${L.median_ai_s.toFixed(2)} 秒(${L.clean_laps}周)` : '-'],
                ['最近の周', (L.recent || []).slice().reverse().map((x) =>
                  `<span class="lap ${x.best ? 'best' : ''} ${x.ai_only && !x.recoveries ? '' : 'muted'}">${x.time_s.toFixed(1)}${x.first ? '(発進)' : ''}${x.ai_only ? '' : '(人)'}${x.recoveries ? '(R)' : ''}</span>`).join(' ') || '-']);
    }
    if (now && now.erpm !== undefined) rows.push(['モーター', `${now.erpm} eRPM`]);
    if (now && now.human) rows.push(['プロポの値', `${now.human[0]} / ${now.human[1]} µs`]);
  }
  $('#dr-stats').innerHTML = rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join('');
  const h = st.history || { t: [] };
  drawChart($('#dr-chart-steer'), [
    { x: h.t, y: h.raw_steer || [], color: COLORS.aux, label: '予測' },
    { x: h.t, y: h.out_steer || [], color: COLORS.steering, label: '出力' },
  ], { xMin: -10, xMax: 0, minSpan: 100 });
  drawChart($('#dr-chart-thr'), [
    { x: h.t, y: h.raw_throttle || [], color: COLORS.aux, label: '予測' },
    { x: h.t, y: h.out_throttle || [], color: COLORS.throttle, label: '出力' },
  ], { xMin: -10, xMax: 0, minSpan: 100 });
  const training = state.status && state.status.training;
  $('#btn-dr-start').disabled = st.running || !drModel() || !!training;
  $('#btn-dr-start').title = training ? '学習中は走行できません' : '';
  $('#btn-dr-stop').disabled = !st.running;
  $('#btn-float-stop').classList.toggle('hidden', !st.running);
  document.body.classList.toggle('driving', !!st.running);
  $('#dr-model').disabled = st.running;
  $('#dr-dry').disabled = st.running; $('#dr-record').disabled = st.running;
}

// ------------------------------------------------------------ 起動
loadModes();
loadSound();
showTab('live');
pollStatus();
pollValues();
loadScenes();

// デバッグ・自動テスト用(ブラウザのコンソールから操作できるように)
window.jetformer = { state, showTab, loadSound, openScene, showFrame, closeModal, loadScenes, loadDatasets, loadJobs, loadModes, loadTrain, pollTrain,
                     loadDrive, pollDrive, stopDrive };
