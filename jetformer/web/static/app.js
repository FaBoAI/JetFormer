'use strict';

const $ = (s) => document.querySelector(s);
const state = { tab: 'live', scenes: [], selected: new Set(), status: null, modal: null, playing: false };
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
  ctx.fillText(`${x0.toFixed(1)}s`, pad.l, h - 4);
  ctx.fillText(`${x1.toFixed(1)}s`, w - pad.r - 30, h - 4);
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
  const video = $('#live-video');
  if (name === 'live') { if (!video.src.includes('live.mjpg')) video.src = '/api/live.mjpg?fps=15&_=' + Date.now(); }
  else video.removeAttribute('src');                      // 見ていない時は映像を止めて帯域を節約
  if (name === 'scenes') loadScenes();
  if (name === 'datasets') { loadDatasets(); loadJobs(); updateDsSelected(); }
}
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

// ------------------------------------------------------------ 状態(1秒ごと)
const SENSOR_LABEL = { CAM_FRONT: 'カメラ', TOF_FRONT: 'ToF', IMU: 'IMU', VEHICLE: '車両(PWM)' };
const EXPECTED_HZ = { CAM_FRONT: 30, TOF_FRONT: 30, IMU: 100, VEHICLE: 50 };

function sensorExtra(name, s) {
  const parts = [];
  if (name === 'CAM_FRONT' && s.dropped !== undefined) parts.push(`破棄 ${s.dropped}`, s.exposure_us ? `露出 ${(s.exposure_us / 1000).toFixed(1)}ms` : '');
  if (name === 'TOF_FRONT') { parts.push(`破損 ${s.bad_frames}`, `欠落 ${s.skipped}`); }
  if (name === 'IMU') parts.push(s.mag ? '地磁気あり' : '地磁気なし');
  if (name === 'VEHICLE') {
    parts.push(`方式 ${s.protocol}`);
    if (s.protocol === 'legacy') parts.push('<span class="warn-text">値の取り違えあり</span>');
  }
  return parts.filter(Boolean).join(' / ');
}

function renderStatus(st) {
  state.status = st;
  const rec = st.recorder;
  const ind = $('#rec-indicator');
  ind.textContent = rec.recording ? `● 収集中 ${rec.log}` : '待機中';
  ind.className = rec.recording ? 'rec-on' : 'rec-off';
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
  if (rec.recording) {
    recState.textContent = `収集中: シーン ${rec.scene_index}${rec.max_scenes ? ' / ' + rec.max_scenes : ''}`;
    recState.className = 'rec-state on';
    $('#scene-progress').style.width = `${Math.min(100, rec.scene_elapsed_s / rec.scene_seconds * 100)}%`;
    const c = rec.current_counts || {};
    $('#rec-detail').textContent = `${rec.scene_elapsed_s.toFixed(1)} / ${rec.scene_seconds} 秒  ` +
      `(カメラ ${c.CAM_FRONT || 0} / ToF ${c.TOF_FRONT || 0} / IMU ${c.IMU || 0} / 車両 ${c.VEHICLE || 0})` +
      (rec.dropped_writes ? `  保存遅れ ${rec.dropped_writes}` : '');
  } else {
    recState.textContent = '待機中'; recState.className = 'rec-state';
    $('#scene-progress').style.width = '0';
    $('#rec-detail').textContent = rec.last_error ? '⚠ ' + rec.last_error : `1シーン = ${st.config.scene_seconds} 秒`;
  }
  $('#btn-rec-start').disabled = rec.recording;
  $('#btn-rec-stop').disabled = !rec.recording;
  $('#btn-sensors-stop').disabled = rec.recording || !st.sensors_running;
  $('#btn-sensors-start').disabled = st.sensors_running;
  const saved = rec.saved_scenes || [];
  $('#session-scenes').innerHTML = saved.length
    ? `<div class="muted">保存したシーン (${saved.length})</div>` + saved.slice().reverse().map((n) => `<div>✓ ${esc(n)}</div>`).join('')
    : '';
  if (saved.length !== state._savedCount) { state._savedCount = saved.length; if (state.tab === 'scenes') loadScenes(); }
}

async function pollStatus() {
  try { renderStatus(await api('GET', '/api/status')); } catch (e) { /* 再接続待ち */ }
  setTimeout(pollStatus, 1000);
}

// ------------------------------------------------------------ ライブ値(0.3秒ごと)
function renderValues(v) {
  const veh = v.vehicle || {}, imu = v.imu || {};
  drawChart($('#chart-pwm'), [
    { x: veh.t || [], y: (veh.steering || []).map((x) => x || null), color: COLORS.steering, label: 'ステア' },
    { x: veh.t || [], y: (veh.throttle || []).map((x) => x || null), color: COLORS.throttle, label: 'スロットル' },
  ], { xMin: -10, xMax: 0, minSpan: 100 });
  drawChart($('#chart-motor'), [{ x: veh.t || [], y: veh.motor_pct || [], color: COLORS.motor, label: 'モーター' }],
    { xMin: -10, xMax: 0, yMin: 0, yMax: 100 });
  drawChart($('#chart-accel'), [
    { x: imu.t || [], y: imu.accel_norm || [], color: COLORS.accel, label: '|a|' },
    { x: imu.t || [], y: imu.accel_x || [], color: COLORS.ax, label: 'x(前後)' },
    { x: imu.t || [], y: imu.accel_y || [], color: COLORS.ay, label: 'y(左右)' },
  ], { xMin: -10, xMax: 0, minSpan: 4 });
  drawChart($('#chart-gyro'), [{ x: imu.t || [], y: imu.gyro_z || [], color: COLORS.gyro, label: 'z' }],
    { xMin: -10, xMax: 0, minSpan: 20 });
  const rows = [];
  if (v.vehicle_now) {
    const p = v.vehicle_now.pwm;
    rows.push(['ステアリング', `${p[0]} µs`], ['スロットル', `${p[1]} µs`], ['補助', `${p[2]} µs`],
              ['モーター', `${v.vehicle_now.motor_pct} %`]);
  }
  if (v.tof) rows.push(['ToF 最小/中央', `${v.tof.min} / ${v.tof.center} ${state.status ? state.status.config.tof_unit : ''}`]);
  if (v.imu_now) {
    rows.push(['加速度', v.imu_now.accel.map((x) => x.toFixed(2)).join(', ') + ' m/s²'],
              ['角速度', v.imu_now.gyro.map((x) => x.toFixed(1)).join(', ') + ' °/s'], ['IMU温度', `${v.imu_now.temp} ℃`]);
  }
  $('#values').innerHTML = rows.length ? rows.map(([k, val]) => `<div class="k">${k}</div><div class="v">${val}</div>`).join('')
    : '<div class="muted">センサー停止中</div>';
}
async function pollValues() {
  if (state.tab === 'live' && document.visibilityState === 'visible') {
    try { renderValues(await api('GET', '/api/live/values')); } catch (e) { /* 無視 */ }
  }
  setTimeout(pollValues, 300);
}

// ------------------------------------------------------------ 収集の操作
$('#btn-rec-start').addEventListener('click', async () => {
  const mode = document.querySelector('input[name=rec-mode]:checked').value;
  const body = { description: $('#rec-desc').value, tags: parseTags($('#rec-tags').value), location: $('#rec-location').value,
                 max_scenes: mode === 'count' ? parseInt($('#rec-count').value, 10) || 1 : null };
  try { const r = await api('POST', '/api/record/start', body); toast(`収集を開始しました (${r.log})`); }
  catch (e) { toast('開始できません: ' + e.message, true); }
});
$('#btn-rec-stop').addEventListener('click', async () => {
  try { await api('POST', '/api/record/stop'); toast('停止しました。最後のシーンを保存しています…'); }
  catch (e) { toast(e.message, true); }
});
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
    if (!q) return true;
    return [s.name, s.description, (s.tags || []).join(' ')].join(' ').toLowerCase().includes(q);
  }).slice().reverse();
}
function sceneBadges(s) {
  const b = [];
  b.push(s.complete ? '<span class="badge ok">20秒</span>' : `<span class="badge warn">${s.duration_s}秒</span>`);
  if (s.vehicle_quality === 'legacy') b.push('<span class="badge warn">PWM:legacy</span>');
  else if (s.vehicle_quality) b.push(`<span class="badge ok">PWM:${esc(s.vehicle_quality)}</span>`);
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
        <div class="muted">${fmtTime(s.start_us)} ・ カメラ ${s.counts.CAM_FRONT} / ToF ${s.counts.TOF_FRONT} / IMU ${s.counts.IMU} / 車両 ${s.counts.VEHICLE}</div>
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
    ['ステアリング', fmt(st.steering_us)], ['スロットル', fmt(st.throttle_us)], ['モーター', fmt(st.motor_pct)],
    ['加速度ピーク', st.accel_peak_ms2 !== undefined ? `${st.accel_peak_ms2} m/s²` : '-'],
    ['映像の最大間隔', `${st.camera_max_gap_ms} ms`], ['車両データ', s.vehicle_quality || '-'],
  ].map(([k, v]) => `<div><span class="muted">${k}:</span> ${esc(v)}</div>`).join('');
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
function closeModal() { state.playing = false; state.modal = null; $('#modal').classList.add('hidden'); }
$('#m-save').addEventListener('click', async () => {
  try {
    await api('PATCH', `/api/scenes/${encodeURIComponent(state.modal.name)}`, { description: $('#m-desc').value, tags: parseTags($('#m-tags').value) });
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
  $('#ds-selected').innerHTML = sel.length
    ? `選択中: <b>${sel.length}</b> シーン / ${(secs / 60).toFixed(1)} 分 / 約 ${sel.reduce((a, s) => a + s.counts.CAM_FRONT, 0)} フレーム`
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
      <div class="muted">${esc(j.message)}</div>${j.error ? `<div class="err">${esc(j.error)}</div>` : ''}</div>`).join('')
    || '<div class="muted small">ありません</div>';
  const running = list.some((j) => j.state === 'running');
  if (running) { clearTimeout(jobTimer); jobTimer = setTimeout(loadJobs, 800); }
  if (state._running && !running) loadDatasets();
  state._running = running;
}
async function loadDatasets() {
  let list = [];
  try { list = await api('GET', '/api/datasets'); } catch (e) { toast(e.message, true); return; }
  $('#dataset-list').innerHTML = list.slice().reverse().map((d) => `
    <div class="ds">
      <h3>${esc(d.name)} <span class="muted small">${esc(d.version)} ・ ${esc(d.created_at)}</span></h3>
      <div class="stats">
        <span>シーン <b>${d.scenes}</b> (学習 ${d.train_scenes} / 検証 ${d.val_scenes})</span>
        <span>フレーム <b>${d.frames}</b></span><span>ToF <b>${d.tof_frames}</b></span>
        <span>時間 <b>${(d.hours * 60).toFixed(1)}</b> 分</span>
        <span>車両データ <b>${esc(d.vehicle_quality.join(', '))}</b></span>
      </div>
      <div class="row">
        ${d.export ? `<a href="/api/exports/${encodeURIComponent(d.export.file)}" download><button class="primary small">⬇ ダウンロード (${(d.export.bytes / 1e6).toFixed(1)} MB)</button></a>
                      <span class="muted small">data/exports/${esc(d.export.file)}(.sha256付き)</span>`
                   : `<button class="small" data-export="${esc(d.name)}">エクスポート(tar作成)</button>`}
        <span class="spacer"></span>
        <button class="small danger" data-del="${esc(d.name)}">削除</button>
      </div>
    </div>`).join('') || '<div class="muted">まだありません</div>';
}
$('#dataset-list').addEventListener('click', async (e) => {
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

// ------------------------------------------------------------ 起動
showTab('live');
pollStatus();
pollValues();
loadScenes();

// デバッグ・自動テスト用(ブラウザのコンソールから操作できるように)
window.jetformer = { state, showTab, openScene, showFrame, closeModal, loadScenes, loadDatasets, loadJobs };
