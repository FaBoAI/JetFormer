'use strict';

const $ = (s) => document.querySelector(s);
const state = { tab: 'live', scenes: [], selected: new Set(), status: null, modal: null, playing: false, modes: [] };
const COLORS = { steering: '#2563eb', throttle: '#dc2626', aux: '#9ca3af', motor: '#16a34a',
                 accel: '#7c3aed', ax: '#0891b2', ay: '#f59e0b', gyro: '#db2777', min: '#dc2626', median: '#2563eb' };

// ------------------------------------------------------------ 共通
async function api(method, url, body, signal) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  if (signal) opt.signal = signal;        // 返事を打ち切る印(走行タブの止まっているときの表示の問い合わせだけ = pollDrive。2026-10-27 fix6)
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
  if (name === 'sim') loadSim();
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
    else parts.push(`v2 / 読み直し ${(s.read_errors || 0) + (s.bus_errors || 0)}${s.skipped_polls ? ` / 飛ばし ${s.skipped_polls}` : ''}`
      + (s.board_resets ? ` / <span class="warn-text">基板の再起動 ${s.board_resets}回(${s.board_reset_ago_s}秒前。バッテリー・電源を確認)</span>` : ''));
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
              ['操縦', n.select ? '<span class="warn-text">AI(学習には使わない)</span>' : '人(プロポ)'],
              ['電池の電圧', vbatText(v.vbat) + (v.vbat && v.vbat.supported && Math.abs((+v.vbat.scale || 1) - 1) > 1e-6
                ? ` <span class="muted">(較正の係数 ${(+v.vbat.scale).toFixed(4)})</span>` : '')]);
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

// ------------------------------------------------------------ 電池の電圧(2026-10-25。基板が flags bit5 と予備のバイトで送るとき)
// 7.90 V 以上はふつう・7.70〜7.90 V は黄・7.70 V 未満は赤「パックを替える」(qualify_plan 3.7。境の値はサーバーの warn_v・bad_v)
// 6.0〜9.0 V の外の値(broken_v)は使わない(走行のゲインにも説明欄にも入れない。2026-10-25 fix.md)。来ていれば赤で出す
function vbatBrokenText(vb) {
  if (!vb || vb.broken_v == null || !Number.isFinite(+vb.broken_v)) return '';
  const r = Array.isArray(vb.range_v) && vb.range_v.length === 2 ? vb.range_v : [6.0, 9.0];
  return ` <b class="vbat-bad-text" id="vbat-broken">電圧の値がおかしい(${(+vb.broken_v).toFixed(2)} V。${(+r[0]).toFixed(1)}〜${(+r[1]).toFixed(1)} V の外なので使いません。配線・分圧を確かめる)</b>`;
}
function vbatText(vb) {
  if (!vb) return '<span class="muted">-(古いサーバー)</span>';
  if (!vb.supported) return '<span class="muted">電圧: 基板が対応していません</span>';
  const broken = vbatBrokenText(vb);
  if (vb.v == null || !Number.isFinite(+vb.v)) return broken ? broken.trim() : `<span class="warn-text">電圧の値が来ていません${vb.age_s != null ? `(最後は ${esc(vb.age_s)} 秒前)` : ''}</span>`;
  const lv = vb.level || 'ok', badV = vb.bad_v ?? 7.70, warnV = vb.warn_v ?? 7.90;
  return `<span class="badge ${lv} vbat-v" id="vbat-badge">${(+vb.v).toFixed(2)} V</span>`
    + (lv === 'bad' ? ` <b class="vbat-bad-text">パックを替える(${(+badV).toFixed(2)} V 未満)</b>`
      : lv === 'warn' ? ` <span class="warn-text">低め(${(+warnV).toFixed(2)} V 未満)</span>` : '') + broken;
}
// 最後の走行の前後の電圧(走行タブの上・走り終わりのメモ)
function vbatRunText(r) {
  if (!r) return '';
  const f = (x) => (x == null || !Number.isFinite(+x) ? '-' : `${(+x).toFixed(2)} V`);
  const parts = [`走る前 ${r.v_before != null ? `${f(r.v_before)}(基板)` : r.human_before ? `${esc(r.human_before)} V(説明欄)` : '-'}`];
  if (r.state === 'waiting') parts.push('走ったあと: 止まって 1 秒たつのを待っています(15 秒まで)');
  else if (r.state === 'done') parts.push(`走ったあと ${f(r.v_after)}`, `走行中の最小 ${f(r.v_min)}`);
  if (r.scenes_written) parts.push(`説明欄に書きました(シーン ${esc(r.scenes_written)} 個)`);
  if (r.note) parts.push(`<span class="muted">${esc(r.note)}</span>`);
  return parts.join(' ・ ');
}
function renderVbatStrip(st) {
  const el = $('#dr-vbat'), vb = st && st.vbat;
  if (!el) return;
  if (!vb) { el.className = 'vbat-strip small'; el.innerHTML = '<span class="muted">電池の電圧: -(古いサーバー)</span>'; return; }
  const r = vb.run;
  const runText = r && (vb.supported || r.v_before != null || r.human_before) ? ` ・ 最後の走行: ${vbatRunText(r)}` : '';
  el.innerHTML = `電池の電圧 ${vbatText(vb)}`
    + (!vb.supported ? ' <span class="muted">(LiPo チェッカーで測って、説明欄の V_before・V_after に書いてください)</span>'
      : st.running ? ' <span class="muted">(走行中は加速で一時的に下がります)</span>' : '') + runText;
  const broken = vb.broken_v != null && Number.isFinite(+vb.broken_v);
  el.className = 'vbat-strip small' + (broken ? ' bad' : vb.supported && (vb.level === 'warn' || vb.level === 'bad') ? ' ' + vb.level : '');
}
async function vbatCalib(body, what) {
  try {
    const r = await api('POST', '/api/vbat/calib', body);
    $('#vbat-calib-note').textContent = body.reset ? '係数を 1.0 に戻しました'
      : `係数 ${(+r.scale).toFixed(4)} を保存しました(基板 ${r.board_v} V → 電圧計 ${r.meter_v} V)`;
    toast(what);
  } catch (e) { $('#vbat-calib-note').innerHTML = `<span class="warn-text">${esc(e.message)}</span>`; toast(e.message, true); }
}
$('#btn-vbat-calib').addEventListener('click', () => {
  const v = +$('#vbat-meter').value;
  if (!$('#vbat-meter').value || !Number.isFinite(v)) { toast('電圧計の値を入れてください', true); return; }
  if (!confirm(`基板の電圧に掛ける係数を、電圧計の値 ${v.toFixed(2)} V に合わせて保存します。\n車は止めたまま(モーターが回っていないとき)にしてください。`)) return;
  vbatCalib({ meter_v: v }, '電圧の係数を保存しました');
});
$('#btn-vbat-calib-reset').addEventListener('click', () => {
  if (confirm('電圧の係数を 1.0(較正なし)に戻します。')) vbatCalib({ reset: true }, '電圧の係数を 1.0 に戻しました');
});

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
                 layout: $('#rec-layout').value, battery: $('#rec-battery').value,
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
  for (const t of s.tags || []) b.push(isSimTag(t) ? '<span class="badge sim">🎮 シミュレーター</span>' : `<span class="badge tag">${esc(t)}</span>`);
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
                                         min_duration: parseFloat($('#ds-min').value), tags: $('#ds-sim').checked ? ['simulator'] : [] });
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

const SIM_TAGS = ['simulator', 'sim', 'simulation', 'emulator', 'シミュレーター', 'シミュレータ', 'エミュレーター', 'エミュレータ'];
const isSimTag = (t) => SIM_TAGS.includes(String(t).trim().toLowerCase()) || SIM_TAGS.includes(String(t).trim());
function sourceBadge(d) {
  if (d.source === 'simulator') return '<span class="badge sim" title="シミュレーター(エミュレーター)で作ったデータ">🎮 シミュレーター</span>';
  if (d.source === 'mixed') return `<span class="badge sim" title="実機とシミュレーターのシーンが混ざっている">🎮 実機 + シミュレーター(${d.sim_scenes} シーン)</span>`;
  return '<span class="badge ok" title="実機で集めたデータ">実機</span>';
}
async function loadDatasets() {
  let list = [];
  try { list = await api('GET', '/api/datasets'); } catch (e) { toast(e.message, true); return; }
  state.datasetList = list;
  if (state.hf === null) await loadHf(false);
  // 1つのデータセットの情報が欠けていても、一覧全体は表示する(検証用に作った特殊なデータセットなど)
  const card = (d) => { if (d.broken) return `<div class="ds"><h3>${esc(d.name)} <span class="badge warn">壊れています</span></h3>
      <div class="err small">データセットの情報を読めません(作成の途中で電源が切れたなど): ${esc(d.error || '')}</div>
      <div class="row"><span class="spacer"></span><button class="small danger" data-del="${esc(d.name)}">削除</button></div></div>`;
    try { return datasetCard(d); } catch (e) {
    return `<div class="ds"><h3>${esc(d.name)}</h3><div class="err small">表示できません: ${esc(e.message)}</div>
      <div class="row"><span class="spacer"></span><button class="small danger" data-del="${esc(d.name)}">削除</button></div></div>`; } };
  $('#dataset-list').innerHTML = list.slice().reverse().map(card).join('') || '<div class="muted">まだありません</div>';
}
function datasetCard(d) {
  const v = (x) => (x === undefined || x === null ? '-' : x);
  return `
    <div class="ds">
      <h3>${esc(d.name)} ${sourceBadge(d)} <span class="muted small">${esc(v(d.version))} ・ ${esc(v(d.created_at))}</span></h3>
      <div class="ds-tags small">${(d.tags || []).filter((t) => !isSimTag(t)).map((t) => `<span class="badge tag">${esc(t)}</span>`).join(' ')}
        <button class="small" data-ds-tags="${esc(d.name)}" title="データセットのタグを付け直す(Hugging Face のカードのタグにもなる)">タグ</button>
        ${d.source === 'mixed' ? '' : `<button class="small" data-ds-sim="${esc(d.name)}" data-on="${d.source === 'simulator' ? 1 : 0}">${d.source === 'simulator' ? '🎮 シミュレーターを外す' : '🎮 シミュレーターにする'}</button>`}</div>
      <div class="stats">
        <span>シーン <b>${v(d.scenes)}</b> (学習 ${v(d.train_scenes)} / 検証 ${v(d.val_scenes)})</span>
        <span>フレーム <b>${v(d.frames)}</b></span><span>ToF <b>${v(d.tof_frames)}</b></span>
        <span>時間 <b>${typeof d.hours === 'number' ? (d.hours * 60).toFixed(1) : '-'}</b> 分</span>
        <span>車両データ <b>${esc((d.vehicle_quality || []).join(', ') || '-')}</b></span>
      </div>
      ${d.import_source ? `<div class="muted small">📥 PC から取り込み: ${esc(d.import_source.file)} ・ ${esc(d.import_source.imported_at)}</div>` : ''}
      <div class="small">${d.scene_count ? `🎞 シーンタブに <b>${d.scene_count}</b> シーン(${esc(d.scene_log)})
          <button class="small" data-ds-view="${esc(d.scene_log)}">シーンタブで見る</button>
          <button class="small" data-ds-expand="${esc(d.name)}" title="データセットから作り直す">作り直す</button>`
        : d.local_scenes ? `<span class="muted">🎞 元のシーン ${d.local_scenes} / ${v(d.scenes)} がシーンタブにあります</span>`
        : `<span class="muted">シーンタブには出ていません</span> <button class="small" data-ds-expand="${esc(d.name)}"
             title="データセットのシーンを、収集したシーンと同じ形にして、シーンタブで1フレームずつ見られるようにする">🎞 シーンに展開</button>`}</div>
      ${d.hf_source ? `<div class="muted small">🤗 取り込み元: <a href="${esc(d.hf_source.url)}" target="_blank" rel="noopener">${esc(d.hf_source.repo_id)}</a>
        ・ ${esc(d.hf_source.downloaded_at)}</div>` : ''}
      <div class="stats">${Object.entries(d.mode_frames || {}).map(([id, n]) => {
        const m = (d.modes || []).find((x) => x.id === +id);
        return `<span><span class="badge mode" style="background:${modeColor(+id)}">${id}: ${esc(m ? m.name : 'ID ' + id)}</span> ${n} フレーム</span>`;
      }).join('')}</div>
      <div class="row">
        <a href="/api/datasets/${encodeURIComponent(d.name)}/download" download="${esc(d.name)}.tar"
           title="その場で tar にしながら送ります(Hugging Face も、Jetson の空き容量も使いません)"><button class="primary small">⬇ PCにダウンロード(tar・約 ${((d.bytes || 0) / 1e6).toFixed(0)} MB)</button></a>
        ${d.export ? `<a href="/api/exports/${encodeURIComponent(d.export.file)}" download><button class="small">保存済みの tar (${(d.export.bytes / 1e6).toFixed(1)} MB)</button></a>
                      <span class="muted small">data/exports/${esc(d.export.file)}(.sha256付き)</span>`
                   : `<button class="small" data-export="${esc(d.name)}" title="data/exports/ に tar と sha256 を作ります">エクスポート(tar作成)</button>`}
        <span class="spacer"></span>
        <button class="small danger" data-del="${esc(d.name)}">削除</button>
      </div>
      ${hfRow(d)}
    </div>`;
}
$('#dataset-list').addEventListener('click', async (e) => {
  const vw = e.target.closest('[data-ds-view]'), ex = e.target.closest('[data-ds-expand]');
  if (vw) { showTab('scenes'); $('#scene-filter').value = vw.dataset.dsView; renderScenes(); return; }
  if (ex) {
    try { await api('POST', `/api/datasets/${encodeURIComponent(ex.dataset.dsExpand)}/scenes`); toast('シーンへの展開を始めました(処理状況に進み具合)'); loadJobs(); }
    catch (err) { toast(err.message, true); }
    return;
  }
  const tg = e.target.closest('[data-ds-tags]'), sm = e.target.closest('[data-ds-sim]');
  if (tg || sm) {
    const name = (tg || sm).dataset.dsTags || (tg || sm).dataset.dsSim;
    const d = (state.datasetList || []).find((x) => x.name === name) || { tags: [] };
    let tags = d.tags || [];
    if (tg) {
      const t = prompt(`${name} のタグ(カンマ区切り。シミュレーターのデータには simulator)`, tags.join(', '));
      if (t === null) return;
      tags = t.split(',').map((x) => x.trim()).filter(Boolean);
    } else {
      tags = sm.dataset.on === '1' ? tags.filter((x) => !isSimTag(x)) : [...tags, 'simulator'];
    }
    try { await api('POST', `/api/datasets/${encodeURIComponent(name)}/tags`, { tags }); loadDatasets(); }
    catch (err) { toast(err.message, true); }
    return;
  }
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
// ------------------------------------------------------------ PC から取り込む(シミュレーターなど)
$('#imp-cmd').textContent = `python3 upload_datasets.py --host ${location.origin} sim_run_01/ sim_run_02/ more.tar\n` +
  `python3 upload_datasets.py --host ${location.origin} --real exported.tar     # 実機のデータとして`;
state.importQueue = [];
state.importing = false;
function importRow(item) {
  const pct = item.total ? Math.round(100 * item.sent / item.total) : 0;
  return `<div class="job"><b>${esc(item.file.name)}</b> <span class="muted">${(item.file.size / 1e6).toFixed(1)} MB ・ ${esc(item.state)}</span>
    <div class="progress blue"><div class="bar" style="width:${pct}%"></div></div>${item.error ? `<div class="err">${esc(item.error)}</div>` : ''}</div>`;
}
function renderImports() { $('#imp-list').innerHTML = state.importQueue.map(importRow).join(''); }
function uploadOne(item) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    const q = `filename=${encodeURIComponent(item.file.name)}&simulator=${item.simulator ? 1 : 0}`;
    xhr.open('PUT', '/api/datasets/import?' + q);
    xhr.upload.onprogress = (e) => { item.sent = e.loaded; item.total = e.total; item.state = '送信中'; renderImports(); };
    xhr.onload = () => {
      if (xhr.status < 300) { item.state = '送信済み → 取り込み中(処理状況に進み具合)'; item.sent = item.total = 1; }
      else { let m = xhr.responseText; try { m = JSON.parse(m).detail || m; } catch (e) { /* そのまま */ } item.state = '失敗'; item.error = m; }
      renderImports(); loadJobs(); resolve();
    };
    xhr.onerror = () => { item.state = '失敗'; item.error = '送信できませんでした(接続が切れた)'; renderImports(); resolve(); };
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.send(item.file);
  });
}
async function runImports() {
  if (state.importing) return;
  state.importing = true;
  for (const item of state.importQueue) {
    if (item.state !== '待ち') continue;
    await uploadOne(item);
  }
  state.importing = false;
}
function queueImports(files) {
  const ok = /\.(tar|tgz|tar\.gz|zip)$/i;
  for (const f of files) {
    if (!ok.test(f.name)) { toast(`${f.name}: .tar・.tar.gz・.zip を選んでください(フォルダは zip か tar にするか、コマンドで送る)`, true); continue; }
    state.importQueue.push({ file: f, simulator: $('#imp-sim').checked, state: '待ち', sent: 0, total: f.size, error: '' });
  }
  renderImports(); runImports();
}
$('#imp-files').addEventListener('change', (e) => { queueImports([...e.target.files]); e.target.value = ''; });
const dz = $('#imp-drop');
['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
dz.addEventListener('drop', (e) => queueImports([...e.dataTransfer.files]));
// ------------------------------------------------------------ Hugging Face からダウンロード
async function searchHubDatasets() {
  const author = $('#hfdl-author').value.trim();
  $('#hfdl-list').innerHTML = '<div class="muted small">探しています…</div>';
  let list = [];
  try { list = await api('GET', '/api/hf/hub_datasets?author=' + encodeURIComponent(author) + '&kind=' + $('#hfdl-kind').value); }
  catch (e) { $('#hfdl-list').innerHTML = ''; toast(e.message, true); return; }
  $('#hfdl-list').innerHTML = list.map((d) => `
    <div class="hfdl-item"><a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.repo_id)}</a>
      ${d.private ? '<span class="badge tag">非公開</span>' : ''}${d.simulator ? ' <span class="badge sim">🎮 シミュレーター</span>' : ''}
      <span class="muted small">${esc((d.last_modified || '').replace('T', ' '))}</span><span class="spacer"></span>
      <button class="small" data-hfdl-pick="${esc(d.repo_id)}">選ぶ</button></div>`).join('')
    || '<div class="muted small">見つかりませんでした(JetFormer のタグが付いたデータセットだけを探します)</div>';
}
async function checkHubRepo() {
  const repo = $('#hfdl-repo').value.trim();
  if (!repo) { toast('リポジトリを入れてください', true); return null; }
  $('#hfdl-files').textContent = '確認しています…';
  let r;
  try { r = await api('GET', '/api/hf/hub_files?repo_id=' + encodeURIComponent(repo)); }
  catch (e) { $('#hfdl-files').textContent = ''; toast(e.message, true); return null; }
  $('#hfdl-files').innerHTML = r.files.length ? r.files.map((f) => `<div>📦 <b>${esc(f.name)}</b> ${(f.size / 1e6).toFixed(1)} MB
      ${f.exists ? '<span class="warn-text">— 同じ名前のデータセットがあります(削除してから取り込んでください)</span>' : ''}</div>`).join('')
    : '<span class="warn-text">学習用の tar(nuscenes/*.tar)がありません。Dataset Viewer 用の parquet だけのリポジトリは取り込めません</span>';
  return r.files;
}
$('#btn-hfdl-search').addEventListener('click', searchHubDatasets);
$('#btn-hfdl-check').addEventListener('click', checkHubRepo);
$('#hfdl-list').addEventListener('click', (e) => {
  const pick = e.target.closest('[data-hfdl-pick]');
  if (pick) { $('#hfdl-repo').value = pick.dataset.hfdlPick; checkHubRepo(); }
});
$('#btn-hfdl-start').addEventListener('click', async () => {
  const files = await checkHubRepo();
  if (!files || !files.length) return;
  if (files[0].exists) { toast('同じ名前のデータセットがあります', true); return; }
  if (!confirm(`${$('#hfdl-repo').value.trim()} の ${files[0].name}(${(files[0].size / 1e6).toFixed(1)} MB)をダウンロードして取り込みます。`)) return;
  try { await api('POST', '/api/hf/download', { repo_id: $('#hfdl-repo').value.trim() }); toast('ダウンロードを始めました(処理状況に進み具合が出ます)'); loadJobs(); }
  catch (err) { toast(err.message, true); }
});


// ------------------------------------------------------------ 学習
const TRAIN_RATE = 200, VAL_RATE = 650;     // Orin Nano実測(v0): 学習 約200枚/秒、検証 約650枚/秒(方策の目安が無いとき)
const fmtDur = (s) => (s === null || s === undefined) ? '-' : s >= 3600 ? `${Math.floor(s / 3600)}時間${Math.round(s % 3600 / 60)}分`
  : s >= 60 ? `${Math.floor(s / 60)}分${Math.round(s % 60)}秒` : `${Math.round(s)}秒`;
state.datasets = [];
state.policies = [];
async function loadTrain() {
  try { state.datasets = await api('GET', '/api/datasets'); } catch (e) { return; }
  try { state.policies = await api('GET', '/api/policies'); } catch (e) { state.policies = []; }
  const sel = $('#tr-dataset'), cur = sel.value;
  sel.innerHTML = state.datasets.filter((d) => !d.broken).slice().reverse().map((d) => `<option value="${esc(d.name)}">${d.source === 'simulator' ? '🎮 ' : d.source === 'mixed' ? '🎮+ ' : ''}${esc(d.name)}(${d.scenes}シーン / ${d.frames}フレーム${d.source === 'simulator' ? '・シミュレーター' : d.source === 'mixed' ? '・実機+シミュレーター' : ''})</option>`).join('')
    || '<option value="">データセットがありません</option>';
  if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  renderPolicySelect();
  updateEstimate(); loadRuns(); pollTrain();
}
// モデルの種類(方策の登録簿 GET /api/policies)。準備ができていないものは選べず、理由を出す
const trPolicy = () => state.policies.find((x) => x.name === $('#tr-policy').value);
function renderPolicySelect() {
  const sel = $('#tr-policy'), cur = sel.value || 'jetformer_v0';
  sel.innerHTML = state.policies.map((p) => `<option value="${esc(p.name)}" ${p.ready ? '' : 'disabled'}>${esc(p.title)}`
    + `${p.status === 'experimental' ? '(試験的)' : ''}${p.ready ? '' : ' — 使えません'}</option>`).join('')
    || '<option value="jetformer_v0">TransFuser軽量型(JetFormer v0・今の標準)</option>';
  const ok = (v) => [...sel.options].some((o) => o.value === v && !o.disabled);
  sel.value = ok(cur) ? cur : 'jetformer_v0';
  renderPolicyInfo();
}
function renderPolicyInfo() {
  const p = trPolicy();
  if (!p) { $('#tr-policy-info').innerHTML = ''; $('#tr-policy-opts').innerHTML = ''; return; }
  const h = p.hints || {};
  const notReady = state.policies.filter((x) => !x.ready).map((x) => `<div class="muted">${esc(x.why_not)}</div>`).join('');
  $('#tr-policy-info').innerHTML = `<div class="muted">${esc(p.description || '')}</div>`
    + `<div>推論の目安 約 <b>${h.infer_ms ?? '-'}</b> ms(TensorRT・走行と同じ呼び方)・パラメータ <b>${h.params_m ?? '-'}</b>M`
    + `${h.params_trainable_m ? `(学習するのは ${h.params_trainable_m}M)` : ''}・出力 ${esc((p.outputs || ['actions']).join(', '))}</div>`
    + (p.requires_pretrained ? '<div class="warn-text">事前学習の重みが必要です(「ImageNet事前学習の重みを使う」を入れたまま。重みは自動では取りません)</div>' : '')
    + (p.postprocess ? '<div class="muted">走行タブで目標速度 → 制御器(速度の上限)に切り替えられます。既定は操作の枝のまま</div>' : '')
    + (notReady ? `<details><summary class="small muted">選べないモデルの種類</summary>${notReady}</details>` : '');
  $('#tr-policy-opts').innerHTML = (p.options || []).map((o) => {
    const d = p.defaults[o.key];
    const id = `tr-popt-${o.key}`;
    if (o.type === 'bool') return `<label class="inline"><input type="checkbox" id="${id}" data-popt="${esc(o.key)}" ${d ? 'checked' : ''}> ${esc(o.label || o.key)}</label>`;
    if (o.type === 'choice') {
      return `<label>${esc(o.label || o.key)}<select id="${id}" data-popt="${esc(o.key)}">${(o.choices || []).map((c) =>
        `<option value="${esc(JSON.stringify(c))}" ${JSON.stringify(c) === JSON.stringify(d) ? 'selected' : ''}>${esc(Array.isArray(c) ? c.join('・') : String(c))}${JSON.stringify(c) === JSON.stringify(d) ? '(既定)' : ''}</option>`).join('')}</select></label>`;
    }
    return `<label>${esc(o.label || o.key)}(既定 ${esc(String(d))})<input id="${id}" data-popt="${esc(o.key)}" type="number" value="${esc(String(d))}"`
      + `${o.min != null ? ` min="${o.min}"` : ''}${o.max != null ? ` max="${o.max}"` : ''} step="${o.type === 'int' ? 1 : 'any'}" class="num"></label>`;
  }).join('');
}
function readPolicyCfg() {
  // 既定値から変えた項目だけ送る(何も変えなければ null = 既定の形)
  const p = trPolicy();
  if (!p) return null;
  const cfg = {};
  for (const el of document.querySelectorAll('#tr-policy-opts [data-popt]')) {
    const key = el.dataset.popt, o = (p.options || []).find((x) => x.key === key) || {};
    let v;
    if (o.type === 'bool') v = el.checked;
    else if (o.type === 'choice') v = JSON.parse(el.value);
    else v = o.type === 'int' ? parseInt(el.value, 10) : +el.value;
    if (JSON.stringify(v) !== JSON.stringify(p.defaults[key])) cfg[key] = v;
  }
  return Object.keys(cfg).length ? cfg : null;
}
$('#tr-policy').addEventListener('change', () => { renderPolicyInfo(); updateEstimate(); });
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
  const h = (trPolicy() || {}).hints || {};       // 方策の目安(bench の実測)。無ければ v0 の実測
  const perEpoch = nTrain / (h.train_rate || TRAIN_RATE) + nVal / (h.val_rate || VAL_RATE) + 3;
  const total = perEpoch * epochs + 30 + ($('#tr-engine').checked ? (h.engine_s || 70) : 0);
  $('#tr-estimate').innerHTML = `学習 最大 <b>${nTrain}</b> 枚 / 検証 最大 <b>${nVal}</b> 枚<br>1エポック 約 <b>${fmtDur(perEpoch)}</b> → 合計 約 <b>${fmtDur(total)}</b>`
    + '<div class="muted small">操作値が無効・AI走行中のフレームは除外されるため、実際はこれより少し短くなります</div>';
}
['#tr-dataset', '#tr-stride', '#tr-epochs', '#tr-engine'].forEach((s) => $(s).addEventListener('change', updateEstimate));
$('#tr-epochs').addEventListener('input', updateEstimate);
$('#btn-tr-start').addEventListener('click', async () => {
  const seed = $('#tr-seed').value.trim();
  const body = { dataset: $('#tr-dataset').value, stride: +$('#tr-stride').value, epochs: +$('#tr-epochs').value,
                 batch: +$('#tr-batch').value, lr: +$('#tr-lr').value, pretrained: $('#tr-pretrained').checked,
                 build_engine: $('#tr-engine').checked, command_input: !$('#tr-nocmd').checked,
                 mask_impacts: $('#tr-mask-impacts').checked,
                 policy: $('#tr-policy').value || 'jetformer_v0', policy_cfg: readPolicyCfg(),
                 impact_g: $('#tr-impact-g').value ? +$('#tr-impact-g').value : null, seed: seed === '' ? null : parseInt(seed, 10) };
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
    const emu = (r.files || []).includes('jetformer.onnx') && (r.files || []).includes('model_config.json')
      ? `<a href="/api/models/${encodeURIComponent(r.name)}/bundle.zip" download="${esc(r.name)}.zip"
           title="ONNX・model_config.json・入出力の仕様(io_spec.json)・README・onnxruntime で動かす例を1つの zip に(Mac のエミュレーター用)"><button class="primary small">⬇ エミュレーター用(zip)</button></a>
         ${(r.files || []).includes('best.pth') ? `<a href="/api/models/${encodeURIComponent(r.name)}/bundle.zip?pth=true" download="${esc(r.name)}_pth.zip"
           title="上の一式に PyTorch の重み(best.pth)も入れる"><button class="small">＋重み(.pth)</button></a>` : ''}` : '';
    const files = emu + ' ' + (r.files || []).filter((f) => f !== 'run.json').map((f) =>
      `<a href="/api/runs/${encodeURIComponent(r.name)}/files/${f}" download><button class="small">⬇ ${f}</button></a>`).join(' ');
    const pol = r.params.policy || 'jetformer_v0';
    const polBadge = `<span class="badge policy ${pol === 'jetformer_v0' ? 'v0' : 'exp'}" title="モデルの種類">${esc(r.policy_title || pol)}</span>`;
    return `<div class="ds">
      <h3>${esc(r.name)} <span class="badge ${cls}">${txt}</span> ${polBadge}</h3>
      <div class="stats">
        <span>データセット <b>${esc(r.params.dataset)}</b></span><span>間引き <b>1/${r.params.stride}</b></span>
        <span>エポック <b>${r.epochs_done ?? '-'}</b> / ${r.params.epochs}</span>
        <span>最良の検証誤差 <b>${r.best_val_us ?? '-'}</b> µs${r.best_epoch ? `(${r.best_epoch}エポック目)` : ''}</span>
        <span>時間 <b>${fmtDur(r.duration_s)}</b></span><span>学習 ${r.train_samples ?? '-'} 枚</span>
        ${r.params.command_input === false ? '<span class="badge tag">今の操作値なし</span>' : ''}
        ${r.params.impact_g != null && Math.abs(r.params.impact_g - 2.5) < 1e-6 ? '<span class="badge tag">衝突 2.5G</span>' : ''}
        ${r.params.seed != null ? `<span class="badge tag">種 ${esc(String(r.params.seed))}</span>` : ''}
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
// pending: 「基準の設定」・ベンチマークの候補で画面に入れ、まだサーバーへ送っていない項目(走行開始で送る。表示の同期で戻さない)
state.drive = { models: [], st: null, paramsLoaded: false, pending: {} };
const DR_PARAMS = [
  ['#dr-steer-gain', 'steer_gain', (v) => (+v).toFixed(2)],
  ['#dr-throttle-gain', 'throttle_gain', (v) => (+v).toFixed(2)],
  ['#dr-trim', 'steer_trim_us', (v) => `${v > 0 ? '+' : ''}${(+v).toFixed(0)} µs`],
  ['#dr-look', 'lookahead', (v) => `${v}(約${Math.round(v * 33)}ms先)`],
  ['#dr-osc', 'osc_percent', (v) => `${v > 0 ? '+' : ''}${(+v).toFixed(1)} %`],
  ['#dr-rev-us', 'rev_us', (v) => `中立+${(+v).toFixed(0)} µs`],
  ['#dr-rev-s', 'rev_s', (v) => `${(+v).toFixed(2)} 秒`],
  // TCP の制御器(tcp だけ)・予測の重ね合わせ(どのモデルにも)
  ['#dr-speed-cap', 'speed_cap', (v) => (+v > 0 ? `${(+v).toFixed(2)} m/s(約 ${Math.round(+v / 0.5e-3)} eRPM)` : '上限なし')],
  ['#dr-speed-scale', 'speed_scale', (v) => (+v).toFixed(2)],
  ['#dr-ens-j', 'ensemble_j', (v) => (+v > 0 ? `${v} フレーム前まで` : 'オフ')],
  ['#dr-ens-m', 'ensemble_m', (v) => (+v).toFixed(1)],
];
const drModel = () => state.drive.models.find((m) => m.name === $('#dr-model').value);
// 走行タブで選ぶモデル(DOM を使わない。tests/test_drive_model_select.py が node で確かめる)。
// 走行中のモデル → 初回は「推奨のモデル」(評価で決めたもの。一覧の先頭)→ 前回のモデル → 今の選択 → 一覧の最初の走れるもの。
// 前は「前回 → 一番新しい学習結果」で、評価の前に使った古いモデルや、比較の実験で作った新しいモデルが選ばれていた。
// 走れないもの(drivable false: エンジンが精度の確認に落ちた・入力が作れないなど)は選ばない。
function pickDriveModel(models, { running = '', last = '', cur = '', first = false } = {}) {
  const ok = (name) => !!name && models.some((m) => m.name === name && m.drivable !== false);
  const rec = models.find((m) => m.recommended && m.drivable !== false);
  const order = first ? [running, rec && rec.name, last, cur] : [running, cur, rec && rec.name, last];
  for (const want of order) if (ok(want)) return want;
  const any = models.find((m) => m.drivable !== false);
  return any ? any.name : '';
}
// 走行タブの表示の問い合わせの返事を待つ長さ(2026-10-27 fix6 直す 6-2 回目 w2-ui-w1-ui-r62-1)。止まっているとき(drIdle)だけ使う。
// サーバーの生存確認(token の問い合わせが途絶えて 5 秒で止める)と同じ 5 秒
const DR_IDLE_WAIT_MS = 5000;
// 止まっている = この画面が最後に読んだ status が走行中でなく、この画面が走行を始めた印(token。付けた問い合わせが生存確認になる)も無い。
// 走行中・走行を始めた画面では、表示の問い合わせの出し方(回数・間隔・打ち切らない)を前とまったく同じにする
const drIdle = () => !state.drive.token && !(state.drive.st && state.drive.st.running);
async function loadDrive() {
  // モデルの一覧の GET が失敗した・DR_IDLE_WAIT_MS のあいだ返事が無い(サーバーが起き直している・通信の途切れ)とき、止まっていれば
  // 表示の問い合わせの輪を始め、次に同期するまで ▶・基準の設定・ブロックを断る(drLost)。前は toast を出して return するだけで、
  // 輪は走行タブでも止まったまま(pollDrive は走行タブか走行中のときしか次を掛けない)・paramsLoaded は true のまま・▶ も押せて、
  // 天井の横の文・スイッチと ▶ の確認の文が前の表示のまま(curl で入れた左 4.2 を読まず 4.1・起き直したあとも起き直す前の 4.2・
  // 200 µs)だった(2026-10-27 fix6 反証 w2-ui-w1-ui-r62-1)。一覧の GET は打ち切らない(遅いだけなら、返事が来たところで前と同じに
  // 続ける)。一覧が通るときは前とまったく同じ(期限のタイマーは返事で消す)
  // 一覧を読めなかった・遅れた印(modelsMiss)と、輪の 2 回目の同期で 1 回だけ読み直す数(modelsRetry)も付ける(2026-10-27 fix7 反証
  // w2-ui-w2-ui-r72-ui-2。下の retryDriveModels)。印は一覧が通ったら消す。一覧が通るときは前とまったく同じ
  // 走行タブを開いた・戻ったとき、止まっていれば(drIdle)、このあとに出した表示の問い合わせが通るまで paramsLoaded を下ろす(▶・基準の
  // 設定・ブロックは drNotSynced が「サーバーの設定を読み込んでいます」で断る = 画面を開いた直後と同じ)。2026-10-28 fix8 反証 8-1 回目
  // stale-status-after-drive-tab-return: 走行タブを離れている間は表示の問い合わせの輪が止まる(pollDrive は走行タブか走行中のときしか
  // 次を掛けない)のに paramsLoaded は true のままで、戻った直後(下の一覧の GET を待つ間。Wi-Fi が弱いと最大 5 秒)の ▶ が離れたときの
  // 古い status の確認の文(見張り ON・左 4.1)を出し、車は離れている間に curl・別の画面で変えたサーバーの値(見張り OFF・左 4.2)で
  // 走った。drTabGen は戻る前に出した問い合わせの返事を「読み直した」に数えないための番号(pollDrive)。問い合わせ・送る物・版は前と
  // 同じ。走行中・token のある画面は輪が走行タブの外でも回っているので前と同じ
  if (drIdle()) { state.drive.paramsLoaded = false; state.drive.drTabGen = (state.drive.drTabGen || 0) + 1; }
  const lost = () => { state.drive.modelsMiss = true; state.drive.modelsRetry = 2; if (drIdle()) { drLost(); pollDrive(); } };
  const late = setTimeout(lost, DR_IDLE_WAIT_MS);
  state.drive.modelsBusy = (state.drive.modelsBusy || 0) + 1;       // 一覧の GET を待っている数(待っている間は読み直さない)
  try { state.drive.models = await api('GET', '/api/drive/models'); } catch (e) {
    state.drive.modelsBusy--; clearTimeout(late); toast(e.message, true); missDriveModels(); lost(); return;
  }
  state.drive.modelsBusy--; state.drive.modelsMiss = false;
  clearTimeout(late);
  if (!state.drive.st) { try { state.drive.st = await api('GET', '/api/drive/status'); } catch (e) { /* 後で */ } }
  fillDriveModels();
  pollDrive();
  loadBench();
}
// モデルの一覧(state.drive.models)を選ぶ欄に入れ、選ぶモデルを決める(loadDrive の中にあったものをそのまま関数にした。fix7。
// 走行タブを開いたとき(loadDrive)と、一覧を読めなかったあとの読み直し(reloadDriveModels)が使う)
function fillDriveModels() {
  const sel = $('#dr-model'), cur = sel.value;
  sel.innerHTML = state.drive.models.map((m) =>
    `<option value="${esc(m.name)}" ${m.drivable === false ? 'disabled' : ''}>${m.recommended ? '★推奨 ' : ''}「${esc(m.policy_title || 'JetFormer v0')}」${esc(m.name)}`
    + `${m.best_val_us != null ? `(検証誤差 ${m.best_val_us}µs)` : ''}${m.drivable === false ? ` — 走れません: ${esc(m.why_not || '')}` : ''}</option>`).join('')
    || '<option value="">走行できるモデルがありません(学習タブで「TensorRTエンジンを作る」)</option>';
  const st = state.drive.st;
  const first = !state.drive.restored;
  const last = first && st && st.last ? st.last.run : '';                       // 初回だけ前回使ったモデルを候補にする
  const want = pickDriveModel(state.drive.models, { running: st && st.running ? st.run : '', last, cur, first });
  if (want) sel.value = want;
  const rec = state.drive.models.find((m) => m.recommended);
  if (first && rec && want === rec.name && last && last !== rec.name) state.drive.restoreNote = `前回のモデル(${last})ではなく、推奨のモデルを選びました`;
  renderDriveModel();
  state.drive.restored = true;
}
// 走行タブを開いたときにモデルの一覧の GET が失敗した・5 秒返らなかったあと(2026-10-27 fix7 反証 w2-ui-w2-ui-r72-ui-2)。前は一覧を
// タブを替えるまで読み直さず、表示の問い合わせの輪は回って「同期済み」になるので、初めて開いたときは選ぶ欄が空・ベンチマークも
// 読まないまま、「基準の設定」が「モデル … が走行の一覧にありません(TensorRT エンジンが必要)」とまちがった理由を出した(エンジンは
// ある。学習タブでエンジンを作り直すなどのまちがった手当てに進ませる)。今は:
//  - 止まっていて(drIdle)走行タブにいるとき、一覧の GET を待っていなければ、輪の 2 回目の同期(失敗のすぐあとの同期の次 = 約 1 秒後)
//    で一覧を 1 回だけ読み直す(retryDriveModels → reloadDriveModels。表示の問い合わせの輪は増やさない = pollDrive は呼ばない)。
//    通れば loadDrive と同じに選ぶ欄とベンチマークを入れる。通らなければそのまま(もう読み直さない。タブを開き直すと loadDrive がまた読む)
//  - 読めていない間、止まっていれば「基準の設定」・ブロック・▶ の「一覧にありません(TensorRT エンジンが必要)」「モデルを選んでください」
//    のかわりに DR_MODELS_MISS(読めていない・開き直すか読み直す)。選ぶ欄が空なら、その欄にも短い文を出す
//  走行中・token のある画面・一覧が通るときは前とまったく同じ(問い合わせ・文・送る物)
const DR_MODELS_MISS = 'モデルの一覧を読めていません(一覧の問い合わせが通らないか遅れています。TensorRT エンジンが無いという意味ではありません)。'
  + '少し待っても「モデル」の欄が空なら、ほかのタブ → 走行タブで開き直すか、画面を読み直してください';
const drModelsMissing = () => !!state.drive.modelsMiss && drIdle();
function missDriveModels() {                          // 選ぶ欄が空のとき(初めて開いた)だけ、欄に短い文を出す
  const sel = $('#dr-model');
  if (!sel.options.length) sel.innerHTML = '<option value="">— モデルの一覧を読めていません(通信の途切れ)。少し待つか、ほかのタブ → 走行タブ —</option>';
}
function retryDriveModels() {                         // pollDrive が同期できたあとに呼ぶ
  const d = state.drive;
  if (!d.modelsMiss || d.modelsBusy > 0 || !drIdle() || state.tab !== 'drive' || !(d.modelsRetry > 0)) return;
  if (--d.modelsRetry === 0) reloadDriveModels();
}
async function reloadDriveModels() {
  let models;
  state.drive.modelsBusy = (state.drive.modelsBusy || 0) + 1;
  try { models = await api('GET', '/api/drive/models'); } catch (e) { return; } finally { state.drive.modelsBusy--; }
  if (!Array.isArray(models)) return;                 // 本文の無い返事は読めなかったのと同じ
  state.drive.models = models; state.drive.modelsMiss = false;
  fillDriveModels();
  loadBench();
}
function renderDriveModel() {
  const m = drModel();
  const modeSel = $('#dr-mode'), cur = modeSel.value;
  const isTcp = !!(m && m.postprocess && m.postprocess.kind === 'tcp_controller');
  $('#dr-tcp').classList.toggle('hidden', !isTcp);
  if (!m) { $('#dr-model-info').textContent = ''; modeSel.innerHTML = ''; return; }
  const names = Object.fromEntries((m.modes || []).map((x) => [x.id, x.name]));
  const pp = m.postprocess && m.postprocess.kind ? (m.postprocess.kind === 'tcp_controller' ? 'TCP の制御器' : m.postprocess.kind) : 'なし';
  const ec = m.engine_check;
  $('#dr-model-info').innerHTML = (m.recommended ? `<div><span class="badge ok">★推奨のモデル</span>${m.recommended_reason ? ` ${esc(m.recommended_reason)}` : ''}`
      + `${state.drive.restoreNote ? `<div class="muted">${esc(state.drive.restoreNote)}</div>` : ''}</div>` : '')
    + `モデルの種類 <b>${esc(m.policy_title || 'JetFormer v0')}</b> ・ 出力 ${esc((m.outputs || ['actions']).join(', '))} ・ 後処理: ${esc(pp)}`
    + ` ・ エンジン ${esc(m.engine_precision || 'fp16')}`
    + (m.drivable === false ? `<div class="warn-text">走行できません: ${esc(m.why_not || '')}</div>` : '')
    // 精度の確認に落ちたエンジンは drivable false(上の理由に出る)。ここは速さだけ落ちたときの警告
    + (ec && ec.ok === false && ec.accuracy_ok !== false ? `<div class="warn-text">このエンジンは速さの確認に落ちています${ec.run_numpy_median_ms != null
      ? `(推論 中央値 ${(+ec.run_numpy_median_ms).toFixed(2)} ms)` : ''}。走れますが、30fps を守れないことがあります</div>` : '')
    + (ec && ec.note ? `<div class="muted small">${esc(ec.note)}</div>` : '')
    + `<div>データセット ${esc(m.dataset)} ・ 間引き 1/${m.stride ?? '-'} ・ ${esc(m.created_at || '')}</div>`
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
$('#dr-model').addEventListener('change', async () => {
  state.drive.restoreNote = '';                 // 人が選び直した(「推奨のモデルを選びました」は出さない)
  renderDriveModel();
  // モデルを替えたら TCP の出し方は「操作の枝だけ」に戻す(試験的な出し方を別のモデルへ持ち越さない)
  if (+$('#dr-tcp-mode').value !== 0) {
    $('#dr-tcp-mode').value = '0';
    try { await api('POST', '/api/drive/params', { tcp_mode: 0 }); } catch (e) { toast(e.message, true); }
    updateTcpWarn();
  }
  updateNoPostNote();
});
$('#dr-record').addEventListener('change', () => $('#dr-record-opts').classList.toggle('hidden', !$('#dr-record').checked));
// 補助のスイッチ(チェックボックス)。変えたらすぐ送る(走行中も反映)
const DR_FLAGS = [['#dr-esc-reverse', 'esc_reverse'], ['#dr-recovery', 'recovery'], ['#dr-start-kick', 'start_kick']];
// 送った項目の返事の順番(2026-10-26 fix3 r1-state3-1)。drAckAt[項目] = その項目の送信の返事が来たときの drSync.clock。表示の
// 問い合わせ(pollDrive)はその前の clock を since として持ち、since より後に返事が来た項目は、その問い合わせの値で画面を戻さない
// (問い合わせがサーバーに届いたのが送信より前なら、古い値のことがある)
const drAckAt = {}, drSync = { clock: 0 };
const drAcked = (keys) => { const c = ++drSync.clock; for (const k of keys) drAckAt[k] = c; };
// 画面の欄の値をサーバーへ送る(2026-10-26 fix3 r1-state3-2)。送る前に keys の pending(ブロック・基準の設定で入れて、まだ
// 送っていない印)を消し、送れなかった(通信の途切れ・400 など)ら元に戻す。前は印を消したまま失敗したので、次の表示の同期
// (約 1 秒後)で、ブロックで OFF にした欄(押していない自動合わせも)にサーバーに残った前の本の値が黙って戻り、走行開始で
// それを送っていた(天井の ON の送信が失敗 → 自動合わせ ON を送る)。途中で基準の設定・取り消し・走行開始(pending を作り直す)
// をしたときは、そちらを正とする(戻さない)
// 送信中の印(2026-10-27 fix5 反証 4 回目 t2): drPostN[項目] = その項目を送っていて、まだ返事が来ていない数(下の drBusy と同じ文で
// 作る)。返事を待つ間に届いた表示の問い合わせの返事(押す前のサーバーの値)では、その項目の欄・スイッチを戻さない(setDriveParams の
// keep。スライダーの drSending と同じ考え)。前は返事が来たあとの印(drAckAt)しか無く、2-4(天井 + 自動合わせ)のあと 2-5 で天井を
// ON にした送信の途中に古い問い合わせの返事が来ると、自動合わせのスイッチが前の本の ON に約 1 秒戻り、その間の ▶ で自動合わせ ON を
// 送った(発進の前進の ON も、同じ形で OFF に戻って OFF を送った)
const postDriveParams = async (body, keys) => {
  const pend = state.drive.pending, had = keys.filter((k) => pend[k]), sent = Object.keys(body);
  for (const k of keys) delete pend[k];
  for (const k of sent) drPostN[k] = (drPostN[k] || 0) + 1;
  try { return await api('POST', '/api/drive/params', body); }
  catch (e) { if (state.drive.pending === pend) for (const k of had) pend[k] = true; throw e; }
  finally { for (const k of sent) drPostN[k] = Math.max(0, (drPostN[k] || 0) - 1); drAcked(sent); }
};
$('#dr-tof-guard').addEventListener('change', async () => {
  try { await postDriveParams({ tof_guard: +$('#dr-tof-guard').value }, ['tof_guard']); } catch (e) { toast(e.message, true); }
});
$('#dr-tcp-mode').addEventListener('change', async () => {
  try { await postDriveParams({ tcp_mode: +$('#dr-tcp-mode').value }, ['tcp_mode']); } catch (e) { toast(e.message, true); }
  updateTcpWarn();
});
function updateTcpWarn() {
  // 出し方 1〜3 の目標速度の追従には速度ゲインも掛かる(速度の上限はゲインで割り戻すので、ゲインに関係なく効く)
  $('#dr-tcp-warn').classList.toggle('hidden', !(+$('#dr-tcp-mode').value >= 1 && Math.abs(+$('#dr-throttle-gain').value - 1) > 1e-6));
}
$('#dr-throttle-gain').addEventListener('input', updateTcpWarn);
// 後処理の無いモデルで、速度の上限・TCP の出し方が設定に残っているとき: 効かないことを出す(スライダーは tcp のときだけ表示)
function updateNoPostNote() {
  const m = drModel(), st = state.drive.st || {}, p = st.params || {};
  const hasPost = !!(m && m.postprocess);
  const cap = +(p.speed_cap || 0), mode = +(p.tcp_mode || 0);
  const show = !!m && !hasPost && (cap > 0 || mode >= 1);
  $('#dr-nopost-note').classList.toggle('hidden', !show);
  if (show) {
    $('#dr-nopost-text').textContent = `${cap > 0 ? `速度の上限 ${cap.toFixed(2)} m/s` : ''}${cap > 0 && mode >= 1 ? '・' : ''}`
      + `${mode >= 1 ? `TCP の出し方(${TCP_MODES[mode] || mode})` : ''}は、このモデル(後処理なし)では効きません。`;
  }
}
$('#btn-dr-nopost-reset').addEventListener('click', async () => {
  try {
    const p = await postDriveParams({ speed_cap: 0, tcp_mode: 0 }, ['speed_cap', 'tcp_mode']);
    if (state.drive.st) state.drive.st.params = p;
    setDriveParams(p);
  } catch (e) { toast(e.message, true); }
  updateNoPostNote();
});
for (const [sel, key] of DR_FLAGS) {
  $(sel).addEventListener('change', async () => {
    const el = $(sel), on = el.checked, st = state.drive.st || {};
    // 発進の補助は走行中に ON にできない(drive.set_params も断る。2026-10-26 fix3 r1-rs1-2: 走行中に 0 → 1 にすると、2 秒の窓が
    // AI の区間の途中で新しく開き、リカバリーの中止・後退の途中を上書きして前進を出していた)。OFF にするのは走行中でもできる
    if (key === 'start_kick' && on && st.running) {
      el.checked = false;
      toast('走行中は発進の補助(AI側にしたら走り出す)を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)', true);
      return;
    }
    // 送れなかったら押す前に戻す(ほかのスイッチと同じ。pending も postDriveParams が戻す。2026-10-26 fix3 r1-state3-2)
    try { await postDriveParams({ [key]: on ? 1 : 0 }, [key]); } catch (e) { toast(e.message, true); el.checked = !on; }
  });
}
function readDriveParams() {
  const p = {};
  for (const [sel, key] of DR_PARAMS) p[key] = +$(sel).value;
  for (const [sel, key] of DR_FLAGS) p[key] = $(sel).checked ? 1 : 0;
  p.tof_guard = +$('#dr-tof-guard').value;
  p.tcp_mode = +$('#dr-tcp-mode').value;
  p.zone_caps = $('#dr-zone-caps').checked ? 1 : 0;     // 区間の速度上限(古いサーバーは知らない項目として読み飛ばす)
  p.speed_ceiling = $('#dr-speed-ceiling').checked ? 1 : 0;                              // 速さの天井(2026-10-25)
  p.speed_auto = $('#dr-speed-ceiling').checked && $('#dr-speed-auto').checked ? 1 : 0;  // 自動合わせは天井が ON のときだけ
  p.launch_boost = $('#dr-launch-boost').checked ? 1 : 0;                                  // 発進の前進(2026-10-26)
  return p;
}
// スライダーは動かした項目だけを送る(別の画面の値を巻き戻さない)。操作中・送信中でなければサーバーの値に合わせる。
// 送信中の印は数で持つ(2026-10-26 fix3 r1-state3-1)。前は「送信中」の 1 ビットを送るたびに立てて返事で消したので、±0.01 を 2 回
// 押したとき、1 回目の返事が 2 回目のタイマー(120 ms)より先に戻ると印が消え、その間の表示の同期でスライダーがサーバーの古い値に
// 戻り、2 回目はその値を送った(0.92 のつもりが 0.91。確認の文も 0.91)。今は動かすたびに drSeq を 1 増やし、送った返事がその番号に
// 届くまで送信中のまま。同じ項目の送信は 1 本ずつ(前の返事を待ってから、そのときのスライダーの値を送る)。送信の返事より前に
// 出た表示の問い合わせの返事でも戻さない(drAckAt と pollDrive の since)
// drPostN = postDriveParams(スイッチ・選ぶ欄)の項目ごとの送信中の数(2026-10-27 fix5 t2。上の postDriveParams の注)。これも drBusy と同じ文
const drBusy = {}, drTimers = {}, drSeq = {}, drAck = {}, drInflight = {}, drAgain = {}, drPend = {}, drPostN = {},
  drSending = (key) => (drSeq[key] || 0) !== (drAck[key] || 0);      // drBusy と同じ文(関数を切り出して試す台本も読めるように)
async function drSend(key, el) {
  if (drInflight[key]) { drAgain[key] = true; return; }      // 前の送信の返事を待ってから、もう一度(そのときの値)
  drInflight[key] = true;
  try {
    do {
      drAgain[key] = false;
      const seq = drSeq[key] || 0;
      try { await api('POST', '/api/drive/params', { [key]: +el.value }); }
      catch (e) {
        toast(e.message, true);
        // 送れなかった: ブロック・基準の設定で入れた印(pending)を戻す(そのあと動かしていなければ。r1-state3-2 と同じ)
        if (drPend[key] && drPend[key] === state.drive.pending && seq === (drSeq[key] || 0)) drPend[key][key] = true;
      }
      if (seq === (drSeq[key] || 0)) delete drPend[key];
      drAck[key] = Math.max(drAck[key] || 0, seq);
      drAcked([key]);
    } while (drAgain[key]);
  } finally { drInflight[key] = false; }
}
for (const [sel, key, fmt] of DR_PARAMS) {
  const el = $(sel);
  el.addEventListener('pointerdown', () => { drBusy[key] = (drBusy[key] || 0) | 1; });
  for (const ev of ['pointerup', 'pointercancel', 'change']) el.addEventListener(ev, () => { drBusy[key] = (drBusy[key] || 0) & ~1; });
  el.addEventListener('input', () => {
    $(sel + '-v').textContent = fmt(el.value);
    if (state.drive.pending[key]) drPend[key] = state.drive.pending;     // 送れなかったときに戻す印
    delete state.drive.pending[key];                // 動かした項目は、今まで通りすぐ送る
    drSeq[key] = (drSeq[key] || 0) + 1;
    clearTimeout(drTimers[key]);
    drTimers[key] = setTimeout(() => drSend(key, el), 120);
  });
}
// 速度ゲインの ±0.01 のボタン(2026-10-26 fix3 state-2)。スライダーは 0.01 刻み(0〜2 の 200 段)で、スマホの指では 0.92 などに
// 合わせにくいので、スライダーを 1 段ずつ動かす。動かしたあとはスライダーを動かしたのと同じ(input の処理 = すぐ送る・走行中も)
const GAIN_STEP = 0.01, GAIN_MIN = 0, GAIN_MAX = 2;              // index.html の #dr-throttle-gain・drive.PARAM_LIMITS と同じ
function gainStep(v, d) {                                         // 0.01 の倍数に丸めて範囲の中へ(DOM を使わない。試験で確かめる)
  const x = Math.round(((Number.isFinite(+v) ? +v : 1) + d) / GAIN_STEP) * GAIN_STEP;
  return +Math.min(GAIN_MAX, Math.max(GAIN_MIN, x)).toFixed(2);
}
for (const [id, d] of [['#btn-dr-tg-down', -GAIN_STEP], ['#btn-dr-tg-up', GAIN_STEP]]) {
  if (!$(id)) continue;                                           // 古い index.html(キャッシュ)でも画面全体を止めない
  $(id).addEventListener('click', () => {
    const el = $('#dr-throttle-gain');
    el.value = String(gainStep(el.value, d));
    el.dispatchEvent(new Event('input'));
  });
}
// since = 表示の問い合わせを出したときの drSync.clock(pollDrive だけが渡す)。その後に送信の返事が来た項目は、この値で戻さない
// (問い合わせのほうが古いかもしれない。次の同期で合わせる。2026-10-26 fix3 r1-state3-1)。since なし = 今まで通り。
// スイッチ・選ぶ欄を送っている間(postDriveParams の返事を待つ間。drPostN)も戻さない(2026-10-27 fix5 t2。返事のあとは since で守る)
function setDriveParams(p, since) {
  const pend = state.drive.pending || {};         // 基準の設定で入れて、まだ送っていない項目は戻さない
  const keep = (k) => !!pend[k] || (drPostN[k] || 0) > 0 || (since !== undefined && (drAckAt[k] || 0) > since);
  for (const [sel, key] of DR_FLAGS) if (p[key] !== undefined && !keep(key)) $(sel).checked = !!+p[key];
  if (p.tof_guard !== undefined && !keep('tof_guard') && document.activeElement !== $('#dr-tof-guard')) $('#dr-tof-guard').value = String(p.tof_guard);
  if (p.tcp_mode !== undefined && !keep('tcp_mode') && document.activeElement !== $('#dr-tcp-mode')) $('#dr-tcp-mode').value = String(p.tcp_mode);
  if (p.zone_caps !== undefined && !keep('zone_caps')) $('#dr-zone-caps').checked = +p.zone_caps === 1;
  if (p.speed_ceiling !== undefined && !keep('speed_ceiling')) $('#dr-speed-ceiling').checked = +p.speed_ceiling === 1;
  if (p.speed_auto !== undefined && !keep('speed_auto')) $('#dr-speed-auto').checked = +p.speed_auto === 1;
  if (p.launch_boost !== undefined && !keep('launch_boost')) $('#dr-launch-boost').checked = +p.launch_boost === 1;
  for (const [sel, key, fmt] of DR_PARAMS) {
    if (p[key] === undefined || drBusy[key] || drSending(key) || keep(key)) continue;
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
// 画面の版と、この画面が読み込んだ app.js の更新時刻(2026-10-27 fix3 r2-r2-state-3)。版は走行開始の body の client と同じ数
// (server.py の DRIVE_CLIENT_VERSION)。更新時刻は index.html が付ける「app.js?v=秒」(server.py の index)を、読み込んだときの
// script の src から読む(読めなければ 0 = サーバーは版だけを見る)。走行開始で build として送り、▶ の横に「画面の版 N(app.js
// MM/DD HH:MM)」と出す(手順書 3 章の「新しい画面の印」。前は 20:25〜23:11 の画面と見分けられなかった)。
// 9 = 2026-10-27 fix5(反証 4 回目 t1・t2・t3): 走行開始で画面が読んだサーバーの起動の時刻(boot_at)を送る(サーバーが起き直していれば
// 409)・表示の問い合わせが通らない間は ▶・基準の設定・ブロックを断る・スイッチの送信中は表示の同期で戻さない
const DRIVE_CLIENT = 9;
const PAGE_BUILD = (() => {
  try {
    const m = /[?&]v=(\d+)/.exec((document.currentScript && document.currentScript.src) || '');
    return m ? +m[1] : 0;
  } catch (e) { return 0; }
})();
(() => {
  const el = $('#dr-client-ver');
  if (!el) return;                                    // 古い index.html(キャッシュ)
  const d = new Date(PAGE_BUILD * 1000), p2 = (x) => String(x).padStart(2, '0');
  el.textContent = `画面の版 ${DRIVE_CLIENT}` + (PAGE_BUILD ? `(app.js ${p2(d.getMonth() + 1)}/${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())})` : '');
})();
// 走行開始を断られた・届かなかった文の出し直し(2026-10-27 fix6 w1-ui-ui61-2 の文の形のまま関数にした。fix7 で noteServerBoot も使う)。
// st: 同期した status(止まっていれば頭に「走行は始まっていません(停止中)。」)。m: 断った文。bootRefusal: 起き直しの 409 の文なら
// それだけ、ほかの断りなら後ろに起き直しの知らせ(state.drive.bootToast)を付ける
function startRefusalText(st, m, bootRefusal) {
  return (st && !st.running ? '走行は始まっていません(停止中)。' : '') + (bootRefusal ? m : `${m.replace(/。$/, '')}。${state.drive.bootToast}`);
}
$('#btn-dr-start').addEventListener('click', async () => {
  const m = drModel();
  if (!m) { toast(drModelsMissing() ? DR_MODELS_MISS : 'モデルを選んでください', true); return; }     // 一覧を読めていない(fix7)
  // 画面を開いた直後、サーバーの設定(トリム・ゲイン・スイッチ)を一度も読んでいないうちは始めない(読む前の欄は index.html の
  // 初めの値 = トリム 0・速度 1.00 で、それを送ってしまう。2026-10-26 fix3)。表示の問い合わせが失敗したあと(サーバーが落ちた・
  // 起き直している)も、次に同期するまで始めない(drNotSynced。2026-10-27 fix5 t1)
  if (drNotSynced()) return;
  const dry = $('#dr-dry').checked;
  // 「モーターに出力しない」のまま、説明欄がベンチマークのブロックの本(block=・official=1)のときは聞く(2026-10-27 fix3 直す 3 回目
  // r3-r3-state-2: dry では下の確認の文が出ないので、朝の dry run の印が残ったままブロックの本を ▶ すると、推論の確認で黙って始まった)
  const dDesc = $('#dr-desc').value;
  if (dry && (descField(dDesc, 'block') || /(^|\s)official=1(\s|$)/.test(dDesc))
      && !confirm('「モーターに出力しない」が ON です(推論の確認。出力は中立で、CH3 を AI 側にしても車は動きません)。\n'
        + `説明欄はベンチマークのブロックの本です(block=${descField(dDesc, 'block') || '?'})。\n\n`
        + 'このまま推論の確認で始めますか?(本番の走行なら キャンセル →「モーターに出力しない」を外してから ▶)')) return;
  const isTcp = !!(m.postprocess && m.postprocess.kind === 'tcp_controller');
  const tcpMode = +$('#dr-tcp-mode').value, cap = +$('#dr-speed-cap').value;
  // TCP の出し方 1〜3 は机上の比較で評価していない(制御器のスロットルの写像はまだ粗い)ので、始める前に確かめる
  const tcpLine = isTcp && tcpMode >= 1 ? `\n\n注意: TCP の出し方が「${TCP_MODES[tcpMode] || tcpMode}」です。比較で評価したのは「操作の枝だけ」です(車で試すのはこちらを勧めます)。` : '';
  const capLine = isTcp && cap > 0 ? `\n速度の上限 ${cap.toFixed(2)} m/s` : '';
  const zoneLine = $('#dr-zone-caps').checked ? `\n区間の速度上限: ON(${zoneSpecText(state.drive.st)})${zoneUnknownCap(state.drive.st) != null ? '\n  位相が不確かなとき(1 周目のほぼ全部)も上限が掛かります' : ''}` : '';
  const prepLine = prepStartLines();
  const speedLine = speedStartLines(state.drive.st) + launchStartLine(state.drive.st);
  const trimLine = trimGuardStartLines(state.drive.st) + revStartLine();
  if (!dry && !confirm(`AI走行を開始します。\nモデル: ${m.name}\nジャンル: ${modeName(+$('#dr-mode').value)} / 速度ゲイン ${(+$('#dr-throttle-gain').value).toFixed(2)}${trimLine}${capLine}${zoneLine}${speedLine}${tcpLine}${prepLine}\n\nプロポのCH3をAI側(LED紫)にすると車が動きます。周りに人や物がないことを確認してください。\nこの画面は開いたままにしてください(閉じる・スリープ・通信断が5秒続くと止まります)。`)) return;
  // client: 画面の版(server.py の DRIVE_CLIENT_VERSION。7 = 2026-10-26 の画面の直し。fix3 state-5。8 = 2026-10-26 夜〜10/27 の直し
  // (r1-state3-1・3-2・rs1-2・state3-4・r2-r2-state-1・2・3)。9 = 2026-10-27 fix5(t1・t2・t3)。古い画面からは走行を始めない)。build: この画面の app.js の更新時刻
  // (PAGE_BUILD。サーバーは今のファイルより古ければ断る = 版を上げ忘れても古い画面で走らない。2026-10-27 fix3 r2-r2-state-3)。
  // boot_at: この画面が最後に読んだ status の boot_at(drive.Driver を作った時刻)。サーバーは今の起動の時刻と違えば断る(409)。確認の文の
  // 見張り・残す・補助なし・天井の左・発進の前進の強さは走行開始で送らずサーバーの値のままなので、起き直す前の表示(state.drive.st)の
  // ままの文で、既定に戻った値で走らないように(2026-10-27 fix5 反証 4 回目 t1。確認の文を読んでいる間に起き直したときも)
  const body = { client: 9, build: PAGE_BUILD, boot_at: state.drive.bootAt ?? null, run: m.name, mode_id: +$('#dr-mode').value, dry_run: dry,
                 record: $('#dr-record').checked, description: $('#dr-desc').value || `AI走行 ${m.name}`, tags: parseTags($('#dr-tags').value),
                 params: readDriveParams() };
  const seq = state.drive.startSeq = (state.drive.startSeq || 0) + 1;     // この ▶ の番号(下の断りの文の出し直しで使う。fix6)
  try {
    const r = await api('POST', '/api/drive/start', body);
    setDriveToken(r.token);
    benchSent();                                    // 基準の設定で入れた値は、走行開始で送って保存された
    toast(dry ? '推論の確認を開始しました(出力は中立)' : (r.ai_no ? `AI走行 No.${r.ai_no} を開始しました` : '走行を開始しました')); pollDrive();
    if (r.plan_diffs && r.plan_diffs.length) toast(`ブロックの計画と違うので「試し」(cand=EXP)として記録します: ${r.plan_diffs.join('、')}`, true);
  } catch (e) {
    toast(e.message, true);
    // 断られた・届かなかった: すぐ同期して、表示と確認の文を今のサーバーに(t1)。同期の返事で起動し直しの知らせ(noteServerBoot の
    // トースト)が出ると、断った文(「…始めません」)が数十 ms で消え、見える文に走行を始めていないことが出なかった(2026-10-27 fix6
    // 反証 w1-ui-ui61-2: 起き直しの 409 のあと「…既定に戻っています」だけが残った)。同期のあと、起き直しで断った文(server.py の
    // _boot_refusal)か、同期のあいだに知らせが出たときは、断った文をもう一度出す(どの返事の順でも最後に残る文が同じになるように)。
    // 頭に、同期した status で止まっていれば「走行は始まっていません(停止中)。」。断った文に起動し直しが無ければ知らせを後ろに付ける。
    // ほかの断り(古い画面・学習中など)で知らせが出ないときは前と同じ。同期を待つあいだに次の ▶ を押したときは出し直さない(前の ▶ の
    // 文で、次の ▶ の知らせ = 走り出した知らせを消さない。遅れて届いた古い返事は止まっている)。表示だけ(送った body・断る決まり・同期は前と同じ)
    // 同期も通らなかった(サーバーが落ちたまま・起き直している途中・通信の途切れ)ときは、知らせはまだ出ていないので、断った文を覚えて
    // おき、次に同期が通って起き直しの知らせ(noteServerBoot)が出たところで、まだ次の ▶ を押していなければ 1 回だけ同じ形で出し直す
    // (2026-10-27 fix7 反証 w2-ui-w2-ui-r72-ui-1: 前は同期が通る場合だけで、落ちている間に OK を押すと、あとの知らせ「…既定に戻って
    // います」だけが最後に残った。起き直しの 409 のあと同期だけ途切れたときも、知らせが出し直した文を消した)。次に同期が通っても知らせが
    // 出なければ(起き直していない。Wi-Fi の途切れだけ)覚えた文は捨てる(前と同じ: 断った文がそのまま残る)
    const nBoot = state.drive.bootToasts || 0;
    let synced = false;
    try { synced = await pollDrive(); } finally {
      const m = String(e.message), bootRefusal = /^管理画面のサーバーが .*に起動し直しました\(/.test(m);
      if (seq === state.drive.startSeq && (bootRefusal || (state.drive.bootToasts || 0) !== nBoot)) {
        toast(startRefusalText(state.drive.st, m, bootRefusal), true);
      }
      if (!synced && seq === state.drive.startSeq && (state.drive.bootToasts || 0) === nBoot) state.drive.startRefused = { seq, m, bootRefusal };
    }
  }
});
// サーバーの設定をまだ読んでいない・表示の問い合わせが失敗したあと次に同期していない: 知らせて true(▶・基準の設定・ブロックは断る)。
// 文は、一度も同期していない(画面を開いた直後)なら前と同じ「読み込んでいます」、同期したあとに失敗したなら「つながっていません」
function drNotSynced() {
  if (state.drive.paramsLoaded) return false;
  toast(state.drive.lostAt && state.drive.synced ? 'サーバーにつながっていません(表示の問い合わせが通りません)。表示が戻って同期してから押してください。'
    + '「起動し直しました」が出たら、止まっているときに朝の設定を入れ直す(手順書 7 章)'
    : 'サーバーの設定を読み込んでいます。少し待ってから押してください', true);
  return true;
}
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
  const since = drSync.clock;         // この問い合わせのあとに返事が来た送信の項目は、この返事で戻さない(r1-state3-1)
  const gen = state.drive.drTabGen;   // 走行タブに戻った番号(loadDrive。fix8)。返事までに変わったら、戻る前に出した問い合わせ
  // 止まっているとき(drIdle)だけ、返事を DR_IDLE_WAIT_MS で打ち切って失敗にする(drLost と次の問い合わせ。2026-10-27 fix6
  // w2-ui-w1-ui-r62-1)。前は返事が来ないまま止まると輪も止まり、drLost も呼ばれず、表示は前のまま ▶ も押せた。本文を読む途中で
  // 打ち切ったとき(api は本文なしの null を返す)も失敗にする。走行中・token のある画面は打ち切らない(生存確認の問い合わせは前と同じ)
  const ac = drIdle() && typeof AbortController === 'function' ? new AbortController() : null;
  const cut = ac ? setTimeout(() => ac.abort(), DR_IDLE_WAIT_MS) : null;
  try {
    st = await api('GET', '/api/drive/status' + tok, undefined, ac ? ac.signal : undefined);
    if (ac && ac.signal.aborted) throw new Error('打ち切り');
  } catch (e) { st = null; drLost(); }
  if (ac) clearTimeout(cut);
  if (st && !st.running && state.drive.token) setDriveToken(null);
  if (st) renderDrive(st, since);
  // 走行タブに戻る前に出した問い合わせの返事(戻った直後に届いた)は、表示はするが「読み直した」に数えない(次の問い合わせまで ▶・
  // 基準の設定・ブロックは断ったまま。2026-10-28 fix8 stale-status-after-drive-tab-return)
  if (st && gen !== state.drive.drTabGen && drIdle()) state.drive.paramsLoaded = false;
  if (st) retryDriveModels();         // 走行タブを開いたときにモデルの一覧を読めなかったら、2 回目の同期で 1 回読み直す(fix7)
  const running = st ? st.running : (state.drive.st && state.drive.st.running);
  if (running || state.tab === 'drive') driveTimer = setTimeout(pollDrive, running ? 250 : 1000);
  return !!st;                        // 同期できたか(▶ の catch が使う。fix7)
}
// 表示の問い合わせが失敗した(サーバーが落ちた・起き直している・通信の途切れ。2026-10-27 fix5 反証 4 回目 t1): 次に同期する
// (renderDrive が params を読む)まで、サーバーの設定を読んでいない扱いにする(▶・基準の設定・ブロックは drNotSynced で断る)。
// 前は表示が古いまま(st = null だけ)で、起き直した直後・確認の文を読んでいる間に起き直したときに、確認の文は起き直す前の値
// (見張り ON・左 4.2・200 µs)のまま、車は既定の値(見張り OFF・左 4.1・250 µs)で走った。走行中は止める操作を変えない
function drLost() {
  state.drive.paramsLoaded = false;
  state.drive.lostAt = Date.now();
  const st = state.drive.st;
  if (st && st.running) return;
  const b = $('#btn-dr-start');
  b.disabled = true; b.title = 'サーバーにつながっていません';
  const d = new Date(state.drive.lostAt), p2 = (x) => String(x).padStart(2, '0');
  $('#dr-message').innerHTML = `<div class="warn-text">サーバーにつながっていません(表示の問い合わせが ${p2(d.getHours())}:${p2(d.getMinutes())}:`
    + `${p2(d.getSeconds())} に失敗)。つながって表示が戻るまで、走行開始・基準の設定・ベンチマークは押せません</div>`;
}
const PHASE = { idle: ['', '停止中'], starting: ['warn', '準備中'], driving: ['ok', '走行中'], error: ['warn', 'エラー'] };
function gaugePos(us) { return `${Math.max(0, Math.min(100, (us - 1000) / 10))}%`; }
// 管理画面のサーバーの起動し直し(2026-10-26 fix3 直す 1 回目 r1-state3-4): status の boot_at(drive.Driver を作った時刻)が、この画面で
// 前に見た値から変わったら知らせる。起動のたびに既定へ戻る項目(drive.RESET_ON_BOOT: 試験の選択肢は OFF・天井の値・発進の前進の値
// など)が、朝に入れた値のつもりのまま走らないように(前はスイッチが黙って OFF に戻り、確認の文から行が消えるだけだった)。
// 表示だけ(送る値・スイッチは変えない)。boot_at の無いサーバー(再起動の前の古い版)では何もしない。画面を読み直すと消える
// 文はそのときのサーバーの値(st.params)から作る(2026-10-27 fix7 反証 w1-ui-r7-ui-1): 前は値を見ずに、トーストは「…既定に戻って
// います」、走行タブの文は試験の選択肢を全部「…は OFF」と書き、天井の左・発進の前進の強さは今の値を既定の値のように出した。ライブの
// タブにいる間に起き直し、走行タブを開く前に curl・別の画面で朝の設定を入れ直すと、ON を OFF と書き、4.2・200 を既定と読ませた
// (知らせのとおり見張りを「入れ直す」つもりで押すと OFF になる)。今は、起動のたびに既定に戻る項目(drive.RESET_ON_BOOT)が全部既定
// なら前とまったく同じ文。既定と違う項目があれば、トーストは今の値を挙げて「スイッチと確認の文で読む」、走行タブの文は ON の選択肢を
// 「ON(起動のあとに入れ直し済み)」、天井の左・発進の前進の強さを「今 X(既定 Y)」と書く。走行タブの文は表示のたびに今の値から作り直す
// (知らせのあとに入れ直したときも、文がスイッチと同じ)。送る物・断る決まり・スイッチ・確認の文は変えない
// 既定の値は drive.DEFAULT_PARAMS の RESET_ON_BOOT の分と同じ(tests/test_ui_fix7.py が照らす)。並びは文に出す順
const BOOT_DEFAULTS = { rec_back_guard: 0, rec_hold_keep: 0, launch_no_kick: 0, ceil_left_mps: 4.1, launch_us: 250.0,
  speed_ceiling: 0, ceil_all_mps: 4.5, speed_auto: 0, auto_target_mps: 4.27, launch_boost: 0, launch_mps: 2.5, launch_max_s: 1.2,
  zone_caps: 0, tcp_mode: 0, speed_scale: 1.0 };
const BOOT_FLAGS = ['rec_back_guard', 'rec_hold_keep', 'launch_no_kick', 'speed_ceiling', 'speed_auto', 'launch_boost', 'zone_caps'];
const BOOT_UNITS = { ceil_left_mps: ' m/s', launch_us: ' µs', ceil_all_mps: ' m/s', auto_target_mps: ' m/s', launch_mps: ' m/s', launch_max_s: ' 秒' };
const bootDiffers = (p, k) => p[k] !== undefined && p[k] !== null && !(Math.abs(+p[k] - BOOT_DEFAULTS[k]) < 1e-9);
const BOOT_NAMES = { ceil_left_mps: '天井の左旋回中', ceil_all_mps: '天井の全体' };     // ほかは試験の選択肢の名前か BENCH_LABEL
function bootItemText(k, v) {                         // 既定と違う項目の 1 つ(「リカバリーの後ろの見張り ON」「天井の左旋回中 4.2 m/s」)
  const name = BOOT_NAMES[k] || (OPT_SWITCHES.some((o) => o.key === k) ? optName(k) : (BENCH_LABEL[k] || k));
  if (BOOT_FLAGS.includes(k) && (+v === 0 || +v === 1)) return `${name} ${+v === 1 ? 'ON' : 'OFF'}`;
  if (k === 'tcp_mode') return `${name}「${TCP_MODES[v] || v}」`;
  return `${name} ${v}${BOOT_UNITS[k] || ''}`;
}
function bootToastText(hm, p) {
  const diff = Object.keys(BOOT_DEFAULTS).filter((k) => bootDiffers(p, k)).map((k) => bootItemText(k, p[k]));
  if (!diff.length) return `管理画面のサーバーが ${hm} に起動し直しました。朝に入れた設定(後ろの見張り・左の値・発進の前進の強さなど)は既定に戻っています`;
  return `管理画面のサーバーが ${hm} に起動し直しました。起動のあとに入れ直した設定があります(今は ${diff.join('・')})。`
    + '朝の設定はスイッチと走行開始の確認の文で読む';
}
function bootNoteText(hm, p) {
  const has = OPT_SWITCHES.filter((o) => p[o.key] !== undefined && p[o.key] !== null);
  const on = has.filter((o) => +p[o.key] === 1).map((o) => o.name), off = has.filter((o) => +p[o.key] === 0).map((o) => o.name);
  const odd = has.filter((o) => +p[o.key] !== 0 && +p[o.key] !== 1).map((o) => `${o.name}は ${p[o.key]}`);
  const opts = (on.length ? [`${on.join('・')}は ON(起動のあとに入れ直し済み)`] : []).concat(off.length ? [`${off.join('・')}は OFF`] : [], odd);
  const val = (k, name, unit) => (bootDiffers(p, k) ? `${name} 今 ${p[k]} ${unit}(既定 ${BOOT_DEFAULTS[k]})` : `${name} ${p[k] ?? '-'} ${unit}`);
  const shown = new Set(OPT_SWITCHES.map((o) => o.key).concat(['ceil_left_mps', 'launch_us']));
  const other = Object.keys(BOOT_DEFAULTS).filter((k) => !shown.has(k) && bootDiffers(p, k)).map((k) => bootItemText(k, p[k]));
  return `管理画面のサーバーが ${hm} に起動し直しました。起動のたびに既定に戻るもの: `
    + (opts.length ? `${opts.join('、')}、` : '')
    + `${val('ceil_left_mps', '天井の左旋回中', 'm/s')}、${val('launch_us', '発進の前進の強さ', 'µs')} など`
    + (other.length ? `(ほかに今は既定と違うもの: ${other.join('・')})` : '') + '。'
    + '朝に入れた値は、止まっているときに入れ直し、スイッチと走行開始の確認の文で読む(手順書 7 章)。画面を読み直すとこの文は消えます';
}
function noteServerBoot(st) {
  // ▶ の catch の同期が通らなかったときに覚えた断りの文(2026-10-27 fix7 反証 w2-ui-w2-ui-r72-ui-1)。同期が通ったらここで捨て、
  // 起き直しの知らせを出したときだけ、まだ次の ▶ を押していなければ知らせのあとに出し直す(同期が通る場合の ▶ の catch と同じ文)
  const rs = state.drive.startRefused;
  state.drive.startRefused = null;
  const b = st && st.boot_at;
  if (b === undefined || b === null) return;
  const prev = state.drive.bootAt;
  state.drive.bootAt = b;
  if (prev === undefined || prev === null || prev === b) return;
  const d = new Date(+b * 1000), p = st.params || {};
  const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  state.drive.bootHm = hm;                            // 走行タブの文(bootNoteText)は renderDrive が表示のたびに今の st.params で作る
  const msg = bootToastText(hm, p);
  state.drive.bootToast = msg;                        // 走行開始の断りの文を出し直すときに使う(▶ の catch。2026-10-27 fix6)。トーストと同じ文
  state.drive.bootToasts = (state.drive.bootToasts || 0) + 1;
  toast(msg, true);
  if (rs && rs.seq === state.drive.startSeq) toast(startRefusalText(st, rs.m, rs.bootRefusal), true);
}
function renderDrive(st, since) {
  const prev = state.drive.st;
  state.drive.st = st;
  // 試験の選択肢(スイッチを押すとすぐ送る。走行開始では送らない)は、送信の返事より前に出た問い合わせの返事(since より後に返事が
  // 来た項目)なら、前の値(送信の返事の値)のままにする。スイッチ(renderOptSwitches)と走行開始の確認の文(trimGuardStartLines)は
  // st.params を読むので(2026-10-27 fix3 r2-r2-state-2。setDriveParams の since と同じ考え。次の問い合わせで合う)
  if (since !== undefined && prev && prev.params && st.params) {
    for (const o of OPT_SWITCHES) {
      if ((drAckAt[o.key] || 0) > since && prev.params[o.key] !== undefined) st.params[o.key] = prev.params[o.key];
    }
  }
  noteServerBoot(st);                 // サーバーの起動し直しの知らせ(2026-10-26 fix3 r1-state3-4。表示だけ)
  if (st.running && Object.keys(state.drive.pending).length) benchSent();     // 走り出したらサーバーの値が正
  if (st.params) { setDriveParams(st.params, since); state.drive.paramsLoaded = true; state.drive.synced = true; state.drive.lostAt = null; }
  if (prev && prev.running && !st.running && state.tab === 'scenes') loadScenes();
  const [cls, txt] = PHASE[st.phase] || ['', st.phase];
  const ph = $('#dr-phase');
  ph.className = 'badge ' + cls;
  ph.textContent = (st.phase === 'driving' && st.dry_run ? '推論の確認中' : txt) + (st.ai_no ? ` ・ AI走行 No.${st.ai_no}` : '');
  $('#dr-message').innerHTML = esc(st.message || '') + (st.error ? `<div class="warn-text">${esc(st.error)}</div>` : '')
    + (st.recording_note ? `<div class="muted">${esc(st.recording_note)}</div>` : '')
    + (st.running && st.recording ? '<div><span class="badge warn">● 収集中</span></div>' : '')
    + (st.running && !st.controller ? '<div class="warn-text">別の画面(またはこの画面の以前の表示)が走行を操作しています。この画面を閉じても走行は止まりません。止めるときは停止ボタンを押してください。</div>' : '')
    + (state.drive.bootHm ? `<div class="warn-text">${esc(bootNoteText(state.drive.bootHm, st.params || {}))}</div>` : '');
  const now = st.running ? st.now : null;
  const who = $('#dr-who');
  if (now) {
    if (now.hold) who.textContent = now.select ? 'CH3: AI側 ・ 中立で停止中' : 'CH3: 人側 ・ 中立で待機中';
    else if (now.assist === 'リカバリー中止') who.textContent = 'CH3: AI側 ・ リカバリーをやめて中立(CH3を人側へ戻して助けてください)';
    else if (now.assist === 'リカバリー') who.textContent = 'CH3: AI側 ・ ぶつかったので自動で下がっています';
    else if (now.assist === '発進の前進') who.textContent = 'CH3: AI側 ・ 発進の前進(強い前進を出しています)';
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
              ['ToFガード', `${now && now.ttc != null ? `衝突まで ${now.ttc.toFixed(2)} 秒` : '-'}`
                + `${now && now.guard ? ` ・ <b class="warn-text">${esc(now.guard)}</b>` : ''} ・ 減速 ${s.guard_slow ?? 0}回 / ブレーキ ${s.guard_brake ?? 0}回`],
              ['補助', `リカバリー ${s.recoveries ?? 0}回 ・ 後退の受け付け ${s.esc_assists ?? 0}回`
                + (s.giveups ? ` ・ <span class="warn-text">中止 ${s.giveups}回</span>` : '')
                + (now && now.assist ? ` ・ <b class="warn-text">${esc(now.assist)}中</b>` : '')],
              ['走行時間', fmtDur(st.elapsed_s)],
              ['AI の時間', aiTimeText(s.ai_s)]);
    const L = st.laps || {};
    if (L.laps !== undefined) {
      rows.push(['周回(ジャイロ)', `${L.laps}周${L.direction ? `(${L.direction})` : ''} ・ 今の周 ${Math.min(359, L.progress_deg || 0)}°`],
                ['ベスト(ぶつからなかった周)', L.best_ai_s != null ? `<b>${L.best_ai_s.toFixed(2)} 秒</b> ・ 中央値 ${L.median_ai_s.toFixed(2)} 秒(${L.clean_laps}周)` : (L.timing ? '-' : '1周目で区切りの位置を決めています')],
                ['AIだけの周', L.ai_laps ? `${L.ai_laps}周 ・ 中央値 ${L.median_ai_all_s.toFixed(2)} 秒(ぶつかった周も含む)` : '-'],
                ['最近の周', (L.recent || []).slice().reverse().map((x) =>
                  `<span class="lap ${x.best ? 'best' : ''} ${x.clean ? '' : 'muted'}">${x.time_s.toFixed(1)}${x.first ? '(最初)' : ''}${x.ai_only ? '' : '(人)'}${x.impacts ? `(衝${x.impacts})` : ''}${x.recoveries ? '(R)' : ''}</span>`).join(' ') || '-']);
    }
    if (now && now.erpm !== undefined) rows.push(['モーター', `${now.erpm} eRPM`]);
    if (now && now.post) rows.push(['TCP の制御器', tcpLiveText(now.post)]);
    if (now && now.ens_j) rows.push(['予測の重ね合わせ', `${now.ens_j} フレーム前まで`]);
    if (now && now.zone) rows.push(['区間の速度上限', zoneLiveText(now.zone, s)]);
    if (now && now.ceil) rows.push(['速さの天井', ceilLiveText(now.ceil, s)]);
    if (now && now.auto) rows.push(['速さのゲインの自動合わせ', autoLiveText(now.auto, st)]);
    if (!st.running && st.auto_end) rows.push(['自動合わせ(走り終わり)', autoEndText(st.auto_end)]);
    if (now && now.human) rows.push(['プロポの値', `${now.human[0]} / ${now.human[1]} µs`]);
  }
  $('#dr-stats').innerHTML = rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join('');
  $('#dr-tcp-live').innerHTML = now && now.post ? tcpLiveText(now.post) : (st.running ? '' : '<span class="muted">走行中は予測の目標速度などをここに出します</span>');
  updateTcpWarn();
  updateNoPostNote();
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
  const dm = drModel();
  $('#btn-dr-start').disabled = st.running || !dm || dm.drivable === false || !!training;
  $('#btn-dr-start').title = training ? '学習中は走行できません' : (dm && dm.drivable === false ? `このモデルでは走行できません: ${dm.why_not || ''}` : '');
  $('#btn-dr-stop').disabled = !st.running;
  $('#btn-float-stop').classList.toggle('hidden', !st.running);
  document.body.classList.toggle('driving', !!st.running);
  $('#dr-model').disabled = st.running;
  $('#dr-dry').disabled = st.running; $('#dr-record').disabled = st.running;
  renderZoneSwitch(st);
  renderSpeedSwitches(st);
  renderLaunchSwitch(st);
  renderOptSwitches(st);              // 試験の選択肢(後ろの見張りなど。2026-10-26 fix3)
  // 発進の補助は走行中は OFF にだけできる(サーバーも 0 → 1 を断る。2026-10-26 fix3 r1-rs1-2)
  $('#dr-start-kick').disabled = !!st.running && !(st.params && +st.params.start_kick === 1);
  renderVbatStrip(st);
  renderPrepHints();
  $('#btn-dr-baseline').disabled = !!st.running;
  if (state.bench.shownRunning !== !!st.running) { state.bench.shownRunning = !!st.running; renderBench(); }
  renderAfter(st);
}

// 走り終わりのメモ(2026-10-24、10/23 の課題「電池・電圧・止めた理由が記録に無い」)。ベンチマークの説明欄(bench=)で
// 収集した走行が止まったら出す。POST /api/bench/after がその記録のシーンの説明に書き、夜の集計が重ねて読む。
const AFTER_DONE_KEY = 'jetformer.bench.afterDone';
function afterDone(log) { try { return localStorage.getItem(AFTER_DONE_KEY) === log; } catch (e) { return false; } }
function descField(desc, key) {
  const head = String(desc || '').replace(/(^|\s)memo=.*$/, '');
  const m = head.match(new RegExp(`(?:^|\\s)${key}=(\\S*)`));
  const v = m ? m[1].replace(/^#/, '') : '';
  return /^[_#]*$/.test(v) ? '' : v;
}
function renderAfter(st) {
  const box = $('#dr-after'), log = st && st.record_log;
  const show = !!(log && !st.running && !st.dry_run && /(^|\s)bench=/.test(st.description || '') && !afterDone(log));
  box.classList.toggle('hidden', !show);
  if (show && box.dataset.log === log) fillAfterVoltage(st);       // 走ったあとの基板の電圧は止まってから届く
  if (!show || box.dataset.log === log) return;
  box.dataset.log = log;                          // 記録が替わったときだけ欄を入れ直す(書いている途中の値を消さない)
  const desc = st.description || '', ai = ((st.stats || {}).ai_s) || 0;
  $('#dr-after-log').textContent = `${log}${st.ai_no ? ` ・ AI走行 No.${st.ai_no}` : ''}`;
  $('#dr-after-ai').innerHTML = `AI の時間 ${Math.round(ai)} 秒` + (ai < 240 ? ' <span class="warn-text">(240 秒に届いていません。止めた理由を選んでください)</span>' : '');
  $('#dr-after-battery').value = descField(desc, 'battery');
  $('#dr-after-vb').value = descField(desc, 'V_before');
  $('#dr-after-va').value = '';
  $('#dr-after-reason').value = ai >= 240 ? '240秒走った' : '';
  $('#dr-after-memo').value = '';
  $('#dr-after-va').dataset.auto = '';
  $('#dr-after-vnote').textContent = '';
  fillAfterVoltage(st);
}
// 走り終わりのメモに、基板の電圧(走ったあと)を前もって入れる。人が書いた値・書いている途中の欄は替えない
function fillAfterVoltage(st) {
  const r = st && st.vbat && st.vbat.run, el = $('#dr-after-va');
  if (!r || (r.log && r.log !== st.record_log)) return;
  if (r.v_after != null && el !== document.activeElement && (!el.value || el.dataset.auto === '1')) {
    el.value = (+r.v_after).toFixed(2); el.dataset.auto = '1';
  }
  const f = (x) => (x == null ? '-' : `${(+x).toFixed(2)} V`);
  $('#dr-after-vnote').textContent = r.state === 'waiting' ? '基板の電圧: 止まってから 1 秒たつのを待っています(15 秒まで)'
    : (r.v_before != null || r.v_after != null || r.v_min != null)
      ? `基板の電圧: 走る前 ${f(r.v_before)}・走ったあと ${f(r.v_after)}・走行中の最小 ${f(r.v_min)}(空の欄に入れました。電圧計の値があれば書き替えてください)` : '';
}
$('#dr-after-va').addEventListener('input', () => { $('#dr-after-va').dataset.auto = ''; });
$('#btn-dr-after').addEventListener('click', async () => {
  const log = $('#dr-after').dataset.log;
  const body = { log, battery: $('#dr-after-battery').value.trim(), V_before: $('#dr-after-vb').value, V_after: $('#dr-after-va').value,
                 reason: $('#dr-after-reason').value || null, memo: $('#dr-after-memo').value };
  if (!body.reason) { toast('止めた理由を選んでください', true); return; }
  try {
    const r = await api('POST', '/api/bench/after', body);
    try { localStorage.setItem(AFTER_DONE_KEY, log); } catch (e) { /* 覚えられなくても書けている */ }
    $('#dr-after').classList.add('hidden');
    toast(`記録 ${log} のシーン ${r.scenes} 個にメモを書きました`);
  } catch (e) { toast(e.message, true); }
});
$('#btn-dr-after-skip').addEventListener('click', () => {
  try { localStorage.setItem(AFTER_DONE_KEY, $('#dr-after').dataset.log || ''); } catch (e) { /* 閉じるだけ */ }
  $('#dr-after').classList.add('hidden');
});

// AI の時間(CH3 が AI 側で dry run でない時間の和。drive.py の stats.ai_s。道具の数える時間と同じ決め方)。
// ベンチマークは「数える時間 + 1 分」を過ぎたら次の直線で CH3 を人側へ戻して止める(240 秒 → 5:00)。2026-10-21 Fix-all
const AI_STOP_S = 300;
function fmtMinSec(x) { const t = Math.max(0, Math.floor(+x || 0)); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`; }
function aiTimeText(ai) {
  if (ai === undefined || ai === null) return '-(古いサーバー)';
  const head = `<b id="dr-ai-time">${fmtMinSec(ai)}</b>`;
  if (ai >= AI_STOP_S) return `${head} ・ <b class="warn-text">5:00 を過ぎました: 次の直線で CH3 を人側へ戻して止める(数える 240 秒のとき)</b>`;
  return `${head} <span class="muted">(止める目安: 数える 240 秒 → 5:00・300 秒 → 6:00・180 秒 → 4:00。人の操縦と dry run は入らない)</span>`;
}

// TCP の制御器の今の値(目標速度・上限の後・今の速さ・曲がれる速さの目安・出力を決めた枝)
const TCP_MODES = { 0: '操作の枝', 1: 'ハイブリッド', 2: '混ぜる', 3: '通過点' };
function tcpLiveText(q) {
  const v = (x) => (x == null ? '-' : `${x.toFixed(2)} m/s`);
  const e = (x) => (x == null ? '' : `(${Math.round(x / 0.5e-3)} eRPM)`);
  return `予測の目標速度 <b>${v(q.v_hat)}</b> ・ 上限の後 <b>${v(q.v_cmd)}</b>${q.cap ? `(上限 ${q.cap.toFixed(2)})` : ''}`
    + ` ・ 今 ${v(q.v_now)}${e(q.v_now)} ・ 曲がれる目安 ${v(q.v_curv)} ・ ${esc(TCP_MODES[q.mode] || String(q.mode))}(${esc(q.branch || '')})`;
}

// ------------------------------------------------------------ 走る日の準備(基準の設定・走る前のチェック)・ベンチマークの候補・区間の速度上限
// 「基準の設定」・候補のブロックは、画面の欄に値を入れるだけ(走行は始めない)。入れた値は走行開始(推論の確認も)で送り、
// 今まで通り保存される。欄を動かした項目は、今まで通りすぐ送る。区間の速度上限はここでは ON にしない(スイッチで、確かめてから)
// 基準の値の控え(/api/bench/candidates が無い古いサーバーのとき)。web/bench.py の fallback_baseline と同じ(試験で確かめる)
const BENCH_FALLBACK = {
  run: 'clean_driving_20261011-nocmd-s3-20261011-024626', mode_id: 5, source: '10/18 plan.md 3.4',
  params: { steer_gain: 1.0, throttle_gain: 1.0, steer_trim_us: 0.0, lookahead: 3, osc_percent: 0.0, esc_reverse: 1, recovery: 1,
            rev_us: 300.0, rev_s: 0.4, tof_guard: 1, speed_cap: 0.0, speed_scale: 1.0, tcp_mode: 0, ensemble_j: 0, ensemble_m: 0.5,
            zone_caps: 0, start_kick: 1,
            speed_ceiling: 0, ceil_all_mps: 4.5, ceil_left_mps: 4.1, speed_auto: 0, auto_target_mps: 4.27, auto_vref_v: 8.35,
            launch_boost: 0, launch_us: 250.0, launch_mps: 2.5, launch_max_s: 1.2, rec_back_guard: 0,
            rec_hold_keep: 0, launch_no_kick: 0 },
};
const BENCH_LABEL = { steer_gain: 'ステア', throttle_gain: '速度', steer_trim_us: 'トリム', lookahead: '先読み', osc_percent: '時計補正',
                      esc_reverse: '後退の前に中立', recovery: 'リカバリー', rev_us: '後退の強さ', rev_s: '下がる時間', tof_guard: 'ToF ガード',
                      speed_cap: '上限(speed_cap)', speed_scale: '目標速度の倍率', tcp_mode: 'TCP の出し方', ensemble_j: '重ね合わせ',
                      ensemble_m: '重ね合わせの強さ', zone_caps: '区間の速度上限', start_kick: '発進の補助',
                      speed_ceiling: '速さの天井', ceil_all_mps: '天井(全体)', ceil_left_mps: '天井(左旋回中)',
                      speed_auto: '速さのゲインの自動合わせ', auto_target_mps: '自動合わせの目標', auto_vref_v: '自動合わせの基準の電圧',
                      launch_boost: '発進の前進', launch_us: '発進の前進の強さ', launch_mps: '発進の前進の速さ', launch_max_s: '発進の前進の時間' };
const TOF_GUARD_LABEL = { 0: 'OFF', 1: '記録だけ', 2: '減速', 3: '減速+ハンドル' };
const RISK_BADGE = { low: 'ok', mid: 'warn', high: 'bad' };
const ZONE_MODE_LABEL = { pass: 'そのまま', cap: '上限で切った', neutral: '中立にした', brake: 'ブレーキ' };
state.bench = { data: null, applied: null, prev: null, shownRunning: null };

// 基準の値 → 画面に入れる値(DOM を使わない。tests/test_bench_ui.py が node で確かめる)。
// steer: その日のステア(「この日のステア」の選択。'' なら基準どおり)。区間の速度上限は ON にしない
function benchFieldValues(baseline, steer) {
  const b = baseline || {};
  const p = Object.assign({}, b.params || {});
  const notes = [];
  if (+p.zone_caps === 1) notes.push('区間の速度上限は ON にしていません(ON にするのは G1 に合格し、ユーザーが了承したあと。スイッチで入れてください)');
  p.zone_caps = 0;
  // 速さの天井・自動合わせ(2026-10-25)も、基準の設定・ブロックでは ON にしない(スイッチで、確かめてから)
  for (const [k, name] of [['speed_ceiling', '速さの天井'], ['speed_auto', '速さのゲインの自動合わせ']]) {
    if (+p[k] === 1) notes.push(`${name}は ON にしていません(スイッチで入れてください)`);
    if (p[k] !== undefined) p[k] = 0;
  }
  // 発進の前進(2026-10-26)も ON にしない(スイッチで、確かめてから)。値(launch_us など)はそのまま
  if (+p.launch_boost === 1) notes.push('発進の前進は ON にしていません(スイッチで入れてください)');
  if (p.launch_boost !== undefined) p.launch_boost = 0;
  // トリム(steer_trim_us)は車の左右の直し(car_check の「直す」)。基準の設定・ブロックでは変えない(画面の欄・サーバーの値の
  // まま。2026-10-26 fix2 state2-1: 前は基準の 0 で上書きし、走行開始で黙って 0 を送っていた)。ブロックの params に書いたときだけ入る。
  // リカバリーの後ろの見張り(rec_back_guard)も入れない。今は試験の選択肢のスイッチ(OPT_SWITCHES。2026-10-26 fix3)で入れ、走行開始では
  // 送らない(サーバーの値のまま)。ほかの試験の選択肢(rec_hold_keep・launch_no_kick)も benchApply が f.params から外す(ブロックでも)
  for (const k of ['steer_trim_us', 'rec_back_guard']) delete p[k];
  if (steer !== undefined && steer !== null && steer !== '' && Number.isFinite(+steer)) p.steer_gain = +steer;
  return { run: b.run || '', mode_id: b.mode_id == null ? 5 : +b.mode_id, params: p, notes };
}
// 画面の値と入れる値の違い [[項目, 今, 入れる値], ...](DOM を使わない。tests/test_bench_ui.py が node で確かめる)
function benchDiff(cur, want) {
  const out = [];
  for (const k of Object.keys(want || {})) {
    if (cur[k] === undefined || want[k] === undefined || want[k] === null) continue;
    if (Math.abs(+cur[k] - +want[k]) > 1e-6) out.push([k, +cur[k], +want[k]]);
  }
  return out;
}
function benchFmt(k, v) {
  if (k === 'esc_reverse' || k === 'recovery' || k === 'zone_caps' || k === 'speed_ceiling' || k === 'speed_auto'
      || k === 'launch_boost') return +v ? 'ON' : 'OFF';
  if (k === 'tof_guard') return TOF_GUARD_LABEL[+v] || String(v);
  if (k === 'tcp_mode') return TCP_MODES[+v] || String(v);
  if (k === 'lookahead' || k === 'ensemble_j' || k === 'rev_us' || k === 'steer_trim_us') return (+v).toFixed(0);
  return (+v).toFixed(2);                         // ゲイン・秒・m/s など(1.0 も「1.00」と出す)
}
const benchDiffText = (diff) => diff.map(([k, a, b]) => `${BENCH_LABEL[k] || k} ${benchFmt(k, a)} → ${benchFmt(k, b)}`).join('、');
function benchBaseline() {
  const d = state.bench.data;
  return d && d.baseline ? d.baseline : BENCH_FALLBACK;
}
async function loadBench() {
  let d = null;
  try { d = await api('GET', '/api/bench/candidates'); } catch (e) { d = null; }     // 古いサーバー(404)ではパネルを出さない
  state.bench.data = d;
  renderBench();
}
function renderBaseInfo() {
  const f = benchFieldValues(benchBaseline(), $('#dr-base-steer').value);
  const p = f.params, base = benchBaseline();
  const isB = f.run === BENCH_FALLBACK.run;
  $('#dr-base-info').textContent = `基準(${base.source || '-'}): モデル ${isB ? 'B' : ''}(${f.run || '-'})・ジャンル ${f.mode_id}`
    + `・ステア ${benchFmt('steer_gain', p.steer_gain)}${$('#dr-base-steer').value ? '(この日)' : ''}・速度 ${benchFmt('throttle_gain', p.throttle_gain)}`
    + `・先読み ${benchFmt('lookahead', p.lookahead)}・ToF ガード ${benchFmt('tof_guard', p.tof_guard)}・重ね合わせ ${benchFmt('ensemble_j', p.ensemble_j)}`
    + `・speed_cap ${+p.speed_cap ? benchFmt('speed_cap', p.speed_cap) : '0(上限なし)'}`
    + `・区間の速度上限 OFF。ほかは drive.py の既定(リカバリー・後退は ON)。押しても走行は始めません。`;
}
// 「AI走行」のカードの「ベンチマーク」の選択(2026-10-23 ユーザーの依頼: ベンチマーク 1, 2, 3… の順に select で選ぶ)。
// 今日のブロックを並べる。選ぶとブロックのボタンと同じ(モデル・基準の値・説明欄)。
// 今日(サーバーの日付 d.today)のブロックが無いときは、ほかの日に落とさない(2026-10-27 fix3 直す 3 回目 r3-r3-state-1): 前は黙って
// 「ブロックのある最初の日」(10/23)を並べ、Jetson の時計が 11/03 でない朝(電源を切ると RTC が戻り、ネットが無いと NTP で合わない)に
// 「ベンチマーク 2」を選ぶと 10/23 の SLOW-SEC1・ゲイン 1.00・説明欄 day=2026-10-23 が入った。今は注意の 1 行だけで選べない
function renderBenchPick(d) {
  const sel = $('#dr-bench-pick'), days = (d && d.exists && d.days) || [];
  const di = days.findIndex((x) => x.date === d.today && x.blocks.length);
  const other = di < 0 ? days.filter((x) => x.blocks.length).map((x) => x.date) : [];
  $('#dr-bench-pick-wrap').classList.toggle('hidden', di < 0 && !other.length);
  if (di < 0) {
    if (other.length) {
      sel.innerHTML = `<option value="">— 今日(サーバーの日付 ${esc(d.today || '?')})のベンチマークはありません。Jetson の時刻を`
        + `確かめてください(ブロックのある日: ${esc(other.join('・'))}。手順書 3 章)—</option>`;
      sel.value = '';
      sel.disabled = true;
    }
    return;
  }
  const ap = state.bench.applied || {};
  // 選ぶ欄は毎回「選ぶ」の行に戻す。ブラウザは選ばれている項目を選び直しても change を出さないので、今のブロックを
  // 選んだままにすると「同じブロックを選び直して入れ直す」ができなかった(2026-10-26 fix2 state2-3)。今のブロックは先頭の行に出す
  const cur = ap.day === di && ap.i !== undefined ? days[di].blocks[ap.i] : null;
  sel.innerHTML = `<option value="">${cur ? `— 今: ベンチマーク${esc(cur.block)}: ${esc(cur.id)}${cur.tune ? ` ・ ${esc(cur.tune)}` : ''}`
    + '(同じ番号をもう一度選ぶと入れ直す)—' : '— ベンチマークを選ぶ —'}</option>` + days[di].blocks.map((b, bi) =>
    `<option value="${di}:${bi}"${b.drivable ? '' : ' disabled'}>ベンチマーク${esc(b.block)}: ${esc(b.id)} ${esc(b.label || '')}`
    + `${b.tune ? ` ・ ${esc(b.tune)}` : ''}${b.drivable ? '' : '(走れません)'}</option>`).join('');
  sel.value = '';
  sel.disabled = !!(state.drive.st && state.drive.st.running);
}
// モデルを手で替えて、選んだベンチマークのモデルと違ってしまったら、ベンチマークの選択を外す(説明欄と食い違わないように知らせる)
$('#dr-model').addEventListener('change', () => {
  const d = state.bench.data, ap = state.bench.applied;
  const block = ap && ap.i !== undefined && d && d.days ? ((d.days[ap.day] || {}).blocks || [])[ap.i] : null;
  if (block && block.run !== $('#dr-model').value) {
    state.bench.applied = null; renderBench();
    toast(`ベンチマーク${block.block}(${block.id})のモデルではなくなりました。説明欄を確かめてください`, true);
  }
});
$('#dr-bench-pick').addEventListener('change', (e) => {
  const d = state.bench.data, v = e.target.value;
  if (!v || !d) return;
  const [di, bi] = v.split(':').map(Number);
  const block = ((d.days[di] || {}).blocks || [])[bi];
  if (!block || !block.drivable) return;
  if ((d.days[di] || {}).date !== d.today) { e.target.value = ''; return; }   // 選ぶ欄は今日のブロックだけ(r3-r3-state-1。古い一覧の値も入れない)
  // 入れなかった(サーバーの設定をまだ読んでいない・走行中・モデルが無い)ときは説明欄に official=1 を足さず、選ぶ欄を「選ぶ」の行に
  // 戻す(同じ番号をもう一度選べるように。2026-10-27 fix3 直す 3 回目 r3-r3-state-3。前は入れなかったときも今の説明欄に足した)
  if (!benchApply(block, { day: di, i: bi })) { e.target.value = ''; return; }
  // 本番の印(official=1)と予定の周(plan_laps=3)を説明欄に足す(夜の集計で試しの走行と分ける)
  const desc = $('#dr-desc').value;
  if (!/(^|\s)official=1(\s|$)/.test(desc)) {
    const i = desc.search(/(^|\s)memo=/);
    $('#dr-desc').value = i < 0 ? `${desc} official=1`.trim()
      : `${desc.slice(0, i).trimEnd()} official=1 ${desc.slice(i).trimStart()}`;
  }
});
// 「モデル(学習結果)」の一覧にも、ベンチマークの番号を付けて先頭に並べる(値は学習結果の名前のまま。選択は変えない)。
// 今日のブロックが無いときは番号を付けない(ほかの日の番号に落とさない。r3-r3-state-1)
function labelDriveModels(d) {
  const sel = $('#dr-model'), days = (d && d.exists && d.days) || [], tags = {};
  const di = days.findIndex((x) => x.date === d.today && x.blocks.length);
  if (di >= 0) days[di].blocks.forEach((b) => { (tags[b.run] = tags[b.run] || { nums: [], id: b.id, label: b.label || '' }).nums.push(b.block); });
  const cur = sel.value, opts = [...sel.options];
  for (const o of opts) {
    if (o.dataset.base === undefined) o.dataset.base = o.textContent;
    const t = tags[o.value];
    o.textContent = t ? `ベンチマーク${t.nums.join('・')}: ${t.id} ${t.label} ― ${o.dataset.base}` : o.dataset.base;
  }
  opts.filter((o) => tags[o.value]).sort((a, b) => tags[b.value].nums[0] - tags[a.value].nums[0])
    .forEach((o) => sel.insertBefore(o, sel.firstChild));
  sel.value = cur;
}
function renderBench() {
  renderBaseInfo();
  const d = state.bench.data, card = $('#bench-card');
  renderBenchPick(d);
  labelDriveModels(d);
  const show = !!(d && d.exists);
  card.classList.toggle('hidden', !show);
  if (!show) return;
  $('#bench-head').textContent = [d.bench, d.protocol_sha256 ? `手順 sha256 ${String(d.protocol_sha256).slice(0, 12)}…` : '',
                                  d.created_at ? `作成 ${d.created_at}` : ''].filter(Boolean).join(' ・ ');
  // 今日(サーバーの日付)のブロックが無い: 選ぶ欄は選べない(r3-r3-state-1)。カードのボタンはほかの日のブロックも押せる(確認の文あり)
  const noToday = (d.days || []).some((x) => x.blocks.length) && !(d.days || []).some((x) => x.date === d.today && x.blocks.length);
  $('#bench-error').innerHTML = (d.error ? `<div>data/bench_candidates.json を読めません: ${esc(d.error)}</div>` : '')
    + (d.warnings || []).map((w) => `<div>${esc(w)}</div>`).join('')
    + (noToday ? `<div>今日(サーバーの日付 ${esc(d.today || '?')})のブロックはありません(「今日」の印が無い)。ベンチマークの日なら `
      + 'Jetson の時刻を確かめてください(端末で date。違えばネットにつなぐか sudo date -s で合わせて、画面を読み直す。手順書 3 章)。'
      + '記録の名前もこの時刻で付きます</div>' : '');
  const running = !!(state.drive.st && state.drive.st.running);
  const ap = state.bench.applied || {};
  $('#bench-days').innerHTML = (d.days || []).map((day, di) => {
    const bad = day.blocks.filter((b) => !b.drivable);
    return `<div class="bench-day"><div class="small"><b>${esc(day.date)}</b>${day.date === d.today ? ' <span class="badge ok">今日</span>' : ''}`
      + `${day.note ? ` <span class="muted">${esc(day.note)}</span>` : ''}</div>`
      + (day.blocks.length ? `<div class="bench-blocks">${day.blocks.map((b, bi) =>
        `<button class="small${ap.day === di && ap.i === bi ? ' active' : ''}" data-bench-day="${di}" data-bench-i="${bi}"`
        + `${b.drivable && !running ? '' : ' disabled'} title="${esc(b.drivable ? `${b.label} ・ ${b.run}` : `走れません: ${b.why_not || ''}`)}">`
        + `${esc(b.block)}. ${esc(b.id)}</button>`).join('')}</div>` : '<div class="muted small">ブロックはまだありません</div>')
      + bad.map((b) => `<div class="warn-text small bench-why">ブロック ${esc(b.block)}(${esc(b.id)}): 走れません — ${esc(b.why_not || '')}</div>`).join('')
      + '</div>';
  }).join('') || (d.ok ? '<div class="muted small">日ごとのブロック(days)がありません</div>' : '');
  if (running) $('#bench-days').insertAdjacentHTML('beforeend', '<div class="muted small bench-why">走行中は選べません(止めてから)</div>');
  const cands = d.candidates || [];
  $('#bench-cands').innerHTML = cands.length ? '<table class="bench-table"><tr><th>id</th><th>名前・仮説・モデル</th><th>危なさ</th><th>走れるか</th></tr>'
    + cands.map((c) => `<tr><td><b>${esc(c.id)}</b></td><td>${esc(c.label)}${c.policy_title ? ` <span class="muted">${esc(c.policy_title)}</span>` : ''}`
      + `<div>${esc(c.hypothesis)}</div><div class="run">${esc(c.run || '-')}</div></td>`
      + `<td>${c.risk ? `<span class="badge ${RISK_BADGE[c.risk] || ''}">${esc(c.risk)}</span>` : '-'}</td>`
      + `<td>${c.drivable ? '<span class="badge ok">走れる</span>' : `<span class="badge bad">走れない</span><div class="warn-text">${esc(c.why_not || '')}</div>`}</td></tr>`).join('')
    + '</table>' : '<div class="muted small">候補がありません</div>';
}
// 画面の欄に値を入れる(送らない)。サーバーの値と違う項目は pending にして、表示の同期で戻さない
function benchSetFields(p) {
  const srv = (state.drive.st && state.drive.st.params) || {};
  const mark = (k) => {
    if (srv[k] !== undefined && Math.abs(+srv[k] - +p[k]) > 1e-6) state.drive.pending[k] = true;
    else delete state.drive.pending[k];
  };
  for (const [sel, key, fmt] of DR_PARAMS) {
    if (p[key] === undefined) continue;
    $(sel).value = p[key]; $(sel + '-v').textContent = fmt($(sel).value); mark(key);
  }
  for (const [sel, key] of DR_FLAGS) if (p[key] !== undefined) { $(sel).checked = !!+p[key]; mark(key); }
  if (p.tof_guard !== undefined) { $('#dr-tof-guard').value = String(p.tof_guard); mark('tof_guard'); }
  if (p.tcp_mode !== undefined) { $('#dr-tcp-mode').value = String(p.tcp_mode); mark('tcp_mode'); }
  $('#dr-zone-caps').checked = false; mark('zone_caps');
  for (const [sel, key] of [['#dr-speed-ceiling', 'speed_ceiling'], ['#dr-speed-auto', 'speed_auto'], ['#dr-launch-boost', 'launch_boost']]) {
    $(sel).checked = false;
    if (srv[key] !== undefined && +srv[key] !== 0) state.drive.pending[key] = true; else delete state.drive.pending[key];
  }
  if (state.drive.st) { renderSpeedSwitches(state.drive.st); renderLaunchSwitch(state.drive.st); }   // OFF にしたことをすぐ出す
}
// block: ベンチマークのブロック(null なら「基準の設定」= モデル B)。戻り値: 画面に入れたら true(断ったら false)
function benchApply(block, where) {
  const st = state.drive.st || {};
  if (st.running) { toast('走行中は変えられません(止めてから)', true); return false; }
  // 画面を開いた直後、サーバーの設定をまだ一度も読んでいないうちは入れない(▶ と同じ。2026-10-27 fix3 直す 3 回目 r3-r3-state-3:
  // 前は状態の問い合わせが失敗している間に押すと、サーバーの値が分からず天井・自動合わせ・発進の前進・ゲインを pending にしないので、
  // 最初の同期で前の本の値(天井 ON・0.97 など)に黙って戻り、▶ でそれを送った)
  // 表示の問い合わせが失敗したあと、次に同期するまでも断る(▶ と同じ drNotSynced。2026-10-27 fix5 t1)
  if (drNotSynced()) return false;
  const f = benchFieldValues(benchBaseline(), $('#dr-base-steer').value);
  if (block && block.params) f.params = Object.assign({}, f.params, block.params);       // ブロックの上書き(速度ゲインなど)
  if (block && block.mode_id != null) f.mode_id = +block.mode_id;
  const run = block ? block.run : f.run;
  const m = state.drive.models.find((x) => x.name === run);
  // 一覧を読めていない(走行タブを開いたときの一覧の GET が失敗・遅れた)ときは「エンジンが必要」と言わない(fix7 w2-ui-w2-ui-r72-ui-2)
  if (!m) { toast(drModelsMissing() ? DR_MODELS_MISS : `モデル ${run} が走行の一覧にありません(TensorRT エンジンが必要)`, true); return false; }
  if (m.drivable === false) { toast(`このモデルでは走行できません: ${m.why_not || ''}`, true); return false; }
  // ほかの日のブロック(カードのボタン。今日 = サーバーの日付のブロックではない)は確かめてから入れる(r3-r3-state-1。選ぶ欄は今日だけ)
  const bday = block && where && state.bench.data ? ((state.bench.data.days || [])[where.day] || {}).date : null;
  if (bday && bday !== state.bench.data.today
      && !confirm(`このブロックは ${bday} の段です(今日 = サーバーの日付 ${state.bench.data.today || '?'})。\n`
        + `説明欄は ${bday} の本になり、記録の名前はサーバーの時刻で付きます。ベンチマークの日なら、キャンセルして Jetson の時刻を`
        + '合わせてから画面を読み直してください(手順書 3 章)。\n\nこのまま入れますか?')) return false;
  const before = { run: $('#dr-model').value, mode: $('#dr-mode').value, desc: $('#dr-desc').value, record: $('#dr-record').checked,
                   dry: $('#dr-dry').checked, params: readDriveParams() };
  if (!state.bench.prev) state.bench.prev = before;            // 取り消しで戻す値(最初に入れる前のもの)
  const changes = [];
  if (before.run !== run) changes.push(`モデル ${before.run || '-'} → ${run}`);
  $('#dr-model').value = run;
  state.drive.restoreNote = '';
  renderDriveModel();
  const notes = f.notes.slice();
  const modeOk = [...$('#dr-mode').options].some((o) => o.value === String(f.mode_id) && !o.disabled);
  if (modeOk) $('#dr-mode').value = String(f.mode_id);
  else notes.push(`このモデルではジャンル ${f.mode_id} を選べません。ジャンルを確かめてください`);
  if (before.mode !== $('#dr-mode').value) changes.push(`ジャンル ${before.mode || '-'} → ${$('#dr-mode').value}`);
  const diff = benchDiff(before.params, f.params);
  const trimKeep = +before.params.steer_trim_us;
  if (f.params.steer_trim_us === undefined && Number.isFinite(trimKeep) && Math.abs(trimKeep) > 1e-6) {
    notes.push(`トリム ${trimKeep > 0 ? '+' : ''}${trimKeep.toFixed(0)} µs はそのままです`
      + '(車の左右の直し。基準の設定・ブロックでは変えません。0 に戻すときはスライダーで)');
  }
  // 試験の選択肢(後ろの見張りなど。2026-10-26 fix3)は基準の設定・ブロックでは変えない(ON にも OFF にもしない。スイッチで)。
  // ブロックの params に書いてあっても入れない(画面の欄も無いので前から入らなかった。ここで外して、ON なら注を出す)
  const stP = (state.drive.st && state.drive.st.params) || {};
  for (const o of OPT_SWITCHES) {
    if (block && block.params && +block.params[o.key] === 1) notes.push(`${o.name}はブロックでは ON にしません(スイッチで入れてください)`);
    delete f.params[o.key];
    if (+stP[o.key] === 1) notes.push(`${o.name}は ON のままです(基準の設定・ブロックでは変えません。OFF にするときはスイッチで)`);
  }
  benchSetFields(f.params);
  if (block) {
    $('#dr-record').checked = true;                            // 説明欄は収集するときだけ記録に残る
    $('#dr-record-opts').classList.remove('hidden');
    if (!before.record) changes.push('収集 OFF → ON');
    $('#dr-desc').value = block.description || '';
    changes.push('説明欄に雛形');
    // ブロックは本番の走行: 朝の dry run の「モーターに出力しない」が残っていたら外す(2026-10-27 fix3 直す 3 回目 r3-r3-state-2。
    // 前は残ったまま ▶ で確認の文が出ず、推論の確認(出力は中立)で始まって 1-1 の 1 本を失った)。取り消すで戻る
    if (before.dry) {
      $('#dr-dry').checked = false;
      changes.push('「モーターに出力しない」ON → OFF(ブロックは本番の走行。推論の確認にするならもう一度入れる)');
    }
  } else if (descField(before.desc, 'bench') && descField(before.desc, 'block')) {
    // 基準の設定はベンチマークの本ではない(2026-10-27 fix3 r2-r2-state-1): 前に選んだブロックの雛形(bench=・block=N・cand=・
    // official=1)が説明欄に残ると、その走行(朝の車輪を浮かせた発進の前進の確かめなど)が公式のブロックの本(AI No.1)として
    // 記録され、夜の道具 ①〜⑤ が回の本に数えた。雛形を消す(走行開始は「AI走行 モデル名」になる。取り消すで戻る)
    $('#dr-desc').value = '';
    changes.push(`説明欄のベンチマークの雛形(ブロック ${descField(before.desc, 'block')}${/(^|\s)official=1(\s|$)/.test(before.desc) ? '・official=1' : ''})を消した`);
    if (before.record) notes.push('「走行中のデータを収集する」は ON のままです。試しの走行なら外してください(手順書 3 章)');
  }
  state.bench.applied = block ? { day: where.day, i: where.i } : { baseline: true };
  updateTcpWarn(); updateNoPostNote(); renderBench(); renderPrepHints();
  const what = block ? `ブロック ${block.block}(${block.id} ${block.label})` : '基準の設定';
  const all = changes.concat(diff.length ? [benchDiffText(diff)] : []);
  $('#dr-base-note-text').innerHTML = `<b>${esc(what)}</b>を画面に入れました。`
    + (all.length ? `変えた所: ${esc(all.join('、'))}。` : '画面の値はすでに同じでした。')
    + (Object.keys(state.drive.pending).length ? ' まだサーバーへは送っていません。走行開始(推論の確認も)で送り、保存されます。' : '')
    + (block ? ' 説明欄の battery・V_before・check を書き足してください。' : '')
    + notes.map((n) => `<div class="warn-text">${esc(n)}</div>`).join('');
  $('#dr-base-note').classList.remove('hidden');
  toast(`${what}を画面に入れました(走行は始めていません)`);
  return true;
}
// 走行開始で送った・走行が始まった: 入れた値はサーバーの値になった
function benchSent() {
  state.drive.pending = {};
  state.bench.prev = null;
  $('#dr-base-note').classList.add('hidden');
}
$('#btn-dr-baseline').addEventListener('click', () => benchApply(null));
$('#dr-base-steer').addEventListener('change', () => { saveDaySteer($('#dr-base-steer').value); renderBaseInfo(); renderPrepHints(); });
// 「この日のステア」は、その日のあいだだけ、この画面(ブラウザ)に覚える。再読み込みや Jetson の立ち上げ直しのあとも同じ値に戻す。
// 日付が変わったら「基準どおり」。サーバーには送らない(送るのは今までどおり走行開始のときだけ)。2026-10-21 Fix-all
const DAY_STEER_KEY = 'jetformer.bench.daySteer';
function localDay() { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; }
function saveDaySteer(v) {
  try { if (v) localStorage.setItem(DAY_STEER_KEY, JSON.stringify({ day: localDay(), value: v })); else localStorage.removeItem(DAY_STEER_KEY); }
  catch (e) { /* 覚えられなくても画面は動く */ }
}
function restoreDaySteer() {
  try {
    const o = JSON.parse(localStorage.getItem(DAY_STEER_KEY) || 'null'), sel = $('#dr-base-steer');
    if (o && o.day === localDay() && [...sel.options].some((x) => x.value === o.value)) sel.value = o.value;
  } catch (e) { /* 読めなければ基準どおり */ }
}
restoreDaySteer();
$('#btn-dr-base-undo').addEventListener('click', () => {
  const prev = state.bench.prev, st = state.drive.st || {};
  state.drive.pending = {};
  if (prev && !st.running) {
    $('#dr-model').value = prev.run; renderDriveModel();
    if ([...$('#dr-mode').options].some((o) => o.value === prev.mode && !o.disabled)) $('#dr-mode').value = prev.mode;
    $('#dr-desc').value = prev.desc; $('#dr-record').checked = prev.record;
    $('#dr-record-opts').classList.toggle('hidden', !prev.record);
    if (prev.dry !== undefined) $('#dr-dry').checked = prev.dry;     // r3-r3-state-2(ブロックで外した「モーターに出力しない」も戻す)
  }
  if (st.params) setDriveParams(st.params);
  state.bench.prev = null; state.bench.applied = null;
  $('#dr-base-note').classList.add('hidden');
  updateTcpWarn(); updateNoPostNote(); renderBench(); renderPrepHints();
  toast('入れた値を取り消しました(サーバーの値に戻しました)');
});
$('#bench-days').addEventListener('click', (e) => {
  const b = e.target.closest('[data-bench-day]');
  const d = state.bench.data;
  if (!b || b.disabled || !d) return;
  const di = +b.dataset.benchDay, bi = +b.dataset.benchI;
  const block = ((d.days[di] || {}).blocks || [])[bi];
  if (block && block.drivable) benchApply(block, { day: di, i: bi });
});

// 走る前のチェック(この画面の中だけ。保存しない)
const drChecks = () => [...document.querySelectorAll('.dr-check')];
function renderCheckCount() {
  const cs = drChecks(), n = cs.filter((c) => c.checked).length;
  const b = $('#dr-check-count');
  b.textContent = `${n} / ${cs.length}`;
  b.className = 'badge' + (n === cs.length ? ' ok' : ' warn');
}
drChecks().forEach((c) => c.addEventListener('change', renderCheckCount));
$('#btn-dr-check-clear').addEventListener('click', () => { drChecks().forEach((c) => { c.checked = false; }); renderCheckCount(); });
renderCheckCount();
function renderPrepHints() {
  const s = state.status || {};
  const free = s.disk ? s.disk.free_gb : null;
  $('#dr-check-disk').innerHTML = free == null ? '' : `<span class="${free >= 10 ? 'ok-text' : 'warn-text'}">今 ${esc(free)} GB</span>`;
  $('#dr-check-heavy').innerHTML = s.training ? '<span class="warn-text">学習が動いています</span>'
    : (s.training === false ? '<span class="ok-text">学習は動いていません</span> <span class="muted">(ワークフロー・jf-heavy は端末で確かめる)</span>' : '');
  const f = benchFieldValues(benchBaseline(), $('#dr-base-steer').value);
  const diff = benchDiff(readDriveParams(), f.params);
  const mode = $('#dr-mode').value;
  const bad = (mode && mode !== String(f.mode_id) ? [`ジャンル ${mode}(基準 ${f.mode_id})`] : [])
    .concat(diff.map(([k, a, b]) => `${BENCH_LABEL[k] || k} ${benchFmt(k, a)}(基準 ${benchFmt(k, b)})`));
  $('#dr-check-base').innerHTML = (bad.length ? `<span class="warn-text">基準と違う: ${esc(bad.join('、'))}</span>`
    : '<span class="ok-text">画面の値は基準と同じ</span> <span class="muted">(モデルはブロックごとに替わる)</span>') + trimNote() + optNote(state.drive.st);
}
// トリムが 0 でないときの文(基準の設定・ブロックでは変えない値なので、基準と同じかの判定とは別に出す。2026-10-26 fix2 state2-1)
function trimNote() {
  const v = +$('#dr-trim').value;
  return Number.isFinite(v) && Math.abs(v) > 1e-6
    ? ` <span class="muted">・トリム ${v > 0 ? '+' : ''}${v.toFixed(0)} µs(車の左右の直し。基準の設定・ブロックでは変えない)</span>` : '';
}
// 走行開始の確認に足す行: トリムが 0 でない・リカバリーの後ろの見張りが ON(どちらも 0・OFF なら行を出さない)
function trimGuardStartLines(st) {
  const v = +$('#dr-trim').value, p = (st && st.params) || {};
  return (Number.isFinite(v) && Math.abs(v) > 1e-6 ? `\nトリム: ${v > 0 ? '+' : ''}${v.toFixed(0)} µs(車の左右の直し。基準の設定・ブロックでは変わりません)` : '')
    + (+p.rec_back_guard === 1 ? '\nリカバリーの後ろの見張り: ON(下がった時間をどの段でも数える・後ろに当たったらすぐ前へ)' : '')
    // 2026-10-26 fix3: 後ろの見張りとこの 2 つは、試験の選択肢のスイッチ(OPT_SWITCHES)か API で入れる(走行開始では送らない =
    // サーバーの値のまま。行もサーバーの値から)。OFF なら行を出さない
    + (+p.rec_hold_keep === 1 ? '\n中立止めのあともリカバリーを残す: ON(リカバリー中止は CH3 を人側にするまで解けない)' : '')
    + (+p.launch_no_kick === 1 ? '\n発進の前進を出さない区間は発進の補助も出さない: ON'
       + ($('#dr-launch-boost').checked ? '' : '(この本は発進の前進が OFF なので、何も変わりません)') : '');
}
// 走行開始の確認に足す行: リカバリーの下がる量(rev_s・rev_us。走行開始で送る値 = スライダー)が、10/23・24 と同じ 0.40 秒・300 µs
// (drive.py の既定・bench_candidates.json の基準)でないとき(2026-10-27 fix5 反証 4 回目 t3。approvals 10 の (b) を選んだ日に読む。
// 同じなら行を出さない = 前と同じ)。基準の設定・ブロックは今まで通り基準の値を入れる(トリムと違い、変えない項目にはしない)
const REV_USUAL = { s: +BENCH_FALLBACK.params.rev_s, us: +BENCH_FALLBACK.params.rev_us };
function revStartLine() {
  const s = +$('#dr-rev-s').value, us = +$('#dr-rev-us').value;
  if (!Number.isFinite(s) || !Number.isFinite(us) || (Math.abs(s - REV_USUAL.s) < 1e-6 && Math.abs(us - REV_USUAL.us) < 1e-6)) return '';
  return `\nリカバリーの下がる量: ${s.toFixed(2)} 秒・中立+${us.toFixed(0)} µs(10/23・24 は ${REV_USUAL.s.toFixed(2)} 秒・${REV_USUAL.us.toFixed(0)} µs)`;
}
// 走行開始の確認に足す行: チェックが残っている・説明欄の電池が空欄
function prepStartLines() {
  const cs = drChecks(), n = cs.filter((c) => c.checked).length;
  const lines = [];
  if (n < cs.length) lines.push(`走る前のチェックが ${n} / ${cs.length} です`);
  if ($('#dr-record').checked && /battery=#__/.test($('#dr-desc').value)) lines.push('説明欄の電池の番号がまだ空欄です(走ったあと、シーンの説明でも直せます)');
  return lines.length ? '\n\n' + lines.join('\n') : '';
}

// 区間の速度上限(drive.py の zone_caps。既定 OFF・起動のたびに OFF)
const zoneSupported = (st) => !!(st && st.zone_caps_spec && st.params && st.params.zone_caps !== undefined);
function zoneSpecText(st) {
  const sp = st && st.zone_caps_spec;
  if (!sp || !sp.caps_mps) return 'v1.1: R1 2.5・GM 2.2・DE 3.0 m/s';
  const c = sp.caps_mps, n = (x) => (x == null || !Number.isFinite(+x) ? '-' : (+x).toFixed(1));
  return `${sp.version || ''}: R1 ${n(c.R1)}・GM ${n(c.GM)}・DE ${n(c.DE)} m/s${c.unknown != null && Number.isFinite(+c.unknown) ? `。4 つめ: 位相が不確かなとき(1 周目のほぼ全部など)も ${n(c.unknown)}` : ''}`;
}
// v1.1 の 4 つめの上限(位相が不確か = unknown)。ユーザーの了承は 3 区間(R1・GM・DE)だけなので、ON の確認で必ず示す
const ZONE_UNKNOWN_FALLBACK = '4 つめの上限: 位相が不確か(unknown)のときも 3.0 m/s で切ります。走行の始めから最初の R1 を見つけるまで(1 周目のほぼ全部)・R1 から 16 秒を超えたとき・1 秒以上止まったあと。ユーザーの了承は 3 区間(R1・GM・DE)だけです。ON の了承をもらうときに、これも含めて確かめてください';
function zoneUnknownCap(st) {                 // m/s(4 つめの上限が無ければ null)。サーバーが説明を返さないときは v1.1 の 3.0
  const sp = st && st.zone_caps_spec;
  if (!sp || !sp.caps_mps) return 3.0;
  const u = sp.caps_mps.unknown;
  return u != null && Number.isFinite(+u) ? +u : null;
}
function zoneUnknownNote(st) {
  const sp = st && st.zone_caps_spec;
  if (sp && typeof sp.unknown_note === 'string') return sp.unknown_note;      // '' = 4 つめの上限なし
  return zoneUnknownCap(st) != null ? ZONE_UNKNOWN_FALLBACK : '';
}
function renderZoneSwitch(st) {
  const el = $('#dr-zone-caps'), ok = zoneSupported(st);
  const srvOn = ok && +st.params.zone_caps === 1;
  el.disabled = !ok || (!!st.running && !srvOn);              // 走行中は OFF にだけできる
  if (!ok) el.checked = false;
  $('#dr-zone-spec').textContent = `(${zoneSpecText(st)})`;
  const unk = zoneUnknownNote(st), unkEl = $('#dr-zone-unknown');
  if (unkEl) { unkEl.textContent = unk; unkEl.hidden = !unk; }
  el.closest('.zone-panel').classList.toggle('on', el.checked);
  const note = $('#dr-zone-note');
  if (!ok) note.innerHTML = '<span class="muted">今のサーバーは区間の速度上限に対応していません(管理画面のサーバーを再起動すると使えます)</span>';
  else if (st.zone_caps_error) note.innerHTML = srvOn
    ? `<span class="warn-text">計算が失敗したため、この走行では上限を外しました: ${esc(st.zone_caps_error)}</span>`
    : `<span class="warn-text">OFF(記録だけ)の影の判定が計算の失敗で止まりました。車への出力は変わりません: ${esc(st.zone_caps_error)}</span>`;
  else if (state.drive.pending.zone_caps) note.innerHTML = '<span class="muted">基準の設定で OFF にしました(走行開始で送ります)</span>';
  else note.innerHTML = srvOn ? '<b class="warn-text">ON: 車への出力に上限を掛けます</b>' : '';
}
$('#dr-zone-caps').addEventListener('change', async () => {
  const el = $('#dr-zone-caps'), st = state.drive.st || {};
  const on = el.checked;
  if (on) {
    if (st.running) { el.checked = false; toast('走行中は区間の速度上限を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)', true); return; }
    const unk = zoneUnknownNote(st);
    if (!confirm(`区間の速度上限を ON にします(${zoneSpecText(st)})。\n\n${unk ? unk + '。\n\n' : ''}ON にするのは、10/23 の記録で G1 に合格し、ユーザーが了承したあとです。\n了承は${unk ? '、3 区間に加えて 4 つめ(位相が不確かなときの上限)にも' : ''}出ていますか?`)) { el.checked = false; return; }
  }
  try {
    const p = await postDriveParams({ zone_caps: on ? 1 : 0 }, ['zone_caps']);    // 送れなければ pending も戻す(r1-state3-2)
    if (p && p.zone_caps === undefined) { el.checked = false; toast('今のサーバーは区間の速度上限に対応していません(再起動が必要)', true); }
    else if (state.drive.st) state.drive.st.params = p;
    if (p && p.zone_caps !== undefined) toast(on ? '区間の速度上限を ON にしました' : '区間の速度上限を OFF にしました');
  } catch (e) { toast(e.message, true); el.checked = !on; }
  if (state.drive.st) renderZoneSwitch(state.drive.st);
});
function zoneLiveText(z, s) {
  if (z.error) return z.on ? `<span class="warn-text">計算が失敗したので外しました(${esc(z.error)})</span>`
    : `<span class="warn-text">OFF(記録だけ)の影の判定が計算の失敗で止まりました。出力は変わりません(${esc(z.error)})</span>`;
  const lim = z.limit != null && Number.isFinite(+z.limit) ? `上限 ${(+z.limit).toFixed(2)} m/s` : '上限なし';
  return `${z.on ? '<b class="warn-text">ON</b>' : 'OFF(記録だけ)'} ・ 区間 ${esc(z.zone || '-')} ・ ${lim}`
    + `${z.mode ? ` ・ ${esc(ZONE_MODE_LABEL[z.mode] || z.mode)}${z.on ? '' : '(ON なら)'}` : ''}`
    + ` ・ 出力を変えた ${s.zone_applied ?? 0} フレーム ・ ブレーキ ${s.zone_brakes ?? 0} 回`;
}

// 速さの天井・速さのゲインの自動合わせ(drive.py の speed_ceiling・speed_auto。2026-10-25。既定 OFF・起動のたびに OFF・
// 走行中は ON にできない・OFF でも影の判定を記録)。文の値はサーバーの speed_ctrl_spec(drive.speed_ctrl_spec。走行ループが
// 使う値から作る)から出す。古いサーバーで欄が無いときだけ params・控えの値(SPEED_FALLBACK)
const speedSupported = (st) => !!(st && st.speed_ctrl_spec && st.params && st.params.speed_ceiling !== undefined);
const autoSupported = (st) => speedSupported(st) && st.params.speed_auto !== undefined;
const SPEED_FALLBACK = { all: 4.5, left: 4.0, steer: 100, gz: 45, hold: 0.25, over: 0.2, target: 4.2, mlo: 0.85, mhi: 1.2,
                         raise: 1.1, step: 0.03, vref: 8.35, lapA: 14.52, lapB: 1.50, band: 0.10 };
function speedSpec(st) {
  const sp = (st && st.speed_ctrl_spec) || {}, p = (st && st.params) || {};
  const c = sp.ceiling || {}, a = sp.auto || {}, lr = c.left_rule || {};
  const num = (...xs) => { for (const x of xs) if (x !== undefined && x !== null && x !== '' && Number.isFinite(+x)) return +x; return null; };
  const mr = Array.isArray(a.m_range) ? a.m_range : [];
  const F = SPEED_FALLBACK;
  return {
    all: num(c.all_mps, p.ceil_all_mps, F.all), left: num(c.left_mps, p.ceil_left_mps, F.left),
    steer: num(lr.steer_us, F.steer), gz: num(lr.gz_dps, F.gz), hold: num(lr.hold_s, F.hold), over: num(c.neutral_over_mps, F.over),
    target: num(a.target_mps, p.auto_target_mps, F.target), mlo: num(mr[0], F.mlo), mhi: num(mr[1], F.mhi),
    raise: num(a.raise_max_gain, F.raise), step: num(a.step_max, F.step), vref: num(a.vref_v, p.auto_vref_v, F.vref),
    lapA: F.lapA, lapB: F.lapB, band: num(a.deadband_s, F.band),
    // 周の直しの物差しは周の時間(2026-10-25 fix.md)。目標の周の時間はサーバーの値(無ければ 14.52 − 1.50 × 目標の速さ)
    lapTarget: num(a.target_lap_s, F.lapA - F.lapB * num(a.target_mps, p.auto_target_mps, F.target)),
    ceilRule: typeof c.rule_text === 'string' ? c.rule_text : '', ceilNote: typeof c.note === 'string' ? c.note : '',
    autoNote: typeof a.note === 'string' ? a.note : '', sources: a.sources || null, warnText: sp.auto_warn_text || {},
  };
}
const ceilSpecText = (sp) => `全体 ${sp.all.toFixed(1)}・左旋回中 ${sp.left.toFixed(1)} m/s`;
const autoLapText = (sp) => `目標 1 周 ${sp.lapTarget.toFixed(2)} 秒(周の速さ p90 ${sp.target.toFixed(2)} m/s 相当)`;
function vbatSourceText(st) {
  const vb = st && st.vbat;
  if (vb && vb.supported) return '基板の電圧(走り始めから)';
  // 説明欄の V_before の読み方はサーバー(jetformer/speed_auto.py の v_before_of)と同じ(2026-10-28 fix8 ui r2
  // rehearsal-r2-vbefore-screen-vs-server。前は「V_before= のあとが数字で始まるか」だけを見たので、831・8,31・0.83・9.5 でも「説明欄」と
  // 出て、サーバーは使わず電圧の源「なし」で走った。全角の ８.３１ は逆): Python の str.split() と同じ空白で区切った「V_before=」(使えなければ
  // 「Vb_before=」)の後ろを、末尾の V・v を除いて Python の float() と同じ決まりで数にし、6.0〜9.0 V のときだけ使う
  const PY_SPACE = /[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/;
  const VDESC = [6.0, 9.0];                                      // speed_auto.AutoParams の vdesc_lo・vdesc_hi
  const pyFloat = (s) => {                                       // Python の float(s) の値(読めない = null)
    let t = '';
    for (const ch of s) {
      if (PY_SPACE.test(ch)) {                                   // 空白の字は ' ' にして、あとで前後だけ除く(Python と同じ)
        t += ' ';
      } else if (/\p{Nd}/u.test(ch)) {                           // Unicode の 10 進の字(全角の ８ など)は 0〜9 に(Python と同じ)
        const c = ch.codePointAt(0);
        let z = c;
        while (z > 0 && /\p{Nd}/u.test(String.fromCodePoint(z - 1))) z -= 1;
        t += String((c - z) % 10);
      } else if (ch.codePointAt(0) > 127) {
        return null;
      } else {
        t += ch;
      }
    }
    t = t.replace(/^ +| +$/g, '');
    if (/^[+-]?(inf|infinity)$/i.test(t)) return t[0] === '-' ? -Infinity : Infinity;
    if (/^[+-]?nan$/i.test(t)) return NaN;
    if (!/^[+-]?(\d(_?\d)*(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?$/.test(t)) return null;
    return Number(t.replace(/_/g, ''));
  };
  const vals = {};
  for (const tok of ($('#dr-desc').value || '').split(PY_SPACE)) {
    for (const key of ['V_before', 'Vb_before']) {
      if (tok.startsWith(key + '=') && !(key in vals)) {
        const x = pyFloat(tok.slice(key.length + 1).replace(/[Vv]+$/, ''));
        if (x !== null) vals[key] = x;
      }
    }
  }
  const ok = ['V_before', 'Vb_before'].some((k) => k in vals && Number.isFinite(vals[k]) && vals[k] > VDESC[0] && vals[k] < VDESC[1]);
  return ok ? '説明欄の V_before(走行のあいだ一定)' : 'なし(1.00 倍。周ごとの直しだけ)';
}
function ceilConfirmText(sp) {
  const rule = sp.ceilRule || `左旋回 = 車に出す舵 +${sp.steer} µs 以上 か ヨー +${sp.gz} deg/s 以上(どちらも切れてから ${sp.hold} 秒は左旋回のまま)`;
  const note = sp.ceilNote || `前進を弱めるだけ。ブレーキなし・舵は変えない。天井 + ${sp.over.toFixed(1)} m/s を超えたら中立(惰性)。当たりを 0 にはしない(DE の当たりには効かない)。OFF にするまで続く(管理画面の再起動で OFF)`;
  return `速さの天井を ON にします(${ceilSpecText(sp)})。\n\n${rule}。\n${note}。\n`
    + '走行中は ON にできません(OFF にはできます)。ON は次の走行にも残ります(OFF にするか、管理画面を起動し直すと OFF)。\n\nユーザーの了承は出ていますか?';
}
function autoConfirmText(sp, st) {
  const note = sp.autoNote || `速さのゲインに ${sp.mlo.toFixed(2)}〜${sp.mhi.toFixed(2)} 倍を掛けます(上げるのは ${sp.raise.toFixed(2)} まで)。天井が ON のときだけ。OFF にするまで続く(管理画面の再起動で OFF)`;
  return `速さのゲインの自動合わせを ON にします(${autoLapText(sp)})。\n\n`
    + `電圧の源: ${vbatSourceText(st)}\n範囲: 速度ゲインの ${sp.mlo.toFixed(2)}〜${sp.mhi.toFixed(2)} 倍(上げるのは ${sp.raise.toFixed(2)} まで)。\n`
    + `${note}。\n走行中は ON にできません(OFF にはできます)。ON は次の走行にも残ります(OFF にするか、管理画面を起動し直すと OFF)。\n\nユーザーの了承は出ていますか?`;
}
function speedStartLines(st) {
  const sp = speedSpec(st), lines = [];
  if ($('#dr-speed-ceiling').checked) lines.push(`速さの天井: ON(${ceilSpecText(sp)}。前進を弱めるだけ)`);
  if ($('#dr-speed-ceiling').checked && $('#dr-speed-auto').checked) lines.push(`速さのゲインの自動合わせ: ON(目標 1 周 ${sp.lapTarget.toFixed(2)} 秒・電圧の源 ${vbatSourceText(st)})`);
  return lines.length ? '\n' + lines.join('\n') : '';
}
function renderSpeedSwitches(st) {
  const ce = $('#dr-speed-ceiling'), au = $('#dr-speed-auto'), ok = speedSupported(st), aok = autoSupported(st);
  const running = !!(st && st.running), p = (st && st.params) || {};
  const srvCeil = ok && +p.speed_ceiling === 1, srvAuto = aok && +p.speed_auto === 1;
  ce.disabled = !ok || (running && !srvCeil);                 // 走行中は OFF にだけできる
  if (!ok) ce.checked = false;
  if (!aok || !ce.checked) au.checked = false;                // 自動合わせは天井が ON のときだけ
  au.disabled = !aok || !ce.checked || (running && !srvAuto);
  const sp = speedSpec(st);
  $('#dr-ceil-spec').textContent = `(${ceilSpecText(sp)})`;
  $('#dr-auto-spec').textContent = `(${autoLapText(sp)})`;
  $('#dr-speed-panel').classList.toggle('on', ce.checked);
  const errs = (st && st.speed_ctrl_error) || {};             // drive.py: {ceil: 理由, auto: 理由}(失敗した部分だけ)
  const err = typeof errs === 'string' ? errs : errs.ceil, aerr = typeof errs === 'string' ? '' : errs.auto;
  const cn = $('#dr-ceil-note'), an = $('#dr-auto-note');
  if (!ok) cn.innerHTML = '<span class="muted">今のサーバーは速さの天井に対応していません(管理画面のサーバーを再起動すると使えます)</span>';
  else if (err) cn.innerHTML = `<span class="warn-text">${srvCeil ? '計算が失敗したため、この走行では天井・自動合わせを外しました' : 'OFF(記録だけ)の影の判定が計算の失敗で止まりました。車への出力は変わりません'}: ${esc(err)}</span>`;
  else if (state.drive.pending.speed_ceiling) cn.innerHTML = '<span class="muted">基準の設定で OFF にしました(走行開始で送ります)</span>';
  else cn.innerHTML = srvCeil ? '<b class="warn-text">ON: 車への出力に速さの天井を掛けます</b>' : '';
  if (ok && !aok) an.innerHTML = '<span class="muted">今のサーバーは自動合わせに対応していません</span>';
  else if (aerr) an.innerHTML = `<span class="warn-text">自動合わせの計算が失敗したため、速度ゲインは設定のままです: ${esc(aerr)}</span>`;
  else if (ok && !ce.checked) an.innerHTML = '<span class="muted">天井が OFF のあいだは押せません</span>';
  else an.innerHTML = srvAuto ? '<b class="warn-text">ON: 速度ゲインを電圧と周の時間で合わせます</b>' : '';
}
$('#dr-speed-ceiling').addEventListener('change', async () => {
  const el = $('#dr-speed-ceiling'), st = state.drive.st || {}, on = el.checked, auWas = $('#dr-speed-auto').checked;
  if (on) {
    if (st.running) { el.checked = false; toast('走行中は速さの天井を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)', true); return; }
    if (!confirm(ceilConfirmText(speedSpec(st)))) { el.checked = false; return; }
  } else $('#dr-speed-auto').checked = false;
  try {
    // 天井を ON にしたら、自動合わせは OFF から始める(サーバーに残った ON も消す)。前は { speed_ceiling: 1 } だけを送り、
    // ブロック・基準の設定で OFF にした自動合わせが、次の表示の同期(約 1 秒後)で黙って ON に戻っていた(2026-10-26 fix3 state-1)。
    // 自動合わせを使う本(2-3・2-4)は、今まで通り天井のあとに自動合わせのスイッチで入れる(別の POST)。
    // 送れなかったら、天井・自動合わせのスイッチと pending を押す前に戻す(前は pending を消したままで、同期でサーバーに残った
    // 前の本の天井・自動合わせの ON が画面に戻り、走行開始で送っていた。2026-10-26 fix3 r1-state3-2)
    const p = await postDriveParams(on ? { speed_ceiling: 1, speed_auto: 0 } : { speed_ceiling: 0, speed_auto: 0 }, ['speed_ceiling', 'speed_auto']);
    if (p && p.speed_ceiling === undefined) { el.checked = false; toast('今のサーバーは速さの天井に対応していません(再起動が必要)', true); }
    else {
      if (state.drive.st) state.drive.st.params = p;
      toast(on ? '速さの天井を ON にしました' : '速さの天井を OFF にしました(自動合わせも OFF)');
    }
  } catch (e) { toast(e.message, true); el.checked = !on; $('#dr-speed-auto').checked = auWas; }
  if (state.drive.st) renderSpeedSwitches(state.drive.st);
});
$('#dr-speed-auto').addEventListener('change', async () => {
  const el = $('#dr-speed-auto'), st = state.drive.st || {}, on = el.checked;
  if (on) {
    if (!$('#dr-speed-ceiling').checked) { el.checked = false; toast('自動合わせは、速さの天井が ON のときだけ使えます', true); return; }
    if (st.running) { el.checked = false; toast('走行中は自動合わせを ON にできません。止めてから ON にしてください(1 走行 = 1 設定)', true); return; }
    if (!confirm(autoConfirmText(speedSpec(st), st))) { el.checked = false; return; }
  }
  try {
    const p = await postDriveParams({ speed_auto: on ? 1 : 0 }, ['speed_auto']);    // 送れなければ pending も戻す(r1-state3-2)
    if (p && p.speed_auto === undefined) { el.checked = false; toast('今のサーバーは自動合わせに対応していません(再起動が必要)', true); }
    else {
      if (state.drive.st) state.drive.st.params = p;
      toast(on ? '速さのゲインの自動合わせを ON にしました' : '速さのゲインの自動合わせを OFF にしました');
    }
  } catch (e) { toast(e.message, true); el.checked = !on; }
  if (state.drive.st) renderSpeedSwitches(state.drive.st);
});
// 発進の前進(drive.py の launch_boost・assist.LaunchBoost。2026-10-26。既定 OFF・起動のたびに OFF・走行中は ON にできない)。
// 文の値はサーバーの launch_spec(drive.launch_spec。走行ループが使う値から作る)から出す。古いサーバーで欄が無ければ押せない
const launchSupported = (st) => !!(st && st.params && st.params.launch_boost !== undefined);
const LAUNCH_FALLBACK = { us: 250, mps: 2.5, max_s: 1.2 };
function launchSpec(st) {
  const sp = (st && st.launch_spec) || {}, p = (st && st.params) || {}, F = LAUNCH_FALLBACK;
  const num = (...xs) => { for (const x of xs) if (x !== undefined && x !== null && x !== '' && Number.isFinite(+x)) return +x; return null; };
  return { us: num(sp.us, p.launch_us, F.us), mps: num(sp.mps, p.launch_mps, F.mps), max_s: num(sp.max_s, p.launch_max_s, F.max_s),
           note: typeof sp.note === 'string' ? sp.note : '' };
}
const launchSpecText = (sp) => `前進 ${sp.us.toFixed(0)} µs 以上・${sp.mps.toFixed(1)} m/s まで・最大 ${sp.max_s.toFixed(1)} 秒`;
function launchConfirmText(sp) {
  const note = sp.note || `CH3 を AI 側にした最初のフレームで、止まっていて前が 1 m 以上空いているときだけ、スロットルを 中立 − ${sp.us.toFixed(0)} µs より前に出します(前進だけ・舵はモデルのまま)。既定は OFF・管理画面を起動するたびに OFF`;
  return `発進の前進を ON にします(${launchSpecText(sp)})。\n\n${note}。\n\n`
    + '試験の選択肢です。LH の入口に少し速く入ります。コースで ON にするのは、ユーザーの了承と、車輪を浮かせた確かめのあとです。\n'
    + 'ToF ガードが OFF のときは始めません。走行中は ON にできません(OFF にはできます)。\n\nユーザーの了承は出ていますか?';
}
function launchStartLine(st) {
  return $('#dr-launch-boost').checked ? `\n発進の前進: ON(${launchSpecText(launchSpec(st))}。AI 側にした最初のフレームから強い前進)` : '';
}
function renderLaunchSwitch(st) {
  const el = $('#dr-launch-boost'), ok = launchSupported(st);
  const running = !!(st && st.running), srvOn = ok && +st.params.launch_boost === 1;
  el.disabled = !ok || (running && !srvOn);                   // 走行中は OFF にだけできる
  if (!ok) el.checked = false;
  const sp = launchSpec(st);
  $('#dr-launch-spec').textContent = `(${launchSpecText(sp)})`;
  $('#dr-launch-panel').classList.toggle('on', el.checked);
  const note = $('#dr-launch-note'), lv = (st && st.launch) || {}, s = (st && st.stats) || {};
  if (!ok) note.innerHTML = '<span class="muted">今のサーバーは発進の前進に対応していません(管理画面のサーバーを再起動すると使えます)</span>';
  else if (state.drive.pending.launch_boost) note.innerHTML = '<span class="muted">基準の設定で OFF にしました(走行開始で送ります)</span>';
  else if (srvOn || (running && (lv.count || lv.last_end))) {
    const live = running ? ` ・ 出した ${s.launch_boosts ?? lv.count ?? 0} 回${lv.active ? ' ・ <b>出しています</b>' : ''}`
      + `${lv.last_end ? ` ・ 最後の終わり: ${esc(lv.last_end)}` : ''}${lv.last_skip ? ` ・ 始めなかった: ${esc(lv.last_skip)}` : ''}` : '';
    note.innerHTML = (srvOn ? '<b class="warn-text">ON: AI 側にした最初のフレームから強い前進を出します</b>' : 'OFF') + live;
  } else note.innerHTML = '';
}
$('#dr-launch-boost').addEventListener('change', async () => {
  const el = $('#dr-launch-boost'), st = state.drive.st || {}, on = el.checked;
  if (on) {
    if (st.running) { el.checked = false; toast('走行中は発進の前進を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)', true); return; }
    if (!confirm(launchConfirmText(launchSpec(st)))) { el.checked = false; return; }
  }
  try {
    const p = await postDriveParams({ launch_boost: on ? 1 : 0 }, ['launch_boost']);    // 送れなければ pending も戻す(r1-state3-2)
    if (p && p.launch_boost === undefined) { el.checked = false; toast('今のサーバーは発進の前進に対応していません(再起動が必要)', true); }
    else {
      if (state.drive.st) state.drive.st.params = p;
      toast(on ? '発進の前進を ON にしました' : '発進の前進を OFF にしました');
    }
  } catch (e) { toast(e.message, true); el.checked = !on; }
  if (state.drive.st) renderLaunchSwitch(state.drive.st);
});
// 試験の選択肢のスイッチ(2026-10-26 fix3): リカバリーの後ろの見張り(rec_back_guard。fix2 safety2-1)・中立止めのあともリカバリーを
// 残す(rec_hold_keep。fix3 safety1-1)・発進の前進を出さない区間は発進の補助も出さない(launch_no_kick。fix3 safety1-2)。
// drive.py ではどれも既定 0・起動のたびに 0。作りは発進の前進のスイッチと同じ(既定 OFF・注意書き・ON の確認の文・走行中は ON に
// できない(OFF はできる)・走行開始の確認の文に ON の行(trimGuardStartLines))。発進の前進と違うところ:
// - 基準の設定・ブロックでは変えない(ON にも OFF にもしない)。トリムと同じく車・日の設定(fix2 state2-1 の決め)。手順書 3 章の
//   「朝に 1 回入れる」のまま使えるように(OFF にすると、朝に入れた ON がブロックを押すたびに黙って消える)
// - 走行開始では送らない(readDriveParams に入れない)。スイッチを押したらすぐ送り、走行開始のときはサーバーの値のまま
//   (前と同じ。API で入れても、画面の古い表示で上書きしない)。スイッチの表示はサーバーの値に合わせる
// - サーバーに無い項目(再起動の前の古いサーバー)はスイッチを出さず、「今のサーバーには無い」の 1 行だけ出す
const OPT_SWITCHES = [
  { key: 'rec_back_guard', sel: '#dr-rec-back-guard', group: 'rec', name: 'リカバリーの後ろの見張り',
    on: 'ON: リカバリーで下がった時間をどの段でも数え、後ろに当たったらすぐ前へ出します',
    text: 'リカバリーの「ブレーキ」「待つ」「中立」の段でも、後ろへ動いていたら下がった時間として数え、下がる時間に届いたら惰性にします。'
      + '後ろへ動いている(直前まで動いていた)ときの衝撃(2.5G 以上)は、後ろに当たったとしてすぐ前へ出します。'
      + '模型と記録の再生でしか確かめていません(実車の ESC ではまだ)。使う日は最初のリカバリーを人が見て、'
      + '1 m 以上下がる・後ろの物に近づくときはすぐ CH3 を人側へ' },
  // rec_hold_keep は後ろの見張りが ON のときだけ(drive._opt_rules。見張りを OFF にするとサーバーがこれも OFF にする)
  { key: 'rec_hold_keep', sel: '#dr-rec-hold-keep', group: 'rec', name: '中立止めのあともリカバリーを残す', needs: 'rec_back_guard',
    on: 'ON: AI 側のままの中立止めではリカバリーを戻しません(中止は CH3 を人側にするまで解けない)',
    text: '後ろの見張りが ON のときだけ使えます(後ろの見張りを OFF にすると、これも OFF)。AI 側のままの中立止め(カメラの遅れ・I2C の失敗など。1 フレームでも)で、リカバリーの中止・途中の段・ぶつかった印を消しません。'
      + 'OFF(今まで)は中立止め 1 回で全部消え、中止が解けるとモデルの弱い前進が壁を押し続けることがあります。'
      + '戻すのは CH3 を人側にしたとき・人側の中立止め・走行の開始。机上の試験だけで、実車ではまだです' },
  { key: 'launch_no_kick', sel: '#dr-launch-no-kick', group: 'launch', name: '発進の前進を出さない区間は発進の補助も出さない',
    on: 'ON: 発進の前進を出さないと決めた区間では、発進の補助も出しません',
    text: '発進の前進が ON で、前が近い・ToF が無い・車両の値が古いなどで「出さない」と決めた区間では、'
      + '発進の補助(止まっていてモデルが進まないときの弱い前進を最大 2 秒)も出しません。発進の前進が OFF のときは何も変えません。'
      + '机上の試験だけで、実車ではまだです' },
];
const optSupported = (st, key) => !!(st && st.params && st.params[key] !== undefined);
const optName = (key) => (OPT_SWITCHES.find((o) => o.key === key) || { name: key }).name;
const optBusy = {};                                   // 送っている間は表示の同期でスイッチを戻さない
function optConfirmText(o) {
  return `${o.name}を ON にします。\n\n${o.text}。\n\n`
    + '試験の選択肢です。既定は OFF で、管理画面を起動するたびに OFF に戻ります。走行中は ON にできません(OFF にはできます)。'
    + '基準の設定・ブロックでは変わりません(ON は OFF にするまで次の走行にも残ります)。\n\nユーザーの了承は出ていますか?';
}
function renderOptSwitches(st) {
  const running = !!(st && st.running), p = (st && st.params) || {}, missing = { rec: [], launch: [] };
  let recOn = false;
  for (const o of OPT_SWITCHES) {
    const el = $(o.sel), line = $(`#dr-opt-line-${o.key}`), note = $(o.sel + '-note');
    if (!el) continue;                                  // 古い index.html(キャッシュ)
    const ok = optSupported(st, o.key), srvOn = ok && +p[o.key] === 1;
    if (line) line.classList.toggle('hidden', !ok);
    if (!ok) {
      el.checked = false; el.disabled = true; missing[o.group].push(o.name);
      if (note) note.innerHTML = '';
      continue;
    }
    const needOk = !o.needs || +p[o.needs] === 1;       // 先に ON にしておく選択肢(rec_hold_keep → 後ろの見張り)
    if (!optBusy[o.key]) el.checked = srvOn;
    el.disabled = (running && !srvOn) || (!needOk && !srvOn);   // 走行中は OFF にだけできる
    if (note) note.innerHTML = srvOn ? `<b class="warn-text">${esc(o.on)}</b>`
      : (!needOk ? `<span class="muted">${esc(optName(o.needs))}が OFF のあいだは押せません</span>` : '');
    if (srvOn && o.group === 'rec') recOn = true;
  }
  for (const g of ['rec', 'launch']) {
    const m = $(`#dr-opt-${g}-missing`);
    if (m) m.textContent = missing[g].length ? `今のサーバーには無い選択肢: ${missing[g].join('・')}(管理画面のサーバーを再起動すると出ます)` : '';
  }
  const panel = $('#dr-opt-rec-panel');
  if (panel) panel.classList.toggle('on', recOn);
}
for (const o of OPT_SWITCHES) {
  const el0 = $(o.sel);
  if (!el0) continue;
  el0.addEventListener('change', async () => {
    const el = $(o.sel), st = state.drive.st || {}, on = el.checked;
    const before = Object.assign({}, st.params || {});
    if (on) {
      if (st.running) { el.checked = false; toast(`走行中は${o.name}を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)`, true); return; }
      if (o.needs && +before[o.needs] !== 1) { el.checked = false; toast(`${o.name}は、${optName(o.needs)}が ON のときだけ使えます(先に${optName(o.needs)}を ON に)`, true); return; }
      if (!confirm(optConfirmText(o))) { el.checked = false; return; }
    }
    optBusy[o.key] = true;
    try {
      const p = await api('POST', '/api/drive/params', { [o.key]: on ? 1 : 0 });
      if (p && p[o.key] === undefined) { el.checked = false; toast(`今のサーバーは${o.name}に対応していません(再起動が必要)`, true); }
      else {
        if (state.drive.st) state.drive.st.params = p;
        // OFF にしたら、それを要る選択肢もサーバーが OFF にする(後ろの見張り → 中立止めのあとも残す)
        const also = OPT_SWITCHES.filter((x) => x.needs === o.key && +before[x.key] === 1 && +p[x.key] !== 1).map((x) => x.name);
        toast(`${o.name}を ${on ? 'ON' : 'OFF'} にしました${also.length ? `(${also.join('・')}も OFF)` : ''}`);
      }
    } catch (e) { toast(e.message, true); el.checked = !on; }
    // 返事の順番(2026-10-27 fix3 r2-r2-state-2): この送信の返事より前に出た表示の問い合わせの返事では、この項目とそれを要る項目
    // (後ろの見張り → 中立止めのあとも残す)を戻さない(renderDrive の since。天井などの postDriveParams と同じ)。前は返事のあとすぐ
    // optBusy を下ろすだけで、古い問い合わせの返事がスイッチと確認の文(trimGuardStartLines)を約 1 秒、前の値に戻した
    drAcked([o.key].concat(OPT_SWITCHES.filter((x) => x.needs === o.key).map((x) => x.key)));
    optBusy[o.key] = false;
    if (state.drive.st) renderOptSwitches(state.drive.st);
  });
}
// 走る前のチェックの行に足す文: ON の試験の選択肢(基準の設定・ブロックでは変えない値なので、基準と同じかの判定とは別に出す)
function optNote(st) {
  const p = (st && st.params) || {};
  const on = OPT_SWITCHES.filter((o) => +p[o.key] === 1).map((o) => o.name);
  return on.length ? ` <span class="muted">・${esc(on.join('・'))}: ON(基準の設定・ブロックでは変えない)</span>` : '';
}
const fmtN = (x, d = 2) => (x === undefined || x === null || !Number.isFinite(+x) ? '-' : (+x).toFixed(d));
// 走行中の表示(drive.py の now.ceil: on・error・active・left・cap・v・mode(名前)・applied・thr_us)
function ceilLiveText(c, s) {
  if (c.error) return c.on ? `<span class="warn-text">計算が失敗したので外しました(${esc(c.error)})</span>`
    : `<span class="warn-text">OFF(記録だけ)の影の判定が計算の失敗で止まりました。出力は変わりません(${esc(c.error)})</span>`;
  const mode = c.mode ? String(c.mode) : '';
  return `${c.on ? '<b class="warn-text">ON</b>' : 'OFF(記録だけ)'} ・ 天井 ${c.active ? `${fmtN(c.cap)} m/s` : '-'}${c.left ? '(左旋回中)' : ''}`
    + ` ・ 速さ ${fmtN(c.v)} m/s${mode ? ` ・ ${esc(mode)}${c.on ? '' : '(ON なら)'}` : ''}`
    + ` ・ 出力を変えた ${s.ceil_applied ?? 0} フレーム(うち中立 ${s.ceil_neutral ?? 0})${s.ceil_ms != null ? ` ・ 計算 ${fmtN(s.ceil_ms, 3)} ms` : ''}`;
}
const SA_SRC = { 0: 'なし(1.00)', 1: '基板の電圧', 2: '説明欄の V_before' };
// 走行中の表示(drive.py の now.auto: on・error・m・m_ff・m_fb・gain・v・src・lap_p90・lap_s・laps_used・warn・warn_text)
function autoLiveText(a, st) {
  if (a.error) return `<span class="warn-text">計算が失敗したので外しました(速度ゲインは設定のまま。${esc(a.error)})</span>`;
  const sources = speedSpec(st).sources || SA_SRC;
  const src = sources[String(a.src)] || SA_SRC[a.src] || '-';
  return `${a.on ? '<b class="warn-text">ON</b>' : 'OFF(記録だけ)'} ・ m ${fmtN(a.m, 3)}(電圧 ${fmtN(a.m_ff, 3)} × 周 ${fmtN(a.m_fb, 3)})`
    + ` ・ ゲイン ${fmtN(a.gain, 3)}${a.on ? '' : '(ON なら)'} ・ 電圧 ${a.v != null ? `${fmtN(a.v)} V` : '-'}(${esc(src)})`
    + (a.lap_p90 != null ? ` ・ 前の周 p90 ${fmtN(a.lap_p90)} m/s・${fmtN(a.lap_s)} 秒` : '')
    + ` ・ 使った周 ${a.laps_used ?? 0}` + (a.warn_text ? ` ・ <b class="warn-text">${esc(a.warn_text)}</b>` : '');
}
// 走り終わり(drive.py の auto_end。speed_auto.end_summary)。速度ゲインを変えるのは人(1 走行 = 1 設定)
function autoEndText(e) {
  if (e.error) return `<span class="warn-text">計算の失敗: ${esc(e.error)}</span>`;
  const parts = [`${e.on ? 'ON' : 'OFF(記録だけ)'}`, `使った周 ${e.laps_used ?? 0}`];
  if (e.m_fb_end != null) parts.push(`終わりの周の直し ${fmtN(e.m_fb_end, 3)}`);
  // 目安(2026-10-25 fix.md): 使った周ごとに「目標の周の時間に合ったはずのゲイン」の中央値。ON・OFF とも両方を出す
  const nl = e.suggest_laps ? `(${esc(e.suggest_laps)} 周から${e.target_lap_s != null ? `・目標 1 周 ${fmtN(e.target_lap_s)} 秒` : ''})` : '';
  if (e.suggest_throttle_gain_auto != null) parts.push(`次も自動合わせなら速度ゲインの目安 <b>${fmtN(e.suggest_throttle_gain_auto, 3)}</b>${nl}`);
  else if (e.on && e.suggest_throttle_gain != null) parts.push(`次も自動合わせなら速度ゲインの目安 <b>${fmtN(e.suggest_throttle_gain, 3)}</b>`);
  if (e.suggest_throttle_gain_fixed != null) parts.push(`使わないなら <b>${fmtN(e.suggest_throttle_gain_fixed, 3)}</b>`);
  if (e.warn_text) parts.push(`<b class="warn-text">${esc(e.warn_text)}</b>`);
  return parts.join(' ・ ') + ' <span class="muted">(画面に出すだけ。速度ゲインは人が変える)</span>';
}

// ------------------------------------------------------------ シミュレーター(CourseLab)
state.sim = { timer: null, st: null, courses: [], models: [], defaultCourse: 'default' };
const SIM_BRIDGE_LABEL = { stopped: '停止中', starting: '起動中…', running: '動作中', stopping: '停止中…', failed: '失敗' };
const SIM_KIND_LABEL = { release: '標準', snapshot: '同梱', 'trusted-js': 'アップロード', data: 'アップロード・簡易描画' };
const simMB = (b) => (b / 1e6).toFixed(b >= 1e8 ? 0 : 1) + ' MB';
const simJsonError = (xhr) => { try { const d = JSON.parse(xhr.responseText); return d.errors || [d.detail || xhr.statusText]; } catch (e) { return [xhr.responseText || ('HTTP ' + xhr.status)]; } };

async function loadSim() {
  await Promise.all([loadSimStatus(), loadSimCourses(), loadSimModels(), loadSimDatasets(), loadSimJobs(), loadSimRl()]);
  clearTimeout(state.sim.timer);
  let n = 0;
  const tick = async () => {
    if (state.tab !== 'sim') return;
    await loadSimStatus(); await loadSimJobs();
    if ($('#sim-log-box').open) loadSimLog();
    if (++n % 3 === 0) loadSimRlPlans();
    state.sim.timer = setTimeout(tick, 2000);
  };
  state.sim.timer = setTimeout(tick, 2000);
}
async function loadSimStatus() {
  let s;
  try { s = await api('GET', '/api/sim/status'); } catch (e) { $('#sim-release').textContent = '状態を取得できません: ' + e.message; return; }
  const prev = state.sim.st; state.sim.st = s;
  renderSimRelease(s); renderSimEnv(s); renderSimBridge(s);
  if (prev && prev.release.id !== s.release.id) loadSimCourses();
}
function renderSimRelease(s) {
  const r = s.release;
  $('#sim-inbox-dir').textContent = s.inbox_dir;
  $('#sim-rel-pill').innerHTML = r.installed ? '<span class="badge ok">導入済み</span>' : '<span class="badge warn">未導入</span>';
  $('#sim-release').innerHTML = r.installed
    ? `使っている版: <b>${esc(r.id)}</b>(${esc((r.created_at || '').slice(0, 16).replace('T', ' '))} 作成・${esc((r.installed_at || '').slice(0, 16).replace('T', ' '))} 導入)<br>`
      + `Default コース: ${esc(r.default_world_version || '?')}` + (r.warnings && r.warnings.length ? `<div class="muted">${r.warnings.map(esc).join('<br>')}</div>` : '')
    : 'まだ導入していません。下の ZIP の「導入」を押してください。';
  const inbox = s.inbox || [];
  $('#sim-inbox').innerHTML = `<div class="card-title" style="margin-top:8px"><small>${esc(s.inbox_dir)} にある ZIP</small></div>` + (inbox.map((z) => `
    <div class="sim-row"><span class="grow"><b>${esc(z.file)}</b> <span class="muted">${simMB(z.bytes)}</span>
      ${z.kind === 'simulator' ? `<span class="badge sim">シミュレーター本体 ${esc(z.release_id)}</span>` : `<span class="badge tag">対象外</span>`}
      ${z.problem ? `<div class="muted small">${esc(z.problem)}</div>` : ''}</span>
      ${z.kind === 'simulator' ? (z.installed ? '<span class="badge ok">導入済み</span>'
        : `<button class="small primary" data-sim-install="${esc(z.file)}" ${s.op_running ? 'disabled' : ''}>導入</button>`) : ''}</div>`).join('')
    || '<div class="muted small">ZIP がありません(Jetson のこのフォルダーにシミュレーターの ZIP を置いてください)</div>');
  const rels = s.releases || [];
  $('#sim-releases').innerHTML = rels.length < 2 ? '' : '<div class="card-title" style="margin-top:8px"><small>導入済みの版</small></div>' + rels.map((x) => `
    <div class="sim-row"><span class="grow">${esc(x.id)} ${x.active ? '<span class="badge ok">使用中</span>' : ''} <span class="muted small">${x.bytes ? simMB(x.bytes) : ''}</span></span>
      ${x.active ? '' : `<button class="small" data-sim-activate="${esc(x.id)}">この版を使う</button><button class="small danger" data-sim-delrel="${esc(x.id)}">削除</button>`}</div>`).join('');
}
function renderSimEnv(s) {
  const e = s.env;
  const p = e.packages || {};
  $('#sim-env-pill').innerHTML = e.ready ? '<span class="badge ok">準備済み</span>' : e.problem ? '<span class="badge bad">要対応</span>' : '<span class="badge warn">未準備</span>';
  $('#sim-env').innerHTML = e.ready
    ? `Python ${esc(e.python_version)} / onnxruntime ${esc(p.onnxruntime)} / numpy ${esc(p.numpy)} / opencv ${esc(p['opencv-python-headless'])}`
      + `<span class="muted">(${esc((e.created_at || '').slice(0, 16).replace('T', ' '))} ${e.source === 'offline' ? '・手元の wheel から' : ''})</span>`
    : (e.problem ? `<span class="warn-text">${esc(e.problem)}</span>` : '未準備です。「準備する」を押してください(本体の導入が先です)。');
  const busy = s.op_running || ['starting', 'running', 'stopping'].includes(s.bridge.state);
  $('#btn-sim-setup').classList.toggle('hidden', e.ready);
  $('#btn-sim-setup').disabled = busy || !s.release.installed;
  $('#btn-sim-resetup').disabled = busy || !s.release.installed;
}
function renderSimBridge(s) {
  const b = s.bridge;
  const cls = { running: 'run', failed: 'bad', starting: 'warn', stopping: 'warn' }[b.state] || 'tag';
  $('#sim-bridge-pill').innerHTML = `<span class="badge ${cls}">${esc(SIM_BRIDGE_LABEL[b.state] || b.state)}</span>`;
  const up = b.started_at ? Math.round(Date.now() / 1000 - b.started_at) : null;
  const lines = [];
  if (b.state === 'running') {
    lines.push(`127.0.0.1:${b.port} ・ PID ${b.pid} ・ 動作 ${up >= 3600 ? (up / 3600).toFixed(1) + ' 時間' : up >= 60 ? Math.round(up / 60) + ' 分' : up + ' 秒'}${b.rss_mb ? ' ・ メモリ ' + Math.round(b.rss_mb) + ' MB' : ''}`);
    lines.push(b.policy_loaded ? `モデル: <b>${esc(b.policy_name || '')}</b>` : 'モデル: 未読み込み(シミュレーターの画面で読み込みます)');
    if (b.recording) lines.push(`<span class="warn-text">収集中(${esc(b.frames ?? '')} フレーム)</span>`);
  }
  if (b.restarts) lines.push(`<span class="warn-text">異常終了のため ${b.restarts} 回起動し直しました</span>`);
  if (b.last_error) lines.push(`<span class="warn-text">${esc(b.last_error)}</span>`);
  if (!s.release.installed) lines.push('先にシミュレーター本体を導入してください。');
  else if (!s.env.ready) lines.push('先に Python 環境を準備してください。');
  if (s.busy && s.busy.training && b.state === 'running') lines.push('<span class="muted">学習中は推論が遅くなります。</span>');
  $('#sim-bridge').innerHTML = lines.join('<br>');
  $('#btn-sim-start').disabled = !(s.release.installed && s.env.ready) || ['starting', 'running', 'stopping'].includes(b.state);
  $('#btn-sim-stop').disabled = !['starting', 'running', 'failed'].includes(b.state);
}
async function loadSimLog() {
  try {
    const r = await api('GET', '/api/sim/bridge/log?lines=200');
    const sup = r.suppressed || {};
    $('#sim-log').textContent = r.lines.join('\n') + `\n(毎フレームの記録は省略: 推論 ${sup.predict || 0} 件・収集 ${sup.frame || 0} 件)`;
  } catch (e) { $('#sim-log').textContent = e.message; }
}
$('#sim-log-box').addEventListener('toggle', () => { if ($('#sim-log-box').open) loadSimLog(); });
async function loadSimCourses() {
  let r;
  try { r = await api('GET', '/api/sim/courses'); } catch (e) { $('#sim-courses').textContent = e.message; return; }
  state.sim.courses = r.courses; state.sim.defaultCourse = r.default;
  $('#sim-courses').innerHTML = r.courses.map((c) => {
    const st = c.stats || {};
    return `<div class="sim-course ${c.is_default ? 'default' : ''}">
      <div><b>${esc(c.name)}</b> ${c.is_default ? '<span class="badge ok">既定</span>' : ''}
        <span class="badge ${c.renderer === 'generic' ? 'warn' : 'sim'}">${esc(SIM_KIND_LABEL[c.kind] || c.kind)}</span></div>
      <div class="meta">${esc(c.world_version || '')} ・ hash ${esc(c.world_hash || '未計算')}
        ${st.segments != null ? ` ・ 線分 ${st.segments}・柱 ${st.disks}・路面 ${st.surfaces}・ゲート ${st.gates}` : ''}
        ${c.size_m ? ` ・ ${Number(c.size_m[0]).toFixed(2)} × ${Number(c.size_m[1]).toFixed(2)} m` : ''}</div>
      <div class="meta">使える教師: ${esc((c.features.teachers || []).join('・'))}${c.features.collection_plan ? ' ・ 20分・60シーンの収集条件' : ''}</div>
      ${(c.warnings || []).map((w) => `<div class="warn-text">${esc(w)}</div>`).join('')}
      <div class="row tight">
        <button class="small primary" data-sim-open="${esc(c.id)}">開く</button>
        ${c.is_default ? '' : `<button class="small" data-sim-default="${esc(c.id)}">既定にする</button>`}
        <a class="small" href="/api/sim/courses/${encodeURIComponent(c.id)}/package.zip" download>書き出し(雛形)</a>
        ${c.deletable ? `<button class="small danger" data-sim-delcourse="${esc(c.id)}">削除</button>` : ''}
      </div></div>`;
  }).join('') || '<div class="muted small">コースがありません(シミュレーター本体を導入すると Default が出ます)</div>';
  const sel = $('#sim-open-course'), cur = sel.value;
  sel.innerHTML = r.courses.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}${c.renderer === 'generic' ? '(簡易描画)' : ''}${c.is_default ? ' ★' : ''}</option>`).join('');
  sel.value = r.courses.some((c) => c.id === cur) ? cur : r.default;
  updateSimOpenNote();
}
async function loadSimModels() {
  try { state.sim.models = await api('GET', '/api/sim/models'); } catch (e) { state.sim.models = []; }
  const sel = $('#sim-open-model'), cur = sel.value;
  sel.innerHTML = '<option value="">なし(手動・デモ・あとで読み込む)</option>' + state.sim.models.map((m) =>
    `<option value="${esc(m.name)}" ${m.sim_compatible ? '' : 'disabled'}>${esc(m.name)}${m.sim_compatible ? '' : '(' + esc(m.reason) + ')'}</option>`).join('');
  if ([...sel.options].some((o) => o.value === cur && !o.disabled)) sel.value = cur;
}
function updateSimOpenNote() {
  const c = state.sim.courses.find((x) => x.id === $('#sim-open-course').value);
  $('#sim-open-note').textContent = !c ? '' : c.renderer === 'generic'
    ? '簡易描画のコースです(JetFormer の汎用描画。カメラ画像は Default と同じにはなりません)。使える教師は低速・高速だけです。'
    : c.id === 'default' ? '' : '使える教師は低速・高速だけです(RC 加速教師・軌跡最適化教師・20分の収集条件は Default 専用)。';
}
$('#sim-open-course').addEventListener('change', updateSimOpenNote);
function openSim(cid, run) {
  const s = state.sim.st;
  if (s && s.release.installed && s.env.ready && ['stopped', 'failed'].includes(s.bridge.state)) {
    api('POST', '/api/sim/bridge/start').then(loadSimStatus).catch((e) => toast('ブリッジを起動できません: ' + e.message, true));
  }
  window.open(`/sim/c/${encodeURIComponent(cid)}/` + (run ? `?model=${encodeURIComponent(run)}` : ''), '_blank', 'noopener');
}
$('#btn-sim-open').addEventListener('click', () => {
  if (!state.sim.st || !state.sim.st.release.installed) { toast('先にシミュレーター本体を導入してください', true); return; }
  openSim($('#sim-open-course').value || state.sim.defaultCourse, $('#sim-open-model').value);
});
async function loadSimDatasets() {
  let r;
  try { r = await api('GET', '/api/sim/datasets'); } catch (e) { $('#sim-datasets').textContent = e.message; return; }
  const rows = r.datasets.map((d) => `
    <div class="sim-row"><span class="grow"><b>${esc(d.filename)}</b>
      <span class="badge ${d.kind === 'collection' ? 'sim' : 'tag'}">${d.kind === 'collection' ? 'まとめ' : 'シーン'}</span>
      ${d.imported ? `<span class="badge ok">取り込み済み: ${esc(d.import_name)}</span>` : ''}
      <div class="muted small">${simMB(d.bytes)} ・ ${esc(d.created_at.replace('T', ' '))}${d.scenes != null ? ` ・ ${d.scenes} シーン` : ''}${d.frames != null ? ` ・ ${d.frames} フレーム` : ''}${d.duration_s != null ? ` ・ ${d.duration_s} 秒` : ''}</div></span>
      <button class="small primary" data-sim-import="${esc(d.session_id)}" ${d.imported ? 'disabled' : ''}>JetFormer に取り込む</button>
      <button class="small danger" data-sim-deldata="${esc(d.session_id)}">削除</button></div>`).join('');
  const inc = r.incomplete.filter((x) => !x.recording);
  const incRows = inc.length ? `<div class="muted small" style="margin-top:6px">保存されていない作業フォルダー ${inc.length} 件(${simMB(inc.reduce((a, x) => a + x.bytes, 0))}):
      ${inc.map((x) => `<button class="small" data-sim-deldata="${esc(x.session_id)}">${esc(x.session_id.slice(0, 8))} を削除</button>`).join(' ')}</div>` : '';
  $('#sim-datasets').innerHTML = (rows || '<div class="muted small">まだありません(シミュレーターの「データセット収集」で作ると、ここに出ます)</div>') + incRows
    + (r.free_gb != null ? `<div class="muted small">空き ${r.free_gb} GB(シミュレーターのまとめ ZIP は「20 GiB + 元データの 2 倍」の空きが必要です)</div>` : '');
}
async function loadSimJobs() {
  let list = [];
  try { list = await api('GET', '/api/jobs'); } catch (e) { return; }
  const mine = list.filter((j) => ['sim_install', 'sim_setup', 'import'].includes(j.kind)).slice(0, 6);
  $('#sim-jobs').innerHTML = mine.map((j) => `
    <div class="job"><b>${esc(j.title)}</b> <span class="muted">${j.state === 'running' ? '実行中' : j.state === 'done' ? '完了' : 'エラー'}</span>
      <div class="progress blue"><div class="bar" style="width:${Math.round(j.progress * 100)}%"></div></div>
      <div class="muted">${esc(j.message)}</div>${j.error ? `<div class="err">${esc(j.error)}</div>` : ''}</div>`).join('')
    || '<div class="muted small">ありません</div>';
  const running = mine.some((j) => j.state === 'running');
  if (state.sim.jobsRunning && !running) { loadSimCourses(); loadSimDatasets(); loadSimRl(); }
  state.sim.jobsRunning = running;
}
async function simAction(fn, okMsg) {
  try { await fn(); if (okMsg) toast(okMsg); } catch (e) { toast(e.message, true); }
  loadSimStatus(); loadSimJobs();
}
$('#btn-sim-setup').addEventListener('click', () => simAction(() => api('POST', '/api/sim/setup', { force: false }), '準備を始めました(処理状況に進み具合が出ます)'));
$('#btn-sim-resetup').addEventListener('click', () => {
  if (!confirm('ブリッジ用の Python 環境(data/sim/venv)を消して作り直します。よろしいですか?')) return;
  simAction(() => api('POST', '/api/sim/setup', { force: true }), '作り直しを始めました');
});
$('#btn-sim-start').addEventListener('click', () => simAction(() => api('POST', '/api/sim/bridge/start'), 'ブリッジを起動しています'));
$('#btn-sim-stop').addEventListener('click', async () => {
  try { await api('POST', '/api/sim/bridge/stop', {}); toast('ブリッジを止めました'); }
  catch (e) {
    if (!confirm(e.message + '\n\nそれでも止めますか?')) return;
    try { await api('POST', '/api/sim/bridge/stop', { force: true }); toast('ブリッジを止めました'); } catch (e2) { toast(e2.message, true); }
  }
  loadSimStatus();
});
$('#sim-course-file').addEventListener('change', (e) => {
  const f = e.target.files[0]; e.target.value = '';
  if (!f) return;
  const out = $('#sim-course-upload');
  const xhr = new XMLHttpRequest();
  xhr.open('PUT', `/api/sim/courses/import?filename=${encodeURIComponent(f.name)}&replace=${$('#sim-course-replace').checked ? 1 : 0}`);
  xhr.upload.onprogress = (ev) => { out.textContent = `送信中 ${Math.round(100 * ev.loaded / (ev.total || 1))}%`; };
  xhr.onload = () => {
    if (xhr.status < 300) {
      const r = JSON.parse(xhr.responseText); const c = r.course;
      out.innerHTML = `<span class="badge ok">${r.already_registered ? '登録済みでした' : '登録しました'}</span> ${esc(c.name)}(${esc(c.world_version)}・hash ${esc(c.world_hash)}・${esc(SIM_KIND_LABEL[c.kind] || c.kind)})`
        + (c.warnings || []).map((w) => `<div class="warn-text">${esc(w)}</div>`).join('');
      loadSimCourses();
    } else {
      out.innerHTML = '<span class="badge bad">登録できません</span><ul class="sim-errors">' + simJsonError(xhr).map((m) => `<li>${esc(m)}</li>`).join('') + '</ul>';
    }
  };
  xhr.onerror = () => { out.textContent = '送信できませんでした(接続が切れた)'; };
  xhr.setRequestHeader('Content-Type', 'application/zip');
  xhr.send(f);
});
$('#tab-sim').addEventListener('click', async (e) => {
  const t = (sel) => e.target.closest(sel);
  let b;
  if ((b = t('[data-sim-install]'))) { simAction(() => api('POST', '/api/sim/install', { file: b.dataset.simInstall, activate: true }), '導入を始めました(処理状況に進み具合が出ます)'); }
  else if ((b = t('[data-sim-activate]'))) { await simAction(() => api('POST', `/api/sim/releases/${encodeURIComponent(b.dataset.simActivate)}/activate`), 'この版を使います'); loadSimCourses(); }
  else if ((b = t('[data-sim-delrel]'))) { if (confirm(`導入済みの版 ${b.dataset.simDelrel} を削除します。`)) simAction(() => api('DELETE', `/api/sim/releases/${encodeURIComponent(b.dataset.simDelrel)}`), '削除しました'); }
  else if ((b = t('[data-sim-open]'))) { openSim(b.dataset.simOpen, $('#sim-open-model').value); }
  else if ((b = t('[data-sim-default]'))) { try { await api('POST', `/api/sim/courses/${encodeURIComponent(b.dataset.simDefault)}/default`); loadSimCourses(); } catch (err) { toast(err.message, true); } }
  else if ((b = t('[data-sim-delcourse]'))) {
    if (!confirm(`コース ${b.dataset.simDelcourse} を削除します(開いているシミュレーターは、次に読み込むときに使えなくなります)。`)) return;
    try {
      await api('DELETE', `/api/sim/courses/${encodeURIComponent(b.dataset.simDelcourse)}`);
      $('#sim-course-upload').textContent = '';           // 「登録しました」の表示を残さない
      toast(`コース ${b.dataset.simDelcourse} を削除しました`); loadSimCourses();
    } catch (err) { toast(err.message, true); }
  }
  else if ((b = t('[data-sim-import]'))) { await simAction(() => api('POST', `/api/sim/datasets/${b.dataset.simImport}/import`), '取り込みを始めました(処理状況に進み具合が出ます)'); }
  else if ((b = t('[data-sim-deldata]'))) {
    if (!confirm('シミュレーターの作業フォルダー(ZIP・展開したもの)を削除します。JetFormer に取り込んだデータセットは残ります。')) return;
    try { await api('DELETE', `/api/sim/datasets/${b.dataset.simDeldata}`); loadSimDatasets(); } catch (err) { toast(err.message, true); }
  }
});

// ------------------------------------------------------------ シミュレーター: 強化学習モード(jetformer/sim/rl_modes.py)
state.simRl = { modes: [], minutes: {}, plans: [] };
const RL_BADGE = { accepted: 'ok', not_accepted: 'warn', pending: 'sim', not_published: 'tag', broken: 'bad', untested: 'tag', training: 'sim' };
const rlNum = (v, d = 2) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
async function loadSimRl() {
  let r;
  try { r = await api('GET', '/api/sim/rl/modes'); }
  catch (e) { $('#sim-rl-modes').innerHTML = `<div class="muted small">${esc(e.message)}</div>`; $('#sim-rl-card').classList.toggle('hidden', /止められて/.test(e.message)); return; }
  state.simRl.modes = r.modes; state.simRl.limits = { max: r.max_minutes, def: r.default_minutes, perMin: r.scenes_per_minute };
  const sel = r.modes.filter((m) => m.selectable).length;
  $('#sim-rl-pill').innerHTML = `<span class="badge rl">${sel} モードを選べます</span>`;
  $('#sim-rl-modes').innerHTML = r.modes.map((m) => {
    const c = m.card || {};
    const nums = [];
    if (c.lap_from_stop_s != null) nums.push(['停止から 1 周', `${rlNum(c.lap_from_stop_s)} 秒` + (c.flying_lap_s != null ? `(走りながら ${rlNum(c.flying_lap_s)} 秒)` : '')]);
    if (c.tests) nums.push(['衝突', c.tests.text_ja]);
    if (c.min_clearance_cm) nums.push(['最小すき間', `${rlNum(c.min_clearance_cm.value, c.min_clearance_cm.value < 0.1 ? 2 : 1)} cm(${c.min_clearance_cm.text_ja})`]);
    for (const x of c.main || []) nums.push([x.label_ja, x.value_ja]);
    const ds = m.datasets || { count: 0 };
    nums.push(['集めたデータ', (ds.count ? `${ds.count} データセット・${ds.scenes} シーン・${rlNum(ds.minutes, 1)} 分` : 'まだありません')
      + (ds.superseded?.count ? `(ほかに前の版の方策で記録した ${ds.superseded.count} データセット)` : '')]);
    const checks = (c.checks || []).map((k) => `<li>${k.pass === true ? '✅' : k.pass === false ? '❌' : '⏳'} ${esc(k.label_ja)}</li>`).join('');
    const notes = (c.notes_ja || []).map((x) => `<li>${esc(x)}</li>`).join('');
    const probs = (m.problems || []).map((x) => `<div class="err small">${esc(x)}</div>`).join('') + (m.warnings || []).map((x) => `<div class="warn-text small">${esc(x)}</div>`).join('');
    return `<div class="sim-rl-mode ${m.selectable ? '' : 'off'}">
      <div class="head"><b>${esc(m.name)}</b><span class="badge ${RL_BADGE[m.state] || 'tag'}">${esc(m.label)}</span>
        <span class="grow"></span>${m.selectable ? `<button class="small primary" data-sim-rl-open="${esc(m.key)}">このモードで開く</button>` : ''}</div>
      <div class="muted small">${esc(m.short || '')}</div>
      ${c.headline_ja ? `<div class="headline">${esc(c.headline_ja)}</div>` : ''}
      <div class="nums">${nums.map(([a, b]) => `<span>${esc(a)}</span><span>${esc(b)}</span>`).join('')}</div>
      ${probs}
      <details><summary>試験の結果と注意</summary>${c.verdict_ja ? `<div class="muted">${esc(c.verdict_ja)}</div>` : ''}<ul>${checks}</ul><ul>${notes}</ul>
        ${m.policy_sha256 ? `<div class="muted">方策 ${esc(m.policy_sha256.slice(0, 12))}…${m.pool ? `・記録のシーンの組 ${m.pool.scenes}(候補の ${Math.round(100 * (m.pool.acceptance_rate || 0))}% が合格)` : ''}</div>` : ''}</details>
    </div>`;
  }).join('') || '<div class="muted small">登録されたモードがありません(data/sim_rl/modes/)</div>';
  const planModes = r.modes.filter((m) => m.plan_ok);
  for (const m of planModes) if (!(m.key in state.simRl.minutes)) state.simRl.minutes[m.key] = m.state === 'accepted' ? r.default_minutes : 0;
  $('#sim-rl-plan-form').innerHTML = planModes.length ? `<table class="sim-rl-plan-table"><tr><th>モード</th><th>分</th><th>シーン</th></tr>${planModes.map((m) => `
    <tr><td>${esc(m.name)} <span class="badge ${RL_BADGE[m.state] || 'tag'}">${esc(m.label)}</span></td>
      <td><input type="number" min="0" max="${m.pool.max_minutes}" step="1" value="${state.simRl.minutes[m.key]}" data-sim-rl-min="${esc(m.key)}"></td>
      <td class="muted" data-sim-rl-scenes="${esc(m.key)}">${state.simRl.minutes[m.key] * r.scenes_per_minute}</td></tr>`).join('')}</table>`
    : '<div class="muted small">プランに使えるモードがありません</div>';
  state.simRl.plans = (r.plans || []);
  renderSimRlPlans();
}
async function loadSimRlPlans() {
  try { state.simRl.plans = (await api('GET', '/api/sim/rl/plans')).plans; renderSimRlPlans(); } catch (e) { /* なし */ }
}
const RL_PLAN_STATE = { ready: '未開始', running: '記録中', stopped: '止めた', finished: '記録済み', error: 'エラー' };
function renderSimRlPlans() {
  const plans = (state.simRl.plans || []).slice(0, 6);
  $('#sim-rl-plans').innerHTML = plans.length ? '<div class="card-title" style="margin-top:10px"><small>最近のプラン</small></div>' + plans.map((p) => {
    const modes = Object.entries(p.modes || {}).map(([k, v]) => {
      const imp = (p.imports || {})[k];
      const nm = (state.simRl.modes.find((m) => m.key === k) || {}).name || k;
      return `${esc(nm)} ${v.done}/${v.planned}${v.failed ? `(外れ ${v.failed})` : ''}${imp ? ` ・ 取り込み ${imp.state === 'done' ? '済み ' + esc(imp.dataset) : imp.state === 'running' ? '中' : 'エラー ' + esc(imp.error || '')}` : ''}`;
    }).join('<br>');
    return `<div class="sim-row"><span class="grow"><b>${esc(p.id)}</b> <span class="badge tag">${esc(RL_PLAN_STATE[p.state] || p.state)}</span>
      <div class="muted small">${modes}${p.message ? '<br>' + esc(p.message) : ''}</div></span>
      <a class="small" href="${esc(p.url)}" target="_blank" rel="noopener">開く</a></div>`;
  }).join('') : '';
}
$('#sim-rl-card').addEventListener('input', (e) => {
  const inp = e.target.closest('[data-sim-rl-min]'); if (!inp) return;
  const k = inp.dataset.simRlMin, v = Math.max(0, Math.min(Number(inp.max) || 20, Math.round(Number(inp.value) || 0)));
  state.simRl.minutes[k] = v;
  const cell = document.querySelector(`[data-sim-rl-scenes="${k}"]`); if (cell) cell.textContent = v * ((state.simRl.limits || {}).perMin || 3);
});
$('#sim-rl-card').addEventListener('click', (e) => {
  const b = e.target.closest('[data-sim-rl-open]'); if (!b) return;
  const s = state.sim.st;
  if (s && s.bridge.state !== 'running' && s.env.ready) api('POST', '/api/sim/bridge/start').catch(() => {});
  window.open(`/sim/c/default/?rl=${encodeURIComponent(b.dataset.simRlOpen)}`, '_blank', 'noopener');
});
$('#btn-sim-rl-plan').addEventListener('click', async () => {
  const minutes = {};
  for (const [k, v] of Object.entries(state.simRl.minutes)) if (v > 0) minutes[k] = v;
  if (!Object.keys(minutes).length) { toast('どのモードも 0 分です', true); return; }
  const body = { course: 'default', minutes, val_ratio: Number($('#sim-rl-val').value || 0.2), delete_sources: $('#sim-rl-delete').checked };
  const seed = $('#sim-rl-seed').value.trim();
  if (seed !== '') body.seed = Math.round(Number(seed));
  const w = window.open('', '_blank');
  try {
    const p = await api('POST', '/api/sim/rl/plans', body);
    const s = state.sim.st;
    if (s && s.bridge.state !== 'running' && s.env.ready) api('POST', '/api/sim/bridge/start').catch(() => {});
    toast(`プラン ${p.id} を作りました(${p.scenes} シーン)。新しいタブで「開始」を押してください`);
    if (w) w.location.href = p.url; else window.open(p.url, '_blank');
    loadSimRlPlans();
  } catch (err) { if (w) w.close(); toast('プランを作れません: ' + err.message, true); }
});

// ------------------------------------------------------------ 起動
loadModes();
loadSound();
showTab('live');
pollStatus();
pollValues();
loadScenes();

// デバッグ・自動テスト用(ブラウザのコンソールから操作できるように)
window.jetformer = { state, showTab, loadSound, openScene, showFrame, closeModal, loadScenes, loadDatasets, loadJobs, loadModes, loadTrain, pollTrain,
                     loadDrive, pollDrive, stopDrive, loadSim };


// ------------------------------------------------------------ センサーの向きの点検(ロードマップ S0)
function renderSensorCheck(r) {
  const el = $('#sensor-check-result');
  if (!r || !r.time) return;
  const t = new Date(r.time * 1000).toLocaleTimeString();
  const tof = r.tof_floor_pct ? `遠い床 ${r.tof_floor_pct['9'] ?? '-'}% ・ 近い床 ${r.tof_floor_pct['18'] ?? '-'}%(目安 ±${r.tof_tolerance_pct}%)` : 'ToF: 基準なし';
  const cam = r.bumper_rows !== undefined ? `バンパー ${r.bumper_rows} 行(学習データ ${r.bumper_target[0]}〜${r.bumper_target[1]} 行)` : 'カメラ: 映像なし';
  el.className = 'small';
  el.innerHTML = `<span class="badge ${r.ok ? 'ok' : 'warn'}">${r.ok ? 'OK' : '要確認'}</span> ${esc(t)} ・ ${esc(cam)} ・ ${esc(tof)}`
    + (r.notes || []).map((n) => `<div class="${n.startsWith('注意') ? 'muted' : 'warn-text'}">${esc(n)}</div>`).join('');
}
$('#btn-sensor-check').addEventListener('click', async () => {
  const b = $('#btn-sensor-check');
  b.disabled = true;
  $('#sensor-check-result').textContent = '測っています(約1秒)…';
  try { renderSensorCheck(await api('POST', '/api/sensor_check')); } catch (e) { toast(e.message, true); }
  b.disabled = false;
});
api('GET', '/api/sensor_check').then(renderSensorCheck).catch(() => {});


// ------------------------------------------------------------ 交代の理由(ロードマップ S3)
async function sendTakeoverReason(reason) {
  try {
    const r = await api('POST', '/api/drive/takeover_reason', { reason });
    toast(`交代の理由: ${r.label}${r.since_takeover_s != null ? `(交代の ${r.since_takeover_s} 秒後)` : '(直前の交代なし)'}${r.recorded ? '' : ' ※収集していないので記録には残りません'}`);
    $('#dr-tk-info').textContent = ` ・ 記録 ${(state.drive.tkCount = (state.drive.tkCount || 0) + 1)} 件`;
  } catch (e) { toast(e.message, true); }
}
document.querySelectorAll('[data-tk]').forEach((b) => b.addEventListener('click', () => sendTakeoverReason(b.dataset.tk)));
document.addEventListener('keydown', (e) => {
  if (state.tab !== 'drive' || !(state.drive.st && state.drive.st.running)) return;
  if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  const reason = { 1: 'collision', 2: 'stuck', 3: 'preemptive', 4: 'other' }[e.key];
  if (reason) { e.preventDefault(); sendTakeoverReason(reason); }
});
