// 設計の段(2026-10-14、sim_dart_20261014): 先生の候補のすき間の分布と、DART の乱しから先生が立て直せるかを Node で測る。
// modes の作業のランタイム(rl/runtime/mode_runtime.mjs)と元の sim-core.js を「読むだけ」で使う(ファイルは変えない)。
//   node measure_teachers.mjs clearance <out.json>
//   node measure_teachers.mjs dart <teacher> <noise_name> <n_scenes> <out.json>
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const MODES = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const {activeWorldFor, simConfig} = await import(pathToFileURL(MODES + '/snapshot/rl/env_core.mjs').href);
const {createModeTeacher} = await import(pathToFileURL(MODES + '/rl/runtime/mode_runtime.mjs').href);
const {commandToAction} = await import(pathToFileURL(MODES + '/rl/runtime/features.mjs').href);
const {perturbAllowed} = await import(pathToFileURL(MODES + '/rl/runtime/perturb.mjs').href);
const {realGateMisses} = await import(pathToFileURL(MODES + '/rl/gate_miss.mjs').href);
const core = await import(pathToFileURL(MODES + '/snapshot/sim/dist/sim-core.js').href);
const BASE_WORLD = JSON.parse(fs.readFileSync(MODES + '/snapshot/sim/world-v3.json', 'utf8'));
const readJ = p => JSON.parse(fs.readFileSync(p, 'utf8'));

export const TEACHERS = {
  fastest: {mode: 'fastest', weights: MODES + '/bundle/fastest/policy.json', params: {}},
  no_wall: {mode: 'no_wall', weights: MODES + '/bundle/no_wall/policy.json', params: {}},
  line_c0: {mode: 'line', weights: MODES + '/candidates/line_r2/policy.json', params: {line: {family: 'const', c0: 0}, c_v: 1.0}},
  line_l3: {mode: 'line', weights: MODES + '/candidates/line_r2/policy.json', params: {line: {family: 'const', c0: 0.3}, c_v: 1.0}},
  line_l2: {mode: 'line', weights: MODES + '/candidates/line_r2/policy.json', params: {line: {family: 'const', c0: 0.2}, c_v: 1.0}},
  line_r3: {mode: 'line', weights: MODES + '/candidates/line_r2/policy.json', params: {line: {family: 'const', c0: -0.3}, c_v: 1.0}},
};
const wcache = {};
const W = p => (wcache[p] ??= readJ(p));
// 評価と同じ開始(sim_rl_bestlap_20261013 eval: E0〜E5)
export const STARTS = {E0: [0, 0, 0], E1: [0, .03, 3], E2: [0, -.03, -3], E3: [0, .08, -5], E4: [0, -.08, 5], E5: [20, 0, 0]};

// 1 回走らせる。noise: (f, ctx) → {dSteer, speedFactor} か null。applied(実際に sim に渡した操作)を教師の「前回の行動」に返す
export function runEpisode({teacher, start, episodeMode = 'single', timeLimit = 15, noise = null, seed = 1}) {
  const T = TEACHERS[teacher];
  const {world, variation} = activeWorldFor(BASE_WORLD, start[0], start[1], start[2]);
  const cfg = simConfig({episodeMode: episodeMode === 'single' ? 'single' : 'continuous', timeLimit, seed});
  const sim = core.createSimulation(world, cfg);
  let snap = sim.reset({seed, scenario: 'baseline'});
  if (snap.state.terminal) return {invalid_spawn: true};
  const cfgN = JSON.parse(JSON.stringify(snap.config));
  const t = createModeTeacher(world, cfgN, variation, {mode: T.mode, weights: W(T.weights), params: T.params});
  const speedCap = 4.5 * (t.params.c_v ?? 1);
  let applied = [0, 0];
  const rows = [], gatesPassedAt = [];
  let lastGates = 0;
  const route = world.route, targets = route.slice(1).concat([route[0]]), gcount = [];
  const gateCtx = (st, idx) => { const i = idx % targets.length, tg = targets[i], bf = i ? targets[i - 1] : route[0];
    const L = Math.hypot(tg[0] - bf[0], tg[1] - bf[1]), nx = (tg[0] - bf[0]) / L, nz = (tg[1] - bf[1]) / L;
    return {along: (st.x - tg[0]) * nx + (st.z - tg[1]) * nz, lateral: -(st.x - tg[0]) * nz + (st.z - tg[1]) * nx}; };
  while (!snap.state.terminal) {
    const st = snap.state;
    gcount.push(`${snap.metrics.completed_laps ?? 0}:${snap.metrics.passed_gates ?? 0}`);
    const out = t.action({state: st, metrics: snap.metrics, config: cfgN}, {applied});
    const lg = t.log.at(-1);
    let cmd = {steering: out.steering, speed: out.speed}, nz = null;
    if (noise) {
      nz = noise(st.frame, {s: lg.s, v: st.speed, dmin: lg.dmin, gate: gateCtx(st, snap.metrics.route_index ?? 0)});
      if (nz) cmd = {steering: Math.max(-.52, Math.min(.52, cmd.steering + (nz.dSteer ?? 0))), speed: Math.max(0, Math.min(speedCap, cmd.speed * (nz.speedFactor ?? 1)))};
    }
    const a = commandToAction(cmd.steering, cmd.speed, speedCap, .52);
    applied = [a.a0, a.a1];
    rows.push({f: st.frame, x: st.x, z: st.z, yaw: st.yaw, v: st.speed, s: lg.s, ey: lg.ey, dmin: lg.dmin, sh: lg.sh, noisy: !!nz, tSteer: out.steering, tSpeed: out.speed, steer: cmd.steering, speed: cmd.speed});
    snap = sim.step(cmd);
    const g = snap.metrics.passed_gates;
    if (g !== lastGates) { gatesPassedAt.push({gate: g, f: st.frame}); lastGates = g; }
  }
  const st = snap.state, m = snap.metrics, rep = sim.report();
  return {teacher, start, reason: st.reason, frames: st.frame, elapsed: m.elapsed, collisions: m.collisions, passed_gates: m.passed_gates,
    collision: m.collision_events?.[0] ? {t: m.collision_events[0].time, collider: m.collision_events[0].collider, kind: m.collision_events[0].kind} : null,
    laps: (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions})),
    crossings: m.required_crossing_events?.length ?? null, rows, world, trace: rep.trace, shield_frames: t.shieldFrames,
    gate_misses: realGateMisses(rows.map(q => q.s), gcount).misses};
}

// ゲートの横の余裕(sim-core の advanceGates と同じ定義。フレームの終わりの姿勢で近似)
function gateMargins(world, trace) {
  const route = world.route, out = [];
  let i = 0, prev = {x: world.spawn.x, z: world.spawn.z};
  const targets = route.slice(1).concat([route[0]]);
  for (const q of trace) {
    if (i >= targets.length) break;
    const target = targets[i], before = i ? targets[i - 1] : route[0];
    const L = Math.hypot(target[0] - before[0], target[1] - before[1]), nx = (target[0] - before[0]) / L, nz = (target[1] - before[1]) / L;
    const sg = p => (p.x - target[0]) * nx + (p.z - target[1]) * nz;
    const a = sg(prev), b = sg(q);
    if (a < 0 && b >= 0) {
      const fr = -a / (b - a), x = prev.x + (q.x - prev.x) * fr, z = prev.z + (q.z - prev.z) * fr;
      const lat = Math.abs(-(x - target[0]) * nz + (z - target[1]) * nx);
      if (lat <= .65) { out.push({gate_idx: i, margin_m: .65 - lat}); i++; }
    }
    prev = q;
  }
  return out;
}
const minOf = a => a.length ? Math.min(...a) : null;
function summarize(r) {
  const rows = r.rows;
  // 最初の直線(ゲート 0 の開始で、車の x が初めて 2.6 m を超えるまで)。S-R がぶつかった区間
  let fs = [];
  if (r.start[0] === 0) { for (const q of rows) { if (q.x > 2.6) break; fs.push(q.dmin); } }
  const gm = gateMargins(r.world, r.trace);
  const gmin = gm.length ? gm.reduce((a, b) => (b.margin_m < a.margin_m ? b : a)) : null;
  // s の 0.5 m ごとの最小すき間(周の中の位置 s mod L)
  const L = 26.119648724030924, bins = {};
  for (const q of rows) { const k = Math.floor((((q.s % L) + L) % L) / .5); bins[k] = Math.min(bins[k] ?? 9, q.dmin); }
  return {teacher: r.teacher, start: r.start, reason: r.reason, frames: r.frames, lap_s: r.reason === 'route_complete' && r.collisions === 0 ? +r.elapsed.toFixed(3) : null,
    collisions: r.collisions, collision: r.collision, passed_gates: r.passed_gates, min_clearance_m: minOf(rows.map(q => q.dmin)),
    first_straight_min_clearance_m: fs.length ? Math.min(...fs) : null, min_gate_margin: gmin, shield_frames: r.shield_frames,
    clearance_by_s_bin05: bins, frac_frames_dmin_lt_10cm: rows.filter(q => q.dmin < .10).length / rows.length,
    frac_frames_dmin_lt_15cm: rows.filter(q => q.dmin < .15).length / rows.length};
}

// ---- DART の乱し(候補)----
function rngFrom(seed) { let s = seed >>> 0; return () => { s = (s + 0x9e3779b9) >>> 0; let z = s; z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0; z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0; return ((z ^ (z >>> 15)) >>> 0) / 4294967296; }; }
export const NOISES = {
  // 舵のかたより(バースト): 大きさ U(a)・続く長さ U(n) フレーム・左右等確率。終わってから U(gap) 秒あけて次。速さの乱しは 25% の回で × U(sf)
  burst_s: {amp: [.04, .10], len: [6, 15], gap: [.6, 1.6], first: [.2, .8], speedP: .25, sf: [.7, .95]},
  burst_m: {amp: [.06, .15], len: [6, 15], gap: [.6, 1.6], first: [.2, .8], speedP: .25, sf: [.7, .95]},
  burst_l: {amp: [.10, .22], len: [8, 18], gap: [.6, 1.6], first: [.2, .8], speedP: .25, sf: [.6, .95]},
  // 見張りつき(guard): 今のすき間 d_min ≥ g0 のときだけ始め、d_min < g1 になったら打ち切る(先生が必ず立て直せるように)
  burst_m_g: {amp: [.06, .15], len: [6, 15], gap: [.6, 1.6], first: [.2, .8], speedP: .25, sf: [.7, .95], guard: [.12, .06]},
  burst_l_g: {amp: [.10, .22], len: [8, 18], gap: [.6, 1.6], first: [.2, .8], speedP: .25, sf: [.6, .95], guard: [.12, .06]},
  burst_xl_g: {amp: [.10, .25], len: [8, 20], gap: [.5, 1.4], first: [.15, .6], speedP: .25, sf: [.65, .95], guard: [.10, .05]},
  // ゲートの見張りつき(gateGuard: [手前の距離 m, 打ち切る |lateral| m])
  burst_m_gg: {amp: [.06, .15], len: [6, 15], gap: [.6, 1.6], first: [.2, .8], speedP: .25, sf: [.7, .95], guard: [.12, .06], gateGuard: [1.2, .45, .30]},
  burst_xl_gg: {amp: [.10, .25], len: [8, 20], gap: [.5, 1.4], first: [.15, .6], speedP: .25, sf: [.65, .95], guard: [.10, .05], gateGuard: [1.2, .45, .30]},
  burst_l_gg: {amp: [.08, .20], len: [6, 18], gap: [.5, 1.4], first: [.15, .6], speedP: .25, sf: [.65, .95], guard: [.10, .05], gateGuard: [1.2, .45, .30]},
  burst_ml_g: {amp: [.06, .20], len: [6, 18], gap: [.5, 1.4], first: [.15, .6], speedP: .25, sf: [.65, .95], guard: [.10, .05]},
  // 連続(OU)の舵の雑音: σ・相関時間 τ 秒
  ou_s: {ou: {sigma: .05, tau: .3}},
  ou_m: {ou: {sigma: .08, tau: .4}},
};
export function makeNoise(name, seed, {zoneCheck = true, scale = 1} = {}) {
  const P = NOISES[name], rnd = rngFrom(seed * 2654435761 + 12345);
  if (P.ou) {
    let x = 0; const a = Math.exp(-1 / 30 / P.ou.tau), b = P.ou.sigma * Math.sqrt(1 - a * a);
    const g = () => { const u1 = Math.max(1e-12, rnd()), u2 = rnd(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); };
    return f => { x = a * x + b * g(); return f < 6 ? null : {dSteer: scale * x}; };
  }
  const U = ([lo, hi]) => lo + (hi - lo) * rnd();
  let next = Math.round(U(P.first) * 30), active = null; const events = [];
  const fn = (f, ctx) => {
    if (active && f >= active.end) { next = f + Math.round(U(P.gap) * 30); active = null; }
    if (!active && f >= next) {
      if (zoneCheck && !perturbAllowed(ctx.s)) return null;
      if (P.guard && !(ctx.dmin >= P.guard[0])) return null;
      // ゲートの見張り: 次のゲートの手前 gateGuard[0] m の中で、ゲートの中心から横に gateGuard[2] m より外なら始めない
      if (P.gateGuard && ctx.gate && ctx.gate.along > -P.gateGuard[0] && ctx.gate.along <= 0 && Math.abs(ctx.gate.lateral) > P.gateGuard[2]) return null;   // 狭い所(使える幅 < 0.60 m・黒シート・裏の直線)では始めない(modes の perturb.mjs と同じ物差し)
      const amp = scale * U(P.amp) * (rnd() < .5 ? -1 : 1), len = Math.round(U(P.len)), sp = rnd() < P.speedP ? U(P.sf) : 1;
      active = {start: f, end: f + len, dSteer: amp, speedFactor: sp}; events.push(active);
    }
    if (active && P.gateGuard && ctx.gate && ctx.gate.along > -P.gateGuard[0] && ctx.gate.along <= 0 && Math.abs(ctx.gate.lateral) > P.gateGuard[1]) { active.aborted = f; active.end = f; next = f + Math.round(U(P.gap) * 30); active = null; return null; }
    if (active && P.guard && ctx.dmin < P.guard[1]) { active.aborted = f; active.end = f; next = f + Math.round(U(P.gap) * 30); active = null; return null; }
    return active ? {dSteer: active.dSteer, speedFactor: active.speedFactor} : null;
  };
  fn.events = events;
  return fn;
}
// 同じ開始・乱しなしの走りとの横のずれ(s で合わせ、参照ループ R からの横ずれ e_y の差)
function deviationFromClean(rows, clean) {
  const cs = clean.map(q => q.s), ce = clean.map(q => q.ey);
  let j = 0; const out = [];
  for (const q of rows) {
    while (j < cs.length - 2 && cs[j + 1] < q.s) j++;
    if (q.s < cs[0] || q.s > cs.at(-1)) { out.push(null); continue; }
    const fr = (q.s - cs[j]) / Math.max(1e-9, cs[j + 1] - cs[j]);
    out.push(Math.abs(q.ey - (ce[j] + (ce[j + 1] - ce[j]) * Math.min(1, Math.max(0, fr)))));
  }
  return out;
}
const pct = (a, p) => { const b = a.filter(x => x != null).sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.floor(p * b.length))] : null; };

const [cmd, ...args] = import.meta.url.endsWith(process.argv[1]?.split("/").pop() ?? "") ? process.argv.slice(2) : [];
if (cmd === 'clearance') {
  const res = [];
  for (const teacher of Object.keys(TEACHERS)) for (const [k, st] of Object.entries(STARTS)) {
    const r = runEpisode({teacher, start: st, episodeMode: 'single', timeLimit: 15});
    const s = summarize(r); s.start_id = k; res.push(s);
    console.log(teacher, k, s.reason, s.lap_s, 'min', s.min_clearance_m?.toFixed(3), 'first', s.first_straight_min_clearance_m?.toFixed(3), 'gate', s.min_gate_margin?.margin_m.toFixed(3), s.min_gate_margin?.gate_idx, 'sh', s.shield_frames);
  }
  fs.writeFileSync(args[0], JSON.stringify({made: new Date().toISOString(), core: 'original sim-core.js (modes snapshot)', results: res}, null, 1));
} else if (cmd === 'dart') {
  const [teacher, noiseName, nStr, outPath, scaleStr] = args;
  const n = +nStr, scale = +(scaleStr ?? 1), res = [];
  for (let i = 0; i < n; i++) {
    // 記録と同じ開始の範囲(ゲート 0・±3 cm・±3°)。i から決まる
    const r0 = rngFrom(1000 + i), st = [0, (r0() * 2 - 1) * .03, (r0() * 2 - 1) * 3];
    const clean = runEpisode({teacher, start: st, episodeMode: 'continuous', timeLimit: 20, seed: 1 + i});
    const nz = makeNoise(noiseName, 7000 + i, {scale});
    const r = runEpisode({teacher, start: st, episodeMode: 'continuous', timeLimit: 20, seed: 1 + i, noise: nz});
    const dev = deviationFromClean(r.rows, clean.rows);
    // 1 フレームごとの「今の線からのずれ」と、乱しの回ごとの最大のずれ
    const evMax = (nz.events ?? []).map(e => { let m = 0; for (let f = e.start; f < Math.min(r.rows.length, e.end + 45); f++) if (dev[f] != null) m = Math.max(m, dev[f]); return m; });
    const firstStraight = r.rows.filter(q => q.x <= 2.6 && q.f < 60).map(q => dev[q.f]).filter(x => x != null);
    const lapsValid = r.laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
    res.push({i, start: st, reason: r.reason, frames: r.frames, collisions: r.collisions, collision: r.collision, passed_gates: r.passed_gates, laps_valid: lapsValid,
      laps: r.laps.map(l => +l.elapsed.toFixed(3)), clean_laps: clean.laps.map(l => +l.elapsed.toFixed(3)), clean_reason: clean.reason,
      noisy_frac: r.rows.filter(q => q.noisy).length / r.rows.length, n_events: (nz.events ?? []).length,
      dev_p50: pct(dev, .5), dev_p90: pct(dev, .9), dev_max: pct(dev, 1), ev_max_p50: pct(evMax, .5), ev_max_p90: pct(evMax, .9),
      ev_frac_5_25cm: evMax.length ? evMax.filter(x => x >= .05 && x <= .25).length / evMax.length : null,
      first_straight_dev_max: firstStraight.length ? Math.max(...firstStraight) : null,
      min_clearance_m: minOf(r.rows.map(q => q.dmin)), shield_frames: r.shield_frames,
      clean_min_clearance_m: minOf(clean.rows.map(q => q.dmin)), clean_gate_margin_min: minOf(gateMargins(clean.world, clean.trace).map(g => g.margin_m)),
      clean_first_straight_min: (() => { let m = 9; for (const q of clean.rows) { if (q.x > 2.6) break; m = Math.min(m, q.dmin); } return m; })(),
      aborted_events: (nz.events ?? []).filter(e => e.aborted != null).length, gate_misses: r.gate_misses, clean_gate_misses: clean.gate_misses});
    const q = res.at(-1);
    console.log(teacher, noiseName, scale, i, q.reason, 'gm', q.gate_misses, 'col', q.collisions, 'validlaps', q.laps_valid, 'laps', q.laps.join('/'), 'ev', q.n_events, 'evmax p50/p90', q.ev_max_p50?.toFixed(3), q.ev_max_p90?.toFixed(3), 'fs', q.first_straight_dev_max?.toFixed(3), 'dmin', q.min_clearance_m?.toFixed(3));
  }
  const ok = res.filter(q => q.collisions === 0 && q.reason === 'collection_duration_complete' && q.gate_misses === 0);
  const summary = {teacher, noise: noiseName, scale, n, ok: ok.length, valid_laps_total: res.reduce((a, q) => a + q.laps_valid, 0), laps_total: res.reduce((a, q) => a + q.laps.length, 0),
    noisy_frac_mean: res.reduce((a, q) => a + q.noisy_frac, 0) / n, ev_max_p50_med: pct(res.map(q => q.ev_max_p50), .5), ev_max_p90_med: pct(res.map(q => q.ev_max_p90), .5),
    ev_frac_5_25cm_mean: res.reduce((a, q) => a + (q.ev_frac_5_25cm ?? 0), 0) / n, first_straight_dev_max_med: pct(res.map(q => q.first_straight_dev_max), .5),
    min_clearance_min: minOf(res.map(q => q.min_clearance_m)), clean_min_clearance_min: minOf(res.map(q => q.clean_min_clearance_m)),
    clean_gate_margin_min: minOf(res.map(q => q.clean_gate_margin_min)), clean_first_straight_min: minOf(res.map(q => q.clean_first_straight_min)),
    clean_lap1_mean: res.filter(q => q.clean_laps[0]).reduce((a, q) => a + q.clean_laps[0], 0) / Math.max(1, res.filter(q => q.clean_laps[0]).length),
    shield_frames_total: res.reduce((a, q) => a + (q.shield_frames ?? 0), 0), events_total: res.reduce((a, q) => a + q.n_events, 0), aborted_total: res.reduce((a, q) => a + q.aborted_events, 0), lap1_mean: res.filter(q => q.laps[0]).reduce((a, q) => a + q.laps[0], 0) / Math.max(1, res.filter(q => q.laps[0]).length)};
  console.log('SUMMARY', JSON.stringify(summary));
  fs.writeFileSync(outPath, JSON.stringify({made: new Date().toISOString(), summary, NOISE: NOISES[noiseName], runs: res}, null, 1));
}
