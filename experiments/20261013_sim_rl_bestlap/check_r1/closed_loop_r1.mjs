// 書き出した方策を、RL の環境を通さずに閉ループで走らせて反証を試みる(2026-10-13、確認の段 r1)
//
//   node check_r1/closed_loop_r1.mjs [--quick]
//
// 物理は元の dist/sim-core.js、コースは scene.js から作った現行 v3、設定と開始は app.js の式(common_r1.mjs)。
//   A. REC60(記録の 60 条件・連続 20 秒)の周回時間を作り直して、RL の段の値と比べる
//   B. 決定性(同じ条件を 3 回 + 別のプロセスで 1 回、操作列の sha256)
//   C. 開始の格子(ゲート 0、横 ±15 cm・向き ±10°)と、学習の範囲の外(横 ±20〜30 cm・向き ±12〜20°)
//   D. 全 33 ゲートからの開始(連続 20 秒で 2 周とも有効か)
//   E. 制御遅延 1 フレーム(アプリの固定遅延: 最初は中立、以後 1 フレーム前の指令)
//   F. 物理のずれ(方策は知らない): 前進ゲイン・舵のゲイン・μ・ブレーキ・応答の時定数・舵の速さ・加速の上限
//   G. 離散化: sim-core の小区間を細かく(maxSpeed 10 にすると小区間が 1.45 倍細かくなる。指令は 4.5 以下なので他は同じ)、
//      自前の積分で小区間 1/8(閉ループ)
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {loadOwnPolicy, runClosedLoop, PLAN60, STD_PLAN, browserConfig, activeWorld, SCENE_WORLD, features, toCommand, hashActions,
  writeJson, REPORTS, runTeacher, core} from './common_r1.mjs';
import {createOwnSim} from './own_physics_r1.mjs';

const policy = loadOwnPolicy();
const quick = process.argv.includes('--quick');
const t0 = Date.now();
const log = (...a) => console.error(`[${((Date.now() - t0) / 1000).toFixed(0)} s]`, ...a);
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const pct = (a, p) => { const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * (b.length - 1) + .5))]; };
const out = {schema: 'rl-check-r1-closed-loop/v1', created_at: new Date().toISOString()};

// 子プロセスとして呼ばれたとき(決定性の確認): 標準の単周の操作列の sha256 だけを出す
if (process.argv.includes('--child-hash')) {
  const r = runClosedLoop(policy, {episodeMode: 'single'});
  console.log(hashActions(r.report.actions));
  process.exit(0);
}

const brief = s => ({valid: s.valid, reason: s.reason, laps: s.laps.map(l => +l.elapsed.toFixed(6)), gates: s.passed_gates, collisions: s.collisions});

// ---- A. REC60 ----
{
  const rows = [], cmdAll = [];
  for (const c of PLAN60) {
    const r = runClosedLoop(policy, {plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], episodeMode: 'continuous', keepReport: false});
    rows.push({name: c.name, scenario: c.scenario, ...brief(r.summary)});
    cmdAll.push(r.cmdLog);
  }
  const l1 = rows.map(r => r.laps[0]).filter(Number.isFinite), l2 = rows.map(r => r.laps[1]).filter(Number.isFinite);
  // 操作の荒さ(REC60 の全フレーム)
  const rate = [], edge = [], spd = [];
  let toggles = 0, frames = 0;
  for (const log_ of cmdAll) for (let i = 0; i < log_.length; i++) {
    frames++;
    if (i) rate.push(Math.abs(log_[i].steering - log_[i - 1].steering) * 30);
    edge.push(Math.abs(log_[i].steering) >= .494 ? 1 : 0); spd.push(log_[i].speed);
    if (i && Math.abs(log_[i].speed - log_[i - 1].speed) > 3) toggles++;
  }
  out.REC60 = {valid_two_laps: rows.filter(r => r.valid).length, n: rows.length, lap1_mean: mean(l1), lap1_min: Math.min(...l1), lap1_max: Math.max(...l1),
    lap2_mean: mean(l2), lap2_min: Math.min(...l2), lap2_max: Math.max(...l2), score: mean(rows.map(r => r.laps[0] + r.laps[1])),
    noise_vs_baseline_identical: (() => { const b = rows.filter(r => r.scenario === 'baseline'), n = rows.filter(r => r.scenario === 'noise'); return {baseline_lap1_mean: mean(b.map(r => r.laps[0])), noise_lap1_mean: mean(n.map(r => r.laps[0]))}; })(),
    command_stats: {frames, steer_cmd_rate_p95: pct(rate, .95), steer_cmd_rate_max: Math.max(...rate), steer_edge_frac: mean(edge),
      speed_cmd_at_cap_frac: mean(spd.map(v => v >= 4.5 * .99 ? 1 : 0)), speed_cmd_zero_frac: mean(spd.map(v => v <= 4.5 * .01 ? 1 : 0)),
      speed_cmd_big_jumps_per_s: toggles / (frames / 30)},
    rows};
  log('A REC60', out.REC60.valid_two_laps, out.REC60.lap1_mean, out.REC60.lap2_mean, JSON.stringify(out.REC60.command_stats));
}

// ---- B. 決定性 ----
{
  const hashes = [];
  for (let i = 0; i < 3; i++) hashes.push(hashActions(runClosedLoop(policy, {episodeMode: 'single'}).report.actions));
  const child = execFileSync(process.execPath, [fileURLToPath(import.meta.url), '--child-hash'], {encoding: 'utf8'}).trim();
  const rec = JSON.parse(fs.readFileSync(REPORTS + '/rl/final/records/rl_main_STD_single.json', 'utf8'));
  out.determinism = {in_process: hashes, child_process: child, record: hashActions(rec.actions), all_equal: new Set([...hashes, child, hashActions(rec.actions)]).size === 1};
  log('B determinism', out.determinism.all_equal);
}

// ---- C. 開始の格子 ----
{
  const grid = [], outside = [];
  const offs = quick ? [-.12, 0, .12] : [-.15, -.12, -.09, -.06, -.03, 0, .03, .06, .09, .12, .15];
  const yaws = quick ? [-8, 0, 8] : [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10];
  for (const o of offs) for (const y of yaws) {
    const r = runClosedLoop(policy, {offset: o, yawDeg: y, episodeMode: 'single', keepReport: false});
    grid.push({offset: o, yaw: y, ...(r.invalid_spawn ? {invalid_spawn: true} : brief(r.summary))});
  }
  for (const o of [-.3, -.25, -.2, .2, .25, .3]) for (const y of [-20, -12, 0, 12, 20]) {
    const r = runClosedLoop(policy, {offset: o, yawDeg: y, episodeMode: 'single', keepReport: false});
    outside.push({offset: o, yaw: y, ...(r.invalid_spawn ? {invalid_spawn: true} : brief(r.summary))});
  }
  const ok = grid.filter(g => g.valid), times = ok.map(g => g.laps[0]);
  out.start_grid = {n: grid.length, valid: ok.length, invalid_spawn: grid.filter(g => g.invalid_spawn).length, lap_min: Math.min(...times), lap_max: Math.max(...times),
    failures: grid.filter(g => !g.valid), rows: grid};
  out.start_outside_training = {n: outside.length, valid: outside.filter(g => g.valid).length, invalid_spawn: outside.filter(g => g.invalid_spawn).length,
    failures: outside.filter(g => !g.valid && !g.invalid_spawn), rows: outside};
  log('C grid', out.start_grid.valid, '/', out.start_grid.n, 'outside', out.start_outside_training.valid, '/', out.start_outside_training.n);
}

// ---- D. 全ゲートからの開始(連続 20 秒) ----
{
  const rows = [];
  for (let g = 0; g < 33; g++) for (const [o, y] of [[0, 0], [.08, 6], [-.08, -6]]) {
    const r = runClosedLoop(policy, {gate: g, offset: o, yawDeg: y, episodeMode: 'continuous', keepReport: false});
    rows.push({gate: g, offset: o, yaw: y, ...(r.invalid_spawn ? {invalid_spawn: true} : brief(r.summary))});
    if (quick && g > 3) break;
  }
  out.all_gates = {n: rows.length, valid: rows.filter(r => r.valid).length, invalid_spawn: rows.filter(r => r.invalid_spawn).length,
    failures: rows.filter(r => !r.valid && !r.invalid_spawn), rows};
  log('D all gates', out.all_gates.valid, '/', out.all_gates.n, 'invalid spawn', out.all_gates.invalid_spawn);
}

// ---- E/F/G. 遅延・物理のずれ・離散化 ----
{
  const cases = [
    ['nominal', {}],
    ['delay1', {delay: 1}],
    ['speed_gain_1.01(ONNX の前進ゲイン)', {speedGain: 1.01}], ['speed_gain_0.97', {speedGain: .97}], ['speed_gain_1.03', {speedGain: 1.03}],
    ['steer_gain_0.95', {steerGain: .95}], ['steer_gain_1.05', {steerGain: 1.05}],
    ['mu_x0.95', {simOverrides: {frictionMultiplier: .95}}], ['mu_x0.90', {simOverrides: {frictionMultiplier: .9}}], ['mu_x1.05', {simOverrides: {frictionMultiplier: 1.05}}],
    ['brake_1.0', {simOverrides: {brakeDeceleration: 1.0}}], ['brake_1.5', {simOverrides: {brakeDeceleration: 1.5}}],
    ['driveTau_0.30', {simOverrides: {driveTau: .3}}], ['driveTau_0.50', {simOverrides: {driveTau: .5}}],
    ['steerRate_1.0', {simOverrides: {steerRate: 1.0}}], ['steerRate_1.5', {simOverrides: {steerRate: 1.5}}],
    ['accelCap_4', {simOverrides: {driveAccelerationCap: 4}}],
    ['finer_substep_maxSpeed10', {simOverrides: {maxSpeed: 10}}],
    ['delay1+speed_gain_1.01', {delay: 1, speedGain: 1.01}],
  ];
  const rows = [];
  for (const [name, opt] of cases) {
    const s = runClosedLoop(policy, {episodeMode: 'single', keepReport: false, ...opt});
    const c = runClosedLoop(policy, {episodeMode: 'continuous', keepReport: false, ...opt});
    const row = {name, single: brief(s.summary), continuous: brief(c.summary)};
    // 同じずれで RC 教師(教師も知らない)
    try {
      row.teacher_single = brief(runTeacher({episodeMode: 'single', ...opt}).summary);
      row.teacher_continuous = brief(runTeacher({episodeMode: 'continuous', ...opt}).summary);
    } catch (e) { row.teacher_error = e.message; }
    rows.push(row);
    log('EF', name, JSON.stringify(rows.at(-1).single.laps), rows.at(-1).single.reason, JSON.stringify(rows.at(-1).continuous.laps), rows.at(-1).continuous.reason, '| teacher', JSON.stringify(rows.at(-1).teacher_single?.laps), rows.at(-1).teacher_single?.reason, JSON.stringify(rows.at(-1).teacher_continuous?.laps), rows.at(-1).teacher_continuous?.reason, rows.at(-1).teacher_error ?? '');
  }
  out.perturbations = rows;
  // 遅延 1 で REC60 も(参考)
  const d1 = [];
  for (const c of PLAN60) {
    const r = runClosedLoop(policy, {plan: c, gate: c['start-gate'], offset: c['spawn-offset'], yawDeg: c['spawn-yaw'], episodeMode: 'continuous', delay: 1, keepReport: false});
    d1.push(brief(r.summary));
    if (quick && d1.length >= 5) break;
  }
  out.REC60_delay1 = {n: d1.length, valid: d1.filter(r => r.valid).length, lap1_mean: mean(d1.filter(r => r.laps[0]).map(r => r.laps[0])),
    lap2_mean: mean(d1.filter(r => r.laps[1]).map(r => r.laps[1])), failures: d1.filter(r => !r.valid)};
  log('E REC60 delay1', out.REC60_delay1.valid, '/', out.REC60_delay1.n);
}

// ---- G2. 自前の積分(小区間 1/8)で閉ループ ----
function ownClosedLoop({refine = 8, episodeMode = 'single', gate = 0, offset = 0, yawDeg = 0}) {
  const cfg = core_cfg(episodeMode);
  const active = activeWorld(SCENE_WORLD, gate, offset, yawDeg);
  const own = createOwnSim(active, cfg, {refine, interp: 4});
  const tracker = features.createFeatureTracker(active, cfg);
  let st = own.state;
  tracker.resetProgress(st.x, st.z, st.route_index);
  const obs = new Float64Array(79);
  tracker.observe(st, st.route_index, 0, 0, obs);
  let frames = 0;
  while (!own.state.terminal && frames < 20 * 30 + 5) {
    const a = policy(obs);
    own.step(toCommand(a[0], a[1]));
    st = own.state;
    tracker.advanceProgress(st.x, st.z);
    tracker.observe(st, st.route_index, a[0], a[1], obs);
    frames++;
  }
  const r = own.result();
  return {refine, episodeMode, reason: r.reason, laps: r.lap_events.map(l => +l.elapsed.toFixed(6)), gates: r.passed_gates, collisions: r.collisions_own.length,
    min_clearance_m: Math.min(r.stats.minClear, r.stats.minClearInterp), sheet_required_ok: r.sheet_events.filter(e => e.dir === 'required' && e.sim_rule_ok).length,
    sheet_required_bad: r.sheet_events.filter(e => e.dir === 'required' && !e.sim_rule_ok).length};
}
function core_cfg(episodeMode) {
  // sim の正規化をかけた設定(gateHalfWidth・substep など既定の値も入る)
  const raw = browserConfig(STD_PLAN, {episodeMode});
  return core.createSimulation(activeWorld(SCENE_WORLD), raw).snapshot().config;
}
out.own_integrator_closed_loop = [ownClosedLoop({refine: 1}), ownClosedLoop({refine: 8}), ownClosedLoop({refine: 8, episodeMode: 'continuous'})];
log('G own', JSON.stringify(out.own_integrator_closed_loop));

writeJson(quick ? 'closed_loop_quick.json' : 'closed_loop.json', out);
log('done');
