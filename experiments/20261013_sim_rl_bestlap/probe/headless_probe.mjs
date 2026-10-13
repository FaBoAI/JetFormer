#!/usr/bin/env node
// CourseLab シミュレーターをWebGLなしでNodeから直接ステップ実行できることの確認用プローブ。
// - 物理・コースは ../sim/ にコピーした利用者のコード(dist/sim-core.js など)をそのまま使い、変更しない。
// - 教師(正解位置を使う)で1周目・2周目のラップを再現し、README の数値と比較する。
// - 学習済みONNX(8.964秒)の保存済み操作列を再生して、物理の再現性も確認する。
// - 1プロセスでのステップ速度とメモリを測る(RL環境の設計用)。
// 使い方: node probe/headless_probe.mjs [学習済み走行JSON] [出力JSON]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const simRoot = path.resolve(here, '../sim');
const {loadCourseWorld} = await import(path.join(simRoot, 'qa/load-course-world.mjs'));
const {createSimulation, DEFAULTS} = await import(path.join(simRoot, 'dist/sim-core.js'));
const {WORLD_VERSION} = await import(path.join(simRoot, 'dist/world.js'));
const {createRcTrainingTeacher, RC_COLLECTION_PLAN, RC_TEACHER_METADATA} = await import(path.join(simRoot, 'dist/rc-speed-controller.js'));
const {createOptimizedTeacher} = await import(path.join(simRoot, 'dist/optimized-reference.js'));
const {createControlDelayQueue} = await import(path.join(simRoot, 'dist/control-delay.js'));

const learnedLapPath = process.argv[2] ?? '/home/jetson/sim/extracted/course-simulator/results/shortcut-rc-learned-lap.json';
const outputPath = process.argv[3] ?? '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/probe_result.json';
const out = {schema: 'sim-rl-bestlap-headless-probe/v1', node: process.version, world_version: WORLD_VERSION};
const log = (...a) => console.log(...a);
const r6 = x => Math.round(x * 1e6) / 1e6;

// ---- コース(ブラウザと同じ scene.js から three.js の幾何だけを作る) ----
let t0 = performance.now();
const world = await loadCourseWorld();
out.world_load_ms = r6(performance.now() - t0);
const worldJson = JSON.stringify(world);
out.world_sha256 = crypto.createHash('sha256').update(worldJson).digest('hex');
fs.writeFileSync(path.join(simRoot, 'world-v3.json'), worldJson + '\n');
out.world_summary = {
  segments: world.segments.length, disks: world.disks.length, surfaces: world.surfaces.map(s => [s.id, s.mu]),
  bounds: world.bounds, gates: world.route.length, spawn: world.spawn, required_crossings: world.requiredCrossings,
};
log(`コース ${WORLD_VERSION}: ゲート${world.route.length}・線分${world.segments.length}・柱/脚${world.disks.length} (読込 ${out.world_load_ms.toFixed(0)} ms)`);

// ---- ゲート幾何: targets=[route[1..], route[0]]、通過線は進行方向に垂直、半幅0.65 m ----
const targets = [...world.route.slice(1), world.route[0]];
out.gates = targets.map((p, i) => {
  const before = i ? targets[i - 1] : world.route[0];
  const len = Math.hypot(p[0] - before[0], p[1] - before[1]);
  return {index: i, x: p[0], z: p[1], normal: [r6((p[0] - before[0]) / len), r6((p[1] - before[1]) / len)], segment_m: r6(len)};
});
out.route_length_m = r6(out.gates.reduce((s, g) => s + g.segment_m, 0));

// ---- RC教師(README の 8.710〜8.758 / 8.359〜8.407 秒の元)と同じ設定 ----
const RC_PHYSICS = {dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1,
  longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2};
// app.js の reset() と同じ開始姿勢(開始ゲート0、左右ずれ・角度)
function spawnWorld(offset = 0, yawDeg = 0) {
  const first = world.route[0], next = world.route[1], heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const turn = yawDeg * Math.PI / 180;
  return {world: {...world, spawn: {x: first[0] + offset * Math.sin(heading), z: first[1] - offset * Math.cos(heading), yaw: heading + turn}},
    variation: {gate: 0, left_offset_m: offset, left_yaw_deg: turn * 180 / Math.PI}};
}
function runRcContinuous(plan) {
  const {world: w, variation} = spawnWorld(plan['spawn-offset'], plan['spawn-yaw']);
  const config = {...RC_PHYSICS, seed: plan.seed, timeLimit: plan.limit, noiseStd: plan.scenario === 'noise' ? .025 : 0, episodeMode: 'continuous-laps-v1'};
  const sim = createSimulation(w, config);
  sim.reset({seed: config.seed, scenario: plan.scenario});
  const teacher = createRcTrainingTeacher(w, sim.snapshot().config, variation);
  let snap = sim.snapshot();
  while (!snap.state.terminal) snap = sim.step(teacher.action(snap));
  return {name: plan.name, split: plan['dataset-split'], frames: snap.state.frame, reason: snap.state.reason, collisions: snap.metrics.collisions,
    laps: snap.metrics.lap_events.map(l => ({lap: l.lap_index, elapsed_s: l.elapsed, gates: l.passed_gates, shortcut_t: l.required_crossing_events[0]?.time ?? null})),
    final_speed: snap.state.speed, current_lap_gates: snap.metrics.current_lap_passed_gates};
}
t0 = performance.now();
const rcRuns = RC_COLLECTION_PLAN.map(runRcContinuous);
const rcMs = performance.now() - t0;
const lap1 = rcRuns.map(r => r.laps[0]?.elapsed_s), lap2 = rcRuns.map(r => r.laps[1]?.elapsed_s);
out.rc_teacher_continuous = {
  scenes: rcRuns.length, total_frames: rcRuns.reduce((s, r) => s + r.frames, 0), wall_ms: r6(rcMs),
  all_two_laps: rcRuns.every(r => r.laps.length >= 2 && r.laps.every(l => l.gates === 33)),
  collisions_total: rcRuns.reduce((s, r) => s + r.collisions, 0), reasons: [...new Set(rcRuns.map(r => r.reason))],
  lap1_min_s: Math.min(...lap1), lap1_max_s: Math.max(...lap1), lap2_min_s: Math.min(...lap2), lap2_max_s: Math.max(...lap2),
  readme_lap1: [8.710, 8.758], readme_lap2: [8.359, 8.407], nominal_scene: rcRuns[0], scenes_detail: rcRuns,
};
log(`RC教師(連続20秒×60シーン): 1周目 ${Math.min(...lap1).toFixed(3)}〜${Math.max(...lap1).toFixed(3)} s / 2周目 ${Math.min(...lap2).toFixed(3)}〜${Math.max(...lap2).toFixed(3)} s` +
  ` / 衝突合計 ${out.rc_teacher_continuous.collisions_total} / 全シーン2周以上 ${out.rc_teacher_continuous.all_two_laps} (README: 8.710〜8.758 / 8.359〜8.407)`);
log(`  標準開始 rc20-mode8-train-00: ${rcRuns[0].laps.map(l => `${l.lap}周目 ${l.elapsed_s.toFixed(4)} s(黒シート ${l.shortcut_t.toFixed(3)} s)`).join(', ')} 600フレーム後 ${rcRuns[0].current_lap_gates}/33ゲート`);

// ---- RC教師 単周(停止から、ゴールで終了)。メタデータの nominal_single_lap_s と比較 ----
{
  const {world: w, variation} = spawnWorld(0, 0);
  const sim = createSimulation(w, {...RC_PHYSICS, seed: 1, timeLimit: 20, noiseStd: 0});
  const teacher = createRcTrainingTeacher(w, sim.snapshot().config, variation);
  let snap = sim.snapshot();
  while (!snap.state.terminal) snap = sim.step(teacher.action(snap));
  out.rc_teacher_single_lap = {elapsed_s: snap.metrics.elapsed, reason: snap.state.reason, success: snap.state.success, gates: snap.metrics.passed_gates,
    collisions: snap.metrics.collisions, frames: snap.state.frame, shortcut: snap.metrics.required_crossing_events[0], metadata_nominal_single_lap_s: RC_TEACHER_METADATA.nominal_single_lap_s,
    max_lateral_acceleration: snap.metrics.max_lateral_acceleration, traction_limited_steps: snap.metrics.traction_limited_steps, distance_m: snap.metrics.distance};
  log(`RC教師(単周・停止から): ${snap.metrics.elapsed.toFixed(4)} s ${snap.state.reason} ゲート${snap.metrics.passed_gates}/33 衝突${snap.metrics.collisions} (メタデータ ${RC_TEACHER_METADATA.nominal_single_lap_s.toFixed(4)} s)`);
  // 1フレーム遅延キューを通した場合(ONNX評価と同じ適用タイミング)
  const sim2 = createSimulation(w, {...RC_PHYSICS, seed: 1, timeLimit: 20, noiseStd: 0});
  const teacher2 = createRcTrainingTeacher(w, sim2.snapshot().config, variation), queue = createControlDelayQueue();
  queue.configure(1);
  let s2 = sim2.snapshot();
  while (!s2.state.terminal) s2 = sim2.step(queue.apply(teacher2.action(s2), {frame: s2.state.frame, sim_time: s2.state.sim_time, dt: s2.config.dt}).action);
  out.rc_teacher_single_lap_delay1 = {elapsed_s: s2.metrics.elapsed, reason: s2.state.reason, gates: s2.metrics.passed_gates, collisions: s2.metrics.collisions,
    collision_events: s2.metrics.collision_events};
  log(`RC教師(単周・制御遅延1フレーム): ${s2.metrics.elapsed.toFixed(4)} s ${s2.state.reason} ゲート${s2.metrics.passed_gates}/33 衝突${s2.metrics.collisions}`);
}

// ---- 最適化参照教師 optimized-reference.js(従来の一定加速度モデル用・単周専用) ----
{
  const config = {dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19, wheelbase: .26,
    stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, seed: 1, timeLimit: 30};
  const sim = createSimulation(world, config);
  const teacher = createOptimizedTeacher(world, sim.snapshot().config, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
  let snap = sim.snapshot();
  while (!snap.state.terminal) snap = sim.step(teacher.action(snap));
  out.optimized_reference_single_lap = {physics: 'constant-acceleration-v1 (1.2 m/s2)', elapsed_s: snap.metrics.elapsed, reason: snap.state.reason,
    gates: snap.metrics.passed_gates, collisions: snap.metrics.collisions, reference_points: teacher.reference.points.length,
    reference_length_m: teacher.reference.length, planned_time_s: teacher.reference.planned_time_s,
    reference_max_speed: Math.max(...teacher.reference.points.map(p => p.speed))};
  log(`最適化参照教師(一定加速度1.2 m/s²・単周): ${snap.metrics.elapsed.toFixed(4)} s ${snap.state.reason} ゲート${snap.metrics.passed_gates}/33 衝突${snap.metrics.collisions} 参照長 ${teacher.reference.length.toFixed(3)} m`);
  // 同じ参照をRC応答物理で走らせた場合(参考: 物理モデルが違うと参照は使えない)
  const simRc = createSimulation(world, {...RC_PHYSICS, seed: 1, timeLimit: 30, noiseStd: 0});
  let s = simRc.snapshot();
  const teacherRc = createOptimizedTeacher(world, config, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
  while (!s.state.terminal) s = simRc.step(teacherRc.action(s));
  out.optimized_reference_on_rc_physics = {elapsed_s: s.metrics.elapsed, reason: s.state.reason, gates: s.metrics.passed_gates, collisions: s.metrics.collisions,
    collision_events: s.metrics.collision_events};
  log(`  同じ参照をRC応答物理で走行: ${s.metrics.elapsed.toFixed(4)} s ${s.state.reason} ゲート${s.metrics.passed_gates}/33`);
}

// ---- 学習済みONNX(8.964117秒)の保存操作列をそのまま再生 ----
if (fs.existsSync(learnedLapPath)) {
  const report = JSON.parse(fs.readFileSync(learnedLapPath, 'utf8'));
  const sim = createSimulation(report.world, report.config);
  sim.reset({seed: report.seed, scenario: report.scenario});
  let snap = sim.snapshot();
  for (const a of report.actions) snap = sim.step(a);
  const sameWorld = JSON.stringify(report.world) === JSON.stringify(world);
  let maxTraceDiff = 0;
  const trace = sim.report().trace;
  for (let i = 0; i < trace.length; i++) for (const k of ['x', 'z', 'yaw', 'speed', 'steering', 'time']) maxTraceDiff = Math.max(maxTraceDiff, Math.abs(trace[i][k] - report.trace[i][k]));
  out.learned_onnx_replay = {source: learnedLapPath, frames: report.actions.length, elapsed_s: snap.metrics.elapsed, saved_elapsed_s: report.metrics.elapsed,
    reason: snap.state.reason, gates: snap.metrics.passed_gates, collisions: snap.metrics.collisions, shortcut: snap.metrics.required_crossing_events[0],
    max_trace_abs_diff: maxTraceDiff, report_world_equals_current_scene: sameWorld, config: report.config,
    delay: report.context?.execution, start_variation: report.context?.start_variation};
  log(`学習済みONNX 8.964秒走行の操作列を再生: ${snap.metrics.elapsed.toFixed(6)} s (保存値 ${report.metrics.elapsed.toFixed(6)}) ${snap.state.reason} ゲート${snap.metrics.passed_gates}/33 最大軌跡差 ${maxTraceDiff.toExponential(2)} 現行コース一致 ${sameWorld}`);
}

// ---- 終了条件の確認: 全開直進で衝突 ----
{
  const sim = createSimulation(world, {...RC_PHYSICS, seed: 1, timeLimit: 20, noiseStd: 0});
  let snap = sim.snapshot();
  while (!snap.state.terminal) snap = sim.step({steering: 0, speed: 7.24});
  out.full_throttle_straight = {reason: snap.state.reason, time_s: snap.state.sim_time, frame: snap.state.frame, speed_at_end: snap.state.speed,
    collision: snap.metrics.collision_events[0], gates: snap.metrics.passed_gates, x: snap.state.x, z: snap.state.z};
  log(`全開直進: ${snap.state.reason} ${snap.state.sim_time.toFixed(3)} s ゲート${snap.metrics.passed_gates} 衝突相手 ${snap.metrics.collision_events[0]?.collider}(${snap.metrics.collision_events[0]?.kind})`);
}
// ゲート外側を通るとゲートは進まない(半幅0.65 m)。停止したままだと time_limit。
{
  const sim = createSimulation(world, {...RC_PHYSICS, seed: 1, timeLimit: 2, noiseStd: 0});
  let snap = sim.snapshot();
  while (!snap.state.terminal) snap = sim.step({steering: 0, speed: 0});
  out.idle = {reason: snap.state.reason, time_s: snap.state.sim_time, frame: snap.state.frame};
}

// ---- 途中状態からの開始(initialState)。単周モードのみ・成功扱いにはならない ----
{
  const {world: w, variation} = spawnWorld(0, 0);
  const ref = createSimulation(w, {...RC_PHYSICS, seed: 1, timeLimit: 20, noiseStd: 0});
  const teacher = createRcTrainingTeacher(w, ref.snapshot().config, variation);
  let snap = ref.snapshot();
  for (let i = 0; i < 120; i++) snap = ref.step(teacher.action(snap));
  const st = snap.state, initialState = {x: st.x, z: st.z, yaw: st.yaw, speed: st.speed, steering: st.steering, routeIndex: snap.metrics.route_index};
  const sim = createSimulation(w, {...RC_PHYSICS, seed: 1, timeLimit: 20, noiseStd: 0});
  sim.reset({initialState});
  let s = sim.snapshot();
  // RC教師は initialState 非対応の契約なので、120フレーム走った同じ教師(参照位置を保持)で続きを追従する
  const steps0 = s.state.frame;
  while (!s.state.terminal) s = sim.step(teacher.action(s));
  out.initial_state_resume = {initialState, after_frames: 120, reason: s.state.reason, success: s.state.success, passed_gates_in_episode: s.metrics.passed_gates,
    route_index: s.metrics.route_index, elapsed_s: s.metrics.elapsed, start_frame: steps0, total_from_stop_s: 120 / 30 + s.metrics.elapsed, required_crossing_events: s.metrics.required_crossing_events.length};
  log(`initialState(120フレーム後の姿勢・速度・routeIndex=${initialState.routeIndex})から再開: ${s.state.reason} 残り ${s.metrics.elapsed.toFixed(4)} s (合計 ${(4 + s.metrics.elapsed).toFixed(4)} s) success=${s.state.success}`);
  let refused = null;
  try { createSimulation(w, {...RC_PHYSICS, episodeMode: 'continuous-laps-v1'}).reset({initialState}); } catch (e) { refused = e.message; }
  out.initial_state_resume.continuous_refused = refused;
}

// ---- 決定性: 同じ入力で同じ結果 ----
{
  const a = runRcContinuous(RC_COLLECTION_PLAN[0]), b = runRcContinuous(RC_COLLECTION_PLAN[0]);
  out.deterministic = JSON.stringify(a) === JSON.stringify(b);
}

// ---- 速度とメモリ(1プロセス) ----
function bench(label, frames, fn) {
  if (global.gc) global.gc();
  const m0 = process.memoryUsage();
  const start = performance.now();
  const n = fn(frames);
  const ms = performance.now() - start;
  const m1 = process.memoryUsage();
  const r = {label, steps: n, wall_ms: r6(ms), steps_per_s: Math.round(n / (ms / 1000)), rss_mb: r6(m1.rss / 2 ** 20), heap_used_mb_delta: r6((m1.heapUsed - m0.heapUsed) / 2 ** 20)};
  log(`  ${label}: ${r.steps_per_s} steps/s (${n}ステップ ${ms.toFixed(0)} ms) RSS ${r.rss_mb.toFixed(0)} MB`);
  return r;
}
log('ステップ速度(1プロセス・1コア):');
const {world: wNom, variation: vNom} = spawnWorld(0, 0);
const cfgCont = {...RC_PHYSICS, seed: 1, timeLimit: 20, noiseStd: 0, episodeMode: 'continuous-laps-v1'};
out.bench = [];
// 1) 教師行動+step(戻り値の snapshot = JSON 深いコピーを含む)。20秒エピソードを繰り返す
out.bench.push(bench('教師+step(20秒連続エピソード×20)', 0, () => {
  let n = 0;
  for (let e = 0; e < 20; e++) {
    const sim = createSimulation(wNom, cfgCont), teacher = createRcTrainingTeacher(wNom, sim.snapshot().config, vNom);
    let s = sim.snapshot();
    while (!s.state.terminal) { s = sim.step(teacher.action(s)); n++; }
  }
  return n;
}));
// 2) 一定行動(教師計算なし)のstepだけ。衝突しない低速の円運動で12000ステップ
out.bench.push(bench('stepのみ(衝突なし・stopOnCollision=false)', 0, () => {
  const sim = createSimulation(wNom, {...RC_PHYSICS, seed: 1, timeLimit: 3600, noiseStd: 0, stopOnCollision: false});
  let n = 0;
  for (let i = 0; i < 12000; i++) { sim.step({steering: .3, speed: 1.0}); n++; }
  return n;
}));
// 3) step + observe(13本の距離レイ)
out.bench.push(bench('step+observe(13レイ)', 0, () => {
  const sim = createSimulation(wNom, cfgCont), teacher = createRcTrainingTeacher(wNom, sim.snapshot().config, vNom);
  let s = sim.snapshot(), n = 0;
  for (let e = 0; e < 5; e++) {
    sim.reset({seed: e}); s = sim.snapshot();
    const t = createRcTrainingTeacher(wNom, sim.snapshot().config, vNom);
    while (!s.state.terminal) { sim.observe(); s = sim.step(t.action(s)); n++; }
  }
  return n;
}));
// 4) 環境の作り直し(createSimulation)と reset のコスト
{
  const start = performance.now();
  for (let i = 0; i < 200; i++) createSimulation(wNom, cfgCont);
  out.create_simulation_ms = r6((performance.now() - start) / 200);
  const sim = createSimulation(wNom, cfgCont), s2 = performance.now();
  for (let i = 0; i < 2000; i++) sim.reset({seed: i});
  out.reset_ms = r6((performance.now() - s2) / 2000);
  const s3 = performance.now();
  for (let i = 0; i < 20; i++) createRcTrainingTeacher(wNom, sim.snapshot().config, vNom);
  out.create_rc_teacher_ms = r6((performance.now() - s3) / 20);
  log(`  createSimulation ${out.create_simulation_ms.toFixed(3)} ms / reset ${out.reset_ms.toFixed(4)} ms / RC教師生成 ${out.create_rc_teacher_ms.toFixed(1)} ms`);
}
// 5) サブステップ数(maxSpeed から決まる)
{
  const vr = Math.hypot(.42, .19) / 2, rate = 7.24 + vr * 7.24 / .26 * Math.tan(.52), h = Math.min(.01, .012 / rate);
  out.substep = {max_motion_rate_mps: rate, h_s: h, substeps_per_frame: Math.ceil((1 / 30) / h), note: 'maxSpeed=7.24 のとき 1フレーム(33.3 ms)を約1.1 msの小区間に分けて衝突・ゲートを判定'};
  const rate16 = 1.6 + vr * 1.6 / .26 * Math.tan(.52);
  out.substep.substeps_if_maxspeed_1_6 = Math.ceil((1 / 30) / Math.min(.01, .012 / rate16));
  log(`  サブステップ: h=${(h * 1000).toFixed(3)} ms → 1フレーム ${out.substep.substeps_per_frame} 回 (maxSpeed 1.6 なら ${out.substep.substeps_if_maxspeed_1_6} 回)`);
}
out.memory_final = Object.fromEntries(Object.entries(process.memoryUsage()).map(([k, v]) => [k, r6(v / 2 ** 20)]));
out.defaults = DEFAULTS;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, JSON.stringify(out, null, 1) + '\n');
log(`結果: ${outputPath}`);
