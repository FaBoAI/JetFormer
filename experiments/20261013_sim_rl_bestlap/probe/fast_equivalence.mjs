#!/usr/bin/env node
// 高速版 sim/fast-sim-core.mjs が元の sim-core.js と完全に同じ結果を出すことの確認と、速度の内訳。
// - RC教師 60シーン(連続20秒)、学習済みONNX 8.964秒走行の再生、ランダム操作(衝突・接触継続あり)を
//   両方で走らせ、report() の JSON 文字列が一致するかを比べる。
// 使い方: node probe/fast_equivalence.mjs [学習済み走行JSON] [出力JSON]
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const simRoot = path.resolve(here, '../sim');
const {loadCourseWorld} = await import(path.join(simRoot, 'qa/load-course-world.mjs'));
const original = await import(path.join(simRoot, 'dist/sim-core.js'));
const fast = await import(path.join(simRoot, 'fast-sim-core.mjs'));
const {createRcTrainingTeacher, RC_COLLECTION_PLAN} = await import(path.join(simRoot, 'dist/rc-speed-controller.js'));
const learnedLapPath = process.argv[2] ?? '/home/jetson/sim/extracted/course-simulator/results/shortcut-rc-learned-lap.json';
const outputPath = process.argv[3] ?? '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/fast_equivalence.json';

const world = await loadCourseWorld();
const RC = {dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19, wheelbase: .26,
  stopOnCollision: true, frictionMultiplier: 1, longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2};
function spawnWorld(offset, yawDeg) {
  const f = world.route[0], n = world.route[1], h = Math.atan2(-(n[1] - f[1]), n[0] - f[0]), t = yawDeg * Math.PI / 180;
  return {w: {...world, spawn: {x: f[0] + offset * Math.sin(h), z: f[1] - offset * Math.cos(h), yaw: h + t}}, v: {gate: 0, left_offset_m: offset, left_yaw_deg: t * 180 / Math.PI}};
}
// 決定的な擬似乱数(xorshift)
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }

const result = {schema: 'fast-sim-core-equivalence/v1', cases: 0, mismatches: [], timing: {}};
function compareRun(label, make, drive) {
  const a = make(original), b = make(fast);
  drive(a.sim, a.ctx); drive(b.sim, b.ctx);
  const ra = JSON.stringify(a.sim.report()), rb = JSON.stringify(b.sim.report());
  result.cases++;
  if (ra !== rb) result.mismatches.push(label);
}
// 1) RC教師 60シーン
for (const plan of RC_COLLECTION_PLAN) {
  compareRun(plan.name, core => {
    const {w, v} = spawnWorld(plan['spawn-offset'], plan['spawn-yaw']);
    const sim = core.createSimulation(w, {...RC, seed: plan.seed, timeLimit: 20, noiseStd: plan.scenario === 'noise' ? .025 : 0, episodeMode: 'continuous-laps-v1'});
    sim.reset({seed: plan.seed, scenario: plan.scenario});
    return {sim, ctx: createRcTrainingTeacher(w, sim.snapshot().config, v)};
  }, (sim, teacher) => { let s = sim.snapshot(); while (!s.state.terminal) s = sim.step(teacher.action(s)); });
}
// 2) 学習済みONNX走行の再生
const learned = JSON.parse(fs.readFileSync(learnedLapPath, 'utf8'));
compareRun('learned-onnx-8.964', core => {
  const sim = core.createSimulation(learned.world, learned.config); sim.reset({seed: learned.seed, scenario: learned.scenario});
  return {sim, ctx: null};
}, sim => { for (const a of learned.actions) sim.step(a); });
// 3) ランダム操作(衝突で止まる/止まらない両方。接触・壁沿い・境界を含む)
for (let k = 0; k < 40; k++) {
  const stop = k % 2 === 0;
  compareRun(`random-${k}-stop${stop}`, core => {
    const sim = core.createSimulation(world, {...RC, seed: k, timeLimit: 30, noiseStd: 0, stopOnCollision: stop});
    return {sim, ctx: rng(1000 + k)};
  }, (sim, r) => {
    let steer = 0, speed = 0;
    for (let i = 0; i < 900; i++) {
      if (i % 10 === 0) { steer = (r() * 2 - 1) * .52; speed = r() * 7.24 * (r() < .1 ? -0.1 : 1); }
      const s = sim.step({steering: steer, speed});
      if (s.state.terminal) break;
    }
  });
}
result.pass = result.mismatches.length === 0;
console.log(`一致確認: ${result.cases}ケース中 不一致 ${result.mismatches.length} → ${result.pass ? '完全一致' : '不一致あり ' + result.mismatches.join(',')}`);

// ---- 速度の内訳(教師操作を事前に記録して物理だけを測る) ----
const {w: wNom, v: vNom} = spawnWorld(0, 0);
const cfg = {...RC, seed: 1, timeLimit: 20, noiseStd: 0, episodeMode: 'continuous-laps-v1'};
const actions = [];
{
  const sim = original.createSimulation(wNom, cfg), t = createRcTrainingTeacher(wNom, sim.snapshot().config, vNom);
  let s = sim.snapshot();
  while (!s.state.terminal) { const a = t.action(s); actions.push(a); s = sim.step(a); }
}
function time(label, reps, fn) {
  const start = performance.now(); let n = 0;
  for (let r = 0; r < reps; r++) n += fn();
  const ms = performance.now() - start;
  result.timing[label] = {steps: n, ms: Math.round(ms), steps_per_s: Math.round(n / ms * 1000)};
  console.log(`  ${label}: ${result.timing[label].steps_per_s} steps/s`);
}
console.log('速度(1プロセス、20秒=600フレームの教師操作列を再生):');
time('元: step(snapshot付き)', 10, () => { const sim = original.createSimulation(wNom, cfg); for (const a of actions) sim.step(a); return actions.length; });
time('高速: step(snapshot付き)', 10, () => { const sim = fast.createSimulation(wNom, cfg); for (const a of actions) sim.step(a); return actions.length; });
time('高速: step(lite)+peek', 30, () => { const sim = fast.createSimulation(wNom, cfg); for (const a of actions) { sim.step(a, true); sim.peek(); } return actions.length; });
time('高速: step(lite)+peek+RC教師action', 10, () => {
  const sim = fast.createSimulation(wNom, cfg), t = createRcTrainingTeacher(wNom, sim.snapshot().config, vNom);
  let n = 0; while (!sim.peek().state.terminal) { sim.step(t.action(sim.peek()), true); n++; } return n;
});
time('高速: step(lite)+observe(13レイ)', 5, () => { const sim = fast.createSimulation(wNom, cfg); for (const a of actions) { sim.step(a, true); sim.observe(); } return actions.length; });
result.memory_mb = Object.fromEntries(Object.entries(process.memoryUsage()).map(([k, v]) => [k, Math.round(v / 2 ** 20)]));
console.log(`メモリ: RSS ${result.memory_mb.rss} MB, heapUsed ${result.memory_mb.heapUsed} MB`);
fs.writeFileSync(outputPath, JSON.stringify(result, null, 1) + '\n');
if (!result.pass) process.exitCode = 1;
