#!/usr/bin/env node
// 記録の段: ブラウザ(Firefox)で記録した閉ループの走りと、Node の予想(同じ差し込みモジュール・元の sim-core.js)が、
// どこから・どれだけずれるかを調べる(2026-10-14)
//
//   node record/divergence_probe.mjs <シーンのフォルダー> [...] [--out divergence.json]
//
//  1) 記録の開始姿勢から Node で閉ループに走らせ(predict_rec60.mjs と同じ)、記録の操作・軌跡とフレームごとに比べる:
//     操作の差が 1e-15・1e-12・1e-9・1e-6・1e-3 を初めて超えるフレーム、位置の差の最大
//  2) 感度: Node の閉ループで、開始の x を 1e-12 m・1e-9 m だけずらしたときの周回時間(ずれの増え方が方策と物理の性質かを見る)
// データは変えない。
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {baseWorld, activeWorldFor, configFromCondition, createSimulation, SIM_ROOT} from '../lib/app_emulation.mjs';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
const outPath = oi >= 0 ? args.splice(oi, 2)[1] : null;
const teacher = await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rl/rl-teacher-bestlap.js')).href);
const plan = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/recording/plan_rlbest60.json', 'utf8'));
const world0 = await baseWorld();

function sceneRoot(p) {
  p = path.resolve(p);
  if (fs.existsSync(path.join(p, 'simulation/session.json'))) return p;
  const c = fs.readdirSync(p).map(x => path.join(p, x)).filter(x => fs.existsSync(path.join(x, 'simulation/session.json')));
  if (c.length !== 1) throw new Error('シーンのフォルダーが見つかりません: ' + p);
  return c[0];
}

// 記録と同じ条件で Node の閉ループ(spawnDx だけ開始の x をずらす)
function closedLoop(cond, spawnDx = 0) {
  const {config, scenario} = configFromCondition(cond);
  const {world, variation} = activeWorldFor(JSON.parse(JSON.stringify(world0)), cond['start-gate'], cond['spawn-offset'], cond['spawn-yaw']);
  world.spawn = {...world.spawn, x: world.spawn.x + spawnDx};
  const sim = createSimulation(world, config);
  sim.reset({seed: config.seed, scenario});
  // 開始の検査は ±1e-12 の範囲で行うので、ずらしたときは検査を通すために標準の開始で作った world を教師に渡す(観測は sim の状態から作る)
  const tWorld = spawnDx ? {...world, spawn: {...world.spawn, x: world.spawn.x - spawnDx}} : world;
  const t = teacher.createRcTrainingTeacher(tWorld, sim.snapshot().config, variation);
  const actions = [], states = [];
  while (!sim.snapshot().state.terminal) {
    const a = t.action(sim.snapshot());
    actions.push(a); sim.step(a); const s = sim.snapshot().state; states.push({x: s.x, z: s.z, yaw: s.yaw, speed: s.speed});
  }
  const r = sim.report();
  return {actions, states, laps: (r.metrics.lap_events ?? []).map(l => l.elapsed), collisions: r.metrics.collisions, reason: r.outcome.reason};
}

const results = [];
for (const arg of args) {
  const root = sceneRoot(arg);
  const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
  const session = read('simulation/session.json'), report = read('simulation/evaluation-report.json');
  const cond = plan.find(c => c.name === session.display_name);
  const node = closedLoop(cond);
  const thresholds = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2];
  const first = Object.fromEntries(thresholds.map(t => [String(t), null]));
  let maxPos = 0, maxPosFrame = -1;
  const n = Math.min(node.actions.length, report.actions.length);
  for (let i = 0; i < n; i++) {
    const d = Math.max(Math.abs(node.actions[i].steering - report.actions[i].steering), Math.abs(node.actions[i].speed - report.actions[i].speed));
    for (const t of thresholds) if (first[String(t)] === null && d > t) first[String(t)] = i;
    const tr = report.trace[i], s = node.states[i];
    const dp = Math.hypot(tr.x - s.x, tr.z - s.z);
    if (dp > maxPos) { maxPos = dp; maxPosFrame = i; }
  }
  const recLaps = (report.metrics.lap_events ?? []).map(l => l.elapsed);
  const sens = [1e-12, 1e-9].map(dx => { const r = closedLoop(cond, dx); return {spawn_dx_m: dx, laps: r.laps, collisions: r.collisions, reason: r.reason}; });
  results.push({scene: session.display_name, dataset: root, frames: n, browser_laps: recLaps, node_laps: node.laps,
    lap_abs_diff_s: recLaps.map((x, k) => Math.abs(x - node.laps[k])), first_frame_action_diff_exceeds: first,
    max_position_diff_m: maxPos, max_position_diff_frame: maxPosFrame, node_sensitivity: sens});
  process.stderr.write(`[ずれ] ${session.display_name}: 操作の差が 1e-15 を超える最初 ${first['1e-15']}・1e-9 ${first['1e-9']}・1e-3 ${first['0.001']} フレーム、` +
    `周 ブラウザ ${recLaps.map(x => x.toFixed(4))} / Node ${node.laps.map(x => x.toFixed(4))}\n`);
}
const out = {schema: 'rl-bestlap-browser-node-divergence/v1', created_at: new Date().toISOString(), scenes: results};
if (outPath) fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
