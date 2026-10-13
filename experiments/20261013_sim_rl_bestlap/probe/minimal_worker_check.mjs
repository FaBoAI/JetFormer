#!/usr/bin/env node
// three.js を読み込まず world-v3.json と高速版コアだけでRL用ワーカーが動くかとメモリを確認する。
// 使い方: node probe/minimal_worker_check.mjs
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url)), simRoot = path.resolve(here, '../sim');
const {createSimulation} = await import(path.join(simRoot, 'fast-sim-core.mjs'));
const world = JSON.parse(fs.readFileSync(path.join(simRoot, 'world-v3.json'), 'utf8'));
const learned = JSON.parse(fs.readFileSync(process.argv[2] ?? '/home/jetson/sim/extracted/course-simulator/results/shortcut-rc-learned-lap.json', 'utf8'));
const sameWorld = JSON.stringify(world) === JSON.stringify(learned.world);
const sim = createSimulation(world, learned.config);
sim.reset({seed: learned.seed, scenario: learned.scenario});
for (const a of learned.actions) sim.step(a, true);
const {state, metrics} = sim.peek();
const hashSame = sim.report().world_hash === learned.world_hash;
// 重い報告用データ(learned)を捨ててから常駐メモリを測る
const actions = learned.actions.map(a => ({steering: a.steering, speed: a.speed}));
let n = 0; const t0 = performance.now();
for (let r = 0; r < 40; r++) { const s = createSimulation(world, learned.config); for (const a of actions) { s.step(a, true); n++; } }
const sps = Math.round(n / (performance.now() - t0) * 1000);
if (global.gc) global.gc();
const m = process.memoryUsage();
console.log(JSON.stringify({world_json_equals_report_world: sameWorld, world_hash_equal: hashSame, elapsed_s: metrics.elapsed, reason: state.reason,
  gates: metrics.passed_gates, steps_per_s_lite: sps, rss_mb: Math.round(m.rss / 2 ** 20), heap_used_mb: Math.round(m.heapUsed / 2 ** 20)}));
