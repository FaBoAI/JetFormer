#!/usr/bin/env node
// 1 プロセス(1 コア)の中の時間の内訳(2026-10-13、RL 環境の段)
//   node rl/bench_node.mjs [出力.json]   (既定 /home/jetson/jetformer_reports/sim_rl_bestlap_20261013/env/bench_node.json)
// 教師の 1 周分の操作列を使って、環境の 1 ステップ(物理 + 観測 + 報酬)・物理だけ・観測・レイ・すき間・射影・リセットの時間を測る。
import fs from 'node:fs';
import os from 'node:os';
import * as E from './env_core.mjs';
import * as F from './features.mjs';

const out = process.argv[2] ?? '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/env/bench_node.json';
const core = await E.loadCore('fast'), orig = await E.loadCore('original'), T = await E.loadRcTeacher();
const env = new E.RlEnv({core, teacherModule: T, teacherActions: true});
env.reset({gate: 0, seed: 1}); const acts = [];
while (!env.done) { const c = env.teacherCommand; acts.push([c.steering, c.speed]); env.stepCommand(c.steering, c.speed); }
const time = (fn, n) => { fn(); const t = performance.now(); const k = fn(n); return (performance.now() - t) / k; };
const res = {schema: 'rl-bestlap-node-bench/v1', created_at: new Date().toISOString(), loadavg_before: os.loadavg(), node: process.version};
const e2 = new E.RlEnv({core});
res.env_step_us = 1000 * time(() => { let n = 0; for (let r = 0; r < 10; r++) { e2.reset({gate: 0, seed: 1}); for (const [s, v] of acts) { if (e2.done) break; e2.stepCommand(s, v); n++; } } return n; });
const e3 = new E.RlEnv({core: orig, coreKind: 'original'});
res.env_step_original_core_us = 1000 * time(() => { let n = 0; for (let r = 0; r < 2; r++) { e3.reset({gate: 0, seed: 1}); for (const [s, v] of acts) { if (e3.done) break; e3.stepCommand(s, v); n++; } } return n; });
const sim = core.createSimulation(e2.world, e2.config);
res.sim_step_fast_us = 1000 * time(() => { let n = 0; for (let r = 0; r < 10; r++) { sim.reset({seed: 1}); for (const [s, v] of acts) { sim.step({steering: s, speed: v}, true); n++; if (sim.peek().state.terminal) break; } } return n; });
const tr = F.createFeatureTracker(e2.world, e2.config), o = new Float64Array(79), st = {x: 0, z: 2.1, yaw: .1, speed: 3, steering: .1};
tr.resetProgress(-.85, 2.1, 0);
res.observe_us = 1000 * time(() => { for (let i = 0; i < 20000; i++) { st.x = -.85 + (i % 100) * .01; tr.observe(st, 0, 0, 0, o); } return 20000; });
res.rays_us = 1000 * time(() => { for (let i = 0; i < 20000; i++) F.castRays(tr.geometry, -.85 + (i % 100) * .01, 2.1, .1, o, 56); return 20000; });
res.rays_reference_us = 1000 * time(() => { for (let i = 0; i < 5000; i++) F.castRaysReference(tr.geometry, -.85 + (i % 100) * .01, 2.1, .1, o, 56); return 5000; });
res.clearance_us = 1000 * time(() => { for (let i = 0; i < 20000; i++) F.clearance(tr.geometry, -.85 + (i % 100) * .01, 2.1, .1); return 20000; });
res.reset_ms = time(() => { for (let i = 0; i < 200; i++) e2.reset({gate: i % 33, left_offset_m: .05, seed: i}); return 200; });
res.teacher_reset_ms = time(() => { for (let i = 0; i < 20; i++) env.reset({gate: 0, seed: i}); return 20; });
res.env_steps_per_s_single_process = 1e6 / res.env_step_us;
res.rss_mb = process.memoryUsage().rss / 2 ** 20;
res.loadavg_after = os.loadavg();
fs.writeFileSync(out, JSON.stringify(res, null, 1) + '\n');
console.log(JSON.stringify(res, null, 1));
