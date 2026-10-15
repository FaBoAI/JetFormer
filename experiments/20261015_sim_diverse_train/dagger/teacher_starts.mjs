#!/usr/bin/env node
// DAgger の準備(sim_diverse_train_20261015、2026-10-15。protocol.json dagger.states_starts.why): 状態集めの開始 D0〜D11(と確かめに E0〜E11)で、
// 先生 T_fast・T_margin(10/14 の web/teacher_core.mjs の写し、遅延 0)が Node の元の sim-core.js で単周を完走するかを確かめる。
//   node teacher_starts.mjs <計画の JSON(eval_common.condition の並び)> <出力.json>
// 世界・設定はアプリの reset() と同じ作り方(lib/app_emulation.mjs の写し)。E0〜E11 は 10/14 の先生の周回時間と同じになるはず。
import fs from 'node:fs';
import {baseWorld, activeWorldFor, configFromCondition, createSimulation} from '../lib/app_emulation.mjs';
import {createTeacherCore} from '../web/teacher_core.mjs';
import WF from '../web/rt/weights_fast.mjs';
import WM from '../web/rt/weights_margin.mjs';

const [planPath, outPath] = process.argv.slice(2);
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
const W = {fast: WF, margin: WM};
const base = await baseWorld();
const out = [];
for (const c of plan) {
  for (const kind of ['fast', 'margin']) {
    const {config, scenario} = configFromCondition(c);
    const {world, variation} = activeWorldFor(base, c['start-gate'], c['spawn-offset'], c['spawn-yaw']);
    const sim = createSimulation(world, config);
    let snap = sim.reset({seed: config.seed, scenario});
    const cfg = JSON.parse(JSON.stringify(snap.config));
    const core = createTeacherCore(kind, world, cfg, W[kind]);
    let prev = [0, 0], f = 0, dmin = Infinity;
    while (!snap.state.terminal) {
      const st = snap.state, ri = snap.metrics.route_index;
      if (f === 0) core.begin(st, ri); else core.advance(st, ri);
      const d = core.decide(st, prev[0], prev[1]);
      dmin = Math.min(dmin, d.dmin ?? Infinity);
      prev = [d.a0, d.a1];
      snap = sim.step({steering: d.steering, speed: d.speed});
      f++;
    }
    const rep = sim.report();
    const ok = rep.outcome.reason === 'route_complete' && rep.metrics.collisions === 0;
    out.push({name: c.name, start: [c['start-gate'], c['spawn-offset'], c['spawn-yaw']], variation, teacher: kind, reason: rep.outcome.reason,
      completed: ok, lap_s: ok ? rep.metrics.elapsed : null, passed_gates: rep.metrics.passed_gates, frames: rep.actions.length, min_clearance_m: dmin});
    console.error(`${c.name} ${kind}: ${rep.outcome.reason} ${ok ? rep.metrics.elapsed.toFixed(3) : ''} ゲート ${rep.metrics.passed_gates}`);
  }
}
fs.writeFileSync(outPath, JSON.stringify({schema: 'jf-sim-diverse-dagger-teacher-starts/v1', created_at: new Date().toISOString(), rows: out}, null, 1) + '\n');
console.log(outPath, out.filter(r => r.completed).length, '/', out.length);
