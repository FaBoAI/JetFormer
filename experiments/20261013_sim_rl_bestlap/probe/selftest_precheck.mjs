#!/usr/bin/env node
// 自己テスト用の差し替え教師(速度×0.9・開始ずれ拡大)が、ブラウザで走らせる予定の条件で衝突せず走るかを Node で先に確かめる
// 使い方: node probe/selftest_precheck.mjs
import {runCondition} from '../lib/app_emulation.mjs';
const dist = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/sim/dist/';
const selftest = await import(dist + 'rl/rl-teacher-selftest.js');
const original = await import(dist + 'rc-speed-controller.js');
const plan = original.RC_COLLECTION_PLAN;
const conds = [];
for (const [offset, yaw] of [[0.10, 6], [-0.08, -5], [0.2, 12], [-0.2, -15], [0.3, 25], [-0.3, -25]])
  for (const base of [plan[0], plan[49]]) conds.push({...base, 'spawn-offset': offset, 'spawn-yaw': yaw});
for (const c of conds) {
  const r = await runCondition(c, selftest);
  const m = r.report.metrics;
  console.log(JSON.stringify({scenario: c.scenario, offset: c['spawn-offset'], yaw: c['spawn-yaw'], reason: r.report.outcome.reason, frames: r.report.final_state.frame,
    collisions: m.collisions, laps: (m.lap_events || []).map(l => +l.elapsed.toFixed(3)), gates: m.passed_gates, metadata: r.teacher.metadata.id}));
}
