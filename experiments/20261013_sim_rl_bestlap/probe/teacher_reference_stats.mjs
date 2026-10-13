#!/usr/bin/env node
// 教師の参照軌跡(最適化参照・RC教師の単周/連続)の点の間隔・長さ・速度・μ・横加速度の上限を数える
// 使い方: node probe/teacher_reference_stats.mjs
import {baseWorld, configFromCondition, activeWorldFor, createSimulation} from '../lib/app_emulation.mjs';
const dist = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/sim/dist/';
const opt = await import(dist + 'optimized-reference.js');
const rc = await import(dist + 'rc-speed-controller.js');
const world = await baseWorld();
const stat = (pts, label) => {
  const ds = pts.slice(1).map((p, i) => p.s - pts[i].s), v = pts.map(p => p.speed);
  const lat = pts.map(p => p.speed ** 2 * Math.abs(p.curvature));
  const mu = {}; for (const p of pts) mu[p.mu] = (mu[p.mu] || 0) + 1;
  const steer = pts.map(p => Math.abs(p.steering));
  return {label, points: pts.length, length_m: +pts.at(-1).s.toFixed(3), ds_median_m: +ds.sort((a, b) => a - b)[ds.length >> 1].toFixed(4),
    speed_max: +Math.max(...v).toFixed(3), speed_mean: +(v.reduce((a, b) => a + b) / v.length).toFixed(3),
    lateral_acc_max: +Math.max(...lat).toFixed(3), lateral_acc_over_mu_g_max: +Math.max(...pts.map((p, i) => lat[i] / (p.mu * 9.80665))).toFixed(3),
    steering_abs_max_rad: +Math.max(...steer).toFixed(3), mu_point_counts: mu};
};
const out = {};
out.optimized = {...stat(opt.OPTIMIZED_REFERENCE.points, 'optimized-reference(一定加減速の物理用)'), metadata_time_s: opt.OPTIMIZED_TEACHER_METADATA.recipe.time};
const plan0 = rc.RC_COLLECTION_PLAN[0];
for (const mode of ['single-lap', 'continuous-laps-v1']) {
  const c = {...plan0, 'episode-mode': mode, limit: mode === 'single-lap' ? 30 : 20};
  const {config, scenario} = configFromCondition(c), {world: w, variation} = activeWorldFor(world, 0, 0, 0);
  const sim = createSimulation(w, config); sim.reset({seed: config.seed, scenario});
  const t = rc.createRcTrainingTeacher(w, sim.snapshot().config, variation);
  const pts = t.reference.points;
  const s = stat(pts, 'RC教師 ' + mode);
  s.planned_time_s = +t.reference.planned_time_s.toFixed(3);
  s.options = t.reference.options;
  // 指令(command)と参照速度の差: 1 周走らせて記録
  let maxCmd = 0, n = 0, cmdOverRef = 0;
  while (!sim.snapshot().state.terminal) { const snap = sim.snapshot(), a = t.action(snap); maxCmd = Math.max(maxCmd, a.speed); if (a.speed > snap.state.speed + 0.5) cmdOverRef++; sim.step(a); n++; }
  const r = sim.report();
  s.run = {frames: n, reason: r.outcome.reason, max_speed_command: +maxCmd.toFixed(3), frames_command_gt_speed_plus_0_5: cmdOverRef,
    laps: (r.metrics.lap_events || []).map(l => +l.elapsed.toFixed(3)), elapsed: +r.metrics.elapsed.toFixed(3),
    traction_limited_steps: r.metrics.traction_limited_steps, max_lateral_acceleration: +r.metrics.max_lateral_acceleration.toFixed(3)};
  if (mode === 'continuous-laps-v1') { s.loop_length_m = t.reference.points.length && +(t.reference.length / 4).toFixed(3); }
  out[mode] = s;
}
console.log(JSON.stringify(out, null, 1));
