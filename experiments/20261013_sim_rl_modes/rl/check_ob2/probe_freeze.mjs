// 障害物よけの確認 r2: 「開始で止まったまま動かない」回の下調べ(2026-10-14、確認の段 r2)。障害物なしのコースで、
// 記録・試験の開始の分布の範囲(ゲート 0〜32・左 −0.15〜+0.15 m・向き −10〜+10°)の格子 33 × 7 × 5 = 1,155 の開始から 3 秒(90 フレーム)走らせ、
// R の上の進み(自前の射影)が 0.6 m 未満なら「止まる」。止まった開始は 20 秒(連続周回)でも走らせ、valid_episode・周の数を見る。
// r2 の方策・最速の方策・r1 の方策で同じ格子。
//   node probe_freeze.mjs → check_obstacle_r2/probe_freeze.json
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, NOMINAL, policyFor, RT, startClearance, runOne} from './run_ob2.mjs';
import {createProjector} from '../check_ln/own_extra.mjs';

const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_obstacle_r2';
const core = await import('/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5/dist/sim-core.js');
const MODE = {obstacle: 'obstacle', obstacle_r1: 'obstacle', fastest: 'fastest'};
function short(pol, st, nFrames = 90) {
  const P = policyFor(pol), {world, variation} = activeWorld(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  const sim = core.createSimulation(world, {...NOMINAL, seed: 4600000, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed: 4600000, scenario: 'baseline'});
  if (snap.state.terminal) return {invalid_spawn: true};
  const t = RT.createModeTeacher(world, snap.config, variation, {mode: MODE[pol], weights: P.weights, policySha256: P.sha, params: {}});
  const pj = createProjector();
  let s0 = null, s = null, maxSp = 0, k = 0;
  while (!snap.state.terminal && k < nFrames) {
    const r = k === 0 ? pj.reset(snap.state.x, snap.state.z, snap.state.yaw) : pj.advance(snap.state.x, snap.state.z);
    if (k === 0) s0 = r.s; s = r.s;
    const a = t.action({state: snap.state, metrics: snap.metrics, config: snap.config});
    maxSp = Math.max(maxSp, snap.state.speed);
    snap = sim.step({steering: a.steering, speed: a.speed}); k++;
  }
  return {progress_m: s - s0, frames: k, terminal: snap.state.terminal, reason: snap.state.reason, max_speed: maxSp, ey0: null};
}
const grid = [];
for (let g = 0; g < 33; g++) for (const off of [-.15, -.10, -.05, 0, .05, .10, .15]) for (const yaw of [-10, -5, 0, 5, 10]) grid.push({gate: g, left_offset_m: off, left_yaw_deg: yaw});
const res = {grid: grid.length, policies: {}};
for (const pol of ['obstacle', 'fastest', 'obstacle_r1']) {
  const frozen = [], early_col = [];
  let n = 0;
  for (const st of grid) {
    if (!(startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) > 1e-9)) continue;
    n++;
    const r = short(pol, st);
    if (r.invalid_spawn) continue;
    if (r.terminal) early_col.push({...st, reason: r.reason, frames: r.frames});
    else if (r.progress_m < .6) frozen.push({...st, progress_m: +r.progress_m.toFixed(3), max_speed: +r.max_speed.toFixed(3)});
  }
  const full = [];
  if (pol === 'obstacle') for (const st of frozen) {
    const r = await runOne({policy: pol, start: {...st, seed: 4600000}, layout: null, keepFrames: false});
    full.push({...st, valid_episode: r.valid_episode, laps: r.laps.length, reason: r.reason, distance: +r.distance.toFixed(3), s_travel: +r.s_travel.toFixed(3)});
  }
  const byGate = {};
  for (const f of frozen) byGate[f.gate] = (byGate[f.gate] ?? 0) + 1;
  res.policies[pol] = {starts: n, frozen: frozen.length, frozen_by_gate: byGate, early_terminal: early_col.length, early_terminal_rows: early_col.slice(0, 20), frozen_rows: frozen, full_20s: full};
  console.log(pol, JSON.stringify({starts: n, frozen: frozen.length, byGate, early: early_col.length, full_valid: full.filter(f => f.valid_episode).length, full_zero_lap: full.filter(f => !f.laps).length}));
}
fs.writeFileSync(`${OUT}/probe_freeze.json`, JSON.stringify(res, null, 1));
