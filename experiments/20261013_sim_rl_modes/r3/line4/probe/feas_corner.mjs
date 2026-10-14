// 角の開始から「前へだけ」で抜けられるか(下調べ。2026-10-14 line r4)。止まった開始の姿勢から、舵 5 × 速さ 3 × 10 フレームの操作を 4 段まで深さ優先で調べ、
// すき間 ≥ thr のまま R の上で 1.0 m 以上進み、そのあと予備の操縦(最大すき間の線の pure pursuit + ブレーキ)で止まれる列があるかを数える。
import {createCarModel, createShield} from '../rt4/shield.mjs';
import {prepareGeometry, clearance, projectWindow, LOOP_LENGTH} from '../rt4/features.mjs';
import {activeWorldFor, simConfig} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';
import {BASE_WORLD} from '../episode_ln4.mjs';
const [gate, off, yaw, thr] = [+process.argv[2], +process.argv[3], +process.argv[4], +(process.argv[5] ?? .04)];
const {world} = activeWorldFor(BASE_WORLD, gate, off, yaw);
const geom = prepareGeometry(world), cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: 1});
const model = createCarModel(geom, cfg), sp = world.spawn;
const poses = new Float64Array(3 * model.maxSubsteps);
const s0 = projectWindow(sp.x, sp.z, 0, LOOP_LENGTH).s;
const STEERS = [-.52, -.26, 0, .26, .52], SPEEDS = [.5, 1.0, 1.5], SEG = 10, DEPTH = 4;
let found = 0, best = null, nodes = 0;
function seg(st, d, v) {
  let minC = Infinity;
  for (let k = 0; k < SEG; k++) {
    const r = model.stepFrame(st, d, v, null, poses);
    for (let i = 0; i < r.n; i++) minC = Math.min(minC, clearance(geom, poses[3 * i], poses[3 * i + 1], poses[3 * i + 2], model.L, model.W, .5));
  }
  return minC;
}
function brakeOk(st) {   // 舵をまっすぐにしてブレーキで止まるまで(すき間 ≥ thr)
  const q = {...st}; let minC = Infinity;
  for (let k = 0; k < 200 && Math.abs(q.speed) > .02; k++) { const r = model.stepFrame(q, 0, 0, null, poses); for (let i = 0; i < r.n; i++) minC = Math.min(minC, clearance(geom, poses[3*i], poses[3*i+1], poses[3*i+2], model.L, model.W, .5)); }
  return minC >= thr;
}
function dfs(st, depth, path, minC) {
  nodes++;
  const ds = ((projectWindow(st.x, st.z, s0 - 1, s0 + 3).s - s0) % LOOP_LENGTH + LOOP_LENGTH) % LOOP_LENGTH;
  if (ds >= 1.0 && ds < 5 && brakeOk(st)) { found++; if (!best || minC > best.minC) best = {path: [...path], minC, ds}; return; }
  if (depth === DEPTH) return;
  for (const d of STEERS) for (const v of SPEEDS) {
    const q = {...st}; const c = seg(q, d, v);
    if (c < thr) continue;
    path.push([d, v]); dfs(q, depth + 1, path, Math.min(minC, c)); path.pop();
  }
}
const st0 = {x: sp.x, z: sp.z, yaw: sp.yaw, speed: 0, steering: 0};
const c0 = clearance(geom, sp.x, sp.z, sp.yaw, model.L, model.W, .5);
dfs(st0, 0, [], c0);
console.log(JSON.stringify({gate, off, yaw, thr, start_clearance: c0, nodes, found_paths: found, best}));
