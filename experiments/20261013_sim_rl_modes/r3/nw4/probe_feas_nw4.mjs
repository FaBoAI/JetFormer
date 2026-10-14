// no_wall r4 の下調べ: 止まった状態(または開始)から、前へだけ・後ろへ下がってから前へ、の操縦で 8 cm を保って抜けられるかを総当たりで調べる
// node probe_feas_nw4.mjs gate off yaw frame
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, NOMINAL} from './run_nw4r.mjs';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const core = await import(`${REL}/dist/sim-core.js`);
const RT = await import(`${EXP}/r3/nw/runtime_v4/mode_runtime.mjs`);
const {createCarModel} = await import(`${EXP}/r3/nw/runtime_v4/shield.mjs`);
const {clearance, prepareGeometry, projectWindow} = await import(`${EXP}/r3/nw/runtime_v4/features.mjs`);
const [g, o, y, F] = process.argv.slice(2).map(Number);
const {world, variation} = activeWorld(BASE_WORLD, g, o, y);
const cfg = {...NOMINAL, seed: 1, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
const sim = core.createSimulation(world, cfg);
let snap = sim.reset({seed: 1, scenario: 'baseline'});
const w = JSON.parse(fs.readFileSync(`${EXP}/r3/candidates/no_wall_r3/policy.json`, 'utf8'));
const t = RT.createModeTeacher(world, snap.config, variation, {mode: 'no_wall', weights: w});
while (snap.state.frame < F) { const out = t.action(snap); snap = sim.step({steering: out.steering, speed: out.speed}); }
const S0 = {...snap.state};
const geom = prepareGeometry(world), model = createCarModel(geom, snap.config);
const proj = {};
const sOf = st => { projectWindow(st.x, st.z, 0, 40, proj); return proj.s; };
console.log('state', F, 'x', S0.x.toFixed(3), 'z', S0.z.toFixed(3), 'yaw', S0.yaw.toFixed(3), 'v', S0.speed.toFixed(3), 'δ', S0.steering.toFixed(3), 'c', clearance(geom, S0.x, S0.z, S0.yaw, .42, .19, .5).toFixed(4), 's', sOf(S0).toFixed(3));
const poses = new Float64Array(3 * model.maxSubsteps);
function run(plan) {   // plan = [[steer, speed, frames], ...]
  const st = {x: S0.x, z: S0.z, yaw: S0.yaw, speed: S0.speed, steering: S0.steering};
  let minC = 9, back = 0, fr = 0;
  for (const [d, v, n] of plan) for (let k = 0; k < n; k++) {
    const r = model.stepFrame(st, d, v, null, poses); fr++;
    for (let i = 0; i < r.n; i++) minC = Math.min(minC, clearance(geom, poses[3 * i], poses[3 * i + 1], poses[3 * i + 2], .42, .19, .5));
    if (st.speed < 0) back++;
  }
  return {minC, s: sOf(st), v: st.speed, fr};
}
const S = [-.52, -.39, -.26, -.13, 0, .13, .26, .39, .52];
let best = null;
// 前へだけ: 舵を止まったまま k フレーム合わせてから、速さ v で 45 フレーム
for (const d of S) for (const v of [.3, .6, 1.0, 1.5]) for (const k of [0, 10]) {
  const r = run([[d, 0, k], [d, v, 45]]);
  if (!best || r.minC > best.minC) best = {...r, d, v, k};
}
console.log('前へだけ(一定の舵)の最大の最小すき間', JSON.stringify(best));
let okF = [];
for (const d1 of S) for (const d2 of S) for (const v of [.4, .8]) {
  const r = run([[d1, v, 15], [d2, v, 30]]);
  if (r.minC >= .08) okF.push({d1, d2, v, minC: +r.minC.toFixed(4), s: +r.s.toFixed(2)});
}
console.log('前へだけ(舵 2 段)で 8 cm 以上', okF.length, JSON.stringify(okF.slice(0, 5)));
// 後ろへ: 舵 dr で速さ −vr を n フレーム → 止まる → 舵 df で前へ v
const okR = [];
for (const dr of S) for (const vr of [-.3, -.5]) for (const n of [10, 20, 30]) for (const df of S) for (const v of [.5, 1.0]) {
  const r = run([[dr, vr, n], [dr, 0, 15], [df, 0, 8], [df, v, 40]]);
  if (r.minC >= .08) okR.push({dr, vr, n, df, v, minC: +r.minC.toFixed(4), s: +r.s.toFixed(2), fr: r.fr});
}
okR.sort((a, b) => b.s - a.s);
console.log('後ろへ下がってから前へで 8 cm 以上', okR.length, JSON.stringify(okR.slice(0, 8)));
