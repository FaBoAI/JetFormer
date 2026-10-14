// 診断: シールドありの 1 回をフレーム F まで走らせ、その状態で予備の操縦だけの先読みの軌跡とすき間を表示する(2026-10-13)
import fs from 'node:fs';
import path from 'node:path';
import {buildJobs} from '../eval_modes.mjs';
import {readJson, BASE_WORLD, loadCore} from '../episode.mjs';
import {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS} from '../../snapshot/rl/env_core.mjs';
import {createModeTeacher} from '../runtime/mode_runtime.mjs';
import {createShield, createCarModel} from '../runtime/shield.mjs';
import {prepareGeometry, clearance, projectWindow, loopPointAt} from '../runtime/features.mjs';
import {tableAt} from '../runtime/line_profiles.mjs';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const J = buildJobs('no_wall', 'validation').filter(j => j.test === opt('--test', 'NW-N60'));
const job = J[+opt('--idx', '0')], F = +opt('--frame', '188');
const weights = readJson(opt('--policy'));
const core = await loadCore('fast');
const start = job.spec.start;
const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: start.seed});
const sim = core.createSimulation(world, cfg);
sim.reset({seed: cfg.seed, scenario: 'baseline'});
let snap = sim.peek();
const cfgN = JSON.parse(JSON.stringify(snap.config));
const teacher = createModeTeacher(world, cfgN, variation, {mode: 'no_wall', weights, overrides: {shield: false}});
const geom = prepareGeometry(world);
const shield = createShield({geometry: geom, config: cfgN});
while (!snap.state.terminal && snap.state.frame < F) { const a = teacher.action({state: snap.state, metrics: snap.metrics, config: cfgN}); sim.step(a, true); snap = sim.peek(); }
const st = {...snap.state};
const lg = teacher.log.at(-1);
console.log('frame', st.frame, 'x', st.x.toFixed(3), 'z', st.z.toFixed(3), 'yaw', st.yaw.toFixed(3), 'v', st.speed.toFixed(3), 'steer', st.steering.toFixed(3), 's', lg.s.toFixed(3), 'dmin', clearance(geom, st.x, st.z, st.yaw, .42, .19, .5).toFixed(3));
// 予備の操縦だけで 30 フレーム(標準の模型)
const model = createCarModel(geom, cfgN);
const q = {x: st.x, z: st.z, yaw: st.yaw, speed: st.speed, steering: st.steering};
let s = lg.s;
const proj = {s: 0, d: 0, px: 0, pz: 0, tx: 1, tz: 0}, pt = {x: 0, z: 0, tx: 0, tz: 0};
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const poses = new Float64Array(3 * model.maxSubsteps);
for (let j = 0; j < 30; j++) {
  projectWindow(q.x, q.z, s - .5, s + 1.5, proj); s = proj.s;
  const sT = proj.s + (+opt('--la', '0.6')) + (+opt('--k', '0')) * Math.abs(q.speed), e = tableAt('emid', sT);
  loopPointAt(sT, pt);
  const tx = pt.x + e * pt.tz, tz = pt.z - e * pt.tx, dx = tx - q.x, dz = tz - q.z;
  const alpha = wrap(Math.atan2(-dz, dx) - q.yaw), ld = Math.max(Math.hypot(dx, dz), .3);
  const steer = Math.max(-.52, Math.min(.52, Math.atan2(2 * .26 * Math.sin(alpha), ld)));
  const r = model.stepFrame(q, steer, 0, null, poses);
  let mc = Infinity; for (let k = 0; k < r.n; k++) mc = Math.min(mc, clearance(geom, poses[3 * k], poses[3 * k + 1], poses[3 * k + 2], .42, .19, .5));
  // e_y の実際と目標
  const ey = (q.x - proj.px) * proj.tz + (q.z - proj.pz) * -proj.tx;
  console.log(`  j=${j} s=${s.toFixed(2)} emid(sT)=${e.toFixed(3)} ey=${ey.toFixed(3)} steer_cmd=${steer.toFixed(3)} steer=${q.steering.toFixed(3)} v=${q.speed.toFixed(2)} x=${q.x.toFixed(2)} z=${q.z.toFixed(2)} yaw=${q.yaw.toFixed(3)} clear=${mc.toFixed(3)} rhoL=${tableAt('rhoL', s).toFixed(3)} rhoR=${tableAt('rhoR', s).toFixed(3)}`);
}
