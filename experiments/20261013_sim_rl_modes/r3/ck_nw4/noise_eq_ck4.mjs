// 確認 r4: アプリの場面「noise」(noiseStd 0.025、レイの距離だけに雑音)と「dim」(明るさだけ)で、教師の操作が baseline と同じか(2 つの開始)
import fs from 'node:fs';
import {coreO, makeStart, NOMINAL_CFG} from './own_ck4.mjs';
const CAND = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r4';
const RT = await import(`${CAND}/runtime/mode_runtime.mjs`);
const W = JSON.parse(fs.readFileSync(`${CAND}/policy.json`, 'utf8'));
async function run(gate, off, yaw, noiseStd, scenario) {
  const {world, variation} = makeStart(gate, off, yaw);
  const cfg = {...NOMINAL_CFG, seed: 7, timeLimit: 20, episodeMode: 'continuous-laps-v1', noiseStd};
  const sim = coreO.createSimulation(world, cfg);
  let snap = sim.reset({seed: 7, scenario});
  const t = RT.createModeTeacher(world, sim.snapshot().config, variation, {mode: 'no_wall', weights: W});
  const acts = [];
  while (!snap.state.terminal) { const o = t.action(snap); acts.push(o.steering, o.speed); snap = sim.step(o); }
  return acts;
}
for (const [g, o, y] of [[3, .1, -7], [23, -.05, 13]]) {
  const a = await run(g, o, y, 0, 'baseline'), b = await run(g, o, y, .025, 'noise'), c = await run(g, o, y, 0, 'dim');
  console.log(JSON.stringify({start: [g, o, y], frames: a.length / 2, noise_same: a.length === b.length && a.every((v, i) => v === b[i]), dim_same: a.length === c.length && a.every((v, i) => v === c[i])}));
}
