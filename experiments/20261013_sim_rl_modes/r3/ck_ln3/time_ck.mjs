// シールドの 1 フレームの計算時間(Node、混んだ CPU。参考): 試験の P37・H48 と STD 左寄り 0.8 で、シールドあり/なしの教師の action() の時間
import fs from 'node:fs';
import {REL, EXP, CAND, BASE_WORLD, SHIELD_PARAMS, spawnFor, PHYS} from './ck_run.mjs';
const core = await import(`${REL}/dist/sim-core.js`);
const RT = await import(`${EXP}/r3/line/rt/mode_runtime.mjs`);
const weights = JSON.parse(fs.readFileSync(`${CAND}/policy.json`, 'utf8'));
const ev = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3/eval_test_r3.json', 'utf8'));
const rows = Object.values(ev.tests).flat().filter(r => ['P37', 'H48', 'P0'].includes(r.id));
const out = {};
for (const t of rows) for (const sh of [true, false]) {
  const {world, variation} = spawnFor(BASE_WORLD, t.start.gate, t.start.left_offset_m, t.start.left_yaw_deg);
  const sim = core.createSimulation(world, {...PHYS, seed: t.start.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed: t.start.seed});
  globalThis.JF_R3 = sh ? {modeShield: {line: JSON.parse(JSON.stringify(SHIELD_PARAMS))}} : undefined;
  const T = RT.createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: 'line', weights, params: {line: t.line, c_v: t.c_v}});
  globalThis.JF_R3 = undefined;
  const ms = [];
  while (!snap.state.terminal) { const a = performance.now(); const o = T.action(snap); ms.push(performance.now() - a); snap = sim.step(o); }
  ms.sort((a, b) => a - b);
  out[`${t.id}_${sh ? 'shield' : 'noshield'}`] = {mean: ms.reduce((a, b) => a + b, 0) / ms.length, p50: ms[ms.length >> 1], p95: ms[Math.floor(ms.length * .95)], max: ms.at(-1)};
}
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/timing.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
