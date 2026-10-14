// 離脱復活の学習の段(r1): protocol acceptance.recover.reported「舵のパルス 12 フレームの生き残り(probe と同じ条件)」。2026-10-13
// probe/kicks.mjs の A と同じ条件: 標準の開始・連続 20 秒、1.0〜19.0 秒の 0.5 秒おき 37 時刻 × 左右、舵を端(±0.52 rad)に N フレーム固定(速さは方策のまま)。
// 生き残り = 20 秒を衝突なしで走り切る(ゲートの外しも数える。probe の「生き残り」は外しなしも条件)。元の sim-core.js。
//   node rl/recover_r1/probe_pulse12.mjs --mode recover|fastest --policy <policy.json> [--frames 12] --out <結果.json>
import fs from 'node:fs';
import {readJson, BASE_WORLD, loadCore} from '../episode.mjs';
import {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS} from '../../snapshot/rl/env_core.mjs';
import {createModeTeacher} from '../runtime/mode_runtime.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = readJson(args.policy), mode = args.mode ?? 'recover', N = +(args.frames ?? 12);
const core = await loadCore(args.core ?? 'original');
const rows = [];
for (let k = 0; k < 37; k++) for (const sign of [1, -1]) {
  const t = 1 + .5 * k, f0 = Math.round(t * 30);
  const {world, variation} = activeWorldFor(BASE_WORLD, 0, 0, 0);
  const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: 1});
  const sim = core.createSimulation(world, cfg);
  let snap = sim.reset({seed: 1, scenario: 'baseline'});
  const cfgN = {...JSON.parse(JSON.stringify(snap.config)), ...Object.fromEntries(PERTURB_KEYS.map(k2 => [k2, PHYSICS[k2]]))};
  const teacher = createModeTeacher(world, cfgN, variation, {mode, weights, overrides: {perturbation: false}});
  let lastRi = 0, stuck = 0;
  while (!snap.state.terminal) {
    const st = snap.state, out = teacher.action({state: st, metrics: snap.metrics, config: cfgN});
    const cmd = st.frame >= f0 && st.frame < f0 + N ? {steering: sign * .52, speed: out.speed} : {steering: out.steering, speed: out.speed};
    snap = sim.step(cmd);
  }
  const m = snap.metrics, laps = (m.lap_events ?? []);
  const gateMiss = mode === 'recover' ? teacher.wrapper.misses > 0 : laps.some(l => l.passed_gates !== 33);
  const noColl = snap.state.reason === 'collection_duration_complete' && m.collisions === 0;
  rows.push({t, sign, reason: snap.state.reason, frames: snap.state.frame, collisions: m.collisions, gate_miss: gateMiss, survived: noColl,
    survived_no_miss: noColl && !gateMiss, laps: laps.map(l => l.elapsed)});
}
const sum = k => rows.filter(r => r[k]).length;
const res = {mode, policy: args.policy, frames: N, runs: rows.length, survived: sum('survived'), survived_no_miss: sum('survived_no_miss'),
  left: rows.filter(r => r.sign > 0 && r.survived).length, right: rows.filter(r => r.sign < 0 && r.survived).length, rows};
console.log(`${mode} 舵のパルス ${N} フレーム: 衝突なし ${res.survived}/74(左 ${res.left}/37・右 ${res.right}/37)、外しもなし ${res.survived_no_miss}/74`);
fs.writeFileSync(args.out, JSON.stringify(res));
