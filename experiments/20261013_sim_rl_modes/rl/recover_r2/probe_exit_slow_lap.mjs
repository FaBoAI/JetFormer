// 黒シートの出口を遅く通ると、停止から 1 周(C6 の RC-STD)がどれだけのびるか(2026-10-14、学習の段 recover r2。目安)。
//   選んだ離脱復活の方策(乱しなし)を STD の開始・単周で元の sim-core.js で走らせ、R 上の s が [a, 8.3) の間だけ速さの指令に上限 v_cap をかける。
//   方策はこの上限で学習していないので、学習した方策ならもっと短くできるかもしれない(上限の目安ではなく、だいたいの大きさ)。
//   node rl/recover_r2/probe_exit_slow_lap.mjs <policy.json> <出力.json>
import fs from 'node:fs';
import crypto from 'node:crypto';
import {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS} from '../../snapshot/rl/env_core.mjs';
import {BASE_WORLD, loadCore, readJson} from '../episode.mjs';
import {createModeTeacher} from '../runtime/mode_runtime.mjs';
import {LOOP_LENGTH} from '../runtime/features.mjs';
import {perturbAllowed, RECORD_SPEC} from '../runtime/perturb.mjs';

const [polPath, outPath] = process.argv.slice(2);
const weights = readJson(polPath), sha = crypto.createHash('sha256').update(fs.readFileSync(polPath)).digest('hex');
const core = await loadCore('original');
// zoneCap: 乱しを始めてよい位置(記録の決まり)すべてで速さの指令に上限をかける(出口の上限 vcap と別)
async function lap(vcap, a, zoneCap = null) {
  const {world, variation} = activeWorldFor(BASE_WORLD, 0, 0, 0);
  const sim = core.createSimulation(world, simConfig({episodeMode: 'single', timeLimit: 20, seed: 1}));
  let snap = sim.reset({seed: 1, scenario: 'baseline'});
  const cfgN = {...JSON.parse(JSON.stringify(snap.config)), ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))};
  const t = createModeTeacher(world, cfgN, variation, {mode: 'recover', weights, policySha256: sha, overrides: {perturbation: false}});
  const atExit = [];
  while (!snap.state.terminal) {
    const out = t.action({state: snap.state, metrics: snap.metrics, config: cfgN});
    const s = t.log.at(-1).s, sm = ((s % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH;
    let sp = out.speed;
    if (vcap != null && sm >= a && sm < 8.3) sp = Math.min(sp, vcap);
    if (zoneCap != null && perturbAllowed(s, RECORD_SPEC)) sp = Math.min(sp, zoneCap);
    if (sm >= 8.0 && sm < 8.25) atExit.push(snap.state.speed);
    snap = sim.step({steering: out.steering, speed: sp});
  }
  const m = snap.metrics, ok = snap.state.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33;
  return {v_cap: vcap, from_s: a, zone_cap: zoneCap, lap_complete: ok, lap_s: ok ? m.elapsed : null, reason: snap.state.reason, collisions: m.collisions,
    speed_at_exit_max: atExit.length ? Math.max(...atExit) : null};
}
const out = [];
out.push(await lap(null, 0));
for (const a of [5.0, 6.0, 7.0]) for (const v of [2.0, 1.5, 1.3, 1.0]) out.push(await lap(v, a));
for (const z of [3.0, 2.6, 2.4, 2.0]) out.push(await lap(1.5, 6.0, z));
fs.writeFileSync(outPath, JSON.stringify({schema: 'jf-sim-rl-recover-exit-slow-lap/v1', policy_sha256: sha, runs: out}, null, 1));
for (const r of out) console.log(JSON.stringify(r));
