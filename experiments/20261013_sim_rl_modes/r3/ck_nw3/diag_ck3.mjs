// 1 回をフレームごとに見る(標準の物理・遅延なし): シールドの替え方・予想・自前のすき間
// node diag_ck3.mjs gate off yaw [from to] [shield=1]
import {core, BASE_WORLD, startWorld, shapes, clearanceAt} from './own_ck3.mjs';
import {NOMINAL, SHIELD_PARAMS} from './run_ck3.mjs';
import fs from 'node:fs';
const C = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/no_wall_r3';
const RT = await import(`${C}/runtime/mode_runtime.mjs`);
const [g, o, y, a = 0, b = 60, sh = '1'] = process.argv.slice(2).map(Number);
const {world, variation} = startWorld(BASE_WORLD, g, o, y);
const S = shapes(world);
const cfg = {...NOMINAL, seed: 1, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
const sim = core.createSimulation(world, cfg);
let snap = sim.reset({seed: 1, scenario: 'baseline'});
const w = JSON.parse(fs.readFileSync(`${C}/policy.json`, 'utf8'));
const t = RT.createModeTeacher(world, snap.config, variation, {mode: 'no_wall', weights: w, ...(sh ? {shieldParams: SHIELD_PARAMS} : {overrides: {shield: false}})});
let minC = 9, minF = -1;
while (!snap.state.terminal) {
  const st = snap.state, f = st.frame;
  const out = t.action(snap), lg = t.log.at(-1);
  const c = clearanceAt(S, st.x, st.z, st.yaw);
  if (c < minC) { minC = c; minF = f; }
  if (f >= a && f <= b) console.log(f, `c=${(100 * c).toFixed(2)}`, `v=${st.speed.toFixed(2)}`, `δ=${st.steering.toFixed(3)}`, `prop?`, `cmd=(${out.steering.toFixed(3)},${out.speed.toFixed(2)})`,
    `sh=${out.info.jf.sh}`, JSON.stringify(lg.shield ? {fb: lg.shield.fb, nom: lg.shield.nom?.toFixed?.(4), wq: lg.shield.wq?.toFixed?.(4), pess: lg.shield.pess?.toFixed?.(4), n: lg.shield.n} : null));
  snap = sim.step({steering: out.steering, speed: out.speed});
}
console.log('min frame-start clearance', (100 * minC).toFixed(3), 'at', minF, 'reason', snap.state.reason, 'laps', snap.metrics.completed_laps);
