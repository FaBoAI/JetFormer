// no_wall r4: 1 回をフレームごとに見る(標準の物理・遅延なし)。node diag_nw4.mjs gate off yaw [from to] [rt=v5|v4] [shield=1]
import fs from 'node:fs';
import {BASE_WORLD, activeWorld, NOMINAL, startClearance} from './run_nw4r.mjs';
import {buildObstacles, ownClearance} from './own_geom.mjs';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const core = await import(`${REL}/dist/sim-core.js`);
const a = process.argv.slice(2);
const [g, o, y] = a.slice(0, 3).map(Number), from = +(a[3] ?? 0), to = +(a[4] ?? 60), rtn = a[5] ?? 'v5', sh = (a[6] ?? '1') === '1';
const pol = process.env.POLICY ?? `${EXP}/r3/candidates/no_wall_r3/policy.json`;
const RT = await import(rtn === 'v4' ? `${EXP}/r3/nw/runtime_v4/mode_runtime.mjs` : `${EXP}/r3/nw4/runtime_v5/mode_runtime.mjs`);
const {world, variation} = activeWorld(BASE_WORLD, g, o, y);
const obs = buildObstacles(world);
const cfg = {...NOMINAL, seed: 1, timeLimit: +(process.env.T ?? 20), episodeMode: 'continuous-laps-v1'};
const sim = core.createSimulation(world, cfg);
let snap = sim.reset({seed: 1, scenario: 'baseline'});
const w = JSON.parse(fs.readFileSync(pol, 'utf8'));
const sp = process.env.SP ? JSON.parse(process.env.SP) : undefined;
const t = RT.createModeTeacher(world, snap.config, variation, {mode: 'no_wall', weights: w, ...(sh ? (sp ? {shieldParams: sp} : {}) : {overrides: {shield: false}})});
let minC = 9, minF = -1, back = 0, shN = 0, ms = 0;
console.log('start clearance', (100 * startClearance(g, o, y)).toFixed(2));
while (!snap.state.terminal) {
  const st = snap.state, f = st.frame;
  const t0 = process.hrtime.bigint();
  const out = t.action(snap), lg = t.log.at(-1);
  ms += Number(process.hrtime.bigint() - t0) / 1e6;
  const c = ownClearance(obs, st.x, st.z, st.yaw);
  if (c < minC) { minC = c; minF = f; }
  if (st.speed < -0.01) back++;
  if (out.info.jf.sh) shN++;
  if (f >= from && f <= to) console.log(f, `c=${(100 * c).toFixed(2)}`, `v=${st.speed.toFixed(2)}`, `δ=${st.steering.toFixed(3)}`, `s=${lg.s.toFixed(2)}`, `cmd=(${out.steering.toFixed(3)},${out.speed.toFixed(2)})`,
    `sh=${out.info.jf.sh}`, JSON.stringify(lg.shield ? {fb: lg.shield.fb, nom: lg.shield.nom?.toFixed?.(4), wq: lg.shield.wq?.toFixed?.(4), pess: lg.shield.pess?.toFixed?.(4), n: lg.shield.n} : null));
  snap = sim.step({steering: out.steering, speed: out.speed});
}
const m = sim.report().metrics;
console.log('min clearance', (100 * minC).toFixed(3), 'at', minF, 'reason', snap.state.reason, 'laps', m.completed_laps, 'collisions', m.collisions, 'back frames', back, 'sh', shN, 'ms/frame', (ms / snap.state.frame).toFixed(2));
