// 単周の initialState から始めた物理が、連続周回の同じ状態から同じ操作で進めた物理と同じかを確かめる(設計の段の下調べ)
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const MODES = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const {activeWorldFor, simConfig} = await import(pathToFileURL(MODES + '/snapshot/rl/env_core.mjs').href);
const core = await import(pathToFileURL(MODES + '/snapshot/sim/dist/sim-core.js').href);
const M = await import('/home/jetson/jetformer/experiments/20261014_sim_dart_margin/design/measure_teachers.mjs');
const BASE = JSON.parse(fs.readFileSync(MODES + '/snapshot/sim/world-v3.json', 'utf8'));
const r = M.runEpisode({teacher: 'fastest', start: [0, .01, 1], episodeMode: 'continuous', timeLimit: 20, seed: 3, noise: M.makeNoise('burst_m_gg', 1)});
const {world} = activeWorldFor(BASE, 0, .01, 1);
let worst = 0, worstYaw = 0, n = 0;
const acts = r.rows.map(q => ({steering: q.steer, speed: q.speed}));
// 連続周回の状態を取り直す(state の全部の欄)
const sim = core.createSimulation(world, simConfig({episodeMode: 'continuous', timeLimit: 20, seed: 3}));
let snap = sim.reset({seed: 3, scenario: 'baseline'});
const states = [JSON.parse(JSON.stringify({state: snap.state, metrics: {route_index: snap.metrics.route_index}}))];
for (const a of acts) { snap = sim.step(a); states.push(JSON.parse(JSON.stringify({state: snap.state, metrics: {route_index: snap.metrics.route_index}}))); if (snap.state.terminal) break; }
console.log('state keys', Object.keys(states[10].state).join(','));
for (const t of [5, 50, 120, 200, 333, 450, 520]) { let w = 0, wk = -1;
  const s0 = states[t].state;
  const sim2 = core.createSimulation(world, simConfig({episodeMode: 'single', timeLimit: 60, seed: 3}));
  let sn = sim2.reset({seed: 3, scenario: 'baseline', initialState: {x: s0.x, z: s0.z, yaw: s0.yaw, speed: s0.speed, steering: s0.steering, routeIndex: states[t].metrics.route_index % 33}});
  for (let k = 0; k < 60 && t + k < acts.length; k++) {
    sn = sim2.step(acts[t + k]);
    const ref = states[t + k + 1]?.state; if (!ref) break;
    const d = Math.hypot(sn.state.x - ref.x, sn.state.z - ref.z), dy = Math.abs(sn.state.yaw - ref.yaw);
    worst = Math.max(worst, d); worstYaw = Math.max(worstYaw, dy); n++; if (d > w) { w = d; wk = k; }
    if (sn.state.terminal || k === 59) { if (!sn.state.terminal) console.log('  t', t, 'max', w, 'at k', wk, 'ri', states[t].metrics.route_index); console.log("  t", t, "w", w, "at k", wk, "route_index", states[t].metrics.route_index); console.log('  t', t, 'single terminal at k', k, sn.state.reason); break; }
  }
}
console.log('compared', n, 'steps; max pos diff', worst, 'max yaw diff', worstYaw);
