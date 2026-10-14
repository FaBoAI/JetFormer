#!/usr/bin/env node
// 経路いろいろ r2 の報告用: 組の開始の「R に対する向きのずれ」(最初の観測の heading_error)と開始のすき間を数える(走らせない)。
//   node rl/line_r2/start_heading.mjs [--set test|validation|fresh]
import {buildJobs, loadLineR2} from '../eval_modes.mjs';
import {createObsWrapper} from '../runtime/obs_modes.mjs';
import {activeWorldFor, simConfig} from '../../snapshot/rl/env_core.mjs';
import {BASE_WORLD} from '../episode.mjs';
import * as core from '../../snapshot/sim/dist/sim-core.js';
const set = process.argv.includes('--set') ? process.argv[process.argv.indexOf('--set') + 1] : 'test';
const jobs = buildJobs('line', set, {protocolLineR2: loadLineR2()}).filter(j => j.test === 'LN-P60');
const out = [];
for (const j of jobs) {
  const st = j.spec.start, {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  const sim = core.createSimulation(world, simConfig({episodeMode: 'continuous', seed: st.seed}));
  const snap = sim.reset({seed: st.seed});
  const w = createObsWrapper({wrapper: 'cond97', world, config: snap.config, lineFactory: null});
  w.reset(snap.state, snap.metrics.route_index ?? 0);
  const aux = {}; w.tracker.observe(snap.state, w.gate.obsIdx, 0, 0, new Float64Array(79), aux);
  out.push({id: j.id, gate: st.gate, epsi0: aux.heading_error, ey0: aux.lateral_error, s0: aux.s, clear0: st.initial_clearance_m});
}
const big = out.filter(o => Math.abs(o.epsi0) > 1.0);
console.log(JSON.stringify({set, n: out.length, abs_epsi_gt_0p5: out.filter(o => Math.abs(o.epsi0) > .5).length, abs_epsi_gt_1: big.length,
  big: big.map(o => ({id: o.id, gate: o.gate, epsi0: +o.epsi0.toFixed(3), s0: +o.s0.toFixed(2)})), max_abs_epsi: Math.max(...out.map(o => Math.abs(o.epsi0)))}));
