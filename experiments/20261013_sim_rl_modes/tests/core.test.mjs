// 高速版 + 摩擦円(rl/fast_core_modes.mjs): 摩擦円なしはベストラップの高速版とビット単位で同じ、摩擦円ありは own_physics_r1 と 1e-9 で同じ
import test from 'node:test';
import assert from 'node:assert/strict';
import {EXP, fastestWeights} from './helpers.mjs';
import * as modes from '../rl/fast_core_modes.mjs';
import * as fastBest from '../snapshot/sim/fast-sim-core.mjs';
import * as original from '../snapshot/sim/dist/sim-core.js';
import {createOwnSim} from '../snapshot/check_r1/own_physics_r1.mjs';
import {activeWorldFor, simConfig} from '../snapshot/rl/env_core.mjs';
import {runModeEpisode, BASE_WORLD} from '../rl/episode.mjs';
import {createRng} from '../rl/runtime/rng.mjs';

async function actionsOf(start) {
  const r = await runModeEpisode({mode: 'fastest', weights: fastestWeights(), start, audit: false});
  return r.actions;
}
function runCore(core, world, cfg, actions, lite) {
  const sim = core.createSimulation(world, cfg); sim.reset({seed: cfg.seed});
  for (const a of actions) { if (lite) sim.step(a, true); else sim.step(a); }
  return sim.report();
}
test('摩擦円なし: ベストラップの高速版・元のコアとビット単位で同じ', async () => {
  for (const st of [{gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, {gate: 17, left_offset_m: .1, left_yaw_deg: -6, seed: 5}]) {
    const acts = await actionsOf(st);
    const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
    const cfg = simConfig({seed: st.seed});
    const a = runCore(modes, world, cfg, acts, true), b = runCore(fastBest, world, cfg, acts, true), c = runCore(original, world, cfg, acts, false);
    assert.deepEqual(a.trace, b.trace);
    assert.deepEqual(a.trace, c.trace);
    assert.deepEqual(a.metrics, c.metrics);
  }
});
test('摩擦円あり: own_physics_r1 の frictionCircle と 1e-9 で同じ(乱暴な操作も含む)', async () => {
  const acts = await actionsOf({gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1});
  const rng = createRng(3);
  const rough = Array.from({length: 600}, (_, i) => ({steering: (rng.uniform() < .5 ? -1 : 1) * .52 * rng.uniform(), speed: i % 40 < 20 ? 4.5 : 0}));
  for (const A of [acts, rough]) {
    const {world} = activeWorldFor(BASE_WORLD, 0, 0, 0);
    const cfg = simConfig({seed: 1});
    const sim = modes.createSimulation(world, {...cfg, frictionCircle: true}); sim.reset({seed: 1});
    for (const a of A) { sim.step(a, true); if (sim.peek().state.terminal) break; }
    const rep = sim.report();
    const own = createOwnSim(world, rep.config, {frictionCircle: true});
    for (const a of rep.actions) own.step(a);
    const tr = own.result().trace;
    assert.equal(tr.length, rep.trace.length);
    let m = 0;
    for (let i = 0; i < tr.length; i++) for (const k of ['x', 'z', 'yaw', 'speed', 'steering']) m = Math.max(m, Math.abs(tr[i][k] - rep.trace[i][k]));
    assert.ok(m <= 1e-9, '差 ' + m);
    assert.ok(sim.peek().friction_circle_limited > 0, '摩擦円が効いていない');
  }
});
