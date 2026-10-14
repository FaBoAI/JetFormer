// シールドの車の式(sim-core の写し)と、シールドの決定性
import test from 'node:test';
import assert from 'node:assert/strict';
import {fastestWeights} from './helpers.mjs';
import {createCarModel, createShield} from '../rl/runtime/shield.mjs';
import {runModeEpisode, BASE_WORLD} from '../rl/episode.mjs';
import {prepareGeometry} from '../rl/runtime/features.mjs';
import {activeWorldFor, simConfig} from '../snapshot/rl/env_core.mjs';
import * as original from '../snapshot/sim/dist/sim-core.js';

test('車の式: 600 フレームの操作列で sim-core との位置の差 ≤ 1e-6 m', async () => {
  for (const st of [{gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, {gate: 9, left_offset_m: -.1, left_yaw_deg: 8, seed: 2}]) {
    const r = await runModeEpisode({mode: 'fastest', weights: fastestWeights(), start: st, audit: false});
    const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
    const cfg = simConfig({seed: st.seed}), sim = original.createSimulation(world, cfg);
    const snap = sim.reset({seed: st.seed});
    const model = createCarModel(prepareGeometry(world), snap.config);
    const s = {x: snap.state.x, z: snap.state.z, yaw: snap.state.yaw, speed: 0, steering: 0};
    let t = 0, m = 0;
    for (let i = 0; i < r.actions.length; i++) {
      const res = model.stepFrame(s, r.actions[i].steering, r.actions[i].speed, null, null, Math.min(snap.config.dt, snap.config.timeLimit - t));
      t = Math.min(snap.config.timeLimit, t + res.elapsed);
      const q = r.trace[i];
      m = Math.max(m, Math.hypot(s.x - q.x, s.z - q.z), Math.abs(s.speed - q.v));
    }
    assert.equal(r.actions.length, 600);
    assert.ok(m <= 1e-6, '差 ' + m);
  }
});
test('シールド: 同じ入力には同じ答え(乱数・時刻を使わない)、2 回呼んでも同じ', () => {
  const {world} = activeWorldFor(BASE_WORLD, 0, 0, 0);
  const cfg = {...simConfig({}), substep: .01};
  const g = prepareGeometry(world), sh = createShield({geometry: g, config: cfg});
  const st = {x: world.spawn.x + .4, z: world.spawn.z, yaw: world.spawn.yaw, speed: 3.5, steering: .2};
  const a = sh.filter(st, 0.6, {steering: .2, speed: 4.5}, {steering: .52, speed: 4.5});
  const b = sh.filter(st, 0.6, {steering: .2, speed: 4.5}, {steering: .52, speed: 4.5});
  assert.deepEqual(a, b);
  const sh2 = createShield({geometry: g, config: cfg});
  assert.deepEqual(sh2.filter(st, 0.6, {steering: .2, speed: 4.5}, {steering: .52, speed: 4.5}), a);
});
test('シールド: 広い所のふつうの指令はそのまま通す', () => {
  const {world} = activeWorldFor(BASE_WORLD, 0, 0, 0);
  const sh = createShield({geometry: prepareGeometry(world), config: {...simConfig({}), substep: .01}});
  const st = {x: world.spawn.x, z: world.spawn.z, yaw: world.spawn.yaw, speed: 0, steering: 0};
  const r = sh.filter(st, 0, {steering: 0, speed: 0}, {steering: 0, speed: 2.0});
  assert.equal(r.replaced, false);
});
