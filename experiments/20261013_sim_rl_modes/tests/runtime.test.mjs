// mode_runtime.mjs(Node の試験とブラウザの教師で同じファイル)の約束
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {fastestWeights, widenWeights, EXP, sha256File} from './helpers.mjs';
import {createModeTeacher, sceneParams, MODE_KEYS, TEACHER_SCHEMA} from '../rl/runtime/mode_runtime.mjs';
import {createObsWrapper} from '../rl/runtime/obs_modes.mjs';
import {createFeatureTracker, OBS_DIM} from '../rl/runtime/features.mjs';
import {runModeEpisode, BASE_WORLD} from '../rl/episode.mjs';
import {activeWorldFor, simConfig} from '../snapshot/rl/env_core.mjs';
import {worldWithObstacles} from '../rl/runtime/layouts.mjs';
import * as original from '../snapshot/sim/dist/sim-core.js';

const W = fastestWeights();
const weightsFor = key => key === 'line' || key === 'speed' ? widenWeights(W, 97, 'cond97', 'cond') : key === 'obstacle' ? widenWeights(W, 95, 'obst95', 'obstacle') : W;
test('最速: 元のコアで停止から 1 周 8.091953367107205 秒(ベストラップの値と 1e-6 以内)', async () => {
  const r = await runModeEpisode({mode: 'fastest', weights: W, start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, episodeMode: 'single'});
  assert.ok(Math.abs(r.lap_from_stop - 8.091953367107205) <= 1e-6, String(r.lap_from_stop));
});
test('runtime と features.mjs: base79 の観測は追跡器の 79 と同じ。base79v はゲートを外すまで同じ', () => {
  const {world} = activeWorldFor(BASE_WORLD, 5, .1, 5), cfg = simConfig({seed: 3});
  const sim = original.createSimulation(world, cfg); let snap = sim.reset({seed: 3});
  const tr = createFeatureTracker(world, snap.config), a = createObsWrapper({wrapper: 'base79', world, config: snap.config}), b = createObsWrapper({wrapper: 'base79v', world, config: snap.config});
  const o0 = new Float64Array(79), o1 = new Float64Array(79), o2 = new Float64Array(79);
  for (let f = 0; f < 200; f++) {
    const st = snap.state, ri = snap.metrics.route_index;
    if (f === 0) { tr.resetProgress(st.x, st.z, ri); a.reset(st, ri); b.reset(st, ri); } else { tr.advanceProgress(st.x, st.z); a.advance(st, ri); b.advance(st, ri); }
    tr.observe(st, ri, .1, -.2, o0); a.observe(st, .1, -.2, o1); b.observe(st, .1, -.2, o2);
    assert.deepEqual(Array.from(o1), Array.from(o0));
    if (!b.missed) assert.deepEqual(Array.from(o2), Array.from(o0));
    snap = sim.step(sim.demoAction());
  }
});
test('同じフレームで 2 回呼ばれても同じ答え・フレーム 0 に p・禁止の鍵が無い・metadata はシーンで変わらない', () => {
  for (const key of MODE_KEYS) {
    const {world: active, variation} = activeWorldFor(BASE_WORLD, 0, 0, 0);
    const world = key === 'obstacle' ? worldWithObstacles(active, [{shape: 'post', x: 1.2, z: 2.1, r: .06, h: .3}]) : active;
    const metas = [];
    for (const seed of [11, 12]) {
      const cfg = simConfig({seed}), sim = original.createSimulation(world, cfg);
      let snap = sim.reset({seed});
      const t = createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: key, weights: weightsFor(key), policySha256: 'x'.repeat(64)});
      metas.push(JSON.stringify(t.metadata));
      for (let f = 0; f < 40; f++) {
        const a = t.action(snap), b = t.action(snap);
        assert.deepEqual(a, b);
        const jf = a.info.jf;
        assert.equal(jf.m, key); assert.equal(jf.f, f); assert.ok(jf.l === 0 || jf.l === 1);
        if (f === 0) assert.ok(jf.p && jf.p.seed === seed); else assert.equal(jf.p, undefined);
        const txt = JSON.stringify(a.info);
        assert.ok(!/steering_pwm|throttle_pwm/.test(txt) && !/"\//.test(txt), txt);
        assert.ok(txt.length < 400, key + ' info が大きい ' + txt.length);
        snap = sim.step(a);
      }
      assert.equal(t.metadata.schema, TEACHER_SCHEMA);
    }
    assert.equal(metas[0], metas[1], key + ' の metadata がシーンで違う');
  }
});
test('シーンの条件: 種から決定的、指定があれば指定のまま', () => {
  assert.deepEqual(sceneParams('line', 5), sceneParams('line', 5));
  assert.notDeepEqual(sceneParams('line', 5), sceneParams('line', 6));
  assert.equal(sceneParams('speed', 9, {c_v: .625}).c_v, .625);
  assert.deepEqual(sceneParams('line', 1, {line: {family: 'inside', a: .7}}).line, {family: 'inside', a: .7});
});
test('ブラウザに配るファイルは features.mjs・reference_loop・policy_forward がベストラップとバイト単位で同じ', () => {
  for (const f of ['features.mjs', 'reference_loop.mjs', 'reference_loop.json', 'policy_forward.mjs'])
    assert.equal(sha256File(`${EXP}/rl/runtime/${f}`), sha256File(`${EXP}/snapshot/rl/${f}`));
  for (const f of fs.readdirSync(`${EXP}/rl/runtime`).filter(f => f.endsWith('.mjs'))) {
    const src = fs.readFileSync(`${EXP}/rl/runtime/${f}`, 'utf8').replace(/\/\/.*$/gm, '');   // 注釈は除く
    assert.ok(!/from 'node:|require\(|Math\.random|Date\.now|performance\.now/.test(src), f + ' が純粋な ESM ではない');
  }
});
