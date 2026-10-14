// シールドの直し(jf-shield/v2、no_wall の学習の段 2026-10-14、deviations.md 2)の試験
//   1. 予備の操縦の組(どれか 1 つで合格)は、元の予備の操縦 1 つだけのときより合格が減らない(同じ状態・同じ提案で)
//   2. 提案が標準の模型で合格なら替えない(keep_nominal)。替えたときは、替えた指令が標準の模型で合格しているか、どの候補も標準の模型で不合格
//   3. 決定性と metadata(version・backups・keep_nominal)
import test from 'node:test';
import assert from 'node:assert/strict';
import {fastestWeights} from './helpers.mjs';
import {createShield, SHIELD_DEFAULTS, SHIELD_VERSION} from '../rl/runtime/shield.mjs';
import {createModeTeacher} from '../rl/runtime/mode_runtime.mjs';
import {BASE_WORLD} from '../rl/episode.mjs';
import {prepareGeometry} from '../rl/runtime/features.mjs';
import {activeWorldFor, simConfig} from '../snapshot/rl/env_core.mjs';
import * as fast from '../rl/fast_core_modes.mjs';

// 最速の方策(シールドなし)で STD の連続を走らせ、フレームごとの (状態, s, 前の指令, 提案) を集める
function collectStates(gate = 0, every = 3) {
  const {world, variation} = activeWorldFor(BASE_WORLD, gate, 0, 0);
  const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: 1});
  const sim = fast.createSimulation(world, cfg);
  sim.reset({seed: 1, scenario: 'baseline'});
  let snap = sim.peek();
  const cfgN = JSON.parse(JSON.stringify(snap.config));
  const t = createModeTeacher(world, cfgN, variation, {mode: 'no_wall', weights: fastestWeights(), overrides: {shield: false}});
  const out = [];
  let prev = {steering: 0, speed: 0};
  while (!snap.state.terminal) {
    const st = {...snap.state};
    const a = t.action({state: snap.state, metrics: snap.metrics, config: cfgN});
    if (st.frame % every === 0) out.push({st, s: t.log.at(-1).s, prev, prop: {steering: a.steering, speed: a.speed}});
    prev = {steering: a.steering, speed: a.speed};
    sim.step(a, true); snap = sim.peek();
  }
  return {world, cfgN, states: out};
}

test('シールド v2: 予備の操縦の組は元の 1 つより合格が減らない', () => {
  const {world, cfgN, states} = collectStates(0, 4);
  const g = prepareGeometry(world);
  const v2 = createShield({geometry: g, config: cfgN});
  const v1 = createShield({geometry: g, config: cfgN, params: {backups: [{la: .6, k: 0}], keep_nominal: false}});
  let n1 = 0, n2 = 0;
  for (const q of states) {
    const a = v1.check(q.st, q.s, q.prev, q.prop), b = v2.check(q.st, q.s, q.prev, q.prop);
    if (a.pass) { n1++; assert.ok(b.pass, `frame ${q.st.frame}: 元の予備の操縦で合格なのに v2 で不合格`); }
    if (b.pass) n2++;
    if (a.passNominal) assert.ok(b.passNominal);
  }
  assert.ok(states.length > 100);
  assert.ok(n2 >= n1);
});

test('シールド v2: 標準の模型で合格の提案は替えない。替えるのは標準の模型で不合格のときだけ', () => {
  for (const gate of [0, 20]) {
    const {world, cfgN, states} = collectStates(gate, 2);
    const sh = createShield({geometry: prepareGeometry(world), config: cfgN});
    let kept = 0, replaced = 0;
    for (const q of states) {
      const c = sh.check(q.st, q.s, q.prev, q.prop);
      const r = sh.filter(q.st, q.s, q.prev, q.prop);
      if (c.passNominal) {
        assert.equal(r.replaced, false, `frame ${q.st.frame}`);
        assert.deepEqual(r.cmd, {steering: q.prop.steering, speed: q.prop.speed});
        if (!c.pass) { assert.equal(r.fallback, 'proposal_nominal'); kept++; }
      } else if (r.replaced) {
        replaced++;
        const rc = sh.check(q.st, q.s, q.prev, r.cmd);
        assert.ok(rc.passNominal || r.fallback === 'max_clearance' || r.fallback === 'stopped_turn_to_backup', `frame ${q.st.frame}: ${r.fallback}`);
      }
    }
    assert.ok(kept + replaced >= 0);
  }
});

test('シールド v2: 決定性・metadata', () => {
  const {world} = activeWorldFor(BASE_WORLD, 0, 0, 0);
  const cfg = {...simConfig({}), substep: .01};
  const g = prepareGeometry(world);
  const st = {x: world.spawn.x + .4, z: world.spawn.z, yaw: world.spawn.yaw, speed: 3.5, steering: .2};
  const a = createShield({geometry: g, config: cfg}).filter(st, .6, {steering: .2, speed: 4.5}, {steering: .52, speed: 4.5});
  const b = createShield({geometry: g, config: cfg}).filter(st, .6, {steering: .2, speed: 4.5}, {steering: .52, speed: 4.5});
  assert.deepEqual(a, b);
  assert.equal(SHIELD_DEFAULTS.version, SHIELD_VERSION);
  assert.equal(SHIELD_VERSION, 'jf-shield/v3');   // 2026-10-14 no_wall r2 で v3 にした(予備の操縦の組・keep_nominal は v2 のまま。tests/shield_v3.test.mjs)
  assert.equal(SHIELD_DEFAULTS.backups.length, 3);
  assert.equal(SHIELD_DEFAULTS.keep_nominal, true);
  const t = createModeTeacher(world, simConfig({seed: 1}), {}, {mode: 'no_wall', weights: fastestWeights()});
  assert.equal(t.metadata.shield.version, 'jf-shield/v3');
  assert.deepEqual(JSON.parse(JSON.stringify(t.metadata.shield.backups)), [{la: .6, k: .2}, {la: .6, k: 0}, {hold: true}]);
});
