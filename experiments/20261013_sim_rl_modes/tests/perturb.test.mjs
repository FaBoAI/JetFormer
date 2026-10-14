// 乱しの予定と学習除外の決まり
import test from 'node:test';
import assert from 'node:assert/strict';
import {createPerturber, perturbAllowed, maskFromLearn, RECORD_SPEC, TRAIN_SPEC} from '../rl/runtime/perturb.mjs';

const ctxAt = f => ({s: 2 + f * .12, ey: .02, dmin: .3, x: 0, z: 0, yaw: 0, speedCap: 4.5, policy: {steering: .1, speed: 3}});
test('同じ種なら同じ乱し、違う種なら違う。始める位置は決まりのとおり', () => {
  const run = seed => { const p = createPerturber({seed}); const out = []; for (let f = 0; f < 600; f++) out.push(JSON.stringify(p.step(f, ctxAt(f)))); return {out, ev: p.events}; };
  const a = run(7), b = run(7), c = run(8);
  assert.deepEqual(a.out, b.out);
  assert.notDeepEqual(a.out, c.out);
  for (const e of a.ev) {
    assert.ok(perturbAllowed(e.s, RECORD_SPEC), '許されない位置で始まった ' + e.s);
    assert.ok(e.ps / 30 <= RECORD_SPEC.no_start_after_s + 1e-9);
    assert.ok(['steer_pulse', 'excursion', 'speed_pulse'].includes(e.type));
  }
  assert.ok(a.ev.length >= 3, '出来事が少ない ' + a.ev.length);
});
test('始めてよい位置: 黒シート・奥の直線の区間では始めない', () => {
  for (let s = 5.9; s <= 8.0; s += .1) assert.equal(perturbAllowed(s, RECORD_SPEC), false);
  for (let s = 14.6; s <= 19.3; s += .1) assert.equal(perturbAllowed(s, RECORD_SPEC), false);
  for (let s = 5.9; s <= 8.0; s += .1) assert.equal(perturbAllowed(s, TRAIN_SPEC), false);
});
test('学習から外す決まり: 乱しの区間 [ps, pe) なら k ∈ [ps − 10, pe − 2]', () => {
  const learn = new Array(600).fill(1);
  for (let j = 100; j < 110; j++) learn[j] = 0;   // ps 100, pe 110
  const m = maskFromLearn(learn);
  for (let k = 0; k < 600; k++) assert.equal(m[k], k >= 90 && k <= 108 ? 1 : 0, 'k ' + k);
});
test('学習の乱しは状態のキックを含み、強さの倍率が効く', () => {
  const p = createPerturber({spec: TRAIN_SPEC, seed: 3, scale: .5}), q = createPerturber({spec: TRAIN_SPEC, seed: 3, scale: 1.2});
  for (let f = 0; f < 600; f++) { p.step(f, ctxAt(f)); q.step(f, ctxAt(f)); }
  const types = new Set([...p.events, ...q.events].map(e => e.type));
  assert.ok(types.size >= 2);
  const pe = p.events.find(e => e.type === 'steer_pulse'), qe = q.events.find(e => e.type === 'steer_pulse' && e.n === pe?.n);
  if (pe && qe) assert.ok(pe.frames <= qe.frames);
});
