// 本当のゲートの外しの数え方(rl/gate_miss.mjs、2026-10-14 recover r2)
import test from 'node:test';
import assert from 'node:assert/strict';
import {realGateMisses, MAX_GATE_GAP_M, GATE_MISS_THRESHOLD_M} from '../rl/gate_miss.mjs';

test('ゲートの間の最大と、しきい値', () => {
  assert.ok(Math.abs(MAX_GATE_GAP_M - 1.5186) < 1e-3);
  assert.ok(Math.abs(GATE_MISS_THRESHOLD_M - (MAX_GATE_GAP_M + .5)) < 1e-12);
});

test('ふつうの走り(1.5 m ごとに数えが増える)は外しにならない', () => {
  const s = Array.from({length: 600}, (_, f) => .05 * f), g = s.map(x => String(Math.floor(x / 1.5)));
  const r = realGateMisses(s, g);
  assert.equal(r.misses, 0);
  assert.ok(r.stuck_max_m < 1.51);
});

test('数えが止まったまま 2.02 m を越えたら 1 回(止まりの間は 1 回だけ)', () => {
  const s = Array.from({length: 600}, (_, f) => .05 * f);
  const g = s.map(x => x < 10 ? String(Math.floor(x / 1.5)) : x < 20 ? 'stuck' : String(100 + Math.floor(x / 1.5)));
  const r = realGateMisses(s, g);
  assert.equal(r.misses, 1);
  assert.ok(r.stuck_max_m > 9.9 && r.stuck_max_m < 10.6);
  assert.equal(r.first_frame, r.frames[0]);
});

test('学習の段 r1 の数え方(laps の数)では見えない外しを数える', () => {
  // sim は外したゲートで止まり、1 周(26 m)後に同じゲートを通ると数えが進む
  const s = Array.from({length: 600}, (_, f) => .06 * f);
  const g = s.map(x => x < 5 ? String(Math.floor(x / 1.5)) : x < 31 ? '3' : String(Math.floor((x - 26) / 1.5)));
  assert.equal(realGateMisses(s, g).misses, 1);
});
