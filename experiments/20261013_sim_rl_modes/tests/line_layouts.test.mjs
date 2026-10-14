// 経路の形・使える幅の表・障害物の配置
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {EXP, readJson} from './helpers.mjs';
import {createLineProfile, tableAt, LINE_FAMILIES} from '../rl/runtime/line_profiles.mjs';
import {ROOM_TABLE_SHA256} from '../rl/runtime/room_table.mjs';
import {buildRoomTable} from '../rl/build_room_table.mjs';
import {prepareGeometry, LOOP_LENGTH, loopPointAt} from '../rl/runtime/features.mjs';
import {geometryWithObstacles, worldWithObstacles, checkPassage, obstaclesFromWorld, generateLayout, buildBaseClearanceGrid} from '../rl/runtime/layouts.mjs';
import {obstacleFeatures} from '../rl/runtime/obs_modes.mjs';
import {createRng} from '../rl/runtime/rng.mjs';

const world = readJson(`${EXP}/snapshot/sim/world-v3.json`);
test('使える幅の表: 作り直すと同じ sha256', () => {
  const {body} = buildRoomTable();
  assert.equal(crypto.createHash('sha256').update(JSON.stringify(body)).digest('hex'), ROOM_TABLE_SHA256);
});
test('経路の形: e* はいつも使える区間 [−ρ_R, ρ_L] の中、c は [−1, 1]、ランダムは種で決まる', () => {
  for (const fam of LINE_FAMILIES) for (const p of [{family: fam, c0: .9, a: .9, seed: 4}, {family: fam, c0: -.9, a: .5, seed: 5}]) {
    const lp = createLineProfile(p, 3.3);
    for (let s = 3.3; s < 3.3 + 2 * LOOP_LENGTH; s += .37) {
      const e = lp.eStar(s), lo = -tableAt('rhoR', s), hi = tableAt('rhoL', s);
      assert.ok(e >= Math.min(lo, hi) - 1e-12 && e <= Math.max(lo, hi) + 1e-12, `${fam} s ${s} e ${e}`);
      assert.ok(Math.abs(lp.c(s)) <= 1);
    }
  }
  const a = createLineProfile({family: 'random', seed: 9}, 1), b = createLineProfile({family: 'random', seed: 9}, 1);
  for (let s = 1; s < 30; s += 1.1) assert.equal(a.eStar(s), b.eStar(s));
  const ins = createLineProfile({family: 'inside', a: .8}), out = createLineProfile({family: 'outside', a: .8});
  let d = 0; for (let s = 0; s < LOOP_LENGTH; s += .1) d += Math.abs(ins.c(s) + out.c(s)); assert.ok(d < 1e-9, 'イン と アウト が逆向きでない');
});
test('障害物: 形は prepareGeometry と同じ、world との往復、細い柱も入力に入る', () => {
  const obs = [{shape: 'post', x: 1.0, z: 2.0, r: .07, h: .3}, {shape: 'box', x: -2, z: 1.5, length: .3, yaw: .7, r: .05, h: .25}];
  const w2 = worldWithObstacles(world, obs);
  const a = geometryWithObstacles(prepareGeometry(world), obs), b = prepareGeometry(w2);
  assert.deepEqual(JSON.parse(JSON.stringify(a)), JSON.parse(JSON.stringify(b)));
  const back = obstaclesFromWorld(w2);
  assert.equal(back.length, 2);
  assert.ok(Math.abs(back[1].length - .3) < 1e-12 && Math.abs(back[0].r - .07) < 1e-12);
  const out = new Float64Array(16);
  obstacleFeatures(obs, .0, 2.0, 0, out, 0);   // 正面 1 m の柱
  assert.ok(Math.abs(out[0] - 1 / 3) < 1e-9 && out[3] === 1);
});
test('通り道の検査器: 道をふさぐ配置を拒否し、ふつうの配置は通す。作った配置の組は全部通る', () => {
  const g = prepareGeometry(world), pt = {x: 0, z: 0, tx: 0, tz: 0};
  loopPointAt(2.0, pt);
  const wall = [];
  for (let o = -1.0; o <= 1.0; o += .15) wall.push({shape: 'post', x: pt.x + o * pt.tz, z: pt.z - o * pt.tx, r: .1, h: .3});
  assert.equal(checkPassage(g, wall).ok, false);
  assert.equal(checkPassage(g, [{shape: 'post', x: pt.x, z: pt.z, r: .05, h: .3}]).ok, true);
  for (const name of ['validation', 'test', 'recording']) {
    const set = readJson(`${EXP}/rl/obstacle_sets/${name}.json`);
    assert.ok(set.layouts.length >= 30);
    for (const l of set.layouts) { assert.ok(l.obstacles.length >= 1 && l.obstacles.length <= 4); assert.equal(l.passage, true); }
  }
  // 同じ種なら同じ配置
  const grid = buildBaseClearanceGrid(g);
  const A = generateLayout(createRng(5), {baseGeometry: g, nominalLine: null, grid}), B = generateLayout(createRng(5), {baseGeometry: g, nominalLine: null, grid});
  assert.deepEqual(A, B);
});
