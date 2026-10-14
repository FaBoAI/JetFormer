// シールド v3(jf-shield/v3、no_wall の作り直し r2 2026-10-14、deviations.md)の試験
//   1. 走りながらの見積もり: 元の sim-core.js をいろいろな物理(舵の速さ・遅延 0/1/2・舵のゲイン・ブレーキ)で動かし、シールドの見積もりが本当の値になる
//   2. 速さの見張り: 遅い舵 + 遅延なら上限 = v_max·T_標準/T(ブレーキが弱ければ ·√(b/b_標準))、標準の物理なら上限なし(替えない)
//   3. 決定性: 同じ snapshot の列に同じ答え。フレームが飛んでも例外にならず、前の指令を使う(v2 と同じ動き)
//   4. 標準の物理では v2 と同じ判定(見積もった模型 = 標準の模型)。no_wall の方策で STD を走らせ、替えたフレーム 0・操作はシールドなしと同じ
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {EXP, readJson} from './helpers.mjs';
import {createShield, SHIELD_DEFAULTS, SHIELD_VERSION} from '../rl/runtime/shield.mjs';
import {createModeTeacher} from '../rl/runtime/mode_runtime.mjs';
import {BASE_WORLD} from '../rl/episode.mjs';
import {prepareGeometry} from '../rl/runtime/features.mjs';
import {activeWorldFor, simConfig} from '../snapshot/rl/env_core.mjs';

const core = await import(path.join(EXP, 'snapshot/sim/dist/sim-core.js'));

// 決めた操作の列(舵を大きく振る・止める・小さく動かす)を、シールドを通して元のコアで走らせる。遅延・ゲインはテストの物理(sim に渡す前)
function drive({steerRate = 1.2, brake = 1.2, delay = 0, gain = 1, frames = 75, gate = 0}) {
  const {world} = activeWorldFor(BASE_WORLD, gate, 0, 0);
  const cfgN = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: 3});
  const sim = core.createSimulation(world, {...cfgN, steerRate, brakeDeceleration: brake});
  let snap = sim.reset({seed: 3, scenario: 'baseline'});
  const sh = createShield({geometry: prepareGeometry(world), config: JSON.parse(JSON.stringify(cfgN))});
  const q = [], outs = [];
  let prev = {steering: 0, speed: 0};
  for (let f = 0; f < frames && !snap.state.terminal; f++) {
    const u = f < 8 ? .45 : f < 20 ? -.4 : f < 30 ? .1 * Math.sin(f) : f < 45 ? .3 : .05 * Math.cos(f / 3);
    const sp = f < 30 ? 1.2 : f < 50 ? 0 : 1.0;
    const r = sh.filter(snap.state, 0, prev, {steering: u, speed: sp});
    outs.push(JSON.parse(JSON.stringify({cmd: r.cmd, replaced: r.replaced, fallback: r.fallback ?? null, ident: r.ident})));
    prev = r.cmd;
    let ap = r.cmd;
    if (delay) { q.push(ap); ap = q.length > delay ? q.shift() : {steering: 0, speed: 0}; }
    snap = sim.step({steering: ap.steering * gain, speed: ap.speed});
  }
  return {sh, outs, idn: sh.identified()};
}

test('シールド v3: 版・既定の値(悲観の模型 = DR-T の悪い端)', () => {
  assert.equal(SHIELD_VERSION, 'jf-shield/v3');
  assert.equal(SHIELD_DEFAULTS.version, 'jf-shield/v3');
  assert.deepEqual({...SHIELD_DEFAULTS.pessimistic}, {steerRate: .8, delay: 1, brakeDeceleration: .7, frictionMultiplier: .85});
  assert.equal(SHIELD_DEFAULTS.identify.enabled, true);
  assert.equal(SHIELD_DEFAULTS.governor.enabled, true);
  assert.equal(SHIELD_DEFAULTS.governor.brake, true);
  assert.equal(SHIELD_DEFAULTS.keep_nominal, true);
  assert.equal(SHIELD_DEFAULTS.c_nominal, .07);
  assert.equal(SHIELD_DEFAULTS.c_pessimistic, .05);
  assert.equal(SHIELD_DEFAULTS.H, 24);
});

test('シールド v3: 走りながらの見積もりが本当の舵の速さ・遅延・ゲイン・ブレーキになる', () => {
  for (const c of [{steerRate: 1.2, delay: 0, gain: 1}, {steerRate: .8, delay: 1, gain: .94, brake: .7}, {steerRate: 1.5, delay: 0, gain: 1.06, brake: 2.0},
    {steerRate: 1.0, delay: 2, gain: 1}, {steerRate: .85, delay: 1, gain: 1.03, brake: 1.6}]) {
    const {idn, sh} = drive(c);
    const tag = JSON.stringify(c);
    assert.equal(idn.known, true, tag);
    assert.ok(Math.abs(idn.m.steerRate - c.steerRate) < 1e-6, `${tag}: 舵の速さ ${idn.m.steerRate}`);
    assert.equal(idn.delay, c.delay, `${tag}: 遅延 ${idn.delay} 仮説 ${JSON.stringify(sh.identState.hyps.map(h => [h.d, h.ok, h.why]))}`);
    assert.ok(Math.abs(idn.m.steerGain - c.gain) < 1e-4, `${tag}: ゲイン ${idn.m.steerGain}`);
    if (c.brake) assert.ok(Math.abs(idn.m.brakeDeceleration - c.brake) < 1e-6, `${tag}: ブレーキ ${idn.m.brakeDeceleration}`);
    assert.equal(sh.identState.broken, false, tag);
  }
});

test('シールド v3: 速さの見張りは遅い舵 + 遅延で上限をかけ、標準の物理ではかけない', () => {
  const slow = drive({steerRate: .8, delay: 1, frames: 40});
  const last = slow.outs.at(-1);
  const cap = 4.5 * (.35 / 1.2) / (1 / 30 + .35 / .8);
  // 見積もりのあとのフレームで速さ 1.2 → 上限(約 2.79)より小さいので替えない。上限を超える提案だけ替える
  const st = {...slow.sh.identState};
  assert.ok(st.rMax > .79 && st.rMax < .81);
  const {world} = activeWorldFor(BASE_WORLD, 0, 0, 0);
  const cfgN = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: 3});
  const sim = core.createSimulation(world, {...cfgN, steerRate: .8});
  let snap = sim.reset({seed: 3, scenario: 'baseline'});
  const sh = createShield({geometry: prepareGeometry(world), config: JSON.parse(JSON.stringify(cfgN))});
  const q = [];
  let prev = {steering: 0, speed: 0}, capped = 0, seen = null;
  for (let f = 0; f < 20; f++) {
    const r = sh.filter(snap.state, 0, prev, {steering: f < 6 ? .4 : 0, speed: 4.5});
    if (f >= 4 && r.cap !== undefined) { seen = r.cap; if (r.cmd.speed <= r.cap + 1e-12) capped++; assert.ok(r.replaced); assert.match(r.fallback, /governor/); }
    prev = r.cmd; q.push(r.cmd); const ap = q.length > 1 ? q.shift() : {steering: 0, speed: 0};
    snap = sim.step(ap);
  }
  assert.ok(seen !== null && Math.abs(seen - cap) < 1e-6, `上限 ${seen} / ${cap}`);
  assert.ok(capped >= 10);
  assert.ok(last.ident.d === 1);
  // 標準の物理: 上限なし
  const nom = drive({steerRate: 1.2, delay: 0, frames: 40});
  assert.ok(nom.outs.every(o => !(o.fallback ?? '').includes('governor')));
});

test('シールド v3: 決定性(同じ列に同じ答え)・フレームが飛んでも動く', () => {
  const a = drive({steerRate: .9, delay: 1, gain: .97, frames: 50}), b = drive({steerRate: .9, delay: 1, gain: .97, frames: 50});
  assert.deepEqual(a.outs, b.outs);
  const {world} = activeWorldFor(BASE_WORLD, 0, 0, 0);
  const cfg = {...simConfig({}), substep: .01};
  const g = prepareGeometry(world);
  const sh = createShield({geometry: g, config: cfg});
  for (const f of [0, 2, 5, 9]) {
    const st = {x: world.spawn.x + .1 * f, z: world.spawn.z, yaw: world.spawn.yaw, speed: .3 * f, steering: .05 * f, frame: f};
    const r = sh.filter(st, .1 * f, {steering: .1, speed: 2}, {steering: .2, speed: 2});
    assert.ok(r.cmd && Number.isFinite(r.cmd.steering) && Number.isFinite(r.cmd.speed));
  }
  assert.equal(sh.identified().known, false);
});

test('シールド v3: 標準の物理では見積もった模型 = 標準の模型で、no_wall の方策の STD は替えたフレーム 0(シールドなしと同じ操作)', () => {
  const pol = path.join(EXP, 'bundle/no_wall/policy.json');
  if (!fs.existsSync(pol)) return;
  const weights = readJson(pol);
  const run = (overrides) => {
    const {world, variation} = activeWorldFor(BASE_WORLD, 0, 0, 0);
    const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: 1});
    const sim = core.createSimulation(world, cfg);
    let snap = sim.reset({seed: 1, scenario: 'baseline'});
    const t = createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: 'no_wall', weights, overrides});
    const acts = [];
    while (!snap.state.terminal) { const a = t.action(snap); acts.push([a.steering, a.speed]); snap = sim.step(a); }
    return {acts, t};
  };
  const on = run({}), off = run({shield: false});
  assert.equal(on.t.shieldFrames, 0);
  assert.deepEqual(on.acts, off.acts);
  const ids = on.t.log.filter(q => q.shield?.id?.known).map(q => q.shield.id);
  assert.ok(ids.length > 500);
  for (const q of ids) { assert.ok(Math.abs(q.sr - 1.2) < 1e-6); assert.equal(q.d, 0); assert.ok(Math.abs(q.g - 1) < 1e-6 || q.g === 1); }
  assert.equal(on.t.metadata.shield.version, 'jf-shield/v3');
});
