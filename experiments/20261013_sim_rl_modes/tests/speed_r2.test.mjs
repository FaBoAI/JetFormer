// 速さいろいろ第 2 回(protocol_speed_r2.json。2026-10-14 学習の段 speed r2)の学習の環境・試験の組の試験。
// cond(経路いろいろ)の学習の設定・乱数の列は変わらないこと、speed の設定が追補の決まりどおりであることを確かめる。
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as modesCore from '../rl/fast_core_modes.mjs';
import {ModeEnv, trainConfig, SPEED_R2_DEFAULTS} from '../rl/env_modes.mjs';
import {createRng} from '../rl/runtime/rng.mjs';
import {buildJobs, loadSpeedR2, haltonStarts} from '../rl/eval_modes.mjs';

const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const R2 = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/protocol_speed_r2.json', 'utf8'));

test('speed の学習の設定 = protocol_speed_r2.json training の値', () => {
  const c = trainConfig('speed');
  assert.equal(c.wrapper, 'cond97'); assert.equal(c.virtual, true); assert.equal(c.gate_miss_terminal, true);
  assert.deepEqual(c.cond.line_fixed, {family: 'const', c0: 0});
  assert.deepEqual([...c.cond.c_v_levels], [1.0, .875, .75, .625, .5]);
  assert.equal(c.cond.c_v_levels_p, .8);
  assert.equal(c.reward.kind, 'cond');
  assert.equal(c.reward.tracking_w, .06); assert.equal(c.reward.tracking_scale_m, .15);
  assert.equal(c.reward.clearance_barrier_w, .2); assert.equal(c.reward.clearance_barrier_start_m, .09); assert.equal(c.reward.terminal_clearance_m, .03);
  // 開始の分布(protocol の start_mix と同じ割合・範囲・すき間の下限)
  assert.deepEqual(c.start_mix.map(m => [m.p, m.left_offset_m, m.left_yaw_deg, m.min_initial_clearance_m]),
    R2.training.start_mix.map(m => [m.p, m.left_offset_m, m.left_yaw_deg, m.min_initial_clearance_m]));
  assert.equal(R2.training.reward.tracking.startsWith('−0.06'), true);
  assert.equal(SPEED_R2_DEFAULTS.clearance_barrier_start_m, .09);
});

test('cond(経路いろいろ)の設定には壁・すき間の終了の鍵が無い(今までと同じ)', () => {
  const c = trainConfig('cond');
  assert.equal(c.reward.clearance_barrier_w, undefined);
  assert.equal(c.reward.terminal_clearance_m, undefined);
  assert.equal(c.cond.line_fixed, undefined);
  assert.equal(c.start_mix, undefined);
  assert.equal(c.reward.tracking_w, .08);
});

// (直す前の env_modes.mjs との乱数の列・報酬の一致は rl/speed_r2/compare_before.mjs で確かめる。写しを rl/ に置く必要があるので試験の外)

test('speed: 線はいつも const c0 = 0、c_v はおよそ 80% が 5 段、開始のすき間は成分の下限以上', () => {
  const env = new ModeEnv({core: modesCore, run: 'speed'}), rng = createRng(5);
  let lv = 0; const N = 300;
  for (let k = 0; k < N; k++) {
    env.resetRandom(rng);
    assert.deepEqual(env.line, {family: 'const', c0: 0});
    assert.ok(env.cV >= .5 - 1e-12 && env.cV <= 1 + 1e-12);
    if ([1.0, .875, .75, .625, .5].includes(env.cV)) lv++;
    const comp = env.cfg.start_mix[env.start.component];
    assert.ok(env.aux.d_min >= comp.min_initial_clearance_m - 1e-12, `開始のすき間 ${env.aux.d_min}`);
  }
  assert.ok(lv / N > .72 && lv / N < .88, `5 段の割合 ${lv / N}`);
});

test('speed: すき間の壁の罰と、3 cm 未満で −10 の終了', () => {
  const env = new ModeEnv({core: modesCore, run: 'speed'});
  assert.equal(env.reset(createRng(3), STD), true);
  // すき間を作り物にして、壁の罰の値を確かめる(wrapper.observe の aux.d_min を上書き)
  const obs = env.wrapper.observe.bind(env.wrapper);
  let fake = .05;
  env.wrapper.observe = (st, a0, a1, out, aux) => { obs(st, a0, a1, out, aux); aux.d_min = fake; return out; };
  const before = env.stats.reward_parts.barrier;
  let r = env.step(0, 0);
  assert.equal(r.terminated, false);
  assert.ok(Math.abs((env.stats.reward_parts.barrier - before) - -(.2 * ((.09 - .05) / .04) ** 2)) < 1e-12);
  fake = .029;
  r = env.step(0, 0);
  assert.equal(r.terminated, true); assert.equal(r.reason, 'clearance'); assert.equal(env.stats.reward_parts.terminal, -10);
});

test('第 2 回の試験・検証・報告の組: 回の数・開始のすき間 ≥ 0.12 m・種・第 1 回の試験の組と重ならない', () => {
  const P = loadSpeedR2();
  for (const set of ['test', 'validation', 'fresh']) {
    const J = buildJobs('speed', set, {protocolR2: P}), p = J.filter(j => j.test === 'SP-P60');
    assert.equal(p.length, R2.sets[set].count, set);
    assert.equal(J.filter(j => j.test === 'SP-STD3').length, 3);
    p.forEach((j, i) => {
      assert.ok(j.spec.start.initial_clearance_m >= .12, `${set} ${j.id} ${j.spec.start.initial_clearance_m}`);
      assert.equal(j.spec.start.seed, R2.sets[set].seed_base + i);
      assert.equal(j.spec.c_v, [1.0, .875, .75, .625, .5][i % 5]);
    });
  }
  const t2 = buildJobs('speed', 'test', {protocolR2: P}).filter(j => j.test === 'SP-P60');
  assert.ok(t2.every(j => j.spec.start.halton_index >= 25001 && j.spec.start.halton_index < 30000));
  const t1 = haltonStarts('speed', 'test', 60);
  assert.ok(Math.max(...t1.map(s => s.halton_index)) < 25001);
  // 第 1 回の試験の組は今までと同じ(開始のすき間の欄を足さない)
  assert.equal(t1[0].initial_clearance_m, undefined);
  assert.equal(buildJobs('speed', 'test').filter(j => j.test === 'SP-P60').length, 60);
});
