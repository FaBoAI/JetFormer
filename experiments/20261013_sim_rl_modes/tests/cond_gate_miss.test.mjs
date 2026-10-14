// cond(経路いろいろ・速さいろいろ)の学習の環境: 仮想のゲートの外しで −10 で終わる(modes.json line.training.reward.terminal = 「base と同じ」)。
// 2026-10-14 学習の段 speed で直したところの試験(deviations.md「cond の学習の環境で…」)。recover の決まり(−3 で続ける)は変えていない。
import test from 'node:test';
import assert from 'node:assert/strict';
import * as modesCore from '../rl/fast_core_modes.mjs';
import {ModeEnv, trainConfig} from '../rl/env_modes.mjs';
import {createRng} from '../rl/runtime/rng.mjs';

const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};

test('cond(と obstacle)の設定に gate_miss_terminal があり、base・no_wall・recover には無い', () => {
  assert.equal(trainConfig('cond').gate_miss_terminal, true);
  assert.equal(trainConfig('obstacle').gate_miss_terminal, true);   // obstacle の段が同じ直しを入れた(2026-10-14 01:34)
  for (const run of ['base', 'no_wall', 'recover']) assert.ok(!trainConfig(run).gate_miss_terminal, run);
});

test('cond: 仮想のゲートの外し(missEvent)で reason gate_miss・終わりの罰 −10 で終わる', () => {
  const env = new ModeEnv({core: modesCore, run: 'cond'});
  assert.equal(env.reset(createRng(3), STD, {line: {family: 'const', c0: 0}, c_v: 1}), true);
  // 数フレームはふつうに進む(外しなし)
  for (let k = 0; k < 5; k++) { const r = env.step(0, 0); assert.equal(r.terminated, false); }
  // 次のフレームで仮想のゲートを外したことにする(wrapper.advance の結果に missEvent を立てる)
  const adv = env.wrapper.advance.bind(env.wrapper);
  env.wrapper.advance = (st, ri) => { const g = adv(st, ri); g.missEvent = true; return g; };
  const r = env.step(0, 0);
  assert.equal(r.terminated, true);
  assert.equal(r.reason, 'gate_miss');
  assert.equal(env.stats.reward_parts.terminal, -10);
  assert.equal(env.stats.gate_misses, 1);
});

test('recover: 仮想のゲートの外しでは終わらない(−3 の罰で続ける)のは変わらない', () => {
  const env = new ModeEnv({core: modesCore, run: 'recover'});
  assert.equal(env.reset(createRng(3), STD), true);
  const adv = env.wrapper.advance.bind(env.wrapper);
  let once = true;
  env.wrapper.advance = (st, ri) => { const g = adv(st, ri); if (once) { g.missEvent = true; once = false; } return g; };
  const r = env.step(0, 0);
  assert.equal(r.terminated, false);
  assert.equal(env.stats.reward_parts.gate_miss, -3);
});
