// 学習の環境(rl/env_modes.mjs): base はベストラップの RlEnv と同じ値、モードごとの観測の次元・差し込み・物理のずれ
import test from 'node:test';
import assert from 'node:assert/strict';
import {fastestWeights} from './helpers.mjs';
import * as modesCore from '../rl/fast_core_modes.mjs';
import {ModeEnv, sampleDrSpec, DR, trainConfig} from '../rl/env_modes.mjs';
import {RlEnv, loadCore, loadBaseWorld} from '../snapshot/rl/env_core.mjs';
import {createRng} from '../rl/runtime/rng.mjs';
import {createPolicy} from '../rl/runtime/policy_forward.mjs';

test('base: ベストラップの RlEnv と観測・報酬・終了がビット単位で同じ', async () => {
  const core = await loadCore('fast'), act = createPolicy(fastestWeights());
  const a = new ModeEnv({core: modesCore, run: 'base'}), b = new RlEnv({core, coreKind: 'fast', baseWorld: loadBaseWorld()});
  for (const st of [{gate: 0, left_offset_m: .05, left_yaw_deg: 2, seed: 4}, {gate: 21, left_offset_m: -.1, left_yaw_deg: -6, seed: 9}]) {
    assert.equal(a.reset(createRng(1), st), true); assert.equal(b.reset(st), true);
    assert.deepEqual(Array.from(a.obs), Array.from(b.obs));
    const out = [0, 0];
    let n = 0;
    while (!a.done && !b.done) {
      act(b.obs, out); const x = out[0] * .9 + .05, y = out[1];
      const ra = a.step(x, y), rb = b.step(x, y); n++;
      assert.equal(ra.reward, rb.reward); assert.equal(ra.terminated, rb.terminated); assert.equal(ra.truncated, rb.truncated); assert.equal(ra.reason, rb.reason);
      assert.deepEqual(Array.from(a.obs), Array.from(b.obs));
    }
    assert.equal(a.done, b.done);
    assert.ok(n > 100);
  }
});
test('観測の次元と差し込みの印(recover)、物理のずれの引く数は一定', () => {
  const dims = {no_wall: 79, recover: 79, cond: 97, obstacle: 95};
  for (const [run, d] of Object.entries(dims)) {
    const env = new ModeEnv({core: modesCore, run});
    env.resetRandom(createRng(2));
    assert.equal(env.obs.length, d);
    let inj = 0;
    for (let i = 0; i < 400 && !env.done; i++) { const r = env.step(0, -.2); inj += r.injected ? 1 : 0; }
    if (run !== 'recover') assert.equal(inj, 0);
  }
  const r1 = createRng(5), r2 = createRng(5);
  sampleDrSpec(r1, DR['DR-NW-train']); sampleDrSpec(r2, DR['DR-T']);
  assert.equal(r1.state, r2.state);
  // DR-NW-train の割合(遅延 2・摩擦円・標準)
  const rng = createRng(11), xs = Array.from({length: 4000}, () => sampleDrSpec(rng, DR['DR-NW-train']));
  const frac = f => xs.filter(f).length / xs.length;
  assert.ok(Math.abs(frac(x => x.nominal) - .15) < .03);
  assert.ok(frac(x => x.delay === 2) > .08 && frac(x => x.frictionCircle) > .2);
});
test('recover の学習: 乱しの行動が入り、ゲートを外しても終わらない(終わりの理由に gate_miss が無い)', () => {
  const env = new ModeEnv({core: modesCore, run: 'recover'}), act = createPolicy(fastestWeights()), rng = createRng(4), out = [0, 0];
  let inj = 0, reasons = new Set();
  for (let ep = 0; ep < 6; ep++) {
    env.resetRandom(rng);
    while (!env.done) { act(env.obs, out); inj += env.step(out[0], out[1]).injected ? 1 : 0; }
    reasons.add(env.summary().reason);
  }
  assert.ok(inj > 0);
  assert.ok(!reasons.has('gate_miss'));
  assert.equal(trainConfig('recover').reward.gate_miss_penalty, -3);
});
