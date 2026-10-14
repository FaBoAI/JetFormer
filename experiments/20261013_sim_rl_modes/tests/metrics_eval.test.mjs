// ばらつきの物差し(VM1 は教師 REC60 で 7.58 ± 0.05 µs)と、試験の組の回の数(protocol のとおり)
import test from 'node:test';
import assert from 'node:assert/strict';
import {VM1, VM3, sceneFromEpisode} from '../rl/variation_metrics.mjs';
import {buildJobs} from '../rl/eval_modes.mjs';
import {RlEnv, loadCore, loadRcTeacher, loadBaseWorld} from '../snapshot/rl/env_core.mjs';
import {loadRec60} from '../snapshot/rl/eval_sets.mjs';

test('VM1: RC 教師 × 記録の 60 条件 = 7.58 ± 0.05 µs(今日の報告の値)', async () => {
  const core = await loadCore('fast'), T = await loadRcTeacher();
  const env = new RlEnv({core, coreKind: 'fast', baseWorld: loadBaseWorld(), mode: 'eval', teacherModule: T, teacherActions: true});
  const scenes = [];
  for (const c of loadRec60()) {
    env.reset(c);
    const frames = [];
    while (!env.done) { const cmd = env.teacherCommand; frames.push({st: cmd.steering, s: env.aux.s, ey: env.aux.lateral_error, epsi: env.aux.heading_error, v: env.view.state.speed, dmin: env.aux.d_min}); env.stepCommand(cmd.steering, cmd.speed); }
    scenes.push({frames, s0: frames[0].s});
  }
  const v = VM1(scenes);
  assert.ok(Math.abs(v - 7.58) <= .05, 'VM1 ' + v);
  const v3 = VM3(scenes).mean_abs_dev;
  assert.ok(Math.abs(v3 - .0039) < .002, 'VM3 ' + v3);
});
test('試験の組の回の数(protocol test_sets)', () => {
  const count = (m, t) => buildJobs(m, 'test').filter(j => t.includes(j.test)).length;
  assert.equal(count('fastest', ['F-REC60']), 60);
  assert.equal(count('no_wall', ['NW-N60', 'NW-STD', 'NW-G', 'NW-R200', 'NW-FC10']), 340);
  assert.equal(count('no_wall', ['NW-G']), 68);
  assert.equal(count('recover', ['RC-P100']), 100);
  assert.equal(count('recover', ['RC-KICK']), 148);
  assert.equal(count('line', ['LN-P60']), 60);
  assert.equal(count('line', ['LN-FAM6']), 6);
  assert.equal(count('speed', ['SP-P60']), 60);
  assert.equal(count('obstacle', ['OB-T100']), 100);
  // 検証の組は 1/4(切り上げ)で、試験と開始・種が重ならない
  const tv = buildJobs('recover', 'validation').filter(j => j.test === 'RC-P100'), tt = buildJobs('recover', 'test').filter(j => j.test === 'RC-P100');
  assert.equal(tv.length, 25);
  const key = j => JSON.stringify(j.spec.start);
  assert.equal(tv.filter(v => tt.some(t => key(t) === key(v))).length, 0);
});
