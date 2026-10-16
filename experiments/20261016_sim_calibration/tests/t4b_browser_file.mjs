// 試験 T4b: ブラウザ用の静的なファイルそのもの(sim/dist/_calib/)を読み込み、REC60 の 1 条件を教師で走らせて、Node の較正版と同じ report になるかを確かめる
//   node tests/t4b_browser_file.mjs
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const F = '/home/jetson/jetformer_reports/sim_calibration_20261016/sim/dist/_calib/sim-core-jf-calib-v1-real-20261010.js';
const text = fs.readFileSync(F, 'utf8');
const mod = await import(`data:text/javascript;base64,${Buffer.from(text).toString('base64')}`);
const {loadCalibratedCore, loadProfile} = await import('/home/jetson/jetformer/experiments/20261016_sim_calibration/calib_core.mjs');
const {profile, sha256} = loadProfile('P_cal');
const node = await loadCalibratedCore(profile, {fast: false, profileSha256: sha256, name: profile.name});
const EXP13 = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap';
const {RlEnv, loadBaseWorld} = await import(pathToFileURL(EXP13 + '/rl/env_core.mjs').href);
const {loadRec60, teacherController} = await import(pathToFileURL(EXP13 + '/rl/eval_sets.mjs').href);
const teacher = await teacherController();
const rec = loadRec60();
const out = [];
for (const m of [mod, node]) {
  const env = new RlEnv({core: m, coreKind: 'original', baseWorld: loadBaseWorld(), mode: 'eval', episodeMode: 'continuous', timeLimit: 20});
  env.reset(rec[0]);
  const ctrl = teacher.create({world: env.world, config: JSON.parse(JSON.stringify(env.config)), variation: env.variation, env});
  while (!env.done) { const c = ctrl.action(env); env.stepCommand(c.steering, c.speed); }
  const r = env.sim.report();
  out.push(JSON.stringify(r));
  console.log(JSON.stringify(r.metrics.lap_events.map(l => l.elapsed.toFixed(3))), r.metrics.passed_gates, r.outcome.reason, r.metrics.completed_laps, r.metrics.collisions, r.final_state.sim_time.toFixed(2), JSON.stringify(r.model.calibration));
}
console.log('same report:', out[0] === out[1]);
