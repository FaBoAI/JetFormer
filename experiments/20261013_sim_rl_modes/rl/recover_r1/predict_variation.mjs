// 離脱復活の学習の段(r1): ばらつきの物差しの Node の予想(判定は記録の段。protocol variation_metrics.where_judged)。2026-10-13
//   教師: RC 教師 × REC60 の 0〜14 番(15 シーン・9,000 フレーム。tests/metrics_eval.test.mjs と同じ作り方、高速版 = 元のコアとビット単位で同じ)
//   方策: 記録の決まりの開始(recover.recording.start_distribution の Halton 1〜、種 recording_base = 700000 + 1000·3 + i)で、
//         元の sim-core.js で走らせ、衝突した回は捨てて次の番号に替える(記録と同じ)。有効な 15 シーンを集める
//   node rl/recover_r1/predict_variation.mjs --policy <policy.json> [--scenes 15] --out <結果.json>
import fs from 'node:fs';
import {runModeEpisode, readJson, BASE_WORLD} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {activeWorldFor} from '../../snapshot/rl/env_core.mjs';
import {prepareGeometry, clearance} from '../runtime/features.mjs';
import {maskFromLearn} from '../runtime/perturb.mjs';
import {VM1, VM3, VM7, VM6, sceneFromEpisode} from '../variation_metrics.mjs';
import {RlEnv, loadCore, loadRcTeacher, loadBaseWorld} from '../../snapshot/rl/env_core.mjs';
import {loadRec60} from '../../snapshot/rl/eval_sets.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = readJson(args.policy), nScenes = +(args.scenes ?? 15);
// 教師 15 シーン
const core = await loadCore('fast'), T = await loadRcTeacher();
const env = new RlEnv({core, coreKind: 'fast', baseWorld: loadBaseWorld(), mode: 'eval', teacherModule: T, teacherActions: true});
const teacher = [];
for (const c of loadRec60().slice(0, 15)) {
  env.reset(c);
  const frames = [];
  while (!env.done) { const cmd = env.teacherCommand; frames.push({st: cmd.steering, s: env.aux.s, ey: env.aux.lateral_error, epsi: env.aux.heading_error, v: env.view.state.speed, dmin: env.aux.d_min}); env.stepCommand(cmd.steering, cmd.speed); }
  teacher.push({frames, s0: frames[0].s});
}
// 方策: 記録の開始の分布(開始の衝突なし)、衝突した回は替える
const dist = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution, geom = prepareGeometry(BASE_WORLD);
const scenes = [], tried = [];
for (let h = 1, i = 0; scenes.length < nScenes && h < 10000; h++) {
  const st = haltonStart(h, dist);
  const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  if (!(clearance(geom, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5) > 1e-9)) continue;
  const seed = 703000 + i; i++;
  const r = await runModeEpisode({mode: 'recover', weights, start: {...st, seed}, coreKind: 'original', audit: true});
  tried.push({halton: h, seed, valid: r.valid_episode, reason: r.reason, frames: r.frames, events: r.events?.length ?? 0});
  if (!r.valid_episode) continue;
  const sc = sceneFromEpisode(r);
  sc.mask = maskFromLearn(r.log.map(q => q.l));
  scenes.push(sc);
}
const t7 = VM7(teacher), p7 = VM7(scenes);
const res = {policy: args.policy, teacher: {scenes: teacher.length, VM1_us: VM1(teacher), VM3_m: VM3(teacher), VM7: t7},
  policy_scenes: {scenes: scenes.length, tried: tried.length, valid_rate: scenes.length / tried.length, VM1_us: VM1(scenes), VM3_m: VM3(scenes), VM6: VM6(scenes), VM7: p7,
    min_clearance: Math.min(...scenes.map(s => s.min_clearance ?? Infinity))},
  vm7_ratio: p7.cells / t7.cells, tried};
console.log(JSON.stringify({teacher_vm7: t7, policy_vm7: p7, ratio: res.vm7_ratio, VM6: res.policy_scenes.VM6, valid_rate: res.policy_scenes.valid_rate, tried: tried.length}));
fs.writeFileSync(args.out, JSON.stringify(res));
