// 障害物よけ(obstacle)の学習の段(r1): 記録のプランを Node(元の sim-core.js・同じ mode_runtime.mjs)で前もって走らせた、ばらつきの物差しの予想。
// 判定は記録の段(protocol variation_metrics.where_judged)。ここは「記録できなかったときに報告する Node の予想」と同じ作り方。2026-10-14
//   記録の決まり(modes.json obstacle.recording): 開始 = Halton(ゲート U{0..32}・左右 ±0.15 m・向き ±10°)、開始の前後 1.5 m に障害物が無い配置だけ、
//   シーンごとに recording 組(30 配置)から違う配置。種 = recording_base = 700000 + 1000·6 + 100·(プランの通し番号 mod 10) + i。
//   ここでは シーン i に配置 i を割り当て、その配置で開始の条件を満たす次の Halton の番号を使う。600 フレーム・衝突 0 にならないシーンは
//   同じ配置で次の Halton の番号に替える(3 回まで。だめなら次の配置へ。recording_common.prefilter と同じ「替える」)。
//   教師の基準: RC 教師 × REC60 の 0〜14 番(rl/recover_r1/predict_variation.mjs と同じ作り方)。
//   node rl/ob/predict_variation.mjs --policy <policy.json> [--scenes 15] [--plan 0] --out <結果.json>
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {activeWorldFor, RlEnv, loadCore, loadRcTeacher, loadBaseWorld} from '../../snapshot/rl/env_core.mjs';
import {prepareGeometry, clearance, projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';
import {geometryWithObstacles, startClearOfObstacles} from '../runtime/layouts.mjs';
import {VM1, VM2, VM3, VM4, VM5, VM7, sceneFromEpisode} from '../variation_metrics.mjs';
import {loadRec60} from '../../snapshot/rl/eval_sets.mjs';
import {passageStats, summarizePassages, obstacleEy} from './ob_common.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = readJson(args.policy), nScenes = +(args.scenes ?? 15), planNo = +(args.plan ?? 0);
const lays = readJson(path.join(here, '../obstacle_sets/recording.json')).layouts;
// 教師 15 シーン(高速版 = 元のコアとビット単位で同じ。試験で確認済み)
const core = await loadCore('fast'), T = await loadRcTeacher();
const env = new RlEnv({core, coreKind: 'fast', baseWorld: loadBaseWorld(), mode: 'eval', teacherModule: T, teacherActions: true});
const teacher = [];
for (const c of loadRec60().slice(0, 15)) {
  env.reset(c);
  const frames = [];
  while (!env.done) { const cmd = env.teacherCommand; frames.push({st: cmd.steering, s: env.aux.s, ey: env.aux.lateral_error, epsi: env.aux.heading_error, v: env.view.state.speed, dmin: env.aux.d_min}); env.stepCommand(cmd.steering, cmd.speed); }
  teacher.push({frames, s0: frames[0].s});
}
const dist = CATALOGUE.modes.find(m => m.key === 'obstacle').recording.start_distribution, geom = prepareGeometry(BASE_WORLD);
const scenes = [], tried = [], passes = [], sceneTraces = [];
const LOOPL = LOOP_LENGTH;
let h = 1, li = 0;
const seedOf = i => 700000 + 1000 * 6 + 100 * (planNo % 10) + i;
while (scenes.length < nScenes && li < lays.length) {
  const lay = lays[li];
  let ok = false;
  for (let attempt = 0; attempt < 3 && !ok; attempt++) {
    // この配置で開始の条件を満たす次の Halton の番号
    let st = null;
    for (; h < 10000; h++) {
      const c = haltonStart(h, dist);
      const {world} = activeWorldFor(BASE_WORLD, c.gate, c.left_offset_m, c.left_yaw_deg);
      const g = geometryWithObstacles(geom, lay.obstacles);
      if (!(clearance(g, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5) > 1e-9)) continue;
      if (!startClearOfObstacles(projectWindow(world.spawn.x, world.spawn.z, 0, LOOP_LENGTH).s, lay.obstacles)) continue;
      st = c; h++; break;
    }
    if (!st) break;
    const seed = seedOf(scenes.length);
    const r = await runModeEpisode({mode: 'obstacle', weights, start: {...st, seed}, params: {layout_id: lay.id}, layout: lay, coreKind: 'original', audit: true});
    const ps = passageStats(r, lay.obstacles);
    tried.push({layout: lay.id, n_obstacles: lay.obstacles.length, halton: st.halton_index ?? null, start: st, seed, valid: r.valid_episode, reason: r.reason, frames: r.frames,
      min_clearance: r.min_clearance, laps: r.laps.map(l => l.elapsed), passes: ps.length});
    if (!r.valid_episode) continue;
    ok = true;
    const sc = sceneFromEpisode(r);
    sc.layout = lay.id;
    scenes.push(sc);
    // 図のための軽い列(2 フレームおき): 絶対の s・横ずれ・速さ。と配置の障害物の s・横の位置
    sceneTraces.push({layout: lay.id, start: st, seed, laps: r.laps.map(l => l.elapsed), min_clearance: r.min_clearance,
      obstacles: lay.obstacles.map(o => ({shape: o.shape, s: +((o.s_ref % LOOPL) + LOOPL) % LOOPL, ey: +obstacleEy(o).toFixed(4), r: o.r, length: o.length ?? null})),
      f: r.log.filter((_, i) => i % 2 === 0).map(q => [+q.s.toFixed(3), +q.ey.toFixed(4), +q.v.toFixed(3)]),
      xz: r.trace.filter((_, i) => i % 2 === 0).map(q => [+q.x.toFixed(3), +q.z.toFixed(3), +q.v.toFixed(3)])});
    passes.push(...ps.map(p => ({...p, layout: lay.id})));
  }
  li++;
}
const t7 = VM7(teacher), p7 = VM7(scenes);
const res = {schema: 'jf-sim-rl-obstacle-variation-prediction/v1', created_at: new Date().toISOString(), policy: args.policy, plan_no: planNo,
  rule: 'シーン i に recording 組の配置を順に割り当て、Halton の開始(前後 1.5 m に障害物なし)。有効にならないシーンは同じ配置で次の開始(3 回まで)→ 次の配置',
  teacher: {scenes: teacher.length, VM1_us: VM1(teacher), VM3_m: VM3(teacher), VM7: t7},
  policy_scenes: {scenes: scenes.length, tried: tried.length, valid_rate: scenes.length / Math.max(1, tried.length), distinct_layouts: new Set(scenes.map(s => s.layout)).size,
    VM1_us: VM1(scenes), VM2_us: VM2(scenes), VM3_m: VM3(scenes), VM4_mps: VM4(scenes), VM5: VM5(scenes), VM7: p7,
    lap1_s: scenes.map(s => s.laps?.[0]).filter(Number.isFinite), min_clearance: Math.min(...scenes.map(s => s.min_clearance ?? Infinity))},
  judged_preview: null, vm7_ratio: p7.cells / t7.cells, passages: summarizePassages(passes), passages_list: passes, tried, scene_traces: sceneTraces};
const vm3 = res.policy_scenes.VM3_m?.mean_abs_dev ?? null;
res.judged_preview = {VM3_min_0_04: {value: vm3, pass: vm3 != null && vm3 >= .04}, distinct_layouts_min_12_of_15: {value: res.policy_scenes.distinct_layouts,
  pass: scenes.length >= 15 && res.policy_scenes.distinct_layouts >= 12}, note: '予想(Node)。判定は記録した見本で行う'};
console.log(JSON.stringify({scenes: scenes.length, tried: tried.length, distinct: res.policy_scenes.distinct_layouts, VM1: res.policy_scenes.VM1_us,
  VM3: vm3, teacher_VM3: res.teacher.VM3_m?.mean_abs_dev, vm7: p7.cells, teacher_vm7: t7.cells, ratio: res.vm7_ratio, passages: res.passages}));
fs.writeFileSync(args.out, JSON.stringify(res));
