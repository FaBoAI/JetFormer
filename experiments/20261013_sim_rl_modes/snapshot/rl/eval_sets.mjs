// RL の評価の条件(REC60・STD・ROB24・閉ループの開始 E0〜E5)と、1 エピソードを走らせる道具(2026-10-13、RL 環境の段)
//
// 走らせるもの(controller)は次のどれか:
//   teacherController()          RC 教師(createRcTrainingTeacher。約束の範囲 = ゲート 0・±3 cm・±3° の外では作れない)
//   teacherOutsideContract()     参考用: 教師の参照を標準の開始で作り、ゲート 0 のずれた開始から走らせる(約束の外。合否には使わない)
//   replayController(actions)    保存した操作列をそのまま適用する
//   policyController(fn)         方策 fn(観測 Float64Array(79)) → [a0, a1](PPO の段が JS の方策をここに渡す)
// 判定は sim の結果だけを使う(mode 'eval': ゲート逃し・停止・逆走で途中終了しない。アプリと同じ)。
import fs from 'node:fs';
import {RlEnv, loadCore, loadRcTeacher, loadBaseWorld, PATHS, checkAsset, EPISODE_SECONDS, inTeacherContract} from './env_core.mjs';

// ---- 条件 ----
export function loadRec60() {
  checkAsset('plan60');
  const plan = JSON.parse(fs.readFileSync(PATHS.plan60, 'utf8'));
  if (plan.length !== 60) throw new Error('eval_sets: 記録の計画が 60 条件ではありません');
  return plan.map(c => ({name: c.name, split: c['dataset-split'], gate: c['start-gate'] ?? 0, left_offset_m: c['spawn-offset'] ?? 0,
    left_yaw_deg: c['spawn-yaw'] ?? 0, seed: c.seed, scenario: c.scenario ?? 'baseline', noiseStd: c.scenario === 'noise' ? .025 : 0,
    limit: c.limit}));
}
export const STD = Object.freeze({name: 'STD', gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1, scenario: 'baseline', noiseStd: 0});
// protocol.json evaluation_sets_rl.ROB24
export const ROB24 = Object.freeze([
  ...[[-.1, -6], [-.1, 6], [-.05, -6], [-.05, 6], [.05, -6], [.05, 6], [.1, -6], [.1, 6]].map(([o, y]) => ({gate: 0, left_offset_m: o, left_yaw_deg: y})),
  ...[4, 8, 12, 16, 20, 24, 28, 31].flatMap(g => [{gate: g, left_offset_m: .05, left_yaw_deg: 3}, {gate: g, left_offset_m: -.05, left_yaw_deg: -3}]),
].map((c, i) => Object.freeze({name: `ROB24-${String(i).padStart(2, '0')}`, seed: 1, scenario: 'baseline', noiseStd: 0, ...c})));
// protocol.json closed_loop_eval.starts_single_lap
export const CLOSED_LOOP_STARTS = Object.freeze([
  {name: 'E0', gate: 0, left_offset_m: 0, left_yaw_deg: 0}, {name: 'E1', gate: 0, left_offset_m: .03, left_yaw_deg: 3},
  {name: 'E2', gate: 0, left_offset_m: -.03, left_yaw_deg: -3}, {name: 'E3', gate: 0, left_offset_m: .08, left_yaw_deg: -5},
  {name: 'E4', gate: 0, left_offset_m: -.08, left_yaw_deg: 5}, {name: 'E5', gate: 20, left_offset_m: 0, left_yaw_deg: 0},
].map(c => Object.freeze({seed: 1, scenario: 'baseline', noiseStd: 0, ...c})));

// ---- 走らせるもの ----
export async function teacherController() {
  const T = await loadRcTeacher();
  return {name: 'rc-teacher', create: ({world, config, variation}) => {
    const t = T.createRcTrainingTeacher(world, config, variation);
    return {action: env => t.action(env.view)};
  }};
}
export async function teacherOutsideContract() {
  const T = await loadRcTeacher();
  return {name: 'rc-teacher-outside-contract', create: ({world, config, variation}) => {
    if (variation.gate !== 0) throw new Error('eval_sets: 教師の参照はゲート 0 から始まるので、ほかのゲートからは走れません');
    const first = world.route[0], next = world.route[1], heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
    // 参照は開始姿勢によらない(ゲート 0 の位置から作る)ので、標準の開始姿勢で作って、ずれた開始から使う
    const t = T.createRcTrainingTeacher({...world, spawn: {x: first[0], z: first[1], yaw: heading}}, config, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
    return {action: env => t.action(env.view)};
  }};
}
export function replayController(actions, name = 'replay') {
  return {name, create: () => { let i = 0; return {action: () => { const a = actions[i++]; return {steering: a.steering, speed: a.speed}; }}; }};
}
// fn(観測, env) → [a0, a1](a ∈ [−1, 1]²。範囲外は env が切る)
export function policyController(fn, name = 'policy') {
  return {name, create: ctx => ({action: env => { const a = fn(env.obs, env, ctx); return {a0: a[0], a1: a[1]}; }})};
}

// ---- 1 エピソード ----
const envCache = new Map();
async function envFor(coreKind, episodeMode, variant, timeLimit) {
  const key = [coreKind, episodeMode, variant, timeLimit].join('/');
  if (!envCache.has(key)) envCache.set(key, new RlEnv({core: await loadCore(coreKind), coreKind, baseWorld: loadBaseWorld(), variant, mode: 'eval',
    episodeMode, timeLimit}));
  return envCache.get(key);
}
// start: {gate, left_offset_m, left_yaw_deg, seed, scenario, noiseStd}。invalidSpawnRule 'next-gate' なら重なるときに次のゲートへ
export async function runEpisode({start, controller, coreKind = 'fast', episodeMode = 'continuous', variant = 'main',
  timeLimit = EPISODE_SECONDS, record = false, invalidSpawnRule = 'fail', delayFrames = 0}) {
  const env = await envFor(coreKind, episodeMode, variant, timeLimit);
  let used = {...start}, moved = 0;
  for (;;) {
    if (env.reset(used)) break;
    if (invalidSpawnRule !== 'next-gate' || moved >= 32) return {name: start.name, start, invalid_spawn: true};
    moved++; used = {...used, gate: ((start.gate ?? 0) + moved) % 33};
  }
  const ctrl = controller.create({world: env.world, config: JSON.parse(JSON.stringify(env.config)), variation: env.variation, env});
  let queued = null;   // 制御遅延(参考の評価だけ。delayFrames 1 = アプリの固定遅延と同じく最初は中立 {0, 0})
  while (!env.done) {
    let c = ctrl.action(env);
    if (delayFrames === 1) { const applied = queued ?? {steering: 0, speed: 0}; queued = c; c = applied; }
    else if (delayFrames !== 0) throw new Error('eval_sets: delayFrames は 0 か 1 です');
    if ('a0' in c) env.step(c.a0, c.a1); else env.stepCommand(c.steering, c.speed);
  }
  const s = env.summary();
  const out = {name: start.name, ...s, start_requested: start, start_moved_gates: moved, controller: controller.name, delay_frames: delayFrames};
  out.valid_two_laps = episodeMode === 'continuous' && s.sim_reason === 'collection_duration_complete' && s.collisions === 0 && s.completed_laps >= 2
    && s.laps.every(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0);
  out.lap_complete = episodeMode === 'single' && s.sim_reason === 'route_complete' && s.collisions === 0;
  if (record) out.report = env.report();
  return out;
}

const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const median = a => { const b = [...a].sort((x, y) => x - y), n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : null; };
export function lapStats(values) {
  return values.length ? {n: values.length, mean: mean(values), median: median(values), min: Math.min(...values), max: Math.max(...values)} : {n: 0};
}
// REC60 の集計(selection.eligible の REC60 の部分と score)
export function summarizeRec60(results) {
  const lap1 = results.map(r => r.laps[0]?.elapsed).filter(Number.isFinite), lap2 = results.map(r => r.laps[1]?.elapsed).filter(Number.isFinite);
  const ok = results.filter(r => r.valid_two_laps).length;
  return {scenes: results.length, valid_two_laps: ok, all_valid: ok === results.length, collisions: results.reduce((s, r) => s + r.collisions, 0),
    min_clearance: Math.min(...results.map(r => r.min_clearance)), lap1: lapStats(lap1), lap2: lapStats(lap2),
    score_mean_lap1_plus_lap2: lap1.length === results.length && lap2.length === results.length ? mean(results.map(r => r.laps[0].elapsed + r.laps[1].elapsed)) : null,
    reasons: Object.fromEntries([...new Set(results.map(r => r.reason))].map(k => [k, results.filter(r => r.reason === k).length]))};
}
export function summarizeRob24(results) {
  const ok = results.filter(r => !r.invalid_spawn && r.valid_two_laps).length;
  return {conditions: results.length, valid_two_laps: ok, required: 22, pass: ok >= 22,
    moved: results.filter(r => r.start_moved_gates).map(r => ({name: r.name, from: r.start_requested.gate, to: r.start.gate})),
    failures: results.filter(r => r.invalid_spawn || !r.valid_two_laps).map(r => ({name: r.name, reason: r.invalid_spawn ? 'invalid_spawn' : r.reason, laps: r.completed_laps}))};
}

// 方策(controller)を REC60・STD(単周と連続)・ROB24 で評価する(selection の判定に使う形)
export async function evaluateController(controller, {coreKind = 'fast', variant = 'main', sets = ['REC60', 'STD', 'ROB24'], record = false} = {}) {
  const out = {controller: controller.name, core: coreKind, variant};
  if (sets.includes('REC60')) {
    const rows = [];
    for (const c of loadRec60()) rows.push(await runEpisode({start: c, controller, coreKind, variant, record}));
    out.REC60 = {summary: summarizeRec60(rows), scenes: rows};
  }
  if (sets.includes('STD')) {
    const single = await runEpisode({start: STD, controller, coreKind, variant, episodeMode: 'single', record});
    const cont = await runEpisode({start: STD, controller, coreKind, variant, episodeMode: 'continuous', record});
    out.STD = {single_lap_from_stop: single.lap_complete ? single.laps[0].elapsed : null, single, flying_lap: cont.valid_two_laps ? cont.laps[1].elapsed : null,
      continuous: cont};
  }
  if (sets.includes('ROB24')) {
    const rows = [];
    for (const c of ROB24) rows.push(await runEpisode({start: c, controller, coreKind, variant, invalidSpawnRule: 'next-gate', record}));
    out.ROB24 = {summary: summarizeRob24(rows), scenes: rows};
  }
  if (out.REC60 && out.STD && out.ROB24) {
    const r = out.REC60.summary;
    out.eligible = r.all_valid && r.min_clearance >= .03 && out.STD.single.lap_complete && out.ROB24.summary.pass;
    out.score = r.score_mean_lap1_plus_lap2;
  }
  return out;
}
export {inTeacherContract};
