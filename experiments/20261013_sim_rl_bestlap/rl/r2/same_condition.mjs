#!/usr/bin/env node
// r2(独立な確認 r1 の must-fix): 学習済み ONNX の 8.964 秒と「同じ条件」の RL の周回時間を、RL の段の道具(元の sim-core.js)で測る。
// 2026-10-13。ONNX の 8.964117 秒は、制御遅延 1 フレーム・前進ゲイン 1.01 の条件の値(README・results/shortcut-rc-learned-lap.json)。
// 遅延 1 のときの観測の「前回の行動」の約束を 2 つとも測る:
//   applied = 実際に適用した 1 フレーム前の指令(RL の段の eval_sets の delayFrames 1 と同じ)
//   fresh   = 方策が今出した値(確認 r1 の runClosedLoop と同じ)
//   node rl/r2/same_condition.mjs --out <json>
import fs from 'node:fs';
import {RlEnv, loadCore, loadBaseWorld, loadRcTeacher} from '../env_core.mjs';
import {actionToCommand} from '../features.mjs';
import {loadPolicy} from '../policy_forward.mjs';
import {STD, loadRec60} from '../eval_sets.mjs';

const args = process.argv.slice(2);
const opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const out = opt('--out'), policyPath = opt('--policy', '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/best/main/policy.json');
const core = await loadCore('original'), baseWorld = loadBaseWorld();
const policy = await loadPolicy(policyPath);

// 1 回走らせる。delay 0/1・speedGain・約束 conv
export function runOnce({episodeMode = 'single', start = STD, delay = 1, speedGain = 1, steerGain = 1, conv = 'applied', timeLimit}) {
  const env = new RlEnv({core, coreKind: 'original', baseWorld, mode: 'eval', episodeMode, timeLimit: timeLimit ?? (episodeMode === 'single' ? 15 : 20)});
  if (!env.reset(start)) return {invalid_spawn: true};
  let queued = actionToCommand(0, -1);
  const cmds = [];
  while (!env.done) {
    const a = policy(env.obs), c = actionToCommand(a[0], a[1]);
    let ap = c;
    if (delay === 1) { ap = queued; queued = c; }
    const prev = conv === 'fresh' ? c : ap;
    env.stepCommand(ap.steering * steerGain, ap.speed * speedGain, prev.a0, prev.a1);
    cmds.push([ap.steering, ap.speed]);
  }
  const s = env.summary();
  const valid = episodeMode === 'single'
    ? s.sim_reason === 'route_complete' && s.collisions === 0 && s.passed_gates === 33 && s.laps[0]?.crossings === 1
    : s.sim_reason === 'collection_duration_complete' && s.collisions === 0 && s.laps.length >= 2 && s.laps.every(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0);
  return {valid, reason: s.sim_reason, laps: s.laps.map(l => l.elapsed), gates: s.passed_gates, collisions: s.collisions, min_clearance: s.min_clearance,
    first_collision: s.collision_events?.[0] ?? null};
}
async function teacherOnce({episodeMode = 'single', delay = 1, speedGain = 1}) {
  const T = await loadRcTeacher();
  const env = new RlEnv({core, coreKind: 'original', baseWorld, mode: 'eval', episodeMode, timeLimit: episodeMode === 'single' ? 15 : 20});
  env.reset(STD);
  const t = T.createRcTrainingTeacher(env.world, JSON.parse(JSON.stringify(env.config)), env.variation);
  let queued = {steering: 0, speed: 0};
  while (!env.done) {
    let c = t.action(env.view);
    if (delay === 1) { const ap = queued; queued = c; c = ap; }
    env.stepCommand(c.steering, c.speed * speedGain);
  }
  const s = env.summary();
  return {reason: s.sim_reason, laps: s.laps.map(l => l.elapsed), gates: s.passed_gates, collisions: s.collisions};
}

const rows = [];
for (const conv of ['applied', 'fresh']) {
  for (const [name, delay, speedGain] of [['遅延 0(標準)', 0, 1], ['遅延 1', 1, 1], ['遅延 1 + 前進ゲイン 1.01(ONNX と同じ条件)', 1, 1.01], ['前進ゲイン 1.01 だけ', 0, 1.01]]) {
    if (delay === 0 && conv === 'fresh') continue;   // 遅延 0 では 2 つの約束は同じ
    const single = runOnce({episodeMode: 'single', delay, speedGain, conv});
    const cont = runOnce({episodeMode: 'continuous', delay, speedGain, conv});
    rows.push({name, delay, speed_gain: speedGain, prev_action_convention: delay ? conv : 'same', single, continuous: cont});
    console.log(conv, name, single.reason, single.laps.map(x => x.toFixed(6)), '|', cont.reason, cont.laps.map(x => x.toFixed(6)));
  }
}
// REC60(記録の 60 条件)を遅延 1 で(両方の約束)
const rec = loadRec60();
const rec60 = {};
for (const conv of ['applied', 'fresh']) for (const speedGain of [1, 1.01]) {
  const res = rec.map(c => ({name: c.name, ...runOnce({episodeMode: 'continuous', start: c, delay: 1, speedGain, conv})}));
  const ok = res.filter(r => r.valid), m = a => a.reduce((x, y) => x + y, 0) / a.length;
  rec60[`${conv}_gain${speedGain}`] = {n: res.length, valid: ok.length, lap1_mean: ok.length ? m(ok.map(r => r.laps[0])) : null,
    lap2_mean: ok.length ? m(ok.map(r => r.laps[1])) : null, failures: res.filter(r => !r.valid).map(r => ({name: r.name, reason: r.reason, laps: r.laps, gates: r.gates, first_collision: r.first_collision}))};
  console.log('REC60 delay1', conv, 'gain', speedGain, rec60[`${conv}_gain${speedGain}`].valid, '/60', rec60[`${conv}_gain${speedGain}`].lap1_mean, rec60[`${conv}_gain${speedGain}`].lap2_mean);
}
const teacher = {delay1_gain1_01_single: await teacherOnce({delay: 1, speedGain: 1.01}), delay1_single: await teacherOnce({delay: 1}),
  delay0_single: await teacherOnce({delay: 0})};
console.log('teacher', JSON.stringify(teacher));
const ONNX = 8.964117142565293;
const pick = (conv, d, g) => rows.find(r => r.delay === d && r.speed_gain === g && (d === 0 || r.prev_action_convention === conv))?.single.laps[0] ?? null;
const cmp = {onnx_from_stop_s: ONNX, onnx_condition: '制御遅延 1 フレーム・前進ゲイン 1.01(学習済み ONNX の記録の条件)',
  rl_delay1_applied_s: pick('applied', 1, 1), rl_delay1_fresh_s: pick('fresh', 1, 1),
  rl_delay1_gain101_applied_s: pick('applied', 1, 1.01), rl_delay1_gain101_fresh_s: pick('fresh', 1, 1.01), rl_delay0_s: pick('applied', 0, 1)};
cmp.diff_same_condition_s = [cmp.rl_delay1_gain101_applied_s, cmp.rl_delay1_gain101_fresh_s].map(x => ONNX - x);
cmp.diff_delay1_only_s = [cmp.rl_delay1_applied_s, cmp.rl_delay1_fresh_s].map(x => ONNX - x);
cmp.diff_range_s = [Math.min(...cmp.diff_same_condition_s, ...cmp.diff_delay1_only_s), Math.max(...cmp.diff_same_condition_s, ...cmp.diff_delay1_only_s)];
cmp.wrong_previous_claim = {text: '学習済み ONNX より 0.872 秒速い(遅延 0 の RL 8.092 秒と、遅延 1・ゲイン 1.01 の ONNX 8.964 秒を並べた)', diff_s: ONNX - cmp.rl_delay0_s};
console.log(JSON.stringify(cmp, null, 1));
if (out) fs.writeFileSync(out, JSON.stringify({schema: 'rl-bestlap-same-condition/v1', created_at: new Date().toISOString(), core: 'original',
  policy: policyPath, note: '学習済み ONNX の 8.964 秒と同じ条件(遅延 1・前進ゲイン 1.01)での RL の周回時間。確認 r1 の must-fix への対応', comparison: cmp, rows, rec60_delay1: rec60, teacher}, null, 1) + '\n');
