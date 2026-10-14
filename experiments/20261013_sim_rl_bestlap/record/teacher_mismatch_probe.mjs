#!/usr/bin/env node
// 記録の段: verify_recording.mjs の「方策の再計算」がフレーム 1 つだけで合わなかったシーンを調べる(2026-10-14)
//
//   node record/teacher_mismatch_probe.mjs <シーンのフォルダー> [--out 結果.json]
//
//  verify_recording.mjs は、記録した操作を Node の元の sim-core.js で再生した状態(ブラウザの状態との差 ~1e-13)から方策を計算し直す。
//  ここでは同じ方策の観測を 2 通りの状態で作って比べる:
//    A) Node で再生した状態(verify_recording.mjs と同じ)
//    B) ブラウザが記録した状態そのもの(frames.jsonl の pose_pre_action・前のフレームの state_post_action。route_index は A と同じ)
//  方策の出力は、ブラウザが読んだ差し込みモジュール dist/rl/rl-teacher-bestlap.js そのもの(A・B で 1 つずつ作る)で計算する。
//  観測の成分の比較(どれが跳んだか)だけは features.mjs で同じ計算を並べて行う。
//  記録の操作と A・B それぞれの方策の出力の差、合わないフレームで観測のどの成分が跳んだかを書く。データは変えない。
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {baseWorld, activeWorldFor, createSimulation, SIM_ROOT} from '../lib/app_emulation.mjs';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
const outPath = oi >= 0 ? args.splice(oi, 2)[1] : null;
const F = await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rl/features.mjs')).href);
const {createPolicy} = await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rl/policy_forward.mjs')).href);
const W = (await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rl/policy_weights.mjs')).href)).default;
const NAMES = [];
for (let i = 0; i < F.OBS_DIM; i++) {
  const ix = F.OBS_INDEX;
  NAMES.push(i < 3 ? ['speed', 'steering', 'yaw_rate'][i] : i < 5 ? `prev_a${i - 3}` : i < 8 ? ['lateral_error', 'heading_sin', 'heading_cos'][i - 5]
    : i < 40 ? `lookahead_${(i - 8) % 2 ? 'left' : 'fwd'}_${Math.floor((i - 8) / 2)}` : i < 56 ? `lookahead_mu_${i - 40}` : i < 75 ? `ray_${-90 + 10 * (i - 56)}deg`
      : i === 75 ? 'center_mu' : i === 76 ? 'gate_along' : i === 77 ? 'gate_lateral' : 'd_min');
  void ix;
}

let root = path.resolve(args[0]);
if (!fs.existsSync(path.join(root, 'simulation/session.json')))
  root = fs.readdirSync(root).map(p => path.join(root, p)).find(p => fs.existsSync(path.join(p, 'simulation/session.json')));
const read = p => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const session = read('simulation/session.json'), report = read('simulation/evaluation-report.json');
const rows = fs.readFileSync(path.join(root, 'simulation/frames.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const v = session.context.start_variation;
const {world} = activeWorldFor(await baseWorld(), v.gate, v.left_offset_m, v.left_yaw_deg);
world.spawn = {...rows[0].pose_pre_action};
const sim = createSimulation(world, report.config);
sim.reset({seed: report.seed, scenario: report.scenario});
const cfg = sim.snapshot().config;
const act = createPolicy(W);
const teacherMod = await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rl/rl-teacher-bestlap.js')).href);
const variation = {gate: v.gate, left_offset_m: v.left_offset_m, left_yaw_deg: v.left_yaw_deg};
const tA = teacherMod.createRcTrainingTeacher(world, cfg, variation), tB = teacherMod.createRcTrainingTeacher(world, cfg, variation);
const mk = () => ({tracker: F.createFeatureTracker(world, cfg), obs: new Float64Array(F.OBS_DIM), a: [0, 0], p0: 0, p1: 0});
const A = mk(), B = mk();
function step(T, st, routeIndex, frame) {
  if (frame === 0) T.tracker.resetProgress(st.x, st.z, routeIndex); else T.tracker.advanceProgress(st.x, st.z);
  T.tracker.observe(st, routeIndex, T.p0, T.p1, T.obs);
  act(T.obs, T.a);
  const c = F.actionToCommand(T.a[0], T.a[1], 4.5, cfg.maxSteer);
  T.p0 = c.a0; T.p1 = c.a1;
  return c;
}
const bad = [];
let maxA = 0, maxB = 0, maxStateDiff = 0;
for (let i = 0; i < rows.length; i++) {
  const snap = sim.snapshot(), st = snap.state, ri = snap.metrics.route_index;
  const bst = i === 0 ? {...st, ...rows[0].pose_pre_action}
    : {...st, ...Object.fromEntries(['x', 'z', 'yaw', 'speed', 'steering'].map(k => [k, rows[i - 1].state_post_action[k]]))};
  for (const k of ['x', 'z', 'yaw', 'speed', 'steering']) maxStateDiff = Math.max(maxStateDiff, Math.abs(bst[k] - st[k]));
  step(A, st, ri, i); step(B, bst, ri, i);   // 観測の成分の比較用(features.mjs)
  const ca = tA.action({...snap, state: st}), cb = tB.action({...snap, state: bst}), rec = report.actions[i];
  const ea = Math.max(Math.abs(ca.steering - rec.steering), Math.abs(ca.speed - rec.speed));
  const eb = Math.max(Math.abs(cb.steering - rec.steering), Math.abs(cb.speed - rec.speed));
  maxA = Math.max(maxA, ea); maxB = Math.max(maxB, eb);
  if (ea > 1e-9 || eb > 1e-9) {
    const jumps = [];
    for (let k = 0; k < F.OBS_DIM; k++) { const d = A.obs[k] - B.obs[k]; if (Math.abs(d) > 1e-9) jumps.push({index: k, name: NAMES[k], node_replay: A.obs[k], browser_state: B.obs[k], diff: d}); }
    bad.push({frame: i, recorded: rec, from_node_replay_state: {steering: ca.steering, speed: ca.speed, err: ea},
      from_browser_state: {steering: cb.steering, speed: cb.speed, err: eb},
      state_diff: Object.fromEntries(['x', 'z', 'yaw', 'speed', 'steering'].map(k => [k, bst[k] - st[k]])), obs_jumps_node_vs_browser_state: jumps});
  }
  sim.step(rec);
}
const out = {schema: 'rl-bestlap-teacher-mismatch-probe/v1', scene: session.display_name, dataset: root, frames: rows.length,
  max_state_diff_browser_vs_node_replay: maxStateDiff, max_err_from_node_replay_state: maxA, max_err_from_browser_state: maxB, mismatched_frames: bad};
if (outPath) fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
