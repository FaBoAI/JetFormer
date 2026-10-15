#!/usr/bin/env node
// 正解の付け替え(1 シーン)— DAgger 用の写し(2026-10-15、sim_diverse_train_20261015。protocol.json dagger.procedure 5)。
// 10/14 の relabel/relabel_scene.mjs の写しで、変えたのは 3 つだけ:
//   (a) 乱しを乱数の決まりから作り直す代わりに、記録の info.dart を読む。info.dart.dagger があるフレームは「JetFormer の操作を再生した
//       フレーム(replay)」かどうかを舵・速さの両方の乱しの印にする(前回の行動 = 実際に適用した操作)。dagger が無いフレーム(10/14 の D1 など)は
//       元と同じく info.dart.noise の有無・倍率で決める
//   (b) 先生のファイルは今日の写しの dist/dagger/(dagger/prepare_record.py が web/ から写したもの。teacher_core・rt は 10/14 と同じ)
//   (c) 開始の約束: 開始が DAgger の計画にあれば dagger-teacher.js の検査(ゲート 0・5・15・24・31、±6 cm・±3°)、無ければ元の RC 教師の検査
// --noise は none | dagger | dart-recorded(記録の info.dart をそのまま読む)。
//
// 以下は 10/14 の元の説明:
//
//   node relabel_scene.mjs --scene <simulation/scenes/<id> のフォルダー> --teacher fast|margin [--noise N_fast|N_margin|none]
//        [--states recorded|replay] [--horizon 10] [--validity 60] [--reference-stride 0] [--out 結果.json]
//
// 決まり: 記録した全フレーム t(t < N − 10)の正解 future_actions[t, k](k = 0..9)を、本当の状態 s_{t+1} から先生だけで運転したときの
// 操作 â_{t+1+k} にする。s_{t+1} は s_t に実際に適用した(乱し入りの)操作 ã_t を 1 フレーム進めた状態 = 記録された次の状態。
//   1) 開始から記録の操作を元の sim-core.js で再生し、記録の状態(frames.jsonl の pose_pre_action・state_post_action)との差 ≤ 1e-9 を確かめる(R2)。
//      sim の route_index(目標ゲート)とゲートの数えは再生から取る(記録に無いため)
//   2) 先生(web/teacher_core.mjs、ブラウザと同じファイル)を開始から同じ順に進める。「前回の行動」は実際に適用した操作
//      (乱していない成分は先生自身の値。乱しの有無は記録の info.dart.noise。無ければ乱しなし)
//   3) 各 t で s_{t+1} から分かれ(fork)、先生だけで validity フレーム(60 = 2 秒)走らせる。最初の horizon(10)個を正解、
//      全体で 衝突・本当のゲートの外し(modes の gate_miss.mjs の決まり)・すき間 2 cm 未満 が無いことを「立て直せる」とする。
//      物理の分かれ方: sim-core の単周の reset({initialState: {x, z, yaw, speed, steering, routeIndex: 0}}) を、経路を今の目標ゲート
//      (再生の route_index)で回した world で行う(単周は周の最後のゲートで止まるため、60 フレームの中で周の終わりに届かないようにする)。
//      先生の分かれ方: teacher_core の fork()(begin / advance に渡した位置の列を新しい core に流し直す)。
//   --states recorded(既定): s_t は記録の状態(ブラウザが計算した値そのもの。画像はこの状態から描かれた)。replay: Node の再生の状態。
//   --reference-stride n(> 0): n フレームおきの t で「基準の作り方」(連続周回の sim を開始から t まで再生し、新しい先生を開始から t+1 まで
//      全部やり直し、そのまま先生だけで horizon 走らせる。O(N²))でも正解を作り、fork の正解と比べる(--states replay のとき差 0 のはず)。
// 出力(JSON): labels[t] = [δ_1, v_1, …, δ_10, v_10](SI 単位。PWM にするのは relabel_dataset.py が収集器の action_to_pwm で行う)、
//   valid_teacher[t]、確かめの数え。ユーザーのファイル・データセットは変えない(読むだけ)。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {baseWorld, activeWorldFor, createSimulation, SIM_ROOT} from '../lib/app_emulation.mjs';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i < 0 ? def : args[i + 1]; };
const sceneDir = opt('--scene');
const kind = opt('--teacher');
const noiseSpec = opt('--noise', 'none');
const statesMode = opt('--states', 'recorded');
const H = Number(opt('--horizon', 10)), V = Number(opt('--validity', 60));
const refStride = Number(opt('--reference-stride', 0));
const noFork = args.includes('--no-fork');   // 検証だけ(再生・先生の再計算・乱しの再計算・ゲートの外し)。正解は作らない
const outPath = opt('--out', null);
const MIN_CLEAR = Number(opt('--min-clearance', .02));
if (!sceneDir || !['fast', 'margin'].includes(kind) || !['recorded', 'replay'].includes(statesMode) || !['none', 'dagger', 'dart-recorded'].includes(noiseSpec)) {
  console.error('使い方: node relabel_scene.mjs --scene <dir> --teacher fast|margin [--noise N_fast|N_margin|none] [--states recorded|replay]');
  process.exit(2);
}
const DART = path.join(SIM_ROOT, 'dist/dagger');   // ブラウザが読んだファイルそのもの(prepare_record.py が web/ から写し、sha256 を確かめた)
const imp = p => import(pathToFileURL(path.join(DART, p)).href);
const {createTeacherCore, previousAction} = await imp('teacher_core.mjs');
const {GATE_MISS_THRESHOLD_M, realGateMisses} = await imp('gate_miss.mjs');
const W = (await imp(kind === 'fast' ? 'rt/weights_fast.mjs' : 'rt/weights_margin.mjs')).default;
const rc = await import(pathToFileURL(path.join(SIM_ROOT, 'dist/rc-speed-controller.js')).href);
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const TOOL_FILES = ['teacher_core.mjs', 'dagger-teacher.js', 'dagger_plan.mjs', 'gate_miss.mjs', 'rt/features.mjs', 'rt/obs_modes.mjs', 'rt/line_profiles.mjs', 'rt/policy_forward.mjs',
  kind === 'fast' ? 'rt/weights_fast.mjs' : 'rt/weights_margin.mjs'];

const read = p => JSON.parse(fs.readFileSync(path.join(sceneDir, p), 'utf8'));
const session = read('session.json'), report = read('evaluation-report.json');
const rows = fs.readFileSync(path.join(sceneDir, 'frames.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const N = rows.length;
const fail = msg => { const o = {schema: 'jf-sim-dart-relabel-scene/v1', pass: false, scene: path.basename(sceneDir), error: msg};
  if (outPath) fs.writeFileSync(outPath, JSON.stringify(o)); console.error('付け替え失敗: ' + msg); process.exit(1); };
if (report.actions.length !== N) fail('frames.jsonl と actions のフレーム数が違います');
const variation = session.context.start_variation;
const base = await baseWorld();
const {world} = activeWorldFor(base, variation.gate, variation.left_offset_m, variation.left_yaw_deg);
for (const k of ['x', 'z', 'yaw']) if (Math.abs(world.spawn[k] - rows[0].pose_pre_action[k]) > 1e-9) fail('開始姿勢が記録と違います: ' + k);
world.spawn = {...rows[0].pose_pre_action};   // verify_recording.mjs と同じく、記録の開始姿勢を正とする
const NG = world.route.length;

// ---- 1) 再生(R2) ----
const sim = createSimulation(world, report.config);
sim.reset({seed: report.seed, scenario: report.scenario});
const replay = [], ri = [], gcount = [];
const err = {pre: 0, post: 0, command: 0};
for (let i = 0; i < N; i++) {
  const b = sim.snapshot();
  for (const k of ['x', 'z', 'yaw']) err.pre = Math.max(err.pre, Math.abs(b.state[k] - rows[i].pose_pre_action[k]));
  for (const k of ['steering', 'speed']) err.command = Math.max(err.command, Math.abs(rows[i].command_si[k] - report.actions[i][k]));
  replay.push({...b.state}); ri.push(b.metrics.route_index); gcount.push(`${b.metrics.completed_laps ?? 0}:${b.metrics.passed_gates ?? 0}`);
  sim.step(report.actions[i]);
  const a = sim.snapshot().state;
  for (const k of ['x', 'z', 'yaw', 'speed', 'steering']) err.post = Math.max(err.post, Math.abs(a[k] - rows[i].state_post_action[k]));
}
{ const b = sim.snapshot(); replay.push({...b.state}); ri.push(b.metrics.route_index); gcount.push(`${b.metrics.completed_laps ?? 0}:${b.metrics.passed_gates ?? 0}`); }
const replayTol = 1e-9;
if (err.pre > replayTol || err.post > replayTol) fail(`記録の操作の再生が記録の状態と合いません(前 ${err.pre}・後 ${err.post})`);
if (err.command > 1e-12) fail('frames.jsonl の command_si と actions が違います');

// 先生に渡す状態 s_0..s_N
const pick = st => ({x: st.x, z: st.z, yaw: st.yaw, speed: st.speed, steering: st.steering, frame: st.frame});
const S = [];
for (let i = 0; i <= N; i++) {
  if (statesMode === 'replay') S.push(pick(replay[i]));
  // フレーム 0: 位置・向きは記録(ブラウザの sim が正規化した開始姿勢。Node の正規化とは最後の桁が違うことがある)、速さ・舵は再生(0)
  else if (i === 0) S.push({...pick(replay[0]), x: rows[0].pose_pre_action.x, z: rows[0].pose_pre_action.z, yaw: rows[0].pose_pre_action.yaw});
  else S.push(pick(rows[i - 1].state_post_action));
}
if (S[0].speed !== 0 || S[0].steering !== 0) fail('フレーム 0 の速さ・舵が 0 ではありません');

// ---- 乱しの記録(info.dart) ----
const dartInfo = rows.map(r => r.action_info?.dart ?? null);
const hasDart = dartInfo.some(x => x);
if (noiseSpec !== 'none' && !hasDart) fail('乱しありの指定ですが、記録に info.dart がありません');
if (rows.some(r => r.action_info?.jf)) fail('記録に action_info.jf があります(label_mask と混ざる)');
const dag = dartInfo.map(x => x?.dagger ?? null);
if (noiseSpec === 'dagger' && !dag.every(Boolean)) fail('DAgger の指定ですが、info.dart.dagger の無いフレームがあります');
const noisy = dartInfo.map((x, i) => dag[i] ? !!dag[i].replay : !!x?.noise);
const speedNoisy = dartInfo.map((x, i) => dag[i] ? !!dag[i].replay : (!!x?.noise && x.noise[1] !== 1));
// DAgger: 再生した操作 = 記録の操作(info.dart.dagger.applied と actions が同じ)
let daggerAppliedMax = 0;
for (let i = 0; i < N; i++) if (dag[i]) daggerAppliedMax = Math.max(daggerAppliedMax, Math.abs(dag[i].applied[0] - report.actions[i].steering), Math.abs(dag[i].applied[1] - report.actions[i].speed));
if (daggerAppliedMax > 1e-12) fail('info.dart.dagger.applied と記録の操作が違います: ' + daggerAppliedMax);
const p0 = dartInfo[0]?.p ?? null;

// ---- 2) 先生を開始から進める ----
const dgMod = await imp('dagger-teacher.js');
const vv = {gate: variation.gate, left_offset_m: variation.left_offset_m, left_yaw_deg: variation.left_yaw_deg};
const {DAGGER_PLAN} = await imp('dagger_plan.mjs');
const inPlan = !!DAGGER_PLAN.scenes[dgMod.startKey(vv)];
const actual = inPlan ? dgMod.makeSlot({original: rc}).assertRcTeacherContract(world, sim.snapshot().config, vv)
  : rc.assertRcTeacherContract(world, sim.snapshot().config, vv);
const core = createTeacherCore(kind, world, actual, W);
const nzRe = null;   // DAgger の写し: 乱しは作り直さず、記録の info.dart を読む
const r2 = {clean_vs_info_max: 0, clean_vs_applied_unnoisy_max: 0, noise_flag_mismatch: 0, noise_value_mismatch_max: 0, info_frames: 0};
const clean = [], prevAfter = [], sArr = [], dminArr = [];
const labels = [], validTeacher = [], forkStats = {first_label_vs_main_bitwise: 0, collisions: 0, gate_misses: 0, low_clearance: 0, early_end: 0,
  min_clearance_m: Infinity, invalid_frames: []};
let prev = [0, 0], lastGateS = 0;
const maxSteer = core.maxSteer, speedCap = core.speedCap;

function forkFrom(i, prevI, lastGateSAtI) {
  // s_i から先生だけで V フレーム。i = t + 1
  const fc = core.fork();
  const k0 = ri[i] % NG;
  const {world: wf} = activeWorldFor(base, (variation.gate + k0) % NG, 0, 0);
  const cfg = {...report.config, timeLimit: 600};
  delete cfg.episodeMode;
  const fs_ = createSimulation(wf, cfg);
  const s0 = S[i];
  const snap0 = fs_.reset({seed: report.seed, scenario: report.scenario, initialState: {x: s0.x, z: s0.z, yaw: s0.yaw, speed: s0.speed, steering: s0.steering, routeIndex: 0}});
  if (snap0.state.terminal) return {labels: [], ok: false, why: 'invalid_spawn'};
  let p = prevI, lastS = lastGateSAtI, lastG = 0, minD = Infinity, why = null;
  const lab = [];
  let first = null;
  for (let j = 0; j < V; j++) {
    const sn = j === 0 ? null : fs_.snapshot();
    const st = j === 0 ? s0 : pick(sn.state);
    const tri = j === 0 ? ri[i] : (k0 + sn.metrics.route_index) % NG;
    if (j > 0) fc.advance(st, tri);
    const d = fc.decide(st, p[0], p[1]);
    if (j === 0) first = d;
    if (j < H) lab.push(d.steering, d.speed);
    minD = Math.min(minD, d.dmin);
    const g = j === 0 ? 0 : sn.metrics.passed_gates;
    if (g !== lastG) { lastS = d.s; lastG = g; }
    if (d.s - lastS > GATE_MISS_THRESHOLD_M && !why) why = 'gate_miss';
    p = [d.a0, d.a1];
    fs_.step({steering: d.steering, speed: d.speed});
    const after = fs_.snapshot();
    if (after.metrics.collisions > 0) { why = why ?? 'collision'; break; }
    if (after.state.terminal) { why = why ?? ('terminal:' + after.state.reason); break; }
  }
  if (!why && minD < MIN_CLEAR) why = 'low_clearance';
  while (lab.length < 2 * H) lab.push(lab.at(-2) ?? 0, lab.at(-1) ?? 0);   // 早く終わった(立て直せない)ときの埋め。学習には使わない
  return {labels: lab, ok: !why, why, minD, first};
}

for (let i = 0; i < N; i++) {
  if (i === 0) core.begin(S[0], ri[0]); else core.advance(S[i], ri[i]);
  const d = core.decide(S[i], prev[0], prev[1]);
  clean.push(d); sArr.push(d.s); dminArr.push(d.dmin);
  if (i > 0 && gcount[i] !== gcount[i - 1]) lastGateS = d.s;
  if (i === 0) lastGateS = d.s;
  const applied = report.actions[i];
  if (dartInfo[i]) {
    r2.info_frames++;
    r2.clean_vs_info_max = Math.max(r2.clean_vs_info_max, Math.abs(d.steering - dartInfo[i].clean[0]), Math.abs(d.speed - dartInfo[i].clean[1]));
  }
  if (!noisy[i]) r2.clean_vs_applied_unnoisy_max = Math.max(r2.clean_vs_applied_unnoisy_max, Math.abs(d.steering - applied.steering), Math.abs(d.speed - applied.speed));
  if (nzRe) {
    const n = nzRe(i, d);
    if (!!n !== noisy[i]) r2.noise_flag_mismatch++;
    else if (n) r2.noise_value_mismatch_max = Math.max(r2.noise_value_mismatch_max, Math.abs(n.dSteer - dartInfo[i].noise[0]), Math.abs(n.speedFactor - dartInfo[i].noise[1]));
  }
  // 3) t = i − 1 の正解: s_i から分かれる(prev = フレーム i の観測に使う「前回の行動」 = 実際に適用した ã_{i−1} から)
  if (!noFork && i >= 1 && i - 1 < N - H) {
    const t = i - 1;
    const r = forkFrom(i, prev, lastGateS);
    if (r.first && r.first.steering === d.steering && r.first.speed === d.speed) forkStats.first_label_vs_main_bitwise++;
    labels[t] = r.labels;
    validTeacher[t] = r.ok;
    if (!r.ok) { forkStats.invalid_frames.push([t, r.why]); if (r.why === 'collision') forkStats.collisions++; else if (r.why === 'gate_miss') forkStats.gate_misses++; else if (r.why === 'low_clearance') forkStats.low_clearance++; else forkStats.early_end++; }
    if (r.minD != null) forkStats.min_clearance_m = Math.min(forkStats.min_clearance_m, r.minD);
  }
  prev = previousAction(d, applied, noisy[i], speedNoisy[i], speedCap, maxSteer);
  prevAfter.push(prev);
}
// t = N − H − 1 の分かれは i = N − H で済む(i < N)。残り(t ≥ N − H)は元のまま(学習に使わない)

// ---- 基準の作り方(O(N²))との比べ ----
let reference = null;
if (refStride > 0 && !noFork) {
  const diffs = [];
  for (let t = 0; t < N - H; t += refStride) {
    const sim2 = createSimulation(world, report.config);
    sim2.reset({seed: report.seed, scenario: report.scenario});
    const c2 = createTeacherCore(kind, world, actual, W);
    let p2 = [0, 0];
    for (let i = 0; i <= t; i++) {
      const sn = sim2.snapshot();
      if (i === 0) c2.begin(sn.state, sn.metrics.route_index); else c2.advance(sn.state, sn.metrics.route_index);
      const d = c2.decide(sn.state, p2[0], p2[1]);
      p2 = previousAction(d, report.actions[i], noisy[i], speedNoisy[i], speedCap, maxSteer);
      sim2.step(report.actions[i]);
    }
    const lab = [];
    for (let k = 0; k < H; k++) {
      const sn = sim2.snapshot();
      c2.advance(sn.state, sn.metrics.route_index);
      const d = c2.decide(sn.state, p2[0], p2[1]);
      lab.push(d.steering, d.speed);
      p2 = [d.a0, d.a1];
      sim2.step({steering: d.steering, speed: d.speed});
    }
    let m = 0;
    for (let q = 0; q < lab.length; q++) m = Math.max(m, Math.abs(lab[q] - labels[t][q]));
    diffs.push([t, m]);
  }
  reference = {stride: refStride, compared_frames: diffs.length, max_abs_diff_si: Math.max(...diffs.map(x => x[1])),
    frames_over_1e9: diffs.filter(x => x[1] > 1e-9).length, worst: diffs.sort((a, b) => b[1] - a[1]).slice(0, 5)};
}

// 乱しの印と正解: 乱していないフレーム t+1 では labels[t][0..1] = 記録の操作(R3 の SI 版。PWM 版は relabel_dataset.py)
const r3 = {unnoisy_next: 0, unnoisy_next_max_si: 0, noisy_next: 0, noisy_next_mean_abs_steer: 0};
for (let t = 0; t < N - H && !noFork; t++) {
  const a = report.actions[t + 1];
  const e = Math.max(Math.abs(labels[t][0] - a.steering), Math.abs(labels[t][1] - a.speed));
  if (!noisy[t + 1]) { r3.unnoisy_next++; r3.unnoisy_next_max_si = Math.max(r3.unnoisy_next_max_si, e); }
  else { r3.noisy_next++; r3.noisy_next_mean_abs_steer += Math.abs(labels[t][0] - a.steering); }
}
if (r3.noisy_next) r3.noisy_next_mean_abs_steer /= r3.noisy_next;

const out = {schema: 'jf-sim-diverse-dagger-relabel-scene/v1', pass: true, scene: path.basename(sceneDir), session_id: session.session_id ?? null,
  display_name: session.display_name ?? null, teacher: kind, noise: noiseSpec, states: statesMode, horizon: H, validity_frames: V, min_clearance_m: MIN_CLEAR,
  frames: N, labelled_frames: N - H, noise_params: p0, noisy_frames: noisy.filter(Boolean).length, dagger_frames: dag.filter(Boolean).length,
  dagger_replay_frames: dag.filter(x => x?.replay).length, dagger_applied_vs_actions_max: daggerAppliedMax, contract: inPlan ? 'dagger' : 'rc',
  replay_max_abs_error: err, r2, r3, no_fork: noFork,
  main_gate_misses: realGateMisses(sArr, gcount.slice(0, N)), main_min_clearance_m: Math.min(...dminArr),
  collisions_replay: sim.snapshot().metrics.collisions, outcome_replay: sim.snapshot().state.reason, fork: {...forkStats, invalid_count: forkStats.invalid_frames.length}, reference,
  teacher_id: session.context?.teacher?.id ?? null,
  tool: {relabel_scene_sha256: sha(new URL(import.meta.url).pathname), files_sha256: Object.fromEntries(TOOL_FILES.map(f => [f, sha(path.join(DART, f))]))},
  labels: labels, valid_teacher: validTeacher,
  clean_actions: clean.map(d => [d.steering, d.speed])};
if (outPath) fs.writeFileSync(outPath, JSON.stringify(out));
const brief = {...out}; delete brief.labels; delete brief.valid_teacher; delete brief.clean_actions; brief.fork = {...out.fork, invalid_frames: out.fork.invalid_frames.slice(0, 10)};
console.log(JSON.stringify(brief));
