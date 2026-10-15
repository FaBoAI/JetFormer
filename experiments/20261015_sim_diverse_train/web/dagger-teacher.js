// DAgger の記録の差し込み(2026-10-15、sim_diverse_train_20261015。protocol.json dagger.procedure 4)
//
// 10/14 の dart-teacher.js(DART の先生の差し込み)を写して作り替えたもの。アプリの「RC加速教師」の差し込み口で走り、1 シーンごとに:
//   フレーム f < k: 学習した JetFormer が閉ループの評価(状態集め D0〜D11、遅延 1)で実際に適用した操作をそのまま再生する
//                   (遅延 0 で同じ操作の列を与えるので、状態は状態集めの走行と同じになる。G2 で確かめる)
//   フレーム f ≥ k: 先生(dagger_plan.mjs の teacher。T_fast か T_margin。teacher_core.mjs は 10/14 の写しのまま)が 20 秒まで運転する(乱しは足さない)
// 先生はフレーム 0 から毎フレーム観測を進める(f < k でも)。f < k の「前回の行動」は実際に適用した(JetFormer の)操作。
// 毎フレーム info.dart = {v, f, clean: [先生のきれいな操作 δ, v], noise: [適用 − きれい の舵, 適用 / きれい の速さ] | null, ev, dagger: {replay, k, applied}}
// を返す(アプリが frames.jsonl の action_info に残す。付け替えの写しは dagger.applied と replay を読む)。jf は出さない。
// 開始の約束: 元の RC 教師の検査(assertRcTeacherContract)で、コースの形・物理・20 秒を確かめる(経路をゲート 0 の並びに戻して渡す)。
//   開始はゲート 0・5・15・24・31、左右 ±6 cm・向き ±3° の中で、dagger_plan.mjs にある開始だけを受け付ける。
// 乱数・時刻は使わない。同じフレームで 2 回呼ばれたら同じ値。フレームが飛んだら例外。
import {createTeacherCore, previousAction, SPEED_CAP, TEACHER_CORE_VERSION} from './teacher_core.mjs';
import {FEATURES_VERSION, REFERENCE_LOOP_SHA256, LOOP_LENGTH} from './rt/features.mjs';
import {DAGGER_PLAN} from './dagger_plan.mjs';
import WF from './rt/weights_fast.mjs';
import WM from './rt/weights_margin.mjs';

export const DAGGER_TEACHER_VERSION = 'jf-sim-diverse-dagger-teacher/v1';
const W = {fast: WF, margin: WM};
export const startKey = v => `${v.gate ?? 0}|${(+(v.left_offset_m ?? 0)).toFixed(4)}|${(+(v.left_yaw_deg ?? 0)).toFixed(4)}`;
const GATES = [0, 5, 15, 24, 31];

export function makeSlot({original}) {
  const kind = DAGGER_PLAN.teacher;
  if (!['fast', 'margin'].includes(kind)) throw new Error('DAgger: 先生の指定が違います: ' + kind);
  const METADATA = Object.freeze({
    schema: 'course-external-teacher-slot/v1',
    id: DAGGER_PLAN.teacher_id,
    controller_kind: 'rl_policy_ground_truth_observation',
    not_learned_policy: false,
    learned_policy_kind: 'reinforcement-learning-ppo',
    uses_ground_truth: true,
    world_version: original.RC_TEACHER_METADATA.world_version,
    teacher: kind === 'fast' ? 'T_fast' : 'T_margin',
    teacher_params: kind === 'margin' ? {line: {family: 'const', c0: .3}, c_v: 1.0, line_profile: 'v2'} : null,
    weights_sha256: DAGGER_PLAN.weights_sha256,
    dagger: {source_arm: DAGGER_PLAN.source_arm, source_onnx_sha256: DAGGER_PLAN.source_onnx_sha256, plan_sha256: DAGGER_PLAN.plan_sha256,
      versions: {teacher: DAGGER_TEACHER_VERSION, core: TEACHER_CORE_VERSION}},
    features_version: FEATURES_VERSION,
    reference_loop_sha256: REFERENCE_LOOP_SHA256,
    action_limits: {steering_rad: .52, speed_mps: [0, SPEED_CAP]},
    control_delay_frames: 0,
    protocol_sha256: DAGGER_PLAN.protocol_sha256,
    start_limits: {gate: GATES, left_offset_m: [-.06, .06], left_yaw_deg: [-3, 3]},
    label_note: 'フレーム k までは学習した JetFormer の操作(状態集めの再生)。学習の正解は取り込みのあと先生の操作に付け替える(relabel)',
  });
  const calls = {createRcTrainingTeacher: 0, assertRcTeacherContract: 0, actions: 0};
  globalThis.__rlTeacherProbe = {loaded: true, module: 'dagger-teacher.js', calls, metadata_id: METADATA.id, teacher: kind};
  globalThis.__daggerLog = globalThis.__daggerLog ?? {};

  function assertRcTeacherContract(world, config, variation = {}) {
    calls.assertRcTeacherContract++;
    const g = variation.gate ?? 0, left = variation.left_offset_m ?? 0, yaw = variation.left_yaw_deg ?? 0;
    if (!GATES.includes(g) || !Number.isFinite(left) || Math.abs(left) > .06 + 1e-12 || !Number.isFinite(yaw) || Math.abs(yaw) > 3 + 1e-12)
      throw new Error('DAgger の開始はゲート 0・5・15・24・31、左右 ±6 cm・向き ±3° 以内です。');
    if (!DAGGER_PLAN.scenes[startKey(variation)]) throw new Error('DAgger: 計画にない開始です: ' + startKey(variation));
    // 経路をゲート 0 の並びにして、開始をゲート 0 の真ん中にし、元の検査(コースの形・物理・20 秒)に通す。
    // アプリは 2 か所で呼ぶ: (1) 連続試験の前の計画の確かめ(app.js 507 行)は、経路を回していない元の world(開始姿勢は元の経路の点 0 から作る)と
    // variation.gate を渡す。(2) 各条件の reset のあとは、経路を開始ゲートで回した world を渡す。どちらでも通るよう、まずそのまま、
    // だめなら回した経路を元に戻して確かめる(2026-10-15 14:50、ブラウザの記録でゲート 5 の開始が (1) で止まったので直した)
    const n = world.route.length;
    const std = route0 => { const a0 = route0[0], b0 = route0[1]; return {x: a0[0], z: a0[1], yaw: Math.atan2(-(b0[1] - a0[1]), b0[0] - a0[0])}; };
    const check0 = route0 => original.assertRcTeacherContract({...world, route: route0, spawn: std(route0)}, config, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
    let actual;
    try {
      actual = check0(world.route);
    } catch (e) {
      if (g === 0) throw e;
      actual = check0([...world.route.slice(n - g), ...world.route.slice(0, n - g)]);
    }
    // 実際の開始姿勢が、開始のゲート・ずれから作ったものと同じか(app.js の reset と同じ式)
    const first = world.route[0], next = world.route[1], heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
    const expected = {x: first[0] + left * Math.sin(heading), z: first[1] - left * Math.cos(heading), yaw: heading + yaw * Math.PI / 180};
    for (const k of ['x', 'z', 'yaw'])
      if (!Number.isFinite(world.spawn?.[k]) || Math.abs(world.spawn[k] - expected[k]) > 1e-12) throw new Error('DAgger: 開始姿勢とメタデータが一致しません: ' + k);
    return actual;
  }

  function createRcTrainingTeacher(world, config, variation = {}) {
    calls.createRcTrainingTeacher++;
    const actual = assertRcTeacherContract(world, config, variation);
    const key = startKey(variation), sc = DAGGER_PLAN.scenes[key];
    const core = createTeacherCore(kind, world, actual, W[kind]);
    let lastF = -1, memo = null, prevA = [0, 0];
    const entry = {key, name: sc.name, k: sc.k, frames: 0, replay_frames: 0, created: calls.createRcTrainingTeacher, log: []};
    globalThis.__daggerLog[key] = entry;
    function action(snapshot) {
      const st = snapshot.state, f = st.frame, ri = snapshot.metrics?.route_index ?? 0;
      if (f === lastF && memo) return JSON.parse(JSON.stringify(memo));
      if (f === 0 && lastF === -1) core.begin(st, ri);
      else if (f === lastF + 1) core.advance(st, ri);
      else throw new Error(`DAgger の先生: フレームが飛びました(前 ${lastF} → 今 ${f})。`);
      const clean = core.decide(st, prevA[0], prevA[1]);
      const replay = f < sc.k;
      let cmd;
      if (replay) {
        const a = sc.actions[f];
        if (!a) throw new Error(`DAgger: フレーム ${f} の再生の操作がありません(k = ${sc.k}、操作 ${sc.actions.length})`);
        cmd = {steering: a[0], speed: a[1]};
        prevA = previousAction(clean, cmd, true, true, core.speedCap, core.maxSteer);
      } else {
        cmd = {steering: clean.steering, speed: clean.speed};
        prevA = [clean.a0, clean.a1];
      }
      const noise = replay ? [cmd.steering - clean.steering, clean.speed > 0 ? cmd.speed / clean.speed : null] : null;
      const info = {dart: {v: 1, f, clean: [clean.steering, clean.speed], noise, ev: replay ? 0 : -1,
        dagger: {replay, k: sc.k, applied: [cmd.steering, cmd.speed]}}};
      if (f === 0) info.dart.p = {dagger: true, start: key, name: sc.name, k: sc.k, teacher: kind, source_arm: DAGGER_PLAN.source_arm};
      memo = {steering: cmd.steering, speed: cmd.speed, info};
      entry.frames = f + 1;
      if (replay) entry.replay_frames = f + 1;
      entry.log.push({f, x: st.x, z: st.z, yaw: st.yaw, speed: st.speed, replay, steering: cmd.steering, cmd_speed: cmd.speed,
        clean_steering: clean.steering, clean_speed: clean.speed, s: clean.s, dmin: clean.dmin, g: `${snapshot.metrics?.completed_laps ?? 0}:${snapshot.metrics?.passed_gates ?? 0}`});
      lastF = f;
      calls.actions++;
      return JSON.parse(JSON.stringify(memo));
    }
    const continuous = actual.episodeMode === 'continuous-laps-v1';
    return {
      action,
      reference: {schema: 'jf-sim-dagger-reference/v1', length: LOOP_LENGTH, planned_time_s: DAGGER_PLAN.planned_single_lap_s},
      options: {policy: METADATA.id, teacher: METADATA.teacher, speed_cap_mps: SPEED_CAP, max_steer_rad: actual.maxSteer, deterministic: true,
        observation: kind === 'fast' ? 'privileged-ground-truth-79' : 'privileged-ground-truth-cond97', control_delay_frames: 0},
      // シーンごとの k は options に入れない(context.teacher に入り、集約 ZIP は全シーンの context.teacher が同じことを求めるため。k は info.dart.p・dagger にある)
      metadata: {...METADATA, episode_mode: continuous ? 'continuous-laps-v1' : 'single-lap-v1'},
    };
  }
  return {assertRcTeacherContract, createRcTrainingTeacher, METADATA};
}
