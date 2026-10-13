// 外部の方策(強化学習など)を、アプリの「RC加速教師」の差し込み口で走らせる仕組みの自己テスト(2026-10-13)
//
// コピーした dist の rl.html の import map で、app.js が読む './rc-speed-controller.js' をこのファイルに差し替える。
// ユーザーのファイル(app.js・sim-core.js・dataset_recorder.py など)は一切変えないので、収集データの形式はアプリ本来のもの。
//
// 自己テストの方策: 元の RC 教師の操舵をそのまま使い、速度指令だけを SELFTEST_POLICY.speed_scale 倍する
//   (元の教師と「違う操作」が、アプリ自身の収集器で PWM ラベルとして記録されることを確かめるため)。
// 開始の制限: 元は「ゲート0・左右±3cm・向き±3°」。ここでは画面の入力範囲(左右±0.3m・向き±25°)まで広げる
//   (ずれた開始からの立て直しを記録できることの確認)。ゲート0以外は、元の教師の参照軌跡の探索が
//   ゲート0から始まるため、この自己テストでは扱わない(RL の方策は経路全体から現在位置を探すので任意のゲートでよい)。
//
// アプリがこの差し込み口に求めること(app.js を読んで確認):
//   - createRcTrainingTeacher(world, config, {gate, left_offset_m, left_yaw_deg}) が {action, reference, options, metadata} を返す
//     ・action(snapshot) → {steering [rad], speed [m/s]}。snapshot は sim.snapshot()(正解の位置・速度・ゲート番号を含む)。
//       観測(カメラ・ToF)の取得の後、同じフレームの操作として sim.step() に渡され、そのまま PWM ラベルになる。
//     ・reference.length / reference.planned_time_s(表示と記録用。無いと reset で例外)
//     ・metadata は context.teacher に展開されて session.json に入る。集約 ZIP は全シーンで context.teacher が
//       同じことを要求するので、シーンごとに変わる値(開始姿勢など)は入れない。
//     ・単周モードの動的な初期状態(initial-state)を使うときは seek(snapshot) も要る(この自己テストは非対応)。
//   - assertRcTeacherContract(world, config, variation) は、連続試験の計画の検証(batch-run)でも呼ばれる。
//   - RC_COLLECTION_PLAN(「20分・60シーンの収集条件を用意」)は元のものをそのまま再エクスポートする。
import * as original from '../rc-speed-controller.js';
import {ROUTE} from '../world.js';
export * from '../rc-speed-controller.js';

export const SELFTEST_POLICY = Object.freeze({speed_scale: 0.9});
export const RL_TEACHER_METADATA = Object.freeze({
  schema: 'course-external-teacher-slot/v1',
  id: 'rl-wrapper-selftest-v1',
  controller_kind: 'scripted_selftest_in_teacher_slot',
  not_learned_policy: true,
  uses_ground_truth: true,
  world_version: original.RC_TEACHER_METADATA.world_version,
  base_teacher_id: original.RC_TEACHER_METADATA.id,
  base_reference_sha256: original.RC_TEACHER_METADATA.base_reference_sha256,
  policy: SELFTEST_POLICY,
  start_limits: {gate: [0], left_offset_m: [-0.3, 0.3], left_yaw_deg: [-25, 25]},
  note: '自己テスト: 元のRC教師の操舵 + 速度指令×0.9。差し替えの仕組み(rl.html の import map)の確認用で、学習データには使わない。',
});

const calls = {createRcTrainingTeacher: 0, assertRcTeacherContract: 0, actions: 0};
globalThis.__rlTeacherProbe = {loaded: true, module: 'rl-teacher-selftest.js', calls, metadata_id: RL_TEACHER_METADATA.id};

const headingOf = route => Math.atan2(-(route[1][1] - route[0][1]), route[1][0] - route[0][0]);

// 元の契約(コース形状・33ゲート・物理・雑音・20秒)を確かめるための「標準の開始姿勢」の world
function canonicalWorld(world) {
  const route = ROUTE.map(p => [...p]);
  return {...world, route, spawn: {x: route[0][0], z: route[0][1], yaw: headingOf(route)}};
}

export function assertRcTeacherContract(world, config, variation = {}) {
  calls.assertRcTeacherContract++;
  if (Object.hasOwn(variation, 'initialState')) throw Error('この差し替え教師(自己テスト)は途中状態からの開始に対応していません。');
  const gate = variation.gate ?? 0, left = variation.left_offset_m ?? 0, yaw = variation.left_yaw_deg ?? 0;
  if (gate !== 0) throw Error('この差し替え教師(自己テスト)の開始はゲート0だけです。');
  if (!Number.isFinite(left) || Math.abs(left) > .3 + 1e-12 || !Number.isFinite(yaw) || Math.abs(yaw) > 25 + 1e-9)
    throw Error('この差し替え教師(自己テスト)の開始ずれは左右±0.3m・向き±25°以内です。');
  // コース・物理などは元の検査をそのまま使う(開始姿勢だけ標準に置き換えて渡す)
  const actual = original.assertRcTeacherContract(canonicalWorld(world), config, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
  if (JSON.stringify(world.route) !== JSON.stringify(ROUTE)) throw Error('コースの経路(ゲートの並び)が標準と違います。');
  const heading = headingOf(world.route), first = world.route[0];
  const expected = {x: first[0] + left * Math.sin(heading), z: first[1] - left * Math.cos(heading), yaw: heading + yaw * Math.PI / 180};
  for (const k of ['x', 'z', 'yaw'])
    if (!Number.isFinite(world.spawn?.[k]) || Math.abs(world.spawn[k] - expected[k]) > 1e-9) throw Error('開始姿勢と開始ずれの記録が一致しません: ' + k);
  return actual;
}

export function createRcTrainingTeacher(world, config, variation = {}) {
  calls.createRcTrainingTeacher++;
  const actual = assertRcTeacherContract(world, config, variation);
  // 元の教師の参照軌跡は開始姿勢に依存しない(参照はゲート0の位置から作る)ので、標準の開始姿勢で作って使う
  const base = original.createRcTrainingTeacher(canonicalWorld(world), config, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
  const scale = SELFTEST_POLICY.speed_scale;
  function action(snapshot) {
    calls.actions++;
    const a = base.action(snapshot);
    return {steering: a.steering, speed: a.speed * scale};
  }
  const continuous = actual.episodeMode === 'continuous-laps-v1';
  return {
    action,
    // 表示と記録に使う最小限(元の4周ぶんの参照軌跡 約1.1MiB は context に入れない)
    reference: {schema: 'rl-wrapper-selftest-reference/v1', length: base.reference.length, planned_time_s: base.reference.planned_time_s},
    options: {...base.options, selftest_speed_scale: scale},
    metadata: {...RL_TEACHER_METADATA, episode_mode: continuous ? 'continuous-laps-v1' : 'single-lap-v1'},
  };
}
