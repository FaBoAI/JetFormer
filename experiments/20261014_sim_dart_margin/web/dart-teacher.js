// DART・余裕の線の先生を、アプリの「RC加速教師」の差し込み口で走らせる(2026-10-14、sim_dart_20261014 の準備の段)
//
// ベストラップの rl-teacher-bestlap.js(record の段)を写して広げたもの。違いは 3 つ:
//   1. 先生は teacher_core.mjs(fast = ベストラップの RL 方策と同じ計算、margin = 経路いろいろ line r2 を c0 = +0.3 で走らせる計算)
//   2. DART の乱し(dart_noise.mjs)を先生の操作に足す。乱しは (シーンの種, データセットの塩, 引き直しの回 k) とフレームごとの状態だけで決まる
//   3. 毎フレーム info.dart = {v, f, clean: [δ, v], noise: [dδ, 倍率] | null, ev} を返す(アプリが frames.jsonl の action_info に残す)。
//      jf は出さない(label_mask.py の印と混ざらないように)。フレーム 0 だけ info.dart.p にシーンの乱しの条件を入れる。
// 入口(entry-D1.js・entry-D2.js・entry-D3.js)が makeSlot(…) を呼び、app.js が読む createRcTrainingTeacher・assertRcTeacherContract を出す。
// 開始・物理の約束は元の RC 教師の検査(assertRcTeacherContract)をそのまま使う(ゲート 0・±3 cm・±3°・v3 コース・RC 応答の物理・20 秒)。
// 乱数・時刻は使わない。同じフレームで 2 回呼ばれたら同じ値。フレームが飛んだら例外。
import {createDartDriver, DART_DRIVER_VERSION} from './dart_driver.mjs';
import {TEACHER_CORE_VERSION, SPEED_CAP} from './teacher_core.mjs';
import {DART_NOISE_VERSION, NOISE_SPECS} from './dart_noise.mjs';
import {FEATURES_VERSION, REFERENCE_LOOP_SHA256, LOOP_LENGTH} from './rt/features.mjs';
import {DART_MANIFEST} from './dart_manifest.mjs';

// dataset: 'D1' | 'D2' | 'D3'。kind: 'fast' | 'margin'。noiseSpec: 'N_fast' | 'N_margin' | null。kTable: {シーンの種: {k, scale}}(記録の前の Node の予想で決めたもの)
export function makeSlot({dataset, kind, weights, noiseSpec = null, salt = null, kTable = {}, original}) {
  const D = DART_MANIFEST.datasets[dataset];
  if (!D || D.kind !== kind || (D.noise ?? null) !== noiseSpec || (D.salt ?? null) !== salt) throw new Error('DART の先生: 入口の設定が manifest と違います: ' + dataset);
  // context.teacher に入る識別(全シーンで同じ。集約 ZIP は全シーンの context.teacher が同じことを求める)。シーンごとの k は info.dart.p に入れる
  const METADATA = Object.freeze({
    schema: 'course-external-teacher-slot/v1',
    id: D.teacher_id,
    controller_kind: 'rl_policy_ground_truth_observation',
    not_learned_policy: false,
    learned_policy_kind: 'reinforcement-learning-ppo',
    uses_ground_truth: true,
    world_version: original.RC_TEACHER_METADATA.world_version,
    teacher: kind === 'fast' ? 'T_fast' : 'T_margin',
    teacher_params: kind === 'margin' ? {line: {family: 'const', c0: .3}, c_v: 1.0, line_profile: 'v2'} : null,
    policy_sha256: D.policy_sha256,
    weights_sha256: D.weights_sha256,
    noise: noiseSpec ? {...NOISE_SPECS[noiseSpec], salt} : null,
    dart_versions: {driver: DART_DRIVER_VERSION, core: TEACHER_CORE_VERSION, noise: DART_NOISE_VERSION},
    files_sha256: DART_MANIFEST.files,
    features_version: FEATURES_VERSION,
    reference_loop_sha256: REFERENCE_LOOP_SHA256,
    action_limits: {steering_rad: .52, speed_mps: [0, SPEED_CAP]},
    control_delay_frames: 0,
    protocol_sha256: DART_MANIFEST.protocol_sha256,
    start_limits: {gate: [0], left_offset_m: [-.03, .03], left_yaw_deg: [-3, 3]},
    label_note: noiseSpec ? '記録の操作は乱し入り。学習の正解は取り込みのあと先生の操作に付け替える(relabel)' : '乱しなし。記録の操作 = 先生の操作',
    note: D.note,
  });
  const calls = {createRcTrainingTeacher: 0, assertRcTeacherContract: 0, actions: 0};
  globalThis.__rlTeacherProbe = {loaded: true, module: 'dart-teacher.js', dataset, calls, metadata_id: METADATA.id, policy_sha256: METADATA.policy_sha256};
  globalThis.__dartLog = globalThis.__dartLog ?? {};
  // ブラウザの取り直しで k を渡す口。WebDriver の実行スクリプトが作ったオブジェクトはページのコードから読めないことがある
  // (準備の段の試しで、globalThis.__dartOverride に直接入れた k が効かなかった)ので、JSON の文字列を受け取り、ページの側でオブジェクトにする
  globalThis.__dartSetOverride = json => { globalThis.__dartOverride = JSON.parse(String(json)); return JSON.stringify(globalThis.__dartOverride); };

  function assertRcTeacherContract(world, config, variation = {}) {
    calls.assertRcTeacherContract++;
    return original.assertRcTeacherContract(world, config, variation);
  }
  function createRcTrainingTeacher(world, config, variation = {}) {
    calls.createRcTrainingTeacher++;
    const actual = assertRcTeacherContract(world, config, variation);
    const seed = actual.seed >>> 0;
    // 引き直しの回: ブラウザの取り直し(globalThis.__dartOverride)→ 計画の表 → 0
    const pick = noiseSpec ? (globalThis.__dartOverride?.[seed] ?? kTable[seed] ?? {k: 0, scale: 1}) : null;
    const noise = noiseSpec ? {spec: noiseSpec, seed, salt, k: pick.k ?? 0, scale: pick.scale ?? 1} : null;
    const driver = createDartDriver({kind, world, config: actual, weights, noise});
    // シーンの記録(browser_collect.py が連続試験のあとに読み、シーンごとの横の記録 sidecar に書く。log は Node の予想でも使う)
    const entry = {dataset, seed, noise, created: calls.createRcTrainingTeacher, frames: 0, noisy_frames: 0, events: driver.events, log: driver.log};
    globalThis.__dartLog[seed] = entry;
    function action(snapshot) {
      const before = driver.log.length;
      const out = driver.action(snapshot);
      if (out.info.dart.f === 0) out.info.dart.p = {dataset, seed, salt, k: noise?.k ?? null, scale: noise?.scale ?? null, spec: noiseSpec, teacher: kind};
      if (driver.log.length > before) { entry.frames = driver.log.length; if (out.info.dart.noise) entry.noisy_frames++; }
      calls.actions++;
      return out;
    }
    const continuous = actual.episodeMode === 'continuous-laps-v1';
    return {
      action,
      reference: {schema: 'jf-sim-dart-reference/v1', length: LOOP_LENGTH, planned_time_s: D.planned_single_lap_s},
      options: {policy: METADATA.id, teacher: METADATA.teacher, speed_cap_mps: SPEED_CAP, max_steer_rad: actual.maxSteer, deterministic: true,
        observation: kind === 'fast' ? 'privileged-ground-truth-79' : 'privileged-ground-truth-cond97', control_delay_frames: 0, noise: noiseSpec},
      metadata: {...METADATA, episode_mode: continuous ? 'continuous-laps-v1' : 'single-lap-v1'},
    };
  }
  return {assertRcTeacherContract, createRcTrainingTeacher, METADATA};
}
