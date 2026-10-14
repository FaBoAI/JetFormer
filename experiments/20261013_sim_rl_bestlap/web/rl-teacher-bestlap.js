// 強化学習(PPO)のベストラップの方策を、アプリの「RC加速教師」の差し込み口で走らせる(2026-10-13、記録の段)
//
// コピーした dist の rl.html の import map で、app.js が読む './rc-speed-controller.js' をこのファイルに差し替える
// (install は record/install_bestlap.py)。ユーザーのファイル(app.js・sim-core.js・dataset_recorder.py など)は一切変えないので、
// 収集データの形式・集約 ZIP はアプリ本来のもの。違うのは「誰が運転したか」(教師の操作)だけ。
//
// 方策: rl/best/main(main 種 1・10,010,624 ステップ。protocol.json の選び方で選んだもの)。
//   観測は ./features.mjs(RL の Node 環境と同じファイル)で正解の状態から 79 次元を作り、./policy_forward.mjs で
//   重み ./policy_weights.mjs の MLP を計算する(展開時の平均 = 決定的)。行動 a ∈ [−1, 1]² → 舵 0.52·a0 [rad]・速さ 4.5·(a1+1)/2 [m/s]。
// RL の環境(rl/env_core.mjs RlEnv、制御遅延 0)と同じ順番で計算する:
//   フレーム 0: resetProgress(x, z, route_index) → observe(前回の行動 0, 0)
//   フレーム t: advanceProgress(x, z) → observe(前回の行動 = フレーム t−1 の a(切ったあとの値))
// 決定的にするための約束: 同じフレーム番号で 2 回呼ばれたら同じ値を返す(フレーム番号で覚える)。フレームが飛んだら例外
// (アプリは 1 フレームずつ進めるので起きない。起きたら記録が方策と合わなくなるので止める)。乱数・時刻は使わない。
//
// 開始の約束は元の RC 教師と同じ(ゲート 0・左右 ±3 cm・向き ±3°・v3 コース・RC 応答の物理・雑音 0 / 25 mm・20 秒)。
// assertRcTeacherContract は元の関数をそのまま呼ぶ(記録の 60 条件は元の計画と同じ開始)。
import * as original from '../rc-speed-controller.js';
import {createFeatureTracker, actionToCommand, OBS_DIM, FEATURES_VERSION, REFERENCE_LOOP_SHA256, LOOP_LENGTH} from './features.mjs';
import {createPolicy} from './policy_forward.mjs';
import WEIGHTS from './policy_weights.mjs';
import {BESTLAP_MANIFEST} from './bestlap_manifest.mjs';
export * from '../rc-speed-controller.js';

const SPEED_CAP = 4.5;   // main の速度の上限(protocol.json action.speed_cap_mps)

// context.teacher に入る識別(全シーンで同じ。集約 ZIP は全シーンの context.teacher が同じことを求める)
export const RL_TEACHER_METADATA = Object.freeze({
  schema: 'course-external-teacher-slot/v1',
  id: 'rl-bestlap-ppo-v1',
  controller_kind: 'rl_policy_ground_truth_observation',
  // 元の RC 教師は not_learned_policy: true(台本の追従器)。この教師は強化学習で学習した方策なので false にする。
  // ただし観測は正解の状態(位置・向き・コースの形)で、カメラは使わない(JetFormer の学習済み AI とは別物)
  not_learned_policy: false,
  learned_policy_kind: 'reinforcement-learning-ppo',
  uses_ground_truth: true,
  world_version: original.RC_TEACHER_METADATA.world_version,
  variant: BESTLAP_MANIFEST.variant,
  run: BESTLAP_MANIFEST.run,
  seed: BESTLAP_MANIFEST.seed,
  checkpoint_steps: BESTLAP_MANIFEST.checkpoint_steps,
  policy_sha256: BESTLAP_MANIFEST.policy_json_sha256,
  weights_sha256: BESTLAP_MANIFEST.files['policy_weights.mjs'],
  features_sha256: BESTLAP_MANIFEST.files['features.mjs'],
  policy_forward_sha256: BESTLAP_MANIFEST.files['policy_forward.mjs'],
  teacher_module_sha256: BESTLAP_MANIFEST.files['rl-teacher-bestlap.js'],
  features_version: FEATURES_VERSION,
  reference_loop_sha256: REFERENCE_LOOP_SHA256,
  reward_version: BESTLAP_MANIFEST.reward_version,
  action_limits: {steering_rad: .52, speed_mps: [0, SPEED_CAP]},
  control_delay_frames: 0,
  protocol_sha256: BESTLAP_MANIFEST.protocol_sha256,
  start_limits: {gate: [0], left_offset_m: [-.03, .03], left_yaw_deg: [-3, 3]},
  note: '強化学習(PPO)のベストラップの方策。観測は正解の状態(79 次元)で、カメラ・ToF は使わない。記録のラベルは方策の操作そのもの(遅延 0)。',
});

const calls = {createRcTrainingTeacher: 0, assertRcTeacherContract: 0, actions: 0, memo_hits: 0};
globalThis.__rlTeacherProbe = {loaded: true, module: 'rl-teacher-bestlap.js', calls, metadata_id: RL_TEACHER_METADATA.id,
  policy_sha256: RL_TEACHER_METADATA.policy_sha256};

// コース・物理・開始の検査は元の RC 教師のものをそのまま使う(範囲も同じ)
export function assertRcTeacherContract(world, config, variation = {}) {
  calls.assertRcTeacherContract++;
  return original.assertRcTeacherContract(world, config, variation);
}

export function createRcTrainingTeacher(world, config, variation = {}) {
  calls.createRcTrainingTeacher++;
  const actual = assertRcTeacherContract(world, config, variation);
  const act = createPolicy(WEIGHTS);
  if (WEIGHTS.obs_dim !== OBS_DIM) throw Error('方策の入力の次元が観測と違います');
  // 観測の物差し(車体の大きさ・軸距・摩擦の倍率)は sim の正規化済みの設定(RL の環境と同じ view.config)
  const tracker = createFeatureTracker(world, actual);
  const obs = new Float64Array(OBS_DIM), a = [0, 0];
  let lastFrame = -1, lastCommand = null, prevA0 = 0, prevA1 = 0;
  function action(snapshot) {
    const st = snapshot.state, frame = st.frame, routeIndex = snapshot.metrics.route_index;
    if (frame === lastFrame) { calls.memo_hits++; return {...lastCommand}; }
    if (frame === 0) {
      tracker.resetProgress(st.x, st.z, routeIndex);
      prevA0 = 0; prevA1 = 0;
    } else if (frame === lastFrame + 1) {
      tracker.advanceProgress(st.x, st.z);
    } else {
      throw Error(`RL 教師: フレームが飛びました(前 ${lastFrame} → 今 ${frame})。方策の状態を再現できないので止めます。`);
    }
    tracker.observe(st, routeIndex, prevA0, prevA1, obs);
    act(obs, a);
    const c = actionToCommand(a[0], a[1], SPEED_CAP, actual.maxSteer);
    prevA0 = c.a0; prevA1 = c.a1;   // 次のフレームの「前回の行動」(RL の環境と同じく切ったあとの値)
    lastFrame = frame; lastCommand = {steering: c.steering, speed: c.speed};
    calls.actions++;
    return {...lastCommand};
  }
  const continuous = actual.episodeMode === 'continuous-laps-v1';
  return {
    action,
    // 表示と記録用(reference_length_m・planned_time_s として context.teacher に入る)。全シーンで同じ値
    reference: {schema: 'rl-bestlap-reference/v1', length: LOOP_LENGTH, planned_time_s: BESTLAP_MANIFEST.std_single_lap_s},
    options: {policy: 'rl-bestlap-ppo-v1', speed_cap_mps: SPEED_CAP, max_steer_rad: actual.maxSteer, deterministic: true,
      observation: 'privileged-ground-truth-79', control_delay_frames: 0},
    metadata: {...RL_TEACHER_METADATA, episode_mode: continuous ? 'continuous-laps-v1' : 'single-lap-v1'},
  };
}
