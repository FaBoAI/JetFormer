// JetFormer: シミュレーターの「RC加速教師」の差し込み口を、強化学習モード(RL の方策)の教師に差し替える(2026-10-14、sim_rl_modes の統合)。
//
// JetFormer が返すページの import map で、app.js が読む ./rc-speed-controller.js をこのファイルにしている(scopes で _jf/ の中からは元のファイル)。
// ユーザーのシミュレーターのファイル(app.js・sim-core.js・rc-speed-controller.js・収集器)は変えない。
//   - 強化学習モードを選んでいない(JETFORMER_RL.mode が無い・'rc')ときは、元の createRcTrainingTeacher・assertRcTeacherContract をそのまま呼ぶ
//     (今までと同じ動き)。元のモジュールの名前は全部そのまま再エクスポートする。
//   - 選んでいるときは、Node の試験と同じファイル ./mode_runtime.mjs の createModeTeacher で走る(観測・方策・安全シールド・乱し・経路の形)。
//     重みは sim-rl.js が JetFormer の /api/sim/rl/modes/<key>/policy.json から読み、JETFORMER_RL.weights[key] に置く(サーバーが sha256 を確かめる)。
//   - 教師は snapshot の写ししか受け取らない(状態には触れない)。乱数は config.seed とフレーム番号だけから作る(Node の予想・再生と同じ操作)。
//
// 開始の約束(RL の観測は Default(v3・33 ゲート)の参照ループ専用): 経路は標準の経路を開始ゲートで回したもの、開始ゲート 0〜32・左右 ±0.3 m・
// 角度 ±25 度(アプリの範囲)。物理・センサーの雑音・20 秒の連続周回は、元の RC 教師の検査に「ゲート 0 に回し戻した world」を渡して確かめる。
// 「低摩擦」(μ × 0.4)は学習・試験した物理の範囲の外なので使わない(元の検査も frictionMultiplier 1 を求める)。
import * as original from '../../rc-speed-controller.js';
import {createModeTeacher, RUNTIME_VERSION, TEACHER_SCHEMA, MODE_KEYS} from './mode_runtime.mjs';
import {routeOffset} from './features.mjs';
export * from '../../rc-speed-controller.js';

export const JF_TEACHER_VERSION = 'jf-sim-rl-teacher-slot/v1';
const RL = globalThis.JETFORMER_RL || (globalThis.JETFORMER_RL = {mode: 'rc', weights: {}, modes: {}, sceneParams: null});
const calls = {createRcTrainingTeacher: 0, assertRcTeacherContract: 0, rl_teachers: 0, actions: 0, errors: 0};
const base = Array.isArray(original.RC_COLLECTION_PLAN) && original.RC_COLLECTION_PLAN[0] ? JSON.parse(JSON.stringify(original.RC_COLLECTION_PLAN[0])) : null;
RL.teacherModule = Object.freeze({loaded: true, version: JF_TEACHER_VERSION, runtime: RUNTIME_VERSION, schema: TEACHER_SCHEMA, modes: [...MODE_KEYS]});
RL.basePlan = base;            // 記録の条件(センサー・物理・RC 応答)。sim-rl.js がモードを選んだときと収集プランで使う
RL.calls = calls;

function activeKey() {
  const k = RL.mode;
  if (!k || k === 'rc') return null;
  if (!MODE_KEYS.includes(k)) throw Error('強化学習モード: 知らないモードです: ' + k);
  if (!RL.modes?.[k] || !RL.weights?.[k]) throw Error('強化学習モード: ' + k + ' の方策がまだ読み込まれていません。読み込みが終わってからリセットしてください');
  return k;
}

function rotateBack(route, k) {
  const n = route.length;
  return Array.from({length: n}, (_, i) => route[(i - k + n) % n]);
}

// 強化学習モードの開始・物理の検査。戻り値は元の検査が返す(正規化した)設定
export function rlContract(key, world, config, variation = {}) {
  if (!world || !Array.isArray(world.route)) throw Error('強化学習モード: コースの経路がありません');
  let k;
  try { k = routeOffset(world.route); }
  catch (e) { throw Error('強化学習モードは Default コース(v3・33 ゲート)だけで使えます(' + e.message + ')'); }
  if (variation && Object.hasOwn(variation, 'initialState')) throw Error('強化学習モードは途中の状態からの開始(initialState)に対応していません');
  const gate = variation?.gate ?? 0, left = variation?.left_offset_m ?? 0, yaw = variation?.left_yaw_deg ?? 0;
  if (!Number.isInteger(gate) || gate < 0 || gate >= world.route.length) throw Error('強化学習モードの開始ゲートは 0〜' + (world.route.length - 1) + ' です');
  if (!Number.isFinite(left) || Math.abs(left) > .3 + 1e-12 || !Number.isFinite(yaw) || Math.abs(yaw) > 25 + 1e-12) {
    throw Error('強化学習モードの開始は 左右 ±0.3 m・角度 ±25 度 以内です');
  }
  // アプリはリセットで経路を開始ゲートで回し(k = gate)、連続試験の計画の検査では回さない world(k = 0)を渡す
  if (k !== 0 && k !== gate) throw Error('強化学習モード: 開始ゲートと経路の回し方が一致しません');
  const first = world.route[0], next = world.route[1], heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const want = {x: first[0] + left * Math.sin(heading), z: first[1] - left * Math.cos(heading), yaw: heading + yaw * Math.PI / 180};
  for (const q of ['x', 'z', 'yaw']) {
    if (!Number.isFinite(world.spawn?.[q]) || Math.abs(world.spawn[q] - want[q]) > 1e-9) throw Error('強化学習モード: 開始姿勢と開始の指定が一致しません: ' + q);
  }
  const fm = config?.frictionMultiplier ?? 1;
  if (fm !== 1) throw Error('強化学習モードでは「低摩擦」の場面は使えません(学習・試験した物理の範囲の外です)。場面を baseline・noise・dim にしてください');
  const std = rotateBack(world.route, k), f0 = std[0], f1 = std[1], h0 = Math.atan2(-(f1[1] - f0[1]), f1[0] - f0[0]);
  const stdWorld = {...world, route: std, spawn: {x: f0[0], z: f0[1], yaw: h0}};
  const actual = original.assertRcTeacherContract(stdWorld, config, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
  if (key !== 'obstacle' && (world.disks || []).concat(world.segments || []).some(o => o && o.kind === 'obstacle')) {
    throw Error('障害物のあるコースで使えるのは「障害物よけ」だけです');
  }
  return actual;
}

export function assertRcTeacherContract(world, config, variation = {}) {
  calls.assertRcTeacherContract++;
  const key = activeKey();
  if (!key) return original.assertRcTeacherContract(world, config, variation);
  return rlContract(key, world, config, variation);
}

// シーンの条件(プランの指定 → ページの設定 → 無ければ種から決める)
function paramsFor(key) {
  if (RL.sceneParams && RL.sceneParams.mode === key) return {...RL.sceneParams.params};
  const s = RL.settings?.[key];
  return s ? {...s} : {};
}

export function createRcTrainingTeacher(world, config, variation = {}) {
  calls.createRcTrainingTeacher++;
  const key = activeKey();
  if (!key) return original.createRcTrainingTeacher(world, config, variation);
  const actual = rlContract(key, world, config, variation);
  const entry = RL.modes[key];
  const t = createModeTeacher(world, config, variation, {mode: key, weights: RL.weights[key], params: paramsFor(key),
    policySha256: entry.policy_sha256, plannedTimeS: entry.planned_time_s ?? null});
  calls.rl_teachers++;
  const live = {mode: key, name: entry.name, frame: -1, ph: 'drive', l: 1, ev: 0, cl_mm: null, min_cl_mm: null, shield_frames: 0,
    perturb_frames: 0, events: 0, params: t.params, seed: config?.seed ?? null, error: null};
  RL.live = live;
  let lastF = -1;
  function action(snapshot) {
    try {
      const out = t.action(snapshot);
      const jf = out.info?.jf || {};
      if (jf.f !== lastF) {
        lastF = jf.f; calls.actions++;
        live.frame = jf.f; live.ph = jf.ph; live.l = jf.l; live.ev = jf.ev; live.cl_mm = jf.cl;
        live.min_cl_mm = live.min_cl_mm == null ? jf.cl : Math.min(live.min_cl_mm, jf.cl);
        if (jf.sh) live.shield_frames++;
        if (jf.l === 0) live.perturb_frames++;
        live.events = Math.max(live.events, jf.ev || 0);
      }
      return out;
    } catch (e) {
      calls.errors++; live.error = String(e && e.message || e);
      throw e;
    }
  }
  const continuous = actual.episodeMode === 'continuous-laps-v1';
  return {action, reference: t.reference, options: t.options, params: t.params,
    metadata: {...t.metadata, episode_mode: continuous ? 'continuous-laps-v1' : 'single-lap-v1', teacher_slot: JF_TEACHER_VERSION}};
}
