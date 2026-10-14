// RL の環境 1 つ分(Node、ヘッドレス)。design.md 2.1〜2.6 節 / protocol.json environment・reward・episode のとおり(2026-10-13)
//
//  - 物理: experiments/…/sim/fast-sim-core.mjs(元の dist/sim-core.js を読み込み時に 3 か所だけ置換した高速版。結果は元と同じ)
//          または core: 'original' で元の sim/dist/sim-core.js(ユーザーの dist とバイト単位で同じコピー)。
//  - コース: sim/world-v3.json(ブラウザと同じ scene.js から作った v3 コース)。開始はアプリの reset と同じ手順
//          (経路を開始ゲート k で回し、左へ off [m]、向き +turn)。
//  - 観測: rl/features.mjs(ブラウザと同じファイル)。報酬・終了: この下の stepCommand。
// ユーザーのファイルは読むだけで変えない。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createFeatureTracker, prepareGeometry, actionToCommand, commandToAction, OBS_DIM, G, FEATURES_VERSION, REFERENCE_LOOP_SHA256,
  LOOP_LENGTH, PROJECTION_WINDOW} from './features.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const EXP_ROOT = path.resolve(here, '..');
export const USER_SIM_ROOT = '/home/jetson/sim/extracted/course-simulator';
export const PATHS = Object.freeze({
  fastCore: path.join(EXP_ROOT, 'sim/fast-sim-core.mjs'),
  originalCore: path.join(EXP_ROOT, 'sim/dist/sim-core.js'),
  world: path.join(EXP_ROOT, 'sim/world-v3.json'),
  rcTeacher: path.join(EXP_ROOT, 'sim/dist/rc-speed-controller.js'),
  controlDelay: path.join(EXP_ROOT, 'sim/dist/control-delay.js'),
  plan60: path.join(USER_SIM_ROOT, 'qa/rc-collection-plan-60.json'),
  learnedLap: path.join(USER_SIM_ROOT, 'results/shortcut-rc-learned-lap.json'),
  userCore: path.join(USER_SIM_ROOT, 'dist/sim-core.js'),
});
// protocol.json files.* と同じ値
export const EXPECTED_SHA256 = Object.freeze({
  fastCore: '227ceccc7970ebf1aea98e4ea8686b612397b212e77382c216e5ec85096df0cc',
  originalCore: 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e',
  world: '57de53a55e5366c004e26443d8e23722b015a9ac865044023a3ac57db55f9dc2',
  plan60: '1c5bb22087b5913531e6ebc76e6ae3da99c04ef1feaebfb1f2e74eaa68160dfb',
  rcTeacher: '4a32ab56369b3d0c9a76f16fe7144eb895677b72ceeefc8483645f91ae96f417',
});
export const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export function checkAsset(name) {
  const actual = sha256File(PATHS[name]);
  if (actual !== EXPECTED_SHA256[name]) throw new Error(`env_core: ${name} の sha256 が手順と違います: ${actual}`);
  return actual;
}

// 物理(20 分データと同じ。変えない)。protocol.json environment.physics
export const PHYSICS = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42,
  vehicleWidth: .19, wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0,
  longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
export const VARIANTS = Object.freeze({main: Object.freeze({speed_cap_mps: 4.5}), open: Object.freeze({speed_cap_mps: 7.24})});
// 報酬・終了(protocol.json reward)。パイロットのあとに 1 回だけ直してよい(直したら deviations.md)
// 2026-10-13 パイロットのあとの 1 回の調整(deviations.md「パイロットのあとの調整」):
//   1. 進みの上限 progress_gate_margin: 進み s は「今の目標ゲートの s + 0.75 m」までしか数えない。
//      パイロットの方策は、ゲート 3 の線を越えずに内側の空き地へ出て、射影 s だけを前へ進めて報酬を得ていた(射影の飛び)。
//      教師の走りでは s − 目標ゲートの s は最大 0.34 m(REC60 の全フレーム)なので、正しい走りの報酬は変わらない。
//   2. 停止・逆走の罰を −5 → −10(衝突と同じ)。パイロットの終わり方の 6〜7 割が停止で、難所の手前で止まると衝突より得になっていた。
export const REWARD = Object.freeze({
  progress_min: -.3, progress_slack: .05, progress_gate_margin: .75,
  smooth_steer: .02, smooth_speed: .01,
  traction: .02, traction_free: 1.1,
  clearance: .05, clearance_target: .05,
  terminal: Object.freeze({collision: -10, gate_miss: -10, required_crossing_missing: -10, lap_distance_insufficient: -10, stall: -10, wrong_way: -10}),
  stall_speed: .2, stall_after_s: 1, stall_frames: 60, wrong_way_m: 1,
});
// 調整前(パイロット)の値。比べる・再現するためだけに残す(RlEnv の reward オプションに渡すと前の報酬になる)
export const REWARD_PILOT = Object.freeze({...REWARD, progress_gate_margin: Infinity,
  terminal: Object.freeze({...REWARD.terminal, stall: -5, wrong_way: -5})});
export const REWARD_VERSION = 'rl-bestlap-reward-v2-gatecap';
// 開始の乱し(protocol.json episode.start_distribution)
export const START_DISTRIBUTION = Object.freeze({gate0_fraction: .5, gate_range: [0, 32], left_offset_m: [-.12, .12], left_yaw_deg: [-8, 8]});

// ---- 物理のずれのランダム化(r2 の追加実験。手順の外の参考。既定では使わない = 本番の結果は変わらない) ----
// 2026-10-13 r2: 独立な確認(rl_check_r1)で、選んだ方策が舵の速さ 1.0 rad/s・ブレーキ ≤ 1.0 m/s² などでぶつかると分かった。
// 「速く・ぶつからない」ための更新の候補として、回ごとに物理を変えて学習し直す(ドメインのランダム化)。
// 観測は標準の物理の値のまま作る(方策は今の物理を知らない = 実車と同じ)。
export const PERTURB_KEYS = Object.freeze(['steerRate', 'brakeDeceleration', 'driveTau', 'driveAccelerationCap', 'frictionMultiplier']);
export const DR_SPECS = Object.freeze({
  // nominal_fraction の回は標準の物理。それ以外は各値を一様に引く。delay1_fraction の回は制御遅延 1 フレーム
  r2: Object.freeze({nominal_fraction: .2, steerRate: [.85, 1.7], brakeDeceleration: [.8, 2.0], driveTau: [.3, .5],
    driveAccelerationCap: [4, 6], frictionMultiplier: [.88, 1.08], delay1_fraction: .5, steerGain: [.95, 1.05], speedGain: [.97, 1.03]}),
});
export function sampleDr(rng, spec) {
  if (!spec) return null;
  // 引く回数をいつも同じにする(標準の回でも同じ数だけ引く)
  const u = () => rng.uniform();
  const pick = (r, x) => r[0] + (r[1] - r[0]) * x;
  const nominal = u() < spec.nominal_fraction;
  const xs = {steerRate: u(), brakeDeceleration: u(), driveTau: u(), driveAccelerationCap: u(), frictionMultiplier: u(),
    delay: u(), steerGain: u(), speedGain: u()};
  if (nominal) return {physics: {}, delay: 0, steerGain: 1, speedGain: 1, nominal: true};
  const physics = {};
  for (const k of PERTURB_KEYS) physics[k] = pick(spec[k], xs[k]);
  return {physics, delay: xs.delay < spec.delay1_fraction ? 1 : 0, steerGain: pick(spec.steerGain, xs.steerGain),
    speedGain: pick(spec.speedGain, xs.speedGain), nominal: false};
}
export const EPISODE_SECONDS = 20;

let baseWorldCache = null;
export function loadBaseWorld() {
  if (!baseWorldCache) { checkAsset('world'); baseWorldCache = JSON.parse(fs.readFileSync(PATHS.world, 'utf8')); }
  return baseWorldCache;
}
const coreCache = {};
export async function loadCore(kind = 'fast') {
  if (!coreCache[kind]) {
    if (kind === 'fast') { checkAsset('fastCore'); checkAsset('originalCore'); coreCache[kind] = await import(pathToFileURL(PATHS.fastCore).href); }
    else if (kind === 'original') { checkAsset('originalCore'); coreCache[kind] = await import(pathToFileURL(PATHS.originalCore).href); }
    else throw new Error('env_core: core は fast か original です');
  }
  return coreCache[kind];
}
let teacherModuleCache = null;
export async function loadRcTeacher() {
  if (!teacherModuleCache) { checkAsset('rcTeacher'); teacherModuleCache = await import(pathToFileURL(PATHS.rcTeacher).href); }
  return teacherModuleCache;
}

// app.js の reset(): 開始ゲートで経路を回転し、開始ずれを左右・向きに加える(lib/app_emulation.mjs と同じ式)
export function activeWorldFor(world, gate = 0, offsetM = 0, yawDeg = 0) {
  const startIndex = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(startIndex), ...world.route.slice(0, startIndex)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const offset = Math.max(-.3, Math.min(.3, offsetM || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  const active = {...world, route, spawn: {x: first[0] + offset * Math.sin(heading), z: first[1] - offset * Math.cos(heading), yaw: heading + turn}};
  return {world: active, variation: {gate: startIndex, left_offset_m: offset, left_yaw_deg: turn * 180 / Math.PI}};
}

// RC 教師の約束の範囲(ゲート 0・左右 ±3 cm・向き ±3°)
export const inTeacherContract = v => v.gate === 0 && Math.abs(v.left_offset_m) <= .03 + 1e-12 && Math.abs(v.left_yaw_deg) <= 3 + 1e-12;

// 決定的な乱数(splitmix32)。環境ごとに (種, 環境の通し番号) から作る
export function mix32(a, b = 0) {
  let h = (Math.imul(a >>> 0, 0x9e3779b1) ^ Math.imul((b >>> 0) + 0x7f4a7c15, 0x85ebca6b)) >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d) >>> 0; h ^= h >>> 15; h = Math.imul(h, 0x846ca68b) >>> 0; h ^= h >>> 16;
  return h >>> 0;
}
export function createRng(seed) {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state; z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0; z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0; z = (z ^ (z >>> 15)) >>> 0;
    return z;
  };
  return {uint32: next, uniform: (lo = 0, hi = 1) => lo + (hi - lo) * (next() / 4294967296), get state() { return state; }};
}
// 開始の乱し: 半分はゲート 0、半分はゲート 0〜32 から一様。横ずれ U(−0.12, 0.12) m・向き U(−8°, 8°)
export function sampleStart(rng, dist = START_DISTRIBUTION) {
  const gate = rng.uniform() < dist.gate0_fraction ? 0 : Math.min(dist.gate_range[1], Math.floor(rng.uniform(dist.gate_range[0], dist.gate_range[1] + 1)));
  return {gate, left_offset_m: rng.uniform(...dist.left_offset_m), left_yaw_deg: rng.uniform(...dist.left_yaw_deg), seed: rng.uint32()};
}

export function simConfig({episodeMode = 'continuous', timeLimit = EPISODE_SECONDS, seed = 1, noiseStd = 0} = {}) {
  const config = {...PHYSICS, seed, timeLimit, noiseStd};
  if (episodeMode === 'continuous') config.episodeMode = 'continuous-laps-v1';
  else if (episodeMode !== 'single') throw new Error('env_core: episodeMode は continuous か single です');
  return config;
}

// 環境 1 つ。mode 'train' は RL 用の終了(ゲート逃し・停止・逆走)も使う。mode 'eval' は sim の終了だけ(アプリと同じ)
export class RlEnv {
  constructor({core, coreKind = 'fast', baseWorld = loadBaseWorld(), variant = 'main', mode = 'train', episodeMode = 'continuous',
    timeLimit = EPISODE_SECONDS, teacherModule = null, teacherActions = false, record = false, reward = REWARD} = {}) {
    if (!core) throw new Error('env_core: core(loadCore の結果)が要ります');
    if (!VARIANTS[variant]) throw new Error('env_core: variant は main か open です');
    if (!['train', 'eval'].includes(mode)) throw new Error('env_core: mode は train か eval です');
    Object.assign(this, {core, coreKind, baseWorld, variant, mode, episodeMode, timeLimit, teacherModule, teacherActions, record, reward});
    this.speedCap = VARIANTS[variant].speed_cap_mps;
    this.geometry = prepareGeometry(baseWorld);
    this.obs = new Float64Array(OBS_DIM);
    this.aux = {};
    this.teacherAction = new Float64Array([NaN, NaN]);
    this.episodes = 0;
    this.sim = null;
  }
  // 開始(停止状態)。invalid_spawn なら false を返す(呼ぶ側が引き直す)
  // perturb(r2 の追加実験だけ): {physics: {steerRate…}, delay: 0|1, steerGain, speedGain}。null なら今までと同じ
  reset({gate = 0, left_offset_m = 0, left_yaw_deg = 0, seed = 1, scenario = 'baseline', noiseStd = 0, perturb = null} = {}) {
    const {world, variation} = activeWorldFor(this.baseWorld, gate, left_offset_m, left_yaw_deg);
    const nominalConfig = simConfig({episodeMode: this.episodeMode, timeLimit: this.timeLimit, seed, noiseStd});
    const physics = perturb?.physics ?? {};
    for (const k of Object.keys(physics)) if (!PERTURB_KEYS.includes(k)) throw new Error('env_core: 変えられない物理の値です: ' + k);
    const config = {...nominalConfig, ...physics};
    const sim = this.core.createSimulation(world, config);
    sim.reset({seed: config.seed, scenario});
    this.sim = sim; this.world = world; this.variation = variation; this.scenario = scenario;
    this.view = this.coreKind === 'fast' ? sim.peek() : sim.snapshot();
    this.config = this.view.config;
    // 観測・報酬の μ は標準の物理の値で作る(方策に今の物理を教えない)
    const trackerConfig = Object.keys(physics).length ? {...this.config, ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))} : this.config;
    this.perturb = perturb && (Object.keys(physics).length || perturb.delay || (perturb.steerGain ?? 1) !== 1 || (perturb.speedGain ?? 1) !== 1)
      ? {physics: {...physics}, delay: perturb.delay | 0, steerGain: perturb.steerGain ?? 1, speedGain: perturb.speedGain ?? 1} : null;
    if (this.perturb && ![0, 1].includes(this.perturb.delay)) throw new Error('env_core: delay は 0 か 1 です');
    this.queued = null;
    const st = this.view.state;
    this.start = {gate: variation.gate, left_offset_m: variation.left_offset_m, left_yaw_deg: variation.left_yaw_deg, seed: config.seed, scenario};
    if (this.perturb) this.start.perturb = this.perturb;
    if (st.terminal) { this.invalid = true; return false; }
    this.invalid = false;
    this.tracker = createFeatureTracker(world, trackerConfig, {geometry: this.geometry});
    this.tracker.resetProgress(st.x, st.z, this.view.metrics.route_index);
    this.prevA0 = 0; this.prevA1 = 0; this.stallFrames = 0;
    this.s0 = this.tracker.s;
    // 今の目標ゲートの s(周をまたいで増え続ける値)。resetProgress の探索範囲と同じく s − 0.5 以上で最も近いもの
    const nGates = this.tracker.targetS.length;
    this.gateIdx = (this.view.metrics.route_index ?? 0) % nGates;
    const g0 = this.tracker.targetS[this.gateIdx];
    this.gateS = g0 + LOOP_LENGTH * Math.ceil((this.s0 - PROJECTION_WINDOW.behind - g0) / LOOP_LENGTH);
    this.sCap = Math.min(this.s0, this.gateS + this.reward.progress_gate_margin);
    this.sCap0 = this.sCap;
    this.tracker.observe(st, this.view.metrics.route_index, 0, 0, this.obs, this.aux);
    this.stats = {return: 0, frames: 0, min_clearance: this.aux.d_min, max_speed: 0, gate_miss_frame: null, lap_count: 0,
      reward_parts: {progress: 0, smooth: 0, traction: 0, clearance: 0, terminal: 0}};
    this.done = false; this.episodes++;
    this.teacher = null;
    if (this.teacherActions && this.teacherModule && inTeacherContract(variation)) {
      this.teacher = this.teacherModule.createRcTrainingTeacher(world, JSON.parse(JSON.stringify(this.config)), variation);
    }
    this.updateTeacherAction();
    return true;
  }
  updateTeacherAction() {
    if (!this.teacher || this.view.state.terminal) { this.teacherAction[0] = NaN; this.teacherAction[1] = NaN; return; }
    const cmd = this.teacher.action(this.view), a = commandToAction(cmd.steering, cmd.speed, this.speedCap, this.config.maxSteer);
    this.teacherAction[0] = a.a0; this.teacherAction[1] = a.a1;
    this.teacherCommand = cmd;
  }
  // 方策の行動 a ∈ [−1, 1]²
  step(a0, a1) {
    const c = actionToCommand(a0, a1, this.speedCap, this.config.maxSteer);
    if (!this.perturb) return this.stepCommand(c.steering, c.speed, c.a0, c.a1);
    // r2: 制御遅延 1 なら 1 フレーム前の行動を適用する(最初は中立 = 舵 0・速さ 0。アプリの固定遅延・eval_sets と同じ)。
    // 観測の「前回の行動」は適用した行動(eval_sets の delayFrames 1 と同じ約束)。ゲインは適用する指令にだけ掛ける
    let ap = c;
    if (this.perturb.delay === 1) { ap = this.queued ?? actionToCommand(0, -1, this.speedCap, this.config.maxSteer); this.queued = c; }
    return this.stepCommand(ap.steering * this.perturb.steerGain, ap.speed * this.perturb.speedGain, ap.a0, ap.a1);
  }
  // アプリが適用する単位で 1 フレーム進める(教師・保存した操作の再生用)。a0n/a1n は観測の「前回の行動」に入れる値
  stepCommand(steering, speed, a0n = null, a1n = null) {
    if (this.done || this.invalid) throw new Error('env_core: 終わったエピソードは reset してから進めます');
    if (a0n === null || a1n === null) { const a = commandToAction(steering, speed, this.speedCap, this.config.maxSteer); a0n = a.a0; a1n = a.a1; }
    const lapsBefore = this.view.metrics.completed_laps ?? 0;
    // 高速版の peek() は state・metrics を参照で返すが、sim は 1 小区間ごとに state を新しいオブジェクトに置き換えるので毎回取り直す
    if (this.coreKind === 'fast') { this.sim.step({steering, speed}, true); this.view = this.sim.peek(); }
    else this.view = this.sim.step({steering, speed});
    const st = this.view.state, m = this.view.metrics, dt = this.config.dt, REWARD = this.reward;
    const s = this.tracker.advanceProgress(st.x, st.z);
    this.tracker.observe(st, m.route_index, a0n, a1n, this.obs, this.aux);
    // 目標ゲートが進んだら、その分だけゲートの s を進める(連続周回では周の終わりに route_index が 0 に戻る。単周の完走では 33)
    const nGates = this.tracker.targetS.length, ts = this.tracker.targetS, gi = (m.route_index ?? 0) % nGates;
    for (let guard = 0; this.gateIdx !== gi && guard < nGates; guard++) {
      const nx = (this.gateIdx + 1) % nGates;
      this.gateS += ((ts[nx] - ts[this.gateIdx]) % LOOP_LENGTH + LOOP_LENGTH) % LOOP_LENGTH;
      this.gateIdx = nx;
    }
    // ---- 報酬(protocol.json reward.per_step。進みは「目標ゲートの s + 余白」で頭打ち = 調整 1) ----
    const sCap = Math.min(s, this.gateS + REWARD.progress_gate_margin);
    const progress = Math.max(REWARD.progress_min, Math.min(sCap - this.sCap, st.speed * dt + REWARD.progress_slack));
    this.sCap = sCap;
    const smooth = REWARD.smooth_steer * (a0n - this.prevA0) ** 2 + REWARD.smooth_speed * (a1n - this.prevA1) ** 2;
    const rho = st.speed * st.speed * Math.abs(Math.tan(st.steering)) / (this.config.wheelbase * this.aux.mu * G);
    const traction = REWARD.traction * Math.max(0, rho - REWARD.traction_free);
    const clear = REWARD.clearance * Math.max(0, (REWARD.clearance_target - this.aux.d_min) / REWARD.clearance_target);
    let reward = progress - smooth - traction - clear, terminal = 0;
    // ---- 終了 ----
    let terminated = false, truncated = false, reason = null;
    if (st.terminal) {
      reason = st.reason;
      if (reason === 'collection_duration_complete' || reason === 'time_limit') truncated = true;
      else { terminated = true; terminal = REWARD.terminal[reason] ?? 0; }   // route_complete(単周の完走)は罰なし
    } else {
      if (this.aux.gate_along >= 0 && this.stats.gate_miss_frame === null) this.stats.gate_miss_frame = st.frame;
      if (st.sim_time >= REWARD.stall_after_s - 1e-9 && st.speed < REWARD.stall_speed) this.stallFrames++; else this.stallFrames = 0;
      if (this.mode === 'train') {
        if (this.aux.gate_along >= 0) reason = 'gate_miss';
        else if (this.stallFrames >= REWARD.stall_frames) reason = 'stall';
        else if (s < this.tracker.sMax - REWARD.wrong_way_m) reason = 'wrong_way';
        if (reason) { terminated = true; terminal = REWARD.terminal[reason]; }
      }
    }
    reward += terminal;
    this.prevA0 = a0n; this.prevA1 = a1n;
    const S = this.stats;
    S.return += reward; S.frames++; S.reward_parts.progress += progress; S.reward_parts.smooth -= smooth; S.reward_parts.traction -= traction;
    S.reward_parts.clearance -= clear; S.reward_parts.terminal += terminal;
    if (this.aux.d_min < S.min_clearance) S.min_clearance = this.aux.d_min;
    if (st.speed > S.max_speed) S.max_speed = st.speed;
    if ((m.completed_laps ?? 0) !== lapsBefore) S.lap_count = m.completed_laps;
    this.done = terminated || truncated;
    this.last = {reward, terminated, truncated, reason, progress, smooth, traction, clearance: clear, terminal_reward: terminal, rho, s};
    if (!this.done) this.updateTeacherAction(); else { this.teacherAction[0] = NaN; this.teacherAction[1] = NaN; }
    return this.last;
  }
  // エピソードの要約(JSON にできる形)
  summary() {
    const st = this.view.state, m = this.view.metrics;
    const laps = (m.lap_events ?? []).map(l => ({lap: l.lap_index, elapsed: l.elapsed, start_time: l.start_time, end_time: l.end_time,
      collisions: l.collisions, gates: l.passed_gates, crossings: l.required_crossing_events.length, distance: l.distance}));
    if (this.episodeMode === 'single' && st.reason === 'route_complete')
      laps.push({lap: 1, elapsed: m.elapsed, start_time: 0, end_time: m.elapsed, collisions: m.collisions, gates: m.passed_gates,
        crossings: m.required_crossing_events.length, distance: m.distance});
    return {episode: this.episodes, start: this.start, variant: this.variant, mode: this.mode, episode_mode: this.episodeMode, core: this.coreKind,
      frames: st.frame, sim_time: st.sim_time, reason: this.last?.reason ?? st.reason, sim_reason: st.reason, terminated: !!this.last?.terminated,
      truncated: !!this.last?.truncated, return: this.stats.return, reward_parts: this.stats.reward_parts, progress_m: this.tracker.s - this.s0,
      progress_capped_m: this.sCap - this.sCap0, reward_version: this.reward === REWARD ? REWARD_VERSION : 'custom',
      laps, completed_laps: m.completed_laps ?? (st.reason === 'route_complete' ? 1 : 0), passed_gates: m.passed_gates,
      current_lap_gates: m.current_lap_passed_gates ?? null, collisions: m.collisions, collision_events: m.collision_events,
      min_clearance: this.stats.min_clearance, max_speed: this.stats.max_speed, gate_miss_frame: this.stats.gate_miss_frame,
      max_lateral_acceleration: m.max_lateral_acceleration, traction_limited_steps: m.traction_limited_steps};
  }
  report() { return {...this.sim.report(), world: this.world, context: {start_variation: this.variation, controller: 'rl-env', core: this.coreKind,
    variant: this.variant, features_version: FEATURES_VERSION, reference_loop_sha256: REFERENCE_LOOP_SHA256,
    reward_version: this.reward === REWARD ? REWARD_VERSION : 'custom'}}; }
}

// 開始を引く(invalid_spawn なら引き直す)。戻り値は引き直した回数
// extra(r2): 開始に足す値(perturb など)。null なら今までと同じ
export function resetRandom(env, rng, dist = START_DISTRIBUTION, maxTries = 100, extra = null) {
  for (let tries = 0; tries < maxTries; tries++) if (env.reset(extra ? {...sampleStart(rng, dist), ...extra} : sampleStart(rng, dist))) return tries;
  throw new Error('env_core: 開始を 100 回引いても衝突しない開始になりません');
}
