// モードの学習の環境(Node、ヘッドレス)。2026-10-13、sim_rl_modes の実装の段。design.md 4〜5 章・modes.json の training のとおり。
// ベストラップの RlEnv(snapshot/rl/env_core.mjs、変えない)と同じ流れを写し、モード別に次を足した:
//   - 観測の包み(rl/runtime/obs_modes.mjs = ブラウザと同じファイル): base79 / base79v(仮想ゲート)/ cond97 / obst95
//   - no_wall: すき間の約束(12 cm から罰・7.5 cm で終了)、ρ_req・摩擦円 ρ_comb の罰、なめらかさ 3 倍、物理のずれ DR-NW-train
//              (舵の速さ・ブレーキ・時定数・加速の上限・μ・遅延 0/1/2・ゲイン・摩擦円 30%・舵の白色雑音)
//   - recover: 乱し(rl/runtime/perturb.mjs の TRAIN_SPEC: 舵のパルス・わざと外す・ブレーキ/踏み込み・状態のキック)、強さのカリキュラム、
//              ゲートを外しても終わらない(−3)。差し込んだ行動のステップは injected = 1(PPO が方策の損失から外す)
//   - cond(line・speed): 経路の形 e*(s) と速さの倍率 c_v を回ごとに引き、速さの上限 4.5·c_v・追従の罰
//   - obstacle: 回ごとの障害物の配置(10% は無し)、足した障害物とのすき間 < 8 cm の罰
//   - obstacle_r2(2026-10-14): obstacle と同じ観測・配置に、開始の空き 1.5 m・全部の障害物の壁・近づく速さの罰・3 cm の終了(OBSTACLE_R2_DEFAULTS)
// 物理は rl/fast_core_modes.mjs(ベストラップの高速版と同じ置き換え + 摩擦円)。'base' はベストラップの環境と同じ値になる(試験で確かめる)。
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS, REWARD as BASE_REWARD, createRng, mix32} from '../snapshot/rl/env_core.mjs';
import {actionToCommand, prepareGeometry, clearance, projectWindow, G, LOOP_LENGTH} from './runtime/features.mjs';
import {createObsWrapper, OBS_WRAPPERS} from './runtime/obs_modes.mjs';
import {createPerturber, TRAIN_SPEC} from './runtime/perturb.mjs';
import {createLineProfile, sampleLineParams} from './runtime/line_profiles.mjs';
import {geometryWithObstacles, worldWithObstacles, obstacleOnlyGeometry, generateLayout, buildBaseClearanceGrid, startClearOfObstacles} from './runtime/layouts.mjs';
import {appliedAction, BASE_SPEED_CAP, modeSpec} from './runtime/mode_runtime.mjs';
import {CATALOGUE} from './runtime/catalogue.mjs';
import {gaussian} from './runtime/rng.mjs';
import {passageTargets, activeTarget} from './ob2/guidance.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const EXP = path.resolve(here, '..');
let baseWorld = null;
export function loadBaseWorld() { return baseWorld ??= JSON.parse(fs.readFileSync(path.join(EXP, 'snapshot/sim/world-v3.json'), 'utf8')); }
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const DR = CATALOGUE.common.dr_specs;

// ---- 開始の分布(modes.json modes[].training.start_distribution.mix)----
export const START_BASE = Object.freeze({mix: [{p: 1, gate: 'base', left_offset_m: [-.12, .12], left_yaw_deg: [-8, 8]}]});
export function startMixFor(run) {
  if (run === 'base') return START_BASE.mix;
  const key = run === 'cond' ? 'line' : run;
  return modeSpec(key).training.start_distribution.mix;
}
// 1 つの開始を引く(引く数はいつも同じ: 成分・ゲート 2・左右・向き・種)
export function sampleStartMix(rng, mix) {
  const u = rng.uniform(), uG = rng.uniform(), uG2 = rng.uniform(), uO = rng.uniform(), uY = rng.uniform(), seed = rng.uint32();
  let acc = 0, comp = mix[mix.length - 1], ci = mix.length - 1;
  for (let i = 0; i < mix.length; i++) { acc += mix[i].p; if (u < acc) { comp = mix[i]; ci = i; break; } }
  let gate;
  if (typeof comp.gate === 'number') gate = comp.gate;
  else if (String(comp.gate).startsWith('U')) gate = Math.min(32, Math.floor(uG2 * 33));
  else gate = uG < .5 ? 0 : Math.min(32, Math.floor(uG2 * 33));   // 'base': 半分ゲート 0・半分 U{0..32}(ベストラップと同じ割合)
  const [o0, o1] = comp.left_offset_m, [y0, y1] = comp.left_yaw_deg;
  return {gate, left_offset_m: o0 + (o1 - o0) * uO, left_yaw_deg: y0 + (y1 - y0) * uY, seed, component: ci, min_initial_clearance_m: comp.min_initial_clearance_m ?? null};
}

// ---- 物理のずれ(DR-NW-train / DR-T)。引く数はいつも同じ ----
export function sampleDrSpec(rng, spec) {
  const u = () => rng.uniform(), pick = (r, x) => r[0] + (r[1] - r[0]) * x;
  const xNom = u(), xs = {steerRate: u(), brakeDeceleration: u(), driveTau: u(), driveAccelerationCap: u(), frictionMultiplier: u()};
  const xDelay = u(), xSg = u(), xVg = u(), xFc = u(), xNoise = u();
  const noise = spec.steer_white_noise && xNoise < spec.steer_white_noise.fraction ? spec.steer_white_noise.sigma_rad : 0;
  if (spec.nominal_fraction && xNom < spec.nominal_fraction) return {physics: {}, delay: 0, steerGain: 1, speedGain: 1, frictionCircle: false, steerNoise: noise, nominal: true};
  const physics = {};
  for (const k of PERTURB_KEYS) physics[k] = pick(spec[k], xs[k]);
  let delay = 0, acc = 0;
  for (const [d, p] of Object.entries(spec.delay_frames_p)) { acc += p; if (xDelay < acc) { delay = +d; break; } }
  return {physics, delay, steerGain: pick(spec.steerGain, xSg), speedGain: pick(spec.speedGain, xVg), frictionCircle: xFc < (spec.friction_circle_fraction ?? 0),
    steerNoise: noise, nominal: false};
}

// ---- 速さいろいろ r2 の学習の既定の値(protocol_speed_r2.json training と同じ。2026-10-14)----
// start_mix: 50% = 記録・試験の開始の分布(ゲート U{0..32}・±0.2 m・±15°・開始のすき間 ≥ 12 cm)、
//            25% = ベストラップの開始(半分ゲート 0・±0.12 m・±8°、≥ 8 cm)、25% = 広め(±0.25 m・±20°、≥ 10 cm)
export const SPEED_R2_DEFAULTS = Object.freeze({
  line_fixed: Object.freeze({family: 'const', c0: 0}),
  c_v_levels: Object.freeze([1.0, .875, .75, .625, .5]), c_v_levels_p: .8, c_v_range: Object.freeze([.5, 1.0]),
  start_mix: Object.freeze([
    Object.freeze({p: .5, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12}),
    Object.freeze({p: .25, gate: 'base', left_offset_m: [-.12, .12], left_yaw_deg: [-8, 8], min_initial_clearance_m: .08}),
    Object.freeze({p: .25, gate: 'U{0..32}', left_offset_m: [-.25, .25], left_yaw_deg: [-20, 20], min_initial_clearance_m: .10})]),
  tracking_w: .06, tracking_scale_m: .15,
  clearance_barrier_w: .2, clearance_barrier_start_m: .09, terminal_clearance_m: .03,
});

// ---- 経路いろいろ r2 の学習の既定の値(protocol_line_r2.json training と同じ。2026-10-14、学習の段 line r2)----
// 目標の線は v2(line_profiles.mjs: 開始のつなぎ 2 m・R の曲がりの内側の上限・なめらかさ)。
// start_mix: 60% = 記録・試験の開始の分布(ゲート U{0..32}・±0.2 m・±15°・開始のすき間 ≥ 12 cm)、20% = ベストラップの開始(≥ 8 cm)、20% = 広め(≥ 10 cm)。
// c_v: 80% は U(0.9, 1.0)(走行中・試験と同じ)、20% は U(0.6, 0.9)。
// 報酬: v2-gatecap の進みに追従の掛け算 f(e) = 1 / (1 + (e/0.08)²) を掛ける(進みが正のときだけ。線から外れるほど進みの得が減る = 速さを落としてでも線に乗る)
//       + 追従の足し算 −0.05·min(1, (e/0.15)²) + すき間の壁 d_min < 6 cm で −0.2·min(4, ((0.06 − d)/0.03)²) + d_min < 3 cm で −10 で終える。
export const LINE_R2_DEFAULTS = Object.freeze({
  line_profile: 'v2',
  c_v_mix: Object.freeze([Object.freeze({p: .8, range: [.9, 1.0]}), Object.freeze({p: .2, range: [.6, .9]})]),
  start_mix: Object.freeze([
    Object.freeze({p: .6, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .12}),
    Object.freeze({p: .2, gate: 'base', left_offset_m: [-.12, .12], left_yaw_deg: [-8, 8], min_initial_clearance_m: .08}),
    Object.freeze({p: .2, gate: 'U{0..32}', left_offset_m: [-.25, .25], left_yaw_deg: [-20, 20], min_initial_clearance_m: .10})]),
  progress_track_sigma_m: .08, tracking_w: .05, tracking_scale_m: .15,
  clearance_barrier_w: .2, clearance_barrier_start_m: .06, barrier_scale_m: .03, terminal_clearance_m: .03,
});

// ---- 障害物よけ r2 の学習の既定の値(protocol_obstacle_r2.json training と同じ。2026-10-14、学習の段 obstacle r2)----
// 確認 obstacle r1 の指摘(早い衝突は 3〜4 m 先の障害物に 3.2 m/s のまま当たる・有効な回でも 0.8 cm まで寄る)に合わせて:
//   配置: 開始の前後の空きを試験・記録と同じ 1.5 m に(r1 は layouts.mjs の余白で 1.8 m)。1〜4 個・10% は無しは同じ
//   開始: 40% = ベストラップの開始(半分ゲート 0)、40% = 記録・試験の開始の分布(U{0..32}・±0.15 m・±10°)、20% = 広め(±0.2 m・±15°)。どれも開始のすき間 ≥ 4 cm
//   報酬: v2-gatecap + 足した障害物の壁(8 cm から −0.2·min(4, ((0.08 − d)/0.04)²)。r1 は 0.1)
//         + 全部の障害物(コース + 足したもの)の壁 d_min < 6 cm で −0.2·min(4, ((0.06 − d)/0.03)²)(line r2 と同じ)
//         + 近づく速さの罰 −0.05·max(0, v − v_ok(d_obs))²、v_ok = 1.5 + 10·max(0, d_obs − 0.05) m/s(足した障害物に近いほど遅く)
//         + d_min < 3 cm で −10 で終える(O2)+ ゲートの外し・衝突は −10 で終える(r1 と同じ)
export const OBSTACLE_R2_DEFAULTS = Object.freeze({
  layouts: Object.freeze({none_p: .1, count: Object.freeze([1, 4]), start_gap_m: 1.5}),
  start_mix: Object.freeze([
    Object.freeze({p: .4, gate: 'base', left_offset_m: [-.12, .12], left_yaw_deg: [-8, 8], min_initial_clearance_m: .04}),
    Object.freeze({p: .4, gate: 'U{0..32}', left_offset_m: [-.15, .15], left_yaw_deg: [-10, 10], min_initial_clearance_m: .04}),
    Object.freeze({p: .2, gate: 'U{0..32}', left_offset_m: [-.2, .2], left_yaw_deg: [-15, 15], min_initial_clearance_m: .04})]),
  obstacle_barrier_w: .2, obstacle_barrier_start_m: .08, barrier_scale_m: .04,
  clearance_barrier_w: .2, clearance_barrier_start_m: .06, clearance_barrier_scale_m: .03, terminal_clearance_m: .03,
  approach_w: .05, approach_v0_mps: 1.5, approach_d0_m: .05, approach_k_per_s: 10,
});

// ---- 障害物よけ r2 の後半(obstacle_r2b、2026-10-14 学習の段 obstacle r2、protocol_obstacle_r2.json の追記 r2b)----
// r2 の前半(obstacle_r2)の 600 万までで分かったこと: (1) 決定的な方策が障害物の手前で止まる・はう(0.2 m/s を少し越える速さでは「止まる」終了に
// かからない)。valid_episode の定義では有効に数えられるが見本にならない。(2) 衝突は 3.3〜4.2 m/s のまま、よける幅が足りずに当たる・通れない側へ行く。
// → 前半の値に (a) 進みの見張り: 1.5 秒以降、直近 60 フレーム(2 秒)の R の進みが 0.6 m 未満なら −10 で終える('slow')、
//   (b) 通り道の目標(rl/ob2/guidance.mjs、学習だけ): 次の障害物の手前 2.5 m から 0.3 m 過ぎるまで −0.05·min(1, ((e_y − e_t)/0.15)²)·ramp
//   (ramp は 1.0 m より近いと 1、2.5 m で 0)、(c) 障害物なしの回を 10% → 20%、を足す。観測・走行中の仕組みは同じ
export const OBSTACLE_R2B_DEFAULTS = Object.freeze({
  none_p: .2,
  guidance: Object.freeze({w: .05, scale_m: .15, ahead_m: 2.5, full_m: 1.0, behind_m: .3, margin_m: .12}),
  slow: Object.freeze({after_s: 1.5, window_frames: 60, min_progress_m: .6}),
});

// ---- 学習の設定(modes.json の値。tuning で 1 回だけ変えてよい値を上書きする)----
export function trainConfig(run, tuning = {}) {
  const T = tuning ?? {};
  if (run === 'base') return {run, mode: 'fastest', wrapper: 'base79', virtual: false, reward: {kind: 'base'}};
  if (run === 'no_wall') {
    const m = modeSpec('no_wall');
    return {run, mode: 'no_wall', wrapper: m.policy.obs_wrapper, virtual: false, dr: DR[m.training.dr], reward: {kind: 'no_wall',
      smooth_steer: Math.abs(T.smooth_steer ?? -.06), smooth_speed: Math.abs(T.smooth_speed ?? -.03), traction_req_w: T.traction_req_w ?? .2, traction_req_free: .9,
      friction_circle_w: T.friction_circle_w ?? .2, friction_circle_free: .9, clearance_barrier_w: T.clearance_barrier_w ?? .25,
      clearance_barrier_start_m: T.clearance_barrier_start_m ?? .12, barrier_scale_m: .04, barrier_cap: 4, terminal_clearance_m: T.terminal_clearance_m ?? .075},
      // no_wall の作り直し(r2、2026-10-14、deviations.md): hard_dr = 遅い舵 + 遅延の回を増やす(DR-NW-train の範囲の中で割合だけ変える)、
      // governor = 走行中のシールド v3 の速さの見張りと同じ上限を学習でもかける(本当の舵の速さ・遅延から。走行中は見積もりから)。どちらも tuning に無ければ今までと同じ
      hard_dr: T.hard_dr ?? null, governor: T.governor ?? null};
  }
  if (run === 'recover') {
    // 1 回だけの調整: 種類の割合(types_p)・幅の条件(zone_width_m)は学習の決まりの写しを変える
    const spec = {...TRAIN_SPEC, types: T.types_p ?? TRAIN_SPEC.types, zone: {...TRAIN_SPEC.zone, width_min_m: T.zone_width_m ?? TRAIN_SPEC.zone.width_min_m}};
    return {run, mode: 'recover', wrapper: 'base79v', virtual: true, perturb: {spec, types_p: T.types_p ?? null,
      curriculum: {from: .5, to: T.magnitude_curriculum_max ?? 1.2, steps: 3_000_000}, zone_width_m: T.zone_width_m ?? null},
      reward: {kind: 'recover', gate_miss_penalty: T.gate_miss_penalty ?? -3}};
  }
  if (run === 'cond') {
    const m = modeSpec('line');
    // gate_miss_terminal: modes.json line.training.reward.terminal = 「base と同じ」(v2-gatecap はゲートの外しを −10 で終える)。観測は仮想のゲート番号のまま、
    // 外した(missEvent)ら −10 で終える(2026-10-14 学習の段 speed で直した。これが無いと cond はゲートを飛ばす走りを覚えた。deviations.md)
    return {run, mode: 'line', wrapper: 'cond97', virtual: true, gate_miss_terminal: true, cond: {families_p: m.training.conditioning.families_p, c_v_one_p: .4, c_v_range: [.5, 1.0]},
      reward: {kind: 'cond', tracking_w: T.tracking_w ?? .08, tracking_scale_m: T.tracking_scale_m ?? .10}};
  }
  if (run === 'speed') {
    // 速さいろいろ r2(2026-10-14、protocol_speed_r2.json の training。確認 speed r1 の指摘 3): cond97 の方策を「速さいろいろだけ」の条件で学び直す。
    // 線は走行中と同じ const c0 = 0(R)、c_v は記録の 5 段を多めに、開始は記録の分布(開始のすき間 ≥ 12 cm)を中心に。
    // 報酬は cond と同じ(v2-gatecap + 追従)に、すき間の壁(d0 から −w·min(4, ((d0 − d)/0.04)²))と「d_min < 3 cm で −10 で終える」を足す。
    // 走行中の仕組み(mode_runtime.mjs・cond97)は変えない。経路いろいろ(cond)の学習の値も変えない
    const S = SPEED_R2_DEFAULTS;
    return {run, mode: 'speed', wrapper: 'cond97', virtual: true, gate_miss_terminal: true,
      cond: {line_fixed: S.line_fixed, c_v_levels: T.c_v_levels ?? S.c_v_levels, c_v_levels_p: T.c_v_levels_p ?? S.c_v_levels_p, c_v_one_p: 0, c_v_range: S.c_v_range},
      start_mix: T.start_mix ?? S.start_mix,
      reward: {kind: 'cond', tracking_w: T.tracking_w ?? S.tracking_w, tracking_scale_m: T.tracking_scale_m ?? S.tracking_scale_m,
        clearance_barrier_w: T.clearance_barrier_w ?? S.clearance_barrier_w, clearance_barrier_start_m: T.clearance_barrier_start_m ?? S.clearance_barrier_start_m,
        barrier_scale_m: .04, barrier_cap: 4, terminal_clearance_m: T.terminal_clearance_m ?? S.terminal_clearance_m}};
  }
  if (run === 'line') {
    // 経路いろいろ r2(2026-10-14、protocol_line_r2.json training。確認 line r1 の指摘 1〜3): cond97 の方策を経路いろいろだけで学び直す。
    // 目標の線 v2・記録の開始の分布(開始のすき間 ≥ 12 cm)を中心に・進みに追従の掛け算・すき間の壁と 3 cm の終了。走行中の仕組みは mode_runtime.mjs(v2 の線)
    const D = LINE_R2_DEFAULTS, m = modeSpec('line');
    return {run, mode: 'line', wrapper: 'cond97', virtual: true, gate_miss_terminal: true, line_profile: D.line_profile,
      cond: {families_p: T.families_p ?? m.training.conditioning.families_p, c_v_mix: D.c_v_mix},
      start_mix: T.start_mix ?? D.start_mix,
      reward: {kind: 'cond', tracking_w: T.tracking_w ?? D.tracking_w, tracking_scale_m: T.tracking_scale_m ?? D.tracking_scale_m,
        progress_track_sigma_m: T.progress_track_sigma_m ?? D.progress_track_sigma_m,
        clearance_barrier_w: T.clearance_barrier_w ?? D.clearance_barrier_w, clearance_barrier_start_m: T.clearance_barrier_start_m ?? D.clearance_barrier_start_m,
        barrier_scale_m: D.barrier_scale_m, barrier_cap: 4, terminal_clearance_m: T.terminal_clearance_m ?? D.terminal_clearance_m}};
  }
  if (run === 'obstacle') {
    // gate_miss_terminal: modes.json obstacle.training.reward.terminal = 「base と同じ」(v2-gatecap はゲートの外しを −10 で終える)。cond と同じ直し
    // (2026-10-14 学習の段 obstacle。障害物を「ゲートの外を回って」よける走りを覚えないように。deviations.md)
    return {run, mode: 'obstacle', wrapper: 'obst95', virtual: true, gate_miss_terminal: true, layouts: {none_p: .1, count: [1, T.count_per_layout_max ?? 4]},
      reward: {kind: 'obstacle', obstacle_barrier_w: T.obstacle_barrier_w ?? .1, obstacle_barrier_start_m: .08, barrier_scale_m: .04, barrier_cap: 4}};
  }
  if (run === 'obstacle_r2' || run === 'obstacle_r2b') {
    // 障害物よけ r2(2026-10-14、protocol_obstacle_r2.json training)。観測・走行中の仕組み(obst95・mode_runtime.mjs)は r1 と同じ。r1 の run 'obstacle' の値は変えない
    // obstacle_r2b(r2 の後半): 進みの見張り・通り道の目標・障害物なし 20%(OBSTACLE_R2B_DEFAULTS)を足す
    const D = OBSTACLE_R2_DEFAULTS, B = run === 'obstacle_r2b' ? OBSTACLE_R2B_DEFAULTS : null;
    return {run, mode: 'obstacle', wrapper: 'obst95', virtual: true, gate_miss_terminal: true, ...(B ? {slow_progress: B.slow} : {}),
      layouts: {none_p: B ? B.none_p : D.layouts.none_p, count: [D.layouts.count[0], T.count_per_layout_max ?? D.layouts.count[1]], start_gap_m: D.layouts.start_gap_m},
      start_mix: T.start_mix ?? D.start_mix,
      reward: {kind: 'obstacle', r2: true, obstacle_barrier_w: T.obstacle_barrier_w ?? D.obstacle_barrier_w, obstacle_barrier_start_m: D.obstacle_barrier_start_m,
        barrier_scale_m: D.barrier_scale_m, barrier_cap: 4,
        clearance_barrier_w: D.clearance_barrier_w, clearance_barrier_start_m: D.clearance_barrier_start_m, clearance_barrier_scale_m: D.clearance_barrier_scale_m,
        terminal_clearance_m: D.terminal_clearance_m,
        approach_w: D.approach_w, approach_v0_mps: D.approach_v0_mps, approach_d0_m: D.approach_d0_m, approach_k_per_s: D.approach_k_per_s,
        ...(B ? {guidance: B.guidance} : {})}};
  }
  throw new Error('env_modes: 知らない学習の種類です: ' + run);
}

let nominalLineFastest = null, gatePosts = null, baseGrid = null, assetsLoaded = false;
function obstacleAssets(baseGeom) {
  if (!assetsLoaded) {
    assetsLoaded = true;
    const p = path.join(here, 'nominal_line_fastest.json');
    nominalLineFastest = fs.existsSync(p) ? Float64Array.from(JSON.parse(fs.readFileSync(p, 'utf8')).ey_mean.map(v => v ?? 0)) : null;
    gatePosts = loadBaseWorld().disks.filter(d => d.kind === 'gate-post').map(d => ({x: d.x, z: d.z}));
    baseGrid = buildBaseClearanceGrid(baseGeom);
  }
  return {nominalLine: nominalLineFastest, gatePosts, grid: baseGrid};
}

export class ModeEnv {
  constructor({core, run = 'base', mode = 'train', episodeMode = 'continuous', timeLimit = 20, tuning = {}, envIndex = 0} = {}) {
    if (!core) throw new Error('env_modes: core(fast_core_modes.mjs)が要ります');
    Object.assign(this, {core, run, mode, episodeMode, timeLimit, envIndex});
    this.cfg = trainConfig(run, tuning);
    this.dim = OBS_WRAPPERS[this.cfg.wrapper].dim;
    this.baseWorld = loadBaseWorld();
    this.baseGeometry = prepareGeometry(this.baseWorld);
    this.reward = BASE_REWARD;
    this.obs = new Float64Array(this.dim);
    this.aux = {};
    this.episodes = 0; this.sim = null; this.globalSteps = 0;
    this.perturbScale = this.cfg.perturb ? this.cfg.perturb.curriculum.from : 1;   // カリキュラムの始め(Python が setProgress で進める)
    this.mix = this.cfg.start_mix ?? startMixFor(run);   // speed(r2)は学習の設定の start_mix
  }
  // 学習の進み(Python から。カリキュラム用)
  setProgress(globalSteps) {
    this.globalSteps = globalSteps;
    const c = this.cfg.perturb?.curriculum;
    if (c) this.perturbScale = c.from + (c.to - c.from) * Math.min(1, globalSteps / c.steps);
  }
  // 開始(停止状態)。start が無ければモードの分布から rng で引く。invalid なら false
  reset(rng, start = null, extra = {}) {
    const st0 = start ?? sampleStartMix(rng, this.mix);
    // 回ごとの条件(引く数はいつも同じ)
    const dr = this.cfg.dr ? (extra.dr !== undefined ? extra.dr : sampleDrSpec(rng, this.cfg.dr)) : (extra.dr ?? null);
    let line = null, cV = 1;
    if (this.cfg.cond) {
      const C = this.cfg.cond, lp = sampleLineParams(rng, C.families_p ?? {}), uc = rng.uniform(), uv = rng.uniform();
      // speed(r2): 線は走行中と同じ const c0 = 0、c_v は c_v_levels_p の割合で 5 段から等確率(残りは一様)。引く数は cond と同じ
      line = extra.line ?? C.line_fixed ?? lp;
      if (C.c_v_levels) cV = extra.c_v ?? (uc < C.c_v_levels_p ? C.c_v_levels[Math.min(C.c_v_levels.length - 1, Math.floor(uv * C.c_v_levels.length))]
        : C.c_v_range[0] + (C.c_v_range[1] - C.c_v_range[0]) * uv);
      else if (C.c_v_mix) {   // line(r2): 成分を uc で選び、その範囲で uv(引く数は cond と同じ)
        let acc = 0, comp = C.c_v_mix[C.c_v_mix.length - 1];
        for (const q of C.c_v_mix) { acc += q.p; if (uc < acc) { comp = q; break; } }
        cV = extra.c_v ?? comp.range[0] + (comp.range[1] - comp.range[0]) * uv;
      }
      else cV = extra.c_v ?? (uc < C.c_v_one_p ? 1 : C.c_v_range[0] + (C.c_v_range[1] - C.c_v_range[0]) * uv);
    }
    // no_wall r2: 遅い舵 + 遅延の回を増やす(引く数はいつも 3 つ。hard_dr が無ければ引かない = 今までと同じ乱数の列)
    const H = this.cfg.hard_dr;
    if (H && dr && extra.dr === undefined) {
      const uH = rng.uniform(), uS = rng.uniform(), uD = rng.uniform();
      if (!dr.nominal && uH < H.fraction) {
        dr.physics = {...dr.physics, steerRate: H.steerRate[0] + (H.steerRate[1] - H.steerRate[0]) * uS};
        dr.delay = uD < H.delay2_p ? 2 : 1; dr.hard = true;
      }
    }
    const pSeed = rng.uint32();
    const {world: active, variation} = activeWorldFor(this.baseWorld, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
    let obstacles = extra.obstacles ?? null;
    if (this.cfg.layouts && !extra.obstacles) {
      const uNone = rng.uniform(), lrng = createRng(rng.uint32());
      if (uNone >= this.cfg.layouts.none_p) {
        const a = obstacleAssets(this.baseGeometry);
        const sp = projectWindow(active.spawn.x, active.spawn.z, 0, LOOP_LENGTH).s;
        // r2(start_gap_m): 開始の前後の空きを試験・記録と同じ startClearOfObstacles(1.5 m)で決める(layouts.mjs の余白 0.3 m を足さない)。
        // 置くときは開始を見ずに置き、開始の前後 1.5 m に入ったものを除く(通り道の保証は障害物を除いても崩れない)
        const gap = this.cfg.layouts.start_gap_m ?? null;
        obstacles = generateLayout(lrng, {baseGeometry: this.baseGeometry, nominalLine: a.nominalLine, gatePosts: a.gatePosts, count: this.cfg.layouts.count,
          startS: gap == null ? sp : null, grid: a.grid, maxTries: 40}).obstacles;
        if (gap != null) obstacles = obstacles.filter(o => startClearOfObstacles(sp, [o]));
      } else obstacles = [];
    }
    const world = obstacles?.length ? worldWithObstacles(active, obstacles) : active;
    const geometry = obstacles?.length ? geometryWithObstacles(this.baseGeometry, obstacles) : this.baseGeometry;
    const nominalConfig = simConfig({episodeMode: this.episodeMode, timeLimit: this.timeLimit, seed: st0.seed ?? 1});
    const physics = dr?.physics ?? {};
    const simCfg = {...nominalConfig, ...physics, ...(dr?.frictionCircle ? {frictionCircle: true} : {})};
    const sim = this.core.createSimulation(world, simCfg);
    sim.reset({seed: simCfg.seed, scenario: 'baseline'});
    this.sim = sim; this.world = world; this.variation = variation; this.view = sim.peek();
    this.config = this.view.config;
    this.trackerConfig = {...this.config, ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))};
    this.dr = dr; this.obstacles = obstacles ?? []; this.line = line; this.cV = cV; this.speedCap = BASE_SPEED_CAP * cV;
    this.start = {gate: variation.gate, left_offset_m: variation.left_offset_m, left_yaw_deg: variation.left_yaw_deg, seed: simCfg.seed, component: st0.component ?? null};
    if (dr && !dr.nominal) this.start.perturb = {physics: {...physics}, delay: dr.delay, steerGain: dr.steerGain, speedGain: dr.speedGain, frictionCircle: dr.frictionCircle};
    if (dr) this.start.steer_noise = dr.steerNoise;
    if (line) { this.start.line = line; this.start.c_v = cV; }
    if (this.obstacles.length) this.start.obstacles = this.obstacles.length;
    const st = this.view.state;
    if (st.terminal) { this.invalid = true; return false; }
    this.invalid = false;
    this.wrapper = createObsWrapper({wrapper: this.cfg.wrapper, world, config: this.trackerConfig, geometry,
      lineFactory: line ? (s0 => createLineProfile(line, s0, {version: this.cfg.line_profile ?? 'v1'})) : null, cV, obstacles: this.obstacles});
    const g = this.wrapper.reset(st, this.view.metrics.route_index);
    this.obstGeom = this.obstacles.length ? obstacleOnlyGeometry(this.obstacles) : null;
    // obstacle_r2b だけ: 通り道の目標(学習だけ)と進みの見張りの履歴
    this.passTargets = this.cfg.reward?.guidance && this.obstacles.length ? passageTargets(obstacleAssets(this.baseGeometry).grid, this.obstacles,
      {margin: this.cfg.reward.guidance.margin_m}) : null;
    this.sHist = this.cfg.slow_progress ? [] : null;
    this.wrapper.observe(st, 0, 0, this.obs, this.aux);
    if (st0.min_initial_clearance_m != null && this.aux.d_min < st0.min_initial_clearance_m) { this.invalid = true; return false; }
    this.prevA0 = 0; this.prevA1 = 0; this.stallFrames = 0;
    this.s0 = g.s; this.sCap = Math.min(g.s, g.gateS + this.reward.progress_gate_margin); this.sCap0 = this.sCap;
    this.queue = []; this.noiseRng = createRng(mix32(simCfg.seed, 0x401e));
    this.perturber = this.cfg.perturb ? createPerturber({spec: this.cfg.perturb.spec, seed: pSeed, scale: this.perturbScale, maxSteer: this.config.maxSteer,
      wheelbase: this.config.wheelbase}) : null;
    this.stats = {return: 0, frames: 0, min_clearance: this.aux.d_min, max_speed: 0, gate_miss_frame: null, gate_misses: 0, injected: 0, kicks: 0,
      track_abs_sum: 0, track_n: 0, track_gt12: 0, track_progress_loss: 0, rho_req_gt1: 0, rho_comb_gt1: 0, moving: 0, min_obstacle_clearance: Infinity,
      reward_parts: {progress: 0, smooth: 0, traction: 0, clearance: 0, barrier: 0, friction: 0, tracking: 0, obstacle: 0, gate_miss: 0, terminal: 0,
        ...(this.cfg.reward?.progress_track_sigma_m ? {track_progress: 0} : {}), ...(this.cfg.reward?.approach_w ? {approach: 0} : {}),
        ...(this.cfg.reward?.guidance ? {guidance: 0} : {})}};
    this.done = false; this.episodes++;
    return true;
  }
  resetRandom(rng, maxTries = 100) {
    for (let t = 0; t < maxTries; t++) if (this.reset(rng)) return t;
    throw new Error('env_modes: 開始を 100 回引いても衝突しない開始になりません');
  }
  // 方策の行動 a ∈ [−1, 1]²。戻り値に injected(差し込んだ行動なら 1)
  step(a0, a1) {
    if (this.done || this.invalid) throw new Error('env_modes: 終わったエピソードは reset してから進めます');
    const c = actionToCommand(a0, a1, this.speedCap, this.config.maxSteer);
    let cmd = {steering: c.steering, speed: c.speed}, injected = false;
    if (this.perturber) {
      const st = this.view.state;
      const r = this.perturber.step(st.frame, {s: this.aux.s, ey: this.aux.lateral_error, dmin: this.aux.d_min, x: st.x, z: st.z, yaw: st.yaw,
        speedCap: this.speedCap, policy: cmd});
      if (r.inj) { cmd = {steering: r.inj.steering ?? cmd.steering, speed: r.inj.speed ?? cmd.speed}; injected = true; }
      if (r.kick) this.applyKick(r.kick);
    }
    // no_wall r2: シールド v3 の速さの見張りと同じ上限(T = d·dt + θ/r、上限 = v_max·T_標準/T。標準の物理・3 フレーム目より前は上限なし)
    const GV = this.cfg.governor;
    if (GV && this.dr && !this.dr.nominal && this.view.state.frame >= (GV.min_frame ?? 3)) {
      const r = this.dr.physics?.steerRate ?? PHYSICS.steerRate, d = this.dr.delay ?? 0;
      const T = d * this.config.dt + GV.theta_rad / r, Tn = GV.theta_rad / PHYSICS.steerRate;
      if (T > Tn * (1 + 1e-9)) { const cap = GV.v_max * Tn / T; if (cmd.speed > cap) { cmd = {steering: cmd.steering, speed: cap}; this.stats.governed = (this.stats.governed ?? 0) + 1; } }
    }
    const [an0, an1] = appliedAction(cmd, c, this.speedCap, this.config.maxSteer);
    let ap = {steering: cmd.steering, speed: cmd.speed, a0: an0, a1: an1};
    const dr = this.dr;
    if (dr && dr.delay) { this.queue.push(ap); ap = this.queue.length > dr.delay ? this.queue.shift() : {steering: 0, speed: 0, a0: 0, a1: -1}; }
    let steering = ap.steering * (dr?.steerGain ?? 1), speed = ap.speed * (dr?.speedGain ?? 1);
    if (dr?.steerNoise) steering += dr.steerNoise * gaussian(this.noiseRng);
    if (injected) this.stats.injected++;
    const out = this.stepCommand(steering, speed, ap.a0, ap.a1);
    out.injected = injected;
    return out;
  }
  // 状態のキック(高速版の peek().state を書き換える)。当たる・5 cm 未満になるキックは戻す
  applyKick(k) {
    const s = this.sim.peek().state, keep = {x: s.x, z: s.z, yaw: s.yaw, speed: s.speed};
    if (k.yaw) s.yaw = wrap(s.yaw + k.yaw);
    if (k.lateral) { s.x += k.lateral * -Math.sin(keep.yaw); s.z += k.lateral * -Math.cos(keep.yaw); }
    if (k.speed_factor) s.speed = clamp(s.speed * k.speed_factor, 0, this.config.maxSpeed);
    const geom = this.obstacles.length ? geometryWithObstacles(this.baseGeometry, this.obstacles) : this.baseGeometry;
    if (clearance(geom, s.x, s.z, s.yaw, this.config.vehicleLength, this.config.vehicleWidth, .5) < .05) Object.assign(s, keep);
    else this.stats.kicks++;
  }
  stepCommand(steering, speed, a0n, a1n) {
    const R = this.reward, RC = this.cfg.reward, dt = this.config.dt;
    const prevV = this.view.state.speed, prevYaw = this.view.state.yaw;
    const lapsBefore = this.view.metrics.completed_laps ?? 0;
    this.sim.step({steering, speed}, true); this.view = this.sim.peek();
    const st = this.view.state, m = this.view.metrics;
    const g = this.wrapper.advance(st, m.route_index);
    this.wrapper.observe(st, a0n, a1n, this.obs, this.aux);
    const aux = this.aux, s = g.s;
    const sCap = Math.min(s, g.gateS + R.progress_gate_margin);
    const progress = Math.max(R.progress_min, Math.min(sCap - this.sCap, st.speed * dt + R.progress_slack));
    this.sCap = sCap;
    const ws = RC.kind === 'no_wall' ? RC.smooth_steer : R.smooth_steer, wv = RC.kind === 'no_wall' ? RC.smooth_speed : R.smooth_speed;
    const smooth = ws * (a0n - this.prevA0) ** 2 + wv * (a1n - this.prevA1) ** 2;
    const rho = st.speed * st.speed * Math.abs(Math.tan(st.steering)) / (this.config.wheelbase * aux.mu * G);
    let traction, clear, barrier = 0, friction = 0, tracking = 0, obstacle = 0, gateMiss = 0;
    if (RC.kind === 'no_wall') {
      traction = RC.traction_req_w * Math.max(0, rho - RC.traction_req_free);
      clear = 0;
      if (aux.d_min < RC.clearance_barrier_start_m) barrier = RC.clearance_barrier_w * Math.min(RC.barrier_cap, ((RC.clearance_barrier_start_m - aux.d_min) / RC.barrier_scale_m) ** 2);
      if (st.speed > .3) {
        const al = (st.speed - prevV) / dt, om = wrap(st.yaw - prevYaw) / dt, rc = Math.hypot(al, st.speed * om) / (aux.mu * G);
        friction = RC.friction_circle_w * Math.max(0, rc - RC.friction_circle_free);
        this.stats.moving++; if (rc > 1) this.stats.rho_comb_gt1++; if (rho > 1) this.stats.rho_req_gt1++;
      }
    } else {
      traction = R.traction * Math.max(0, rho - R.traction_free);
      clear = R.clearance * Math.max(0, (R.clearance_target - aux.d_min) / R.clearance_target);
    }
    let trackProgress = 0;
    if (RC.kind === 'cond') {
      const e = aux.lateral_error - aux.e_star;
      tracking = RC.tracking_w * Math.min(1, (e / RC.tracking_scale_m) ** 2);
      // line(r2)だけ: 進みに追従の掛け算 f(e) = 1/(1 + (e/σ)²)。進みが正のときだけ、減ったぶんを trackProgress に(cond・speed の設定には鍵が無いので今までと同じ)
      if (RC.progress_track_sigma_m && progress > 0) trackProgress = progress * (1 - 1 / (1 + (e / RC.progress_track_sigma_m) ** 2));
      if (RC.progress_track_sigma_m && s - this.s0 >= 3 && Math.abs(e) > .12) this.stats.track_gt12++;
      // speed(r2)だけ: すき間の壁(cond の設定には鍵が無いので今までと同じ)
      if (RC.clearance_barrier_w && aux.d_min < RC.clearance_barrier_start_m)
        barrier = RC.clearance_barrier_w * Math.min(RC.barrier_cap, ((RC.clearance_barrier_start_m - aux.d_min) / RC.barrier_scale_m) ** 2);
      if (s - this.s0 >= 3) { this.stats.track_abs_sum += Math.abs(e); this.stats.track_n++; }
    }
    let approach = 0;
    if (this.obstGeom) {
      const d = clearance(this.obstGeom, st.x, st.z, st.yaw, this.config.vehicleLength, this.config.vehicleWidth, .5);
      if (d < this.stats.min_obstacle_clearance) this.stats.min_obstacle_clearance = d;
      if (d < RC.obstacle_barrier_start_m) obstacle = RC.obstacle_barrier_w * Math.min(RC.barrier_cap, ((RC.obstacle_barrier_start_m - d) / RC.barrier_scale_m) ** 2);
      // obstacle r2 だけ: 足した障害物に近いときの速さの罰(r1 の設定には鍵が無いので今までと同じ)
      if (RC.approach_w) { const vOk = RC.approach_v0_mps + RC.approach_k_per_s * Math.max(0, d - RC.approach_d0_m); if (st.speed > vOk) approach = RC.approach_w * (st.speed - vOk) ** 2; }
    }
    // obstacle r2 だけ: 全部の障害物(コース + 足したもの)のすき間の壁
    if (RC.kind === 'obstacle' && RC.r2 && aux.d_min < RC.clearance_barrier_start_m)
      barrier = RC.clearance_barrier_w * Math.min(RC.barrier_cap, ((RC.clearance_barrier_start_m - aux.d_min) / RC.clearance_barrier_scale_m) ** 2);
    // obstacle_r2b だけ: 通り道の目標(次の障害物の手前 2.5 m から 0.3 m 過ぎるまで)
    let guide = 0;
    if (this.passTargets) {
      const G2 = RC.guidance, at = activeTarget(this.passTargets, s, {ahead: G2.ahead_m, full: G2.full_m, behind: G2.behind_m});
      if (at) guide = G2.w * Math.min(1, ((aux.lateral_error - at.t.e_target) / G2.scale_m) ** 2) * at.ramp;
    }
    let reward = progress - trackProgress - smooth - traction - clear - barrier - friction - tracking - obstacle - approach - guide, terminal = 0;
    let terminated = false, truncated = false, reason = null;
    if (st.terminal) {
      reason = st.reason;
      if (reason === 'collection_duration_complete' || reason === 'time_limit') truncated = true;
      else { terminated = true; terminal = R.terminal[reason] ?? 0; }
    } else {
      if (aux.gate_along >= 0 && this.stats.gate_miss_frame === null && !this.cfg.virtual) this.stats.gate_miss_frame = st.frame;
      if (g.missEvent) { this.stats.gate_misses++; if (this.stats.gate_miss_frame === null) this.stats.gate_miss_frame = st.frame; }
      if (st.sim_time >= R.stall_after_s - 1e-9 && st.speed < R.stall_speed) this.stallFrames++; else this.stallFrames = 0;
      if (this.mode === 'train') {
        if (!this.cfg.virtual && aux.gate_along >= 0) reason = 'gate_miss';
        else if (this.cfg.virtual && this.cfg.gate_miss_terminal && g.missEvent) reason = 'gate_miss';
        else if (this.stallFrames >= R.stall_frames) reason = 'stall';
        else if (s < this.wrapper.tracker.sMax - R.wrong_way_m) reason = 'wrong_way';
        else if (RC.kind === 'no_wall' && aux.d_min < RC.terminal_clearance_m) reason = 'clearance';
        else if (RC.kind === 'cond' && RC.terminal_clearance_m != null && aux.d_min < RC.terminal_clearance_m) reason = 'clearance';   // speed(r2)だけ
        else if (RC.kind === 'obstacle' && RC.r2 && aux.d_min < RC.terminal_clearance_m) reason = 'clearance';   // obstacle r2 だけ
        else if (this.sHist && st.sim_time >= this.cfg.slow_progress.after_s - 1e-9 && this.sHist.length >= this.cfg.slow_progress.window_frames
          && s - this.sHist[this.sHist.length - this.cfg.slow_progress.window_frames] < this.cfg.slow_progress.min_progress_m) reason = 'slow';   // obstacle_r2b だけ
        if (reason) { terminated = true; terminal = R.terminal[reason] ?? -10; }
        else if (this.cfg.virtual && g.missEvent && RC.gate_miss_penalty) gateMiss = -RC.gate_miss_penalty;
      }
    }
    reward += terminal - gateMiss;
    if (this.sHist) { this.sHist.push(s); if (this.sHist.length > 64) this.sHist.shift(); }
    this.prevA0 = a0n; this.prevA1 = a1n;
    const S = this.stats;
    S.return += reward; S.frames++;
    const P = S.reward_parts;
    P.progress += progress; P.smooth -= smooth; P.traction -= traction; P.clearance -= clear; P.barrier -= barrier; P.friction -= friction;
    P.tracking -= tracking; P.obstacle -= obstacle; if (RC.approach_w) P.approach -= approach; if (RC.guidance) P.guidance -= guide; if (trackProgress) { P.track_progress -= trackProgress; S.track_progress_loss += trackProgress; } P.gate_miss -= gateMiss; P.terminal += terminal;
    if (aux.d_min < S.min_clearance) S.min_clearance = aux.d_min;
    if (st.speed > S.max_speed) S.max_speed = st.speed;
    if ((m.completed_laps ?? 0) !== lapsBefore) S.lap_count = m.completed_laps;
    this.done = terminated || truncated;
    this.last = {reward, terminated, truncated, reason, progress, s};
    return this.last;
  }
  summary() {
    const st = this.view.state, m = this.view.metrics, S = this.stats;
    const laps = (m.lap_events ?? []).map(l => ({lap: l.lap_index, elapsed: l.elapsed, collisions: l.collisions, gates: l.passed_gates,
      crossings: l.required_crossing_events.length}));
    if (this.episodeMode === 'single' && st.reason === 'route_complete') laps.push({lap: 1, elapsed: m.elapsed, collisions: m.collisions, gates: m.passed_gates,
      crossings: m.required_crossing_events.length});
    return {episode: this.episodes, run: this.run, start: this.start, frames: st.frame, reason: this.last?.reason ?? st.reason, sim_reason: st.reason,
      return: S.return, reward_parts: S.reward_parts, progress_m: this.wrapper.tracker.s - this.s0, laps, completed_laps: m.completed_laps ?? 0,
      collisions: m.collisions, min_clearance: S.min_clearance, max_speed: S.max_speed, gate_miss_frame: S.gate_miss_frame, gate_misses: S.gate_misses,
      injected_frames: S.injected, kicks: S.kicks, events: this.perturber ? this.perturber.events.length : 0, perturb_scale: this.perturber ? this.perturbScale : null,
      tracking_abs_mean: S.track_n ? S.track_abs_sum / S.track_n : null,
      ...(this.cfg.reward.progress_track_sigma_m ? {tracking_gt12_frac: S.track_n ? S.track_gt12 / S.track_n : null, c_v: this.cV, family: this.line?.family ?? null} : {}),
      rho_req_gt1_frac: S.moving ? S.rho_req_gt1 / S.moving : null,
      rho_comb_gt1_frac: S.moving ? S.rho_comb_gt1 / S.moving : null, min_obstacle_clearance: Number.isFinite(S.min_obstacle_clearance) ? S.min_obstacle_clearance : null,
      ...(this.cfg.hard_dr || this.cfg.governor ? {hard: !!this.dr?.hard, governed_frames: S.governed ?? 0} : {})};
  }
}
