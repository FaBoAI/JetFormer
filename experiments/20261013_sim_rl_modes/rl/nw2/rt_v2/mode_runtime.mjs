// 強化学習モードの教師(走行中の仕組みの全部)。2026-10-13、sim_rl_modes。
// **Node の試験(元の sim-core.js を動かす)とブラウザの教師(JetFormer の _jf/rl/teacher.js)が同じこのファイルを使う**(純粋な ESM)。
// 試験した走り = 記録する走り にするため、観測の包み・方策・シールド・乱しの予定・経路の形は全部ここで決める。
// 乱数は config.seed(シーンの種)とフレーム番号だけから作り、時刻を使わない。同じフレームで 2 回呼ばれても同じ答えを返す。
//
//   const t = createModeTeacher(world, config, variation, {mode: 'no_wall', weights, params, policySha256});
//   t.action(snapshot) → {steering, speed, info: {jf: {v, m, f, ph, l, ev, cl, es?, sh?, p?}}}
//   t.metadata(context.teacher に入る。モードの中の全シーンで同じ)・t.reference・t.options
//
// world: アプリの activeWorld(開始ゲートで回した経路。障害物よけでは kind 'obstacle' の障害物を含む)。config: アプリの config(標準の物理)。
// snapshot: sim.snapshot() の形 {state, metrics, config}。
import {OBS_DIM, FEATURES_VERSION, REFERENCE_LOOP_SHA256, LOOP_LENGTH, actionToCommand, commandToAction, prepareGeometry, routeOffset} from './features.mjs';
import {createPolicy, POLICY_SCHEMA} from './policy_forward.mjs';
import {createObsWrapper, OBS_WRAPPERS} from './obs_modes.mjs';
import {createShield, SHIELD_DEFAULTS} from './shield.mjs';
import {createPerturber, RECORD_SPEC} from './perturb.mjs';
import {createLineProfile, LINE_FAMILIES} from './line_profiles.mjs';
import {obstaclesFromWorld, worldWithoutObstacles} from './layouts.mjs';
import {CATALOGUE, CATALOGUE_SHA256} from './catalogue.mjs';
import {ROOM_TABLE_SHA256} from './room_table.mjs';
import {createRng, mix32} from './rng.mjs';

export const RUNTIME_VERSION = 'jf-sim-rl-mode-runtime/v1';
export const TEACHER_SCHEMA = 'jetformer-sim-rl-teacher/v1';
export const MODE_POLICY_SCHEMA = 'jf-sim-rl-mode-policy/v1';
export const BASE_SPEED_CAP = 4.5;
export const MODE_KEYS = Object.freeze(CATALOGUE.modes.map(m => m.key));
export const MODE_NUMBER = Object.freeze({fastest: 1, no_wall: 2, recover: 3, line: 4, speed: 5, obstacle: 6});   // protocol.json modes.mode_number

export function modeSpec(key) {
  const m = CATALOGUE.modes.find(x => x.key === key);
  if (!m) throw new Error('mode_runtime: 知らないモードです: ' + key);
  return m;
}
// モードごとの走行中の仕組み(modes.json の runtime を機械が読める形に)
export function runtimeSpec(key) {
  const m = modeSpec(key);
  return {key, obs_wrapper: m.policy.obs_wrapper, shield: !!m.runtime.shield, perturbation: m.runtime.perturbation ? 'record' : null,
    conditioning: key === 'line' || key === 'speed' ? key : key === 'obstacle' ? 'layout' : null,
    speed_cap_base_mps: BASE_SPEED_CAP, speed_cap_scaled: key === 'line' || key === 'speed'};
}
// 重み(rl-bestlap-policy/v1 または jf-sim-rl-mode-policy/v1)→ policy_forward の形
export function policyFromWeights(weights, expectDim = null) {
  if (!weights) throw new Error('mode_runtime: 方策の重みがありません');
  const w = weights.schema === MODE_POLICY_SCHEMA ? {...weights, schema: POLICY_SCHEMA} : weights;
  if (w.schema !== POLICY_SCHEMA) throw new Error('mode_runtime: 重みの schema が違います: ' + weights.schema);
  if (expectDim != null && w.obs_dim !== expectDim) throw new Error(`mode_runtime: 重みの観測の次元 ${w.obs_dim} がモードの ${expectDim} と違います`);
  return createPolicy(w);
}

// 記録の「経路いろいろ」の形の割り当て(modes.json line.recording.per_scene_params.family_plan_15 と同じ割合)。シーンの種から決まる(プランが無いとき)
const LINE_PLAN15 = [['const_left', 2], ['const_right', 2], ['const_center', 1], ['inside', 2], ['outside', 2], ['late_apex', 2], ['early_apex', 1], ['random', 3]];
export function lineParamsFor(kind, rng) {
  const u1 = rng.uniform(), u2 = rng.uniform(), seed = rng.uint32();
  if (kind === 'const_left') return {family: 'const', c0: .6 + .3 * u1};
  if (kind === 'const_right') return {family: 'const', c0: -.9 + .3 * u1};
  if (kind === 'const_center') return {family: 'const', c0: -.2 + .4 * u1};
  if (kind === 'random') return {family: 'random', seed};
  if (LINE_FAMILIES.includes(kind)) return {family: kind, a: .5 + .4 * u2};
  throw new Error('mode_runtime: 知らない経路の形の指定です: ' + kind);
}
export const SPEED_LEVELS = Object.freeze([1.0, .875, .75, .625, .5]);
// シーンの条件 p を、プランの指定(params)とシーンの種(seed)から決める(指定の無い値は種から決定的に)
export function sceneParams(key, seed, params = {}) {
  const rng = createRng(mix32(seed >>> 0, 0x5ce2e));
  const p = {mode: key, seed: seed >>> 0};
  if (key === 'line') {
    let line = params.line;
    if (!line || line === 'auto') {
      const total = LINE_PLAN15.reduce((a, [, w]) => a + w, 0); let u = rng.uniform() * total, kind = 'random';
      for (const [k, w] of LINE_PLAN15) { if (u < w) { kind = k; break; } u -= w; }
      line = lineParamsFor(kind, rng);
    } else if (typeof line === 'string') line = lineParamsFor(line, rng);
    p.line = line;
    p.c_v = params.c_v ?? +(0.9 + 0.1 * rng.uniform()).toFixed(4);
  } else if (key === 'speed') {
    p.line = {family: 'const', c0: 0};
    p.c_v = params.c_v ?? SPEED_LEVELS[Math.floor(rng.uniform() * SPEED_LEVELS.length) % SPEED_LEVELS.length];
  } else if (key === 'recover') {
    p.perturb_class = params.perturb_class ?? 'auto';
  } else if (key === 'obstacle') {
    p.layout_id = params.layout_id ?? null;
  }
  if (params.speed_limit != null) p.speed_limit = +params.speed_limit;
  if (params.start) p.start = params.start;
  return p;
}
// 適用した操作 → 観測の「前回の行動」(方策の値のままなら方策の値、変えたなら単位から戻す)
export function appliedAction(cmd, policyCmd, speedCap, maxSteer = .52) {
  const a = commandToAction(cmd.steering, cmd.speed, speedCap, maxSteer);
  return [cmd.steering === policyCmd.steering ? policyCmd.a0 : a.a0, cmd.speed === policyCmd.speed ? policyCmd.a1 : a.a1];
}

export function createModeTeacher(world, config, variation = {}, {mode, weights, params = {}, policySha256 = null, obstacleSet = null,
  shieldParams = {}, plannedTimeS = null, policy = null, overrides = {}} = {}) {
  // overrides(Node の試験だけ): {shield: false}(NW-OFF)・{perturbation: false}(乱しなしの線)。ブラウザでは使わない
  const key = mode, M = modeSpec(key), R = runtimeSpec(key);
  if (!world || !Array.isArray(world.route)) throw new Error('mode_runtime: world がありません');
  routeOffset(world.route);   // Default(v3)の経路を回したものでなければ例外(RL の観測は v3 専用)
  const seed = (config?.seed ?? 1) >>> 0;
  const p = sceneParams(key, seed, params);
  const wrapperName = R.obs_wrapper, dim = OBS_WRAPPERS[wrapperName].dim;
  const act = policy ?? policyFromWeights(weights, dim);
  const cV = p.c_v ?? 1, speedCap = BASE_SPEED_CAP * cV, maxSteer = config?.maxSteer ?? .52;
  const obstacles = obstaclesFromWorld(world);
  if (key !== 'obstacle' && obstacles.length && !overrides.allow_obstacles) throw new Error('mode_runtime: 障害物のあるコースで使えるのは「障害物よけ」だけです');
  const geometry = prepareGeometry(world);
  const lineFactory = p.line ? (s0 => createLineProfile(p.line, s0)) : null;
  const wrapper = createObsWrapper({wrapper: wrapperName, world, config, geometry, lineFactory, cV, obstacles});
  const shield = R.shield && overrides.shield !== false ? createShield({geometry, config, params: shieldParams}) : null;
  const perturber = R.perturbation && overrides.perturbation !== false ? createPerturber({spec: RECORD_SPEC, seed, cls: p.perturb_class, maxSteer, wheelbase: config?.wheelbase ?? .26}) : null;
  const obs = new Float64Array(dim), aux = {}, pa = [0, 0];
  let lastF = -1, memo = null, prevA = [0, 0], prevCmd = {steering: 0, speed: 0}, shieldCount = 0, frames = 0;
  const sha12 = policySha256 ? String(policySha256).slice(0, 12) : null;
  const log = [];
  function action(snapshot, extra = null) {
    const st = snapshot.state, m = snapshot.metrics ?? {}, f = st.frame;
    if (memo && f === lastF) return JSON.parse(JSON.stringify(memo));
    if (f === 0) { wrapper.reset(st, m.route_index ?? 0); prevA = [0, 0]; prevCmd = {steering: 0, speed: 0}; }
    else if (f === lastF + 1) wrapper.advance(st, m.route_index ?? 0);
    else throw new Error(`mode_runtime: フレームが飛びました(${lastF} → ${f})`);
    // Node の試験で制御遅延をかけたとき: 観測の「前回の行動」は実際に適用した行動(r2 と同じ約束)。ブラウザでは使わない
    if (extra?.applied) prevA = [extra.applied[0], extra.applied[1]];
    wrapper.observe(st, prevA[0], prevA[1], obs, aux);
    act(obs, pa);
    const pc = actionToCommand(pa[0], pa[1], speedCap, maxSteer);
    let cmd = {steering: pc.steering, speed: pc.speed}, ph = 'drive', l = 1, ev = 0, sh = 0;
    // 速さの上限(速さいろいろの代わりの方法: 学習しない上限 {4.5, 4.0, 3.5, 3.0, 2.5} m/s。protocol acceptance.fallbacks.speed)
    if (p.speed_limit != null && cmd.speed > p.speed_limit) cmd.speed = p.speed_limit;
    if (perturber) {
      const r = perturber.step(f, {s: aux.s, ey: aux.lateral_error, dmin: aux.d_min, x: st.x, z: st.z, yaw: st.yaw, speedCap, policy: cmd});
      if (r.inj) { cmd = {steering: r.inj.steering ?? cmd.steering, speed: r.inj.speed ?? cmd.speed}; ph = 'perturb'; l = 0; }
      ev = r.ev;
    }
    let shDbg = null;
    if (shield) {
      const r = shield.filter(st, aux.s, prevCmd, cmd);
      shDbg = {fb: r.fallback ?? null, nom: r.predicted?.nominal ?? null, pess: r.predicted?.pessimistic ?? null, n: r.checked};
      if (r.replaced) { cmd = r.cmd; sh = 1; ph = 'shield'; shieldCount++; }
    }
    prevA = appliedAction(cmd, pc, speedCap, maxSteer);
    prevCmd = cmd;
    const jf = {v: 1, m: key, f, ph, l, ev, cl: Math.round(aux.d_min * 1000)};
    if (aux.e_star !== undefined) jf.es = Math.round(aux.e_star * 1000);
    if (shield) jf.sh = sh;
    if (f === 0) jf.p = {...p, policy: sha12, runtime: RUNTIME_VERSION};
    frames++;
    const out = {steering: cmd.steering, speed: cmd.speed, info: {jf}};
    log.push({f, s: aux.s, ey: aux.lateral_error, epsi: aux.heading_error, dmin: aux.d_min, es: aux.e_star, v: st.speed, ph, l, ev, sh,
      a0: prevA[0], a1: prevA[1], steering: cmd.steering, speed: cmd.speed, missed: wrapper.missed, shield: shDbg});
    lastF = f; memo = out;
    return JSON.parse(JSON.stringify(out));
  }
  const metadata = {schema: TEACHER_SCHEMA, id: `jf-rl-${key}-v1`, mode: key, policy_sha256: policySha256, obs_wrapper: wrapperName,
    features_version: FEATURES_VERSION, reference_loop_sha256: REFERENCE_LOOP_SHA256, speed_cap_mps: R.speed_cap_scaled ? `${BASE_SPEED_CAP}·c_v` : BASE_SPEED_CAP,
    runtime_version: RUNTIME_VERSION, catalogue_sha256: CATALOGUE_SHA256, room_table_sha256: ROOM_TABLE_SHA256, obstacle_set: obstacleSet ?? null};
  if (R.shield) metadata.shield = {...SHIELD_DEFAULTS, ...shieldParams};
  if (Object.keys(overrides).length) metadata.overrides_node_test_only = {...overrides};
  return {action, metadata, reference: {length: LOOP_LENGTH, planned_time_s: plannedTimeS}, options: {mode: key}, params: p,
    get log() { return log; }, get events() { return perturber ? perturber.events : []; }, get shieldFrames() { return shieldCount; },
    get frames() { return frames; }, wrapper, spec: R};
}
export {worldWithoutObstacles};
