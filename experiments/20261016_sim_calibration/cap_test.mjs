// 区間の速度上限の試験(S4、2026-10-16)。Node だけ。protocol.json cap_test のとおり。
//
//   node cap_test.mjs --plan primary|secondary|extra|all [--shard i/n] [--only id,...]
//
// 1 条件 = 運転手 × 上限 × 物理 × ブレーキ × 世界 × 電池。24 個の開始(S24)を 60 秒の連続周回で走らせ(最初の衝突で終わる)、
// cap_test/runs/<条件>.json に書く。既にある条件は飛ばす(電源断のあとは同じコマンドで続きから)。
// 物理: P_cal = 較正版のコア(profiles/P_cal.json)、P_default = 「元の式 + 遅れ 0」の実行部(元と同じ動き。試験 T1b)。
//       どちらも fast(10/13 の高速化。T1 で元と同じ)で、上限の層は calibControl({filter}) の口に cap_runtime の createCapFilter を差す。
// アプリの遅れ 1 フレーム(最初は中立)。上限の層は遅れのあとの指令に掛かる(コアの step の始め)。
// ユーザーのファイルは読むだけ。実車・PCA9685・I2C には触れない。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {loadCalibratedCore, loadProfile, origReplayProfile} from './calib_core.mjs';
import {compileProfile, governorStep, governorInitialState, GOVERNOR_V1_SIM} from './calib_runtime.mjs';
import {createLoop, createProgressTracker, createCapFilter, zoneOfS, SECTIONS_R, CAP_RUNTIME_VERSION} from './cap_runtime.mjs';
import {choosePath, speedProfile, physicsMaps, createPPDriver, REFERENCE_LOOP_JSON, PP_VERSION} from './pp_driver.mjs';
import {prepareGeometry, clearance} from '/home/jetson/jetformer/jetformer/sim/static/rl/features.mjs';
import {createModeTeacher} from '/home/jetson/jetformer/jetformer/sim/static/rl/mode_runtime.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const OUT = '/home/jetson/jetformer_reports/sim_calibration_20261016/cap_test';
export const WORLD_V3 = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/sim/world-v3.json';
const WORLD_V3_SHA = '57de53a55e5366c004e26443d8e23722b015a9ac865044023a3ac57db55f9dc2';
const POLICY = {fastest: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee', speed: '6f4d5e8d245f4f3e076c4a5db94c9da684b5bb0ba061b999d372ab6098898ac0'};
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const readJson = p => JSON.parse(fs.readFileSync(p, 'utf8'));
export const PHYSICS_CONFIG = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42,
  vehicleWidth: .19, wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0,
  longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
export const EPISODE = Object.freeze({timeLimit: 60, episodeMode: 'continuous-laps-v1', seed: 1, delayFrames: 1});
// 芝の斜めの板(W+、protocol matrix.secondary)
export const W_PLUS_RAIL = Object.freeze({a: [4.37, -2.76], b: [4.50, -1.82], radius: .0095, height: .089, kind: 'rail'});

// ------------------------------------------------------------------ 世界・開始
export function loadWorld(kind = 'W0') {
  const text = fs.readFileSync(WORLD_V3);
  if (sha(text) !== WORLD_V3_SHA) throw new Error('cap_test: world-v3.json の sha256 が違います');
  const w = JSON.parse(text);
  if (kind === 'W+') w.segments = [...w.segments, {...W_PLUS_RAIL, a: [...W_PLUS_RAIL.a], b: [...W_PLUS_RAIL.b]}];
  else if (kind !== 'W0') throw new Error('cap_test: 世界は W0 か W+ です');
  return w;
}
// app.js の reset(): 開始ゲートで経路を回し、開始ずれを左右・向きに加える(app_emulation.mjs と同じ)
export function activeWorldFor(world, gate = 0, offsetM = 0, yawDeg = 0) {
  const startIndex = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(startIndex), ...world.route.slice(0, startIndex)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const offset = Math.max(-.3, Math.min(.3, offsetM || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  const active = {...world, route, spawn: {x: first[0] + offset * Math.sin(heading), z: first[1] - offset * Math.cos(heading), yaw: heading + turn}};
  return {world: active, variation: {gate: startIndex, left_offset_m: offset, left_yaw_deg: turn * 180 / Math.PI}};
}
function halton(i, b) { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; }
// S24: Halton(底 2・3・5)。開始のすき間 < 0.12 m は次の点に替える
export function makeStarts(world, n = 24) {
  const geom = prepareGeometry(world), starts = [], skipped = [];
  for (let i = 1; starts.length < n && i < 1000; i++) {
    const gate = Math.min(32, Math.floor(halton(i, 2) * 33)), off = -.15 + .3 * halton(i, 3), yaw = -10 + 20 * halton(i, 5);
    const {world: a} = activeWorldFor(world, gate, off, yaw);
    const c = clearance(geom, a.spawn.x, a.spawn.z, a.spawn.yaw, .42, .19, .5);
    const rec = {halton_index: i, gate, left_offset_m: +off.toFixed(6), left_yaw_deg: +yaw.toFixed(6), clearance_m: +c.toFixed(4)};
    if (c < .12) skipped.push(rec); else starts.push({id: starts.length, ...rec});
  }
  return {schema: 'jf-cap-test-starts/v1', rule: 'Halton(2,3,5) i = 1, 2, …: gate = floor(h2·33), offset = −0.15 + 0.3 h3 [m], yaw = −10 + 20 h5 [deg]。すき間 < 0.12 m は次の点', world_sha256: WORLD_V3_SHA, starts, skipped};
}

// ------------------------------------------------------------------ 条件
export const DRIVERS = ['PP-1.0', 'PP-1.2', 'PP-1.4', 'RL-fastest', 'RL-speed-0.75'];
export function condId(c) {
  return [c.driver, c.gov, c.physics, c.brake === 'nominal' ? null : 'brake-' + c.brake, c.world === 'W0' ? null : c.world, c.kBatt === 1 ? null : 'kb' + c.kBatt]
    .filter(Boolean).join('__');
}
const C = (driver, gov, o = {}) => ({driver, gov, physics: 'P_cal', brake: 'nominal', world: 'W0', kBatt: 1, ...o});
export function plan(name) {
  const out = [];
  if (name === 'primary' || name === 'all') for (const d of DRIVERS) for (const g of ['OFF', 'ON-O25']) out.push(C(d, g));
  if (name === 'secondary' || name === 'all') {
    for (const d of DRIVERS) out.push(C(d, 'ON-O275'));
    for (const b of ['low', 'high']) for (const d of ['PP-1.2', 'PP-1.4', 'RL-fastest']) out.push(C(d, 'ON-O25', {brake: b}));
    for (const d of DRIVERS) for (const g of ['OFF', 'ON-O25']) out.push(C(d, g, {physics: 'P_default'}));
    for (const d of ['PP-1.2', 'RL-fastest']) for (const g of ['OFF', 'ON-O25']) out.push(C(d, g, {world: 'W+'}));
    for (const d of ['PP-1.2', 'RL-fastest']) out.push(C(d, 'ON-G25'));
  }
  if (name === 'extra' || name === 'all') {
    for (const kb of [.8, 1.12]) for (const d of ['PP-1.2', 'RL-fastest']) for (const g of ['OFF', 'ON-O25']) out.push(C(d, g, {kBatt: kb}));
    // P_default の 2.75(較正で結論が変わるかを 2.75 でも見る)
    for (const d of ['PP-1.2', 'PP-1.4', 'RL-fastest']) out.push(C(d, 'ON-O275', {physics: 'P_default'}));
  }
  return out;
}
export function govParams(gov) {
  if (gov === 'ON-O25' || gov === 'ON-G25') return {...GOVERNOR_V1_SIM};
  if (gov === 'ON-O275') return {...GOVERNOR_V1_SIM, cap_R1_app: 2.75, cap_R1_turn: 2.75};
  return null;
}

// ------------------------------------------------------------------ 準備(コア・経路・方策)
const cache = {};
async function coreFor(physics, brake) {
  const key = physics + '/' + brake;
  if (!cache[key]) {
    if (physics === 'P_cal') {
      const {profile, sha256} = loadProfile('P_cal');
      cache[key] = {core: await loadCalibratedCore(profile, {fast: true, profileSha256: sha256, brakeScenario: brake}), compiled: compileProfile(profile, {brakeScenario: brake}), profileSha256: sha256};
    } else {
      if (brake !== 'nominal') throw new Error('cap_test: P_default のブレーキは名目だけです');
      cache[key] = {core: await loadCalibratedCore(origReplayProfile(0, 'P_default'), {fast: true}), compiled: null, profileSha256: null};
    }
  }
  return cache[key];
}
let ppPathCache = null;
export function ppPath() {
  if (!ppPathCache) ppPathCache = choosePath(loadWorld('W0'));
  return ppPathCache;
}
const policyCache = {};
function policyFor(key) {
  if (!policyCache[key]) {
    const p = `/home/jetson/jetformer/data/sim_rl/modes/${key}/policy.json`, text = fs.readFileSync(p);
    if (sha(text) !== POLICY[key]) throw new Error('cap_test: 方策の sha256 が違います: ' + key);
    policyCache[key] = JSON.parse(text);
  }
  return policyCache[key];
}
const loop = createLoop(readJson(REFERENCE_LOOP_JSON));

// ------------------------------------------------------------------ 1 走行
const SEC_ORDER = ['HS', 'LH', 'R1', 'GM', 'DE', 'SW'];
function sectionOfS(s) {
  const u = ((s % loop.L) + loop.L) % loop.L;
  if (u >= 5.906 && u < 8.216) return 'R1';   // 衝突の割り当て(protocol metrics.collision_attribution)
  for (const [k, [a, b]] of Object.entries(SECTIONS_R)) if (u >= a && u < b) return k.startsWith('HS') ? 'HS' : k;
  // 区間の間のすき間(約 0.02 m)は次の区間
  for (const k of SEC_ORDER) { const [a] = k === 'HS' ? SECTIONS_R.HS_b : SECTIONS_R[k]; if (u < a) return k; }
  return 'HS';
}
export async function runEpisode(c, start, {keepTrace = false} = {}) {
  const {core, compiled} = await coreFor(c.physics, c.brake);
  const base = loadWorld(c.world);
  const {world, variation} = activeWorldFor(base, start.gate, start.left_offset_m, start.left_yaw_deg);
  const config = {...PHYSICS_CONFIG, seed: EPISODE.seed, timeLimit: EPISODE.timeLimit, episodeMode: EPISODE.episodeMode};
  const sim = core.createSimulation(world, config);
  sim.reset({seed: config.seed, scenario: 'baseline'});
  let snap = sim.peek();
  if (snap.state.terminal) return {start: start.id, invalid_spawn: true, reason: snap.state.reason};
  if (c.kBatt !== 1) sim.calibControl({kBatt: c.kBatt});
  const gp = govParams(c.gov);
  const cap = gp ? createCapFilter({loop, route: world.route, governorStep, governorInitialState, params: gp, zoneMode: c.gov === 'ON-G25' ? 'gyro' : 'oracle', name: c.gov}) : null;
  if (cap) sim.calibControl({filter: cap.filter, filterName: cap.name});
  const cfgN = JSON.parse(JSON.stringify(snap.config));
  // 運転手
  let driver, isRl = false;
  if (c.driver.startsWith('PP-')) {
    const k = +c.driver.slice(3), pp = ppPath();
    const maps = physicsMaps(c.physics === 'P_cal' ? 'cal' : 'default', compiled);
    driver = createPPDriver(pp.path, speedProfile(pp.path, base, maps, k), maps, {k, wheelbase: config.wheelbase});
  } else {
    isRl = true;
    const mode = c.driver === 'RL-fastest' ? 'fastest' : 'speed', params = c.driver === 'RL-fastest' ? {} : {c_v: .75};
    driver = createModeTeacher(world, cfgN, variation, {mode, weights: policyFor(mode), params, policySha256: POLICY[mode]});
  }
  const tr = createProgressTracker(loop, world.route);
  const T = [], S = [], V = [], ZO = [];
  let trMaxD = 0;
  const queue = [];
  let appliedPrev = null;
  while (!snap.state.terminal) {
    const st = snap.state;
    const s = st.frame === 0 ? tr.reset(st.x, st.z, 0) : tr.advance(st.x, st.z);
    if (tr.d > trMaxD) trMaxD = tr.d;
    T.push(st.sim_time); S.push(s); V.push(st.speed); ZO.push(zoneOfS(s, loop.L));
    let ap;
    if (isRl) {
      const out = driver.action({state: st, metrics: snap.metrics, config: cfgN}, {applied: appliedPrev ?? [0, 0]});
      const lg = driver.log.at(-1);
      ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    } else {
      const out = driver.action(st);
      ap = {steering: out.steering, speed: out.speed, a: null};
    }
    // アプリの遅れ 1 フレーム(最初は中立)
    queue.push(ap);
    ap = queue.length > EPISODE.delayFrames ? queue.shift() : {steering: 0, speed: 0, a: [0, -1]};
    appliedPrev = ap.a;
    sim.step({steering: ap.steering, speed: ap.speed}, true);
    snap = sim.peek();
  }
  // 最後の状態も入れる
  const stF = snap.state, sF = tr.advance(stF.x, stF.z);
  T.push(stF.sim_time); S.push(sF); V.push(stF.speed); ZO.push(zoneOfS(sF, loop.L));
  const m = snap.metrics;
  const ce = m.collision_events?.[0] ?? null;
  let collision = null;
  if (ce) {
    const pr = loop.projectWindow(ce.x, ce.z, sF - 1.0, sF + 0.5, {s: 0, d: 0});
    // 速さは衝突のフレームの始まりの値(衝突のあと sim は止める)
    collision = {t: ce.time, frame: ce.frame, x: ce.x, z: ce.z, s: pr.s, s_mod: ((pr.s % loop.L) + loop.L) % loop.L, section: sectionOfS(pr.s), collider: ce.collider, kind: ce.kind,
      v: V.length >= 2 ? V[V.length - 2] : stF.speed, zone: zoneOfS(pr.s, loop.L)};
  }
  const res = {start: start.id, gate: start.gate, reason: stF.reason, frames: stF.frame, sim_time: stF.sim_time, distance_m: m.distance,
    laps_driven: m.distance / loop.L, completed_laps: m.completed_laps ?? 0, collisions: m.collisions, collision,
    complete: stF.reason === 'collection_duration_complete' && m.collisions === 0,
    lap_events: (m.lap_events ?? []).map(l => ({lap: l.lap_index, elapsed: l.elapsed, start_time: l.start_time, collisions: l.collisions, gates: l.passed_gates})),
    tracker: {jumps: tr.jumps, max_dist_m: +trMaxD.toFixed(3), s_progress_m: +(sF - S[0]).toFixed(3)},
    passes: passEvents(T, S, V, collision), max_v: Math.max(...V), reverse_frames: V.filter(v => v < -.05).length};
  if (cap) Object.assign(res, governorSummary(cap.log, ZO, config.dt));
  if (keepTrace) {
    const tr2 = sim.report().trace;
    res.trace = {t: T.map(x => +x.toFixed(4)), s: S.map(x => +x.toFixed(4)), v: V.map(x => +x.toFixed(4)), x: tr2.map(q => +q.x.toFixed(4)), z: tr2.map(q => +q.z.toFixed(4)),
      gov: cap ? cap.log.map(q => ({m: q.mode, z: q.zone, c: q.cap, u: +q.u.toFixed(3), u2: +q.u2.toFixed(3)})) : null};
  }
  return res;
}

// 区間の通過(R1 の進入の速さ = R1 の始まり s = 6.206 の手前 0.4 秒の最大。実車の entry_erpm と同じ定義)
function crossTime(T, S, target) {
  for (let i = 1; i < S.length; i++) if (S[i - 1] < target && S[i] >= target) { const f = (target - S[i - 1]) / (S[i] - S[i - 1]); return {i, t: T[i - 1] + f * (T[i] - T[i - 1])}; }
  return null;
}
function maxVIn(T, V, t0, t1) { let m = -Infinity; for (let i = 0; i < T.length; i++) if (T[i] >= t0 - 1e-9 && T[i] <= t1 + 1e-9) m = Math.max(m, V[i]); return m === -Infinity ? null : m; }
function vAt(T, V, t) { for (let i = 1; i < T.length; i++) if (T[i] >= t) { const f = (t - T[i - 1]) / (T[i] - T[i - 1] || 1); return V[i - 1] + f * (V[i] - V[i - 1]); } return V[V.length - 1]; }
export function passEvents(T, S, V, collision) {
  const L = loop.L, sMin = S[0], sMax = S[S.length - 1];
  const out = {R1: [], GM: [], DE: []};
  const m0 = Math.floor((sMin - 8.216) / L), m1 = Math.ceil(sMax / L);
  for (let m = m0; m <= m1; m++) {
    const o = m * L;
    // R1: 帰属の区間 [5.906, 8.216] に手前から入った通過(R1 の始まりを越えた、またはその手前 0.3 m〜R1 で衝突)。区間の中からの開始は数えない
    const cR = crossTime(T, S, o + 6.206), colR = collision && collision.section === 'R1' && collision.s >= o + 5.906 - 1e-9 && collision.s < o + 8.216;
    if (S[0] < o + 5.906 && (cR || colR)) {
      const tRef = cR ? cR.t : collision.t;
      const done = crossTime(T, S, o + 8.216);
      out.R1.push({lap_offset: m, t: +tRef.toFixed(4), entry_v: +maxVIn(T, V, tRef - .4, tRef).toFixed(4), v_at_start: cR ? +vAt(T, V, cR.t).toFixed(4) : null,
        reached_start: !!cR, collision: !!colR, completed: !!done && !colR});
    }
    const cG = crossTime(T, S, o + 8.237), colG = collision && collision.section === 'GM' && collision.s >= o + 8.237 && collision.s < o + 12.601;
    if (S[0] < o + 8.237 && (cG || colG)) {
      out.GM.push({lap_offset: m, t: cG ? +cG.t.toFixed(4) : null, start_v: cG ? +vAt(T, V, cG.t).toFixed(4) : null, collision: !!colG, completed: !!crossTime(T, S, o + 12.601) && !colG});
    }
    const cD = crossTime(T, S, o + 12.626), colD = collision && collision.section === 'DE' && collision.s >= o + 12.626 && collision.s < o + 16.149;
    if (S[0] < o + 12.626 && (cD || colD)) {
      let mx = -Infinity; for (let i = 0; i < S.length; i++) if (S[i] >= o + 12.626 && S[i] < o + 16.149) mx = Math.max(mx, V[i]);
      out.DE.push({lap_offset: m, max_v: Number.isFinite(mx) ? +mx.toFixed(4) : null, collision: !!colD, completed: !!crossTime(T, S, o + 16.149) && !colD});
    }
  }
  return out;
}

// 上限の層の記録 → 作動・不変条件
export function governorSummary(log, zoneOracle, dt) {
  const zones = {};
  let brakeFrames = 0, brakeBad = 0, brakeBadOracle = 0, brakeStartLow = 0, uUp = 0, roundTrip02 = 0, roundTrip05 = 0, brakeStarts = 0;
  let lastBrakeEnd = null, prevBrake = false;
  for (let i = 0; i < log.length; i++) {
    const q = log[i], z = q.zone, zo = zoneOracle[i];
    const Z = zones[z] ??= {frames: 0, active: 0, cap: 0, neutral: 0, brake: 0, cut_mps_s: 0};
    Z.frames++;
    if (q.mode !== 'pass') Z.active++;
    if (q.mode === 'cap') Z.cap++;
    if (q.mode === 'neutral') Z.neutral++;
    if (q.mode === 'brake') { Z.brake++; brakeFrames++; }
    Z.cut_mps_s += (q.u - q.u2) * dt;
    if (q.mode === 'brake' && (z === 'R1_turn' || z === 'GM' || z === 'DE')) brakeBad++;
    if (q.mode === 'brake' && (zo === 'R1_turn' || zo === 'GM' || zo === 'DE')) brakeBadOracle++;
    if (q.bs) { brakeStarts++; if (q.v < 1.0) brakeStartLow++; if (lastBrakeEnd !== null && q.t - lastBrakeEnd < .2 - 1e-9) roundTrip02++; if (lastBrakeEnd !== null && q.t - lastBrakeEnd < .5 - 1e-9) roundTrip05++; }
    if (q.u2 > q.u + 1e-12) uUp++;
    const isB = q.mode === 'brake';
    if (prevBrake && !isB) lastBrakeEnd = q.t;
    prevBrake = isB;
  }
  for (const Z of Object.values(zones)) Z.cut_mps_s = +Z.cut_mps_s.toFixed(4);
  return {governor: {zones, frames: log.length, brake_frames: brakeFrames, brake_starts: brakeStarts},
    invariants: {brake_in_R1turn_GM_DE: brakeBad, brake_in_R1turn_GM_DE_by_position: brakeBadOracle, brake_start_below_1mps: brakeStartLow,
      brake_neutral_roundtrip_lt_0_2s: roundTrip02, brake_neutral_roundtrip_lt_0_5s: roundTrip05, u_increased_frames: uUp}};
}

// ------------------------------------------------------------------ 条件を走らせる(既にあれば飛ばす)
export async function runCondition(c, starts, {force = false, traceStarts = [0, 1]} = {}) {
  const id = condId(c), p = path.join(OUT, 'runs', id + '.json');
  if (!force && fs.existsSync(p)) return {id, skipped: true};
  const t0 = Date.now(), runs = [];
  for (const st of starts) runs.push(await runEpisode(c, st, {keepTrace: traceStarts.includes(st.id)}));
  const res = {schema: 'jf-cap-test-run/v1', id, condition: c, gov_params: govParams(c.gov), runtime: {cap: CAP_RUNTIME_VERSION, pp: PP_VERSION},
    episode: EPISODE, physics_config: PHYSICS_CONFIG, created_at: new Date().toISOString(), wall_s: (Date.now() - t0) / 1000, runs};
  const tmp = p + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(res));
  fs.renameSync(tmp, p);
  return {id, wall_s: res.wall_s};
}

async function main() {
  const args = process.argv.slice(2), get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  fs.mkdirSync(path.join(OUT, 'runs'), {recursive: true});
  const sp = path.join(OUT, 'starts.json');
  const made = makeStarts(loadWorld('W0'));
  if (fs.existsSync(sp)) {
    const old = readJson(sp);
    if (JSON.stringify(old.starts) !== JSON.stringify(made.starts)) throw new Error('cap_test: starts.json が作り直したものと違います');
  } else fs.writeFileSync(sp, JSON.stringify(made, null, 1));
  const pathInfo = path.join(OUT, 'pp_path.json');
  if (!fs.existsSync(pathInfo)) {
    const pp = ppPath();
    fs.writeFileSync(pathInfo, JSON.stringify({use: pp.use, sigma_m: pp.sigma_m, checks: pp.checks, n: pp.path.n, length_m: pp.path.L,
      x: Array.from(pp.path.x, v => +v.toFixed(4)), z: Array.from(pp.path.z, v => +v.toFixed(4))}, null, 0));
  }
  let list = plan(get('--plan', 'primary'));
  const only = get('--only', null);
  if (only) list = list.filter(c => only.split(',').includes(condId(c)));
  const shard = get('--shard', '0/1').split('/').map(Number);
  list = list.filter((_, i) => i % shard[1] === shard[0]);
  const nStarts = +get('--starts', '24');
  for (const c of list) {
    const r = await runCondition(c, made.starts.slice(0, nStarts), {force: args.includes('--force')});
    console.log(new Date().toISOString().slice(11, 19), r.id, r.skipped ? '(既にある)' : `${r.wall_s.toFixed(1)} 秒`);
  }
}
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) await main();
