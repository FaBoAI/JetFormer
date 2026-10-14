// no_wall r2(学習の段の作り直し)用に確認 r1 の run_nw.mjs を写して、runtime・方策・シールドの値を環境変数で替えられるようにしたもの(2026-10-14)。
//   NW2_RT = runtime のフォルダー(既定 web/rl)・NW2_POLICY = no_wall の policy.json(既定 bundle/no_wall)・NW2_SHIELD = shieldParams の JSON
// 元: no_wall の確認 r1: 1 回を走らせる(2026-10-14、確認の段)。学習・評価のコード(episode.mjs・eval_modes.mjs・env_core.mjs・lap_audit.mjs)を使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs(ブラウザと同じ scene.js から作る)を app.js の reset と同じ式で開始ゲートへ回す
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル)。摩擦円の回だけは元のコアに無いので、実装の写し(fast_core_modes.mjs の fastfc)で動かし、
//         自前の積分(摩擦円つき)で軌跡が合うことを確かめる
//   操作: 書き出したブラウザの組 web/rl/mode_runtime.mjs と bundle/<mode>/policy.json(ブラウザで走るものと同じ)
//   すき間・ρ・ラップ: 自前(own_geom.mjs)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {auditEpisode, buildObstacles, ownClearance, surfaceMuAt} from './own_geom.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export const SIM_CORE_SHA = sha(`${REL}/dist/sim-core.js`);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(`${REL}/dist/sim-core.js`);
let fcCore = null;
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
export const RT_DIR = process.env.NW2_RT || `${EXP}/web/rl`;
const RT = await import(`${RT_DIR}/mode_runtime.mjs`);
const SHIELD_PARAMS = process.env.NW2_SHIELD ? JSON.parse(process.env.NW2_SHIELD) : {};
const G = 9.80665;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const policies = {};
export function policyFor(mode) {
  if (!policies[mode]) { const p = mode === 'no_wall' && process.env.NW2_POLICY ? process.env.NW2_POLICY : `${EXP}/bundle/${mode}/policy.json`; policies[mode] = {weights: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p)}; }
  return policies[mode];
}
// 物理(ブラウザの記録と同じ標準の値。RC_COLLECTION_PLAN[0]・modes.json common.physics_nominal)
export const NOMINAL = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
// app.js の reset と同じ(dist/app.js 111〜114 行)
export function activeWorld(world, gate, offset, yawDeg) {
  const startIndex = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(startIndex), ...world.route.slice(0, startIndex)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...world, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}},
    variation: {gate: startIndex, left_offset_m: off, left_yaw_deg: turn * 180 / Math.PI}};
}
export function startClearance(gate, offset, yawDeg) {
  const {world} = activeWorld(BASE_WORLD, gate, offset, yawDeg);
  return ownClearance(buildObstacles(world), world.spawn.x, world.spawn.z, world.spawn.yaw);
}

// spec: {mode, start: {gate, left_offset_m, left_yaw_deg, seed}, episodeMode: 'continuous'|'single', dr: {physics, delay, steerGain, speedGain, frictionCircle},
//        shield: true|false, keepPath: bool}
export async function runOne(spec) {
  const mode = spec.mode ?? 'no_wall', P = policyFor(mode);
  const st0 = spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const episodeMode = spec.episodeMode ?? 'continuous';
  const base = {...NOMINAL, seed, timeLimit: 20};
  if (episodeMode === 'continuous') base.episodeMode = 'continuous-laps-v1';
  const dr = spec.dr ?? {};
  const fc = !!dr.frictionCircle;
  const simCfg = {...base, ...(dr.physics ?? {}), ...(fc ? {frictionCircle: true} : {})};
  if (fc && !fcCore) fcCore = await import(`${EXP}/rl/fast_core_modes.mjs`);
  const sim = (fc ? fcCore : core).createSimulation(world, simCfg);
  let snap = sim.reset({seed, scenario: 'baseline'});
  if (fc) snap = JSON.parse(JSON.stringify(sim.peek()));
  if (snap.state.terminal) return {invalid_spawn: true};
  // 教師に渡す config: 同じコース・同じ種の「標準の物理」の sim の snapshot().config(ブラウザの記録と同じ)
  const nomSim = core.createSimulation(world, base); const cfgN = nomSim.reset({seed, scenario: 'baseline'}).config;
  const teacher = RT.createModeTeacher(world, cfgN, variation, {mode, weights: P.weights, policySha256: P.sha, shieldParams: {...SHIELD_PARAMS, ...(spec.shieldParams ?? {})}, overrides: spec.shield === false ? {shield: false} : {}});
  const delay = dr.delay ?? 0, sg = dr.steerGain ?? 1, vg = dr.speedGain ?? 1;
  const q = [];
  let appliedPrev = [0, 0];
  const labels = {l0: 0, sh: 0, ph: {}, bad_keys: 0, frames: 0, fb: {}, pess_fail_kept: 0};
  let firstShieldFrame = null;
  const path = spec.keepPath ? [] : null;
  while (!snap.state.terminal) {
    const st = snap.state;
    const out = teacher.action({state: st, metrics: snap.metrics, config: cfgN}, delay ? {applied: appliedPrev} : null);
    const jf = out.info.jf;
    labels.frames++; if (jf.l !== 1) labels.l0++; if (jf.sh === 1) { labels.sh++; if (firstShieldFrame === null) firstShieldFrame = st.frame; } labels.ph[jf.ph] = (labels.ph[jf.ph] ?? 0) + 1;
    if (JSON.stringify(out.info).match(/steering_pwm|throttle_pwm|"\//)) labels.bad_keys++;
    const lg = teacher.log.at(-1);
    if (lg.shield && lg.shield.fb) { labels.fb[lg.shield.fb] = (labels.fb[lg.shield.fb] ?? 0) + 1; if (lg.shield.fb === 'proposal_nominal') labels.pess_fail_kept++; }
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { q.push(ap); ap = q.length > delay ? q.shift() : {steering: 0, speed: 0, a: [0, -1]}; appliedPrev = ap.a; }
    if (path) path.push([st.x, st.z, st.speed, lg.s, lg.ey, out.steering, out.speed, jf.sh ?? 0, st.steering, lg.shield?.fb ?? null]);
    if (fc) { sim.step({steering: ap.steering * sg, speed: ap.speed * vg}, true); snap = sim.peek(); }
    else snap = sim.step({steering: ap.steering * sg, speed: ap.speed * vg});
  }
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  // 自前のすき間(元のコアの回は sim の config、摩擦円の回は自前の積分も摩擦円で)
  const au = auditEpisode(world, rep.config, world.spawn, rep.actions, rep.trace, {frictionCircle: fc, skipLastErr: fs_.reason !== 'collection_duration_complete'});
  // ρ(N6 の定義): ステップ後の v・δ、μ = 車体の中心の路面 × frictionMultiplier。ρ_comb は v > 0.3 のフレーム
  const sf = world.surfaces, fm = rep.config.frictionMultiplier, wb = rep.config.wheelbase, dt = rep.config.dt;
  let nReq = 0, req1 = 0, nComb = 0, comb1 = 0, reqMax = 0, combMax = 0;
  let prev = {speed: 0, yaw: world.spawn.yaw};
  for (const tq of rep.trace) {
    const mu = surfaceMuAt(sf, tq.x, tq.z, fm);
    if (tq.speed > .3) {
      const rr = tq.speed * tq.speed * Math.abs(Math.tan(tq.steering)) / (wb * mu * G);
      nReq++; if (rr > 1) req1++; reqMax = Math.max(reqMax, rr);
      const al = (tq.speed - prev.speed) / dt, alat = tq.speed * wrap(tq.yaw - prev.yaw) / dt, rc = Math.hypot(al, alat) / (mu * G);
      nComb++; if (rc > 1) comb1++; combMax = Math.max(combMax, rc);
    }
    prev = tq;
  }
  // ラップ(sim の lap_events)と、自前の黒シートの横断の確かめ(各周で z = 0 を −z 向きに、車幅ぶん x 1.26〜2.36 の中で 1 回以上)
  const laps = (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions,
    distance: l.distance, start_frame: l.start_frame, end_frame: l.end_frame}));
  const ownCross = [];
  for (let i = 1; i < rep.trace.length; i++) {
    const a = rep.trace[i - 1], b = rep.trace[i];
    if (a.z > 0 && b.z <= 0) {
      const u = a.z / (a.z - b.z), x = a.x + (b.x - a.x) * u, y = a.yaw + wrap(b.yaw - a.yaw) * u;
      const half = .21 * Math.abs(Math.cos(y)) + .095 * Math.abs(Math.sin(y));
      if (x > 0) ownCross.push({frame: i, x, inside: x - half >= 1.26 - 1e-9 && x + half <= 2.36 + 1e-9});
    }
  }
  const res = {mode, start: {...st0}, variation, episode_mode: episodeMode, dr: spec.dr ?? null, shield: spec.shield !== false,
    reason: fs_.reason, frames: fs_.frame, collisions: m.collisions, collision: m.collision_events?.[0] ?? null,
    laps, completed_laps: m.completed_laps ?? null, passed_gates: m.passed_gates,
    lap_from_stop: episodeMode === 'single' && fs_.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1 ? m.elapsed : null,
    min_clearance: au.min_clearance, min_frame: au.min_frame, min_kind: au.min_kind, audit_pos_err: au.max_pos_err, overlap_frames: au.overlap_frames,
    rho: {n: nReq, req_gt1: req1, comb_n: nComb, comb_gt1: comb1, req_max: reqMax, comb_max: combMax},
    shield_frames: teacher.shieldFrames, first_shield_frame: firstShieldFrame, labels, policy_sha256: P.sha, rt_dir: RT_DIR, shield_params: teacher.metadata.shield ?? null, own_crossings: ownCross,
    mean_running_speed: (() => { const v = rep.trace.filter(t => t.speed > .3).map(t => t.speed); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0; })(),
    distance: m.distance};
  res.valid_episode = episodeMode === 'continuous' && fs_.reason === 'collection_duration_complete' && fs_.frame === 600 && m.collisions === 0;
  res.valid_laps = laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  if (path) res.path = path;
  return res;
}
