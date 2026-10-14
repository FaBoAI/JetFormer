// no_wall の確認 r2: 1 回を走らせる(2026-10-14、確認の段 r2)。確認 r1 の run_nw.mjs(sha256 f17e3589…)を元に、この段で足したもの:
//   - 長い連続周回(timeLimit を 20 秒より長く。sim-core は 3,600 秒まで)
//   - runtime のフォルダー(既定 web/rl = 書き出したブラウザの組)と方策のファイル(既定 bundle/no_wall/policy.json)を spec で替えられる
//   - シールドの記録: 見積もりが確かになったフレーム・速さの見張りが働いたフレーム・fallback の種類・見積もりの値(最後)
//   - ρ を「指令の舵(適用した指令 × ゲイン)」でも数える(N6 の参考)
// 学習・評価のコード(episode.mjs・eval_modes.mjs・env_core.mjs・env_modes.mjs・lap_audit.mjs・nw2/)は使わない。
// コース = ユーザーのリリースの qa/load-course-world.mjs、物理 = ユーザーのリリースの dist/sim-core.js(摩擦円の回だけ fast_core_modes.mjs の fastfc)、
// すき間・自前の積分 = ../check_nw/own_geom.mjs(確認 r1 の自前の形、sha256 667f044a…)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {auditEpisode, buildObstacles, ownClearance, surfaceMuAt} from '../check_nw/own_geom.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export const SIM_CORE_SHA = sha(`${REL}/dist/sim-core.js`);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(`${REL}/dist/sim-core.js`);
let fcCore = null;
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
const RTS = {};
async function runtimeFrom(dir) { if (!RTS[dir]) RTS[dir] = await import(`${dir}/mode_runtime.mjs`); return RTS[dir]; }
const G = 9.80665;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const policies = {};
export function policyFile(mode, file = null) {
  const p = file ?? `${EXP}/bundle/${mode}/policy.json`;
  if (!policies[p]) policies[p] = {weights: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p), path: p};
  return policies[p];
}
export const NOMINAL = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
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

// spec: {mode, start: {gate, left_offset_m, left_yaw_deg, seed}, episodeMode, dr: {physics, delay, steerGain, speedGain, frictionCircle},
//        shield, keepPath, timeLimit, rtDir, policyPath, scenario}
export async function runOne(spec) {
  const mode = spec.mode ?? 'no_wall', P = policyFile(mode, spec.policyPath ?? null);
  const RT = await runtimeFrom(spec.rtDir ?? `${EXP}/web/rl`);
  const st0 = spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const episodeMode = spec.episodeMode ?? 'continuous';
  const T = spec.timeLimit ?? 20;
  const base = {...NOMINAL, seed, timeLimit: T};
  if (episodeMode === 'continuous') base.episodeMode = 'continuous-laps-v1';
  const dr = spec.dr ?? {};
  const fc = !!dr.frictionCircle;
  const simCfg = {...base, ...(dr.physics ?? {}), ...(fc ? {frictionCircle: true} : {})};
  if (fc && !fcCore) fcCore = await import(`${EXP}/rl/fast_core_modes.mjs`);
  const sim = (fc ? fcCore : core).createSimulation(world, simCfg);
  let snap = sim.reset({seed, scenario: spec.scenario ?? 'baseline'});
  if (fc) snap = JSON.parse(JSON.stringify(sim.peek()));
  if (snap.state.terminal) return {invalid_spawn: true};
  // 教師に渡す config: 標準の物理の sim の snapshot().config(ブラウザの記録と同じ。今の物理を教師に教えない)
  const nomSim = core.createSimulation(world, base); const cfgN = nomSim.reset({seed, scenario: 'baseline'}).config;
  const teacher = RT.createModeTeacher(world, cfgN, variation, {mode, weights: P.weights, policySha256: P.sha, overrides: spec.shield === false ? {shield: false} : {}});
  const delay = dr.delay ?? 0, sg = dr.steerGain ?? 1, vg = dr.speedGain ?? 1;
  const q = [];
  let appliedPrev = [0, 0];
  const labels = {l0: 0, sh: 0, ph: {}, bad_keys: 0, frames: 0, fb: {}, pess_fail_kept: 0, governor: 0, known_frame: null, broken_frames: 0,
    max_run_sh: 0, first_sh: null, ident_last: null, cap_min: null, sh_lt5s: 0};
  let runSh = 0;
  const path = spec.keepPath ? [] : null;
  const cmdApplied = [];
  while (!snap.state.terminal) {
    const st = snap.state;
    const out = teacher.action({state: st, metrics: snap.metrics, config: cfgN}, delay ? {applied: appliedPrev} : null);
    const jf = out.info.jf;
    labels.frames++; if (jf.l !== 1) labels.l0++;
    if (jf.sh === 1) { labels.sh++; runSh++; labels.max_run_sh = Math.max(labels.max_run_sh, runSh); if (labels.first_sh === null) labels.first_sh = st.frame; if (st.frame < 150) labels.sh_lt5s++; } else runSh = 0;
    labels.ph[jf.ph] = (labels.ph[jf.ph] ?? 0) + 1;
    if (JSON.stringify(out.info).match(/steering_pwm|throttle_pwm|"\//)) labels.bad_keys++;
    const lg = teacher.log.at(-1);
    if (lg.shield) {
      const fb = lg.shield.fb;
      if (fb) { labels.fb[fb] = (labels.fb[fb] ?? 0) + 1; if (fb === 'proposal_nominal') labels.pess_fail_kept++; if (fb.includes('governor')) labels.governor++; }
      const id = lg.shield.id;
      if (id) { if (id.known && labels.known_frame === null) labels.known_frame = st.frame; if (id.broken) labels.broken_frames++; labels.ident_last = id; }
    }
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { q.push(ap); ap = q.length > delay ? q.shift() : {steering: 0, speed: 0, a: [0, -1]}; appliedPrev = ap.a; }
    cmdApplied.push(Math.max(-.52, Math.min(.52, ap.steering * sg)));
    if (path) path.push([st.x, st.z, st.speed, lg.s, lg.ey, out.steering, out.speed, jf.sh ?? 0]);
    if (fc) { sim.step({steering: ap.steering * sg, speed: ap.speed * vg}, true); snap = sim.peek(); }
    else snap = sim.step({steering: ap.steering * sg, speed: ap.speed * vg});
  }
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  const au = auditEpisode(world, rep.config, world.spawn, rep.actions, rep.trace, {frictionCircle: fc, skipLastErr: fs_.reason !== 'collection_duration_complete'});
  const sf = world.surfaces, fm = rep.config.frictionMultiplier, wb = rep.config.wheelbase, dt = rep.config.dt;
  let nReq = 0, req1 = 0, nComb = 0, comb1 = 0, reqMax = 0, combMax = 0, cmd1 = 0;
  let prev = {speed: 0, yaw: world.spawn.yaw};
  rep.trace.forEach((tq, i) => {
    const mu = surfaceMuAt(sf, tq.x, tq.z, fm);
    if (tq.speed > .3) {
      const rr = tq.speed * tq.speed * Math.abs(Math.tan(tq.steering)) / (wb * mu * G);
      nReq++; if (rr > 1) req1++; reqMax = Math.max(reqMax, rr);
      const rq = tq.speed * tq.speed * Math.abs(Math.tan(cmdApplied[i] ?? 0)) / (wb * mu * G); if (rq > 1) cmd1++;
      const al = (tq.speed - prev.speed) / dt, alat = tq.speed * wrap(tq.yaw - prev.yaw) / dt, rc = Math.hypot(al, alat) / (mu * G);
      nComb++; if (rc > 1) comb1++; combMax = Math.max(combMax, rc);
    }
    prev = tq;
  });
  const laps = (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions,
    distance: l.distance, start_frame: l.start_frame, end_frame: l.end_frame}));
  const ownCross = [];
  for (let i = 1; i < rep.trace.length; i++) {
    const a = rep.trace[i - 1], b = rep.trace[i];
    if (a.z > 0 && b.z <= 0) {
      const u = a.z / (a.z - b.z), x = a.x + (b.x - a.x) * u, y = a.yaw + wrap(b.yaw - a.yaw) * u;
      const half = .21 * Math.abs(Math.cos(y)) + .095 * Math.abs(Math.sin(y));
      if (x > 0) ownCross.push({frame: i, x: +x.toFixed(4), inside: x - half >= 1.26 - 1e-9 && x + half <= 2.36 + 1e-9});
    }
  }
  const nFrames = Math.round(T * 30);
  const res = {mode, start: {...st0}, variation, episode_mode: episodeMode, time_limit: T, dr: spec.dr ?? null, shield: spec.shield !== false, policy_sha: P.sha.slice(0, 12),
    reason: fs_.reason, frames: fs_.frame, collisions: m.collisions, collision: m.collision_events?.[0] ?? null,
    laps, completed_laps: m.completed_laps ?? null, passed_gates: m.passed_gates,
    lap_from_stop: episodeMode === 'single' && fs_.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1 ? m.elapsed : null,
    min_clearance: au.min_clearance, min_frame: au.min_frame, min_kind: au.min_kind, audit_pos_err: au.max_pos_err, overlap_frames: au.overlap_frames,
    rho: {n: nReq, req_gt1: req1, comb_n: nComb, comb_gt1: comb1, req_max: reqMax, comb_max: combMax, cmd_gt1: cmd1},
    shield_frames: teacher.shieldFrames, labels, own_crossings: ownCross,
    mean_running_speed: (() => { const v = rep.trace.filter(t => t.speed > .3).map(t => t.speed); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0; })(),
    distance: m.distance};
  res.valid_episode = episodeMode === 'continuous' && fs_.reason === 'collection_duration_complete' && fs_.frame === nFrames && m.collisions === 0;
  res.valid_laps = laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  if (path) res.path = path;
  if (spec.keepFrames) res.per_frame_clearance = au.per_frame.map(c => +c.toFixed(5));
  return res;
}
