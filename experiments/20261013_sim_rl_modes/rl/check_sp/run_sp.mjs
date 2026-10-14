// 速さいろいろ(speed)の確認 r1: 1 回を走らせる(2026-10-14、確認の段)。学習・評価のコード(episode.mjs・eval_modes.mjs・env_core.mjs・
// env_modes.mjs・lap_audit.mjs・variation_metrics.mjs)を使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs(ブラウザと同じ scene.js から作る)を app.js の reset と同じ式で開始ゲートへ回す
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル、sha256 aaeb2379…549e)。物理のずれの回は遅延・ゲイン・物理の値だけ(摩擦円は使わない)
//   操作: ブラウザの組 web/rl/mode_runtime.mjs + 書き出した方策(speed = train/speed/export/policy.json、最速 = bundle/fastest/policy.json)
//   すき間: own_geom.mjs(no_wall の確認の段の自前の多角形の距離。学習の段のものではない)+ own_extra.mjs の点の確かめ
//   s・e_y・マス: own_extra.mjs の自前の R への射影
import fs from 'node:fs';
import crypto from 'node:crypto';
import {auditEpisode, buildObstacles, ownClearance, ownStepFrame, nearObstacles, surfaceMuAt} from './own_geom.mjs';
import {sampleClearance, createProjector, binFirstPass, LOOP_L_OWN} from './own_extra.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export const SIM_CORE_SHA = sha(`${REL}/dist/sim-core.js`);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(`${REL}/dist/sim-core.js`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
const RT = await import(`${EXP}/web/rl/mode_runtime.mjs`);
const G = 9.80665;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const POLICY_PATH = {speed: `${REP}/train/speed/export/policy.json`, fastest: `${EXP}/bundle/fastest/policy.json`};
const policies = {};
export function policyFor(kind) {
  if (!policies[kind]) { const p = POLICY_PATH[kind]; policies[kind] = {weights: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p), path: p}; }
  return policies[kind];
}
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
const OB = buildObstacles(BASE_WORLD);
export function startClearance(gate, offset, yawDeg) {
  const {world} = activeWorld(BASE_WORLD, gate, offset, yawDeg);
  return ownClearance(OB, world.spawn.x, world.spawn.z, world.spawn.yaw);
}
const quant = (arr, p) => { const b = [...arr].sort((x, y) => x - y); if (!b.length) return null; const i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
const steerUs = a0 => 1442 + a0 * (a0 >= 0 ? 558 : 442);

// spec: {policy: 'speed'|'fastest', start: {gate, left_offset_m, left_yaw_deg, seed}, params: {c_v} | {speed_limit}, episodeMode: 'continuous'|'single',
//        dr: {physics, delay, steerGain, speedGain}, keepPath}
export async function runOne(spec) {
  const kind = spec.policy ?? 'speed', P = policyFor(kind), mode = kind;
  const st0 = spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const episodeMode = spec.episodeMode ?? 'continuous';
  const base = {...NOMINAL, seed, timeLimit: 20};
  if (episodeMode === 'continuous') base.episodeMode = 'continuous-laps-v1';
  const dr = spec.dr ?? {};
  const simCfg = {...base, ...(dr.physics ?? {})};
  const sim = core.createSimulation(world, simCfg);
  let snap = sim.reset({seed, scenario: 'baseline'});
  if (snap.state.terminal) return {invalid_spawn: true, start: st0};
  const nomSim = core.createSimulation(world, base); const cfgN = nomSim.reset({seed, scenario: 'baseline'}).config;
  const teacher = RT.createModeTeacher(world, cfgN, variation, {mode, weights: P.weights, policySha256: P.sha, params: spec.params ?? {}});
  const capPlanned = mode === 'speed' ? 4.5 * (spec.params?.c_v ?? NaN) : (spec.params?.speed_limit ?? 4.5);
  const delay = dr.delay ?? 0, sg = dr.steerGain ?? 1, vg = dr.speedGain ?? 1;
  const q = [];
  let appliedPrev = [0, 0];
  const labels = {frames: 0, l0: 0, ph: {}, sh_key: 0, bad_keys: 0, p0: null, es_vals: new Set(), cmd_speed_max: 0, cmd_over_cap: 0, cmd_steer_absmax: 0};
  const pre = [];   // 操作の前の状態(フレーム k の観測の時点)
  while (!snap.state.terminal) {
    const st = snap.state;
    const out = teacher.action({state: st, metrics: snap.metrics, config: cfgN}, delay ? {applied: appliedPrev} : null);
    const jf = out.info.jf;
    labels.frames++; if (jf.l !== 1) labels.l0++; labels.ph[jf.ph] = (labels.ph[jf.ph] ?? 0) + 1; if ('sh' in jf) labels.sh_key++;
    if (jf.es !== undefined) labels.es_vals.add(jf.es);
    if (st.frame === 0) labels.p0 = jf.p ?? null;
    if (JSON.stringify(out.info).match(/steering_pwm|throttle_pwm|"\//)) labels.bad_keys++;
    labels.cmd_speed_max = Math.max(labels.cmd_speed_max, out.speed);
    if (out.speed > capPlanned + 1e-9) labels.cmd_over_cap++;
    labels.cmd_steer_absmax = Math.max(labels.cmd_steer_absmax, Math.abs(out.steering));
    const lg = teacher.log.at(-1);
    pre.push({f: st.frame, x: st.x, z: st.z, yaw: st.yaw, v: st.speed, st_cmd: out.steering, sp_cmd: out.speed, rt_s: lg.s, rt_ey: lg.ey});
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { q.push(ap); ap = q.length > delay ? q.shift() : {steering: 0, speed: 0, a: [0, -1]}; appliedPrev = ap.a; }
    snap = sim.step({steering: ap.steering * sg, speed: ap.speed * vg});
  }
  labels.es_vals = [...labels.es_vals];
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  // 自前のすき間(多角形の距離。小区間ごと + 12 cm 未満は間を 4 分割)
  const au = auditEpisode(world, rep.config, world.spawn, rep.actions, rep.trace, {skipLastErr: fs_.reason !== 'collection_duration_complete'});
  // 点の確かめ: 自前の積分を回し直し、正確な値が 6 cm 未満の小区間の姿勢を点でも測る
  const chk = {n: 0, max_exact_minus_sample: -Infinity, min_sample: Infinity, max_sample_minus_exact: -Infinity};
  {
    const stp = {x: world.spawn.x, z: world.spawn.z, yaw: world.spawn.yaw, speed: 0, steering: 0};
    const c0 = ownClearance(OB, stp.x, stp.z, stp.yaw), s0 = sampleClearance(OB, stp.x, stp.z, stp.yaw);
    chk.start_exact = c0; chk.start_sample = s0;
    let t = 0;
    for (let f = 0; f < rep.actions.length; f++) {
      const near = nearObstacles(OB, stp.x, stp.z, 1.0);
      const poses = ownStepFrame(stp, rep.actions[f], rep.config, OB.surfaces, {timeLeft: rep.config.timeLimit - t});
      t += rep.config.dt;
      for (const p of poses) {
        const e = ownClearance(near, p.x, p.z, p.yaw);
        if (e < .06) {
          const s = sampleClearance(near, p.x, p.z, p.yaw);
          chk.n++; chk.min_sample = Math.min(chk.min_sample, s);
          chk.max_exact_minus_sample = Math.max(chk.max_exact_minus_sample, e - s);
          chk.max_sample_minus_exact = Math.max(chk.max_sample_minus_exact, s - e);
        }
      }
    }
  }
  // 自前の R への射影(操作の前の状態)
  const pj = createProjector();
  const frames = pre.map((q, k) => {
    const r = k === 0 ? pj.reset(q.x, q.z, q.yaw) : pj.advance(q.x, q.z);
    return {f: q.f, s: r.s, ey: r.ey, v: q.v, st: steerUs(q.st_cmd / .52), rt_s: q.rt_s, rt_ey: q.rt_ey};
  });
  // 自前の s とランタイムの s の差(参考: 同じ R なら小さい)
  let sDiff = 0, eyDiff = 0;
  for (const fr of frames) { sDiff = Math.max(sDiff, Math.abs((fr.s - frames[0].s) - (fr.rt_s - frames[0].rt_s))); eyDiff = Math.max(eyDiff, Math.abs(fr.ey - fr.rt_ey)); }
  const run = frames.filter(fr => fr.v > .3).map(fr => fr.v);
  // ρ_req(定義: ステップ後の v・δ、μ = 車体の中心の路面 × frictionMultiplier)
  const sf = world.surfaces, fm = rep.config.frictionMultiplier, wb = rep.config.wheelbase;
  let nReq = 0, req1 = 0;
  for (const tq of rep.trace) if (tq.speed > .3) { const mu = surfaceMuAt(sf, tq.x, tq.z, fm); nReq++; if (tq.speed ** 2 * Math.abs(Math.tan(tq.steering)) / (wb * mu * G) > 1) req1++; }
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
  // 停止(走り出しのあと 1 秒より後で v < 0.3 のフレーム)
  const stops = frames.filter(fr => fr.f >= 30 && fr.v < .3).length;
  // 目標の線 R への追従(開始から 3 m 以降、|e_y − e*|、e* = 0)
  const trk = frames.filter(fr => fr.s - frames[0].s >= 3).map(fr => Math.abs(fr.ey));
  const minPre = au.min_frame >= 0 ? pre[Math.min(au.min_frame, pre.length - 1)] : pre[0];
  const pmin = au.min_frame >= 0 ? frames[Math.min(au.min_frame, frames.length - 1)] : frames[0];
  const res = {policy: kind, policy_sha256: P.sha, start: {...st0}, variation, params: spec.params ?? {}, c_v: spec.c_v ?? null, episode_mode: episodeMode, dr: spec.dr ?? null,
    reason: fs_.reason, frames: fs_.frame, collisions: m.collisions, collision: m.collision_events?.[0] ?? null,
    laps, passed_gates: m.passed_gates, completed_laps: m.completed_laps ?? null,
    lap_from_stop: episodeMode === 'single' && fs_.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1 ? m.elapsed : null,
    single_crossings: episodeMode === 'single' ? m.required_crossing_events.length : null,
    min_clearance: au.min_clearance, min_frame: au.min_frame, min_kind: au.min_kind, audit_pos_err: au.max_pos_err, overlap_frames: au.overlap_frames,
    min_at: {x: minPre.x, z: minPre.z, v: minPre.v, s_from_start: pmin.s - frames[0].s, s_mod: ((pmin.s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN},
    start_clearance: chk.start_exact, sample_check: chk,
    clear_after_3m: Math.min(...au.per_frame.filter((_, k) => frames[Math.min(k + 1, frames.length - 1)].s - frames[0].s >= 3)),
    per_frame_lt3: au.per_frame.filter(c => c < .03).length,
    speed: {mean_running: run.length ? run.reduce((a, b) => a + b, 0) / run.length : 0, p95: quant(run, .95), max: Math.max(0, ...rep.trace.map(t => t.speed)), n_running: run.length},
    stops_after_1s: stops, tracking_R: {median: quant(trk, .5), p95: quant(trk, .95), mean: trk.length ? trk.reduce((a, b) => a + b, 0) / trk.length : null},
    rho_req_gt1: nReq ? req1 / nReq : 0, own_crossings: ownCross, labels, proj_check: {max_ds_vs_runtime: sDiff, max_dey_vs_runtime: eyDiff},
    distance: m.distance};
  res.valid_episode = episodeMode === 'continuous' && fs_.reason === 'collection_duration_complete' && fs_.frame === 600 && m.collisions === 0;
  res.valid_laps = laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  if (episodeMode === 'continuous') {
    res.prof = {v: binFirstPass(frames, 'v').map(x => x == null ? null : +x.toFixed(4)), ey: binFirstPass(frames, 'ey').map(x => x == null ? null : +x.toFixed(4)),
      st: binFirstPass(frames, 'st').map(x => x == null ? null : +x.toFixed(2))};
    res.st_time = frames.map(fr => +fr.st.toFixed(2));
  }
  if (spec.keepPath) res.path = frames.filter((_, i) => i % 2 === 0).map(fr => [+pre[fr.f].x.toFixed(4), +pre[fr.f].z.toFixed(4), +fr.v.toFixed(3), +fr.s.toFixed(3), +fr.ey.toFixed(4)]);
  return res;
}
