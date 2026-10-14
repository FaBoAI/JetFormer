// 離脱復活 r4 の確認(反証)の段: 1 回を走らせる(2026-10-15)。確認 r3 の ck_rc3/ck_run.mjs(sha256 89c787b9…)の写しを r4 の候補に替えた。
// 学習・評価のコード(r3/rc4 の episode_rc4.mjs・eval_rc4.mjs・variation_metrics_rc4.mjs・gate_miss_rc4.mjs・predict_scenes_rc4.mjs・
// promote/check_recorded_scenes_rc4.mjs、r3/rc の *_rc3.mjs、rl/ の episode.mjs・eval_modes.mjs・lap_audit.mjs・env_core.mjs)を使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs。開始は dist/app.js の reset(111〜114 行)を読み直して書いた式
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル、sha256 を照合)。キックの回だけ、最後の return に state を返す関数を
//         足しただけの写し(物理は 1 文字も変えない)を data: URL で読む
//   操作: 確かめる候補そのもの(r3/candidates/recover_r4/runtime/mode_runtime.mjs + policy.json)。CK_RTDIR で写しに替えられる(強さを上限に固定する試しだけ)
//   r3 の方策・最速の方策は同じ r4 の走行中の仕組み(表 v2・シールド r4)で走らせる(比べだけ)
//   s・横ずれ・向きのずれ: reference_loop.json の点列への自前の射影。すき間: ck_geom.mjs(確認 r1 の自前の形と積分の写し、sha256 は記録)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {buildObstacles, ownClearance, ownStepFrame, nearObstacles, auditEpisode} from './ck_geom.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
export const CAND = `${EXP}/r3/candidates/recover_r4`;
export const RTDIR = process.env.CK_RTDIR || `${CAND}/runtime`;
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const CORE_PATH = `${REL}/dist/sim-core.js`;
export const SIM_CORE_SHA = sha(CORE_PATH);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(CORE_PATH);
const src = fs.readFileSync(CORE_PATH, 'utf8');
const RET = '  return { reset, step, observe, snapshot, report, demoAction, setPaused };';
if (src.split(RET).length !== 2) throw new Error('置き換えの場所が 1 つではありません');
const kickCore = await import(`data:text/javascript;base64,${Buffer.from(src.replace(RET, '  return { reset, step, observe, snapshot, report, demoAction, setPaused, __state: () => state };')).toString('base64')}`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
const RT = await import(`${RTDIR}/mode_runtime.mjs`);
const FEAT = await import(`${RTDIR}/features.mjs`);
const PERT = await import(`${RTDIR}/perturb.mjs`);
export const perturbAllowed = PERT.perturbAllowed;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// ---- 自前の R への射影 ----
const REF = JSON.parse(fs.readFileSync(`${CAND}/runtime/reference_loop.json`, 'utf8'));
const RX = REF.x, RZ = REF.z, RS = REF.s, RN = REF.n;
export const LOOP = REF.length_m;
const SEGL = Array.from({length: RN}, (_, i) => i < RN - 1 ? RS[i + 1] - RS[i] : LOOP - RS[RN - 1]);
function projSeg(i, x, z) {
  const j = (i + 1) % RN, dx = RX[j] - RX[i], dz = RZ[j] - RZ[i], l2 = dx * dx + dz * dz;
  const u = Math.max(0, Math.min(1, ((x - RX[i]) * dx + (z - RZ[i]) * dz) / l2));
  const px = RX[i] + dx * u, pz = RZ[i] + dz * u, len = Math.sqrt(l2);
  return {d2: (x - px) ** 2 + (z - pz) ** 2, sLocal: RS[i] + u * SEGL[i], px, pz, tx: dx / len, tz: dz / len};
}
const idxOf = sl => { let lo = 0, hi = RN - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (RS[m] <= sl) lo = m; else hi = m - 1; } return lo; };
export function project(x, z, yaw, prevS = null) {
  let best = null, bestLap = 0;
  if (prevS == null) { for (let i = 0; i < RN; i++) { const r = projSeg(i, x, z); if (!best || r.d2 < best.d2) best = r; } }
  else {
    const a = prevS - .6, b = prevS + 1.6;
    let lap = Math.floor(a / LOOP), i = idxOf(a - lap * LOOP);
    for (let g = 0; g < RN + 2; g++) {
      if (lap * LOOP + RS[i] > b) break;
      const r = projSeg(i, x, z);
      if (!best || r.d2 < best.d2) { best = r; bestLap = lap; }
      i++; if (i === RN) { i = 0; lap++; }
    }
  }
  const s = best.sLocal + bestLap * LOOP;
  return {s, ey: (x - best.px) * best.tz + (z - best.pz) * (-best.tx), epsi: wrap(yaw - Math.atan2(-best.tz, best.tx))};
}
// dist/app.js reset と同じ開始
export function activeWorld(world, gate, offset, yawDeg) {
  const startIndex = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(startIndex), ...world.route.slice(0, startIndex)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...world, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}},
    variation: {gate: startIndex, left_offset_m: off, left_yaw_deg: turn * 180 / Math.PI}};
}
const OB_BASE = buildObstacles(BASE_WORLD);
export function startClearance(gate, offset, yawDeg) {
  const {world} = activeWorld(BASE_WORLD, gate, offset, yawDeg);
  return ownClearance(OB_BASE, world.spawn.x, world.spawn.z, world.spawn.yaw);
}
// 元のゲート(route[i])の R 上の s(自前の射影)
export const GATE_S_OWN = BASE_WORLD.route.map(p => project(p[0], p[1], 0).s);
export const NOMINAL = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
export const POLICY_PATHS = {r4: `${CAND}/policy.json`, r3: `${EXP}/r3/candidates/recover_r3/policy.json`, fastest: `${EXP}/bundle/fastest/policy.json`};
const policies = {};
function policyFor(key) {
  if (!policies[key]) policies[key] = {weights: JSON.parse(fs.readFileSync(POLICY_PATHS[key], 'utf8')), sha: sha(POLICY_PATHS[key])};
  return policies[key];
}
const PH = {drive: 0, perturb: 1, shield: 2, kick: 3};

// spec: {policy: 'r4'|'r3'|'fastest', start: {gate, left_offset_m, left_yaw_deg, seed}, episodeMode: 'continuous'|'single', perturb: bool,
//        params: {perturb_class?, perturb_version?}, kick: {frame, yaw?, lateral?}, maxFrames?: n(途中で止める。止まりの格子だけ)}
export async function runOne(spec) {
  const P = policyFor(spec.policy ?? 'r4');
  const st0 = spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const episodeMode = spec.episodeMode ?? 'continuous';
  const base = {...NOMINAL, seed, timeLimit: 20};
  if (episodeMode === 'continuous') base.episodeMode = 'continuous-laps-v1';
  const C = spec.kick ? kickCore : core;
  const sim = C.createSimulation(world, base);
  let snap = sim.reset({seed, scenario: 'baseline'});
  const startClr = ownClearance(OB_BASE, world.spawn.x, world.spawn.z, world.spawn.yaw);
  if (snap.state.terminal) return {invalid_spawn: true, spec, start_clearance: startClr};
  const cfgN = snap.config;
  const inner = RT.policyFromWeights(P.weights, 79);
  const raw = [0, 0];
  const policy = (obs, out) => { inner(obs, out); raw[0] = out[0]; raw[1] = out[1]; };
  // spec.mode(20:50 に足した。最速のモードそのもの 'fastest' との比べだけ。既定 'recover')
  // spec.overrides(20:30 に足した。シールドを切った比べ {shield: false} だけで使う。ほかの回は今までどおり)
  const overrides = {...(spec.perturb === false ? {perturbation: false} : {}), ...(spec.overrides ?? {})};
  const teacher = RT.createModeTeacher(world, cfgN, variation, {mode: spec.mode ?? 'recover', weights: P.weights, policySha256: P.sha, policy, params: spec.params ?? {}, overrides});
  const F = [];
  let prevS = null, kicked = null, p0 = null;
  const lab = {changed_not_perturb_not_shield: 0, l0_not_perturb: 0, perturb_l1: 0, shield_l0: 0, l0_count: 0, sh_count: 0, info_bad_keys: 0};
  const maxFrames = spec.maxFrames ?? Infinity;
  while (!snap.state.terminal && snap.state.frame < maxFrames) {
    const st = snap.state, m = snap.metrics;
    const out = teacher.action({state: st, metrics: m, config: cfgN});
    const jf = out.info.jf, lg = teacher.log.at(-1);
    if (st.frame === 0) p0 = jf.p ?? null;
    const pc = FEAT.actionToCommand(raw[0], raw[1], 4.5, cfgN.maxSteer);
    const changed = Math.abs(out.steering - pc.steering) > 1e-12 || Math.abs(out.speed - pc.speed) > 1e-12;
    if (changed && jf.ph !== 'perturb' && jf.sh !== 1) lab.changed_not_perturb_not_shield++;
    if (jf.l === 0 && jf.ph !== 'perturb') lab.l0_not_perturb++;
    if (jf.ph === 'perturb' && jf.l !== 0) lab.perturb_l1++;
    if (jf.sh === 1 && jf.l !== 1) lab.shield_l0++;
    if (jf.l === 0) lab.l0_count++;
    if (jf.sh === 1) lab.sh_count++;
    if (/steering_pwm|throttle_pwm/.test(JSON.stringify(out.info))) lab.info_bad_keys++;
    const pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    F.push({f: st.frame, x: st.x, z: st.z, yaw: st.yaw, v: st.speed, s: pr.s, ey: pr.ey, epsi: pr.epsi, st: out.steering, sp: out.speed,
      pst: pc.steering, psp: pc.speed, ch: changed ? 1 : 0, l: jf.l, ph: PH[jf.ph] ?? 9, ev: jf.ev, sh: jf.sh ?? 0, g: m.passed_gates, lap: m.completed_laps ?? 0,
      fb: jf.sh === 1 ? (lg.shield?.fb ?? 'x') : null, rs: lg.s, rey: lg.ey});
    // キックの時刻: kick.frame(時刻の格子・一様な時刻)か、kick.s_target(r4 の確認で足した: min_frame 以降に R 上の s mod L が
    // [s_target, s_target + 0.4] に入った最初のフレーム。スタートのゲートの区間を狙うため)
    let fire = false;
    if (spec.kick && !kicked) {
      if (spec.kick.frame != null) fire = st.frame === spec.kick.frame;
      else if (spec.kick.s_target != null && st.frame >= (spec.kick.min_frame ?? 45)) { const sm = ((pr.s % LOOP) + LOOP) % LOOP; fire = sm >= spec.kick.s_target && sm <= spec.kick.s_target + .4; }
    }
    if (fire) {
      const s = sim.__state();
      if (spec.kick.yaw) s.yaw = wrap(s.yaw + spec.kick.yaw);
      if (spec.kick.lateral) { s.x += spec.kick.lateral * -Math.sin(s.yaw); s.z += spec.kick.lateral * -Math.cos(s.yaw); }
      kicked = {frame: st.frame, x: s.x, z: s.z, yaw: s.yaw, v: s.speed, s_pre: pr.s, ey_pre: pr.ey, rt_s: lg.s, allowed_rt: perturbAllowed(lg.s), allowed_own: perturbAllowed(pr.s),
        clr_after: ownClearance(OB_BASE, s.x, s.z, s.yaw)};
    }
    snap = sim.step({steering: out.steering, speed: out.speed});
  }
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  const fin = project(fs_.x, fs_.z, fs_.yaw, prevS);
  const res = {spec, policy_sha: P.sha, variation, start_clearance: startClr, reason: fs_.reason, frames: fs_.frame, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: m.collision_events[0].time, collider: m.collision_events[0].collider, kind: m.collision_events[0].kind} : null,
    passed_gates: m.passed_gates, completed_laps: m.completed_laps ?? null,
    laps: (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions})),
    events: teacher.events.map(e => ({n: e.n, type: e.type, cls: e.cls, kind: e.kind ?? null, sign: e.sign, ps: e.ps, pe: e.pe, pe_planned: e.pe_planned, aborted: e.aborted,
      s: e.s, v: e.v, frames: e.frames ?? null, k: e.k ?? null, d: e.d ?? null, T: e.T ?? null, req: e.req ?? null, fallback: e.fallback ?? null, wait: e.wait_frames ?? null})),
    p0, labels: lab, kick: kicked, final: {s: fin.s, ey: fin.ey, x: fs_.x, z: fs_.z, v: fs_.speed}, shield_frames: teacher.shieldFrames,
    metadata_perturbation: teacher.metadata.perturbation ?? null, metadata_shield: teacher.metadata.shield ?? null};
  res.valid_episode = episodeMode === 'continuous' && fs_.reason === 'collection_duration_complete' && fs_.frame === 600 && m.collisions === 0;
  if (episodeMode === 'single') res.lap_from_stop = fs_.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1 ? m.elapsed : null;
  // すき間(自前)。キックの回は自前の状態にもキックを入れて積分し直す
  if (!spec.kick) {
    const au = auditEpisode(world, rep.config, world.spawn, rep.actions, rep.trace, {skipLastErr: fs_.reason !== 'collection_duration_complete' && fs_.reason !== 'route_complete'});
    Object.assign(res, {min_clearance: au.min_clearance, min_frame: au.min_frame, min_kind: au.min_kind, audit_pos_err: au.max_pos_err, overlap_frames: au.overlap_frames});
    res.clear_pf = au.per_frame;
  } else {
    const ob = buildObstacles(world), cfg = rep.config, own = {x: world.spawn.x, z: world.spawn.z, yaw: world.spawn.yaw, speed: 0, steering: 0};
    let minC = Infinity, over = 0, maxErr = 0, t = 0; const pf = [];
    for (let f = 0; f < rep.actions.length; f++) {
      if (kicked && f === kicked.frame) { own.yaw = wrap(own.yaw + (spec.kick.yaw ?? 0)); if (spec.kick.lateral) { own.x += spec.kick.lateral * -Math.sin(own.yaw); own.z += spec.kick.lateral * -Math.cos(own.yaw); } }
      const near = nearObstacles(ob, own.x, own.z, 1.0);
      const poses = ownStepFrame(own, rep.actions[f], cfg, ob.surfaces, {timeLeft: cfg.timeLimit - t}); t += cfg.dt;
      let fMin = Infinity; for (const p of poses) fMin = Math.min(fMin, ownClearance(near, p.x, p.z, p.yaw));
      if (f < rep.actions.length - 1 || fs_.reason === 'collection_duration_complete') { const tr = rep.trace[f]; if (tr) maxErr = Math.max(maxErr, Math.hypot(tr.x - own.x, tr.z - own.z)); }
      minC = Math.min(minC, fMin); if (fMin <= 0) over++; pf.push(fMin);
    }
    Object.assign(res, {min_clearance: minC, overlap_frames: over, audit_pos_err: maxErr});
    res.clear_pf = pf;
  }
  res.frames_data = F;
  return res;
}
