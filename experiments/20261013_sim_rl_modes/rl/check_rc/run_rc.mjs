// 離脱復活(recover)の確認 r1: 1 回を走らせる(2026-10-14、確認の段)。学習・評価のコード(episode.mjs・eval_modes.mjs・env_core.mjs・
// lap_audit.mjs・variation_metrics.mjs・build_nominal_line.mjs)を使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs(ブラウザと同じ scene.js から作る)を dist/app.js の reset と同じ式で開始ゲートへ回す
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル)。キックの回だけ、この確認で書いた 1 行の置き換え(state を返す関数を足すだけ)の写し
//   操作: 書き出したブラウザの組 web/rl/mode_runtime.mjs と bundle/<mode>/policy.json(ブラウザで走るものと同じ)。
//         方策の生の出力をこちらで横取りし、適用した操作と比べる(ラベルの漏れの確かめ)
//   s・横ずれ・向きのずれ: reference_loop.json の点列への自前の射影。すき間: check_nw/own_geom.mjs(no_wall の確認の段の自前の形と積分)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {auditEpisode, buildObstacles, ownClearance, ownStepFrame, nearObstacles} from '../check_nw/own_geom.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const CORE_PATH = `${REL}/dist/sim-core.js`;
export const SIM_CORE_SHA = sha(CORE_PATH);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(CORE_PATH);
// キック用: 元のソースの最後の return に state を返す関数を足すだけ(物理は 1 文字も変えない)
const src = fs.readFileSync(CORE_PATH, 'utf8');
const RET = '  return { reset, step, observe, snapshot, report, demoAction, setPaused };';
if (src.split(RET).length !== 2) throw new Error('置き換えの場所が 1 つではありません');
const kickSrc = src.replace(RET, '  return { reset, step, observe, snapshot, report, demoAction, setPaused, __state: () => state };');
const kickCore = await import(`data:text/javascript;base64,${Buffer.from(kickSrc).toString('base64')}`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
const RT = await import(`${EXP}/web/rl/mode_runtime.mjs`);
const FEAT = await import(`${EXP}/web/rl/features.mjs`);
const {perturbAllowed} = await import(`${EXP}/web/rl/perturb.mjs`);
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

// ---- 自前の R(reference_loop.json の点列)への射影 ----
const REF = JSON.parse(fs.readFileSync(`${EXP}/web/rl/reference_loop.json`, 'utf8'));
const RX = REF.x, RZ = REF.z, RS = REF.s, RN = REF.n;
export const LOOP = REF.length_m;
const SEGL = Array.from({length: RN}, (_, i) => i < RN - 1 ? RS[i + 1] - RS[i] : LOOP - RS[RN - 1]);
function projSeg(i, x, z) {
  const j = (i + 1) % RN, dx = RX[j] - RX[i], dz = RZ[j] - RZ[i], l2 = dx * dx + dz * dz;
  const u = Math.max(0, Math.min(1, ((x - RX[i]) * dx + (z - RZ[i]) * dz) / l2));
  const px = RX[i] + dx * u, pz = RZ[i] + dz * u, len = Math.sqrt(l2);
  return {d2: (x - px) ** 2 + (z - pz) ** 2, sLocal: RS[i] + u * SEGL[i], px, pz, tx: dx / len, tz: dz / len};
}
const idxOf = sLocal => { let lo = 0, hi = RN - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (RS[m] <= sLocal) lo = m; else hi = m - 1; } return lo; };
// 前のフレームの s(周をまたいだ値)を中心に −0.6〜+1.6 m を探す。prevS = null なら全部
export function project(x, z, yaw, prevS = null) {
  let best = null, bestLap = 0;
  if (prevS == null) {
    for (let i = 0; i < RN; i++) { const r = projSeg(i, x, z); if (!best || r.d2 < best.d2) best = r; }
  } else {
    const a = prevS - .6, b = prevS + 1.6;
    let lap = Math.floor(a / LOOP), i = idxOf(a - lap * LOOP);
    for (let g = 0; g < RN + 2; g++) {
      const start = lap * LOOP + RS[i];
      if (start > b) break;
      const r = projSeg(i, x, z);
      if (!best || r.d2 < best.d2) { best = r; bestLap = lap; }
      i++; if (i === RN) { i = 0; lap++; }
    }
  }
  const s = best.sLocal + bestLap * LOOP;
  const ey = (x - best.px) * best.tz + (z - best.pz) * (-best.tx);   // R の左が正
  const epsi = wrap(yaw - Math.atan2(-best.tz, best.tx));
  return {s, ey, epsi};
}

// app.js の reset と同じ(dist/app.js 111〜114 行)
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
export const NOMINAL = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
const policies = {};
function policyFor(key) {
  if (!policies[key]) { const p = `${EXP}/bundle/${key}/policy.json`; policies[key] = {weights: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p)}; }
  return policies[key];
}

// spec: {mode: 'recover'|'fastest', policy: 'recover'|'fastest', start: {gate, left_offset_m, left_yaw_deg, seed}, episodeMode, perturb: true|false,
//        classes: null|['弱','中'], kick: {frame, yaw?, lateral?}, keep: 'full'|'path'|null}
export async function runOne(spec) {
  const mode = spec.mode ?? 'recover', P = policyFor(spec.policy ?? mode);
  const st0 = spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const episodeMode = spec.episodeMode ?? 'continuous';
  const base = {...NOMINAL, seed, timeLimit: 20};
  if (episodeMode === 'continuous') base.episodeMode = 'continuous-laps-v1';
  const C = spec.kick ? kickCore : core;
  const sim = C.createSimulation(world, base);
  let snap = sim.reset({seed, scenario: 'baseline'});
  if (snap.state.terminal) return {invalid_spawn: true, spec};
  const cfgN = snap.config;
  // 方策の生の出力を横取り
  const inner = RT.policyFromWeights(P.weights, 79);
  const raw = [0, 0];
  const policy = (obs, out) => { inner(obs, out); raw[0] = out[0]; raw[1] = out[1]; };
  const overrides = spec.perturb === false && mode === 'recover' ? {perturbation: false} : {};
  const params = spec.classes ? {perturb_class: spec.classes} : {};
  const teacher = RT.createModeTeacher(world, cfgN, variation, {mode, weights: P.weights, policySha256: P.sha, policy, params, overrides});
  const F = [];
  let prevS = null, kicked = null;
  const labelBad = {changed_but_l1: 0, l0_outside_event: 0, l0_count: 0, info_bad_keys: 0, ev_mismatch: 0};
  let p0 = null;
  while (!snap.state.terminal) {
    const st = snap.state, m = snap.metrics;
    const out = teacher.action({state: st, metrics: m, config: cfgN});
    const jf = out.info.jf;
    if (st.frame === 0) p0 = jf.p ?? null;
    const pc = FEAT.actionToCommand(raw[0], raw[1], 4.5, cfgN.maxSteer);
    const changed = Math.abs(out.steering - pc.steering) > 1e-12 || Math.abs(out.speed - pc.speed) > 1e-12;
    if (changed && jf.l === 1) labelBad.changed_but_l1++;
    if (jf.l === 0) labelBad.l0_count++;
    if (/steering_pwm|throttle_pwm|"\//.test(JSON.stringify(out.info))) labelBad.info_bad_keys++;
    const pr = project(st.x, st.z, st.yaw, prevS); prevS = pr.s;
    F.push({f: st.frame, x: st.x, z: st.z, yaw: st.yaw, v: st.speed, d: st.steering, s: pr.s, ey: pr.ey, epsi: pr.epsi,
      st: out.steering, sp: out.speed, pst: pc.steering, psp: pc.speed, ch: changed ? 1 : 0, l: jf.l, ph: jf.ph, ev: jf.ev, g: m.passed_gates,
      rs: teacher.log.at(-1).s, rey: teacher.log.at(-1).ey});
    if (spec.kick && st.frame === spec.kick.frame && !kicked) {
      const s = sim.__state();
      if (spec.kick.yaw) s.yaw = wrap(s.yaw + spec.kick.yaw);
      if (spec.kick.lateral) { s.x += spec.kick.lateral * -Math.sin(s.yaw); s.z += spec.kick.lateral * -Math.cos(s.yaw); }
      kicked = {frame: st.frame, x: s.x, z: s.z, yaw: s.yaw, s_pre: pr.s, allowed: perturbAllowed(teacher.log.at(-1).s)};
    }
    snap = sim.step({steering: out.steering, speed: out.speed});
  }
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  const res = {spec, variation, reason: fs_.reason, frames: fs_.frame, collisions: m.collisions, collision: m.collision_events?.[0] ?? null,
    passed_gates: m.passed_gates, completed_laps: m.completed_laps ?? null,
    laps: (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions, distance: l.distance})),
    events: teacher.events.map(e => ({n: e.n, type: e.type, cls: e.cls, ps: e.ps, pe: e.pe, pe_planned: e.pe_planned, aborted: e.aborted, s: e.s, sign: e.sign,
      k: e.k ?? null, frames: e.frames ?? null, d: e.d ?? null, T: e.T ?? null, kind: e.kind ?? null})),
    virtual_misses: teacher.wrapper.misses ?? null, p0, labels: labelBad, kick: kicked};
  res.valid_episode = episodeMode === 'continuous' && fs_.reason === 'collection_duration_complete' && fs_.frame === 600 && m.collisions === 0;
  res.valid_laps = res.laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  if (episodeMode === 'single') res.lap_from_stop = fs_.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1 ? m.elapsed : null;
  // すき間(自前)。キックの回は、キックを自前の状態にも入れて積分し直す
  if (!spec.kick) {
    const au = auditEpisode(world, rep.config, world.spawn, rep.actions, rep.trace, {skipLastErr: fs_.reason !== 'collection_duration_complete' && fs_.reason !== 'route_complete'});
    Object.assign(res, {min_clearance: au.min_clearance, min_frame: au.min_frame, min_kind: au.min_kind, audit_pos_err: au.max_pos_err, overlap_frames: au.overlap_frames});
    res.clear_pf = au.per_frame;
  } else {
    const ob = buildObstacles(world), cfg = rep.config, own = {x: world.spawn.x, z: world.spawn.z, yaw: world.spawn.yaw, speed: 0, steering: 0};
    let minC = Infinity, over = 0, maxErr = 0, t = 0;
    for (let f = 0; f < rep.actions.length; f++) {
      if (kicked && f === kicked.frame) { own.yaw = wrap(own.yaw + (spec.kick.yaw ?? 0)); if (spec.kick.lateral) { own.x += spec.kick.lateral * -Math.sin(own.yaw); own.z += spec.kick.lateral * -Math.cos(own.yaw); } }
      const near = nearObstacles(ob, own.x, own.z, 1.0);
      const poses = ownStepFrame(own, rep.actions[f], cfg, ob.surfaces, {timeLeft: cfg.timeLimit - t}); t += cfg.dt;
      let fMin = Infinity; for (const p of poses) fMin = Math.min(fMin, ownClearance(near, p.x, p.z, p.yaw));
      if (f < rep.actions.length - 1 || fs_.reason === 'collection_duration_complete') { const tr = rep.trace[f]; if (tr) maxErr = Math.max(maxErr, Math.hypot(tr.x - own.x, tr.z - own.z)); }
      minC = Math.min(minC, fMin); if (fMin <= 0) over++;
    }
    Object.assign(res, {min_clearance: minC, overlap_frames: over, audit_pos_err: maxErr});
  }
  // 黒シートの横断(自前): z = 0 を −z 向きに、x > 0、車幅ぶん x 1.26〜2.36 の中
  const cross = [];
  for (let i = 1; i < rep.trace.length; i++) {
    const a = rep.trace[i - 1], b = rep.trace[i];
    if (a.z > 0 && b.z <= 0) {
      const u = a.z / (a.z - b.z), x = a.x + (b.x - a.x) * u, y = a.yaw + wrap(b.yaw - a.yaw) * u;
      const half = .21 * Math.abs(Math.cos(y)) + .095 * Math.abs(Math.sin(y));
      if (x > 0) cross.push({frame: i, x, inside: x - half >= 1.26 - 1e-9 && x + half <= 2.36 + 1e-9});
    }
  }
  res.own_crossings = cross;
  res.frames_data = F;
  return res;
}
