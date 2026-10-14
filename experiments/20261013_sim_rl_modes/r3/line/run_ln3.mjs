// r3 の下調べ(2026-10-14): 確認 line r2 の run_ln2.mjs(sha256 22e16cec…)の写し。違いは、走らせる仕組みを r3/runtime にした(JF_RT=web なら元の web/rl)ことと、
// 環境変数 JF_R3(JSON)を globalThis.JF_R3 に入れること(r3/runtime の試作の切り替え)だけ。
// 経路いろいろ(line)の確認 r2: 1 回を走らせる(2026-10-14、確認の段)。学習・評価のコード(episode.mjs・eval_modes.mjs・env_modes.mjs・
// lap_audit.mjs・variation_metrics.mjs・gate_miss.mjs・line_r2/*)を使わない。r1 の確認の run_ln.mjs を r2 向けに直したもの:
//   コース: ユーザーのリリースの qa/load-course-world.mjs を app.js の reset と同じ式で開始ゲートへ回す
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル、sha256 aaeb2379…549e)。標準の物理だけ
//   操作: ブラウザに配る組 web/rl/mode_runtime.mjs + 候補 candidates/line_r2/policy.json(1ae6be07…)。比べる相手は bundle/fastest(7c43cb53…)
//   すき間: check_ln/own_geom.mjs(自前の多角形の距離、小区間ごと)+ own_extra.mjs の 4 mm おきの点で確かめ
//   s・e_y: own_extra.mjs の自前の R への射影。e*: own_line_v2.mjs(protocol_line_r2 の文言から。random の c(s) だけランタイム)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {auditEpisode, buildObstacles, ownClearance, ownStepFrame, nearObstacles} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_ln/own_geom.mjs';
import {sampleClearance, createProjector, binFirstPass, LOOP_L_OWN} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_ln/own_extra.mjs';
import {REF_LOOP, rAt} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_ln/own_line.mjs';
import {createOwnV2} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/check_ln2/own_line_v2.mjs';

const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export const SIM_CORE_SHA = sha(`${REL}/dist/sim-core.js`);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(`${REL}/dist/sim-core.js`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
if (process.env.JF_R3) globalThis.JF_R3 = JSON.parse(process.env.JF_R3);
const RTDIR = process.env.JF_RT === 'web' ? `${EXP}/web/rl` : `${EXP}/r3/runtime`;
const RT = await import(`${RTDIR}/mode_runtime.mjs`);
const LP = await import(`${RTDIR}/line_profiles.mjs`);
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const POLICY_PATH = {line: `${EXP}/candidates/line_r2/policy.json`, line_r1: `${EXP}/candidates/line/policy.json`, fastest: `${EXP}/bundle/fastest/policy.json`};
const EXPECT_SHA = {line: '1ae6be0793914094dbabfbdc325905ab8af01f12ea3bc7b2638b05c6bd0167ae', line_r1: '2df85c569eedf8fe41a07dcfa06348f6d8be17dfcf8e53d427bc71ade268655b',
  fastest: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee'};
const policies = {};
export function policyFor(kind) {
  const p = POLICY_PATH[kind] ?? kind;
  if (!policies[p]) policies[p] = {weights: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p), path: p};
  const P = policies[p];
  if (EXPECT_SHA[kind] && P.sha !== EXPECT_SHA[kind]) throw new Error(`${kind} の方策の sha256 が違います`);
  return P;
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
export const OB = buildObstacles(BASE_WORLD);
export function startPose(gate, offset, yawDeg) { return activeWorld(BASE_WORLD, gate, offset, yawDeg).world.spawn; }
export function startClearance(gate, offset, yawDeg) { const sp = startPose(gate, offset, yawDeg); return ownClearance(OB, sp.x, sp.z, sp.yaw); }
// 開始の車の向きと R の向きのずれ(自前の射影の最初の区間の向き)
export function startHeadingVsR(gate, offset, yawDeg) {
  const sp = startPose(gate, offset, yawDeg), pj = createProjector(), r = pj.reset(sp.x, sp.z, sp.yaw);
  const a = rAt(r.s - .05), b = rAt(r.s + .05), psiR = Math.atan2(-(b.z - a.z), b.x - a.x);
  return {dpsi: wrap(sp.yaw - psiR), s: ((r.s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN, ey: r.ey};
}
// R の上(e = 0、R の向き)の静的なすき間
export function rStaticClearance(s) { const a = rAt(s - .05), b = rAt(s + .05), r = rAt(s); return ownClearance(OB, r.x, r.z, Math.atan2(-(b.z - a.z), b.x - a.x)); }
const quant = (arr, p) => { const b = [...arr].sort((x, y) => x - y); if (!b.length) return null; const i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
const steerUs = a0 => 1442 + a0 * (a0 >= 0 ? 558 : 442);
const GATE_S = REF_LOOP.gate_s;
const r4 = x => x == null || !Number.isFinite(x) ? null : +x.toFixed(4);

// spec: {policy: 'line'|'fastest'|<path>, mode, start: {gate, left_offset_m, left_yaw_deg, seed}, params: {line, c_v}, keepFrames, keepPath}
export async function runOne(spec) {
  const kind = spec.policy ?? 'line', P = policyFor(kind), mode = spec.mode ?? (kind.startsWith('line') ? 'line' : 'fastest');
  const st0 = spec.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = activeWorld(BASE_WORLD, st0.gate ?? 0, st0.left_offset_m ?? 0, st0.left_yaw_deg ?? 0);
  const base = {...NOMINAL, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'};
  const sim = core.createSimulation(world, base);
  let snap = sim.reset({seed, scenario: 'baseline'});
  if (snap.state.terminal) return {invalid_spawn: true, start: st0};
  const cfgN = snap.config;
  const teacher = RT.createModeTeacher(world, cfgN, variation, {mode, weights: P.weights, policySha256: P.sha, params: spec.params ?? {}});
  const lineP = teacher.params.line ?? null, cV = teacher.params.c_v ?? null;
  const labels = {frames: 0, l0: 0, ph: {}, ev_nonzero: 0, sh_key: 0, es_missing: 0, bad_keys: 0, p0: null, bytes_max: 0, cmd_over_cap: 0, meta_line_profile: teacher.metadata.line_profile ?? null};
  const pre = [], gateCnt = [];
  while (!snap.state.terminal) {
    const st = snap.state;
    gateCnt.push(snap.metrics.passed_gates ?? 0);
    const out = teacher.action({state: st, metrics: snap.metrics, config: cfgN});
    const jf = out.info.jf, js = JSON.stringify(out.info);
    labels.frames++; if (jf.l !== 1) labels.l0++; labels.ph[jf.ph] = (labels.ph[jf.ph] ?? 0) + 1; if ('sh' in jf) labels.sh_key++;
    if (jf.ev) labels.ev_nonzero++;
    if (mode === 'line' && jf.es === undefined) labels.es_missing++;
    if (st.frame === 0) labels.p0 = jf.p ?? null;
    if (js.match(/steering_pwm|throttle_pwm|"\//)) labels.bad_keys++;
    if (st.frame > 0) labels.bytes_max = Math.max(labels.bytes_max, js.length);
    if (cV != null && out.speed > 4.5 * cV + 1e-9) labels.cmd_over_cap++;
    const lg = teacher.log.at(-1);
    pre.push({f: st.frame, x: st.x, z: st.z, yaw: st.yaw, v: st.speed, st_cmd: out.steering, sp_cmd: out.speed, rt_s: lg.s, rt_ey: lg.ey,
      es_rt: jf.es === undefined ? null : lg.es, es_mm: jf.es ?? null, cl_mm: jf.cl});
    snap = sim.step({steering: out.steering, speed: out.speed});
  }
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  const au = auditEpisode(world, rep.config, world.spawn, rep.actions, rep.trace, {skipLastErr: fs_.reason !== 'collection_duration_complete'});
  const chk = {n: 0, max_exact_minus_sample: -Infinity};
  {
    const stp = {x: world.spawn.x, z: world.spawn.z, yaw: world.spawn.yaw, speed: 0, steering: 0};
    let t = 0;
    for (let f = 0; f < rep.actions.length; f++) {
      const near = nearObstacles(OB, stp.x, stp.z, 1.0);
      const poses = ownStepFrame(stp, rep.actions[f], rep.config, OB.surfaces, {timeLeft: rep.config.timeLimit - t});
      t += rep.config.dt;
      for (const p of poses) {
        const e = ownClearance(near, p.x, p.z, p.yaw);
        if (e < .06) { const s = sampleClearance(near, p.x, p.z, p.yaw); chk.n++; chk.max_exact_minus_sample = Math.max(chk.max_exact_minus_sample, e - s); }
      }
    }
  }
  // 自前の R への射影と自前の e*(v2: 最初のフレームの自前の e_y・s で開始のつなぎ)
  const pj = createProjector();
  const proj = pre.map((q, k) => k === 0 ? pj.reset(q.x, q.z, q.yaw) : pj.advance(q.x, q.z));
  const s0 = proj[0].s, e0 = proj[0].ey;
  let own = null;
  if (mode === 'line' && lineP) {
    const ver = teacher.params.line_profile ?? 'v1';
    if (ver !== 'v2') throw new Error('line の候補が v2 で動いていません');
    const cFn = lineP.family === 'random' ? LP.createLineProfile(lineP, s0, {version: 'v1'}).c : null;
    own = createOwnV2(lineP, s0, e0, {cFn});
  }
  const sp0 = world.spawn, a0 = rAt(s0 - .05), b0 = rAt(s0 + .05), dpsi0 = wrap(sp0.yaw - Math.atan2(-(b0.z - a0.z), b0.x - a0.x));
  const frames = pre.map((q, k) => ({f: q.f, s: proj[k].s, ey: proj[k].ey, v: q.v, st: steerUs(q.st_cmd / .52), sp: q.sp_cmd, rt_s: q.rt_s, rt_ey: q.rt_ey,
    es_rt: q.es_rt, es_own: own ? own.eStar(proj[k].s) : null, es_mm: q.es_mm, clr: au.per_frame[k] ?? null, t: k / 30}));
  let sDiff = 0, eyDiff = 0, esDiffMax = 0, esMmDiffMax = 0, nEs = 0;
  for (const fr of frames) {
    sDiff = Math.max(sDiff, Math.abs((fr.s - s0) - (fr.rt_s - frames[0].rt_s)));
    eyDiff = Math.max(eyDiff, Math.abs(fr.ey - fr.rt_ey));
    if (fr.es_own != null && fr.es_rt != null) { esDiffMax = Math.max(esDiffMax, Math.abs(fr.es_own - fr.es_rt)); nEs++; }
    if (fr.es_mm != null && fr.es_rt != null) esMmDiffMax = Math.max(esMmDiffMax, Math.abs(fr.es_mm / 1000 - fr.es_rt));
  }
  const after3 = frames.filter(fr => fr.s - s0 >= 3);
  const trkOwn = after3.filter(fr => fr.es_own != null).map(fr => Math.abs(fr.ey - fr.es_own));
  const trkRt = after3.filter(fr => fr.es_rt != null).map(fr => Math.abs(fr.ey - fr.es_rt));
  const laps = (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions,
    distance: l.distance, start_frame: l.start_frame, end_frame: l.end_frame}));
  const sEnd = frames.at(-1).s, gatesCrossedS = [];
  for (let n = Math.floor((s0 - 1) / LOOP_L_OWN); n <= Math.floor(sEnd / LOOP_L_OWN) + 1; n++)
    for (const gs of GATE_S) { const a = n * LOOP_L_OWN + gs; if (a > s0 + .3 && a <= sEnd) gatesCrossedS.push(a); }
  gatesCrossedS.sort((a, b) => a - b);
  let stuck = 0;
  for (let k = 0; k < frames.length; k++) { const need = gatesCrossedS.filter(a => a + .5 < frames[k].s).length; if (gateCnt[k] < need) stuck = Math.max(stuck, need - gateCnt[k]); }
  const ownCross = [];
  for (let i = 1; i < rep.trace.length; i++) {
    const a = rep.trace[i - 1], b = rep.trace[i];
    if (a.z > 0 && b.z <= 0) {
      const u = a.z / (a.z - b.z), x = a.x + (b.x - a.x) * u, y = a.yaw + wrap(b.yaw - a.yaw) * u;
      const half = .21 * Math.abs(Math.cos(y)) + .095 * Math.abs(Math.sin(y));
      if (x > 0) ownCross.push({frame: i, x: +x.toFixed(4), inside: x - half >= 1.26 - 1e-9 && x + half <= 2.36 + 1e-9});
    }
  }
  const run = frames.filter(fr => fr.v > .3).map(fr => fr.v);
  const mf = au.min_frame >= 0 ? Math.min(au.min_frame, frames.length - 1) : 0, minPre = frames[mf];
  const clrWhere = pred => { const a = au.per_frame.filter((_, k) => pred(k)); return a.length ? Math.min(...a) : null; };
  const res = {policy: kind, mode, policy_sha256: P.sha, start: {...st0}, variation, params: teacher.params, family: spec.family ?? lineP?.family ?? null,
    reason: fs_.reason, frames: fs_.frame, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: m.collision_events[0].time, collider: m.collision_events[0].collider, kind: m.collision_events[0].kind} : null,
    laps, passed_gates: m.passed_gates, gates_by_s: gatesCrossedS.length, gate_stuck_max: stuck,
    min_clearance: au.min_clearance, min_frame: au.min_frame, min_t: au.min_frame >= 0 ? au.min_frame / 30 : 0, min_kind: au.min_kind,
    min_s_from_start: minPre.s - s0, min_s_mod: ((minPre.s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN, min_v: minPre.v, min_ey: minPre.ey, min_es: minPre.es_own ?? minPre.es_rt,
    min_R_static_clearance: rStaticClearance(minPre.s),
    start_s_mod: ((s0 % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN, start_ey: e0, start_dpsi_vs_R: dpsi0,
    clr_first_1p5s: clrWhere(k => k / 30 < 1.5), clr_after_1p5s: clrWhere(k => k / 30 >= 1.5), clr_after_3s: clrWhere(k => k / 30 >= 3),
    clr_after_blend_2m: clrWhere(k => frames[Math.min(k, frames.length - 1)].s - s0 >= 2), clr_after_3m: clrWhere(k => frames[Math.min(k, frames.length - 1)].s - s0 >= 3),
    audit_pos_err: au.max_pos_err, overlap_frames: au.overlap_frames, start_clearance: ownClearance(OB, sp0.x, sp0.z, sp0.yaw), sample_check: chk,
    speed: {mean_running: run.length ? run.reduce((a, b) => a + b, 0) / run.length : 0, p95: quant(run, .95), max: Math.max(0, ...rep.trace.map(t => t.speed))},
    tracking_own: trkOwn.length ? {n: trkOwn.length, median: quant(trkOwn, .5), p95: quant(trkOwn, .95)} : null,
    tracking_rt: trkRt.length ? {n: trkRt.length, median: quant(trkRt, .5), p95: quant(trkRt, .95)} : null,
    es_check: {n: nEs, max_abs_own_minus_rt: esDiffMax, max_abs_info_mm_minus_rt: esMmDiffMax}, proj_check: {max_ds_vs_runtime: sDiff, max_dey_vs_runtime: eyDiff},
    own_crossings: ownCross, labels, distance: m.distance, s_travel: sEnd - s0};
  res.valid_episode = fs_.reason === 'collection_duration_complete' && fs_.frame === 600 && m.collisions === 0;
  res.valid_laps = laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  res.prof = {ey: binFirstPass(frames, 'ey').map(r4), v: binFirstPass(frames, 'v').map(r4)};
  // フレームの列: [t, s − s0, s mod L, e_y, e*_rt, e*_own, すき間, v, 舵 µs, 速さの指令]
  if (spec.keepFrames !== false) res.fr = frames.map(fr => [+fr.t.toFixed(4), r4(fr.s - s0), r4(((fr.s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN), r4(fr.ey), r4(fr.es_rt), r4(fr.es_own), r4(fr.clr), r4(fr.v), +fr.st.toFixed(1), r4(fr.sp)]);
  if (spec.keepPath) res.path = pre.map(q => [+q.x.toFixed(4), +q.z.toFixed(4), +q.yaw.toFixed(4)]);
  return res;
}
