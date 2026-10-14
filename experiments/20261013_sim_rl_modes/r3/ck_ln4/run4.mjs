// 確認 line r4(2026-10-14 夜): 1 回を走らせる自前の走らせ方。
// 学習・評価のコード(r3/line4b/*・r3/line4/*・r3/line/*・rl/*)も、前の確認のコード(r3/ck_ln3*/*)も使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs(ブラウザと同じ scene.js から作る)
//   開始:   dist/app.js の reset の式(経路を開始ゲートから回し、spawn = first + offset·(sin h, −cos h)、yaw = h + turn)をここで書いた spawnAt()
//   物理:   リリースの dist/sim-core.js(元のファイル。sha256 を毎回確かめる)、modes.json common.physics_nominal + real-drive-response-estimate-v1
//   操作:   確かめる相手そのもの = 候補 candidates/line_r4/ の runtime/mode_runtime.mjs + policy.json + mode.json の runtime(modeRuntime)
//   すき間: geom4.mjs(自前の幾何)を、自前の積分の小区間ごとに。さらに「障害物を 3 cm 太らせた world」を sim-core.js にそのまま渡して
//           同じ操作の列を流し直し、sim 自身の当たり判定で 3 cm 未満を数える(自前の幾何とは別の道具)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {buildObstacles, nearList, clearance, inflatedWorld, loadLoop} from './geom4.mjs';

export const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
export const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
export const CAND = `${EXP}/r3/candidates/line_r4`;
export const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const need = (c, m) => { if (!c) throw new Error(m); };
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

export const EXPECT = {
  sim_core: 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e',
  policy_line: '5c197a2beb90cf96701180a1a176bde795cad9c2cfdd1271b6f1fa68981bcd8c',
  policy_fastest: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee',
  shield_mjs: '130ed5b29a3dbae60777bf91e4eff9bdc23a4719b69e9c6057ddee14bc939cdf',
  mode_runtime: '2fa7146d72310b9bb3605b14af9f4de31a9068c1001f3e7c46687d2d47da65cd',
  protocol_line_r4b: 'c7a1f67706c1c5cd02f143ca4f364238bbc082da361e9456d6cf1ebe6887d17c',
};
need(sha(`${REL}/dist/sim-core.js`) === EXPECT.sim_core, 'sim-core.js の sha256 が違います');
// 走行中の仕組みの場所。既定は候補の runtime/。CK4_RT を渡したとき(昇格の取り違えの試し: shield.mjs だけを別の版に替えた写し)だけ別の場所
export const RT = process.env.CK4_RT || `${CAND}/runtime`;
if (!process.env.CK4_RT) need(sha(`${RT}/shield.mjs`) === EXPECT.shield_mjs, '候補の shield.mjs の sha256 が違います');
need(sha(`${RT}/mode_runtime.mjs`) === EXPECT.mode_runtime, '候補の mode_runtime.mjs の sha256 が違います');
export const RT_SHIELD_SHA = sha(`${RT}/shield.mjs`);
need(sha(`${REP}/r3/protocol_line_r4b.json`) === EXPECT.protocol_line_r4b, 'protocol_line_r4b.json の sha256 が違います');

const core = await import(pathToFileURL(`${REL}/dist/sim-core.js`).href);
const {loadCourseWorld} = await import(pathToFileURL(`${REL}/qa/load-course-world.mjs`).href);
export const WORLD = await loadCourseWorld();
export const OB = buildObstacles(WORLD);
export const LOOP = loadLoop(`${CAND}/runtime/reference_loop.json`);
const MR = await import(pathToFileURL(`${RT}/mode_runtime.mjs`).href);

// 候補の mode.json の runtime(シールドの切り替えと値)が、固めた追補の値と一字一句同じこと
const MODE = JSON.parse(fs.readFileSync(`${CAND}/mode.json`, 'utf8'));
const PROT = JSON.parse(fs.readFileSync(`${REP}/r3/protocol_line_r4b.json`, 'utf8'));
need(MODE.runtime.shield === true, 'mode.json runtime.shield が true ではありません');
need(JSON.stringify(MODE.runtime.shield_params) === JSON.stringify(PROT.runtime.shield.mode_runtime.shield_params), 'mode.json の shield_params が追補 r4b と違います');
const FAST_MODE = JSON.parse(fs.readFileSync(`${EXP}/bundle/fastest/mode.json`, 'utf8'));

const POL = {line: [`${CAND}/policy.json`, EXPECT.policy_line, MODE.runtime], fastest: [`${EXP}/bundle/fastest/policy.json`, EXPECT.policy_fastest, FAST_MODE.runtime ?? null]};
const pcache = {};
function pol(k) {
  if (!pcache[k]) { const [p, want, rt] = POL[k], h = sha(p); need(h === want, `${k} の方策の sha256 が違います`); pcache[k] = {w: JSON.parse(fs.readFileSync(p, 'utf8')), sha: h, rt}; }
  return pcache[k];
}

const PN = JSON.parse(fs.readFileSync(`${REP}/modes.json`, 'utf8')).common.physics_nominal;
export const PHYS = {dt: PN.dt, maxSpeed: PN.maxSpeed, maxSteer: PN.maxSteer, steerRate: PN.steerRate, brakeDeceleration: PN.brakeDeceleration,
  driveTau: PN.driveTau, driveAccelerationCap: PN.driveAccelerationCap, frictionMultiplier: PN.frictionMultiplier, vehicleLength: PN.vehicleLength,
  vehicleWidth: PN.vehicleWidth, wheelbase: PN.wheelbase, longitudinalModel: 'real-drive-response-estimate-v1', stopOnCollision: true, noiseStd: 0};

// app.js の reset の開始(この段で書いた)
export function spawnAt(gate, offset, yawDeg, world = WORLD) {
  const n = world.route.length, g = Math.max(0, Math.min(n - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(g), ...world.route.slice(0, g)], first = route[0], next = route[1];
  const h = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...world, route, spawn: {x: first[0] + off * Math.sin(h), z: first[1] - off * Math.cos(h), yaw: h + turn}},
    variation: {gate: g, left_offset_m: off, left_yaw_deg: turn * 180 / Math.PI}};
}
// 開始の確かめ(開始のすき間・R に対する向きのずれ)
export function startInfo(gate, offset, yawDeg) {
  const {world} = spawnAt(gate, offset, yawDeg), p = world.spawn, pr = LOOP.proj(p.x, p.z);
  return {clearance: clearance(OB, p.x, p.z, p.yaw), dpsi: wrap(p.yaw - LOOP.heading(pr.s)), s: pr.s, ey: pr.ey, x: p.x, z: p.z, yaw: p.yaw};
}

// 自前の小区間の積分(sim-core.js の step を読んで書いた。当たり判定はしない)
function surfMu(x, z) { let mu = .8; for (const s of WORLD.surfaces ?? []) if (Math.abs(x - s.x) <= s.w / 2 && Math.abs(z - s.z) <= s.d / 2) mu = Number.isFinite(s.mu) ? s.mu : .8; return mu * PHYS.frictionMultiplier; }
const appr = (v, t, d) => v + Math.max(-d, Math.min(d, t - v));
function substeps(st, steer, speed, rem, cfg) {
  const vr = Math.hypot(cfg.vehicleLength, cfg.vehicleWidth) / 2;
  const rate = cfg.maxSpeed + vr * cfg.maxSpeed / cfg.wheelbase * Math.tan(cfg.maxSteer);
  const sp = Math.max(-cfg.maxSpeed, Math.min(cfg.maxSpeed, speed)), de = Math.max(-cfg.maxSteer, Math.min(cfg.maxSteer, steer));
  const poses = [];
  let t = 0;
  while (t < rem - 1e-12) {
    const h = Math.min(cfg.substep, .012 / rate, rem - t), mu = surfMu(st.x, st.z);
    const acc = st.speed >= 0 && sp > st.speed ? Math.min(cfg.driveAccelerationCap, mu * 9.80665, (sp - st.speed) / cfg.driveTau) : cfg.brakeDeceleration;
    const v = appr(st.speed, sp, acc * h), d = appr(st.steering, de, cfg.steerRate * h);
    const lim = mu * 9.80665 / Math.max(Math.abs(v), .01), w = Math.max(-lim, Math.min(lim, v / cfg.wheelbase * Math.tan(d)));
    const ym = st.yaw + w * h / 2;
    st.x += v * Math.cos(ym) * h; st.z -= v * Math.sin(ym) * h; st.yaw = wrap(st.yaw + w * h); st.speed = v; st.steering = d;
    poses.push([st.x, st.z, st.yaw]);
    t += h;
  }
  return poses;
}

// 障害物を太らせた world で同じ操作の列を流し直し、sim 自身の当たり判定で最初に当たったフレームを返す
const inflCache = {};
export function inflatedFirstHit(gate, offset, yawDeg, seed, cmds, margin) {
  if (!inflCache[margin]) inflCache[margin] = inflatedWorld(WORLD, margin);
  const {world} = spawnAt(gate, offset, yawDeg, inflCache[margin]);
  const sim = core.createSimulation(world, {...PHYS, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed});
  if (snap.state.terminal) return {frame: -1, reason: snap.state.reason};
  for (const [st, sp] of cmds) {
    snap = sim.step({steering: st, speed: sp});
    if (snap.metrics.collisions > 0) { const e = snap.metrics.collision_events[0]; return {frame: e.frame, t: +e.time.toFixed(4), id: e.collider}; }
    if (snap.state.terminal) break;
  }
  return null;
}

const BIN = .05;
export const NB = Math.round(LOOP.L / BIN);
// 絶対の s(mod L)の 5 cm のマスに、開始から 3 m 以降の最初の通過の値(マスの中心に一番近いフレームの線形補間)
function firstPassBins(sArr, vArr, s0) {
  const out = new Array(NB).fill(null);
  for (let k = 0; k + 1 < sArr.length; k++) {
    const a = sArr[k], b = sArr[k + 1];
    if (!(b > a) || a - s0 < 3) continue;
    for (let j = Math.ceil(a / BIN); j * BIN < b; j++) {
      const m = ((j % NB) + NB) % NB;
      if (out[m] === null) out[m] = Math.round((vArr[k] + (vArr[k + 1] - vArr[k]) * (j * BIN - a) / (b - a)) * 1e4) / 1e4;
    }
  }
  return out;
}

const JF_KEYS = new Set(['v', 'm', 'f', 'ph', 'l', 'ev', 'cl', 'es', 'sh', 'p']);
// job: {id, set, policy: 'line'|'fastest', shield: bool, gate, off, yaw, seed, line, c_v, keep, infl: [0.03, ...]}
export function runJob(job) {
  const pk = job.policy ?? 'line', P = pol(pk), mode = pk;
  const seed = job.seed >>> 0;
  const {world, variation} = spawnAt(job.gate, job.off, job.yaw);
  const sim = core.createSimulation(world, {...PHYS, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed, scenario: 'baseline'});
  const sp0 = world.spawn, pr0 = LOOP.proj(sp0.x, sp0.z);
  const head = {id: job.id, set: job.set, policy: pk, shield: !!job.shield, gate: variation.gate, off: variation.left_offset_m, yaw: variation.left_yaw_deg, seed,
    line: job.line ?? null, c_v: job.c_v ?? null,
    start: {c: +clearance(OB, sp0.x, sp0.z, sp0.yaw).toFixed(5), dpsi: +wrap(sp0.yaw - LOOP.heading(pr0.s)).toFixed(4), s: +pr0.s.toFixed(3), ey: +pr0.ey.toFixed(4)}};
  if (snap.state.terminal) return {...head, invalid_spawn: true, valid: false, valid_r3: false};
  const cfg = snap.config;
  const params = pk === 'line' ? {line: job.line, c_v: job.c_v} : {};
  const T = MR.createModeTeacher(world, JSON.parse(JSON.stringify(cfg)), variation, {mode, weights: P.w, policySha256: P.sha, params: JSON.parse(JSON.stringify(params)),
    modeRuntime: P.rt ? JSON.parse(JSON.stringify(P.rt)) : null, overrides: pk === 'line' && !job.shield ? {shield: false} : {}});
  const cV = T.params.c_v ?? 1;
  const fol = LOOP.follower();
  const own = {x: sp0.x, z: sp0.z, yaw: sp0.yaw, speed: 0, steering: 0};
  const lab = {n: 0, l_not1: 0, ev_not0: 0, sh_missing: 0, sh1: 0, sh_ph_mismatch: 0, es_missing: 0, cl_missing: 0, p0_bad: 0, p_later: 0, unknown_key: 0,
    bad_text: 0, over_cap: 0, steer_out: 0, nonfinite: 0, max_bytes: 0, neg_cmd: 0, p0: null};
  const F = [], cmds = [];
  let integMax = 0, minC = Infinity, minF = -1, minId = null;
  while (!snap.state.terminal) {
    const S = snap.state, f = S.frame;
    const o = T.action({state: S, metrics: snap.metrics, config: cfg});
    const jf = o.info.jf, txt = JSON.stringify(o.info), lg = T.log[T.log.length - 1];
    lab.n++;
    if (jf.l !== 1) lab.l_not1++;
    if (jf.ev !== 0) lab.ev_not0++;
    if (pk === 'line') {
      if (job.shield) { if (!('sh' in jf)) lab.sh_missing++; else { if (jf.sh === 1) lab.sh1++; if ((jf.sh === 1) !== (jf.ph === 'shield')) lab.sh_ph_mismatch++; } }
      if (!('es' in jf)) lab.es_missing++;
    }
    if (!('cl' in jf)) lab.cl_missing++;
    for (const k of Object.keys(jf)) if (!JF_KEYS.has(k)) lab.unknown_key++;
    if (f === 0) {
      const p = jf.p ?? {};
      lab.p0 = p;
      const ok = p.mode === undefined ? true : p.mode === mode;
      const okLine = pk !== 'line' || (JSON.stringify(p.line) === JSON.stringify(job.line) && p.c_v === job.c_v && p.line_profile === 'v2');
      if (!ok || !okLine || p.policy !== P.sha.slice(0, 12)) lab.p0_bad++;
    } else if ('p' in jf) lab.p_later++;
    if (/pwm|PWM|\/home\/|\\\\/.test(txt)) lab.bad_text++;
    lab.max_bytes = Math.max(lab.max_bytes, txt.length);
    if (o.speed > 4.5 * cV + 1e-9) lab.over_cap++;
    if (o.speed < 0) lab.neg_cmd++;
    if (Math.abs(o.steering) > cfg.maxSteer + 1e-12) lab.steer_out++;
    if (!Number.isFinite(o.speed) || !Number.isFinite(o.steering)) lab.nonfinite++;
    const pj = f === 0 ? fol.first(S.x, S.z) : fol.next(S.x, S.z);
    // 自前の積分(sim の状態に合わせ直さない。ずれは integMax に)
    const rem = Math.min(cfg.dt, cfg.timeLimit - S.sim_time);
    const b0 = [own.x, own.z, own.yaw];
    const poses = substeps(own, o.steering, o.speed, rem, cfg);
    cmds.push([o.steering, o.speed]);
    snap = sim.step({steering: o.steering, speed: o.speed});
    const near = nearList(OB, b0[0], b0[1], 1.0);
    let fm = Infinity, fid = null, prev = b0, pc = clearance(near, b0[0], b0[1], b0[2]);
    if (pc < fm) fm = pc;
    for (const q of poses) {
      const w = {}, c = clearance(near, q[0], q[1], q[2], w);
      if (Math.min(c, pc) < .08) for (const u of [1 / 3, 2 / 3]) {
        const w2 = {}, c2 = clearance(near, prev[0] + (q[0] - prev[0]) * u, prev[1] + (q[1] - prev[1]) * u, prev[2] + wrap(q[2] - prev[2]) * u, w2);
        if (c2 < fm) { fm = c2; fid = w2.id; }
      }
      if (c < fm) { fm = c; fid = w.id; }
      prev = q; pc = c;
    }
    const T2 = snap.state;
    if (snap.metrics.collisions === 0) integMax = Math.max(integMax, Math.hypot(T2.x - own.x, T2.z - own.z), Math.abs(wrap(T2.yaw - own.yaw)), Math.abs(T2.speed - own.speed));
    if (fm < minC) { minC = fm; minF = f; minId = fid; }
    F.push({f, x: S.x, z: S.z, yaw: S.yaw, v: S.speed, st: o.steering, sp: o.speed, s: pj.s, ey: pj.ey, es: lg.es, rt_ey: lg.ey, rt_s: lg.s,
      sh: jf.sh ?? 0, fb: lg.shield?.fb ?? null, c: fm, cid: fid});
  }
  const rep = sim.report(), m = rep.metrics, fin = rep.final_state;
  // 止まり: フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上続く
  let run = 0, stallFirst = null, stallLongest = 0;
  for (const r of F) { if (r.f >= 30 && Math.abs(r.v) < .2) { run++; if (run > stallLongest) stallLongest = run; if (run === 30 && stallFirst === null) stallFirst = r.f - 29; } else run = 0; }
  const laps = (m.lap_events ?? []).map(l => ({t: +l.elapsed.toFixed(4), gates: l.passed_gates, cross: l.required_crossing_events.length, col: l.collisions, f0: l.start_frame, f1: l.end_frame}));
  const validLaps = laps.filter(l => l.gates === 33 && l.cross === 1 && l.col === 0).length;
  const s0 = F[0].s;
  const trk = [], eyDiff = [];
  for (const r of F) {
    if (r.s - s0 < 3) continue;
    if (Number.isFinite(r.es)) trk.push(Math.abs(r.ey - r.es));
    if (Number.isFinite(r.rt_ey)) eyDiff.push(Math.abs(r.ey - r.rt_ey));
  }
  const q = (a, p) => { if (!a.length) return null; const b = Float64Array.from(a).sort(), i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
  const fb = {};
  for (const r of F) if (r.fb) fb[r.fb] = (fb[r.fb] ?? 0) + 1;
  const shF = F.filter(r => r.sh === 1).map(r => r.f);
  const escF = F.filter(r => r.fb === 'escape_fwd' || r.fb === 'escape_rev').length;
  const brakeF = F.filter(r => r.fb === 'brake_first').length;
  const revF = F.filter(r => r.v < -.01).length;
  const mf = F[Math.max(0, Math.min(minF, F.length - 1))];
  const res = {...head, rt_shield: RT_SHIELD_SHA.slice(0, 8), invalid_spawn: false, reason: fin.reason, frames: fin.frame, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: +m.collision_events[0].time.toFixed(4), frame: m.collision_events[0].frame, id: m.collision_events[0].collider} : null,
    laps, valid_laps: validLaps, lap1: laps[0]?.t ?? null, completed_laps: m.completed_laps ?? 0,
    min_c: +Math.min(minC, m.collisions > 0 ? 0 : Infinity).toFixed(6), min_c_own: +minC.toFixed(6), min_f: minF, min_id: minId, min_v: +mf.v.toFixed(3),
    min_s_mod: +(((mf.s % LOOP.L) + LOOP.L) % LOOP.L).toFixed(3), min_c_first3s: +Math.min(...F.filter(r => r.f < 90).map(r => r.c)).toFixed(6),
    stall: stallFirst !== null, stall_first: stallFirst, stall_longest: stallLongest,
    sh_frames: shF.length, sh_list: shF.slice(0, 20), sh_frac: shF.length / Math.max(1, F.length), fb, esc_frames: escF, brake_first_frames: brakeF, reverse_frames: revF,
    integ_err: integMax, s_travel: +(F[F.length - 1].s - s0).toFixed(3),
    trk: trk.length ? {n: trk.length, med: +q(trk, .5).toFixed(5), p95: +q(trk, .95).toFixed(5)} : null,
    trk_mm10: trk.map(x => Math.round(x * 1e4)),
    ey_vs_runtime_max: eyDiff.length ? +Math.max(...eyDiff).toFixed(6) : null,
    v_run: (() => { const a = F.filter(r => r.v > .3).map(r => r.v); return a.length ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(4) : 0; })(),
    labels: lab, meta_shield: T.metadata.shield ? {version: T.metadata.shield.version, select: T.metadata.shield.select, c_nominal: T.metadata.shield.c_nominal} : null,
    meta_overrides: T.metadata.overrides_node_test_only ?? null, meta_line_profile: T.metadata.line_profile ?? null,
    bins_ey: firstPassBins(F.map(r => r.s), F.map(r => r.ey), s0)};
  res.valid = fin.reason === 'collection_duration_complete' && fin.frame === 600 && m.collisions === 0;
  res.valid_r3 = res.valid && !res.stall && validLaps >= 1;
  // 3 cm(と 4 cm)の太らせた world で流し直す(sim 自身の当たり判定)
  res.infl = {};
  for (const mg of job.infl ?? [.03]) res.infl[mg] = inflatedFirstHit(job.gate, job.off, job.yaw, seed, cmds, mg);
  if (job.keep) res.fr = F.map(r => [r.f, +r.x.toFixed(4), +r.z.toFixed(4), +r.yaw.toFixed(4), +r.v.toFixed(3), +r.st.toFixed(4), +r.sp.toFixed(3),
    +(r.s - s0).toFixed(3), +r.ey.toFixed(4), Number.isFinite(r.es) ? +r.es.toFixed(4) : null, +r.c.toFixed(4), r.sh, r.fb, r.cid]);
  return res;
}
