// no_wall 確認 r4: 1 回を走らせて自前で判定する(2026-10-14/15、確認の段)。学習・評価の走らせ方(r3/nw4/run_nw4r.mjs・run_sets_nw4.mjs・judge_nw4.py)と
// 確認 r3 の走らせ方(r3/ck_nw3/run_ck3.mjs)は使わない。
//   物理 = own_ck4.mjs の coreH(ユーザーの元の sim-core.js + 小区間の姿勢の引っかけ + 摩擦円)。摩擦円でない回は元の sim-core.js を同時に動かして毎フレーム比べる
//   操作 = 候補の一式 r3/candidates/no_wall_r4/runtime/mode_runtime.mjs(shield.mjs v5)と policy.json。シールドの値は**渡さない**
//          (ブラウザの teacher.js と同じ = shield.mjs の既定。既定が mode.json・決まりの値と同じことは params_eq_ck4.mjs で確かめる)
//   物理のずれ = 遅延(n フレーム前の指令、最初は中立 = 速さ 0)・ゲイン(適用する指令に掛ける)・観測の前回の行動 = 適用した行動(modes.json dr_specs.conventions)
//   ラベルの出どころ = 影の教師(シールドなし、同じ観測)で方策そのものの指令を作り、sh = 0 のフレームの指令が方策の指令と同じかを確かめる
// 使い方(子): node run_ck4.mjs --jobs jobs.jsonl --shard k --nshards K --out rows.jsonl
import fs from 'node:fs';
import crypto from 'node:crypto';
import {coreO, coreH, BASE_WORLD, makeStart, obstacles, clearOwn, muOf, NOMINAL_CFG, SIM_CORE_SHA, CORE_H_SHA} from './own_ck4.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
export const CAND = `${EXP}/r3/candidates/no_wall_r4`;
export const FAST = {rt: `${EXP}/web/rl`, policy: `${EXP}/bundle/fastest/policy.json`};
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const RTS = {}, POL = {};
async function rt(dir) { return RTS[dir] ??= await import(`${dir}/mode_runtime.mjs`); }
function pol(p) { return POL[p] ??= {w: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p)}; }
const G = 9.80665;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const FORBID = /steering_pwm|throttle_pwm|"\//;

// spec: {id, set, mode('no_wall'|'fastest'), start: {gate, off, yaw}, phys: {steerRate, brakeDeceleration, driveTau, driveAccelerationCap, frictionMultiplier}|null,
//        delay, sg, vg, fc, T, episode('continuous'|'single'), shield(true/false), seed, keepPath, shadow(true/false), shieldParams(試験だけ)}
export async function runOne(spec) {
  const mode = spec.mode ?? 'no_wall';
  const rtDir = spec.rtDir ?? (mode === 'fastest' ? FAST.rt : `${CAND}/runtime`);
  const RT = await rt(rtDir);
  const P = pol(spec.policy ?? (mode === 'fastest' ? FAST.policy : `${CAND}/policy.json`));
  const {world, variation} = makeStart(spec.start.gate, spec.start.off, spec.start.yaw);
  const O = obstacles(world);
  const T = spec.T ?? 20, episode = spec.episode ?? 'continuous', seed = (spec.seed ?? 1) >>> 0;
  const cfgBase = {...NOMINAL_CFG, seed, timeLimit: T, ...(episode === 'continuous' ? {episodeMode: 'continuous-laps-v1'} : {})};
  const cfgRun = {...cfgBase, ...(spec.phys ?? {})};
  const fc = !!spec.fc;
  const simH = coreH.createSimulation(world, fc ? {...cfgRun, __ck4fc: true} : cfgRun);
  const simO = fc ? null : coreO.createSimulation(world, cfgRun);
  let snap = simH.reset({seed, scenario: 'baseline'});
  if (simO) simO.reset({seed, scenario: 'baseline'});
  const startC = clearOwn(O, world.spawn.x, world.spawn.z, world.spawn.yaw, 2.0);
  if (snap.state.terminal) return {id: spec.id, set: spec.set, invalid_spawn: true, start: spec.start, start_clearance: startC};
  const teacherCfg = coreO.createSimulation(world, cfgBase).snapshot().config;   // ブラウザ: sim.snapshot().config(標準の物理。今の物理は教えない)
  const opts = {mode, weights: P.w, policySha256: P.sha};
  if (mode === 'no_wall' && spec.shield === false) opts.overrides = {shield: false};
  if (mode === 'no_wall' && spec.shieldParams) opts.shieldParams = spec.shieldParams;
  const teacher = RT.createModeTeacher(world, teacherCfg, variation, opts);
  const shadow = mode === 'no_wall' && spec.shadow !== false && spec.shield !== false
    ? RT.createModeTeacher(world, teacherCfg, variation, {mode, weights: P.w, policySha256: P.sha, overrides: {shield: false}}) : null;
  const delay = spec.delay ?? 0, sg = spec.sg ?? 1, vg = spec.vg ?? 1;
  const q = [];
  let applied = spec.applied0 ?? [0, -1];   // 遅延の回のフレーム 0 の「前回の行動」= 中立(速さ 0)。この段の約束(試験の作り直しだけ学習の段の約束 [0, 0])
  let lastA = [0, 0];             // 遅延なし: 教師が観測に使う前回の行動(教師の log の a0・a1)
  const poses = [];
  coreH.__ck4SetPose((x, z, y) => { poses.push(x, z, y); });
  const L = {frames: 0, sh: 0, l0: 0, bad: 0, fb: {}, esc: 0, escFrames: 0, probe: 0, gov: 0, revCmd: 0, shMismatch: 0, shNoChange: 0, lockMismatch: 0, maxRunSh: 0};
  const ms = [];
  let run = 0, prevFb = '';
  let minC = startC, minF = 0, minKind = null, overlap = 0;
  const who = {};
  let px = world.spawn.x, pz = world.spawn.z, pyaw = world.spawn.yaw, pc = startC;
  const trace = [];
  const path = spec.keepPath ? [] : null;
  const diag = spec.diag ? [] : null;   // diag_ck4.mjs だけ: フレームごとのシールドの内訳
  while (!snap.state.terminal) {
    const f = snap.state.frame;
    const t0 = process.hrtime.bigint();
    const out = teacher.action(snap, delay ? {applied} : null);
    ms.push(Number(process.hrtime.bigint() - t0) / 1e6);
    const jf = out.info.jf, lg = teacher.log.at(-1);
    L.frames++;
    if (jf.l !== 1) L.l0++;
    if (FORBID.test(JSON.stringify(out.info))) L.bad++;
    const fb = lg.shield?.fb ?? '';
    if (jf.sh === 1) {
      L.sh++; run++; L.maxRunSh = Math.max(L.maxRunSh, run);
      L.fb[fb || 'none'] = (L.fb[fb || 'none'] ?? 0) + 1;
      if (fb.includes('brake_probe')) L.probe++;
      if (fb.includes('governor')) L.gov++;
    } else run = 0;
    if (fb.startsWith('escape_')) { L.escFrames++; if (!prevFb.startsWith('escape_')) L.esc++; }
    prevFb = fb;
    if (out.speed < 0) L.revCmd++;
    // ラベルの出どころ: 影の教師(シールドなし)に同じ「前回の行動」を渡す → 方策そのものの指令
    if (shadow) {
      const so = shadow.action(snap, {applied: delay ? applied : (f === 0 ? [0, 0] : lastA)});
      const same = so.steering === out.steering && so.speed === out.speed;
      if (jf.sh !== 1 && !same) L.shMismatch++;
      if (jf.sh === 1 && same) L.shNoChange++;
    }
    lastA = [lg.a0, lg.a1];
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { q.push(ap); ap = q.length > delay ? q.shift() : {steering: 0, speed: 0, a: [0, -1]}; applied = ap.a; }
    const act = {steering: ap.steering * sg, speed: ap.speed * vg};
    poses.length = 0;
    snap = simH.step(act);
    if (simO) {
      const so = simO.step(act).state, s = snap.state;
      if (so.x !== s.x || so.z !== s.z || so.yaw !== s.yaw || so.speed !== s.speed || so.steering !== s.steering || so.terminal !== s.terminal || so.reason !== s.reason) L.lockMismatch++;
    }
    // すき間: 受け入れた小区間の姿勢と、近いときはその間(4 点)
    let fmin = Infinity;
    for (let k = 0; k < poses.length; k += 3) {
      const x = poses[k], z = poses[k + 1], yaw = poses[k + 2];
      let c = clearOwn(O, x, z, yaw, .4, who), kind = who.kind;
      if (Math.min(c, pc) < .12) {
        for (let j = 1; j < 5; j++) {
          const u = j / 5, ci = clearOwn(O, px + (x - px) * u, pz + (z - pz) * u, pyaw + wrap(yaw - pyaw) * u, .4, who);
          if (ci < c) { c = ci; kind = who.kind; }
        }
      }
      if (c < fmin) fmin = c;
      if (c < minC) { minC = c; minF = f + 1; minKind = kind; }
      px = x; pz = z; pyaw = yaw; pc = c;
    }
    if (fmin <= 0) overlap++;
    const s = snap.state;
    trace.push([s.frame, s.x, s.z, s.yaw, s.speed, s.steering, s.surface]);
    if (path) path.push([+s.x.toFixed(4), +s.z.toFixed(4), +s.speed.toFixed(3), jf.sh ?? 0]);
    if (diag) diag.push({yaw: s.yaw, st: +s.steering.toFixed(3), cmd: [+out.steering.toFixed(3), +out.speed.toFixed(3)], fb, nom: lg.shield?.nom ?? null, wq: lg.shield?.wq ?? null, pess: lg.shield?.pess ?? null, id: lg.shield?.id ? {sr: +lg.shield.id.sr.toFixed(3), d: lg.shield.id.d, b: +lg.shield.id.b.toFixed(3)} : null, ms: +ms.at(-1).toFixed(1)});
  }
  coreH.__ck4SetPose(null);
  const rep = simH.report(), m = rep.metrics, fsx = rep.final_state;
  // 止まり(protocol_no_wall_r3.json definitions_added.stall): フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上続く
  let sRun = 0, sMax = 0, sFirst = null, revFrames = 0;
  for (const [fr, , , , v] of trace) {
    if (v < -.01) revFrames++;
    if (fr >= 30 && Math.abs(v) < .2) { sRun++; if (sRun > sMax) sMax = sRun; if (sRun === 30 && sFirst === null) sFirst = fr - 29; } else sRun = 0;
  }
  // ρ(protocol definitions rho_req・rho_comb)
  let n = 0, req1 = 0, comb1 = 0, reqMax = 0, combMax = 0, pv = 0, py = world.spawn.yaw;
  for (const [, x, z, yaw, v, d] of trace) {
    if (v > .3) {
      const mu = muOf(O, x, z, cfgRun.frictionMultiplier);
      const rr = v * v * Math.abs(Math.tan(d)) / (.26 * mu * G);
      const aL = (v - pv) / cfgRun.dt, aT = v * wrap(yaw - py) / cfgRun.dt, cc = Math.hypot(aL, aT) / (mu * G);
      n++; if (rr > 1) req1++; if (cc > 1) comb1++; if (rr > reqMax) reqMax = rr; if (cc > combMax) combMax = cc;
    }
    pv = v; py = yaw;
  }
  // 周(sim が数えたもの + 黒シートの横断が車幅ぶんの開口の中か)
  const laps = (m.lap_events ?? []).map(l => ({el: +l.elapsed.toFixed(4), g: l.passed_gates, x: l.required_crossing_events.length, c: l.collisions,
    sf: l.start_frame, ef: l.end_frame, inside: l.required_crossing_events.every(e => {
      const hx = .21 * Math.abs(Math.cos(e.yaw)) + .095 * Math.abs(Math.sin(e.yaw)); return e.x - hx >= 1.26 - 1e-9 && e.x + hx <= 2.36 + 1e-9; })}));
  const validLaps = laps.filter(l => l.g === 33 && l.x === 1 && l.c === 0 && l.inside).length;
  const nF = Math.round(T * 30);
  const validEp = episode === 'continuous' && fsx.reason === 'collection_duration_complete' && fsx.frame === nF && m.collisions === 0;
  const stall = sMax >= 30;
  const msS = ms.slice().sort((a, b) => a - b), pct = p => msS.length ? msS[Math.min(msS.length - 1, Math.floor(p * msS.length))] : null;
  const res = {id: spec.id, set: spec.set, mode, start: spec.start, phys: spec.phys ?? null, delay, sg, vg, fc, T, episode, shield: mode === 'no_wall' && spec.shield !== false,
    seed, policy_sha: P.sha.slice(0, 12), reason: fsx.reason, frames: fsx.frame, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: +m.collision_events[0].time.toFixed(3), f: m.collision_events[0].frame, kind: m.collision_events[0].kind, id: m.collision_events[0].collider} : null,
    laps, valid_laps: validLaps, passed_gates: m.passed_gates, start_clearance: +startC.toFixed(5),
    min_clearance: minC, min_frame: minF, min_kind: minKind, overlap_frames: overlap, lock_mismatch: L.lockMismatch,
    stall, stall_max: sMax, stall_first: sFirst, reverse_frames: revFrames, reverse_cmd_frames: L.revCmd,
    rho: {n, req1, comb1, reqMax: +reqMax.toFixed(3), combMax: +combMax.toFixed(3)},
    sh: L.sh, shield_fraction: L.frames ? L.sh / L.frames : 0, fb: L.fb, probe: L.probe, gov: L.gov, escapes: L.esc, escape_frames: L.escFrames, max_run_sh: L.maxRunSh,
    l0: L.l0, bad_keys: L.bad, label_mismatch_sh0: L.shMismatch, label_sh1_unchanged: L.shNoChange, shadow: !!shadow,
    lap_from_stop: episode === 'single' && fsx.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1 ? m.elapsed : null,
    first_lap: laps[0]?.el ?? null,
    mean_speed: (() => { const v = trace.filter(t => t[4] > .3).map(t => t[4]); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0; })(),
    ms_p50: pct(.5), ms_p99: pct(.99), ms_max: msS.at(-1) ?? null, ms_over33: ms.filter(x => x > 33).length, ms_mean: ms.length ? ms.reduce((a, b) => a + b, 0) / ms.length : null,
    valid_episode: validEp, valid_episode_r3: validEp && !stall && validLaps >= (T >= 120 ? 5 : 1)};
  if (path) res.path = path;
  if (diag) res.diag = diag;
  return res;
}

// ---- 子プロセス(続きから: out にある id は飛ばす) ----
if (process.argv[1] && process.argv[1].endsWith('run_ck4.mjs') && process.argv.includes('--jobs')) {
  const arg = k => { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : null; };
  const jobs = fs.readFileSync(arg('--jobs'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
  const shard = +arg('--shard'), K = +arg('--nshards'), out = arg('--out');
  const done = new Set();
  if (fs.existsSync(out)) for (const l of fs.readFileSync(out, 'utf8').split('\n')) if (l.trim()) { try { done.add(JSON.parse(l).id); } catch {} }
  for (let i = 0; i < jobs.length; i++) {
    if (i % K !== shard || done.has(jobs[i].id)) continue;
    const t = Date.now();
    let r;
    try { r = await runOne(jobs[i]); } catch (e) { r = {id: jobs[i].id, set: jobs[i].set, error: String(e?.stack ?? e)}; }
    r.wall_s = (Date.now() - t) / 1000;
    fs.appendFileSync(out, JSON.stringify(r) + '\n');
  }
}
export {SIM_CORE_SHA, CORE_H_SHA};
