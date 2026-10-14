// no_wall 確認 r3: 1 回を走らせて自前で判定する(2026-10-14、確認の段)。学習・評価・確認 r1/r2 の走らせ方(run_nw*.mjs・episode.mjs・eval_*.mjs)は使わない。
//   物理 = ユーザーの元の dist/sim-core.js(摩擦円の回だけ own_ck3.mjs の自前の置き換え。衝突は全障害物で判定)
//   操作 = 候補の一式 r3/candidates/no_wall_r3/runtime/mode_runtime.mjs(+ shield.mjs v4.1)と policy.json。シールドの値は候補の mode.json の runtime.shield_params(ブラウザと同じ渡し方)
//   物理のずれ = 遅延(n フレーム前の指令、最初は中立)・ゲイン(適用する指令に掛ける)・観測の前回の行動 = 適用した行動(modes.json dr_specs.conventions)
//   すき間・止まり・有効な周・ρ・自前の積分 = own_ck3.mjs(この段で書いたもの)
// 使い方(子): node run_ck3.mjs --jobs jobs.jsonl --shard k --nshards K --out rows.jsonl
import fs from 'node:fs';
import crypto from 'node:crypto';
import {core, coreFc, BASE_WORLD, startWorld, shapes, clearanceAt, integrateFrame, muAt, SIM_CORE_SHA} from './own_ck3.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const CAND = `${EXP}/r3/candidates/no_wall_r3`;
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const MODE_JSON = JSON.parse(fs.readFileSync(`${CAND}/mode.json`, 'utf8'));
export const SHIELD_PARAMS = MODE_JSON.runtime.shield_params;
const RTS = {}, POL = {};
async function rt(dir) { return RTS[dir] ??= await import(`${dir}/mode_runtime.mjs`); }
function pol(p) { return POL[p] ??= {w: JSON.parse(fs.readFileSync(p, 'utf8')), sha: sha(p)}; }
export const NOMINAL = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, longitudinalModel: 'real-drive-response-estimate-v1',
  driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2});
const G = 9.80665;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

export function startClearance(gate, off, yaw) {
  const {world} = startWorld(BASE_WORLD, gate, off, yaw);
  return clearanceAt(shapes(world), world.spawn.x, world.spawn.z, world.spawn.yaw, 1.0);
}

// spec: {id, set, mode('no_wall'|'fastest'), start: {gate, off, yaw}, phys: {...}|null, delay, sg, vg, fc, T, episode('continuous'|'single'),
//        shield(true/false), policy, rtDir, keepPath, scenario, shieldParams}
export async function runOne(spec) {
  const mode = spec.mode ?? 'no_wall';
  const RT = await rt(spec.rtDir ?? `${CAND}/runtime`);
  const P = pol(spec.policy ?? `${CAND}/policy.json`);
  const {world, variation} = startWorld(BASE_WORLD, spec.start.gate, spec.start.off, spec.start.yaw);
  const S = shapes(world);
  const T = spec.T ?? 20, episode = spec.episode ?? 'continuous', seed = (spec.seed ?? 1) >>> 0;
  const base = {...NOMINAL, seed, timeLimit: T, ...(episode === 'continuous' ? {episodeMode: 'continuous-laps-v1'} : {})};
  const fc = !!spec.fc;
  const cfg = {...base, ...(spec.phys ?? {}), ...(fc ? {ckFrictionCircle: true} : {})};
  const sim = (fc ? coreFc : core).createSimulation(world, cfg);
  let snap = sim.reset({seed, scenario: spec.scenario ?? 'baseline'});
  if (snap.state.terminal) return {id: spec.id, set: spec.set, invalid_spawn: true};
  const cfgN = core.createSimulation(world, base).snapshot().config;   // 教師に渡すのは標準の物理(今の物理を教えない)
  const opts = {mode, weights: P.w, policySha256: P.sha};
  if (mode === 'no_wall') { if (spec.shield === false) opts.overrides = {shield: false}; else opts.shieldParams = spec.shieldParams ?? SHIELD_PARAMS; }
  const teacher = RT.createModeTeacher(world, cfgN, variation, opts);
  const delay = spec.delay ?? 0, sg = spec.sg ?? 1, vg = spec.vg ?? 1;
  const q = [];
  let applied = spec.applied0 ?? [0, -1];   // 遅延の回のフレーム 0 の「前回の行動」(この段の既定は中立 = 速さ 0。学習の段の走らせ方は [0, 0])
  const L = {frames: 0, sh: 0, l0: 0, bad: 0, fb: {}, probe: 0, gov: 0, maxRunSh: 0, firstSh: null};
  let run = 0;
  const shFrames = [];
  const t0 = process.hrtime.bigint();
  while (!snap.state.terminal) {
    const out = teacher.action(snap, delay ? {applied} : null);
    const jf = out.info.jf, lg = teacher.log.at(-1);
    L.frames++;
    if (jf.l !== 1) L.l0++;
    if (lg.shield?.fb) { L.fbAll ??= {}; L.fbAll[lg.shield.fb] = (L.fbAll[lg.shield.fb] ?? 0) + 1; }   // 替えなかったフレームも(proposal_nominal・proposal_best = あきらめ)
    if (/steering_pwm|throttle_pwm|"\//.test(JSON.stringify(out.info))) L.bad++;
    if (jf.sh === 1) {
      L.sh++; run++; L.maxRunSh = Math.max(L.maxRunSh, run); if (L.firstSh === null) L.firstSh = snap.state.frame;
      const fb = lg.shield?.fb ?? 'none'; L.fb[fb] = (L.fb[fb] ?? 0) + 1;
      if (fb.includes('brake_probe')) L.probe++;
      if (fb.includes('governor')) L.gov++;
      shFrames.push(snap.state.frame);
    } else run = 0;
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { q.push(ap); ap = q.length > delay ? q.shift() : {steering: 0, speed: 0, a: [0, -1]}; applied = ap.a; }
    snap = sim.step({steering: ap.steering * sg, speed: ap.speed * vg});
  }
  const ctrlMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state, tr = rep.trace, acts = rep.actions;
  // ---- 自前の積分とすき間 ----
  const rc = rep.config;
  let prev = {x: world.spawn.x, z: world.spawn.z, yaw: wrap(world.spawn.yaw), speed: 0, steering: 0}, tPrev = 0;
  let minC = clearanceAt(S, prev.x, prev.z, prev.yaw), minF = 0, minKind = null, posErr = 0, overlap = 0;
  const startC = minC;
  const kOut = {};
  const perFrameMin = spec.keepPath ? [] : null;
  for (let i = 0; i < tr.length; i++) {
    const last = i === tr.length - 1 && fs_.reason !== 'collection_duration_complete';   // 衝突・完走・途中の終了はフレームの途中で止まる
    const poses = [];
    const rem = Math.min(rc.dt, rc.timeLimit - tPrev);
    const end = integrateFrame(S, rc, prev, acts[i], rem, fc, poses);
    if (!last) posErr = Math.max(posErr, Math.hypot(end.x - tr[i].x, end.z - tr[i].z), Math.abs(wrap(end.yaw - tr[i].yaw)));
    let fmin = Infinity, px = prev.x, pz = prev.z, pyaw = prev.yaw, pc = clearanceAt(S, px, pz, pyaw);
    for (const [x, z, yaw] of poses) {
      let c = clearanceAt(S, x, z, yaw, .3, kOut);
      let kind = kOut.kind;
      if (Math.min(c, pc) < .12) {   // 近いときは小区間の間も 4 点
        for (let k = 1; k < 5; k++) {
          const u = k / 5, ci = clearanceAt(S, px + (x - px) * u, pz + (z - pz) * u, pyaw + wrap(yaw - pyaw) * u, .3, kOut);
          if (ci < c) { c = ci; kind = kOut.kind; }
        }
      }
      if (c < fmin) fmin = c;
      if (!last && c < minC) { minC = c; minF = i + 1; minKind = kind; }
      px = x; pz = z; pyaw = yaw; pc = c;
    }
    if (!last && fmin <= 0) overlap++;
    if (perFrameMin) perFrameMin.push(+fmin.toFixed(4));
    // 次のフレームは sim の状態から(積分の差が積み重ならないように)
    prev = {x: tr[i].x, z: tr[i].z, yaw: tr[i].yaw, speed: tr[i].speed, steering: tr[i].steering}; tPrev = tr[i].time;
  }
  // ---- 止まり(definitions_added.stall)----
  let sRun = 0, sMax = 0, sFirst = null;
  for (let i = 0; i < tr.length; i++) {
    const f = i + 1;
    if (f >= 30 && Math.abs(tr[i].speed) < .2) { sRun++; if (sRun > sMax) sMax = sRun; if (sRun === 30 && sFirst === null) sFirst = f - 29; } else sRun = 0;
  }
  // ---- ρ(N6)----
  let n = 0, req1 = 0, comb1 = 0, reqMax = 0, combMax = 0;
  let pv = 0, py = world.spawn.yaw;
  for (const t of tr) {
    if (t.speed > .3) {
      const mu = muAt(S, t.x, t.z, rc.frictionMultiplier);
      const rr = t.speed * t.speed * Math.abs(Math.tan(t.steering)) / (.26 * mu * G);
      const al = (t.speed - pv) / rc.dt, alat = t.speed * wrap(t.yaw - py) / rc.dt, cc = Math.hypot(al, alat) / (mu * G);
      n++; if (rr > 1) req1++; if (cc > 1) comb1++; reqMax = Math.max(reqMax, rr); combMax = Math.max(combMax, cc);
    }
    pv = t.speed; py = t.yaw;
  }
  // ---- 周 ----
  const laps = (m.lap_events ?? []).map(l => ({el: +l.elapsed.toFixed(4), g: l.passed_gates, x: l.required_crossing_events.length, c: l.collisions, sf: l.start_frame, ef: l.end_frame,
    xin: l.required_crossing_events.every(e => { const hx = .21 * Math.abs(Math.cos(e.yaw)) + .095 * Math.abs(Math.sin(e.yaw)); return e.x - hx >= 1.26 - 1e-9 && e.x + hx <= 2.36 + 1e-9; })}));
  const validLaps = laps.filter(l => l.g === 33 && l.x === 1 && l.c === 0 && l.xin).length;
  const nF = Math.round(T * 30);
  const validEp = episode === 'continuous' && fs_.reason === 'collection_duration_complete' && fs_.frame === nF && m.collisions === 0;
  const stall = sMax >= 30;
  const res = {id: spec.id, set: spec.set, mode, start: spec.start, phys: spec.phys ?? null, delay, sg, vg, fc, T, episode, shield: spec.shield !== false,
    policy_sha: P.sha.slice(0, 12), reason: fs_.reason, frames: fs_.frame, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: +m.collision_events[0].time.toFixed(3), kind: m.collision_events[0].kind, id: m.collision_events[0].collider} : null,
    laps, valid_laps: validLaps, passed_gates: m.passed_gates, start_clearance: +startC.toFixed(5),
    min_clearance: minC, min_frame: minF, min_kind: minKind, pos_err: posErr, overlap_frames: overlap,
    stall, stall_max: sMax, stall_first: sFirst,
    rho: {n, req1, comb1, reqMax: +reqMax.toFixed(3), combMax: +combMax.toFixed(3)},
    sh: L.sh, shield_fraction: L.frames ? L.sh / L.frames : 0, fb: L.fb, fb_all: L.fbAll ?? {}, probe: L.probe, gov: L.gov, l0: L.l0, bad_keys: L.bad, max_run_sh: L.maxRunSh, first_sh: L.firstSh,
    lap_from_stop: episode === 'single' && fs_.reason === 'route_complete' && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1 ? m.elapsed : null,
    first_lap: laps[0]?.el ?? null,
    mean_speed: (() => { const v = tr.filter(t => t.speed > .3).map(t => t.speed); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0; })(),
    ms_per_frame: L.frames ? ctrlMs / L.frames : null,
    valid_episode: validEp, valid_episode_r3: validEp && !stall && validLaps >= (T >= 120 ? 5 : 1)};
  if (spec.keepPath) { res.path = tr.map(t => [+t.x.toFixed(4), +t.z.toFixed(4), +t.speed.toFixed(3)]); res.sh_frames = shFrames; res.frame_min = perFrameMin; }
  return res;
}

// ---- 子プロセス ----
if (process.argv[1] && process.argv[1].endsWith('run_ck3.mjs') && process.argv.includes('--jobs')) {
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
export {SIM_CORE_SHA};
