// 確認 line r3: 1 回を走らせる自前の走らせ方(2026-10-14、確認の段)。
// 学習・評価のコード(r3/line/eval_line_r3.mjs・episode_ln3.mjs・run_ln3.mjs・snapshot/rl/env_core.mjs・rl/episode.mjs)を使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs(ブラウザと同じ scene.js)。開始は dist/app.js reset の式を自分で写した
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル、sha256 aaeb2379…549e)。標準の物理(modes.json common.physics_nominal)
//   操作: 候補の走行中の仕組み(r3/line/rt/mode_runtime.mjs + shield.mjs。candidates/line_r3/shield.json の sha256 と照らす)+ 候補の policy.json。
//         シールドの値は candidates/line_r3/mode.json runtime.shield_params(protocol_line_r3.json runtime.shield.params と同じか確かめる)
//   すき間: ck_geom.mjs(自前の凸多角形の距離)。小区間は自前で積分し直し(フレームの終わりで sim の trace と照らす)、12 cm 未満なら間を 4 分割
//   s・e_y: ck_geom.mjs の自前の R への射影(参照ループの点はデータとして読む)
//   e*: ランタイムの値(info.jf.es と教師の log)と、確認 r2 の自前の目標の線 v2(rl/check_ln2/own_line_v2.mjs)の両方
import fs from 'node:fs';
import crypto from 'node:crypto';
import {buildObstacles, clearance, nearSet, sampledClearance, loadLoop} from './ck_geom.mjs';
import {createOwnV2} from '../../rl/check_ln2/own_line_v2.mjs';

export const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
export const CAND = `${EXP}/r3/candidates/line_r3`;
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export const SIM_CORE_SHA = sha(`${REL}/dist/sim-core.js`);
if (SIM_CORE_SHA !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('sim-core.js の sha256 が違います');
const core = await import(`${REL}/dist/sim-core.js`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const BASE_WORLD = await loadCourseWorld();
export const OB = buildObstacles(BASE_WORLD);

// 走行中の仕組み: r3 = 候補の固定の写し(r3/line/rt)、web = いま共有で配っている組(web/rl、line のシールドなし)
const RT_DIR = {r3: `${EXP}/r3/line/rt`, web: `${EXP}/web/rl`};
const SHIELD_JSON = JSON.parse(fs.readFileSync(`${CAND}/shield.json`, 'utf8'));
if (sha(`${RT_DIR.r3}/shield.mjs`) !== SHIELD_JSON.implementation.shield_mjs_sha256) throw new Error('shield.mjs の sha256 が shield.json と違います');
if (sha(`${RT_DIR.r3}/mode_runtime.mjs`) !== SHIELD_JSON.implementation.mode_runtime_sha256) throw new Error('mode_runtime.mjs の sha256 が shield.json と違います');
const RTM = {}, LPM = {};
for (const [k, d] of Object.entries(RT_DIR)) { RTM[k] = await import(`${d}/mode_runtime.mjs`); LPM[k] = await import(`${d}/line_profiles.mjs`); }
export const LOOP = loadLoop(`${RT_DIR.r3}/reference_loop.json`);

const MODE_JSON = JSON.parse(fs.readFileSync(`${CAND}/mode.json`, 'utf8'));
export const SHIELD_PARAMS = MODE_JSON.runtime.shield_params;
{
  const PROT = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_line_r3.json', 'utf8'));
  const a = JSON.stringify(PROT.runtime.shield.params), b = JSON.stringify(SHIELD_PARAMS), c = JSON.stringify(SHIELD_JSON.params);
  if (a !== b || a !== c) throw new Error(`シールドの値が protocol・mode.json・shield.json で違います: ${a} / ${b} / ${c}`);
  if (MODE_JSON.runtime.shield !== true) throw new Error('mode.json runtime.shield が true ではありません');
}
const POLICY = {line: `${CAND}/policy.json`, fastest: `${EXP}/bundle/fastest/policy.json`,
  line_r2: `${EXP}/candidates/line_r2/policy.json`};
const EXPECT = {line: '9f10c7b7a09bf27a13744bb6f66ba71184df7a967589dfc44504e6e3607de52d', fastest: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee',
  line_r2: '1ae6be0793914094dbabfbdc325905ab8af01f12ea3bc7b2638b05c6bd0167ae'};
const pcache = {};
function policyFor(k) {
  if (!pcache[k]) {
    const p = POLICY[k] ?? k, h = sha(p);
    if (EXPECT[k] && h !== EXPECT[k]) throw new Error(`${k} の方策の sha256 が違います: ${h}`);
    pcache[k] = {weights: JSON.parse(fs.readFileSync(p, 'utf8')), sha: h};
  }
  return pcache[k];
}
// modes.json common.physics_nominal(+ 縦の応答の式の名前。アプリの longitudinalConfig と同じ)
export const PHYS = Object.freeze({dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, steerRate: 1.2, brakeDeceleration: 1.2, driveTau: .4, driveAccelerationCap: 6,
  frictionMultiplier: 1, vehicleLength: .42, vehicleWidth: .19, wheelbase: .26, longitudinalModel: 'real-drive-response-estimate-v1', stopOnCollision: true, noiseStd: 0});
// dist/app.js の reset(開始ゲートで経路を回し、左へ offset、向きを turn)を自分で写したもの
export function spawnFor(world, gate, offset, yawDeg) {
  const g = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(g), ...world.route.slice(0, g)], p0 = route[0], p1 = route[1];
  const hd = Math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...world, route, spawn: {x: p0[0] + off * Math.sin(hd), z: p0[1] - off * Math.cos(hd), yaw: hd + turn}},
    variation: {gate: g, left_offset_m: off, left_yaw_deg: turn * 180 / Math.PI}};
}
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
export function startInfo(gate, offset, yawDeg) {
  const {world} = spawnFor(BASE_WORLD, gate, offset, yawDeg), sp = world.spawn;
  const tr = LOOP.tracker(), p = tr.start(sp.x, sp.z, sp.yaw);
  return {clearance: clearance(OB, sp.x, sp.z, sp.yaw), dpsi: wrap(sp.yaw - LOOP.psiAt(p.s)), s: p.s, ey: p.ey, x: sp.x, z: sp.z, yaw: sp.yaw};
}

// 自前の小区間の積分(sim-core.js step の文言の式。衝突は見ない)。st を書き換え、小区間の姿勢の列を返す
const G = 9.80665;
function surfaceMu(world, x, z, fm) {
  let mu = .8;
  for (const s of world.surfaces ?? []) if (Math.abs(x - s.x) <= s.w / 2 && Math.abs(z - s.z) <= s.d / 2) mu = Math.max(.01, Math.min(3, s.mu ?? .8));
  return mu * fm;
}
function integrateFrame(world, cfg, st, cmd, remaining) {
  const steer = Math.max(-cfg.maxSteer, Math.min(cfg.maxSteer, Number.isFinite(cmd.steering) ? cmd.steering : 0));
  const speed = Math.max(-cfg.maxSpeed, Math.min(cfg.maxSpeed, Number.isFinite(cmd.speed) ? cmd.speed : 0));
  const vr = Math.hypot(cfg.vehicleLength, cfg.vehicleWidth) / 2, maxRate = cfg.maxSpeed + vr * cfg.maxSpeed / cfg.wheelbase * Math.tan(cfg.maxSteer);
  const poses = [];
  let el = 0;
  while (el < remaining - 1e-12) {
    const h = Math.min(cfg.substep, .012 / maxRate, remaining - el);
    const mu = surfaceMu(world, st.x, st.z, cfg.frictionMultiplier);
    const rate = st.speed >= 0 && speed > st.speed ? Math.min(cfg.driveAccelerationCap, mu * G, (speed - st.speed) / cfg.driveTau) : cfg.brakeDeceleration;
    const v = st.speed + Math.max(-rate * h, Math.min(rate * h, speed - st.speed));
    const d = st.steering + Math.max(-cfg.steerRate * h, Math.min(cfg.steerRate * h, steer - st.steering));
    const lim = mu * G / Math.max(Math.abs(v), .01), w = Math.max(-lim, Math.min(lim, v / cfg.wheelbase * Math.tan(d)));
    const ym = st.yaw + w * h / 2;
    st.x += v * Math.cos(ym) * h; st.z -= v * Math.sin(ym) * h; st.yaw = wrap(st.yaw + w * h); st.speed = v; st.steering = d;
    poses.push({x: st.x, z: st.z, yaw: st.yaw});
    el += h;
  }
  return poses;
}
const q = (a, p) => { if (!a.length) return null; const b = Float64Array.from(a).sort(), i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
const r4 = x => (x == null || !Number.isFinite(x) ? null : +x.toFixed(4));
const BIN = .05, NBIN = Math.ceil(LOOP.L / BIN);
// 絶対の s mod L の 5 cm のマスに、開始から 3 m 以降の最初の通過の値(マスの左端を通る時の値を前後のフレームで線形補間)
export function firstPass(sArr, vArr, s0, skip = 3) {
  const out = new Array(NBIN).fill(null);
  for (let k = 0; k + 1 < sArr.length; k++) {
    const a = sArr[k], b = sArr[k + 1];
    if (a - s0 < skip || !(b > a)) continue;
    for (let n = Math.ceil(a / BIN); n * BIN < b; n++) {
      const j = ((n % NBIN) + NBIN) % NBIN;
      if (out[j] == null) out[j] = vArr[k] + (vArr[k + 1] - vArr[k]) * (n * BIN - a) / (b - a);
    }
  }
  return out;
}

// job: {id, set, policy: 'line'|'fastest'|'line_r2', rt: 'r3'|'web', shield: bool, start: {gate, left_offset_m, left_yaw_deg, seed}, params: {line, c_v}, family,
//       keepFrames, keepPath, sampleCheck}
export function runOne(job) {
  const pk = job.policy ?? 'line', P = policyFor(pk), mode = pk.startsWith('line') ? 'line' : 'fastest';
  const rtKey = job.rt ?? 'r3', RT = RTM[rtKey];
  const st0 = job.start, seed = (st0.seed ?? 1) >>> 0;
  const {world, variation} = spawnFor(BASE_WORLD, st0.gate, st0.left_offset_m, st0.left_yaw_deg);
  const sim = core.createSimulation(world, {...PHYS, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed, scenario: 'baseline'});
  const sp = world.spawn;
  const base = {id: job.id, set: job.set, policy: pk, policy_sha256: P.sha, rt: rtKey, shield_on: !!job.shield, start: st0, family: job.family ?? null};
  if (snap.state.terminal) return {...base, invalid_spawn: true};
  const cfg = snap.config;
  // シールド: r3 の固定の写しの切り替え(globalThis.JF_R3.modeShield.line)に mode.json の値を入れる。切るときは何も入れない
  globalThis.JF_R3 = job.shield && mode === 'line' ? {modeShield: {line: JSON.parse(JSON.stringify(SHIELD_PARAMS))}} : undefined;
  if (job.shield && rtKey !== 'r3') throw new Error('シールドは r3 の写しでだけかけられます');
  const teacher = RT.createModeTeacher(world, JSON.parse(JSON.stringify(cfg)), variation, {mode, weights: P.weights, policySha256: P.sha, params: job.params ?? {}});
  globalThis.JF_R3 = undefined;
  const tp = teacher.params, lineP = tp.line ?? null, cV = tp.c_v ?? null;
  const tr = LOOP.tracker();
  const own = {x: sp.x, z: sp.z, yaw: sp.yaw, speed: 0, steering: 0};
  const lab = {frames: 0, l_not1: 0, ph: {}, ev_nonzero: 0, sh_missing: 0, sh1: 0, sh_ph_mismatch: 0, es_missing: 0, cl_missing: 0, p0_ok: null, p_after0: 0,
    bad_keys: 0, extra_keys: new Set(), bytes_max: 0, over_cap: 0, steer_out: 0, cmd_nonfinite: 0, cl_vs_own_max: 0};
  const F = [];
  let posErr = 0, simT = 0, minC = Infinity, minF = -1, minWhat = null, samp = {n: 0, max_exact_minus_sample: -Infinity};
  const c0 = clearance(OB, sp.x, sp.z, sp.yaw);
  minC = c0; minF = 0;
  let collided = false;
  while (!snap.state.terminal) {
    const S = snap.state, f = S.frame;
    const out = teacher.action({state: S, metrics: snap.metrics, config: cfg});
    const jf = out.info.jf, js = JSON.stringify(out.info), lg = teacher.log.at(-1);
    // ラベル
    lab.frames++;
    if (jf.l !== 1) lab.l_not1++;
    lab.ph[jf.ph] = (lab.ph[jf.ph] ?? 0) + 1;
    if (jf.ev) lab.ev_nonzero++;
    if (mode === 'line') {
      if (job.shield) { if (!('sh' in jf)) lab.sh_missing++; else { if (jf.sh === 1) lab.sh1++; if ((jf.sh === 1) !== (jf.ph === 'shield')) lab.sh_ph_mismatch++; } }
      if (jf.es === undefined) lab.es_missing++;
    }
    if (jf.cl === undefined) lab.cl_missing++;
    for (const k of Object.keys(jf)) if (!['v', 'm', 'f', 'ph', 'l', 'ev', 'cl', 'es', 'sh', 'p'].includes(k)) lab.extra_keys.add(k);
    if (f === 0) {
      const p = jf.p ?? {};
      lab.p0_ok = p.mode === mode && p.seed === seed && JSON.stringify(p.line ?? null) === JSON.stringify(job.params?.line ?? p.line ?? null)
        && (job.params?.c_v == null || p.c_v === job.params.c_v) && p.policy === P.sha.slice(0, 12) && (mode !== 'line' || p.line_profile === 'v2');
    } else if (jf.p !== undefined) lab.p_after0++;
    if (/pwm|PWM|"\//.test(js)) lab.bad_keys++;
    lab.bytes_max = Math.max(lab.bytes_max, js.length);
    if (cV != null && out.speed > 4.5 * cV + 1e-9) lab.over_cap++;
    if (Math.abs(out.steering) > .52 + 1e-12) lab.steer_out++;
    if (!Number.isFinite(out.steering) || !Number.isFinite(out.speed)) lab.cmd_nonfinite++;
    // 自前の射影
    const pr = f === 0 ? tr.start(S.x, S.z, S.yaw) : tr.step(S.x, S.z);
    const fr = {f, x: S.x, z: S.z, yaw: S.yaw, v: S.speed, steer: out.steering, speed: out.speed, s: pr.s, ey: pr.ey, rt_s: lg.s, rt_ey: lg.ey,
      es_rt: lg.es ?? null, es_mm: jf.es ?? null, sh: jf.sh ?? 0, fb: lg.shield?.fb ?? null, cl_rt: lg.dmin, gates: snap.metrics.passed_gates, clr: Infinity};
    // 自前の小区間の積分とすき間(このフレームの動き)
    const remaining = Math.min(cfg.dt, cfg.timeLimit - simT);
    const before = {x: own.x, z: own.z, yaw: own.yaw};
    const poses = integrateFrame(world, cfg, own, out, remaining);
    snap = sim.step({steering: out.steering, speed: out.speed});
    simT = snap.state.sim_time;
    const near = nearSet(OB, before.x, before.z, 1.2);
    let prev = before, prevC = clearance(near, before.x, before.z, before.yaw), fmin = prevC;
    const hitThisFrame = snap.metrics.collisions > 0 && !collided;
    for (const ps of poses) {
      const w = {}, c = clearance(near, ps.x, ps.z, ps.yaw, w);
      if (Math.min(c, prevC) < .12) {
        for (let k = 1; k < 4; k++) {
          const u = k / 4, cy = clearance(near, prev.x + (ps.x - prev.x) * u, prev.z + (ps.z - prev.z) * u, prev.yaw + wrap(ps.yaw - prev.yaw) * u);
          if (cy < fmin) fmin = cy;
        }
      }
      if (c < fmin) fmin = c;
      if (c < minC) { minC = c; minF = f; minWhat = w; }
      if (job.sampleCheck && c < .06 && c > 0) { const sc = sampledClearance(near, ps.x, ps.z, ps.yaw); samp.n++; samp.max_exact_minus_sample = Math.max(samp.max_exact_minus_sample, c - sc); }
      prev = ps; prevC = c;
      if (hitThisFrame && c <= 0) break;
    }
    fr.clr = fmin;
    if (fmin < minC) { minC = fmin; minF = f; }
    // 自前の積分と sim の照らし合わせ(衝突のフレームは sim が当たる前の姿勢で止まるので除く)
    if (snap.metrics.collisions === 0) {
      const t = snap.state;
      posErr = Math.max(posErr, Math.hypot(t.x - own.x, t.z - own.z), Math.abs(wrap(t.yaw - own.yaw)), Math.abs(t.speed - own.speed));
      own.x = t.x; own.z = t.z; own.yaw = t.yaw; own.speed = t.speed; own.steering = t.steering;
    } else collided = true;
    // ランタイムの d_min(フレームの始めの姿勢、features.mjs)と自前のすき間(同じ姿勢)の差。近いとき(< 0.3 m)だけ
    { const cs = clearance(near, S.x, S.z, S.yaw); if (cs < .3 && Number.isFinite(lg.dmin)) lab.cl_vs_own_max = Math.max(lab.cl_vs_own_max, Math.abs(lg.dmin - cs)); }
    F.push(fr);
  }
  const rep = sim.report(), m = rep.metrics, fs_ = rep.final_state;
  if (m.collisions > 0) { minC = Math.min(minC, 0); }
  const s0 = F[0].s, e0 = F[0].ey, sArr = F.map(r => r.s);
  // 止まり: フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上続く
  let run = 0, stallLong = 0, stallFirst = null;
  for (const r of F) {
    if (r.f >= 30 && Math.abs(r.v) < .2) { run++; if (run >= 30 && stallFirst == null) stallFirst = r.f - 29; stallLong = Math.max(stallLong, run); } else run = 0;
  }
  // 周(sim が数えたもの)と、自前の s で見た周の長さ(ゲートのずる・近道が無いか)
  const laps = (m.lap_events ?? []).map(l => {
    const a = F[Math.min(l.start_frame, F.length - 1)]?.s, b = l.end_frame < F.length ? F[l.end_frame].s : F.at(-1).s;
    return {elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions, start_frame: l.start_frame, end_frame: l.end_frame,
      own_ds: b - a, cross_x: l.required_crossing_events.map(e => e.x)};
  });
  const validLaps = laps.filter(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0).length;
  // 黒シート: trace の z の符号の変わり目を自分で数え、車幅ぶんが開口(requiredCrossings の minX〜maxX)に入っているか
  const RC = (BASE_WORLD.requiredCrossings ?? [])[0];
  const cross = [];
  if (RC) {
    let pz = sp.z, px = sp.x, py = sp.yaw;
    for (const t of rep.trace) {
      const a = (pz - RC.z) * (RC.direction === 1 ? 1 : -1), b = (t.z - RC.z) * (RC.direction === 1 ? 1 : -1);
      if (a < 0 && b >= 0) {
        const u = -a / (b - a), x = px + (t.x - px) * u, y = py + wrap(t.yaw - py) * u, half = .21 * Math.abs(Math.cos(y)) + .095 * Math.abs(Math.sin(y));
        if (x > RC.minX - 1 && x < RC.maxX + 1) cross.push({frame: t.frame, x: +x.toFixed(4), inside: x - half >= RC.minX - 1e-9 && x + half <= RC.maxX + 1e-9});
      }
      pz = t.z; px = t.x; py = t.yaw;
    }
  }
  // e*: ランタイム(log)と自前 v2
  let ownLine = null;
  if (mode === 'line' && lineP) {
    if (tp.line_profile !== 'v2') throw new Error('line が v2 で動いていません');
    const cFn = lineP.family === 'random' ? LPM[rtKey].createLineProfile(lineP, s0, {version: 'v1'}).c : null;
    ownLine = createOwnV2(lineP, s0, e0, {cFn});
  }
  const trk = [], trkOwn = [], trkRtEy = [];
  let esDiff = 0, esMmDiff = 0, eyDiff = 0;
  for (const r of F) {
    if (r.es_rt != null && r.es_mm != null) esMmDiff = Math.max(esMmDiff, Math.abs(r.es_mm / 1000 - r.es_rt));
    if (r.s - s0 < 3) continue;
    eyDiff = Math.max(eyDiff, Math.abs(r.ey - r.rt_ey));
    if (r.es_rt != null) { trk.push(Math.abs(r.ey - r.es_rt)); trkRtEy.push(Math.abs(r.rt_ey - r.es_rt)); }
    if (ownLine) { const e = ownLine.eStar(r.s); r.es_own = e; trkOwn.push(Math.abs(r.ey - e)); if (r.es_rt != null) esDiff = Math.max(esDiff, Math.abs(e - r.es_rt)); }
  }
  const movingV = F.filter(r => r.v > .3).map(r => r.v);
  const shFrames = F.filter(r => r.sh === 1).map(r => r.f);
  const mf = F[Math.min(minF, F.length - 1)];
  const clrWhere = pred => { let v = Infinity; for (const r of F) if (pred(r)) v = Math.min(v, r.clr); return Number.isFinite(v) ? v : null; };
  const big = F.filter(r => r.s - s0 >= 3 && r.es_rt != null && Math.abs(r.ey - r.es_rt) > .12).map(r => +(((r.s % LOOP.L) + LOOP.L) % LOOP.L).toFixed(2));
  const res = {...base, invalid_spawn: false, variation, params: tp, c_v: cV, line: lineP,
    start_info: {clearance: c0, dpsi: wrap(sp.yaw - LOOP.psiAt(s0)), s_mod: ((s0 % LOOP.L) + LOOP.L) % LOOP.L, ey: e0},
    reason: fs_.reason, frames: fs_.frame, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: m.collision_events[0].time, frame: m.collision_events[0].frame, collider: m.collision_events[0].collider, kind: m.collision_events[0].kind,
      v: F[Math.min(m.collision_events[0].frame, F.length - 1)].v, s_mod: ((F[Math.min(m.collision_events[0].frame, F.length - 1)].s % LOOP.L) + LOOP.L) % LOOP.L} : null,
    laps, valid_laps: validLaps, crossings: cross,
    min_clearance: minC, min_frame: minF, min_t: minF / 30, min_what: minWhat, min_v: mf?.v, min_s_from_start: mf ? mf.s - s0 : null,
    min_s_mod: mf ? ((mf.s % LOOP.L) + LOOP.L) % LOOP.L : null, min_ey: mf?.ey, min_es: mf?.es_rt,
    clr_first_3s: clrWhere(r => r.f < 90), clr_after_3s: clrWhere(r => r.f >= 90), clr_after_3m: clrWhere(r => r.s - s0 >= 3),
    stall: stallFirst != null, stall_first: stallFirst, stall_longest: stallLong,
    shield_frames: shFrames.length, shield_at: shFrames.slice(0, 40), shield_fraction: shFrames.length / Math.max(1, F.length),
    shield_fb: F.reduce((a, r) => { if (r.fb) a[r.fb] = (a[r.fb] ?? 0) + 1; return a; }, {}),
    integ_err: posErr, sample_check: samp,
    tracking: trk.length ? {n: trk.length, median: q(trk, .5), p95: q(trk, .95), over12: trk.filter(x => x > .12).length} : null,
    trk_e4: trk.map(x => Math.round(x * 1e4)),
    tracking_own_line: trkOwn.length ? {n: trkOwn.length, median: q(trkOwn, .5), p95: q(trkOwn, .95)} : null,
    tracking_rt_ey: trkRtEy.length ? {median: q(trkRtEy, .5), p95: q(trkRtEy, .95)} : null,
    over12_s_mod: big, es_check: {own_vs_rt_max: esDiff, info_mm_vs_rt_max: esMmDiff}, ey_own_vs_rt_after3m_max: eyDiff,
    speed: {mean_running: movingV.length ? movingV.reduce((a, b) => a + b, 0) / movingV.length : 0, max: Math.max(0, ...F.map(r => r.v))},
    s_travel: F.at(-1).s - s0,
    labels: {...lab, extra_keys: [...lab.extra_keys], meta_line_profile: teacher.metadata.line_profile ?? null, meta_has_shield: !!teacher.metadata.shield}};
  res.valid_episode = fs_.reason === 'collection_duration_complete' && fs_.frame === 600 && m.collisions === 0;
  res.valid_r3 = res.valid_episode && !res.stall && validLaps >= 1;
  res.lap1 = laps[0]?.elapsed ?? null;
  res.bins = {ey: firstPass(sArr, F.map(r => r.ey), s0).map(r4), v: firstPass(sArr, F.map(r => r.v), s0).map(r4), steer: firstPass(sArr, F.map(r => r.steer), s0).map(r4)};
  if (job.keepFrames) res.fr = F.map(r => [r.f, r4(r.x), r4(r.z), r4(r.yaw), r4(r.v), r4(r.steer), r4(r.speed), r4(r.s - s0), r4(r.ey), r4(r.es_rt), r4(r.clr), r.sh, r4(r.rt_s - F[0].rt_s), r4(r.rt_ey)]);
  return res;
}
