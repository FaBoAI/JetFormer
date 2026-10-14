// 確認 line r3(再実行、2026-10-14 夜): 1 回を走らせる自前の走らせ方。
// 学習・評価のコード(r3/line/eval_line_r3.mjs・episode_ln3.mjs・run_ln3.mjs・rl/*)も、前の確認のコード(r3/ck_ln3/*)も使わない。
//   コース: ユーザーのリリースの qa/load-course-world.mjs(ブラウザと同じ scene.js から作る)
//   開始: dist/app.js の reset の 3 行(経路を開始ゲートで回し、左へ offset、向き +yaw)をここで書き直した spawn()
//   物理: ユーザーのリリースの dist/sim-core.js(元のファイル)。modes.json common.physics_nominal
//   操作: 候補そのもの = r3/line/rt の走行中の仕組み(shield.json の sha256 と照らす)+ 候補の policy.json + mode.json のシールドの値
//   すき間・射影・止まり・周・ラベル・追従: geom_b.mjs と、このファイルの自前の計算
import fs from 'node:fs';
import crypto from 'node:crypto';
import {obstaclesOf, nearOf, clr, loadR} from './geom_b.mjs';

export const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
export const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
export const CAND = `${EXP}/r3/candidates/line_r3`;
const RT = `${EXP}/r3/line/rt`;
export const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const must = (c, m) => { if (!c) throw new Error(m); };

must(sha(`${REL}/dist/sim-core.js`) === 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e', 'sim-core.js の sha256 が違います');
const core = await import(`${REL}/dist/sim-core.js`);
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
export const WORLD = await loadCourseWorld();
export const OB = obstaclesOf(WORLD);
export const R = loadR(`${RT}/reference_loop.json`);

// 候補の照らし合わせ
const SHJ = JSON.parse(fs.readFileSync(`${CAND}/shield.json`, 'utf8'));
must(sha(`${RT}/shield.mjs`) === SHJ.implementation.shield_mjs_sha256, 'shield.mjs の sha256 が shield.json と違います');
must(sha(`${RT}/mode_runtime.mjs`) === SHJ.implementation.mode_runtime_sha256, 'mode_runtime.mjs の sha256 が shield.json と違います');
const MODE = JSON.parse(fs.readFileSync(`${CAND}/mode.json`, 'utf8'));
const PROT = JSON.parse(fs.readFileSync(`${REP}/r3/protocol_line_r3.json`, 'utf8'));
must(sha(`${REP}/r3/protocol_line_r3.json`) === '25c853d778f0894ecfac97f5980c4f50986649536df66a48fa9de004fbb0b170', 'protocol_line_r3.json の sha256 が違います');
export const SHIELD = MODE.runtime.shield_params;
must(JSON.stringify(SHIELD) === JSON.stringify(PROT.runtime.shield.params) && JSON.stringify(SHIELD) === JSON.stringify(SHJ.params), 'シールドの値が protocol・mode.json・shield.json で違います');
must(MODE.runtime.shield === true, 'mode.json runtime.shield が true ではありません');
const MR = await import(`${RT}/mode_runtime.mjs`);

const POL = {
  line: [`${CAND}/policy.json`, '9f10c7b7a09bf27a13744bb6f66ba71184df7a967589dfc44504e6e3607de52d'],
  fastest: [`${EXP}/bundle/fastest/policy.json`, null],
};
const pc = {};
function policy(k) {
  if (!pc[k]) {
    const [p, want] = POL[k], h = sha(p);
    must(!want || h === want, `${k} の方策の sha256 が違います: ${h}`);
    pc[k] = {w: JSON.parse(fs.readFileSync(p, 'utf8')), sha: h};
  }
  return pc[k];
}

// 標準の物理(modes.json common.physics_nominal)。縦の応答の値があるので real-drive-response-estimate-v1 が要る
const MJ = JSON.parse(fs.readFileSync(`${REP}/modes.json`, 'utf8'));
const PN = MJ.common.physics_nominal;
export const PHYS = {dt: PN.dt, maxSpeed: PN.maxSpeed, maxSteer: PN.maxSteer, steerRate: PN.steerRate, brakeDeceleration: PN.brakeDeceleration,
  driveTau: PN.driveTau, driveAccelerationCap: PN.driveAccelerationCap, frictionMultiplier: PN.frictionMultiplier, vehicleLength: PN.vehicleLength,
  vehicleWidth: PN.vehicleWidth, wheelbase: PN.wheelbase, longitudinalModel: 'real-drive-response-estimate-v1', stopOnCollision: true, noiseStd: 0};

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
// app.js reset の開始(ここで書き直したもの)
export function spawn(gate, off, yawDeg) {
  const n = WORLD.route.length, g = Math.max(0, Math.min(n - 1, Math.trunc(gate) || 0));
  const route = WORLD.route.slice(g).concat(WORLD.route.slice(0, g));
  const a = route[0], b = route[1], h = Math.atan2(-(b[1] - a[1]), b[0] - a[0]);
  const o = Math.max(-.3, Math.min(.3, off || 0)), t = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {world: {...WORLD, route, spawn: {x: a[0] + o * Math.sin(h), z: a[1] - o * Math.cos(h), yaw: h + t}},
    variation: {gate: g, left_offset_m: o, left_yaw_deg: t * 180 / Math.PI}};
}
export function startOf(gate, off, yawDeg) {
  const {world} = spawn(gate, off, yawDeg), p = world.spawn, pr = R.globalProj(p.x, p.z);
  return {x: p.x, z: p.z, yaw: p.yaw, clearance: clr(OB, p.x, p.z, p.yaw), dpsi: wrap(p.yaw - R.psi(pr.s)), s: pr.s, ey: pr.ey};
}

// 自前の小区間の積分(sim-core.js step の式を読んで書いたもの。衝突は見ない)。st を進め、小区間の姿勢を返す
function muAt(x, z) {
  let mu = .8;
  for (const s of WORLD.surfaces) if (Math.abs(x - s.x) <= s.w / 2 && Math.abs(z - s.z) <= s.d / 2) mu = s.mu ?? .8;
  return mu * PHYS.frictionMultiplier;
}
const toward = (v, t, d) => v < t ? Math.min(t, v + d) : Math.max(t, v - d);
function integ(st, cmd, rem, cfg) {
  const vr = Math.hypot(cfg.vehicleLength, cfg.vehicleWidth) / 2;
  const mr = cfg.maxSpeed + vr * cfg.maxSpeed / cfg.wheelbase * Math.tan(cfg.maxSteer);
  const dmax = Math.min(.01, cfg.dt);
  const sp = Math.max(-cfg.maxSpeed, Math.min(cfg.maxSpeed, cmd.speed)), de = Math.max(-cfg.maxSteer, Math.min(cfg.maxSteer, cmd.steering));
  const out = [];
  let el = 0;
  while (el < rem - 1e-12) {
    const h = Math.min(dmax, .012 / mr, rem - el), mu = muAt(st.x, st.z);
    const rate = st.speed >= 0 && sp > st.speed ? Math.min(cfg.driveAccelerationCap, mu * 9.80665, (sp - st.speed) / cfg.driveTau) : cfg.brakeDeceleration;
    const v = toward(st.speed, sp, rate * h), d = toward(st.steering, de, cfg.steerRate * h);
    const raw = v / cfg.wheelbase * Math.tan(d), lim = mu * 9.80665 / Math.max(Math.abs(v), .01), w = Math.max(-lim, Math.min(lim, raw));
    const ym = st.yaw + w * h / 2;
    st.x += v * Math.cos(ym) * h; st.z -= v * Math.sin(ym) * h; st.yaw = wrap(st.yaw + w * h); st.speed = v; st.steering = d;
    out.push([st.x, st.z, st.yaw]);
    el += h;
  }
  return out;
}

const KEYS = new Set(['v', 'm', 'f', 'ph', 'l', 'ev', 'cl', 'es', 'sh', 'p']);
const BIN = .05;
export const NB = Math.ceil(R.L / BIN);
// 5 cm のマス(絶対の s mod L)に、開始から 3 m 以降の最初の通過の値
function bins(sA, vA, s0) {
  const out = new Array(NB).fill(null);
  for (let k = 0; k + 1 < sA.length; k++) {
    const a = sA[k], b = sA[k + 1];
    if (a - s0 < 3 || !(b > a)) continue;
    for (let j = Math.ceil(a / BIN); j * BIN < b; j++) {
      const m = ((j % NB) + NB) % NB;
      if (out[m] === null) out[m] = +(vA[k] + (vA[k + 1] - vA[k]) * (j * BIN - a) / (b - a)).toFixed(4);
    }
  }
  return out;
}
const qtl = (a, p) => { if (!a.length) return null; const b = [...a].sort((x, y) => x - y), i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };

// job: {id, set, policy: 'line'|'fastest', shield: bool, gate, off, yaw, seed, params: {line, c_v}, keep}
export function run(job) {
  const pk = job.policy ?? 'line', P = policy(pk), mode = pk === 'line' ? 'line' : 'fastest';
  const seed = (job.seed >>> 0);
  const {world, variation} = spawn(job.gate, job.off, job.yaw);
  const sim = core.createSimulation(world, {...PHYS, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed});
  const head = {id: job.id, set: job.set, policy: pk, shield: !!job.shield, gate: variation.gate, off: variation.left_offset_m, yaw: variation.left_yaw_deg, seed};
  const sp = world.spawn, pr0 = R.globalProj(sp.x, sp.z);
  head.start = {clearance: +clr(OB, sp.x, sp.z, sp.yaw).toFixed(5), dpsi: +wrap(sp.yaw - R.psi(pr0.s)).toFixed(4), s: +pr0.s.toFixed(3)};
  if (snap.state.terminal) return {...head, invalid_spawn: true};
  const cfg = snap.config;
  globalThis.JF_R3 = job.shield && mode === 'line' ? {modeShield: {line: JSON.parse(JSON.stringify(SHIELD))}} : undefined;
  const T = MR.createModeTeacher(world, JSON.parse(JSON.stringify(cfg)), variation, {mode, weights: P.w, policySha256: P.sha, params: JSON.parse(JSON.stringify(job.params ?? {}))});
  globalThis.JF_R3 = undefined;
  const tp = T.params, cV = tp.c_v ?? 1;
  const tr = R.tracker();
  const own = {x: sp.x, z: sp.z, yaw: sp.yaw, speed: 0, steering: 0};
  const lab = {n: 0, l_not1: 0, ev: 0, sh_missing: 0, sh1: 0, sh_ph_bad: 0, es_missing: 0, cl_missing: 0, p0_bad: 0, p_later: 0, extra: 0, bad_text: 0,
    over_cap: 0, steer_out: 0, nonfinite: 0, bytes_max: 0};
  const fr = [];
  let integErr = 0, minC = Infinity, minF = 0, minId = null, collided = false;
  while (!snap.state.terminal) {
    const S = snap.state, f = S.frame;
    const o = T.action({state: S, metrics: snap.metrics, config: cfg});
    const jf = o.info.jf, txt = JSON.stringify(o.info);
    // ラベル
    lab.n++;
    if (jf.l !== 1) lab.l_not1++;
    if (jf.ev) lab.ev++;
    if (mode === 'line') {
      if (job.shield) { if (!('sh' in jf)) lab.sh_missing++; else { if (jf.sh === 1) lab.sh1++; if ((jf.sh === 1) !== (jf.ph === 'shield')) lab.sh_ph_bad++; } }
      if (!('es' in jf)) lab.es_missing++;
    }
    if (!('cl' in jf)) lab.cl_missing++;
    if (Object.keys(jf).some(k => !KEYS.has(k))) lab.extra++;
    if (f === 0) {
      const p = jf.p ?? {};
      const ok = p.mode === mode && p.seed === seed && p.policy === P.sha.slice(0, 12) && (mode !== 'line' || (p.line_profile === 'v2'
        && JSON.stringify(p.line) === JSON.stringify(job.params.line) && p.c_v === job.params.c_v));
      if (!ok) lab.p0_bad++;
    } else if ('p' in jf) lab.p_later++;
    if (/pwm|PWM|\/home|\\\\/.test(txt)) lab.bad_text++;
    lab.bytes_max = Math.max(lab.bytes_max, txt.length);
    if (o.speed > 4.5 * cV + 1e-9) lab.over_cap++;
    if (Math.abs(o.steering) > .52 + 1e-12) lab.steer_out++;
    if (!Number.isFinite(o.speed) || !Number.isFinite(o.steering)) lab.nonfinite++;
    // 自前の射影
    const pj = f === 0 ? tr.start(S.x, S.z) : tr.step(S.x, S.z);
    const lg = T.log.at(-1);
    // 自前の積分とすき間
    const rem = Math.min(cfg.dt, cfg.timeLimit - S.sim_time);
    const b0 = [own.x, own.z, own.yaw];
    const poses = integ(own, o, rem, cfg);
    snap = sim.step({steering: o.steering, speed: o.speed});
    const near = nearOf(OB, b0[0], b0[1], 1.0);
    let prev = b0, pc_ = clr(near, b0[0], b0[1], b0[2]), fmin = pc_, fid = null;
    for (const q of poses) {
      const w = {}, c = clr(near, q[0], q[1], q[2], w);
      if (Math.min(c, pc_) < .1) {
        for (const u of [.25, .5, .75]) {
          const w2 = {}, c2 = clr(near, prev[0] + (q[0] - prev[0]) * u, prev[1] + (q[1] - prev[1]) * u, prev[2] + wrap(q[2] - prev[2]) * u, w2);
          if (c2 < fmin) { fmin = c2; fid = w2.id; }
        }
      }
      if (c < fmin) { fmin = c; fid = w.id; }
      prev = q; pc_ = c;
      if (c <= 0) break;
    }
    if (fmin < minC) { minC = fmin; minF = f; minId = fid; }
    // 照らすのは、フレームの途中で終わらなかったフレームだけ(周の数えでの終わり required_crossing_missing などは sim が途中で止める)
    if (snap.metrics.collisions === 0 && (!snap.state.terminal || snap.state.reason === 'collection_duration_complete')) {
      const t = snap.state;
      integErr = Math.max(integErr, Math.hypot(t.x - own.x, t.z - own.z), Math.abs(wrap(t.yaw - own.yaw)), Math.abs(t.speed - own.speed));
      own.x = t.x; own.z = t.z; own.yaw = t.yaw; own.speed = t.speed; own.steering = t.steering;
    } else collided = true;
    fr.push({f, x: S.x, z: S.z, yaw: S.yaw, v: S.speed, st: o.steering, sp: o.speed, s: pj.s, ey: pj.ey, es: jf.es != null ? jf.es / 1000 : null,
      rt_ey: lg.ey, sh: jf.sh ?? 0, fb: lg.shield?.fb ?? null, c: fmin});
  }
  const rep = sim.report(), m = rep.metrics, fin = rep.final_state;
  if (m.collisions > 0) minC = Math.min(minC, 0);
  // 止まり: フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上続く
  let run_ = 0, stallF = null, stallMax = 0;
  for (const r of fr) { if (r.f >= 30 && Math.abs(r.v) < .2) { run_++; stallMax = Math.max(stallMax, run_); if (run_ === 30 && stallF === null) stallF = r.f - 29; } else run_ = 0; }
  const laps = (m.lap_events ?? []).map(l => ({t: +l.elapsed.toFixed(4), gates: l.passed_gates, cross: l.required_crossing_events.length, col: l.collisions, f0: l.start_frame, f1: l.end_frame,
    ds: +((fr[Math.min(l.end_frame, fr.length - 1)]?.s ?? NaN) - (fr[Math.min(l.start_frame, fr.length - 1)]?.s ?? NaN)).toFixed(3)}));
  const vlaps = laps.filter(l => l.gates === 33 && l.cross >= 1 && l.col === 0).length;
  // 追従(開始から 3 m 以降。e* はランタイムの値)
  const s0 = fr[0].s, trk = [], eyd = [];
  for (const r of fr) {
    if (r.s - s0 < 3) continue;
    if (r.es != null) trk.push(Math.abs(r.ey - r.es));
    if (Number.isFinite(r.rt_ey)) eyd.push(Math.abs(r.ey - r.rt_ey));
  }
  const sA = fr.map(r => r.s);
  const shF = fr.filter(r => r.sh === 1).map(r => r.f);
  const fb = {};
  for (const r of fr) if (r.fb) fb[r.fb] = (fb[r.fb] ?? 0) + 1;
  const mfr = fr[Math.min(minF, fr.length - 1)];
  const res = {...head, invalid_spawn: false, params: {line: tp.line ?? null, c_v: tp.c_v ?? null},
    reason: fin.reason, frames: fin.frame, collisions: m.collisions,
    collision: m.collision_events?.[0] ? {t: +m.collision_events[0].time.toFixed(4), frame: m.collision_events[0].frame, id: m.collision_events[0].collider,
      v: +fr[Math.min(m.collision_events[0].frame, fr.length - 1)].v.toFixed(3), s_mod: +(((fr[Math.min(m.collision_events[0].frame, fr.length - 1)].s % R.L) + R.L) % R.L).toFixed(3)} : null,
    laps, valid_laps: vlaps, lap1: laps[0]?.t ?? null,
    min_c: +minC.toFixed(6), min_f: minF, min_id: minId, min_v: mfr ? +mfr.v.toFixed(3) : null, min_s_mod: mfr ? +(((mfr.s % R.L) + R.L) % R.L).toFixed(3) : null,
    min_c_first3s: +Math.min(...fr.filter(r => r.f < 90).map(r => r.c)).toFixed(6),
    stall: stallF !== null, stall_first: stallF, stall_max: stallMax,
    sh_frames: shF.length, sh_first: shF.slice(0, 12), sh_frac: shF.length / Math.max(1, fr.length), fb,
    integ_err: integErr, s_travel: +(fr.at(-1).s - s0).toFixed(3),
    trk: trk.length ? {n: trk.length, med: +qtl(trk, .5).toFixed(5), p95: +qtl(trk, .95).toFixed(5)} : null,
    trk_all: trk.map(x => Math.round(x * 1e4)),
    ey_vs_rt_max: eyd.length ? +Math.max(...eyd).toFixed(6) : null,
    v_run: (() => { const a = fr.filter(r => r.v > .3).map(r => r.v); return a.length ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(4) : 0; })(),
    labels: lab, meta_shield: !!T.metadata.shield, meta_line_profile: T.metadata.line_profile ?? null,
    bins_ey: bins(sA, fr.map(r => r.ey), s0)};
  res.valid = fin.reason === 'collection_duration_complete' && fin.frame === 600 && m.collisions === 0;
  res.valid_r3 = res.valid && !res.stall && vlaps >= 1;
  if (job.keep) res.fr = fr.map(r => [r.f, +r.x.toFixed(4), +r.z.toFixed(4), +r.yaw.toFixed(4), +r.v.toFixed(3), +r.st.toFixed(4), +r.sp.toFixed(3), +(r.s - s0).toFixed(3),
    +r.ey.toFixed(4), r.es, +r.c.toFixed(4), r.sh, r.fb]);
  return res;
}
