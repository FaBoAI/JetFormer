// RL のベストラップの反証の確認(r2)の共通部品(2026-10-13、確認の段 r2)
//
// 方針: RL の env(rl/env_core.mjs)・RL の評価(rl/r2/*.mjs・rl/eval_sets.mjs・rl/lap_audit.mjs)・確認 r1 の道具(check_r1/*)は使わない。
//   - 物理: ユーザーの展開物の dist/sim-core.js をそのまま import する(コピーではなく元のファイル。変えない)
//   - 制御遅延: ユーザーの dist/control-delay.js の createControlDelayQueue(ブラウザの待ち行列そのもの。最初は中立)
//   - コース: ユーザーの qa/load-course-world.mjs(scene.js から作る現行 v3)
//   - 設定・開始: app.js reset() の式をここで書き直す(収集の計画 qa/rc-collection-plan-60.json の値から)
//   - 方策: policy.json の重みを自前の MLP で計算する
//   - 観測: 方策の入力の定義そのものなので rl/features.mjs だけは使う(sha256 を固定して、r2 で変わっていないことを確かめる)
//   - すき間: 自前の厳密な距離(車体の長方形と障害物、フレームごと)と、sim の衝突判定に「太らせた車体」を当てる小区間ごとの上下限
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

export const USER_SIM = '/home/jetson/sim/extracted/course-simulator';
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap';
export const REPORTS = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013';
export const OUT = REPORTS + '/rl_check_r2';
export const POLICIES = Object.freeze({
  selected: {path: REPORTS + '/rl/best/main/policy.json', sha: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee'},
  r2_5M: {path: REPORTS + '/rl/r2/dr/policy.json', sha: '5f6f57168217b7355d2015633d36c6cc50ddbc0d7a0c50ea3f34aeabbf8155a5'},
  r2_1M: {path: REPORTS + '/rl/r2-dr-s11/policies/step_001015808.json', sha: '53b9e242f1c921668bb9d25de6430b68105101942e24933d2502d21a9ef627a7'},
  r2_2M: {path: REPORTS + '/rl/r2-dr-s11/policies/step_002015232.json', sha: '62690e98a8715f8d1163e0262641341106bbae4e150a0246fe47bdb2f9431932'},
  r2_3M: {path: REPORTS + '/rl/r2-dr-s11/policies/step_003014656.json', sha: '083df2643ce9a6fb6d81a10376571be10040bab44dd8fb440a7070663906391f'},
  r2_4M: {path: REPORTS + '/rl/r2-dr-s11/policies/step_004014080.json', sha: '1a2479709e36c739d3646773bfc69b01e545e68c94bfd857da58a830ea64452d'},
});
export const EXPECTED = Object.freeze({
  sim_core: 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e',
  control_delay: 'eb405f7e87ce9581c3a5ccc2850dce67d21750ad49e930a06a972a4aaf72f23c',
  rc_teacher: '4a32ab56369b3d0c9a76f16fe7144eb895677b72ceeefc8483645f91ae96f417',
  scene: '878fc934af2890c4db152041182b8914033224f65a2482bcc46265cdc7d0c806',
  verify_learned_lap: '506674aa160f94d0f24bdc51801783c58367f27b235a1883ddb077eeaf0e740b',
  plan60: '1c5bb22087b5913531e6ebc76e6ae3da99c04ef1feaebfb1f2e74eaa68160dfb',
  features: '3cf941237cdce719ba963036f3f43cec67998b6c07e0295ef3a6d17b8ba38790',
});
export const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export function checkSha(path, expected) {
  const got = sha256File(path);
  if (got !== expected) throw new Error(`sha256 が違います: ${path} ${got}`);
  return got;
}
checkSha(USER_SIM + '/dist/sim-core.js', EXPECTED.sim_core);
checkSha(USER_SIM + '/dist/control-delay.js', EXPECTED.control_delay);
checkSha(USER_SIM + '/dist/rc-speed-controller.js', EXPECTED.rc_teacher);
checkSha(USER_SIM + '/dist/scene.js', EXPECTED.scene);
checkSha(USER_SIM + '/qa/verify-learned-lap.mjs', EXPECTED.verify_learned_lap);
checkSha(USER_SIM + '/qa/rc-collection-plan-60.json', EXPECTED.plan60);
checkSha(EXP + '/rl/features.mjs', EXPECTED.features);

export const core = await import(pathToFileURL(USER_SIM + '/dist/sim-core.js').href);
export const delayMod = await import(pathToFileURL(USER_SIM + '/dist/control-delay.js').href);
export const userVerifier = await import(pathToFileURL(USER_SIM + '/qa/verify-learned-lap.mjs').href);
const {loadCourseWorld} = await import(pathToFileURL(USER_SIM + '/qa/load-course-world.mjs').href);
export const {WORLD_VERSION} = await import(pathToFileURL(USER_SIM + '/dist/world.js').href);
export const rcTeacherMod = await import(pathToFileURL(USER_SIM + '/dist/rc-speed-controller.js').href);
export const features = await import(pathToFileURL(EXP + '/rl/features.mjs').href);
export const WORLD = await loadCourseWorld();
export const PLAN60 = JSON.parse(fs.readFileSync(USER_SIM + '/qa/rc-collection-plan-60.json', 'utf8'));

// ---- app.js reset() の設定(計画の 1 条件の値から) ----
export function browserConfig(v, mode = 'single') {
  const c = {seed: v.seed, dt: 1 / 30, timeLimit: v.limit, maxSpeed: v.maxspeed, maxSteer: v.maxsteer, acceleration: v.acceleration,
    steerRate: v['steer-rate'], vehicleLength: v['car-length'], vehicleWidth: v['car-width'], wheelbase: v.wheelbase,
    stopOnCollision: v['stop-collision'], frictionMultiplier: v.scenario === 'low-friction' ? .4 : 1, noiseStd: v.scenario === 'noise' ? .025 : 0};
  Object.assign(c, core.normalizeLongitudinalConfig({longitudinalModel: v['longitudinal-model'], driveTau: v['drive-tau'],
    driveAccelerationCap: v['drive-acceleration-cap'], brakeDeceleration: v['brake-deceleration']}));
  if (mode === 'continuous') c.episodeMode = 'continuous-laps-v1';
  return c;
}
// 標準の開始(計画のうちゲート 0・ずれ 0・baseline のもの。種 1)
export const STD_PLAN = Object.freeze({...PLAN60.find(c => c['start-gate'] === 0 && c['spawn-offset'] === 0 && c['spawn-yaw'] === 0 && c.scenario === 'baseline'), seed: 1});
// app.js reset() の開始: 経路を開始ゲートで回し、左へ offset、向き +turn
export function startWorld(gate = 0, offset = 0, yawDeg = 0, world = WORLD) {
  const i = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(i), ...world.route.slice(0, i)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {...world, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}};
}

// ---- 自前の MLP(policy.json: x = clip((obs − mean)/std, ±clip) → 層 → [−1, 1]) ----
export function loadMlp(key) {
  const {path, sha} = POLICIES[key];
  checkSha(path, sha);
  const w = JSON.parse(fs.readFileSync(path, 'utf8'));
  const mean = Float64Array.from(w.obs_mean), std = Float64Array.from(w.obs_std), clip = w.obs_clip;
  const layers = w.layers.map(L => ({nIn: L.n_in, nOut: L.n_out, tanh: L.act === 'tanh', w: Float64Array.from(L.w), b: Float64Array.from(L.b)}));
  const fn = obs => {
    let x = new Float64Array(obs.length);
    for (let i = 0; i < obs.length; i++) x[i] = Math.max(-clip, Math.min(clip, (obs[i] - mean[i]) / std[i]));
    for (const L of layers) {
      const y = new Float64Array(L.nOut);
      for (let o = 0; o < L.nOut; o++) {
        let s = L.b[o];
        const off = o * L.nIn;
        for (let i = 0; i < L.nIn; i++) s += L.w[off + i] * x[i];
        y[o] = L.tanh ? Math.tanh(s) : s;
      }
      x = y;
    }
    return [Math.max(-1, Math.min(1, x[0])), Math.max(-1, Math.min(1, x[1]))];
  };
  fn.key = key; fn.sha = sha;
  return fn;
}
export const MAX_STEER = .52, SPEED_CAP = 4.5;

// ---- 閉ループで 1 回走らせる(元の sim-core。RL の env を通さない) ----
// opts:
//   plan / gate / offset / yawDeg / mode('single'|'continuous') / timeLimit
//   delay: 0|1(ユーザーの control-delay.js の待ち行列。最初は中立)
//   prev: 'applied'(観測の前回の行動 = 実際に適用した行動。RL の段の約束)| 'fresh'(= 方策が今出した値。確認 r1 の約束)
//   phys: sim の設定だけを上書き(舵の速さ・ブレーキなど。観測・教師は標準の値のまま = 知らない)
//   steerGain / speedGain: 方策の指令に掛ける(ブリッジのゲインと同じく、待ち行列の前)
export function runPolicy(policy, o = {}) {
  const {plan = STD_PLAN, gate = 0, offset = 0, yawDeg = 0, mode = 'single', delay = 0, prev = 'applied', phys = {}, steerGain = 1, speedGain = 1,
    timeLimit = null} = o;
  const nominal = browserConfig(plan, mode);
  if (timeLimit) nominal.timeLimit = timeLimit;
  const world = startWorld(gate, offset, yawDeg);
  const config = {...nominal, ...phys};
  const sim = core.createSimulation(world, config);
  let snap = sim.reset({seed: config.seed, scenario: plan.scenario});
  if (snap.state.terminal) return {invalid_spawn: true};
  const q = delayMod.createControlDelayQueue(); q.configure(delay);
  const tracker = features.createFeatureTracker(world, nominal);
  tracker.resetProgress(snap.state.x, snap.state.z, snap.metrics.route_index);
  const obs = new Float64Array(79);
  tracker.observe(snap.state, snap.metrics.route_index, 0, 0, obs);
  const cmds = [];
  while (!snap.state.terminal) {
    const a = policy(obs);
    const pred = {steering: MAX_STEER * a[0] * steerGain, speed: SPEED_CAP * (a[1] + 1) / 2 * speedGain, info: {a0: a[0], a1: a[1]}};
    const {action} = q.apply(pred, {frame: snap.state.frame, sim_time: snap.state.sim_time, dt: config.dt});
    cmds.push({steering: action.steering, speed: action.speed, fresh_steering: pred.steering, fresh_speed: pred.speed});
    snap = sim.step({steering: action.steering, speed: action.speed});
    tracker.advanceProgress(snap.state.x, snap.state.z);
    let p0 = a[0], p1 = a[1];
    if (prev === 'applied') {
      const inf = action.info ?? {};
      if (Number.isFinite(inf.a0)) { p0 = inf.a0; p1 = inf.a1; } else { p0 = 0; p1 = -1; }   // 最初の中立 = 舵 0・速さ 0
    }
    tracker.observe(snap.state, snap.metrics.route_index, p0, p1, obs);
  }
  const report = sim.report();
  report.world = world;
  return {report, world, config, nominal, cmds, summary: summarize(report)};
}
// RC 教師(ユーザーの dist/rc-speed-controller.js。標準の設定で作る = 物理のずれを知らない)
export function runTeacher(o = {}) {
  const {plan = STD_PLAN, offset = 0, yawDeg = 0, mode = 'single', delay = 0, phys = {}, steerGain = 1, speedGain = 1} = o;
  const nominal = browserConfig(plan, mode);
  if (mode === 'single') nominal.timeLimit = 20;
  const world = startWorld(0, offset, yawDeg);
  const t = rcTeacherMod.createRcTrainingTeacher(world, nominal, {gate: 0, left_offset_m: offset, left_yaw_deg: yawDeg});
  const config = {...nominal, ...phys};
  const sim = core.createSimulation(world, config);
  let snap = sim.reset({seed: config.seed, scenario: plan.scenario});
  const q = delayMod.createControlDelayQueue(); q.configure(delay);
  const cmds = [];
  while (!snap.state.terminal) {
    const c = t.action(snap);
    const pred = {steering: c.steering * steerGain, speed: c.speed * speedGain};
    const {action} = q.apply(pred, {frame: snap.state.frame, sim_time: snap.state.sim_time, dt: config.dt});
    cmds.push({steering: action.steering, speed: action.speed, fresh_steering: pred.steering, fresh_speed: pred.speed});
    snap = sim.step({steering: action.steering, speed: action.speed});
  }
  const report = sim.report(); report.world = world;
  return {report, world, config, nominal, cmds, summary: summarize(report)};
}

// sim のレポートの要約(周・ゲート・黒シート・衝突)。合否の式は sim の数え方そのまま
export function summarize(r) {
  const m = r.metrics, o = r.outcome, cont = r.config.episodeMode === 'continuous-laps-v1';
  const laps = cont ? (m.lap_events ?? []).map(l => ({t: l.elapsed, gates: l.passed_gates, cross: l.required_crossing_events.length, col: l.collisions}))
    : (o.reason === 'route_complete' ? [{t: m.elapsed, gates: m.passed_gates, cross: m.required_crossing_events.length, col: m.collisions}] : []);
  const valid = cont
    ? o.reason === 'collection_duration_complete' && m.collisions === 0 && laps.length >= 2 && laps.every(l => l.gates === 33 && l.cross === 1 && l.col === 0)
    : o.reason === 'route_complete' && o.success === true && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1;
  const ce = m.collision_events?.[0];
  return {valid, reason: o.reason, laps: laps.map(l => +l.t.toFixed(6)), gates: m.passed_gates, collisions: m.collisions,
    first_collision: ce ? {t: +ce.time.toFixed(3), collider: ce.collider, kind: ce.kind, x: +ce.x.toFixed(3), z: +ce.z.toFixed(3)} : null, frames: r.actions.length};
}
export const hashOf = v => crypto.createHash('sha256').update(JSON.stringify(v)).digest('hex');

// ---- すき間 1: 自前の厳密な距離(フレームの終わりの姿勢ごと) ----
// 車体 = 向きのある長方形(半長 hx・半幅 hz)。障害物 = sim と同じ 3 種(半径つき線分・円・向きのない長方形の脚)と外枠
const toLocal = (p, s) => { const dx = p[0] - s.x, dz = p[1] - s.z, c = Math.cos(s.yaw), n = Math.sin(s.yaw); return [dx * c - dz * n, dx * n + dz * c]; };
const ptRect = (p, hx, hz) => Math.hypot(Math.max(0, Math.abs(p[0]) - hx), Math.max(0, Math.abs(p[1]) - hz));
const ptSeg = (p, a, b) => { const dx = b[0] - a[0], dz = b[1] - a[1], L = dx * dx + dz * dz; const t = L ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / L)) : 0; return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dz); };
function segHitsRect(a, b, hx, hz) {   // 線分が長方形に入るか(Liang–Barsky)
  let lo = 0, hi = 1;
  for (const [o, d, h] of [[a[0], b[0] - a[0], hx], [a[1], b[1] - a[1], hz]]) {
    if (Math.abs(d) < 1e-15) { if (Math.abs(o) > h) return false; continue; }
    let t0 = (-h - o) / d, t1 = (h - o) / d; if (t0 > t1) [t0, t1] = [t1, t0];
    lo = Math.max(lo, t0); hi = Math.min(hi, t1); if (lo > hi) return false;
  }
  return true;
}
function segRect(a, b, hx, hz) {
  if (segHitsRect(a, b, hx, hz)) return 0;
  let d = Math.min(ptRect(a, hx, hz), ptRect(b, hx, hz));
  for (const c of [[-hx, -hz], [-hx, hz], [hx, -hz], [hx, hz]]) d = Math.min(d, ptSeg(c, a, b));
  return d;
}
export function exactClearance(s, world, L = .42, W = .19) {
  const hx = L / 2, hz = W / 2;
  let best = Infinity, who = null;
  for (const [i, g] of (world.segments ?? []).entries()) {
    if (!((g.height ?? .1) > 0)) continue;
    const d = segRect(toLocal(g.a, s), toLocal(g.b, s), hx, hz) - (g.radius ?? .01);
    if (d < best) { best = d; who = `seg-${i} ${g.kind ?? ''}`; }
  }
  for (const [i, k] of (world.disks ?? []).entries()) {
    if (!((k.height ?? .1) > 0)) continue;
    let d;
    if (k.kind === 'post-foot') {   // 向きのない 0.16 × 0.15 m の長方形(sim の normalizedWorld と同じ)。凸多角形どうしの距離
      const fx = .08, fz = .075, fc = [[k.x - fx, k.z - fz], [k.x + fx, k.z - fz], [k.x + fx, k.z + fz], [k.x - fx, k.z + fz]];
      const loc = fc.map(p => toLocal(p, s));
      d = Infinity;
      for (let j = 0; j < 4; j++) d = Math.min(d, segRect(loc[j], loc[(j + 1) % 4], hx, hz));
      // 車の角が脚の中に入る場合(上の辺の判定で 0 になる)も含む
    } else d = ptRect(toLocal([k.x, k.z], s), hx, hz) - (k.radius ?? .02);
    if (d < best) { best = d; who = `disk-${i} ${k.kind ?? ''}`; }
  }
  const b = world.bounds, c = Math.cos(s.yaw), n = Math.sin(s.yaw);
  if (b) for (const [x, z] of [[-hx, -hz], [-hx, hz], [hx, -hz], [hx, hz]].map(([x, z]) => [s.x + x * c + z * n, s.z - x * n + z * c])) {
    const d = Math.min(x - b.minX, b.maxX - x, z - b.minZ, b.maxZ - z);
    if (d < best) { best = d; who = 'bounds'; }
  }
  return {d: best, who};
}
export function traceClearance(report, world) {
  let best = {d: Infinity, who: null, frame: -1};
  for (const t of report.trace) { const r = exactClearance(t, world); if (r.d < best.d) best = {...r, frame: t.frame, x: t.x, z: t.z}; }
  return best;
}

// ---- すき間 2: sim の衝突判定に「太らせた車体」を当てて、同じ操作列を開ループで再生する(小区間ごと) ----
// 太らせた幅 m で衝突しない ⇒ すき間 > m(小区間ごと)。衝突する ⇒ すき間 ≤ √2·m(長方形を太らせると角が √2 倍まで出るため)。
// 小区間の長さ h = min(0.01, 0.012 / 最大の動きの速さ) が変わらないよう、maxSpeed を下げて最大の動きの速さを同じに保つ(指令は 4.64 m/s 以下なので切られない)。
// 黒シートの判定は幅に依るので、再生のコースでは外す(動きには関係しない)。
export function inflatedReplay(report, world, margin) {
  const cfg = report.config;
  const R0 = Math.hypot(cfg.vehicleLength, cfg.vehicleWidth) / 2;
  const rate0 = cfg.maxSpeed + R0 * cfg.maxSpeed / cfg.wheelbase * Math.tan(cfg.maxSteer);
  const L = cfg.vehicleLength + 2 * margin, W = cfg.vehicleWidth + 2 * margin, R1 = Math.hypot(L, W) / 2;
  const maxSpeed = rate0 / (1 + R1 / cfg.wheelbase * Math.tan(cfg.maxSteer));
  const conf = {...cfg, vehicleLength: L, vehicleWidth: W, maxSpeed};
  const sim = core.createSimulation({...world, requiredCrossings: []}, conf);
  let snap = sim.reset({seed: cfg.seed});
  if (snap.state.terminal) return {collided: true, at_spawn: true};
  let maxDiff = 0;
  for (let i = 0; i < report.actions.length && !snap.state.terminal; i++) {
    snap = sim.step(report.actions[i]);
    const a = report.trace[i];
    if (a && !snap.metrics.collisions) maxDiff = Math.max(maxDiff, Math.abs(snap.state.x - a.x), Math.abs(snap.state.z - a.z));
  }
  const m = snap.metrics;
  return {collided: m.collisions > 0, first: m.collision_events[0] ?? null, trace_max_diff: maxDiff, max_speed_used: maxSpeed};
}
// 一連の走り(同じ操作列)について、「どれも衝突しない最大の m」を二分探索で求める
export function inflationBounds(runs, {lo = 0, hi = .08, iters = 11} = {}) {
  const anyHit = m => runs.some(r => inflatedReplay(r.report, r.world, m).collided);
  if (anyHit(lo)) return {lo: null, hi: lo, note: '太らせない再生で衝突(再生が合わない)'};
  for (let k = 0; k < iters; k++) { const mid = (lo + hi) / 2; if (anyHit(mid)) hi = mid; else lo = mid; }
  return {no_collision_margin_m: lo, collision_margin_m: hi, clearance_lower_m: lo, clearance_upper_m: hi * Math.SQRT2};
}

// ---- ゲートの線を通った横の位置(フレームの軌跡を線分でつないで、線を越えた点で求める) ----
export function gateLaterals(report, world) {
  const route = world.route, n = route.length, targets = [...route.slice(1), route[0]];
  const tr = [{x: world.spawn.x, z: world.spawn.z}, ...report.trace];
  const out = [];
  let i = 0;
  for (let k = 1; k < tr.length; k++) {
    const t = targets[i], b = i ? targets[i - 1] : route[0];
    const L = Math.hypot(t[0] - b[0], t[1] - b[1]), nx = (t[0] - b[0]) / L, nz = (t[1] - b[1]) / L;
    const sg = p => (p.x - t[0]) * nx + (p.z - t[1]) * nz;
    const A = sg(tr[k - 1]), B = sg(tr[k]);
    if (A < 0 && B >= 0) {
      const f = -A / (B - A), x = tr[k - 1].x + (tr[k].x - tr[k - 1].x) * f, z = tr[k - 1].z + (tr[k].z - tr[k - 1].z) * f;
      const lat = -(x - t[0]) * nz + (z - t[1]) * nx;
      out.push({gate: i, lateral: lat, ok: Math.abs(lat) <= .65});
      if (Math.abs(lat) <= .65) i = (i + 1) % n;
    }
  }
  return out;
}
// 黒シートの横断の端までの余裕(sim の式: 車体の x への投影の半分)
export function sheetMargins(report, world) {
  const g = world.requiredCrossings[0];
  return (report.metrics.required_crossing_events ?? []).map(e => {
    const half = .42 / 2 * Math.abs(Math.cos(e.yaw)) + .19 / 2 * Math.abs(Math.sin(e.yaw));
    return {x: +e.x.toFixed(4), margin_m: +Math.min(e.x - half - g.minX, g.maxX - (e.x + half)).toFixed(4), dir: e.direction};
  });
}
// 操作の荒さ(走行中 = 0.3 m/s 超のフレーム)
export function roughness(cmds, report) {
  const tr = report.trace, dt = report.config.dt, idx = [];
  for (let i = 1; i < Math.min(cmds.length, tr.length); i++) if (tr[i - 1].speed > .3) idx.push(i);
  const rate = idx.map(i => Math.abs(cmds[i].steering - cmds[i - 1].steering) / dt).sort((a, b) => a - b);
  const q = p => rate[Math.min(rate.length - 1, Math.floor(p * (rate.length - 1)))];
  return {frames: idx.length, steer_cmd_rate_p95: q(.95), steer_cmd_rate_max: rate.at(-1), steer_edge_frac: idx.filter(i => Math.abs(cmds[i].steering) >= .494).length / idx.length,
    speed_zero_frac: idx.filter(i => cmds[i].speed <= .05).length / idx.length};
}
export function writeJson(name, data) {
  fs.mkdirSync(OUT, {recursive: true});
  fs.writeFileSync(OUT + '/' + name, JSON.stringify(data, null, 1) + '\n');
}
// 自前の乱数(mulberry32。RL の段の createRng とは別の列)
export function rng32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
