// RL のベストラップの反証の確認(r3)の共通部品(2026-10-14、確認の段 r3)
//
// r3 の段は、確認 r2 の道具(check_r2/common_r2.mjs)で r3 の主な値を測った。同じ道具で測り直すだけでは道具の誤りを見つけられないので、
// この確認では check_r2 も check_r1 も RL の env・評価も import せずに、閉ループ・設定・開始・方策・すき間・ゲートの余裕を書き直す。
//   - 物理: ユーザーの dist/sim-core.js をそのまま import(sha256 を固定)
//   - 制御遅延: ユーザーの dist/control-delay.js の createControlDelayQueue(ブラウザの待ち行列。最初は中立)
//   - コース: ユーザーの qa/load-course-world.mjs(scene.js から作る現行 v3)
//   - 設定・開始: app.js reset() の式(収集の計画 qa/rc-collection-plan-60.json の値)をここで書き直す
//   - 方策: policy.json の重みを自前の MLP で計算(行列を行ごとに持つ別の書き方)
//   - 観測: 方策の入力の定義なので rl/features.mjs だけは使う(sha256 を固定)。観測の μ・舵の速さなどは標準の値のまま(方策は今の物理を知らない)
//   - すき間: sim の衝突判定そのものに「太らせた車体」を当て、同じ操作列を再生する(フレームの間の小区間もすべて見る)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

export const USER_SIM = '/home/jetson/sim/extracted/course-simulator';
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap';
export const REPORTS = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013';
export const OUT = REPORTS + '/rl_check_r3';
export const sha256Buf = b => crypto.createHash('sha256').update(b).digest('hex');
export const sha256File = p => sha256Buf(fs.readFileSync(p));
const FIXED = {
  [USER_SIM + '/dist/sim-core.js']: 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e',
  [USER_SIM + '/dist/control-delay.js']: 'eb405f7e87ce9581c3a5ccc2850dce67d21750ad49e930a06a972a4aaf72f23c',
  [USER_SIM + '/dist/rc-speed-controller.js']: '4a32ab56369b3d0c9a76f16fe7144eb895677b72ceeefc8483645f91ae96f417',
  [USER_SIM + '/dist/scene.js']: '878fc934af2890c4db152041182b8914033224f65a2482bcc46265cdc7d0c806',
  [USER_SIM + '/qa/verify-learned-lap.mjs']: '506674aa160f94d0f24bdc51801783c58367f27b235a1883ddb077eeaf0e740b',
  [USER_SIM + '/qa/rc-collection-plan-60.json']: '1c5bb22087b5913531e6ebc76e6ae3da99c04ef1feaebfb1f2e74eaa68160dfb',
  [EXP + '/rl/features.mjs']: '3cf941237cdce719ba963036f3f43cec67998b6c07e0295ef3a6d17b8ba38790',
};
for (const [p, h] of Object.entries(FIXED)) if (sha256File(p) !== h) throw new Error('sha256 が違います: ' + p);

// 方策(書き出したファイル。sha256 は r3 の段の provenance と照合する)
export const POLICY_FILES = Object.freeze({
  selected: {path: REPORTS + '/rl/best/main/policy.json', sha: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee'},
  r2_5M: {path: REPORTS + '/rl/r2/dr/policy.json', sha: '5f6f57168217b7355d2015633d36c6cc50ddbc0d7a0c50ea3f34aeabbf8155a5'},
  r3_s21: {path: REPORTS + '/rl/r3/r3-s21/policy.json', sha: '438ad2de35c5f49c4fed4d1ebca4d44c265f97188b907737719af8a47d4c439e'},
  r3_s22: {path: REPORTS + '/rl/r3/r3-s22/policy.json', sha: '3e229a91749353d5b4f886b9abe7f823281e05fa037af3e56baba391f03ed1bf'},
});

export const core = await import(pathToFileURL(USER_SIM + '/dist/sim-core.js').href);
export const delayMod = await import(pathToFileURL(USER_SIM + '/dist/control-delay.js').href);
export const verifier = await import(pathToFileURL(USER_SIM + '/qa/verify-learned-lap.mjs').href);
export const rcMod = await import(pathToFileURL(USER_SIM + '/dist/rc-speed-controller.js').href);
export const {WORLD_VERSION} = await import(pathToFileURL(USER_SIM + '/dist/world.js').href);
const {loadCourseWorld} = await import(pathToFileURL(USER_SIM + '/qa/load-course-world.mjs').href);
export const feat = await import(pathToFileURL(EXP + '/rl/features.mjs').href);
export const BASE_WORLD = await loadCourseWorld();
export const PLAN = JSON.parse(fs.readFileSync(USER_SIM + '/qa/rc-collection-plan-60.json', 'utf8'));

// app.js reset() の config(計画の 1 行から)。longitudinalConfig() は real-drive-response-estimate-v1 の 3 つの値
export function appConfig(row, continuous) {
  const cfg = {
    seed: row.seed, dt: 1 / 30, timeLimit: row.limit, maxSpeed: row.maxspeed, maxSteer: row.maxsteer, acceleration: row.acceleration,
    steerRate: row['steer-rate'], vehicleLength: row['car-length'], vehicleWidth: row['car-width'], wheelbase: row.wheelbase,
    stopOnCollision: row['stop-collision'], frictionMultiplier: row.scenario === 'low-friction' ? .4 : 1, noiseStd: row.scenario === 'noise' ? .025 : 0,
    longitudinalModel: row['longitudinal-model'], driveTau: row['drive-tau'], driveAccelerationCap: row['drive-acceleration-cap'],
    brakeDeceleration: row['brake-deceleration'],
  };
  if (continuous) cfg.episodeMode = 'continuous-laps-v1';
  return cfg;
}
// 標準の開始 STD: ゲート 0・ずれ 0・baseline、種 1(RL の段・確認 r1・r2 と同じ約束)
const stdRow = PLAN.find(r => r.scenario === 'baseline' && r['start-gate'] === 0 && r['spawn-offset'] === 0 && r['spawn-yaw'] === 0);
export const STD = Object.freeze({...stdRow, seed: 1, name: 'STD'});
// app.js reset() の開始の姿勢
export function spawnWorld(gate, offset, yawDeg) {
  const n = BASE_WORLD.route.length, k = Math.max(0, Math.min(n - 1, Math.trunc(gate) || 0));
  const route = BASE_WORLD.route.slice(k).concat(BASE_WORLD.route.slice(0, k));
  const [x0, z0] = route[0], [x1, z1] = route[1], h = Math.atan2(-(z1 - z0), x1 - x0);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {...BASE_WORLD, route, spawn: {x: x0 + off * Math.sin(h), z: z0 - off * Math.cos(h), yaw: h + turn}};
}

// 自前の MLP(行ごとの Float64Array。policy.json の層は w[o * n_in + i])
export function loadPolicy(key, path = null) {
  const file = path ?? POLICY_FILES[key].path, buf = fs.readFileSync(file), sha = sha256Buf(buf);
  if (!path && sha !== POLICY_FILES[key].sha) throw new Error(`方策の sha256 が違います: ${key} ${sha}`);
  const j = JSON.parse(buf.toString('utf8'));
  const mu = j.obs_mean, sd = j.obs_std, clip = j.obs_clip;
  const layers = j.layers.map(L => ({rows: Array.from({length: L.n_out}, (_, o) => Float64Array.from(L.w.slice(o * L.n_in, (o + 1) * L.n_in))),
    b: Float64Array.from(L.b), act: L.act}));
  const f = obs => {
    let h = Array.from(obs, (v, i) => Math.min(clip, Math.max(-clip, (v - mu[i]) / sd[i])));
    for (const L of layers) {
      h = L.rows.map((row, o) => { let s = L.b[o]; for (let i = 0; i < row.length; i++) s += row[i] * h[i]; return L.act === 'tanh' ? Math.tanh(s) : s; });
    }
    return [Math.min(1, Math.max(-1, h[0])), Math.min(1, Math.max(-1, h[1]))];
  };
  f.key = key; f.sha = sha; f.meta = j.meta ?? null;
  return f;
}
export const STEER_MAX = .52, V_CAP = 4.5;

// 閉ループ(元の sim-core・ユーザーの遅延の待ち行列)
// o: row(計画の行)/ gate / offset / yaw / continuous / delay / prev('applied'|'fresh') / phys / steerGain / speedGain / timeLimit
export function drive(pol, o = {}) {
  const row = o.row ?? STD, gate = o.gate ?? row['start-gate'] ?? 0, offset = o.offset ?? row['spawn-offset'] ?? 0, yaw = o.yaw ?? row['spawn-yaw'] ?? 0;
  const nominal = appConfig(row, !!o.continuous);
  if (o.timeLimit) nominal.timeLimit = o.timeLimit;
  const world = spawnWorld(gate, offset, yaw);
  const sim = (o.coreMod ?? core).createSimulation(world, {...nominal, ...(o.phys ?? {})});   // coreMod: 反事実の物理(摩擦円)だけで使う
  let snap = sim.reset({seed: nominal.seed, scenario: row.scenario});
  if (snap.state.terminal) return {invalid_spawn: true};
  const queue = delayMod.createControlDelayQueue(); queue.configure(o.delay ?? 0);
  const tracker = feat.createFeatureTracker(world, nominal), obs = new Float64Array(feat.OBS_DIM);
  tracker.resetProgress(snap.state.x, snap.state.z, snap.metrics.route_index);
  tracker.observe(snap.state, snap.metrics.route_index, 0, 0, obs);
  const sg = o.steerGain ?? 1, vg = o.speedGain ?? 1, cmds = [];
  while (!snap.state.terminal) {
    const [a0, a1] = pol(obs);
    const out = queue.apply({steering: STEER_MAX * a0 * sg, speed: V_CAP * (a1 + 1) / 2 * vg, info: {a0, a1}},
      {frame: snap.state.frame, sim_time: snap.state.sim_time, dt: nominal.dt});
    const act = out.action;
    cmds.push([act.steering, act.speed]);
    snap = sim.step({steering: act.steering, speed: act.speed});
    tracker.advanceProgress(snap.state.x, snap.state.z);
    let p0 = a0, p1 = a1;
    if ((o.prev ?? 'applied') === 'applied' && (o.delay ?? 0) === 1) {
      const inf = act.info ?? {};
      [p0, p1] = Number.isFinite(inf.a0) ? [inf.a0, inf.a1] : [0, -1];
    }
    tracker.observe(snap.state, snap.metrics.route_index, p0, p1, obs);
  }
  const report = sim.report(); report.world = world;
  return {report, world, cmds, judge: judge(report)};
}
// RC 教師(標準の設定で作る。物理のずれは知らない)
export function driveTeacher(o = {}) {
  const row = o.row ?? STD, offset = o.offset ?? row['spawn-offset'] ?? 0, yaw = o.yaw ?? row['spawn-yaw'] ?? 0;
  const nominal = appConfig(row, !!o.continuous); if (!o.continuous) nominal.timeLimit = 20;
  const world = spawnWorld(0, offset, yaw);
  const sim = (o.coreMod ?? core).createSimulation(world, {...nominal, ...(o.phys ?? {})});
  let snap = sim.reset({seed: nominal.seed, scenario: row.scenario});
  const t = rcMod.createRcTrainingTeacher(world, core.createSimulation(world, nominal).snapshot().config, {gate: 0, left_offset_m: offset, left_yaw_deg: yaw});
  const queue = delayMod.createControlDelayQueue(); queue.configure(o.delay ?? 0);
  const cmds = [];
  while (!snap.state.terminal) {
    const c = t.action(snap);
    const act = queue.apply({steering: c.steering * (o.steerGain ?? 1), speed: c.speed * (o.speedGain ?? 1)}, {frame: snap.state.frame, sim_time: snap.state.sim_time, dt: nominal.dt}).action;
    cmds.push([act.steering, act.speed]);
    snap = sim.step({steering: act.steering, speed: act.speed});
  }
  const report = sim.report(); report.world = world;
  return {report, world, cmds, judge: judge(report)};
}
// 合否(sim の数えたもの。単周: route_complete・33 ゲート・黒シート 1 回・衝突 0。連続 20 秒: 2 周以上で各周 33・黒シート 1・衝突 0)
export function judge(r) {
  const m = r.metrics, o = r.outcome, cont = r.config.episodeMode === 'continuous-laps-v1';
  let laps, ok;
  if (cont) {
    laps = (m.lap_events ?? []).map(e => e.elapsed);
    ok = o.reason === 'collection_duration_complete' && m.collisions === 0 && laps.length >= 2 &&
      m.lap_events.every(e => e.passed_gates === 33 && e.required_crossing_events.length === 1 && e.collisions === 0);
  } else {
    laps = o.reason === 'route_complete' ? [m.elapsed] : [];
    ok = o.reason === 'route_complete' && o.success === true && m.finished === true && m.collisions === 0 && m.passed_gates === 33 &&
      m.required_crossing_events.length === 1;
  }
  const c = m.collision_events?.[0];
  return {ok, reason: o.reason, laps, gates: m.passed_gates, collisions: m.collisions, frames: r.actions.length,
    first_collision: c ? {t: +c.time.toFixed(3), collider: c.collider, x: +c.x.toFixed(2), z: +c.z.toFixed(2)} : null};
}
export const hashJson = v => sha256Buf(JSON.stringify(v));

// 太らせた車体での再生(sim の衝突判定。フレームの間の小区間もすべて)。margin で衝突しない ⇒ すき間 > margin
// 小区間の長さが変わらないよう、最大の動きの速さを同じに保つ maxSpeed にする(指令は 4.64 m/s 以下なので切られない)。黒シートは幅で判定するので外す
export function replayInflated(run, margin) {
  const cfg = run.report.config;
  const rad = (L, W) => Math.hypot(L, W) / 2, k = Math.tan(cfg.maxSteer) / cfg.wheelbase;
  const rate = cfg.maxSpeed * (1 + rad(cfg.vehicleLength, cfg.vehicleWidth) * k);
  const L = cfg.vehicleLength + 2 * margin, W = cfg.vehicleWidth + 2 * margin, vmax = rate / (1 + rad(L, W) * k);
  if (vmax < 4.7) throw new Error('太らせすぎ: 指令が切られます');
  const sim = core.createSimulation({...run.world, requiredCrossings: []}, {...cfg, vehicleLength: L, vehicleWidth: W, maxSpeed: vmax});
  let s = sim.reset({seed: cfg.seed});
  if (s.state.terminal) return {hit: true, spawn: true};
  let drift = 0;
  for (let i = 0; i < run.report.actions.length && !s.state.terminal; i++) {
    s = sim.step(run.report.actions[i]);
    if (!s.metrics.collisions) { const t = run.report.trace[i]; drift = Math.max(drift, Math.hypot(s.state.x - t.x, s.state.z - t.z)); }
  }
  return {hit: s.metrics.collisions > 0, first: s.metrics.collision_events[0] ?? null, drift};
}
// 走りの集まりについて、どれも衝突しない最大の太らせ幅(2 分探索)
export function clearanceBounds(runs, hi = .1, iters = 12) {
  const any = m => runs.some(r => replayInflated(r, m).hit);
  if (any(0)) return {error: '太らせない再生で衝突'};
  let lo = 0; if (!any(hi)) return {lower_m: hi, upper_m: null};
  for (let i = 0; i < iters; i++) { const mid = (lo + hi) / 2; if (any(mid)) hi = mid; else lo = mid; }
  return {lower_m: lo, upper_m: hi * Math.SQRT2, hit_margin_m: hi};
}
// ゲートの線(sim と同じ定義: 目標とその前の目標を結ぶ向きに垂直)を通った横の位置を、軌跡から自前で求める
export function gateCrossings(run) {
  const w = run.world, route = w.route, tg = [...route.slice(1), route[0]], pts = [{x: w.spawn.x, z: w.spawn.z}, ...run.report.trace];
  const out = []; let i = 0;
  for (let k = 1; k < pts.length; k++) {
    const t = tg[i], b = i ? tg[i - 1] : route[0], L = Math.hypot(t[0] - b[0], t[1] - b[1]), nx = (t[0] - b[0]) / L, nz = (t[1] - b[1]) / L;
    const a = (pts[k - 1].x - t[0]) * nx + (pts[k - 1].z - t[1]) * nz, c = (pts[k].x - t[0]) * nx + (pts[k].z - t[1]) * nz;
    if (a < 0 && c >= 0) {
      const f = -a / (c - a), x = pts[k - 1].x + (pts[k].x - pts[k - 1].x) * f, z = pts[k - 1].z + (pts[k].z - pts[k - 1].z) * f;
      const lat = Math.abs(-(x - t[0]) * nz + (z - t[1]) * nx);
      out.push({gate: i, lat});
      if (lat <= .65) i = (i + 1) % tg.length;
    }
  }
  return out;
}
// 黒シートの横断の余裕(車体の x への投影の半分を引いた、端までの距離)
export function sheetMargin(run) {
  const g = run.world.requiredCrossings[0];
  return (run.report.metrics.required_crossing_events ?? []).map(e => {
    const half = .21 * Math.abs(Math.cos(e.yaw)) + .095 * Math.abs(Math.sin(e.yaw));
    return Math.min(e.x - half - g.minX, g.maxX - e.x - half);
  });
}
// 操作の荒さ(走行中 = 前のフレームの速さ 0.3 m/s 超)
export function rough(run) {
  const tr = run.report.trace, c = run.cmds, dt = 1 / 30, r = [], edge = [];
  for (let i = 1; i < Math.min(c.length, tr.length); i++) if (tr[i - 1].speed > .3) { r.push(Math.abs(c[i][0] - c[i - 1][0]) / dt); edge.push(Math.abs(c[i][0]) >= .494); }
  r.sort((a, b) => a - b);
  return {p95: r[Math.floor(.95 * (r.length - 1))], edge: edge.filter(Boolean).length / edge.length, n: r.length};
}
export function save(name, data) { fs.mkdirSync(OUT, {recursive: true}); fs.writeFileSync(OUT + '/' + name, JSON.stringify(data, null, 1) + '\n'); }
// 自前の乱数(xorshift128+ 風の別の列。RL の段とも確認 r2 の mulberry32 とも別)
export function rand(seed) {
  let s0 = (seed ^ 0x9E3779B9) >>> 0 || 1, s1 = (Math.imul(seed, 0x85EBCA6B) ^ 0xC2B2AE35) >>> 0 || 2;
  return () => { let x = s0; const y = s1; s0 = y; x ^= x << 23; x ^= x >>> 17; x ^= y ^ (y >>> 26); s1 = x >>> 0; return ((s0 + s1) >>> 0) / 4294967296; };
}
