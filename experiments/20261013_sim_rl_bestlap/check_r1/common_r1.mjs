// RL のベストラップの反証の確認(r1)の共通部品(2026-10-13、確認の段)
//
// 方針: RL の環境(rl/env_core.mjs)・RL の確認器(rl/lap_audit.mjs)・RL が作ったコース(sim/world-v3.json)は使わない。
//   - 物理: ユーザーの展開物の dist/sim-core.js を **そのまま** import する(コピーではなく元のファイル。変えない)
//   - コース: ユーザーの qa/load-course-world.mjs(scene.js から作る現行 v3)
//   - 設定・開始: app.js の reset() と同じ式をここで書き直す(収集の計画 qa/rc-collection-plan-60.json の値から)
//   - 方策: policy.json の重みを自前の MLP で計算する(rl/policy_forward.mjs は照合にだけ使う)
//   - 観測: 方策の入力の定義そのものなので rl/features.mjs を使う(これだけは RL の側のファイル)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

export const USER_SIM = '/home/jetson/sim/extracted/course-simulator';
export const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap';
export const REPORTS = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013';
export const OUT = REPORTS + '/rl_check_r1';
export const POLICY_JSON = REPORTS + '/rl/best/main/policy.json';
export const EXPECTED = Object.freeze({
  sim_core: 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e',
  verify_learned_lap: '506674aa160f94d0f24bdc51801783c58367f27b235a1883ddb077eeaf0e740b',
  plan60: '1c5bb22087b5913531e6ebc76e6ae3da99c04ef1feaebfb1f2e74eaa68160dfb',
  policy_json: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee',
});
export const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
export function checkSha(path, expected) {
  const got = sha256File(path);
  if (got !== expected) throw new Error(`sha256 が違います: ${path} ${got}`);
  return got;
}
checkSha(USER_SIM + '/dist/sim-core.js', EXPECTED.sim_core);
checkSha(USER_SIM + '/qa/verify-learned-lap.mjs', EXPECTED.verify_learned_lap);
checkSha(USER_SIM + '/qa/rc-collection-plan-60.json', EXPECTED.plan60);

export const core = await import(pathToFileURL(USER_SIM + '/dist/sim-core.js').href);
export const userVerifier = await import(pathToFileURL(USER_SIM + '/qa/verify-learned-lap.mjs').href);
const {loadCourseWorld} = await import(pathToFileURL(USER_SIM + '/qa/load-course-world.mjs').href);
const {WORLD_VERSION} = await import(pathToFileURL(USER_SIM + '/dist/world.js').href);
export {WORLD_VERSION};
export const SCENE_WORLD = await loadCourseWorld();   // ブラウザと同じ scene.js から作った現行コース
export const PLAN60 = JSON.parse(fs.readFileSync(USER_SIM + '/qa/rc-collection-plan-60.json', 'utf8'));

// app.js reset() の config(収集の計画の 1 条件の値から)。overrides で物理を変える(感度の確認用)
export function browserConfig(v, {episodeMode = 'continuous', overrides = {}} = {}) {
  const config = {seed: v.seed, dt: 1 / 30, timeLimit: v.limit, maxSpeed: v.maxspeed, maxSteer: v.maxsteer, acceleration: v.acceleration,
    steerRate: v['steer-rate'], vehicleLength: v['car-length'], vehicleWidth: v['car-width'], wheelbase: v.wheelbase,
    stopOnCollision: v['stop-collision'], frictionMultiplier: v.scenario === 'low-friction' ? .4 : 1, noiseStd: v.scenario === 'noise' ? .025 : 0};
  Object.assign(config, core.normalizeLongitudinalConfig({longitudinalModel: v['longitudinal-model'], driveTau: v['drive-tau'],
    driveAccelerationCap: v['drive-acceleration-cap'], brakeDeceleration: v['brake-deceleration']}));
  if (episodeMode === 'continuous') config.episodeMode = 'continuous-laps-v1';
  return {...config, ...overrides};
}
// 標準の条件(計画の 0 番 = ゲート 0・ずれ 0・baseline)。種は 1(RL の記録と同じ)
export const STD_PLAN = Object.freeze({...PLAN60.find(c => c['spawn-offset'] === 0 && c['spawn-yaw'] === 0 && c.scenario === 'baseline'), seed: 1});

// app.js reset() の開始: 経路を開始ゲートで回し、左へ offset、向き +turn
export function activeWorld(world, gate = 0, offset = 0, yawDeg = 0) {
  const startIndex = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(startIndex), ...world.route.slice(0, startIndex)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {...world, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}};
}

// ---- 方策(自前の MLP。policy.json: x = clip((obs − mean)/std, ±clip) → tanh 256 → tanh 256 → 線形 2 → [−1, 1]) ----
export function loadOwnPolicy(path = POLICY_JSON) {
  checkSha(path, EXPECTED.policy_json);
  const w = JSON.parse(fs.readFileSync(path, 'utf8'));
  const mean = w.obs_mean, std = w.obs_std, clip = w.obs_clip;
  const layers = w.layers.map(L => ({nIn: L.n_in, nOut: L.n_out, tanh: L.act === 'tanh', w: Float64Array.from(L.w), b: Float64Array.from(L.b)}));
  return obs => {
    let x = Array.from(obs, (v, i) => Math.max(-clip, Math.min(clip, (v - mean[i]) / std[i])));
    for (const L of layers) {
      const y = new Array(L.nOut);
      for (let o = 0; o < L.nOut; o++) {
        let s = L.b[o];
        for (let i = 0; i < L.nIn; i++) s += L.w[o * L.nIn + i] * x[i];
        y[o] = L.tanh ? Math.tanh(s) : s;
      }
      x = y;
    }
    return [Math.max(-1, Math.min(1, x[0])), Math.max(-1, Math.min(1, x[1]))];
  };
}
// 行動 → アプリの単位(protocol action: 舵 0.52·a0、速さ 4.5·(a1+1)/2)
export const SPEED_CAP = 4.5, MAX_STEER = .52;
export const toCommand = (a0, a1, cap = SPEED_CAP) => ({steering: MAX_STEER * a0, speed: cap * (a1 + 1) / 2});

export const features = await import(pathToFileURL(EXP + '/rl/features.mjs').href);

// ---- 閉ループで 1 回走らせる(元の sim-core。RL の env を通さない) ----
// opts: plan(収集の計画の 1 条件の値)・gate/offset/yawDeg・episodeMode・delay(0 か 1。アプリの固定遅延と同じく最初は中立)・
//       simOverrides(物理だけ変える。方策の観測は標準の値のまま = 方策は知らない)・steerGain/speedGain(指令に掛ける)・
//       world(コースを差し替える)・maxFrames
export function runClosedLoop(policy, {plan = STD_PLAN, gate = 0, offset = 0, yawDeg = 0, episodeMode = 'single', delay = 0,
  simOverrides = {}, steerGain = 1, speedGain = 1, world = SCENE_WORLD, timeLimit = null, keepReport = true} = {}) {
  const nominal = browserConfig(plan, {episodeMode});
  if (timeLimit) nominal.timeLimit = timeLimit;
  const simConfig = {...nominal, ...simOverrides};
  const active = activeWorld(world, gate, offset, yawDeg);
  const sim = core.createSimulation(active, simConfig);
  let snap = sim.reset({seed: simConfig.seed, scenario: plan.scenario});
  if (snap.state.terminal) return {invalid_spawn: true, start: {gate, offset, yawDeg}};
  const tracker = features.createFeatureTracker(active, nominal);   // 観測は標準の物理の値で計算する
  tracker.resetProgress(snap.state.x, snap.state.z, snap.metrics.route_index);
  const obs = new Float64Array(79);
  let prev = [0, 0], queued = null;
  tracker.observe(snap.state, snap.metrics.route_index, 0, 0, obs);
  const cmdLog = [];
  while (!snap.state.terminal) {
    const a = policy(obs);
    let cmd = toCommand(a[0], a[1]);
    cmd = {steering: cmd.steering * steerGain, speed: cmd.speed * speedGain};
    cmdLog.push(cmd);
    let applied = cmd;
    if (delay === 1) { applied = queued ?? {steering: 0, speed: 0}; queued = cmd; }
    snap = sim.step(applied);
    tracker.advanceProgress(snap.state.x, snap.state.z);
    prev = a;
    tracker.observe(snap.state, snap.metrics.route_index, prev[0], prev[1], obs);
  }
  const report = sim.report();
  report.world = active;
  report.context = {start_variation: {gate, left_offset_m: offset, left_yaw_deg: yawDeg}, controller: 'rl-check-r1',
    delay_frames: delay, sim_overrides: simOverrides, steer_gain: steerGain, speed_gain: speedGain};
  return {report: keepReport ? report : null, summary: summarize(report), cmdLog};
}

// sim のレポートの要約(周・ゲート・黒シート・衝突)
export function summarize(report) {
  const m = report.metrics, o = report.outcome;
  const continuous = report.config.episodeMode === 'continuous-laps-v1';
  let laps;
  if (continuous) laps = (m.lap_events ?? []).map(l => ({elapsed: l.elapsed, gates: l.passed_gates, crossings: l.required_crossing_events.length, collisions: l.collisions}));
  else laps = o.reason === 'route_complete' ? [{elapsed: m.elapsed, gates: m.passed_gates, crossings: m.required_crossing_events.length, collisions: m.collisions}] : [];
  const valid = continuous
    ? o.reason === 'collection_duration_complete' && m.collisions === 0 && laps.length >= 2 && laps.every(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0)
    : o.reason === 'route_complete' && o.success === true && m.collisions === 0 && m.passed_gates === 33 && m.required_crossing_events.length === 1;
  return {reason: o.reason, valid, laps, passed_gates: m.passed_gates, collisions: m.collisions, frames: report.actions.length,
    elapsed: m.elapsed, distance: m.distance};
}
export const hashActions = actions => crypto.createHash('sha256').update(JSON.stringify(actions)).digest('hex');
export function writeJson(name, data) {
  fs.mkdirSync(OUT, {recursive: true});
  fs.writeFileSync(OUT + '/' + name, JSON.stringify(data, null, 1) + '\n');
}

// ---- 比べるための RC 教師(ユーザーの dist/rc-speed-controller.js をそのまま使う) ----
// 教師は標準の物理(約束の設定)で作り、sim だけ simOverrides で変える(教師も知らない = 方策と同じ条件)
export const rcTeacherModule = await import(pathToFileURL(USER_SIM + '/dist/rc-speed-controller.js').href);
export function runTeacher({plan = STD_PLAN, offset = 0, yawDeg = 0, episodeMode = 'single', delay = 0, simOverrides = {}, steerGain = 1, speedGain = 1} = {}) {
  const nominal = browserConfig(plan, {episodeMode});
  if (episodeMode === 'single') nominal.timeLimit = 20;
  const active = activeWorld(SCENE_WORLD, 0, offset, yawDeg);
  const t = rcTeacherModule.createRcTrainingTeacher(active, nominal, {gate: 0, left_offset_m: offset, left_yaw_deg: yawDeg});
  const sim = core.createSimulation(active, {...nominal, ...simOverrides});
  let snap = sim.reset({seed: nominal.seed, scenario: plan.scenario}), queued = null;
  while (!snap.state.terminal) {
    let c = t.action(snap);
    c = {steering: c.steering * steerGain, speed: c.speed * speedGain};
    if (delay === 1) { const a = queued ?? {steering: 0, speed: 0}; queued = c; c = a; }
    snap = sim.step(c);
  }
  return {summary: summarize(sim.report())};
}
