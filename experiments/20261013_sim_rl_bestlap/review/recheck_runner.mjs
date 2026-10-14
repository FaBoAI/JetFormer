// 確認の段(review、2026-10-14): ランダムな開始で、選んだ RL の方策に元の sim-core.js を走らせ、レポートを保存する
//
//   node recheck_runner.mjs <poses.json> <出力フォルダー>
//
// poses.json = [{name, gate, offset, yawDeg, mode: 'single'|'continuous'}]
// 方針(ここまでの段の道具を使わない):
//   - 物理・コース: ユーザーの展開物の dist/sim-core.js・qa/load-course-world.mjs をそのまま import(元のファイル。変えない)
//   - 開始: app.js reset() の式(経路を開始ゲートで回し、左へ offset・向き +turn)をここで書き直す
//   - 設定: qa/rc-collection-plan-60.json の標準の条件(ゲート 0・ずれ 0・baseline)の値。RL の約束どおり制御遅延 0
//   - 方策: policy.json の重みを自前の MLP で計算。観測は方策の入力の定義そのものなので rl/features.mjs だけを使う
// 合否はここでは決めない(review/independent_lap_check.py が Python の自前の積分・判定で数え直す)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

const USER_SIM = '/home/jetson/sim/extracted/course-simulator';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap';
const POLICY = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/best/main/policy.json';
const EXPECT = {
  policy: '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee',
  sim_core: 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e',
  features: '3cf941237cdce719ba963036f3f43cec67998b6c07e0295ef3a6d17b8ba38790',
};
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
for (const [k, p] of [['policy', POLICY], ['sim_core', USER_SIM + '/dist/sim-core.js'], ['features', EXP + '/rl/features.mjs']]) {
  if (sha(p) !== EXPECT[k]) throw new Error(`sha256 が違います: ${p}`);
}
const core = await import(pathToFileURL(USER_SIM + '/dist/sim-core.js').href);
const {loadCourseWorld} = await import(pathToFileURL(USER_SIM + '/qa/load-course-world.mjs').href);
const features = await import(pathToFileURL(EXP + '/rl/features.mjs').href);
const WORLD = await loadCourseWorld();
const PLAN60 = JSON.parse(fs.readFileSync(USER_SIM + '/qa/rc-collection-plan-60.json', 'utf8'));
const STD = PLAN60.find(c => c['start-gate'] === 0 && c['spawn-offset'] === 0 && c['spawn-yaw'] === 0 && c.scenario === 'baseline');

// app.js reset() の設定(値は計画の標準の条件。種は baseline なので結果に効かない)
function configFor(mode) {
  const v = STD;
  const c = {seed: 1, dt: 1 / 30, timeLimit: mode === 'single' ? 15 : v.limit, maxSpeed: v.maxspeed, maxSteer: v.maxsteer,
    acceleration: v.acceleration, steerRate: v['steer-rate'], vehicleLength: v['car-length'], vehicleWidth: v['car-width'],
    wheelbase: v.wheelbase, stopOnCollision: v['stop-collision'], frictionMultiplier: 1, noiseStd: 0,
    longitudinalModel: v['longitudinal-model'], driveTau: v['drive-tau'], driveAccelerationCap: v['drive-acceleration-cap'],
    brakeDeceleration: v['brake-deceleration']};
  if (mode === 'continuous') c.episodeMode = 'continuous-laps-v1';
  return c;
}
// app.js reset() の開始の式
function startWorld(gate, offset, yawDeg) {
  const i = Math.max(0, Math.min(WORLD.route.length - 1, Math.trunc(gate) || 0));
  const route = [...WORLD.route.slice(i), ...WORLD.route.slice(0, i)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, offset || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  return {...WORLD, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}};
}
// 自前の MLP: x = clip((obs − mean)/std, ±clip) → 層(tanh)→ [−1, 1] に切る
const W = JSON.parse(fs.readFileSync(POLICY, 'utf8'));
function policy(obs) {
  let x = Array.from(obs, (o, i) => Math.max(-W.obs_clip, Math.min(W.obs_clip, (o - W.obs_mean[i]) / W.obs_std[i])));
  for (const L of W.layers) {
    const y = new Array(L.n_out);
    for (let o = 0; o < L.n_out; o++) {
      let s = L.b[o];
      for (let i = 0; i < L.n_in; i++) s += L.w[o * L.n_in + i] * x[i];
      y[o] = L.act === 'tanh' ? Math.tanh(s) : s;
    }
    x = y;
  }
  return [Math.max(-1, Math.min(1, x[0])), Math.max(-1, Math.min(1, x[1]))];
}

function run(p) {
  const config = configFor(p.mode);
  const world = startWorld(p.gate, p.offset, p.yawDeg);
  const sim = core.createSimulation(world, config);
  let snap = sim.reset({seed: config.seed, scenario: 'baseline'});
  const tracker = features.createFeatureTracker(world, config);
  tracker.resetProgress(snap.state.x, snap.state.z, snap.metrics.route_index);
  const obs = new Float64Array(79);
  tracker.observe(snap.state, snap.metrics.route_index, 0, 0, obs);
  const policyOut = [];
  while (!snap.state.terminal) {
    const a = policy(obs);
    policyOut.push(a);
    snap = sim.step({steering: .52 * a[0], speed: 4.5 * (a[1] + 1) / 2});   // 遅延 0(RL の約束)
    tracker.advanceProgress(snap.state.x, snap.state.z);
    tracker.observe(snap.state, snap.metrics.route_index, a[0], a[1], obs);
  }
  const report = sim.report();
  report.world = world;
  report.context = {controller: 'rl-policy-review', start_variation: {gate: p.gate, left_offset_m: p.offset, left_yaw_deg: p.yawDeg},
    policy_sha256: EXPECT.policy, control_delay_frames: 0};
  report.policy_output = policyOut;
  return report;
}

const [posesPath, outDir] = process.argv.slice(2);
const poses = JSON.parse(fs.readFileSync(posesPath, 'utf8'));
fs.mkdirSync(outDir, {recursive: true});
const rows = [];
for (const p of poses) {
  const r = run(p);
  const file = path.join(outDir, `${p.name}.json`);
  fs.writeFileSync(file, JSON.stringify(r));
  const m = r.metrics;
  const row = {name: p.name, file, reason: r.outcome.reason, success: r.outcome.success, frames: r.actions.length, elapsed: m.elapsed,
    passed_gates: m.passed_gates, collisions: m.collisions, crossings: m.required_crossing_events.length,
    laps: (m.lap_events ?? []).map(l => l.elapsed)};
  rows.push(row);
  console.log(JSON.stringify(row));
}
fs.writeFileSync(path.join(outDir, 'runner_summary.json'), JSON.stringify(rows, null, 1));
