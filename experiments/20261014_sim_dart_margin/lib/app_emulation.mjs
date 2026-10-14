// ブラウザのアプリ(dist/app.js の reset と連続試験)と同じ手順で、Node 上にシミュレーションと教師を作る(ベストラップの lib/app_emulation.mjs の写し。SIM_ROOT だけ今日のコピーに変えた)
//
//  - コース: コピーした sim/qa/load-course-world.mjs(ユーザーの qa と同じ内容)で、画面と同じ world を作る
//  - 物理: コピーした sim/dist/sim-core.js(ユーザーの dist とバイト単位で同じ)。高速版ではなく元のコアを使う
//  - 設定: app.js の reset() の config の作り方・開始ゲートの回転・開始ずれ(±0.3m・±25°に丸める)をそのまま写す
// ユーザーのファイルは読むだけで変更しない。
import path from 'node:path';
import {pathToFileURL} from 'node:url';

export const SIM_ROOT = '/home/jetson/jetformer_reports/sim_dart_20261014/sim';
const url = rel => pathToFileURL(path.join(SIM_ROOT, rel)).href;
const core = await import(url('dist/sim-core.js'));
const {loadCourseWorld} = await import(url('qa/load-course-world.mjs'));
export const {createSimulation} = core;

let cachedWorld = null;
export async function baseWorld() {
  if (!cachedWorld) cachedWorld = await loadCourseWorld();
  return JSON.parse(JSON.stringify(cachedWorld));
}

// app.js の longitudinalConfig() と reset() の config(計画の1条件から)
export function configFromCondition(c) {
  const scenario = c.scenario ?? 'baseline';
  const config = {seed: c.seed, dt: 1 / 30, timeLimit: c.limit, maxSpeed: c.maxspeed, maxSteer: c.maxsteer, acceleration: c.acceleration,
    steerRate: c['steer-rate'], vehicleLength: c['car-length'], vehicleWidth: c['car-width'], wheelbase: c.wheelbase,
    stopOnCollision: c['stop-collision'], frictionMultiplier: scenario === 'low-friction' ? .4 : 1, noiseStd: scenario === 'noise' ? .025 : 0};
  const model = c['longitudinal-model'] ?? 'constant-acceleration-v1';
  Object.assign(config, core.normalizeLongitudinalConfig(model === 'real-drive-response-estimate-v1'
    ? {longitudinalModel: model, driveTau: c['drive-tau'] ?? .4, driveAccelerationCap: c['drive-acceleration-cap'] ?? 6, brakeDeceleration: c['brake-deceleration'] ?? 1.2}
    : {longitudinalModel: model}));
  if ((c['episode-mode'] ?? 'single-lap') === 'continuous-laps-v1') config.episodeMode = 'continuous-laps-v1';
  return {config, scenario};
}

// app.js の reset(): 開始ゲートで経路を回転し、開始ずれを左右・向きに加える
export function activeWorldFor(world, gate = 0, offsetM = 0, yawDeg = 0) {
  const startIndex = Math.max(0, Math.min(world.route.length - 1, Math.trunc(gate) || 0));
  const route = [...world.route.slice(startIndex), ...world.route.slice(0, startIndex)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const offset = Math.max(-.3, Math.min(.3, offsetM || 0)), turn = Math.max(-25, Math.min(25, yawDeg || 0)) * Math.PI / 180;
  const active = {...world, route, spawn: {x: first[0] + offset * Math.sin(heading), z: first[1] - offset * Math.cos(heading), yaw: heading + turn}};
  return {world: active, variation: {gate: startIndex, left_offset_m: offset, left_yaw_deg: turn * 180 / Math.PI}};
}

// 1 条件を最後まで走らせる(app.js の tick と同じ順: 教師の操作 → sim.step)
export async function runCondition(c, teacherModule, {world = null, onFrame = null} = {}) {
  const w = world ?? await baseWorld();
  const {config, scenario} = configFromCondition(c);
  const {world: active, variation} = activeWorldFor(w, c['start-gate'], c['spawn-offset'], c['spawn-yaw']);
  const sim = createSimulation(active, config);
  sim.reset({seed: config.seed, scenario});
  const teacher = teacherModule.createRcTrainingTeacher(active, sim.snapshot().config, variation);
  while (!sim.snapshot().state.terminal) {
    const before = sim.snapshot(), action = teacher.action(before);
    sim.step(action);
    if (onFrame) onFrame(before, action, sim.snapshot());
  }
  return {sim, teacher, report: sim.report(), world: active, variation, config};
}
