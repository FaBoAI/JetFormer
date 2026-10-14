// r3 の経路いろいろ(line_r3、2026-10-14 学習の段): 共有の rl/ の写し。違いは読み込む先(走行中の仕組みは r3/line/rt = r3/runtime の固定の写し、
// snapshot は絶対パス)だけ。元の sha256 は r3/line/sources_sha256_line_r3.txt。
// 学習・試験用の高速版 sim-core(2026-10-13、sim_rl_modes)。snapshot/sim/dist/sim-core.js(ユーザーの元のファイルとバイト単位で同じ写し)を
// 読み込み時に文字列で置き換えて data: URL から読む。元のファイルは変えない。
//  1)〜3) ベストラップの sim/fast-sim-core.mjs と同じ置き換え(近傍の障害物だけ判定・step(action, true) で写しを省く・peek())。
//  4) 摩擦円(modes.json dr_specs.conventions): config に frictionCircle: true を渡した回だけ、縦の加減速を √((μg)² − a_lat²) までにする
//     (check_r1/own_physics_r1.mjs の frictionCircle と同じ式。sim-core の normalizedConfig は知らない鍵を捨てるので、作るときの引数から読む)。
//     frictionCircle を渡さない回は、ベストラップの高速版とビット単位で同じ(試験 tests/core.test.mjs)。
import {readFile} from 'node:fs/promises';
import crypto from 'node:crypto';

const coreUrl = new URL('file:///home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/sim/dist/sim-core.js');
const source = await readFile(coreUrl, 'utf8');
export const SIM_CORE_SHA256 = crypto.createHash('sha256').update(source).digest('hex');
if (SIM_CORE_SHA256 !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('fast_core_modes: sim-core.js の sha256 が違います');
function replaceOnce(text, from, to) {
  const first = text.indexOf(from);
  if (first < 0 || text.indexOf(from, first + from.length) >= 0) throw new Error('fast_core_modes: 置換箇所が1つではありません: ' + from.slice(0, 60));
  return text.slice(0, first) + to + text.slice(first + from.length);
}
let patched = source;
// ---- 1)〜3) ベストラップの高速版と同じ ----
patched = replaceOnce(patched, 'function step(action = {}) {', 'function step(action = {}, lite = false) {');
patched = replaceOnce(patched, '    while (elapsed < remainingTime-1e-12 && !state.terminal) {',
  `    const nearWorld = nearObstacles(state.x, state.z);
    while (elapsed < remainingTime-1e-12 && !state.terminal) {`);
patched = replaceOnce(patched, '      const hit = collisionAt(trial, world, config);', '      const hit = collisionAt(trial, nearWorld, config);');
patched = replaceOnce(patched,
  "    trace.push({ frame: state.frame, time: state.sim_time, x: state.x, z: state.z, yaw: state.yaw, speed: state.speed, steering: state.steering, surface: state.surface });\n    return snapshot();",
  "    trace.push({ frame: state.frame, time: state.sim_time, x: state.x, z: state.z, yaw: state.yaw, speed: state.speed, steering: state.steering, surface: state.surface });\n    return lite ? null : snapshot();");
patched = replaceOnce(patched, '  return { reset, step, observe, snapshot, report, demoAction, setPaused };',
  `  // 近傍の障害物だけを返す。1フレームの移動量は |speed|<=maxSpeed なので maxSpeed*dt 以下。
  const reachBase = vehicleRadius + config.maxSpeed * config.dt + .05;
  const segmentReach = world.segments.map(s => (reachBase + s.radius) ** 2);
  const diskReach = world.disks.map(d => (reachBase + (d.shape === 'rectangle' ? Math.hypot(d.w, d.d) / 2 : d.radius)) ** 2);
  function nearObstacles(x, z) {
    const p = [x, z], segments = [], disks = [];
    for (let i = 0; i < world.segments.length; i++) if (pointSegmentDistanceSquared(p, world.segments[i].a, world.segments[i].b) <= segmentReach[i]) segments.push(world.segments[i]);
    for (let i = 0; i < world.disks.length; i++) { const d = world.disks[i]; if ((d.x - x) ** 2 + (d.z - z) ** 2 <= diskReach[i]) disks.push(d); }
    return { segments, disks, bounds: world.bounds };
  }
  function peek() { return { state, metrics, config, friction_circle: frictionCircle, friction_circle_limited: frictionCircleLimited }; }
  return { reset, step, observe, snapshot, report, demoAction, setPaused, peek, world, targets };`);
// ---- 4) 摩擦円(作るときの引数 frictionCircle: true のときだけ) ----
patched = replaceOnce(patched, 'export function createSimulation(worldInput, configInput = {}) {\n  const world = normalizedWorld(worldInput), config = normalizedConfig(configInput);',
  'export function createSimulation(worldInput, configInput = {}) {\n  const world = normalizedWorld(worldInput), config = normalizedConfig(configInput);\n  const frictionCircle = !!(configInput && configInput.frictionCircle === true); let frictionCircleLimited = 0;');
patched = replaceOnce(patched,
  `      const rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      const limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      const yawRate = clamp(rawRate, -limit, limit);`,
  `      let rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      let limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01);
      let yawRate = clamp(rawRate, -limit, limit);
      if (frictionCircle) {
        const aLat = Math.abs(nextSpeed*yawRate), avail = Math.sqrt(Math.max(0, (surface.mu*9.80665)**2-aLat**2)), aLong = (nextSpeed-state.speed)/h;
        if (Math.abs(aLong) > avail) {
          nextSpeed = state.speed+Math.sign(aLong)*avail*h;
          rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering); limit = surface.mu*9.80665/Math.max(Math.abs(nextSpeed), .01); yawRate = clamp(rawRate, -limit, limit);
          frictionCircleLimited++;
        }
      }`);
const module = await import(`data:text/javascript;base64,${Buffer.from(patched).toString('base64')}`);
export const createSimulation = module.createSimulation;
export const DEFAULTS = module.DEFAULTS;
export const validateInitialState = module.validateInitialState;
export const normalizeLongitudinalConfig = module.normalizeLongitudinalConfig;
