// RL用の高速版 sim-core。元の dist/sim-core.js は変更せず、読み込み時に次の3点だけを
// 文字列置換して data: URL から読み込む(load-course-world.mjs と同じ方式)。
//  1) 衝突判定の近傍絞り込み: 1フレームの開始時に、そのフレーム内で車体が届き得る範囲
//     (車体外接円半径 + maxSpeed*dt + 障害物の大きさ + 余裕5 cm)にある線分・柱だけを判定する。
//     届かない障害物を除くだけなので、判定結果は元と同一(probe/fast_equivalence.mjs で全軌跡一致を確認)。
//  2) step(action, true) で戻り値の深いコピー(snapshot)を省略できる。
//  3) peek() で state / metrics / config を参照で返す(書き換え禁止・読み取り専用で使う)。
// 物理・ゲート・黒シート判定・時間刻み・車体寸法は一切変更しない。
import {readFile} from 'node:fs/promises';

const source = await readFile(new URL('./dist/sim-core.js', import.meta.url), 'utf8');
function replaceOnce(text, from, to) {
  const first = text.indexOf(from);
  if (first < 0 || text.indexOf(from, first + from.length) >= 0) throw new Error('fast-sim-core: 置換箇所が1つではありません: ' + from.slice(0, 60));
  return text.slice(0, first) + to + text.slice(first + from.length);
}
let patched = source;
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
  function peek() { return { state, metrics, config }; }
  return { reset, step, observe, snapshot, report, demoAction, setPaused, peek, world, targets };`);
const module = await import(`data:text/javascript;base64,${Buffer.from(patched).toString('base64')}`);
export const createSimulation = module.createSimulation;
export const DEFAULTS = module.DEFAULTS;
export const validateInitialState = module.validateInitialState;
export const normalizeLongitudinalConfig = module.normalizeLongitudinalConfig;
