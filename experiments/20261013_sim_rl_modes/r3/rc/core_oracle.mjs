// 離脱復活 r3(2026-10-14、学習の段 recover r3): 立て直せる表を作るための sim-core。
// ユーザーの元の sim-core.js(sha256 aaeb2379…549e)を読み込み時に文字列で置き換えて data: URL から読む(元のファイルは変えない)。
// 置き換えは rl/fast_core_modes.mjs の 1)〜3)(近くの障害物だけ判定する・step(action, true) で写しを省く・peek())と同じで物理は 1 文字も変えない。
// 足したのは状態の保存と戻し(__save・__restore: state・metrics・接触の履歴・周の始まりの値)だけ。枝分かれの走りを「正確な状態の写し」から始めるため。
// 元の sim-core.js と同じ走りになることは build_recoverable_table.mjs の最初の自己試験で確かめる(軌跡がビット単位で同じ)。
import {readFile} from 'node:fs/promises';
import crypto from 'node:crypto';

const CORE = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5/dist/sim-core.js';
const source = await readFile(CORE, 'utf8');
export const SIM_CORE_SHA256 = crypto.createHash('sha256').update(source).digest('hex');
if (SIM_CORE_SHA256 !== 'aaeb23798ee4b85f31be6bedf763a519d910ffc76a39f1b85ddaa83fd854549e') throw new Error('core_oracle: sim-core.js の sha256 が違います');
function replaceOnce(text, from, to) {
  const first = text.indexOf(from);
  if (first < 0 || text.indexOf(from, first + from.length) >= 0) throw new Error('core_oracle: 置換箇所が 1 つではありません: ' + from.slice(0, 60));
  return text.slice(0, first) + to + text.slice(first + from.length);
}
let p = source;
p = replaceOnce(p, 'function step(action = {}) {', 'function step(action = {}, lite = false) {');
p = replaceOnce(p, '    while (elapsed < remainingTime-1e-12 && !state.terminal) {',
  `    const nearWorld = nearObstacles(state.x, state.z);
    while (elapsed < remainingTime-1e-12 && !state.terminal) {`);
p = replaceOnce(p, '      const hit = collisionAt(trial, world, config);', '      const hit = collisionAt(trial, nearWorld, config);');
p = replaceOnce(p,
  "    trace.push({ frame: state.frame, time: state.sim_time, x: state.x, z: state.z, yaw: state.yaw, speed: state.speed, steering: state.steering, surface: state.surface });\n    return snapshot();",
  "    if (!lite) trace.push({ frame: state.frame, time: state.sim_time, x: state.x, z: state.z, yaw: state.yaw, speed: state.speed, steering: state.steering, surface: state.surface });\n    return lite ? null : snapshot();");
p = replaceOnce(p, '    actionLog.push({ frame: state.frame, steering, speed });', '    if (!lite) actionLog.push({ frame: state.frame, steering, speed });');
p = replaceOnce(p, '  return { reset, step, observe, snapshot, report, demoAction, setPaused };',
  `  const reachBase = vehicleRadius + config.maxSpeed * config.dt + .05;
  const segmentReach = world.segments.map(s => (reachBase + s.radius) ** 2);
  const diskReach = world.disks.map(d => (reachBase + (d.shape === 'rectangle' ? Math.hypot(d.w, d.d) / 2 : d.radius)) ** 2);
  function nearObstacles(x, z) {
    const p = [x, z], segments = [], disks = [];
    for (let i = 0; i < world.segments.length; i++) if (pointSegmentDistanceSquared(p, world.segments[i].a, world.segments[i].b) <= segmentReach[i]) segments.push(world.segments[i]);
    for (let i = 0; i < world.disks.length; i++) { const d = world.disks[i]; if ((d.x - x) ** 2 + (d.z - z) ** 2 <= diskReach[i]) disks.push(d); }
    return { segments, disks, bounds: world.bounds };
  }
  function peek() { return { state, metrics, config }; }
  function __save() { return copy({ state, metrics, priorContact, lapStartTime, lapStartDistance, lapStartFrame, lapStartCollisions }); }
  function __restore(o) {
    const c = copy(o);
    state = c.state; metrics = c.metrics; priorContact = c.priorContact; lapStartTime = c.lapStartTime; lapStartDistance = c.lapStartDistance;
    lapStartFrame = c.lapStartFrame; lapStartCollisions = c.lapStartCollisions; actionLog = []; trace = [];
  }
  return { reset, step, observe, snapshot, report, demoAction, setPaused, peek, __save, __restore };`);
export const PATCHED_SHA256 = crypto.createHash('sha256').update(p).digest('hex');
const mod = await import(`data:text/javascript;base64,${Buffer.from(p).toString('base64')}`);
export const createSimulation = mod.createSimulation;
export const original = await import(CORE);
