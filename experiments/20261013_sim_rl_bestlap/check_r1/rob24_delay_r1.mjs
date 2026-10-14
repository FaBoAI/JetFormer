// ROB24 をそのまま作り直す + 遅延 1 の「前回の行動」の約束の違いで周回時間がどう変わるか(確認の段 r1)
import {loadOwnPolicy, runClosedLoop, browserConfig, STD_PLAN, activeWorld, SCENE_WORLD, core, features, toCommand, summarize, writeJson} from './common_r1.mjs';
const policy = loadOwnPolicy();
const ROB24 = [...[[-.1, -6], [-.1, 6], [-.05, -6], [-.05, 6], [.05, -6], [.05, 6], [.1, -6], [.1, 6]].map(([o, y]) => ({gate: 0, o, y})),
  ...[4, 8, 12, 16, 20, 24, 28, 31].flatMap(g => [{gate: g, o: .05, y: 3}, {gate: g, o: -.05, y: -3}])];
const rows = [];
for (const c of ROB24) {
  let g = c.gate, r;
  for (let k = 0; k < 33; k++) { r = runClosedLoop(policy, {gate: g, offset: c.o, yawDeg: c.y, episodeMode: 'continuous', keepReport: false}); if (!r.invalid_spawn) break; g = (g + 1) % 33; }
  rows.push({req_gate: c.gate, used_gate: g, offset: c.o, yaw: c.y, valid: r.summary.valid, laps: r.summary.laps.map(l => +l.elapsed.toFixed(4)), reason: r.summary.reason});
}
console.error('ROB24 valid', rows.filter(r => r.valid).length, '/', rows.length, 'moved', rows.filter(r => r.req_gate !== r.used_gate).map(r => `${r.req_gate}->${r.used_gate}`));
// 遅延 1: 観測の「前回の行動」を (a) 方策が今出した値 / (b) 実際に適用した(1 フレーム前の)指令 にする
function delayRun(convention) {
  const cfg = browserConfig(STD_PLAN, {episodeMode: 'single'}), active = activeWorld(SCENE_WORLD);
  const sim = core.createSimulation(active, cfg);
  let snap = sim.reset({seed: cfg.seed, scenario: 'baseline'});
  const tr = features.createFeatureTracker(active, cfg); tr.resetProgress(snap.state.x, snap.state.z, 0);
  const obs = new Float64Array(79); tr.observe(snap.state, 0, 0, 0, obs);
  let queued = null;
  while (!snap.state.terminal) {
    const a = policy(obs), cmd = toCommand(a[0], a[1]);
    const applied = queued ?? {steering: 0, speed: 0}; queued = cmd;
    snap = sim.step(applied);
    tr.advanceProgress(snap.state.x, snap.state.z);
    const p = convention === 'fresh' ? a : [applied.steering / .52, applied.speed / 4.5 * 2 - 1];
    tr.observe(snap.state, snap.metrics.route_index, p[0], p[1], obs);
  }
  return summarize(sim.report());
}
const d = {fresh_output: delayRun('fresh'), applied_command: delayRun('applied')};
console.error('delay1 fresh', JSON.stringify(d.fresh_output.laps), 'applied', JSON.stringify(d.applied_command.laps));
writeJson('rob24_delay.json', {schema: 'rl-check-r1-rob24-delay/v1', rob24: {valid: rows.filter(r => r.valid).length, n: rows.length, rows}, delay1_prev_action_convention: d});
