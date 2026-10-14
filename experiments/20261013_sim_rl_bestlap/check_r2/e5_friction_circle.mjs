#!/usr/bin/env node
// 確認 r2 の 5: 摩擦円の反事実(縦と横の合成を μg まで)での単周。RL の段の主張「r2 の 100〜400 万は完走、500 万は衝突」の照合。
//   node check_r2/e5_friction_circle.mjs
// sim-core.js は縦と横を別々に頭打ちにするので、反事実は確認 r1 の自前の積分(check_r1/own_physics_r1.mjs の frictionCircle)を使う
// (RL の段も同じ積分を使った。ここでは閉ループ・方策・観測を自前で組む)。あわせて、frictionCircle なしの同じ積分で sim と同じ周になることも確かめる。
import {loadMlp, browserConfig, startWorld, STD_PLAN, features, MAX_STEER, SPEED_CAP, writeJson, runPolicy, core} from './common_r2.mjs';
import {createOwnSim} from '../check_r1/own_physics_r1.mjs';

const keys = ['selected', 'r2_1M', 'r2_2M', 'r2_3M', 'r2_4M', 'r2_5M'];
const out = {schema: 'rl-check-r2-friction-circle/v1', created_at: new Date().toISOString(), rows: []};
for (const key of keys) {
  const pol = loadMlp(key);
  for (const fc of [false, true]) {
    const world = startWorld(0, 0, 0);
    const raw = browserConfig(STD_PLAN, 'single'); raw.timeLimit = 15;
    const cfg = core.createSimulation(world, raw).report().config;   // 自前の積分には sim-core が正規化した設定(substep・gateHalfWidth を含む)を渡す
    const own = createOwnSim(world, cfg, {frictionCircle: fc, interp: 4});
    const tr = features.createFeatureTracker(world, cfg), obs = new Float64Array(79);
    let s = own.state; tr.resetProgress(s.x, s.z, s.route_index); tr.observe(s, s.route_index, 0, 0, obs);
    let n = 0;
    while (!own.state.terminal && n++ < 15 * 30 + 5) {
      const a = pol(obs);
      own.step({steering: MAX_STEER * a[0], speed: SPEED_CAP * (a[1] + 1) / 2});
      s = own.state; tr.advanceProgress(s.x, s.z); tr.observe(s, s.route_index, a[0], a[1], obs);
    }
    const r = own.result();
    const row = {policy: key, friction_circle: fc, reason: r.reason, laps: r.lap_events.map(l => +l.elapsed.toFixed(4)), gates: r.passed_gates,
      first_collision: r.collisions_own[0] ?? null, min_clearance_m: r.stats.minClear};
    if (!fc) row.sim_core_single = runPolicy(pol, {mode: 'single'}).summary.laps;
    out.rows.push(row);
    console.log(key, fc ? '摩擦円' : '摩擦円なし', row.reason, JSON.stringify(row.laps), row.gates, row.sim_core_single ? 'sim ' + JSON.stringify(row.sim_core_single) : '', row.first_collision ? JSON.stringify(row.first_collision) : '');
  }
}
writeJson('e5_friction_circle.json', out);
