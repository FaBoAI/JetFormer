// 反事実: 摩擦円(縦と横の合成を μg まで)を入れた自前の物理で、RL と RC 教師を閉ループで走らせる(確認の段 r1)
// どちらもこの変更を知らない(RL の観測・教師の参照は標準の物理のまま)。sim-core.js は変えていない(自前の積分の選択肢)。
import {loadOwnPolicy, browserConfig, STD_PLAN, activeWorld, SCENE_WORLD, core, features, toCommand, rcTeacherModule, writeJson} from './common_r1.mjs';
import {createOwnSim} from './own_physics_r1.mjs';
const policy = loadOwnPolicy();
function cfgFor(mode) { return core.createSimulation(activeWorld(SCENE_WORLD), browserConfig(STD_PLAN, {episodeMode: mode})).snapshot().config; }
function run(who, {frictionCircle, mode = 'single'}) {
  const cfg = cfgFor(mode), active = activeWorld(SCENE_WORLD);
  const own = createOwnSim(active, cfg, {frictionCircle, interp: 4});
  let ctrl;
  if (who === 'rl') {
    const tr = features.createFeatureTracker(active, cfg), obs = new Float64Array(79);
    let st = own.state; tr.resetProgress(st.x, st.z, st.route_index); tr.observe(st, st.route_index, 0, 0, obs);
    ctrl = {act: () => { const a = policy(obs); ctrl.last = a; return toCommand(a[0], a[1]); },
      after: () => { const s = own.state; tr.advanceProgress(s.x, s.z); tr.observe(s, s.route_index, ctrl.last[0], ctrl.last[1], obs); }};
  } else {
    const t = rcTeacherModule.createRcTrainingTeacher(active, cfg, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
    ctrl = {act: () => { const s = own.state; return t.action({state: s, metrics: {route_index: s.route_index}, config: cfg}); }, after: () => {}};
  }
  let n = 0;
  while (!own.state.terminal && n++ < 20 * 30 + 5) { own.step(ctrl.act()); ctrl.after(); }
  const r = own.result();
  return {who, frictionCircle, mode, reason: r.reason, laps: r.lap_events.map(l => +l.elapsed.toFixed(4)), gates: r.passed_gates, collisions: r.collisions_own.length,
    first_collision: r.collisions_own[0] ?? null, friction_circle_limited_frac: (r.stats.friction_circle_limited ?? 0) / r.stats.substeps, min_clearance_m: r.stats.minClear};
}
const rows = [];
for (const who of ['rl', 'teacher']) for (const fc of [false, true]) for (const mode of ['single', 'continuous']) {
  const r = run(who, {frictionCircle: fc, mode}); rows.push(r);
  console.error(who, 'friction_circle', fc, mode, r.reason, JSON.stringify(r.laps), 'coll', r.collisions, JSON.stringify(r.first_collision), 'limited', r.friction_circle_limited_frac.toFixed(3));
}
writeJson('friction_circle.json', {schema: 'rl-check-r1-friction-circle/v1', note: '自前の積分の反事実(摩擦円)。sim-core.js は変えていない', rows});
