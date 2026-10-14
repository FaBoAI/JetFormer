// 追加の障害物(柱・箱)を world に足せるか、sim の衝突・方策の観測(レイ・d_min)・sim.observe() に見えるか、
// いまのベストラップの方策(障害物なしで学習)がそれを避けるか、を確かめる。
//   node probe/obstacles.mjs
import {features, envCore, loadBestPolicy, runPolicyEpisode, writeJson, cloneWorld} from './common.mjs';

const {loopPointAt, prepareGeometry, createFeatureTracker, OBS_DIM, OBS_INDEX} = features;
const base = envCore.loadBaseWorld();
const {act, sha256} = loadBestPolicy();
const pt = {x: 0, z: 0, tx: 0, tz: 0};
function onR(s, off = 0) { loopPointAt(s, pt); return {x: pt.x + off * pt.tz, z: pt.z - off * pt.tx, yaw: Math.atan2(-pt.tz, pt.tx)}; }
function withDisks(disks) { const w = cloneWorld(base); w.disks = [...w.disks, ...disks]; return w; }

const results = {schema: 'sim-rl-modes-probe-obstacles/v1', policy_sha256: sha256, checks: {}, runs: []};

// 1. sim(高速版・元)に足した柱が衝突として数えられるか(車を柱へまっすぐ走らせる)
// 柱は開始(x −0.85, z 2.1、+x 向き)の正面 1.35 m(x 0.5, z 2.1)に置く
for (const coreKind of ['fast', 'original']) {
  const p = {x: .5, z: 2.1};
  const w = withDisks([{x: p.x, z: p.z, radius: .06, height: .3, kind: 'post'}]);
  const core = await envCore.loadCore(coreKind);
  const sim = core.createSimulation({...w, spawn: {x: -0.85, z: 2.1, yaw: 0}}, envCore.simConfig({episodeMode: 'continuous'}));
  sim.reset({seed: 1});
  let snap;
  for (let f = 0; f < 120; f++) { snap = sim.step({steering: 0, speed: 1.5}); if (coreKind === 'fast') snap = sim.snapshot(); if (snap.state.terminal) break; }
  const ev = snap.metrics.collision_events?.[0];
  results.checks[`collision_${coreKind}`] = {reason: snap.state.reason, collider: ev?.collider, kind: ev?.kind, time: ev?.time, n_disks: w.disks.length};
  // sim.observe() のレイ(高さ 8 cm 以上の障害物だけ)
  if (coreKind === 'original') {
    const sim2 = core.createSimulation({...w, spawn: {x: -0.85, z: 2.1, yaw: 0}}, envCore.simConfig({episodeMode: 'continuous'}));
    sim2.reset({seed: 1});
    const o = sim2.observe();
    const sim3 = core.createSimulation({...base, spawn: {x: -0.85, z: 2.1, yaw: 0}}, envCore.simConfig({episodeMode: 'continuous'}));
    sim3.reset({seed: 1});
    const o3 = sim3.observe();
    results.checks.sim_observe_center_ray = {with_post: o.ranges[6].distance, without: o3.ranges[6].distance, rayCount: o.ranges.length};
  }
}
// 1b. 高速版で、作ったあとに world.disks へ足しても衝突には効かない(近傍の表 diskReach は作ったときのもの)ことの確認
{
  const core = await envCore.loadCore('fast');
  const sim = core.createSimulation({...base, spawn: {x: -0.85, z: 2.1, yaw: 0}}, envCore.simConfig({episodeMode: 'continuous'}));
  sim.reset({seed: 1});
  const p = {x: .5, z: 2.1};
  sim.world.disks.push({id: 'late', x: p.x, z: p.z, radius: .06, height: .3, kind: 'post'});
  let snap;
  for (let f = 0; f < 120; f++) { sim.step({steering: 0, speed: 1.5}, true); snap = sim.peek(); if (snap.state.terminal) break; }
  results.checks.fast_late_push = {reason: snap.state.reason, collisions: snap.metrics.collisions, frame: snap.state.frame, collider: snap.metrics.collision_events[0]?.collider, collision_time: snap.metrics.collision_events[0]?.time,
    note: '高速版の sim.world.disks に後から足した柱は nearObstacles の diskReach に無いので判定されない(作り直しが要る)'};
}
// 2. 方策の観測(features.mjs)に見えるか: ゲート 0 の標準の開始で、正面 1.35 m と、右 6°(レイの間)の 1.35 m に柱
for (const [label, p] of [['ahead', {x: .5, z: 2.1}], ['between_rays_right6deg', {x: -.85 + 1.35 * Math.cos(6 * Math.PI / 180), z: 2.1 + 1.35 * Math.sin(6 * Math.PI / 180)}]]) {
  const w = withDisks([{x: p.x, z: p.z, radius: .06, height: .3, kind: 'post'}]);
  const mk = world => {
    const {world: aw} = envCore.activeWorldFor(world, 0, 0, 0);
    const tr = createFeatureTracker(aw, envCore.PHYSICS);
    tr.resetProgress(aw.spawn.x, aw.spawn.z, 0);
    const o = new Float64Array(OBS_DIM), aux = {};
    tr.observe({x: aw.spawn.x, z: aw.spawn.z, yaw: aw.spawn.yaw, speed: 0, steering: 0}, 0, 0, 0, o, aux);
    return {rays: Array.from(o.slice(OBS_INDEX.rays, OBS_INDEX.rays + 19), v => +(v * 3).toFixed(3)), d_min: aux.d_min,
      lookahead_unchanged: Array.from(o.slice(8, 40))};
  };
  const a = mk(base), b = mk(w);
  results.checks['features_see_post_' + label] = {rays_without: a.rays, rays_with: b.rays, center_ray_with: b.rays[9], d_min_without: a.d_min, d_min_with: b.d_min,
    lookahead_same: JSON.stringify(a.lookahead_unchanged) === JSON.stringify(b.lookahead_unchanged)};
}
// 3. いまの方策は避けるか: R の上(と少し横)に柱を置いて、STD の連続 20 秒
const placements = [
  {name: 'straight-s1.0-on-R', s: 1.0, off: 0}, {name: 'straight-s2.0-left0.15', s: 2.0, off: .15}, {name: 'straight-s2.0-right0.15', s: 2.0, off: -.15},
  {name: 'turf-s10.8-on-R', s: 10.8, off: 0}, {name: 'back-s21.0-on-R', s: 21.0, off: 0}, {name: 'final-s25.4-on-R', s: 25.4, off: 0},
];
for (const pl of placements) {
  const p = onR(pl.s, pl.off), w = withDisks([{x: p.x, z: p.z, radius: .05, height: .3, kind: 'post'}]);
  const r = await runPolicyEpisode({act, baseWorld: w});
  const s = r.summary;
  results.runs.push({placement: pl, post: {x: p.x, z: p.z, r: .05}, reason: s.sim_reason, frames: s.frames, laps: s.completed_laps, valid_laps: s.valid_laps,
    collisions: s.collisions, collider: s.collision_events?.[0]?.collider, min_clearance: s.min_clearance});
}
// 基準(柱なし)
{ const r = await runPolicyEpisode({act}); results.runs.push({placement: 'none', reason: r.summary.sim_reason, laps: r.summary.completed_laps,
  valid_laps: r.summary.valid_laps, lap_times: r.summary.laps.map(l => l.elapsed), min_clearance: r.summary.min_clearance}); }
// 4. 障害物が増えたときの 1 ステップの時間(観測のレイ・すき間)
{
  const many = [];
  for (let k = 0; k < 20; k++) { const p = onR(1 + k * 1.2, (k % 3 - 1) * .3); many.push({x: p.x, z: p.z, radius: .05, height: .3, kind: 'post'}); }
  const t0 = performance.now(); const r0 = await runPolicyEpisode({act}); const t1 = performance.now();
  const r1 = await runPolicyEpisode({act, baseWorld: withDisks(many)}); const t2 = performance.now();
  results.checks.step_cost = {base_ms_per_frame: (t1 - t0) / r0.summary.frames, plus20_posts_ms_per_frame: (t2 - t1) / r1.summary.frames,
    note: '方策の計算(MLP 79-256-256-2)込み、1 プロセス'};
}
const out = writeJson('obstacles.json', results);
console.log(JSON.stringify(results, null, 1).slice(0, 6000));
console.log('書いた:', out);
