#!/usr/bin/env node
// r2(手順の外の参考の追加実験): 物理のずれへの強さを、方策ごと・教師で同じ条件に並べて測る。2026-10-13
//
// 独立な確認 r1 で、選んだ方策(main 種 1・1,000 万)が舵の速さ 1.0 rad/s・ブレーキ ≤ 1.0 m/s² などでぶつかると分かった。
// 「速く・ぶつからない」ための更新の候補(ドメインのランダム化で学習を続けた方策)が、どれだけ強くなり、どれだけ遅くなるかを測る。
//
//   node rl/r2/robust_eval.mjs --policy 名前=path.json [--policy …] [--teacher] [--core fast|original] [--sets grid,random,rec60,fc,rough] --out out.json
//
// 走らせ方(どれも観測は標準の物理で作る = 方策・教師は今の物理を知らない):
//   - 物理の値(舵の速さ・ブレーキ・時定数・加速の上限・μ の倍率)は sim の設定を上書きする(rl/env_core.mjs の perturb.physics)
//   - 遅延 1 は 1 フレーム前の指令を適用する(最初は中立)。観測の「前回の行動」は適用した行動(RL の段の eval_sets と同じ約束)
//   - ゲインは適用する指令に掛ける
// sets:
//   grid   : 条件の表(1 つずつ・組み合わせ)を STD の開始で、停止から 1 周(単周)と連続 20 秒の両方
//   random : 物理のずれをランダムに引いた 200 条件(学習の範囲 r2 から 100・学習の範囲の外を含む広い範囲から 100。種は学習と別)、連続 20 秒
//   rec60  : 記録の 60 条件(標準の物理・遅延 0)と、遅延 1
//   fc     : 摩擦円の反事実(確認 r1 の自前の積分 check_r1/own_physics_r1.mjs を読むだけ)
//   rough  : 操作の荒さ(STD の連続 20 秒・標準の物理)
import fs from 'node:fs';
import {RlEnv, loadCore, loadBaseWorld, loadRcTeacher, createRng, DR_SPECS, sampleDr, PERTURB_KEYS, PHYSICS} from '../env_core.mjs';
import {actionToCommand, createFeatureTracker, G} from '../features.mjs';
import {loadPolicy} from '../policy_forward.mjs';
import {STD, loadRec60} from '../eval_sets.mjs';

const args = process.argv.slice(2);
const all = n => args.flatMap((a, i) => a === n ? [args[i + 1]] : []);
const opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const coreKind = opt('--core', 'fast'), out = opt('--out'), sets = (opt('--sets', 'grid,random,rec60,fc,rough')).split(',');
const core = await loadCore(coreKind), baseWorld = loadBaseWorld();

const controllers = [];
for (const spec of all('--policy')) {
  const [name, p] = spec.split('=');
  const fn = await loadPolicy(p);
  controllers.push({name, kind: 'policy', path: p, make: () => ({act: env => { const a = fn(env.obs); return actionToCommand(a[0], a[1]); }})});
}
if (args.includes('--teacher')) {
  const T = await loadRcTeacher();
  controllers.push({name: 'RC 教師', kind: 'teacher', make: env => {
    // 教師は標準の物理の設定で作る(今の物理を知らない)
    const cfg = {...JSON.parse(JSON.stringify(env.config)), ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))};
    const t = T.createRcTrainingTeacher(env.world, cfg, env.variation);
    return {act: e => { const c = t.action(e.view); const a = actionToCommand(c.steering / .52, 2 * c.speed / 4.5 - 1); return {...a, steering: c.steering, speed: c.speed}; }};
  }});
}

// 1 回走らせる
const envs = new Map();
function envFor(episodeMode) {
  if (!envs.has(episodeMode)) envs.set(episodeMode, new RlEnv({core, coreKind, baseWorld, mode: 'eval', episodeMode, timeLimit: episodeMode === 'single' ? 15 : 20}));
  return envs.get(episodeMode);
}
function run(ctrl, {episodeMode = 'single', start = STD, physics = {}, delay = 0, steerGain = 1, speedGain = 1, rough = false}) {
  const env = envFor(episodeMode);
  if (ctrl.kind === 'teacher' && (start.gate !== 0 || Math.abs(start.left_offset_m) > .03 + 1e-12 || Math.abs(start.left_yaw_deg) > 3 + 1e-12)) return {skipped: '教師の約束の外'};
  if (!env.reset({...start, perturb: Object.keys(physics).length ? {physics} : null})) return {invalid_spawn: true};
  const c = ctrl.make(env);
  let queued = actionToCommand(0, -1);
  const R = rough ? {cmd: [], st: []} : null;
  while (!env.done) {
    const cmd = c.act(env);
    let ap = cmd;
    if (delay === 1) { ap = queued; queued = cmd; }
    env.stepCommand(ap.steering * steerGain, ap.speed * speedGain, ap.a0, ap.a1);
    if (R) { const s = env.view.state; R.cmd.push([ap.steering, ap.speed]); R.st.push([s.speed, s.steering, s.yaw, env.aux.mu]); }
  }
  const s = env.summary();
  const valid = episodeMode === 'single'
    ? s.sim_reason === 'route_complete' && s.collisions === 0 && s.passed_gates === 33 && s.laps[0]?.crossings === 1
    : s.sim_reason === 'collection_duration_complete' && s.collisions === 0 && s.laps.length >= 2 && s.laps.every(l => l.gates === 33 && l.crossings === 1 && l.collisions === 0);
  const e = s.collision_events?.[0];
  const r = {valid, reason: s.sim_reason, laps: s.laps.map(l => +l.elapsed.toFixed(6)), gates: s.passed_gates, collisions: s.collisions,
    min_clearance: +s.min_clearance.toFixed(4), first_collision: e ? {t: +e.time.toFixed(3), x: +e.x.toFixed(2), z: +e.z.toFixed(2), collider: e.collider ?? e.kind} : null};
  if (R) r.rough = roughness(R, env.config);
  return r;
}
const q = (a, p) => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.floor(p * (b.length - 1)))] : null; };
function roughness(R, cfg) {
  const dt = cfg.dt, moving = [];
  for (let i = 1; i < R.st.length; i++) if (R.st[i][0] > .3) moving.push(i);
  const dcmd = moving.map(i => Math.abs(R.cmd[i][0] - R.cmd[i - 1][0]) / dt);
  const lat = moving.map(i => { let dy = R.st[i][2] - R.st[i - 1][2]; dy = Math.atan2(Math.sin(dy), Math.cos(dy)); return Math.abs(R.st[i][0] * dy / dt); });
  const rho = moving.map(i => R.st[i][0] ** 2 * Math.abs(Math.tan(R.st[i][1])) / (cfg.wheelbase * R.st[i][3] * G));
  const n = moving.length;
  return {frames_moving: n, steer_edge_frac: moving.filter(i => Math.abs(R.cmd[i][0]) >= .494).length / n, steer_cmd_rate_p95: q(dcmd, .95),
    speed_cmd_zero_frac: moving.filter(i => R.cmd[i][1] <= .05).length / n, speed_cmd_max_frac: moving.filter(i => R.cmd[i][1] >= 4.45).length / n,
    lat_acc_p95: q(lat, .95), lat_acc_max: Math.max(...lat), rho_gt1_frac: rho.filter(x => x > 1).length / n};
}

// ---- 条件の表 ----
// in_range: 学習の範囲 r2(DR_SPECS.r2)の中か。外の値は「学習していない強さ」の確認
const R2 = DR_SPECS.r2;
const inR = (k, v) => v >= R2[k][0] - 1e-12 && v <= R2[k][1] + 1e-12;
const GRID = [{name: '標準', cond: {}}, {name: '遅延 1', cond: {delay: 1}}, {name: '遅延 1 + 前進ゲイン 1.01(ONNX の条件)', cond: {delay: 1, speedGain: 1.01}},
  ...[.8, .9, 1.0, 1.1, 1.3, 1.5, 1.6, 1.8].map(v => ({name: `舵の速さ ${v} rad/s`, cond: {physics: {steerRate: v}}, in_range: inR('steerRate', v)})),
  ...[.6, .7, .8, .9, 1.0, 1.1, 1.5, 2.0, 2.5].map(v => ({name: `ブレーキ ${v} m/s²`, cond: {physics: {brakeDeceleration: v}}, in_range: inR('brakeDeceleration', v)})),
  ...[.8, .85, .9, .95, 1.05, 1.1].map(v => ({name: `μ × ${v}`, cond: {physics: {frictionMultiplier: v}}, in_range: inR('frictionMultiplier', v)})),
  ...[.25, .3, .5, .6].map(v => ({name: `時定数 ${v} 秒`, cond: {physics: {driveTau: v}}, in_range: inR('driveTau', v)})),
  ...[3, 4].map(v => ({name: `加速の上限 ${v} m/s²`, cond: {physics: {driveAccelerationCap: v}}, in_range: inR('driveAccelerationCap', v)})),
  {name: '舵のゲイン 0.95', cond: {steerGain: .95}}, {name: '舵のゲイン 1.05', cond: {steerGain: 1.05}},
  {name: '前進ゲイン 0.97', cond: {speedGain: .97}}, {name: '前進ゲイン 1.03', cond: {speedGain: 1.03}},
  {name: '組み合わせ: 舵の速さ 1.0 + ブレーキ 1.0', cond: {physics: {steerRate: 1.0, brakeDeceleration: 1.0}}},
  {name: '組み合わせ: 舵の速さ 1.0 + 遅延 1', cond: {physics: {steerRate: 1.0}, delay: 1}},
  {name: '組み合わせ: μ × 0.9 + ブレーキ 1.0 + 遅延 1', cond: {physics: {frictionMultiplier: .9, brakeDeceleration: 1.0}, delay: 1}},
  {name: '組み合わせ: 舵 0.9・ブレーキ 0.8・μ 0.9・時定数 0.5・遅延 1(悪い側をすべて)', cond: {physics: {steerRate: .9, brakeDeceleration: .8, frictionMultiplier: .9, driveTau: .5}, delay: 1}},
];
// 学習の範囲の外を含む広い範囲(random の後半)
const WIDE = {nominal_fraction: 0, steerRate: [.7, 2.0], brakeDeceleration: [.6, 2.5], driveTau: [.25, .6], driveAccelerationCap: [3, 6],
  frictionMultiplier: [.8, 1.12], delay1_fraction: .5, steerGain: [.93, 1.07], speedGain: [.95, 1.05]};

const result = {schema: 'rl-bestlap-r2-robust-eval/v1', created_at: new Date().toISOString(), core: coreKind, sets,
  conventions: '観測は標準の物理。遅延 1 は 1 フレーム前の指令(最初は中立)、観測の前回の行動は適用した行動。ゲインは適用する指令に掛ける',
  dr_spec_r2: R2, wide_spec: WIDE, controllers: {}};
for (const ctrl of controllers) {
  const t0 = Date.now(), C = {path: ctrl.path ?? null};
  if (sets.includes('grid')) {
    C.grid = GRID.map(g => ({name: g.name, in_range: g.in_range ?? null, cond: g.cond, single: run(ctrl, {episodeMode: 'single', ...g.cond}), continuous: run(ctrl, {episodeMode: 'continuous', ...g.cond})}));
    const gs = C.grid.slice(1);
    C.grid_summary = {conditions: gs.length, single_valid: gs.filter(r => r.single.valid).length, continuous_valid: gs.filter(r => r.continuous.valid).length,
      both_valid: gs.filter(r => r.single.valid && r.continuous.valid).length};
  }
  if (sets.includes('random') && ctrl.kind === 'policy' || sets.includes('random') && ctrl.kind === 'teacher') {
    const res = {};
    for (const [label, spec, seed] of [['in_range_r2', {...R2, nominal_fraction: 0}, 7001], ['wide', WIDE, 7002]]) {
      const rng = createRng(seed), rows = [];
      for (let i = 0; i < 100; i++) {
        const p = sampleDr(rng, spec);
        const r = run(ctrl, {episodeMode: 'continuous', physics: p.physics, delay: p.delay, steerGain: p.steerGain, speedGain: p.speedGain});
        rows.push({p: {...p.physics, delay: p.delay, steerGain: p.steerGain, speedGain: p.speedGain}, valid: r.valid, reason: r.reason, laps: r.laps});
      }
      const ok = rows.filter(r => r.valid), m = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
      res[label] = {n: rows.length, valid: ok.length, lap1_mean: m(ok.map(r => r.laps[0])), lap2_mean: m(ok.map(r => r.laps[1])), rows};
    }
    C.random = res;
  }
  if (sets.includes('rec60')) {
    const rec = loadRec60(), res = {};
    for (const delay of [0, 1]) {
      const rows = rec.map(c => ({name: c.name, ...run(ctrl, {episodeMode: 'continuous', start: c, delay})}));
      const ok = rows.filter(r => r.valid && !r.skipped), m = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
      res[`delay${delay}`] = {n: rows.filter(r => !r.skipped).length, valid: ok.length, lap1_mean: m(ok.map(r => r.laps[0])), lap2_mean: m(ok.map(r => r.laps[1])),
        min_clearance: ok.length ? Math.min(...ok.map(r => r.min_clearance)) : null, failures: rows.filter(r => !r.valid && !r.skipped).map(r => ({name: r.name, reason: r.reason, laps: r.laps, first_collision: r.first_collision}))};
    }
    C.rec60 = res;
  }
  if (sets.includes('rough')) C.rough = run(ctrl, {episodeMode: 'continuous', rough: true}).rough;
  if (sets.includes('fc')) C.friction_circle = await frictionCircle(ctrl);
  C.wall_s = (Date.now() - t0) / 1000;
  result.controllers[ctrl.name] = C;
  console.log(ctrl.name, JSON.stringify({grid: C.grid_summary, random: C.random && Object.fromEntries(Object.entries(C.random).map(([k, v]) => [k, `${v.valid}/${v.n}`])),
    rec60: C.rec60 && Object.fromEntries(Object.entries(C.rec60).map(([k, v]) => [k, `${v.valid}/${v.n} ${v.lap1_mean?.toFixed(3)} ${v.lap2_mean?.toFixed(3)}`])),
    fc: C.friction_circle?.map(r => `${r.mode} ${r.reason} ${r.laps}`), wall_s: C.wall_s}));
}

// 摩擦円の反事実(確認 r1 の自前の積分を読むだけ。sim-core.js は変えない)
async function frictionCircle(ctrl) {
  const {createOwnSim} = await import('../../check_r1/own_physics_r1.mjs');
  const rows = [];
  for (const mode of ['single', 'continuous']) {
    const env = envFor(mode); env.reset(STD);
    const cfg = JSON.parse(JSON.stringify(env.config)), active = env.world;
    const own = createOwnSim(active, cfg, {frictionCircle: true, interp: 4});
    const tr = createFeatureTracker(active, cfg), obs = new Float64Array(79);
    let st = own.state; tr.resetProgress(st.x, st.z, st.route_index); tr.observe(st, st.route_index, 0, 0, obs);
    let act;
    if (ctrl.kind === 'policy') { const c = ctrl.make(env); act = () => { const cmd = c.act({obs}); return cmd; }; }
    else { const T = await loadRcTeacher(); const t = T.createRcTrainingTeacher(active, cfg, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
      act = () => { const s = own.state; const c = t.action({state: s, metrics: {route_index: s.route_index}, config: cfg}); return {...actionToCommand(c.steering / .52, 2 * c.speed / 4.5 - 1), steering: c.steering, speed: c.speed}; }; }
    let n = 0;
    while (!own.state.terminal && n++ < 20 * 30 + 5) {
      const cmd = act(); own.step({steering: cmd.steering, speed: cmd.speed});
      const s = own.state; tr.advanceProgress(s.x, s.z); tr.observe(s, s.route_index, cmd.a0, cmd.a1, obs);
    }
    const r = own.result();
    rows.push({mode, reason: r.reason, laps: r.lap_events.map(l => +l.elapsed.toFixed(4)), gates: r.passed_gates, collisions: r.collisions_own.length,
      first_collision: r.collisions_own[0] ?? null, min_clearance_m: r.stats.minClear});
  }
  return rows;
}
if (out) fs.writeFileSync(out, JSON.stringify(result, null, 1) + '\n');
