#!/usr/bin/env node
// no_wall r3 の F-ID(2026-10-14): Node で作った snapshot の列と、r3/nw/runtime_v4(シールド v4)の教師の操作。昇格の段がブラウザ(8080)の
// teacher.js に同じ snapshot を 1 フレームずつ与えて、操作が同じか(protocol ui_acceptance U5)を確かめるために使う。形は rl/nw2/write_fid_slow.mjs と同じ。
//   fid_std    標準の物理・STD の開始・連続 20 秒(元の sim-core.js)
//   fid_slow   舵の速さ 0.8・遅延 1・ゲート 16(元の sim-core.js。見積もり・速さの見張りが働く)
//   fid_corner 確認 r2 の隅 K0(舵 1.8・ブレーキ 0.7・時定数 0.28・加速の上限 3.5・μ × 0.85・摩擦円・舵のゲイン 1.06・前進のゲイン 1.04)・ゲート 10
//              (摩擦円は高速版の写し fast_core_nw3.mjs。模型 Q・ブレーキの見張りが働く)
//   node r3/nw/write_fid_nw3.mjs --policy policy.json --outdir <dir>
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const {createModeTeacher} = await import(`${EXP}/r3/nw/runtime_v4/mode_runtime.mjs`);
const core = await import(`${EXP}/snapshot/sim/dist/sim-core.js`);
const fcore = await import(`${EXP}/r3/nw/fast_core_nw3.mjs`);
const {simConfig, activeWorldFor} = await import(`${EXP}/snapshot/rl/env_core.mjs`);
const BASE_WORLD = JSON.parse(fs.readFileSync(`${EXP}/snapshot/sim/world-v3.json`, 'utf8'));
const policyPath = opt('--policy'), weights = JSON.parse(fs.readFileSync(policyPath, 'utf8'));
const sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const outdir = opt('--outdir');
fs.mkdirSync(outdir, {recursive: true});
const CASES = {
  fid_std: {gate: 0, physics: {}, delay: 0, sg: 1, vg: 1, fc: false, note: '標準の物理・STD の開始・連続 20 秒'},
  fid_slow: {gate: 16, physics: {steerRate: .8}, delay: 1, sg: 1, vg: 1, fc: false, note: '物理 舵の速さ 0.8・遅延 1(snapshot に入っている)。シールド v4 の見積もり・速さの見張りが働く'},
  fid_corner: {gate: 10, physics: {steerRate: 1.8, brakeDeceleration: .7, driveTau: .28, driveAccelerationCap: 3.5, frictionMultiplier: .85}, delay: 0, sg: 1.06, vg: 1.04, fc: true,
    note: '確認 r2 の隅 K0(舵 1.8・ブレーキ 0.7・時定数 0.28・加速の上限 3.5・μ × 0.85・摩擦円・舵のゲイン 1.06・前進のゲイン 1.04)・ゲート 10。摩擦円は高速版の写し。模型 Q とブレーキの見張りが働く'},
};
const summary = {};
for (const [name, C] of Object.entries(CASES)) {
  const start = {gate: C.gate, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
  const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
  const cfg = simConfig({episodeMode: 'continuous', seed: start.seed});
  const nomSim = core.createSimulation(world, cfg);
  const cfgN = JSON.parse(JSON.stringify(nomSim.reset({seed: cfg.seed}).config));
  const sim = (C.fc ? fcore : core).createSimulation(world, {...cfg, ...C.physics, ...(C.fc ? {frictionCircle: true} : {})});
  let snap = sim.reset({seed: cfg.seed});
  const t = createModeTeacher(world, cfgN, variation, {mode: 'no_wall', weights, params: {}, policySha256: sha});
  const frames = [], q = [];
  while (!snap.state.terminal) {
    const view = JSON.parse(JSON.stringify({...snap, config: cfgN}));
    const a = t.action(view);
    frames.push({snapshot: view, action: a});
    q.push(a); const ap = q.length > C.delay ? q.shift() : {steering: 0, speed: 0};
    snap = sim.step({steering: ap.steering * C.sg, speed: ap.speed * C.vg});
  }
  const shieldFrames = frames.filter(f => f.action.info.jf.sh === 1).length;
  const doc = {schema: 'jf-sim-rl-fid/v1', mode: 'no_wall', policy_sha256: sha, world, config: cfgN, variation, params: {}, metadata: t.metadata,
    note: `シールド v4 の列: ${C.note}。教師の config は標準の物理`, physics: {...C.physics, delay: C.delay, steerGain: C.sg, speedGain: C.vg, frictionCircle: C.fc}, frames};
  fs.writeFileSync(path.join(outdir, `${name}.json`), JSON.stringify(doc));
  summary[name] = {frames: frames.length, shield_frames: shieldFrames, reason: snap.state.reason, collisions: snap.metrics.collisions};
}
fs.writeFileSync(path.join(outdir, 'fid_summary.json'), JSON.stringify({policy_sha256: sha, shield_version: CASES && (await import(`${EXP}/r3/nw/runtime_v4/shield.mjs`)).SHIELD_VERSION, cases: summary}, null, 1));
console.log(JSON.stringify(summary));
