#!/usr/bin/env node
// F-ID の追加の列(no_wall r2、2026-10-14): シールド v3 の見積もり・速さの見張りが働く列。ブラウザの teacher.js に同じ snapshot を与えて
// Node と同じ操作になるか(protocol ui_acceptance U5「シールドも同じ」)を確かめるため。物理は 舵の速さ 0.8・遅延 1(テストの側で 1 フレーム遅らせる)、
// 開始はゲート 16。教師には標準の物理の config を渡し、extra(適用した行動)は渡さない(ブラウザと同じ呼び方)。形は eval_modes.mjs writeFid と同じ。
//   node rl/nw2/write_fid_slow.mjs --policy bundle/no_wall/policy.json --out fid_slow.json [--runtime rl/runtime] [--gate 16] [--steer-rate 0.8] [--delay 1]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url)), EXP = path.resolve(here, '../..');
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const rtDir = path.resolve(opt('--runtime', path.join(EXP, 'rl/runtime')));
const {createModeTeacher} = await import(pathToFileURL(path.join(rtDir, 'mode_runtime.mjs')).href);
const core = await import(pathToFileURL(path.join(EXP, 'snapshot/sim/dist/sim-core.js')).href);
const {simConfig, activeWorldFor} = await import(pathToFileURL(path.join(EXP, 'snapshot/rl/env_core.mjs')).href);
const {BASE_WORLD} = await import(pathToFileURL(path.join(EXP, 'rl/episode.mjs')).href);
const policyPath = opt('--policy'), weights = JSON.parse(fs.readFileSync(policyPath, 'utf8'));
const sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const gate = +opt('--gate', '16'), sr = +opt('--steer-rate', '0.8'), delay = +opt('--delay', '1');
const start = {gate, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
const cfg = simConfig({episodeMode: 'continuous', seed: start.seed});
const sim = core.createSimulation(world, {...cfg, steerRate: sr});
let snap = sim.reset({seed: cfg.seed});
const cfgN = {...JSON.parse(JSON.stringify(snap.config)), steerRate: cfg.steerRate ?? 1.2};
const t = createModeTeacher(world, cfgN, variation, {mode: 'no_wall', weights, params: {}, policySha256: sha});
const frames = [], q = [];
while (!snap.state.terminal) {
  const view = {...snap, config: cfgN};
  const a = t.action(view);
  frames.push({snapshot: view, action: a});
  q.push(a); const ap = q.length > delay ? q.shift() : {steering: 0, speed: 0};
  snap = sim.step({steering: ap.steering, speed: ap.speed});
}
const shieldFrames = frames.filter(f => f.action.info.jf.sh === 1).length;
fs.writeFileSync(opt('--out'), JSON.stringify({schema: 'jf-sim-rl-fid/v1', mode: 'no_wall', policy_sha256: sha, world, config: cfgN, variation, params: {},
  metadata: t.metadata, note: `シールド v3 の列: 物理 舵の速さ ${sr}・遅延 ${delay}(snapshot に入っている)。教師の config は標準の物理`, physics: {steerRate: sr, delay}, frames}));
console.log(JSON.stringify({frames: frames.length, shield_frames: shieldFrames, reason: snap.state.reason, collisions: snap.metrics.collisions}));
