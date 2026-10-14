#!/usr/bin/env node
// 経路いろいろ r3 の F-ID(2026-10-14、r3 の学習の段 line): STD の開始・左寄り c0 0.8・c_v 1.0 の 600 フレームの snapshot と、
// Node の操作(r3/line/rt/mode_runtime.mjs + すき間シールド protocol_line_r3.json runtime.shield.params)。元の sim-core.js。
// 昇格の段がブラウザの teacher.js に 1 フレームずつ与えて、操作と info.jf が同じことを確かめるのに使う(rl/eval_modes.mjs writeFid と同じ形)。
//   node r3/line/make_fid_ln3.mjs --policy <policy.json> --out fid.json [--start '{"gate":…,"seed":…}' --params '{"line":…,"c_v":…}']
//   (--start・--params: シールドが働く回の F-ID。既定は STD・左寄り 0.8)
import fs from 'node:fs';
import crypto from 'node:crypto';
import {SHIELD_PARAMS} from './eval_line_r3.mjs';
import {BASE_WORLD} from './episode_ln3.mjs';
import {createModeTeacher} from './rt/mode_runtime.mjs';
import {activeWorldFor, simConfig} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';

const arg = k => { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : null; };
const policyPath = arg('--policy'), out = arg('--out');
const weights = JSON.parse(fs.readFileSync(policyPath, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
globalThis.JF_R3 = {modeShield: {line: JSON.parse(JSON.stringify(SHIELD_PARAMS))}};
const core = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/sim/dist/sim-core.js');
const start = arg('--start') ? JSON.parse(arg('--start')) : {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const params = arg('--params') ? JSON.parse(arg('--params')) : {line: {family: 'const', c0: .8}, c_v: 1.0};
const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
const cfg = simConfig({episodeMode: 'continuous', seed: start.seed});
const sim = core.createSimulation(world, cfg);
let snap = sim.reset({seed: cfg.seed});
const t = createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: 'line', weights, params, policySha256: sha});
const frames = [];
let sh = 0;
while (!snap.state.terminal) { const a = t.action(snap); if (a.info.jf.sh) sh++; frames.push({snapshot: snap, action: a}); snap = sim.step(a); }
fs.writeFileSync(out, JSON.stringify({schema: 'jf-sim-rl-fid/v1', mode: 'line', policy_sha256: sha, world, config: snap.config, variation, params, metadata: t.metadata,
  shield: {params: SHIELD_PARAMS, note: 'r3 の試作の切り替え globalThis.JF_R3.modeShield.line にこの値を入れて作った'}, frames}));
console.log(JSON.stringify({out, start, frames: frames.length, shield_frames: sh, shield_at: frames.map((q, k) => q.action.info.jf.sh ? k : -1).filter(k => k >= 0), reason: snap.state.reason, laps: snap.metrics.lap_events?.map(l => +l.elapsed.toFixed(3))}));
