#!/usr/bin/env node
// 経路いろいろ r4b の F-ID(2026-10-14 夜、r3/line4/make_fid_ln4.mjs の写し。読み込む先を rt4b・protocol_line_r4b.json に替えた): 開始・形の 600 フレームの snapshot と、
// Node の操作(r3/line4b/rt4b/mode_runtime.mjs + シールド v5b。シールドは教師に modeRuntime = protocol_line_r4b.json runtime.shield.mode_runtime を渡してかける
// = 昇格のあとのブラウザで teacher.js が mode.json の runtime を渡すのと同じ道)。元の sim-core.js。
// 昇格の段がブラウザの teacher.js に 1 フレームずつ与えて、操作と info.jf が同じことを確かめるのに使う(rl/eval_modes.mjs writeFid と同じ形)。
//   node r3/line4b/make_fid_ln4b.mjs --policy <policy.json> --out fid.json [--start '{"gate":…,"seed":…}' --params '{"line":…,"c_v":…}']
import fs from 'node:fs';
import crypto from 'node:crypto';
import {shieldModeRuntime} from './eval_line_r4b.mjs';
import {BASE_WORLD} from './episode_ln4.mjs';
import {createModeTeacher} from './rt4b/mode_runtime.mjs';
import {activeWorldFor, simConfig} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';

const arg = k => { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : null; };
const policyPath = arg('--policy'), out = arg('--out');
const weights = JSON.parse(fs.readFileSync(policyPath, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const SH = shieldModeRuntime({set: 'test'});
const core = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/sim/dist/sim-core.js');
const start = arg('--start') ? JSON.parse(arg('--start')) : {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const params = arg('--params') ? JSON.parse(arg('--params')) : {line: {family: 'const', c0: .8}, c_v: 1.0};
const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
const cfg = simConfig({episodeMode: 'continuous', seed: start.seed});
const sim = core.createSimulation(world, cfg);
let snap = sim.reset({seed: cfg.seed});
const t = createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: 'line', weights, params, policySha256: sha, modeRuntime: SH.modeRuntime});
const frames = [];
let sh = 0, rev = 0;
while (!snap.state.terminal) { const a = t.action(snap); if (a.info.jf.sh) sh++; if (a.speed < 0) rev++; frames.push({snapshot: snap, action: a}); snap = sim.step(a); }
fs.writeFileSync(out, JSON.stringify({schema: 'jf-sim-rl-fid/v1', mode: 'line', policy_sha256: sha, world, config: snap.config, variation, params, metadata: t.metadata,
  mode_runtime: SH.modeRuntime, shield: {source: SH.source, source_sha256: SH.sha256, note: '教師に modeRuntime(mode.json の runtime と同じ形)を渡して作った'}, frames}));
console.log(JSON.stringify({out, start, frames: frames.length, shield_frames: sh, negative_speed_cmds: rev, shield_at: frames.map((q, k) => q.action.info.jf.sh ? k : -1).filter(k => k >= 0).slice(0, 40),
  reason: snap.state.reason, laps: snap.metrics.lap_events?.map(l => +l.elapsed.toFixed(3)), metadata_shield: !!t.metadata.shield}));
