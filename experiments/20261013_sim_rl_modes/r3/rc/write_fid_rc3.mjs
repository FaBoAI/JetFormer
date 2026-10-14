// 離脱復活 r3 の F-ID(ブラウザと Node の操作の一致の確かめに使う 600 フレームの列。2026-10-14、学習の段 recover r3)。
// rl/eval_modes.mjs の writeFid と同じ形で、走行中の仕組みだけ r3/rc/rt(乱し v2・乱しのフレーム以外のシールド)。昇格の段がブラウザ(teacher.js)に 1 フレームずつ与えて比べる。
//   node r3/rc/write_fid_rc3.mjs --policy <policy.json> --start '{"gate":0,"left_offset_m":0,"left_yaw_deg":0,"seed":1}' --out <fid.json>
import fs from 'node:fs';
import crypto from 'node:crypto';
import {createModeTeacher} from './rt/mode_runtime.mjs';
import {BASE_WORLD} from './episode_rc3.mjs';
const E = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const core = await import(`${E}/snapshot/sim/dist/sim-core.js`);
const {simConfig, activeWorldFor} = await import(`${E}/snapshot/rl/env_core.mjs`);
const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = JSON.parse(fs.readFileSync(args.policy, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(args.policy)).digest('hex');
const start = JSON.parse(args.start);
const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
const cfg = simConfig({episodeMode: 'continuous', seed: start.seed});
const sim = core.createSimulation(world, cfg);
let snap = sim.reset({seed: cfg.seed});
const t = createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: 'recover', weights, policySha256: sha});
const frames = [];
while (!snap.state.terminal) { const a = t.action(snap); frames.push({snapshot: snap, action: a}); snap = sim.step(a); }
fs.writeFileSync(args.out, JSON.stringify({schema: 'jf-sim-rl-fid/v1', mode: 'recover', round: 'r3', policy_sha256: sha, world, config: snap.config, variation, params: t.params,
  metadata: t.metadata, frames}));
console.log(JSON.stringify({frames: frames.length, reason: snap.state.reason, collisions: snap.metrics.collisions, learn0: frames.filter(x => x.action.info.jf.l === 0).length,
  shield: frames.filter(x => x.action.info.jf.sh === 1).length, events: t.events.map(e => ({type: e.type, cls: e.cls, kind: e.kind ?? null, ps: e.ps, pe: e.pe, fallback: e.fallback}))}));
