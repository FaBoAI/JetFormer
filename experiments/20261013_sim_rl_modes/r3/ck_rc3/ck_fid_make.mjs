// 離脱復活 r3 の確認: ブラウザと Node で同じ操作になるか(F-ID)の材料(2026-10-14 21:20)。
// 元の sim-core.js + 候補の runtime で走らせ、各フレームに教師へ渡した {state, metrics, config} と教師の出力 {steering, speed, info} を残す。
//   node ck_fid_make.mjs <出力のフォルダ>
import fs from 'node:fs';
import crypto from 'node:crypto';
import {BASE_WORLD, activeWorld, NOMINAL, CAND, POLICY_PATHS} from './ck_run.mjs';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const core = await import(`${REL}/dist/sim-core.js`);
const RT = await import(`${CAND}/runtime/mode_runtime.mjs`);
const weights = JSON.parse(fs.readFileSync(POLICY_PATHS.r3, 'utf8'));
const psha = crypto.createHash('sha256').update(fs.readFileSync(POLICY_PATHS.r3)).digest('hex');
const out = process.argv[2];
// 乱しあり(出来事 4・5、シールドの替えあり)・止まり(シールドが 600 フレーム)・強さ「強」固定
const CASES = [
  {name: 'FP_P0', start: {gate: 19, left_offset_m: -0.15558166466653348, left_yaw_deg: 11.568411192856729}, seed: 7310000, params: {}},
  {name: 'RP_P1', start: null, seed: 3120001, params: {}},
  {name: 'RP_P84', start: null, seed: 3120084, params: {}},
  {name: 'FS_P5', start: null, seed: 7370005, params: {perturb_class: '強'}},
  {name: 'FP_P106', start: null, seed: 7310106, params: {}},   // 乱しあり + シールドの替え(20 フレーム)
].filter(c => !process.env.CK_ONLY || process.env.CK_ONLY.split(',').includes(c.name));
const EV = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc3/eval_test.json', 'utf8'));
const JOBS = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_recover_r3/jobs.json', 'utf8'));
for (const c of CASES) {
  if (c.name.startsWith('RP_')) { const t = EV.tests['RC-P100-R3'].find(r => r.id === c.name.slice(3)); c.start = t.start; }
  if (c.name === 'FS_P5') { const j = JOBS.find(j => j.set === 'FSTRONG' && j.id === 'P5'); c.start = j.spec.start; }
  if (c.name === 'FP_P106') { const j = JOBS.find(j => j.set === 'FP' && j.id === 'P106'); c.start = j.spec.start; }
  const {world, variation} = activeWorld(BASE_WORLD, c.start.gate, c.start.left_offset_m, c.start.left_yaw_deg);
  const sim = core.createSimulation(world, {...NOMINAL, seed: c.seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed: c.seed, scenario: 'baseline'});
  const config = JSON.parse(JSON.stringify(snap.config));
  const teacher = RT.createModeTeacher(world, snap.config, variation, {mode: 'recover', weights, policySha256: psha, params: c.params});
  const frames = [];
  while (!snap.state.terminal) {
    const s = JSON.parse(JSON.stringify({state: snap.state, metrics: snap.metrics, config: snap.config}));
    const a = teacher.action({state: snap.state, metrics: snap.metrics, config: snap.config});
    frames.push({snapshot: s, action: {steering: a.steering, speed: a.speed, info: a.info}});
    snap = sim.step({steering: a.steering, speed: a.speed});
  }
  const fid = {name: c.name, world, config, variation, params: c.params, policy_sha256: psha, metadata: teacher.metadata, frames,
    events: teacher.events.length, shield_frames: teacher.shieldFrames, reason: snap.state.reason};
  fs.writeFileSync(`${out}/fid_${c.name}.json`, JSON.stringify(fid));
  console.log(c.name, frames.length, 'events', fid.events, 'shield', fid.shield_frames, snap.state.reason);
}
