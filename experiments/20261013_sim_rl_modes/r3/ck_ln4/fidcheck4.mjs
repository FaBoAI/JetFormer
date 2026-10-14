// 確認 line r4: 候補の F-ID 2 つを、自前の走らせ方(リリースの sim-core.js + loadCourseWorld + app.js の開始の式 + 候補の runtime)で
// 1 フレームずつ作り直し、F-ID に書かれた状態・操作・info と照らす(F-ID の world を使わず、自前で作った world で)
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {REL, CAND, PHYS, spawnAt, sha, EXPECT} from './run4.mjs';
const core = await import(pathToFileURL(`${REL}/dist/sim-core.js`).href);
const MR = await import(pathToFileURL(`${CAND}/runtime/mode_runtime.mjs`).href);
const MODE = JSON.parse(fs.readFileSync(`${CAND}/mode.json`, 'utf8'));
const W = JSON.parse(fs.readFileSync(`${CAND}/policy.json`, 'utf8'));
const out = {};
for (const f of ['fid_std_left08.json', 'fid_ln_g30r120_r4b_g8_shield.json']) {
  const F = JSON.parse(fs.readFileSync(`${CAND}/${f}`, 'utf8'));
  const v = F.variation, seed = F.config.seed;
  const {world, variation} = spawnAt(v.gate, v.left_offset_m, v.left_yaw_deg);
  const sim = core.createSimulation(world, {...PHYS, seed, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  let snap = sim.reset({seed, scenario: 'baseline'});
  const T = MR.createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode: 'line', weights: W, policySha256: EXPECT.policy_line,
    params: JSON.parse(JSON.stringify(F.params)), modeRuntime: JSON.parse(JSON.stringify(MODE.runtime))});
  let same = 0, stateMax = 0, actMax = 0, infoDiff = 0, n = 0, shF = 0;
  for (const fr of F.frames) {
    if (snap.state.terminal) break;
    const S = snap.state, R = fr.snapshot.state;
    stateMax = Math.max(stateMax, Math.abs(S.x - R.x), Math.abs(S.z - R.z), Math.abs(S.yaw - R.yaw), Math.abs(S.speed - R.speed));
    const o = T.action({state: S, metrics: snap.metrics, config: snap.config});
    const a = fr.action;
    actMax = Math.max(actMax, Math.abs(o.steering - a.steering), Math.abs(o.speed - a.speed));
    if (JSON.stringify(o.info) !== JSON.stringify(a.info)) infoDiff++;
    if (JSON.stringify(o) === JSON.stringify(a)) same++;
    if (o.info.jf.sh === 1) shF++;
    snap = sim.step({steering: o.steering, speed: o.speed});
    n++;
  }
  out[f] = {frames_file: F.frames.length, frames_replayed: n, exact_same: same, state_max_abs: stateMax, action_max_abs: actMax, info_diff_frames: infoDiff, shield_frames: shF,
    metadata_same: JSON.stringify(T.metadata) === JSON.stringify(F.metadata), policy_sha_in_file: F.policy_sha256 === EXPECT.policy_line,
    world_same_as_release: JSON.stringify(F.world.segments) === JSON.stringify(world.segments) && JSON.stringify(F.world.disks) === JSON.stringify(world.disks) && JSON.stringify(F.world.route) === JSON.stringify(world.route)};
}
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4/fidcheck4.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
