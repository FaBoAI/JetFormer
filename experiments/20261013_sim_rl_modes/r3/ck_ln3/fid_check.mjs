// 確認 line r3: 候補の F-ID 2 つ(fid_std_left08.json・fid_p37_shield.json)を自前でやり直す(2026-10-14、確認の段)。
//  (a) F-ID の world・config から元の sim-core.js で走らせ直し、snapshot の state が全部同じか
//  (b) F-ID の snapshot を 1 フレームずつ新しい教師(r3 の写し + mode.json のシールドの値)に与え、操作と info.jf が全部同じか(ブラウザの教師と同じ使い方)
//  (c) F-ID の world がユーザーのリリースの load-course-world.mjs + app.js の開始の式と同じか
import fs from 'node:fs';
import crypto from 'node:crypto';
import {REL, EXP, CAND, BASE_WORLD, SHIELD_PARAMS, spawnFor} from './ck_run.mjs';
const core = await import(`${REL}/dist/sim-core.js`);
const RT = await import(`${EXP}/r3/line/rt/mode_runtime.mjs`);
const weights = JSON.parse(fs.readFileSync(`${CAND}/policy.json`, 'utf8')), psha = crypto.createHash('sha256').update(fs.readFileSync(`${CAND}/policy.json`)).digest('hex');
const out = {};
for (const name of ['fid_std_left08.json', 'fid_p37_shield.json']) {
  const F = JSON.parse(fs.readFileSync(`${CAND}/${name}`, 'utf8'));
  const v = F.variation, {world} = spawnFor(BASE_WORLD, v.gate, v.left_offset_m, v.left_yaw_deg);
  const worldSame = JSON.stringify(world) === JSON.stringify(F.world);
  // (a)
  const sim = core.createSimulation(F.world, F.config);
  let snap = sim.reset({seed: F.config.seed});
  let stateDiff = 0, k = 0;
  for (const fr of F.frames) {
    const a = JSON.stringify(snap.state), b = JSON.stringify(fr.snapshot.state);
    if (a !== b) stateDiff++;
    snap = sim.step({steering: fr.action.steering, speed: fr.action.speed});
    k++;
  }
  // (b)
  globalThis.JF_R3 = {modeShield: {line: JSON.parse(JSON.stringify(SHIELD_PARAMS))}};
  const t = RT.createModeTeacher(F.world, JSON.parse(JSON.stringify(F.frames[0].snapshot.config)), F.variation, {mode: 'line', weights, params: F.params, policySha256: psha});
  globalThis.JF_R3 = undefined;
  let actDiff = 0, infoDiff = 0, sh = [];
  for (const fr of F.frames) {
    const a = t.action(JSON.parse(JSON.stringify(fr.snapshot)));
    if (a.steering !== fr.action.steering || a.speed !== fr.action.speed) actDiff++;
    if (JSON.stringify(a.info) !== JSON.stringify(fr.action.info)) infoDiff++;
    if (a.info.jf.sh) sh.push(fr.snapshot.state.frame);
  }
  out[name] = {frames: F.frames.length, world_same_as_release_and_appjs: worldSame, sim_state_mismatch_frames: stateDiff, teacher_action_mismatch_frames: actDiff,
    teacher_info_mismatch_frames: infoDiff, shield_frames: sh, policy_sha_matches: F.policy_sha256 === psha, metadata_has_shield: !!F.metadata.shield,
    shield_params_same: JSON.stringify(F.shield.params) === JSON.stringify(SHIELD_PARAMS), reason: snap.state.reason, laps: snap.metrics.lap_events.map(l => l.elapsed)};
}
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/fid_check.json', JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
