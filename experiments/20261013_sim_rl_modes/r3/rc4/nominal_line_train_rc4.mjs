// 離脱復活 r4(2026-10-14): r3/rc/nominal_line_train.mjs の写し(方策を r3 の 8d686076… に)。学習の立て直しの報酬に使う「乱しなしの線」(protocol definitions.nominal_line と同じ 8 回の作り方)を、
// 初期化元 = 離脱復活 r3 の方策(8d686076…)で作る。学習ではシールドを使わないので、シールドなし・乱しなし・元の sim-core.js。
//   node r3/rc4/nominal_line_train_rc4.mjs <出力.json>
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runModeEpisode, readJson} from './episode_rc4.mjs';
import {nominalLine, sceneFromEpisode} from './variation_metrics_rc4.mjs';
import {createRng, mix32} from './rt/rng.mjs';
const P = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/rc4/ref/recover_r3_policy.json';
const weights = readJson(P), sha = crypto.createHash('sha256').update(fs.readFileSync(P)).digest('hex');
const scenes = [], runs = [];
for (let i = 0; i < 8; i++) {
  const seed = 990000 + i, rng = createRng(mix32(seed));
  const start = {gate: 0, left_offset_m: rng.uniform(-.03, .03), left_yaw_deg: rng.uniform(-3, 3), seed};
  const r = await runModeEpisode({mode: 'recover', weights, policySha256: sha, start, coreKind: 'original', audit: false, overrides: {perturbation: false, shield: false}});
  scenes.push(sceneFromEpisode(r)); runs.push({seed, start, reason: r.reason, collisions: r.collisions, laps: r.laps.map(l => l.elapsed)});
}
fs.writeFileSync(process.argv[2], JSON.stringify({...nominalLine(scenes), mode: 'recover', policy_sha256: sha, runs_detail: runs,
  note: '離脱復活 r4 の学習の立て直しの報酬に使う線(初期化元 = 離脱復活 r3 の方策 8d686076…・シールドなし・乱しなし)'}));
console.log(JSON.stringify(runs));
