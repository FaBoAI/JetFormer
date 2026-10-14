// 本当のゲートの外しの数え方の確かめ(2026-10-14 recover r2): 試験の組の RC-P100 のうち指定の回だけを元のコアで走らせ、gate_miss を出す
//   node rl/recover_r2/probe_gate_miss_one.mjs <policy.json> P37 P0 ...
import fs from 'node:fs';
import crypto from 'node:crypto';
import {buildJobs} from '../eval_modes.mjs';
import {runModeEpisode, readJson} from '../episode.mjs';
const [pol, ...ids] = process.argv.slice(2);
const weights = readJson(pol), sha = crypto.createHash('sha256').update(fs.readFileSync(pol)).digest('hex');
const jobs = buildJobs('recover', 'test').filter(j => j.test === 'RC-P100' && ids.includes(j.id));
for (const j of jobs) {
  const r = await runModeEpisode({mode: 'recover', weights, policySha256: sha, start: j.spec.start, params: {}, keepLog: true});
  console.log(j.id, {valid: r.valid_episode, reason: r.reason, frames: r.frames, laps: r.laps.map(l => +l.elapsed.toFixed(2)), gate_miss: r.gate_miss, gate_misses: r.gate_misses,
    stuck: +r.gate_stuck_max_m.toFixed(2), miss_frames: r.gate_miss_frames, virtual: r.gate_misses_virtual});
}
