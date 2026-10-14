// 離脱復活の図のための走りの書き出し(学習の段 recover r1、2026-10-13)。試験の組 RC-P100 と同じ開始・種(seedBase('test', 'recover') + i)で、
// 元の sim-core.js(判定と同じコア)で走らせ、姿勢の列・乱しの区間・ぶつかった位置を JSON に書く。乱しなしの 1 周(STD)も付ける。
//   node rl/recover_r1/dump_traces.mjs --policy <policy.json> --ids 0,1,2,3 --out <traces.json>
import fs from 'node:fs';
import {runModeEpisode, readJson} from '../episode.mjs';
import {haltonStarts, seedBase} from '../eval_modes.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = readJson(args.policy), ids = (args.ids ?? '0,1,2,3').split(',').map(Number), set = args.set ?? 'test';
const hs = haltonStarts('recover', set, Math.max(...ids) + 1), base = seedBase(set, 'recover');
const out = {policy: args.policy, set, episodes: []};
const pack = (r, id) => ({id, start: r.start, reason: r.reason, frames: r.frames, collisions: r.collisions, valid: r.valid_episode, events: r.events,
  collision_xz: r.collisions ? [r.trace.at(-1).x, r.trace.at(-1).z] : null,
  trace: r.trace.map((q, i) => [+q.x.toFixed(4), +q.z.toFixed(4), +q.v.toFixed(3), r.log[i]?.l ?? 1, r.log[i]?.ev ?? 0]), laps: r.laps});
for (const i of ids) {
  const r = await runModeEpisode({mode: 'recover', weights, start: {...hs[i], seed: base + i}, coreKind: args.core ?? 'original', audit: false});
  out.episodes.push(pack(r, `P${i}`));
  console.log(`P${i}`, r.reason, r.frames, 'events', r.events.length, 'valid', r.valid_episode);
}
const std = await runModeEpisode({mode: 'recover', weights, start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, episodeMode: 'single', coreKind: args.core ?? 'original', audit: false, overrides: {perturbation: false}});
out.std = pack(std, 'STD');
console.log('STD', std.reason, std.laps?.[0]?.elapsed ?? std.sim_time);
fs.writeFileSync(args.out, JSON.stringify(out));
