// 経路いろいろ r3 の自分での確かめ(2026-10-14、r3 の学習の段 line。判定には使わない): 試験の組と同じ回を、r3 の下調べの走らせ方
// r3/line/run_ln3.mjs(確認 line r2 の run_ln2.mjs の写し: ユーザーのリリースの sim-core.js とコース、自前の多角形の距離 + 4 mm おきの点ですき間)で走らせ直し、
// 学習の段の走らせ方(episode_ln3.mjs + lap_audit)と終わり方・最小すき間・シールドのフレームをくらべる。
//   JF_R3='{"modeShield":{"line":{…}}}' node r3/line/xcheck_ln3.mjs --policy <policy.json> --shard k/n --out rows.jsonl
import fs from 'node:fs';
import {buildJobs} from './eval_line_r3.mjs';
import {runOne} from './run_ln3.mjs';
const arg = k => { const i = process.argv.indexOf(k); return i >= 0 ? process.argv[i + 1] : null; };
const [sk, sn] = (arg('--shard') ?? '0/1').split('/').map(Number), pol = arg('--policy');
const jobs = buildJobs('test').jobs.filter((_, i) => i % sn === sk);
const fd = fs.openSync(arg('--out'), 'w');
for (const j of jobs) {
  const r = await runOne({policy: pol, mode: 'line', start: j.spec.start, params: j.spec.params, family: j.spec.family, keepFrames: false});
  fs.writeSync(fd, JSON.stringify({test: j.test, id: j.id, reason: r.reason, frames: r.frames, collisions: r.collisions, min_clearance: r.min_clearance, min_t: r.min_t,
    valid_episode: r.valid_episode, valid_laps: r.valid_laps, shield_frames: r.labels.ph.shield ?? 0, labels: r.labels, own_crossings_bad: r.own_crossings.filter(c => !c.inside).length,
    audit_pos_err: r.audit_pos_err, overlap_frames: r.overlap_frames, gate_stuck_max: r.gate_stuck_max, laps: r.laps.map(l => l.elapsed)}) + '\n');
}
fs.closeSync(fd);
