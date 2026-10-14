// 確認 r4 の煙の試験: 試験の行 N0 と、確認 r3 の崩れの開始を 1 回ずつ
import {runOne} from './run_ck4.mjs';
const specs = [
  {id: 'N0', set: 'smoke', start: {gate: 17, off: 0.03557610346209644, yaw: -8.981568}, seed: 6200000},
  {id: 'g20', set: 'smoke', start: {gate: 20, off: -0.13, yaw: -13}, seed: 1},
  {id: 'g23', set: 'smoke', start: {gate: 23, off: 0.05, yaw: 12}, seed: 1},
];
for (const s of specs) {
  const t = Date.now();
  const r = await runOne(s);
  console.log(JSON.stringify({id: r.id, reason: r.reason, coll: r.collisions, minC: r.min_clearance, minF: r.min_frame, kind: r.min_kind, laps: r.valid_laps, sh: r.sh, fb: r.fb, esc: r.escapes, rev: r.reverse_frames, stall: r.stall, lock: r.lock_mismatch, lab: [r.label_mismatch_sh0, r.label_sh1_unchanged], ms: [r.ms_p50, r.ms_p99, r.ms_max], v3: r.valid_episode_r3, wall: (Date.now() - t) / 1000}));
}
