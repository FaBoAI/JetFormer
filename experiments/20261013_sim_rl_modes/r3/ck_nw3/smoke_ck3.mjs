// 確認 r3 の最初の確かめ: 試験の行のいくつかを自前の走らせ方で走らせ、行の値と比べる
import fs from 'node:fs';
import {runOne} from './run_ck3.mjs';
const rows = fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/no_wall_r3/test/rows_test.jsonl', 'utf8').trim().split('\n').map(l => JSON.parse(l));
const want = process.argv.slice(2);
for (const r of rows.filter(r => want.includes(`${r.set}:${r.id}`))) {
  const d = r.dr ?? {};
  const spec = {id: r.id, set: r.set, start: {gate: r.start.gate, off: r.start.left_offset_m, yaw: r.start.left_yaw_deg}, seed: r.start.seed,
    phys: d.physics ?? null, delay: d.delay ?? 0, sg: d.steerGain ?? 1, vg: d.speedGain ?? 1, fc: !!d.frictionCircle, T: r.time_limit,
    episode: r.episode_mode, shield: r.shield};
  const t = Date.now();
  const o = await runOne(spec);
  console.log(JSON.stringify({id: `${r.set}:${r.id}`, wall: (Date.now() - t) / 1000,
    reason: [o.reason, r.reason], frames: [o.frames, r.frames], coll: [o.collisions, r.collisions], sh: [o.sh, r.shield_frames],
    vlaps: [o.valid_laps, r.valid_laps], minc: [+o.min_clearance.toFixed(6), +r.min_clearance.toFixed(6)], kind: [o.min_kind, r.min_kind], stall: [o.stall, r.stall, o.stall_max, r.stall_max_frames],
    vr3: [o.valid_episode_r3, r.valid_episode_r3], pos_err: o.pos_err, overlap: o.overlap_frames, fb: o.fb, probe: o.probe, ms: o.ms_per_frame?.toFixed(2)}));
}
