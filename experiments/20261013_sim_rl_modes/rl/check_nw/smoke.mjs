import {runOne} from './run_nw.mjs';
const t0 = Date.now();
const out = [];
const R58 = {start: {gate: 16, left_offset_m: -0.10726345916103576, left_yaw_deg: 13.341504, seed: 902158},
  dr: {physics: {steerRate: 0.8121330295223743, brakeDeceleration: 1.8718466758029537, driveTau: 0.438332234765403, driveAccelerationCap: 4.387224436737597, frictionMultiplier: 1.011924094252754}, delay: 1, steerGain: 1.037500893175602, speedGain: 0.9709531652927399, frictionCircle: false}};
for (const [name, spec] of [['STD-single', {start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, episodeMode: 'single'}],
  ['STD-cont', {start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}}], ['R58', R58], ['R58-off', {...R58, shield: false}],
  ['FC-STD', {start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, dr: {physics: {}, frictionCircle: true}}]]) {
  const t = Date.now(); const r = await runOne(spec);
  console.log(name, JSON.stringify({reason: r.reason, frames: r.frames, col: r.collisions, minC: r.min_clearance, f: r.min_frame, kind: r.min_kind, err: r.audit_pos_err, ov: r.overlap_frames,
    laps: r.laps.map(l => [+l.elapsed.toFixed(3), l.gates, l.crossings, +l.distance.toFixed(2)]), lfs: r.lap_from_stop, sh: r.shield_frames, lab: r.labels, rho: r.rho, cross: r.own_crossings.length, vrun: r.mean_running_speed}), (Date.now() - t) / 1000 + 's');
}
