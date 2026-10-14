// 確認 r2: 標準の物理で止まったままになる開始(ゲート 17・右 0.2 m・左向き 15°)の走りの中身(進みの s・ゲート・シールドのフレーム)
import {runOne} from './run_nw_r2.mjs';
for (const st of [{gate: 17, left_offset_m: -.2, left_yaw_deg: 15, seed: 2600000}, {gate: 17, left_offset_m: -.2, left_yaw_deg: 12, seed: 2990101}, {gate: 17, left_offset_m: -.15, left_yaw_deg: 15, seed: 2990102}, {gate: 17, left_offset_m: -.2, left_yaw_deg: 10, seed: 2990103}]) {
  const r = await runOne({mode: 'no_wall', start: st, keepPath: true});
  const p = r.path, s0 = p[0][3];
  const tl = [0, 15, 30, 60, 90, 150, 210, 270, 300, 330, 360, 450, 599].filter(i => i < p.length).map(i => `${i}:s${(p[i][3] - s0).toFixed(2)}/v${p[i][2].toFixed(2)}/sh${p[i][7]}`);
  console.log(JSON.stringify({st, reason: r.reason, laps: r.valid_laps, gates: r.passed_gates, dist: +r.distance.toFixed(2), min: +(100 * r.min_clearance).toFixed(2), sh: r.shield_frames, fb: r.labels.fb}));
  console.log(tl.join(' '));
}
