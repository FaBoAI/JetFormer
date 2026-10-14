// r3 の下調べ(no_wall): 確認 r2 で見つけた「止まったまま」の開始などを、r3/runtime の試作(シールド v4・開始のそろえ)で走らせて中身を見る
//   JF_R3='{...}' node r3/nw/probe_cases.mjs
import {runOne} from '../../rl/check_nw_r2/run_nw_r2.mjs';
if (process.env.JF_R3) globalThis.JF_R3 = JSON.parse(process.env.JF_R3);
const RT = process.env.JF_RT === 'web' ? '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/web/rl' : '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/runtime';
const cases = [
  {id: 'FN581', start: {gate: 17, left_offset_m: -0.19765207832679155, left_yaw_deg: 14.691235923673958, seed: 2400581}},
  {id: 'NG-g17o-0.2y15', start: {gate: 17, left_offset_m: -0.2, left_yaw_deg: 15, seed: 2600001}},
  {id: 'NG-g17o-0.1y15', start: {gate: 17, left_offset_m: -0.1, left_yaw_deg: 15, seed: 2600002}},
  {id: 'FD73-nominal', start: {gate: 23, left_offset_m: 0.03225647350773214, left_yaw_deg: 13.315470144152641, seed: 2400673}},
];
for (const c of cases) {
  const r = await runOne({mode: 'no_wall', start: c.start, rtDir: RT, keepPath: false});
  console.log(JSON.stringify({id: c.id, reason: r.reason, col: r.collisions, valid: r.valid_episode, laps: r.valid_laps, min: +r.min_clearance.toFixed(4), minf: r.min_frame,
    sh: r.shield_frames, align: r.labels.ph.align ?? 0, fb: r.labels.fb, v: +r.mean_running_speed.toFixed(2)}));
}
