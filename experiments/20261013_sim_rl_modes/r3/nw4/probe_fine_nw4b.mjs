#!/usr/bin/env node
// no_wall r4b の下調べ(2026-10-14 20 時、再開で足した): ゲート 17 の「右へずれて左を向いた」角を細かい格子(dev4f)で見る。
// 格子は検証の点(±0.18 m・±13°)と試験の点(±0.16・±0.19 m × ±11.5・±14.5°)を含まないように選んだ:
//   左右 {−0.2, −0.185, −0.175, −0.165, −0.155} m × 向き {15, 14, 13.5, 12.5, 12}°。選び・判定には使わない
//   node probe_fine_nw4b.mjs <policy.json> [gate=17]
import {runOne, startClearance} from './run_nw4r.mjs';
const pol = process.argv[2], g = +(process.argv[3] ?? 17);
let n = 0;
for (const o of [-.2, -.185, -.175, -.165, -.155]) for (const y of [15, 14, 13.5, 12.5, 12]) {
  const c = startClearance(g, o, y);
  if (!(c >= .12)) continue;
  const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 6960000 + n++}, policyPath: pol});
  console.log(JSON.stringify({gate: g, off: o, yaw: y, stall: r.stall, collision: r.collision, valid_laps: r.valid_laps, min_cm: +(100 * r.min_clearance).toFixed(2), escapes: r.escapes, policy: r.policy_sha}));
}
