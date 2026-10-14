// 離脱復活 r4 の確認: 足した回の一覧(2026-10-15 01:20)。試験と新しい時刻の格子で落ちたキックが 2 つとも s 12.2〜12.6 m の右向きの向きのキックだったので、
// その付近(s 11.0〜14.0 m)を位置で狙うキックを新しい開始で(FKZ、報告だけ。判定の組ではない)。乱数の種 0x7c0e4d05・種 9,623,000 + i
//   node ck_jobs_extra_r4.mjs <out.json>
import fs from 'node:fs';
import {startClearance} from './ck_run.mjs';
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const J = [];
const u = mulberry32(0x7c0e4d05); let n = 0;
while (n < 120) {
  const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: -.2 + .4 * u(), left_yaw_deg: -15 + 30 * u()};
  const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
  if (!(c >= .03)) continue;
  const sT = 11.0 + 3.0 * u(), sign = u() < 2 / 3 ? -1 : 1, kind = n % 2 === 0 ? 'yaw' : 'lateral';
  J.push({set: 'FKZ', id: `Z${n}`, spec: {policy: 'r4', start: {...st, start_clearance: c, seed: 9623000 + n}, perturb: false, kick: {s_target: sT, min_frame: 45, [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind}});
  n++;
}
fs.writeFileSync(process.argv[2], JSON.stringify(J));
console.log(J.length);
