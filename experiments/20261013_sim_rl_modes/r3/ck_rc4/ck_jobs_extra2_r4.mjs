// 離脱復活 r4 の確認: 足した回(2026-10-15 01:45)。開始のすき間が 3〜5 cm の開始(C6_r4 の乱しなしの組に入りうる、すき間の小さい側の端)を
// 記録の開始の分布から引き直しで 60 個(FUL、乱しなし、報告だけ)。乱数の種 0x7c0e4d07・種 9,624,000 + i
//   node ck_jobs_extra2_r4.mjs <out.json>
import fs from 'node:fs';
import {startClearance} from './ck_run.mjs';
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const J = [];
const u = mulberry32(0x7c0e4d07); let n = 0, draws = 0;
while (n < 60) {
  draws++;
  const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: -.2 + .4 * u(), left_yaw_deg: -15 + 30 * u()};
  const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
  if (!(c >= .03 && c < .05)) continue;
  J.push({set: 'FUL', id: `L${n}`, spec: {policy: 'r4', start: {...st, start_clearance: c, seed: 9624000 + n}, perturb: false}});
  n++;
}
fs.writeFileSync(process.argv[2], JSON.stringify(J));
console.log(J.length, 'draws', draws);
