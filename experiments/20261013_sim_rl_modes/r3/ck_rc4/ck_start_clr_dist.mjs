// 離脱復活 r4 の確認: 記録の開始の分布(ゲート U{0..32}・左右 ±0.2 m・向き ±15°)で、開始のすき間(自前の形)の分布を数える(2026-10-15、報告だけ)
import {startClearance} from './ck_run.mjs';
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const u = mulberry32(0x7c0e4d06), N = 20000; const c = {le0: 0, lt3: 0, b3_35: 0, b35_5: 0, ge5: 0};
for (let i = 0; i < N; i++) {
  const x = startClearance(Math.min(32, Math.floor(u() * 33)), -.2 + .4 * u(), -15 + 30 * u());
  if (!(x > 1e-9)) c.le0++; else if (x < .03) c.lt3++; else if (x < .035) c.b3_35++; else if (x < .05) c.b35_5++; else c.ge5++;
}
console.log(JSON.stringify({N, ...c, frac: Object.fromEntries(Object.entries(c).map(([k, v]) => [k, +(v / N).toFixed(5)]))}));
