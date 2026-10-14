// r3 の下調べ(離脱復活 + シールドの試作): REC60(乱しなし、リリースの rc-collection-plan-60)と、新しい開始 100 回(乱しあり)の一覧
//   node r3/rc/jobs_rc3.mjs <out.json>
// 新しい開始は記録の開始の分布から mulberry32(0x3c3c0001) の一様乱数(重なる開始は飛ばす)、種 3,190,100 + i(試験・検証・記録・確認の種と重ならない)
import fs from 'node:fs';
import {startClearance} from './run_rc3.mjs';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const J = [], add = (set, id, spec) => J.push({set, id, spec});
const plan = JSON.parse(fs.readFileSync(`${REL}/qa/rc-collection-plan-60.json`, 'utf8'));
plan.forEach(c => add('REC60', c.name, {mode: 'recover', perturb: false, start: {gate: c['start-gate'] ?? 0, left_offset_m: c['spawn-offset'] ?? 0, left_yaw_deg: c['spawn-yaw'] ?? 0, seed: c.seed}}));
const rnd = mulberry32(0x3c3c0001);
const fresh = [];
while (fresh.length < 100) {
  const st = {gate: Math.min(32, Math.floor(rnd() * 33)), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
  if (startClearance(st.gate, st.left_offset_m, st.left_yaw_deg) > 1e-9) fresh.push(st);
}
fresh.forEach((st, i) => add('FP100', `R${i}`, {mode: 'recover', start: {...st, seed: 3190100 + i}}));
fs.writeFileSync(process.argv[2], JSON.stringify(J));
console.log('jobs', J.length);
