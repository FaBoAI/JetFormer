// no_wall r4 の学習の開始の一覧(2026-10-14): 記録の開始の分布(ゲート U{0..32}・左右 U(−0.2, 0.2) m・向き U(−15°, 15°))から
// mulberry32(0x4e4a7a01) の一様乱数で引き、開始のすき間(自前の形 own_geom.mjs)が 12〜15 cm のものを 4,000 個。学習だけに使う
// (試験・検証・下調べの乱数とは別)。node gen_lowc_nw4.mjs > starts_lowc_nw4.json
import {startClearance} from '../run_nw4r.mjs';
import {mulberry32} from '../nw4_sets.mjs';
const rnd = mulberry32(0x4e4a7a01), starts = [];
let draws = 0;
while (starts.length < 4000) {
  const g = Math.floor(rnd() * 33), o = -.2 + .4 * rnd(), y = -15 + 30 * rnd(); draws++;
  const c = startClearance(g, o, y);
  if (c >= .12 && c <= .15) starts.push({gate: g, left_offset_m: +o.toFixed(6), left_yaw_deg: +y.toFixed(5), c0: +c.toFixed(5)});
}
const byGate = {}; for (const s of starts) byGate[s.gate] = (byGate[s.gate] ?? 0) + 1;
process.stdout.write(JSON.stringify({schema: 'jf-nw4-train-starts/v1', rng: 'mulberry32(0x4e4a7a01)', rule: '記録の開始の分布から一様、開始のすき間 0.12〜0.15 m',
  draws, count: starts.length, by_gate: byGate, starts}) + '\n');
