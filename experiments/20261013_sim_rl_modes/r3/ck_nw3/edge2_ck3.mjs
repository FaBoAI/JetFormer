// 記録の開始の分布の「すき間 12〜16 cm」の開始を格子で全部(33 ゲート × 左右 {±0.2, ±0.17, ±0.14, ±0.11} × 向き {±15, ±13, ±11}°)
import fs from 'node:fs';
import {startClearance} from './run_ck3.mjs';
const jobs = []; let seed = 4600000;
for (let g = 0; g < 33; g++) for (const off of [-.2, -.17, -.14, -.11, .11, .14, .17, .2]) for (const yaw of [-15, -13, -11, 11, 13, 15]) {
  const c = startClearance(g, off, yaw);
  if (c >= .12 && c <= .16) jobs.push({set: 'EDGE2', id: `EDGE2:g${g}o${off}y${yaw}`, seed: seed++, start: {gate: g, off, yaw, c0: +c.toFixed(5)}});
}
fs.writeFileSync(process.argv[2], jobs.map(j => JSON.stringify(j)).join('\n') + '\n');
console.log(jobs.length);
