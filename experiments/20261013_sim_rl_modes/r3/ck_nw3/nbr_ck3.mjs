// 標準の物理で 5.6 cm まで寄った開始(LOWC:214、ゲート 20・左 −0.124 m・−13.0°)のまわりを細かく調べる(記録の開始の分布の中、開始のすき間 ≥ 12 cm だけ)
import fs from 'node:fs';
import {startClearance} from './run_ck3.mjs';
const jobs = []; let seed = 4500000;
for (const g of [20, 21]) for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) {
  const off = -.16 + .01 * i, yaw = -15 + 1 * j, c = startClearance(g, off, yaw);
  if (c >= .12) jobs.push({set: 'NBR', id: `NBR:g${g}o${off.toFixed(2)}y${yaw}`, seed: seed++, start: {gate: g, off, yaw, c0: +c.toFixed(5)}});
}
fs.writeFileSync(process.argv[2], jobs.map(j => JSON.stringify(j)).join('\n') + '\n');
console.log(jobs.length);
