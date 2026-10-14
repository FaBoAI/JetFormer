#!/usr/bin/env node
// 確認 r3 の 7: ゲートの線(半幅 0.65 m の仮想の線)をどれだけ端で通るか。教師と比べて、ゲートの幅の使い方に差があるか(STD 単周)
import {loadPolicy, drive, driveTeacher, gateCrossings, save} from './common_r3.mjs';
const out = {};
const runs = {teacher: driveTeacher(), ...Object.fromEntries(['selected', 'r2_5M', 'r3_s21', 'r3_s22'].map(k => [k, drive(loadPolicy(k))]))};
for (const [k, r] of Object.entries(runs)) {
  const g = gateCrossings(r).filter(x => x.lat <= .65);
  const top = [...g].sort((a, b) => b.lat - a.lat).slice(0, 4).map(x => `${x.gate}:${x.lat.toFixed(3)}`);
  out[k] = {n: g.length, over_0_5: g.filter(x => x.lat > .5).length, over_0_6: g.filter(x => x.lat > .6).length, mean: g.reduce((a, x) => a + x.lat, 0) / g.length, top};
  console.log(k, JSON.stringify(out[k]));
}
save('e7_gate_edges.json', out);
