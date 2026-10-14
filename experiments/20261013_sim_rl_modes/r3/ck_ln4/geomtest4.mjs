// 確認 line r4: 自前のすき間(geom4.mjs)を、sim-core.js 自身の当たり判定と照らす。
// 障害物の近くの姿勢をランダムにとり(すき間 0〜8 cm を 2,000 個・重なり 1,000 個)、自前のすき間 c に対して「c − 1 mm だけ太らせた world では当たらない・c + 1 mm では当たる」
// (c ≤ 0 なら太らせない world で当たる)を、sim.reset の invalid_spawn で確かめる
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {WORLD, OB, REL, PHYS} from './run4.mjs';
import {clearance, inflatedWorld} from './geom4.mjs';
const core = await import(pathToFileURL(`${REL}/dist/sim-core.js`).href);
let s = 0x2468ace1;
const u = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
const hits = (m, x, z, yaw) => {
  const w = m > 0 ? inflatedWorld(WORLD, m) : WORLD;
  const sim = core.createSimulation({...w, spawn: {x, z, yaw}}, {...PHYS, seed: 1, timeLimit: 20, episodeMode: 'continuous-laps-v1'});
  return sim.reset({seed: 1}).state.terminal;
};
const res = {n: 0, pos: 0, bad: [], by_kind: {}, neg: 0, cmin: Infinity, cmax: -Infinity};
const L = OB.list;
for (let i = 0; i < 200000 && res.pos < 2000; i++) {
  const o = L[Math.floor(u() * L.length)];
  const x = o.cx + (u() - .5) * 1.0, z = o.cz + (u() - .5) * 1.0, yaw = (u() - .5) * 2 * Math.PI;
  const w = {}, c = clearance(OB, x, z, yaw, w);
  if (c > .08 || (c <= 0 && res.neg >= 1000)) continue;
  res.n++;
  const kind = w.id === 'bounds' ? 'bounds' : (L.find(q => q.id === w.id)?.k ?? '?');
  res.by_kind[kind] = (res.by_kind[kind] ?? 0) + 1;
  res.cmin = Math.min(res.cmin, c); res.cmax = Math.max(res.cmax, c);
  if (c <= 0) { res.neg++; if (!hits(0, x, z, yaw)) res.bad.push({x, z, yaw, c, id: w.id, why: 'c<=0 なのに当たらない'}); continue; }
  res.pos++;
  if (c > .001 && hits(c - .001, x, z, yaw)) res.bad.push({x, z, yaw, c, id: w.id, why: 'c−1mm で当たる'});
  if (!hits(c + .001, x, z, yaw)) res.bad.push({x, z, yaw, c, id: w.id, why: 'c+1mm で当たらない'});
}
console.log(JSON.stringify(res));
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4/geomtest4.json', JSON.stringify(res, null, 1));
