// 障害物よけ r2(学習の段、2026-10-14): 試験・検証の組の開始のすき間(車体とコース + 足した障害物、方策は走らせない)。
// O2(有効な回の最小すき間 ≥ 3 cm)が開始の時点で満たせない回が無いかを、学習の前に確かめる。
//   node rl/ob2/start_clearance.mjs
import {buildJobs, haltonStarts} from '../eval_modes.mjs';
import {BASE_WORLD} from '../episode.mjs';
import {activeWorldFor} from '../../snapshot/rl/env_core.mjs';
import {prepareGeometry, clearance, projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';
import {geometryWithObstacles, obstacleS} from '../runtime/layouts.mjs';
const base = prepareGeometry(BASE_WORLD);
const out = {};
for (const set of ['test', 'validation']) {
  const J = buildJobs('obstacle', set);
  const rows = [];
  for (const j of J) {
    const sp = j.spec; if (!sp.layout) continue;
    const st = sp.start, {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
    const g = geometryWithObstacles(base, sp.layout.obstacles);
    const cAll = clearance(g, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5), cBase = clearance(base, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5);
    const s0 = projectWindow(world.spawn.x, world.spawn.z, 0, LOOP_LENGTH).s;
    const ahead = sp.layout.obstacles.map(o => { let d = ((obstacleS(o) - s0) % LOOP_LENGTH + LOOP_LENGTH) % LOOP_LENGTH; return d; });
    rows.push({id: j.id, gate: st.gate, off: st.left_offset_m, yaw: st.left_yaw_deg, c_all: +cAll.toFixed(4), c_base: +cBase.toFixed(4), first_obstacle_ahead_m: +Math.min(...ahead).toFixed(3)});
  }
  rows.sort((a, b) => a.c_all - b.c_all);
  out[set] = {n: rows.length, below_3cm: rows.filter(r => r.c_all < .03).length, below_5cm: rows.filter(r => r.c_all < .05).length, lowest: rows.slice(0, 8),
    first_obstacle_ahead_min: Math.min(...rows.map(r => r.first_obstacle_ahead_m))};
}
console.log(JSON.stringify(out, null, 1));
