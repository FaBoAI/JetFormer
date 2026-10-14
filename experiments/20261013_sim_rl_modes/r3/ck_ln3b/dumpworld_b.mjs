// 確認 line r3(再実行): 図のためにコースの障害物と R の点を書き出す
import fs from 'node:fs';
import {WORLD, OB, R} from './run_b.mjs';
const pts = [];
for (let s = 0; s < R.L; s += .05) pts.push(R.at(s).map(v => +v.toFixed(4)));
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3b/world_b.json', JSON.stringify({ob: OB.list, bounds: OB.bounds, route: WORLD.route, R: pts, rc: WORLD.requiredCrossings}));
console.log(OB.list.length, R.L);
