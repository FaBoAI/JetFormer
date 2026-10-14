// 確認 line r4: 図のための走り直し(フレームつき)。攻めの組で方策だけでは当たる / 3 cm を割る回を、シールドあり・なしで
import fs from 'node:fs';
import {runJob, OB} from './run4.mjs';
const A = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4/advjobs4.json', 'utf8')).jobs;
const ids = process.argv.slice(2);
const out = {obstacles: OB.list.map(o => o.k === 'cap' ? {k: 'cap', a: [o.ax, o.az], b: [o.bx, o.bz], r: o.r, id: o.id} : o.k === 'box' ? {k: 'box', V: o.V, id: o.id} : {k: 'circ', x: o.x, z: o.z, r: o.r, id: o.id}), runs: {}};
for (const id of ids) {
  const j = A.find(q => q.id === id);
  for (const sh of [true, false]) {
    const r = runJob({...j, shield: sh, keep: true});
    out.runs[`${id}:${sh ? 'on' : 'off'}`] = {fr: r.fr.slice(0, 150), min_c: r.min_c, min_f: r.min_f, min_id: r.min_id, collision: r.collision, sh_list: r.sh_list, line: r.line, c_v: r.c_v, gate: r.gate, off: r.off, yaw: r.yaw};
  }
}
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4/traj4.json', JSON.stringify(out));
console.log(Object.entries(out.runs).map(([k, v]) => `${k} ${v.min_c} ${v.min_id} ${JSON.stringify(v.collision)}`).join('\n'));
