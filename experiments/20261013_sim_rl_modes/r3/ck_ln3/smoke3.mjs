import fs from 'node:fs';
import {runOne} from './ck_run.mjs';
const ev = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3/eval_test_r3.json', 'utf8'));
const rows = Object.values(ev.tests).flat().filter(r => ['H48', 'H3', 'H33', 'P37', 'P0', 'E10'].includes(r.id));
for (const t of rows) {
  const r = runOne({id: t.id, set: 'S', policy: 'line', rt: 'r3', shield: true, start: {gate: t.start.gate, left_offset_m: t.start.left_offset_m, left_yaw_deg: t.start.left_yaw_deg, seed: t.start.seed}, params: {line: t.line, c_v: t.c_v}, keepFrames: true});
  let dsMax = 0, eyMax = 0, dsMax1 = 0;
  for (const q of r.fr) { const d = Math.abs(q[7] - q[12]); if (q[0] > 90) { dsMax = Math.max(dsMax, d); eyMax = Math.max(eyMax, Math.abs(q[8] - q[13])); } else dsMax1 = Math.max(dsMax1, d); }
  console.log(t.id, r.laps.map(l => +l.own_ds.toFixed(3)), 's_travel', r.s_travel.toFixed(2), 'trk', r.tracking.median.toFixed(4), r.tracking.p95.toFixed(4), 'own-vs-rt ds(f>90)', dsMax.toFixed(4), 'ey', eyMax.toFixed(4), 'ds(f<=90)', dsMax1.toFixed(3));
}
