// 自前の射影の確かめ: ゲート 17 の大きく横を向いた開始(H48・H3・H33)と P37 などで、周ごとの自前の s の進みが L になるか、e_y がランタイムと合うか
import fs from 'node:fs';
import {runOne} from './ck_run.mjs';
const ev = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3/eval_test_r3.json', 'utf8'));
const rows = Object.values(ev.tests).flat().filter(r => ['H48', 'H3', 'H33', 'P37', 'P0', 'E10'].includes(r.id));
for (const t of rows) {
  const r = runOne({id: t.id, set: 'S', policy: 'line', rt: 'r3', shield: true, start: {gate: t.start.gate, left_offset_m: t.start.left_offset_m, left_yaw_deg: t.start.left_yaw_deg, seed: t.start.seed}, params: {line: t.line, c_v: t.c_v}, keepFrames: true});
  console.log(t.id, r.laps.map(l => +l.own_ds.toFixed(3)), 's_travel', r.s_travel.toFixed(2), 'trk', r.tracking.median.toFixed(4), r.tracking.p95.toFixed(4), 'ey_vs_rt_after3m', r.ey_own_vs_rt_after3m_max.toExponential(2), 'es own-rt', r.es_check.own_vs_rt_max.toExponential(2));
}
