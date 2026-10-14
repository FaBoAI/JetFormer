// 確認 line r3(再実行): 走らせ方の試し。前の確認で崩れた 5 回と、試験の P36・P37 を自前で走らせて照らす
import fs from 'node:fs';
import {run} from './run_b.mjs';
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3';
const rp = JSON.parse(fs.readFileSync(`${REP}/check_line_r3/repro.json`, 'utf8'));
const ev = JSON.parse(fs.readFileSync(`${REP}/line_r3/eval_test_r3.json`, 'utf8'));
const jobs = rp.map(x => ({id: x.id, set: 'REPRO', shield: true, gate: x.start.gate, off: x.start.left_offset_m, yaw: x.start.left_yaw_deg, seed: x.start.seed, params: x.params, _prev: x.own}));
for (const k of [36, 37]) { const t = ev.tests['LN-P60-R3'][k]; jobs.push({id: t.id, set: 'TEST', shield: true, gate: t.start.gate, off: t.start.left_offset_m, yaw: t.start.left_yaw_deg, seed: t.start.seed, params: {line: t.line, c_v: t.c_v}, _prev: t}); }
const t37 = ev.tests['LN-P60-R3'][37];
jobs.push({id: 'P37-noshield', set: 'TEST', shield: false, gate: t37.start.gate, off: t37.start.left_offset_m, yaw: t37.start.left_yaw_deg, seed: t37.start.seed, params: {line: t37.line, c_v: t37.c_v}, _prev: null});
for (const j of jobs) {
  const t0 = Date.now(), r = run(j), p = j._prev;
  console.log(j.id, `${((Date.now() - t0) / 1000).toFixed(1)}s`, r.reason, 'col', r.collision ? JSON.stringify(r.collision) : '-', 'min', r.min_c, r.min_id, 'stall', r.stall_first, 'sh', r.sh_frames,
    'laps', JSON.stringify(r.laps.map(l => l.t)), 'integ', r.integ_err.toExponential(1), 'eyrt', r.ey_vs_rt_max, 'dpsi', r.start.dpsi, 'c0', r.start.clearance, 'lab', JSON.stringify(r.labels));
  if (p) console.log('   prev:', p.reason, 'min', p.min ?? p.min_clearance, 'stall', p.stall_first ?? p.stall_first_frame, 'sh', p.shield_frames, 'laps', JSON.stringify(p.laps?.map?.(l => l.elapsed ?? l) ?? null));
}
