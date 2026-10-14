// 確認 line r4: 道具の試し(FAM6 の c0 と、試験の G30R120 の G0 を走らせ、学習の段の値と照らす)
import {runJob} from './run4.mjs';
const t0 = Date.now();
const a = runJob({id: 'smoke-c0', set: 'smoke', shield: true, gate: 0, off: 0, yaw: 0, seed: 1, line: {family: 'const', c0: 0}, c_v: 1, infl: [.03, .04]});
console.log(JSON.stringify({id: a.id, reason: a.reason, laps: a.laps.map(l => l.t), min_c: a.min_c, min_id: a.min_id, integ: a.integ_err, sh: a.sh_frames, eyrt: a.ey_vs_runtime_max, trk: a.trk, p0: a.labels.p0, lab: {...a.labels, p0: undefined}, meta: a.meta_shield, infl: a.infl, s: a.s_travel, ms: Date.now() - t0}));
const b = runJob({id: 'smoke-g0', set: 'smoke', shield: true, gate: 29, off: 0.18417643836699427, yaw: 13.918949292507023, seed: 9300000, line: {family: 'const', c0: -0.8737}, c_v: 0.9636, infl: [.03]});
console.log(JSON.stringify({id: b.id, reason: b.reason, laps: b.laps.map(l => l.t), min_c: b.min_c, min_id: b.min_id, integ: b.integ_err, sh: b.sh_frames, shl: b.sh_list, fb: b.fb, trk: b.trk, start: b.start, infl: b.infl, eyrt: b.ey_vs_runtime_max, ms: Date.now() - t0}));
