// 崩れた回を、自前の走らせ方(フレームつき)と学習の段の走らせ方 r3/line/run_ln3.mjs(JF_R3 でシールド)の両方で走らせ直す(2026-10-14、確認の段)
//   JF_R3='{"modeShield":{"line":{…}}}' node repro_ck.mjs
import fs from 'node:fs';
import {runOne} from './ck_run.mjs';
const T = await import('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line/run_ln3.mjs');
const CAND = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/line_r3/policy.json';
const cases = [
  {id: 'GR-2', start: {gate: 16, left_offset_m: -.2, left_yaw_deg: 14.5, seed: 4723002}, params: {line: {family: 'const', c0: 0}, c_v: .95}},
  {id: 'C129', start: {gate: 16, left_offset_m: -.2, left_yaw_deg: 15, seed: 4713129}, params: {line: {family: 'random', seed: 2155857802}, c_v: .9}},
  {id: 'C137', start: {gate: 17, left_offset_m: -.2, left_yaw_deg: 15, seed: 4713137}, params: {line: {family: 'const', c0: 0}, c_v: 1}},
  {id: 'GR-7', start: {gate: 16, left_offset_m: -.2, left_yaw_deg: 13, seed: 4723007}, params: {line: {family: 'random', seed: 777}, c_v: .95}},
  {id: 'BOX17-44', start: null, params: null},
];
// BOX17-44 の条件を結果の行から
for (const f of fs.readdirSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/runs').filter(f => f.startsWith('box_cand_')))
  for (const l of fs.readFileSync(`/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/runs/${f}`, 'utf8').split('\n')) if (l.trim()) {
    const r = JSON.parse(l); if (r.id === 'BOX17-44') { cases[4].start = r.start; cases[4].params = {line: r.line, c_v: r.c_v}; }
  }
const out = [];
const jf = globalThis.JF_R3;
for (const c of cases) {
  const a = runOne({id: c.id, set: 'REPRO', policy: 'line', rt: 'r3', shield: true, start: c.start, params: c.params, keepFrames: true});
  globalThis.JF_R3 = jf;
  const b = await T.runOne({policy: CAND, mode: 'line', start: c.start, params: c.params, keepFrames: false});
  const row = {id: c.id, start: c.start, params: c.params,
    own: {reason: a.reason, collision: a.collision, min: a.min_clearance, min_t: a.min_t, min_what: a.min_what, min_v: a.min_v, min_s_mod: a.min_s_mod, stall: a.stall, stall_first: a.stall_first, laps: a.laps.map(l => l.elapsed), shield_frames: a.shield_frames, shield_fb: a.shield_fb},
    training_tool: {reason: b.reason, collisions: b.collisions, collision: b.collision, min: b.min_clearance, min_t: b.min_t, laps: b.laps.map(l => l.elapsed), shield_frames: b.labels.ph.shield ?? 0},
    frames: a.fr.filter(q => q[0] % 3 === 0 || q[11] === 1).slice(0, 220)};
  out.push(row);
  console.log(JSON.stringify({id: c.id, own: row.own, training_tool: row.training_tool}));
}
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/repro.json', JSON.stringify(out));
