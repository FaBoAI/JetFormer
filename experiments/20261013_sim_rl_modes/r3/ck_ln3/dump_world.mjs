// 図のためのコース(柵の線分・柱の足・R の点)を書き出す
import fs from 'node:fs';
import {BASE_WORLD, LOOP} from './ck_run.mjs';
const R = JSON.parse(fs.readFileSync('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/line/rt/reference_loop.json', 'utf8'));
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3/world_fig.json', JSON.stringify({
  segments: BASE_WORLD.segments.map(s => ({a: s.a, b: s.b})), feet: BASE_WORLD.disks.filter(d => d.kind === 'post-foot').map(d => [d.x, d.z]),
  other_disks: BASE_WORLD.disks.filter(d => d.kind !== 'post-foot').map(d => [d.x, d.z, d.radius, d.kind]), bounds: BASE_WORLD.bounds, loop_x: R.x, loop_z: R.z, L: LOOP.L,
  required: BASE_WORLD.requiredCrossings}));
console.log(BASE_WORLD.disks.length, BASE_WORLD.segments.length, JSON.stringify(BASE_WORLD.requiredCrossings), [...new Set(BASE_WORLD.disks.map(d => d.kind))]);
