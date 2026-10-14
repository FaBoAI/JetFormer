// 図のためにコースの形(柵・柱・柱の足)と R・P45 の目標の線を JSON に書く(2026-10-14、確認の段)
import fs from 'node:fs';
import {createOwnV2} from './own_line_v2.mjs';
import {rAt, L_OWN} from '../check_ln/own_line.mjs';
const REL = '/home/jetson/jetformer/data/sim/releases/20261013-144013-6f610dd5';
const {loadCourseWorld} = await import(`${REL}/qa/load-course-world.mjs`);
const w = await loadCourseWorld();
const R = []; for (let s = 0; s < L_OWN; s += .02) { const p = rAt(s); R.push([+p.x.toFixed(4), +p.z.toFixed(4)]); }
const run = fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_line_r2/runs/REPLAY.jsonl', 'utf8').split('\n').filter(Boolean).map(JSON.parse).find(r => r.id === 'P45');
const s0 = run.start_s_mod, e0 = run.start_ey;
const prof = createOwnV2(run.params.line, s0, e0), T = [];
for (let s = s0 - .5; s < s0 + 5; s += .02) { const p = rAt(s), e = prof.eStar(s); T.push([+(p.x + e * p.nx).toFixed(4), +(p.z + e * p.nz).toFixed(4)]); }
const L = []; { const pl = createOwnV2(run.params.line, s0, null); for (let s = s0 - .5; s < s0 + 5; s += .02) { const p = rAt(s), e = pl.eStar(s); L.push([+(p.x + e * p.nx).toFixed(4), +(p.z + e * p.nz).toFixed(4)]); } }
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_line_r2/world_fig.json', JSON.stringify({segments: (w.segments ?? []).filter(s => (s.height ?? .1) > 0).map(s => ({a: s.a, b: s.b, r: s.radius, kind: s.kind})),
  disks: (w.disks ?? []).filter(d => (d.height ?? .1) > 0).map(d => ({x: d.x, z: d.z, r: d.radius, kind: d.kind})), bounds: w.bounds, R, target_blend: T, target_noblend: L,
  p45: {path: run.path, min_frame: run.min_frame, min_clearance: run.min_clearance}}));
console.log('ok', (w.segments ?? []).length, (w.disks ?? []).length);
