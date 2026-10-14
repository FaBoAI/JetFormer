// 動作の確かめ: 試験の P37(シールドが働く回)を自前の走らせ方で走らせ、候補の F-ID・学習の段の行と比べる
import fs from 'node:fs';
import {runOne} from './ck_run.mjs';
const C = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/r3/candidates/line_r3';
const fid = JSON.parse(fs.readFileSync(`${C}/fid_p37_shield.json`, 'utf8'));
const t0 = Date.now();
const r = runOne({id: 'P37', set: 'SMOKE', policy: 'line', rt: 'r3', shield: true, start: {...fid.variation, seed: fid.config.seed}, params: fid.params, keepFrames: true, sampleCheck: true});
let dmax = 0, n = 0;
for (let k = 0; k < fid.frames.length; k++) { const a = fid.frames[k].action, b = r.fr[k]; dmax = Math.max(dmax, Math.abs(a.steering - b[5]), Math.abs(a.speed - b[6])); n++; }
const {fr, bins, ...rest} = r;
console.log(JSON.stringify({ms: Date.now() - t0, fid_cmd_maxdiff_4dp: dmax, n, ...rest}, null, 0).slice(0, 4000));
