#!/usr/bin/env node
// 経路いろいろ r2 の下調べ: 目標の線そのもの(車が目標の上を正確に走ったとき)の L4(線の違い)と横ずれのばらつき(VM3 に近い値)。v1 と v2 を比べる
//   node rl/line_r2/target_l4.mjs
import {createLineProfile} from '../runtime/line_profiles.mjs';
import {LOOP_LENGTH} from '../runtime/features.mjs';
const L = LOOP_LENGTH, FAM6 = {'c-0.8': {family: 'const', c0: -.8}, c0: {family: 'const', c0: 0}, 'c+0.8': {family: 'const', c0: .8},
  inside: {family: 'inside', a: .8}, outside: {family: 'outside', a: .8}, late_apex: {family: 'late_apex', a: .8}};
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
export function targetL4(version) {
  const bins = []; for (let s = 3; s < 3 + L; s += .05) bins.push(s);
  const P = Object.fromEntries(Object.entries(FAM6).map(([k, p]) => { const pr = createLineProfile(p, 0, {version}); if (version === 'v2') pr.anchor(0); return [k, bins.map(s => pr.eStar(s))]; }));
  const d = (a, b) => P[a].map((v, i) => Math.abs(v - P[b][i]));
  const ids = Object.keys(P), pairs = [];
  for (let a = 0; a < ids.length; a++) for (let b = a + 1; b < ids.length; b++) {
    const dd = d(ids[a], ids[b]); let best = 0;
    for (let i = 0; i + 20 <= dd.length; i++) { let s = 0; for (let k = 0; k < 20; k++) s += dd[i + k]; best = Math.max(best, s / 20); }
    pairs.push([`${ids[a]}~${ids[b]}`, +best.toFixed(3)]);
  }
  pairs.sort((x, y) => x[1] - y[1]);
  return {left_right_mean: +mean(d('c+0.8', 'c-0.8')).toFixed(3), inside_outside_mean: +mean(d('inside', 'outside')).toFixed(3), weakest_pairs: pairs.slice(0, 3)};
}
for (const v of ['v1', 'v2']) console.log(v, JSON.stringify(targetL4(v)));
