// 経路いろいろ(line)の確認 r1: 目標の点が R の曲率の中心より先になる所(e*·κ_R ≥ 1)を形の強さごとに探す(2026-10-14、確認の段)。
// κ_R は R の点の列の向きの s ± 0.1 m の差(局所の曲率。表の κ̄ は ±0.5 m の平均なので使わない)。
//   node cusp_scan.mjs > 結果.json
import {eStarOwn, rAt, L} from './own_line.mjs';
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const hd = s => { const r = rAt(s); return Math.atan2(-r.tz, r.tx); };
const kLoc = s => wrap(hd(s + .1) - hd(s - .1)) / .2;
const out = {};
const scan = (nm, p) => {
  let mx = -Infinity, at = null; const zone = [];
  for (let s = 0; s < L; s += .01) {
    const v = eStarOwn(p, s) * kLoc(s);
    if (v > mx) { mx = v; at = s; }
    if (v >= 1) zone.push(s);
  }
  out[nm] = {max_e_kappa: +mx.toFixed(3), at_s: +at.toFixed(2), len_ge1_m: +(zone.length * .01).toFixed(2), s_range: zone.length ? [+zone[0].toFixed(2), +zone.at(-1).toFixed(2)] : null};
};
for (const c0 of [.5, .6, .65, .7, .75, .8, .85, .9]) scan(`const c0=${c0}`, {family: 'const', c0});
for (const c0 of [-.6, -.75, -.9]) scan(`const c0=${c0}`, {family: 'const', c0});
for (const f of ['inside', 'outside', 'late_apex', 'early_apex']) for (const a of [.5, .6, .7, .8, .9]) scan(`${f} a=${a}`, {family: f, a});
console.log(JSON.stringify(out));
