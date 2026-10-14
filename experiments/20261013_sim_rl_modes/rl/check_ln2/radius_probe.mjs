// 目標の線 v2 の曲率半径を、弦の長さを変えて測る(折れ線の細かいでこぼこを除くため)。R そのもの(e = 0)とも比べる
import {createOwnV2, L} from './own_line_v2.mjs';
import {rAt} from '../check_ln/own_line.mjs';
const fams = {R: null, 'c-0.8': {family: 'const', c0: -.8}, 'c+0.8': {family: 'const', c0: .8}, inside: {family: 'inside', a: .8}, 'inside0.9': {family: 'inside', a: .9}, late: {family: 'late_apex', a: .8}, 'c+0.9': {family: 'const', c0: .9}, outside: {family: 'outside', a: .8}};
const res = {};
for (const [nm, p] of Object.entries(fams)) {
  const own = p ? createOwnV2(p, 0, null) : null, pts = [];
  for (let k = 0; k < Math.round(L / .01); k++) { const s = k * .01, e = own ? own.eStar(s) : 0, r = rAt(s); pts.push({s, x: r.x + e * r.nx, z: r.z + e * r.nz}); }
  res[nm] = {};
  for (const h of [4, 10, 20]) {
    let rMin = Infinity, at = 0;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[(i - h + pts.length) % pts.length], b = pts[i], c = pts[(i + h) % pts.length];
      const ax = b.x - a.x, az = b.z - a.z, bx = c.x - b.x, bz = c.z - b.z, cx = c.x - a.x, cz = c.z - a.z;
      const den = Math.hypot(ax, az) * Math.hypot(bx, bz) * Math.hypot(cx, cz), k = den > 0 ? 2 * Math.abs(ax * bz - az * bx) / den : 0;
      if (k > 0 && 1 / k < rMin) { rMin = 1 / k; at = b.s; }
    }
    res[nm][`chord_pm${h}cm`] = [+rMin.toFixed(3), +at.toFixed(2)];
  }
}
console.log(JSON.stringify(res));
