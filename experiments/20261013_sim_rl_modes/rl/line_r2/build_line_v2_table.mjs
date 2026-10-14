#!/usr/bin/env node
// 経路いろいろ r2 の「目標の線 v2」の幅の表を作る(2026-10-14、学習の段 line r2。protocol_line_r2.json target_line)。
// 使える幅の表(room_table、v1 と同じもの)に、R の曲がりの内側の上限を足す:
//   R の局所の曲率 κ(s)(s ± 0.1 m の 3 点の円、左曲がりが正、σ 0.1 m でなめらかに)から、内側へ寄れる量を
//   b(s) = max(0, 1/|κ| − r_min)(r_min = 0.5 m: 線の曲率半径が 0.5 m を割らない。車の舵の上限 0.52 rad の最小半径は 0.455 m)とし、
//   ±0.5 m の最小フィルター → σ 0.25 m のガウスでなめらかにして、左曲がりでは左の上限、右曲がりでは右の上限にする。
//   hi(s) = min(ρ_L(s), b_L(s))、lo(s) = max(−ρ_R(s), −b_R(s))。目標の点が R の曲率の中心より先になる(e·κ ≥ 1)ことがなくなる(確認 line r1 の指摘 3)。
// 出力: rl/runtime/line_v2_table.json と、ブラウザ用の rl/runtime/line_v2_table.mjs(同じ中身。sha256 は JSON の本文)
//   node rl/line_r2/build_line_v2_table.mjs
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loopPointAt, LOOP_LENGTH} from '../runtime/features.mjs';
import {ROOM_TABLE, ROOM_TABLE_SHA256} from '../runtime/room_table.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
export const PARAMS = Object.freeze({r_min_m: 0.7, kappa_chord_m: .1, kappa_sigma_m: .1, min_filter_m: .5, bound_sigma_m: .25, big_m: 2.0, slope_max: 0.25});
const T = ROOM_TABLE, N = T.n, DS = T.ds_m, L = T.loop_length_m;
if (Math.abs(L - LOOP_LENGTH) > 1e-9) throw new Error('room_table と R の長さが違います');
const circ = i => ((i % N) + N) % N;
const gauss = (a, sigmaM) => {
  const sg = sigmaM / DS, h = Math.ceil(3 * sg), w = []; let tot = 0;
  for (let d = -h; d <= h; d++) { const v = Math.exp(-.5 * (d / sg) ** 2); w.push(v); tot += v; }
  return a.map((_, i) => { let s = 0; for (let d = -h; d <= h; d++) s += w[d + h] * a[circ(i + d)]; return s / tot; });
};
const minFilter = (a, m) => { const h = Math.round(m / DS); return a.map((_, i) => { let v = Infinity; for (let d = -h; d <= h; d++) v = Math.min(v, a[circ(i + d)]); return v; }); };
// 3 点の円の符号つき曲率(sim の座標は x 右・z 下。前 = (cos yaw, −sin yaw)。左曲がりを正にする)
export function signedCurv(x0, z0, x1, z1, x2, z2) {
  const ax = x1 - x0, az = z1 - z0, bx = x2 - x1, bz = z2 - z1, cx = x0 - x2, cz = z0 - z2;
  const cr = ax * bz - az * bx, d = Math.hypot(ax, az) * Math.hypot(bx, bz) * Math.hypot(cx, cz);
  return d > 0 ? -2 * cr / d : 0;
}
export function buildLineV2Table(P = PARAMS) {
  const h = P.kappa_chord_m, kRaw = [];
  for (let i = 0; i < N; i++) {
    const s = i * DS, p = [s - h, s, s + h].map(q => loopPointAt(q, {x: 0, z: 0, tx: 0, tz: 0}));
    kRaw.push(signedCurv(p[0].x, p[0].z, p[1].x, p[1].z, p[2].x, p[2].z));
  }
  const kappa = gauss(kRaw, P.kappa_sigma_m);
  const bIn = kappa.map(k => Math.abs(k) > 1e-3 ? Math.min(P.big_m, Math.max(0, 1 / Math.abs(k) - P.r_min_m)) : P.big_m);
  const inL = gauss(minFilter(kappa.map((k, i) => k > 0 ? bIn[i] : P.big_m), P.min_filter_m), P.bound_sigma_m);
  const inR = gauss(minFilter(kappa.map((k, i) => k < 0 ? bIn[i] : P.big_m), P.min_filter_m), P.bound_sigma_m);
  const hi0 = [], lo0 = [];
  for (let i = 0; i < N; i++) { hi0.push(Math.min(T.rhoL[i], inL[i])); lo0.push(Math.max(-T.rhoR[i], -inR[i])); }
  // 傾きの上限(|d hi/ds|・|d lo/ds| ≤ slope_max): 上限は「円すいの下の包絡」hi[i] = min_j (hi0[j] + g·|i − j|·ds)、下限はその逆。
  // 幅が急に開く所(R の曲がりの出口など)で、目標の線が 1 m 足らずで大きく振れるのを防ぐ(線の曲率半径を車が曲がれる大きさに保つ)
  const g = P.slope_max, hi = [], lo = [];
  for (let i = 0; i < N; i++) {
    let h = Infinity, l = -Infinity;
    for (let j = 0; j < N; j++) { let d = Math.abs(i - j); d = Math.min(d, N - d) * DS; h = Math.min(h, hi0[j] + g * d); l = Math.max(l, lo0[j] - g * d); }
    hi.push(h); lo.push(l);
  }
  const r5 = a => a.map(v => +v.toFixed(5));
  return {schema: 'jf-sim-rl-line-v2-table/v1', note: 'protocol_line_r2.json target_line の幅の表。s は R 上の位置(0〜L、周期)。値は格子点の値で、間は線形補間。hi = 左の上限、lo = 右の上限(負が右)',
    loop_length_m: L, n: N, ds_m: DS, room_table_sha256: ROOM_TABLE_SHA256, params: {...P}, kappa: r5(kappa), hi: r5(hi), lo: r5(lo), in_left: r5(inL), in_right: r5(inR)};
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const t = buildLineV2Table(), body = JSON.stringify(t), sha = crypto.createHash('sha256').update(body).digest('hex');
  const out = path.join(here, '../runtime');
  fs.writeFileSync(path.join(out, 'line_v2_table.json'), body);
  fs.writeFileSync(path.join(out, 'line_v2_table.mjs'), `// 自動生成(rl/line_r2/build_line_v2_table.mjs)。手で変えない。sha256(line_v2_table.json の本文) = ${sha}\n` +
    `export const LINE_V2_TABLE_SHA256 = '${sha}';\nexport const LINE_V2_TABLE = Object.freeze(${body});\n`);
  let maxEk = 0, nLoHi = 0;
  for (let i = 0; i < t.n; i++) { maxEk = Math.max(maxEk, t.hi[i] * t.kappa[i], -t.lo[i] * t.kappa[i]); if (t.lo[i] > t.hi[i]) nLoHi++; }
  console.log(JSON.stringify({sha256: sha, n: t.n, max_e_kappa_at_bounds: +maxEk.toFixed(3), lo_gt_hi_points: nLoHi}));
}
