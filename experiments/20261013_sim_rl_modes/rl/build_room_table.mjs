#!/usr/bin/env node
// 使える幅の表(room_table)を作る(2026-10-13、sim_rl_modes の実装の段)。modes.json common.room_table の定義のとおり:
//   R 上の s ごと(5 cm おき)に、車体を R の接線の向きにして横へ 1 cm ずつずらし、全障害物とのすき間 ≥ 0.08 m で動ける左・右の幅。
//   ゲートの s ±0.2 m では |ゲートに対する横| ≤ 0.65 − 0.12、黒シートの区間では車幅の条件(x 1.26〜2.36 m)を満たす範囲に切る。
//   ±0.5 m の最小フィルターのあと、σ 0.4 m のガウスでなめらかにし、左右それぞれ 0.03 m 引いた値を ρ_L(s)・ρ_R(s) とする。
// ほかに: 乱しの始めてよい位置の判定に使う「すき間 5 cm で動ける幅」(corridor.json と同じ作り方)、R の曲率の ±0.5 m の平均 κ̄、取れる最大のすき間。
// 出力: rl/runtime/room_table.json と、ブラウザ用の rl/runtime/room_table.mjs(同じ中身。sha256 は JSON の本文)
//   node rl/build_room_table.mjs
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {prepareGeometry, clearance, loopPointAt, LOOP_LENGTH, GATE_S, CANONICAL_ROUTE, gateStatus} from './runtime/features.mjs';
import {CATALOGUE} from './runtime/catalogue.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const world = JSON.parse(fs.readFileSync(path.join(here, '../snapshot/sim/world-v3.json'), 'utf8'));
const geom = prepareGeometry(world);
const L = LOOP_LENGTH, N = Math.round(L / CATALOGUE.common.room_table.grid_ds_m), DS = L / N;
const OFF_MAX = 1.0, OFF_STEP = .01, NOFF = Math.round(2 * OFF_MAX / OFF_STEP) + 1, I0 = Math.round(OFF_MAX / OFF_STEP);
const C_ROOM = .08, C_W05 = .05, GATE_WIN = .65 - .12, GATE_BAND = .2, MINF = .5, SIGMA = .4, MARGIN = .03;
const VL = .42, VW = .19;
const crossing = world.requiredCrossings[0];
const sheetZone = CATALOGUE.common.zones.black_sheet_zone_s;
const wrapS = s => ((s % L) + L) % L;
const dS = (a, b) => { let d = wrapS(a - b); if (d > L / 2) d -= L; return d; };   // a − b(周を考えた差)

export function buildRoomTable() {
const pt = {x: 0, z: 0, tx: 0, tz: 0}, gs = {along: 0, lateral: 0, index: 0};
const rows = [];
for (let i = 0; i < N; i++) {
  const s = i * DS;
  loopPointAt(s, pt);
  const yaw = Math.atan2(-pt.tz, pt.tx), lx = pt.tz, lz = -pt.tx;
  const cl = new Float64Array(NOFF), clip = new Uint8Array(NOFF);
  // このあたりのゲート(ゲートの s ±0.2 m)
  const gates = [];
  for (let g = 0; g < GATE_S.length; g++) if (Math.abs(dS(s, GATE_S[g])) <= GATE_BAND) gates.push(g);
  const inSheet = s >= sheetZone[0] && s <= sheetZone[1];
  for (let k = 0; k < NOFF; k++) {
    const o = -OFF_MAX + k * OFF_STEP, x = pt.x + o * lx, z = pt.z + o * lz;
    cl[k] = clearance(geom, x, z, yaw, VL, VW, .5);
    let ok = 1;
    for (const g of gates) { gateStatus(CANONICAL_ROUTE, g, x, z, gs); if (Math.abs(gs.lateral) > GATE_WIN) ok = 0; }
    if (inSheet && Math.abs(z - crossing.z) <= .3) {
      const half = VL / 2 * Math.abs(Math.cos(yaw)) + VW / 2 * Math.abs(Math.sin(yaw));
      if (x - half < crossing.minX || x + half > crossing.maxX) ok = 0;
    }
    clip[k] = ok;
  }
  // c 以上のすき間(と切り)の連続区間: 0 を含む区間、0 が入らなければいちばん近い自由な点を含む区間
  const interval = (c, useClip) => {
    const free = k => cl[k] >= c && cl[k] > 1e-9 && (!useClip || clip[k]);
    let center = I0;
    if (!free(I0)) {
      center = -1;
      for (let d = 1; d < NOFF; d++) {
        if (I0 + d < NOFF && free(I0 + d)) { center = I0 + d; break; }
        if (I0 - d >= 0 && free(I0 - d)) { center = I0 - d; break; }
      }
      if (center < 0) return null;
    }
    let lo = center, hi = center;
    while (lo - 1 >= 0 && free(lo - 1)) lo--;
    while (hi + 1 < NOFF && free(hi + 1)) hi++;
    return {right: -OFF_MAX + lo * OFF_STEP, left: -OFF_MAX + hi * OFF_STEP, R_inside: center === I0};
  };
  const r08 = interval(C_ROOM, true), w05 = interval(C_W05, false), any = interval(0, false);
  let maxClear = -1;
  if (any) for (let k = Math.round((any.right + OFF_MAX) / OFF_STEP); k <= Math.round((any.left + OFF_MAX) / OFF_STEP); k++) maxClear = Math.max(maxClear, cl[k]);
  rows.push({s, x: pt.x, z: pt.z, yaw, left08: r08 ? r08.left : 0, right08: r08 ? r08.right : 0, r08_R_inside: r08 ? r08.R_inside : false,
    width05: w05 ? w05.left - w05.right : 0, clear_at_R: cl[I0], max_clear: maxClear, gate_clip: gates.length > 0, sheet_clip: inSheet && Math.abs(pt.z - crossing.z) <= .3});
}
// 最小フィルター(±0.5 m)→ ガウス(σ 0.4 m)→ 0.03 m 引く。左は left08、右は −right08(R の右への幅。R が自由でなければ負になりうる)
const circ = i => ((i % N) + N) % N;
const minFilter = a => { const h = Math.round(MINF / DS); return a.map((_, i) => { let m = Infinity; for (let d = -h; d <= h; d++) m = Math.min(m, a[circ(i + d)]); return m; }); };
const gauss = a => {
  const sg = SIGMA / DS, h = Math.ceil(3 * sg), w = []; let tot = 0;
  for (let d = -h; d <= h; d++) { const v = Math.exp(-.5 * (d / sg) ** 2); w.push(v); tot += v; }
  return a.map((_, i) => { let s = 0; for (let d = -h; d <= h; d++) s += w[d + h] * a[circ(i + d)]; return s / tot; });
};
const rhoL = gauss(minFilter(rows.map(r => r.left08))).map(v => v - MARGIN);
const rhoR = gauss(minFilter(rows.map(r => -r.right08))).map(v => v - MARGIN);
// R の曲率の ±0.5 m の平均 κ̄(s) = (向き(s + 0.5) − 向き(s − 0.5)) / 1.0(左曲がりが正)
const heading = s => { loopPointAt(s, pt); return Math.atan2(-pt.tz, pt.tx); };
const kappa = rows.map(r => { let d = heading(r.s + .5) - heading(r.s - .5); d = Math.atan2(Math.sin(d), Math.cos(d)); return d / 1.0; });
const round = (v, k = 5) => +v.toFixed(k);
const body = {schema: 'jf-sim-rl-room-table/v1', note: 'modes.json common.room_table の定義。s は R 上の位置(0〜L、周期)。値は格子点の値で、間は線形補間',
  loop_length_m: L, n: N, ds_m: DS, clearance_m: C_ROOM, gate_window_m: GATE_WIN, gate_band_m: GATE_BAND, min_filter_m: MINF, sigma_m: SIGMA, margin_m: MARGIN,
  offsets: {max_m: OFF_MAX, step_m: OFF_STEP}, vehicle: {length: VL, width: VW}, black_sheet_zone_s: sheetZone,
  back_straight_zone_s: CATALOGUE.common.zones.back_straight_zone_s,
  rhoL: rhoL.map(v => round(v)), rhoR: rhoR.map(v => round(v)), emid: rhoL.map((v, i) => round((v - rhoR[i]) / 2)),
  left08_raw: rows.map(r => round(r.left08, 3)), right08_raw: rows.map(r => round(r.right08, 3)), width05: rows.map(r => round(r.width05, 3)),
  kappa: kappa.map(v => round(v)), max_clear: rows.map(r => round(r.max_clear, 4)), clear_at_R: rows.map(r => round(r.clear_at_R, 4))};
  return {body, rows, rhoL, rhoR};
}
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
if (isMain) {
const {body, rows, rhoL, rhoR} = buildRoomTable();
const round = (v, k = 5) => +v.toFixed(k);
const json = JSON.stringify(body);
const sha = crypto.createHash('sha256').update(json).digest('hex');
fs.writeFileSync(path.join(here, 'runtime/room_table.json'), json);
fs.writeFileSync(path.join(here, 'runtime/room_table.mjs'), `// 自動生成(rl/build_room_table.mjs)。手で変えない。sha256(room_table.json の本文) = ${sha}\n` +
  `export const ROOM_TABLE_SHA256 = '${sha}';\nexport const ROOM_TABLE = Object.freeze(${json});\n`);
const q = (a, p) => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(p * (b.length - 1))]; };
console.log(JSON.stringify({n: N, ds: DS, sha256: sha, rhoL_q: [0, .05, .5, .95].map(p => round(q(rhoL, p), 3)), rhoR_q: [0, .05, .5, .95].map(p => round(q(rhoR, p), 3)),
  width05_q: [0, .05, .5].map(p => q(rows.map(r => r.width05), p)), max_clear_min: Math.min(...rows.map(r => r.max_clear)),
  R_not_free08: rows.filter(r => !r.r08_R_inside).length, negative_room: rhoL.filter((v, i) => v + rhoR[i] < 0).length}));
}
