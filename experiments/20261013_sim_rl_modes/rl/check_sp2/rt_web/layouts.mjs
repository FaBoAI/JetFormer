// 障害物よけの配置(2026-10-13、sim_rl_modes)。Node とブラウザで同じファイル(純粋な ESM)。modes.json common.obstacle_layouts のとおり:
//   柱(円柱、半径 0.05〜0.10 m・高さ 0.30 m)と箱(カプセル、長さ 0.15〜0.35 m・半径 0.05 m・向き自由・高さ 0.25 m)を 1〜4 個、s で 2 m 以上離して、
//   最速の方策のふだんの線 ē_y(s) + U(−0.15, 0.15) m に置く。黒シートの区間・奥の直線・開始の前後 1.5 m・ゲートの柱から 0.3 m 以内には置かない。
//   通り道の保証: 障害物の s ±0.6 m のどこでも、全障害物とのすき間 ≥ 0.08 m で通れる横の範囲が 0.10 m 以上残り、その範囲がゲートの窓(±0.65 m)と重なる。
// 形の約束: post = {shape: 'post', x, z, r, h}、box = {shape: 'box', x, z, length(カプセルの芯の線分の長さ), yaw(sim の向き), r: 0.05, h}。
// world には kind 'obstacle' の disks(柱)・segments(箱)として足す(sim-core・features.mjs とも、高さ > 0 のものを全部障害物として扱う)。
import {prepareGeometry, clearance, projectWindow, loopPointAt, LOOP_LENGTH} from './features.mjs';
import {ROOM_TABLE} from './room_table.mjs';
import {createRng, mix32} from './rng.mjs';
import {CATALOGUE} from './catalogue.mjs';

export const LAYOUT_SCHEMA = 'jf-sim-rl-obstacle-layout/v1';
const Z = CATALOGUE.common.zones, N = ROOM_TABLE.n, DS = ROOM_TABLE.ds_m, L = LOOP_LENGTH;
const OFF_MAX = 1.0, OFF_STEP = .01, NOFF = Math.round(2 * OFF_MAX / OFF_STEP) + 1;
export const PASSAGE = Object.freeze({half_window_s_m: .6, clearance_m: .08, min_width_m: .10, gate_window_m: .65});
const VL = .42, VW = .19;
const modS = s => ((s % L) + L) % L;
const dS = (a, b) => { let d = modS(a - b); if (d > L / 2) d -= L; return d; };

export function boxEnds(o) {
  const h = o.length / 2, ux = Math.cos(o.yaw), uz = -Math.sin(o.yaw);
  return [[o.x - h * ux, o.z - h * uz], [o.x + h * ux, o.z + h * uz]];
}
// 障害物 → world の項目
export function obstacleToWorldEntry(o) {
  if (o.shape === 'post') return {kind: 'disk', entry: {x: o.x, z: o.z, radius: o.r, height: o.h ?? .3, kind: 'obstacle'}};
  if (o.shape === 'box') { const [a, b] = boxEnds(o); return {kind: 'segment', entry: {a, b, radius: o.r ?? .05, height: o.h ?? .25, kind: 'obstacle'}}; }
  throw new Error('layouts: 知らない形です: ' + o.shape);
}
// world(回した経路の world でよい)に障害物を足した写し(元は変えない)
export function worldWithObstacles(world, obstacles) {
  const segs = [...(world.segments ?? [])], disks = [...(world.disks ?? [])];
  for (const o of obstacles ?? []) { const e = obstacleToWorldEntry(o); (e.kind === 'disk' ? disks : segs).push(e.entry); }
  return {...world, segments: segs, disks};
}
// 形(features.mjs の prepareGeometry と同じ中身)を、キャッシュを通さずに作る(配置を回ごとに変えてもメモリが増えない)
export function geometryWithObstacles(baseGeometry, obstacles) {
  if (!obstacles?.length) return baseGeometry;
  const segments = [...baseGeometry.segments], circles = [...baseGeometry.circles];
  for (const o of obstacles) {
    const e = obstacleToWorldEntry(o).entry;
    if (o.shape === 'post') circles.push({x: e.x, z: e.z, r: e.radius, reach: e.radius, kind: 'obstacle'});
    else {
      const [a, b] = [e.a, e.b], mx = (a[0] + b[0]) / 2, mz = (a[1] + b[1]) / 2;
      segments.push({ax: a[0], az: a[1], bx: b[0], bz: b[1], r: e.radius, cx: mx, cz: mz, reach: Math.hypot(b[0] - a[0], b[1] - a[1]) / 2 + e.radius, kind: 'obstacle'});
    }
  }
  return Object.freeze({segments, circles, rects: baseGeometry.rects, bounds: baseGeometry.bounds, surfaces: baseGeometry.surfaces,
    boundSegments: baseGeometry.boundSegments, counts: {segments: segments.length, circles: circles.length, rects: baseGeometry.rects.length}});
}
// 足した障害物だけの形(外枠は遠くに置く。報酬の d_obs 用)
export function obstacleOnlyGeometry(obstacles) {
  const far = 1e5, empty = {segments: [], circles: [], rects: [], bounds: {minX: -far, maxX: far, minZ: -far, maxZ: far}, surfaces: [], boundSegments: []};
  return geometryWithObstacles(empty, obstacles);
}

// 標準のコースのすき間の格子(R の s の格子 × 横 1 cm、車体は R の接線の向き)。通り道の検査を速くする(学習の環境で 1 回作る)
export function buildBaseClearanceGrid(baseGeometry) {
  const grid = new Float32Array(N * NOFF), pt = {x: 0, z: 0, tx: 0, tz: 0};
  for (let i = 0; i < N; i++) {
    loopPointAt(i * DS, pt);
    const yaw = Math.atan2(-pt.tz, pt.tx);
    for (let k = 0; k < NOFF; k++) { const o = -OFF_MAX + k * OFF_STEP; grid[i * NOFF + k] = clearance(baseGeometry, pt.x + o * pt.tz, pt.z - o * pt.tx, yaw, VL, VW, .5); }
  }
  return grid;
}
export function obstacleS(o) { const p = projectWindow(o.x, o.z, 0, L); return p.s; }
// 通り道の検査。戻り値 {ok, per_obstacle: [{s, ok, min_width}]}。grid が無ければ毎回すき間を計算する(遅いが同じ結果)
export function checkPassage(baseGeometry, obstacles, {grid = null} = {}) {
  const added = obstacleOnlyGeometry(obstacles), pt = {x: 0, z: 0, tx: 0, tz: 0}, out = [];
  let ok = true;
  for (const o of obstacles) {
    const so = obstacleS(o);
    let minWidth = Infinity;
    for (let d = -Math.round(PASSAGE.half_window_s_m / DS); d <= Math.round(PASSAGE.half_window_s_m / DS); d++) {
      const i = ((Math.round(so / DS) + d) % N + N) % N;
      loopPointAt(i * DS, pt);
      const yaw = Math.atan2(-pt.tz, pt.tx);
      let best = 0, run = 0, runLo = 0;
      for (let k = 0; k < NOFF; k++) {
        const off = -OFF_MAX + k * OFF_STEP, x = pt.x + off * pt.tz, z = pt.z - off * pt.tx;
        const base = grid ? grid[i * NOFF + k] : clearance(baseGeometry, x, z, yaw, VL, VW, .5);
        const free = base >= PASSAGE.clearance_m && clearance(added, x, z, yaw, VL, VW, .5) >= PASSAGE.clearance_m;
        if (free) { if (!run) runLo = off; run++; }
        if (!free || k === NOFF - 1) {
          if (run) {
            const lo = runLo, hi = runLo + (run - 1) * OFF_STEP, w = hi - lo;
            // ゲートの窓(R から ±0.65 m)と重なる部分の幅
            const olo = Math.max(lo, -PASSAGE.gate_window_m), ohi = Math.min(hi, PASSAGE.gate_window_m);
            if (ohi - olo >= PASSAGE.min_width_m - 1e-9 && w > best) best = Math.min(w, ohi - olo);
          }
          run = 0;
        }
      }
      minWidth = Math.min(minWidth, best);
    }
    const good = minWidth >= PASSAGE.min_width_m - 1e-9;
    out.push({s: so, ok: good, min_width: minWidth});
    if (!good) ok = false;
  }
  return {ok, per_obstacle: out};
}
// s が置いてよい位置か(黒シートの区間・奥の直線・開始の前後 1.5 m を除く)
export function placementAllowedS(s, startS = null) {
  const m = modS(s);
  for (const z of [Z.black_sheet_zone_s, Z.back_straight_zone_s]) if (m >= z[0] - .3 && m <= z[1] + .3) return false;
  if (startS != null && Math.abs(dS(s, startS)) < 1.5 + .3) return false;
  return true;
}
// 1 つの配置を作る。nominalLine = 最速の方策のふだんの線(room_table の格子の ē_y)。gatePosts = [{x, z}](ゲートの柱)
export function generateLayout(rng, {baseGeometry, nominalLine, gatePosts = [], count = [1, 4], startS = null, grid = null, maxTries = 60, id = null}) {
  const n = Math.min(count[1], count[0] + Math.floor(rng.uniform() * (count[1] - count[0] + 1)));
  const obstacles = [], pt = {x: 0, z: 0, tx: 0, tz: 0};
  for (let tries = 0; obstacles.length < n && tries < maxTries; tries++) {
    const u = Array.from({length: 7}, () => rng.uniform());
    const s = u[0] * L;
    if (!placementAllowedS(s, startS)) continue;
    if (obstacles.some(o => Math.abs(dS(s, o.s_ref)) < 2.0)) continue;
    const i = Math.round(s / DS) % N, lat = (nominalLine ? nominalLine[i] : 0) + (-.15 + .3 * u[1]);
    loopPointAt(s, pt);
    const x = pt.x + lat * pt.tz, z = pt.z - lat * pt.tx;
    const o = u[2] < .5 ? {shape: 'post', x, z, r: .05 + .05 * u[3], h: .3}
      : {shape: 'box', x, z, length: .15 + .2 * u[4], yaw: Math.PI * u[5], r: .05, h: .25};
    if (gatePosts.some(g => Math.hypot(g.x - x, g.z - z) < .3 + (o.shape === 'box' ? o.length / 2 + o.r : o.r))) continue;
    // コースの障害物と重ならない(すき間 5 cm 以上)
    const og = obstacleOnlyGeometry([o]);
    let tooClose = false;
    for (const c of baseGeometry.circles) if (Math.hypot(c.x - x, c.z - z) - c.r < (o.shape === 'box' ? o.length / 2 : 0) + o.r + .05) {
      if (clearance(og, c.x, c.z, 0, 0, 0, .5) < c.r + .05) tooClose = true; }
    for (const c of baseGeometry.rects) if (Math.hypot(c.x - x, c.z - z) < 1 && clearance(og, c.x, c.z, 0, 0, 0, .5) < c.reach + .05) tooClose = true;
    for (const g of baseGeometry.segments) if (Math.hypot(g.cx - x, g.cz - z) - g.reach < 1) {
      // 線分の近くの点をいくつか調べる
      for (let t = 0; t <= 1 && !tooClose; t += .05) { const px = g.ax + (g.bx - g.ax) * t, pz = g.az + (g.bz - g.az) * t; if (clearance(og, px, pz, 0, 0, 0, .5) < g.r + .05) tooClose = true; }
    }
    if (tooClose) continue;
    const trial = [...obstacles, o];
    if (!checkPassage(baseGeometry, [o], {grid}).ok) continue;
    if (!checkPassage(baseGeometry, trial, {grid}).ok) continue;
    o.s_ref = +s.toFixed(4);
    obstacles.push(o);
  }
  for (const o of obstacles) for (const k of ['x', 'z', 'r', 'length', 'yaw']) if (o[k] !== undefined) o[k] = +o[k].toFixed(5);
  return {id, obstacles};
}
// 組(validation・test・recording)の配置。種 seedBase + i
export function generateLayoutSet(name, seedBase, count, opts) {
  const layouts = [];
  for (let i = 0; i < count; i++) {
    const rng = createRng(mix32(seedBase + i));
    const lay = generateLayout(rng, {...opts, id: `${name}-${String(i).padStart(3, '0')}`});
    lay.seed = seedBase + i;
    layouts.push(lay);
  }
  return {schema: LAYOUT_SCHEMA, id: name, base: 'default', seed_base: seedBase, layouts};
}
// world の kind 'obstacle' の項目 → 障害物の一覧(ブラウザでは重ね描きが world に入れたものを読む)
export function obstaclesFromWorld(world) {
  const out = [];
  for (const d of world.disks ?? []) if (d && d.kind === 'obstacle' && d.height > 0) out.push({shape: 'post', x: d.x, z: d.z, r: d.radius, h: d.height});
  for (const g of world.segments ?? []) if (g && g.kind === 'obstacle' && g.height > 0) {
    const dx = g.b[0] - g.a[0], dz = g.b[1] - g.a[1];
    out.push({shape: 'box', x: (g.a[0] + g.b[0]) / 2, z: (g.a[1] + g.b[1]) / 2, length: Math.hypot(dx, dz), yaw: Math.atan2(-dz, dx), r: g.radius, h: g.height});
  }
  return out;
}
// world から足した障害物を除いたもの(元のコースの形の確認用)
export function worldWithoutObstacles(world) {
  return {...world, disks: (world.disks ?? []).filter(d => d?.kind !== 'obstacle'), segments: (world.segments ?? []).filter(g => g?.kind !== 'obstacle')};
}
// 開始の前後 1.5 m に障害物が無いか(記録・試験の開始の条件)
export function startClearOfObstacles(startS, obstacles) {
  return (obstacles ?? []).every(o => Math.abs(dS(o.s_ref ?? obstacleS(o), startS)) >= 1.5);
}
