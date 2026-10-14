// 速さいろいろ(speed)の確認 r1 の自前の道具(2026-10-14、確認の段)。学習・評価のコード(features.mjs・variation_metrics.mjs・lap_audit.mjs)を使わずに書いた。
//   - sampleClearance: 車体の長方形の周を 4 mm おきの点にし、各点と障害物(カプセル・柱・柱の足・外枠)の距離の最小 + 障害物の角・端・中心が車体の中にあるか。
//     own_geom.mjs(多角形どうしの正確な距離)の確かめ用。点で測るので、正しければ「点の値 ≥ 正確な値」で差は数 mm まで
//   - createProjector: 参照ループ R(runtime/reference_loop.json の点の列。データとして読むだけ)への自前の射影。進み s は周をまたいで増え続け、
//     横ずれ e_y は左が正(車体の左 = (−sin ψ, −cos ψ)。舵が正 = 左で ψ が増える)
import fs from 'node:fs';

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
function ptSeg(px, pz, ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az, l2 = dx * dx + dz * dz;
  const t = l2 > 0 ? clamp(((px - ax) * dx + (pz - az) * dz) / l2, 0, 1) : 0;
  return Math.hypot(px - ax - t * dx, pz - az - t * dz);
}
// 車体の 4 隅(前 = (cos ψ, −sin ψ)、横 = (sin ψ, cos ψ)。長方形なので横の向きの符号は関係ない)
function corners(x, z, yaw, L, W) {
  const fx = Math.cos(yaw), fz = -Math.sin(yaw), lx = Math.sin(yaw), lz = Math.cos(yaw), a = L / 2, b = W / 2;
  return [[x + a * fx + b * lx, z + a * fz + b * lz], [x + a * fx - b * lx, z + a * fz - b * lz],
    [x - a * fx - b * lx, z - a * fz - b * lz], [x - a * fx + b * lx, z - a * fz + b * lz]];
}
// 点が車体の中か(車体の座標に直して |前| ≤ L/2・|横| ≤ W/2)
function insideCar(px, pz, x, z, yaw, L, W) {
  const dx = px - x, dz = pz - z, f = dx * Math.cos(yaw) - dz * Math.sin(yaw), l = dx * Math.sin(yaw) + dz * Math.cos(yaw);
  return Math.abs(f) <= L / 2 && Math.abs(l) <= W / 2;
}
function ptRect(px, pz, cx, cz, hw, hd) {
  const dx = Math.max(Math.abs(px - cx) - hw, 0), dz = Math.max(Math.abs(pz - cz) - hd, 0);
  return Math.hypot(dx, dz);
}
// ob = own_geom.mjs buildObstacles(world) の形(caps・circles・rects・bounds)。柱の足は 0.16 × 0.15 m の軸にそろった長方形
export function sampleClearance(ob, x, z, yaw, L = .42, W = .19, step = .004) {
  const C = corners(x, z, yaw, L, W);
  let m = Infinity;
  const pts = [];
  for (let i = 0; i < 4; i++) {
    const a = C[i], b = C[(i + 1) % 4], len = Math.hypot(b[0] - a[0], b[1] - a[1]), n = Math.max(1, Math.ceil(len / step));
    for (let k = 0; k < n; k++) pts.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  }
  for (const [px, pz] of pts) {
    for (const c of ob.caps) m = Math.min(m, ptSeg(px, pz, c.ax, c.az, c.bx, c.bz) - c.r);
    for (const c of ob.circles) m = Math.min(m, Math.hypot(px - c.x, pz - c.z) - c.r);
    for (const r of ob.rects) m = Math.min(m, ptRect(px, pz, r.x, r.z, .08, .075));
    const b = ob.bounds; m = Math.min(m, px - b.minX, b.maxX - px, pz - b.minZ, b.maxZ - pz);
  }
  // 障害物が車体の中に入り込んでいないか(点の間をすり抜ける細いもの)
  for (const c of ob.caps) if (insideCar(c.ax, c.az, x, z, yaw, L, W) || insideCar(c.bx, c.bz, x, z, yaw, L, W)) m = Math.min(m, 0);
  for (const c of ob.circles) if (insideCar(c.x, c.z, x, z, yaw, L, W)) m = Math.min(m, 0);
  for (const r of ob.rects) for (const [px, pz] of r.poly) if (insideCar(px, pz, x, z, yaw, L, W)) m = Math.min(m, 0);
  return m;
}

// ---- R への射影 ----
const REF = JSON.parse(fs.readFileSync('/home/jetson/jetformer/experiments/20261013_sim_rl_modes/rl/runtime/reference_loop.json', 'utf8'));
export const LOOP_L = REF.length_m;
const N = REF.x.length;
const SEG = [];
{
  let s = 0;
  for (let i = 0; i < N; i++) {
    const j = (i + 1) % N, ax = REF.x[i], az = REF.z[i], bx = REF.x[j], bz = REF.z[j], len = Math.hypot(bx - ax, bz - az);
    SEG.push({ax, az, bx, bz, len, s0: s, tx: (bx - ax) / len, tz: (bz - az) / len});
    s += len;
  }
}
export const LOOP_L_OWN = SEG.reduce((a, q) => a + q.len, 0);
function projSeg(i, x, z) {
  const q = SEG[i], t = clamp((x - q.ax) * q.tx + (z - q.az) * q.tz, 0, q.len), px = q.ax + q.tx * t, pz = q.az + q.tz * t;
  // 左 = 進む向き (tx, tz) に対して (tz, −tx)。sim-core は前 = (cos ψ, −sin ψ)・舵が正(左)で ψ が増えるので、左 = (−sin ψ, −cos ψ)
  const ey = (x - px) * q.tz + (z - pz) * (-q.tx);
  return {d: Math.hypot(x - px, z - pz), s: q.s0 + t, ey, i};
}
export function createProjector() {
  let idx = -1, sPrev = 0;
  return {
    // 最初: 向きの合う(進む向きと車の向きの内積 > 0.3)いちばん近い区間
    reset(x, z, yaw) {
      const hx = Math.cos(yaw), hz = -Math.sin(yaw);
      let best = null;
      for (let i = 0; i < N; i++) {
        if (SEG[i].tx * hx + SEG[i].tz * hz < .3) continue;
        const p = projSeg(i, x, z);
        if (!best || p.d < best.d) best = p;
      }
      idx = best.i; sPrev = best.s;
      return {s: best.s, ey: best.ey, d: best.d};
    },
    // 次: 前の区間の −30〜+60 の窓で一番近いもの。s は前の値に一番近い周の値に
    advance(x, z) {
      let best = null;
      for (let k = -30; k <= 60; k++) {
        const i = ((idx + k) % N + N) % N, p = projSeg(i, x, z);
        if (!best || p.d < best.d) best = p;
      }
      idx = best.i;
      const base = sPrev - (((sPrev % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN);
      let s = base + best.s;
      if (s - sPrev > LOOP_L_OWN / 2) s -= LOOP_L_OWN; else if (sPrev - s > LOOP_L_OWN / 2) s += LOOP_L_OWN;
      sPrev = s;
      return {s, ey: best.ey, d: best.d};
    },
  };
}
// 位置でそろえたマス(protocol variation_metrics VM2〜VM4 の文言のとおり): 絶対の s mod L の 0.05 m のマスに、開始から 3 m 以降の最初の通過の値
export const DS = .05, NB = Math.ceil(LOOP_L_OWN / DS);
export function binFirstPass(frames, key, skip = 3) {
  const out = new Array(NB).fill(null), s0 = frames[0].s;
  for (let k = 0; k + 1 < frames.length; k++) {
    const a = frames[k].s, b = frames[k + 1].s;
    if (a - s0 < skip || !(b > a)) continue;
    const val = frames[k][key];
    // [a, b) にある各マスの左端(絶対の s で n·L + j·DS)
    for (let n = Math.floor(a / LOOP_L_OWN); n <= Math.floor(b / LOOP_L_OWN); n++) {
      const j0 = Math.max(0, Math.ceil((a - n * LOOP_L_OWN) / DS - 1e-12));
      for (let j = j0; j < NB; j++) {
        const e = n * LOOP_L_OWN + j * DS;
        if (e >= b) break;
        if (out[j] === null) out[j] = val;
      }
    }
  }
  return out;
}
