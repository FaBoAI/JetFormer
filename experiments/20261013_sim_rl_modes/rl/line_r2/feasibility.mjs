#!/usr/bin/env node
// 経路いろいろ r2 の下調べ(2026-10-14、学習の段 line r2): 目標の線 e*(s) を「その線の上を正確に走る理想の運転手」が走れるかを調べる。
// sim-core の約束(舵の上限 0.52 rad・舵の速さ 1.2 rad/s・ヨーレートの上限 μg/v・ブレーキ 1.2 m/s²・軸距 0.26 m)から、
// 線の上の各点の速さの上限を出し、前後に流して 1 周の時間と「走れない所」(舵が足りない・R の曲率の中心より先)を数える。
//   node rl/line_r2/feasibility.mjs [--profile v1|v2] [--json 出力]
import fs from 'node:fs';
import {loopPointAt, LOOP_LENGTH} from '../runtime/features.mjs';
import {createLineProfile} from '../runtime/line_profiles.mjs';

const L = LOOP_LENGTH, WB = .26, MAXSTEER = .52, STEER_RATE = 1.2, MU_G = .8 * 9.80665, BRAKE = 1.2, ACC = 4.0, VCAP = 4.5;
const pt = {x: 0, z: 0, tx: 0, tz: 0};
// R の点と左の法線(なめらかな接線: s ± 0.05 m の差)
export function rFrame(s) {
  const a = loopPointAt(s - .05, {x: 0, z: 0, tx: 0, tz: 0}), b = loopPointAt(s + .05, {x: 0, z: 0, tx: 0, tz: 0});
  loopPointAt(s, pt);
  let tx = b.x - a.x, tz = b.z - a.z; const n = Math.hypot(tx, tz); tx /= n; tz /= n;
  return {x: pt.x, z: pt.z, tx, tz, lx: tz, lz: -tx};
}
// R の局所の曲率(左曲がりが正。s ± h の 3 点の円)
export function kappaR(s, h = .1) {
  const p = [s - h, s, s + h].map(q => loopPointAt(q, {x: 0, z: 0, tx: 0, tz: 0}));
  return signedCurv(p[0].x, p[0].z, p[1].x, p[1].z, p[2].x, p[2].z);
}
// 3 点の円の符号つき曲率。座標は x 右・z 下向き(sim の約束: 前は (cos yaw, −sin yaw))なので、左曲がりが正になるよう符号を合わせる
export function signedCurv(x0, z0, x1, z1, x2, z2) {
  const ax = x1 - x0, az = z1 - z0, bx = x2 - x1, bz = z2 - z1, cx = x0 - x2, cz = z0 - z2;
  const cr = ax * bz - az * bx, d = Math.hypot(ax, az) * Math.hypot(bx, bz) * Math.hypot(cx, cz);
  return d > 0 ? -2 * cr / d : 0;
}
// e(s) の関数 → 1 周(または [sa, sb])の線の点・曲率・速さの上限・時間
export function analyse(eFn, {sa = 0, sb = L, ds = .05, cV = 1, standing = true} = {}) {
  const n = Math.round((sb - sa) / ds) + 1, X = new Float64Array(n), Z = new Float64Array(n), E = new Float64Array(n), KR = new Float64Array(n), S = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const s = sa + i * ds, f = rFrame(s), e = eFn(s);
    S[i] = s; E[i] = e; X[i] = f.x + e * f.lx; Z[i] = f.z + e * f.lz; KR[i] = kappaR(s);
  }
  const K = new Float64Array(n), D = new Float64Array(n), SEG = new Float64Array(n), VL = new Float64Array(n);
  const k2 = 3;
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - k2), b = Math.min(n - 1, i + k2);
    K[i] = a === i || b === i ? 0 : signedCurv(X[a], Z[a], X[i], Z[i], X[b], Z[b]);
    if (i > 0) SEG[i] = Math.hypot(X[i] - X[i - 1], Z[i] - Z[i - 1]);
  }
  for (let i = 0; i < n; i++) D[i] = Math.atan(WB * K[i]);
  let infeasibleSteer = 0, cusp = 0, maxEK = 0, maxK = 0, backward = 0;
  for (let i = 0; i < n; i++) {
    const ia = Math.max(0, i - 3), ib = Math.min(n - 1, i + 3), dd = ib > ia ? Math.abs(D[ib] - D[ia]) / Math.max(1e-6, (ib - ia) * ds) : 0;
    let v = VCAP * cV;
    if (Math.abs(K[i]) > 1e-6) v = Math.min(v, Math.sqrt(MU_G / Math.abs(K[i])));
    if (dd > 1e-6) v = Math.min(v, STEER_RATE / dd);
    if (Math.abs(D[i]) > MAXSTEER) { infeasibleSteer++; v = Math.min(v, .5); }
    VL[i] = v;
    const ek = E[i] * KR[i]; maxEK = Math.max(maxEK, ek); if (ek >= 1) cusp++;
    maxK = Math.max(maxK, Math.abs(K[i]));
    if (i > 0 && SEG[i] < .2 * ds) backward++;
  }
  // 前へ(加速)・後ろへ(ブレーキ)
  const V = Float64Array.from(VL);
  if (standing) V[0] = 0;
  for (let i = 1; i < n; i++) V[i] = Math.min(V[i], Math.sqrt(V[i - 1] ** 2 + 2 * ACC * SEG[i]));
  for (let i = n - 2; i >= 0; i--) V[i] = Math.min(V[i], Math.sqrt(V[i + 1] ** 2 + 2 * BRAKE * SEG[i + 1]));
  let t = 0;
  for (let i = 1; i < n; i++) t += SEG[i] / Math.max(.05, (V[i] + V[i - 1]) / 2);
  // 参考: 舵の速さの上限を入れない(横の上限・舵の上限・ブレーキだけ)ときの時間
  const V2 = Float64Array.from(VL, (v, i) => Math.min(VCAP * cV, Math.abs(K[i]) > 1e-6 ? Math.sqrt(MU_G / Math.abs(K[i])) : VCAP * cV));
  if (standing) V2[0] = 0;
  for (let i = 1; i < n; i++) V2[i] = Math.min(V2[i], Math.sqrt(V2[i - 1] ** 2 + 2 * ACC * SEG[i]));
  for (let i = n - 2; i >= 0; i--) V2[i] = Math.min(V2[i], Math.sqrt(V2[i + 1] ** 2 + 2 * BRAKE * SEG[i + 1]));
  let t2 = 0;
  for (let i = 1; i < n; i++) t2 += SEG[i] / Math.max(.05, (V2[i] + V2[i - 1]) / 2);
  return {n, time_s: t, time_no_steer_rate_s: t2, length_m: SEG.reduce((a, b) => a + b, 0), max_abs_kappa_path: maxK, max_e_kappaR: maxEK, cusp_points: cusp,
    steer_infeasible_points: infeasibleSteer, collapsed_points: backward, v_min: Math.min(...V.slice(5)), S, E, K, V, X, Z};
}

const FAM = [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}],
  ['inside', {family: 'inside', a: .8}], ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}],
  ['early_apex', {family: 'early_apex', a: .8}], ['c+0.9', {family: 'const', c0: .9}], ['c-0.9', {family: 'const', c0: -.9}],
  ['inside0.9', {family: 'inside', a: .9}], ['late0.9', {family: 'late_apex', a: .9}], ['early0.9', {family: 'early_apex', a: .9}]];

if (process.argv[1] && process.argv[1].endsWith('feasibility.mjs')) {
  const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
  const which = opt('--profile', 'v1');
  const rows = [];
  for (const [nm, p] of FAM) {
    const prof = which === 'v1' ? createLineProfile(p, 0) : createLineProfile({...p}, 0, {version: 'v2', e0: null});
    const r = analyse(s => prof.eStar(s), {sa: 0, sb: L, standing: false});
    const bad = [];
    for (let i = 0; i < r.n; i++) if (Math.abs(Math.atan(WB * r.K[i])) > MAXSTEER || r.E[i] * kappaR(r.S[i]) >= 1) bad.push(+r.S[i].toFixed(2));
    rows.push({name: nm, lap_ideal_s: +r.time_s.toFixed(2), lap_lat_only_s: +r.time_no_steer_rate_s.toFixed(2), max_kappa_path: +r.max_abs_kappa_path.toFixed(2), min_radius_m: +(1 / r.max_abs_kappa_path).toFixed(3),
      max_e_kappaR: +r.max_e_kappaR.toFixed(3), cusp_points: r.cusp_points, steer_infeasible_points: r.steer_infeasible_points, v_min: +r.v_min.toFixed(2),
      bad_s: bad.length ? [bad[0], bad.at(-1)] : null});
  }
  console.table(rows);
  if (opt('--json')) fs.writeFileSync(opt('--json'), JSON.stringify({profile: which, rows}, null, 1));
}
