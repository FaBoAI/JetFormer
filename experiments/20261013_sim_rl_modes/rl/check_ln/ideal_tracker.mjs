// 経路いろいろ(line)の確認 r1: 「目標の線の上を正確に走る車」の追従の誤差(2026-10-14、確認の段)。
// 目標の線の点 P(s) = R(s) + e*(s)·n(s) を 1 cm おきに順に置き、自前の R への射影(own_extra.mjs、ランタイムの射影と同じ値になることは run_ln で確かめた)で
// (s', e_y') に戻し、定義どおり |e_y' − e*(s')| を測る。曲がりの内側で e*·κ_R ≥ 1(R の曲率の中心より先)だと、射影が前へ飛ぶので 0 にならない。
// 方策に関係なく残る誤差 = L3 の下限の目安。
//   node ideal_tracker.mjs REPLAY.jsonl > 結果.json
import fs from 'node:fs';
import {createProjector, LOOP_L_OWN} from './own_extra.mjs';
import {eStarOwn, rAt, REF_LOOP} from './own_line.mjs';
// 窓つきの射影(ランタイムの projectWindow と同じ考え方: [a, b] の中で一番近い点)。窓は「目標の点を作った s」の −0.5〜+1.5 m に固定する
// (順に追うと、尖った所で目標の点が R に対して後ろへ戻るため射影が迷子になる。車が目標の上にいるなら射影はこう答える、の目安)
const NP = REF_LOOP.x.length, SG = [];
{ let s = 0; for (let i = 0; i < NP; i++) { const j = (i + 1) % NP, ax = REF_LOOP.x[i], az = REF_LOOP.z[i], bx = REF_LOOP.x[j], bz = REF_LOOP.z[j], len = Math.hypot(bx - ax, bz - az); SG.push({ax, az, s0: s, len, tx: (bx - ax) / len, tz: (bz - az) / len}); s += len; } }
function projWin(x, z, a, b) {
  let best = null;
  const lap0 = Math.floor(a / LOOP_L_OWN) - 1;
  for (let lap = lap0; lap <= lap0 + 2; lap++) for (const q of SG) {
    const st = lap * LOOP_L_OWN + q.s0;
    if (st + q.len < a || st > b) continue;
    const tLo = Math.max(0, a - st), tHi = Math.min(q.len, b - st);
    const t = Math.max(tLo, Math.min(tHi, (x - q.ax) * q.tx + (z - q.az) * q.tz)), px = q.ax + q.tx * t, pz = q.az + q.tz * t, d = Math.hypot(x - px, z - pz);
    if (!best || d < best.d) best = {d, s: st + t, ey: (x - px) * q.tz + (z - pz) * (-q.tx)};
  }
  return best;
}

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const {createLineProfile} = await import(`${EXP}/web/rl/line_profiles.mjs`);
const quant = (arr, p) => { const b = Float64Array.from(arr).sort(); if (!b.length) return null; const i = p * (b.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
const rows = fs.readFileSync(process.argv[2], 'utf8').trim().split('\n').map(l => JSON.parse(l)).filter(r => r.test === 'LN-P60' || r.test === 'LN-FAM6' || r.test === 'FRESH');
const out = {runs: [], pooled: {}};
const poolS = [], poolT = [], poolActual = [];
for (const r of rows) {
  if (!r.valid_episode && r.test !== 'LN-FAM6') continue;
  const p = r.params.line, sStart = r.fr[0][2];   // 開始の s(mod L)。自前の射影
  // random は節が種の乱数なので、ランタイムの作り手(原点 = 開始の s)を使う。ほかは自前の式
  const prof = p.family === 'random' ? createLineProfile(p, sStart) : null;
  const eStar = s => prof ? prof.eStar(s) : eStarOwn(p, s);
  const pj = createProjector();
  const sEnd = sStart + r.s_travel;
  const errs = [], wts = [];
  // 実際の回の速さ(s mod L の 5 cm のマス、開始から 3 m 以降の最初の通過)で時間の重み
  const vb = r.prof.v;
  let first = true;
  for (let s = sStart; s <= sEnd; s += .01) {
    const q = rAt(s), e = eStar(s), x = q.x + e * q.nx, z = q.z + e * q.nz;
    const pr = projWin(x, z, s - .5, s + 1.5);
    if (s - sStart < 3) continue;
    // 射影の s は周をまたいだ値。e* は random なら同じ原点の値、ほかは mod L
    const err = Math.abs(pr.ey - eStar(pr.s));
    const v = vb[Math.floor((((s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN) / .05)] ?? null;
    errs.push(err); wts.push(v && v > .3 ? .01 / v : .01 / 3);
  }
  // 時間の重みつきの分位(重みの累積)
  const idx = errs.map((_, i) => i).sort((a, b) => errs[a] - errs[b]);
  const tot = wts.reduce((a, b) => a + b, 0);
  const wq = pq => { let c = 0; for (const i of idx) { c += wts[i]; if (c >= pq * tot) return errs[i]; } return errs[idx.at(-1)]; };
  const actual = r.fr.filter(f => f[1] >= 3).map(f => Math.abs(f[3] - (f[4] ?? f[5])));
  out.runs.push({id: r.id, test: r.test, family: r.family, line: p, c_v: r.params.c_v, ideal_median: quant(errs, .5), ideal_p95: quant(errs, .95), ideal_max: Math.max(...errs),
    ideal_frac_gt12: errs.filter(e => e > .12).length / errs.length, ideal_time_p95: wq(.95), actual_p95: quant(actual, .95), actual_median: quant(actual, .5)});
  if (r.test !== 'LN-FAM6') {
    for (let i = 0; i < errs.length; i += 4) poolS.push(errs[i]);
    // 時間の重み: 1/v に比例する回数で入れる(v を 0.5 m/s 刻みに丸め、10 cm 間隔 × 回数)
    for (let i = 0; i < errs.length; i += 10) { const n = Math.max(1, Math.round(wts[i] / (.01 / 4.5))); for (let k = 0; k < n; k++) poolT.push(errs[i]); }
    for (const a of actual) poolActual.push(a);
  }
}
out.pooled = {n_runs: out.runs.filter(r => r.test !== 'LN-FAM6').length, ideal_s_uniform: {median: quant(poolS, .5), p95: quant(poolS, .95), frac_gt12: poolS.filter(e => e > .12).length / poolS.length},
  ideal_time_weighted: {median: quant(poolT, .5), p95: quant(poolT, .95), frac_gt12: poolT.filter(e => e > .12).length / poolT.length},
  actual: {median: quant(poolActual, .5), p95: quant(poolActual, .95), frac_gt12: poolActual.filter(e => e > .12).length / poolActual.length, n: poolActual.length}};
console.log(JSON.stringify(out));
