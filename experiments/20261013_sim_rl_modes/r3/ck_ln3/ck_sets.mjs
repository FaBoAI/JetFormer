// 確認 line r3 の組(2026-10-14、確認の段)。
//  1. 決めた試験の組(LN-P60-R3・LN-HEAD60-R3・LN-EXT60-R3・LN-FAM6)を protocol_line_r3.json の文言から自分で作り直し、学習の段の行と照らす
//     (Halton は自前、mulberry32 は文言の式、開始のすき間・|Δψ| は自前の幾何と射影)。走らせるのは学習の段の行の条件(照らして一致したもの)
//  2. どこにも使っていない新しい組(種 4,710,000〜、乱数の種 0x6c6e33xx)を作る。開始の分布の全体(端・組み合わせ)を狙う
//   node ck_sets.mjs → check_line_r3/sets.json と testset_check.json
import fs from 'node:fs';
import {startInfo} from './ck_run.mjs';

const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r3';
const EV = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3/eval_test_r3.json', 'utf8'));
const PROT = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/protocol_line_r3.json', 'utf8'));
function halton(i, b) { let f = 1, r = 0, n = i; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; }
// mulberry32(文言の名前どおりの、よく知られた式)
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const PLAN15 = [['const_left', 2], ['const_right', 2], ['const_center', 1], ['inside', 2], ['outside', 2], ['late_apex', 2], ['early_apex', 1], ['random', 3]].flatMap(([k, w]) => Array(w).fill(k));
const famOk = (fam, L) => fam === 'const_left' ? L.family === 'const' && L.c0 >= .6 && L.c0 <= .9 : fam === 'const_right' ? L.family === 'const' && L.c0 >= -.9 && L.c0 <= -.6
  : fam === 'const_center' ? L.family === 'const' && Math.abs(L.c0) <= .2 : fam === 'random' ? L.family === 'random' && Number.isInteger(L.seed) : L.family === fam && L.a >= .5 && L.a <= .9;

// ---- 1. 決めた試験の組の作り直し ----
const tc = {};
const rows = EV.tests;
{
  // LN-P60-R3: Halton 29,001〜、開始のすき間 ≥ 0.12 m、種 3,130,000 + i、形 family_plan_15 の順
  const own = [], skipped = [];
  for (let i = 29001; own.length < 60; i++) {
    const g = Math.min(32, Math.floor(halton(i, 2) * 33)), off = -.2 + .4 * halton(i, 3), yaw = -15 + 30 * halton(i, 5), I = startInfo(g, off, yaw);
    if (!(I.clearance >= .12)) { skipped.push({i, c: +I.clearance.toFixed(4)}); continue; }
    own.push({i, g, off, yaw, I});
  }
  const R = rows['LN-P60-R3'], bad = [];
  let maxd = 0;
  R.forEach((r, k) => {
    const o = own[k], d = Math.max(Math.abs(o.off - r.start.left_offset_m), Math.abs(o.yaw - r.start.left_yaw_deg));
    maxd = Math.max(maxd, d);
    if (o.g !== r.start.gate || o.i !== r.start.halton_index || r.start.seed !== 3130000 + k || r.family !== PLAN15[k % 15] || !famOk(r.family, r.line) || !(r.c_v >= .9 && r.c_v <= 1)) bad.push(r.id);
  });
  tc['LN-P60-R3'] = {n: R.length, max_abs_start_diff: maxd, mismatched: bad, skipped, last_index: own.at(-1).i, min_start_clearance: Math.min(...own.map(o => o.I.clearance)),
    large_heading: own.map((o, k) => ({id: `P${k}`, g: o.g, dpsi: +o.I.dpsi.toFixed(3)})).filter(x => Math.abs(x.dpsi) >= 1)};
}
function drawStarts(seed, n, pred) {
  const rnd = mulberry32(seed), out = [];
  let drawn = 0, clrOk = 0;
  while (out.length < n) {
    const g = Math.floor(rnd() * 33), off = -.2 + .4 * rnd(), yaw = -15 + 30 * rnd();
    drawn++;
    const I = startInfo(g, off, yaw);
    if (!(I.clearance >= .12)) continue;
    clrOk++;
    if (!pred(I)) continue;
    out.push({g, off, yaw, I});
  }
  return {out, drawn, clrOk};
}
for (const [name, seed, base, pred] of [['LN-HEAD60-R3', 0x4ead0003, 3131000, I => Math.abs(I.dpsi) >= 1], ['LN-EXT60-R3', 0xe7e70003, 3132000, () => true]]) {
  const {out, drawn, clrOk} = drawStarts(seed, 60, pred), R = rows[name], bad = [];
  let maxd = 0;
  R.forEach((r, k) => {
    const o = out[k], d = Math.max(Math.abs(o.off - r.start.left_offset_m), Math.abs(o.yaw - r.start.left_yaw_deg));
    maxd = Math.max(maxd, d);
    if (o.g !== r.start.gate || r.start.seed !== base + k || !(r.c_v >= .9 && r.c_v <= 1)) bad.push(r.id);
    if (name === 'LN-HEAD60-R3' && (r.family !== PLAN15[k % 15] || !famOk(r.family, r.line))) bad.push(r.id + ':shape');
    if (name === 'LN-EXT60-R3') {
      // i mod 10: const c0 ±0.9・inside/outside/late_apex/early_apex a 0.9 の 6 つ + random(確認 r2 の EXT と同じ割り当て)、c_v は奇数 1.0・偶数 0.9
      const L = r.line, isExt = L.family === 'random' || (L.family === 'const' && Math.abs(Math.abs(L.c0) - .9) < 1e-9) || (L.a != null && Math.abs(L.a - .9) < 1e-9);
      if (!isExt || r.c_v !== (k % 2 ? 1.0 : 0.9)) bad.push(r.id + ':ext');
    }
  });
  tc[name] = {n: R.length, drawn, clearance_ok: clrOk, max_abs_start_diff: maxd, mismatched: bad, min_start_clearance: Math.min(...out.map(o => o.I.clearance)),
    min_abs_dpsi: Math.min(...out.map(o => Math.abs(o.I.dpsi))), gates: out.reduce((a, o) => { a[o.g] = (a[o.g] ?? 0) + 1; return a; }, {}),
    ext_assignment: name === 'LN-EXT60-R3' ? R.slice(0, 10).map(r => r.line.family + (r.line.c0 != null ? r.line.c0 : r.line.a != null ? r.line.a : '')) : undefined};
}
{
  const R = rows['LN-FAM6'];
  tc['LN-FAM6'] = R.map(r => ({id: r.id, start: r.start, line: r.line, c_v: r.c_v}));
}
// 学習・検証・報告の組と、決めた試験の組が重なっていないか(開始の 3 つの値が同じもの)
{
  const key = s => `${s.gate}|${(+s.left_offset_m).toFixed(6)}|${(+s.left_yaw_deg).toFixed(4)}`;
  const test = new Set(Object.entries(rows).filter(([k]) => k !== 'LN-FAM6').flatMap(([, v]) => v.map(r => key(r.start))));
  const fresh = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/line_r3/eval_fresh_r3.json', 'utf8'));
  const fr = Object.values(fresh.tests).flat();
  tc.overlap_test_vs_fresh240 = fr.filter(r => test.has(key(r.start))).length;
}
fs.writeFileSync(`${OUT}/testset_check.json`, JSON.stringify(tc, null, 1));

// ---- 走らせる組 ----
const jobs = [];
const add = (set, id, start, line, c_v, family, extra = {}) => jobs.push({set, id, start, params: {line, c_v}, family, ...extra});
for (const name of ['LN-P60-R3', 'LN-HEAD60-R3', 'LN-EXT60-R3', 'LN-FAM6'])
  for (const r of rows[name]) add('REPLAY', r.id, {gate: r.start.gate, left_offset_m: r.start.left_offset_m, left_yaw_deg: r.start.left_yaw_deg, seed: r.start.seed}, r.line, r.c_v, r.family, {test: name});

// 新しい組の形(family_plan_15 の順。強さは modes.json line_families の範囲の一様乱数、c_v U(0.9, 1.0))
function shapeFor(kind, rnd) {
  const u = rnd();
  if (kind === 'const_left') return {family: 'const', c0: .6 + .3 * u};
  if (kind === 'const_right') return {family: 'const', c0: -.9 + .3 * u};
  if (kind === 'const_center') return {family: 'const', c0: -.2 + .4 * u};
  if (kind === 'random') return {family: 'random', seed: Math.floor(u * 4294967296) >>> 0};
  return {family: kind, a: .5 + .4 * u};
}
// 範囲の端の形(10 通りをくり返す): c0 ±0.9・0、a 0.9 の 4 つ、random 3
const EXT10 = [['const', .9], ['const', -.9], ['inside', .9], ['outside', .9], ['late_apex', .9], ['early_apex', .9], ['random'], ['const', 0], ['random'], ['random']];
function extShape(i, rnd) {
  const [f, v] = EXT10[i % 10];
  if (f === 'const') return {line: {family: 'const', c0: v}, family: v > 0 ? 'const_left' : v < 0 ? 'const_right' : 'const_center'};
  if (f === 'random') return {line: {family: 'random', seed: Math.floor(rnd() * 4294967296) >>> 0}, family: 'random'};
  return {line: {family: f, a: v}, family: f};
}
const fresh = (set, n, sSeed, shSeed, seedBase, pred) => {
  const {out, drawn, clrOk} = drawStarts(sSeed, n, pred), rnd = mulberry32(shSeed);
  out.forEach((o, i) => { const kind = PLAN15[i % 15], line = shapeFor(kind, rnd), cv = +(0.9 + 0.1 * rnd()).toFixed(4);
    add(set, `${set}-${i}`, {gate: o.g, left_offset_m: o.off, left_yaw_deg: o.yaw, seed: seedBase + i}, line, cv, kind, {dpsi: o.I.dpsi, c_start: o.I.clearance}); });
  return {n, drawn, clearance_ok: clrOk};
};
const meta = {};
meta.U300 = fresh('U300', 300, 0x6c6e3301, 0x6c6e3302, 4710000, () => true);
meta.HEAD120 = fresh('HEAD120', 120, 0x6c6e3303, 0x6c6e3304, 4711000, I => Math.abs(I.dpsi) >= 1);
meta.MID120 = fresh('MID120', 120, 0x6c6e3305, 0x6c6e3306, 4712000, I => Math.abs(I.dpsi) >= .6 && Math.abs(I.dpsi) < 1);
meta.NEAR60 = fresh('NEAR60', 60, 0x6c6e3309, 0x6c6e330a, 4715000, I => I.clearance < .15);
// 開始の分布の角と辺: 33 ゲート × (左右 ±0.2 × 向き ±15°)と(左右 0 × 向き ±15°)・(左右 ±0.2 × 向き 0)
{
  const rnd = mulberry32(0x6c6e3307);
  let i = 0, skipped = 0;
  for (let g = 0; g < 33; g++) for (const [off, yaw] of [[-.2, -15], [-.2, 15], [.2, -15], [.2, 15], [0, -15], [0, 15], [-.2, 0], [.2, 0]]) {
    const I = startInfo(g, off, yaw);
    if (!(I.clearance >= .12)) { skipped++; continue; }
    const sh = extShape(i, rnd), cv = Math.floor(i / 10) % 2 ? 1.0 : 0.9;
    add('CORNER', `C${i}`, {gate: g, left_offset_m: off, left_yaw_deg: yaw, seed: 4713000 + i}, sh.line, cv, sh.family, {dpsi: I.dpsi, c_start: I.clearance});
    i++;
  }
  meta.CORNER = {n: i, skipped_low_clearance: skipped};
}
// 危ない所の密な格子: ゲート 6・12・13・16・17・31(大きく横を向く開始と、確認 r2 の E41・試験の P37 の所)× 左右 5 × 向き 5、形は範囲の端
{
  const rnd = mulberry32(0x6c6e330b);
  let i = 0, skipped = 0;
  for (const g of [6, 12, 13, 16, 17, 31]) for (const off of [-.2, -.1, 0, .1, .2]) for (const yaw of [-15, -7.5, 0, 7.5, 15]) {
    const I = startInfo(g, off, yaw);
    if (!(I.clearance >= .12)) { skipped++; continue; }
    const sh = extShape(i + 3, rnd), cv = Math.floor(i / 10) % 2 ? 0.9 : 1.0;
    add('DENSE', `D${i}`, {gate: g, left_offset_m: off, left_yaw_deg: yaw, seed: 4717000 + i}, sh.line, cv, sh.family, {dpsi: I.dpsi, c_start: I.clearance});
    i++;
  }
  meta.DENSE = {n: i, skipped_low_clearance: skipped};
}
// 大きく横を向いた開始 × 範囲の端の形 × c_v の両端
{
  const {out, drawn, clrOk} = drawStarts(0x6c6e3308, 60, I => Math.abs(I.dpsi) >= 1), rnd = mulberry32(0x6c6e330c);
  out.forEach((o, i) => { const sh = extShape(i, rnd), cv = Math.floor(i / 10) % 2 ? 1.0 : 0.9;
    add('EXTHEAD', `XH${i}`, {gate: o.g, left_offset_m: o.off, left_yaw_deg: o.yaw, seed: 4714000 + i}, sh.line, cv, sh.family, {dpsi: o.I.dpsi, c_start: o.I.clearance}); });
  meta.EXTHEAD = {n: 60, drawn, clearance_ok: clrOk};
}
// LN-FAM6 の頑丈さ: 標準の開始の小さなずれ 6 通り + c_v 0.9
{
  const FAM6 = [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}], ['inside', {family: 'inside', a: .8}],
    ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}]];
  let i = 0;
  for (const [vk, off, yaw, cv] of [['o+3', .03, 0, 1], ['o-3', -.03, 0, 1], ['y+3', 0, 3, 1], ['y-3', 0, -3, 1], ['o+3y+3', .03, 3, 1], ['o-3y-3', -.03, -3, 1], ['cv0.9', 0, 0, .9]])
    for (const [nm, line] of FAM6) { add('FAM6V', `${vk}:${nm}`, {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 4716000 + i}, line, cv, nm, {variant: vk}); i++; }
  meta.FAM6V = {n: i};
}
fs.writeFileSync(`${OUT}/sets.json`, JSON.stringify({created_at: new Date().toISOString(), meta, jobs}, null, 0));
console.log(JSON.stringify({testset_check: Object.fromEntries(Object.entries(tc).map(([k, v]) => [k, Array.isArray(v) ? v.length : typeof v === 'object' ? {...v, skipped: v.skipped?.length, large_heading: v.large_heading?.length} : v])), meta, jobs: jobs.length}, null, 0));
