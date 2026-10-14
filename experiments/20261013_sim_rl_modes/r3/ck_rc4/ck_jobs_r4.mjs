// 離脱復活 r4 の確認: 走らせる回の一覧(2026-10-15)。
//   1. 決めた試験の組の自前の作り直し(RP・RK・RKR・RU・REC60・STD1)。開始は自前の Halton(基数 2・3・5)・自前の mulberry32・自前の開始のすき間(ck_geom.mjs)で作り、
//      学習の段の eval_test.json の開始と 1 つずつ照合する(食い違いは数えて出す)
//   2. どこでも使っていない種・開始(種 9,610,000〜9,629,999、乱数の種 0x7c0e4d01〜0x7c0e4d05)で、範囲の全部・角・組み合わせ:
//      FP 新しい一様な開始(乱し v2・強さ auto)/ FU 乱しなし / FC・FUC・FCS 範囲の角(乱しあり・なし・強固定)/ GRID 格子(乱しなし、止まりの探し)/
//      FK 新しい時刻の格子のキック / FKR 新しい開始・一様な時刻のキック / FKB STD のスタートのゲートの区間の全フレームのキック /
//      FKBR 新しい開始でスタートのゲートの区間を位置で狙うキック / FSTRONG・FWEAK・FMED 強さ固定 / FMX 強さの中身を上限に(別の一覧) /
//      FF 最速の方策 + 同じ乱し / FV1 元の乱し v1 / FR3 r3 の方策 + 同じ r4 の仕組み / FNS シールドなし / NOM 乱しなしの線 / FUF 最速のモードそのもの
//   node ck_jobs_r4.mjs <out.json> <out_fmx.json>
import fs from 'node:fs';
import {startClearance, runOne, LOOP} from './ck_run.mjs';

function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function halton(i, b) { let f = 1, r = 0, n = i; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; }
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
const J = [], JX = [], add = (set, id, spec) => J.push({set, id, spec});
const EV = JSON.parse(fs.readFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/rc4/eval_test.json', 'utf8'));
const check = {};

// ---- 1. 決めた試験の組の作り直し ----
function haltonFromOwn(from, count, minC) {
  const out = []; let i = from;
  for (; out.length < count; i++) {
    const st = {gate: Math.min(32, Math.floor(halton(i, 2) * 33)), left_offset_m: -.2 + .4 * halton(i, 3), left_yaw_deg: -15 + 30 * halton(i, 5), halton_index: i};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= minC)) continue;
    out.push({...st, start_clearance: c});
  }
  return {starts: out, next: i};
}
const P = haltonFromOwn(80001, 100, 1e-9), U = haltonFromOwn(P.next, 60, .03);
const same = (a, b) => a.gate === b.gate && Math.abs(a.left_offset_m - b.left_offset_m) < 1e-12 && Math.abs(a.left_yaw_deg - b.left_yaw_deg) < 1e-9;
check.RP_start_mismatch = P.starts.filter((st, i) => !(same(st, EV.tests['RC-P100-R4'][i].start) && st.halton_index === EV.tests['RC-P100-R4'][i].start.halton_index)).length;
check.RU_start_mismatch = U.starts.filter((st, i) => !(same(st, EV.tests['RC-U60-R4'][i].start) && st.halton_index === EV.tests['RC-U60-R4'][i].start.halton_index)).length;
check.RU_min_start_clearance_own = Math.min(...U.starts.map(s => s.start_clearance));
P.starts.forEach((st, i) => add('RP', `P${i}`, {policy: 'r4', start: {...st, seed: 8120000 + i}, perturb: true}));
U.starts.forEach((st, i) => add('RU', `U${i}`, {policy: 'r4', start: {...st, seed: 8122000 + i}, perturb: false}));
// RC-KICK-R4: STD・時刻 1.05 + 0.5k と 1.30 + 0.5k(k = 0〜35)× 左右 × 向き/横、種 8,121,000 + i(決まりの文の並び)
let ki = 0;
const testKickFrames = new Set();
for (const t0 of [1.05, 1.30]) for (let k = 0; k <= 35; k++) for (const sign of [1, -1]) for (const kind of ['yaw', 'lateral']) {
  const t = t0 + .5 * k, frame = Math.round(t * 30);
  testKickFrames.add(frame);
  add('RK', `${kind}${sign > 0 ? '+' : '-'}${t.toFixed(2)}`, {policy: 'r4', start: {...STD, seed: 8121000 + ki++}, perturb: false, kick: {frame, [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
}
for (let k = 0; k <= 35; k++) testKickFrames.add(Math.round((1.4 + .5 * k) * 30));   // 検証の組の時刻も避ける
// RC-KICKR-R4: mulberry32(0x6b1c4004)・記録の開始の分布・開始のすき間 ≥ 0.03 m(自前)、続きで時刻 U(1.5, 19.0)・向き(u < 0.5 なら左)、偶 = 向き / 奇 = 横
{
  const u = mulberry32(0x6b1c4004), out = [];
  for (let guard = 0; out.length < 100; guard++) {
    const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: -.2 + .4 * u(), left_yaw_deg: -15 + 30 * u()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= .03)) continue;
    const t = 1.5 + 17.5 * u(), sign = u() < .5 ? 1 : -1, i = out.length, kind = i % 2 === 0 ? 'yaw' : 'lateral';
    out.push({st: {...st, start_clearance: c, seed: 8123000 + i}, kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
  }
  const T = EV.tests['RC-KICKR-R4'];
  check.RKR_mismatch = out.filter((q, i) => !(same(q.st, T[i].start) && q.kick.frame === T[i].kick.frame && (q.kick.yaw ?? q.kick.lateral) === (T[i].kick.yaw ?? T[i].kick.lateral))).length;
  out.forEach((q, i) => add('RKR', `K${i}`, {policy: 'r4', start: q.st, perturb: false, kick: q.kick, kind: q.kind}));
}
const plan = JSON.parse(fs.readFileSync('/home/jetson/sim/extracted/course-simulator/qa/rc-collection-plan-60.json', 'utf8'));
plan.forEach(c => add('REC60', c.name, {policy: 'r4', start: {gate: c['start-gate'] ?? 0, left_offset_m: c['spawn-offset'] ?? 0, left_yaw_deg: c['spawn-yaw'] ?? 0, seed: c.seed}, perturb: false}));
add('STD1', 'single', {policy: 'r4', start: {...STD, seed: 1}, perturb: false, episodeMode: 'single'});
add('STD1F', 'single', {policy: 'fastest', start: {...STD, seed: 1}, perturb: false, episodeMode: 'single', mode: 'fastest'});

// ---- 2. 新しい種・開始 ----
function freshStarts(rngSeed, n, minC) {
  const u = mulberry32(rngSeed), out = []; let skipped = 0;
  while (out.length < n) {
    const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: -.2 + .4 * u(), left_yaw_deg: -15 + 30 * u()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= minC)) { skipped++; continue; }
    out.push({...st, start_clearance: c});
  }
  return {out, skipped};
}
const fp = freshStarts(0x7c0e4d01, 320, 1e-9);
const FP = fp.out.slice(0, 200), FU = fp.out.slice(200, 320);
check.FP_skipped = fp.skipped;
FP.forEach((st, i) => add('FP', `P${i}`, {policy: 'r4', start: {...st, seed: 9610000 + i}, perturb: true}));
FU.forEach((st, i) => add('FU', `U${i}`, {policy: 'r4', start: {...st, seed: 9611000 + i}, perturb: false}));
// 乱しなしの線(protocol definitions.nominal_line: 種 990000 + i、ゲート 0・±3 cm・±3°)
const un = mulberry32(0x7c0e4d04);
for (let i = 0; i < 8; i++) {
  const o = -.03 + .06 * un(), y = -3 + 6 * un();
  add('NOM-r4', `N${i}`, {policy: 'r4', start: {gate: 0, left_offset_m: o, left_yaw_deg: y, seed: 990000 + i}, perturb: false});
  add('NOM-fastest', `N${i}`, {policy: 'fastest', start: {gate: 0, left_offset_m: o, left_yaw_deg: y, seed: 990000 + i}, perturb: false});
  add('NOM-r3', `N${i}`, {policy: 'r3', start: {gate: 0, left_offset_m: o, left_yaw_deg: y, seed: 990000 + i}, perturb: false});
  add('NOMF', `N${i}`, {policy: 'fastest', mode: 'fastest', start: {gate: 0, left_offset_m: o, left_yaw_deg: y, seed: 990000 + i}, perturb: false});
}
// キック: 新しい時刻の格子(フレーム 36 + 15k = 1.2 + 0.5k 秒、43 + 15k ≈ 1.433 + 0.5k 秒。試験 32・39、検証 42、protocol 30 と重ならない)
let fki = 0;
for (const f0 of [36, 43]) for (let k = 0; k <= 35; k++) for (const sign of [1, -1]) for (const kind of ['yaw', 'lateral']) {
  const frame = f0 + 15 * k;
  add('FK', `${kind}${sign > 0 ? '+' : '-'}f${frame}`, {policy: 'r4', start: {...STD, seed: 9616000 + fki++}, perturb: false, kick: {frame, [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
}
// 新しい開始・一様な時刻のキック(RC-KICKR-R4 と同じ作り方、別の乱数の種)
{
  const u = mulberry32(0x7c0e4d02); let n = 0;
  while (n < 200) {
    const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: -.2 + .4 * u(), left_yaw_deg: -15 + 30 * u()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= .03)) continue;
    const t = 1.5 + 17.5 * u(), sign = u() < .5 ? 1 : -1, kind = n % 2 === 0 ? 'yaw' : 'lateral';
    add('FKR', `K${n}`, {policy: 'r4', start: {...st, start_clearance: c, seed: 9617000 + n}, perturb: false, kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
    n++;
  }
}
// STD のスタートのゲートの区間(s mod L 0.30〜3.70 m)の全部のフレーム(試験・検証のキックのフレームを除く、フレーム 31 以降)× 左右 × 向き/横
{
  const nom = await runOne({policy: 'r4', start: {...STD, seed: 9620000}, perturb: false});
  const bandFrames = nom.frames_data.filter(q => { const m = ((q.s % LOOP) + LOOP) % LOOP; return q.f >= 31 && m >= .30 && m <= 3.70; }).map(q => q.f).filter(f => !testKickFrames.has(f));
  check.FKB_band_frames = bandFrames.length;
  check.FKB_band_frame_ranges = bandFrames.reduce((acc, f) => { const l = acc.at(-1); if (l && f === l[1] + 1) l[1] = f; else acc.push([f, f]); return acc; }, []);
  let bi = 0;
  for (const frame of bandFrames) for (const sign of [-1, 1]) for (const kind of ['yaw', 'lateral'])
    add('FKB', `${kind}${sign > 0 ? '+' : '-'}f${frame}`, {policy: 'r4', start: {...STD, seed: 9621000 + bi++}, perturb: false, kick: {frame, [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
}
// 新しい開始(開始のすき間 ≥ 3 cm)で、1.5 秒以降に初めて s mod L が s_target(0.35〜3.55 m の一様)に来たフレームでキック。右 2/3、向き/横は交互
{
  const u = mulberry32(0x7c0e4d03); let n = 0;
  while (n < 200) {
    const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: -.2 + .4 * u(), left_yaw_deg: -15 + 30 * u()};
    const c = startClearance(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(c >= .03)) continue;
    const sT = .35 + 3.2 * u(), sign = u() < 2 / 3 ? -1 : 1, kind = n % 2 === 0 ? 'yaw' : 'lateral';
    add('FKBR', `B${n}`, {policy: 'r4', start: {...st, start_clearance: c, seed: 9622000 + n}, perturb: false, kick: {s_target: sT, min_frame: 45, [kind]: sign * (kind === 'yaw' ? .3 : .15)}, kind});
    n++;
  }
}
// 範囲の角(ゲート 33 × 左右 ±0.2 m × 向き ±15°)
let ci = 0; const cskip = [];
for (let g = 0; g <= 32; g++) for (const o of [-.2, .2]) for (const y of [-15, 15]) {
  const c = startClearance(g, o, y);
  if (!(c > 1e-9)) { cskip.push([g, o, y]); continue; }
  const id = `C${g}_${o > 0 ? 'L' : 'R'}${y > 0 ? '+' : '-'}`;
  add('FC', id, {policy: 'r4', start: {gate: g, left_offset_m: o, left_yaw_deg: y, start_clearance: c, seed: 9612000 + ci}, perturb: true});
  add('FUC', id, {policy: 'r4', start: {gate: g, left_offset_m: o, left_yaw_deg: y, start_clearance: c, seed: 9613000 + ci}, perturb: false});
  add('FCS', id, {policy: 'r4', start: {gate: g, left_offset_m: o, left_yaw_deg: y, start_clearance: c, seed: 9614000 + ci}, perturb: true, params: {perturb_class: '強'}});
  ci++;
}
check.corner_skipped = cskip;
// 強さ固定・比べ(FP の最初の 100 / 60 の開始)
FP.slice(0, 100).forEach((st, i) => {
  add('FSTRONG', `P${i}`, {policy: 'r4', start: {...st, seed: 9619000 + i}, perturb: true, params: {perturb_class: '強'}});
  add('FF', `P${i}`, {policy: 'fastest', start: {...st, seed: 9610000 + i}, perturb: true});
  add('FV1', `P${i}`, {policy: 'r4', start: {...st, seed: 9610000 + i}, perturb: true, params: {perturb_version: 'v1'}});
  add('FR3', `P${i}`, {policy: 'r3', start: {...st, seed: 9610000 + i}, perturb: true});
  JX.push({set: 'FMX', id: `P${i}`, spec: {policy: 'r4', start: {...st, seed: 9610000 + i}, perturb: true}});
});
FP.slice(0, 60).forEach((st, i) => {
  add('FWEAK', `P${i}`, {policy: 'r4', start: {...st, seed: 9619500 + i}, perturb: true, params: {perturb_class: '弱'}});
  add('FMED', `P${i}`, {policy: 'r4', start: {...st, seed: 9619700 + i}, perturb: true, params: {perturb_class: '中'}});
  add('FNS', `P${i}`, {policy: 'r4', start: {...st, seed: 9610000 + i}, perturb: true, overrides: {shield: false}});
});
FU.slice(0, 100).forEach((st, i) => add('FUF', `U${i}`, {policy: 'fastest', mode: 'fastest', start: {...st, seed: 9611000 + i}, perturb: false}));
// 止まりの格子(乱しなし): ゲート 33 × 左右 5 × 向き 5
let gi = 0;
for (let g = 0; g <= 32; g++) for (const o of [-.2, -.1, 0, .1, .2]) for (const y of [-15, -7.5, 0, 7.5, 15]) {
  const c = startClearance(g, o, y);
  add('GRID', `G${g}_${o}_${y}`, {policy: 'r4', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 9615000 + gi++, start_clearance: c}, perturb: false});
}
fs.writeFileSync(process.argv[2], JSON.stringify(J));
fs.writeFileSync(process.argv[3], JSON.stringify(JX));
const cnt = {}; for (const j of J) cnt[j.set] = (cnt[j.set] ?? 0) + 1;
console.log(JSON.stringify({counts: cnt, total: J.length, fmx: JX.length, check}));
