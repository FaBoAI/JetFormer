#!/usr/bin/env node
// 経路いろいろ(line)第 4 回の下調べ・検証・試験・報告の組と判定(2026-10-14、line r4 の段)。protocol_line_r4.json のとおり。
// r3/line/eval_line_r3.mjs の写しを元に書き直したもの。違い:
//   - 走行中の仕組みは r3/line4/rt4(シールド v5 = r3/nw4/runtime_v5/shield.mjs の写し、mode.json の runtime で切り替える mode_runtime)。
//     シールドは globalThis.JF_R3 ではなく、教師に modeRuntime = {shield: true, shield_params} を渡してかける(昇格のあとのブラウザと同じ道)。
//   - 組: 角の組(LN-CORNER-R4: 33 ゲート × 開始の範囲の 4 つの角の箱)と、確認 r3 で崩れた箱(LN-BOX120-R4: ゲート 16・17 の左右 −0.2〜−0.16 m・向き +12〜+15°)を足した。
//   - 判定: L1〜L8 に、L9(止まり 0)・L10(P60・FAM6 で後ろへ動いたフレーム 0)を足した。L1・L2・L8・L9 は全部の組に。
//
//   node r3/line4/eval_line_r4.mjs --policy <policy.json> --set dev|validation|test|fresh|regression|fastest [--core fast|original] [--jobs N]
//        [--only 組,…] [--out 結果.json] [--keep-path] [--no-shield] [--shield-config <json>(dev だけ。固める前の下調べ)]
//   dev:        下調べ(固める前にシールドの値・学び直しの要否を決めるため。確認 r3 の崩れた 5 回・BOX160・NBC32 と、新しい種の DEV-H60・DEV-HEAD40・DEV-CORNER)
//   validation: LN-V60-R4 + LN-VHEAD40-R4 + LN-VCORNER-R4 + LN-VBOX40-R4 + LN-FAM6(高速版でよい。チェックポイントを選ぶためだけ)
//   test:       LN-P60-R4 + LN-HEAD60-R4 + LN-EXT60-R4 + LN-CORNER-R4 + LN-BOX120-R4 + LN-FAM6(元の sim-core.js。選んだ 1 つにだけ 1 回)
//   fresh:      LN-F300-R4(報告だけ。元の sim-core.js)
//   regression: 確認 r3 の崩れた回と組(REG-FAIL5・REG-NBC32・REG-BOX160。下調べで見たので報告だけ)
//   fastest:    最速の方策(bundle/fastest、シールドなし)を LN-P60-R4 の 60 の開始で(線の違いの報告用)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD} from './episode_ln4.mjs';
import {lineParamsFor} from './rt4/mode_runtime.mjs';
import {haltonStart, createRng, mix32} from './rt4/rng.mjs';
import {prepareGeometry, clearance, projectWindow, loopPointAt, LOOP_LENGTH} from './rt4/features.mjs';
import {activeWorldFor} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';
import {binProfile, spread, quantile, VM1, VM7, sceneFromEpisode, steerUs} from './variation_metrics.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function loadVerified(jsonPath, shaPath, name) {
  const raw = fs.readFileSync(jsonPath), sha = crypto.createHash('sha256').update(raw).digest('hex');
  const want = fs.readFileSync(shaPath, 'utf8').split('\n').map(l => l.trim().split(/\s+/)).find(a => a[1] === name)?.[0];
  if (sha !== want) throw new Error(`eval_line_r4: ${name} の sha256 ${sha} が固めた値 ${want} と違います`);
  return {...JSON.parse(raw), _sha256: sha};
}
const P4_PATH = `${REP}/r3/protocol_line_r4.json`, P4_SHA = `${REP}/r3/protocol_line_r4.sha256`;
export const P4 = fs.existsSync(P4_PATH) && fs.existsSync(P4_SHA) ? loadVerified(P4_PATH, P4_SHA, 'protocol_line_r4.json') : null;
export const P2 = loadVerified(`${REP}/protocol_line_r2.json`, `${REP}/protocol_line_r2.sha256`, 'protocol_line_r2.json');
const DIST = P2.start_distribution;
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};

// ---- 開始 ----
export function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export const PLAN15 = Object.entries({const_left: 2, const_right: 2, const_center: 1, inside: 2, outside: 2, late_apex: 2, early_apex: 1, random: 3}).flatMap(([k, w]) => Array(w).fill(k));
export function lineFor(kind, rnd) {
  const u = rnd();
  if (kind === 'const_left') return {family: 'const', c0: .6 + .3 * u};
  if (kind === 'const_right') return {family: 'const', c0: -.9 + .3 * u};
  if (kind === 'const_center') return {family: 'const', c0: -.2 + .4 * u};
  if (kind === 'random') return {family: 'random', seed: Math.floor(u * 4294967296) >>> 0};
  return {family: kind, a: .5 + .4 * u};
}
const baseGeom = prepareGeometry(BASE_WORLD);
const PA = {x: 0, z: 0, tx: 0, tz: 0}, PB = {x: 0, z: 0, tx: 0, tz: 0};
// 開始の姿勢のすき間(features.mjs clearance、車体 0.42 × 0.19)と、R に対する向きのずれ(R への射影の s ± 0.05 m の点から。protocol_line_r3 definitions_added)
export function startInfo(gate, off, yaw) {
  const {world} = activeWorldFor(BASE_WORLD, gate, off, yaw), sp = world.spawn;
  const c = clearance(baseGeom, sp.x, sp.z, sp.yaw, .42, .19, .5), s = projectWindow(sp.x, sp.z, 0, LOOP_LENGTH).s;
  loopPointAt(s - .05, PA); loopPointAt(s + .05, PB);
  return {clearance: c, dpsi: wrap(sp.yaw - Math.atan2(-(PB.z - PA.z), PB.x - PA.x)), s};
}
// 一様乱数の開始: mulberry32(種) で ゲート ⌊33u⌋・左右 −0.2 + 0.4u・向き −15 + 30u の順(r3 と同じ)
export function drawStarts(count, seed, {minClr = .12, minAbsDpsi = 0} = {}) {
  const rnd = mulberry32(seed), out = [];
  let drawn = 0, clrOk = 0;
  while (out.length < count) {
    const st = {gate: Math.floor(rnd() * 33), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    drawn++;
    if (drawn > 200000) throw new Error('eval_line_r4: 開始が引けません');
    const I = startInfo(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(I.clearance >= minClr)) continue;
    clrOk++;
    if (Math.abs(I.dpsi) < minAbsDpsi) continue;
    out.push({...st, initial_clearance_m: I.clearance, dpsi: I.dpsi, draw_index: drawn - 1});
  }
  return {starts: out, drawn, clr_ok: clrOk};
}
export function haltonSet(from, count, maxI) {
  const out = [];
  let skipped = 0;
  for (let i = from; out.length < count && i < maxI; i++) {
    const st = haltonStart(i, DIST), I = startInfo(st.gate, st.left_offset_m, st.left_yaw_deg);
    if (!(I.clearance > 1e-9) || I.clearance < DIST.min_initial_clearance_m) { skipped++; continue; }
    out.push({...st, initial_clearance_m: I.clearance, dpsi: I.dpsi});
  }
  if (out.length !== count) throw new Error(`eval_line_r4: Halton の開始が ${out.length} 個しか作れません(要 ${count})`);
  return {starts: out, skipped};
}
// 角の箱(protocol_line_r4 definitions_added.corner_box): 左右の符号 so・向きの符号 sy の角で、|左右| ∈ [0.16, 0.20] m・|向き| ∈ [12, 15]°
//   (各ゲートの開始の面積の 1%)。開始のすき間 ≥ 0.12 m になるまで引き直す(上限 maxTries。引けなければ飛ばして記録)。引く順は |左右|・|向き|
export const CORNERS = [[-1, 1], [-1, -1], [1, 1], [1, -1]];
function cornerStart(gate, so, sy, rnd, maxTries = 60) {
  for (let k = 0; k < maxTries; k++) {
    const off = so * (.16 + .04 * rnd()), yaw = sy * (12 + 3 * rnd()), I = startInfo(gate, off, yaw);
    if (I.clearance >= .12) return {gate, left_offset_m: off, left_yaw_deg: yaw, initial_clearance_m: I.clearance, dpsi: I.dpsi, corner: `${so < 0 ? '-' : '+'}off${sy < 0 ? '-' : '+'}yaw`, tries: k + 1};
  }
  return null;
}
// 確認 r3 で崩れた箱(ゲート 16・17、左右 U(−0.20, −0.16)・向き U(12, 15)°)
function boxStart(gate, rnd, maxTries = 200) {
  for (let k = 0; k < maxTries; k++) {
    const off = -.2 + .04 * rnd(), yaw = 12 + 3 * rnd(), I = startInfo(gate, off, yaw);
    if (I.clearance >= .12) return {gate, left_offset_m: off, left_yaw_deg: yaw, initial_clearance_m: I.clearance, dpsi: I.dpsi, corner: '-off+yaw', tries: k + 1};
  }
  throw new Error('eval_line_r4: 箱の開始が引けません ' + gate);
}
const FAM6 = [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}],
  ['inside', {family: 'inside', a: .8}], ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}]];
const EXT6 = [{family: 'const', c0: .9}, {family: 'const', c0: -.9}, {family: 'inside', a: .9}, {family: 'outside', a: .9}, {family: 'late_apex', a: .9}, {family: 'early_apex', a: .9}];
const FASTEST_POLICY = `${EXP}/bundle/fastest/policy.json`;
// 確認 r3 の崩れた 5 回(check_line_r3/repro.json)と、ちょうど角の 32 回(確認 r3 の NBC と同じ定義)
const FAIL5 = [
  ['GR-2', {gate: 16, left_offset_m: -.2, left_yaw_deg: 14.5, seed: 4723002}, {line: {family: 'const', c0: 0}, c_v: .95}],
  ['C129', {gate: 16, left_offset_m: -.2, left_yaw_deg: 15, seed: 4713129}, {line: {family: 'random', seed: 2155857802}, c_v: .9}],
  ['C137', {gate: 17, left_offset_m: -.2, left_yaw_deg: 15, seed: 4713137}, {line: {family: 'const', c0: 0}, c_v: 1}],
  ['GR-7', {gate: 16, left_offset_m: -.2, left_yaw_deg: 13, seed: 4723007}, {line: {family: 'random', seed: 777}, c_v: .95}],
  ['BOX17-44', {gate: 17, left_offset_m: -0.19577976564876737, left_yaw_deg: 14.314614483620971, seed: 4725044}, {line: {family: 'random', seed: 53318559}, c_v: .998}]];
const NBC_SHAPES = [['const_left', {family: 'const', c0: .75}], ['const_right', {family: 'const', c0: -.75}], ['const_center', {family: 'const', c0: 0}], ['inside', {family: 'inside', a: .75}],
  ['outside', {family: 'outside', a: .75}], ['late_apex', {family: 'late_apex', a: .75}], ['early_apex', {family: 'early_apex', a: .75}], ['random', {family: 'random', seed: 777}]];

export function buildJobs(set) {
  const J = [], meta = {};
  const add = (test, id, spec) => J.push({test, id, spec});
  const shaped = (name, prefix, starts, shapeSeed, seedBase) => {
    const rnd = mulberry32(shapeSeed);
    starts.forEach((st, i) => {
      const kind = PLAN15[i % 15], line = lineFor(kind, rnd), cv = +(0.9 + 0.1 * rnd()).toFixed(4);
      add(name, `${prefix}${i}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: seedBase + i,
        initial_clearance_m: st.initial_clearance_m, dpsi: st.dpsi, corner: st.corner ?? null}, params: {line, c_v: cv}, family: kind});
    });
  };
  const haltonJobs = (name, prefix, from, count, seedBase, maxI) => {
    const {starts, skipped} = haltonSet(from, count, maxI);
    meta[name] = {halton_from: from, last_index: starts.at(-1).halton_index, skipped_low_clearance: skipped, seed_base: seedBase};
    starts.forEach((st, i) => {
      const seed = seedBase + i, rng = createRng(mix32(seed, 0x11e)), kind = PLAN15[i % 15];
      add(name, `${prefix}${i}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed, halton_index: st.halton_index,
        initial_clearance_m: st.initial_clearance_m, dpsi: st.dpsi}, params: {line: lineParamsFor(kind, rng), c_v: +(0.9 + 0.1 * rng.uniform()).toFixed(4)}, family: kind});
    });
  };
  const headJobs = (name, prefix, count, startSeed, shapeSeed, seedBase) => {
    const {starts, drawn, clr_ok} = drawStarts(count, startSeed, {minAbsDpsi: 1.0});
    meta[name] = {rng_seed: startSeed, shape_seed: shapeSeed, drawn, clearance_ok: clr_ok, seed_base: seedBase};
    shaped(name, prefix, starts, shapeSeed, seedBase);
  };
  const uniformJobs = (name, prefix, count, startSeed, shapeSeed, seedBase) => {
    const {starts, drawn, clr_ok} = drawStarts(count, startSeed);
    meta[name] = {rng_seed: startSeed, shape_seed: shapeSeed, drawn, clearance_ok: clr_ok, seed_base: seedBase};
    shaped(name, prefix, starts, shapeSeed, seedBase);
  };
  const extJobs = (name, prefix, startSeed, shapeSeed, seedBase) => {
    const {starts, drawn, clr_ok} = drawStarts(60, startSeed), rnd = mulberry32(shapeSeed);
    meta[name] = {rng_seed: startSeed, shape_seed: shapeSeed, drawn, clearance_ok: clr_ok, seed_base: seedBase};
    starts.forEach((st, i) => {
      const line = i % 10 < 6 ? EXT6[i % 10] : {family: 'random', seed: Math.floor(rnd() * 4294967296) >>> 0}, cv = i % 2 ? 1.0 : 0.9;
      const kind = line.family === 'const' ? (line.c0 > 0 ? 'const_left' : 'const_right') : line.family;
      add(name, `${prefix}${i}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: seedBase + i,
        initial_clearance_m: st.initial_clearance_m, dpsi: st.dpsi}, params: {line, c_v: cv}, family: kind});
    });
  };
  // 角の組: ゲートごとに corners の角(全部 or 乱数で k 個)を 1 回ずつ
  const cornerJobs = (name, prefix, startSeed, shapeSeed, seedBase, perGate = 4) => {
    const rnd = mulberry32(startSeed), starts = [], skipped = [];
    for (let g = 0; g < 33; g++) {
      let cs = CORNERS;
      if (perGate < 4) { const order = CORNERS.map(c => [rnd(), c]).sort((a, b) => a[0] - b[0]).map(x => x[1]); cs = order.slice(0, perGate); }
      for (const [so, sy] of cs) { const st = cornerStart(g, so, sy, rnd); if (st) starts.push(st); else skipped.push({gate: g, so, sy}); }
    }
    meta[name] = {rng_seed: startSeed, shape_seed: shapeSeed, seed_base: seedBase, per_gate: perGate, n: starts.length, skipped};
    shaped(name, prefix, starts, shapeSeed, seedBase);
  };
  const boxJobs = (name, prefix, perGate, startSeed, shapeSeed, seedBase) => {
    const rnd = mulberry32(startSeed), starts = [];
    for (const g of [16, 17]) for (let i = 0; i < perGate; i++) starts.push(boxStart(g, rnd));
    meta[name] = {rng_seed: startSeed, shape_seed: shapeSeed, seed_base: seedBase, per_gate: perGate, gates: [16, 17], box: {left_offset_m: [-.2, -.16], left_yaw_deg: [12, 15]}};
    shaped(name, prefix, starts, shapeSeed, seedBase);
  };
  const fam6 = () => { for (const [nm, line] of FAM6) add('LN-FAM6', nm, {start: STD, params: {line, c_v: 1.0}, family: nm}); };
  const fail5 = name => { for (const [id, start, params] of FAIL5) { const I = startInfo(start.gate, start.left_offset_m, start.left_yaw_deg);
    add(name, id, {start: {...start, initial_clearance_m: I.clearance, dpsi: I.dpsi, corner: '-off+yaw'}, params, family: params.line.family}); } };
  const nbc32 = name => { let i = 0; for (const g of [16, 17]) for (const [kind, line] of NBC_SHAPES) for (const cv of [.9, 1.0]) {
    const I = startInfo(g, -.2, 15);
    add(name, `NBC-${i}`, {start: {gate: g, left_offset_m: -.2, left_yaw_deg: 15, seed: 4722000 + i, initial_clearance_m: I.clearance, dpsi: I.dpsi, corner: '-off+yaw'},
      params: {line, c_v: cv}, family: kind}); i++; } };
  // 確認 r3 の BOX16・BOX17(nb_probe.mjs box と同じ作り方: mulberry32(0x6c6e3312 / 0x6c6e3313)、形は種 + 0x100、回の種 4724000 / 4725000 + i)
  const box160 = name => {
    for (const [g, sSeed, base] of [[16, 0x6c6e3312, 4724000], [17, 0x6c6e3313, 4725000]]) {
      const rs = mulberry32(sSeed), rsh = mulberry32(sSeed + 0x100);
      let i = 0;
      while (i < 80) {
        const off = -.2 + .04 * rs(), yaw = 12 + 3 * rs(), I = startInfo(g, off, yaw);
        if (!(I.clearance >= .12)) continue;
        const kind = PLAN15[i % 15], line = lineFor(kind, rsh), cv = +(0.9 + 0.1 * rsh()).toFixed(4);
        add(name, `BOX${g}-${i}`, {start: {gate: g, left_offset_m: off, left_yaw_deg: yaw, seed: base + i, initial_clearance_m: I.clearance, dpsi: I.dpsi, corner: '-off+yaw'},
          params: {line, c_v: cv}, family: kind});
        i++;
      }
    }
  };
  if (set === 'dev') {
    fail5('DEV-FAIL5');
    nbc32('DEV-NBC32');
    box160('DEV-BOX160');
    haltonJobs('DEV-H60', 'DH', 98001, 60, 8930000, 99999);
    headJobs('DEV-HEAD40', 'DV', 40, 0xde740001, 0xde740002, 8931000);
    cornerJobs('DEV-CORNER', 'DC', 0xde740003, 0xde740004, 8932000, 2);
    fam6();
  } else if (set === 'validation') {
    haltonJobs('LN-V60-R4', 'V', 95001, 60, 8530000, 97999);
    headJobs('LN-VHEAD40-R4', 'VH', 40, 0x4ead0013, 0x4ead0014, 8531000);
    cornerJobs('LN-VCORNER-R4', 'VC', 0xc0e40003, 0xc0e40004, 8533000, 2);
    boxJobs('LN-VBOX40-R4', 'VB', 20, 0xb0c40003, 0xb0c40004, 8534000);
    fam6();
  } else if (set === 'test' || set === 'fastest') {
    haltonJobs('LN-P60-R4', 'P', 90001, 60, 8430000, 94999);
    if (set === 'fastest') {
      for (const j of J) { j.spec.mode = 'fastest'; j.spec.policyPath = FASTEST_POLICY; j.spec.noShield = true; j.test = 'LN-P60-R4-FAST'; }
    } else {
      headJobs('LN-HEAD60-R4', 'H', 60, 0x4ead0011, 0x4ead0012, 8431000);
      extJobs('LN-EXT60-R4', 'E', 0xe7e70011, 0xe7e70012, 8432000);
      cornerJobs('LN-CORNER-R4', 'C', 0xc0e40001, 0xc0e40002, 8433000, 4);
      boxJobs('LN-BOX120-R4', 'B', 60, 0xb0c40001, 0xb0c40002, 8434000);
      fam6();
    }
  } else if (set === 'fresh') {
    uniformJobs('LN-F300-R4', 'F', 300, 0x5eed4c01, 0x5eed4c02, 8630000);
  } else if (set === 'regression') {
    fail5('REG-FAIL5');
    nbc32('REG-NBC32');
    box160('REG-BOX160');
  } else throw new Error('eval_line_r4: 知らない組 ' + set);
  return {jobs: J, meta};
}

// ---- 1 回 ----
// 止まり(protocol_line_r3 definitions_added.stall): フレーム 30 以降に |v| < 0.2 m/s が 30 フレーム以上続く区間
export function stallOf(log) {
  let run = 0, best = 0, at = null;
  for (let f = 30; f < log.length; f++) {
    if (Math.abs(log[f].v) < .2) { run++; if (run > best) { best = run; if (run >= 30 && at == null) at = f - 29; } } else run = 0;
  }
  return {stall: best >= 30, longest_frames: best, first_frame: at};
}
const policyCache = new Map();
function policyOf(p) { if (!policyCache.has(p)) policyCache.set(p, {weights: readJson(p), sha: sha256File(p), path: p}); return policyCache.get(p); }
const r4 = x => x == null || !Number.isFinite(x) ? null : +x.toFixed(4);
async function runJob(job, ctx) {
  const sp = job.spec, t0 = Date.now();
  const P = sp.policyPath ? policyOf(sp.policyPath) : ctx.policy;
  const noShield = ctx.noShield || sp.noShield;
  const r = await runModeEpisode({mode: sp.mode ?? 'line', weights: P.weights, policySha256: P.sha, start: sp.start, params: sp.mode === 'fastest' ? {} : sp.params,
    episodeMode: 'continuous', coreKind: ctx.core, overrides: noShield ? {shield: false} : {}, audit: true,
    modeRuntime: sp.mode === 'fastest' || noShield ? null : ctx.modeRuntime});
  const row = {test: job.test, id: job.id, start: sp.start, family: sp.family ?? null, c_v: sp.params?.c_v ?? null, line: sp.params?.line ?? null,
    policy_sha256: P.sha, invalid_spawn: !!r.invalid_spawn};
  if (r.invalid_spawn) return row;
  const log = r.log, st = stallOf(log);
  Object.assign(row, {reason: r.reason, frames: r.frames, collisions: r.collisions, collision: r.collision, laps: r.laps.map(l => l.elapsed), valid_laps: r.valid_laps,
    valid_episode: r.valid_episode, stall: st.stall, stall_longest_frames: st.longest_frames, stall_first_frame: st.first_frame,
    min_clearance: r.min_clearance ?? null, min_clearance_t: r.min_clearance_at?.time ?? null, audit_err: r.audit?.max_pos_err ?? null,
    shield_frames: r.shield_frames, log_frames: log.length, shield_fraction: r.shield_frames / Math.max(1, log.length),
    gate_misses: r.gate_misses, gate_stuck_max_m: r.gate_stuck_max_m, line_profile: r.params?.line_profile ?? null,
    dmin_frame_min: Math.min(...log.map(q => q.dmin)), dmin_first_1p5s: Math.min(...log.slice(0, 46).map(q => q.dmin)),
    dmin_first_3s: Math.min(...log.slice(0, 91).map(q => q.dmin)), epsi0: log[0]?.epsi ?? null,
    reverse_frames: r.trace.filter(q => q.v < -.01).length, neg_speed_cmd_frames: log.filter(q => q.sp < 0).length,
    teacher_shield_meta: r.teacher_metadata?.shield ? {version: r.teacher_metadata.shield.version, c_nominal: r.teacher_metadata.shield.c_nominal} : null});
  row.valid_r3 = row.valid_episode && !row.stall && row.valid_laps >= 1;
  // シールドが替えたフレームの替え方の内訳(と、提案をそのまま通したときの理由)
  const fb = {}, fbKeep = {};
  for (const q of log) {
    const k = q.shield?.fb ?? 'pass';
    if (q.sh === 1) fb[k] = (fb[k] ?? 0) + 1;
    else if (q.shield?.fb) fbKeep[k] = (fbKeep[k] ?? 0) + 1;
  }
  row.shield_fallbacks = fb; row.shield_kept = fbKeep;
  row.escape_frames = log.filter(q => q.sh === 1 && String(q.shield?.fb ?? '').startsWith('escape_')).length;
  if (row.shield_frames) { const f1 = log.find(q => q.sh === 1); row.shield_first_t = f1.f / 30; }
  const run = log.filter(q => q.v > .3).map(q => q.v);
  row.speed = {mean_running: mean(run), p95: quantile(run, .95), max: Math.max(0, ...log.map(q => q.v))};
  const sc = sceneFromEpisode(r), rnd = (a, k) => Array.from(a, v => Number.isNaN(v) ? null : +v.toFixed(k));
  row.prof = {ey: rnd(binProfile(sc, 'ey'), 4), v: rnd(binProfile(sc, 'v'), 3), st: rnd(binProfile(sc, q => steerUs(q.st / .52)), 2)};
  row.vm_frames = sc.frames.filter((q, i) => i % 3 === 0 && q.s - sc.s0 >= 3).map(q => [+q.s.toFixed(3), +q.ey.toFixed(4), +q.epsi.toFixed(4), +q.v.toFixed(3)]);
  row.st_time = log.map(q => +q.st.toFixed(5));
  if (log[0]?.es !== undefined) {
    const s0 = log[0].s, sel = log.filter(q => q.s - s0 >= 3 && q.es != null);
    const te = sel.map(q => Math.abs(q.ey - q.es));
    row.tracking = {n: te.length, median: quantile(te, .5), p95: quantile(te, .95), mean: mean(te)};
    row.tracking_abs = te.map(x => +x.toFixed(4)).filter((_, i) => i % 2 === 0);
    row.tracking_s = sel.map(q => +(((q.s % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH).toFixed(2)).filter((_, i) => i % 2 === 0);
  }
  if (ctx.keepPath) {
    row.path = r.trace.filter((_, i) => i % 2 === 0).map(q => [+q.x.toFixed(3), +q.z.toFixed(3), +q.v.toFixed(2)]);
    row.sh_frames = log.filter(q => q.sh === 1).map(q => q.f);
    row.ey_s = log.filter((_, i) => i % 2 === 0).map(q => [r4(q.s - log[0].s), r4(q.ey), q.es == null ? null : r4(q.es), r4(q.dmin)]);
  }
  row.wall_s = (Date.now() - t0) / 1000;
  return row;
}

// ---- 判定 ----
const minOrNull = a => a.length ? Math.min(...a) : null;
const geq = (v, t) => v != null && v >= t;
const clr0 = r => (r.invalid_spawn || r.collisions > 0 || r.min_clearance == null) ? 0 : r.min_clearance;
const profSpread = (rs, key) => spread(rs.filter(r => r.prof).map(r => Float64Array.from(r.prof[key], v => v == null ? NaN : v)));
function fam6Judge(fam) {
  const F = Object.fromEntries(fam.map(r => [r.id, r]));
  const famOk = fam.length === 6 && fam.every(r => r.valid_r3);
  const prof = id => Float64Array.from(F[id]?.prof?.ey ?? [], v => v == null ? NaN : v);
  let L4 = {pass: false, value: null};
  if (fam.length === 6) {
    const ids = Object.keys(F), P = ids.map(prof), keep = [];
    for (let i = 0; i < (P[0]?.length ?? 0); i++) if (P.every(p => !Number.isNaN(p[i]))) keep.push(i);
    const d = (a, b) => keep.map(i => Math.abs(prof(a)[i] - prof(b)[i]));
    const lr = mean(d('c+0.8', 'c-0.8')), io = mean(d('inside', 'outside')), pairs = [];
    for (let a = 0; a < ids.length; a++) for (let b = a + 1; b < ids.length; b++) {
      const dd = d(ids[a], ids[b]); let best = 0;
      for (let i = 0; i + 20 <= dd.length; i++) { let s = 0; for (let k = 0; k < 20; k++) s += dd[i + k]; best = Math.max(best, s / 20); }
      pairs.push({a: ids[a], b: ids[b], max_1m_avg: best});
    }
    L4 = {value: {left_right_mean: lr, inside_outside_mean: io, min_pair_max_1m_avg: Math.min(...pairs.map(p => p.max_1m_avg)), bins: keep.length}, pairs,
      pass: famOk && lr >= .20 && io >= .10 && pairs.every(p => p.max_1m_avg >= .08)};
  }
  const L6 = {value: F.c0?.laps?.[0] ?? null, pass: !!F.c0?.valid_r3 && F.c0.laps[0] <= 10.0};
  return {L4: {...L4, fam6_valid_r3: famOk}, L6};
}
function trackingJudge(ok) {
  const te = ok.flatMap(r => r.tracking_abs ?? []);
  return {value: {median: quantile(te, .5), p95: quantile(te, .95), frames: te.length}, pass: te.length > 0 && quantile(te, .5) <= .05 && quantile(te, .95) <= .12};
}
const shieldFrac = rs => rs.reduce((a, r) => a + (r.shield_frames ?? 0), 0) / Math.max(1, rs.reduce((a, r) => a + (r.log_frames ?? 0), 0));
const validCount = rs => rs.filter(r => r.valid_r3).length;
// 組の名前(検証・試験): 主 = Halton の組(L3・L5・L7 の組)、ほかは L1・L2・L8・L9 の組
const SETS = {
  validation: {main: 'LN-V60-R4', others: [['LN-VHEAD40-R4', 40, 'ratio'], ['LN-VCORNER-R4', 57, 'ratio95'], ['LN-VBOX40-R4', 40, 'ratio95']], hard: ['LN-VHEAD40-R4', 'LN-VCORNER-R4', 'LN-VBOX40-R4']},
  test: {main: 'LN-P60-R4', others: [['LN-HEAD60-R4', 60, 'ratio'], ['LN-EXT60-R4', 60, 'ratio'], ['LN-CORNER-R4', 116, 'ratio95'], ['LN-BOX120-R4', 120, 'ratio95']],
    hard: ['LN-HEAD60-R4', 'LN-EXT60-R4', 'LN-CORNER-R4', 'LN-BOX120-R4']},
  dev: {main: 'DEV-H60', others: [['DEV-HEAD40', 40, 'ratio'], ['DEV-CORNER', null, 'ratio95'], ['DEV-BOX160', 160, 'ratio95'], ['DEV-NBC32', 32, 'ratio95'], ['DEV-FAIL5', 5, 'ratio95']],
    hard: ['DEV-HEAD40', 'DEV-CORNER', 'DEV-BOX160', 'DEV-NBC32', 'DEV-FAIL5']},
};
// 有効の数の決まり: ratio = 57/60 と同じ割合(⌈0.95·n⌉)、ratio95 = ⌈0.95·n⌉(角・箱の組。protocol_line_r4 acceptance.L1_r4)
const need = n => Math.ceil(0.95 * n - 1e-9);
function summarizeRuns(rs) {
  return {n: rs.length, valid_r3: validCount(rs), collisions: rs.filter(r => r.collisions > 0 || r.invalid_spawn).length, stalls: rs.filter(r => r.stall).length,
    min_all: minOrNull(rs.map(clr0)), min_valid: minOrNull(rs.filter(r => r.valid_r3).map(r => r.min_clearance)), below_3cm_valid: rs.filter(r => r.valid_r3 && r.min_clearance < .03).length,
    below_4cm: rs.filter(r => clr0(r) < .04).length, shield_runs: rs.filter(r => r.shield_frames > 0).length, shield_fraction: shieldFrac(rs),
    shield_gt10pct: rs.filter(r => (r.shield_fraction ?? 0) > .10).length, reverse_runs: rs.filter(r => (r.reverse_frames ?? 0) > 0).length,
    escape_runs: rs.filter(r => (r.escape_frames ?? 0) > 0).length, invalid_other: rs.filter(r => !r.valid_r3 && !r.stall && !(r.collisions > 0)).length};
}
export function judge(set, rows) {
  const T = name => rows.filter(r => r.test === name);
  const J = {}, rep = {};
  const S = SETS[set];
  if (S) {
    const P = T(S.main), F = T('LN-FAM6'), nP = 60;
    const others = S.others.map(([name, n]) => ({name, rs: T(name), n: n ?? T(name).length}));
    const okP = P.filter(r => r.valid_r3);
    const l1 = {[S.main]: `${okP.length}/${P.length}`};
    let l1pass = P.length === nP && okP.length >= 57;
    for (const o of others) { const ok = validCount(o.rs); l1[o.name] = `${ok}/${o.rs.length}`; l1pass = l1pass && o.rs.length === o.n && o.n > 0 && ok >= need(o.n); }
    J.L1_r4 = {value: l1, pass: l1pass};
    const okAll = [...okP, ...others.flatMap(o => o.rs.filter(r => r.valid_r3))];
    const mc = minOrNull(okAll.map(r => r.min_clearance).concat(F.map(clr0)));
    J.L2_r4 = {value: mc, pass: geq(mc, .03)};
    J.L3_tracking = trackingJudge(okP);
    const fj = fam6Judge(F);
    J.L4_distinct = fj.L4;
    const vm3 = profSpread(okP, 'ey').mean_abs_dev;
    J.L5_vm3 = {value: vm3, pass: vm3 != null && vm3 >= .08};
    J.L6_lap = fj.L6;
    const fracP = shieldFrac(P), perRunMax = Math.max(0, ...[...P, ...F].map(r => r.shield_fraction ?? 0));
    J.L7_shield = {value: {P_fraction: fracP, max_per_run_P_FAM6: perRunMax}, pass: P.length === nP && fracP <= .02 && perRunMax <= .10};
    const all = [...P, ...others.flatMap(o => o.rs), ...F];
    const col = all.filter(r => r.invalid_spawn || r.collisions !== 0);
    const nAll = nP + others.reduce((a, o) => a + o.n, 0) + 6;
    J.L8_no_collision = {value: `${all.length - col.length}/${all.length}`, collided: col.map(r => `${r.test}:${r.id}`), pass: col.length === 0 && all.length === nAll};
    const stl = all.filter(r => r.stall);
    J.L9_no_stall = {value: `${all.length - stl.length}/${all.length}`, stalled: stl.map(r => `${r.test}:${r.id}`), pass: stl.length === 0 && all.length === nAll};
    const rev = [...P, ...F].filter(r => (r.reverse_frames ?? 0) > 0);
    J.L10_no_reverse_P_FAM6 = {value: rev.length, runs: rev.map(r => `${r.test}:${r.id}`), pass: rev.length === 0 && P.length === nP && F.length === 6};
    // 選び方の値: むずかしい組(大きく横を向いた開始・角・箱)の最小すき間(衝突・止まりの回は 0)
    const hard = rows.filter(r => S.hard.includes(r.test));
    rep.hard_min_clearance = minOrNull(hard.map(r => r.stall ? 0 : clr0(r)));
    rep.hard_fail = hard.filter(r => r.stall || r.collisions > 0 || r.invalid_spawn || (r.valid_r3 && r.min_clearance < .03)).length;
    rep.by_set = Object.fromEntries([[S.main, P], ...others.map(o => [o.name, o.rs]), ['LN-FAM6', F]].map(([k, rs]) => [k, summarizeRuns(rs)]));
    rep.stalls = stl.map(r => ({test: r.test, id: r.id, gate: r.start.gate, off: r.start.left_offset_m, yaw: r.start.left_yaw_deg, dpsi: r.start.dpsi,
      first_t: r.stall_first_frame / 30, longest_s: r.stall_longest_frames / 30, fb: r.shield_fallbacks}));
    rep.not_valid_r3 = all.filter(r => !r.valid_r3).map(r => ({test: r.test, id: r.id, reason: r.reason, frames: r.frames, stall: r.stall, valid_laps: r.valid_laps,
      collision: r.collision, gate: r.start.gate, off: r.start.left_offset_m, yaw: r.start.left_yaw_deg, dpsi: r.start.dpsi, family: r.family, line: r.line}));
    rep.shield_gt5pct = all.filter(r => (r.shield_fraction ?? 0) > .05).map(r => ({test: r.test, id: r.id, fraction: r.shield_fraction, gate: r.start.gate, dpsi: r.start.dpsi}));
    rep.shield_fraction = Object.fromEntries([[S.main, P], ...others.map(o => [o.name, o.rs]), ['LN-FAM6', F]].map(([k, rs]) => [k, shieldFrac(rs)]));
    rep.shield_fallbacks = all.reduce((acc, r) => { for (const [k, v] of Object.entries(r.shield_fallbacks ?? {})) acc[k] = (acc[k] ?? 0) + v; return acc; }, {});
    rep.shield_kept = all.reduce((acc, r) => { for (const [k, v] of Object.entries(r.shield_kept ?? {})) acc[k] = (acc[k] ?? 0) + v; return acc; }, {});
    rep.reverse = all.filter(r => (r.reverse_frames ?? 0) > 0).map(r => ({test: r.test, id: r.id, frames: r.reverse_frames, escape_frames: r.escape_frames}));
    const lh = all.filter(r => r.start.dpsi != null && Math.abs(r.start.dpsi) >= 1.0);
    rep.large_heading = {n: lh.length, valid_r3: lh.filter(r => r.valid_r3).length, min_clearance: minOrNull(lh.map(clr0)), min_clearance_first_3s_runtime: minOrNull(lh.map(r => r.dmin_first_3s))};
    rep.below_5cm = all.filter(r => clr0(r) < .05).map(r => ({test: r.test, id: r.id, min_clearance: clr0(r), t_s: r.min_clearance_t, valid_r3: r.valid_r3, gate: r.start.gate,
      off: r.start.left_offset_m, yaw: r.start.left_yaw_deg, dpsi: r.start.dpsi, family: r.family, shield_frames: r.shield_frames}));
    rep.by_family = Object.fromEntries([...new Set(P.map(r => r.family))].map(f => { const rs = P.filter(r => r.family === f), ok = rs.filter(r => r.valid_r3);
      return [f, {n: rs.length, valid_r3: ok.length, tracking_median: quantile(ok.flatMap(r => r.tracking_abs ?? []), .5), tracking_p95: quantile(ok.flatMap(r => r.tracking_abs ?? []), .95),
        lap1: mean(ok.map(r => r.laps[0]).filter(Number.isFinite))}]; }));
    const fr = okP.flatMap(r => (r.tracking_abs ?? []).map((e, k) => [e, r.tracking_s?.[k]])), big = fr.filter(([e]) => e > .12);
    const hist = {}; for (const [, s] of big) { const b = Math.floor(s); hist[b] = (hist[b] ?? 0) + 1; }
    rep.tracking_gt12 = {frac: fr.length ? big.length / fr.length : null, n: big.length, by_s_1m: hist};
    rep.lap1 = {P_median: quantile(okP.map(r => r.laps[0]).filter(Number.isFinite), .5), FAM6: Object.fromEntries(F.map(r => [r.id, r.laps[0] ?? null]))};
    rep.speed_mean_running_P = mean(okP.map(r => r.speed.mean_running));
    const vmRows = okP.filter(r => r.vm_frames);
    if (vmRows.length >= 2) {
      const scenes = vmRows.map(r => ({frames: r.vm_frames.map(([s, ey, epsi, v]) => ({s, ey, epsi, v})), s0: -Infinity}));
      rep.variation_P = {rows: vmRows.length, VM1_us: VM1(vmRows.map(r => ({frames: r.st_time.map(st => ({st}))}))), VM2_us: profSpread(vmRows, 'st'),
        VM3_m: profSpread(vmRows, 'ey'), VM4_mps: profSpread(vmRows, 'v'), VM7_cells: VM7(scenes, 3000).cells};
    }
    rep.wall_s_per_run = mean(rows.map(r => r.wall_s).filter(Number.isFinite));
  } else if (set === 'fresh') {
    const R = T('LN-F300-R4');
    rep.blocks = [0, 1, 2, 3, 4].map(b => {
      const rs = R.slice(60 * b, 60 * b + 60), ok = rs.filter(r => r.valid_r3);
      return {block: b, L1: `${ok.length}/${rs.length}`, L1_pass: ok.length >= 57, L2_min_valid: minOrNull(ok.map(r => r.min_clearance)), L2_pass: geq(minOrNull(ok.map(r => r.min_clearance)), .03),
        L3: trackingJudge(ok).value, L3_pass: trackingJudge(ok).pass, L5_vm3: profSpread(ok, 'ey').mean_abs_dev, stalls: rs.filter(r => r.stall).length,
        collisions: rs.filter(r => r.collisions > 0).length, min_all: minOrNull(rs.map(clr0)), shield_fraction: shieldFrac(rs), reverse_runs: rs.filter(r => r.reverse_frames > 0).length};
    });
    rep.all = {...summarizeRuns(R), large_heading: R.filter(r => Math.abs(r.start.dpsi) >= 1).length, large_heading_min: minOrNull(R.filter(r => Math.abs(r.start.dpsi) >= 1).map(clr0)),
      below_5cm: R.filter(r => clr0(r) < .05).map(r => ({id: r.id, min: clr0(r), gate: r.start.gate, dpsi: r.start.dpsi, shield_frames: r.shield_frames})),
      not_valid_r3: R.filter(r => !r.valid_r3).map(r => ({id: r.id, reason: r.reason, stall: r.stall, gate: r.start.gate, family: r.family}))};
  } else if (set === 'regression') {
    rep.by_set = Object.fromEntries(['REG-FAIL5', 'REG-NBC32', 'REG-BOX160'].map(k => [k, summarizeRuns(T(k))]));
    rep.fail5 = T('REG-FAIL5').map(r => ({id: r.id, reason: r.reason, valid_r3: r.valid_r3, stall: r.stall, collisions: r.collisions, min_clearance: clr0(r),
      shield_frames: r.shield_frames, fb: r.shield_fallbacks, reverse_frames: r.reverse_frames}));
  } else if (set === 'fastest') {
    const P = T('LN-P60-R4-FAST');
    rep.fastest = {n: P.length, valid_r3: validCount(P), lap1_median: quantile(P.filter(r => r.valid_r3).map(r => r.laps[0]), .5), min_all: minOrNull(P.map(clr0)),
      vm3: profSpread(P.filter(r => r.valid_r3), 'ey').mean_abs_dev, speed_mean_running: mean(P.filter(r => r.valid_r3).map(r => r.speed.mean_running))};
  }
  const judged = Object.values(J).filter(x => x.pass !== null);
  return {judged: J, all_judged_pass: judged.length > 0 && judged.every(x => x.pass === true), reported: rep};
}

// ---- 実行 ----
export function shieldModeRuntime({set, shieldConfig = null}) {
  if (shieldConfig) {
    if (set !== 'dev' && set !== 'regression') throw new Error('eval_line_r4: --shield-config は下調べ(dev・regression)だけです。検証・試験は protocol_line_r4.json の値で');
    const c = readJson(shieldConfig);
    return {modeRuntime: {shield: true, shield_params: c.shield_params ?? c}, source: shieldConfig, sha256: sha256File(shieldConfig)};
  }
  if (!P4) throw new Error('eval_line_r4: protocol_line_r4.json(固めたもの)がまだありません');
  const mr = P4.runtime.shield.mode_runtime;
  return {modeRuntime: JSON.parse(JSON.stringify(mr)), source: 'protocol_line_r4.json runtime.shield.mode_runtime', sha256: P4._sha256};
}
async function main() {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; }, flag = n => args.includes(n);
  const set = opt('--set', 'validation'), core = opt('--core', set === 'validation' ? 'fast' : 'original');
  const jobs = +opt('--jobs', '1'), only = opt('--only')?.split(','), out = opt('--out'), policyPath = opt('--policy');
  const noShield = flag('--no-shield');
  if (!policyPath && set !== 'fastest') { console.error('使い方: node eval_line_r4.mjs --policy <json> --set dev|validation|test|fresh|regression|fastest'); process.exit(2); }
  const SH = noShield || set === 'fastest' ? null : shieldModeRuntime({set, shieldConfig: opt('--shield-config')});
  const ctx = {core, noShield, keepPath: flag('--keep-path'), policy: policyPath ? policyOf(policyPath) : null, modeRuntime: SH?.modeRuntime ?? null};
  let {jobs: J, meta} = buildJobs(set);
  if (only) J = J.filter(j => only.includes(j.test));
  const shard = opt('--shard');
  if (shard) {
    const [k, n] = shard.split('/').map(Number), rows = [];
    for (let j = k; j < J.length; j += n) { const row = await runJob(J[j], ctx); row._j = j; rows.push(row); }
    fs.writeFileSync(opt('--rows-out'), JSON.stringify(rows));
    return;
  }
  const t0 = Date.now();
  let rows = [];
  if (jobs > 1) {
    const tmp = fs.mkdtempSync(path.join(process.env.JF_TMP ?? '/tmp', 'eval_line_r4_'));
    await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
      const ch = fork(fileURLToPath(import.meta.url), [...args.filter((a, i) => !['--jobs', '--out'].includes(a) && !['--jobs', '--out'].includes(args[i - 1])), '--shard', `${k}/${jobs}`, '--rows-out', `${tmp}/${k}.json`], {stdio: 'inherit'});
      ch.on('exit', c => c === 0 ? res() : rej(new Error('子プロセスが失敗しました ' + c)));
    })));
    for (let k = 0; k < jobs; k++) rows.push(...JSON.parse(fs.readFileSync(`${tmp}/${k}.json`, 'utf8')));
    fs.rmSync(tmp, {recursive: true});
    rows.sort((a, b) => a._j - b._j);
  } else for (const [j, job] of J.entries()) { const r = await runJob(job, ctx); r._j = j; rows.push(r); if ((j + 1) % 20 === 0) console.error(`${j + 1}/${J.length}`); }
  const verdict = judge(set, rows);
  const shieldMjs = path.join(here, 'rt4/shield.mjs'), modeRt = path.join(here, 'rt4/mode_runtime.mjs');
  const res = {schema: 'jf-sim-rl-line-r4-eval/v1', created_at: new Date().toISOString(), set, core,
    policy: ctx.policy ? {path: ctx.policy.path, sha256: ctx.policy.sha, schema: ctx.policy.weights.schema, obs_dim: ctx.policy.weights.obs_dim, meta: ctx.policy.weights.meta ?? null} : null,
    shield: SH ? {mode_runtime: SH.modeRuntime, source: SH.source, source_sha256: SH.sha256, shield_mjs_sha256: sha256File(shieldMjs), mode_runtime_sha256: sha256File(modeRt)} : null,
    protocol_line_r4_sha256: P4?._sha256 ?? null, protocol_line_r2_sha256: P2._sha256, set_meta: meta, runs: rows.length, ...verdict, wall_s: (Date.now() - t0) / 1000,
    tests: Object.fromEntries([...new Set(rows.map(r => r.test))].map(t => [t, rows.filter(r => r.test === t).map(r => { const {vm_frames, st_time, tracking_abs, tracking_s, _j, ...rest} = r; return rest; })]))};
  if (out) { fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true}); fs.writeFileSync(out, JSON.stringify(res)); }
  console.log(JSON.stringify({set, runs: rows.length, all_judged_pass: res.all_judged_pass, wall_s: res.wall_s,
    judged: Object.fromEntries(Object.entries(res.judged).map(([k, v]) => [k, {pass: v.pass, value: v.value}])),
    hard_min: res.reported.hard_min_clearance, hard_fail: res.reported.hard_fail, by_set: res.reported.by_set}));
}
if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
