#!/usr/bin/env node
// 経路いろいろ(line)第 3 回の検証・試験・報告の組と判定(2026-10-14、r3 の学習の段 line)。protocol_line_r3.json のとおり。
// 共有の rl/eval_modes.mjs(line の部分)を元に書き直したもの。走らせ方は r3/line/episode_ln3.mjs(rl/episode.mjs の写し)、
// 走行中の仕組みは r3/line/rt(r3/runtime の固定の写し)。経路いろいろにもすき間シールド(protocol_line_r3.json runtime.shield.params)をかける:
//   mode_runtime.mjs の試作の切り替え globalThis.JF_R3 = {modeShield: {line: params}} を、この中で決まりの値から入れる。
//
//   node r3/line/eval_line_r3.mjs --policy <policy.json> --set validation|test|fresh|fallback|fastest [--core fast|original] [--jobs N]
//        [--only LN-P60-R3,…] [--out 結果.json] [--keep-path] [--no-shield]
//   validation: LN-V60-R3 + LN-VHEAD40-R3 + LN-FAM6(高速版でよい。チェックポイントを選ぶためだけ)
//   test:       LN-P60-R3 + LN-HEAD60-R3 + LN-EXT60-R3 + LN-FAM6(元の sim-core.js。選んだ 1 つにだけ 1 回)
//   fresh:      LN-F240-R3(報告だけ。元の sim-core.js)
//   fallback:   代わりの方法(protocol.json acceptance.fallbacks.line: ベストラップの 7 本の線の混ぜ合わせ、LN-P60-R3 の回 i に i mod 7。シールドなし)
//   fastest:    最速の方策(bundle/fastest)を LN-P60-R3 の 60 の開始で(線の違いの報告用。判定には使わない)
//
// 組の作り方(決まりの文言にない細かい所は、確認 line r2 の jobs_ln2.mjs と同じにした。train_line_r3.md に書く):
//   Halton の組(LN-P60-R3・LN-V60-R3): haltonStart(i, protocol_line_r2.start_distribution)、開始のすき間(features.mjs clearance)≥ 0.12 m を満たさない番号は飛ばす。
//     形・c_v は createRng(mix32(種, 0x11e)) で lineParamsFor(family_plan_15[i mod 15]) と c_v = 0.9 + 0.1·u(r2 と同じ)
//   一様乱数の組(LN-HEAD60-R3・LN-EXT60-R3・LN-VHEAD40-R3・LN-F240-R3): mulberry32(種) で ゲート ⌊33u⌋・左右 −0.2 + 0.4u・向き −15 + 30u の順に引き、
//     開始のすき間 ≥ 0.12 m(と large_heading_start: R に対する向きのずれ ≥ 1.0 rad)を満たすものだけ。形・強さ・c_v は別の mulberry32(種 + 1)
//     (HEAD 0x4ead0004・EXT 0xe7e70004 は決まりの値。VHEAD 0x4ead0006・F240 0x5eed2c44 は決まりに書いていないので同じ「+1」の作り方)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD} from './episode_ln3.mjs';
import {lineParamsFor} from './rt/mode_runtime.mjs';
import {haltonStart, createRng, mix32} from './rt/rng.mjs';
import {prepareGeometry, clearance, projectWindow, loopPointAt, LOOP_LENGTH} from './rt/features.mjs';
import {activeWorldFor} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';
import {binProfile, spread, quantile, VM1, VM7, sceneFromEpisode, steerUs} from './variation_metrics.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function loadVerified(jsonPath, shaPath, name) {
  const raw = fs.readFileSync(jsonPath), sha = crypto.createHash('sha256').update(raw).digest('hex');
  const want = fs.readFileSync(shaPath, 'utf8').split('\n').map(l => l.trim().split(/\s+/)).find(a => a[1] === name)?.[0];
  if (sha !== want) throw new Error(`eval_line_r3: ${name} の sha256 ${sha} が固めた値 ${want} と違います`);
  return {...JSON.parse(raw), _sha256: sha};
}
export const P3 = loadVerified(`${REP}/r3/protocol_line_r3.json`, `${REP}/r3/protocol_r3.sha256`, 'protocol_line_r3.json');
export const P2 = loadVerified(`${REP}/protocol_line_r2.json`, `${REP}/protocol_line_r2.sha256`, 'protocol_line_r2.json');
export const SHIELD_PARAMS = Object.freeze(JSON.parse(JSON.stringify(P3.runtime.shield.params)));
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
export function drawStarts(count, seed, {minClr = .12, minAbsDpsi = 0} = {}) {
  const rnd = mulberry32(seed), out = [];
  let drawn = 0, clrOk = 0;
  while (out.length < count) {
    const st = {gate: Math.floor(rnd() * 33), left_offset_m: -.2 + .4 * rnd(), left_yaw_deg: -15 + 30 * rnd()};
    drawn++;
    if (drawn > 200000) throw new Error('eval_line_r3: 開始が引けません');
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
  if (out.length !== count) throw new Error(`eval_line_r3: Halton の開始が ${out.length} 個しか作れません(要 ${count})`);
  return {starts: out, skipped};
}
const FAM6 = [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}],
  ['inside', {family: 'inside', a: .8}], ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}]];
const EXT6 = [{family: 'const', c0: .9}, {family: 'const', c0: -.9}, {family: 'inside', a: .9}, {family: 'outside', a: .9}, {family: 'late_apex', a: .9}, {family: 'early_apex', a: .9}];
const FALLBACK_POLICIES = ['main-s1/policies/step_010010624', 'main-s1/policies/step_006012928', 'main-s2/policies/step_010010624', 'r2-dr-s11/policies/step_001015808',
  'r3-s21/policies/step_001015808', 'r3-s22/policies/step_004014080', 'main-s0/policies/step_006012928'].map(q => `/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/rl/${q}.json`);
const FASTEST_POLICY = `${EXP}/bundle/fastest/policy.json`;

export function buildJobs(set) {
  const J = [], meta = {};
  const add = (test, id, spec) => J.push({test, id, spec});
  const haltonJobs = (name, spec, maxI) => {
    const {starts, skipped} = haltonSet(spec.halton_from, spec.count, maxI);
    meta[name] = {halton_from: spec.halton_from, last_index: starts.at(-1).halton_index, skipped_low_clearance: skipped};
    starts.forEach((st, i) => {
      const seed = spec.seed_base + i, rng = createRng(mix32(seed, 0x11e)), kind = PLAN15[i % 15];
      add(name, `P${i}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed, halton_index: st.halton_index,
        initial_clearance_m: st.initial_clearance_m, dpsi: st.dpsi}, params: {line: lineParamsFor(kind, rng), c_v: +(0.9 + 0.1 * rng.uniform()).toFixed(4)}, family: kind});
    });
  };
  const headJobs = (name, prefix, count, startSeed, shapeSeed, seedBase) => {
    const {starts, drawn, clr_ok} = drawStarts(count, startSeed, {minAbsDpsi: 1.0}), rnd = mulberry32(shapeSeed);
    meta[name] = {rng_seed: startSeed, shape_seed: shapeSeed, drawn, clearance_ok: clr_ok, large_heading_share_of_clearance_ok: count / clr_ok};
    starts.forEach((st, i) => {
      const kind = PLAN15[i % 15], line = lineFor(kind, rnd), cv = +(0.9 + 0.1 * rnd()).toFixed(4);
      add(name, `${prefix}${i}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: seedBase + i,
        initial_clearance_m: st.initial_clearance_m, dpsi: st.dpsi}, params: {line, c_v: cv}, family: kind});
    });
  };
  const fam6 = () => { for (const [nm, line] of FAM6) add('LN-FAM6', nm, {start: STD, params: {line, c_v: 1.0}, family: nm}); };
  const S = P3.sets;
  if (set === 'validation') {
    haltonJobs('LN-V60-R3', {halton_from: 18001, count: 60, seed_base: 3230000}, 20000);
    headJobs('LN-VHEAD40-R3', 'VH', 40, 0x4ead0005, 0x4ead0006, 3231000);
    fam6();
  } else if (set === 'test' || set === 'fallback' || set === 'fastest') {
    haltonJobs('LN-P60-R3', S.test['LN-P60-R3'], 30000);
    if (set === 'fallback') {
      for (const j of J) { j.spec.mode = 'fastest'; j.spec.policyPath = FALLBACK_POLICIES[+j.id.slice(1) % 7]; j.spec.noShield = true; }
      J.forEach(j => { j.test = 'LN-P60-R3-FB'; });
    } else if (set === 'fastest') {
      for (const j of J) { j.spec.mode = 'fastest'; j.spec.policyPath = FASTEST_POLICY; j.spec.noShield = true; }
      J.forEach(j => { j.test = 'LN-P60-R3-FAST'; });
    } else {
      headJobs('LN-HEAD60-R3', 'H', 60, 0x4ead0003, 0x4ead0004, S.test['LN-HEAD60-R3'].seed_base);
      const {starts, drawn, clr_ok} = drawStarts(60, 0xe7e70003), rnd = mulberry32(0xe7e70004);
      meta['LN-EXT60-R3'] = {rng_seed: 0xe7e70003, shape_seed: 0xe7e70004, drawn, clearance_ok: clr_ok};
      starts.forEach((st, i) => {
        const line = i % 10 < 6 ? EXT6[i % 10] : {family: 'random', seed: Math.floor(rnd() * 4294967296) >>> 0}, cv = i % 2 ? 1.0 : 0.9;
        const kind = line.family === 'const' ? (line.c0 > 0 ? 'const_left' : 'const_right') : line.family;
        add('LN-EXT60-R3', `E${i}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: S.test['LN-EXT60-R3'].seed_base + i,
          initial_clearance_m: st.initial_clearance_m, dpsi: st.dpsi}, params: {line, c_v: cv}, family: kind});
      });
      fam6();
    }
  } else if (set === 'fresh') {
    const {starts, drawn, clr_ok} = drawStarts(240, 0x5eed2c43), rnd = mulberry32(0x5eed2c44);
    meta['LN-F240-R3'] = {rng_seed: 0x5eed2c43, shape_seed: 0x5eed2c44, drawn, clearance_ok: clr_ok};
    starts.forEach((st, i) => {
      const kind = PLAN15[i % 15], line = lineFor(kind, rnd), cv = +(0.9 + 0.1 * rnd()).toFixed(4);
      add('LN-F240-R3', `F${i}`, {start: {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: 3330000 + i,
        initial_clearance_m: st.initial_clearance_m, dpsi: st.dpsi}, params: {line, c_v: cv}, family: kind});
    });
  } else throw new Error('eval_line_r3: 知らない組 ' + set);
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
    episodeMode: 'continuous', coreKind: ctx.core, overrides: noShield ? {shield: false} : {}, audit: true});
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
    dmin_first_3s: Math.min(...log.slice(0, 91).map(q => q.dmin)), epsi0: log[0]?.epsi ?? null});
  row.valid_r3 = row.valid_episode && !row.stall && row.valid_laps >= 1;
  // シールドが替えたフレームの替え方の内訳
  const fb = {};
  for (const q of log) if (q.sh === 1) { const k = q.shield?.fb ?? 'pass'; fb[k] = (fb[k] ?? 0) + 1; }
  row.shield_fallbacks = fb;
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
export function judge(set, rows) {
  const T = name => rows.filter(r => r.test === name);
  const J = {}, rep = {};
  if (set === 'test' || set === 'validation') {
    const isTest = set === 'test';
    const P = T(isTest ? 'LN-P60-R3' : 'LN-V60-R3'), H = T(isTest ? 'LN-HEAD60-R3' : 'LN-VHEAD40-R3'), E = isTest ? T('LN-EXT60-R3') : [], F = T('LN-FAM6');
    const nP = 60, nH = isTest ? 60 : 40, need = n => Math.ceil(57 / 60 * n);
    const okP = P.filter(r => r.valid_r3), okH = H.filter(r => r.valid_r3), okE = E.filter(r => r.valid_r3);
    J.L1_r3 = {value: {P: `${okP.length}/${P.length}`, HEAD: `${okH.length}/${H.length}`, ...(isTest ? {EXT: `${okE.length}/${E.length}`} : {})},
      pass: P.length === nP && okP.length >= need(nP) && H.length === nH && okH.length >= need(nH) && (!isTest || (E.length === 60 && okE.length >= need(60)))};
    const mc = minOrNull([...okP, ...okH, ...okE].map(r => r.min_clearance).concat(F.map(clr0)));
    J.L2_r3 = {value: mc, pass: geq(mc, .03)};
    J.L3_tracking = trackingJudge(okP);
    const fj = fam6Judge(F);
    J.L4_distinct = fj.L4;
    const vm3 = profSpread(okP, 'ey').mean_abs_dev;
    J.L5_vm3 = {value: vm3, pass: vm3 != null && vm3 >= .08};
    J.L6_lap = fj.L6;
    const fracP = shieldFrac(P), perRunMax = Math.max(0, ...[...P, ...F].map(r => r.shield_fraction ?? 0));
    J.L7_shield = {value: {P_fraction: fracP, max_per_run_P_FAM6: perRunMax}, pass: P.length === nP && fracP <= .02 && perRunMax <= .10};
    const all = [...P, ...H, ...E, ...F];
    const col = all.filter(r => r.invalid_spawn || r.collisions !== 0);
    J.L8_no_collision = {value: `${all.length - col.length}/${all.length}`, collided: col.map(r => `${r.test}:${r.id}`), pass: col.length === 0 && all.length === nP + nH + (isTest ? 60 : 0) + 6};
    // 選び方の値(検証の組の LN-VHEAD40-R3 の最小すき間。衝突した回は 0)
    rep.head_min_clearance_all = minOrNull(H.map(clr0));
    rep.head_stall_plus_collision = H.filter(r => r.stall || r.collisions > 0 || r.invalid_spawn).length;
    rep.stalls = all.filter(r => r.stall).map(r => ({test: r.test, id: r.id, gate: r.start.gate, dpsi: r.start.dpsi, first_t: r.stall_first_frame / 30, longest_s: r.stall_longest_frames / 30}));
    rep.not_valid_r3 = all.filter(r => !r.valid_r3).map(r => ({test: r.test, id: r.id, reason: r.reason, frames: r.frames, stall: r.stall, valid_laps: r.valid_laps,
      collision: r.collision, gate: r.start.gate, dpsi: r.start.dpsi, family: r.family}));
    rep.shield_gt5pct = all.filter(r => (r.shield_fraction ?? 0) > .05).map(r => ({test: r.test, id: r.id, fraction: r.shield_fraction, gate: r.start.gate, dpsi: r.start.dpsi}));
    rep.shield_fraction = {P: fracP, HEAD: shieldFrac(H), ...(isTest ? {EXT: shieldFrac(E)} : {}), FAM6: shieldFrac(F), FAM6_frames: F.reduce((a, r) => a + r.shield_frames, 0)};
    rep.shield_fallbacks = all.reduce((acc, r) => { for (const [k, v] of Object.entries(r.shield_fallbacks ?? {})) acc[k] = (acc[k] ?? 0) + v; return acc; }, {});
    const lh = all.filter(r => r.start.dpsi != null && Math.abs(r.start.dpsi) >= 1.0);
    rep.large_heading = {n: lh.length, by_set: Object.fromEntries([...new Set(lh.map(r => r.test))].map(t => [t, lh.filter(r => r.test === t).length])),
      valid_r3: lh.filter(r => r.valid_r3).length, min_clearance: minOrNull(lh.map(clr0)), min_clearance_first_3s_runtime: minOrNull(lh.map(r => r.dmin_first_3s)),
      below_5cm: lh.filter(r => clr0(r) < .05).map(r => ({test: r.test, id: r.id, min: clr0(r), t: r.min_clearance_t, gate: r.start.gate, dpsi: r.start.dpsi}))};
    rep.below_5cm = all.filter(r => clr0(r) < .05).map(r => ({test: r.test, id: r.id, min_clearance: clr0(r), t_s: r.min_clearance_t, valid_r3: r.valid_r3, gate: r.start.gate,
      dpsi: r.start.dpsi, family: r.family}));
    rep.min_clearance_by_set = Object.fromEntries([['P', P], ['HEAD', H], ...(isTest ? [['EXT', E]] : []), ['FAM6', F]].map(([k, rs]) => [k, {all: minOrNull(rs.map(clr0)),
      median: quantile(rs.map(clr0), .5), first_3s_runtime: minOrNull(rs.map(r => r.dmin_first_3s))}]));
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
  } else if (set === 'fresh') {
    const R = T('LN-F240-R3');
    rep.blocks = [0, 1, 2, 3].map(b => {
      const rs = R.slice(60 * b, 60 * b + 60), ok = rs.filter(r => r.valid_r3);
      return {block: b, L1: `${ok.length}/${rs.length}`, L1_pass: ok.length >= 57, L2_min_valid: minOrNull(ok.map(r => r.min_clearance)), L2_pass: geq(minOrNull(ok.map(r => r.min_clearance)), .03),
        L3: trackingJudge(ok).value, L3_pass: trackingJudge(ok).pass, L5_vm3: profSpread(ok, 'ey').mean_abs_dev, stalls: rs.filter(r => r.stall).length,
        collisions: rs.filter(r => r.collisions > 0).length, min_all: minOrNull(rs.map(clr0)), shield_fraction: shieldFrac(rs)};
    });
    rep.all = {n: R.length, valid_r3: validCount(R), min_all: minOrNull(R.map(clr0)), stalls: R.filter(r => r.stall).length, collisions: R.filter(r => r.collisions > 0).length,
      large_heading: R.filter(r => Math.abs(r.start.dpsi) >= 1).length, large_heading_min: minOrNull(R.filter(r => Math.abs(r.start.dpsi) >= 1).map(clr0)),
      below_5cm: R.filter(r => clr0(r) < .05).map(r => ({id: r.id, min: clr0(r), gate: r.start.gate, dpsi: r.start.dpsi})), shield_fraction: shieldFrac(R),
      shield_gt5pct: R.filter(r => r.shield_fraction > .05).map(r => ({id: r.id, fraction: r.shield_fraction, gate: r.start.gate, dpsi: r.start.dpsi}))};
  } else if (set === 'fallback') {
    const P = T('LN-P60-R3-FB'), ok = P.filter(r => r.valid_r3);
    J.L1_valid = {value: `${ok.length}/${P.length}`, valid_episode_protocol: `${P.filter(r => r.valid_episode).length}/${P.length}`, pass: P.length === 60 && ok.length >= 57};
    const mc = minOrNull(ok.map(r => r.min_clearance));
    J.L2_clearance = {value: mc, pass: geq(mc, .03)};
    const vm3 = profSpread(ok, 'ey').mean_abs_dev;
    J.L5_vm3 = {value: vm3, pass: vm3 != null && vm3 >= .08};
  } else if (set === 'fastest') {
    const P = T('LN-P60-R3-FAST');
    rep.fastest = {n: P.length, valid_r3: validCount(P), lap1_median: quantile(P.filter(r => r.valid_r3).map(r => r.laps[0]), .5), min_all: minOrNull(P.map(clr0)),
      vm3: profSpread(P.filter(r => r.valid_r3), 'ey').mean_abs_dev};
  }
  const judged = Object.values(J).filter(x => x.pass !== null);
  return {judged: J, all_judged_pass: judged.length > 0 && judged.every(x => x.pass === true), reported: rep};
}

// ---- 実行 ----
async function main() {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; }, flag = n => args.includes(n);
  const set = opt('--set', 'validation'), core = opt('--core', set === 'validation' ? 'fast' : 'original');
  const jobs = +opt('--jobs', '1'), only = opt('--only')?.split(','), out = opt('--out'), policyPath = opt('--policy');
  const noShield = flag('--no-shield');
  // 走行中のシールド(protocol_line_r3.json runtime.shield.params)。mode_runtime.mjs の試作は createModeTeacher のときにこれを読む
  globalThis.JF_R3 = noShield ? {} : {modeShield: {line: JSON.parse(JSON.stringify(SHIELD_PARAMS))}};
  if (!policyPath && set !== 'fallback' && set !== 'fastest') { console.error('使い方: node eval_line_r3.mjs --policy <json> --set validation|test|fresh|fallback|fastest'); process.exit(2); }
  const ctx = {core, noShield, keepPath: flag('--keep-path'), policy: policyPath ? policyOf(policyPath) : null};
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
    const tmp = fs.mkdtempSync(path.join(process.env.JF_TMP ?? '/tmp', 'eval_line_r3_'));
    await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
      const ch = fork(fileURLToPath(import.meta.url), [...args.filter((a, i) => !['--jobs', '--out'].includes(a) && !['--jobs', '--out'].includes(args[i - 1])), '--shard', `${k}/${jobs}`, '--rows-out', `${tmp}/${k}.json`], {stdio: 'inherit'});
      ch.on('exit', c => c === 0 ? res() : rej(new Error('子プロセスが失敗しました ' + c)));
    })));
    for (let k = 0; k < jobs; k++) rows.push(...JSON.parse(fs.readFileSync(`${tmp}/${k}.json`, 'utf8')));
    fs.rmSync(tmp, {recursive: true});
    rows.sort((a, b) => a._j - b._j);
  } else for (const [j, job] of J.entries()) { const r = await runJob(job, ctx); r._j = j; rows.push(r); if ((j + 1) % 20 === 0) console.error(`${j + 1}/${J.length}`); }
  const verdict = judge(set, rows);
  const shieldMjs = path.join(here, 'rt/shield.mjs'), modeRt = path.join(here, 'rt/mode_runtime.mjs');
  const res = {schema: 'jf-sim-rl-line-r3-eval/v1', created_at: new Date().toISOString(), set, core,
    policy: ctx.policy ? {path: ctx.policy.path, sha256: ctx.policy.sha, schema: ctx.policy.weights.schema, obs_dim: ctx.policy.weights.obs_dim, meta: ctx.policy.weights.meta ?? null} : null,
    shield: noShield ? null : {params: SHIELD_PARAMS, shield_mjs_sha256: sha256File(shieldMjs), mode_runtime_sha256: sha256File(modeRt)},
    protocol_line_r3_sha256: P3._sha256, protocol_line_r2_sha256: P2._sha256, set_meta: meta, runs: rows.length, ...verdict, wall_s: (Date.now() - t0) / 1000,
    tests: Object.fromEntries([...new Set(rows.map(r => r.test))].map(t => [t, rows.filter(r => r.test === t).map(r => { const {vm_frames, st_time, tracking_abs, tracking_s, _j, ...rest} = r; return rest; })]))};
  if (out) { fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true}); fs.writeFileSync(out, JSON.stringify(res)); }
  console.log(JSON.stringify({set, runs: rows.length, all_judged_pass: res.all_judged_pass, wall_s: res.wall_s,
    judged: Object.fromEntries(Object.entries(res.judged).map(([k, v]) => [k, {pass: v.pass, value: v.value}])),
    reported_head: res.reported.head_min_clearance_all, stalls: res.reported.stalls?.length}));
}
if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
