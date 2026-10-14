#!/usr/bin/env node
// モードごとの合否の試験(protocol.json test_sets・acceptance のとおり。2026-10-13、sim_rl_modes の実装の段)
//
//   node rl/eval_modes.mjs --mode <fastest|no_wall|recover|line|speed|obstacle> --policy <policy.json> [--set test|validation]
//        [--core original|fast] [--jobs N] [--only 試験,…] [--out 結果.json]
//        [--shield-c 0.09](no_wall の代わり)[--classes 弱,中](recover の代わり)[--speed-caps](speed の代わり: 学習しない上限)
//        [--fid 出力.json](F-ID: 600 フレームの snapshot と Node の操作。ブラウザの確認に使う)
//        [--protocol speed_r2](速さいろいろ第 2 回の追補の決まり protocol_speed_r2.json の組と S2。--set test|validation|fresh。2026-10-14)
//        [--protocol line_r2](経路いろいろ第 2 回の追補の決まり protocol_line_r2.json の組。--mode line、--set test|validation|fresh。2026-10-14)
// 判定は元の sim-core.js(--core original、既定)。例外: 摩擦円(FC)は高速版の写し、状態のキック(RC-KICK)は高速版。
// 検証の組(--set validation)は、種 validation_base・Halton 10,001〜・回の数は試験の 1/4(切り上げ)。高速版で回してよい(チェックポイントを選ぶため)。
// --jobs N: 回を N 個の子プロセスに分けて走らせる(結果は同じ。並べ方は決まっている)。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD} from './episode.mjs';
import {MODE_NUMBER, lineParamsFor, SPEED_LEVELS} from './runtime/mode_runtime.mjs';
import {CATALOGUE} from './runtime/catalogue.mjs';
import {haltonStart, createRng, mix32} from './runtime/rng.mjs';
import {prepareGeometry, clearance, projectWindow, LOOP_LENGTH} from './runtime/features.mjs';
import {activeWorldFor, PHYSICS} from '../snapshot/rl/env_core.mjs';
import {loadRec60} from '../snapshot/rl/eval_sets.mjs';
import {perturbAllowed, RECORD_SPEC, maskFromLearn} from './runtime/perturb.mjs';
import {worldWithObstacles, geometryWithObstacles, startClearOfObstacles} from './runtime/layouts.mjs';
import {computeNominalLine} from './build_nominal_line.mjs';
import {binProfile, spread, quantile, nominalAt, VM1, VM2, VM3, VM4, VM5, VM7, sceneFromEpisode, steerUs} from './variation_metrics.mjs';
import {sampleDrSpec} from './env_modes.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const PROTOCOL_PATH = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/protocol.json';
const PROTOCOL = JSON.parse(fs.readFileSync(PROTOCOL_PATH, 'utf8'));
// 速さいろいろ第 2 回の追補(protocol.json は変えない)。読むときに protocol_speed_r2.sha256 と照合する(違えば止まる)
export const PROTOCOL_SPEED_R2_PATH = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/protocol_speed_r2.json';
export function loadSpeedR2() {
  const raw = fs.readFileSync(PROTOCOL_SPEED_R2_PATH), sha = crypto.createHash('sha256').update(raw).digest('hex');
  const want = fs.readFileSync(PROTOCOL_SPEED_R2_PATH.replace(/\.json$/, '.sha256'), 'utf8').trim().split(/\s+/)[0];
  if (sha !== want) throw new Error(`eval_modes: protocol_speed_r2.json の sha256 ${sha} が固めた値 ${want} と違います`);
  return {...JSON.parse(raw), _sha256: sha};
}
// 経路いろいろ第 2 回の追補(protocol.json は変えない)。読むときに protocol_line_r2.sha256 と照合する(違えば止まる)
export const PROTOCOL_LINE_R2_PATH = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/protocol_line_r2.json';
export function loadLineR2() {
  const raw = fs.readFileSync(PROTOCOL_LINE_R2_PATH), sha = crypto.createHash('sha256').update(raw).digest('hex');
  const want = fs.readFileSync(PROTOCOL_LINE_R2_PATH.replace(/\.json$/, '.sha256'), 'utf8').trim().split(/\s+/)[0];
  if (sha !== want) throw new Error(`eval_modes: protocol_line_r2.json の sha256 ${sha} が固めた値 ${want} と違います`);
  return {...JSON.parse(raw), _sha256: sha};
}
const DRT = CATALOGUE.common.dr_specs['DR-T'];
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const modeByKey = k => CATALOGUE.modes.find(m => m.key === k);

// ---- 種・開始 ----
export const seedBase = (set, mode) => (set === 'test' ? 900000 : 800000) + 1000 * MODE_NUMBER[mode];
const baseGeom = prepareGeometry(BASE_WORLD);
// Halton の開始(protocol seeds.halton): 重なる・開始の条件を満たさない番号は飛ばす
// from・dist: 速さいろいろ第 2 回(protocol_speed_r2.json)の番号の始まりと開始の分布(開始のすき間の下限つき)。無ければ今までと同じ
export function haltonStarts(mode, set, count, {skip = 0, layoutFor = null, distKey = mode, from = null, dist: distOverride = null} = {}) {
  const dist = distOverride ?? modeByKey(distKey).recording.start_distribution;
  const out = [];
  let i = from ?? (set === 'test' ? 20001 : 10001), valid = 0;
  for (; out.length < count && i < (set === 'test' ? 30000 : 20000); i++) {
    const st = haltonStart(i, dist);
    const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
    const lay = layoutFor ? layoutFor(out.length) : null;
    const g = lay?.obstacles?.length ? geometryWithObstacles(baseGeom, lay.obstacles) : baseGeom;
    const c = clearance(g, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5);
    if (!(c > 1e-9)) continue;
    if (dist.min_initial_clearance_m && c < dist.min_initial_clearance_m) continue;
    if (distOverride) st.initial_clearance_m = c;
    if (lay && !startClearOfObstacles(projectWindow(world.spawn.x, world.spawn.z, 0, LOOP_LENGTH).s, lay.obstacles)) continue;
    if (valid++ < skip) continue;
    out.push(st);
  }
  return out;
}
const q4 = (n, set) => set === 'test' ? n : Math.ceil(n / 4);
// 速さいろいろ第 2 回の報告用の「新しい開始」(Halton を使わない一様乱数。protocol_speed_r2.json sets.fresh)
export function freshStarts(spec, dist) {
  const out = [];
  for (let i = 0; out.length < spec.count && i < 100 * spec.count; i++) {
    const rng = createRng(mix32(spec.rng_seed, i));
    const st = {gate: Math.min(32, Math.floor(rng.uniform() * 33)), left_offset_m: dist.left_offset_m[0] + (dist.left_offset_m[1] - dist.left_offset_m[0]) * rng.uniform(),
      left_yaw_deg: dist.left_yaw_deg[0] + (dist.left_yaw_deg[1] - dist.left_yaw_deg[0]) * rng.uniform(), fresh_index: i};
    const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
    const c = clearance(baseGeom, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5);
    if (!(c > 1e-9) || c < (dist.min_initial_clearance_m ?? 0)) continue;
    st.initial_clearance_m = c;
    out.push(st);
  }
  return out;
}
// r2 の条件の表(rl/r2/robust_eval.mjs の GRID の写し。40 行)。in_range は DR-T(protocol definitions.in_range)
export function gridRows() {
  const G = [{name: '標準', cond: {}}, {name: '遅延 1', cond: {delay: 1}}, {name: '遅延 1 + 前進ゲイン 1.01(ONNX の条件)', cond: {delay: 1, speedGain: 1.01}},
    ...[.8, .9, 1.0, 1.1, 1.3, 1.5, 1.6, 1.8].map(v => ({name: `舵の速さ ${v} rad/s`, cond: {physics: {steerRate: v}}})),
    ...[.6, .7, .8, .9, 1.0, 1.1, 1.5, 2.0, 2.5].map(v => ({name: `ブレーキ ${v} m/s²`, cond: {physics: {brakeDeceleration: v}}})),
    ...[.8, .85, .9, .95, 1.05, 1.1].map(v => ({name: `μ × ${v}`, cond: {physics: {frictionMultiplier: v}}})),
    ...[.25, .3, .5, .6].map(v => ({name: `時定数 ${v} 秒`, cond: {physics: {driveTau: v}}})),
    ...[3, 4].map(v => ({name: `加速の上限 ${v} m/s²`, cond: {physics: {driveAccelerationCap: v}}})),
    {name: '舵のゲイン 0.95', cond: {steerGain: .95}}, {name: '舵のゲイン 1.05', cond: {steerGain: 1.05}},
    {name: '前進ゲイン 0.97', cond: {speedGain: .97}}, {name: '前進ゲイン 1.03', cond: {speedGain: 1.03}},
    {name: '組み合わせ: 舵の速さ 1.0 + ブレーキ 1.0', cond: {physics: {steerRate: 1.0, brakeDeceleration: 1.0}}},
    {name: '組み合わせ: 舵の速さ 1.0 + 遅延 1', cond: {physics: {steerRate: 1.0}, delay: 1}},
    {name: '組み合わせ: μ × 0.9 + ブレーキ 1.0 + 遅延 1', cond: {physics: {frictionMultiplier: .9, brakeDeceleration: 1.0}, delay: 1}},
    {name: '組み合わせ: 舵 0.9・ブレーキ 0.8・μ 0.9・時定数 0.5・遅延 1(悪い側をすべて)', cond: {physics: {steerRate: .9, brakeDeceleration: .8, frictionMultiplier: .9, driveTau: .5}, delay: 1}}];
  const inR = (k, v) => v >= DRT[k][0] - 1e-12 && v <= DRT[k][1] + 1e-12;
  return G.map(r => {
    const c = r.cond;
    let ok = (c.delay ?? 0) <= 1;
    for (const [k, v] of Object.entries(c.physics ?? {})) ok = ok && inR(k, v);
    if (c.steerGain != null) ok = ok && inR('steerGain', c.steerGain);
    if (c.speedGain != null) ok = ok && inR('speedGain', c.speedGain);
    return {...r, in_range: ok};
  });
}
const drFromCond = c => ({physics: c.physics ?? {}, delay: c.delay ?? 0, steerGain: c.steerGain ?? 1, speedGain: c.speedGain ?? 1, frictionCircle: !!c.frictionCircle});

// ---- 試験の回の一覧(決まった順) ----
export function buildJobs(mode, set, opt = {}) {
  const J = [], base = seedBase(set, mode), rec60 = loadRec60(), test = set === 'test';
  const recStart = c => ({gate: c.gate, left_offset_m: c.left_offset_m, left_yaw_deg: c.left_yaw_deg, seed: c.seed, scenario: c.scenario === 'noise' ? 'baseline' : c.scenario});
  const add = (test, id, spec) => J.push({test, id, spec});
  if (mode === 'fastest') {
    add('F-STD', 'single', {start: STD, episodeMode: 'single'});
    add('F-STD', 'continuous', {start: STD});
    rec60.slice(0, q4(60, set)).forEach((c, i) => add('F-REC60', c.name, {start: recStart(c)}));
  } else if (mode === 'no_wall') {
    const recIdx = test ? rec60.map((_, i) => i).filter(i => i % 2 === 0) : rec60.map((_, i) => i).filter(i => i % 2 === 1).slice(0, 8);
    const hN = test ? 30 : 7, hs = haltonStarts(mode, set, hN + q4(200, set));
    for (const i of recIdx) add('NW-N60', rec60[i].name, {start: recStart(rec60[i])});
    hs.slice(0, hN).forEach((st, i) => add('NW-N60', `H${st.halton_index}`, {start: {...st, seed: base + i}}));
    add('NW-STD', 'single', {start: STD, episodeMode: 'single'});
    add('NW-STD', 'continuous', {start: STD});
    const rows = gridRows();
    const inr = rows.map((r, i) => ({r, i})).filter(x => x.r.in_range), sel = test ? inr : inr.filter((_, k) => k % 4 === 0);
    for (const {r, i} of sel) for (const em of ['single', 'continuous']) add('NW-G', `G${String(i).padStart(2, '0')}-${em}`, {start: STD, episodeMode: em, dr: drFromCond(r.cond), name: r.name});
    if (test) for (const [i, r] of rows.entries()) if (!r.in_range) for (const em of ['single', 'continuous']) add('NW-G-OUT', `G${String(i).padStart(2, '0')}-${em}`, {start: STD, episodeMode: em, dr: drFromCond(r.cond), name: r.name});
    hs.slice(hN).forEach((st, i) => {
      const s = base + 100 + i, d = sampleDrSpec(createRng(mix32(s)), DRT);
      add('NW-R200', `R${i}`, {start: {...st, seed: s}, dr: {...d, steerNoise: 0}});
    });
    const fcIdx = test ? [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] : [11, 13, 15];
    for (const i of fcIdx) add('NW-FC10', rec60[i].name, {start: recStart(rec60[i]), dr: {physics: {}, delay: 0, frictionCircle: true}});
    if (test) {
      for (const j of J.filter(x => x.test === 'NW-N60' || x.test === 'NW-R200')) add('NW-OFF', `${j.test}:${j.id}`, {...j.spec, overrides: {shield: false}});
    }
  } else if (mode === 'recover') {
    const n = q4(100, set), hs = haltonStarts(mode, set, n);
    hs.forEach((st, i) => add('RC-P100', `P${i}`, {start: {...st, seed: base + i}, params: {}}));
    if (test || true) {
      const times = Array.from({length: 37}, (_, i) => 1 + .5 * i).filter((_, i) => test || i % 4 === 0);
      for (const t of times) for (const sign of [1, -1]) for (const kind of ['yaw', 'lateral'])
        add('RC-KICK', `${kind}${sign > 0 ? '+' : '-'}${t.toFixed(1)}`, {start: STD, kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)}, overrides: {perturbation: false}, kind});
    }
    add('RC-STD', 'single', {start: STD, episodeMode: 'single', overrides: {perturbation: false}});
    rec60.slice(0, q4(60, set)).forEach(c => add('RC-REC60', c.name, {start: recStart(c), overrides: {perturbation: false}}));
  } else if (mode === 'line') {
    const plan = Object.entries({const_left: 2, const_right: 2, const_center: 1, inside: 2, outside: 2, late_apex: 2, early_apex: 1, random: 3}).flatMap(([k, w]) => Array(w).fill(k));
    // 第 2 回(--protocol line_r2): 開始のすき間 ≥ 0.12 m の分布、組ごとの Halton の始まり・回の数・種(protocol_line_r2.json sets)。形・c_v の作り方は同じ
    const R2 = opt.protocolLineR2, spec = R2 ? R2.sets[set] : null;
    if (R2 && !spec) throw new Error('eval_modes: protocol_line_r2 に組 ' + set + ' がありません');
    const n = spec ? spec.count : q4(60, set);
    const hs = !R2 ? haltonStarts(mode, set, n) : set === 'fresh' ? freshStarts(spec, R2.start_distribution)
      : haltonStarts(mode, set, n, {from: spec.halton_from, dist: R2.start_distribution});
    if (hs.length !== n) throw new Error(`eval_modes: 開始が ${hs.length} 個しか作れません(要 ${n})`);
    const seedOf = i => spec ? spec.seed_base + i : base + i;
    hs.forEach((st, i) => {
      const rng = createRng(mix32(seedOf(i), 0x11e));
      add('LN-P60', `P${i}`, {start: {...st, seed: seedOf(i)}, params: {line: lineParamsFor(plan[i % 15], rng), c_v: +(0.9 + 0.1 * rng.uniform()).toFixed(4)}, family: plan[i % 15]});
    });
    if (!(R2 && set === 'fresh')) for (const [nm, line] of [['c-0.8', {family: 'const', c0: -.8}], ['c0', {family: 'const', c0: 0}], ['c+0.8', {family: 'const', c0: .8}],
      ['inside', {family: 'inside', a: .8}], ['outside', {family: 'outside', a: .8}], ['late_apex', {family: 'late_apex', a: .8}]])
      add('LN-FAM6', nm, {start: STD, params: {line, c_v: 1.0}});
  } else if (mode === 'speed') {
    const caps = opt.speedCaps ? [1.0, 4.0 / 4.5, 3.5 / 4.5, 3.0 / 4.5, 2.5 / 4.5] : SPEED_LEVELS;
    // 第 2 回(--protocol speed_r2): 開始のすき間 ≥ 0.12 m の分布、組ごとの Halton の始まり・回の数・種(protocol_speed_r2.json sets)
    const R2 = opt.protocolR2, spec = R2 ? R2.sets[set] : null;
    if (R2 && !spec) throw new Error('eval_modes: protocol_speed_r2 に組 ' + set + ' がありません');
    const n = spec ? spec.count : q4(60, set);
    const hs = !R2 ? haltonStarts(mode, set, n) : set === 'fresh' ? freshStarts(spec, R2.start_distribution)
      : haltonStarts(mode, set, n, {from: spec.halton_from, dist: R2.start_distribution});
    if (hs.length !== n) throw new Error(`eval_modes: 開始が ${hs.length} 個しか作れません(要 ${n})`);
    const seedOf = i => spec ? spec.seed_base + i : base + i;
    hs.forEach((st, i) => { const cv = caps[i % 5]; add('SP-P60', `P${i}`, {start: {...st, seed: seedOf(i)}, params: opt.speedCaps ? {speed_limit: 4.5 * cv} : {c_v: cv}, c_v: cv}); });
    for (const cv of [1.0, .75, .5]) { const c = opt.speedCaps ? {1: 1, .75: 3.5 / 4.5, .5: 2.5 / 4.5}[cv] : cv;
      add('SP-STD3', `cv${cv}`, {start: STD, episodeMode: 'single', params: opt.speedCaps ? {speed_limit: 4.5 * c} : {c_v: c}, c_v: c}); }
  } else if (mode === 'obstacle') {
    const lays = readJson(path.join(here, `obstacle_sets/${set === 'test' ? 'test' : 'validation'}.json`)).layouts.slice(0, set === 'test' ? 50 : 13);
    const hs = haltonStarts('obstacle', set, lays.length, {layoutFor: k => lays[k]});
    lays.forEach((lay, i) => {
      add('OB-T100', `${lay.id}-STD`, {start: {...STD, seed: base + 2 * i}, layout: lay});
      add('OB-T100', `${lay.id}-H`, {start: {...hs[i], seed: base + 2 * i + 1}, layout: lay});
    });
    add('OB-EMPTY', 'single', {start: STD, episodeMode: 'single'});
  }
  return J;
}

// ---- 1 回 → 結果の行(親で判定できる小さな形) ----
let nominalCache = null;
async function runJob(job, ctx) {
  const {mode, weights, sha, core, opt} = ctx, sp = job.spec;
  const t0 = Date.now();
  const params = {...(sp.params ?? {})};
  if (job.test === 'RC-P100' && opt.classes) params.perturb_class = opt.classes.split(',');
  let W = weights, S = sha;
  if (ctx.policyList && job.test === 'LN-P60') { const k = +job.id.slice(1) % ctx.policyList.length; W = ctx.policyList[k].weights; S = ctx.policyList[k].sha; }
  const r = await runModeEpisode({mode: sp.modeOverride ?? mode, weights: W, policySha256: S, start: sp.start, params, episodeMode: sp.episodeMode ?? 'continuous',
    coreKind: job.test === 'RC-KICK' ? 'fast' : core, dr: sp.dr ?? null, kick: sp.kick ?? null, layout: sp.layout ?? null,
    shieldParams: opt.shieldC ? {c_nominal: opt.shieldC} : {}, overrides: {...(sp.overrides ?? {}), ...(ctx.overrides ?? {})}, audit: job.test !== 'RC-KICK'});
  const row = {test: job.test, id: job.id, start: sp.start, dr: sp.dr ?? null, name: sp.name ?? null, family: sp.family ?? null, c_v: sp.c_v ?? r.params?.c_v ?? null,
    layout_id: sp.layout?.id ?? null, invalid_spawn: !!r.invalid_spawn, wall_s: 0};
  if (r.invalid_spawn) return row;
  Object.assign(row, {reason: r.reason, frames: r.frames, collisions: r.collisions, collision: r.collision, laps: r.laps.map(l => l.elapsed), valid_laps: r.valid_laps,
    valid_episode: r.valid_episode, lap_complete: r.lap_complete ?? null, lap_from_stop: r.lap_from_stop ?? null, min_clearance: r.min_clearance ?? null,
    audit_err: r.audit?.max_pos_err ?? null, shield_frames: r.shield_frames, rho: r.rho, params: r.params,
    dmin_frame_min: Math.min(...r.log.map(q => q.dmin)),
    // 本当のゲートの外し(rl/gate_miss.mjs。2026-10-14 recover r2)と、参考のランタイムの仮想の外し
    gate_miss: r.gate_miss, gate_misses: r.gate_misses, gate_stuck_max_m: r.gate_stuck_max_m, gate_misses_virtual: r.gate_misses_virtual});
  const log = r.log;
  // 走っているフレームの速さ
  const run = log.filter(q => q.v > .3).map(q => q.v);
  row.speed = {mean_running: mean(run), p95: quantile(run, .95)};
  // 経路・速さの物差し用の s のマス(開始から 3 m 以降の最初の通過)
  if (['LN-P60', 'LN-FAM6', 'SP-P60', 'OB-T100', 'F-REC60', 'NW-N60', 'RC-P100'].includes(job.test)) {
    const sc = sceneFromEpisode(r);
    const rnd = (a, k) => Array.from(a, v => Number.isNaN(v) ? null : +v.toFixed(k));
    row.prof = {ey: rnd(binProfile(sc, 'ey'), 4), v: rnd(binProfile(sc, 'v'), 3), st: rnd(binProfile(sc, q => steerUs(q.st / .52)), 2)};
    row.vm_frames = sc.frames.filter((q, i) => i % 3 === 0 && q.s - sc.s0 >= 3).map(q => [+q.s.toFixed(3), +q.ey.toFixed(4), +q.epsi.toFixed(4), +q.v.toFixed(3)]);
    row.st_time = log.map(q => +q.st.toFixed(5));
  }
  if (r.log[0]?.es !== undefined) {
    const s0 = log[0].s, te = log.filter(q => q.s - s0 >= 3 && q.es != null).map(q => Math.abs(q.ey - q.es));
    row.tracking = {n: te.length, median: quantile(te, .5), p95: quantile(te, .95), mean: mean(te)};
    row.tracking_abs = te.map(x => +x.toFixed(4)).filter((_, i) => i % 2 === 0);
    // 経路いろいろ第 2 回の報告(protocol_line_r2.json acceptance_changes.reported): 追従の誤差のフレームの R 上の s(tracking_abs と同じ間引き)と、
    // 開始から 1.5 秒(45 フレーム)の最小すき間・回の最小すき間のフレームの時刻
    if (ctx.opt.protocolLineR2 && job.test.startsWith('LN-')) {
      row.tracking_s = log.filter(q => q.s - s0 >= 3 && q.es != null).map(q => +(((q.s % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH).toFixed(2)).filter((_, i) => i % 2 === 0);
      row.dmin_first_1p5s = Math.min(...log.slice(0, 46).map(q => q.dmin));
      let kMin = 0; for (let k = 1; k < log.length; k++) if (log[k].dmin < log[kMin].dmin) kMin = k;
      row.dmin_frame_min_t = kMin / 30;
      row.line_profile = r.params?.line_profile ?? 'v1';
    }
  }
  if (job.test === 'RC-KICK') {
    const kf = sp.kick.frame, q = log[kf] ?? log.at(-1);
    row.kick_s = q?.s ?? null;
    row.kick_allowed = q ? perturbAllowed(q.s, RECORD_SPEC) : false;
    row.survived = r.reason === 'collection_duration_complete' && r.collisions === 0;
    row.kind = sp.kind;
  }
  if (job.test === 'RC-P100') {
    // 学習から外すフレーム(mask_rule)と出来事ごとの立て直し(protocol definitions.recovered・recovery_success)
    const learn = log.map(q => q.l), mask = maskFromLearn(learn);
    row.mask_frac = mask.reduce((a, b) => a + b, 0) / 600;
    row.events = r.events;
    if (!nominalCache) nominalCache = await computeNominalLine({mode, weights, policySha256: sha, coreKind: core});
    const nl = nominalCache;
    const T = log.length, R = PROTOCOL.definitions;
    // 停止・逆走の印(env と同じ決まり: 1 秒以降に 0.2 m/s 未満が 60 フレーム、s が最大から 1 m 戻る)
    const stallAt = [], wrongAt = [];
    let sf = 0, smax = -Infinity;
    for (let f = 0; f < T; f++) {
      const q = log[f];
      sf = f / 30 >= 1 && q.v < .2 ? sf + 1 : 0; stallAt.push(sf >= 60);
      smax = Math.max(smax, q.s); wrongAt.push(q.s < smax - 1);
    }
    const collF = r.collisions ? r.frames : Infinity;
    row.recovery = r.events.map((ev, k) => {
      const pe = ev.pe ?? T, next = r.events[k + 1]?.ps ?? T;
      let rec = null, runLen = 0;
      for (let f = pe; f < T; f++) {
        const q = log[f], nm = nominalAt(nl, q.s);
        const ok = Math.abs(q.ey - nm.ey) <= .10 && Math.abs(Math.atan2(Math.sin(q.epsi - nm.psi), Math.cos(q.epsi - nm.psi))) <= .15;
        runLen = ok ? runLen + 1 : 0;
        if (runLen >= 15) { rec = f - 14; break; }
      }
      const until = Math.min(next, T);
      let bad = collF <= until;
      for (let f = pe; f < until && !bad; f++) if (stallAt[f] || wrongAt[f]) bad = true;
      const recS = rec == null ? null : (rec - pe) / 30;
      return {n: ev.n, type: ev.type, cls: ev.cls, ps: ev.ps, pe: ev.pe, aborted: ev.aborted, recovered_s: recS, success: recS != null && recS <= 3.0 && !bad, bad_after: bad};
    });
  }
  row.wall_s = (Date.now() - t0) / 1000;
  return row;
}

// ---- 判定 ----
const allOf = rows => rows.every(Boolean);
// 有効な回が 1 つも無いときは null(不合格)。ある時は最小
const minOrNull = a => a.length ? Math.min(...a) : null;
const geq = (v, t) => v != null && v >= t;
function judge(mode, rows, set, opt = {}) {
  const T = name => rows.filter(r => r.test === name);
  const J = {}, rep = {};
  const vmScenes = rs => rs.filter(r => r.prof).map(r => ({frames: [], prof: r.prof}));
  const profSpread = (rs, key) => spread(rs.filter(r => r.prof).map(r => Float64Array.from(r.prof[key], v => v == null ? NaN : v)));
  if (mode === 'fastest') {
    const single = T('F-STD').find(r => r.id === 'single'), cont = T('F-STD').find(r => r.id === 'continuous'), rec = T('F-REC60');
    J.F2_lap = {value: single?.lap_from_stop, reference: 8.091953367107205, pass: single?.lap_from_stop != null && Math.abs(single.lap_from_stop - 8.091953367107205) <= 1e-6,
      note: '7c43cb53… の値(ベストラップの確認 r1)。別の方策なら、その作業の報告の値と比べる'};
    J.F3_rec60 = {value: `${rec.filter(r => r.valid_episode).length}/${rec.length}`, pass: rec.length === q4(60, set) && rec.every(r => r.valid_episode)};
    J.F4_identity = {pass: null, note: 'ブラウザ(8080)の確認で判定する。--fid で Node 側の列を作る'};
    rep.flying_lap = cont?.laps?.[1] ?? null;
    rep.rec60_min_clearance = minOrNull(rec.map(r => r.min_clearance ?? 0));
  } else if (mode === 'no_wall') {
    const n60 = T('NW-N60'), std = T('NW-STD'), g = T('NW-G'), r200 = T('NW-R200'), fc = T('NW-FC10');
    const all = [...n60, ...std, ...g, ...r200, ...fc];
    const nTarget = set === 'test' ? 340 : all.length;
    J.N1_no_collision = {value: `${all.filter(r => r.collisions === 0 && !r.invalid_spawn).length}/${all.length}`, runs: all.length, pass: all.length === nTarget && all.every(r => r.collisions === 0 && !r.invalid_spawn)};
    const minN = minOrNull([...n60, ...std].map(r => r.min_clearance ?? 0)), minR = minOrNull([...g, ...r200, ...fc].map(r => r.min_clearance ?? 0));
    J.N2_clearance_nominal = {value: minN, pass: geq(minN, .08)};
    J.N3_clearance_randomised = {value: minR, pass: geq(minR, .05)};
    const s1 = std.find(r => r.id === 'single');
    J.N4_lap_from_stop = {value: s1?.lap_from_stop ?? null, pass: s1?.lap_complete === true && s1.lap_from_stop <= 12.0};
    const sc = std.find(r => r.id === 'continuous');
    const okLap = r => r.valid_episode && r.valid_laps >= 1;
    J.N5_completion = {value: {n60: `${n60.filter(okLap).length}/${n60.length}`, std: sc ? okLap(sc) : null, r200: `${r200.filter(okLap).length}/${r200.length}`},
      pass: n60.every(okLap) && !!sc && okLap(sc) && r200.filter(okLap).length >= Math.ceil(.95 * r200.length)};
    const rq = n60.reduce((a, r) => a + (r.rho?.req_gt1 ?? 0), 0) / Math.max(1, n60.reduce((a, r) => a + (r.rho?.n ?? 0), 0));
    const rc = n60.reduce((a, r) => a + (r.rho?.comb_gt1 ?? 0), 0) / Math.max(1, n60.reduce((a, r) => a + (r.rho?.comb_n ?? 0), 0));
    J.N6_friction = {value: {rho_req_gt1: rq, rho_comb_gt1: rc}, pass: rq <= .02 && rc <= .02};
    const shf = n60.reduce((a, r) => a + r.shield_frames, 0) / Math.max(1, n60.reduce((a, r) => a + r.frames, 0));
    J.N7_shield = {value: shf, pass: shf <= .05};
    const off = T('NW-OFF');
    rep.NW_OFF = off.length ? {runs: off.length, collisions: off.filter(r => r.collisions).length, min_clearance: Math.min(...off.map(r => r.min_clearance ?? Infinity))} : null;
    rep.R200_shield_fraction = r200.reduce((a, r) => a + r.shield_frames, 0) / Math.max(1, r200.reduce((a, r) => a + r.frames, 0));
    rep.NW_G_OUT = T('NW-G-OUT').map(r => ({id: r.id, name: r.name, reason: r.reason, collisions: r.collisions, min_clearance: r.min_clearance}));
    rep.lap1 = {n60: n60.map(r => r.laps[0]).filter(Number.isFinite)};
    // 画面の文言(protocol acceptance.no_wall.ui_numbers: 衝突は「N 回の試験で衝突 0」、最小すき間は N2 と N3 の両方)。合否(「不合格(参考)」)は publish_mode.py が足す
    // (2026-10-14 no_wall r2 で直した。r1 は N3 の値だけだった = 確認 r1 の指摘)
    const nCol = all.filter(r => r.collisions !== 0 || r.invalid_spawn).length, cm = v => v == null ? '—' : (v * 100).toFixed(1);
    rep.ui_text = `${all.length} 回の試験で衝突 ${nCol}・最小すき間 ${cm(minN)} cm(標準の物理)/ ${cm(minR)} cm(物理のずれ)`;
    rep.ui_numbers = {runs: all.length, collisions: nCol, min_clearance_nominal_m: minN, min_clearance_randomised_m: minR};
  } else if (mode === 'recover') {
    const p = T('RC-P100'), kick = T('RC-KICK'), std = T('RC-STD')[0], rec = T('RC-REC60');
    const nP = q4(100, set);
    J.C1_valid = {value: `${p.filter(r => r.valid_episode).length}/${p.length}`, pass: p.filter(r => r.valid_episode).length >= Math.ceil(.9 * nP)};
    const ev = p.flatMap(r => r.recovery ?? []);
    const succ = ev.filter(e => e.success).length / Math.max(1, ev.length);
    J.C2_success = {value: succ, events: ev.length, pass: ev.length > 0 && succ >= .95};
    const rt = ev.filter(e => e.success).map(e => e.recovered_s);
    J.C3_recovery_time = {value: {median: quantile(rt, .5), p90: quantile(rt, .9)}, pass: rt.length > 0 && quantile(rt, .5) <= 1.2 && quantile(rt, .9) <= 2.5};
    const epm = mean(p.map(r => r.events?.length ?? 0)), mf = mean(p.map(r => r.mask_frac ?? 0));
    J.C4_events_mask = {value: {events_per_episode: epm, masked_fraction: mf}, pass: epm >= 3.5 && mf >= .10 && mf <= .40};
    const ky = kick.filter(r => r.kick_allowed && r.kind === 'yaw'), kl = kick.filter(r => r.kick_allowed && r.kind === 'lateral');
    const sy = ky.filter(r => r.survived).length / Math.max(1, ky.length), sl = kl.filter(r => r.survived).length / Math.max(1, kl.length);
    J.C5_kick = {value: {yaw: `${ky.filter(r => r.survived).length}/${ky.length}`, lateral: `${kl.filter(r => r.survived).length}/${kl.length}`, yaw_rate: sy, lateral_rate: sl},
      pass: ky.length > 0 && kl.length > 0 && sy >= .9 && sl >= .9};
    const recMin = minOrNull(rec.map(r => r.min_clearance ?? 0));
    J.C6_unperturbed = {value: {lap_from_stop: std?.lap_from_stop ?? null, rec60: `${rec.filter(r => r.valid_episode).length}/${rec.length}`, rec60_min_clearance: recMin},
      pass: std?.lap_complete === true && std.lap_from_stop <= 9.0 && rec.length > 0 && rec.every(r => r.valid_episode) && geq(recMin, .03)};
    const by = k => Object.fromEntries([...new Set(ev.map(e => e[k]))].map(v => [v, {n: ev.filter(e => e[k] === v).length, success: ev.filter(e => e[k] === v && e.success).length}]));
    rep.by_class = by('cls'); rep.by_type = by('type');
    // ゲートの外し(protocol: 成功の条件に入れず別に数える)。r1 は「valid_laps < laps.length」で数えていたが、laps は sim が数えた周だけなので
    // いつも 0 だった(確認 recover r1 の指摘)。r2 から sim の数えが止まったまま s が進んだ回を数える(rl/gate_miss.mjs)
    const pv = p.filter(r => r.valid_episode), ks = kick.filter(r => r.survived);
    rep.gate_miss_episodes = pv.filter(r => r.gate_miss).length;
    rep.gate_miss = {rule: 'sim の passed_gates が変わらないまま R 上の s がゲートの間の最大 1.519 m + 0.5 m を越えて進んだ(rl/gate_miss.mjs)',
      p100_valid: `${rep.gate_miss_episodes}/${pv.length}`, p100_valid_ids: pv.filter(r => r.gate_miss).map(r => r.id),
      p100_all: `${p.filter(r => r.gate_miss).length}/${p.length}`,
      kick_survived: `${ks.filter(r => r.gate_miss).length}/${ks.length}`,
      kick_allowed_survived: {yaw: `${ks.filter(r => r.kick_allowed && r.kind === 'yaw' && r.gate_miss).length}/${ks.filter(r => r.kick_allowed && r.kind === 'yaw').length}`,
        lateral: `${ks.filter(r => r.kick_allowed && r.kind === 'lateral' && r.gate_miss).length}/${ks.filter(r => r.kick_allowed && r.kind === 'lateral').length}`},
      rec60: `${rec.filter(r => r.gate_miss).length}/${rec.length}`,
      virtual_counter_episodes_valid: `${pv.filter(r => (r.gate_misses_virtual ?? 0) > 0).length}/${pv.length}`,
      virtual_counter_note: 'ランタイムの仮想の外し(wrapper.misses)は面を越えるのが遅れただけでも数えるので、外しの数には使わない(参考)'};
    rep.kick_all_positions = {yaw: `${kick.filter(r => r.kind === 'yaw' && r.survived).length}/${kick.filter(r => r.kind === 'yaw').length}`,
      lateral: `${kick.filter(r => r.kind === 'lateral' && r.survived).length}/${kick.filter(r => r.kind === 'lateral').length}`};
  } else if (mode === 'line') {
    const p = T('LN-P60'), fam = T('LN-FAM6'), ok = p.filter(r => r.valid_episode);
    const RL2 = opt.protocolLineR2, nWantL = RL2 ? RL2.sets[set].count : q4(60, set);
    J.L1_valid = {value: `${ok.length}/${p.length}`, pass: p.length === nWantL && ok.length >= Math.ceil(57 / 60 * nWantL)};
    const mc = minOrNull([...ok, ...fam].map(r => r.min_clearance));
    J.L2_clearance = {value: mc, pass: geq(mc, .03)};
    const te = ok.flatMap(r => r.tracking_abs ?? []);
    J.L3_tracking = {value: {median: quantile(te, .5), p95: quantile(te, .95)}, pass: te.length > 0 && quantile(te, .5) <= .05 && quantile(te, .95) <= .12};
    const F = Object.fromEntries(fam.map(r => [r.id, r]));
    const famOk = fam.length === 6 && fam.every(r => r.valid_episode);
    const prof = id => Float64Array.from(F[id]?.prof?.ey ?? [], v => v == null ? NaN : v);
    const pairDiff = (a, b) => { const A = prof(a), B = prof(b), d = []; for (let i = 0; i < A.length; i++) if (!Number.isNaN(A[i]) && !Number.isNaN(B[i])) d.push(Math.abs(A[i] - B[i])); return d; };
    const allBins = () => { const ids = Object.keys(F), P = ids.map(prof), keep = []; for (let i = 0; i < (P[0]?.length ?? 0); i++) if (P.every(p => !Number.isNaN(p[i]))) keep.push(i); return keep; };
    let L4 = {pass: false};
    if (famOk) {
      const bins = allBins(), d = (a, b) => bins.map(i => Math.abs(prof(a)[i] - prof(b)[i]));
      const lr = mean(d('c+0.8', 'c-0.8')), io = mean(d('inside', 'outside'));
      const ids = Object.keys(F), pairs = [];
      for (let a = 0; a < ids.length; a++) for (let b = a + 1; b < ids.length; b++) {
        const dd = d(ids[a], ids[b]); let best = 0;
        for (let i = 0; i + 20 <= dd.length; i++) { let s = 0; for (let k = 0; k < 20; k++) s += dd[i + k]; best = Math.max(best, s / 20); }
        pairs.push({a: ids[a], b: ids[b], max_1m_avg: best});
      }
      L4 = {value: {left_right_mean: lr, inside_outside_mean: io, min_pair_max_1m_avg: Math.min(...pairs.map(p => p.max_1m_avg)), bins: bins.length}, pairs,
        pass: lr >= .20 && io >= .10 && pairs.every(p => p.max_1m_avg >= .08)};
    }
    J.L4_distinct = {...L4, fam6_valid: famOk};
    const vm3 = profSpread(ok, 'ey').mean_abs_dev;
    J.L5_vm3 = {value: vm3, pass: vm3 != null && vm3 >= .08};
    J.L6_lap = {value: F.c0?.laps?.[0] ?? null, pass: F.c0?.valid_episode && F.c0.laps[0] <= 10.0};
    rep.by_family = Object.fromEntries([...new Set(p.map(r => r.family))].map(f => [f, {n: p.filter(r => r.family === f).length, valid: p.filter(r => r.family === f && r.valid_episode).length,
      tracking_median: quantile(p.filter(r => r.family === f).flatMap(r => r.tracking_abs ?? []), .5), lap1: mean(p.filter(r => r.family === f).map(r => r.laps[0]).filter(Number.isFinite))}]));
    if (RL2) {
      // 第 2 回の報告(protocol_line_r2.json acceptance_changes.reported)
      for (const [f, v] of Object.entries(rep.by_family)) v.tracking_p95 = quantile(ok.filter(r => r.family === f).flatMap(r => r.tracking_abs ?? []), .95);
      rep.collided = p.filter(r => r.collisions > 0).map(r => ({id: r.id, family: r.family, frames: r.frames, gate: r.start?.gate}));
      rep.below_5cm = [...p, ...fam].filter(r => r.min_clearance != null && r.min_clearance < .05).map(r => ({id: r.id, test: r.test, family: r.family, min_clearance: r.min_clearance,
        t_s: r.dmin_frame_min_t, valid: r.valid_episode}));
      rep.start_1p5s_min_clearance = minOrNull(p.map(r => r.dmin_first_1p5s).filter(Number.isFinite));
      const fr = ok.flatMap(r => (r.tracking_abs ?? []).map((e, k) => [e, r.tracking_s?.[k]])), big = fr.filter(([e]) => e > .12);
      const inS = s => s != null && ((s >= 2 && s <= 4.5) || (s >= 8 && s <= 12.5));
      rep.tracking_gt12 = {frac: fr.length ? big.length / fr.length : null, n: big.length, share_in_S_zones: big.length ? big.filter(([, s]) => inS(s)).length / big.length : null,
        S_zones_s: [[2, 4.5], [8, 12.5]]};
      rep.protocol_line_r2_sha256 = RL2._sha256;
    }
  } else if (mode === 'speed') {
    const p = T('SP-P60'), s3 = T('SP-STD3'), ok = p.filter(r => r.valid_episode);
    const R2 = opt.protocolR2, nWant = R2 ? R2.sets[set].count : q4(60, set);
    J.S1_valid = {value: `${ok.length}/${p.length}`, pass: p.length === nWant && ok.length >= Math.ceil(57 / 60 * nWant)};
    if (R2) {
      // 第 2 回: S2 は全部の回(有効でない回も)。衝突した回・開始できなかった回はすき間 0 として数える(確認 speed r1 の指摘 2)
      const cl = p.map(r => (r.invalid_spawn || r.collisions > 0 || r.min_clearance == null) ? 0 : r.min_clearance);
      const mc = cl.length === nWant ? Math.min(...cl) : null;
      J.S2_clearance = {value: mc, pass: geq(mc, .03), rule: 'protocol_speed_r2: 全部の回、衝突した回はすき間 0',
        below_3cm: p.filter((r, i) => cl[i] < .03).map(r => r.id), below_5cm: p.filter((r, i) => cl[i] < .05).map(r => r.id),
        collided: p.filter(r => r.collisions > 0).map(r => r.id)};
      const cvsAll = [...new Set(p.map(r => r.c_v))].sort((a, b) => a - b);
      rep.clearance_by_c_v = cvsAll.map(cv => { const v = p.filter(r => r.c_v === cv).map(r => (r.collisions > 0 || r.min_clearance == null) ? 0 : r.min_clearance).sort((a, b) => a - b);
        return {c_v: cv, n: v.length, min: v[0] ?? null, median: quantile(v, .5), below_5cm: v.filter(x => x < .05).length}; });
      rep.std3_min_clearance = minOrNull(s3.map(r => (r.collisions > 0 || r.min_clearance == null) ? 0 : r.min_clearance));
      rep.protocol_speed_r2_sha256 = R2._sha256;
    } else {
      const mc = minOrNull(ok.map(r => r.min_clearance));
      J.S2_clearance = {value: mc, pass: geq(mc, .03)};
    }
    const t = Object.fromEntries(s3.map(r => [r.id, r]));
    const t1 = t['cv1']?.lap_from_stop, t75 = t['cv0.75']?.lap_from_stop, t5 = t['cv0.5']?.lap_from_stop;
    J.S3_lap_order = {value: {t1, t75, t5}, pass: s3.every(r => r.lap_complete) && t1 < t75 && t75 < t5 && t5 - t1 >= 3.0};
    const cvs = [...new Set(p.map(r => r.c_v))].sort((a, b) => a - b);
    const per = cvs.map(cv => { const rs = ok.filter(r => r.c_v === cv); const vs = rs.map(r => r.speed.mean_running); return {c_v: cv, n: rs.length, mean_speed: mean(vs), p95_max: rs.length ? Math.max(...rs.map(r => r.speed.p95)) : null}; });
    let mono = true; for (let i = 1; i < per.length; i++) if (!(per[i].mean_speed - per[i - 1].mean_speed >= .05)) mono = false;
    const capOk = per.length > 0 && per.every(x => x.n > 0 && x.p95_max <= 4.5 * x.c_v + .05);
    J.S4_speed_order = {value: per, pass: mono && capOk};
    const vm4 = profSpread(ok, 'v').mean_abs_dev;
    J.S5_vm4 = {value: vm4, pass: vm4 != null && vm4 >= .4};
  } else if (mode === 'obstacle') {
    const p = T('OB-T100'), e = T('OB-EMPTY')[0], ok = p.filter(r => r.valid_episode);
    J.O1_valid = {value: `${ok.length}/${p.length}`, pass: ok.length >= Math.ceil(.95 * p.length)};
    const mc = minOrNull(ok.map(r => r.min_clearance));
    J.O2_clearance = {value: mc, pass: geq(mc, .03)};
    const l1 = ok.map(r => r.laps[0]).filter(Number.isFinite);
    J.O3_lap1_median = {value: quantile(l1, .5), pass: l1.length > 0 && quantile(l1, .5) <= 10.5};
    J.O4_empty = {value: e?.lap_from_stop ?? null, pass: e?.lap_complete === true && e.lap_from_stop <= 9.0};
  }
  // 物差し(報告)
  const vmRows = rows.filter(r => r.vm_frames && (r.valid_episode || r.test === 'RC-P100'));
  if (vmRows.length >= 2) {
    const scenes = vmRows.map(r => ({frames: r.vm_frames.map(([s, ey, epsi, v]) => ({s, ey, epsi, v})), s0: -Infinity}));
    rep.variation = {rows: vmRows.length, VM1_us: VM1(vmRows.map(r => ({frames: r.st_time.map(st => ({st}))}))), VM2_us: profSpread(vmRows, 'st'),
      VM3_m: profSpread(vmRows, 'ey'), VM4_mps: profSpread(vmRows, 'v'), VM7_cells: VM7(scenes.map(s => ({...s, s0: -Infinity})), 3000).cells,
      note: 'VM1・VM2 は舵の PWM [µs]、VM3 は横ずれ [m]、VM4 は速さ [m/s]。VM7 はフレームを 3 つおきに間引いた 3,000 フレームで数えた参考の値(判定は記録の段)'};
  }
  const judged = Object.values(J).filter(x => x.pass !== null);
  return {judged: J, all_judged_pass: judged.length > 0 && judged.every(x => x.pass === true), reported: rep};
}

// ---- 実行 ----
async function main() {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; }, flag = n => args.includes(n);
  const mode = opt('--mode'), policyPath = opt('--policy'), set = opt('--set', 'test'), core = opt('--core', set === 'validation' ? 'fast' : 'original');
  const jobs = +opt('--jobs', '1'), only = opt('--only')?.split(','), out = opt('--out');
  const o = {shieldC: opt('--shield-c') ? +opt('--shield-c') : null, classes: opt('--classes'), speedCaps: flag('--speed-caps')};
  if (opt('--protocol')) {
    if (opt('--protocol') === 'line_r2') {
      if (mode !== 'line') { console.error('--protocol line_r2 は --mode line だけです'); process.exit(2); }
      o.protocolLineR2 = loadLineR2();
    } else {
      if (opt('--protocol') !== 'speed_r2' || mode !== 'speed') { console.error('--protocol speed_r2 は --mode speed だけです'); process.exit(2); }
      o.protocolR2 = loadSpeedR2();
    }
  }
  // --run-mode: 別のモードの方策で同じ試験の組を走らせる(報告用。例 OB-BASE = --mode obstacle --run-mode fastest)
  const overrides = opt('--run-mode') && opt('--run-mode') !== mode ? {allow_obstacles: true} : {};
  if (flag('--no-shield')) overrides.shield = false;   // 報告用(例: 最速の方策・シールドなしで no_wall の試験の組)
  if (!mode || !policyPath) { console.error('使い方: node rl/eval_modes.mjs --mode <key> --policy <json> [--set test|validation] [--out file]'); process.exit(2); }
  const weights = readJson(policyPath), sha = sha256File(policyPath);
  let J = buildJobs(mode, set, o);
  if (only) J = J.filter(j => only.includes(j.test));
  const shard = opt('--shard');
  const ctx = {mode: opt('--run-mode', mode), weights, sha, core, opt: o, overrides};
  // 線の混ぜ合わせ(protocol fallbacks.line): --run-mode fastest --policy-list a.json,b.json,… で LN-P60 の回 i に方策 i mod n を使う(L1・L2・L5 だけを判定)
  if (opt('--policy-list')) ctx.policyList = opt('--policy-list').split(',').map(p => ({path: p, weights: readJson(p), sha: sha256File(p)}));
  if (shard) {
    const [k, n] = shard.split('/').map(Number), rows = [];
    for (let j = k; j < J.length; j += n) { const row = await runJob(J[j], ctx); row._j = j; rows.push(row); }
    fs.writeFileSync(opt('--rows-out'), JSON.stringify(rows));
    return;
  }
  const t0 = Date.now();
  let rows = [];
  if (jobs > 1) {
    const tmp = fs.mkdtempSync('/tmp/eval_modes_');
    await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
      const ch = fork(fileURLToPath(import.meta.url), [...args.filter((a, i) => !['--jobs', '--out'].includes(a) && !['--jobs', '--out'].includes(args[i - 1])), '--shard', `${k}/${jobs}`, '--rows-out', `${tmp}/${k}.json`], {stdio: 'inherit'});
      ch.on('exit', c => c === 0 ? res() : rej(new Error('子プロセスが失敗しました ' + c)));
    })));
    for (let k = 0; k < jobs; k++) rows.push(...JSON.parse(fs.readFileSync(`${tmp}/${k}.json`, 'utf8')));
    fs.rmSync(tmp, {recursive: true});
    rows.sort((a, b) => a._j - b._j);
  } else for (const [j, job] of J.entries()) { const r = await runJob(job, ctx); r._j = j; rows.push(r); if ((j + 1) % 20 === 0) console.error(`${j + 1}/${J.length}`); }
  const verdict = judge(mode, rows, set, o);
  if (ctx.policyList) {   // 代わりの方法: L3・L4・L6 は当てはまらない
    for (const k of ['L3_tracking', 'L4_distinct', 'L6_lap']) if (verdict.judged[k]) verdict.judged[k] = {pass: null, note: '線の混ぜ合わせには当てはまらない(protocol fallbacks.line)'};
    const js = Object.values(verdict.judged).filter(x => x.pass !== null);
    verdict.all_judged_pass = js.every(x => x.pass === true);
    verdict.fallback = {kind: 'line_checkpoint_mix', policies: ctx.policyList.map(p => ({path: p.path, sha256: p.sha}))};
  }
  const res = {schema: 'jf-sim-rl-mode-eval/v1', created_at: new Date().toISOString(), mode, set, core, judged_core_note: PROTOCOL.definitions.core_judged,
    policy: {path: policyPath, sha256: sha, schema: weights.schema, obs_dim: weights.obs_dim},
    options: o.protocolR2 ? {...o, protocolR2: undefined, protocol: 'speed_r2'} : o.protocolLineR2 ? {...o, protocolLineR2: undefined, protocol: 'line_r2'} : o,
    protocol_sha256: sha256File(PROTOCOL_PATH),
    ...(o.protocolR2 ? {protocol_speed_r2_sha256: o.protocolR2._sha256, set_spec: o.protocolR2.sets[set]} : {}),
    ...(o.protocolLineR2 ? {protocol_line_r2_sha256: o.protocolLineR2._sha256, set_spec: o.protocolLineR2.sets[set]} : {}),
    runs: rows.length, ...verdict, wall_s: (Date.now() - t0) / 1000,
    // 走行中のシールドの版と値(no_wall。2026-10-14 r2 で足した: どのシールドで試験したかを残す)
    ...(mode === 'no_wall' ? {shield: {...(await import('./runtime/shield.mjs')).SHIELD_DEFAULTS, ...(o.shieldC ? {c_nominal: o.shieldC} : {}),
      shield_mjs_sha256: sha256File(path.join(here, 'runtime/shield.mjs'))}} : {}),
    tests: Object.fromEntries([...new Set(rows.map(r => r.test))].map(t => [t, rows.filter(r => r.test === t).map(r => { const {prof, vm_frames, st_time, tracking_abs, tracking_s, _j, ...rest} = r; return rest; })]))};
  // F-ID: STD の連続の 600 フレームの snapshot と Node の操作(ブラウザの teacher.js に 1 フレームずつ与えて比べる)
  if (opt('--fid')) await writeFid(mode, weights, sha, opt('--fid'));
  if (out) { fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true}); fs.writeFileSync(out, JSON.stringify(res, null, 1)); }
  console.log(JSON.stringify({mode, set, runs: rows.length, all_judged_pass: res.all_judged_pass, wall_s: res.wall_s,
    judged: Object.fromEntries(Object.entries(res.judged).map(([k, v]) => [k, {pass: v.pass, value: v.value}]))}));
}
export async function writeFid(mode, weights, sha, outPath, {start = STD, params = {}, layout = null} = {}) {
  const {createModeTeacher} = await import('./runtime/mode_runtime.mjs');
  const core = await import('../snapshot/sim/dist/sim-core.js');
  const {simConfig} = await import('../snapshot/rl/env_core.mjs');
  const {world: active, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
  const world = layout ? worldWithObstacles(active, layout.obstacles) : active;
  const cfg = simConfig({episodeMode: 'continuous', seed: start.seed});
  const sim = core.createSimulation(world, cfg);
  let snap = sim.reset({seed: cfg.seed});
  const t = createModeTeacher(world, JSON.parse(JSON.stringify(snap.config)), variation, {mode, weights, params, policySha256: sha});
  const frames = [];
  while (!snap.state.terminal) { const a = t.action(snap); frames.push({snapshot: snap, action: a}); snap = sim.step(a); }
  fs.writeFileSync(outPath, JSON.stringify({schema: 'jf-sim-rl-fid/v1', mode, policy_sha256: sha, world, config: snap.config, variation, params, metadata: t.metadata, frames}));
}
if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
