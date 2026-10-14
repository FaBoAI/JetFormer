#!/usr/bin/env node
// 離脱復活 r3 の試験・検証(2026-10-14、学習の段 recover r3。protocol_recover_r3.json sets・acceptance・definitions_added のとおり)。
//
//   node r3/rc/eval_rc3.mjs --set validation|test|fresh|v1|fallback --policy <policy.json> [--jobs N] [--out 結果.json] [--only 組,…] [--label 名前]
//     validation: RC-V100-R3(Halton 17,001〜・種 3,220,000 + i・乱し v2)+ RC-VKICK-R3(STD・時刻 1.4 + 0.5k・種 3,221,000 + i)+ RC-STD
//                 + RC-REC60(protocol の検証の決まりどおり最初の 15)+ RC-VU30-R3(乱しなし・RC-V100-R3 の続きの Halton 30・種 3,222,000 + i)。高速版
//     test      : RC-P100-R3(Halton 28,001〜・種 3,120,000 + i)+ RC-KICK-R3(STD・時刻 1.25 + 0.5k・種 3,121,000 + i、高速版)+ RC-STD + RC-REC60(60)
//                 + RC-U60-R3(乱しなし・RC-P100-R3 の続きの Halton 60・種 3,122,000 + i)。元の sim-core.js(キックだけ高速版)
//     fresh     : mulberry32(0x3c3c0002) の一様乱数の新しい開始 150 回(記録の開始の分布・開始のすき間 > 0・乱しあり・種 3,320,000 + i)。報告だけ
//     v1        : RC-P100-R3 と同じ開始・種を、protocol の元の乱し(v1)で(RC-P100-R3-V1。報告だけ)
//     fallback  : RC-P100-R3 を強さ「弱・中」だけで(protocol fallbacks.recover。乱し v2 のまま。1 回だけ)
// 走行中の仕組みは r3/rc/rt/mode_runtime.mjs(離脱復活: 乱し v2 + 乱しのフレーム以外のシールド 3.5 cm)。判定は definitions_added の valid_episode_r3
// (止まり・進みを足す)・C5_r3(止まりのないキックの生き残り)・C6_r3(RC-REC60 と RC-U60-R3)・C7(当てた組の強さ・種類のばらつき)。
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD} from './episode_rc3.mjs';
import {CATALOGUE} from './rt/catalogue.mjs';
import {haltonStart} from './rt/rng.mjs';
import {prepareGeometry, clearance, LOOP_LENGTH} from './rt/features.mjs';
import {perturbAllowed, RECORD_SPEC, maskFromLearn} from './rt/perturb.mjs';
import {RECOVERABLE_TABLE_SHA256} from './rt/recoverable_table.mjs';
import {RECOVER_R3} from './rt/mode_runtime.mjs';
import {nominalLine, nominalAt, sceneFromEpisode, binProfile, spread, quantile, VM3, VM7} from './variation_metrics_rc3.mjs';
import {activeWorldFor} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';
import {loadRec60} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/eval_sets.mjs';
import {createRng, mix32} from './rt/rng.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const R3DIR = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3';
const P3_PATH = `${R3DIR}/protocol_recover_r3.json`;
const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
// 追補の決まり・立て直せる表は固めた sha256 と照合する(違えば止まる)
function loadP3() {
  const got = sha256File(P3_PATH);
  const want = Object.fromEntries(fs.readFileSync(`${R3DIR}/protocol_r3.sha256`, 'utf8').trim().split('\n').map(l => l.trim().split(/\s+/)).map(([h, f]) => [f, h]));
  if (want['protocol_recover_r3.json'] !== got) throw new Error(`eval_rc3: protocol_recover_r3.json の sha256 ${got} が固めた値と違います`);
  if (process.env.JF_RC3_SMOKE === '1') { console.error('eval_rc3: 試し(JF_RC3_SMOKE=1)なので立て直せる表の照合を省きます'); return {...JSON.parse(fs.readFileSync(P3_PATH, 'utf8')), _sha256: got, _smoke: true}; }
  const tw = fs.readFileSync(`${R3DIR}/protocol_recover_r3_table.sha256`, 'utf8').split(/\s+/)[0];
  if (tw !== RECOVERABLE_TABLE_SHA256) throw new Error(`eval_rc3: 立て直せる表の sha256 ${RECOVERABLE_TABLE_SHA256} が固めた値 ${tw} と違います`);
  return {...JSON.parse(fs.readFileSync(P3_PATH, 'utf8')), _sha256: got};
}
const P3 = loadP3();
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
const DIST = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution;
const baseGeom = prepareGeometry(BASE_WORLD);
const startClear = st => { const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg); return clearance(baseGeom, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5); };

// Halton の開始(開始のすき間 > 0 を満たさない番号は飛ばす)。戻り値 {starts, next}(next = 続きの番号)
export function haltonFrom(from, count) {
  const out = [];
  let i = from;
  for (; out.length < count && i < from + 100 * count; i++) {
    const st = haltonStart(i, DIST);
    const c = startClear(st);
    if (!(c > 1e-9)) continue;
    out.push({...st, initial_clearance_m: c});
  }
  if (out.length !== count) throw new Error(`eval_rc3: 開始が ${out.length} 個しか作れません`);
  return {starts: out, next: i};
}
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function freshStarts(count = 150) {
  const u = mulberry32(0x3c3c0002), out = [];
  for (let guard = 0; out.length < count && guard < 100 * count; guard++) {
    const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: DIST.left_offset_m[0] + (DIST.left_offset_m[1] - DIST.left_offset_m[0]) * u(),
      left_yaw_deg: DIST.left_yaw_deg[0] + (DIST.left_yaw_deg[1] - DIST.left_yaw_deg[0]) * u(), fresh_index: guard};
    const c = startClear(st);
    if (!(c > 1e-9)) continue;
    out.push({...st, initial_clearance_m: c});
  }
  return out;
}

// ---- 回の一覧(決まった順) ----
export function buildJobs(set) {
  const J = [], rec60 = loadRec60();
  const recStart = c => ({gate: c.gate, left_offset_m: c.left_offset_m, left_yaw_deg: c.left_yaw_deg, seed: c.seed, scenario: c.scenario === 'noise' ? 'baseline' : c.scenario});
  const add = (test, id, spec) => J.push({test, id, spec});
  const kicks = (name, t0, seedBase) => {
    let i = 0;
    for (let k = 0; k <= 35; k++) for (const sign of [1, -1]) for (const kind of ['yaw', 'lateral']) {
      const t = t0 + .5 * k;
      add(name, `${kind}${sign > 0 ? '+' : '-'}${t.toFixed(2)}`, {start: {...STD, seed: seedBase + i++}, kick: {frame: Math.round(t * 30), [kind]: sign * (kind === 'yaw' ? .3 : .15)},
        overrides: {perturbation: false}, kind});
    }
  };
  if (set === 'test' || set === 'v1' || set === 'fallback') {
    const T = P3.sets.test, p = haltonFrom(T['RC-P100-R3'].halton_from, 100);
    const name = set === 'test' ? 'RC-P100-R3' : set === 'v1' ? 'RC-P100-R3-V1' : 'RC-P100-R3-WM';
    const params = set === 'v1' ? {perturb_version: 'v1'} : set === 'fallback' ? {perturb_class: ['弱', '中']} : {};
    p.starts.forEach((st, i) => add(name, `P${i}`, {start: {...st, seed: T['RC-P100-R3'].seed_base + i}, params, perturbed: true}));
    if (set === 'test') {
      kicks('RC-KICK-R3', 1.25, T['RC-KICK-R3'].seed_base);
      add('RC-STD', 'single', {start: STD, episodeMode: 'single', overrides: {perturbation: false}});
      rec60.forEach(c => add('RC-REC60', c.name, {start: recStart(c), overrides: {perturbation: false}}));
      const u = haltonFrom(p.next, 60);
      u.starts.forEach((st, i) => add('RC-U60-R3', `U${i}`, {start: {...st, seed: 3122000 + i}, overrides: {perturbation: false}}));
    }
  } else if (set === 'validation') {
    const p = haltonFrom(17001, 100);
    p.starts.forEach((st, i) => add('RC-V100-R3', `V${i}`, {start: {...st, seed: 3220000 + i}, params: {}, perturbed: true}));
    kicks('RC-VKICK-R3', 1.4, 3221000);
    add('RC-STD', 'single', {start: STD, episodeMode: 'single', overrides: {perturbation: false}});
    rec60.slice(0, 15).forEach(c => add('RC-REC60', c.name, {start: recStart(c), overrides: {perturbation: false}}));
    const u = haltonFrom(p.next, 30);
    u.starts.forEach((st, i) => add('RC-VU30-R3', `VU${i}`, {start: {...st, seed: 3222000 + i}, overrides: {perturbation: false}}));
  } else if (set === 'fresh') {
    freshStarts(150).forEach((st, i) => add('RC-FRESH150-R3', `Q${i}`, {start: {...st, seed: 3320000 + i}, params: {}, perturbed: true}));
  } else throw new Error('eval_rc3: 知らない組 ' + set);
  return J;
}

// ---- 止まり・進み(protocol_recover_r3.json definitions_added) ----
export function stallInfo(log, events) {
  const T = log.length, excl = new Uint8Array(T);
  for (const e of events ?? []) if (e.type === 'speed_pulse' && e.kind === 'brake') for (let f = e.ps; f < Math.min(T, (e.pe ?? T) + 30); f++) excl[f] = 1;
  let run = 0, maxRun = 0, at = null;
  for (let f = 30; f < T; f++) {
    if (excl[f]) continue;   // ブレーキの乱しの区間とそのあと 30 フレームは数えない(続きも切らない = きびしい側)
    if (Math.abs(log[f].v) < .2) { run++; if (run > maxRun) maxRun = run; if (run >= 30 && at == null) at = f - 29; } else run = 0;
  }
  return {stall: maxRun >= 30, max_run_frames: maxRun, at};
}
const progressOf = log => log.length ? log.at(-1).s - log[0].s : 0;

// ---- 1 回 → 結果の行 ----
let nominalCache = null;
async function runJob(job, ctx) {
  const sp = job.spec, t0 = Date.now();
  const r = await runModeEpisode({mode: 'recover', weights: ctx.weights, policySha256: ctx.sha, start: sp.start, params: sp.params ?? {},
    episodeMode: sp.episodeMode ?? 'continuous', coreKind: sp.kick ? 'fast' : ctx.core, kick: sp.kick ?? null, overrides: sp.overrides ?? {}, audit: !sp.kick});
  const row = {test: job.test, id: job.id, start: sp.start, invalid_spawn: !!r.invalid_spawn, wall_s: 0};
  if (r.invalid_spawn) return row;
  const log = r.log, T = log.length;
  Object.assign(row, {reason: r.reason, frames: r.frames, collisions: r.collisions, collision: r.collision, laps: r.laps.map(l => l.elapsed), valid_laps: r.valid_laps,
    valid_episode: r.valid_episode, lap_complete: r.lap_complete ?? null, lap_from_stop: r.lap_from_stop ?? null, min_clearance: r.min_clearance ?? null,
    audit_err: r.audit?.max_pos_err ?? null, shield_frames: r.shield_frames, params: r.params, gate_miss: r.gate_miss, gate_misses: r.gate_misses,
    dmin_frame_min: Math.min(...log.map(q => q.dmin)), events: r.events,
    // 報告だけ(判定は min_clearance): 最小すき間の時刻と、1 秒より後のフレームの終わりのすき間(features)の最小。開始のすき間で決まる回の見分け
    min_clearance_at_s: r.min_clearance_at_s ?? null, dmin_after_1s: log.length > 30 ? Math.min(...log.slice(30).map(q => q.dmin)) : null});
  const si = stallInfo(log, r.events);
  row.stall = si.stall; row.stall_max_run = si.max_run_frames; row.stall_at = si.at;
  row.progress_m = progressOf(log);
  row.progress_ok = row.progress_m >= LOOP_LENGTH - 1e-9;
  row.valid_episode_r3 = !!(r.valid_episode && !si.stall && row.progress_ok);
  const run = log.filter(q => q.v > .3).map(q => q.v);
  row.speed = {mean_running: mean(run), p95: quantile(run, .95)};
  // シールドの作動: 出来事の終わりから 3 秒の中とそれ以外
  const after = new Uint8Array(T);
  for (const e of r.events) for (let f = e.pe ?? T; f < Math.min(T, (e.pe ?? T) + 90); f++) after[f] = 1;
  let shA = 0, nA = 0, shO = 0, nO = 0;
  for (let f = 0; f < T; f++) { if (log[f].ph === 'perturb') continue; if (after[f]) { nA++; shA += log[f].sh ? 1 : 0; } else { nO++; shO += log[f].sh ? 1 : 0; } }
  row.shield = {frames: r.shield_frames, frac: r.shield_frames / Math.max(1, T), after3s: {n: nA, sh: shA}, other: {n: nO, sh: shO}};
  if (sp.kick) {
    const kf = sp.kick.frame, q = log[kf] ?? log.at(-1);
    row.kick_s = q?.s ?? null;
    row.kick_allowed = q ? perturbAllowed(q.s, RECORD_SPEC) : false;
    row.survived = r.reason === 'collection_duration_complete' && r.collisions === 0;
    row.kick_survival_r3 = row.survived && !si.stall;
    row.kind = sp.kind;
    // 報告だけ: キックの直後のフレームで、シールドの標準の模型がどの候補でも 0.8 秒先までにぶつかると見たか(物理として避けられないキックの目安)
    const q1 = log[kf + 1]?.shield ?? null;
    row.kick_after_shield = q1 ? {fb: q1.fb, nom: q1.nom, n: q1.n} : null;
    row.kick_same_step_collision = r.collisions > 0 && r.frames <= kf + 1;   // キックと同じステップでぶつかった(操作はキックの前に決まる)
    row.kick_no_candidate = !!(q1 && (q1.fb === 'proposal_best' || q1.fb === 'max_clearance'));   // どの候補も合格しない(proposal_best・max_clearance)
  }
  if (sp.perturbed) {
    const learn = log.map(q => q.l), mask = maskFromLearn(learn);
    row.mask_frac = mask.reduce((a, b) => a + b, 0) / 600;
    if (!nominalCache) nominalCache = await computeNominal(ctx);
    const nl = nominalCache.line;
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
      return {n: ev.n, type: ev.type, kind: ev.kind, cls: ev.cls, sign: ev.sign, ps: ev.ps, pe: ev.pe, aborted: ev.aborted, s: ev.s, v: ev.v, req: ev.req, fallback: ev.fallback,
        wait_frames: ev.wait_frames, recovered_s: recS, success: recS != null && recS <= 3.0 && !bad, bad_after: bad};
    });
    const sc = sceneFromEpisode(r);
    const rnd = (a, k) => Array.from(a, v => Number.isNaN(v) ? null : +v.toFixed(k));
    row.prof = {ey: rnd(binProfile(sc, 'ey'), 4)};
    row.vm_frames = sc.frames.filter((q, i) => i % 3 === 0 && q.s - sc.s0 >= 3).map(q => [+q.s.toFixed(3), +q.ey.toFixed(4), +q.epsi.toFixed(4), +q.v.toFixed(3)]);
  }
  row.wall_s = (Date.now() - t0) / 1000;
  return row;
}
// 同じ方策・同じ走行中の仕組み(シールドあり)の乱しなしの線(protocol definitions.nominal_line)
async function computeNominal(ctx) {
  const scenes = [], runs = [];
  for (let i = 0; i < 8; i++) {
    const seed = 990000 + i, rng = createRng(mix32(seed));
    const start = {gate: 0, left_offset_m: rng.uniform(-.03, .03), left_yaw_deg: rng.uniform(-3, 3), seed};
    const r = await runModeEpisode({mode: 'recover', weights: ctx.weights, policySha256: ctx.sha, start, coreKind: ctx.core, audit: false, overrides: {perturbation: false}});
    scenes.push(sceneFromEpisode(r)); runs.push({seed, reason: r.reason, collisions: r.collisions, laps: r.laps.map(l => l.elapsed)});
  }
  return {line: nominalLine(scenes), runs};
}

// ---- 判定 ----
const geq = (v, t) => v != null && v >= t;
const zoneOf = s => { const m = ((s % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH; return m >= 8.0 && m <= 8.25 ? 'black_sheet_exit' : m >= 19.30 && m <= 19.55 ? 'back_straight_exit' : 'other'; };
export function judge(set, rows) {
  const T = name => rows.filter(r => r.test === name);
  const J = {}, rep = {};
  const pName = set === 'validation' ? 'RC-V100-R3' : set === 'test' ? 'RC-P100-R3' : set === 'v1' ? 'RC-P100-R3-V1' : set === 'fallback' ? 'RC-P100-R3-WM' : 'RC-FRESH150-R3';
  const p = T(pName), nP = p.length;
  const nValid = p.filter(r => r.valid_episode_r3).length;
  J.C1_r3 = {value: `${nValid}/${nP}`, pass: nP > 0 && nValid >= Math.ceil(.9 * nP),
    note: `valid_episode(元の定義)は ${p.filter(r => r.valid_episode).length}/${nP}。止まり ${p.filter(r => r.stall).length}・進みの不足 ${p.filter(r => r.valid_episode && !r.progress_ok).length}`};
  const ev = p.flatMap(r => r.recovery ?? []);
  const succ = ev.filter(e => e.success).length / Math.max(1, ev.length);
  J.C2 = {value: succ, events: ev.length, success: ev.filter(e => e.success).length, pass: ev.length > 0 && succ >= .95};
  const rt = ev.filter(e => e.success).map(e => e.recovered_s);
  J.C3 = {value: {median: quantile(rt, .5), p90: quantile(rt, .9)}, pass: rt.length > 0 && quantile(rt, .5) <= 1.2 && quantile(rt, .9) <= 2.5};
  const epm = mean(p.map(r => r.events?.length ?? 0)), mf = mean(p.map(r => r.mask_frac ?? 0));
  J.C4 = {value: {events_per_episode: epm, masked_fraction: mf}, pass: epm != null && epm >= 3.5 && mf >= .10 && mf <= .40};
  // C7(新): 当てた組の強さ・種類のばらつき
  const nE = ev.length, share = f => nE ? ev.filter(f).length / nE : 0;
  const clsShare = Object.fromEntries(['弱', '中', '強'].map(c => [c, share(e => e.cls === c)]));
  const typeShare = Object.fromEntries(['steer_pulse', 'excursion', 'speed_pulse'].map(t => [t, share(e => e.type === t)]));
  J.C7 = {value: {class_share: clsShare, type_share: typeShare}, pass: nE > 0 && Object.values(clsShare).every(x => x >= .15) && Object.values(typeShare).every(x => x >= .10)};
  if (set === 'test' || set === 'validation') {
    const kick = T(set === 'test' ? 'RC-KICK-R3' : 'RC-VKICK-R3');
    const ky = kick.filter(r => r.kick_allowed && r.kind === 'yaw'), kl = kick.filter(r => r.kick_allowed && r.kind === 'lateral');
    const sy = ky.filter(r => r.kick_survival_r3).length / Math.max(1, ky.length), sl = kl.filter(r => r.kick_survival_r3).length / Math.max(1, kl.length);
    J.C5_r3 = {value: {yaw: `${ky.filter(r => r.kick_survival_r3).length}/${ky.length}`, lateral: `${kl.filter(r => r.kick_survival_r3).length}/${kl.length}`, yaw_rate: sy, lateral_rate: sl},
      pass: ky.length > 0 && kl.length > 0 && sy >= .9 && sl >= .9};
    const std = T('RC-STD')[0], rec = T('RC-REC60'), u = T(set === 'test' ? 'RC-U60-R3' : 'RC-VU30-R3'), un = [...rec, ...u];
    const minC = un.length ? Math.min(...un.map(r => r.min_clearance ?? 0)) : null;
    J.C6_r3 = {value: {lap_from_stop: std?.lap_from_stop ?? null, rec60: `${rec.filter(r => r.valid_episode_r3).length}/${rec.length}`,
      unperturbed: `${u.filter(r => r.valid_episode_r3).length}/${u.length}`, min_clearance: minC,
      min_clearance_rec60: rec.length ? Math.min(...rec.map(r => r.min_clearance ?? 0)) : null, min_clearance_u: u.length ? Math.min(...u.map(r => r.min_clearance ?? 0)) : null},
      pass: std?.lap_complete === true && std.lap_from_stop <= 9.0 && rec.length > 0 && u.length > 0 && un.every(r => r.valid_episode_r3) && geq(minC, .03)};
    // 報告だけ: 開始のすき間が 3 cm 未満の開始(protocol は「開始のすき間 > 0」だけ)。これがあると C6_r3 は方策によらず満たせない
    rep.c6_spawn_limited = un.filter(r => (r.start?.initial_clearance_m ?? 1) < .03).map(r => ({test: r.test, id: r.id, initial_clearance_m: r.start.initial_clearance_m,
      min_clearance: r.min_clearance, min_clearance_at_s: r.min_clearance_at_s, dmin_after_1s: r.dmin_after_1s}));
    const unOk = un.filter(r => !((r.start?.initial_clearance_m ?? 1) < .03));
    rep.c6_without_spawn_limited = {runs: unOk.length, valid: unOk.filter(r => r.valid_episode_r3).length, min_clearance: unOk.length ? Math.min(...unOk.map(r => r.min_clearance ?? 0)) : null,
      dmin_after_1s_all: un.length ? Math.min(...un.map(r => r.dmin_after_1s ?? 1)) : null,
      note: '報告だけ(判定ではない)。開始のすき間が 3 cm 未満の開始を除いた最小すき間と、全部の回の 1 秒より後のフレームの終わりのすき間(features)の最小'};
    rep.kick_all_positions = {yaw: `${kick.filter(r => r.kind === 'yaw' && r.kick_survival_r3).length}/${kick.filter(r => r.kind === 'yaw').length}`,
      lateral: `${kick.filter(r => r.kind === 'lateral' && r.kick_survival_r3).length}/${kick.filter(r => r.kind === 'lateral').length}`};
    rep.kick_fail_ids = kick.filter(r => r.kick_allowed && !r.kick_survival_r3).map(r => ({id: r.id, reason: r.reason, stall: r.stall, kick_s: r.kick_s,
      frames: r.frames, same_step_collision: r.kick_same_step_collision ?? null, no_candidate_after_kick: r.kick_no_candidate ?? null}));
    rep.kick_fail_no_candidate = {judged_failed: kick.filter(r => r.kick_allowed && !r.kick_survival_r3).length,
      no_candidate: kick.filter(r => r.kick_allowed && !r.kick_survival_r3 && r.kick_no_candidate).length,
      same_step_collision: kick.filter(r => r.kick_allowed && !r.kick_survival_r3 && r.kick_same_step_collision).length,
      note: '報告だけ: キックの直後にシールドの標準の模型がどの候補(23)でも 0.8 秒先までに 3.5 cm を割ると見た(fallback proposal_best・max_clearance)回。物理として避けにくいキックの目安'};
    rep.unperturbed_fail_ids = un.filter(r => !r.valid_episode_r3).map(r => ({test: r.test, id: r.id, reason: r.reason, collisions: r.collisions, stall: r.stall, progress_m: r.progress_m}));
  }
  // 報告
  const by = k => Object.fromEntries([...new Set(ev.map(e => e[k]))].map(v => [v, {n: ev.filter(e => e[k] === v).length, success: ev.filter(e => e[k] === v && e.success).length}]));
  rep.by_class = by('cls'); rep.by_type = by('type');
  rep.by_fallback = by('fallback');
  const reqKey = e => e.req ? `${e.req.type}${e.req.kind ? ':' + e.req.kind : ''}:${e.req.cls}` : null, asKey = e => `${e.type}${e.kind ? ':' + e.kind : ''}:${e.cls}`;
  const reqAs = {};
  for (const e of ev) { const k = `${reqKey(e)} → ${asKey(e)}`; reqAs[k] = (reqAs[k] ?? 0) + 1; }
  rep.requested_to_assigned = Object.fromEntries(Object.entries(reqAs).sort((a, b) => b[1] - a[1]));
  rep.replaced_fraction = nE ? ev.filter(e => e.fallback && e.fallback !== 'requested').length / nE : null;
  rep.wait_frames = {median: quantile(ev.map(e => e.wait_frames ?? 0), .5), p90: quantile(ev.map(e => e.wait_frames ?? 0), .9)};
  const zones = {};
  for (const e of ev) { const z = zoneOf(e.s); zones[z] ??= {n: 0, success: 0}; zones[z].n++; if (e.success) zones[z].success++; }
  rep.start_zones = Object.fromEntries(Object.entries(zones).map(([k, v]) => [k, {...v, share: v.n / Math.max(1, nE)}]));
  rep.failed_events = ev.filter(e => !e.success).map(e => ({type: e.type, kind: e.kind, cls: e.cls, sign: e.sign, s: +(e.s % LOOP_LENGTH).toFixed(2), v: e.v, fallback: e.fallback,
    recovered_s: e.recovered_s, bad_after: e.bad_after}));
  const pv = p.filter(r => r.valid_episode_r3);
  rep.gate_miss = {p_valid: `${pv.filter(r => r.gate_miss).length}/${pv.length}`, ids: pv.filter(r => r.gate_miss).map(r => r.id)};
  rep.invalid_p = p.filter(r => !r.valid_episode_r3).map(r => ({id: r.id, reason: r.reason, collisions: r.collisions, stall: r.stall, progress_m: +(r.progress_m ?? 0).toFixed(2),
    frames: r.frames, collision_s: r.collision ? +r.collision.t.toFixed(2) : null}));
  const sumSh = (k, f) => p.reduce((a, r) => a + (r.shield?.[k]?.[f] ?? 0), 0);
  rep.shield = {after3s: sumSh('after3s', 'sh') / Math.max(1, sumSh('after3s', 'n')), other: sumSh('other', 'sh') / Math.max(1, sumSh('other', 'n')),
    all_rows: rows.reduce((a, r) => a + (r.shield?.frames ?? 0), 0) / Math.max(1, rows.reduce((a, r) => a + (r.frames ?? 0), 0))};
  rep.min_clearance_valid_p = pv.length ? Math.min(...pv.map(r => r.min_clearance ?? 0)) : null;
  if (pv.length >= 2) {
    const scenes = pv.map(r => ({frames: r.vm_frames.map(([s, ey, epsi, v]) => ({s, ey, epsi, v})), s0: -Infinity}));
    rep.variation = {rows: pv.length, VM7_cells_3000: VM7(scenes, 3000).cells, VM3_m: spread(pv.map(r => Float64Array.from(r.prof.ey, v => v == null ? NaN : v))).mean_abs_dev,
      note: '有効な乱しありの回だけ。VM7 は 3 フレームおきに間引いた 3,000 フレームで数えた参考の値(判定は記録の段)'};
  }
  const judged = Object.values(J);
  return {judged: J, all_judged_pass: judged.length > 0 && judged.every(x => x.pass === true), n_passed: judged.filter(x => x.pass).length, n_judged: judged.length, reported: rep};
}

async function main() {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
  const set = opt('--set', 'test'), policyPath = opt('--policy'), jobs = +opt('--jobs', '1'), out = opt('--out'), only = opt('--only')?.split(',');
  const core = opt('--core', set === 'validation' ? 'fast' : 'original');
  if (!policyPath) { console.error('使い方: node r3/rc/eval_rc3.mjs --set validation|test|fresh|v1|fallback --policy <json> [--jobs N] [--out file]'); process.exit(2); }
  const weights = readJson(policyPath), sha = sha256File(policyPath);
  let J = buildJobs(set);
  if (only) J = J.filter(j => only.includes(j.test));
  if (opt('--limit')) { const cnt = {}; J = J.filter(j => (cnt[j.test] = (cnt[j.test] ?? 0) + 1) <= +opt('--limit')); }   // 試しだけ
  const ctx = {weights, sha, core};
  const shard = opt('--shard');
  if (shard) {
    const [k, n] = shard.split('/').map(Number), rows = [];
    for (let j = k; j < J.length; j += n) { const row = await runJob(J[j], ctx); row._j = j; rows.push(row); }
    fs.writeFileSync(opt('--rows-out'), JSON.stringify({rows, nominal: nominalCache?.runs ?? null}));
    return;
  }
  const t0 = Date.now();
  let rows = [], nominalRuns = null;
  if (jobs > 1) {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'eval_rc3_'));
    await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
      const ch = fork(fileURLToPath(import.meta.url), [...args.filter((a, i) => !['--jobs', '--out'].includes(a) && !['--jobs', '--out'].includes(args[i - 1])), '--shard', `${k}/${jobs}`, '--rows-out', `${tmp}/${k}.json`], {stdio: 'inherit'});
      ch.on('exit', c => c === 0 ? res() : rej(new Error('子プロセスが失敗しました ' + c)));
    })));
    for (let k = 0; k < jobs; k++) { const d = JSON.parse(fs.readFileSync(`${tmp}/${k}.json`, 'utf8')); rows.push(...d.rows); nominalRuns ??= d.nominal; }
    fs.rmSync(tmp, {recursive: true});
    rows.sort((a, b) => a._j - b._j);
  } else for (const [j, job] of J.entries()) { const r = await runJob(job, ctx); r._j = j; rows.push(r); if ((j + 1) % 50 === 0) console.error(`${j + 1}/${J.length}`); }
  nominalRuns ??= nominalCache?.runs ?? null;
  const verdict = judge(set, rows);
  const res = {schema: 'jf-sim-rl-recover-r3-eval/v1', created_at: new Date().toISOString(), set, core, label: opt('--label'),
    policy: {path: policyPath, sha256: sha, obs_dim: weights.obs_dim, meta_steps: weights.meta?.steps ?? null},
    protocol_recover_r3_sha256: P3._sha256, recoverable_table_sha256: RECOVERABLE_TABLE_SHA256, runtime: {perturbation: set === 'v1' ? 'jf-perturb-record/v1' : RECOVER_R3.perturbation,
      min_rate: RECOVER_R3.min_rate, shield: RECOVER_R3.shield, mode_runtime_sha256: sha256File(path.join(here, 'rt/mode_runtime.mjs')),
      perturb_v2_sha256: sha256File(path.join(here, 'rt/perturb_v2.mjs')), shield_mjs_sha256: sha256File(path.join(here, 'rt/shield.mjs'))},
    nominal_runs: nominalRuns, runs: rows.length, ...verdict, wall_s: (Date.now() - t0) / 1000,
    tests: Object.fromEntries([...new Set(rows.map(r => r.test))].map(t => [t, rows.filter(r => r.test === t).map(r => { const {prof, vm_frames, _j, ...rest} = r; return rest; })]))};
  if (out) { fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true}); fs.writeFileSync(out, JSON.stringify(res, null, 1)); }
  console.log(JSON.stringify({set, runs: rows.length, all_judged_pass: res.all_judged_pass, n_passed: res.n_passed, wall_s: res.wall_s,
    judged: Object.fromEntries(Object.entries(res.judged).map(([k, v]) => [k, {pass: v.pass, value: v.value}]))}));
}
if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
