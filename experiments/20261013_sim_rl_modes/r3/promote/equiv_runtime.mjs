#!/usr/bin/env node
// 昇格の段(2026-10-15、r3/promote): まとめた走行中の仕組み(rt_merged)が、各モードの試験に使った写し(候補の runtime/)と
// ビット単位で同じ動きをすることを確かめる。
//   - 閉ループ: 同じ開始・種・条件・物理で、元の sim-core.js を 2 つの runtime で走らせ、全フレームの操作(舵・速さ)・info.jf・
//     教師の metadata・乱しの出来事・シールドのフレーム数を比べる(違いは 0 であること)。
//   - 開ループ(F-ID): 保存した snapshot の列を rt_merged に与え、保存した操作・info.jf・metadata と比べる(check_fid.mjs と同じ比べ方)。
// 走らせ方は rl/episode.mjs の写し(物理のずれ・遅延・ゲイン・摩擦円の扱いが同じ)。すき間の測り直しはしない(操作が同じなら同じ)。
//
//   node r3/promote/equiv_runtime.mjs --cases cases.json --out out.json
//   cases.json = {cases: [{mode, a_rt, b_rt, policy, a_mode_runtime?, b_mode_runtime?, start, params, dr}], fids: [{mode, file, rt, policy, mode_runtime?}]}
//     a_rt・b_rt = runtime のフォルダー(mode_runtime.mjs のあるところ)。*_mode_runtime = createModeTeacher に渡す modeRuntime(mode.json の runtime)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS} = await import(pathToFileURL(path.join(EXP, 'snapshot/rl/env_core.mjs')).href);
const BASE_WORLD = JSON.parse(fs.readFileSync(path.join(EXP, 'snapshot/sim/world-v3.json'), 'utf8'));
const cores = {};
async function loadCore(kind) {
  if (!cores[kind]) cores[kind] = await import(pathToFileURL(kind === 'original' ? path.join(EXP, 'snapshot/sim/dist/sim-core.js') : path.join(EXP, 'rl/fast_core_modes.mjs')).href);
  return cores[kind];
}
const rts = new Map();
async function loadRt(dir) {
  const d = path.resolve(dir);
  if (!rts.has(d)) rts.set(d, await import(pathToFileURL(path.join(d, 'mode_runtime.mjs')).href));
  return rts.get(d);
}
const jsonCache = new Map();
function readJson(p) { if (!jsonCache.has(p)) jsonCache.set(p, JSON.parse(fs.readFileSync(p, 'utf8'))); return jsonCache.get(p); }
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

async function run(rt, {mode, policy, start, params = {}, dr = null, modeRuntime = null, overrides = {}}) {
  const fc = !!dr?.frictionCircle;
  const core = await loadCore(fc ? 'fastfc' : 'original');
  const {world, variation} = activeWorldFor(BASE_WORLD, start.gate ?? 0, start.left_offset_m ?? 0, start.left_yaw_deg ?? 0);
  const nominal = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: start.seed ?? 1});
  const physics = dr?.physics ?? {};
  for (const k of Object.keys(physics)) if (!PERTURB_KEYS.includes(k)) throw new Error('変えられない物理の値 ' + k);
  const simCfg = {...nominal, ...physics, ...(fc ? {frictionCircle: true} : {})};
  const sim = core.createSimulation(world, simCfg);
  let snap = sim.reset({seed: simCfg.seed, scenario: start.scenario ?? 'baseline'});
  if (fc) snap = sim.peek();
  if (snap.state.terminal) return {invalid_spawn: true};
  const cfgN = {...JSON.parse(JSON.stringify(snap.config)), ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))};
  delete cfgN.frictionCircle;
  const weights = readJson(policy);
  const opts = {mode, weights, params, policySha256: sha(policy), overrides};
  if (modeRuntime) opts.modeRuntime = modeRuntime;
  const teacher = rt.createModeTeacher(world, cfgN, variation, opts);
  const delay = dr?.delay ?? 0, sg = dr?.steerGain ?? 1, vg = dr?.speedGain ?? 1;
  const queue = [];
  let appliedPrev = null;
  const acts = [], infos = [];
  const t0 = Date.now();
  while (!snap.state.terminal) {
    const out = teacher.action({state: snap.state, metrics: snap.metrics, config: cfgN}, delay ? {applied: appliedPrev ?? [0, 0]} : null);
    acts.push([out.steering, out.speed]); infos.push(JSON.stringify(out.info));
    const lg = teacher.log.at(-1);
    let ap = {steering: out.steering, speed: out.speed, a: [lg.a0, lg.a1]};
    if (delay) { queue.push(ap); ap = queue.length > delay ? queue.shift() : {steering: 0, speed: 0, a: [0, -1]}; appliedPrev = ap.a; }
    if (fc) { sim.step({steering: ap.steering * sg, speed: ap.speed * vg}, true); snap = sim.peek(); }
    else snap = sim.step({steering: ap.steering * sg, speed: ap.speed * vg});
  }
  return {acts, infos, metadata: JSON.stringify(teacher.metadata), events: JSON.stringify(teacher.events), shield_frames: teacher.shieldFrames,
    reason: snap.state.reason, frames: snap.state.frame, collisions: snap.metrics.collisions, laps: (snap.metrics.lap_events ?? []).map(l => +l.elapsed.toFixed(4)),
    reverse_frames: acts.filter(a => a[1] < 0).length, ms: Date.now() - t0};
}

function compare(a, b) {
  if (a.invalid_spawn || b.invalid_spawn) return {same: !!a.invalid_spawn === !!b.invalid_spawn, invalid_spawn: true};
  let first = null, maxS = 0, maxV = 0, infoDiff = 0;
  const n = Math.max(a.acts.length, b.acts.length);
  for (let i = 0; i < n; i++) {
    const x = a.acts[i], y = b.acts[i];
    if (!x || !y) { if (first === null) first = i; continue; }
    const ds = Math.abs(x[0] - y[0]), dv = Math.abs(x[1] - y[1]);
    maxS = Math.max(maxS, ds); maxV = Math.max(maxV, dv);
    if (a.infos[i] !== b.infos[i]) infoDiff++;
    if (first === null && (ds !== 0 || dv !== 0 || a.infos[i] !== b.infos[i])) first = i;
  }
  const same = first === null && a.acts.length === b.acts.length && a.metadata === b.metadata && a.events === b.events && a.shield_frames === b.shield_frames;
  return {same, frames: [a.acts.length, b.acts.length], first_diff_frame: first, max_abs_steering: maxS, max_abs_speed: maxV, info_mismatch: infoDiff,
    metadata_same: a.metadata === b.metadata, events_same: a.events === b.events, shield_frames: [a.shield_frames, b.shield_frames],
    reason: a.reason, collisions: a.collisions, laps: a.laps, reverse_frames: a.reverse_frames, ms: [a.ms, b.ms]};
}

async function fidCheck({mode, file, rt: dir, policy, mode_runtime = null}) {
  const rt = await loadRt(dir);
  const fid = readJson(file);
  const weights = readJson(policy);
  const opts = {mode: fid.mode ?? mode, weights, params: fid.params ?? {}, policySha256: fid.policy_sha256};
  if (fid.overrides && Object.keys(fid.overrides).length) opts.overrides = fid.overrides;
  if (mode_runtime) opts.modeRuntime = mode_runtime;
  const t = rt.createModeTeacher(fid.world, fid.config, fid.variation, opts);
  let maxS = 0, maxV = 0, info = 0;
  for (const fr of fid.frames) {
    const a = t.action(fr.snapshot), b = fr.action;
    maxS = Math.max(maxS, Math.abs(a.steering - b.steering)); maxV = Math.max(maxV, Math.abs(a.speed - b.speed));
    if (JSON.stringify(a.info) !== JSON.stringify(b.info)) info++;
  }
  const strip = m => { const x = {...m}; delete x.episode_mode; delete x.teacher_slot; return JSON.stringify(x); };
  const metaSame = fid.metadata ? strip(t.metadata) === strip(fid.metadata) : null;
  return {mode, file, frames: fid.frames.length, max_abs_steering: maxS, max_abs_speed: maxV, info_mismatch: info, metadata_same: metaSame,
    policy_sha_ok: sha(policy) === fid.policy_sha256, pass: maxS === 0 && maxV === 0 && info === 0 && metaSame !== false && sha(policy) === fid.policy_sha256};
}

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const spec = readJson(path.resolve(opt('--cases')));
const outPath = opt('--out');
const res = {schema: 'jf-sim-rl-runtime-equivalence/v1', created_at: new Date().toISOString(), cases: [], fids: []};
const save = () => fs.writeFileSync(outPath, JSON.stringify(res, null, 1));
for (const f of spec.fids ?? []) {
  const r = await fidCheck(f);
  res.fids.push(r);
  console.error('[fid]', f.mode, path.basename(f.file), r.pass ? 'pass' : 'FAIL', JSON.stringify({s: r.max_abs_steering, v: r.max_abs_speed, info: r.info_mismatch, meta: r.metadata_same}));
  save();
}
let i = 0;
for (const c of spec.cases ?? []) {
  const A = await loadRt(c.a_rt), B = await loadRt(c.b_rt);
  const base = {mode: c.mode, policy: c.policy, start: c.start, params: c.params ?? {}, dr: c.dr ?? null, overrides: c.overrides ?? {}};
  const a = await run(A, {...base, modeRuntime: c.a_mode_runtime ?? null});
  const b = await run(B, {...base, modeRuntime: c.b_mode_runtime ?? null});
  const r = {id: c.id, mode: c.mode, set: c.set ?? null, start: c.start, params: c.params ?? {}, dr: c.dr ?? null, ...compare(a, b)};
  res.cases.push(r);
  i++;
  if (!r.same || i % 10 === 0) console.error('[case]', i, c.mode, c.id, r.same ? 'same' : 'DIFF', JSON.stringify({first: r.first_diff_frame, meta: r.metadata_same, ev: r.events_same, sh: r.shield_frames, reason: r.reason, ms: r.ms}));
  save();
}
const byMode = {};
for (const r of res.cases) { const e = byMode[r.mode] ??= {cases: 0, same: 0, collisions: 0, reverse_runs: 0}; e.cases++; if (r.same) e.same++; if (r.collisions) e.collisions++; if (r.reverse_frames) e.reverse_runs++; }
res.summary = {cases: res.cases.length, same: res.cases.filter(r => r.same).length, fids: res.fids.length, fids_pass: res.fids.filter(r => r.pass).length, by_mode: byMode};
save();
console.error(JSON.stringify(res.summary));
process.exit(res.summary.same === res.summary.cases && res.summary.fids_pass === res.summary.fids ? 0 : 1);
