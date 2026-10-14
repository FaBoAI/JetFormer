#!/usr/bin/env node
// 経路いろいろ r4b の比べる値だけを走らせる(2026-10-14 夜、line r4 の段のやり直し。判定・選び方には使わない。試験の判定のあとに使う)。
// r3/line4/compare_line_r4.mjs の写し。eval_line_r4b.mjs(固めたときのまま変えない)の組の作り方・判定をそのまま読み込み、1 回の走らせ方(runJob)を写して、
// シールドの値を --shield-config で替えられるようにしたもの(eval_line_r4b.mjs は --shield-config を下調べの組だけに限っている)。
//   node r3/line4b/compare_line_r4b.mjs --set test --policy <policy.json> --shield-config <json> [--jobs N] [--keep-path] --out <out.json>
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson} from './episode_ln4.mjs';
import {buildJobs, judge, stallOf, shieldModeRuntime, P4, P4B, P2} from './eval_line_r4b.mjs';
import {LOOP_LENGTH} from './rt4b/features.mjs';
import {quantile, sceneFromEpisode, binProfile, steerUs} from './variation_metrics.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
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

async function main() {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; }, flag = n => args.includes(n);
  const set = opt('--set', 'validation'), core = opt('--core', set === 'validation' ? 'fast' : 'original');
  const jobs = +opt('--jobs', '1'), only = opt('--only')?.split(','), out = opt('--out'), policyPath = opt('--policy');
  const noShield = flag('--no-shield');
  if (!policyPath && set !== 'fastest') { console.error('使い方: node compare_line_r4b.mjs --policy <json> --set dev|validation|test|fresh|regression|fastest'); process.exit(2); }
  const cfgPath = opt('--shield-config');
  // 比べる値だけ: --shield-config の値でシールドをかける(試験の組でも)。結果は compare: true で、判定には使わない
  const SH = noShield || set === 'fastest' ? null : cfgPath ? {modeRuntime: {shield: true, shield_params: (c => c.shield_params ?? c)(readJson(cfgPath))}, source: cfgPath, sha256: sha256File(cfgPath)}
    : shieldModeRuntime({set: set === 'test' ? 'test' : set});
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
    const tmp = fs.mkdtempSync(path.join(process.env.JF_TMP ?? '/tmp', 'cmp_line_r4b_'));
    await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
      const ch = fork(fileURLToPath(import.meta.url), [...args.filter((a, i) => !['--jobs', '--out'].includes(a) && !['--jobs', '--out'].includes(args[i - 1])), '--shard', `${k}/${jobs}`, '--rows-out', `${tmp}/${k}.json`], {stdio: 'inherit'});
      ch.on('exit', c => c === 0 ? res() : rej(new Error('子プロセスが失敗しました ' + c)));
    })));
    for (let k = 0; k < jobs; k++) rows.push(...JSON.parse(fs.readFileSync(`${tmp}/${k}.json`, 'utf8')));
    fs.rmSync(tmp, {recursive: true});
    rows.sort((a, b) => a._j - b._j);
  } else for (const [j, job] of J.entries()) { const r = await runJob(job, ctx); r._j = j; rows.push(r); if ((j + 1) % 20 === 0) console.error(`${j + 1}/${J.length}`); }
  const verdict = judge(set, rows);
  const shieldMjs = path.join(here, 'rt4b/shield.mjs'), modeRt = path.join(here, 'rt4b/mode_runtime.mjs');
  const res = {schema: 'jf-sim-rl-line-r4b-eval/v1', compare: true, compare_note: '比べる値(判定・選び方には使わない)', created_at: new Date().toISOString(), set, core,
    policy: ctx.policy ? {path: ctx.policy.path, sha256: ctx.policy.sha, schema: ctx.policy.weights.schema, obs_dim: ctx.policy.weights.obs_dim, meta: ctx.policy.weights.meta ?? null} : null,
    shield: SH ? {mode_runtime: SH.modeRuntime, source: SH.source, source_sha256: SH.sha256, shield_mjs_sha256: sha256File(shieldMjs), mode_runtime_sha256: sha256File(modeRt)} : null,
    protocol_line_r4_sha256: P4?._sha256 ?? null, protocol_line_r4b_sha256: P4B?._sha256 ?? null, protocol_line_r2_sha256: P2._sha256, set_meta: meta, runs: rows.length, ...verdict, wall_s: (Date.now() - t0) / 1000,
    tests: Object.fromEntries([...new Set(rows.map(r => r.test))].map(t => [t, rows.filter(r => r.test === t).map(r => { const {vm_frames, st_time, tracking_abs, tracking_s, _j, ...rest} = r; return rest; })]))};
  if (out) { fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true}); fs.writeFileSync(out, JSON.stringify(res)); }
  console.log(JSON.stringify({set, runs: rows.length, all_judged_pass: res.all_judged_pass, wall_s: res.wall_s,
    judged: Object.fromEntries(Object.entries(res.judged).map(([k, v]) => [k, {pass: v.pass, value: v.value}])),
    hard_min: res.reported.hard_min_clearance, hard_fail: res.reported.hard_fail, by_set: res.reported.by_set}));
}
if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
