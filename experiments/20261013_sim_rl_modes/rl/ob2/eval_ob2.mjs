#!/usr/bin/env node
// 障害物よけ r2 の「選びの組」の評価(2026-10-14 学習の段 obstacle r2、protocol_obstacle_r2.json sets.selection・selection)。
//   OB-V100-R2 : 選びの配置 50(rl/ob2/selection_layouts.json、種 1,830,000〜)× 2 開始(STD と、記録の開始の分布の Halton 15,001 番から、
//                配置の開始の条件 = 開始の前後 1.5 m に障害物なし・開始のすき間 > 0 を満たすもの)、連続 20 秒
//   OB-EMPTY   : 障害物なしの STD の単周(O4)
//   EMPTY-VAR6 : 障害物なしの単周を STD から ±3 cm・±3° ずらした 6 つ(確認 r1 の指摘 7: O4 が STD ちょうどでしか測られない。選びの好みに使う)
// 判定の式は protocol.json acceptance.obstacle と同じ(O1 ≥ 95%・O2 有効な回の最小すき間 ≥ 3 cm・O3 1 周目の中央値 ≤ 10.5 秒・O4 ≤ 9.0 秒)。
// 1 回の走らせ方は rl/eval_modes.mjs と同じ(rl/episode.mjs runModeEpisode = mode_runtime.mjs + すき間の測り直し)。コアは既定で高速版(検証の組と同じ扱い)。
//   node rl/ob2/eval_ob2.mjs --policy <policy.json> --out <結果.json> [--core fast|original] [--jobs N] [--set selection|validation27]
//   (--set validation27 は protocol の検証の組と同じ 13 配置 × 2 開始。報告の比べ用)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fork} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson} from '../episode.mjs';
import {haltonStarts} from '../eval_modes.mjs';
import {passageStats, summarizePassages, quantile} from '../ob/ob_common.mjs';
import {VM3, sceneFromEpisode} from '../variation_metrics.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), rl = path.resolve(here, '..');
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
export const VAR6 = [['L+3cm', .03, 0], ['L-3cm', -.03, 0], ['yaw+3', 0, 3], ['yaw-3', 0, -3], ['L+3cm_yaw-3', .03, -3], ['L-3cm_yaw+3', -.03, 3]];
const sha256File = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

export function buildJobs(set = 'selection') {
  const J = [], add = (test, id, spec) => J.push({test, id, spec});
  if (set === 'selection') {
    const lays = readJson(path.join(here, 'selection_layouts.json')).layouts;
    const hs = haltonStarts('obstacle', 'validation', lays.length, {layoutFor: k => lays[k], from: 15001});
    if (hs.length !== lays.length) throw new Error('eval_ob2: Halton の開始が足りません');
    lays.forEach((lay, i) => {
      add('OB-V100-R2', `${lay.id}-STD`, {start: {...STD, seed: 2830000 + 2 * i}, layout: lay});
      add('OB-V100-R2', `${lay.id}-H`, {start: {...hs[i], seed: 2830000 + 2 * i + 1}, layout: lay});
    });
  } else if (set === 'validation27') {
    const lays = readJson(path.join(rl, 'obstacle_sets/validation.json')).layouts.slice(0, 13);
    const hs = haltonStarts('obstacle', 'validation', lays.length, {layoutFor: k => lays[k]});
    lays.forEach((lay, i) => {
      add('OB-V100-R2', `${lay.id}-STD`, {start: {...STD, seed: 806000 + 2 * i}, layout: lay});
      add('OB-V100-R2', `${lay.id}-H`, {start: {...hs[i], seed: 806000 + 2 * i + 1}, layout: lay});
    });
  } else throw new Error('eval_ob2: 知らない組 ' + set);
  add('OB-EMPTY', 'single', {start: STD, episodeMode: 'single'});
  for (const [id, off, yaw] of VAR6) add('EMPTY-VAR6', id, {start: {gate: 0, left_offset_m: off, left_yaw_deg: yaw, seed: 1}, episodeMode: 'single'});
  return J;
}

async function runJob(job, ctx) {
  const sp = job.spec, t0 = Date.now();
  const r = await runModeEpisode({mode: 'obstacle', weights: ctx.weights, policySha256: ctx.sha, start: sp.start, params: {layout_id: sp.layout?.id ?? null},
    episodeMode: sp.episodeMode ?? 'continuous', coreKind: ctx.core, layout: sp.layout ?? null, audit: true});
  const row = {test: job.test, id: job.id, layout_id: sp.layout?.id ?? null, n_obstacles: sp.layout?.obstacles?.length ?? 0, start: sp.start,
    invalid_spawn: !!r.invalid_spawn};
  if (r.invalid_spawn) return row;
  const run = r.log.filter(q => q.v > .3).map(q => q.v);
  Object.assign(row, {reason: r.reason, frames: r.frames, collisions: r.collisions, collision: r.collision, laps: r.laps.map(l => l.elapsed), valid_laps: r.valid_laps,
    valid_episode: r.valid_episode, lap_complete: r.lap_complete ?? null, lap_from_stop: r.lap_from_stop ?? null, min_clearance: r.min_clearance ?? null,
    audit_err: r.audit?.max_pos_err ?? null, gate_miss: r.gate_miss, speed_mean_running: run.length ? run.reduce((a, b) => a + b, 0) / run.length : null});
  if (sp.layout) {
    const ps = passageStats(r, sp.layout.obstacles);
    row.passages = ps.map(p => ({k: p.obstacle, side: p.side, completed: p.completed, cl: p.min_clearance_m, v: p.v_at_min, gap: p.lateral_gap_m}));
    if (r.valid_episode) { const sc = sceneFromEpisode(r); row.ey_scene = {s0: sc.s0, f: sc.frames.map(q => [+q.s.toFixed(4), +q.ey.toFixed(4)])}; }
  }
  row.wall_s = (Date.now() - t0) / 1000;
  return row;
}

export function judge(rows) {
  const p = rows.filter(r => r.test === 'OB-V100-R2'), e = rows.find(r => r.test === 'OB-EMPTY'), v6 = rows.filter(r => r.test === 'EMPTY-VAR6');
  const ok = p.filter(r => r.valid_episode), J = {};
  J.O1_valid = {value: `${ok.length}/${p.length}`, n: ok.length, pass: ok.length >= Math.ceil(.95 * p.length)};
  const mc = ok.length ? Math.min(...ok.map(r => r.min_clearance)) : null;
  J.O2_clearance = {value: mc, pass: mc != null && mc >= .03};
  const l1 = ok.map(r => r.laps[0]).filter(Number.isFinite);
  J.O3_lap1_median = {value: quantile(l1, .5), pass: l1.length > 0 && quantile(l1, .5) <= 10.5};
  J.O4_empty = {value: e?.lap_from_stop ?? null, pass: e?.lap_complete === true && e.lap_from_stop <= 9.0};
  const v6t = v6.map(r => r.lap_complete ? r.lap_from_stop : Infinity);
  const rep = {var6: {laps: Object.fromEntries(v6.map((r, i) => [r.id, v6t[i]])), max: v6t.length ? Math.max(...v6t) : null, all_le_9: v6t.every(x => x <= 9.0)}};
  const fails = p.filter(r => !r.valid_episode);
  rep.failures = {n: fails.length, reasons: fails.reduce((a, r) => ((a[r.reason] = (a[r.reason] ?? 0) + 1), a), {}),
    collision_kinds: fails.reduce((a, r) => (r.collision ? ((a[r.collision.kind] = (a[r.collision.kind] ?? 0) + 1), a) : a), {}),
    early_le_2_1s: fails.filter(r => r.frames <= 63).length, ids: fails.map(r => r.id)};
  rep.valid_by_start = {STD: `${ok.filter(r => r.id.endsWith('-STD')).length}/${p.filter(r => r.id.endsWith('-STD')).length}`,
    H: `${ok.filter(r => r.id.endsWith('-H')).length}/${p.filter(r => r.id.endsWith('-H')).length}`};
  rep.valid_by_count = [1, 2, 3, 4].map(n => `${ok.filter(r => r.n_obstacles === n).length}/${p.filter(r => r.n_obstacles === n).length}`);
  const cls = ok.map(r => r.min_clearance).sort((a, b) => a - b);
  rep.valid_min_clearances_lowest = cls.slice(0, 5);
  rep.valid_below_3cm = cls.filter(x => x < .03).length;
  const passes = p.flatMap(r => (r.passages ?? []).map(q => ({...q, min_clearance_m: q.cl, v_at_min: q.v, lateral_gap_m: q.gap})));
  rep.passages = summarizePassages(passes);
  const scenes = ok.filter(r => r.ey_scene).map(r => ({frames: r.ey_scene.f.map(([s, ey]) => ({s, ey})), s0: r.ey_scene.s0}));
  rep.VM3_valid_runs = scenes.length >= 2 ? VM3(scenes) : null;
  rep.lap1_valid = {median: quantile(l1, .5), max: l1.length ? Math.max(...l1) : null};
  rep.speed_mean_running_median = quantile(ok.map(r => r.speed_mean_running), .5);
  const judged = Object.values(J);
  return {judged: J, n_pass: judged.filter(x => x.pass).length, all_judged_pass: judged.every(x => x.pass), reported: rep};
}

async function main() {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
  const policyPath = opt('--policy'), out = opt('--out'), core = opt('--core', 'fast'), jobs = +opt('--jobs', '1'), set = opt('--set', 'selection');
  if (!policyPath) { console.error('使い方: node rl/ob2/eval_ob2.mjs --policy <json> --out <json> [--core fast|original] [--jobs N]'); process.exit(2); }
  const weights = readJson(policyPath), sha = sha256File(policyPath), ctx = {weights, sha, core};
  const all = buildJobs(set), shard = opt('--shard');
  if (shard) {
    const [k, n] = shard.split('/').map(Number), rows = [];
    for (let i = k; i < all.length; i += n) rows.push(await runJob(all[i], ctx));
    fs.writeFileSync(opt('--rows-out'), JSON.stringify(rows));
    return;
  }
  const t0 = Date.now();
  let rows = [];
  if (jobs > 1) {
    const tmp = fs.mkdtempSync(path.join(path.dirname(out ?? '/tmp/x'), '.eval_ob2_'));
    await Promise.all(Array.from({length: jobs}, (_, k) => new Promise((res, rej) => {
      const ch = fork(fileURLToPath(import.meta.url), ['--policy', policyPath, '--core', core, '--set', set, '--shard', `${k}/${jobs}`, '--rows-out', `${tmp}/${k}.json`], {stdio: 'inherit'});
      ch.on('exit', c => c === 0 ? res() : rej(new Error('eval_ob2: 子の終了コード ' + c)));
    })));
    for (let k = 0; k < jobs; k++) rows.push(...JSON.parse(fs.readFileSync(`${tmp}/${k}.json`, 'utf8')));
    fs.rmSync(tmp, {recursive: true, force: true});
    const order = new Map(all.map((j, i) => [j.test + '/' + j.id, i]));
    rows.sort((a, b) => order.get(a.test + '/' + a.id) - order.get(b.test + '/' + b.id));
  } else for (const j of all) rows.push(await runJob(j, ctx));
  const verdict = judge(rows);
  const res = {schema: 'jf-sim-rl-obstacle-r2-selection-eval/v1', created_at: new Date().toISOString(), set, core, policy: {path: policyPath, sha256: sha, meta: weights.meta ?? null},
    selection_layouts_sha256: sha256File(path.join(here, 'selection_layouts.json')), runs: rows.length, ...verdict, wall_s: (Date.now() - t0) / 1000,
    rows: rows.map(({ey_scene, ...r}) => r)};
  if (out) fs.writeFileSync(out, JSON.stringify(res));
  console.log(JSON.stringify({set, core, runs: rows.length, wall_s: res.wall_s, all_judged_pass: res.all_judged_pass, n_pass: res.n_pass,
    judged: Object.fromEntries(Object.entries(res.judged).map(([k, v]) => [k, v.value])), var6_max: res.reported.var6.max, failures: res.reported.failures.reasons,
    early: res.reported.failures.early_le_2_1s, VM3: res.reported.VM3_valid_runs?.mean_abs_dev ?? null, passes: res.reported.passages}));
}
if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
