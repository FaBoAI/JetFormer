#!/usr/bin/env node
// 経路いろいろ(line)の図のための軌跡を書き出す(2026-10-13、学習の段 line)。判定には使わない(判定は rl/eval_modes.mjs)。
//   node rl/line_traj.mjs --policy <policy.json> --out <traj.json> [--set test|validation] [--p60 15] [--core original|fast]
// 試験の組と同じ回(rl/eval_modes.mjs の buildJobs の LN-FAM6 の 6 回と、LN-P60 の最初の N 回 = family_plan_15 の 1 回分)を
// 同じ runModeEpisode(元の sim-core.js + mode_runtime.mjs)で走らせ、各フレームの x・z・速さ と s・e_y・e*(s) を残す。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {buildJobs} from './eval_modes.mjs';
import {runModeEpisode, readJson} from './episode.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const policyPath = opt('--policy'), out = opt('--out'), set = opt('--set', 'test'), nP = +opt('--p60', '15'), core = opt('--core', 'original');
if (!policyPath || !out) { console.error('使い方: node rl/line_traj.mjs --policy <json> --out <json> [--set test] [--p60 15]'); process.exit(2); }
const weights = readJson(policyPath), sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const jobs = buildJobs('line', set);
const sel = [...jobs.filter(j => j.test === 'LN-FAM6'), ...jobs.filter(j => j.test === 'LN-P60').slice(0, nP)];
const runs = [];
for (const job of sel) {
  const sp = job.spec;
  const r = await runModeEpisode({mode: 'line', weights, policySha256: sha, start: sp.start, params: sp.params ?? {}, coreKind: core, audit: true});
  const r4 = v => v == null ? null : +(+v).toFixed(4);
  runs.push({test: job.test, id: job.id, family: sp.family ?? r.params?.line?.family ?? null, params: r.params, start: sp.start,
    valid_episode: r.valid_episode, reason: r.reason, collisions: r.collisions, laps: r.laps.map(l => l.elapsed), min_clearance: r.min_clearance ?? null,
    x: r.trace.map(q => r4(q.x)), z: r.trace.map(q => r4(q.z)), v: r.trace.map(q => +q.v.toFixed(3)),
    s: r.log.map(q => r4(q.s)), ey: r.log.map(q => r4(q.ey)), es: r.log.map(q => r4(q.es)), dmin: r.log.map(q => r4(q.dmin))});
  console.error(`${job.test} ${job.id} 有効 ${r.valid_episode} 1 周目 ${r.laps[0]?.elapsed?.toFixed(3) ?? '-'} すき間 ${r.min_clearance?.toFixed(4)}`);
}
fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true});
fs.writeFileSync(out, JSON.stringify({schema: 'jf-sim-rl-line-traj/v1', created_at: new Date().toISOString(), policy: {path: policyPath, sha256: sha}, set, core,
  note: '軌跡の図のため(判定は rl/eval_modes.mjs の結果)。x・z・v は sim の trace(ステップ後)、s・ey・es・dmin は教師の log(そのフレームの観測の時点)', runs}));
console.log(JSON.stringify({out, runs: runs.length}));
