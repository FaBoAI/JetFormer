#!/usr/bin/env node
// 速さいろいろ r2 の図のための走り(2026-10-14、学習の段 speed r2)。判定・選び方には使わない(図と報告の数字だけ)。
//   node rl/speed_r2/figdata_r2.mjs --policy <r2 の方策.json> --policy-r1 <r1 の方策.json> --eval <r2 の試験の結果.json> --out <figdata.json>
// 元の sim-core.js・mode_runtime.mjs(ブラウザと同じファイル)で:
//   - STD の単周と連続 20 秒を c_v 5 段で(r2 の方策)
//   - 試験の組 SP-P60-R2 の 60 の開始を、r2 の方策と r1 の方策(比べるため)で走らせ直し、フレームごとの s・位置・すき間・速さを残す
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {runModeEpisode, readJson} from '../episode.mjs';
import {SPEED_LEVELS} from '../runtime/mode_runtime.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const pathR2 = opt('--policy'), pathR1 = opt('--policy-r1'), evalPath = opt('--eval'), out = opt('--out');
if (!pathR2 || !pathR1 || !evalPath || !out) { console.error('使い方: node rl/speed_r2/figdata_r2.mjs --policy <json> --policy-r1 <json> --eval <json> --out <json>'); process.exit(2); }
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const P = {r2: {weights: readJson(pathR2), sha: sha(pathR2), path: pathR2}, r1: {weights: readJson(pathR1), sha: sha(pathR1), path: pathR1}};
const ev = readJson(evalPath);
if (ev.policy.sha256 !== P.r2.sha) throw new Error('--eval の結果は別の方策のものです');
const frames = r => r.log.map((q, i) => [+q.s.toFixed(4), +r.trace[i].x.toFixed(4), +r.trace[i].z.toFixed(4), +q.dmin.toFixed(5), +q.v.toFixed(3), +q.ey.toFixed(4)]);
const slim = r => ({reason: r.reason, frames: r.frames, collisions: r.collisions, laps: r.laps.map(l => l.elapsed), valid_episode: r.valid_episode,
  lap_complete: r.lap_complete ?? null, lap_from_stop: r.lap_from_stop ?? null, min_clearance: r.min_clearance ?? null, params: r.params,
  trace: r.trace.map(q => [+q.x.toFixed(4), +q.z.toFixed(4), +q.v.toFixed(3), +q.yaw.toFixed(4)])});
const res = {schema: 'jf-sim-rl-speed-r2-figdata/v1', created_at: new Date().toISOString(), core: 'original',
  policy: {path: pathR2, sha256: P.r2.sha}, policy_r1: {path: pathR1, sha256: P.r1.sha}, eval: evalPath, single: {}, continuous: {}, test: []};
for (const cv of SPEED_LEVELS) {
  const s = await runModeEpisode({mode: 'speed', weights: P.r2.weights, policySha256: P.r2.sha, start: STD, params: {c_v: cv}, episodeMode: 'single'});
  const c = await runModeEpisode({mode: 'speed', weights: P.r2.weights, policySha256: P.r2.sha, start: STD, params: {c_v: cv}, episodeMode: 'continuous'});
  res.single[cv] = slim(s); res.continuous[cv] = {...slim(c), frames: frames(c)};
  console.error(`STD c_v ${cv}: 単周 ${s.lap_from_stop?.toFixed(3)} 秒・連続のすき間 ${(c.min_clearance * 100).toFixed(2)} cm`);
}
for (const r0 of ev.tests['SP-P60']) {
  const row = {id: r0.id, c_v: r0.c_v, start: r0.start, eval_min_clearance: r0.min_clearance};
  for (const k of ['r2', 'r1']) {
    const r = await runModeEpisode({mode: 'speed', weights: P[k].weights, policySha256: P[k].sha, start: r0.start, params: {c_v: r0.c_v}, episodeMode: 'continuous'});
    let m = 0; r.log.forEach((q, i) => { if (q.dmin < r.log[m].dmin) m = i; });
    row[k] = {reason: r.reason, collisions: r.collisions, valid_episode: r.valid_episode, min_clearance: r.min_clearance, laps: r.laps.map(l => l.elapsed),
      frame_min: m, at_min: [r.trace[m].x, r.trace[m].z, r.trace[m].yaw], frames: frames(r)};
  }
  if (Math.abs(row.r2.min_clearance - r0.min_clearance) > 1e-9) throw new Error(`${r0.id}: 走らせ直したすき間が試験の結果と違います`);
  res.test.push(row);
  console.error(`${r0.id} c_v ${r0.c_v}: r2 ${(row.r2.min_clearance * 100).toFixed(2)} cm・r1 ${(row.r1.min_clearance * 100).toFixed(2)} cm`);
}
fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true});
fs.writeFileSync(out, JSON.stringify(res));
console.log(out);
