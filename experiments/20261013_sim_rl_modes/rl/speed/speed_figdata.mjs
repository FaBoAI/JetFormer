#!/usr/bin/env node
// 速さいろいろ(speed)の図のための走り(2026-10-13、sim_rl_modes の学習の段 speed)。
//   node rl/speed/speed_figdata.mjs --policy <cond の方策.json> [--fastest <最速の方策.json>] --out <figdata.json>
// 元の sim-core.js(判定と同じコア)・mode_runtime.mjs(ブラウザと同じファイル)で走らせ、上から見た軌跡と s ごとの速さを書き出す。
//   - STD の単周(停止から 1 周)を c_v 1.0・0.875・0.75・0.625・0.5 で
//   - STD の連続 20 秒を同じ 5 段で(s ごとの速さ・横ずれ)
//   - 比べる: 最速の方策 + 学習しない上限 2.5 m/s(protocol の代わりの方法の一番遅い段)の STD の連続 20 秒
//   - --eval <試験の結果.json>: 試験の組 SP-P60 のうちすき間 3 cm 未満の回を同じ開始・同じ c_v で走らせ直し、最初の 3 秒の軌跡と一番近づいた所を書く(図だけ。選び直しには使わない)
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {runModeEpisode, readJson} from '../episode.mjs';
import {SPEED_LEVELS} from '../runtime/mode_runtime.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const policyPath = opt('--policy'), fastestPath = opt('--fastest'), out = opt('--out'), evalPath = opt('--eval');
if (!policyPath || !out) { console.error('使い方: node rl/speed/speed_figdata.mjs --policy <json> [--fastest <json>] --out <json>'); process.exit(2); }
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const weights = readJson(policyPath), psha = sha(policyPath);
const slim = r => ({reason: r.reason, frames: r.frames, collisions: r.collisions, laps: r.laps.map(l => l.elapsed), valid_episode: r.valid_episode,
  lap_complete: r.lap_complete ?? null, lap_from_stop: r.lap_from_stop ?? null, min_clearance: r.min_clearance ?? null, params: r.params,
  trace: r.trace.map(q => [+q.x.toFixed(4), +q.z.toFixed(4), +q.v.toFixed(3)]),
  log: r.log.map(q => [+q.s.toFixed(4), +q.ey.toFixed(4), +q.v.toFixed(3), +(q.es ?? NaN).toFixed(4), +q.dmin.toFixed(4)])});
const res = {schema: 'jf-sim-rl-speed-figdata/v1', created_at: new Date().toISOString(), core: 'original', policy: {path: policyPath, sha256: psha},
  single: {}, continuous: {}, fallback: null};
for (const cv of SPEED_LEVELS) {
  const t0 = Date.now();
  const s = await runModeEpisode({mode: 'speed', weights, policySha256: psha, start: STD, params: {c_v: cv}, episodeMode: 'single', coreKind: 'original'});
  res.single[cv] = slim(s);
  const c = await runModeEpisode({mode: 'speed', weights, policySha256: psha, start: STD, params: {c_v: cv}, episodeMode: 'continuous', coreKind: 'original'});
  res.continuous[cv] = slim(c);
  console.error(`c_v ${cv}: 単周 ${s.lap_from_stop?.toFixed(3)} 秒・連続 ${c.reason} 周 ${c.laps.map(l => l.elapsed.toFixed(2)).join(', ')}・すき間 ${(c.min_clearance * 100).toFixed(1)} cm(${Date.now() - t0} ms)`);
}
if (fastestPath) {
  const fw = readJson(fastestPath), fsha = sha(fastestPath);
  const c = await runModeEpisode({mode: 'fastest', weights: fw, policySha256: fsha, start: STD, params: {speed_limit: 2.5}, episodeMode: 'continuous', coreKind: 'original'});
  res.fallback = {policy: {path: fastestPath, sha256: fsha}, speed_limit: 2.5, ...slim(c)};
  console.error(`最速 + 上限 2.5 m/s: ${c.reason}・すき間 ${(c.min_clearance * 100).toFixed(2)} cm`);
}
if (evalPath) {
  const ev = readJson(evalPath);
  if (ev.policy.sha256 !== psha) throw new Error('--eval の結果は別の方策のものです');
  res.low_clearance = [];
  for (const r0 of ev.tests['SP-P60'].filter(r => r.min_clearance != null && r.min_clearance < .03)) {
    const r = await runModeEpisode({mode: 'speed', weights, policySha256: psha, start: r0.start, params: {c_v: r0.c_v}, episodeMode: 'continuous', coreKind: 'original'});
    let k = 0; r.log.forEach((q, i) => { if (q.dmin < r.log[k].dmin) k = i; });
    res.low_clearance.push({id: r0.id, c_v: r0.c_v, start: r0.start, min_clearance: r.min_clearance, eval_min_clearance: r0.min_clearance, frame_min: k,
      s_from_start_m: r.log[k].s - r.log[0].s, start_dmin: r.log[0].dmin, trace: r.trace.slice(0, 90).map(q => [+q.x.toFixed(4), +q.z.toFixed(4), +q.v.toFixed(3)]),
      at_min: [r.trace[k].x, r.trace[k].z, r.trace[k].yaw]});
    console.error(`${r0.id}: c_v ${r0.c_v}・開始のすき間 ${(r.log[0].dmin * 100).toFixed(1)} cm・一番近い ${(r.min_clearance * 100).toFixed(2)} cm(${k} フレーム目、開始から ${(r.log[k].s - r.log[0].s).toFixed(2)} m)`);
  }
}
fs.mkdirSync(path.dirname(path.resolve(out)), {recursive: true});
fs.writeFileSync(out, JSON.stringify(res));
console.log(out);
