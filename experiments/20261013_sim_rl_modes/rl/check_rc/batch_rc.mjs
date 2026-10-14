// 離脱復活の確認 r1: 一覧の一部(shard)を走らせ、1 回 1 行の JSONL に書く(2026-10-14)。
//   node rl/check_rc/batch_rc.mjs <jobs.json> <shard> <nshards> <out.jsonl> [set,set,...]
import fs from 'node:fs';
import {runOne} from './run_rc.mjs';
const [jobsPath, shard, nsh, outPath, sets] = process.argv.slice(2);
const only = sets ? new Set(sets.split(',')) : null;
const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter(j => !only || only.has(j.set));
const fd = fs.openSync(outPath, 'w');
const r4 = x => Math.round(x * 1e4) / 1e4;
let n = 0;
const t0 = Date.now();
for (let i = +shard; i < jobs.length; i += +nsh) {
  const j = jobs[i];
  const r = await runOne(j.spec);
  if (r.invalid_spawn) { fs.writeSync(fd, JSON.stringify({set: j.set, id: j.id, invalid_spawn: true}) + '\n'); continue; }
  const F = r.frames_data;
  let dS = 0, dE = 0;
  for (const q of F) { dS = Math.max(dS, Math.abs(q.s - q.rs)); dE = Math.max(dE, Math.abs(q.ey - q.rey)); }
  const row = {set: j.set, id: j.id, spec: j.spec, variation: r.variation, reason: r.reason, frames: r.frames, collisions: r.collisions,
    collision: r.collision ? {t: r.collision.time, collider: r.collision.collider, kind: r.collision.kind} : null,
    passed_gates: r.passed_gates, completed_laps: r.completed_laps, laps: r.laps, valid_episode: r.valid_episode, valid_laps: r.valid_laps,
    lap_from_stop: r.lap_from_stop ?? null, events: r.events, virtual_misses: r.virtual_misses, p0: r.p0, labels: r.labels, kick: r.kick,
    min_clearance: r.min_clearance, min_frame: r.min_frame ?? null, min_kind: r.min_kind ?? null, audit_pos_err: r.audit_pos_err, overlap_frames: r.overlap_frames,
    own_crossings: r.own_crossings, proj_vs_runtime: {ds: dS, dey: dE},
    pf: {s: F.map(q => r4(q.s)), ey: F.map(q => r4(q.ey)), epsi: F.map(q => r4(q.epsi)), v: F.map(q => Math.round(q.v * 1e3) / 1e3),
      x: F.map(q => r4(q.x)), z: F.map(q => r4(q.z)),
      st: F.map(q => r4(q.st)), sp: F.map(q => Math.round(q.sp * 1e3) / 1e3), l: F.map(q => q.l), ev: F.map(q => q.ev), ch: F.map(q => q.ch), g: F.map(q => q.g),
      cl: r.clear_pf ? r.clear_pf.map(c => r4(c)) : null}};
  fs.writeSync(fd, JSON.stringify(row) + '\n');
  n++;
  if (n % 25 === 0) console.log(`[shard ${shard}] ${n} 回 ${((Date.now() - t0) / 1000).toFixed(0)} 秒`);
}
fs.closeSync(fd);
console.log(`[shard ${shard}] 終わり ${n} 回 ${((Date.now() - t0) / 1000).toFixed(0)} 秒`);
