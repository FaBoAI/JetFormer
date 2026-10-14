// 離脱復活 r4 の確認: 一覧の一部(shard)を走らせ、1 回 1 行の JSONL に書く(2026-10-15、確認 r3 の ck_batch.mjs の写し)
//   node ck_batch.mjs <jobs.json> <shard> <nshards> <out.jsonl> [set,set,...]
import fs from 'node:fs';
import {runOne} from './ck_run.mjs';
const [jobsPath, shard, nsh, outPath, sets] = process.argv.slice(2);
const only = sets ? new Set(sets.split(',')) : null;
const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter(j => !only || only.has(j.set));
// 途中から続ける: 出力に既にある id は飛ばす
const done = new Set();
if (fs.existsSync(outPath)) for (const l of fs.readFileSync(outPath, 'utf8').split('\n')) if (l.trim()) { try { const r = JSON.parse(l); done.add(r.set + '/' + r.id); } catch {} }
const fd = fs.openSync(outPath, 'a');
const r4 = x => Math.round(x * 1e4) / 1e4;
let n = 0; const t0 = Date.now();
for (let i = +shard; i < jobs.length; i += +nsh) {
  const j = jobs[i];
  if (done.has(j.set + '/' + j.id)) continue;
  const r = await runOne(j.spec);
  if (r.invalid_spawn) { fs.writeSync(fd, JSON.stringify({set: j.set, id: j.id, spec: j.spec, invalid_spawn: true, start_clearance: r.start_clearance}) + '\n'); continue; }
  const F = r.frames_data;
  let dS = 0, dE = 0;
  for (const q of F) { dS = Math.max(dS, Math.abs(q.s - q.rs)); dE = Math.max(dE, Math.abs(q.ey - q.rey)); }
  const row = {set: j.set, id: j.id, spec: j.spec, policy_sha: r.policy_sha.slice(0, 12), variation: r.variation, start_clearance: r.start_clearance, reason: r.reason, frames: r.frames, collisions: r.collisions,
    collision: r.collision, passed_gates: r.passed_gates, completed_laps: r.completed_laps, laps: r.laps, valid_episode: r.valid_episode, lap_from_stop: r.lap_from_stop ?? null,
    events: r.events, p0: r.p0, labels: r.labels, kick: r.kick, final: r.final, shield_frames: r.shield_frames, meta_pert: r.metadata_perturbation, meta_shield: r.metadata_shield,
    min_clearance: r.min_clearance, min_frame: r.min_frame ?? null, min_kind: r.min_kind ?? null, audit_pos_err: r.audit_pos_err, overlap_frames: r.overlap_frames,
    proj_vs_runtime: {ds: dS, dey: dE},
    pf: {s: F.map(q => r4(q.s)), ey: F.map(q => r4(q.ey)), epsi: F.map(q => r4(q.epsi)), v: F.map(q => Math.round(q.v * 1e3) / 1e3),
      x: F.map(q => r4(q.x)), z: F.map(q => r4(q.z)), st: F.map(q => r4(q.st)), sp: F.map(q => Math.round(q.sp * 1e3) / 1e3),
      pst: F.map(q => r4(q.pst)), psp: F.map(q => Math.round(q.psp * 1e3) / 1e3),
      l: F.map(q => q.l), ph: F.map(q => q.ph), ev: F.map(q => q.ev), sh: F.map(q => q.sh), g: F.map(q => q.g), lap: F.map(q => q.lap),
      fb: F.map(q => q.fb), cl: r.clear_pf ? r.clear_pf.map(c => r4(c)) : null}};
  fs.writeSync(fd, JSON.stringify(row) + '\n');
  n++;
  if (n % 20 === 0) console.log(`[shard ${shard}] ${n} 回 ${((Date.now() - t0) / 1000).toFixed(0)} 秒`);
}
fs.closeSync(fd);
console.log(`[shard ${shard}] 終わり ${n} 回 ${((Date.now() - t0) / 1000).toFixed(0)} 秒`);
