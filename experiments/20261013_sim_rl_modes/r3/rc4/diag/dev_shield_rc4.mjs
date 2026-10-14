// 離脱復活 r4 の下調べ(2026-10-14、学習の段 recover r4。選び・試験には使わない): シールドの「すくみをほどく(unstick)」を離脱復活で使うか。
// 確認 recover r3 の止まり 14 の開始(シールドが止めたまま・方策の止まり)と、新しい開始(下調べの乱数 mulberry32(0x6dd0e401)、種 8,900,000 + i)を、
// r3 の方策・r4 の走行中の仕組み(rc4/rt)で、シールドの値だけ替えて走らせる。
//   node r3/rc4/diag/dev_shield_rc4.mjs --cases stall_cases.json --fresh 40 --variants A,B,C --policy <policy.json> --out rows.jsonl [--perturbed] [--shard k/n]
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runModeEpisode, readJson, BASE_WORLD} from '../episode_rc4.mjs';
import {CATALOGUE} from '../rt/catalogue.mjs';
import {activeWorldFor} from '/home/jetson/jetformer/experiments/20261013_sim_rl_modes/snapshot/rl/env_core.mjs';
import {prepareGeometry, clearance, LOOP_LENGTH} from '../rt/features.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const VARIANTS = {
  A: {},   // r3 のまま(unstick 切り)
  B: {unstick: {enabled: true, after: 15, speeds: [.5, 1.0], c: .03, hold: 90, exit_margin: .01}},
  C: {unstick: {enabled: true, after: 15, speeds: [.5, 1.0], c: .02, hold: 90, exit_margin: .01}},
  D: {unstick: {enabled: true, after: 10, speeds: [.3, .6, 1.0], c: .03, hold: 120, exit_margin: .02}},
};
const DIST = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution;
const geom = prepareGeometry(BASE_WORLD);
const startClear = st => { const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg); return clearance(geom, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5); };
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const cases = opt('--cases') ? readJson(opt('--cases')).map(c => ({id: c.id, start: c.start})) : [];
const nFresh = +opt('--fresh', '0'), u = mulberry32(0x6dd0e401);
for (let i = 0, g = 0; i < nFresh && g < 100000; g++) {
  const st = {gate: Math.min(32, Math.floor(u() * 33)), left_offset_m: DIST.left_offset_m[0] + (DIST.left_offset_m[1] - DIST.left_offset_m[0]) * u(),
    left_yaw_deg: DIST.left_yaw_deg[0] + (DIST.left_yaw_deg[1] - DIST.left_yaw_deg[0]) * u()};
  if (!(startClear(st) > 1e-9)) continue;
  cases.push({id: `DEV${i}`, start: {...st, seed: 8900000 + i}}); i++;
}
const variants = (opt('--variants', 'A,B')).split(',');
const policyPath = opt('--policy'), weights = readJson(policyPath), sha = crypto.createHash('sha256').update(fs.readFileSync(policyPath)).digest('hex');
const perturbed = args.includes('--perturbed');
const [SK, SN] = (opt('--shard', '0/1')).split('/').map(Number);
const out = fs.openSync(opt('--out'), 'a');
const jobs = [];
for (const c of cases) for (const v of variants) jobs.push({c, v});
for (let j = SK; j < jobs.length; j += SN) {
  const {c, v} = jobs[j], t0 = Date.now();
  const r = await runModeEpisode({mode: 'recover', weights, policySha256: sha, start: c.start, shieldParams: VARIANTS[v],
    overrides: perturbed ? {} : {perturbation: false}});
  if (r.invalid_spawn) { fs.writeSync(out, JSON.stringify({id: c.id, v, invalid_spawn: true}) + '\n'); continue; }
  const log = r.log;
  let run = 0, maxRun = 0;
  for (let f = 30; f < log.length; f++) { if (Math.abs(log[f].v) < .2) { run++; maxRun = Math.max(maxRun, run); } else run = 0; }
  const fb = {};
  for (const q of log) if (q.shield?.fb) fb[q.shield.fb] = (fb[q.shield.fb] ?? 0) + 1;
  fs.writeSync(out, JSON.stringify({id: c.id, v, start: c.start, start_clearance: startClear(c.start), reason: r.reason, frames: r.frames, collisions: r.collisions,
    valid_episode: r.valid_episode, stall_run: maxRun, stall: maxRun >= 30, progress_m: log.at(-1).s - log[0].s, min_clearance: r.min_clearance,
    dmin_after_1s: Math.min(...log.slice(30).map(q => q.dmin)), shield_frames: r.shield_frames, fallbacks: fb, events: r.events.length,
    gate_miss: r.gate_miss, laps: r.laps.map(l => l.elapsed), wall_s: (Date.now() - t0) / 1000}) + '\n');
}
fs.closeSync(out);
