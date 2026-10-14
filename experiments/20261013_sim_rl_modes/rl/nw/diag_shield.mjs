// no_wall の学習の段の診断(2026-10-13): シールドありの 1 回を走らせ、シールドが動いたフレームと衝突の前を表示する
//   node rl/nw/diag_shield.mjs --policy <policy.json> [--test NW-N60] [--idx 0] [--core fast] [--no-shield]
import fs from 'node:fs';
import {buildJobs} from '../eval_modes.mjs';
import {runModeEpisode, readJson} from '../episode.mjs';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const J = buildJobs('no_wall', opt('--set', 'validation')).filter(j => j.test === opt('--test', 'NW-N60'));
const job = J[+opt('--idx', '0')];
const weights = readJson(opt('--policy'));
const r = await runModeEpisode({mode: 'no_wall', weights, start: job.spec.start, episodeMode: job.spec.episodeMode ?? 'continuous', coreKind: opt('--core', 'fast'),
  dr: job.spec.dr ?? null, overrides: args.includes('--no-shield') ? {shield: false} : {}});
console.log(job.id, r.reason, r.frames, 'col', r.collisions, JSON.stringify(r.collision), 'minc', r.min_clearance, 'sh', r.shield_frames);
const L = r.log, cf = r.frames;
const show = new Set();
L.forEach((q, i) => { if (q.sh) for (let k = i - 3; k <= i + 3; k++) show.add(k); });
for (let k = cf - 25; k < cf; k++) show.add(k);
for (const i of [...show].filter(i => i >= 0 && i < L.length).sort((a, b) => a - b)) {
  const q = L[i], t = r.trace[i];
  console.log(`${String(q.f).padStart(4)} s=${q.s.toFixed(2)} ey=${q.ey.toFixed(3)} dmin=${q.dmin.toFixed(3)} v=${q.v.toFixed(2)} ph=${q.ph} st=${q.st.toFixed(3)} sp=${q.sp.toFixed(2)} a=${q.a0.toFixed(2)},${q.a1.toFixed(2)} ${q.shield ? JSON.stringify(q.shield) : ''} x=${t?.x.toFixed(2)} z=${t?.z.toFixed(2)}`);
}
