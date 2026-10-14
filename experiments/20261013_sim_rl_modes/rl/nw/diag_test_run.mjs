// 診断(2026-10-14): 試験の組の 1 回(id 指定)を元のコアで走らせ、最小すき間の近くとシールドの動きを表示する。判定には使わない
import {buildJobs} from '../eval_modes.mjs';
import {runModeEpisode, readJson} from '../episode.mjs';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const job = buildJobs('no_wall', opt('--set', 'test')).find(j => j.test === opt('--test', 'NW-R200') && j.id === opt('--id'));
const r = await runModeEpisode({mode: 'no_wall', weights: readJson(opt('--policy')), start: job.spec.start, episodeMode: job.spec.episodeMode ?? 'continuous',
  coreKind: 'original', dr: job.spec.dr ?? null, overrides: args.includes('--no-shield') ? {shield: false} : {}});
console.log(job.id, JSON.stringify(job.spec.start), r.reason, 'minc', r.min_clearance, 'sh', r.shield_frames, 'col', JSON.stringify(r.collision));
let k = 0; r.log.forEach((q, i) => { if (q.dmin < r.log[k].dmin) k = i; });
const show = new Set(); for (let i = k - 30; i <= k + 5; i++) show.add(i); r.log.forEach((q, i) => { if (q.sh) show.add(i); });
for (const i of [...show].filter(i => i >= 0 && i < r.log.length).sort((a, b) => a - b)) { const q = r.log[i], t = r.trace[i];
  console.log(q.f, 's', q.s.toFixed(2), 'ey', q.ey.toFixed(3), 'epsi', q.epsi.toFixed(3), 'dmin', q.dmin.toFixed(3), 'v', q.v.toFixed(2), 'st', q.st.toFixed(2), 'sp', q.sp.toFixed(2), 'ph', q.ph,
    q.shield ? `fb=${q.shield.fb} nom=${q.shield.nom?.toFixed(3)} pess=${q.shield.pess?.toFixed(3)}` : '', 'x', t.x.toFixed(2), 'z', t.z.toFixed(2)); }
