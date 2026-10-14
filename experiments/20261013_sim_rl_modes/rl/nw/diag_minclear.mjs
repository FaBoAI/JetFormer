// 診断: 検証の組の 1 回を走らせ、最小すき間のフレームの近くを表示する(2026-10-14)
import {buildJobs} from '../eval_modes.mjs';
import {runModeEpisode, readJson} from '../episode.mjs';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const job = buildJobs('no_wall', opt('--set', 'validation')).find(j => j.id === opt('--id'));
const r = await runModeEpisode({mode: 'no_wall', weights: readJson(opt('--policy')), start: job.spec.start, episodeMode: job.spec.episodeMode ?? 'continuous',
  coreKind: 'fast', dr: job.spec.dr ?? null});
console.log(job.id, JSON.stringify(job.spec.start), r.reason, 'minc', r.min_clearance);
let k = 0; r.log.forEach((q, i) => { if (q.dmin < r.log[k].dmin) k = i; });
for (let i = Math.max(0, k - 8); i < Math.min(r.log.length, k + 5); i++) { const q = r.log[i], t = r.trace[i];
  console.log(q.f, 's', q.s.toFixed(2), 'ey', q.ey.toFixed(3), 'epsi', q.epsi.toFixed(3), 'dmin', q.dmin.toFixed(3), 'v', q.v.toFixed(2), 'st', q.st.toFixed(2), 'sp', q.sp.toFixed(2), 'x', t.x.toFixed(2), 'z', t.z.toFixed(2)); }
