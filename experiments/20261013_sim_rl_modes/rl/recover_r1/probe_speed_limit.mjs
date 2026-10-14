// 離脱復活の学習の段(r1)の下調べ: 方策に速さの上限を掛けると、記録の乱しのあとで有効な回が増えるか(試験・検証の組と重ならない種・開始で)。
//   node rl/recover_r1/probe_speed_limit.mjs --policy <policy.json> --limits none,4.0,3.5,3.0 --n 40 --out <結果.json>
import fs from 'node:fs';
import {runModeEpisode, readJson} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const policyPath = args.policy, n = +(args.n ?? 40), limits = (args.limits ?? 'none').split(',');
const weights = readJson(policyPath);
const dist = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution;
const out = {policy: policyPath, n, by_limit: {}};
for (const lim of limits) {
  const rows = [];
  for (let i = 0; i < n; i++) {
    const st = haltonStart(40001 + i, dist);
    const r = await runModeEpisode({mode: 'recover', weights, start: {...st, seed: 500000 + i}, params: lim === 'none' ? {} : {speed_limit: +lim},
      coreKind: 'fast', audit: false, keepLog: true});
    if (r.invalid_spawn) continue;
    const coll = r.collisions ? r.frames : null;
    const failEv = coll != null ? [...r.events].reverse().find(e => e.ps <= coll) : null;
    const lap1 = r.laps[0]?.elapsed ?? null;
    rows.push({i, valid: r.valid_episode, frames: r.frames, events: r.events.length, fail: failEv ? {type: failEv.type, cls: failEv.cls, s: +(failEv.s % 26.1196).toFixed(2), dt: coll - failEv.ps} : null,
      lap1, vmean: r.log.reduce((a, q) => a + q.v, 0) / r.log.length});
  }
  const v = rows.filter(r => r.valid).length;
  const fails = rows.filter(r => r.fail).map(r => `${r.fail.type}/${r.fail.cls}@${r.fail.s}`);
  out.by_limit[lim] = {valid: `${v}/${rows.length}`, events_mean: rows.reduce((a, r) => a + r.events, 0) / rows.length,
    vmean: rows.reduce((a, r) => a + r.vmean, 0) / rows.length, fails, rows};
  console.log(lim, out.by_limit[lim].valid, 'events', out.by_limit[lim].events_mean.toFixed(2), 'vmean', out.by_limit[lim].vmean.toFixed(2), fails.join(' '));
}
if (args.out) fs.writeFileSync(args.out, JSON.stringify(out));
