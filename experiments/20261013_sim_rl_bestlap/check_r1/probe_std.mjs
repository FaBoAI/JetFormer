// 標準の単周・連続を自前のループで走らせ、RL の記録(rl/final/records)と操作列が同じかを見る(試し)
import fs from 'node:fs';
import {loadOwnPolicy, runClosedLoop, REPORTS, hashActions} from './common_r1.mjs';
const policy = loadOwnPolicy();
for (const [mode, file] of [['single', 'rl_main_STD_single.json'], ['continuous', 'rl_main_STD_continuous.json']]) {
  const t0 = Date.now();
  const r = runClosedLoop(policy, {episodeMode: mode});
  const rec = JSON.parse(fs.readFileSync(`${REPORTS}/rl/final/records/${file}`, 'utf8'));
  let maxDiff = 0;
  const n = Math.min(rec.actions.length, r.report.actions.length);
  for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(rec.actions[i].steering - r.report.actions[i].steering), Math.abs(rec.actions[i].speed - r.report.actions[i].speed));
  console.log(mode, JSON.stringify(r.summary.laps), r.summary.reason, 'frames', r.report.actions.length, rec.actions.length, 'maxdiff', maxDiff,
    'same_hash', hashActions(rec.actions) === hashActions(r.report.actions), 'ms', Date.now() - t0);
}
