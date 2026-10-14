// 障害物よけ(obstacle)の報告「障害物の横を通った回数とすき間」(protocol acceptance.obstacle.reported)。2026-10-14、obstacle の学習の段 r1
// 試験の組 OB-T100(rl/eval_modes.mjs の buildJobs と同じ回・同じ開始・同じ種)を元の sim-core.js で走らせ直し、通過ごとに
// 通った向き(障害物の左 / 右)・その障害物とのすき間(フレームの値)・速さを数える。判定には使わない(判定は eval_modes.mjs の結果)。
//   node rl/ob/passages_test.mjs --policy <policy.json> [--set test|validation] [--run-mode fastest] --out <結果.json>
import fs from 'node:fs';
import {runModeEpisode, readJson} from '../episode.mjs';
import {buildJobs} from '../eval_modes.mjs';
import {passageStats, summarizePassages} from './ob_common.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const W = readJson(opt('--policy')), set = opt('--set', 'test'), runMode = opt('--run-mode', 'obstacle');
const jobs = buildJobs('obstacle', set).filter(j => j.test === 'OB-T100');
const rows = [], all = [];
for (const j of jobs) {
  const sp = j.spec;
  const r = await runModeEpisode({mode: runMode, weights: W, start: sp.start, params: runMode === 'obstacle' ? {layout_id: sp.layout.id} : {}, layout: sp.layout,
    overrides: runMode !== 'obstacle' ? {allow_obstacles: true} : {}, audit: false});
  const ps = passageStats(r, sp.layout.obstacles);
  all.push(...ps.map(p => ({...p, run: j.id})));
  rows.push({id: j.id, layout: sp.layout.id, n_obstacles: sp.layout.obstacles.length, reason: r.reason, frames: r.frames, valid: r.valid_episode,
    collision: r.collision, laps: r.laps.map(l => l.elapsed), passes: ps.filter(p => p.completed).length,
    sides: ps.filter(p => p.completed).map(p => p.side[0]).join(''), min_pass_clearance: Math.min(...ps.map(p => p.min_clearance_m), Infinity)});
}
// 同じ配置の障害物を 2 回目以降に通ったときに、1 回目と違う向きを通った割合(線が毎回同じかどうか)
const byKey = {};
for (const p of all.filter(p => p.completed)) (byKey[`${p.run}|${p.obstacle}`] ??= []).push(p.side);
const multi = Object.values(byKey).filter(a => a.length >= 2);
const res = {schema: 'jf-ob-passages/v1', created_at: new Date().toISOString(), policy: opt('--policy'), run_mode: runMode, set, runs: rows.length,
  valid: rows.filter(r => r.valid).length, summary: summarizePassages(all),
  side_change_within_run: {obstacles_passed_twice_or_more: multi.length, changed_side: multi.filter(a => new Set(a).size > 1).length},
  rows, passes: all};
console.log(JSON.stringify({runs: res.runs, valid: res.valid, summary: res.summary, side_change_within_run: res.side_change_within_run}));
fs.writeFileSync(opt('--out'), JSON.stringify(res));
