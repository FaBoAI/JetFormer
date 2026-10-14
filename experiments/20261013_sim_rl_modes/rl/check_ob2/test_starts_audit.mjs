// 障害物よけの確認 r2: 試験の組 OB-T100 の開始と配置を、方策を走らせずに調べる(2026-10-14、確認の段 r2)。試験の 2 回の枠は使わない。
//   試験の開始(STD と Halton 20,001〜)は rl/eval_modes.mjs haltonStarts を読むだけ(方策・sim は動かさない)で作り、
//   開始から最初の障害物までの R 上の距離(自前の射影)を、選びの組・確認 r2 の新しい配置の開始と同じ物差しで数える。
//   node test_starts_audit.mjs → check_obstacle_r2/test_starts_audit.json
import fs from 'node:fs';
import {projectGlobal, activeWorld, BASE_WORLD, startClearance} from './run_ob2.mjs';
import {LOOP_L_OWN} from '../check_ln/own_extra.mjs';
import {buildSet} from './jobs_ob2.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const REP = '/home/jetson/jetformer_reports/sim_rl_modes_20261013';
const OUT = `${REP}/check_obstacle_r2`;
const {haltonStarts} = await import(`${EXP}/rl/eval_modes.mjs`);
const modL = s => ((s % LOOP_L_OWN) + LOOP_L_OWN) % LOOP_L_OWN;
function ahead(st, obstacles) {
  const {world} = activeWorld(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  const s0 = projectGlobal(world.spawn.x, world.spawn.z).s;
  const ds = obstacles.map(o => modL(projectGlobal(o.x, o.z).s - s0)).sort((a, b) => a - b);
  return {first_m: +ds[0].toFixed(3), all_m: ds.map(x => +x.toFixed(3)), start_clearance: +startClearance(st.gate, st.left_offset_m, st.left_yaw_deg, obstacles).toFixed(4)};
}
const bucket = d => d < 2.5 ? '1.5-2.5' : d < 3.5 ? '2.5-3.5' : d < 5 ? '3.5-5' : '5+';
function summarize(rows) {
  const h = {'1.5-2.5': 0, '2.5-3.5': 0, '3.5-5': 0, '5+': 0};
  rows.forEach(r => h[bucket(r.first_m)]++);
  return {n: rows.length, by_bucket: h, within_3_5: rows.filter(r => r.first_m < 3.5).length, min_first: Math.min(...rows.map(r => r.first_m)),
    min_start_clearance: Math.min(...rows.map(r => r.start_clearance))};
}
const res = {};
// 試験の組
const test = JSON.parse(fs.readFileSync(`${EXP}/rl/obstacle_sets/test.json`, 'utf8')).layouts.slice(0, 50);
const hs = haltonStarts('obstacle', 'test', test.length, {layoutFor: k => test[k]});
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
const tRows = [];
test.forEach((lay, i) => {
  tRows.push({id: `${lay.id}-STD`, kind: 'STD', n: lay.obstacles.length, ...ahead(STD, lay.obstacles)});
  tRows.push({id: `${lay.id}-H`, kind: 'H', n: lay.obstacles.length, halton: hs[i].halton_index ?? null, start: hs[i], ...ahead(hs[i], lay.obstacles)});
});
res.test = {all: summarize(tRows), STD: summarize(tRows.filter(r => r.kind === 'STD')), H: summarize(tRows.filter(r => r.kind === 'H')),
  count_hist: [1, 2, 3, 4].map(k => test.filter(l => l.obstacles.length === k).length), rows: tRows};
// 選びの組(学習の段の結果の行の開始)
const sel = JSON.parse(fs.readFileSync(`${REP}/train/obstacle/r2/final/selected_selection_original.json`, 'utf8')).rows.filter(r => r.test === 'OB-V100-R2');
const sl = Object.fromEntries(JSON.parse(fs.readFileSync(`${EXP}/rl/ob2/selection_layouts.json`, 'utf8')).layouts.map(l => [l.id, l]));
const sRows = sel.map(r => ({id: r.id, kind: r.id.endsWith('STD') ? 'STD' : 'H', n: sl[r.layout_id].obstacles.length, valid_trainer: r.valid_episode, ...ahead(r.start, sl[r.layout_id].obstacles)}));
res.selection = {all: summarize(sRows), STD: summarize(sRows.filter(r => r.kind === 'STD')), H: summarize(sRows.filter(r => r.kind === 'H')),
  count_hist: [1, 2, 3, 4].map(k => Object.values(sl).filter(l => l.obstacles.length === k).length), rows: sRows};
// 確認 r2 の新しい配置
const fr = buildSet('FRESH');
const fRows = fr.map(j => ({id: j.id, kind: j.id.endsWith('STD') ? 'STD' : 'R', n: j.spec.layout.obstacles.length, ...ahead(j.spec.start, j.spec.layout.obstacles)}));
res.fresh2 = {all: summarize(fRows), STD: summarize(fRows.filter(r => r.kind === 'STD')), R: summarize(fRows.filter(r => r.kind === 'R')),
  count_hist: [1, 2, 3, 4].map(k => fr.filter((j, i) => i % 2 === 0 && j.spec.layout.obstacles.length === k).length), rows: fRows};
fs.writeFileSync(`${OUT}/test_starts_audit.json`, JSON.stringify(res, null, 1));
for (const k of Object.keys(res)) console.log(k, JSON.stringify({all: res[k].all, STD: res[k].STD, other: res[k].H ?? res[k].R, count_hist: res[k].count_hist}));
