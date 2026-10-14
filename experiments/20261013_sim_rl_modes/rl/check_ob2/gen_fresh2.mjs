// 障害物よけの確認 r2: 新しい配置 150 を作る(2026-10-14、確認の段 r2)。分布は protocol の決まり(modes.json common.obstacle_layouts)そのもの =
// 試験の組を作った layouts.mjs の generateLayoutSet を、新しい種 4,130,000〜で呼ぶ(学習 = 回ごとの乱数、記録 730000〜・検証 830000〜・試験 930000〜・
// 選びの組 1,830,000〜・確認 r1 の 1,930,000〜 と重ならない)。入力は rl/build_obstacle_sets.mjs と同じ(最速のふだんの線・ゲートの柱・STD の前後に置かない)。
// コースはユーザーのリリースの qa/load-course-world.mjs から作る(snapshot の world-v3.json と同じかも確かめる)。
//   node gen_fresh2.mjs
import fs from 'node:fs';
import crypto from 'node:crypto';
import {BASE_WORLD} from './run_ob2.mjs';

const EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes';
const OUT = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/check_obstacle_r2';
const {generateLayoutSet, buildBaseClearanceGrid} = await import(`${EXP}/web/rl/layouts.mjs`);
const {prepareGeometry, projectWindow, LOOP_LENGTH} = await import(`${EXP}/web/rl/features.mjs`);
const snap = JSON.parse(fs.readFileSync(`${EXP}/snapshot/sim/world-v3.json`, 'utf8'));
const strip = w => JSON.stringify({segments: w.segments, disks: w.disks, surfaces: w.surfaces, bounds: w.bounds, route: w.route, spawn: w.spawn});
const sameWorld = strip(snap) === strip(BASE_WORLD);
const g = prepareGeometry(BASE_WORLD), grid = buildBaseClearanceGrid(g);
const nominal = JSON.parse(fs.readFileSync(`${EXP}/rl/nominal_line_fastest.json`, 'utf8'));
const gatePosts = BASE_WORLD.disks.filter(d => d.kind === 'gate-post').map(d => ({x: d.x, z: d.z}));
const stdS = projectWindow(BASE_WORLD.spawn.x, BASE_WORLD.spawn.z, 0, LOOP_LENGTH).s;
const set = generateLayoutSet('fresh2', 4130000, 150, {baseGeometry: g, nominalLine: nominal.ey_mean, gatePosts, count: [1, 4], startS: stdS, grid});
set.note = '確認の段 r2 の新しい配置(試験の組と同じ作り方・新しい種 4,130,000〜)。STD の前後に置かない(試験の組と同じ)';
set.base_world_same_as_snapshot = sameWorld;
set.nominal_line_sha256 = crypto.createHash('sha256').update(fs.readFileSync(`${EXP}/rl/nominal_line_fastest.json`)).digest('hex');
fs.writeFileSync(`${OUT}/fresh2_layouts.json`, JSON.stringify(set, null, 1));
const hist = {}; for (const l of set.layouts) hist[l.obstacles.length] = (hist[l.obstacles.length] ?? 0) + 1;
console.log(JSON.stringify({layouts: set.layouts.length, hist, same_world_as_snapshot: sameWorld, std_s: stdS}));
