#!/usr/bin/env node
// 障害物よけ r2 の「選びの組」OB-V100-R2 の配置(2026-10-14 学習の段 obstacle r2、protocol_obstacle_r2.json sets.selection)。
// 試験・検証の組と同じ作り方(rl/build_obstacle_sets.mjs と同じ引数: 最速のふだんの線 + U(−0.15, 0.15) m・通り道の保証・STD の前後に置かない)で、
// 新しい種 1,830,000 + i の 50 配置。試験(930000〜)・検証(830000〜)・記録(730000〜)・確認 r1 の新しい配置(1,930,000〜)と重ならない。
//   node rl/ob2/build_selection_set.mjs
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {generateLayoutSet, buildBaseClearanceGrid, checkPassage} from '../runtime/layouts.mjs';
import {prepareGeometry, projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), rl = path.resolve(here, '..');
const world = JSON.parse(fs.readFileSync(path.join(rl, '../snapshot/sim/world-v3.json'), 'utf8'));
const g = prepareGeometry(world), grid = buildBaseClearanceGrid(g);
const nominal = JSON.parse(fs.readFileSync(path.join(rl, 'nominal_line_fastest.json'), 'utf8'));
const gatePosts = world.disks.filter(d => d.kind === 'gate-post').map(d => ({x: d.x, z: d.z}));
const stdS = projectWindow(world.spawn.x, world.spawn.z, 0, LOOP_LENGTH).s;
const set = generateLayoutSet('selection', 1830000, 50, {baseGeometry: g, nominalLine: nominal.ey_mean, gatePosts, count: CATALOGUE.common.obstacle_layouts.count_per_layout,
  startS: stdS, grid});
for (const lay of set.layouts) { lay.passage = checkPassage(g, lay.obstacles).ok; lay.n = lay.obstacles.length; }
set.nominal_line_sha256 = crypto.createHash('sha256').update(fs.readFileSync(path.join(rl, 'nominal_line_fastest.json'))).digest('hex');
set.std_start_excluded = true; set.created_by = 'rl/ob2/build_selection_set.mjs(obstacle r2)';
const out = path.join(here, 'selection_layouts.json');
fs.writeFileSync(out, JSON.stringify(set, null, 1));
const hist = [0, 0, 0, 0, 0]; set.layouts.forEach(l => hist[l.n]++);
console.log(JSON.stringify({layouts: set.layouts.length, all_passage: set.layouts.every(l => l.passage), count_hist: hist,
  sha256: crypto.createHash('sha256').update(fs.readFileSync(out)).digest('hex')}));
