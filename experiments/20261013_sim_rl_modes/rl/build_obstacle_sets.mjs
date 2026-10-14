#!/usr/bin/env node
// 障害物よけの配置の組(modes.json common.obstacle_layouts.sets: validation 30(種 830000〜)・test 50(930000〜)・recording 30(730000〜))を作る。
// 置く位置は最速の方策のふだんの線(rl/nominal_line_fastest.json)+ U(−0.15, 0.15) m。通り道は layouts.mjs の検査器で保証。
// test・validation は STD の開始(ゲート 0)の前後 1.5 m に置かない(OB-T100 はどの配置も STD から走るため)。2026-10-13
//   node rl/build_obstacle_sets.mjs
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {generateLayoutSet, buildBaseClearanceGrid, checkPassage, worldWithObstacles} from './runtime/layouts.mjs';
import {prepareGeometry, projectWindow, LOOP_LENGTH} from './runtime/features.mjs';
import {CATALOGUE} from './runtime/catalogue.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const world = JSON.parse(fs.readFileSync(path.join(here, '../snapshot/sim/world-v3.json'), 'utf8'));
const g = prepareGeometry(world), grid = buildBaseClearanceGrid(g);
const nominal = JSON.parse(fs.readFileSync(path.join(here, 'nominal_line_fastest.json'), 'utf8'));
const gatePosts = world.disks.filter(d => d.kind === 'gate-post').map(d => ({x: d.x, z: d.z}));
const stdS = projectWindow(world.spawn.x, world.spawn.z, 0, LOOP_LENGTH).s;
const count = CATALOGUE.common.obstacle_layouts.count_per_layout;
const out = {};
for (const [name, base, n, excludeStd] of [['validation', 830000, 30, true], ['test', 930000, 50, true], ['recording', 730000, 30, false]]) {
  const set = generateLayoutSet(name, base, n, {baseGeometry: g, nominalLine: nominal.ey_mean, gatePosts, count, startS: excludeStd ? stdS : null, grid});
  // 検査器でもう一度(グリッドを使わない遅い方)
  for (const lay of set.layouts) { lay.passage = checkPassage(g, lay.obstacles).ok; lay.n = lay.obstacles.length; }
  set.nominal_line_sha256 = crypto.createHash('sha256').update(fs.readFileSync(path.join(here, 'nominal_line_fastest.json'))).digest('hex');
  set.std_start_excluded = excludeStd; set.created_by = 'rl/build_obstacle_sets.mjs';
  fs.writeFileSync(path.join(here, `obstacle_sets/${name}.json`), JSON.stringify(set, null, 1));
  out[name] = {layouts: set.layouts.length, all_pass: set.layouts.every(l => l.passage), counts: set.layouts.map(l => l.n).join('')};
}
console.log(JSON.stringify(out));
