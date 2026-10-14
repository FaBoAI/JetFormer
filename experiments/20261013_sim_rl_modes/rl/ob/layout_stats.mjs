// 障害物の配置の組(validation・test・recording)の中身の集計(2026-10-14、obstacle の学習の段 r1。報告用)
//   node rl/ob/layout_stats.mjs --out <結果.json>
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {checkPassage, buildBaseClearanceGrid} from '../runtime/layouts.mjs';
import {prepareGeometry} from '../runtime/features.mjs';
import {BASE_WORLD} from '../episode.mjs';
import {obstacleEy, quantile} from './ob_common.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const g = prepareGeometry(BASE_WORLD), grid = buildBaseClearanceGrid(g);
const nominal = JSON.parse(fs.readFileSync(path.join(here, '../nominal_line_fastest.json'), 'utf8'));
const out = {};
for (const name of ['validation', 'test', 'recording']) {
  const set = JSON.parse(fs.readFileSync(path.join(here, `../obstacle_sets/${name}.json`), 'utf8'));
  const lays = name === 'validation' ? set.layouts.slice(0, 13) : set.layouts;
  const counts = {}, shapes = {post: 0, box: 0}, widths = [], offs = [];
  for (const lay of lays) {
    counts[lay.obstacles.length] = (counts[lay.obstacles.length] ?? 0) + 1;
    const cp = checkPassage(g, lay.obstacles, {grid});
    for (const [k, o] of lay.obstacles.entries()) {
      shapes[o.shape]++; widths.push(cp.per_obstacle[k].min_width);
      const i = Math.round(o.s_ref / nominal.ds_m) % nominal.n; offs.push(obstacleEy(o) - (nominal.ey_mean[i] ?? 0));
    }
  }
  out[name] = {layouts: lays.length, obstacles: widths.length, count_hist: counts, shapes,
    passage_min_width_m: {min: Math.min(...widths), p10: quantile(widths, .1), median: quantile(widths, .5)},
    offset_from_fastest_line_m: {min: Math.min(...offs), max: Math.max(...offs), abs_median: quantile(offs.map(Math.abs), .5)}};
}
console.log(JSON.stringify(out, null, 1));
if (opt('--out')) fs.writeFileSync(opt('--out'), JSON.stringify(out, null, 1));
