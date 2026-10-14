#!/usr/bin/env node
// 障害物よけ r2 の下調べ(判定に使わない): 配置の各障害物で、障害物の s ±0.6 m の通り道(すき間 ≥ 8 cm で動ける横の範囲、ゲートの窓の中)を
// 障害物の左・右に分けて、その幅の最小を出す(layouts.mjs checkPassage と同じ格子)。2026-10-14 学習の段 obstacle r2
//   node rl/ob2/passage_sides.mjs <配置の組.json> <出力.json>
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {prepareGeometry, clearance, loopPointAt, LOOP_LENGTH} from '../runtime/features.mjs';
import {obstacleOnlyGeometry, obstacleS, buildBaseClearanceGrid} from '../runtime/layouts.mjs';
import {ROOM_TABLE} from '../runtime/room_table.mjs';
import {obstacleEy} from '../ob/ob_common.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const world = JSON.parse(fs.readFileSync(path.join(here, '../../snapshot/sim/world-v3.json'), 'utf8'));
const g = prepareGeometry(world), grid = buildBaseClearanceGrid(g), N = ROOM_TABLE.n, DS = ROOM_TABLE.ds_m;
const OFF_MAX = 1.0, STEP = .01, NOFF = Math.round(2 * OFF_MAX / STEP) + 1;
export function sides(obstacles, o) {
  const added = obstacleOnlyGeometry(obstacles), so = obstacleS(o), eyo = obstacleEy(o), pt = {x: 0, z: 0, tx: 0, tz: 0};
  let L = Infinity, R = Infinity;
  for (let d = -4; d <= 4; d++) {   // 障害物の s ±0.2 m(障害物が R の横の位置 eyo をふさいでいる所)
    const i = ((Math.round(so / DS) + d) % N + N) % N;
    loopPointAt(i * DS, pt);
    const yaw = Math.atan2(-pt.tz, pt.tx);
    let bestL = 0, bestR = 0, run = 0, lo = 0;
    for (let k = 0; k < NOFF; k++) {
      const off = -OFF_MAX + k * STEP, x = pt.x + off * pt.tz, z = pt.z - off * pt.tx;
      const free = Math.abs(off) <= .65 && grid[i * NOFF + k] >= .08 && clearance(added, x, z, yaw, .42, .19, .5) >= .08;
      if (free) { if (!run) lo = off; run++; }
      if (!free || k === NOFF - 1) {
        if (run) { const hi = lo + (run - 1) * STEP, w = hi - lo, mid = (lo + hi) / 2; if (mid > eyo) bestL = Math.max(bestL, w); else bestR = Math.max(bestR, w); }
        run = 0;
      }
    }
    L = Math.min(L, bestL); R = Math.min(R, bestR);
  }
  return {s: so, ey: eyo, left_w: L, right_w: R};
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const set = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const out = {};
  for (const lay of set.layouts) out[lay.id] = lay.obstacles.map(o => sides(lay.obstacles, o));
  fs.writeFileSync(process.argv[3], JSON.stringify(out));
  const all = Object.values(out).flat();
  console.log(JSON.stringify({obstacles: all.length, left_only: all.filter(q => q.left_w >= .1 && q.right_w < .1).length, right_only: all.filter(q => q.right_w >= .1 && q.left_w < .1).length,
    both: all.filter(q => q.left_w >= .1 && q.right_w >= .1).length}));
}
