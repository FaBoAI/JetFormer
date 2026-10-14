// 障害物よけ(obstacle)の図のための軌跡の書き出し(2026-10-14、obstacle の学習の段 r1)。判定には使わない(試験の数字は eval_modes.mjs)。
//   node rl/ob/dump_traces.mjs --policy <obstacle の policy.json> --fastest <最速の policy.json> --out <traces.json> [--layouts 6]
// 回(元の sim-core.js、標準の物理、連続 20 秒):
//   - recording 組の配置 0〜(--layouts − 1)番、開始は記録の決まり(Halton、前後 1.5 m に障害物なし。rl/ob/predict_variation.mjs と同じ選び方の最初の候補)で
//     obstacle の方策と最速の方策(比べる。足した障害物があるので Node の試験の上書き allow_obstacles を使う)
//   - 障害物なしの STD の連続(obstacle・最速)
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {activeWorldFor} from '../../snapshot/rl/env_core.mjs';
import {prepareGeometry, clearance, projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';
import {geometryWithObstacles, startClearOfObstacles, worldWithObstacles} from '../runtime/layouts.mjs';
import {REFERENCE_LOOP} from '../runtime/reference_loop.mjs';
import {passageStats, obstacleEy} from './ob_common.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const W = readJson(opt('--policy')), F = readJson(opt('--fastest')), nLay = +opt('--layouts', '6');
const lays = readJson(path.join(here, '../obstacle_sets/recording.json')).layouts;
const dist = CATALOGUE.modes.find(m => m.key === 'obstacle').recording.start_distribution, geom = prepareGeometry(BASE_WORLD);
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const out = {schema: 'jf-ob-traces/v1', created_at: new Date().toISOString(), policy: opt('--policy'), fastest: opt('--fastest'),
  course: {segments: BASE_WORLD.segments.map(s => ({a: s.a, b: s.b, r: s.radius, kind: s.kind})), disks: BASE_WORLD.disks.map(d => ({x: d.x, z: d.z, r: d.radius, kind: d.kind})),
    surfaces: BASE_WORLD.surfaces, route: BASE_WORLD.route, bounds: BASE_WORLD.bounds, reference: {x: REFERENCE_LOOP.x, z: REFERENCE_LOOP.z}},
  runs: []};
async function one(id, label, mode, weights, start, layout) {
  const res = await runModeEpisode({mode, weights, start, params: mode === 'obstacle' ? {layout_id: layout?.id ?? null} : {}, layout,
    overrides: mode !== 'obstacle' && layout ? {allow_obstacles: true} : {}});
  if (res.invalid_spawn) { console.error('開始が無効', id); return; }
  const {world: active} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
  const g = prepareGeometry(layout ? worldWithObstacles(active, layout.obstacles) : active);
  const tr = res.trace.map(q => ({x: +q.x.toFixed(4), z: +q.z.toFixed(4), v: +q.v.toFixed(3), d: +clearance(g, q.x, q.z, q.yaw, .42, .19, .5).toFixed(4)}));
  const passes = layout ? passageStats(res, layout.obstacles) : [];
  out.runs.push({id, label, mode, start, layout_id: layout?.id ?? null, obstacles: layout ? layout.obstacles.map(o => ({...o, ey: +obstacleEy(o).toFixed(4)})) : [],
    reason: res.reason, frames: res.frames, collisions: res.collisions, collision: res.collision, min_clearance: res.min_clearance, valid: res.valid_episode,
    laps: res.laps.map(l => l.elapsed), passes, trace: tr});
  console.error(`${id} ${res.reason} 衝突 ${res.collisions} すき間 ${((res.min_clearance ?? 0) * 100).toFixed(1)} cm 周 ${res.laps.map(l => l.elapsed.toFixed(2)).join(',')} 通過 ${passes.length}`);
}
// --val-ids a,b,…: 検証の組(rl/eval_modes.mjs の buildJobs と同じ回・開始・種)から選んだ回を走らせる(recording の代わり)
const valIds = opt('--val-ids')?.split(',') ?? null;
if (valIds) {
  const {buildJobs} = await import('../eval_modes.mjs');
  const jobs = buildJobs('obstacle', 'validation').filter(j => valIds.includes(j.id));
  for (const j of jobs) {
    await one(`ob_${j.id}`, `障害物よけ・${j.id}`, 'obstacle', W, j.spec.start, j.spec.layout);
    await one(`fastest_${j.id}`, `最速・${j.id}`, 'fastest', F, j.spec.start, j.spec.layout);
  }
}
let h = 1;
for (const lay of valIds ? [] : lays.slice(0, nLay)) {
  let st = null;
  for (; h < 10000; h++) {
    const c = haltonStart(h, dist);
    const {world} = activeWorldFor(BASE_WORLD, c.gate, c.left_offset_m, c.left_yaw_deg);
    if (!(clearance(geometryWithObstacles(geom, lay.obstacles), world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5) > 1e-9)) continue;
    if (!startClearOfObstacles(projectWindow(world.spawn.x, world.spawn.z, 0, LOOP_LENGTH).s, lay.obstacles)) continue;
    st = c; h++; break;
  }
  const start = {gate: st.gate, left_offset_m: st.left_offset_m, left_yaw_deg: st.left_yaw_deg, seed: 706900 + out.runs.length};
  await one(`ob_${lay.id}`, `障害物よけ・${lay.id}`, 'obstacle', W, start, lay);
  await one(`fastest_${lay.id}`, `最速・${lay.id}`, 'fastest', F, start, lay);
}
await one('ob_empty', '障害物よけ・障害物なし', 'obstacle', W, STD, null);
await one('fastest_empty', '最速・障害物なし', 'fastest', F, STD, null);
fs.writeFileSync(opt('--out'), JSON.stringify(out));
