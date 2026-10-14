// no_wall の図のための軌跡の書き出し(2026-10-14、no_wall の学習の段)。判定には使わない(試験の数字は eval_modes.mjs)。
//   node rl/nw/dump_traces.mjs --policy <no_wall の policy.json> --fastest <最速の policy.json> --out <traces.json>
// 書き出すもの: コースの形(レール・柱・面)、R の線、各回の軌跡 (x, z, 速さ, フレームのすき間, シールドの作動) と衝突・最小すき間。
// 回: no_wall の STD 連続(標準の物理)・最速の STD 連続(比べる線)・no_wall の物理をずらした回(GRID の範囲の中から 5 つ + 摩擦円)・
//     no_wall の NW-N60 の試験の Halton の開始 6 つ・シールドなしの no_wall の STD 連続。
import fs from 'node:fs';
import {runModeEpisode, readJson, BASE_WORLD} from '../episode.mjs';
import {gridRows} from '../eval_modes.mjs';
import {prepareGeometry, clearance} from '../runtime/features.mjs';
import {activeWorldFor} from '../../snapshot/rl/env_core.mjs';
import {REFERENCE_LOOP} from '../runtime/reference_loop.mjs';

const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const W = readJson(opt('--policy')), F = readJson(opt('--fastest'));
const STD = {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const rows = gridRows();
const pick = names => rows.filter(r => names.some(n => r.name.startsWith(n)));
const drOf = c => ({physics: c.physics ?? {}, delay: c.delay ?? 0, steerGain: c.steerGain ?? 1, speedGain: c.speedGain ?? 1, frictionCircle: !!c.frictionCircle});
const runs = [
  {id: 'nw_std', label: 'no_wall・標準の物理', mode: 'no_wall', weights: W, start: STD},
  {id: 'fastest_std', label: '最速・標準の物理', mode: 'fastest', weights: F, start: STD},
  {id: 'nw_std_noshield', label: 'no_wall・シールドなし', mode: 'no_wall', weights: W, start: STD, overrides: {shield: false}},
  ...pick(['舵の速さ 0.8 ', 'ブレーキ 0.7 ', 'μ × 0.85', '組み合わせ: 舵 0.9', '遅延 1']).filter(r => r.in_range)
    .map((r, i) => ({id: `nw_grid${i}`, label: `no_wall・${r.name}`, mode: 'no_wall', weights: W, start: STD, dr: drOf(r.cond)})),
  {id: 'nw_fc', label: 'no_wall・摩擦円', mode: 'no_wall', weights: W, start: STD, dr: {physics: {}, delay: 0, frictionCircle: true}},
  ...[{gate: 7, left_offset_m: .15, left_yaw_deg: -12}, {gate: 13, left_offset_m: -.18, left_yaw_deg: 10}, {gate: 19, left_offset_m: .1, left_yaw_deg: 14},
    {gate: 24, left_offset_m: -.12, left_yaw_deg: -9}, {gate: 29, left_offset_m: .18, left_yaw_deg: 6}, {gate: 3, left_offset_m: -.05, left_yaw_deg: -14}]
    .map((s, i) => ({id: `nw_start${i}`, label: `no_wall・開始 ゲート ${s.gate}`, mode: 'no_wall', weights: W, start: {...s, seed: 5000 + i}})),
];
const out = {schema: 'jf-nw-traces/v1', created_at: new Date().toISOString(), policy: opt('--policy'), fastest: opt('--fastest'),
  course: {segments: BASE_WORLD.segments.map(s => ({a: s.a, b: s.b, r: s.radius, kind: s.kind})), disks: BASE_WORLD.disks.map(d => ({x: d.x, z: d.z, r: d.radius, kind: d.kind})),
    surfaces: BASE_WORLD.surfaces, route: BASE_WORLD.route, bounds: BASE_WORLD.bounds,
    reference: {x: REFERENCE_LOOP.x, z: REFERENCE_LOOP.z}},
  runs: []};
for (const r of runs) {
  const res = await runModeEpisode({mode: r.mode, weights: r.weights, start: r.start, dr: r.dr ?? null, overrides: r.overrides ?? {}});
  if (res.invalid_spawn) { console.error('開始が無効', r.id); continue; }
  const {world} = activeWorldFor(BASE_WORLD, r.start.gate, r.start.left_offset_m, r.start.left_yaw_deg);
  const g = prepareGeometry(world);
  const tr = res.trace.map((q, i) => ({x: +q.x.toFixed(4), z: +q.z.toFixed(4), v: +q.v.toFixed(3), d: +clearance(g, q.x, q.z, q.yaw, .42, .19, .5).toFixed(4),
    sh: res.log[i]?.sh ?? 0, s: res.log[i] ? +res.log[i].s.toFixed(3) : null}));
  out.runs.push({id: r.id, label: r.label, mode: r.mode, start: r.start, dr: r.dr ?? null, overrides: r.overrides ?? null, reason: res.reason, frames: res.frames,
    collisions: res.collisions, collision: res.collision, min_clearance: res.min_clearance, shield_frames: res.shield_frames,
    laps: res.laps.map(l => l.elapsed), trace: tr});
  console.error(`${r.id} ${res.reason} 衝突 ${res.collisions} すき間 ${(res.min_clearance * 100).toFixed(1)} cm シールド ${res.shield_frames} 周 ${res.laps.map(l => l.elapsed.toFixed(2)).join(',')}`);
}
fs.writeFileSync(opt('--out'), JSON.stringify(out));
