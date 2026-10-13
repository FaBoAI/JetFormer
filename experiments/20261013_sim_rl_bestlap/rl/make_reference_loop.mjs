#!/usr/bin/env node
// 観測の物差しにする参照ループ R を作る(2026-10-13、RL 環境の段)
//
//   node rl/make_reference_loop.mjs        → rl/reference_loop.json と rl/reference_loop.mjs を書く
//   node rl/make_reference_loop.mjs --check → 書かずに、今のファイルと同じものができるかだけ確かめる
//
// 元: RC 教師の単周の参照軌跡(createRcTrainingTeacher・標準の開始の reference.points、1,092 点・26.119 m)。
// 単周の参照は終点が始点から横に 5.3 cm ずれて終わる(向きも 7° 違う)ので、そのまま結ぶと横向きの短い線分ができる。
// そこで作者の連続周回の教師と同じ閉じ方(createPeriodicBase: 始点・終点の前後 1.5 m を 3 次エルミートでつなぎ、
// ゲート 0 の位置を向き 0 で通す)で閉じた 1 周分(referenceLap 0 の 1,091 点・26.1196 m)を使う。
// R は「座標の物差しと先読みの形」にだけ使い、R に沿うことへの報酬はない(design.md 2.3 節)。
// ユーザーのファイルは読むだけ(コピーした sim/dist を使う。コピーが元と同じことは tests で確かめる)。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const simRoot = path.resolve(here, '../sim');
const world = JSON.parse(fs.readFileSync(path.join(simRoot, 'world-v3.json'), 'utf8'));
const {createRcTrainingTeacher} = await import(path.join(simRoot, 'dist/rc-speed-controller.js'));
const {ROUTE} = await import(path.join(simRoot, 'dist/world.js'));

const first = world.route[0], next = world.route[1];
const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
const nominal = {...world, spawn: {x: first[0], z: first[1], yaw: heading}};
const physics = {dt: 1 / 30, maxSpeed: 7.24, maxSteer: .52, acceleration: 1.2, steerRate: 1.2, vehicleLength: .42, vehicleWidth: .19,
  wheelbase: .26, stopOnCollision: true, frictionMultiplier: 1, noiseStd: 0, seed: 1, timeLimit: 20,
  longitudinalModel: 'real-drive-response-estimate-v1', driveTau: .4, driveAccelerationCap: 6, brakeDeceleration: 1.2};
const single = createRcTrainingTeacher(nominal, physics, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
const periodic = createRcTrainingTeacher(nominal, {...physics, episodeMode: 'continuous-laps-v1'}, {gate: 0, left_offset_m: 0, left_yaw_deg: 0});
const loop = periodic.reference.points.filter(p => p.referenceLap === 0);
const length = periodic.reference.points.find(p => p.referenceLap === 1).s;
const sp = single.reference.points;

// ゲート(targets = route[1..32] + route[0])の R 上の位置 s: 前のゲートの s から先へ順に、いちばん近い点を探す
const targets = [...ROUTE.slice(1), ROUTE[0]];
function nearestS(p, from, span) {
  let best = {d: Infinity, s: null};
  for (let i = 0; i < loop.length; i++) {
    const a = loop[i], b = loop[(i + 1) % loop.length], sa = a.s, sb = i + 1 < loop.length ? b.s : length;
    let rel = sa - from; rel = ((rel % length) + length) % length;
    if (rel > span) continue;
    const dx = b.x - a.x, dz = b.z - a.z, u = Math.max(0, Math.min(1, ((p[0] - a.x) * dx + (p[1] - a.z) * dz) / (dx * dx + dz * dz)));
    const d = Math.hypot(p[0] - a.x - u * dx, p[1] - a.z - u * dz);
    if (d < best.d) best = {d, s: sa + u * (sb - sa)};
  }
  return best;
}
const gateS = [], gateDist = [];
let cursor = 0;
for (let g = 0; g < targets.length; g++) {
  const span = g === targets.length - 1 ? length : 4;   // 最後(ゲート 0 の位置)は s = 0 付近に戻る
  const r = nearestS(targets[g], cursor, span);
  gateS.push(g === targets.length - 1 ? 0 : r.s); gateDist.push(r.d); cursor = r.s;
}
const data = {
  schema: 'rl-bestlap-reference-loop/v1',
  source: 'createRcTrainingTeacher(standard start, continuous-laps-v1).reference.points の referenceLap 0(= 単周の参照を createPeriodicBase で閉じたもの)',
  world_version: 'indoor-course-eval-3-turf-board-removed',
  rc_teacher_base_reference_sha256: periodic.metadata.base_reference_sha256,
  seam_width_m: 1.5,
  n: loop.length,
  length_m: length,
  single_lap_reference: {points: sp.length, length_m: sp.at(-1).s, end_gap_m: Math.hypot(sp.at(-1).x - sp[0].x, sp.at(-1).z - sp[0].z),
    end_yaw_minus_start_yaw_rad: sp.at(-1).yaw - sp[0].yaw},
  x: loop.map(p => p.x), z: loop.map(p => p.z), s: loop.map(p => p.s),
  route: ROUTE.map(p => [...p]),
  gate_s: gateS,
  gate_nearest_distance_m: gateDist,
};
const json = JSON.stringify(data);
const sha = crypto.createHash('sha256').update(json).digest('hex');
const mjs = `// 自動生成(rl/make_reference_loop.mjs)。手で変えない。sha256(JSON 本文) = ${sha}\nexport const REFERENCE_LOOP_SHA256 = '${sha}';\nexport const REFERENCE_LOOP = ${json};\n`;
const jsonPath = path.join(here, 'reference_loop.json'), mjsPath = path.join(here, 'reference_loop.mjs');
if (process.argv.includes('--check')) {
  const same = fs.readFileSync(jsonPath, 'utf8') === json && fs.readFileSync(mjsPath, 'utf8') === mjs;
  console.log(JSON.stringify({same, sha256: sha}));
  process.exit(same ? 0 : 1);
}
fs.writeFileSync(jsonPath, json);
fs.writeFileSync(mjsPath, mjs);
console.log(JSON.stringify({sha256: sha, n: data.n, length_m: length, single_lap_reference: data.single_lap_reference,
  gate_nearest_distance_max_m: Math.max(...gateDist)}, null, 1));
