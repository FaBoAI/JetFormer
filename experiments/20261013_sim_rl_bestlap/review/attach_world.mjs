// 確認の段(review): データセットのシーンの evaluation-report.json に、コース(ユーザーの qa/load-course-world.mjs)を
// app.js reset() の式で開始ゲート・ずれに合わせて回した world を付けて書き出す(independent_lap_check.py が読むため)
//   node attach_world.mjs <出力フォルダー> <evaluation-report.json> [...]
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const USER_SIM = '/home/jetson/sim/extracted/course-simulator';
const {loadCourseWorld} = await import(pathToFileURL(USER_SIM + '/qa/load-course-world.mjs').href);
const W = await loadCourseWorld();
const [outDir, ...files] = process.argv.slice(2);
fs.mkdirSync(outDir, {recursive: true});
for (const f of files) {
  const r = JSON.parse(fs.readFileSync(f, 'utf8'));
  const v = r.context?.start_variation ?? {gate: 0, left_offset_m: 0, left_yaw_deg: 0};
  const i = Math.max(0, Math.min(W.route.length - 1, Math.trunc(v.gate) || 0));
  const route = [...W.route.slice(i), ...W.route.slice(0, i)], first = route[0], next = route[1];
  const heading = Math.atan2(-(next[1] - first[1]), next[0] - first[0]);
  const off = Math.max(-.3, Math.min(.3, v.left_offset_m || 0)), turn = Math.max(-25, Math.min(25, v.left_yaw_deg || 0)) * Math.PI / 180;
  r.world = {...W, route, spawn: {x: first[0] + off * Math.sin(heading), z: first[1] - off * Math.cos(heading), yaw: heading + turn}};
  const name = path.basename(path.dirname(f));
  fs.writeFileSync(path.join(outDir, name + '.json'), JSON.stringify(r));
  console.log(name, JSON.stringify(v));
}
