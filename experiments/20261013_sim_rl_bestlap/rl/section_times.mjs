#!/usr/bin/env node
// 停止から 1 周の記録(final_eval.mjs の records/*.json)から、ゲートごとの通過時刻・区間の時間・路面ごとの時間を出す(2026-10-13、RL の学習の段)
//
//   node rl/section_times.mjs --out sections.json 名前=記録.json [名前=記録.json ...]
//
// ゲートの通過時刻: フレームの軌跡で、目標ゲートの線(前のゲート → 目標の向きに垂直)を越えたフレームの間を直線で補う(目安。sim は 1 フレームを 31 に分ける)。
// 区間はコースの部分ごと: 発進〜ゲート 3 / U ターン 3〜8 / 黒シートの近道 8〜15 / 芝・PTFE 15〜20 / 奥の直線 20〜26 / 青マットのコーナー〜ゴール 26〜33。
import fs from 'node:fs';

const args = process.argv.slice(2);
const oi = args.indexOf('--out');
const out = oi >= 0 ? args.splice(oi, 2)[1] : null;
const SECTIONS = [['発進〜3', 0, 3], ['U ターン 3〜8', 3, 8], ['黒シート 8〜15', 8, 15], ['芝・PTFE 15〜20', 15, 20], ['奥の直線 20〜26', 20, 26], ['青マット〜ゴール 26〜33', 26, 33]];

function gateTimes(rep) {
  const route = rep.world.route, n = route.length, tr = rep.trace;
  const times = [0];
  let k = 0, prev = {x: rep.world.spawn.x, z: rep.world.spawn.z, time: 0};
  for (const t of tr) {
    while (k < n) {
      const target = route[(k + 1) % n], before = route[k];
      const len = Math.hypot(target[0] - before[0], target[1] - before[1]), nx = (target[0] - before[0]) / len, nz = (target[1] - before[1]) / len;
      const a0 = (prev.x - target[0]) * nx + (prev.z - target[1]) * nz, a1 = (t.x - target[0]) * nx + (t.z - target[1]) * nz;
      if (a1 < 0) break;
      const u = a0 < 0 ? -a0 / (a1 - a0) : 0;
      times.push(prev.time + u * (t.time - prev.time));
      k++;
    }
    prev = t;
  }
  return times;   // times[g] = ゲート g(経路の番号。33 = ゴールのゲート 0)を越えた時刻
}

const result = {schema: 'rl-bestlap-section-times/v1', sections: SECTIONS.map(s => s[0]), runs: {}};
for (const a of args) {
  const [name, file] = a.split('=');
  const rep = JSON.parse(fs.readFileSync(file, 'utf8'));
  const g = gateTimes(rep), surf = {};
  for (const t of rep.trace) surf[t.surface] = (surf[t.surface] ?? 0) + rep.config.dt;
  result.runs[name] = {file, lap_s: rep.metrics.elapsed, gate_times: g,
    sections_s: SECTIONS.map(([, a0, b0]) => (g[b0] ?? NaN) - (g[a0] ?? NaN)), surface_time_s: surf,
    max_speed: Math.max(...rep.trace.map(t => t.speed)), distance_m: rep.metrics.distance};
}
const names = Object.keys(result.runs);
console.log(['区間', ...names].join('\t'));
SECTIONS.forEach(([label], i) => console.log([label, ...names.map(n => result.runs[n].sections_s[i].toFixed(3))].join('\t')));
console.log(['1 周', ...names.map(n => result.runs[n].lap_s.toFixed(3))].join('\t'));
console.log(['距離 m', ...names.map(n => result.runs[n].distance_m.toFixed(2))].join('\t'));
for (const n of names) console.log(n, '路面', JSON.stringify(Object.fromEntries(Object.entries(result.runs[n].surface_time_s).map(([k, v]) => [k, +v.toFixed(3)]))));
if (out) fs.writeFileSync(out, JSON.stringify(result, null, 1) + '\n');
