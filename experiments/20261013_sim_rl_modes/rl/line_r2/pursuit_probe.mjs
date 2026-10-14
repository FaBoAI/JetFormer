#!/usr/bin/env node
// 経路いろいろ r2 の下調べ: 目標の線 v2 を「純追従(pure pursuit)+ 一定の速さ」の手書きの運転で追えるか(元の物理 = 高速版のコア)。
// 学習していない運転で、R の上の s ごとの追従の誤差を出す(目標の線がその速さで走れるかの目安)。
//   node rl/line_r2/pursuit_probe.mjs [--v 1.5] [--ld 0.45] [--family const --c0 0]
import {ModeEnv} from '../env_modes.mjs';
import * as core from '../fast_core_modes.mjs';
import {createRng} from '../runtime/rng.mjs';
import {loopPointAt, LOOP_LENGTH} from '../runtime/features.mjs';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf(k); return i < 0 ? d : args[i + 1]; };
const V = +opt('--v', 1.5), LD = +opt('--ld', .45), fam = opt('--family', 'const');
const line = fam === 'const' ? {family: 'const', c0: +opt('--c0', 0)} : fam === 'random' ? {family: 'random', seed: +opt('--seed', 7)} : {family: fam, a: +opt('--a', .8)};
const env = new ModeEnv({core, run: 'line', mode: 'eval'});
env.reset(createRng(1), {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}, {line, c_v: 1});
const p = {x: 0, z: 0, tx: 0, tz: 0}, L = LOOP_LENGTH;
const target = s => { const a = loopPointAt(s - .05, {x: 0, z: 0, tx: 0, tz: 0}), b = loopPointAt(s + .05, {x: 0, z: 0, tx: 0, tz: 0}); loopPointAt(s, p);
  let tx = b.x - a.x, tz = b.z - a.z; const n = Math.hypot(tx, tz); tx /= n; tz /= n; const e = env.wrapper.line.eStar(s); return [p.x + e * tz, p.z - e * tx]; };
const rows = [];
while (!env.done) {
  const st = env.view.state, s = env.wrapper.tracker.s;
  // 先の点: s + LD(弧長の近似)
  const [gx, gz] = target(s + LD);
  const dx = gx - st.x, dz = gz - st.z, fwd = dx * Math.cos(st.yaw) - dz * Math.sin(st.yaw), left = -dx * Math.sin(st.yaw) - dz * Math.cos(st.yaw);
  const alpha = Math.atan2(left, fwd), d = Math.hypot(dx, dz);
  const steer = Math.max(-.52, Math.min(.52, Math.atan(2 * .26 * Math.sin(alpha) / Math.max(.1, d))));
  // 行動に直す(actionToCommand の逆: a0 = steer/0.52、a1 = 速さ / 上限 × 2 − 1)
  env.step(steer / .52, V / 4.5 * 2 - 1);
  if (env.wrapper.tracker.s - env.s0 >= 3) rows.push({s: ((env.aux.s % L) + L) % L, e: Math.abs(env.aux.lateral_error - env.aux.e_star), v: env.view.state.speed});
}
const sm = env.summary();
const q = (a, f) => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(f * (b.length - 1))]; };
console.log(JSON.stringify({line, V, LD, reason: sm.reason, laps: sm.laps.map(l => +l.elapsed.toFixed(2)), p95: q(rows.map(r => r.e), .95), med: q(rows.map(r => r.e), .5)}));
for (let a = 0; a < 26; a += 1) { const m = rows.filter(r => r.s >= a && r.s < a + 1); if (m.length > 5) { const p9 = q(m.map(r => r.e), .95); if (p9 > .06) console.log(`  s ${a}-${a + 1}: p95 ${(p9 * 100).toFixed(1)} cm, v ${(m.reduce((x, r) => x + r.v, 0) / m.length).toFixed(2)}`); } }
