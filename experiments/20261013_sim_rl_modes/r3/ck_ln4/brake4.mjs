// 確認 line r4: 確認 r3 の指摘 2(合格する候補が無いとき方策の指令を通さない)を、シールドを直接呼んで確かめる。
// 柵(z = −2.8 の下の直線)へ向かって速く走っている姿勢をいくつも作り、候補の値(select 'v5b')と no_wall の値(select 'v5')で、
// 返す指令・fallback を比べる。どの候補も止まりきれない姿勢では、v5b は速さ 0 を返し、方策の提案(速さ > 0)を通さないはず
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {CAND, WORLD, LOOP, PHYS} from './run4.mjs';
const {prepareGeometry} = await import(pathToFileURL(`${CAND}/runtime/features.mjs`).href);
const {createShield} = await import(pathToFileURL(`${CAND}/runtime/shield.mjs`).href);
const MODE = JSON.parse(fs.readFileSync(`${CAND}/mode.json`, 'utf8'));
const geometry = prepareGeometry(WORLD);
const config = {...PHYS, seed: 1, timeLimit: 20, episodeMode: 'continuous-laps-v1', substep: .01};
const rows = [];
for (const sel of ['v5b', 'v5']) {
  for (const gap of [.15, .3, .5, .8, 1.2]) for (const v of [1.5, 2.5, 3.5]) for (const dyaw of [0, .2, -.2]) {
    const sh = createShield({geometry, config, params: {...JSON.parse(JSON.stringify(MODE.runtime.shield_params)), select: sel}});
    // 柵の上の点 (0.8, −2.8) の手前 gap(車の前の端から)。向きは −z(yaw = π/2)から dyaw
    const yaw = Math.PI / 2 + dyaw, x = .8, z = -2.8 + .0095 + gap + .21;
    const state = {x, z, yaw, speed: v, steering: 0, frame: 5, sim_time: 5 / 30};
    const s = LOOP.proj(x, z).s;
    const prop = {steering: 0, speed: v + .5};
    const r = sh.filter(state, s, {steering: 0, speed: v}, prop);
    rows.push({sel, gap, v, dyaw, out_speed: r.cmd.speed, out_steer: r.cmd.steering, replaced: r.replaced, fb: r.fallback ?? null});
  }
}
const sum = {};
for (const r of rows) { const k = `${r.sel}:${r.fb}`; sum[k] = sum[k] ?? {n: 0, speed_pos: 0, kept_proposal: 0}; sum[k].n++; if (r.out_speed > 1e-9) sum[k].speed_pos++; if (!r.replaced) sum[k].kept_proposal++; }
const nopass_v5b = rows.filter(r => r.sel === 'v5b' && r.fb === 'brake_first');
const res = {rows, summary: sum, v5b_brake_first_n: nopass_v5b.length, v5b_brake_first_positive_speed: nopass_v5b.filter(r => r.out_speed > 1e-9).length,
  v5_maxe_positive_speed_kept_proposal: rows.filter(r => r.sel === 'v5' && r.fb === 'max_e' && !r.replaced && r.out_speed > 1e-9).length};
fs.writeFileSync('/home/jetson/jetformer_reports/sim_rl_modes_20261013/r3/check_line_r4/brake4.json', JSON.stringify(res, null, 1));
console.log(JSON.stringify({summary: sum, v5b_brake_first_n: res.v5b_brake_first_n, v5b_brake_first_positive_speed: res.v5b_brake_first_positive_speed, v5_maxe_kept: res.v5_maxe_positive_speed_kept_proposal}));
