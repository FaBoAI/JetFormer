// 離脱復活 r4(2026-10-14、報告だけ): スタートのゲートの区間(s 0.30〜3.70 m)の右の余地を、方策ごとに乱しなしで測る(図 recover_r4_band_line.png の材料)。
// 開始は STD と protocol の乱しなしの線の作り方(種 990000 + i、ゲート 0・±0.03 m・±3°)の 8 つ。r4 の走行中の仕組み(シールドあり)・元の sim-core.js。
//   node r3/rc4/diag/band_line_rc4.mjs --out band_line.json <名前>=<policy.json> …
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runModeEpisode, readJson} from '../episode_rc4.mjs';
import {createRng, mix32} from '../rt/rng.mjs';
import {tableAt} from '../rt/line_profiles.mjs';
import {LOOP_LENGTH} from '../rt/features.mjs';

const args = process.argv.slice(2), oi = args.indexOf('--out'), out = args[oi + 1];
const pols = args.filter((a, i) => i !== oi && i !== oi + 1).map(a => a.split('='));
const starts = [{gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 1}];
for (let i = 0; i < 8; i++) { const seed = 990000 + i, rng = createRng(mix32(seed)); starts.push({gate: 0, left_offset_m: rng.uniform(-.03, .03), left_yaw_deg: rng.uniform(-3, 3), seed}); }
const res = {schema: 'jf-rc4-band-line/v1', band: [.30, 3.70], policies: {}};
for (const [name, p] of pols) {
  const weights = readJson(p), sha = crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const pts = [];
  const laps = [];
  for (const st of starts) {
    const r = await runModeEpisode({mode: 'recover', weights, policySha256: sha, start: st, overrides: {perturbation: false}, audit: false});
    laps.push(r.laps.map(l => l.elapsed));
    for (const q of r.log) {
      const sm = ((q.s % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH;
      if (sm > 4.5 || q.f < 30) continue;
      pts.push([+sm.toFixed(3), +q.ey.toFixed(4), +(q.ey + tableAt('rhoR', q.s)).toFixed(4), +q.dmin.toFixed(4), +q.v.toFixed(3)]);
    }
  }
  res.policies[name] = {path: p, sha256: sha, laps, points: pts, cols: ['s_mod_L', 'ey', 'room_right', 'dmin', 'v']};
}
fs.writeFileSync(out, JSON.stringify(res));
console.log(Object.fromEntries(Object.entries(res.policies).map(([k, v]) => {
  const b = v.points.filter(q => q[0] >= .3 && q[0] <= 3.7);
  return [k, {n: b.length, room_right_min: Math.min(...b.map(q => q[2])), room_right_mean: b.reduce((a, q) => a + q[2], 0) / b.length, dmin_min: Math.min(...b.map(q => q[3]))}];
})));
