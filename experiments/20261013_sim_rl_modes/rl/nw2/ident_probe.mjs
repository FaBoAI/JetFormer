// 舵の速さを走りながら見積もれるか(r1 の方策・シールドあり)。|Δδ| が舵の速さ × dt にどれだけ当たるか
import {runOne} from './run_nw2.mjs';
const rows = [];
const starts = [[0, 0, 0], [16, 0, 0], [21, .08, 12], [8, -.1, -10], [3, .05, 5], [27, 0, 8]];
for (const [label, dr] of [['nom', {}], ['s08d1', {physics: {steerRate: .8}, delay: 1}], ['s15d0', {physics: {steerRate: 1.5}}], ['s08d0g094', {physics: {steerRate: .8}, steerGain: .94}]]) {
  for (const [g, o, y] of starts) {
    const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1800000 + g}, dr, keepPath: true});
    const rate = dr.physics?.steerRate ?? 1.2, dt = 1 / 30;
    const d = r.path.map(p => p[8]);
    let sat = 0, firstSat = null, maxd = 0;
    for (let i = 1; i < d.length; i++) { const dd = Math.abs(d[i] - d[i - 1]); maxd = Math.max(maxd, dd / dt); if (dd > rate * dt * (1 - 1e-6)) { sat++; if (firstSat === null) firstSat = i; } }
    rows.push({label, g, o, y, col: r.collisions, min: +(r.min_clearance * 100).toFixed(2), frames: d.length, sat, firstSat, maxRateObs: +maxd.toFixed(4), sh: r.shield_frames});
    console.log(JSON.stringify(rows.at(-1)));
  }
}
