// 1 回のフレームごとの中身(方策の指令・シールド・見積もり)
import {runOne} from './run_nw2.mjs';
const [g, o, y, sr, delay] = process.argv.slice(2).map(Number);
const RT = await import(process.env.NW2_RT + '/mode_runtime.mjs');
const r = await runOne({mode: 'no_wall', start: {gate: g, left_offset_m: o, left_yaw_deg: y, seed: 1810000 + g}, dr: {physics: {steerRate: sr}, delay}, keepPath: true});
console.log(JSON.stringify({col: r.collisions, coll: r.collision, min: r.min_clearance, sh: r.shield_frames, fb: r.labels.fb}));
r.path.forEach((p, f) => { if (f % 3 === 0 || p[7]) console.log(f, 'v', p[2].toFixed(2), 's', p[3].toFixed(2), 'ey', p[4].toFixed(3), 'cmd', p[5].toFixed(3), p[6].toFixed(2), 'sh', p[7], 'δ', p[8].toFixed(3), p[9] ?? ''); });
