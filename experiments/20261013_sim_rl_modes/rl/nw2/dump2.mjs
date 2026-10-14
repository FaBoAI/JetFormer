// 1 回のフレームごとの中身(spec を JSON で)
import {runOne} from './run_nw2.mjs';
const spec = JSON.parse(process.argv[2]);
const r = await runOne({mode: 'no_wall', ...spec, keepPath: true});
console.log(JSON.stringify({col: r.collisions, coll: r.collision, min: r.min_clearance, sh: r.shield_frames, fb: r.labels.fb, valid: r.valid_episode, laps: r.valid_laps}));
const every = +(process.argv[3] ?? 3), from = +(process.argv[4] ?? 0), to = +(process.argv[5] ?? 9999);
r.path.forEach((p, f) => { if (f >= from && f <= to && (f % every === 0 || p[7])) console.log(f, 'x', p[0].toFixed(2), 'z', p[1].toFixed(2), 'v', p[2].toFixed(2), 's', p[3].toFixed(2), 'ey', p[4].toFixed(3), 'cmd', p[5].toFixed(3), p[6].toFixed(2), 'sh', p[7], 'δ', p[8].toFixed(3), p[9] ?? ''); });
