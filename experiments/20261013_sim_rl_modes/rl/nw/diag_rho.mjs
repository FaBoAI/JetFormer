// 診断: 1 回の ρ_req・ρ_comb > 1 のフレームの位置と中身(2026-10-14)
import {runModeEpisode, readJson, BASE_WORLD} from '../episode.mjs';
import {prepareGeometry, surfaceMu, G} from '../runtime/features.mjs';
import {activeWorldFor} from '../../snapshot/rl/env_core.mjs';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const start = {gate: +opt('--gate', '0'), left_offset_m: 0, left_yaw_deg: 0, seed: 1};
const r = await runModeEpisode({mode: opt('--mode', 'no_wall'), weights: readJson(opt('--policy')), start});
const {world} = activeWorldFor(BASE_WORLD, start.gate, 0, 0);
const g = prepareGeometry(world), wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
let prev = null, n = 0, nc = 0, nr = 0;
const fullTrace = r.trace;
for (let i = 0; i < fullTrace.length; i++) {
  const q = fullTrace[i], mu = surfaceMu(g, q.x, q.z, 1);
  if (q.v > .3 && prev) {
    n++;
    const al = (q.v - prev.v) * 30, om = wrap(q.yaw - prev.yaw) * 30, alat = q.v * om, rc = Math.hypot(al, alat) / (mu * G);
    const rq = q.v * q.v * Math.abs(Math.tan(q.d)) / (.26 * mu * G);
    if (rc > 1 || rq > 1) { if (rc > 1) nc++; if (rq > 1) nr++;
      console.log(i, 's', r.log[i].s.toFixed(2), 'mu', mu, 'v', q.v.toFixed(2), 'a_long', al.toFixed(2), 'a_lat', alat.toFixed(2), 'rc', rc.toFixed(2), 'rq', rq.toFixed(2), 'cmd', r.actions[i].speed.toFixed(2)); }
  }
  prev = q;
}
console.log('running', n, 'comb>1', nc, 'req>1', nr);
