// 離脱復活の学習の段(r1)の下調べ: 記録の乱し(RECORD_SPEC)を、R を追う簡単な運転(pure pursuit・一定の速さ)に入れたら、どの速さなら戻れるか。
// 方策に関係なく「その速さで乱しを受けて戻れるか」(物理の上限の目安)を見る。高速版のコア、試験・検証の組と重ならない種・開始。
//   node rl/recover_r1/probe_pp.mjs --speeds 1.0,1.5,2.0,2.5,3.0 --n 40 [--cls 強] [--ld 0.6] --out <結果.json>
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {BASE_WORLD, EXP} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {activeWorldFor, simConfig} from '../../snapshot/rl/env_core.mjs';
import {loopPointAt, projectWindow, LOOP_LENGTH, prepareGeometry, clearance} from '../runtime/features.mjs';
import {createPerturber, RECORD_SPEC} from '../runtime/perturb.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const n = +(args.n ?? 40), speeds = (args.speeds ?? '1.5,2.0,2.5').split(',').map(Number), Ld = +(args.ld ?? .6), cls = args.cls ?? null;
const core = await import(pathToFileURL(path.join(EXP, 'rl/fast_core_modes.mjs')).href);
const dist = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const pt = {x: 0, z: 0, tx: 0, tz: 0}, pr = {s: 0, d: 0, px: 0, pz: 0, tx: 1, tz: 0};
const res = {Ld, cls, by_speed: {}};
for (const V of speeds) {
  let valid = 0, tot = 0, evs = 0, fails = [];
  const byCls = {};
  for (let i = 0; i < n; i++) {
    const start = {...haltonStart(40001 + i, dist), seed: 500000 + i};
    const {world} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
    const geom = prepareGeometry(world);
    const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: start.seed});
    const sim = core.createSimulation(world, cfg);
    sim.reset({seed: cfg.seed, scenario: 'baseline'});
    let st = sim.peek().state;
    if (st.terminal) continue;
    tot++;
    projectWindow(st.x, st.z, 0, LOOP_LENGTH, pr);
    let s = pr.s;
    const pert = createPerturber({spec: RECORD_SPEC, seed: start.seed, cls, maxSteer: .52, wheelbase: .26});
    while (!st.terminal) {
      projectWindow(st.x, st.z, s - .5, s + 1.5, pr); s = pr.s;
      const ey = (st.x - pr.px) * pr.tz - (st.z - pr.pz) * pr.tx;
      loopPointAt(s + Ld, pt);
      const dx = pt.x - st.x, dz = pt.z - st.z, alpha = wrap(Math.atan2(-dz, dx) - st.yaw), ld = Math.max(Math.hypot(dx, dz), .2);
      let cmd = {steering: Math.max(-.52, Math.min(.52, Math.atan2(2 * .26 * Math.sin(alpha), ld))), speed: V};
      const dmin = clearance(geom, st.x, st.z, st.yaw, .42, .19, .5);
      const r = pert.step(st.frame, {s, ey, dmin, x: st.x, z: st.z, yaw: st.yaw, speedCap: 4.5, policy: cmd});
      if (r.inj) cmd = {steering: r.inj.steering ?? cmd.steering, speed: r.inj.speed ?? cmd.speed};
      sim.step(cmd, true); st = sim.peek().state;
    }
    const ok = st.reason === 'collection_duration_complete';
    if (ok) valid++;
    evs += pert.events.length;
    for (const e of pert.events) { byCls[`${e.type}/${e.cls}`] ??= {n: 0, fail: 0}; byCls[`${e.type}/${e.cls}`].n++; }
    if (!ok) {
      const e = [...pert.events].reverse().find(q => q.ps <= st.frame);
      if (e) { byCls[`${e.type}/${e.cls}`].fail++; fails.push(`${e.type}/${e.cls}@${(e.s % LOOP_LENGTH).toFixed(2)}`); } else fails.push(`${st.reason}(乱しの前)`);
    }
  }
  res.by_speed[V] = {valid: `${valid}/${tot}`, events_mean: evs / tot, byCls, fails};
  console.log(`V=${V} 有効 ${valid}/${tot} 出来事 ${(evs / tot).toFixed(2)}/回`, JSON.stringify(byCls), fails.join(' '));
}
if (args.out) fs.writeFileSync(args.out, JSON.stringify(res));
