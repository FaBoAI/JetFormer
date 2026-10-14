// 離脱復活の学習の段(r1)の下調べ: 方策が乱しのあとにぶつかった出来事で、「乱しのあとの操作を替えれば」ぶつからずに済んだか(物理として戻れるか)。
// 同じ回を高速版のコアで操作列から作り直し、乱しの終わり pe(と乱しの始め ps)から簡単な操縦に替えて 3 秒走らせる:
//   pure pursuit(R の前方 Ld m の点、Ld 0.4〜1.2)× 速さの指令(0・1.0・2.0・2.5・3.5 m/s・その時の速さのまま)、乱しの間のブレーキの有無
//   node rl/recover_r1/probe_feasible.mjs --policy <policy.json> --n 40 --out <結果.json>
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD, EXP} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {activeWorldFor, simConfig} from '../../snapshot/rl/env_core.mjs';
import {loopPointAt, projectWindow, LOOP_LENGTH} from '../runtime/features.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = readJson(args.policy), n = +(args.n ?? 40), hbase = +(args.halton ?? 40001), sbase = +(args.seed ?? 500000);
const core = await import(pathToFileURL(path.join(EXP, 'rl/fast_core_modes.mjs')).href);
const dist = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const pt = {x: 0, z: 0, tx: 0, tz: 0}, pr = {s: 0, d: 0, px: 0, pz: 0, tx: 1, tz: 0};

function makeSim(start) {
  const {world} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
  const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: start.seed});
  const sim = core.createSimulation(world, cfg);
  sim.reset({seed: cfg.seed, scenario: 'baseline'});
  return sim;
}
// 操作列 actions[0..k-1] を流したあと、ctl(st, sPrev) の操縦で H フレーム。ぶつからなければ true
function branch(start, actions, k, ctl, H, sAt) {
  const sim = makeSim(start);
  for (let f = 0; f < k; f++) { sim.step(actions[f], true); if (sim.peek().state.terminal) return {ok: false, at: f}; }
  let s = sAt;
  for (let h = 0; h < H; h++) {
    const st = sim.peek().state;
    if (st.terminal) return {ok: st.reason === 'collection_duration_complete', at: k + h};
    projectWindow(st.x, st.z, s - .5, s + 1.5, pr); s = pr.s;
    sim.step(ctl(st, s, h), true);
  }
  const st = sim.peek().state;
  return {ok: !(st.terminal && st.reason === 'collision'), at: k + H};
}
const pp = (Ld, v) => (st, s) => {
  loopPointAt(s + Ld, pt);
  const dx = pt.x - st.x, dz = pt.z - st.z, alpha = wrap(Math.atan2(-dz, dx) - st.yaw), ld = Math.max(Math.hypot(dx, dz), .2);
  return {steering: Math.max(-.52, Math.min(.52, Math.atan2(2 * .26 * Math.sin(alpha), ld))), speed: v === 'keep' ? st.speed : v};
};
const out = [];
let nFail = 0, nSaved = 0, nSavedAfterPe = 0;
for (let i = 0; i < n; i++) {
  const st0 = {...haltonStart(hbase + i, dist), seed: sbase + i};
  const r = await runModeEpisode({mode: 'recover', weights, start: st0, coreKind: 'fast', audit: false, keepLog: true});
  if (r.invalid_spawn || !r.collisions) continue;
  const coll = r.frames, ev = [...r.events].reverse().find(e => e.ps <= coll);
  if (!ev) continue;
  nFail++;
  const pe = ev.pe ?? coll, actions = r.actions, s_pe = r.log[Math.min(pe, r.log.length - 1)].s, s_ps = r.log[ev.ps].s;
  const H = 90;
  const tried = [];
  let saved = null, savedPe = null;
  // (1) pe から替える
  for (const Ld of [.4, .6, .8, 1.2]) for (const v of [0, 1.0, 2.0, 2.5, 3.5, 'keep']) {
    if (pe >= coll) break;
    const b = branch(st0, actions, pe, pp(Ld, v), H, s_pe);
    tried.push({from: 'pe', Ld, v, ok: b.ok});
    if (b.ok && !savedPe) savedPe = {Ld, v};
  }
  // (2) 乱しの間はブレーキ(舵は差し込みのまま)+ pe から替える
  if (!savedPe) {
    const brakeActs = actions.map((a, f) => (f >= ev.ps && f < pe ? {steering: a.steering, speed: 0} : a));
    for (const Ld of [.4, .6, .8, 1.2]) for (const v of [0, 1.0, 2.0, 'keep']) {
      const b = branch(st0, brakeActs, pe, pp(Ld, v), H, s_pe);
      tried.push({from: 'pe+brake', Ld, v, ok: b.ok});
      if (b.ok && !saved) saved = {Ld, v, brake: true};
    }
  }
  if (savedPe) nSavedAfterPe++;
  if (savedPe || saved) nSaved++;
  const v_ps = r.log[ev.ps].v, v_pe = r.log[Math.min(pe, r.log.length - 1)].v;
  out.push({i, start: st0, ev, coll, s_mod: +(s_ps % LOOP_LENGTH).toFixed(2), v_ps, v_pe, savedPe, saved, tried});
  console.log(`P${i} ${ev.type}/${ev.cls} s=${(s_ps % LOOP_LENGTH).toFixed(2)} v=${v_ps.toFixed(2)} ps=${ev.ps} pe=${pe} coll=${coll} → pe から: ${savedPe ? JSON.stringify(savedPe) : 'だめ'} / ブレーキつき: ${saved ? JSON.stringify(saved) : (savedPe ? '-' : 'だめ')}`);
}
console.log(`失敗 ${nFail}、pe から替えて助かる ${nSavedAfterPe}、乱しの間のブレーキも入れて助かる ${nSaved}`);
if (args.out) fs.writeFileSync(args.out, JSON.stringify({policy: args.policy, n, nFail, nSavedAfterPe, nSaved, rows: out}));
