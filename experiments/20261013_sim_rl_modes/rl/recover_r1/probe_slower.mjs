// 離脱復活の学習の段(r1)の下調べ(2): 舵のパルスでぶつかった出来事で、「乱しが始まった時の速さが c 倍だったら」戻れたか。
// 同じ回を高速版のコアで操作列から作り直し、乱しの始まり ps で速さを c 倍に書き換え(状態の書き換え)、乱しの間は差し込みの舵 + 速さの指令は方策(上限 c·v)、
// 乱しのあとは方策(上限 c·v を 1 秒、そのあと上限なし)で 3 秒走らせる。方策の観測は mode_runtime の教師(乱しを止めたもの)に毎フレーム与える。
//   node rl/recover_r1/probe_slower.mjs --policy <policy.json> --n 40 --out <結果.json>
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {runModeEpisode, readJson, BASE_WORLD, EXP} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {activeWorldFor, simConfig, PHYSICS, PERTURB_KEYS} from '../../snapshot/rl/env_core.mjs';
import {createModeTeacher} from '../runtime/mode_runtime.mjs';
import {LOOP_LENGTH} from '../runtime/features.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = readJson(args.policy), n = +(args.n ?? 40), hbase = +(args.halton ?? 40001), sbase = +(args.seed ?? 500000);
const factors = (args.factors ?? '1,0.8,0.6,0.45,0.3').split(',').map(Number);
const core = await import(pathToFileURL(path.join(EXP, 'rl/fast_core_modes.mjs')).href);
const dist = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution;

function branch(start, actions, ev, c, H = 90) {
  const {world, variation} = activeWorldFor(BASE_WORLD, start.gate, start.left_offset_m, start.left_yaw_deg);
  const cfg = simConfig({episodeMode: 'continuous', timeLimit: 20, seed: start.seed});
  const sim = core.createSimulation(world, cfg);
  sim.reset({seed: cfg.seed, scenario: 'baseline'});
  let snap = sim.peek();
  const cfgN = {...JSON.parse(JSON.stringify(snap.config)), ...Object.fromEntries(PERTURB_KEYS.map(k => [k, PHYSICS[k]]))};
  const teacher = createModeTeacher(world, cfgN, variation, {mode: 'recover', weights, overrides: {perturbation: false}});
  const pe = ev.pe;
  let cap = Infinity;
  for (let f = 0; f < pe + H; f++) {
    const st = snap.state;
    if (st.terminal) return {ok: st.reason === 'collection_duration_complete', at: f};
    const out = teacher.action({state: st, metrics: snap.metrics, config: cfgN});
    let cmd;
    if (f < ev.ps) cmd = actions[f];
    else {
      if (f === ev.ps) { const s = sim.peek().state; s.speed *= c; cap = c < 1 ? Math.max(.3, s.speed) : Infinity; }
      if (f < pe) cmd = {steering: actions[f].steering, speed: Math.min(out.speed, cap)};
      else cmd = {steering: out.steering, speed: f < pe + 30 ? Math.min(out.speed, cap) : out.speed};
    }
    sim.step(cmd, true); snap = sim.peek();
  }
  return {ok: !(snap.state.terminal && snap.state.reason === 'collision'), at: pe + H};
}
const rows = [];
const okBy = Object.fromEntries(factors.map(c => [c, 0]));
let nEv = 0, nDuring = 0;
for (let i = 0; i < n; i++) {
  const st0 = {...haltonStart(hbase + i, dist), seed: sbase + i};
  const r = await runModeEpisode({mode: 'recover', weights, start: st0, coreKind: 'fast', audit: false, keepLog: true});
  if (r.invalid_spawn || !r.collisions) continue;
  const coll = r.frames, ev = [...r.events].reverse().find(e => e.ps <= coll);
  if (!ev || ev.type !== 'steer_pulse') continue;
  if (ev.pe == null) { console.log(`P${i} ${ev.cls} 乱しの最中にぶつかった(pe なし。数えない)`); nDuring++; continue; }
  nEv++;
  const res = {};
  for (const c of factors) { const b = branch(st0, r.actions, ev, c); res[c] = b.ok; if (b.ok) okBy[c]++; }
  const v = r.log[ev.ps].v;
  rows.push({i, ev, v_ps: v, s_mod: +(r.log[ev.ps].s % LOOP_LENGTH).toFixed(2), res});
  console.log(`P${i} ${ev.cls} s=${(r.log[ev.ps].s % LOOP_LENGTH).toFixed(2)} v=${v.toFixed(2)} frames=${ev.pe - ev.ps} → ` + factors.map(c => `${c}:${res[c] ? '○' : '×'}`).join(' '));
}
console.log(`舵のパルスでぶつかった ${nEv} 件(ほかに乱しの最中にぶつかった ${nDuring} 件)。速さを c 倍にして助かった数: ` + factors.map(c => `${c}→${okBy[c]}`).join(' '));
if (args.out) fs.writeFileSync(args.out, JSON.stringify({policy: args.policy, n, factors, nEv, nDuring, okBy, rows}));
