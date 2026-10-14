// 離脱復活の記録のプランとデータセットの説明に書く数字(2026-10-14、学習の段 recover r2。確認 recover r1 の指摘 5)。
//   記録の決まりの開始(recover.recording.start_distribution の Halton 1〜・開始の衝突なし、種 recording_base = 700000 + 1000·3 + 100·0 + i)を
//   元の sim-core.js・ブラウザと同じ mode_runtime.mjs で、決めた数の候補だけ走らせ(衝突で替えない = 全部数える)、
//   (1) 20 秒を走り切る割合、(2) 走り切ったシーンのうち本当のゲートの外しがある割合、(3) 乱しの始まりの位置(R 上の s)の集まり を出す。
//   node rl/recover_r2/recording_facts.mjs --policy <policy.json> [--candidates 96] --out <結果.json>
import fs from 'node:fs';
import crypto from 'node:crypto';
import {runModeEpisode, readJson, BASE_WORLD} from '../episode.mjs';
import {haltonStart} from '../runtime/rng.mjs';
import {CATALOGUE} from '../runtime/catalogue.mjs';
import {activeWorldFor} from '../../snapshot/rl/env_core.mjs';
import {prepareGeometry, clearance, LOOP_LENGTH} from '../runtime/features.mjs';
import {maskFromLearn} from '../runtime/perturb.mjs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => (x.startsWith('--') ? a.concat([[x.slice(2), arr[i + 1]]]) : a), []));
const weights = readJson(args.policy), sha = crypto.createHash('sha256').update(fs.readFileSync(args.policy)).digest('hex');
const nCand = +(args.candidates ?? 96);
const dist = CATALOGUE.modes.find(m => m.key === 'recover').recording.start_distribution, geom = prepareGeometry(BASE_WORLD);
const runs = [];
for (let h = 1, i = 0; runs.length < nCand && h < 10000; h++) {
  const st = haltonStart(h, dist);
  const {world} = activeWorldFor(BASE_WORLD, st.gate, st.left_offset_m, st.left_yaw_deg);
  if (!(clearance(geom, world.spawn.x, world.spawn.z, world.spawn.yaw, .42, .19, .5) > 1e-9)) continue;
  const seed = 703000 + i; i++;
  const r = await runModeEpisode({mode: 'recover', weights, policySha256: sha, start: {...st, seed}, coreKind: 'original', audit: true});
  const mask = maskFromLearn(r.log.map(q => q.l));
  runs.push({halton: h, seed, start: st, valid: r.valid_episode, reason: r.reason, frames: r.frames, collision: r.collision, laps: r.laps.map(l => +l.elapsed.toFixed(3)),
    gate_miss: r.gate_miss, gate_misses: r.gate_misses, gate_stuck_max_m: +r.gate_stuck_max_m.toFixed(3), min_clearance: r.min_clearance,
    events: r.events.map(e => ({type: e.type, cls: e.cls, ps: e.ps, pe: e.pe, s_mod: +(((e.s % LOOP_LENGTH) + LOOP_LENGTH) % LOOP_LENGTH).toFixed(3)})),
    mask_frac: mask.reduce((a, b) => a + b, 0) / 600});
  process.stderr.write(`${runs.length}/${nCand}\r`);
}
const ok = runs.filter(r => r.valid), evAll = runs.flatMap(r => r.events), evOk = ok.flatMap(r => r.events);
const share = (evs, a, b) => evs.length ? evs.filter(e => e.s_mod >= a && e.s_mod < b).length / evs.length : null;
const hist = evs => { const h = {}; for (const e of evs) { const k = (Math.floor(e.s_mod / .25) * .25).toFixed(2); h[k] = (h[k] ?? 0) + 1; }
  return Object.entries(h).sort((x, y) => y[1] - x[1]).slice(0, 5).map(([k, n]) => ({s_from: +k, n, share: n / evs.length})); };
// 15 シーン(記録の 5 分)を集めるのに要る候補の数(この列の順で)
let need15 = null; { let c = 0; for (let k = 0; k < runs.length; k++) { if (runs[k].valid) c++; if (c === 15) { need15 = k + 1; break; } } }
const wilson = (k, n, z = 1.96) => { const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, w = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [c - w, c + w]; };
const res = {schema: 'jf-sim-rl-recover-recording-facts/v1', created_at: new Date().toISOString(), policy: args.policy, policy_sha256: sha,
  rule: 'recover.recording.start_distribution の Halton 1〜(開始の衝突なし)・種 703000 + i・元の sim-core.js・強さ auto・衝突しても替えずに全部数える',
  candidates: runs.length, completed: ok.length, completion_rate: ok.length / runs.length, completion_ci95: wilson(ok.length, runs.length),
  candidates_needed_for_first_15: need15,
  completed_with_gate_miss: ok.filter(r => r.gate_miss).length, gate_miss_rate_completed: ok.length ? ok.filter(r => r.gate_miss).length / ok.length : null,
  gate_miss_ci95: ok.length ? wilson(ok.filter(r => r.gate_miss).length, ok.length) : null,
  first15: (() => { const f = ok.slice(0, 15); return {scenes: f.length, with_gate_miss: f.filter(r => r.gate_miss).length, events_per_scene: f.reduce((a, r) => a + r.events.length, 0) / Math.max(1, f.length),
    masked_fraction: f.reduce((a, r) => a + r.mask_frac, 0) / Math.max(1, f.length), min_clearance: Math.min(...f.map(r => r.min_clearance ?? Infinity))}; })(),
  event_starts: {all: {n: evAll.length, black_sheet_exit_8_00_8_25: share(evAll, 8.0, 8.25), back_straight_exit_19_25_19_50: share(evAll, 19.25, 19.5),
      back_straight_exit_19_25_19_55: share(evAll, 19.25, 19.55), top: hist(evAll)},
    completed: {n: evOk.length, black_sheet_exit_8_00_8_25: share(evOk, 8.0, 8.25), back_straight_exit_19_25_19_50: share(evOk, 19.25, 19.5), top: hist(evOk)}},
  runs};
fs.writeFileSync(args.out, JSON.stringify(res, null, 1));
const {runs: _r, ...head} = res;
console.log(JSON.stringify(head, null, 1));
