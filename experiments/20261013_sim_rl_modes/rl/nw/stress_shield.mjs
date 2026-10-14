// 次の段(作り直し)のための調べ(2026-10-14、no_wall の学習の段): 試験の組を使わずに、舵が遅く遅延 1 の物理と斜めの開始の組(検証の Halton の開始・検証の種)で、
// シールドの版を比べる。--tree rl(v2)/ rlv(案。rl の写しに train/no_wall/diag/shield_v3_idea.diff を当てたもの。調べたあと消した)。判定には使わない
//   node rl/nw/stress_shield.mjs --tree rl --policy <policy.json> --n 120 --out <結果.json>
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const EXP = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const tree = path.join(EXP, opt('--tree', 'rl'));
const {runModeEpisode, readJson} = await import(pathToFileURL(path.join(tree, 'episode.mjs')).href);
const {haltonStarts} = await import(pathToFileURL(path.join(tree, 'eval_modes.mjs')).href);
const {sampleDrSpec} = await import(pathToFileURL(path.join(tree, 'env_modes.mjs')).href);
const {CATALOGUE} = await import(pathToFileURL(path.join(tree, 'runtime/catalogue.mjs')).href);
const {createRng, mix32} = await import(pathToFileURL(path.join(tree, 'runtime/rng.mjs')).href);
const DRT = CATALOGUE.common.dr_specs['DR-T'];
const n = +opt('--n', '120'), W = readJson(opt('--policy'));
const starts = haltonStarts('no_wall', 'validation', n + 57).slice(57);   // 検証の組(最初の 7 + 50 = 57 個)と重ならない検証の範囲の Halton
const rows = [];
for (let i = 0; i < starts.length; i++) {
  const seed = 800000 + 2000 * 2 + 500 + i;   // 検証の種の範囲(試験の 900000 台と重ならない)
  const d = sampleDrSpec(createRng(mix32(seed)), DRT);
  const u = createRng(mix32(seed, 0x51));
  d.physics.steerRate = .8 + .1 * u.uniform(); d.physics.brakeDeceleration = .7 + .5 * u.uniform(); d.delay = 1; d.steerNoise = 0;
  const r = await runModeEpisode({mode: 'no_wall', weights: W, start: {...starts[i], seed}, dr: d, keepLog: true});
  const fb = {}; for (const q of r.log) if (q.shield?.fb) fb[q.shield.fb] = (fb[q.shield.fb] ?? 0) + 1;
  rows.push({i, start: starts[i], dr: d, reason: r.reason, collisions: r.collisions, min_clearance: r.min_clearance, valid: r.valid_episode && r.valid_laps >= 1,
    shield_frames: r.shield_frames, fallbacks: fb});
}
const mc = rows.map(r => r.min_clearance).sort((a, b) => a - b);
const sum = {tree: opt('--tree', 'rl'), n: rows.length, collisions: rows.filter(r => r.collisions).length, below5cm: rows.filter(r => r.min_clearance < .05).length,
  below8cm: rows.filter(r => r.min_clearance < .08).length, min: mc[0], p5: mc[Math.floor(.05 * mc.length)], median: mc[Math.floor(.5 * mc.length)],
  valid: rows.filter(r => r.valid).length, shield_frac: rows.reduce((a, r) => a + r.shield_frames, 0) / (600 * rows.length)};
fs.writeFileSync(opt('--out'), JSON.stringify({summary: sum, rows}, null, 1));
console.log(JSON.stringify(sum));
