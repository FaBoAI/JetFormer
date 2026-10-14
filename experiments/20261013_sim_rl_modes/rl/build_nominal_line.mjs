#!/usr/bin/env node
// 方策の乱しなしの線 ē_y(s)・ψ̄(s)(protocol definitions.nominal_line: 8 回、種 990000 + i、ゲート 0・±3 cm・±3°、連続 20 秒)を作る。
// 障害物よけの配置(最速の方策のふだんの線の上に置く)と、離脱復活の「戻った」の判定に使う。2026-10-13
//   node rl/build_nominal_line.mjs --mode fastest --policy <policy.json> --out rl/nominal_line_fastest.json
import fs from 'node:fs';
import {runModeEpisode, readJson} from './episode.mjs';
import {nominalLine, sceneFromEpisode} from './variation_metrics.mjs';
import {createRng, mix32} from './runtime/rng.mjs';
import crypto from 'node:crypto';

export async function computeNominalLine({mode, weights, policySha256 = null, coreKind = 'original', params = {}}) {
  const scenes = [], runs = [];
  for (let i = 0; i < 8; i++) {
    const seed = 990000 + i, rng = createRng(mix32(seed));
    const start = {gate: 0, left_offset_m: rng.uniform(-.03, .03), left_yaw_deg: rng.uniform(-3, 3), seed};
    // 乱しなし(recover は乱しを止める。シールドはモードのまま)
    const r = await runModeEpisode({mode, weights, policySha256, start, params, coreKind, audit: false, overrides: {perturbation: false}});
    scenes.push(sceneFromEpisode(r)); runs.push({seed, start, reason: r.reason, collisions: r.collisions, laps: r.laps.map(l => l.elapsed)});
  }
  return {...nominalLine(scenes), mode, policy_sha256: policySha256, runs_detail: runs};
}
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
  const p = opt('--policy'), out = opt('--out'), mode = opt('--mode', 'fastest');
  const w = readJson(p), sha = crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const line = await computeNominalLine({mode, weights: w, policySha256: sha});
  fs.writeFileSync(out, JSON.stringify(line));
  console.log('書いた', out, 'runs', line.runs_detail.map(r => r.reason + ' ' + r.laps.map(x => x.toFixed(2)).join('/')).join(' | '));
}
