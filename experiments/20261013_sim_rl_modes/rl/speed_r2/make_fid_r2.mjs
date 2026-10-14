#!/usr/bin/env node
// 速さいろいろ r2 の F-ID の列(STD の連続 20 秒の 600 フレームの snapshot と Node の操作)を c_v 1.0・0.5 で書く(2026-10-14 学習の段 speed r2)。
// 統合の段がブラウザの teacher.js に同じ snapshot を与えて操作を比べる(rl/check_fid.mjs と同じ比べ方)。
//   node rl/speed_r2/make_fid_r2.mjs <policy.json> <出力のディレクトリ>
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {writeFid} from '../eval_modes.mjs';

const [pp, outDir] = process.argv.slice(2);
const W = JSON.parse(fs.readFileSync(pp, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(pp)).digest('hex');
fs.mkdirSync(outDir, {recursive: true});
for (const cv of [1.0, .5]) {
  const out = path.join(outDir, `fid_std_speed_r2_cv${cv}.json`);
  await writeFid('speed', W, sha, out, {params: {c_v: cv}});
  console.log(out);
}
