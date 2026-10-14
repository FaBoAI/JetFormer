#!/usr/bin/env node
// 障害物よけ r2 の F-ID の列(600 フレームの snapshot と Node の操作)を書く(2026-10-14 学習の段 obstacle r2)。
// r1 の候補と同じ validation-000 の配置(柱 2・箱 1)・STD の開始・連続 20 秒。統合の段がブラウザの teacher.js(仮想コース default-obstacles)に
// 同じ snapshot を与えて操作を比べる(rl/check_fid.mjs と同じ比べ方)。
//   node rl/ob2/make_fid_r2.mjs <policy.json> <出力.json>
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {writeFid} from '../eval_modes.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const [pp, out] = process.argv.slice(2);
const W = JSON.parse(fs.readFileSync(pp, 'utf8')), sha = crypto.createHash('sha256').update(fs.readFileSync(pp)).digest('hex');
const lay = JSON.parse(fs.readFileSync(path.join(here, '../obstacle_sets/validation.json'), 'utf8')).layouts[0];
await writeFid('obstacle', W, sha, out, {start: {gate: 0, left_offset_m: 0, left_yaw_deg: 0, seed: 806000}, params: {layout_id: lay.id}, layout: lay});
console.log(out);
