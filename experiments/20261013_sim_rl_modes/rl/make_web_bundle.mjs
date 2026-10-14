#!/usr/bin/env node
// ブラウザに配るファイルの組(rl/runtime/ の全部)を写し、sha256 の一覧 manifest.json を作る(2026-10-13、sim_rl_modes)。
// 統合の段はこの組を jetformer/sim/static/rl/ に置く(teacher.js が import する)。Node の試験と同じファイルであることを sha256 で確かめられる。
//   node rl/make_web_bundle.mjs [--out web/rl]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {RUNTIME_VERSION} from './runtime/mode_runtime.mjs';
import {CATALOGUE_SHA256} from './runtime/catalogue.mjs';
import {ROOM_TABLE_SHA256} from './runtime/room_table.mjs';
import {FEATURES_VERSION, REFERENCE_LOOP_SHA256} from './runtime/features.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2), oi = args.indexOf('--out');
const out = path.resolve(oi >= 0 ? args[oi + 1] : path.join(here, '../web/rl'));
const src = path.join(here, 'runtime');
fs.mkdirSync(out, {recursive: true});
const files = {};
for (const f of fs.readdirSync(src).sort()) {
  if (!/\.(mjs|json)$/.test(f)) continue;
  const buf = fs.readFileSync(path.join(src, f));
  fs.writeFileSync(path.join(out, f), buf);
  files[f] = {sha256: crypto.createHash('sha256').update(buf).digest('hex'), bytes: buf.length};
}
const manifest = {schema: 'jf-sim-rl-web-bundle/v1', created_at: new Date().toISOString(), runtime_version: RUNTIME_VERSION, entry: 'mode_runtime.mjs',
  export: 'createModeTeacher(world, config, variation, {mode, weights, params, policySha256, obstacleSet})', features_version: FEATURES_VERSION,
  reference_loop_sha256: REFERENCE_LOOP_SHA256, catalogue_sha256: CATALOGUE_SHA256, room_table_sha256: ROOM_TABLE_SHA256, source: src,
  note: 'teacher.js(統合の段)は ./rc-speed-controller.js の差し替えとして、このフォルダーの mode_runtime.mjs を import する。重みは /api/sim/rl/modes/<key>/policy.json を fetch して渡す',
  files};
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 1) + '\n');
console.log(`書いた: ${out}(${Object.keys(files).length} ファイル)`);
