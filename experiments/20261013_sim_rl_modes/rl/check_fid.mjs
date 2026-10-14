#!/usr/bin/env node
// F-ID(protocol test_sets.fastest.F-ID・ui_acceptance U5)の Node 側: 保存した 600 フレームの snapshot を 1 フレームずつ mode_runtime.mjs に与え、
// 保存した操作と比べる(|Δ舵| ≤ 1e-6 rad・|Δ速さ| ≤ 1e-6 m/s、info.jf も同じ)。ブラウザの確認(統合の段)は同じ fid ファイルを teacher.js に与える。
//   node rl/check_fid.mjs --fid fid.json --policy policy.json [--runtime rl/runtime/mode_runtime.mjs]
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const args = process.argv.slice(2), opt = (n, d = null) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const fid = JSON.parse(fs.readFileSync(opt('--fid'), 'utf8')), weights = JSON.parse(fs.readFileSync(opt('--policy'), 'utf8'));
const rt = await import(pathToFileURL(path.resolve(opt('--runtime', new URL('./runtime/mode_runtime.mjs', import.meta.url).pathname))).href);
const t = rt.createModeTeacher(fid.world, fid.config, fid.variation, {mode: fid.mode, weights, params: fid.params, policySha256: fid.policy_sha256});
let maxS = 0, maxV = 0, infoDiff = 0, first = null;
fid.frames.forEach((fr, i) => {
  const a = t.action(fr.snapshot), b = fr.action;
  const ds = Math.abs(a.steering - b.steering), dv = Math.abs(a.speed - b.speed);
  maxS = Math.max(maxS, ds); maxV = Math.max(maxV, dv);
  if (JSON.stringify(a.info) !== JSON.stringify(b.info)) infoDiff++;
  if (first === null && (ds > 1e-6 || dv > 1e-6)) first = i;
});
const pass = maxS <= 1e-6 && maxV <= 1e-6 && infoDiff === 0;
console.log(JSON.stringify({frames: fid.frames.length, max_abs_steering: maxS, max_abs_speed: maxV, info_mismatch: infoDiff, first_frame_over: first, pass}));
process.exit(pass ? 0 : 1);
