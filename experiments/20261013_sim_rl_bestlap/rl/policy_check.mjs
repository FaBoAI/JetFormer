#!/usr/bin/env node
// JS の方策(rl/policy_forward.mjs)で、保存した観測の行動を計算する(Python との一致の試験用。2026-10-13、RL の学習の段)
//
//   node rl/policy_check.mjs <重み.json> <観測.f32(float32 × N × 79)> <出力.f64(float64 × N × 2)>
import fs from 'node:fs';
import {loadPolicy} from './policy_forward.mjs';

const [weightsPath, obsPath, outPath] = process.argv.slice(2);
if (!outPath) { console.error('使い方: node rl/policy_check.mjs weights.json obs.f32 out.f64'); process.exit(2); }
const fn = await loadPolicy(weightsPath);
const buf = fs.readFileSync(obsPath);
const obs = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const n = obs.length / 79, out = new Float64Array(n * 2), o = new Float64Array(79);
for (let i = 0; i < n; i++) {
  for (let j = 0; j < 79; j++) o[j] = obs[i * 79 + j];
  const a = fn(o);
  out[2 * i] = a[0]; out[2 * i + 1] = a[1];
}
fs.writeFileSync(outPath, Buffer.from(out.buffer));
console.log(JSON.stringify({n}));
