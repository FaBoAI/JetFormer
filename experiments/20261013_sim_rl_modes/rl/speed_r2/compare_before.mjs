#!/usr/bin/env node
// 速さいろいろ r2 で env_modes.mjs を変えたあと、ほかの run(base・no_wall・recover・cond・obstacle)の学習の環境が直す前と同じに動くことを確かめる
// (2026-10-14 学習の段 speed r2)。直す前の写し(train/speed/r2/before/env_modes.mjs)を rl/ に一時的に置いて import し、同じ種・同じ行動の列で
// 開始・線・c_v・観測・報酬・終わりの理由を比べる。終わったら一時の写しを消す。
//   node rl/speed_r2/compare_before.mjs
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import * as core from '../fast_core_modes.mjs';
import {createRng} from '../runtime/rng.mjs';

const here = path.dirname(fileURLToPath(import.meta.url)), rl = path.resolve(here, '..');
const before = '/home/jetson/jetformer_reports/sim_rl_modes_20261013/train/speed/r2/before/env_modes.mjs';
const tmp = path.join(rl, `.env_modes_before_${process.pid}.mjs`);
fs.copyFileSync(before, tmp);
try {
  const A = await import(pathToFileURL(path.join(rl, 'env_modes.mjs')).href), B = await import(pathToFileURL(tmp).href);
  let checked = 0;
  for (const run of ['base', 'no_wall', 'recover', 'cond', 'obstacle']) {
    const ea = new A.ModeEnv({core, run}), eb = new B.ModeEnv({core, run});
    const ra = createRng(1234), rb = createRng(1234), act = createRng(99);
    for (let ep = 0; ep < 4; ep++) {
      ea.resetRandom(ra); eb.resetRandom(rb);
      if (JSON.stringify(ea.start) !== JSON.stringify(eb.start)) throw new Error(`${run}: 開始が違います`);
      for (let k = 0; k < 200 && !ea.done; k++) {
        const a0 = act.uniform(-.3, .3), a1 = act.uniform(-.2, 1);
        const x = ea.step(a0, a1), y = eb.step(a0, a1);
        if (x.reward !== y.reward || x.reason !== y.reason || x.terminated !== y.terminated) throw new Error(`${run}: ${ep} ${k} の報酬・終わりが違います`);
        for (let j = 0; j < ea.obs.length; j++) if (ea.obs[j] !== eb.obs[j]) throw new Error(`${run}: 観測が違います`);
        checked++;
      }
      if (JSON.stringify(ea.summary()) !== JSON.stringify(eb.summary())) throw new Error(`${run}: 集計が違います`);
    }
    console.log(`${run}: 直す前と同じ(4 回)`);
  }
  console.log(`全部同じ: ${checked} ステップ`);
} finally { fs.rmSync(tmp, {force: true}); }
